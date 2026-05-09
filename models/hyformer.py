"""
HyFormer: Revisiting the Roles of Sequence Modeling and Feature Interaction
in CTR Prediction (ByteDance, arXiv:2601.12681).

This module implements the paper's core idea for this repository:
alternate Query Decoding over behavior sequences with RankMixer-style
Query Boosting over global query tokens and non-sequential tokens.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rankmixer import RankMixerBlock


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class SwiGLU(nn.Module):
    def __init__(self, dim: int, expansion: int = 4, dropout: float = 0.0):
        super().__init__()
        inner_dim = dim * expansion
        self.w12 = nn.Linear(dim, inner_dim * 2)
        self.w3 = nn.Linear(inner_dim, dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = self.w12(x).chunk(2, dim=-1)
        return self.dropout(self.w3(F.silu(x1) * x2))


class SequenceFFNBlock(nn.Module):
    """Attention-free sequence representation update for layer-wise K/V."""

    def __init__(self, hidden_dim: int, expansion: int = 2, dropout: float = 0.0):
        super().__init__()
        self.norm = RMSNorm(hidden_dim)
        self.ffn = SwiGLU(hidden_dim, expansion=expansion, dropout=dropout)

    def forward(self, seq_tokens: torch.Tensor) -> torch.Tensor:
        return seq_tokens + self.ffn(self.norm(seq_tokens))


class QueryGenerator(nn.Module):
    """Generate global query tokens from non-sequential tokens and seq pooling."""

    def __init__(
        self,
        num_nonseq_tokens: int,
        num_query_tokens: int,
        hidden_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_query_tokens = num_query_tokens
        input_dim = (num_nonseq_tokens + 1) * hidden_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim * 2, num_query_tokens * hidden_dim),
        )

    def forward(
        self,
        nonseq_tokens: torch.Tensor,
        seq_tokens: torch.Tensor,
        seq_mask: torch.Tensor,
    ) -> torch.Tensor:
        B, _, D = nonseq_tokens.shape
        mask = seq_mask.unsqueeze(-1).to(seq_tokens.dtype)
        denom = mask.sum(dim=1).clamp_min(1.0)
        seq_pool = (seq_tokens * mask).sum(dim=1) / denom
        global_info = torch.cat([nonseq_tokens.flatten(1), seq_pool], dim=-1)
        return self.net(global_info).view(B, self.num_query_tokens, D)


class QueryDecoding(nn.Module):
    """Cross-attend global query tokens to behavior sequence K/V states."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.q_norm = RMSNorm(hidden_dim)
        self.seq_norm = RMSNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(
        self,
        queries: torch.Tensor,
        seq_tokens: torch.Tensor,
        seq_mask: torch.Tensor,
    ) -> torch.Tensor:
        key_padding_mask = ~seq_mask
        all_padding = key_padding_mask.all(dim=1)
        if all_padding.any():
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[all_padding, 0] = False
        decoded, _ = self.attn(
            self.q_norm(queries),
            self.seq_norm(seq_tokens),
            self.seq_norm(seq_tokens),
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        return queries + self.dropout(decoded)


class QueryBoosting(nn.Module):
    """RankMixer-style token mixing over decoded queries and NS tokens."""

    def __init__(
        self,
        num_query_tokens: int,
        num_nonseq_tokens: int,
        hidden_dim: int,
        ffn_expansion: int = 4,
    ):
        super().__init__()
        self.num_query_tokens = num_query_tokens
        self.num_nonseq_tokens = num_nonseq_tokens
        self.total_tokens = num_query_tokens + num_nonseq_tokens
        if hidden_dim % self.total_tokens != 0:
            raise ValueError(
                "HyFormer QueryBoosting requires hidden_dim divisible by "
                f"num_query_tokens + num_nonseq_tokens; got hidden_dim={hidden_dim}, "
                f"tokens={self.total_tokens}."
            )
        self.block = RankMixerBlock(self.total_tokens, hidden_dim, ffn_expansion)

    def forward(
        self,
        queries: torch.Tensor,
        nonseq_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mixed = self.block(torch.cat([queries, nonseq_tokens], dim=1))
        return (
            mixed[:, :self.num_query_tokens],
            mixed[:, self.num_query_tokens:],
        )


class HyFormerBlock(nn.Module):
    def __init__(
        self,
        num_query_tokens: int,
        num_nonseq_tokens: int,
        hidden_dim: int,
        num_heads: int,
        ffn_expansion: int = 4,
        seq_ffn_expansion: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.seq_encoder = SequenceFFNBlock(
            hidden_dim,
            expansion=seq_ffn_expansion,
            dropout=dropout,
        )
        self.query_decoding = QueryDecoding(hidden_dim, num_heads, dropout)
        self.query_boosting = QueryBoosting(
            num_query_tokens,
            num_nonseq_tokens,
            hidden_dim,
            ffn_expansion=ffn_expansion,
        )

    def forward(
        self,
        queries: torch.Tensor,
        nonseq_tokens: torch.Tensor,
        seq_tokens: torch.Tensor,
        seq_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        seq_tokens = self.seq_encoder(seq_tokens)
        queries = self.query_decoding(queries, seq_tokens, seq_mask)
        queries, nonseq_tokens = self.query_boosting(queries, nonseq_tokens)
        return queries, nonseq_tokens, seq_tokens
