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


class SequenceSelfAttentionBlock(nn.Module):
    """Lightweight full-sequence encoder for layer-wise K/V states."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        expansion: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.attn_norm = RMSNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn_norm = RMSNorm(hidden_dim)
        self.ffn = SwiGLU(hidden_dim, expansion=expansion, dropout=dropout)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, seq_tokens: torch.Tensor, seq_mask: torch.Tensor) -> torch.Tensor:
        key_padding_mask = ~seq_mask
        all_padding = key_padding_mask.all(dim=1)
        if all_padding.any():
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[all_padding, 0] = False
        attn_out, _ = self.attn(
            self.attn_norm(seq_tokens),
            self.attn_norm(seq_tokens),
            self.attn_norm(seq_tokens),
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        seq_tokens = seq_tokens + self.dropout(attn_out)
        seq_tokens = seq_tokens + self.ffn(self.ffn_norm(seq_tokens))
        return seq_tokens * seq_mask.unsqueeze(-1).to(seq_tokens.dtype)


class SequenceEncoderBlock(nn.Module):
    """Selectable sequence K/V encoder: paper supports capacity/latency variants."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        expansion: int = 2,
        dropout: float = 0.0,
        encoder_type: str = "attention",
    ):
        super().__init__()
        self.encoder_type = encoder_type.lower()
        if self.encoder_type == "attention":
            self.encoder = SequenceSelfAttentionBlock(
                hidden_dim, num_heads, expansion=expansion, dropout=dropout
            )
        elif self.encoder_type == "ffn":
            self.encoder = SequenceFFNBlock(
                hidden_dim, expansion=expansion, dropout=dropout
            )
        elif self.encoder_type == "none":
            self.encoder = nn.Identity()
        else:
            raise ValueError(
                "seq_encoder_type must be one of: attention, ffn, none; "
                f"got {encoder_type}"
            )

    def forward(self, seq_tokens: torch.Tensor, seq_mask: torch.Tensor) -> torch.Tensor:
        if self.encoder_type == "attention":
            return self.encoder(seq_tokens, seq_mask)
        if self.encoder_type == "ffn":
            seq_tokens = self.encoder(seq_tokens)
            return seq_tokens * seq_mask.unsqueeze(-1).to(seq_tokens.dtype)
        return seq_tokens * seq_mask.unsqueeze(-1).to(seq_tokens.dtype)


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


class MultiSequenceQueryGenerator(nn.Module):
    """
    Generate sequence-specific global query tokens from NS tokens and per-seq pools.

    This follows the HyFormer paper more closely than a single merged sequence:
    each behavior sequence receives its own query set, while all query sets are
    conditioned on the same non-sequential context and cross-sequence pooling.
    """

    def __init__(
        self,
        num_nonseq_tokens: int,
        num_sequences: int,
        query_tokens_per_sequence: int,
        hidden_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_sequences = num_sequences
        self.query_tokens_per_sequence = query_tokens_per_sequence
        self.total_query_tokens = num_sequences * query_tokens_per_sequence
        input_dim = (num_nonseq_tokens + num_sequences) * hidden_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim * 2, self.total_query_tokens * hidden_dim),
        )

    @staticmethod
    def _masked_pool(seq_tokens: torch.Tensor, seq_mask: torch.Tensor) -> torch.Tensor:
        mask = seq_mask.unsqueeze(-1).to(seq_tokens.dtype)
        denom = mask.sum(dim=1).clamp_min(1.0)
        return (seq_tokens * mask).sum(dim=1) / denom

    def forward(
        self,
        nonseq_tokens: torch.Tensor,
        seq_tokens_list: list[torch.Tensor],
        seq_masks: list[torch.Tensor],
    ) -> torch.Tensor:
        B, _, D = nonseq_tokens.shape
        seq_pools = [
            self._masked_pool(seq_tokens, seq_mask)
            for seq_tokens, seq_mask in zip(seq_tokens_list, seq_masks)
        ]
        global_info = torch.cat([nonseq_tokens.flatten(1), *seq_pools], dim=-1)
        return self.net(global_info).view(B, self.total_query_tokens, D)


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
        seq_encoder_type: str = "ffn",
    ):
        super().__init__()
        self.seq_encoder = SequenceEncoderBlock(
            hidden_dim,
            num_heads,
            expansion=seq_ffn_expansion,
            dropout=dropout,
            encoder_type=seq_encoder_type,
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
        seq_tokens = self.seq_encoder(seq_tokens, seq_mask)
        queries = self.query_decoding(queries, seq_tokens, seq_mask)
        queries, nonseq_tokens = self.query_boosting(queries, nonseq_tokens)
        return queries, nonseq_tokens, seq_tokens


class MultiSequenceHyFormerBlock(nn.Module):
    """
    Paper-style HyFormer block for multiple heterogeneous behavior sequences.

    Each sequence keeps independent K/V states and query tokens during Query
    Decoding. Query Boosting then mixes all decoded global tokens with NS tokens,
    enabling cross-sequence and sequence/non-sequence interaction without forcing
    the raw sequences into one merged stream.
    """

    def __init__(
        self,
        num_sequences: int,
        query_tokens_per_sequence: int,
        num_nonseq_tokens: int,
        hidden_dim: int,
        num_heads: int,
        ffn_expansion: int = 4,
        seq_ffn_expansion: int = 2,
        dropout: float = 0.0,
        seq_encoder_type: str = "attention",
    ):
        super().__init__()
        self.num_sequences = num_sequences
        self.query_tokens_per_sequence = query_tokens_per_sequence
        total_query_tokens = num_sequences * query_tokens_per_sequence
        self.seq_encoders = nn.ModuleList([
            SequenceEncoderBlock(
                hidden_dim,
                num_heads,
                expansion=seq_ffn_expansion,
                dropout=dropout,
                encoder_type=seq_encoder_type,
            )
            for _ in range(num_sequences)
        ])
        self.query_decoders = nn.ModuleList([
            QueryDecoding(hidden_dim, num_heads, dropout)
            for _ in range(num_sequences)
        ])
        self.query_boosting = QueryBoosting(
            total_query_tokens,
            num_nonseq_tokens,
            hidden_dim,
            ffn_expansion=ffn_expansion,
        )

    def forward(
        self,
        queries: torch.Tensor,
        nonseq_tokens: torch.Tensor,
        seq_tokens_list: list[torch.Tensor],
        seq_masks: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        query_chunks = queries.split(self.query_tokens_per_sequence, dim=1)
        decoded_queries = []
        next_seq_tokens = []
        for idx, (query, seq_tokens, seq_mask) in enumerate(
            zip(query_chunks, seq_tokens_list, seq_masks)
        ):
            seq_tokens = self.seq_encoders[idx](seq_tokens, seq_mask)
            query = self.query_decoders[idx](query, seq_tokens, seq_mask)
            decoded_queries.append(query)
            next_seq_tokens.append(seq_tokens)
        queries = torch.cat(decoded_queries, dim=1)
        queries, nonseq_tokens = self.query_boosting(queries, nonseq_tokens)
        return queries, nonseq_tokens, next_seq_tokens
