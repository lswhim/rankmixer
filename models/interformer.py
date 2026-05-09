"""
InterFormer: Effective Heterogeneous Interaction Learning for CTR Prediction
(Meta, arXiv:2411.09852).

This implementation follows the paper's three-part block:
- Interaction Arch: behavior-aware interaction over non-seq tokens plus
  sequence summaries.
- Sequence Arch: context-aware sequence modeling with personalized FFN and MHA.
- Cross Arch: gated summaries exchanged between the two arches.

For this repository, the Interaction Arch uses the existing RankMixer block as
the feature interaction backbone.
"""

import torch
import torch.nn as nn

from .rankmixer import RankMixerBlock


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class FeedForward(nn.Module):
    def __init__(self, hidden_dim: int, expansion: int = 4, dropout: float = 0.0):
        super().__init__()
        inner_dim = hidden_dim * expansion
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, inner_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(inner_dim, hidden_dim),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TokenSummaryGate(nn.Module):
    """Project many tokens to a small number of gated summary tokens."""

    def __init__(self, num_input_tokens: int, num_summary_tokens: int, hidden_dim: int):
        super().__init__()
        self.token_proj = nn.Linear(num_input_tokens, num_summary_tokens)
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        summary = self.token_proj(tokens.transpose(1, 2)).transpose(1, 2)
        return summary * torch.sigmoid(self.gate(summary))


class PersonalizedFFN(nn.Module):
    """Condition sequence tokens on non-sequence summaries.

    The paper describes a personalized projection f(X_sum)S. A full per-sample
    D x D projection is expensive, so this implementation uses FiLM-style
    feature-wise scaling and bias from the summarized non-sequence context.
    """

    def __init__(self, hidden_dim: int, expansion: int = 4, dropout: float = 0.0):
        super().__init__()
        self.norm_seq = RMSNorm(hidden_dim)
        self.norm_ctx = RMSNorm(hidden_dim)
        self.condition = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * expansion),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim * expansion, hidden_dim * 2),
        )
        self.out = FeedForward(hidden_dim, expansion=expansion, dropout=dropout)

    def forward(self, seq_tokens: torch.Tensor, nonseq_summary: torch.Tensor) -> torch.Tensor:
        context = self.norm_ctx(nonseq_summary).mean(dim=1)
        scale, bias = self.condition(context).chunk(2, dim=-1)
        x = self.norm_seq(seq_tokens)
        x = x * (1.0 + scale.unsqueeze(1)) + bias.unsqueeze(1)
        return seq_tokens + self.out(x)


class PMAPooling(nn.Module):
    """Pooling by Multihead Attention with learnable seed queries."""

    def __init__(
        self,
        num_pma_tokens: int,
        hidden_dim: int,
        num_heads: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.seed = nn.Parameter(torch.zeros(1, num_pma_tokens, hidden_dim))
        nn.init.normal_(self.seed, std=0.02)
        self.norm = RMSNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )

    def forward(self, seq_tokens: torch.Tensor, seq_mask: torch.Tensor) -> torch.Tensor:
        B = seq_tokens.size(0)
        key_padding_mask = ~seq_mask
        all_padding = key_padding_mask.all(dim=1)
        if all_padding.any():
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[all_padding, 0] = False
        out, _ = self.attn(
            self.seed.expand(B, -1, -1),
            self.norm(seq_tokens),
            self.norm(seq_tokens),
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        return out


class SequenceSummary(nn.Module):
    """Build S_sum from CLS, PMA, and recent interacted tokens."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_pma_tokens: int = 2,
        num_recent_tokens: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_recent_tokens = num_recent_tokens
        self.pma = PMAPooling(num_pma_tokens, hidden_dim, num_heads, dropout)
        total_tokens = 1 + num_pma_tokens + num_recent_tokens
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.total_tokens = total_tokens

    def forward(self, seq_tokens: torch.Tensor, seq_mask: torch.Tensor) -> torch.Tensor:
        cls_token = seq_tokens[:, :1]
        pma_tokens = self.pma(seq_tokens, seq_mask)
        body = seq_tokens[:, 1:]
        body_mask = seq_mask[:, 1:]
        B, _, D = body.shape
        recent = []
        for offset in range(self.num_recent_tokens):
            idx = (body_mask.long().sum(dim=1) - 1 - offset).clamp_min(0)
            recent_token = body[torch.arange(B, device=body.device), idx]
            recent_token = recent_token.masked_fill(
                (body_mask.long().sum(dim=1) <= offset).unsqueeze(-1),
                0.0,
            )
            recent.append(recent_token)
        recent_tokens = torch.stack(list(reversed(recent)), dim=1)
        summary = torch.cat([cls_token, pma_tokens, recent_tokens], dim=1)
        return summary * torch.sigmoid(self.gate(summary))


class SequenceArch(nn.Module):
    """Context-aware sequence modeling: PFFN followed by self-attention."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        ffn_expansion: int = 4,
        dropout: float = 0.0,
        max_seq_len: int = 256,
    ):
        super().__init__()
        self.pffn = PersonalizedFFN(hidden_dim, expansion=ffn_expansion, dropout=dropout)
        self.norm_attn = RMSNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.norm_ffn = RMSNorm(hidden_dim)
        self.ffn = FeedForward(hidden_dim, expansion=ffn_expansion, dropout=dropout)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_seq_len + 1, hidden_dim))
        nn.init.normal_(self.pos_emb, std=0.02)

    def forward(
        self,
        seq_tokens: torch.Tensor,
        seq_mask: torch.Tensor,
        nonseq_summary: torch.Tensor,
    ) -> torch.Tensor:
        x = self.pffn(seq_tokens, nonseq_summary)
        x = x + self.pos_emb[:, :x.size(1)]
        key_padding_mask = ~seq_mask
        all_padding = key_padding_mask.all(dim=1)
        if all_padding.any():
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[all_padding, 0] = False
        attn_out, _ = self.attn(
            self.norm_attn(x),
            self.norm_attn(x),
            self.norm_attn(x),
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + self.dropout(attn_out)
        x = x + self.ffn(self.norm_ffn(x))
        return x * seq_mask.unsqueeze(-1).to(x.dtype)


class InteractionArch(nn.Module):
    """Behavior-aware non-sequence interaction with sequence summaries."""

    def __init__(
        self,
        num_nonseq_tokens: int,
        num_seq_summary_tokens: int,
        hidden_dim: int,
        ffn_expansion: int = 4,
    ):
        super().__init__()
        self.num_nonseq_tokens = num_nonseq_tokens
        self.total_tokens = num_nonseq_tokens + num_seq_summary_tokens
        if hidden_dim % self.total_tokens != 0:
            raise ValueError(
                "InterFormer InteractionArch requires hidden_dim divisible by "
                f"num_nonseq_tokens + num_seq_summary_tokens; got hidden_dim={hidden_dim}, "
                f"tokens={self.total_tokens}."
            )
        self.interaction = RankMixerBlock(self.total_tokens, hidden_dim, ffn_expansion)
        self.project_back = nn.Sequential(
            RMSNorm(hidden_dim),
            FeedForward(hidden_dim, expansion=ffn_expansion),
        )

    def forward(
        self,
        nonseq_tokens: torch.Tensor,
        seq_summary: torch.Tensor,
    ) -> torch.Tensor:
        mixed = self.interaction(torch.cat([nonseq_tokens, seq_summary], dim=1))
        updated = mixed[:, :self.num_nonseq_tokens]
        return nonseq_tokens + self.project_back(updated)


class InterFormerBlock(nn.Module):
    def __init__(
        self,
        num_nonseq_tokens: int,
        num_nonseq_summary_tokens: int,
        hidden_dim: int,
        num_heads: int,
        num_pma_tokens: int = 2,
        num_recent_tokens: int = 1,
        ffn_expansion: int = 4,
        dropout: float = 0.0,
        max_seq_len: int = 256,
    ):
        super().__init__()
        self.nonseq_summary = TokenSummaryGate(
            num_nonseq_tokens,
            num_nonseq_summary_tokens,
            hidden_dim,
        )
        self.seq_summary = SequenceSummary(
            hidden_dim,
            num_heads,
            num_pma_tokens=num_pma_tokens,
            num_recent_tokens=num_recent_tokens,
            dropout=dropout,
        )
        self.interaction_arch = InteractionArch(
            num_nonseq_tokens,
            self.seq_summary.total_tokens,
            hidden_dim,
            ffn_expansion=ffn_expansion,
        )
        self.sequence_arch = SequenceArch(
            hidden_dim,
            num_heads,
            ffn_expansion=ffn_expansion,
            dropout=dropout,
            max_seq_len=max_seq_len,
        )

    def forward(
        self,
        nonseq_tokens: torch.Tensor,
        seq_tokens: torch.Tensor,
        seq_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x_sum = self.nonseq_summary(nonseq_tokens)
        seq_tokens = self.sequence_arch(seq_tokens, seq_mask, x_sum)
        s_sum = self.seq_summary(seq_tokens, seq_mask)
        nonseq_tokens = self.interaction_arch(nonseq_tokens, s_sum)
        return nonseq_tokens, seq_tokens
