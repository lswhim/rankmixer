"""
InterFormer: Effective Heterogeneous Interaction Learning for CTR Prediction
(Meta, arXiv:2411.09852).

This implementation follows the paper's three-part block:
- Interaction Arch: behavior-aware interaction over non-seq tokens plus
  sequence summaries.
- Sequence Arch: context-aware sequence modeling with personalized FFN and MHA.
- Cross Arch: gated summaries exchanged between the two arches.

The Interaction Arch follows the paper's experimental choice: DHEN with DOT
product and DCN branches ensembled together.
"""

import torch
import torch.nn as nn


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
            nn.SiLU(),
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
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        summary = self.token_proj(tokens.transpose(1, 2)).transpose(1, 2)
        return summary * torch.sigmoid(self.gate(summary))


class PersonalizedFFN(nn.Module):
    """Condition sequence tokens on non-sequence summaries as f(X_sum)S."""

    def __init__(self, hidden_dim: int, expansion: int = 4, dropout: float = 0.0):
        super().__init__()
        self.norm_seq = RMSNorm(hidden_dim)
        self.norm_ctx = RMSNorm(hidden_dim)
        self.condition = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * expansion),
            nn.SiLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim * expansion, hidden_dim * hidden_dim),
        )
        self.bias = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, seq_tokens: torch.Tensor, nonseq_summary: torch.Tensor) -> torch.Tensor:
        context = self.norm_ctx(nonseq_summary).mean(dim=1)
        B = seq_tokens.size(0)
        D = seq_tokens.size(-1)
        weight = self.condition(context).view(B, D, D)
        bias = self.bias(context).unsqueeze(1)
        x = self.norm_seq(seq_tokens)
        return seq_tokens + torch.bmm(x, weight) + bias


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


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def apply_rope(x: torch.Tensor) -> torch.Tensor:
    """Apply rotary position embedding to [B, H, T, Dh]."""
    T = x.size(-2)
    Dh = x.size(-1)
    device = x.device
    dtype = x.dtype
    inv_freq = 1.0 / (
        10000 ** (torch.arange(0, Dh, 2, device=device, dtype=torch.float32) / Dh)
    )
    pos = torch.arange(T, device=device, dtype=torch.float32)
    freqs = torch.einsum("t,d->td", pos, inv_freq)
    emb = torch.repeat_interleave(freqs, 2, dim=-1).to(dtype)
    cos = emb.cos()[None, None, :, :]
    sin = emb.sin()[None, None, :, :]
    return (x * cos) + (_rotate_half(x) * sin)


class RoPEMultiheadAttention(nn.Module):
    """Multi-head self-attention with RoPE on Q/K."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        if self.head_dim % 2 != 0:
            raise ValueError("RoPE requires an even head_dim")
        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3)
        self.out = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, seq_mask: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        q = apply_rope(q)
        k = apply_rope(k)

        attn = torch.matmul(q, k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        key_mask = seq_mask[:, None, None, :]
        attn = attn.masked_fill(~key_mask, -1e9)
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.out(out)


class SequenceSummary(nn.Module):
    """Build S_sum from CLS, PMA, and recent interacted tokens."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_cls_tokens: int = 4,
        num_pma_tokens: int = 2,
        num_recent_tokens: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_cls_tokens = num_cls_tokens
        self.num_recent_tokens = num_recent_tokens
        self.pma = PMAPooling(num_pma_tokens, hidden_dim, num_heads, dropout)
        total_tokens = num_cls_tokens + num_pma_tokens + num_recent_tokens
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.total_tokens = total_tokens

    def forward(self, seq_tokens: torch.Tensor, seq_mask: torch.Tensor) -> torch.Tensor:
        cls_token = seq_tokens[:, :self.num_cls_tokens]
        pma_tokens = self.pma(seq_tokens, seq_mask)
        body = seq_tokens[:, self.num_cls_tokens:]
        body_mask = seq_mask[:, self.num_cls_tokens:]
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
        num_cls_tokens: int = 4,
    ):
        super().__init__()
        self.pffn = PersonalizedFFN(hidden_dim, expansion=ffn_expansion, dropout=dropout)
        self.norm_attn = RMSNorm(hidden_dim)
        self.attn = RoPEMultiheadAttention(hidden_dim, num_heads, dropout)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.norm_ffn = RMSNorm(hidden_dim)
        self.ffn = FeedForward(hidden_dim, expansion=ffn_expansion, dropout=dropout)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_seq_len + num_cls_tokens, hidden_dim))
        nn.init.normal_(self.pos_emb, std=0.02)

    def forward(
        self,
        seq_tokens: torch.Tensor,
        seq_mask: torch.Tensor,
        nonseq_summary: torch.Tensor,
    ) -> torch.Tensor:
        x = self.pffn(seq_tokens, nonseq_summary)
        x = x + self.pos_emb[:, :x.size(1)]
        attn_out = self.attn(self.norm_attn(x), seq_mask)
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
        dhen_layers: int = 3,
        dcn_rank: int = 32,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_nonseq_tokens = num_nonseq_tokens
        self.total_tokens = num_nonseq_tokens + num_seq_summary_tokens
        flat_dim = self.total_tokens * hidden_dim
        output_dim = num_nonseq_tokens * hidden_dim
        num_pairs = self.total_tokens * (self.total_tokens - 1) // 2

        self.dot_proj = nn.Sequential(
            nn.Linear(num_pairs, output_dim),
            nn.SiLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )
        self.dcn_layers = nn.ModuleList([
            LowRankCrossLayer(flat_dim, dcn_rank) for _ in range(dhen_layers)
        ])
        self.dcn_proj = nn.Linear(flat_dim, output_dim)
        self.out = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(
        self,
        nonseq_tokens: torch.Tensor,
        seq_summary: torch.Tensor,
    ) -> torch.Tensor:
        tokens = torch.cat([nonseq_tokens, seq_summary], dim=1)
        dot_features = []
        for i in range(self.total_tokens):
            for j in range(i + 1, self.total_tokens):
                dot_features.append((tokens[:, i] * tokens[:, j]).sum(dim=-1, keepdim=True))
        dot_out = self.dot_proj(torch.cat(dot_features, dim=-1))

        x0 = tokens.flatten(1)
        x = x0
        for layer in self.dcn_layers:
            x = layer(x0, x)
        dcn_out = self.dcn_proj(x)

        updated = (dot_out + dcn_out).view(
            nonseq_tokens.size(0),
            self.num_nonseq_tokens,
            -1,
        )
        return nonseq_tokens + self.out(updated)


class LowRankCrossLayer(nn.Module):
    """DCNv2-style low-rank cross layer used in DHEN's DCN branch."""

    def __init__(self, input_dim: int, rank: int = 32):
        super().__init__()
        rank = min(rank, input_dim)
        self.u = nn.Linear(input_dim, rank, bias=False)
        self.v = nn.Linear(rank, input_dim, bias=True)

    def forward(self, x0: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return x0 * self.v(self.u(x)) + x


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
        dhen_layers: int = 3,
        dcn_rank: int = 32,
        dropout: float = 0.0,
        max_seq_len: int = 256,
        num_cls_tokens: int = 4,
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
            num_cls_tokens=num_cls_tokens,
            num_pma_tokens=num_pma_tokens,
            num_recent_tokens=num_recent_tokens,
            dropout=dropout,
        )
        self.interaction_arch = InteractionArch(
            num_nonseq_tokens,
            self.seq_summary.total_tokens,
            hidden_dim,
            dhen_layers=dhen_layers,
            dcn_rank=dcn_rank,
            dropout=dropout,
        )
        self.sequence_arch = SequenceArch(
            hidden_dim,
            num_heads,
            ffn_expansion=ffn_expansion,
            dropout=dropout,
            max_seq_len=max_seq_len,
            num_cls_tokens=num_cls_tokens,
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
