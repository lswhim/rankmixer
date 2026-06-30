"""
OneTrans: Unified Feature Interaction and Sequence Modeling with One Transformer
(ByteDance + NTU, arXiv:2510.26104, WWW 2026).

Core components implemented here:
- Mixed parameterization (shared QKV/FFN for S-tokens, per-token for NS-tokens)
- Pre-RMSNorm OneTrans block with mixed causal multi-head attention + mixed FFN
- Optional pyramid stack that progressively keeps fewer S-token queries per layer

Not implemented in this module (CTR wrapper / serving stubs):
- Cross-request KV cache (see OneTransCTR notes)
- FlashAttention / mixed-precision hooks (handled by training infra)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class TokenLayout:
    """
    OneTrans unified token sequence layout:

        [ S₁  S₂  …  Sₙ  |  NS₁  NS₂  …  NSₘ ]
        └─ sequence ────┘  └── non-sequence ──┘

    S-tokens (sequence): shared Q/K/V and FFN weights across all positions.
    NS-tokens (non-sequence): each position has its own Q/K/V and FFN weights.
    """

    num_sequence_tokens: int
    num_non_sequence_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.num_sequence_tokens + self.num_non_sequence_tokens

    def split(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Split [B, S+NS, D] into sequence and non-sequence segments."""
        n_seq = self.num_sequence_tokens
        return x[:, :n_seq], x[:, n_seq:]


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class MixedLinear(nn.Module):
    """
    Mixed linear projection (OneTrans Eqn. 11–12).

    QKV strategy:
    - **S-tokens** (sequence): one shared ``Linear(dim, dim)`` for every position.
    - **NS-tokens** (non-sequence): position ``i`` uses its own weight ``ns_weight[i]``.

    Call ``forward_split(seq_tokens, ns_tokens)`` when S/NS segments are already
    separated; ``forward(x, num_sequence_tokens)`` splits a concatenated tensor.
    """

    def __init__(self, dim: int, num_non_sequence_tokens: int, bias: bool = True):
        super().__init__()
        self.dim = dim
        self.num_non_sequence_tokens = num_non_sequence_tokens
        # Shared by all sequence tokens (S₁…Sₙ).
        self.shared = nn.Linear(dim, dim, bias=bias)
        # One weight matrix per non-sequence token (NS₁…NSₘ).
        self.ns_weight = nn.Parameter(
            torch.empty(num_non_sequence_tokens, dim, dim)
        )
        if bias:
            self.ns_bias = nn.Parameter(torch.zeros(num_non_sequence_tokens, dim))
        else:
            self.register_parameter("ns_bias", None)
        nn.init.xavier_uniform_(self.ns_weight)
        if self.ns_bias is not None:
            nn.init.zeros_(self.ns_bias)

    def forward_split(
        self,
        seq_tokens: torch.Tensor,
        ns_tokens: torch.Tensor,
    ) -> torch.Tensor:
        # S: shared W, b  →  y_s = x_s @ W^T + b
        seq_out = self.shared(seq_tokens)
        if self.num_non_sequence_tokens == 0:
            return seq_out
        # NS: per-token W_i, b_i  →  y_ns[i] = x_ns[i] @ W_i^T + b_i
        ns_out = torch.einsum("bnd, ndo -> bno", ns_tokens, self.ns_weight)
        if self.ns_bias is not None:
            ns_out = ns_out + self.ns_bias.unsqueeze(0)
        return torch.cat([seq_out, ns_out], dim=1)

    def forward(self, x: torch.Tensor, num_sequence_tokens: int) -> torch.Tensor:
        seq_tokens, ns_tokens = x[:, :num_sequence_tokens], x[:, num_sequence_tokens:]
        return self.forward_split(seq_tokens, ns_tokens)


class MixedFFN(nn.Module):
    """
    Mixed feed-forward network: shared FFN for S-tokens, per-token FFN for NS-tokens.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_non_sequence_tokens: int,
        expansion: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        inner_dim = hidden_dim * expansion
        self.num_non_sequence_tokens = num_non_sequence_tokens
        # Shared two-layer FFN for all sequence tokens.
        self.s_ffn = nn.Sequential(
            nn.Linear(hidden_dim, inner_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(inner_dim, hidden_dim),
        )
        if num_non_sequence_tokens > 0:
            # Per NS-token FFN weights.
            self.ns_w1 = nn.Parameter(
                torch.empty(num_non_sequence_tokens, hidden_dim, inner_dim)
            )
            self.ns_b1 = nn.Parameter(torch.zeros(num_non_sequence_tokens, inner_dim))
            self.ns_w2 = nn.Parameter(
                torch.empty(num_non_sequence_tokens, inner_dim, hidden_dim)
            )
            self.ns_b2 = nn.Parameter(torch.zeros(num_non_sequence_tokens, hidden_dim))
            for t in range(num_non_sequence_tokens):
                nn.init.kaiming_uniform_(self.ns_w1[t], a=math.sqrt(5))
                nn.init.kaiming_uniform_(self.ns_w2[t], a=math.sqrt(5))

    def forward_split(
        self,
        seq_tokens: torch.Tensor,
        ns_tokens: torch.Tensor,
    ) -> torch.Tensor:
        seq_out = self.s_ffn(seq_tokens)
        if self.num_non_sequence_tokens == 0:
            return seq_out
        h = torch.einsum("bnd, ndi -> bni", ns_tokens, self.ns_w1) + self.ns_b1.unsqueeze(0)
        h = F.gelu(h)
        ns_out = torch.einsum("bni, nio -> bno", h, self.ns_w2) + self.ns_b2.unsqueeze(0)
        return torch.cat([seq_out, ns_out], dim=1)

    def forward(self, x: torch.Tensor, num_sequence_tokens: int) -> torch.Tensor:
        seq_tokens, ns_tokens = x[:, :num_sequence_tokens], x[:, num_sequence_tokens:]
        return self.forward_split(seq_tokens, ns_tokens)


class MixedCausalAttention(nn.Module):
    """
    Causal multi-head attention with mixed Q/K/V/out projections (paper Sec. 3.3.1).

    QKV layout mirrors ``MixedLinear``: S-tokens share projections; each NS-token
    has dedicated Q, K, V, and output projections.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_non_sequence_tokens: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"
            )
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.num_non_sequence_tokens = num_non_sequence_tokens
        self.scale = self.head_dim ** -0.5

        self.q_proj = MixedLinear(hidden_dim, num_non_sequence_tokens, bias=True)
        self.k_proj = MixedLinear(hidden_dim, num_non_sequence_tokens, bias=True)
        self.v_proj = MixedLinear(hidden_dim, num_non_sequence_tokens, bias=True)
        self.out_proj = MixedLinear(hidden_dim, num_non_sequence_tokens, bias=True)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        return x.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, H, L, Dh = x.shape
        return x.transpose(1, 2).contiguous().view(B, L, H * Dh)

    def forward(
        self,
        x: torch.Tensor,
        num_sequence_tokens: int,
        attn_mask: torch.Tensor,
        query_indices: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Explicit S | NS split before mixed QKV projection.
        layout = TokenLayout(num_sequence_tokens, self.num_non_sequence_tokens)
        seq_tokens, ns_tokens = layout.split(x)

        q_full = self._split_heads(self.q_proj.forward_split(seq_tokens, ns_tokens))
        k_full = self._split_heads(self.k_proj.forward_split(seq_tokens, ns_tokens))
        v_full = self._split_heads(self.v_proj.forward_split(seq_tokens, ns_tokens))

        if query_indices is None:
            q = q_full
            query_idx = torch.arange(x.size(1), device=x.device)
        else:
            q = q_full.index_select(2, query_indices)
            query_idx = query_indices

        attn_logits = torch.matmul(q, k_full.transpose(-2, -1)) * self.scale

        q_pos = query_idx.view(1, 1, -1, 1)
        k_pos = torch.arange(x.size(1), device=x.device).view(1, 1, 1, -1)
        causal_mask = k_pos > q_pos
        attn_logits = attn_logits.masked_fill(causal_mask, float("-inf"))

        if key_padding_mask is not None:
            pad_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_logits = attn_logits.masked_fill(pad_mask, float("-inf"))

        if attn_mask is not None:
            attn_logits = attn_logits.masked_fill(attn_mask, float("-inf"))

        all_masked = torch.isinf(attn_logits).all(dim=-1, keepdim=True)
        attn_probs = torch.softmax(attn_logits, dim=-1)
        attn_probs = attn_probs.masked_fill(all_masked, 0.0)
        attn_probs = self.dropout(attn_probs)

        out = torch.matmul(attn_probs, v_full)
        out = self._merge_heads(out)

        if query_indices is None:
            num_seq_in_out = num_sequence_tokens
            seq_out, ns_out = out[:, :num_sequence_tokens], out[:, num_sequence_tokens:]
        else:
            num_seq_in_out = int((query_indices < num_sequence_tokens).sum().item())
            seq_out = out[:, :num_seq_in_out]
            ns_out = out[:, num_seq_in_out:]

        return self.out_proj.forward_split(seq_out, ns_out)


class OneTransBlock(nn.Module):
    """
    Pre-RMSNorm causal block (paper Eqn. 4-5):
      Z = MixedMHA(RMSNorm(X)) + X
      X' = MixedFFN(RMSNorm(Z)) + Z
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_non_sequence_tokens: int,
        ffn_expansion: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.attn_norm = RMSNorm(hidden_dim)
        self.ffn_norm = RMSNorm(hidden_dim)
        self.attn = MixedCausalAttention(
            hidden_dim, num_heads, num_non_sequence_tokens, dropout=dropout
        )
        self.ffn = MixedFFN(
            hidden_dim, num_non_sequence_tokens, expansion=ffn_expansion, dropout=dropout
        )

    def forward(
        self,
        x: torch.Tensor,
        num_sequence_tokens: int,
        attn_mask: torch.Tensor,
        query_indices: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, int]:
        residual = x
        if query_indices is not None:
            residual = x.index_select(1, query_indices)

        h = self.attn(
            self.attn_norm(x),
            num_sequence_tokens=num_sequence_tokens,
            attn_mask=attn_mask,
            query_indices=query_indices,
            key_padding_mask=key_padding_mask,
        )
        z = residual + h

        num_seq_after_attn = num_sequence_tokens
        if query_indices is not None:
            num_seq_after_attn = int((query_indices < num_sequence_tokens).sum().item())

        out = z + self.ffn(self.ffn_norm(z), num_seq_after_attn)
        return out, num_seq_after_attn


def build_pyramid_query_indices(
    layer_idx: int,
    num_layers: int,
    initial_num_sequence_tokens: int,
    current_num_sequence_tokens: int,
    num_non_sequence_tokens: int,
    min_sequence_queries: int,
) -> torch.Tensor:
    """
    Pyramid schedule (paper Sec. 3.4): keep the tail of S-tokens as queries;
    NS-tokens always remain in the query set.
    """
    if num_layers <= 1:
        seq_queries = current_num_sequence_tokens
    else:
        keep_ratio = 1.0 - (layer_idx + 1) / num_layers
        seq_queries = max(
            min_sequence_queries,
            int(math.ceil(initial_num_sequence_tokens * keep_ratio)),
        )
        seq_queries = min(seq_queries, current_num_sequence_tokens)
    seq_idx = torch.arange(
        current_num_sequence_tokens - seq_queries, current_num_sequence_tokens
    )
    ns_idx = torch.arange(
        current_num_sequence_tokens,
        current_num_sequence_tokens + num_non_sequence_tokens,
    )
    return torch.cat([seq_idx, ns_idx])


class OneTransStack(nn.Module):
    """Stack of OneTrans blocks with optional pyramid token pruning."""

    def __init__(
        self,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        num_non_sequence_tokens: int,
        ffn_expansion: int = 4,
        dropout: float = 0.0,
        use_pyramid: bool = False,
        pyramid_min_s_queries: int = 1,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.num_non_sequence_tokens = num_non_sequence_tokens
        self.use_pyramid = use_pyramid
        self.pyramid_min_s_queries = pyramid_min_s_queries
        self.blocks = nn.ModuleList([
            OneTransBlock(
                hidden_dim,
                num_heads,
                num_non_sequence_tokens,
                ffn_expansion=ffn_expansion,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

    def forward(
        self,
        x: torch.Tensor,
        num_sequence_tokens: int,
        seq_valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        num_ns = self.num_non_sequence_tokens
        key_padding_mask = torch.cat([
            ~seq_valid_mask,
            torch.zeros(
                x.size(0), num_ns, dtype=torch.bool, device=x.device
            ),
        ], dim=1)

        all_padding = key_padding_mask.all(dim=1)
        if all_padding.any():
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[all_padding, 0] = False

        initial_num_sequence_tokens = num_sequence_tokens
        attn_mask = None
        for layer_idx, block in enumerate(self.blocks):
            query_indices = None
            if self.use_pyramid:
                query_indices = build_pyramid_query_indices(
                    layer_idx,
                    self.num_layers,
                    initial_num_sequence_tokens,
                    num_sequence_tokens,
                    num_ns,
                    self.pyramid_min_s_queries,
                ).to(x.device)

            x, num_sequence_tokens = block(
                x,
                num_sequence_tokens=num_sequence_tokens,
                attn_mask=attn_mask,
                query_indices=query_indices,
                key_padding_mask=key_padding_mask,
            )

            if query_indices is not None:
                key_padding_mask = key_padding_mask.index_select(1, query_indices)

        return x
