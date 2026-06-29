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
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


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
    Mixed linear map from Eqn. (11)-(12):
    S-tokens (index < num_s) share one weight; each NS-token has its own weight.
    """

    def __init__(self, dim: int, num_ns_tokens: int, bias: bool = True):
        super().__init__()
        self.dim = dim
        self.num_ns = num_ns_tokens
        self.shared = nn.Linear(dim, dim, bias=bias)
        self.ns_weight = nn.Parameter(torch.empty(num_ns_tokens, dim, dim))
        if bias:
            self.ns_bias = nn.Parameter(torch.zeros(num_ns_tokens, dim))
        else:
            self.register_parameter("ns_bias", None)
        nn.init.xavier_uniform_(self.ns_weight)
        if self.ns_bias is not None:
            nn.init.zeros_(self.ns_bias)

    def forward(self, x: torch.Tensor, num_s: int) -> torch.Tensor:
        s_part = self.shared(x[:, :num_s])
        if self.num_ns == 0:
            return s_part
        ns_part = x[:, num_s:]
        ns_out = torch.einsum("bnd, ndo -> bno", ns_part, self.ns_weight)
        if self.ns_bias is not None:
            ns_out = ns_out + self.ns_bias.unsqueeze(0)
        return torch.cat([s_part, ns_out], dim=1)


class MixedFFN(nn.Module):
    """Mixed feed-forward: one shared FFN for S-tokens, per-token FFN for NS-tokens."""

    def __init__(
        self,
        hidden_dim: int,
        num_ns_tokens: int,
        expansion: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        inner_dim = hidden_dim * expansion
        self.num_ns = num_ns_tokens
        self.s_ffn = nn.Sequential(
            nn.Linear(hidden_dim, inner_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(inner_dim, hidden_dim),
        )
        if num_ns_tokens > 0:
            self.ns_w1 = nn.Parameter(torch.empty(num_ns_tokens, hidden_dim, inner_dim))
            self.ns_b1 = nn.Parameter(torch.zeros(num_ns_tokens, inner_dim))
            self.ns_w2 = nn.Parameter(torch.empty(num_ns_tokens, inner_dim, hidden_dim))
            self.ns_b2 = nn.Parameter(torch.zeros(num_ns_tokens, hidden_dim))
            for t in range(num_ns_tokens):
                nn.init.kaiming_uniform_(self.ns_w1[t], a=math.sqrt(5))
                nn.init.kaiming_uniform_(self.ns_w2[t], a=math.sqrt(5))

    def forward(self, x: torch.Tensor, num_s: int) -> torch.Tensor:
        s_out = self.s_ffn(x[:, :num_s])
        if self.num_ns == 0:
            return s_out
        ns = x[:, num_s:]
        h = torch.einsum("bnd, ndi -> bni", ns, self.ns_w1) + self.ns_b1.unsqueeze(0)
        h = F.gelu(h)
        ns_out = torch.einsum("bni, nio -> bno", h, self.ns_w2) + self.ns_b2.unsqueeze(0)
        return torch.cat([s_out, ns_out], dim=1)


class MixedCausalAttention(nn.Module):
    """Standard causal MHA with mixed Q/K/V/out projections (paper Sec. 3.3.1)."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_ns_tokens: int,
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
        self.num_ns = num_ns_tokens
        self.scale = self.head_dim ** -0.5

        self.q_proj = MixedLinear(hidden_dim, num_ns_tokens, bias=True)
        self.k_proj = MixedLinear(hidden_dim, num_ns_tokens, bias=True)
        self.v_proj = MixedLinear(hidden_dim, num_ns_tokens, bias=True)
        self.out_proj = MixedLinear(hidden_dim, num_ns_tokens, bias=True)
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
        num_s: int,
        attn_mask: torch.Tensor,
        query_indices: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        q_full = self._split_heads(self.q_proj(x, num_s))
        k_full = self._split_heads(self.k_proj(x, num_s))
        v_full = self._split_heads(self.v_proj(x, num_s))

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
        num_s_out = num_s if query_indices is None else int((query_indices < num_s).sum().item())
        return self.out_proj(out, num_s_out)


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
        num_ns_tokens: int,
        ffn_expansion: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.attn_norm = RMSNorm(hidden_dim)
        self.ffn_norm = RMSNorm(hidden_dim)
        self.attn = MixedCausalAttention(
            hidden_dim, num_heads, num_ns_tokens, dropout=dropout
        )
        self.ffn = MixedFFN(
            hidden_dim, num_ns_tokens, expansion=ffn_expansion, dropout=dropout
        )

    def forward(
        self,
        x: torch.Tensor,
        num_s: int,
        attn_mask: torch.Tensor,
        query_indices: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, int]:
        residual = x
        if query_indices is not None:
            residual = x.index_select(1, query_indices)

        h = self.attn(
            self.attn_norm(x),
            num_s=num_s,
            attn_mask=attn_mask,
            query_indices=query_indices,
            key_padding_mask=key_padding_mask,
        )
        z = residual + h

        num_s_out = num_s
        if query_indices is not None:
            num_s_out = int((query_indices < num_s).sum().item())

        out = z + self.ffn(self.ffn_norm(z), num_s_out)
        return out, num_s_out


def build_pyramid_query_indices(
    layer_idx: int,
    num_layers: int,
    initial_num_s: int,
    current_num_s: int,
    num_ns: int,
    min_s_queries: int,
) -> torch.Tensor:
    """
    Pyramid schedule (paper Sec. 3.4): keep the tail of S-tokens as queries;
    NS-tokens always remain in the query set.
    """
    if num_layers <= 1:
        s_queries = current_num_s
    else:
        keep_ratio = 1.0 - (layer_idx + 1) / num_layers
        s_queries = max(min_s_queries, int(math.ceil(initial_num_s * keep_ratio)))
        s_queries = min(s_queries, current_num_s)
    s_idx = torch.arange(current_num_s - s_queries, current_num_s)
    ns_idx = torch.arange(current_num_s, current_num_s + num_ns)
    return torch.cat([s_idx, ns_idx])


class OneTransStack(nn.Module):
    """Stack of OneTrans blocks with optional pyramid token pruning."""

    def __init__(
        self,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        num_ns_tokens: int,
        ffn_expansion: int = 4,
        dropout: float = 0.0,
        use_pyramid: bool = False,
        pyramid_min_s_queries: int = 1,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.num_ns = num_ns_tokens
        self.use_pyramid = use_pyramid
        self.pyramid_min_s_queries = pyramid_min_s_queries
        self.blocks = nn.ModuleList([
            OneTransBlock(
                hidden_dim,
                num_heads,
                num_ns_tokens,
                ffn_expansion=ffn_expansion,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

    def forward(
        self,
        x: torch.Tensor,
        num_s: int,
        s_valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        num_ns = self.num_ns
        key_padding_mask = torch.cat([
            ~s_valid_mask,
            torch.zeros(
                x.size(0), num_ns, dtype=torch.bool, device=x.device
            ),
        ], dim=1)

        all_padding = key_padding_mask.all(dim=1)
        if all_padding.any():
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[all_padding, 0] = False

        initial_num_s = num_s
        attn_mask = None
        for layer_idx, block in enumerate(self.blocks):
            query_indices = None
            if self.use_pyramid:
                query_indices = build_pyramid_query_indices(
                    layer_idx,
                    self.num_layers,
                    initial_num_s,
                    num_s,
                    num_ns,
                    self.pyramid_min_s_queries,
                ).to(x.device)

            x, num_s = block(
                x,
                num_s=num_s,
                attn_mask=attn_mask,
                query_indices=query_indices,
                key_padding_mask=key_padding_mask,
            )

            if query_indices is not None:
                key_padding_mask = key_padding_mask.index_select(1, query_indices)

        return x
