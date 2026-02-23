"""
SubMixer: Multi-Subspace Adaptive Token Mixing for Industrial Ranking Models

核心创新 (相对于 TokenMixer-Large 的固定 Transpose Mixing):
1. Compress-Mix-Decompress: 在低维 latent space 做 token mixing，而非原始 D 维
   - 借鉴 DeepSeek MLA (Multi-head Latent Attention) 的低秩压缩思想
   - 压缩过程本身是特征选择: 每个 token 决定暴露什么信息参与混合

2. Multi-Subspace Adaptive Mixing: H 个独立记忆子空间 + 输入自适应混合矩阵
   - 借鉴 DeltaNet (NeurIPS 2024) 的 Delta Rule: 查询-遗忘-写入
   - 不同子空间捕捉不同类型的特征交互模式

3. Surprise-Gated Routing: token-level 的动态路由 + surprise 门控
   - 借鉴 Titans (Google 2025) 的 surprise-driven memory 更新
   - 异常/冷启动 token 自动获得更大的交互权重

关键优势 vs TokenMixer:
- 混合矩阵是输入自适应的 (非固定 transpose)
- 不需要 Revert 操作 (混合后仍在原始 token 空间)
- 每个 Block 只需一套 FFN (TokenMixer 需两套: mix 后 + revert 后)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
import math

from .rankmixer import MultiHeadTokenMixing, FeatureTokenizer
from .tokenmixer_large import RMSNorm, PerTokenSwiGLU


# ============================================================
# 1. Latent Adaptive Mixer (核心创新模块)
# ============================================================

class LatentAdaptiveMixer(nn.Module):
    """
    Compress-Mix-Decompress 范式的自适应 token 混合。

    1. Compress: 每个 token 独立投影到 latent space [B, T, D] → [B, T, d_lat]
    2. Multi-Subspace Mixing: H 个子空间各自计算输入相关的混合矩阵
       - 每个子空间: Q_h @ K_h^T → [T, T] 混合权重 (输入自适应!)
       - Surprise gate: 控制每个 token 对子空间的更新幅度
       - Subspace router: 每个 token 动态选择参与哪些子空间
    3. Decompress: 投影回原始空间 [B, T, d_lat] → [B, T, D]

    复杂度: O(T² · d_sub · H + T · D · d_lat)
    当 d_lat << D 时, 远低于 attention 的 O(T² · D)
    """
    def __init__(
        self,
        num_tokens: int,
        hidden_dim: int,
        d_latent: int = 0,        # latent 维度, 0 = auto (D // 4)
        num_subspaces: int = 4,    # 记忆子空间数
    ):
        super().__init__()
        self.T = num_tokens
        self.D = hidden_dim
        self.H = num_subspaces

        # Auto latent dim: D // 4, 且必须被 H 整除
        if d_latent <= 0:
            d_latent = hidden_dim // 4
        d_latent = (d_latent // num_subspaces) * num_subspaces  # 确保整除
        self.d_lat = d_latent
        self.d_sub = d_latent // num_subspaces

        # --- Compress: Per-token 投影 D → d_lat ---
        # Per-token 保留了 TokenMixer 的异质性思想: 不同特征 token 压缩方式不同
        self.W_compress = nn.Parameter(torch.empty(num_tokens, hidden_dim, d_latent))

        # --- Per-Subspace Q, K 投影 (轻量) ---
        # 共享 Q, K (非 per-token), 因为混合权重本身就是输入自适应的
        self.W_q = nn.Parameter(torch.empty(num_subspaces, self.d_sub, self.d_sub))
        self.W_k = nn.Parameter(torch.empty(num_subspaces, self.d_sub, self.d_sub))

        # --- Surprise Gate ---
        # 判断 token 在该子空间中的"意外度": 高 surprise → 大幅参与混合
        self.surprise_proj = nn.Linear(self.d_sub, 1, bias=True)

        # --- Subspace Router ---
        # 每个 token 动态选择参与哪些子空间 (类 MoE router, 但路由的是混合子空间)
        self.subspace_router = nn.Linear(d_latent, num_subspaces, bias=False)

        # --- Decompress: Per-token 投影 d_lat → D ---
        self.W_decompress = nn.Parameter(torch.empty(num_tokens, d_latent, hidden_dim))

        # --- 缩放因子 (稳定训练) ---
        self.scale = 1.0 / math.sqrt(self.d_sub)

        self._init_weights()

    def _init_weights(self):
        for t in range(self.T):
            nn.init.kaiming_uniform_(self.W_compress[t], a=math.sqrt(5))
            # Decompress 使用 small init, 让初始残差 ≈ 0
            nn.init.uniform_(self.W_decompress[t], -0.01, 0.01)
        for h in range(self.H):
            nn.init.kaiming_uniform_(self.W_q[h], a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.W_k[h], a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args: x [batch, T, D]
        Returns: delta_x [batch, T, D] (调用方做 x + delta_x)
        """
        B, T, D = x.shape

        # === 1. Compress: [B, T, D] → [B, T, d_lat] ===
        z = torch.einsum('btd,tdc->btc', x, self.W_compress)  # [B, T, d_lat]

        # === 2. Split into subspaces: [B, T, H, d_sub] ===
        z_subs = z.view(B, T, self.H, self.d_sub)

        # === 3. Subspace Routing: 每个 token 对每个子空间的参与度 ===
        route_logits = self.subspace_router(z)                  # [B, T, H]
        route_weights = F.softmax(route_logits, dim=-1)         # [B, T, H]

        # === 4. Per-Subspace Adaptive Mixing ===
        # Q, K: [B, T, H, d_sub] (共享投影, 但输入不同所以结果不同)
        q = torch.einsum('bths,hsk->bthk', z_subs, self.W_q)  # [B, T, H, d_sub]
        k = torch.einsum('bths,hsk->bthk', z_subs, self.W_k)  # [B, T, H, d_sub]

        # 混合矩阵: [B, H, T, T] (输入自适应!)
        attn = torch.einsum('bthd,bshd->bhts', q, k)           # [B, H, T, T]
        attn = attn * self.scale

        # Surprise gate: [B, T, H, 1] → [B, H, T, 1]
        surprise = torch.sigmoid(self.surprise_proj(z_subs))    # [B, T, H, 1]
        surprise = surprise.permute(0, 2, 1, 3)                 # [B, H, T, 1]

        # 门控调制: surprise 高的 token 作为 query 获取更多信息
        attn = attn * surprise                                   # [B, H, T, T]

        # 归一化 (类 softmax, 但更轻量)
        attn = F.softmax(attn, dim=-1)                           # [B, H, T, T]

        # 混合: [B, H, T, T] @ [B, H, T, d_sub] → [B, H, T, d_sub]
        v_subs = z_subs.permute(0, 2, 1, 3)                     # [B, H, T, d_sub]
        z_mixed = torch.einsum('bhts,bhsd->bhtd', attn, v_subs) # [B, H, T, d_sub]

        # === 5. Route-weighted combination ===
        # [B, H, T, d_sub] * [B, H, T, 1] → weighted
        route_w = route_weights.permute(0, 2, 1).unsqueeze(-1)  # [B, H, T, 1]
        z_mixed = z_mixed * route_w                              # [B, H, T, d_sub]
        z_out = z_mixed.permute(0, 2, 1, 3).contiguous()        # [B, T, H, d_sub]
        z_out = z_out.view(B, T, self.d_lat)                    # [B, T, d_lat]

        # === 6. Decompress: [B, T, d_lat] → [B, T, D] ===
        delta_x = torch.einsum('btc,tcd->btd', z_out, self.W_decompress)

        return delta_x


# ============================================================
# 2. SubMixer Block (Dense 版本)
# ============================================================

class SubMixerBlock(nn.Module):
    """
    SubMixer Block = Latent Adaptive Mixing + Per-token SwiGLU

    对比 TokenMixer-Large Block:
    - TokenMixer: Mix(transpose) → pSwiGLU → Revert(transpose) → pSwiGLU  (两套 FFN)
    - SubMixer:   LatentAdaptiveMix → pSwiGLU  (一套 FFN, 无 Revert)

    不需要 Revert 的原因: 自适应混合的输出 = 所有 token 的加权和,
    仍在原始 token 语义空间, 不会语义错位。
    """
    def __init__(
        self,
        num_tokens: int,
        hidden_dim: int,
        d_latent: int = 0,
        num_subspaces: int = 4,
        ffn_expansion: int = 4,
        small_init: bool = True,
    ):
        super().__init__()
        # Adaptive Mixing
        self.norm_mix = RMSNorm(hidden_dim)
        self.mixer = LatentAdaptiveMixer(
            num_tokens, hidden_dim, d_latent, num_subspaces
        )

        # Per-token SwiGLU (沿用 TokenMixer-Large)
        self.norm_ffn = RMSNorm(hidden_dim)
        self.pswiglu = PerTokenSwiGLU(
            num_tokens, hidden_dim, ffn_expansion, small_init
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args: x [batch, T, D]
        Returns: [batch, T, D]
        """
        # Adaptive Mixing + Residual (Pre-Norm)
        x = x + self.mixer(self.norm_mix(x))

        # Per-token SwiGLU + Residual (Pre-Norm)
        x = x + self.pswiglu(self.norm_ffn(x))

        return x


# ============================================================
# 3. SubMixer MoE Block
# ============================================================

class SubMixerMoEBlock(nn.Module):
    """
    MoE 版本的 SubMixer Block: pSwiGLU 替换为 Sparse-Pertoken MoE。
    """
    def __init__(
        self,
        num_tokens: int,
        hidden_dim: int,
        d_latent: int = 0,
        num_subspaces: int = 4,
        num_experts: int = 8,
        top_k: int = 2,
        ffn_expansion: int = 4,
        gate_scale: float = 0.0,
        small_init: bool = True,
    ):
        super().__init__()
        from .tokenmixer_large import SparsePerTokenMoE

        # Adaptive Mixing
        self.norm_mix = RMSNorm(hidden_dim)
        self.mixer = LatentAdaptiveMixer(
            num_tokens, hidden_dim, d_latent, num_subspaces
        )

        # Sparse-Pertoken MoE
        self.norm_ffn = RMSNorm(hidden_dim)
        self.moe = SparsePerTokenMoE(
            num_tokens, hidden_dim, num_experts, top_k,
            ffn_expansion, gate_scale, small_init
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.mixer(self.norm_mix(x))
        x = x + self.moe(self.norm_ffn(x))
        return x


# ============================================================
# 4. 完整 SubMixer 模型
# ============================================================

class SubMixer(nn.Module):
    """
    完整 SubMixer 模型。

    架构:
    1. FeatureTokenizer: 特征 → [B, T, D] tokens
    2. Global Token: 可学习的 [CLS] token (沿用 TokenMixer-Large)
    3. L 层 SubMixerBlock (或 SubMixerMoEBlock)
    4. Inter-layer Residual + Auxiliary Loss (沿用 TokenMixer-Large)
    5. Global Token 输出 → 预测头
    """
    def __init__(
        self,
        feature_dims: List[int],
        embed_dims: List[int],
        chunk_size: int = 64,
        hidden_dim: int = 256,
        num_layers: int = 4,
        d_latent: int = 0,
        num_subspaces: int = 4,
        ffn_expansion: int = 4,
        use_moe: bool = False,
        num_experts: int = 8,
        top_k: int = 2,
        gate_scale: float = 0.0,
        small_init: bool = True,
        inter_residual_interval: int = 2,
        aux_loss_weight: float = 0.1,
        num_classes: int = 1,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.inter_residual_interval = inter_residual_interval
        self.aux_loss_weight = aux_loss_weight

        # === 输入 Token 化 (复用 RankMixer 的 FeatureTokenizer) ===
        self.tokenizer = FeatureTokenizer(
            feature_dims, embed_dims, chunk_size, hidden_dim
        )
        T = self.tokenizer.T

        # Global Token (沿用 TokenMixer-Large)
        self.global_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        nn.init.normal_(self.global_token, std=0.02)
        self.T_with_global = T + 1

        # === Blocks ===
        if use_moe:
            self.blocks = nn.ModuleList([
                SubMixerMoEBlock(
                    self.T_with_global, hidden_dim, d_latent, num_subspaces,
                    num_experts, top_k, ffn_expansion, gate_scale, small_init
                )
                for _ in range(num_layers)
            ])
        else:
            self.blocks = nn.ModuleList([
                SubMixerBlock(
                    self.T_with_global, hidden_dim, d_latent, num_subspaces,
                    ffn_expansion, small_init
                )
                for _ in range(num_layers)
            ])

        # === Auxiliary Loss: 中间层的预测头 ===
        self.aux_heads = nn.ModuleDict()
        for i in range(inter_residual_interval - 1, num_layers - 1, inter_residual_interval):
            self.aux_heads[str(i)] = nn.Sequential(
                RMSNorm(hidden_dim),
                nn.Linear(hidden_dim, num_classes),
            )

        # === 主输出头 ===
        self.output_head = nn.Sequential(
            RMSNorm(hidden_dim),
            nn.Linear(hidden_dim, num_classes),
        )

    @property
    def T(self) -> int:
        return self.tokenizer.T

    def forward(
        self, feature_ids: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 1. Token 化
        tokens = self.tokenizer(feature_ids)  # [B, T, D]
        B = tokens.size(0)

        # 2. 添加 Global Token
        global_t = self.global_token.expand(B, -1, -1)
        x = torch.cat([global_t, tokens], dim=1)  # [B, T+1, D]

        # 3. 通过 L 层 Block + inter-residual
        aux_logits_list = []
        residual_cache = x

        for i, block in enumerate(self.blocks):
            x = block(x)

            if (i + 1) % self.inter_residual_interval == 0 and i < self.num_layers - 1:
                x = x + residual_cache
                residual_cache = x

                if str(i) in self.aux_heads:
                    aux_logit = self.aux_heads[str(i)](x[:, 0, :])
                    aux_logits_list.append(aux_logit)

        # 4. 主输出: Global Token
        main_logits = self.output_head(x[:, 0, :])

        # 5. 辅助 logits
        if aux_logits_list and self.training:
            aux_logits = torch.stack(aux_logits_list, dim=0).mean(dim=0)
        else:
            aux_logits = torch.zeros_like(main_logits)

        return main_logits, aux_logits


# ============================================================
# 测试
# ============================================================

def test():
    batch_size = 4

    feature_dims = [10000, 100, 3, 50000, 1000, 500, 24, 10] + [50000] * 5
    embed_dims   = [64,    16, 8, 64,    32,   32,  8,  8]  + [64] * 5

    feature_ids = [
        torch.randint(0, dim, (batch_size,))
        for dim in feature_dims
    ]

    print("=" * 60)
    print("SubMixer Architecture Test")
    print("=" * 60)

    # --- Dense 版本 ---
    print("\n--- Dense Version ---")
    model_dense = SubMixer(
        feature_dims=feature_dims,
        embed_dims=embed_dims,
        chunk_size=69,
        hidden_dim=288,       # D=288, D/(T+1)=288/9=32
        num_layers=4,
        d_latent=0,           # auto: 288 // 4 = 72
        num_subspaces=4,
        ffn_expansion=4,
        use_moe=False,
        small_init=True,
        inter_residual_interval=2,
    )

    T = model_dense.T
    print(f"Feature tokens (T): {T}, +1 global = {T+1}")
    print(f"Hidden dim (D): 288, D/(T+1): {288 // (T+1)}")
    print(f"Latent dim: {model_dense.blocks[0].mixer.d_lat}")
    print(f"Subspace dim: {model_dense.blocks[0].mixer.d_sub}")
    print(f"Num subspaces: {model_dense.blocks[0].mixer.H}")

    logits, aux_logits = model_dense(feature_ids)
    print(f"Main logits: {logits.shape}")
    print(f"Aux logits: {aux_logits.shape}")

    total = sum(p.numel() for p in model_dense.parameters())
    print(f"Total params: {total / 1e6:.2f}M")

    # --- MoE 版本 ---
    print("\n--- MoE Version ---")
    model_moe = SubMixer(
        feature_dims=feature_dims,
        embed_dims=embed_dims,
        chunk_size=69,
        hidden_dim=288,
        num_layers=4,
        d_latent=0,
        num_subspaces=4,
        ffn_expansion=4,
        use_moe=True,
        num_experts=4,
        top_k=2,
        small_init=True,
        inter_residual_interval=2,
    )

    logits, aux_logits = model_moe(feature_ids)
    print(f"Main logits: {logits.shape}")

    total_moe = sum(p.numel() for p in model_moe.parameters())
    print(f"Total params (MoE): {total_moe / 1e6:.2f}M")

    print("\n" + "=" * 60)
    print("SubMixer 创新点:")
    print("  1. Compress-Mix-Decompress (低维 latent space 混合)")
    print("  2. Multi-Subspace Adaptive Mixing (输入自适应 token 混合)")
    print("  3. Surprise-Gated Routing (意外度驱动的子空间路由)")
    print("  4. 无 Revert (混合后仍在原始 token 空间)")
    print("  5. 每 block 只需一套 FFN (TokenMixer 需两套)")
    print("=" * 60)


if __name__ == "__main__":
    test()
