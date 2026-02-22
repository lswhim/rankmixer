"""
TokenMixer-Large: Scaling Up Large Ranking Models in Industrial Recommenders
(ByteDance, arXiv:2602.06563)

在 RankMixer (arXiv:2507.15551) 基础上的系统性改进:
1. Mixing & Reverting: 先混合再还原, 保证残差连接语义对齐
2. Per-token SwiGLU: 替换 GELU FFN, 使用 gate/up/down 三路结构
3. RMSNorm + Pre-Norm: 替换 LayerNorm + Post-Norm, 提升深层稳定性
4. Down-matrix Small Init: FC_down 初始化方差=0.01, 初期近似恒等映射
5. Inter-layer Residual + Auxiliary Loss: 跨层残差 + 辅助损失, 支撑深层训练
6. Sparse-Pertoken MoE: Top-k softmax routing + shared expert + gate scaling α
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
import math

from .rankmixer import MultiHeadTokenMixing, FeatureTokenizer


# ============================================================
# 1. RMSNorm (替换 LayerNorm)
# ============================================================

class RMSNorm(nn.Module):
    """
    论文选择 RMSNorm 替代 LayerNorm, 去除均值中心化, 减少计算量。
    RMS(x) = sqrt(mean(x^2) + eps)
    RMSNorm(x) = x / RMS(x) * gamma
    """
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x / rms * self.gamma


# ============================================================
# 2. Per-token SwiGLU (替换 Per-token FFN with GELU)
# ============================================================

class PerTokenSwiGLU(nn.Module):
    """
    论文 Section 3.3: Per-token SwiGLU

    pSwiGLU(x_t) = FC_down^t( Swish(FC_gate^t(x_t)) ⊙ FC_up^t(x_t) )

    其中 FC_i^t(x) = W_i^t @ x_t + b_i^t, 每个 token 独立参数。
    W_gate, W_up ∈ R^{D x nD}, W_down ∈ R^{nD x D}

    Down-matrix Small Init: FC_down 用 Xavier Uniform stddev=0.01 初始化,
    使得训练初期 F(x) ≈ 0, 残差分支 F(x)+x ≈ x (近似恒等映射)。
    """
    def __init__(self, num_tokens: int, hidden_dim: int, expansion: int = 4,
                 small_init: bool = True):
        super().__init__()
        self.T = num_tokens
        self.D = hidden_dim
        self.nD = hidden_dim * expansion

        # W_gate: [T, D, nD], b_gate: [T, nD]
        self.W_gate = nn.Parameter(torch.empty(num_tokens, hidden_dim, self.nD))
        self.b_gate = nn.Parameter(torch.zeros(num_tokens, self.nD))

        # W_up: [T, D, nD], b_up: [T, nD]
        self.W_up = nn.Parameter(torch.empty(num_tokens, hidden_dim, self.nD))
        self.b_up = nn.Parameter(torch.zeros(num_tokens, self.nD))

        # W_down: [T, nD, D], b_down: [T, D]
        self.W_down = nn.Parameter(torch.empty(num_tokens, self.nD, hidden_dim))
        self.b_down = nn.Parameter(torch.zeros(num_tokens, hidden_dim))

        # 初始化
        for t in range(num_tokens):
            nn.init.kaiming_uniform_(self.W_gate[t], a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.W_up[t], a=math.sqrt(5))
            if small_init:
                # Down-matrix Small Init: stddev=0.01
                nn.init.uniform_(self.W_down[t], -0.01, 0.01)
            else:
                nn.init.kaiming_uniform_(self.W_down[t], a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args: x [batch, T, D]
        Returns: [batch, T, D]
        """
        # gate: [batch, T, nD]
        gate = torch.einsum('btd,tdh->bth', x, self.W_gate) + self.b_gate
        gate = F.silu(gate)  # Swish = SiLU

        # up: [batch, T, nD]
        up = torch.einsum('btd,tdh->bth', x, self.W_up) + self.b_up

        # element-wise multiply
        h = gate * up  # [batch, T, nD]

        # down: [batch, T, D]
        out = torch.einsum('bth,thd->btd', h, self.W_down) + self.b_down

        return out


# ============================================================
# 3. Mixing & Reverting 操作
# ============================================================

# Mixing 操作复用 RankMixer 的 MultiHeadTokenMixing (逻辑完全相同)
MixingOperation = MultiHeadTokenMixing


class RevertingOperation(nn.Module):
    """
    无参数的 Reverting 操作 (Mixing 的逆操作)。

    将 H^next ∈ R^{H x (T*D/H)} 还原回 X^revert ∈ R^{T x D}:
    1. reshape: [H, T, D/H]
    2. 转置: [T, H, D/H]
    3. 展平: [T, D]
    """
    def __init__(self, num_tokens: int, hidden_dim: int):
        super().__init__()
        self.T = num_tokens
        self.H = num_tokens
        self.D = hidden_dim
        self.head_dim = hidden_dim // num_tokens

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[batch, H, T*D/H] -> [batch, T, D]"""
        B = x.size(0)
        # [B, H, T, head_dim]
        x = x.view(B, self.H, self.T, self.head_dim)
        # transpose back -> [B, T, H, head_dim]
        x = x.transpose(1, 2).contiguous()
        # flatten -> [B, T, D]
        x = x.view(B, self.T, self.D)
        return x


# ============================================================
# 4. TokenMixer-Large Block (Mixing & Reverting + Per-token SwiGLU)
# ============================================================

class TokenMixerLargeBlock(nn.Module):
    """
    论文 Section 3.3: TokenMixer-Large Block

    使用 Pre-Norm (而非 RankMixer 的 Post-Norm), 保证深层稳定性。

    Step 1 - Mixing:
        H = Mix(X)                          # 无参数 reshape+transpose
        H_next = X + pSwiGLU(Norm(H))       # Pre-Norm + 残差 (注: 残差加的是 H 而非 X)

    Step 2 - Reverting:
        X_revert = Revert(H_next)           # 无参数逆操作
        X_next = X + pSwiGLU(Norm(X_revert)) # Pre-Norm + 残差 (跳过整个 block)

    注意: 最外层残差连接的是原始输入 X, 实现了 "跨 block 残差"。
    """
    def __init__(self, num_tokens: int, hidden_dim: int,
                 ffn_expansion: int = 4, small_init: bool = True):
        super().__init__()
        self.mix = MixingOperation(num_tokens, hidden_dim)
        self.revert = RevertingOperation(num_tokens, hidden_dim)

        # Mixing 阶段的 Norm + pSwiGLU
        self.norm_mix = RMSNorm(hidden_dim)
        self.pswiglu_mix = PerTokenSwiGLU(
            num_tokens, hidden_dim, ffn_expansion, small_init
        )

        # Reverting 阶段的 Norm + pSwiGLU
        self.norm_revert = RMSNorm(hidden_dim)
        self.pswiglu_revert = PerTokenSwiGLU(
            num_tokens, hidden_dim, ffn_expansion, small_init
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args: x [batch, T, D]
        Returns: [batch, T, D]
        """
        # Step 1: Mixing
        h = self.mix(x)                              # [B, T, D]
        h = h + self.pswiglu_mix(self.norm_mix(h))   # Pre-Norm + 残差

        # Step 2: Reverting
        x_revert = self.revert(h)                    # [B, T, D]
        x_next = x + self.pswiglu_revert(self.norm_revert(x_revert))  # 跨 block 残差

        return x_next


# ============================================================
# 5. Sparse-Pertoken MoE (替换 RankMixer 的 ReLU-DTSI MoE)
# ============================================================

class TopKRouter(nn.Module):
    """
    TokenMixer-Large 的 Top-k Softmax Router。

    替换 RankMixer 的 ReLU routing, 使用标准 Top-k + Softmax:
    - 可预测的激活专家数 (固定 top_k)
    - 统一 sparse-train / sparse-infer
    """
    def __init__(self, hidden_dim: int, num_experts: int, top_k: int = 2):
        super().__init__()
        self.gate = nn.Linear(hidden_dim, num_experts)
        self.top_k = top_k
        self.num_experts = num_experts

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args: x [batch*T, D]
        Returns:
            gate_values: [batch*T, num_experts] (只有 top_k 个非零)
            indices: [batch*T, top_k]
        """
        logits = self.gate(x)  # [N, E]
        top_vals, top_idx = torch.topk(logits, self.top_k, dim=-1)  # [N, k]
        top_vals = F.softmax(top_vals, dim=-1)  # softmax over top-k

        # scatter 到 full expert 维度 (显式对齐 dtype, 兼容 fp16 autocast)
        gate_out = torch.zeros_like(logits).to(top_vals.dtype)
        gate_out.scatter_(1, top_idx, top_vals)

        return gate_out, top_idx


class SparsePerTokenMoE(nn.Module):
    """
    论文 Section 3.4: Sparse-Pertoken MoE

    S-P MoE(x) = α * Σ_{j in top-k} g_j(x) * Expert_j(x) + SharedExpert(x)

    改进点 vs RankMixer MoE:
    1. Top-k Softmax routing (替换 ReLU routing)
    2. Shared Expert: 每个 token 有一个始终激活的共享专家
    3. Gate Value Scaling α: 放大稀疏路由梯度, α ≈ E/k
    4. Sparse-Train + Sparse-Infer: 训练和推理均稀疏 (替换 DTSI)
    5. Down-matrix Small Init
    """
    def __init__(
        self,
        num_tokens: int,
        hidden_dim: int,
        num_experts: int = 8,
        top_k: int = 2,
        ffn_expansion: int = 4,
        gate_scale: float = 0.0,  # 0 = auto (E/k)
        small_init: bool = True,
    ):
        super().__init__()
        self.T = num_tokens
        self.D = hidden_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.nD = hidden_dim * ffn_expansion

        # Gate value scaling α
        self.alpha = gate_scale if gate_scale > 0 else float(num_experts) / top_k

        # Router
        self.router = TopKRouter(hidden_dim, num_experts, top_k)

        # Per-token Routed Experts: 每个 token 有 num_experts 个专家
        # 每个专家用 SwiGLU (gate/up/down)
        # 每个专家的隐藏维度 = nD / num_experts (保持总参数量不变)
        expert_nD = self.nD // num_experts

        # W_gate: [T, E, D, expert_nD]
        self.W_gate_e = nn.Parameter(torch.empty(num_tokens, num_experts, hidden_dim, expert_nD))
        self.b_gate_e = nn.Parameter(torch.zeros(num_tokens, num_experts, expert_nD))
        self.W_up_e = nn.Parameter(torch.empty(num_tokens, num_experts, hidden_dim, expert_nD))
        self.b_up_e = nn.Parameter(torch.zeros(num_tokens, num_experts, expert_nD))
        self.W_down_e = nn.Parameter(torch.empty(num_tokens, num_experts, expert_nD, hidden_dim))
        self.b_down_e = nn.Parameter(torch.zeros(num_tokens, num_experts, hidden_dim))

        # Per-token Shared Expert (始终激活, 不经过 router)
        self.shared_expert = PerTokenSwiGLU(
            num_tokens, hidden_dim, ffn_expansion, small_init
        )

        # 初始化
        for t in range(num_tokens):
            for e in range(num_experts):
                nn.init.kaiming_uniform_(self.W_gate_e[t, e], a=math.sqrt(5))
                nn.init.kaiming_uniform_(self.W_up_e[t, e], a=math.sqrt(5))
                if small_init:
                    nn.init.uniform_(self.W_down_e[t, e], -0.01, 0.01)
                else:
                    nn.init.kaiming_uniform_(self.W_down_e[t, e], a=math.sqrt(5))

    def _routed_expert_forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:    [batch, T, D]
            gate: [batch, T, E] (sparse, 只有 top_k 非零)
        Returns: [batch, T, D]
        """
        # gate path: [B,T,D] @ [T,E,D,nD/E] -> [B,T,E,nD/E]
        g = torch.einsum('btd,tedk->btek', x, self.W_gate_e) + self.b_gate_e.unsqueeze(0)
        g = F.silu(g)

        u = torch.einsum('btd,tedk->btek', x, self.W_up_e) + self.b_up_e.unsqueeze(0)

        h = g * u  # [B, T, E, nD/E]

        out = torch.einsum('btek,tekd->bted', h, self.W_down_e) + self.b_down_e.unsqueeze(0)

        # 加权求和 (gate 已经是 sparse 的)
        out = torch.einsum('bte,bted->btd', gate, out)  # [B, T, D]
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args: x [batch, T, D]
        Returns: [batch, T, D]
        """
        B, T, D = x.shape

        # Router
        gate, _ = self.router(x.reshape(-1, D))  # [B*T, E]
        gate = gate.view(B, T, self.num_experts)  # [B, T, E]

        # Routed experts with gate scaling
        routed_out = self.alpha * self._routed_expert_forward(x, gate)

        # Shared expert (always active)
        shared_out = self.shared_expert(x)

        return routed_out + shared_out


# ============================================================
# 6. TokenMixer-Large MoE Block
# ============================================================

class TokenMixerLargeMoEBlock(nn.Module):
    """
    MoE 版本的 TokenMixer-Large Block。

    与 Dense 版本的区别: pSwiGLU 替换为 Sparse-Pertoken MoE。
    """
    def __init__(self, num_tokens: int, hidden_dim: int,
                 num_experts: int = 8, top_k: int = 2,
                 ffn_expansion: int = 4, gate_scale: float = 0.0,
                 small_init: bool = True):
        super().__init__()
        self.mix = MixingOperation(num_tokens, hidden_dim)
        self.revert = RevertingOperation(num_tokens, hidden_dim)

        # Mixing 阶段
        self.norm_mix = RMSNorm(hidden_dim)
        self.moe_mix = SparsePerTokenMoE(
            num_tokens, hidden_dim, num_experts, top_k,
            ffn_expansion, gate_scale, small_init
        )

        # Reverting 阶段
        self.norm_revert = RMSNorm(hidden_dim)
        self.moe_revert = SparsePerTokenMoE(
            num_tokens, hidden_dim, num_experts, top_k,
            ffn_expansion, gate_scale, small_init
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.mix(x)
        h = h + self.moe_mix(self.norm_mix(h))

        x_revert = self.revert(h)
        x_next = x + self.moe_revert(self.norm_revert(x_revert))

        return x_next


# ============================================================
# 7. 完整 TokenMixer-Large 模型
# ============================================================

class TokenMixerLarge(nn.Module):
    """
    完整的 TokenMixer-Large 模型。

    在 RankMixer 基础上的改进:
    1. Mixing & Reverting Block (替换 Token Mixing + FFN)
    2. Per-token SwiGLU (替换 Per-token GELU FFN)
    3. RMSNorm + Pre-Norm (替换 LayerNorm + Post-Norm)
    4. Down-matrix Small Init
    5. Inter-layer Residual (每 inter_residual_interval 层添加跨层残差)
    6. Auxiliary Loss (中间层输出辅助预测)
    7. Sparse-Pertoken MoE (可选, 替换 Dense pSwiGLU)
    """
    def __init__(
        self,
        feature_dims: List[int],
        embed_dims: List[int],
        chunk_size: int = 64,
        hidden_dim: int = 256,
        num_layers: int = 4,
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

        # Global Token (TokenMixer-Large 新增)
        self.global_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        nn.init.normal_(self.global_token, std=0.02)
        self.T_with_global = T + 1  # 加上 global token

        # === Blocks ===
        if use_moe:
            self.blocks = nn.ModuleList([
                TokenMixerLargeMoEBlock(
                    self.T_with_global, hidden_dim, num_experts, top_k,
                    ffn_expansion, gate_scale, small_init
                )
                for _ in range(num_layers)
            ])
        else:
            self.blocks = nn.ModuleList([
                TokenMixerLargeBlock(
                    self.T_with_global, hidden_dim, ffn_expansion, small_init
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
        """
        Args:
            feature_ids: List of [batch] tensors
        Returns:
            logits: [batch, num_classes]
            aux_loss: 辅助损失 (需要在外部与 label 计算 BCE)
        """
        # 1. Token 化 (复用 FeatureTokenizer)
        tokens = self.tokenizer(feature_ids)  # [B, T, D]
        B = tokens.size(0)

        # 2. 添加 Global Token
        global_t = self.global_token.expand(B, -1, -1)  # [B, 1, D]
        x = torch.cat([global_t, tokens], dim=1)  # [B, T+1, D]

        # 3. 通过 L 层 Block, 带 inter-residual
        aux_logits_list = []
        residual_cache = x  # 用于 inter-layer residual

        for i, block in enumerate(self.blocks):
            x = block(x)

            # Inter-layer residual: 每 interval 层加一次跨层残差
            # 但不在最后一层 (避免低层原始信息干扰高层抽象)
            if (i + 1) % self.inter_residual_interval == 0 and i < self.num_layers - 1:
                x = x + residual_cache
                residual_cache = x

                # Auxiliary Loss: 在跨层残差点计算辅助预测
                if str(i) in self.aux_heads:
                    aux_logit = self.aux_heads[str(i)](
                        x[:, 0, :]  # 用 global token 做预测
                    )
                    aux_logits_list.append(aux_logit)

        # 4. 主输出: 用 global token
        main_logits = self.output_head(x[:, 0, :])  # [B, num_classes]

        # 5. 聚合辅助 logits
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
    # 总维度 = 552

    feature_ids = [
        torch.randint(0, dim, (batch_size,))
        for dim in feature_dims
    ]

    print("=" * 60)
    print("TokenMixer-Large Architecture Test")
    print("=" * 60)

    # --- Dense 版本 ---
    print("\n--- Dense Version ---")
    model_dense = TokenMixerLarge(
        feature_dims=feature_dims,
        embed_dims=embed_dims,
        chunk_size=69,        # ceil(552/69)=8, T=8
        hidden_dim=288,       # D=288, D/(T+1)=288/9=32
        num_layers=4,
        ffn_expansion=4,
        use_moe=False,
        small_init=True,
        inter_residual_interval=2,
        aux_loss_weight=0.1,
    )

    T = model_dense.T
    print(f"Feature tokens (T): {T}, +1 global = {T+1}")
    print(f"Hidden dim (D): 288, D/(T+1): {288 // (T+1)}")

    logits, aux_logits = model_dense(feature_ids)
    print(f"Main logits: {logits.shape}")
    print(f"Aux logits: {aux_logits.shape}")

    total = sum(p.numel() for p in model_dense.parameters())
    print(f"Total params: {total / 1e6:.2f}M")

    # --- MoE 版本 ---
    print("\n--- MoE Version ---")
    model_moe = TokenMixerLarge(
        feature_dims=feature_dims,
        embed_dims=embed_dims,
        chunk_size=69,
        hidden_dim=288,
        num_layers=4,
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
    print("改进点 vs RankMixer:")
    print("  1. Mixing & Reverting (语义对齐的残差连接)")
    print("  2. Per-token SwiGLU (替换 GELU FFN)")
    print("  3. RMSNorm + Pre-Norm (替换 LayerNorm + Post-Norm)")
    print("  4. Down-matrix Small Init (训练初期近似恒等)")
    print("  5. Global Token (类似 BERT [CLS])")
    print("  6. Inter-layer Residual + Auxiliary Loss")
    print("  7. Sparse-Pertoken MoE (Top-k + Shared Expert + α scaling)")
    print("=" * 60)


if __name__ == "__main__":
    test()
