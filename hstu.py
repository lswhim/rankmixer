"""
HSTU: Hierarchical Sequential Transduction Unit
(Meta, ICML 2024, arXiv:2402.17152)

"Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers
for Generative Recommendations"

核心架构 (每层 HSTU Layer 包含三个子层):
1. Pointwise 投影层: U, V, Q, K = Split(SiLU(f1(X)))
2. Pointwise 空间聚合层: A(X)V(X) = φ2(QK^T + rab) ⊙ V  (无 Softmax!)
3. Pointwise 转换层: Y = f2(Norm(A(X)V(X)) ⊙ U)

关键设计:
- 去除 Softmax 归一化, 保留兴趣强度信号 (类似 DIN)
- 相对位置+时间编码 rab^{p,t} 作为 attention bias
- U 门控: 原始特征控制上下文信息流动
- 复用 FeatureTokenizer 做输入 token 化

适配为 CTR 排序模型:
- 输入: 特征 token 化后的 [B, T, D]
- 不使用 causal mask (排序场景所有特征同时可见)
- 输出: mean pooling → 预测头
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
import math

from rankmixer import FeatureTokenizer


# ============================================================
# 1. RMSNorm
# ============================================================

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x / rms * self.gamma


# ============================================================
# 2. Relative Attention Bias (位置 + 时间)
# ============================================================

class LearnedRelativeBias(nn.Module):
    """
    论文中 rab^{p,t}: 相对位置偏置。
    在 CTR 排序场景中只使用位置偏置 (无时间序列)。
    可学习的 per-head relative position bias, 类似 ALiBi 但可学习。
    """
    def __init__(self, num_heads: int, max_tokens: int):
        super().__init__()
        # 可学习的相对位置 bias table: [2*max_tokens-1]
        self.num_heads = num_heads
        self.max_tokens = max_tokens
        self.bias_table = nn.Parameter(
            torch.zeros(num_heads, 2 * max_tokens - 1)
        )
        nn.init.normal_(self.bias_table, std=0.02)

        # 预计算相对位置索引
        coords = torch.arange(max_tokens)
        relative_coords = coords.unsqueeze(0) - coords.unsqueeze(1)  # [T, T]
        relative_coords += max_tokens - 1  # shift to [0, 2T-2]
        self.register_buffer("relative_position_index", relative_coords)

    def forward(self, seq_len: int) -> torch.Tensor:
        """
        Returns: [num_heads, seq_len, seq_len]
        """
        idx = self.relative_position_index[:seq_len, :seq_len]  # [T, T]
        bias = self.bias_table[:, idx.reshape(-1)].view(
            self.num_heads, seq_len, seq_len
        )
        return bias


# ============================================================
# 3. HSTU Layer
# ============================================================

class HSTULayer(nn.Module):
    """
    论文核心: HSTU Layer

    三个子层:
    1. Pointwise 投影: U, V, Q, K = Split(SiLU(f1(X)))
       - f1 是一个线性投影, 输出维度 = 2D + 2*head_dim*num_heads
       - U, V ∈ R^D 用于门控和 value
       - Q, K ∈ R^{head_dim * num_heads} 用于注意力

    2. Pointwise 空间聚合 (无 Softmax 注意力):
       attn_weights = φ2(QK^T / sqrt(d_k) + rab)   # φ2 = SiLU
       context = attn_weights ⊙ V                    # element-wise (广播)

    3. Pointwise 转换 (门控 + 投影):
       Y = f2(Norm(context) ⊙ U)
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        max_tokens: int = 64,
        ffn_expansion: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        assert hidden_dim % num_heads == 0

        # === Sub-layer 1: Pointwise 投影 ===
        # 输出: U(D) + V(D) + Q(D) + K(D) = 4D
        self.f1 = nn.Linear(hidden_dim, 4 * hidden_dim)

        # === Sub-layer 2: 相对位置偏置 ===
        self.rel_bias = LearnedRelativeBias(num_heads, max_tokens)

        # === Sub-layer 3: Pointwise 转换 ===
        self.norm = RMSNorm(hidden_dim)
        self.f2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * ffn_expansion),
            nn.SiLU(),
            nn.Linear(hidden_dim * ffn_expansion, hidden_dim),
        )

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # 缩放因子
        self.scale = self.head_dim ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args: x [batch, T, D]
        Returns: [batch, T, D]
        """
        B, T, D = x.shape
        H = self.num_heads
        d_k = self.head_dim

        # === Sub-layer 1: Pointwise 投影 + SiLU ===
        projected = F.silu(self.f1(x))  # [B, T, 4D]
        U, V, Q, K = projected.split(D, dim=-1)  # 各 [B, T, D]

        # reshape Q, K 为 multi-head: [B, H, T, d_k]
        Q = Q.view(B, T, H, d_k).transpose(1, 2)  # [B, H, T, d_k]
        K = K.view(B, T, H, d_k).transpose(1, 2)  # [B, H, T, d_k]
        V_heads = V.view(B, T, H, d_k).transpose(1, 2)  # [B, H, T, d_k]

        # === Sub-layer 2: Pointwise 空间聚合 (无 Softmax) ===
        # QK^T / sqrt(d_k): [B, H, T, T]
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        # 加入 relative position bias: [H, T, T]
        rel_bias = self.rel_bias(T)  # [H, T, T]
        attn_scores = attn_scores + rel_bias.unsqueeze(0)

        # φ2 = SiLU (替代 Softmax!, 保留兴趣强度)
        attn_weights = F.silu(attn_scores)  # [B, H, T, T]
        attn_weights = self.dropout(attn_weights)

        # 加权聚合: [B, H, T, T] @ [B, H, T, d_k] -> [B, H, T, d_k]
        context = torch.matmul(attn_weights, V_heads)  # [B, H, T, d_k]

        # reshape 回 [B, T, D]
        context = context.transpose(1, 2).contiguous().view(B, T, D)

        # === Sub-layer 3: Pointwise 转换 (门控) ===
        # Y = f2(Norm(context) ⊙ U)
        gated = self.norm(context) * U  # [B, T, D]
        output = self.f2(gated)  # [B, T, D]

        # 残差连接
        return x + self.dropout(output)


# ============================================================
# 4. 完整 HSTU 模型 (适配 CTR 排序)
# ============================================================

class HSTU(nn.Module):
    """
    完整的 HSTU 模型, 适配为 CTR 排序任务。

    架构:
    1. FeatureTokenizer: 特征 → T 个 token [B, T, D]
    2. L 层 HSTULayer
    3. Mean Pooling → 预测头

    与 RankMixer/TokenMixer-Large 的对比:
    - Token 间信息交互: 无 Softmax 注意力 (vs 无参数 reshape mixing)
    - FFN: 共享参数 (vs Per-token 独立参数)
    - 门控: U 门控机制 (vs 无门控 / SwiGLU)
    - 位置编码: 可学习的相对位置偏置 (vs 无位置编码)
    """

    def __init__(
        self,
        feature_dims: List[int],
        embed_dims: List[int],
        chunk_size: int = 64,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        ffn_expansion: int = 4,
        dropout: float = 0.0,
        num_classes: int = 1,
    ):
        super().__init__()
        self.num_layers = num_layers

        # 输入 Token 化 (复用 FeatureTokenizer)
        self.tokenizer = FeatureTokenizer(
            feature_dims, embed_dims, chunk_size, hidden_dim
        )
        T = self.tokenizer.T
        max_tokens = T + 16  # 留一些余量

        # HSTU Layers
        self.layers = nn.ModuleList([
            HSTULayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                max_tokens=max_tokens,
                ffn_expansion=ffn_expansion,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        # 输出头
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
            aux_loss: placeholder (0), 保持接口一致
        """
        # Token 化: [B, T, D]
        x = self.tokenizer(feature_ids)

        # L 层 HSTU
        for layer in self.layers:
            x = layer(x)

        # Mean Pooling → 预测
        x = x.mean(dim=1)  # [B, D]
        logits = self.output_head(x)  # [B, num_classes]

        return logits, torch.tensor(0.0, device=logits.device)


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
    print("HSTU Architecture Test")
    print("=" * 60)

    model = HSTU(
        feature_dims=feature_dims,
        embed_dims=embed_dims,
        chunk_size=70,        # ceil(552/70)=8, T=8
        hidden_dim=256,       # D=256, 必须被 num_heads 整除
        num_layers=4,
        num_heads=8,
        ffn_expansion=4,
        dropout=0.0,
    )

    T = model.T
    print(f"Feature tokens (T): {T}")
    print(f"Hidden dim (D): 256, Heads: 8, Head dim: {256 // 8}")

    logits, aux = model(feature_ids)
    print(f"Logits: {logits.shape}")

    total = sum(p.numel() for p in model.parameters())
    print(f"Total params: {total / 1e6:.2f}M")

    print("\n改进点 vs standard Transformer:")
    print("  1. 无 Softmax 注意力 (SiLU 替代, 保留兴趣强度)")
    print("  2. U 门控机制 (原始特征控制上下文信息流动)")
    print("  3. 可学习的相对位置偏置")
    print("  4. SiLU 非线性投影 (Q/K/V/U 共用)")
    print("=" * 60)


if __name__ == "__main__":
    test()
