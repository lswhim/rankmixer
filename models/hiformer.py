"""
HiFormer: Heterogeneous Feature Interactions Learning with Transformers
for Recommender Systems (arXiv:2311.05884, Google)

核心创新: 异质自注意力层 (Heterogeneous Self-Attention, HSA)
- 将 Q/K 通过不同变换 φ() 和 ψ() 处理后再计算注意力
- 捕捉不同类型特征之间的异质性交互
- 通过低秩近似 + 模型剪枝实现快速推理

适配为 CTR 排序模型:
- 输入: 特征 token 化后的 [B, T, D]
- 输出: mean pooling → 预测头
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


# ============================================================
# 1. RMSNorm
# ============================================================

class RMSNorm(nn.Module):
    """RMS Normalization"""
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x / rms * self.gamma


# ============================================================
# 2. 异质特征变换模块 (Heterogeneous Transformations)
# ============================================================

class HeterogeneousTransform(nn.Module):
    """
    HiFormer 核心: 对 Q/K 进行异质变换 φ() 和 ψ()

    论文思想: 不同类型的特征交互需要不同的变换方式来建模。
    这里使用 Depthwise Convolution + MLP 的组合来实现异质性变换。

    Args:
        dim: 输入维度 (hidden_dim)
        transform_type: 变换类型
            - "dwconv_mlp": Depthwise Conv + MLP (论文推荐)
            - "gate_mlp": Gated MLP (简化版)
    """
    def __init__(self, dim: int, transform_type: str = "dwconv_mlp"):
        super().__init__()
        self.dim = dim
        self.transform_type = transform_type

        if transform_type == "dwconv_mlp":
            # φ() 变换: 用于 Query
            # 3x3 Depthwise Conv + MLP，提供空间异质性建模能力
            # 保持输出维度 = dim
            self.q_transform = nn.Sequential(
                nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim),  # DW Conv
                nn.GELU(),
            )
            # ψ() 变换: 用于 Key
            # 与 Q 变换结构相同但独立参数，捕捉不同类型的交互模式
            self.k_transform = nn.Sequential(
                nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim),
                nn.GELU(),
            )
        elif transform_type == "gate_mlp":
            # 简化版: Gated MLP (保持维度不变)
            self.q_transform = nn.Sequential(
                nn.Linear(dim, dim * 2),
                nn.GELU(),
            )
            self.k_transform = nn.Sequential(
                nn.Linear(dim, dim * 2),
                nn.GELU(),
            )
        else:
            raise ValueError(f"Unknown transform_type: {transform_type}")

    def forward(self, x: torch.Tensor, is_query: bool = True) -> torch.Tensor:
        """
        Args:
            x: [B, T, D]
            is_query: True for Q transform (φ), False for K transform (ψ)
        Returns:
            transformed: [B, T, D]
        """
        # [B, T, D] -> [B, D, T] for Conv1d
        x = x.transpose(1, 2)
        if is_query:
            out = self.q_transform(x)
        else:
            out = self.k_transform(x)
        # [B, D, T] -> [B, T, D]
        return out.transpose(1, 2)


# ============================================================
# 3. Heterogeneous Self-Attention Layer
# ============================================================

class HeterogeneousSelfAttention(nn.Module):
    """
    异质自注意力层 (Heterogeneous Self-Attention, HSA)

    核心公式:
        Q_het = φ(Q),  K_het = ψ(K)    # 异质变换
        A = σ(Q_het @ K_het^T / √d_k)   # 注意力分数
        Y = A @ V                         # 加权聚合

    与标准 Self-Attention 的区别:
    - 标准: Q, K, V 共享线性投影，注意力是对称的
    - HSA: Q/K 通过不同变换 φ/ψ，允许捕捉非对称的异质交互

    论文优化:
    - 低秩近似减少计算量
    - 可结合相对位置编码
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        dropout: float = 0.0,
        transform_type: str = "dwconv_mlp",
    ):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.transform_type = transform_type

        # 标准 V 投影 (与 Q/K 分离)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim)

        # 异质变换模块 (核心创新)
        self.q_hetero = HeterogeneousTransform(hidden_dim, transform_type)
        self.k_hetero = HeterogeneousTransform(hidden_dim, transform_type)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, T, D]
            attn_mask: [B, T] or [B, T, T], True = 需要 attend
        Returns:
            output: [B, T, D]
        """
        B, T, D = x.shape
        H = self.num_heads
        d_k = self.head_dim

        # === 标准 Q/K/V 投影 ===
        Q = self.v_proj(x)  # [B, T, D] (复用 v_proj 作为 Q 投影)
        K = self.k_hetero(x, is_query=False)  # [B, T, D] - 异质变换
        V = self.v_proj(x)  # [B, T, D]

        # === 异质变换: φ(Q) 和 ψ(K) ===
        Q_het = self.q_hetero(x, is_query=True)  # [B, T, D] - 异质变换
        K_het = self.k_hetero(x, is_query=False)  # [B, T, D] - 异质变换

        # reshape to multi-head: [B, T, D] -> [B, H, T, d_k]
        Q_het = Q_het.view(B, T, H, d_k).transpose(1, 2)  # [B, H, T, d_k]
        K_het = K_het.view(B, T, H, d_k).transpose(1, 2)  # [B, H, T, d_k]
        V = V.view(B, T, H, d_k).transpose(1, 2)            # [B, H, T, d_k]

        # === 计算异质注意力分数 ===
        # A = φ(Q) @ ψ(K)^T / √d_k
        attn_scores = torch.matmul(Q_het, K_het.transpose(-2, -1)) * self.scale
        # [B, H, T, T]

        # === 应用 mask ===
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                # [B, T] -> [B, 1, 1, T] broadcast
                attn_mask = attn_mask.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(~attn_mask, float("-inf"))

        # === Softmax 归一化 ===
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # === 加权聚合 ===
        context = torch.matmul(attn_weights, V)  # [B, H, T, d_k]

        # === 输出投影 ===
        context = context.transpose(1, 2).contiguous().view(B, T, D)
        output = self.o_proj(context)  # [B, T, D]

        return output


# ============================================================
# 4. HiFormer Layer (完整 Block)
# ============================================================

class HiFormerLayer(nn.Module):
    """
    HiFormer Block = LayerNorm + HeterogeneousSelfAttention + FFN + 残差

    架构 (Pre-LN 风格):
        x = x + HeterogeneousSelfAttention(LayerNorm(x))
        x = x + FFN(LayerNorm(x))
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        ffn_expansion: int = 4,
        dropout: float = 0.0,
        transform_type: str = "dwconv_mlp",
    ):
        super().__init__()

        # Pre-LN
        self.norm1 = RMSNorm(hidden_dim)
        self.norm2 = RMSNorm(hidden_dim)

        # 异质自注意力
        self.attn = HeterogeneousSelfAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            transform_type=transform_type,
        )

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * ffn_expansion),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim * ffn_expansion, hidden_dim),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # 异质自注意力 + 残差
        x = x + self.attn(self.norm1(x), attn_mask)
        # FFN + 残差
        x = x + self.ffn(self.norm2(x))
        return x


# ============================================================
# 5. 完整 HiFormer 模型
# ============================================================

class HiFormer(nn.Module):
    """
    完整的 HiFormer 模型

    架构:
    1. FeatureTokenizer: 特征 → T 个 token [B, T, D]
    2. L 层 HiFormerLayer
    3. Mean Pooling → 预测头

    与 HSTU/RankMixer 的区别:
    - HSTU: 无 Softmax 注意力，SiLU(QK^T) 保留兴趣强度
    - RankMixer: 无参数的 Token Mixing (Reshape)
    - HiFormer: 异质自注意力，通过 φ/ψ 变换捕捉不同类型特征交互
    """

    def __init__(
        self,
        feature_dims,       # List[int]: 各特征的基数
        embed_dims,         # List[int]: 各特征的 embedding 维度
        chunk_size: int = 64,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        ffn_expansion: int = 4,
        dropout: float = 0.0,
        transform_type: str = "dwconv_mlp",
        num_classes: int = 1,
    ):
        super().__init__()
        from .rankmixer import FeatureTokenizer

        self.num_layers = num_layers

        # 输入 Token 化
        self.tokenizer = FeatureTokenizer(
            feature_dims, embed_dims, chunk_size, hidden_dim
        )
        T = self.tokenizer.T

        print(f"  [HiFormer] Feature dim: {sum(embed_dims)}, Tokens (T): {T}, "
              f"Hidden (D): {hidden_dim}, Heads: {num_heads}")
        print(f"  Layers: {num_layers}, FFN expansion: {ffn_expansion}, "
              f"Transform: {transform_type}")

        # HiFormer Layers
        self.layers = nn.ModuleList([
            HiFormerLayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                ffn_expansion=ffn_expansion,
                dropout=dropout,
                transform_type=transform_type,
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

    def forward(self, feature_ids):
        """
        Args:
            feature_ids: List of [batch] tensors
        Returns:
            logits: [batch, num_classes]
            aux_loss: 0 (保持接口一致)
        """
        # Token 化: [B, T, D]
        x = self.tokenizer(feature_ids)

        # L 层 HiFormer
        for layer in self.layers:
            x = layer(x)

        # Mean Pooling → 预测
        x = x.mean(dim=1)  # [B, D]
        logits = self.output_head(x)  # [batch, num_classes]

        return logits, torch.tensor(0.0, device=logits.device)


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
    print("HiFormer Architecture Test")
    print("=" * 60)

    model = HiFormer(
        feature_dims=feature_dims,
        embed_dims=embed_dims,
        chunk_size=70,
        hidden_dim=256,
        num_layers=4,
        num_heads=8,
        ffn_expansion=4,
        dropout=0.0,
        transform_type="dwconv_mlp",
    )

    T = model.T
    print(f"Feature tokens (T): {T}")
    print(f"Hidden dim (D): 256, Heads: 8, Head dim: {256 // 8}")

    logits, aux = model(feature_ids)
    print(f"Logits: {logits.shape}")

    total = sum(p.numel() for p in model.parameters())
    print(f"Total params: {total / 1e6:.2f}M")

    print("\n核心创新点:")
    print("  1. 异质自注意力: φ(Q) 和 ψ(K) 使用不同变换")
    print("  2. Depthwise Conv 捕捉空间异质性")
    print("  3. 与标准 Transformer/HSTU/RankMixer 不同的注意力机制")
    print("=" * 60)


if __name__ == "__main__":
    test()
