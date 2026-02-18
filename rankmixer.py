"""
RankMixer: Scaling Up Ranking Models in Industrial Recommenders
(ByteDance, arXiv:2507.15551)

严格按照论文公式复现核心架构：
1. 特征分组 → 拼接 → 按固定维度 d 切分成 T 个 token → 投影到 D 维
2. Multi-head Token Mixing: 将 token 切头后跨 token 重排（无参数操作, H=T）
3. Per-token FFN: 每个 token 拥有独立的 FFN 参数（不共享）
4. Sparse-MoE 扩展: ReLU routing + Dense-Train/Sparse-Infer (DTSI)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
import math


# ============================================================
# 1. 输入层: 特征分组 → 拼接 → 切分 → 投影
# ============================================================

class FeatureTokenizer(nn.Module):
    """
    论文 Section 3.2: 基于语义的特征分组与 Token 化

    步骤:
    1. 将各特征域通过各自的 Embedding 得到向量
    2. 按语义分组后拼接: e_input = [e_1; e_2; ...; e_N]
    3. 按固定维度 d 切分成 T 个片段
    4. 每个片段通过 Proj 映射到模型隐藏维度 D

    公式: x_i = Proj(e_input[d*(i-1) : d*i]),  i = 1, ..., T
    """

    def __init__(
        self,
        feature_dims: List[int],    # 各特征域的 vocab size
        embed_dims: List[int],      # 各特征域的 embedding 维度
        chunk_size: int,            # 切分维度 d
        hidden_dim: int,            # 模型隐藏维度 D
    ):
        super().__init__()
        self.chunk_size = chunk_size
        self.hidden_dim = hidden_dim

        # 各特征域独立的 Embedding
        self.embeddings = nn.ModuleList([
            nn.Embedding(vocab, edim)
            for vocab, edim in zip(feature_dims, embed_dims)
        ])

        total_embed_dim = sum(embed_dims)

        # 拼接后的总维度，需要 padding 到 chunk_size 的整数倍
        self.num_tokens = math.ceil(total_embed_dim / chunk_size)  # T
        self.padded_dim = self.num_tokens * chunk_size

        # Projection: 将每个 chunk (d 维) 映射到隐藏维度 D
        self.proj = nn.Linear(chunk_size, hidden_dim)

    @property
    def T(self) -> int:
        return self.num_tokens

    def forward(self, feature_ids: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            feature_ids: List of [batch] tensors, 每个是一个特征域的 ID
        Returns:
            [batch, T, D] 的 feature tokens
        """
        # 1. 各自嵌入
        embedded = [emb(fid) for emb, fid in zip(self.embeddings, feature_ids)]

        # 2. 拼接: [batch, total_embed_dim]
        e_input = torch.cat(embedded, dim=-1)

        # 3. Padding 到 chunk_size 的整数倍
        if e_input.size(-1) < self.padded_dim:
            pad_size = self.padded_dim - e_input.size(-1)
            e_input = F.pad(e_input, (0, pad_size))

        # 4. 切分: [batch, T, chunk_size]
        batch_size = e_input.size(0)
        tokens = e_input.view(batch_size, self.num_tokens, self.chunk_size)

        # 5. 投影: [batch, T, D]
        tokens = self.proj(tokens)

        return tokens


# ============================================================
# 2. Multi-head Token Mixing (无参数的重排操作)
# ============================================================

class MultiHeadTokenMixing(nn.Module):
    """
    论文 Section 3.3: Multi-head Token Mixing

    替代自注意力，本质是一个无参数的 reshape/transpose 操作。
    - 将每个 token x_t 切分为 H 个头: [x_t^1 || x_t^2 || ... || x_t^H]
    - 对每个头 h，将所有 token 的第 h 部分拼接: s^h = Concat(x_1^h, ..., x_T^h)
    - 论文设置 H = T，使得输出维度与输入一致（可直接残差连接）

    公式:
        [x_t^(1) || ... || x_t^(H)] = SplitHead(x_t)
        s^h = Concat(x_1^h, x_2^h, ..., x_T^h)

    当 H = T 时:
        输入: [batch, T, D]  (T tokens, 每个 D 维)
        每个 token 切成 T 个头, 每头 D/T 维
        跨 token 拼接: 每个新 token 由所有旧 token 的对应头拼接
        输出: [batch, T, D]  (维度不变)

    本质就是 reshape(B, T, T, D//T) → transpose(1,2) → reshape(B, T, D)
    """

    def __init__(self, num_tokens: int, hidden_dim: int):
        super().__init__()
        self.T = num_tokens   # H = T
        self.D = hidden_dim
        self.head_dim = hidden_dim // num_tokens

        assert hidden_dim % num_tokens == 0, \
            f"hidden_dim({hidden_dim}) must be divisible by num_tokens({num_tokens})"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, T, D]
        Returns:
            [batch, T, D] (token 间信息已混合)
        """
        batch = x.size(0)

        # [batch, T, T, head_dim]  (T tokens, 每个切成 T 个头)
        x = x.view(batch, self.T, self.T, self.head_dim)

        # 跨 token 拼接: transpose token 维和 head 维
        # [batch, T, T, head_dim] → [batch, T, T, head_dim]
        x = x.transpose(1, 2).contiguous()

        # 展平回 [batch, T, D]
        x = x.view(batch, self.T, self.D)

        return x


# ============================================================
# 3. Per-token FFN (每个 token 独立参数，不共享)
# ============================================================

class PerTokenFFN(nn.Module):
    """
    论文 Section 3.3: Per-token FFN

    与标准 Transformer 共享一个 FFN 不同，这里每个 token 有独立的 FFN 参数。
    目的: 不同 token 对应不同的语义特征子空间，独立参数可避免高频特征主导。

    公式:
        v_t = f_pffn^{t,2}(Gelu(f_pffn^{t,1}(s_t)))

        其中 f_pffn^{t,i}(x) = x @ W_pffn^{t,i} + b_pffn^{t,i}
        W^{t,1} ∈ R^{D x kD},  W^{t,2} ∈ R^{kD x D}
    """

    def __init__(self, num_tokens: int, hidden_dim: int, expansion: int = 4):
        super().__init__()
        self.T = num_tokens
        self.D = hidden_dim
        self.kD = hidden_dim * expansion

        # 为每个 token 分配独立的 W1, b1, W2, b2
        # 用一个大的参数表示，按 token 索引
        # W1: [T, D, kD],  b1: [T, kD]
        self.W1 = nn.Parameter(torch.empty(num_tokens, hidden_dim, self.kD))
        self.b1 = nn.Parameter(torch.zeros(num_tokens, self.kD))

        # W2: [T, kD, D],  b2: [T, D]
        self.W2 = nn.Parameter(torch.empty(num_tokens, self.kD, hidden_dim))
        self.b2 = nn.Parameter(torch.zeros(num_tokens, hidden_dim))

        # 初始化
        for i in range(num_tokens):
            nn.init.kaiming_uniform_(self.W1[i], a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.W2[i], a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, T, D]
        Returns:
            [batch, T, D]
        """
        # x: [batch, T, D]
        # 第一层: [batch, T, D] @ [T, D, kD] → [batch, T, kD]
        # 使用 einsum 实现 per-token 的矩阵乘法
        h = torch.einsum('btd, tdh -> bth', x, self.W1) + self.b1  # [batch, T, kD]
        h = F.gelu(h)

        # 第二层: [batch, T, kD] @ [T, kD, D] → [batch, T, D]
        out = torch.einsum('bth, thd -> btd', h, self.W2) + self.b2  # [batch, T, D]

        return out


# ============================================================
# 4. RankMixer Block (Token Mixing + Per-token FFN)
# ============================================================

class RankMixerBlock(nn.Module):
    """
    论文公式:
        S_{n-1} = LN(TokenMixing(X_{n-1}) + X_{n-1})
        X_n     = LN(PFFN(S_{n-1}) + S_{n-1})
    """

    def __init__(self, num_tokens: int, hidden_dim: int, ffn_expansion: int = 4):
        super().__init__()
        self.token_mixing = MultiHeadTokenMixing(num_tokens, hidden_dim)
        self.pffn = PerTokenFFN(num_tokens, hidden_dim, ffn_expansion)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args & Returns: [batch, T, D]
        """
        # Token Mixing + Residual + LN
        s = self.norm1(self.token_mixing(x) + x)

        # Per-token FFN + Residual + LN
        out = self.norm2(self.pffn(s) + s)

        return out


# ============================================================
# 5. Sparse-MoE: ReLU Routing + DTSI
# ============================================================

class ReLURouter(nn.Module):
    """
    论文 Section 4.1: ReLU Routing

    摒弃 Top-k + Softmax，使用 ReLU 门控。
    不同 token 可以激活不同数量的专家。

    G_{i,j} = ReLU(h(s_i))

    稀疏性由 L1 正则控制:
        L_reg = λ * Σ_i Σ_j G_{i,j}
    """

    def __init__(self, hidden_dim: int, num_experts: int):
        super().__init__()
        self.gate = nn.Linear(hidden_dim, num_experts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch*T, D]
        Returns:
            gate_values: [batch*T, num_experts] (非负，ReLU 后)
        """
        return F.relu(self.gate(x))


class PerTokenMoEFFN(nn.Module):
    """
    论文 Section 4: 将 Per-token FFN 扩展为 Sparse-MoE

    每个 token 有自己的一组专家 (Per-token experts)。
    使用 ReLU routing + DTSI (Dense-Train, Sparse-Infer) 策略。

    公式:
        G_{i,j} = ReLU(h(s_i))
        v_i = Σ_{j=1}^{N_e} G_{i,j} * e_{i,j}(s_i)

    DTSI 策略:
        - 训练时: 使用 h_train (密集路由, 所有专家都收到梯度)
        - 推理时: 使用 h_infer (稀疏路由, 降低计算成本)
    """

    def __init__(
        self,
        num_tokens: int,
        hidden_dim: int,
        num_experts: int = 8,
        ffn_expansion: int = 4,
        l1_lambda: float = 0.01,
    ):
        super().__init__()
        self.T = num_tokens
        self.D = hidden_dim
        self.num_experts = num_experts
        self.kD = hidden_dim * ffn_expansion
        self.l1_lambda = l1_lambda

        # 每个 token 有 num_experts 个专家
        # Expert weights: 每个 token 每个 expert 有独立的 W1, W2
        # W1: [T, num_experts, D, kD],  W2: [T, num_experts, kD, D]
        self.W1 = nn.Parameter(
            torch.empty(num_tokens, num_experts, hidden_dim, self.kD)
        )
        self.b1 = nn.Parameter(
            torch.zeros(num_tokens, num_experts, self.kD)
        )
        self.W2 = nn.Parameter(
            torch.empty(num_tokens, num_experts, self.kD, hidden_dim)
        )
        self.b2 = nn.Parameter(
            torch.zeros(num_tokens, num_experts, hidden_dim)
        )

        # 初始化
        for t in range(num_tokens):
            for e in range(num_experts):
                nn.init.kaiming_uniform_(self.W1[t, e], a=math.sqrt(5))
                nn.init.kaiming_uniform_(self.W2[t, e], a=math.sqrt(5))

        # DTSI: 训练路由器 和 推理路由器
        self.router_train = ReLURouter(hidden_dim, num_experts)
        self.router_infer = ReLURouter(hidden_dim, num_experts)

    def _expert_forward(
        self, x: torch.Tensor, gate: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x:    [batch, T, D]
            gate: [batch, T, num_experts]
        Returns:
            [batch, T, D]
        """
        # 对每个 token t, 每个 expert j:
        #   h = Gelu(x_t @ W1[t,j] + b1[t,j])
        #   e_out = h @ W2[t,j] + b2[t,j]
        #   最终: v_t = Σ_j gate[t,j] * e_out

        # x: [batch, T, D],  W1: [T, E, D, kD]
        # 沿 D 维收缩: btd, tedk -> btek
        h = torch.einsum('btd, tedk -> btek', x, self.W1)  # [batch, T, E, kD]
        h = h + self.b1.unsqueeze(0)  # [batch, T, E, kD]
        h = F.gelu(h)

        # h: [batch, T, E, kD],  W2: [T, E, kD, D]
        # 沿 kD 维收缩: btek, tekd -> bted
        out = torch.einsum('btek, tekd -> bted', h, self.W2)  # [batch, T, E, D]
        out = out + self.b2.unsqueeze(0)  # [batch, T, E, D]

        # 加权求和: gate [batch, T, E] * out [batch, T, E, D]
        out = torch.einsum('bte, bted -> btd', gate, out)  # [batch, T, D]

        return out

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch, T, D]
        Returns:
            output: [batch, T, D]
            reg_loss: L1 稀疏正则损失
        """
        batch, T, D = x.shape

        if self.training:
            # DTSI: 训练时密集路由 (所有专家收到梯度)
            gate_train = self.router_train(
                x.reshape(-1, D)
            ).view(batch, T, self.num_experts)  # [batch, T, E]

            # 同时训练推理路由器
            gate_infer = self.router_infer(
                x.reshape(-1, D)
            ).view(batch, T, self.num_experts)

            # 用训练路由器的门控值前向
            output = self._expert_forward(x, gate_train)

            # L1 正则: 控制推理路由器的稀疏性
            reg_loss = self.l1_lambda * gate_infer.sum() / (batch * T)
        else:
            # 推理时仅用稀疏路由器
            gate = self.router_infer(
                x.reshape(-1, D)
            ).view(batch, T, self.num_experts)  # [batch, T, E]

            output = self._expert_forward(x, gate)
            reg_loss = torch.tensor(0.0, device=x.device)

        return output, reg_loss


class RankMixerMoEBlock(nn.Module):
    """
    带 MoE 的 RankMixer Block:
        S_{n-1} = LN(TokenMixing(X_{n-1}) + X_{n-1})
        X_n     = LN(MoE_PFFN(S_{n-1}) + S_{n-1})
    """

    def __init__(
        self,
        num_tokens: int,
        hidden_dim: int,
        num_experts: int = 8,
        ffn_expansion: int = 4,
        l1_lambda: float = 0.01,
    ):
        super().__init__()
        self.token_mixing = MultiHeadTokenMixing(num_tokens, hidden_dim)
        self.moe_pffn = PerTokenMoEFFN(
            num_tokens, hidden_dim, num_experts, ffn_expansion, l1_lambda
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        s = self.norm1(self.token_mixing(x) + x)
        pffn_out, reg_loss = self.moe_pffn(s)
        out = self.norm2(pffn_out + s)
        return out, reg_loss


# ============================================================
# 6. 完整 RankMixer 模型
# ============================================================

class RankMixer(nn.Module):
    """
    完整的 RankMixer 模型

    模型参数量: ~2kLTD^2  (密集版)
    FLOPs:     ~4kLTD^2

    扩展维度:
    - T: token 数量 (特征分组切分)
    - D: 隐藏维度
    - L: 层数
    - E: 专家数 (MoE 版本)
    """

    def __init__(
        self,
        feature_dims: List[int],     # 各特征域 vocab size
        embed_dims: List[int],       # 各特征域 embedding dim
        chunk_size: int = 64,        # 切分维度 d
        hidden_dim: int = 256,       # 隐藏维度 D
        num_dense_layers: int = 3,   # 密集 Block 层数
        num_moe_layers: int = 1,     # MoE Block 层数
        ffn_expansion: int = 4,      # FFN 扩展倍率 k
        num_experts: int = 8,        # 专家数 (MoE)
        l1_lambda: float = 0.01,     # L1 正则系数
        num_classes: int = 1,        # 输出数 (CTR=1)
    ):
        super().__init__()

        # 输入 Token 化
        self.tokenizer = FeatureTokenizer(
            feature_dims, embed_dims, chunk_size, hidden_dim
        )
        T = self.tokenizer.T  # token 数量

        # 密集 RankMixer Blocks
        self.dense_blocks = nn.ModuleList([
            RankMixerBlock(T, hidden_dim, ffn_expansion)
            for _ in range(num_dense_layers)
        ])

        # MoE RankMixer Blocks
        self.moe_blocks = nn.ModuleList([
            RankMixerMoEBlock(T, hidden_dim, num_experts, ffn_expansion, l1_lambda)
            for _ in range(num_moe_layers)
        ])

        # Output Pooling → 预测
        self.output = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self, feature_ids: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            feature_ids: List of [batch] tensors
        Returns:
            logits: [batch, num_classes]
            reg_loss: MoE L1 正则损失
        """
        # Token 化: [batch, T, D]
        x = self.tokenizer(feature_ids)

        # 密集层
        for block in self.dense_blocks:
            x = block(x)

        # MoE 层
        total_reg_loss = 0
        for block in self.moe_blocks:
            x, reg_loss = block(x)
            total_reg_loss = total_reg_loss + reg_loss

        # 输出: 平均池化 → 线性
        x = x.mean(dim=1)  # [batch, D]
        logits = self.output(x)  # [batch, num_classes]

        return logits, total_reg_loss


# ============================================================
# 测试
# ============================================================

def test():
    batch_size = 4

    # 定义特征 (模拟工业场景的异构特征)
    # 用户特征: user_id(10k), age(100), gender(3)
    # 物品特征: item_id(50k), category(1000), brand(500)
    # 上下文特征: hour(24), device(10)
    # 交互历史: hist_item_1..5 (各50k)
    feature_dims = [10000, 100, 3, 50000, 1000, 500, 24, 10] + [50000] * 5
    embed_dims   = [64,    16, 8, 64,    32,   32,  8,  8]  + [64] * 5
    # 总维度 = 552, chunk_size=64 → T = ceil(552/64) = 9
    # D 必须被 T 整除 (因为 H=T), 所以 D=9*32=288

    # 模拟输入
    feature_ids = [
        torch.randint(0, dim, (batch_size,))
        for dim in feature_dims
    ]

    model = RankMixer(
        feature_dims=feature_dims,
        embed_dims=embed_dims,
        chunk_size=70,       # d: 切分维度 (ceil(552/70)=8)
        hidden_dim=256,      # D: 隐藏维度 (必须被 T=8 整除, 256/8=32)
        num_dense_layers=3,  # L_dense
        num_moe_layers=1,    # L_moe
        ffn_expansion=4,     # k
        num_experts=4,       # E
        l1_lambda=0.01,
        num_classes=1,
    )

    print("=" * 60)
    print("RankMixer Architecture Test")
    print("=" * 60)

    T = model.tokenizer.T
    D = 256
    print(f"特征域数: {len(feature_dims)}")
    print(f"Embedding 总维度: {sum(embed_dims)}")
    print(f"Token 数 (T): {T}")
    print(f"隐藏维度 (D): {D}")
    print(f"Head dim (D/T): {D // T}")
    print()

    # 前向
    logits, reg_loss = model(feature_ids)
    print(f"输出 logits: {logits.shape}")
    print(f"MoE reg loss: {reg_loss.item():.6f}")
    print()

    # 参数统计
    total = sum(p.numel() for p in model.parameters())
    tokenizer_params = sum(p.numel() for p in model.tokenizer.parameters())
    dense_params = sum(p.numel() for p in model.dense_blocks.parameters())
    moe_params = sum(p.numel() for p in model.moe_blocks.parameters())
    output_params = sum(p.numel() for p in model.output.parameters())

    print(f"总参数量:        {total / 1e6:.2f}M")
    print(f"  Tokenizer:     {tokenizer_params / 1e6:.2f}M")
    print(f"  Dense blocks:  {dense_params / 1e6:.2f}M")
    print(f"  MoE blocks:    {moe_params / 1e6:.2f}M")
    print(f"  Output:        {output_params / 1e6:.2f}M")

    # 论文公式验证: #Param ≈ 2kLTD^2
    k, L = 4, 4  # total layers
    theoretical = 2 * k * L * T * D * D
    print(f"\n论文理论参数量 (2kLTD²): {theoretical / 1e6:.2f}M (不含 embedding)")


if __name__ == "__main__":
    test()
