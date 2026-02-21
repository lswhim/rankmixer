"""
CTR 模型适配层: 将底层模块 (RankMixer / TokenMixer-Large / HSTU / Transformer)
适配到 KuaiVideo CTR 预测任务。

包含:
- DINAttention: Target-Aware 序列注意力
- BaseCTR: 公共基类 (embedding + DIN + chunk tokenization)
- RankMixerCTR
- TokenMixerLargeCTR
- TransformerCTR
- HSTUCTR
- build_model: 统一构建入口
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .rankmixer import (
    MultiHeadTokenMixing, PerTokenFFN, ReLURouter, PerTokenMoEFFN,
    RankMixerBlock, RankMixerMoEBlock,
)


# ============================================================
# DIN Attention (Target-Aware 序列编码)
# ============================================================

class DINAttention(nn.Module):
    """
    Deep Interest Network (DIN) 的 target-aware attention.
    将用户行为序列编码为一个 target-aware 的 embedding 向量。
    attention(e_i, e_target) = softmax(MLP(e_i, e_target, e_i - e_target, e_i * e_target))
    """

    def __init__(self, emb_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.attn_mlp = nn.Sequential(
            nn.Linear(emb_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, seq_emb, target_emb, seq_mask):
        """
        Args:
            seq_emb:    [B, S, E]  序列 item embeddings
            target_emb: [B, E]     候选 item embedding
            seq_mask:   [B, S]     bool, True = valid position
        Returns:
            out: [B, E]  weighted sum of seq_emb
        """
        S = seq_emb.size(1)
        target_exp = target_emb.unsqueeze(1).expand_as(seq_emb)  # [B, S, E]

        attn_input = torch.cat([
            seq_emb, target_exp,
            seq_emb - target_exp,
            seq_emb * target_exp,
        ], dim=-1)  # [B, S, 4E]

        attn_score = self.attn_mlp(attn_input).squeeze(-1)  # [B, S]

        # mask padding positions
        attn_score = attn_score.masked_fill(~seq_mask, float("-inf"))

        # 全为 padding 时 softmax 会 NaN，用 0 代替
        all_masked = ~seq_mask.any(dim=-1, keepdim=True)  # [B, 1]
        attn_weight = torch.softmax(attn_score, dim=-1)   # [B, S]
        attn_weight = attn_weight.masked_fill(all_masked, 0.0)

        out = (attn_weight.unsqueeze(-1) * seq_emb).sum(dim=1)  # [B, E]
        return out


# ============================================================
# CTR 模型基类 (公共: embedding + DIN序列编码 + tokenization + proj)
# ============================================================

class BaseCTR(nn.Module):
    """RankMixer/TokenMixer 共用: embedding → DIN序列编码 → chunk → proj"""

    def __init__(self, cfg):
        super().__init__()
        data_cfg = cfg["data"]
        emb_cfg = cfg["embedding"]
        model_cfg = cfg["model"]

        user_emb_dim = emb_cfg["user_emb_dim"]
        item_emb_dim = emb_cfg["item_emb_dim"]
        self.use_pretrained = data_cfg.get("use_pretrained_emb", True)
        item_vis_dim = data_cfg["item_vis_dim"] if self.use_pretrained else 0
        chunk_size = model_cfg["chunk_size"]
        hidden_dim = model_cfg["hidden_dim"]
        self.embedding_regularizer = data_cfg.get("embedding_regularizer", 0.0)

        self.user_emb = nn.Embedding(data_cfg["num_users"], user_emb_dim)
        self.item_emb = nn.Embedding(data_cfg["item_hash_size"], item_emb_dim)

        # Pretrained embedding 投影层 (与 FuxiCTR 对齐: nn.Linear(64,64,bias=False))
        if self.use_pretrained:
            self.vis_proj = nn.Linear(item_vis_dim, item_vis_dim, bias=False)

        # DIN 序列注意力: 双通道 (item_id_emb + item_vis_emb) → DIN
        # 序列 embedding 维度 = item_emb_dim + item_vis_dim (与 FuxiCTR 对齐)
        seq_emb_dim = item_emb_dim + item_vis_dim if self.use_pretrained else item_emb_dim
        self.pos_din_attn = DINAttention(seq_emb_dim, hidden_dim=64)
        self.neg_din_attn = DINAttention(seq_emb_dim, hidden_dim=64)

        # total_dim: user_emb + item_emb + item_vis + pos_seq_enc + neg_seq_enc
        # 无 is_like/is_follow (与 FuxiCTR benchmark 对齐)
        total_dim = (user_emb_dim + item_emb_dim + item_vis_dim
                     + seq_emb_dim + seq_emb_dim)
        self.num_tokens = math.ceil(total_dim / chunk_size)
        self.padded_dim = self.num_tokens * chunk_size
        self.chunk_size = chunk_size
        self.hidden_dim = hidden_dim
        self.total_dim = total_dim

        self.proj = nn.Linear(chunk_size, hidden_dim)

    def _get_embedding_reg_loss(self):
        """计算 embedding L2 正则损失 (与 FuxiCTR 对齐: λ/p * ||W||_p^p, p=2)"""
        if self.embedding_regularizer <= 0:
            return torch.tensor(0.0)
        reg = self.embedding_regularizer / 2.0 * (
            self.user_emb.weight.norm(2).pow(2) +
            self.item_emb.weight.norm(2).pow(2)
        )
        return reg

    def _tokenize(self, user_ids, item_ids, item_vis,
                  pos_items, pos_lens, neg_items, neg_lens,
                  pos_items_vis, neg_items_vis):
        """
        公共前处理 (与 FuxiCTR benchmark 对齐):
        embedding → vis_proj → DIN双通道序列编码(pos+neg) → concat → pad → chunk → proj → [B, T, D]
        """
        u_emb = self.user_emb(user_ids)       # [B, user_emb_dim]
        i_emb = self.item_emb(item_ids)       # [B, item_emb_dim]

        if self.use_pretrained:
            # 投影 target item visual embedding
            item_vis_proj = self.vis_proj(item_vis)  # [B, vis_dim]

            # DIN: pos 双通道 (item_id_emb + item_vis_emb)
            pos_id_emb = self.item_emb(pos_items)           # [B, S, item_emb_dim]
            pos_vis_proj = self.vis_proj(pos_items_vis)      # [B, S, vis_dim]
            pos_seq_emb = torch.cat([pos_id_emb, pos_vis_proj], dim=-1)  # [B, S, seq_emb_dim]
            target_seq = torch.cat([i_emb, item_vis_proj], dim=-1)       # [B, seq_emb_dim]

            # DIN: neg 双通道
            neg_id_emb = self.item_emb(neg_items)           # [B, S, item_emb_dim]
            neg_vis_proj = self.vis_proj(neg_items_vis)      # [B, S, vis_dim]
            neg_seq_emb = torch.cat([neg_id_emb, neg_vis_proj], dim=-1)  # [B, S, seq_emb_dim]
        else:
            item_vis_proj = None
            pos_seq_emb = self.item_emb(pos_items)
            neg_seq_emb = self.item_emb(neg_items)
            target_seq = i_emb

        pos_mask = torch.arange(pos_items.size(1), device=pos_items.device).unsqueeze(0) < pos_lens.unsqueeze(1)
        pos_enc = self.pos_din_attn(pos_seq_emb, target_seq, pos_mask)  # [B, seq_emb_dim]

        neg_mask = torch.arange(neg_items.size(1), device=neg_items.device).unsqueeze(0) < neg_lens.unsqueeze(1)
        neg_enc = self.neg_din_attn(neg_seq_emb, target_seq, neg_mask)  # [B, seq_emb_dim]

        parts = [u_emb, i_emb]
        if self.use_pretrained:
            parts.append(item_vis_proj)
        parts.extend([pos_enc, neg_enc])
        e_input = torch.cat(parts, dim=-1)

        B = e_input.size(0)
        if e_input.size(-1) < self.padded_dim:
            e_input = F.pad(e_input, (0, self.padded_dim - e_input.size(-1)))

        tokens = e_input.view(B, self.num_tokens, self.chunk_size)
        return self.proj(tokens)  # [B, T, D]


# ============================================================
# RankMixer CTR 模型
# ============================================================

class RankMixerCTR(BaseCTR):
    def __init__(self, cfg):
        super().__init__(cfg)
        model_cfg = cfg["model"]
        T = self.num_tokens
        hidden_dim = self.hidden_dim

        num_dense = model_cfg["num_dense_layers"]
        num_moe = model_cfg["num_moe_layers"]
        ffn_exp = model_cfg["ffn_expansion"]
        num_experts = model_cfg["num_experts"]
        l1_lambda = model_cfg["l1_lambda"]

        print(f"  [RankMixer] Feature dim: {self.total_dim}, Tokens (T): {T}, "
              f"Hidden (D): {hidden_dim}, D/T: {hidden_dim // T}")
        print(f"  Dense layers: {num_dense}, MoE layers: {num_moe}, "
              f"Experts: {num_experts}, FFN expansion: {ffn_exp}")

        self.dense_blocks = nn.ModuleList([
            RankMixerBlock(T, hidden_dim, ffn_exp) for _ in range(num_dense)
        ])
        self.moe_blocks = nn.ModuleList([
            RankMixerMoEBlock(T, hidden_dim, num_experts, ffn_exp, l1_lambda)
            for _ in range(num_moe)
        ])
        self.output_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, user_ids, item_ids, item_vis,
                pos_items, pos_lens, neg_items, neg_lens,
                pos_items_vis, neg_items_vis):
        x = self._tokenize(user_ids, item_ids, item_vis,
                           pos_items, pos_lens, neg_items, neg_lens,
                           pos_items_vis, neg_items_vis)

        for blk in self.dense_blocks:
            x = blk(x)

        total_reg = torch.tensor(0.0, device=x.device)
        for blk in self.moe_blocks:
            x, reg = blk(x)
            total_reg = total_reg + reg

        # 加上 embedding L2 正则
        total_reg = total_reg + self._get_embedding_reg_loss().to(x.device)

        x = x.mean(dim=1)
        logits = self.output_head(x).squeeze(-1)
        return logits, total_reg


# ============================================================
# TokenMixer-Large CTR 模型 (arXiv:2602.06563)
# ============================================================

class TokenMixerLargeCTR(BaseCTR):
    def __init__(self, cfg):
        super().__init__(cfg)
        from .tokenmixer_large import (
            TokenMixerLargeBlock, TokenMixerLargeMoEBlock, RMSNorm,
        )

        model_cfg = cfg["model"]
        T = self.num_tokens
        hidden_dim = self.hidden_dim
        T_global = T + 1

        print(f"  [TokenMixer-Large] Feature dim: {self.total_dim}, Tokens (T): {T}, "
              f"+1 global = {T_global}")
        print(f"  Hidden (D): {hidden_dim}, D/(T+1): {hidden_dim // T_global}")

        num_layers = model_cfg["num_layers"]
        ffn_expansion = model_cfg["ffn_expansion"]
        use_moe = model_cfg["use_moe"]
        small_init = model_cfg.get("small_init", True)
        inter_residual_interval = model_cfg.get("inter_residual_interval", 2)
        self.inter_residual_interval = inter_residual_interval
        self.num_layers = num_layers
        self.aux_loss_weight = model_cfg.get("aux_loss_weight", 0.1)

        print(f"  Layers: {num_layers}, MoE: {use_moe}, "
              f"Experts: {model_cfg.get('num_experts', '-')}, "
              f"FFN expansion: {ffn_expansion}")

        self.global_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        nn.init.normal_(self.global_token, std=0.02)

        if use_moe:
            num_experts = model_cfg["num_experts"]
            top_k = model_cfg.get("top_k", 2)
            gate_scale = model_cfg.get("gate_scale", 0.0)
            self.blocks = nn.ModuleList([
                TokenMixerLargeMoEBlock(
                    T_global, hidden_dim, num_experts, top_k,
                    ffn_expansion, gate_scale, small_init
                ) for _ in range(num_layers)
            ])
        else:
            self.blocks = nn.ModuleList([
                TokenMixerLargeBlock(T_global, hidden_dim, ffn_expansion, small_init)
                for _ in range(num_layers)
            ])

        self.aux_heads = nn.ModuleDict()
        for i in range(inter_residual_interval - 1, num_layers - 1, inter_residual_interval):
            self.aux_heads[str(i)] = nn.Sequential(
                RMSNorm(hidden_dim),
                nn.Linear(hidden_dim, 1),
            )

        self.output_head = nn.Sequential(
            RMSNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, user_ids, item_ids, item_vis,
                pos_items, pos_lens, neg_items, neg_lens,
                pos_items_vis, neg_items_vis):
        tokens = self._tokenize(user_ids, item_ids, item_vis,
                                pos_items, pos_lens, neg_items, neg_lens,
                                pos_items_vis, neg_items_vis)
        B = tokens.size(0)

        global_t = self.global_token.expand(B, -1, -1)
        x = torch.cat([global_t, tokens], dim=1)

        aux_logits_list = []
        residual_cache = x

        for i, block in enumerate(self.blocks):
            x = block(x)

            if (i + 1) % self.inter_residual_interval == 0 and i < self.num_layers - 1:
                x = x + residual_cache
                residual_cache = x

                if str(i) in self.aux_heads:
                    aux_logit = self.aux_heads[str(i)](x[:, 0, :]).squeeze(-1)
                    aux_logits_list.append(aux_logit)

        main_logits = self.output_head(x[:, 0, :]).squeeze(-1)

        if aux_logits_list and self.training:
            aux_logits = torch.stack(aux_logits_list, dim=0).mean(dim=0)
        else:
            aux_logits = torch.zeros_like(main_logits)

        return main_logits, aux_logits


# ============================================================
# Vanilla Transformer CTR 模型 (Baseline)
# 标准 Multi-Head Self-Attention + FFN, 作为最简对照组
# ============================================================

class TransformerCTR(BaseCTR):
    """
    最简单的 Transformer baseline:
    - 继承 BaseCTR 复用 embedding + DIN 序列编码 + chunk tokenization
    - 标准 Pre-LN Transformer: LayerNorm → MHSA → residual → LayerNorm → FFN → residual
    - Mean pooling → 预测头
    - 无 MoE、无 global token、无花哨设计
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        model_cfg = cfg["model"]
        T = self.num_tokens
        hidden_dim = self.hidden_dim

        num_layers = model_cfg["num_layers"]
        num_heads = model_cfg["num_heads"]
        ffn_expansion = model_cfg.get("ffn_expansion", 4)
        dropout = model_cfg.get("dropout", 0.0)

        print(f"  [Transformer] Feature dim: {self.total_dim}, Tokens (T): {T}, "
              f"Hidden (D): {hidden_dim}, Heads: {num_heads}, "
              f"Head dim: {hidden_dim // num_heads}")
        print(f"  Layers: {num_layers}, FFN expansion: {ffn_expansion}, "
              f"Dropout: {dropout}")

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * ffn_expansion,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-LN
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers,
            enable_nested_tensor=False,
        )

        self.output_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, user_ids, item_ids, item_vis,
                pos_items, pos_lens, neg_items, neg_lens,
                pos_items_vis, neg_items_vis):
        x = self._tokenize(user_ids, item_ids, item_vis,
                           pos_items, pos_lens, neg_items, neg_lens,
                           pos_items_vis, neg_items_vis)
        x = self.encoder(x)
        x = x.mean(dim=1)
        logits = self.output_head(x).squeeze(-1)
        return logits, torch.tensor(0.0, device=logits.device)


# ============================================================
# HSTU CTR 模型 (arXiv:2402.17152)
# 原文做法: Content-Action 交替序列 + Target-Aware Attention
# 不继承 BaseCTR, 独立实现序列输入
# ============================================================

class HSTUCTR(nn.Module):
    """
    HSTU 序列转导模型 (忠实于原文)

    输入序列构造:
      [Φ_0, a_0, Φ_1, a_1, ..., Φ_{k-1}, a_{k-1}, Φ_target]
    其中:
      - Φ_i = 历史交互 item 的 embedding (content token)
      - a_i = 对应行为的 embedding (action token, 这里统一为 click)
      - Φ_target = 当前候选 item (放在末尾)

    用户特征 (user_emb) 作为序列的第一个 prefix token。
    输出取 Φ_target 位置 (最后一个 token) 的 hidden state。
    """

    def __init__(self, cfg):
        super().__init__()
        from .hstu import HSTULayer, RMSNorm as HSTURMSNorm

        data_cfg = cfg["data"]
        emb_cfg = cfg["embedding"]
        model_cfg = cfg["model"]

        user_emb_dim = emb_cfg["user_emb_dim"]
        item_emb_dim = emb_cfg["item_emb_dim"]
        self.use_pretrained = data_cfg.get("use_pretrained_emb", True)
        item_vis_dim = data_cfg["item_vis_dim"] if self.use_pretrained else 0
        hidden_dim = model_cfg["hidden_dim"]
        num_layers = model_cfg["num_layers"]
        num_heads = model_cfg["num_heads"]
        ffn_expansion = model_cfg.get("ffn_expansion", 4)
        dropout = model_cfg.get("dropout", 0.0)
        max_seq_len = data_cfg.get("max_seq_len", 100)
        self.embedding_regularizer = data_cfg.get("embedding_regularizer", 0.0)

        self.hidden_dim = hidden_dim

        # Embeddings
        self.user_emb = nn.Embedding(data_cfg["num_users"], user_emb_dim)
        self.item_emb = nn.Embedding(data_cfg["item_hash_size"], item_emb_dim)

        # Pretrained embedding 投影层 (与 FuxiCTR 对齐)
        if self.use_pretrained:
            self.vis_proj = nn.Linear(item_vis_dim, item_vis_dim, bias=False)

        # Action embedding: click=0, like=1, follow=2, skip/neg=3
        num_action_types = 4
        self.action_emb = nn.Embedding(num_action_types, item_emb_dim)

        # 投影层: content token (item_emb + item_vis → hidden_dim)
        self.content_proj = nn.Linear(item_emb_dim + item_vis_dim, hidden_dim)
        # 投影层: action token (item_emb_dim → hidden_dim)
        self.action_proj = nn.Linear(item_emb_dim, hidden_dim)
        # 投影层: user prefix token (user_emb → hidden_dim, 无 is_like/is_follow)
        self.user_proj = nn.Linear(user_emb_dim, hidden_dim)

        # 序列最长: 1 (user prefix) + max_seq_len * 2 (pos content+action)
        #         + max_seq_len * 2 (neg content+action) + 1 (target)
        max_tokens = 1 + max_seq_len * 4 + 1 + 8  # 留余量
        self.max_seq_len = max_seq_len

        print(f"  [HSTU] Hidden (D): {hidden_dim}, Heads: {num_heads}, "
              f"Head dim: {hidden_dim // num_heads}")
        print(f"  Layers: {num_layers}, Max seq tokens: {max_tokens}, "
              f"FFN expansion: {ffn_expansion}, Dropout: {dropout}")

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

        self.output_head = nn.Sequential(
            HSTURMSNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
        )

    def _get_embedding_reg_loss(self):
        """计算 embedding L2 正则损失 (与 FuxiCTR 对齐: λ/p * ||W||_p^p, p=2)"""
        if self.embedding_regularizer <= 0:
            return torch.tensor(0.0)
        reg = self.embedding_regularizer / 2.0 * (
            self.user_emb.weight.norm(2).pow(2) +
            self.item_emb.weight.norm(2).pow(2)
        )
        return reg

    def forward(self, user_ids, item_ids, item_vis,
                pos_items, pos_lens, neg_items, neg_lens,
                pos_items_vis, neg_items_vis):
        """
        构造 Content-Action 交替序列并通过 HSTU Layers。
        序列: [user_prefix, pos_Φ_0, pos_a_0, ..., neg_Φ_0, neg_a_0, ..., Φ_target]
        取最后一个 token (target content) 的输出做预测。
        """
        B = user_ids.size(0)
        device = user_ids.device
        S = pos_items.size(1)  # max_seq_len

        # --- 构造各种 token ---
        # 1. User prefix token: user_emb → [B, hidden_dim]
        u_emb = self.user_emb(user_ids)                            # [B, user_emb_dim]
        user_token = self.user_proj(u_emb)

        # 2. Pos content tokens for history: item_emb + item_vis (双通道)
        pos_item_emb = self.item_emb(pos_items)                    # [B, S, item_emb_dim]
        if self.use_pretrained:
            pos_vis_proj = self.vis_proj(pos_items_vis)             # [B, S, vis_dim]
            pos_content_input = torch.cat([pos_item_emb, pos_vis_proj], dim=-1)
        else:
            pos_content_input = pos_item_emb
        pos_content_tokens = self.content_proj(pos_content_input)  # [B, S, D]

        # 3. Pos action tokens (click = 0)
        pos_action_ids = torch.zeros(B, S, dtype=torch.long, device=device)
        pos_action_tokens = self.action_proj(self.action_emb(pos_action_ids))  # [B, S, D]

        # 4. Neg content tokens for history (双通道)
        neg_item_emb = self.item_emb(neg_items)                    # [B, S, item_emb_dim]
        if self.use_pretrained:
            neg_vis_proj = self.vis_proj(neg_items_vis)             # [B, S, vis_dim]
            neg_content_input = torch.cat([neg_item_emb, neg_vis_proj], dim=-1)
        else:
            neg_content_input = neg_item_emb
        neg_content_tokens = self.content_proj(neg_content_input)  # [B, S, D]

        # 5. Neg action tokens (neg/skip = 3)
        neg_action_ids = torch.full((B, S), 3, dtype=torch.long, device=device)
        neg_action_tokens = self.action_proj(self.action_emb(neg_action_ids))  # [B, S, D]

        # 6. Target content token: item_emb + item_vis (双通道)
        i_emb = self.item_emb(item_ids)                            # [B, item_emb_dim]
        if self.use_pretrained:
            target_vis = self.vis_proj(item_vis)                   # [B, vis_dim]
            target_token = self.content_proj(torch.cat([i_emb, target_vis], dim=-1))
        else:
            target_token = self.content_proj(i_emb)

        # --- 构造交替序列 ---
        # Pos interleaved: [Φ_0, a_0, Φ_1, a_1, ...] → [B, 2S, D]
        pos_interleaved = torch.stack([pos_content_tokens, pos_action_tokens], dim=2)
        pos_interleaved = pos_interleaved.view(B, S * 2, self.hidden_dim)

        # Neg interleaved: [Φ_0, a_0, Φ_1, a_1, ...] → [B, 2S, D]
        neg_interleaved = torch.stack([neg_content_tokens, neg_action_tokens], dim=2)
        neg_interleaved = neg_interleaved.view(B, S * 2, self.hidden_dim)

        # 拼接: [user(1), pos(2S), neg(2S), target(1)] = [B, 4S+2, D]
        full_seq = torch.cat([
            user_token.unsqueeze(1),       # [B, 1, D]
            pos_interleaved,               # [B, 2S, D]
            neg_interleaved,               # [B, 2S, D]
            target_token.unsqueeze(1),     # [B, 1, D]
        ], dim=1)  # [B, 4S+2, D]

        # --- 构造 attention mask ---
        # valid 长度: 1 (user) + pos_len*2 + neg_len*2 + 1 (target)
        valid_lens = 1 + pos_lens * 2 + neg_lens * 2 + 1  # [B]
        total_len = full_seq.size(1)       # 4S+2

        # padding mask: [B, total_len], True = valid
        pos_indices = torch.arange(total_len, device=device).unsqueeze(0)
        padding_mask = pos_indices < valid_lens.unsqueeze(1)

        # attention mask: [B, 1, total_len, total_len]
        attn_mask = padding_mask.unsqueeze(1).unsqueeze(2)
        attn_mask = attn_mask.expand(B, 1, total_len, total_len)

        # --- HSTU Layers ---
        x = full_seq
        for layer in self.layers:
            x = layer(x, attn_mask=attn_mask)

        # --- 取 target 位置的输出 ---
        target_idx = (valid_lens - 1).unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1]
        target_idx = target_idx.expand(B, 1, self.hidden_dim)     # [B, 1, D]
        target_hidden = x.gather(1, target_idx).squeeze(1)         # [B, D]

        logits = self.output_head(target_hidden).squeeze(-1)
        reg_loss = self._get_embedding_reg_loss().to(logits.device)
        return logits, reg_loss


# ============================================================
# 构建模型 (统一入口)
# ============================================================

def build_model(cfg) -> nn.Module:
    arch = cfg["model"].get("arch", "rankmixer")
    if arch == "tokenmixer_large":
        return TokenMixerLargeCTR(cfg)
    elif arch == "hstu":
        return HSTUCTR(cfg)
    elif arch == "transformer":
        return TransformerCTR(cfg)
    else:
        return RankMixerCTR(cfg)
