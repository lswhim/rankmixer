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
from torch.utils.checkpoint import checkpoint as ckpt_fn

from .rankmixer import (
    MultiHeadTokenMixing, PerTokenFFN, ReLURouter, PerTokenMoEFFN,
    RankMixerBlock, RankMixerMoEBlock,
)
from .dmin import Dice, MLPBlock


def _build_output_head(input_dim, hidden_units, activation="Dice", dropout=0.1):
    """
    构建 BARS 风格的预测头 MLP (对齐 DIN/DMIN benchmark):
    [Linear → Dice → Dropout] × N → Linear(1)
    """
    return nn.Sequential(
        MLPBlock(input_dim, hidden_units,
                 hidden_activation=activation,
                 dropout_rate=dropout),
        nn.Linear(hidden_units[-1], 1),
    )


# ============================================================
# DIN Attention (Target-Aware 序列编码)
# ============================================================

class DINAttention(nn.Module):
    """
    Deep Interest Network (DIN) 的 target-aware attention (对齐 FuxiCTR).
    拼接顺序: [target, seq, target-seq, target*seq] (与 FuxiCTR TargetAttention 一致)
    hidden_units 可配置，默认 [256, 128] (增大容量，对齐 BARS benchmark)
    """

    def __init__(self, emb_dim: int, hidden_units=None):
        super().__init__()
        if hidden_units is None:
            hidden_units = [256, 128]
        layers = []
        input_dim = emb_dim * 4
        for h in hidden_units:
            layers.append(nn.Linear(input_dim, h))
            layers.append(nn.ReLU())
            input_dim = h
        layers.append(nn.Linear(input_dim, 1))
        self.attn_mlp = nn.Sequential(*layers)

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

        # P4: 拼接顺序对齐 FuxiCTR: [target, seq, target-seq, target*seq]
        attn_input = torch.cat([
            target_exp, seq_emb,
            target_exp - seq_emb,
            target_exp * seq_emb,
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
        self.use_user_feature = data_cfg.get("use_user_feature", True)
        self.use_pretrained = data_cfg.get("use_pretrained_emb", True)
        item_vis_dim = data_cfg.get("item_vis_dim", 0) if self.use_pretrained else 0
        self.use_category = data_cfg.get("num_categories", 0) > 0
        self.use_item_sequence = data_cfg.get("pos_item_col", "pos_items") is not None
        self.cfg_data_has_pos_cate = data_cfg.get("pos_cate_col") is not None
        cate_emb_dim = emb_cfg.get("cate_emb_dim", item_emb_dim)
        extra_cat_emb_dim = emb_cfg.get("extra_cat_emb_dim", item_emb_dim)
        numeric_emb_dim = emb_cfg.get("numeric_emb_dim", item_emb_dim)
        chunk_size = model_cfg["chunk_size"]
        hidden_dim = model_cfg["hidden_dim"]
        self.embedding_regularizer = data_cfg.get("embedding_regularizer", 0.0)
        self.extra_cat_configs = data_cfg.get("extra_categorical_cols", [])
        self.extra_cat_names = [c["name"] for c in self.extra_cat_configs]
        self.extra_seq_configs = data_cfg.get("extra_sequence_cols", [])
        self.numeric_cols = data_cfg.get("numeric_cols", [])

        if self.use_user_feature:
            self.user_emb = nn.Embedding(data_cfg["num_users"], user_emb_dim)
        self.item_emb = nn.Embedding(data_cfg["item_hash_size"], item_emb_dim)
        if self.use_category:
            self.cate_emb = nn.Embedding(data_cfg["num_categories"], cate_emb_dim)
        self.extra_cat_embs = nn.ModuleDict({
            c["name"]: nn.Embedding(c["vocab_size"], c.get("embedding_dim", extra_cat_emb_dim))
            for c in self.extra_cat_configs
        })
        self.numeric_projs = nn.ModuleDict({
            name: nn.Linear(1, numeric_emb_dim)
            for name in self.numeric_cols
        })

        # P2: xavier_uniform_ 初始化 embedding (对齐 FuxiCTR)
        if self.use_user_feature:
            nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)
        if self.use_category:
            nn.init.xavier_uniform_(self.cate_emb.weight)
        for emb in self.extra_cat_embs.values():
            nn.init.xavier_uniform_(emb.weight)

        # Pretrained embedding 投影层 (与 FuxiCTR 对齐: nn.Linear(64,64,bias=False))
        if self.use_pretrained:
            self.vis_proj = nn.Linear(item_vis_dim, item_vis_dim, bias=False)

        # P1: DIN 序列注意力, hidden_units 可配置 (默认 [256, 128], 增大容量)
        seq_emb_dim = 0
        if self.use_item_sequence:
            seq_emb_dim += item_emb_dim
        if self.use_item_sequence and self.use_pretrained:
            seq_emb_dim += item_vis_dim
        if self.use_category and data_cfg.get("pos_cate_col") is not None:
            seq_emb_dim += cate_emb_dim
        for c in self.extra_seq_configs:
            share = c["share_embedding"]
            if share == "item_id":
                seq_emb_dim += item_emb_dim
            elif share == "cate_id":
                seq_emb_dim += cate_emb_dim
            else:
                seq_emb_dim += self.extra_cat_embs[share].embedding_dim
        din_hidden_units = model_cfg.get("din_hidden_units", [256, 128])
        self.pos_din_attn = DINAttention(seq_emb_dim, hidden_units=din_hidden_units)

        # total_dim: optional user_emb + item_emb + optional item_vis/category + pos_seq_enc
        total_dim = item_emb_dim + seq_emb_dim
        if self.use_user_feature:
            total_dim += user_emb_dim
        if self.use_pretrained:
            total_dim += item_vis_dim
        if self.use_category:
            total_dim += cate_emb_dim
        total_dim += sum(self.extra_cat_embs[name].embedding_dim for name in self.extra_cat_names)
        total_dim += len(self.numeric_cols) * numeric_emb_dim
        self.num_tokens = math.ceil(total_dim / chunk_size)
        self.padded_dim = self.num_tokens * chunk_size
        self.chunk_size = chunk_size
        self.hidden_dim = hidden_dim
        self.total_dim = total_dim

        self.proj = nn.Linear(chunk_size, hidden_dim)

    def _extra_cat_embedding_dict(self, extra_cat_ids):
        if not self.extra_cat_names:
            return {}
        return {
            name: self.extra_cat_embs[name](extra_cat_ids[:, idx])
            for idx, name in enumerate(self.extra_cat_names)
        }

    def _lookup_shared_seq_embedding(self, share_name, ids):
        if share_name == "item_id":
            return self.item_emb(ids)
        if share_name == "cate_id":
            return self.cate_emb(ids)
        return self.extra_cat_embs[share_name](ids)

    def _lookup_shared_target_embedding(self, share_name, i_emb, c_emb, extra_cat_embs):
        if share_name == "item_id":
            return i_emb
        if share_name == "cate_id":
            return c_emb
        return extra_cat_embs[share_name]

    def _get_embedding_reg_loss(self, u_emb, i_emb):
        """
        P0: 计算 batch-level embedding L2 正则损失。
        只对当前 batch 中实际用到的 embedding 向量做正则, 而非整个 weight 矩阵。
        公式: λ/2 * (mean(||u_emb||^2) + mean(||i_emb||^2))
        """
        if self.embedding_regularizer <= 0:
            return torch.tensor(0.0, device=i_emb.device)
        terms = [i_emb.norm(2, dim=-1).pow(2).mean()]
        if u_emb is not None:
            terms.append(u_emb.norm(2, dim=-1).pow(2).mean())
        reg = self.embedding_regularizer / 2.0 * sum(terms)
        return reg

    def _tokenize(self, user_ids, item_ids, item_vis,
                  pos_items, pos_lens, neg_items, neg_lens,
                  pos_items_vis, neg_items_vis,
                  cate_ids=None, pos_cates=None, neg_cates=None,
                  extra_cat_ids=None, extra_seq_ids=None, extra_seq_lens=None,
                  numeric_vals=None):
        """
        公共前处理 (与 BARS benchmark 对齐):
        embedding → vis_proj → DIN双通道序列编码(仅pos) → concat → pad → chunk → proj → [B, T, D]
        neg 序列不参与建模 (与 BARS benchmark 所有模型一致)

        Returns: (tokens [B, T, D], u_emb [B, E], i_emb [B, E])
        """
        u_emb = self.user_emb(user_ids) if self.use_user_feature else None
        i_emb = self.item_emb(item_ids)       # [B, item_emb_dim]
        c_emb = self.cate_emb(cate_ids) if self.use_category else None
        extra_cat_embs = self._extra_cat_embedding_dict(extra_cat_ids)

        seq_parts = []
        target_parts = []
        seq_mask = None
        if self.use_item_sequence:
            pos_id_emb = self.item_emb(pos_items)
            seq_parts.append(pos_id_emb)
            target_parts.append(i_emb)
            seq_mask = torch.arange(pos_items.size(1), device=pos_items.device).unsqueeze(0) < pos_lens.unsqueeze(1)

        if self.use_item_sequence and self.use_pretrained:
            # 投影 target item visual embedding
            item_vis_proj = self.vis_proj(item_vis)  # [B, vis_dim]

            # DIN: pos 双通道 (item_id_emb + item_vis_emb)
            pos_vis_proj = self.vis_proj(pos_items_vis)      # [B, S, vis_dim]
            seq_parts.append(pos_vis_proj)
            target_parts.append(item_vis_proj)
        else:
            item_vis_proj = None

        if self.use_category and pos_cates is not None and self.cfg_data_has_pos_cate:
            seq_parts.append(self.cate_emb(pos_cates))
            target_parts.append(c_emb)
            if seq_mask is None:
                seq_mask = torch.arange(pos_cates.size(1), device=pos_cates.device).unsqueeze(0) < pos_lens.unsqueeze(1)

        if self.extra_seq_configs:
            for idx, seq_cfg in enumerate(self.extra_seq_configs):
                ids = extra_seq_ids[:, idx, :]
                lens = extra_seq_lens[:, idx]
                seq_parts.append(self._lookup_shared_seq_embedding(seq_cfg["share_embedding"], ids))
                target_parts.append(self._lookup_shared_target_embedding(
                    seq_cfg["share_embedding"], i_emb, c_emb, extra_cat_embs,
                ))
                mask_i = torch.arange(ids.size(1), device=ids.device).unsqueeze(0) < lens.unsqueeze(1)
                seq_mask = mask_i if seq_mask is None else (seq_mask | mask_i)

        pos_seq_emb = torch.cat(seq_parts, dim=-1)
        target_seq = torch.cat(target_parts, dim=-1)

        pos_enc = self.pos_din_attn(pos_seq_emb, target_seq, seq_mask)  # [B, seq_emb_dim]

        parts = []
        if self.use_user_feature:
            parts.append(u_emb)
        parts.append(i_emb)
        if self.use_pretrained:
            parts.append(item_vis_proj)
        if self.use_category:
            parts.append(c_emb)
        parts.extend(extra_cat_embs[name] for name in self.extra_cat_names)
        if self.numeric_cols:
            for idx, name in enumerate(self.numeric_cols):
                parts.append(self.numeric_projs[name](numeric_vals[:, idx:idx + 1]))
        parts.append(pos_enc)
        e_input = torch.cat(parts, dim=-1)

        B = e_input.size(0)
        if e_input.size(-1) < self.padded_dim:
            e_input = F.pad(e_input, (0, self.padded_dim - e_input.size(-1)))

        tokens = e_input.view(B, self.num_tokens, self.chunk_size)
        return self.proj(tokens), u_emb, i_emb  # [B, T, D], [B, E], [B, E]


# ============================================================
# RankMixer CTR 模型
# ============================================================

class RankMixerCTR(BaseCTR):
    def __init__(self, cfg):
        super().__init__(cfg)
        model_cfg = cfg["model"]
        T = self.num_tokens
        hidden_dim = self.hidden_dim
        self._gradient_checkpointing = False

        # mode 二选一: "dense" 或 "moe"
        mode = model_cfg.get("mode", "dense").lower()
        assert mode in ("dense", "moe"), \
            f"RankMixer model.mode must be 'dense' or 'moe', got '{mode}'"
        num_layers = model_cfg["num_layers"]
        ffn_exp = model_cfg["ffn_expansion"]

        # P3: output_head MLP (BARS 风格: [1024,512,256] + Dice + Dropout)
        output_head_units = model_cfg.get("output_head_units", [1024, 512, 256])
        head_activation = model_cfg.get("head_activation", "Dice")
        head_dropout = model_cfg.get("head_dropout", 0.1)

        print(f"  [RankMixer] Feature dim: {self.total_dim}, Tokens (T): {T}, "
              f"Hidden (D): {hidden_dim}, D/T: {hidden_dim // T}")

        self.mode = mode
        if mode == "dense":
            print(f"  Mode: dense, Layers: {num_layers}, FFN expansion: {ffn_exp}")
            self.blocks = nn.ModuleList([
                RankMixerBlock(T, hidden_dim, ffn_exp) for _ in range(num_layers)
            ])
        else:  # moe
            num_experts = model_cfg["num_experts"]
            l1_lambda = model_cfg["l1_lambda"]
            print(f"  Mode: moe, Layers: {num_layers}, Experts: {num_experts}, "
                  f"FFN expansion: {ffn_exp}, L1 λ: {l1_lambda}")
            self.blocks = nn.ModuleList([
                RankMixerMoEBlock(T, hidden_dim, num_experts, ffn_exp, l1_lambda)
                for _ in range(num_layers)
            ])

        # P3: BARS 风格预测头 (MLPBlock + Dice + Dropout → Linear(1))
        self.output_head = _build_output_head(
            hidden_dim, output_head_units, head_activation, head_dropout
        )

    def enable_gradient_checkpointing(self):
        self._gradient_checkpointing = True

    def forward(self, user_ids, item_ids, item_vis,
                pos_items, pos_lens, neg_items, neg_lens,
                pos_items_vis, neg_items_vis,
                cate_ids=None, pos_cates=None, neg_cates=None,
                extra_cat_ids=None, extra_seq_ids=None, extra_seq_lens=None,
                numeric_vals=None):
        x, u_emb, i_emb = self._tokenize(user_ids, item_ids, item_vis,
                           pos_items, pos_lens, neg_items, neg_lens,
                           pos_items_vis, neg_items_vis,
                            cate_ids, pos_cates, neg_cates,
                            extra_cat_ids, extra_seq_ids, extra_seq_lens,
                            numeric_vals)

        total_reg = torch.tensor(0.0, device=x.device)
        if self.mode == "dense":
            for blk in self.blocks:
                if self._gradient_checkpointing and self.training:
                    x = ckpt_fn(blk, x, use_reentrant=False)
                else:
                    x = blk(x)
        else:  # moe
            for blk in self.blocks:
                if self._gradient_checkpointing and self.training:
                    x, reg = ckpt_fn(blk, x, use_reentrant=False)
                else:
                    x, reg = blk(x)
                total_reg = total_reg + reg

        # P0: batch-level embedding L2 正则
        total_reg = total_reg + self._get_embedding_reg_loss(u_emb, i_emb)

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
        self._gradient_checkpointing = False

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

        # P3: BARS 风格预测头 (主输出)
        output_head_units = model_cfg.get("output_head_units", [1024, 512, 256])
        head_activation = model_cfg.get("head_activation", "Dice")
        head_dropout = model_cfg.get("head_dropout", 0.1)
        self.output_head = _build_output_head(
            hidden_dim, output_head_units, head_activation, head_dropout
        )

    def enable_gradient_checkpointing(self):
        self._gradient_checkpointing = True

    def forward(self, user_ids, item_ids, item_vis,
                pos_items, pos_lens, neg_items, neg_lens,
                pos_items_vis, neg_items_vis):
        tokens, u_emb, i_emb = self._tokenize(user_ids, item_ids, item_vis,
                                pos_items, pos_lens, neg_items, neg_lens,
                                pos_items_vis, neg_items_vis)
        B = tokens.size(0)

        global_t = self.global_token.expand(B, -1, -1)
        x = torch.cat([global_t, tokens], dim=1)

        aux_logits_list = []
        residual_cache = x

        for i, block in enumerate(self.blocks):
            if self._gradient_checkpointing and self.training:
                x = ckpt_fn(block, x, use_reentrant=False)
            else:
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

        # P0: 将 batch-level embedding reg 加到 aux_logits 的 "meta" 中
        # 训练循环会单独处理 tokenmixer_large 的 aux_logits (用于辅助 loss)
        # 所以这里返回第三个值: embedding reg loss
        emb_reg = self._get_embedding_reg_loss(u_emb, i_emb)

        return main_logits, aux_logits, emb_reg


# ============================================================
# Vanilla Transformer CTR 模型 (Baseline)
# 标准 Multi-Head Self-Attention + FFN, 作为最简对照组
# ============================================================

class TransformerCTR(BaseCTR):
    """
    最简单的 Transformer baseline:
    - 继承 BaseCTR 复用 embedding + DIN 序列编码 + chunk tokenization
    - 标准 Pre-LN Transformer: LayerNorm → MHSA → residual → LayerNorm → FFN → residual
    - Mean pooling → MLP 预测头
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        model_cfg = cfg["model"]
        T = self.num_tokens
        hidden_dim = self.hidden_dim
        self._gradient_checkpointing = False

        num_layers = model_cfg["num_layers"]
        num_heads = model_cfg["num_heads"]
        ffn_expansion = model_cfg.get("ffn_expansion", 4)
        dropout = model_cfg.get("dropout", 0.0)

        # P3: output_head MLP (BARS 风格)
        output_head_units = model_cfg.get("output_head_units", [1024, 512, 256])
        head_activation = model_cfg.get("head_activation", "Dice")
        head_dropout = model_cfg.get("head_dropout", 0.1)

        print(f"  [Transformer] Feature dim: {self.total_dim}, Tokens (T): {T}, "
              f"Hidden (D): {hidden_dim}, Heads: {num_heads}, "
              f"Head dim: {hidden_dim // num_heads}")
        print(f"  Layers: {num_layers}, FFN expansion: {ffn_expansion}, "
              f"Dropout: {dropout}")

        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * ffn_expansion,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,  # Pre-LN
            )
            for _ in range(num_layers)
        ])

        # P3: BARS 风格预测头
        self.output_head = _build_output_head(
            hidden_dim, output_head_units, head_activation, head_dropout
        )

    def enable_gradient_checkpointing(self):
        self._gradient_checkpointing = True

    def forward(self, user_ids, item_ids, item_vis,
                pos_items, pos_lens, neg_items, neg_lens,
                pos_items_vis, neg_items_vis):
        x, u_emb, i_emb = self._tokenize(user_ids, item_ids, item_vis,
                           pos_items, pos_lens, neg_items, neg_lens,
                           pos_items_vis, neg_items_vis)
        for layer in self.layers:
            if self._gradient_checkpointing and self.training:
                x = ckpt_fn(layer, x, use_reentrant=False)
            else:
                x = layer(x)
        x = x.mean(dim=1)
        logits = self.output_head(x).squeeze(-1)
        reg_loss = self._get_embedding_reg_loss(u_emb, i_emb)
        return logits, reg_loss


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
        self._gradient_checkpointing = False

        user_emb_dim = emb_cfg["user_emb_dim"]
        item_emb_dim = emb_cfg["item_emb_dim"]
        self.use_pretrained = data_cfg.get("use_pretrained_emb", True)
        item_vis_dim = data_cfg.get("item_vis_dim", 0) if self.use_pretrained else 0
        cate_emb_dim = emb_cfg.get("cate_emb_dim", item_emb_dim)
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

        # P2: xavier_uniform_ 初始化 embedding (对齐 FuxiCTR)
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)

        # Pretrained embedding 投影层 (与 FuxiCTR 对齐)
        if self.use_pretrained:
            self.vis_proj = nn.Linear(item_vis_dim, item_vis_dim, bias=False)

        # Action embedding: click=0, like=1, follow=2, skip/neg=3
        num_action_types = 4
        self.action_emb = nn.Embedding(num_action_types, item_emb_dim)
        nn.init.xavier_uniform_(self.action_emb.weight)

        # 投影层: content token (item_emb + item_vis → hidden_dim)
        self.content_proj = nn.Linear(item_emb_dim + item_vis_dim, hidden_dim)
        # 投影层: action token (item_emb_dim → hidden_dim)
        self.action_proj = nn.Linear(item_emb_dim, hidden_dim)
        # 投影层: user prefix token (user_emb → hidden_dim, 无 is_like/is_follow)
        self.user_proj = nn.Linear(user_emb_dim, hidden_dim)

        # 序列最长: 1 (user prefix) + max_seq_len * 2 (pos content+action) + 1 (target)
        # 与 BARS benchmark 对齐: neg 序列不参与建模
        max_tokens = 1 + max_seq_len * 2 + 1 + 8  # 留余量
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

        # P3: BARS 风格预测头 (可配置)
        output_head_units = model_cfg.get("output_head_units", [1024, 512, 256])
        head_activation = model_cfg.get("head_activation", "Dice")
        head_dropout = model_cfg.get("head_dropout", 0.1)
        self.output_head = _build_output_head(
            hidden_dim, output_head_units, head_activation, head_dropout
        )

    def _get_embedding_reg_loss(self, u_emb, i_emb):
        """P0: batch-level embedding L2 正则损失"""
        if self.embedding_regularizer <= 0:
            return torch.tensor(0.0, device=u_emb.device)
        reg = self.embedding_regularizer / 2.0 * (
            u_emb.norm(2, dim=-1).pow(2).mean() +
            i_emb.norm(2, dim=-1).pow(2).mean()
        )
        return reg

    def enable_gradient_checkpointing(self):
        self._gradient_checkpointing = True

    def forward(self, user_ids, item_ids, item_vis,
                pos_items, pos_lens, neg_items, neg_lens,
                pos_items_vis, neg_items_vis):
        """
        构造 Content-Action 交替序列并通过 HSTU Layers。
        与 BARS benchmark 对齐: 只使用 pos 序列, neg 序列不参与建模。
        序列: [user_prefix, pos_Φ_0, pos_a_0, ..., Φ_target]
        取最后一个 token (target content) 的输出做预测。
        """
        B = user_ids.size(0)
        device = user_ids.device
        S = pos_items.size(1)  # max_seq_len

        # --- 构造各种 token ---
        u_emb = self.user_emb(user_ids)                            # [B, user_emb_dim]
        user_token = self.user_proj(u_emb)

        pos_item_emb = self.item_emb(pos_items)                    # [B, S, item_emb_dim]
        if self.use_pretrained:
            pos_vis_proj = self.vis_proj(pos_items_vis)             # [B, S, vis_dim]
            pos_content_input = torch.cat([pos_item_emb, pos_vis_proj], dim=-1)
        else:
            pos_content_input = pos_item_emb
        pos_content_tokens = self.content_proj(pos_content_input)  # [B, S, D]

        pos_action_ids = torch.zeros(B, S, dtype=torch.long, device=device)
        pos_action_tokens = self.action_proj(self.action_emb(pos_action_ids))  # [B, S, D]

        i_emb = self.item_emb(item_ids)                            # [B, item_emb_dim]
        if self.use_pretrained:
            target_vis = self.vis_proj(item_vis)                   # [B, vis_dim]
            target_token = self.content_proj(torch.cat([i_emb, target_vis], dim=-1))
        else:
            target_token = self.content_proj(i_emb)

        # --- 构造交替序列 (仅 pos) ---
        pos_interleaved = torch.stack([pos_content_tokens, pos_action_tokens], dim=2)
        pos_interleaved = pos_interleaved.view(B, S * 2, self.hidden_dim)

        full_seq = torch.cat([
            user_token.unsqueeze(1),       # [B, 1, D]
            pos_interleaved,               # [B, 2S, D]
            target_token.unsqueeze(1),     # [B, 1, D]
        ], dim=1)  # [B, 2S+2, D]

        # --- 构造 attention mask ---
        valid_lens = 1 + pos_lens * 2 + 1  # [B]
        total_len = full_seq.size(1)       # 2S+2

        pos_indices = torch.arange(total_len, device=device).unsqueeze(0)
        padding_mask = pos_indices < valid_lens.unsqueeze(1)

        attn_mask = padding_mask.unsqueeze(1).unsqueeze(2)
        attn_mask = attn_mask.expand(B, 1, total_len, total_len)

        # --- HSTU Layers ---
        x = full_seq
        for layer in self.layers:
            if self._gradient_checkpointing and self.training:
                x = ckpt_fn(layer, x, attn_mask, use_reentrant=False)
            else:
                x = layer(x, attn_mask=attn_mask)

        # --- 取 target 位置的输出 ---
        target_idx = (valid_lens - 1).unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1]
        target_idx = target_idx.expand(B, 1, self.hidden_dim)     # [B, 1, D]
        target_hidden = x.gather(1, target_idx).squeeze(1)         # [B, D]

        logits = self.output_head(target_hidden).squeeze(-1)
        # P0: batch-level embedding reg
        reg_loss = self._get_embedding_reg_loss(u_emb, i_emb)
        return logits, reg_loss


# ============================================================
# HiFormer CTR 模型 (arXiv:2311.05884)
# 异质自注意力: 通过 φ(Q) 和 ψ(K) 不同变换捕捉异质性特征交互
# ============================================================

class HiFormerCTR(BaseCTR):
    """
    HiFormer CTR 模型

    核心创新: 异质自注意力层 (Heterogeneous Self-Attention)
    - Q/K 通过不同的变换 φ() 和 ψ() 处理
    - 使用 Depthwise Conv + MLP 实现异质性变换
    - 允许捕捉不同类型特征之间的非对称交互

    架构:
    - 继承 BaseCTR: 复用 embedding + DIN 序列编码 + chunk tokenization
    - L 层 HiFormerLayer (异质自注意力 + FFN)
    - Mean pooling → MLP 预测头
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        from .hiformer import HiFormerLayer

        model_cfg = cfg["model"]
        T = self.num_tokens
        hidden_dim = self.hidden_dim
        self._gradient_checkpointing = False

        num_layers = model_cfg["num_layers"]
        num_heads = model_cfg["num_heads"]
        ffn_expansion = model_cfg.get("ffn_expansion", 4)
        dropout = model_cfg.get("dropout", 0.0)
        transform_type = model_cfg.get("transform_type", "dwconv_mlp")

        # P3: output_head MLP (BARS 风格)
        output_head_units = model_cfg.get("output_head_units", [1024, 512, 256])
        head_activation = model_cfg.get("head_activation", "Dice")
        head_dropout = model_cfg.get("head_dropout", 0.1)

        print(f"  [HiFormer] Feature dim: {self.total_dim}, Tokens (T): {T}, "
              f"Hidden (D): {hidden_dim}, Heads: {num_heads}, "
              f"Head dim: {hidden_dim // num_heads}")
        print(f"  Layers: {num_layers}, FFN expansion: {ffn_expansion}, "
              f"Dropout: {dropout}, Transform: {transform_type}")

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

        # P3: BARS 风格预测头
        self.output_head = _build_output_head(
            hidden_dim, output_head_units, head_activation, head_dropout
        )

    def enable_gradient_checkpointing(self):
        self._gradient_checkpointing = True

    def forward(self, user_ids, item_ids, item_vis,
                pos_items, pos_lens, neg_items, neg_lens,
                pos_items_vis, neg_items_vis):
        x, u_emb, i_emb = self._tokenize(user_ids, item_ids, item_vis,
                           pos_items, pos_lens, neg_items, neg_lens,
                           pos_items_vis, neg_items_vis)

        for layer in self.layers:
            if self._gradient_checkpointing and self.training:
                x = ckpt_fn(layer, x, use_reentrant=False)
            else:
                x = layer(x)

        x = x.mean(dim=1)
        logits = self.output_head(x).squeeze(-1)
        reg_loss = self._get_embedding_reg_loss(u_emb, i_emb)
        return logits, reg_loss


# ============================================================
# HyFormer CTR 模型 (ByteDance, arXiv:2601.12681)
# Query Decoding + Query Boosting
# ============================================================

class HyFormerCTR(BaseCTR):
    """
    ByteDance HyFormer adapted to the KuaiVideo CTR pipeline.

    It keeps non-sequential features as tokens, keeps the positive behavior
    sequence as sequence tokens, and alternates:
    - Query Decoding: global query tokens cross-attend to sequence K/V.
    - Query Boosting: RankMixer-style mixing over queries and NS tokens.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        from .hyformer import HyFormerBlock, QueryGenerator

        data_cfg = cfg["data"]
        emb_cfg = cfg["embedding"]
        model_cfg = cfg["model"]

        hidden_dim = self.hidden_dim
        chunk_size = self.chunk_size
        user_emb_dim = emb_cfg["user_emb_dim"]
        item_emb_dim = emb_cfg["item_emb_dim"]
        item_vis_dim = data_cfg.get("item_vis_dim", 0) if self.use_pretrained else 0
        cate_emb_dim = emb_cfg.get("cate_emb_dim", item_emb_dim)
        numeric_emb_dim = emb_cfg.get("numeric_emb_dim", item_emb_dim)

        seq_input_dim = 0
        if self.use_item_sequence:
            seq_input_dim += item_emb_dim
        if self.use_item_sequence and self.use_pretrained:
            seq_input_dim += item_vis_dim
        if self.use_category and self.cfg_data_has_pos_cate:
            seq_input_dim += cate_emb_dim
        for c in self.extra_seq_configs:
            share = c["share_embedding"]
            if share == "item_id":
                seq_input_dim += item_emb_dim
            elif share == "cate_id":
                seq_input_dim += cate_emb_dim
            else:
                seq_input_dim += self.extra_cat_embs[share].embedding_dim

        self._gradient_checkpointing = False
        self.num_query_tokens = model_cfg.get("num_query_tokens", 6)
        self.nonseq_dim = item_emb_dim
        if self.use_user_feature:
            self.nonseq_dim += user_emb_dim
        if self.use_pretrained:
            self.nonseq_dim += item_vis_dim
        if self.use_category:
            self.nonseq_dim += cate_emb_dim
        self.nonseq_dim += sum(self.extra_cat_embs[name].embedding_dim for name in self.extra_cat_names)
        self.nonseq_dim += len(self.numeric_cols) * numeric_emb_dim
        self.num_nonseq_tokens = math.ceil(self.nonseq_dim / chunk_size)
        self.nonseq_padded_dim = self.num_nonseq_tokens * chunk_size

        num_layers = model_cfg["num_layers"]
        num_heads = model_cfg["num_heads"]
        ffn_expansion = model_cfg.get("ffn_expansion", 4)
        seq_ffn_expansion = model_cfg.get("seq_ffn_expansion", 2)
        dropout = model_cfg.get("dropout", 0.0)

        # Reuse BaseCTR.proj for non-sequential chunk projection.
        self.seq_proj = nn.Linear(seq_input_dim, hidden_dim)
        self.query_generator = QueryGenerator(
            self.num_nonseq_tokens,
            self.num_query_tokens,
            hidden_dim,
            dropout=dropout,
        )
        self.blocks = nn.ModuleList([
            HyFormerBlock(
                num_query_tokens=self.num_query_tokens,
                num_nonseq_tokens=self.num_nonseq_tokens,
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                ffn_expansion=ffn_expansion,
                seq_ffn_expansion=seq_ffn_expansion,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        output_head_units = model_cfg.get("output_head_units", [1024, 512, 256])
        head_activation = model_cfg.get("head_activation", "Dice")
        head_dropout = model_cfg.get("head_dropout", 0.1)
        self.output_head = _build_output_head(
            hidden_dim, output_head_units, head_activation, head_dropout
        )

        boost_tokens = self.num_query_tokens + self.num_nonseq_tokens
        print(f"  [HyFormer] NS dim: {self.nonseq_dim}, NS tokens: {self.num_nonseq_tokens}, "
              f"Query tokens: {self.num_query_tokens}, Boost tokens: {boost_tokens}")
        print(f"  Hidden (D): {hidden_dim}, Heads: {num_heads}, Layers: {num_layers}, "
              f"FFN expansion: {ffn_expansion}, Seq FFN expansion: {seq_ffn_expansion}")

    def enable_gradient_checkpointing(self):
        self._gradient_checkpointing = True

    def _tokenize_nonseq_and_seq(self, user_ids, item_ids, item_vis,
                                 pos_items, pos_lens, neg_items, neg_lens,
                                 pos_items_vis, neg_items_vis,
                                 cate_ids=None, pos_cates=None, neg_cates=None,
                                 extra_cat_ids=None, extra_seq_ids=None, extra_seq_lens=None,
                                 numeric_vals=None):
        u_emb = self.user_emb(user_ids) if self.use_user_feature else None
        i_emb = self.item_emb(item_ids)
        c_emb = self.cate_emb(cate_ids) if self.use_category else None
        extra_cat_embs = self._extra_cat_embedding_dict(extra_cat_ids)

        nonseq_parts = []
        if self.use_user_feature:
            nonseq_parts.append(u_emb)
        nonseq_parts.append(i_emb)

        seq_parts = []
        seq_mask = None
        if self.use_item_sequence:
            pos_id_emb = self.item_emb(pos_items)
            seq_parts.append(pos_id_emb)
            seq_mask = (
                torch.arange(pos_items.size(1), device=pos_items.device).unsqueeze(0)
                < pos_lens.unsqueeze(1)
            )

        if self.use_item_sequence and self.use_pretrained:
            item_vis_proj = self.vis_proj(item_vis)
            pos_vis_proj = self.vis_proj(pos_items_vis)
            seq_parts.append(pos_vis_proj)
            nonseq_parts.append(item_vis_proj)

        if self.use_category:
            nonseq_parts.append(c_emb)
            if self.cfg_data_has_pos_cate:
                seq_parts.append(self.cate_emb(pos_cates))
                if seq_mask is None:
                    seq_mask = torch.arange(pos_cates.size(1), device=pos_cates.device).unsqueeze(0) < pos_lens.unsqueeze(1)

        for name in self.extra_cat_names:
            nonseq_parts.append(extra_cat_embs[name])
        for idx, name in enumerate(self.numeric_cols):
            nonseq_parts.append(self.numeric_projs[name](numeric_vals[:, idx:idx + 1]))

        if self.extra_seq_configs:
            for idx, seq_cfg in enumerate(self.extra_seq_configs):
                ids = extra_seq_ids[:, idx, :]
                lens = extra_seq_lens[:, idx]
                seq_parts.append(self._lookup_shared_seq_embedding(seq_cfg["share_embedding"], ids))
                mask_i = torch.arange(ids.size(1), device=ids.device).unsqueeze(0) < lens.unsqueeze(1)
                seq_mask = mask_i if seq_mask is None else (seq_mask | mask_i)

        nonseq_input = torch.cat(nonseq_parts, dim=-1)
        seq_raw = torch.cat(seq_parts, dim=-1)

        B = nonseq_input.size(0)
        if nonseq_input.size(-1) < self.nonseq_padded_dim:
            nonseq_input = F.pad(
                nonseq_input,
                (0, self.nonseq_padded_dim - nonseq_input.size(-1)),
            )
        nonseq_chunks = nonseq_input.view(B, self.num_nonseq_tokens, self.chunk_size)
        nonseq_tokens = self.proj(nonseq_chunks)

        seq_tokens = self.seq_proj(seq_raw)
        seq_tokens = seq_tokens * seq_mask.unsqueeze(-1).to(seq_tokens.dtype)
        return nonseq_tokens, seq_tokens, seq_mask, u_emb, i_emb

    def forward(self, user_ids, item_ids, item_vis,
                pos_items, pos_lens, neg_items, neg_lens,
                pos_items_vis, neg_items_vis,
                cate_ids=None, pos_cates=None, neg_cates=None,
                extra_cat_ids=None, extra_seq_ids=None, extra_seq_lens=None,
                numeric_vals=None):
        nonseq_tokens, seq_tokens, seq_mask, u_emb, i_emb = self._tokenize_nonseq_and_seq(
            user_ids, item_ids, item_vis,
            pos_items, pos_lens, neg_items, neg_lens,
            pos_items_vis, neg_items_vis,
            cate_ids, pos_cates, neg_cates,
            extra_cat_ids, extra_seq_ids, extra_seq_lens, numeric_vals,
        )

        queries = self.query_generator(nonseq_tokens, seq_tokens, seq_mask)
        for block in self.blocks:
            if self._gradient_checkpointing and self.training:
                queries, nonseq_tokens, seq_tokens = ckpt_fn(
                    block, queries, nonseq_tokens, seq_tokens, seq_mask,
                    use_reentrant=False,
                )
            else:
                queries, nonseq_tokens, seq_tokens = block(
                    queries, nonseq_tokens, seq_tokens, seq_mask
                )

        logits = self.output_head(queries.mean(dim=1)).squeeze(-1)
        reg_loss = self._get_embedding_reg_loss(u_emb, i_emb)
        return logits, reg_loss


# ============================================================
# InterFormer CTR 模型 (Meta, arXiv:2411.09852)
# Interaction Arch + Sequence Arch + Cross Arch
# ============================================================

class InterFormerCTR(BaseCTR):
    """
    InterFormer adapted to the KuaiVideo CTR pipeline.

    The model preserves non-sequence and sequence tokens through stacked blocks,
    while exchanging compact gated summaries between both modes at every layer.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        from .interformer import InterFormerBlock, TokenSummaryGate

        data_cfg = cfg["data"]
        emb_cfg = cfg["embedding"]
        model_cfg = cfg["model"]

        hidden_dim = self.hidden_dim
        user_emb_dim = emb_cfg["user_emb_dim"]
        item_emb_dim = emb_cfg["item_emb_dim"]
        cate_emb_dim = emb_cfg.get("cate_emb_dim", item_emb_dim)
        numeric_emb_dim = emb_cfg.get("numeric_emb_dim", item_emb_dim)
        item_vis_dim = data_cfg.get("item_vis_dim", 0) if self.use_pretrained else 0

        self._gradient_checkpointing = False
        self.use_neg_sequence = model_cfg.get("use_neg_sequence", True)
        self.num_nonseq_tokens = 1
        if self.use_user_feature:
            self.num_nonseq_tokens += 1
        if self.use_pretrained:
            self.num_nonseq_tokens += 1
        if self.use_category:
            self.num_nonseq_tokens += 1
        self.num_nonseq_tokens += len(self.extra_cat_names)
        self.num_nonseq_tokens += len(self.numeric_cols)

        num_layers = model_cfg["num_layers"]
        num_heads = model_cfg["num_heads"]
        ffn_expansion = model_cfg.get("ffn_expansion", 4)
        dropout = model_cfg.get("dropout", 0.0)
        num_nonseq_summary_tokens = model_cfg.get("num_nonseq_summary_tokens", 1)
        num_cls_tokens = model_cfg.get("num_cls_tokens", 4)
        num_pma_tokens = model_cfg.get("num_pma_tokens", 2)
        num_recent_tokens = model_cfg.get("num_recent_tokens", 2)
        dhen_layers = model_cfg.get("dhen_layers", 3)
        dcn_rank = model_cfg.get("dcn_rank", 32)
        max_seq_len = data_cfg.get("max_seq_len", 100)

        if self.use_user_feature:
            self.user_token_proj = nn.Linear(user_emb_dim, hidden_dim)
        self.item_token_proj = nn.Linear(item_emb_dim, hidden_dim)
        if self.use_pretrained:
            self.item_vis_token_proj = nn.Linear(item_vis_dim, hidden_dim)
        if self.use_category:
            self.cate_token_proj = nn.Linear(cate_emb_dim, hidden_dim)
        self.extra_cat_token_proj = nn.ModuleDict({
            name: nn.Linear(self.extra_cat_embs[name].embedding_dim, hidden_dim)
            for name in self.extra_cat_names
        })
        self.numeric_token_proj = nn.ModuleDict({
            name: nn.Linear(numeric_emb_dim, hidden_dim)
            for name in self.numeric_cols
        })

        if self.use_item_sequence:
            self.item_seq_proj = nn.Linear(item_emb_dim, hidden_dim)
        if self.use_pretrained:
            self.vis_seq_proj = nn.Linear(item_vis_dim, hidden_dim)
        if self.use_category and self.cfg_data_has_pos_cate:
            self.cate_seq_proj = nn.Linear(cate_emb_dim, hidden_dim)
        self.extra_seq_proj = nn.ModuleDict()
        for c in self.extra_seq_configs:
            share = c["share_embedding"]
            if share == "item_id":
                in_dim = item_emb_dim
            elif share == "cate_id":
                in_dim = cate_emb_dim
            else:
                in_dim = self.extra_cat_embs[share].embedding_dim
            self.extra_seq_proj[c["name"]] = nn.Linear(in_dim, hidden_dim)

        fields_per_sequence = 0
        if self.use_item_sequence:
            fields_per_sequence += 1
        if self.use_item_sequence and self.use_pretrained:
            fields_per_sequence += 1
        if self.use_category and self.cfg_data_has_pos_cate:
            fields_per_sequence += 1
        num_seq_fields = fields_per_sequence * (2 if self.use_neg_sequence else 1)
        num_seq_fields += len(self.extra_seq_configs)
        self.seq_mask_net = nn.Sequential(
            nn.Linear(num_seq_fields * hidden_dim, num_seq_fields * hidden_dim),
            nn.SiLU(),
            nn.Linear(num_seq_fields * hidden_dim, num_seq_fields * hidden_dim),
            nn.Sigmoid(),
        )
        self.seq_lce = nn.Linear(num_seq_fields * hidden_dim, hidden_dim)
        self.initial_cls = TokenSummaryGate(
            self.num_nonseq_tokens,
            num_cls_tokens,
            hidden_dim,
        )
        self.blocks = nn.ModuleList([
            InterFormerBlock(
                num_nonseq_tokens=self.num_nonseq_tokens,
                num_nonseq_summary_tokens=num_nonseq_summary_tokens,
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                num_pma_tokens=num_pma_tokens,
                num_recent_tokens=num_recent_tokens,
                ffn_expansion=ffn_expansion,
                dhen_layers=dhen_layers,
                dcn_rank=dcn_rank,
                dropout=dropout,
                max_seq_len=max_seq_len,
                num_cls_tokens=num_cls_tokens,
            )
            for _ in range(num_layers)
        ])

        output_head_units = model_cfg.get("output_head_units", [1024, 512, 256])
        head_activation = model_cfg.get("head_activation", "Dice")
        head_dropout = model_cfg.get("head_dropout", 0.1)
        self.output_head = _build_output_head(
            hidden_dim * 2, output_head_units, head_activation, head_dropout
        )

        seq_summary_tokens = num_cls_tokens + num_pma_tokens + num_recent_tokens
        interaction_tokens = self.num_nonseq_tokens + seq_summary_tokens
        print(f"  [InterFormer] NS tokens: {self.num_nonseq_tokens}, CLS tokens: {num_cls_tokens}, "
              f"Seq summary tokens: {seq_summary_tokens}, Interaction tokens: {interaction_tokens}")
        print(f"  Hidden (D): {hidden_dim}, Heads: {num_heads}, Layers: {num_layers}, "
              f"PMA: {num_pma_tokens}, Recent: {num_recent_tokens}, "
              f"DHEN layers: {dhen_layers}, FFN expansion: {ffn_expansion}")

    def enable_gradient_checkpointing(self):
        self._gradient_checkpointing = True

    def _tokenize_nonseq_and_seq(self, user_ids, item_ids, item_vis,
                                 pos_items, pos_lens, neg_items, neg_lens,
                                 pos_items_vis, neg_items_vis,
                                 cate_ids=None, pos_cates=None, neg_cates=None,
                                 extra_cat_ids=None, extra_seq_ids=None, extra_seq_lens=None,
                                 numeric_vals=None):
        u_emb = self.user_emb(user_ids) if self.use_user_feature else None
        i_emb = self.item_emb(item_ids)
        c_emb = self.cate_emb(cate_ids) if self.use_category else None
        extra_cat_embs = self._extra_cat_embedding_dict(extra_cat_ids)

        nonseq_tokens = []
        if self.use_user_feature:
            nonseq_tokens.append(self.user_token_proj(u_emb).unsqueeze(1))
        nonseq_tokens.append(self.item_token_proj(i_emb).unsqueeze(1))
        item_vis_proj = None
        if self.use_item_sequence and self.use_pretrained:
            item_vis_proj = self.vis_proj(item_vis)
            pos_id_emb = self.item_emb(pos_items)
            pos_vis_proj = self.vis_proj(pos_items_vis)
            neg_id_emb = self.item_emb(neg_items)
            neg_vis_proj = self.vis_proj(neg_items_vis)
            nonseq_tokens.append(self.item_vis_token_proj(item_vis_proj).unsqueeze(1))
        elif self.use_item_sequence:
            pos_id_emb = self.item_emb(pos_items)
            neg_id_emb = self.item_emb(neg_items)
        if self.use_category:
            nonseq_tokens.append(self.cate_token_proj(c_emb).unsqueeze(1))
            if self.cfg_data_has_pos_cate:
                pos_cate_emb = self.cate_emb(pos_cates)
                neg_cate_emb = self.cate_emb(neg_cates)
        for name in self.extra_cat_names:
            nonseq_tokens.append(self.extra_cat_token_proj[name](extra_cat_embs[name]).unsqueeze(1))
        for idx, name in enumerate(self.numeric_cols):
            num_emb = self.numeric_projs[name](numeric_vals[:, idx:idx + 1])
            nonseq_tokens.append(self.numeric_token_proj[name](num_emb).unsqueeze(1))
        nonseq_tokens = torch.cat(nonseq_tokens, dim=1)

        seq_fields = []
        seq_mask = None
        if self.use_item_sequence:
            pos_mask = (
                torch.arange(pos_items.size(1), device=pos_items.device).unsqueeze(0)
                < pos_lens.unsqueeze(1)
            )
            neg_mask = (
                torch.arange(neg_items.size(1), device=neg_items.device).unsqueeze(0)
                < neg_lens.unsqueeze(1)
            )
            seq_fields.append(self.item_seq_proj(pos_id_emb))
            seq_mask = pos_mask
        if self.use_item_sequence and self.use_pretrained:
            seq_fields.append(self.vis_seq_proj(pos_vis_proj))
        if self.use_category and self.cfg_data_has_pos_cate:
            seq_fields.append(self.cate_seq_proj(pos_cate_emb))
            if seq_mask is None:
                seq_mask = torch.arange(pos_cates.size(1), device=pos_cates.device).unsqueeze(0) < pos_lens.unsqueeze(1)
        if self.use_item_sequence and self.use_neg_sequence:
            seq_fields.append(self.item_seq_proj(neg_id_emb))
            seq_mask = seq_mask | neg_mask
            if self.use_pretrained:
                seq_fields.append(self.vis_seq_proj(neg_vis_proj))
            if self.use_category and self.cfg_data_has_pos_cate:
                seq_fields.append(self.cate_seq_proj(neg_cate_emb))
        if self.extra_seq_configs:
            for idx, seq_cfg in enumerate(self.extra_seq_configs):
                ids = extra_seq_ids[:, idx, :]
                lens = extra_seq_lens[:, idx]
                seq_fields.append(
                    self.extra_seq_proj[seq_cfg["name"]](
                        self._lookup_shared_seq_embedding(seq_cfg["share_embedding"], ids)
                    )
                )
                mask_i = torch.arange(ids.size(1), device=ids.device).unsqueeze(0) < lens.unsqueeze(1)
                seq_mask = mask_i if seq_mask is None else (seq_mask | mask_i)

        seq_cat = torch.cat(seq_fields, dim=-1)
        seq_tokens = self.seq_lce(seq_cat * self.seq_mask_net(seq_cat))
        seq_tokens = seq_tokens * seq_mask.unsqueeze(-1).to(seq_tokens.dtype)

        cls_token = self.initial_cls(nonseq_tokens)
        cls_mask = torch.ones(
            nonseq_tokens.size(0),
            cls_token.size(1),
            dtype=torch.bool,
            device=seq_mask.device,
        )
        seq_tokens = torch.cat([cls_token, seq_tokens], dim=1)
        seq_mask = torch.cat([cls_mask, seq_mask], dim=1)
        return nonseq_tokens, seq_tokens, seq_mask, u_emb, i_emb

    def forward(self, user_ids, item_ids, item_vis,
                pos_items, pos_lens, neg_items, neg_lens,
                pos_items_vis, neg_items_vis,
                cate_ids=None, pos_cates=None, neg_cates=None,
                extra_cat_ids=None, extra_seq_ids=None, extra_seq_lens=None,
                numeric_vals=None):
        nonseq_tokens, seq_tokens, seq_mask, u_emb, i_emb = self._tokenize_nonseq_and_seq(
            user_ids, item_ids, item_vis,
            pos_items, pos_lens, neg_items, neg_lens,
            pos_items_vis, neg_items_vis,
            cate_ids, pos_cates, neg_cates,
            extra_cat_ids, extra_seq_ids, extra_seq_lens, numeric_vals,
        )

        for block in self.blocks:
            if self._gradient_checkpointing and self.training:
                nonseq_tokens, seq_tokens = ckpt_fn(
                    block, nonseq_tokens, seq_tokens, seq_mask,
                    use_reentrant=False,
                )
            else:
                nonseq_tokens, seq_tokens = block(nonseq_tokens, seq_tokens, seq_mask)

        final_repr = torch.cat([nonseq_tokens.mean(dim=1), seq_tokens[:, 0]], dim=-1)
        logits = self.output_head(final_repr).squeeze(-1)
        reg_loss = self._get_embedding_reg_loss(u_emb, i_emb)
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
    elif arch == "hiformer":
        return HiFormerCTR(cfg)
    elif arch == "hyformer":
        return HyFormerCTR(cfg)
    elif arch == "interformer":
        return InterFormerCTR(cfg)
    elif arch == "dmin":
        from .dmin import DMINCTR
        return DMINCTR(cfg)
    else:
        return RankMixerCTR(cfg)
