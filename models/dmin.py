"""
DMIN (Deep Multi-Interest Network) 严格复现 FuxiCTR 实现。
对照 BARS benchmark DMIN_kuaivideo_x1 最佳配置:
  num_heads=2, use_behavior_refiner=False, use_pos_emb=False,
  attention_dropout=0.2, dnn=[1024,512,256], dnn_act=Dice,
  attention_hidden_units=[512,256], net_dropout=0.1,
  aux_loss_lambda=0, enable_sum_pooling=False

参考: https://github.com/reczoo/FuxiCTR/blob/v2.3.7/model_zoo/DMIN/src/DMIN.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Dice 激活函数 (严格复现 FuxiCTR)
# ============================================================

class Dice(nn.Module):
    """Data-dependent activation (DIN 论文)"""
    def __init__(self, input_dim, eps=1e-9):
        super().__init__()
        self.bn = nn.BatchNorm1d(input_dim, affine=False, momentum=0.01, eps=eps)
        self.alpha = nn.Parameter(torch.zeros(input_dim))

    def forward(self, x):
        # x: [*, input_dim]
        orig_shape = x.shape
        x_flat = x.view(-1, orig_shape[-1])
        p = torch.sigmoid(self.bn(x_flat))
        out = p * x_flat + self.alpha * (1 - p) * x_flat
        return out.view(orig_shape)


# ============================================================
# MLP Block (严格复现 FuxiCTR MLP_Block)
# ============================================================

class MLPBlock(nn.Module):
    def __init__(self, input_dim, hidden_units, hidden_activation="ReLU",
                 output_dim=None, output_activation=None,
                 dropout_rate=0.0, batch_norm=False):
        super().__init__()
        layers = []
        dims = [input_dim] + list(hidden_units)
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if batch_norm:
                layers.append(nn.BatchNorm1d(dims[i + 1]))
            if hidden_activation == "Dice":
                layers.append(Dice(dims[i + 1]))
            elif hidden_activation == "ReLU":
                layers.append(nn.ReLU())
            elif hidden_activation in ("SiLU", "Swish"):
                layers.append(nn.SiLU())
            elif hidden_activation == "PReLU":
                layers.append(nn.PReLU())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
        if output_dim is not None:
            layers.append(nn.Linear(dims[-1], output_dim))
        if output_activation == "Sigmoid":
            layers.append(nn.Sigmoid())
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


# ============================================================
# Scaled Dot-Product Attention (严格复现 FuxiCTR)
# ============================================================

class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout_rate=0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None

    def forward(self, query, key, value, scale=None, mask=None):
        # query/key/value: [B, num_heads, S, head_dim]
        scores = torch.matmul(query, key.transpose(-2, -1))
        if scale is not None:
            scores = scores / scale
        if mask is not None:
            scores = scores.masked_fill(~mask, -1e9)
        attn_weights = torch.softmax(scores, dim=-1)
        if self.dropout is not None:
            attn_weights = self.dropout(attn_weights)
        output = torch.matmul(attn_weights, value)
        return output, attn_weights


# ============================================================
# Target Attention (DIN-style, 严格复现 FuxiCTR TargetAttention)
# ============================================================

class TargetAttention(nn.Module):
    """
    FuxiCTR TargetAttention: 4路拼接 [target, seq, target-seq, target*seq] → MLP → softmax
    注: BARS 最佳配置 use_pos_emb=False
    """
    def __init__(self, model_dim, attention_hidden_units=(512, 256),
                 attention_activation="ReLU", attention_dropout=0.0):
        super().__init__()
        self.attn_mlp = MLPBlock(
            input_dim=model_dim * 4,
            hidden_units=attention_hidden_units,
            hidden_activation=attention_activation,
            output_dim=1,
            dropout_rate=attention_dropout,
            batch_norm=False,
        )

    def forward(self, sequence_emb, target_emb, mask=None):
        """
        sequence_emb: [B, S, D]
        target_emb:   [B, D]
        mask:         [B, S] bool, True=valid
        """
        seq_len = sequence_emb.size(1)
        target_exp = target_emb.unsqueeze(1).expand(-1, seq_len, -1)

        din_concat = torch.cat([
            target_exp, sequence_emb,
            target_exp - sequence_emb,
            target_exp * sequence_emb,
        ], dim=-1)  # [B, S, 4D]

        attn_score = self.attn_mlp(
            din_concat.view(-1, 4 * target_exp.size(-1))
        )  # [B*S, 1]
        attn_score = attn_score.view(-1, seq_len)  # [B, S]

        if mask is not None:
            attn_score = attn_score.masked_fill(~mask, -1e9)
        attn_score = torch.softmax(attn_score, dim=-1)

        output = (attn_score.unsqueeze(-1) * sequence_emb).sum(dim=1)  # [B, D]
        return output


# ============================================================
# Multi-Interest Extractor Layer (严格复现 FuxiCTR)
# ============================================================

class MultiInterestExtractorLayer(nn.Module):
    """
    FuxiCTR MultiInterestExtractorLayer:
    多头 QKV self-attention → 每头独立 W_o + LayerNorm + FFN + residual → TargetAttention
    """
    def __init__(self, model_dim, num_heads=2, attn_dropout=0.0, net_dropout=0.0,
                 layer_norm=True, attention_hidden_units=(512, 256),
                 attention_activation="ReLU"):
        super().__init__()
        assert model_dim % num_heads == 0
        self.head_dim = model_dim // num_heads
        self.num_heads = num_heads
        self.scale = self.head_dim ** 0.5

        self.W_qkv = nn.Linear(model_dim, 3 * model_dim, bias=False)
        self.attention = ScaledDotProductAttention(attn_dropout)

        # 每头独立的: W_o, dropout, layer_norm, FFN
        self.W_o = nn.ModuleList([
            nn.Linear(self.head_dim, model_dim, bias=False)
            for _ in range(num_heads)
        ])
        self.dropout = nn.ModuleList([
            nn.Dropout(net_dropout) for _ in range(num_heads)
        ]) if net_dropout > 0 else None
        self.layer_norm = nn.ModuleList([
            nn.LayerNorm(model_dim) for _ in range(num_heads)
        ]) if layer_norm else None
        ffn_dim = model_dim * 2
        self.ffn = nn.ModuleList([
            nn.Sequential(
                nn.Linear(model_dim, ffn_dim),
                nn.ReLU(),
                nn.Linear(ffn_dim, model_dim),
            ) for _ in range(num_heads)
        ])

        # 每头独立的 TargetAttention
        self.target_attention = nn.ModuleList([
            TargetAttention(
                model_dim,
                attention_hidden_units=attention_hidden_units,
                attention_activation=attention_activation,
                attention_dropout=attn_dropout,
            ) for _ in range(num_heads)
        ])

    def forward(self, sequence_emb, target_emb, attn_mask=None, pad_mask=None):
        """
        sequence_emb: [B, S, D]
        target_emb:   [B, D]
        attn_mask:    [B*H, S, S] bool, True=valid (用于 self-attention)
        pad_mask:     [B, S] bool, True=valid (用于 target attention)
        Returns: list of [B, D], length=num_heads
        """
        query, key, value = torch.chunk(self.W_qkv(sequence_emb), chunks=3, dim=-1)

        B = query.size(0)
        query = query.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        attn, _ = self.attention(query, key, value, scale=self.scale, mask=attn_mask)
        # attn: [B, H, S, head_dim]

        attn_heads = torch.chunk(attn, chunks=self.num_heads, dim=1)  # H x [B, 1, S, head_dim]
        interests = []
        for idx, h_head in enumerate(attn_heads):
            s = self.W_o[idx](h_head.squeeze(1))  # [B, S, D]
            if self.dropout is not None:
                s = self.dropout[idx](s)
            s = s + sequence_emb  # residual
            if self.layer_norm is not None:
                s = self.layer_norm[idx](s)
            head_out = self.ffn[idx](s)
            head_out = head_out + s  # residual
            interest_emb = self.target_attention[idx](head_out, target_emb, mask=pad_mask)
            interests.append(interest_emb)  # [B, D]
        return interests


# ============================================================
# DMIN CTR Model
# ============================================================

class DMINCTR(nn.Module):
    """
    严格复现 FuxiCTR DMIN (BARS 最佳配置):
    - aux_loss_lambda=0 → neg序列不参与模型
    - use_behavior_refiner=False, use_pos_emb=False, enable_sum_pooling=False
    - 只对 pos 序列做 Multi-Interest Extraction
    - 拼接: multi-interest outputs + 2D feature embeddings
    - DNN (Dice) → 预测
    - embedding L2 正则 (与 FuxiCTR 对齐)

    FuxiCTR feature_dim = 448:
      sum_emb_out_dim(448) + model_dim*(num_heads-1)(128) - neg_seq_emb(128) = 448
    forward concat: interests(2*128=256) + user(64) + item(64) + item_vis(64) = 448
    """

    def __init__(self, cfg):
        super().__init__()
        data_cfg = cfg["data"]
        emb_cfg = cfg["embedding"]
        model_cfg = cfg["model"]

        user_emb_dim = emb_cfg["user_emb_dim"]
        item_emb_dim = emb_cfg["item_emb_dim"]
        self.use_pretrained = data_cfg.get("use_pretrained_emb", True)
        item_vis_dim = data_cfg["item_vis_dim"] if self.use_pretrained else 0
        self.embedding_regularizer = data_cfg.get("embedding_regularizer", 0.0)

        # Embeddings
        self.user_emb = nn.Embedding(data_cfg["num_users"], user_emb_dim)
        self.item_emb = nn.Embedding(data_cfg["item_hash_size"], item_emb_dim)

        # P2: xavier_uniform_ 初始化 embedding (对齐 FuxiCTR)
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)

        if self.use_pretrained:
            self.vis_proj = nn.Linear(item_vis_dim, item_vis_dim, bias=False)

        # DMIN 配置
        num_heads = model_cfg.get("num_heads", 2)
        attn_dropout = model_cfg.get("attention_dropout", 0.2)
        net_dropout = model_cfg.get("net_dropout", 0.1)
        dnn_hidden_units = model_cfg.get("dnn_hidden_units", [1024, 512, 256])
        dnn_activation = model_cfg.get("dnn_activation", "Dice")
        attention_hidden_units = model_cfg.get("attention_hidden_units", [512, 256])
        attention_activation = model_cfg.get("attention_activation", "ReLU")

        # 序列 embedding 维度:
        # target_field = (item_id, item_emb) → model_dim = 64 + 64 = 128
        # sequence_field = (pos_items, pos_items_emb) → 也是 128
        seq_emb_dim = item_emb_dim + item_vis_dim if self.use_pretrained else item_emb_dim

        # 只对 pos 序列做 Multi-Interest Extraction (aux_loss_lambda=0, neg 不参与)
        self.multi_interest = MultiInterestExtractorLayer(
            model_dim=seq_emb_dim,
            num_heads=num_heads,
            attn_dropout=attn_dropout,
            net_dropout=net_dropout,
            layer_norm=True,
            attention_hidden_units=attention_hidden_units,
            attention_activation=attention_activation,
        )

        # feature_dim (严格与 FuxiCTR 对齐):
        # FuxiCTR 计算: sum_emb_out_dim + model_dim*(num_heads-1) - emb_dim*len(neg_seq_fields)
        # sum_emb_out_dim = 7个特征 * 64 = 448 (user_id, item_id, item_emb,
        #   pos_items, neg_items, pos_items_emb, neg_items_emb 各 64)
        # + model_dim * (num_heads - 1) = 128 * 1 = 128
        # - neg_seq_fields: 64 * 2 = 128 (neg_items + neg_items_emb)
        # = 448 + 128 - 128 = 448
        #
        # 在 forward 中, concat_emb 实际包含:
        #   1. multi-interest outputs (num_heads 个 model_dim 向量)
        #   2. 所有 ndim==2 的 feature embeddings (排除 neg_seq_field)
        # 我们的独立实现等价于:
        #   interests (num_heads * seq_emb_dim) + user(64) + item(64) + item_vis(64)
        # = 2*128 + 64*3 = 256 + 192 = 448
        interest_dim = num_heads * seq_emb_dim
        base_2d_dim = user_emb_dim + item_emb_dim
        if self.use_pretrained:
            base_2d_dim += item_vis_dim
        feature_dim = interest_dim + base_2d_dim

        print(f"  [DMIN] seq_emb_dim={seq_emb_dim}, num_heads={num_heads}, "
              f"feature_dim={feature_dim}")

        self.dnn = MLPBlock(
            input_dim=feature_dim,
            hidden_units=dnn_hidden_units,
            hidden_activation=dnn_activation,
            output_dim=1,
            output_activation=None,
            dropout_rate=net_dropout,
            batch_norm=False,
        )

        self.num_heads = num_heads

    def _get_embedding_reg_loss(self, u_emb, i_emb):
        """P0: batch-level embedding L2 正则"""
        if self.embedding_regularizer <= 0:
            return torch.tensor(0.0, device=u_emb.device)
        reg = self.embedding_regularizer / 2.0 * (
            u_emb.norm(2, dim=-1).pow(2).mean() +
            i_emb.norm(2, dim=-1).pow(2).mean()
        )
        return reg

    def _get_mask(self, seq_ids, seq_lens):
        """
        构造 padding mask 和 self-attention causal mask (严格复现 FuxiCTR)
        """
        B, S = seq_ids.shape
        device = seq_ids.device

        pad_mask = seq_ids.ne(0) & seq_lens.gt(0).unsqueeze(1)  # [B, S] True=valid

        # FuxiCTR attn_mask: padding + causal, 对角线保留
        padding_mask_2d = (~pad_mask).unsqueeze(1).expand(B, S, S)
        diag_zeros = ~torch.eye(S, device=device).bool().unsqueeze(0).expand(B, S, S)
        attn_mask_inv = padding_mask_2d & diag_zeros
        causal_mask = torch.triu(torch.ones(S, S, device=device), 1).bool().unsqueeze(0).expand(B, S, S)
        attn_mask_inv = attn_mask_inv | causal_mask
        attn_mask = ~attn_mask_inv

        attn_mask = attn_mask.unsqueeze(1).expand(B, self.num_heads, S, S)
        return pad_mask, attn_mask

    def forward(self, user_ids, item_ids, item_vis,
                pos_items, pos_lens, neg_items, neg_lens,
                pos_items_vis, neg_items_vis,
                cate_ids=None, pos_cates=None, neg_cates=None,
                extra_cat_ids=None, extra_seq_ids=None, extra_seq_lens=None,
                numeric_vals=None):
        # Embeddings
        u_emb = self.user_emb(user_ids)          # [B, 64]
        i_emb = self.item_emb(item_ids)          # [B, 64]

        # pos 序列 embeddings (双通道: ID + vis)
        pos_id_emb = self.item_emb(pos_items)    # [B, S, 64]

        if self.use_pretrained:
            item_vis_proj = self.vis_proj(item_vis)           # [B, 64]
            pos_vis = self.vis_proj(pos_items_vis)            # [B, S, 64]
            pos_seq_emb = torch.cat([pos_id_emb, pos_vis], dim=-1)  # [B, S, 128]
            target_seq = torch.cat([i_emb, item_vis_proj], dim=-1)  # [B, 128]
        else:
            item_vis_proj = None
            pos_seq_emb = pos_id_emb
            target_seq = i_emb

        # Masks
        pos_pad_mask, pos_attn_mask = self._get_mask(pos_items, pos_lens)

        # Multi-Interest Extraction (只对 pos 序列)
        B = user_ids.size(0)
        pos_interests = self.multi_interest(
            pos_seq_emb, target_seq,
            attn_mask=pos_attn_mask, pad_mask=pos_pad_mask,
        )  # list of [B, D], len=num_heads

        # Feature concatenation (严格与 FuxiCTR 对齐):
        # 1. multi-interest outputs (num_heads 个 model_dim 向量)
        # 2. 所有 ndim==2 的 feature embeddings (排除 neg_seq_field)
        concat_parts = []
        concat_parts.extend(pos_interests)
        concat_parts.append(u_emb)          # user_id: 64
        concat_parts.append(i_emb)          # item_id: 64
        if self.use_pretrained:
            concat_parts.append(item_vis_proj)  # item_emb: 64

        concat_emb = torch.cat(concat_parts, dim=-1)  # [B, feature_dim]

        # DNN prediction
        logits = self.dnn(concat_emb).squeeze(-1)  # [B]

        reg_loss = self._get_embedding_reg_loss(u_emb, i_emb)
        return logits, reg_loss
