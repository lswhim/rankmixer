"""
在 KuaiVideo_x1 数据集上进行 CTR 预测训练
支持三种模型架构:
  - RankMixer (arXiv:2507.15551)
  - TokenMixer-Large (arXiv:2602.06563)
  - HSTU (arXiv:2402.17152)
通过 config YAML 中的 model.arch 字段选择
"""

import os
import sys
import csv
import time
import math
import argparse
import numpy as np
import h5py
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset
from typing import Tuple
from sklearn.metrics import roc_auc_score, log_loss


# ============================================================
# 配置加载
# ============================================================

def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def get_device(cfg_device: str) -> str:
    if cfg_device == "auto":
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"
    return cfg_device


# ============================================================
# 预训练 Embedding 加载
# ============================================================

def load_pretrained_embeddings(cfg):
    data_cfg = cfg["data"]
    project_root = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(project_root, data_cfg["data_dir"])

    print("加载预训练 embedding ...")

    with h5py.File(os.path.join(data_dir, data_cfg["user_emb_file"]), "r") as f:
        user_keys = f["key"][:]
        user_vals = f["value"][:].astype(np.float32)

    num_users = data_cfg["num_users"]
    user_vis_dim = data_cfg["user_vis_dim"]
    user_emb_table = np.zeros((num_users, user_vis_dim), dtype=np.float32)
    for k, v in zip(user_keys, user_vals):
        if k < num_users:
            user_emb_table[k] = v

    with h5py.File(os.path.join(data_dir, data_cfg["item_emb_file"]), "r") as f:
        item_keys = f["key"][:]
        item_vals = f["value"][:].astype(np.float32)

    item_hash_size = data_cfg["item_hash_size"]
    item_vis_dim = data_cfg["item_vis_dim"]
    item_emb_table = np.zeros((item_hash_size, item_vis_dim), dtype=np.float32)
    for k, v in zip(item_keys, item_vals):
        hk = int(k) % item_hash_size
        item_emb_table[hk] = v

    print(f"  User visual emb: {user_emb_table.shape}")
    print(f"  Item visual emb: {item_emb_table.shape}")
    return user_emb_table, item_emb_table


# ============================================================
# 数据集
# ============================================================

class KuaiVideoIterDataset(IterableDataset):
    def __init__(self, csv_path, user_vis_emb, item_vis_emb, cfg, max_samples=None):
        self.csv_path = csv_path
        self.user_vis_emb = user_vis_emb
        self.item_vis_emb = item_vis_emb
        self.num_users = cfg["data"]["num_users"]
        self.item_hash_size = cfg["data"]["item_hash_size"]
        self.max_samples = max_samples
        self.max_seq_len = cfg["data"].get("max_seq_len", 20)

    def __iter__(self):
        count = 0
        max_seq = self.max_seq_len
        with open(self.csv_path, "r") as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                if self.max_samples and count >= self.max_samples:
                    return
                user_id = int(row[1])
                item_id = int(row[2])
                label = float(row[3])
                uid = min(user_id, self.num_users - 1)
                iid_hash = item_id % self.item_hash_size
                user_vis = self.user_vis_emb[uid]
                item_vis = self.item_vis_emb[iid_hash]

                # 解析 pos_items 行为序列 (col 6, "^" 分隔)
                raw_pos = row[6] if len(row) > 6 else ""
                if raw_pos:
                    pos_ids = [int(x) % self.item_hash_size for x in raw_pos.split("^")]
                else:
                    pos_ids = []
                # 截断到 max_seq_len
                pos_ids = pos_ids[-max_seq:]  # 取最近的
                seq_len = len(pos_ids)
                # pad 到 max_seq_len
                if seq_len < max_seq:
                    pos_ids = pos_ids + [0] * (max_seq - seq_len)

                yield (
                    uid, iid_hash,
                    torch.tensor(user_vis, dtype=torch.float32),
                    torch.tensor(item_vis, dtype=torch.float32),
                    torch.tensor(pos_ids, dtype=torch.long),
                    seq_len,
                    label,
                )
                count += 1


def collate_fn(batch):
    user_ids, item_ids, user_vis, item_vis, seq_items, seq_lens, labels = zip(*batch)
    return (
        torch.tensor(user_ids, dtype=torch.long),
        torch.tensor(item_ids, dtype=torch.long),
        torch.stack(user_vis),
        torch.stack(item_vis),
        torch.stack(seq_items),                     # [B, max_seq_len]
        torch.tensor(seq_lens, dtype=torch.long),   # [B]
        torch.tensor(labels, dtype=torch.float32),
    )


# ============================================================
# RankMixer 模型组件 (从 rankmixer.py 导入)
# ============================================================

from rankmixer import (
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
        user_vis_dim = data_cfg["user_vis_dim"]
        item_vis_dim = data_cfg["item_vis_dim"]
        chunk_size = model_cfg["chunk_size"]
        hidden_dim = model_cfg["hidden_dim"]

        self.user_emb = nn.Embedding(data_cfg["num_users"], user_emb_dim)
        self.item_emb = nn.Embedding(data_cfg["item_hash_size"], item_emb_dim)

        # DIN 序列注意力: 将行为序列编码为一个 item_emb_dim 维向量
        self.din_attn = DINAttention(item_emb_dim, hidden_dim=64)

        # total_dim 现在多了一路 seq_emb (item_emb_dim)
        total_dim = user_emb_dim + item_emb_dim + user_vis_dim + item_vis_dim + item_emb_dim
        self.num_tokens = math.ceil(total_dim / chunk_size)
        self.padded_dim = self.num_tokens * chunk_size
        self.chunk_size = chunk_size
        self.hidden_dim = hidden_dim
        self.total_dim = total_dim

        self.proj = nn.Linear(chunk_size, hidden_dim)

    def _tokenize(self, user_ids, item_ids, user_vis, item_vis, seq_items, seq_lens):
        """
        公共前处理:
        embedding → DIN序列编码 → concat → pad → chunk → proj → [B, T, D]
        """
        u_emb = self.user_emb(user_ids)       # [B, user_emb_dim]
        i_emb = self.item_emb(item_ids)       # [B, item_emb_dim]

        # DIN: 行为序列编码
        seq_emb_lookup = self.item_emb(seq_items)   # [B, S, item_emb_dim]
        seq_mask = torch.arange(seq_items.size(1), device=seq_items.device).unsqueeze(0) < seq_lens.unsqueeze(1)
        seq_enc = self.din_attn(seq_emb_lookup, i_emb, seq_mask)  # [B, item_emb_dim]

        e_input = torch.cat([u_emb, i_emb, user_vis, item_vis, seq_enc], dim=-1)

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

    def forward(self, user_ids, item_ids, user_vis, item_vis, seq_items, seq_lens):
        x = self._tokenize(user_ids, item_ids, user_vis, item_vis, seq_items, seq_lens)

        for blk in self.dense_blocks:
            x = blk(x)

        total_reg = torch.tensor(0.0, device=x.device)
        for blk in self.moe_blocks:
            x, reg = blk(x)
            total_reg = total_reg + reg

        x = x.mean(dim=1)
        logits = self.output_head(x).squeeze(-1)
        return logits, total_reg


# ============================================================
# TokenMixer-Large CTR 模型 (arXiv:2602.06563)
# ============================================================

class TokenMixerLargeCTR(BaseCTR):
    def __init__(self, cfg):
        super().__init__(cfg)
        from tokenmixer_large import (
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

    def forward(self, user_ids, item_ids, user_vis, item_vis, seq_items, seq_lens):
        tokens = self._tokenize(user_ids, item_ids, user_vis, item_vis, seq_items, seq_lens)
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

    用户特征 (user_emb + user_vis) 作为序列的第一个 prefix token。
    输出取 Φ_target 位置 (最后一个 token) 的 hidden state。
    """

    def __init__(self, cfg):
        super().__init__()
        from hstu import HSTULayer, RMSNorm as HSTURMSNorm

        data_cfg = cfg["data"]
        emb_cfg = cfg["embedding"]
        model_cfg = cfg["model"]

        user_emb_dim = emb_cfg["user_emb_dim"]
        item_emb_dim = emb_cfg["item_emb_dim"]
        user_vis_dim = data_cfg["user_vis_dim"]
        item_vis_dim = data_cfg["item_vis_dim"]
        hidden_dim = model_cfg["hidden_dim"]
        num_layers = model_cfg["num_layers"]
        num_heads = model_cfg["num_heads"]
        ffn_expansion = model_cfg.get("ffn_expansion", 4)
        dropout = model_cfg.get("dropout", 0.0)
        max_seq_len = data_cfg.get("max_seq_len", 20)

        self.hidden_dim = hidden_dim

        # Embeddings
        self.user_emb = nn.Embedding(data_cfg["num_users"], user_emb_dim)
        self.item_emb = nn.Embedding(data_cfg["item_hash_size"], item_emb_dim)

        # Action embedding: 目前 pos_items 都是 click, 用一个可学习的 action token
        # 预留多种 action type (click=0, like=1, follow=2, skip=3)
        num_action_types = 4
        self.action_emb = nn.Embedding(num_action_types, item_emb_dim)

        # 投影层: content token (item_emb + item_vis → hidden_dim)
        self.content_proj = nn.Linear(item_emb_dim + item_vis_dim, hidden_dim)
        # 投影层: action token (item_emb_dim → hidden_dim)
        self.action_proj = nn.Linear(item_emb_dim, hidden_dim)
        # 投影层: user prefix token (user_emb + user_vis → hidden_dim)
        self.user_proj = nn.Linear(user_emb_dim + user_vis_dim, hidden_dim)

        # 序列最长: 1 (user prefix) + max_seq_len * 2 (content+action) + 1 (target)
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

        self.output_head = nn.Sequential(
            HSTURMSNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, user_ids, item_ids, user_vis, item_vis, seq_items, seq_lens):
        """
        构造 Content-Action 交替序列并通过 HSTU Layers。
        取最后一个 token (target content) 的输出做预测。
        """
        B = user_ids.size(0)
        device = user_ids.device
        S = seq_items.size(1)  # max_seq_len

        # --- 构造各种 token ---
        # 1. User prefix token: [B, hidden_dim]
        u_emb = self.user_emb(user_ids)                            # [B, user_emb_dim]
        user_token = self.user_proj(torch.cat([u_emb, user_vis], dim=-1))  # [B, D]

        # 2. Content tokens for history: item_emb + item_vis
        seq_item_emb = self.item_emb(seq_items)                    # [B, S, item_emb_dim]
        # 历史 item 的 visual embedding: 需要通过 item_vis_emb 查表
        # 但训练时 item_vis_emb 不在 GPU 上，所以用零代替历史 item 的 vis
        # (原文也没有要求历史 item 必须有 vis feature)
        seq_vis_placeholder = torch.zeros(B, S, item_vis.size(-1), device=device)
        seq_content_input = torch.cat([seq_item_emb, seq_vis_placeholder], dim=-1)  # [B, S, emb+vis]
        seq_content_tokens = self.content_proj(seq_content_input)  # [B, S, D]

        # 3. Action tokens for history (all click = 0)
        action_ids = torch.zeros(B, S, dtype=torch.long, device=device)
        seq_action_tokens = self.action_proj(self.action_emb(action_ids))  # [B, S, D]

        # 4. Target content token: item_emb + item_vis
        i_emb = self.item_emb(item_ids)                            # [B, item_emb_dim]
        target_token = self.content_proj(torch.cat([i_emb, item_vis], dim=-1))  # [B, D]

        # --- 构造交替序列: [user, Φ_0, a_0, Φ_1, a_1, ..., Φ_target] ---
        # 交织 content 和 action: [B, S*2, D]
        interleaved = torch.stack([seq_content_tokens, seq_action_tokens], dim=2)
        interleaved = interleaved.view(B, S * 2, self.hidden_dim)  # [B, 2S, D]

        # 拼接: [user_token(1), interleaved(2S), target_token(1)] = [B, 2S+2, D]
        full_seq = torch.cat([
            user_token.unsqueeze(1),       # [B, 1, D]
            interleaved,                   # [B, 2S, D]
            target_token.unsqueeze(1),     # [B, 1, D]
        ], dim=1)  # [B, 2S+2, D]

        # --- 构造 attention mask ---
        # valid 长度: 1 (user) + seq_len*2 (content+action pairs) + 1 (target)
        valid_lens = 1 + seq_lens * 2 + 1  # [B]
        total_len = full_seq.size(1)       # 2S+2

        # padding mask: [B, total_len], True = valid
        pos_indices = torch.arange(total_len, device=device).unsqueeze(0)  # [1, total_len]
        padding_mask = pos_indices < valid_lens.unsqueeze(1)                # [B, total_len]

        # attention mask: [B, 1, total_len, total_len]
        # 每个 token 只能 attend 到 valid (非 padding) 的位置
        attn_mask = padding_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, total_len]
        attn_mask = attn_mask.expand(B, 1, total_len, total_len)  # [B, 1, T, T]

        # --- HSTU Layers ---
        x = full_seq
        for layer in self.layers:
            x = layer(x, attn_mask=attn_mask)

        # --- 取 target 位置的输出 ---
        # target 位置 = valid_lens - 1 (最后一个 valid token)
        target_idx = (valid_lens - 1).unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1]
        target_idx = target_idx.expand(B, 1, self.hidden_dim)     # [B, 1, D]
        target_hidden = x.gather(1, target_idx).squeeze(1)         # [B, D]

        logits = self.output_head(target_hidden).squeeze(-1)
        return logits, torch.tensor(0.0, device=logits.device)


# ============================================================
# 构建模型 (统一入口)
# ============================================================

def build_model(cfg) -> nn.Module:
    arch = cfg["model"].get("arch", "rankmixer")
    if arch == "tokenmixer_large":
        return TokenMixerLargeCTR(cfg)
    elif arch == "hstu":
        return HSTUCTR(cfg)
    else:
        return RankMixerCTR(cfg)


# ============================================================
# 评估
# ============================================================

def evaluate(model, dataloader, device):
    model.eval()
    all_labels, all_preds = [], []
    with torch.no_grad():
        for batch in dataloader:
            uids, iids, u_vis, i_vis, seq_items, seq_lens, labels = batch
            logits, _ = model(
                uids.to(device), iids.to(device),
                u_vis.to(device), i_vis.to(device),
                seq_items.to(device), seq_lens.to(device),
            )
            probs = torch.sigmoid(logits).cpu().numpy()
            all_labels.extend(labels.numpy().tolist())
            all_preds.extend(probs.tolist())

    all_labels = np.array(all_labels)
    all_preds = np.clip(np.array(all_preds), 1e-7, 1 - 1e-7)
    return roc_auc_score(all_labels, all_preds), log_loss(all_labels, all_preds)


# ============================================================
# 训练主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str,
        default=os.path.join(os.path.dirname(__file__), "config", "kuaivideo_small.yaml"),
        help="配置文件路径"
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = get_device(cfg.get("device", "auto"))
    project_root = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(project_root, cfg["data"]["data_dir"])
    train_cfg = cfg["training"]
    log_cfg = cfg["logging"]
    model_cfg = cfg["model"]
    arch = model_cfg.get("arch", "rankmixer")

    print("=" * 60)
    arch_names = {"rankmixer": "RankMixer", "tokenmixer_large": "TokenMixer-Large", "hstu": "HSTU"}
    print(f"{arch_names.get(arch, arch)} on KuaiVideo_x1")
    print(f"Config: {args.config}")
    print(f"Device: {device}")
    print("=" * 60)

    print("\n--- Model Config ---")
    for k, v in model_cfg.items():
        print(f"  {k}: {v}")
    print("\n--- Training Config ---")
    for k, v in train_cfg.items():
        print(f"  {k}: {v}")

    user_vis_emb, item_vis_emb = load_pretrained_embeddings(cfg)

    print("\n构建模型 ...")
    model = build_model(cfg).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  总参数量: {total_params / 1e6:.2f}M")

    # RankMixer 理论参数量
    if arch == "rankmixer":
        T = model.num_tokens
        D = model_cfg["hidden_dim"]
        k = model_cfg["ffn_expansion"]
        L_dense = model_cfg["num_dense_layers"]
        L_moe = model_cfg["num_moe_layers"]
        E = model_cfg["num_experts"]
        dense_theory = 2 * k * L_dense * T * D * D
        moe_theory = 2 * k * L_moe * T * D * D * E
        print(f"  论文理论 Dense 参数量 (2kL_dTD²): {dense_theory / 1e6:.2f}M")
        print(f"  论文理论 MoE 参数量 (2kL_mTD²E): {moe_theory / 1e6:.2f}M")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"]
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=train_cfg["num_epochs"],
        eta_min=train_cfg["scheduler_eta_min"]
    )
    criterion = nn.BCEWithLogitsLoss()

    train_csv = os.path.join(data_dir, cfg["data"]["train_file"])
    test_csv = os.path.join(data_dir, cfg["data"]["test_file"])

    best_auc = 0.0
    save_path = os.path.join(project_root, log_cfg["save_path"])
    warmup_steps = train_cfg.get("warmup_steps", 0)

    # TokenMixer-Large 的辅助损失权重
    aux_loss_weight = model_cfg.get("aux_loss_weight", 0.1) if arch == "tokenmixer_large" else 0.0

    for epoch in range(train_cfg["num_epochs"]):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{train_cfg['num_epochs']}  "
              f"(lr={optimizer.param_groups[0]['lr']:.2e})")
        print(f"{'='*60}")

        train_dataset = KuaiVideoIterDataset(
            train_csv, user_vis_emb, item_vis_emb, cfg
        )
        train_loader = DataLoader(
            train_dataset, batch_size=train_cfg["batch_size"],
            collate_fn=collate_fn, num_workers=0,
        )

        model.train()
        epoch_loss = 0.0
        epoch_samples = 0
        step = 0
        t0 = time.time()
        global_step = epoch * 2700

        for batch in train_loader:
            uids, iids, u_vis, i_vis, seq_items, seq_lens, labels = batch
            uids = uids.to(device)
            iids = iids.to(device)
            u_vis = u_vis.to(device)
            i_vis = i_vis.to(device)
            seq_items = seq_items.to(device)
            seq_lens = seq_lens.to(device)
            labels = labels.to(device)

            cur_global = global_step + step
            if warmup_steps > 0 and cur_global < warmup_steps:
                warmup_lr = train_cfg["learning_rate"] * (cur_global + 1) / warmup_steps
                for pg in optimizer.param_groups:
                    pg["lr"] = warmup_lr

            optimizer.zero_grad()
            main_logits, aux_output = model(uids, iids, u_vis, i_vis, seq_items, seq_lens)

            main_loss = criterion(main_logits, labels)

            if arch == "tokenmixer_large":
                # aux_output 是辅助 logits
                if aux_output.abs().sum() > 0:
                    aux_loss = criterion(aux_output, labels)
                    loss = main_loss + aux_loss_weight * aux_loss
                else:
                    loss = main_loss
                    aux_loss = torch.tensor(0.0)
            elif arch == "rankmixer":
                # aux_output 是 MoE L1 正则损失
                loss = main_loss + aux_output
                aux_loss = aux_output
            else:
                # HSTU: 无辅助损失
                loss = main_loss
                aux_loss = torch.tensor(0.0)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=train_cfg["grad_clip_norm"]
            )
            optimizer.step()

            bs = labels.size(0)
            epoch_loss += main_loss.item() * bs
            epoch_samples += bs
            step += 1

            if step % log_cfg["log_interval"] == 0:
                avg = epoch_loss / epoch_samples
                elapsed = time.time() - t0
                speed = epoch_samples / elapsed
                aux_name = {"tokenmixer_large": "Aux", "rankmixer": "Reg"}.get(arch, "Aux")
                aux_val = aux_loss.item() if torch.is_tensor(aux_loss) else aux_loss
                print(
                    f"  Step {step:5d} | "
                    f"Samples: {epoch_samples:>8d} | "
                    f"Loss: {avg:.4f} | "
                    f"Main: {main_loss.item():.4f} | "
                    f"{aux_name}: {aux_val:.4f} | "
                    f"Speed: {speed:.0f} s/s | "
                    f"LR: {optimizer.param_groups[0]['lr']:.2e}"
                )

        avg_loss = epoch_loss / max(epoch_samples, 1)
        elapsed = time.time() - t0
        print(f"\n  Epoch {epoch+1} 训练完成: Loss={avg_loss:.4f}, "
              f"Samples={epoch_samples}, Time={elapsed:.1f}s")

        scheduler.step()

        print("\n  评估中 ...")
        eval_samples = log_cfg["eval_samples"] if log_cfg["eval_samples"] > 0 else None
        test_dataset = KuaiVideoIterDataset(
            test_csv, user_vis_emb, item_vis_emb, cfg, max_samples=eval_samples
        )
        test_loader = DataLoader(
            test_dataset, batch_size=train_cfg["batch_size"] * 2,
            collate_fn=collate_fn, num_workers=0,
        )
        auc, logloss = evaluate(model, test_loader, device)
        print(f"  Test AUC: {auc:.4f} | Test LogLoss: {logloss:.4f}")

        if log_cfg.get("save_best", True):
            if auc > best_auc:
                best_auc = auc
                torch.save(model.state_dict(), save_path)
                print(f"  ** 新最佳模型已保存 (AUC={auc:.4f}) → {save_path}")
        else:
            torch.save(model.state_dict(), save_path)
            print(f"  模型已保存 → {save_path}")

    print(f"\n{'='*60}")
    print(f"训练完成! Best Test AUC: {best_auc:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
