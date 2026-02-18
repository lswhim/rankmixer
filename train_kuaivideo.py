"""
在 KuaiVideo_x1 数据集上进行 CTR 预测训练
支持两种模型架构:
  - RankMixer (arXiv:2507.15551)
  - TokenMixer-Large (arXiv:2602.06563)
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

    def __iter__(self):
        count = 0
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
                yield (
                    uid, iid_hash,
                    torch.tensor(user_vis, dtype=torch.float32),
                    torch.tensor(item_vis, dtype=torch.float32),
                    label,
                )
                count += 1


def collate_fn(batch):
    user_ids, item_ids, user_vis, item_vis, labels = zip(*batch)
    return (
        torch.tensor(user_ids, dtype=torch.long),
        torch.tensor(item_ids, dtype=torch.long),
        torch.stack(user_vis),
        torch.stack(item_vis),
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
# RankMixer CTR 模型
# ============================================================

class RankMixerCTR(nn.Module):
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
        num_dense = model_cfg["num_dense_layers"]
        num_moe = model_cfg["num_moe_layers"]
        ffn_exp = model_cfg["ffn_expansion"]
        num_experts = model_cfg["num_experts"]
        l1_lambda = model_cfg["l1_lambda"]

        self.user_emb = nn.Embedding(data_cfg["num_users"], user_emb_dim)
        self.item_emb = nn.Embedding(data_cfg["item_hash_size"], item_emb_dim)

        total_dim = user_emb_dim + item_emb_dim + user_vis_dim + item_vis_dim
        self.num_tokens = math.ceil(total_dim / chunk_size)
        self.padded_dim = self.num_tokens * chunk_size
        self.chunk_size = chunk_size
        self.hidden_dim = hidden_dim
        self.proj = nn.Linear(chunk_size, hidden_dim)

        T = self.num_tokens
        print(f"  [RankMixer] Feature dim: {total_dim}, Tokens (T): {T}, "
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

    def forward(self, user_ids, item_ids, user_vis, item_vis):
        u_emb = self.user_emb(user_ids)
        i_emb = self.item_emb(item_ids)
        e_input = torch.cat([u_emb, i_emb, user_vis, item_vis], dim=-1)

        B = e_input.size(0)
        if e_input.size(-1) < self.padded_dim:
            e_input = F.pad(e_input, (0, self.padded_dim - e_input.size(-1)))

        tokens = e_input.view(B, self.num_tokens, self.chunk_size)
        x = self.proj(tokens)

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

class TokenMixerLargeCTR(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        from tokenmixer_large import (
            TokenMixerLargeBlock, TokenMixerLargeMoEBlock, RMSNorm,
        )

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

        total_dim = user_emb_dim + item_emb_dim + user_vis_dim + item_vis_dim
        self.num_tokens = math.ceil(total_dim / chunk_size)
        self.padded_dim = self.num_tokens * chunk_size
        self.chunk_size = chunk_size
        self.hidden_dim = hidden_dim

        T = self.num_tokens
        T_global = T + 1

        print(f"  [TokenMixer-Large] Feature dim: {total_dim}, Tokens (T): {T}, "
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

        self.proj = nn.Linear(chunk_size, hidden_dim)

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

    def forward(self, user_ids, item_ids, user_vis, item_vis):
        u_emb = self.user_emb(user_ids)
        i_emb = self.item_emb(item_ids)
        e_input = torch.cat([u_emb, i_emb, user_vis, item_vis], dim=-1)

        B = e_input.size(0)
        if e_input.size(-1) < self.padded_dim:
            e_input = F.pad(e_input, (0, self.padded_dim - e_input.size(-1)))

        tokens = e_input.view(B, self.num_tokens, self.chunk_size)
        tokens = self.proj(tokens)

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
# 构建模型 (统一入口)
# ============================================================

def build_model(cfg) -> nn.Module:
    arch = cfg["model"].get("arch", "rankmixer")
    if arch == "tokenmixer_large":
        return TokenMixerLargeCTR(cfg)
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
            uids, iids, u_vis, i_vis, labels = batch
            logits, _ = model(
                uids.to(device), iids.to(device),
                u_vis.to(device), i_vis.to(device)
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
    print(f"{'TokenMixer-Large' if arch == 'tokenmixer_large' else 'RankMixer'} on KuaiVideo_x1")
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
            uids, iids, u_vis, i_vis, labels = batch
            uids = uids.to(device)
            iids = iids.to(device)
            u_vis = u_vis.to(device)
            i_vis = i_vis.to(device)
            labels = labels.to(device)

            cur_global = global_step + step
            if warmup_steps > 0 and cur_global < warmup_steps:
                warmup_lr = train_cfg["learning_rate"] * (cur_global + 1) / warmup_steps
                for pg in optimizer.param_groups:
                    pg["lr"] = warmup_lr

            optimizer.zero_grad()
            main_logits, aux_output = model(uids, iids, u_vis, i_vis)

            main_loss = criterion(main_logits, labels)

            if arch == "tokenmixer_large":
                # aux_output 是辅助 logits
                if aux_output.abs().sum() > 0:
                    aux_loss = criterion(aux_output, labels)
                    loss = main_loss + aux_loss_weight * aux_loss
                else:
                    loss = main_loss
                    aux_loss = torch.tensor(0.0)
            else:
                # aux_output 是 MoE L1 正则损失
                loss = main_loss + aux_output
                aux_loss = aux_output

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
                aux_name = "Aux" if arch == "tokenmixer_large" else "Reg"
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
