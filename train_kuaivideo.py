"""
在 KuaiVideo_x1 数据集上进行 CTR 预测训练
支持四种模型架构:
  - RankMixer (arXiv:2507.15551)
  - TokenMixer-Large (arXiv:2602.06563)
  - HSTU (arXiv:2402.17152)
  - Vanilla Transformer (Baseline)
通过 config YAML 中的 model.arch 字段选择

多卡训练:
  torchrun --nproc_per_node=N train_kuaivideo.py --config config/xxx.yaml
单卡训练:
  python train_kuaivideo.py --config config/xxx.yaml
"""

import os
import sys
import csv
import time
import math
import random
from datetime import datetime
import argparse
import numpy as np
import h5py
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, IterableDataset
from typing import Tuple
from sklearn.metrics import roc_auc_score, log_loss
import wandb


# ============================================================
# 分布式工具
# ============================================================

def is_dist():
    """是否处于分布式环境"""
    return dist.is_available() and dist.is_initialized()


def get_rank():
    return dist.get_rank() if is_dist() else 0


def get_world_size():
    return dist.get_world_size() if is_dist() else 1


def is_main_process():
    return get_rank() == 0


def setup_distributed():
    """初始化分布式环境 (由 torchrun 设置环境变量)"""
    if "RANK" not in os.environ:
        return  # 单卡模式, 不初始化
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)


def cleanup_distributed():
    if is_dist():
        dist.destroy_process_group()


# ============================================================
# 配置加载
# ============================================================

def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def get_device(cfg_device: str) -> str:
    if is_dist():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        return f"cuda:{local_rank}"
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
    """加载预训练 embedding。与 FuxiCTR benchmark 对齐: 只加载 item visual emb，不用 user visual emb。"""
    data_cfg = cfg["data"]
    use_pretrained = data_cfg.get("use_pretrained_emb", True)

    if not use_pretrained:
        if is_main_process():
            print("跳过预训练 embedding 加载 (use_pretrained_emb=false)")
        return None

    project_root = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(project_root, data_cfg["data_dir"])

    if is_main_process():
        print("加载预训练 item visual embedding ...")

    with h5py.File(os.path.join(data_dir, data_cfg["item_emb_file"]), "r") as f:
        item_keys = f["key"][:]
        item_vals = f["value"][:].astype(np.float32)

    item_hash_size = data_cfg["item_hash_size"]
    item_vis_dim = data_cfg["item_vis_dim"]
    item_emb_table = np.zeros((item_hash_size, item_vis_dim), dtype=np.float32)
    for k, v in zip(item_keys, item_vals):
        hk = int(k) % item_hash_size
        item_emb_table[hk] = v

    if is_main_process():
        print(f"  Item visual emb: {item_emb_table.shape}")
    return item_emb_table


# ============================================================
# 数据集
# ============================================================

class KuaiVideoIterDataset(IterableDataset):
    def __init__(self, csv_path, item_vis_emb, cfg,
                 max_samples=None, shuffle_buffer=0):
        self.csv_path = csv_path
        self.item_vis_emb = item_vis_emb
        self.num_users = cfg["data"]["num_users"]
        self.item_hash_size = cfg["data"]["item_hash_size"]
        self.max_samples = max_samples
        self.max_seq_len = cfg["data"].get("max_seq_len", 100)
        self.shuffle_buffer = shuffle_buffer
        self.use_pretrained = item_vis_emb is not None
        if self.use_pretrained:
            self.item_vis_dim = item_vis_emb.shape[1]
        else:
            self.item_vis_dim = 0

    def _parse_row(self, row, max_seq):
        """解析 CSV 行为样本 tuple (与 FuxiCTR benchmark 对齐: 无 is_like/is_follow, 双通道序列)"""
        user_id = int(row[1])
        item_id = int(row[2])
        label = float(row[3])
        # is_like = row[4], is_follow = row[5] — 与 FuxiCTR 对齐, 不使用
        uid = min(user_id, self.num_users - 1)
        iid_hash = item_id % self.item_hash_size

        if self.use_pretrained:
            item_vis = self.item_vis_emb[iid_hash]
        else:
            item_vis = np.zeros(0, dtype=np.float32)

        raw_pos = row[6] if len(row) > 6 else ""
        if raw_pos:
            pos_ids = [int(x) % self.item_hash_size for x in raw_pos.split("^")]
        else:
            pos_ids = []
        pos_ids = pos_ids[-max_seq:]
        pos_len = len(pos_ids)
        if pos_len < max_seq:
            pos_ids = pos_ids + [0] * (max_seq - pos_len)

        raw_neg = row[7] if len(row) > 7 else ""
        if raw_neg:
            neg_ids = [int(x) % self.item_hash_size for x in raw_neg.split("^")]
        else:
            neg_ids = []
        neg_ids = neg_ids[-max_seq:]
        neg_len = len(neg_ids)
        if neg_len < max_seq:
            neg_ids = neg_ids + [0] * (max_seq - neg_len)

        # 序列视觉 embedding (双通道: ID + visual)
        if self.use_pretrained:
            pos_vis = self.item_vis_emb[pos_ids]  # [max_seq, vis_dim]
            neg_vis = self.item_vis_emb[neg_ids]  # [max_seq, vis_dim]
        else:
            pos_vis = np.zeros((max_seq, 0), dtype=np.float32)
            neg_vis = np.zeros((max_seq, 0), dtype=np.float32)

        return (
            uid, iid_hash,
            torch.tensor(item_vis, dtype=torch.float32),
            torch.tensor(pos_ids, dtype=torch.long),
            pos_len,
            torch.tensor(neg_ids, dtype=torch.long),
            neg_len,
            torch.tensor(pos_vis, dtype=torch.float32),
            torch.tensor(neg_vis, dtype=torch.float32),
            label,
        )

    def __iter__(self):
        rank = get_rank()
        world_size = get_world_size()
        count = 0
        max_seq = self.max_seq_len
        buf = []
        buf_size = self.shuffle_buffer

        with open(self.csv_path, "r") as f:
            reader = csv.reader(f)
            next(reader)
            for row_idx, row in enumerate(reader):
                if self.max_samples and count >= self.max_samples:
                    break
                if row_idx % world_size != rank:
                    continue

                sample = self._parse_row(row, max_seq)

                if buf_size > 0:
                    buf.append(sample)
                    if len(buf) >= buf_size:
                        random.shuffle(buf)
                        for s in buf:
                            yield s
                            count += 1
                            if self.max_samples and count >= self.max_samples:
                                return
                        buf = []
                else:
                    yield sample
                    count += 1

        # flush remaining buffer
        if buf:
            random.shuffle(buf)
            for s in buf:
                yield s
                count += 1
                if self.max_samples and count >= self.max_samples:
                    return


def collate_fn(batch):
    (user_ids, item_ids, item_vis,
     pos_items, pos_lens, neg_items, neg_lens,
     pos_vis, neg_vis, labels) = zip(*batch)
    return (
        torch.tensor(user_ids, dtype=torch.long),
        torch.tensor(item_ids, dtype=torch.long),
        torch.stack(item_vis),
        torch.stack(pos_items),                      # [B, max_seq_len]
        torch.tensor(pos_lens, dtype=torch.long),    # [B]
        torch.stack(neg_items),                      # [B, max_seq_len]
        torch.tensor(neg_lens, dtype=torch.long),    # [B]
        torch.stack(pos_vis),                        # [B, max_seq_len, vis_dim]
        torch.stack(neg_vis),                        # [B, max_seq_len, vis_dim]
        torch.tensor(labels, dtype=torch.float32),
    )

from models.ctr_models import (
    DINAttention, BaseCTR, RankMixerCTR, TokenMixerLargeCTR,
    TransformerCTR, HSTUCTR, build_model,
)


# ============================================================
# 评估
# ============================================================

def compute_gauc(user_ids, labels, preds):
    """
    计算 Group AUC (GAUC): 按 user_id 分组计算 AUC, 再按样本数加权平均。
    只对同时包含正负样本的用户分组计算。
    """
    from collections import defaultdict
    user_data = defaultdict(lambda: ([], []))
    for uid, label, pred in zip(user_ids, labels, preds):
        user_data[uid][0].append(label)
        user_data[uid][1].append(pred)

    total_auc = 0.0
    total_count = 0
    for uid, (u_labels, u_preds) in user_data.items():
        u_labels = np.array(u_labels)
        u_preds = np.array(u_preds)
        # 只有同时包含正负样本才能计算 AUC
        if len(set(u_labels)) < 2:
            continue
        try:
            auc = roc_auc_score(u_labels, u_preds)
            n = len(u_labels)
            total_auc += auc * n
            total_count += n
        except ValueError:
            continue

    if total_count == 0:
        return 0.0
    return total_auc / total_count


def evaluate(model, dataloader, device):
    """
    评估模型。DDP 模式下每个 rank 各自跑一部分数据，
    通过 all_gather 汇总到 rank 0 计算全局指标。
    返回 (auc, gauc, logloss)。
    """
    raw_model = model.module if hasattr(model, "module") else model
    raw_model.eval()
    all_labels, all_preds, all_uids = [], [], []
    with torch.no_grad():
        for batch in dataloader:
            (uids, iids, i_vis,
             pos_items, pos_lens, neg_items, neg_lens,
             pos_vis, neg_vis, labels) = batch
            logits, _ = raw_model(
                uids.to(device), iids.to(device),
                i_vis.to(device),
                pos_items.to(device), pos_lens.to(device),
                neg_items.to(device), neg_lens.to(device),
                pos_vis.to(device), neg_vis.to(device),
            )
            probs = torch.sigmoid(logits).cpu()
            all_labels.append(labels)
            all_preds.append(probs)
            all_uids.append(uids)

    all_labels = torch.cat(all_labels)
    all_preds = torch.cat(all_preds)
    all_uids = torch.cat(all_uids)

    if is_dist():
        # all-gather across ranks
        world_size = get_world_size()
        local_size = torch.tensor([all_labels.size(0)], dtype=torch.long, device=device)
        sizes_list = [torch.zeros_like(local_size) for _ in range(world_size)]
        dist.all_gather(sizes_list, local_size)
        max_size = max(s.item() for s in sizes_list)

        # pad to max_size for uniform gather
        def pad_to(t, n):
            if t.size(0) < n:
                return torch.cat([t, torch.zeros(n - t.size(0), dtype=t.dtype)])
            return t[:n]

        padded_labels = pad_to(all_labels, max_size).to(device)
        padded_preds = pad_to(all_preds, max_size).to(device)
        padded_uids = pad_to(all_uids.float(), max_size).to(device)

        gathered_labels = [torch.zeros_like(padded_labels) for _ in range(world_size)]
        gathered_preds = [torch.zeros_like(padded_preds) for _ in range(world_size)]
        gathered_uids = [torch.zeros_like(padded_uids) for _ in range(world_size)]
        dist.all_gather(gathered_labels, padded_labels)
        dist.all_gather(gathered_preds, padded_preds)
        dist.all_gather(gathered_uids, padded_uids)

        if is_main_process():
            final_labels, final_preds, final_uids = [], [], []
            for i in range(world_size):
                n = sizes_list[i].item()
                final_labels.append(gathered_labels[i][:n].cpu())
                final_preds.append(gathered_preds[i][:n].cpu())
                final_uids.append(gathered_uids[i][:n].cpu())
            all_labels = torch.cat(final_labels).numpy()
            all_preds = torch.cat(final_preds).numpy()
            all_uids = torch.cat(final_uids).numpy().astype(int)
        else:
            return 0.0, 0.0, 0.0  # 非 rank 0 不需要结果
    else:
        all_labels = all_labels.numpy()
        all_preds = all_preds.numpy()
        all_uids = all_uids.numpy().astype(int)

    all_preds = np.clip(all_preds, 1e-7, 1 - 1e-7)
    auc = roc_auc_score(all_labels, all_preds)
    gauc = compute_gauc(all_uids, all_labels, all_preds)
    logloss = log_loss(all_labels, all_preds)
    return auc, gauc, logloss


# ============================================================
# 训练主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str,
        default=os.path.join(os.path.dirname(__file__), "config", "rankmixer_small.yaml"),
        help="配置文件路径"
    )
    args = parser.parse_args()

    # --- 分布式初始化 ---
    setup_distributed()
    rank = get_rank()
    world_size = get_world_size()

    cfg = load_config(args.config)
    device = get_device(cfg.get("device", "auto"))
    project_root = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(project_root, cfg["data"]["data_dir"])
    train_cfg = cfg["training"]
    log_cfg = cfg["logging"]
    model_cfg = cfg["model"]
    arch = model_cfg.get("arch", "rankmixer")

    # --- wandb 初始化 (仅 rank 0) ---
    if is_main_process():
        wandb_cfg = cfg.get("wandb", {})
        wandb_project = wandb_cfg.get("project", "rankmixer")
        wandb_name = wandb_cfg.get("name", None)
        if wandb_name is None:
            config_basename = os.path.splitext(os.path.basename(args.config))[0]
            wandb_name = config_basename
        wandb_tags = wandb_cfg.get("tags", [arch])

        wandb.init(
            project=wandb_project,
            name=wandb_name,
            tags=wandb_tags,
            config={
                "arch": arch,
                "model": model_cfg,
                "training": train_cfg,
                "data": {k: v for k, v in cfg["data"].items()
                         if k not in ("data_dir", "train_file", "test_file",
                                      "user_emb_file", "item_emb_file")},
                "embedding": cfg["embedding"],
                "world_size": world_size,
            },
        )

    if is_main_process():
        print("=" * 60)
        arch_names = {"rankmixer": "RankMixer", "tokenmixer_large": "TokenMixer-Large", "hstu": "HSTU", "transformer": "Transformer"}
        print(f"{arch_names.get(arch, arch)} on KuaiVideo_x1")
        print(f"Config: {args.config}")
        print(f"Device: {device} | World size: {world_size}")
        print("=" * 60)

        print("\n--- Model Config ---")
        for k, v in model_cfg.items():
            print(f"  {k}: {v}")
        print("\n--- Training Config ---")
        for k, v in train_cfg.items():
            print(f"  {k}: {v}")

    item_vis_emb = load_pretrained_embeddings(cfg)

    if is_main_process():
        print("\n构建模型 ...")
    model = build_model(cfg).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    if is_main_process():
        print(f"  总参数量: {total_params / 1e6:.2f}M")
        wandb.log({"model/total_params_M": total_params / 1e6}, step=0)

    # RankMixer 理论参数量
    if arch == "rankmixer" and is_main_process():
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

    # --- DDP 包装 ---
    if is_dist():
        model = DDP(model, device_ids=[int(os.environ.get("LOCAL_RANK", 0))])

    # --- 优化器 (与 FuxiCTR benchmark 对齐: Adam, 无 weight_decay) ---
    opt_name = train_cfg.get("optimizer", "adam").lower()
    weight_decay = train_cfg.get("weight_decay", 0.0)
    if opt_name == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=train_cfg["learning_rate"],
            weight_decay=weight_decay,
        )
    else:
        optimizer = torch.optim.Adam(
            model.parameters(), lr=train_cfg["learning_rate"],
            weight_decay=weight_decay,
        )

    # --- 学习率调度 ---
    lr_decay_factor = train_cfg.get("lr_decay_factor", 0.1)
    lr_min = train_cfg.get("scheduler_eta_min", 1e-6)
    criterion = nn.BCEWithLogitsLoss()

    train_csv = os.path.join(data_dir, cfg["data"]["train_file"])
    test_csv = os.path.join(data_dir, cfg["data"]["test_file"])

    best_auc = 0.0
    best_monitor = -1.0  # monitor = gAUC + AUC (与 FuxiCTR 对齐)
    no_improve_count = 0
    early_stop_patience = train_cfg.get("early_stop_patience", 5)
    # 保存路径: ckpt/方法名/时间戳/best.pt
    config_basename = os.path.splitext(os.path.basename(args.config))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(project_root, "ckpt", config_basename, timestamp)
    if is_main_process():
        os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "best.pt")

    # TokenMixer-Large 的辅助损失权重
    aux_loss_weight = model_cfg.get("aux_loss_weight", 0.1) if arch == "tokenmixer_large" else 0.0

    global_step = 0

    for epoch in range(train_cfg["num_epochs"]):
        if is_main_process():
            print(f"\n{'='*60}")
            print(f"Epoch {epoch + 1}/{train_cfg['num_epochs']}  "
                  f"(lr={optimizer.param_groups[0]['lr']:.2e})")
            print(f"{'='*60}")

        train_dataset = KuaiVideoIterDataset(
            train_csv, item_vis_emb, cfg,
            shuffle_buffer=train_cfg.get("shuffle_buffer", 50000),
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

        for batch in train_loader:
            (uids, iids, i_vis,
             pos_items, pos_lens, neg_items, neg_lens,
             pos_vis, neg_vis, labels) = batch
            uids = uids.to(device)
            iids = iids.to(device)
            i_vis = i_vis.to(device)
            pos_items = pos_items.to(device)
            pos_lens = pos_lens.to(device)
            neg_items = neg_items.to(device)
            neg_lens = neg_lens.to(device)
            pos_vis = pos_vis.to(device)
            neg_vis = neg_vis.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            main_logits, aux_output = model(
                uids, iids, i_vis,
                pos_items, pos_lens, neg_items, neg_lens,
                pos_vis, neg_vis,
            )

            main_loss = criterion(main_logits, labels)

            if arch == "tokenmixer_large":
                if aux_output.abs().sum() > 0:
                    aux_loss = criterion(aux_output, labels)
                    loss = main_loss + aux_loss_weight * aux_loss
                else:
                    loss = main_loss
                    aux_loss = torch.tensor(0.0)
            elif arch == "rankmixer":
                loss = main_loss + aux_output
                aux_loss = aux_output
            else:
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
            global_step += 1

            # wandb step-level logging (rank 0 only)
            aux_val = aux_loss.item() if torch.is_tensor(aux_loss) else aux_loss
            if is_main_process():
                wandb.log({
                    "train/loss": loss.item(),
                    "train/main_loss": main_loss.item(),
                    "train/aux_loss": aux_val,
                    "train/lr": optimizer.param_groups[0]["lr"],
                    "train/epoch": epoch + 1,
                }, step=global_step)

            if step % log_cfg["log_interval"] == 0 and is_main_process():
                avg = epoch_loss / epoch_samples
                elapsed = time.time() - t0
                speed = epoch_samples / elapsed
                aux_name = {"tokenmixer_large": "Aux", "rankmixer": "Reg"}.get(arch, "Aux")
                print(
                    f"  Step {step:5d} | "
                    f"Samples: {epoch_samples:>8d} | "
                    f"Loss: {avg:.4f} | "
                    f"Main: {main_loss.item():.4f} | "
                    f"{aux_name}: {aux_val:.4f} | "
                    f"Speed: {speed:.0f} s/s | "
                    f"LR: {optimizer.param_groups[0]['lr']:.2e}"
                )
                wandb.log({
                    "train/avg_loss": avg,
                    "train/speed_samples_per_sec": speed,
                }, step=global_step)

        # 同步 epoch_loss 和 epoch_samples (可选, 让 rank 0 拿到全局统计)
        if is_dist():
            stats = torch.tensor([epoch_loss, float(epoch_samples)], device=device)
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
            epoch_loss = stats[0].item()
            epoch_samples = int(stats[1].item())

        avg_loss = epoch_loss / max(epoch_samples, 1)
        elapsed = time.time() - t0
        if is_main_process():
            print(f"\n  Epoch {epoch+1} 训练完成: Loss={avg_loss:.4f}, "
                  f"Samples={epoch_samples}, Time={elapsed:.1f}s")

            wandb.log({
                "epoch/train_loss": avg_loss,
                "epoch/train_samples": epoch_samples,
                "epoch/train_time_s": elapsed,
                "epoch/epoch": epoch + 1,
            }, step=global_step)

        # --- 评估 ---
        if is_main_process():
            print("\n  评估中 ...")
        eval_samples = log_cfg["eval_samples"] if log_cfg["eval_samples"] > 0 else None
        test_dataset = KuaiVideoIterDataset(
            test_csv, item_vis_emb, cfg, max_samples=eval_samples
        )
        test_loader = DataLoader(
            test_dataset, batch_size=train_cfg["batch_size"] * 2,
            collate_fn=collate_fn, num_workers=0,
        )
        auc, gauc, logloss = evaluate(model, test_loader, device)

        # Early stopping: rank 0 判断是否提升，广播给所有 rank
        should_stop = False
        if is_main_process():
            print(f"  Test AUC: {auc:.4f} | Test GAUC: {gauc:.4f} | Test LogLoss: {logloss:.4f}")

            wandb.log({
                "eval/auc": auc,
                "eval/gauc": gauc,
                "eval/logloss": logloss,
                "eval/epoch": epoch + 1,
            }, step=global_step)

            if log_cfg.get("save_best", True):
                # Monitor = gAUC + AUC (与 FuxiCTR monitor: {"gAUC": 1, "AUC": 1} 对齐)
                current_monitor = gauc + auc
                if current_monitor > best_monitor:
                    best_monitor = current_monitor
                    best_auc = auc
                    no_improve_count = 0
                    raw_model = model.module if hasattr(model, "module") else model
                    torch.save(raw_model.state_dict(), save_path)
                    print(f"  ** 新最佳模型已保存 (AUC={auc:.4f}, gAUC={gauc:.4f}, monitor={current_monitor:.4f}) → {save_path}")
                    wandb.run.summary["best_auc"] = auc
                    wandb.run.summary["best_gauc"] = gauc
                    wandb.run.summary["best_monitor"] = current_monitor
                    wandb.run.summary["best_epoch"] = epoch + 1
                else:
                    no_improve_count += 1
                    print(f"  Monitor 未提升 ({no_improve_count}/{early_stop_patience}), "
                          f"current={current_monitor:.4f}, best={best_monitor:.4f}")
                    # reduce_lr_on_plateau: 指标不提升时降低学习率 (与 FuxiCTR 对齐)
                    if train_cfg.get("reduce_lr_on_plateau", True):
                        for pg in optimizer.param_groups:
                            pg["lr"] = max(pg["lr"] * lr_decay_factor, lr_min)
                        print(f"  Reduce LR on plateau → {optimizer.param_groups[0]['lr']:.2e}")
                    if no_improve_count >= early_stop_patience:
                        print(f"  ** Early stopping: 连续 {early_stop_patience} 个 epoch 无提升")
                        should_stop = True
            else:
                raw_model = model.module if hasattr(model, "module") else model
                torch.save(raw_model.state_dict(), save_path)
                print(f"  模型已保存 → {save_path}")

        # DDP: 广播 early stop 信号
        if is_dist():
            stop_tensor = torch.tensor([1 if should_stop else 0], dtype=torch.long, device=device)
            dist.broadcast(stop_tensor, src=0)
            should_stop = stop_tensor.item() == 1
            dist.barrier()
        
        if should_stop:
            break

    if is_main_process():
        print(f"\n{'='*60}")
        print(f"训练完成! Best Test AUC: {best_auc:.4f}")
        print(f"{'='*60}")

        wandb.run.summary["final_best_auc"] = best_auc
        wandb.finish()

    cleanup_distributed()


if __name__ == "__main__":
    main()
