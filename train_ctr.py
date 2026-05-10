"""
BARS-style CTR prediction training entrypoint.
支持多种模型架构:
  - RankMixer (arXiv:2507.15551)
  - TokenMixer-Large (arXiv:2602.06563)
  - HSTU (arXiv:2402.17152)
  - Vanilla Transformer (Baseline)
通过 config YAML 中的 model.arch 字段选择

多卡训练:
  torchrun --nproc_per_node=N train_ctr.py --config config/xxx.yaml
单卡训练:
  python train_ctr.py --config config/xxx.yaml
"""

import os
import sys
import time
import random
from datetime import datetime
import argparse
import numpy as np
import yaml
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
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

def _deep_merge(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    includes = cfg.pop("includes", []) or []
    if isinstance(includes, str):
        includes = [includes]

    merged = {}
    config_dir = os.path.dirname(config_path)
    for include_path in includes:
        full_path = include_path
        if not os.path.isabs(full_path):
            full_path = os.path.normpath(os.path.join(config_dir, include_path))
        merged = _deep_merge(merged, load_config(full_path))
    return _deep_merge(merged, cfg)


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


from data.bars_dataset import (
    CTRMapDataset as KuaiVideoMapDataset,
    load_pretrained_embeddings,
    make_collate_fn,
)

from models.ctr_models import build_model


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
            (group_ids, uids, iids, i_vis,
             pos_items, pos_lens, neg_items, neg_lens,
             pos_vis, neg_vis,
             cate_ids, pos_cates, neg_cates,
             extra_cat_ids, extra_seq_ids, extra_seq_lens, numeric_vals,
             labels) = batch
            with torch.no_grad():
                output = raw_model(
                    uids.to(device), iids.to(device),
                    i_vis.to(device),
                    pos_items.to(device), pos_lens.to(device),
                    neg_items.to(device), neg_lens.to(device),
                    pos_vis.to(device), neg_vis.to(device),
                    cate_ids.to(device), pos_cates.to(device), neg_cates.to(device),
                    extra_cat_ids.to(device), extra_seq_ids.to(device),
                    extra_seq_lens.to(device), numeric_vals.to(device),
                )
                logits = output[0]
            probs = torch.sigmoid(logits.float()).cpu()
            all_labels.append(labels)
            all_preds.append(probs)
            all_uids.append(group_ids)

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


def compute_monitor_value(monitor_cfg, auc, gauc, logloss):
    """Compute BARS/FuxiCTR-style early-stop monitor from metric config."""
    metrics = {"AUC": auc, "gAUC": gauc, "logloss": logloss}
    if monitor_cfg is None:
        return gauc + auc
    if isinstance(monitor_cfg, str):
        return metrics[monitor_cfg]
    if isinstance(monitor_cfg, dict):
        return sum(metrics[name] * weight for name, weight in monitor_cfg.items())
    raise ValueError(f"Unsupported monitor config: {monitor_cfg}")


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

    # --- 随机种子 (可复现) ---
    seed = cfg.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if is_main_process():
        print(f"  Random seed: {seed}")

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
        arch_names = {"rankmixer": "RankMixer", "tokenmixer_large": "TokenMixer-Large", "hstu": "HSTU", "transformer": "Transformer", "hiformer": "HiFormer", "hyformer": "HyFormer", "interformer": "InterFormer", "dmin": "DMIN"}
        print(f"{arch_names.get(arch, arch)} on {cfg['data'].get('data_dir', 'dataset')}")
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

    # 梯度检查点: 用时间换空间, 降低显存占用
    use_grad_ckpt = train_cfg.get("gradient_checkpointing", False)
    if use_grad_ckpt and hasattr(model, "enable_gradient_checkpointing"):
        model.enable_gradient_checkpointing()
        if is_main_process():
            print("  Gradient Checkpointing: ON")

    # RankMixer 理论参数量
    if arch == "rankmixer" and is_main_process():
        T = model.num_tokens
        D = model_cfg["hidden_dim"]
        k = model_cfg["ffn_expansion"]
        L = model_cfg["num_layers"]
        mode = model_cfg.get("mode", "dense").lower()
        if mode == "dense":
            theory = 2 * k * L * T * D * D
            print(f"  论文理论 Dense 参数量 (2kLTD²): {theory / 1e6:.2f}M  "
                  f"[mode=dense, L={L}, T={T}, D={D}, k={k}]")
        else:
            E = model_cfg["num_experts"]
            theory = 2 * k * L * T * D * D * E
            print(f"  论文理论 MoE 参数量 (2kLTD²·E): {theory / 1e6:.2f}M  "
                  f"[mode=moe, L={L}, T={T}, D={D}, k={k}, E={E}]")

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

    # 全量加载数据到内存 (与 FuxiCTR 对齐: 全局 shuffle)
    if is_main_process():
        print("\n加载训练集 ...")
    train_samples = log_cfg.get("train_samples", 0)
    train_samples = train_samples if train_samples > 0 else None
    train_dataset = KuaiVideoMapDataset(train_csv, item_vis_emb, cfg, max_samples=train_samples)
    if is_main_process():
        print("加载测试集 ...")
    eval_samples = log_cfg["eval_samples"] if log_cfg["eval_samples"] > 0 else None
    test_dataset = KuaiVideoMapDataset(test_csv, item_vis_emb, cfg, max_samples=eval_samples)

    # 创建 collate_fn (embedding 查表在 collate 阶段批量完成)
    collate = make_collate_fn(item_vis_emb)

    # DDP: 使用 DistributedSampler
    if is_dist():
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, shuffle=True
        )
    else:
        train_sampler = None

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

    # 梯度累积: micro_batch_size = batch_size / accum_steps
    accum_steps = train_cfg.get("gradient_accumulation_steps", 1)
    micro_batch_size = train_cfg["batch_size"] // accum_steps
    if is_main_process() and accum_steps > 1:
        print(f"\n梯度累积: {accum_steps} steps, "
              f"effective batch_size={train_cfg['batch_size']}, "
              f"micro_batch_size={micro_batch_size}")

    global_step = 0

    for epoch in range(train_cfg["num_epochs"]):
        if is_main_process():
            print(f"\n{'='*60}")
            print(f"Epoch {epoch + 1}/{train_cfg['num_epochs']}  "
                  f"(lr={optimizer.param_groups[0]['lr']:.2e})")
            print(f"{'='*60}")

        # DataLoader: micro_batch_size per step, accum_steps 次累积 = effective batch_size
        if is_dist():
            train_sampler.set_epoch(epoch)
        train_loader = DataLoader(
            train_dataset, batch_size=micro_batch_size,
            shuffle=(train_sampler is None),  # 非 DDP 时全局 shuffle
            sampler=train_sampler,
            collate_fn=collate, num_workers=4, pin_memory=True,
        )

        model.train()
        epoch_loss = 0.0
        epoch_samples = 0
        step = 0          # micro-step (每个 batch 算一次)
        optim_step = 0    # optimizer step (每 accum_steps 个 micro-step 算一次)
        t0 = time.time()
        optimizer.zero_grad()

        for batch in train_loader:
            (group_ids, uids, iids, i_vis,
             pos_items, pos_lens, neg_items, neg_lens,
             pos_vis, neg_vis,
             cate_ids, pos_cates, neg_cates,
             extra_cat_ids, extra_seq_ids, extra_seq_lens, numeric_vals,
             labels) = batch
            uids = uids.to(device)
            iids = iids.to(device)
            i_vis = i_vis.to(device)
            pos_items = pos_items.to(device)
            pos_lens = pos_lens.to(device)
            neg_items = neg_items.to(device)
            neg_lens = neg_lens.to(device)
            pos_vis = pos_vis.to(device)
            neg_vis = neg_vis.to(device)
            cate_ids = cate_ids.to(device)
            pos_cates = pos_cates.to(device)
            neg_cates = neg_cates.to(device)
            extra_cat_ids = extra_cat_ids.to(device)
            extra_seq_ids = extra_seq_ids.to(device)
            extra_seq_lens = extra_seq_lens.to(device)
            numeric_vals = numeric_vals.to(device)
            labels = labels.to(device)

            # DDP: 非累积最后一步时关闭梯度同步，节省通信开销
            is_accum_last = ((step + 1) % accum_steps == 0)
            from contextlib import nullcontext
            sync_ctx = model.no_sync() if (is_dist() and not is_accum_last) else nullcontext()

            with sync_ctx:
                model_output = model(
                    uids, iids, i_vis,
                    pos_items, pos_lens, neg_items, neg_lens,
                    pos_vis, neg_vis,
                    cate_ids, pos_cates, neg_cates,
                    extra_cat_ids, extra_seq_ids, extra_seq_lens, numeric_vals,
                )

                if arch == "tokenmixer_large":
                    # TokenMixer-Large returns (main_logits, aux_logits, emb_reg)
                    main_logits, aux_logits_tm, emb_reg = model_output
                    main_loss = criterion(main_logits, labels)
                    if aux_logits_tm.abs().sum() > 0:
                        aux_loss = criterion(aux_logits_tm, labels)
                        loss = main_loss + aux_loss_weight * aux_loss
                    else:
                        loss = main_loss
                        aux_loss = torch.tensor(0.0)
                    loss = loss + emb_reg
                else:
                    # Other models return (main_logits, aux_output)
                    main_logits, aux_output = model_output
                    main_loss = criterion(main_logits, labels)
                    loss = main_loss + aux_output
                    aux_loss = aux_output

                # 梯度累积: loss 除以 accum_steps 保证有效梯度与不累积时一致
                scaled_loss = loss / accum_steps

            # 反向传播
            scaled_loss.backward()

            bs = labels.size(0)
            epoch_loss += main_loss.item() * bs
            epoch_samples += bs
            step += 1

            # 累积够 accum_steps 步后执行 optimizer.step()
            if is_accum_last:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=train_cfg["grad_clip_norm"]
                )
                optimizer.step()
                optimizer.zero_grad()
                optim_step += 1
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

                if optim_step % log_cfg["log_interval"] == 0 and is_main_process():
                    avg = epoch_loss / epoch_samples
                    elapsed = time.time() - t0
                    speed = epoch_samples / elapsed
                    aux_name = {"tokenmixer_large": "Aux", "rankmixer": "Reg", "dmin": "Reg"}.get(arch, "Reg")
                    print(
                        f"  Step {optim_step:5d} | "
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

        # 处理尾部不足 accum_steps 的剩余梯度
        if step % accum_steps != 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=train_cfg["grad_clip_norm"]
            )
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

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
        test_loader = DataLoader(
            test_dataset, batch_size=train_cfg["batch_size"] * 2,
            shuffle=False, collate_fn=collate, num_workers=4, pin_memory=True,
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
                monitor_cfg = train_cfg.get("monitor", log_cfg.get("monitor"))
                current_monitor = compute_monitor_value(monitor_cfg, auc, gauc, logloss)
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
