import os
import pickle
import time
import zipfile
from collections import Counter

import h5py
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist


def is_main_process():
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0


def load_pretrained_embeddings(cfg):
    """Load feature-level pretrained embeddings declared by the dataset config."""
    data_cfg = cfg["data"]
    if not data_cfg.get("use_pretrained_emb", True):
        if is_main_process():
            print("跳过预训练 embedding 加载 (use_pretrained_emb=false)")
        return None

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, data_cfg["data_dir"])

    if is_main_process():
        print("加载预训练 item visual embedding ...", flush=True)

    with h5py.File(os.path.join(data_dir, data_cfg["item_emb_file"]), "r") as f:
        item_keys = f["key"][:]
        item_vals = f["value"][:].astype(np.float32)

    item_hash_size = data_cfg.get("raw_item_hash_size", data_cfg["item_hash_size"])
    item_vis_dim = data_cfg["item_vis_dim"]
    item_emb_table = np.zeros((item_hash_size, item_vis_dim), dtype=np.float32)
    item_emb_table[(item_keys.astype(np.int64) % item_hash_size)] = item_vals

    if is_main_process():
        print(f"  Item visual emb: {item_emb_table.shape}", flush=True)
    return item_emb_table


def _parse_seq_column(series, vocab_size, max_seq_len, value_map=None):
    ids_arr = np.zeros((len(series), max_seq_len), dtype=np.int32)
    lens_arr = np.zeros(len(series), dtype=np.int32)
    for i, val in enumerate(series):
        if not isinstance(val, str) or val == "":
            continue
        parts = val.split("^")
        if value_map is None:
            seq = [int(x) % vocab_size for x in parts]
        else:
            mapping = value_map["mapping"]
            oov_idx = int(value_map["oov_idx"])
            seq = [mapping.get(int(x), oov_idx) for x in parts]
        seq = seq[-max_seq_len:]
        ids_arr[i, :len(seq)] = seq
        lens_arr[i] = len(seq)
    return ids_arr, lens_arr


def _collect_seq_counts(series, counter):
    for val in series:
        if isinstance(val, str) and val:
            counter.update(int(x) for x in val.split("^") if x != "")


def _build_single_feature_map(counter, vocab_size, min_count):
    kept = sorted(k for k, v in counter.items() if v >= min_count)
    kept = kept[:max(int(vocab_size) - 2, 0)]
    return {
        "mapping": {raw_id: idx + 1 for idx, raw_id in enumerate(kept)},
        "oov_idx": int(vocab_size) - 1,
        "vocab_size": int(vocab_size),
    }


def _feature_map_specs(data_cfg):
    specs = {}
    user_col = data_cfg.get("user_col", "user_id")
    item_col = data_cfg.get("item_col", "item_id")
    cate_col = data_cfg.get("cate_col")

    if data_cfg.get("use_user_feature", True):
        specs[user_col] = {"vocab_size": data_cfg["num_users"], "scalar_cols": [user_col], "seq_cols": []}
    specs[item_col] = {"vocab_size": data_cfg["item_hash_size"], "scalar_cols": [item_col], "seq_cols": []}
    if cate_col and data_cfg.get("num_categories", 0) > 0:
        specs[cate_col] = {"vocab_size": data_cfg["num_categories"], "scalar_cols": [cate_col], "seq_cols": []}

    for col, share in (
        (data_cfg.get("pos_item_col"), item_col),
        (data_cfg.get("neg_item_col"), item_col),
        (data_cfg.get("pos_cate_col"), cate_col),
        (data_cfg.get("neg_cate_col"), cate_col),
    ):
        if col and share in specs:
            specs[share]["seq_cols"].append(col)

    for col_cfg in data_cfg.get("extra_categorical_cols", []):
        specs[col_cfg["name"]] = {
            "vocab_size": col_cfg["vocab_size"],
            "scalar_cols": [col_cfg["name"]],
            "seq_cols": [],
        }
    for col_cfg in data_cfg.get("extra_sequence_cols", []):
        share = col_cfg["share_embedding"]
        if share in specs:
            specs[share]["seq_cols"].append(col_cfg["name"])
    return specs


def _build_or_load_feature_maps(cfg, data_dir):
    data_cfg = cfg["data"]
    if not data_cfg.get("use_bars_feature_encoder", False):
        return None

    cache_version = data_cfg.get("cache_version", "v1")
    map_path = os.path.join(data_dir, f"feature_encoder_{cache_version}.pkl")
    if os.path.exists(map_path):
        if is_main_process():
            print(f"  加载 BARS feature encoder: {map_path}")
        with open(map_path, "rb") as f:
            return pickle.load(f)

    train_csv = os.path.join(data_dir, data_cfg["train_file"])
    min_count = int(data_cfg.get("min_categr_count", 10))
    specs = _feature_map_specs(data_cfg)
    usecols = []
    for spec in specs.values():
        for col in spec["scalar_cols"] + spec["seq_cols"]:
            if col and col not in usecols:
                usecols.append(col)

    if is_main_process():
        print(f"  构建 BARS feature encoder (min_categr_count={min_count}): {train_csv}")
        t0 = time.time()

    counters = {name: Counter() for name in specs}
    reader = pd.read_csv(
        train_csv,
        usecols=usecols,
        header=0,
        nrows=data_cfg.get("feature_encoder_samples"),
        chunksize=int(data_cfg.get("feature_encoder_chunksize", 500000)),
        engine="c",
        na_filter=False,
    )
    if isinstance(reader, pd.DataFrame):
        reader = [reader]
    for chunk in reader:
        for name, spec in specs.items():
            for col in spec["scalar_cols"]:
                counters[name].update(chunk[col].astype(np.int64).tolist())
            for col in spec["seq_cols"]:
                _collect_seq_counts(chunk[col], counters[name])

    feature_maps = {
        name: _build_single_feature_map(counters[name], spec["vocab_size"], min_count)
        for name, spec in specs.items()
    }
    with open(map_path, "wb") as f:
        pickle.dump(feature_maps, f, protocol=pickle.HIGHEST_PROTOCOL)

    if is_main_process():
        sizes = {k: len(v["mapping"]) + 2 for k, v in feature_maps.items()}
        print(f"  BARS feature encoder 已保存: {map_path} ({time.time() - t0:.1f}s)")
        print(f"  Remap vocab sizes: {sizes}")
    return feature_maps


def _map_scalar_column(series, value_map, vocab_size):
    if value_map is None:
        return np.minimum(series.values.astype(np.int64), vocab_size - 1).astype(np.int32)
    return series.map(value_map["mapping"]).fillna(value_map["oov_idx"]).values.astype(np.int32)


def _preprocess_csv_to_cache(csv_path, cache_path, cfg, feature_maps=None, max_samples=None):
    if is_main_process():
        print(f"  预处理 CSV → 缓存 (pandas 加速): {csv_path}")
        t0 = time.time()

    data_cfg = cfg["data"]
    num_users = data_cfg["num_users"]
    item_hash_size = data_cfg["item_hash_size"]
    num_categories = data_cfg.get("num_categories", 0)
    max_seq_len = data_cfg.get("max_seq_len", 100)
    user_col = data_cfg.get("user_col", "user_id")
    item_col = data_cfg.get("item_col", "item_id")
    label_col = data_cfg.get("label_col", "is_click")
    cate_col = data_cfg.get("cate_col")
    pos_item_col = data_cfg.get("pos_item_col", "pos_items")
    neg_item_col = data_cfg.get("neg_item_col", "neg_items")
    pos_cate_col = data_cfg.get("pos_cate_col")
    neg_cate_col = data_cfg.get("neg_cate_col")
    extra_cat_cols = data_cfg.get("extra_categorical_cols", [])
    extra_seq_cols = data_cfg.get("extra_sequence_cols", [])
    numeric_cols = data_cfg.get("numeric_cols", [])

    usecols = [user_col, item_col, label_col]
    for col in (cate_col, pos_item_col, neg_item_col, pos_cate_col, neg_cate_col):
        if col and col not in usecols:
            usecols.append(col)
    for col_cfg in extra_cat_cols + extra_seq_cols:
        col = col_cfg["name"]
        if col not in usecols:
            usecols.append(col)
    for col in numeric_cols:
        if col not in usecols:
            usecols.append(col)

    df = pd.read_csv(csv_path, usecols=usecols, header=0, nrows=max_samples, engine="c", na_filter=False)
    if is_main_process():
        print(f"    CSV 读取完成: {len(df)} 行, 耗时 {time.time() - t0:.1f}s")

    group_ids = df[user_col].values.astype(np.int64).astype(np.int32)
    raw_item_hash_size = data_cfg.get("raw_item_hash_size", item_hash_size)
    raw_iids = (df[item_col].values.astype(np.int64) % raw_item_hash_size).astype(np.int32)
    user_map = feature_maps.get(user_col) if feature_maps else None
    item_map = feature_maps.get(item_col) if feature_maps else None
    cate_map = feature_maps.get(cate_col) if (feature_maps and cate_col) else None

    uids = _map_scalar_column(df[user_col], user_map, num_users)
    iids = _map_scalar_column(df[item_col], item_map, item_hash_size)
    labels = df[label_col].values.astype(np.float32)
    cids = (
        _map_scalar_column(df[cate_col], cate_map, num_categories)
        if cate_col and num_categories > 0
        else np.zeros(len(df), dtype=np.int32)
    )

    if pos_item_col:
        if is_main_process():
            print(f"    解析 {pos_item_col} 序列 ...")
        raw_pos_ids, _ = _parse_seq_column(df[pos_item_col], raw_item_hash_size, max_seq_len)
        pos_ids, pos_lens = _parse_seq_column(df[pos_item_col], item_hash_size, max_seq_len, item_map)
    else:
        raw_pos_ids = np.zeros((len(df), max_seq_len), dtype=np.int32)
        pos_ids = np.zeros_like(raw_pos_ids)
        pos_lens = np.zeros(len(df), dtype=np.int32)

    if neg_item_col:
        if is_main_process():
            print(f"    解析 {neg_item_col} 序列 ...")
        raw_neg_ids, _ = _parse_seq_column(df[neg_item_col], raw_item_hash_size, max_seq_len)
        neg_ids, neg_lens = _parse_seq_column(df[neg_item_col], item_hash_size, max_seq_len, item_map)
    else:
        raw_neg_ids = np.zeros_like(raw_pos_ids)
        neg_ids = np.zeros_like(pos_ids)
        neg_lens = np.zeros_like(pos_lens)

    if pos_cate_col and num_categories > 0:
        if is_main_process():
            print(f"    解析 {pos_cate_col} 序列 ...")
        pos_cates, _ = _parse_seq_column(df[pos_cate_col], num_categories, max_seq_len, cate_map)
    else:
        pos_cates = np.zeros_like(pos_ids)

    if neg_cate_col and num_categories > 0:
        if is_main_process():
            print(f"    解析 {neg_cate_col} 序列 ...")
        neg_cates, _ = _parse_seq_column(df[neg_cate_col], num_categories, max_seq_len, cate_map)
    else:
        neg_cates = np.zeros_like(neg_ids)

    extra_cat_arr = (
        np.stack([
            _map_scalar_column(
                df[c["name"]],
                feature_maps.get(c["name"]) if feature_maps else None,
                int(c["vocab_size"]),
            )
            for c in extra_cat_cols
        ], axis=1)
        if extra_cat_cols
        else np.zeros((len(df), 0), dtype=np.int32)
    )

    extra_seq_ids, extra_seq_lens = [], []
    for c in extra_seq_cols:
        if is_main_process():
            print(f"    解析 {c['name']} 序列 ...")
        ids, lens = _parse_seq_column(
            df[c["name"]],
            int(c["vocab_size"]),
            max_seq_len,
            feature_maps.get(c["share_embedding"]) if feature_maps else None,
        )
        extra_seq_ids.append(ids)
        extra_seq_lens.append(lens)
    if extra_seq_ids:
        extra_seq_ids = np.stack(extra_seq_ids, axis=1)
        extra_seq_lens = np.stack(extra_seq_lens, axis=1)
    else:
        extra_seq_ids = np.zeros((len(df), 0, max_seq_len), dtype=np.int32)
        extra_seq_lens = np.zeros((len(df), 0), dtype=np.int32)

    numeric_vals = (
        np.stack([df[col].values.astype(np.float32) for col in numeric_cols], axis=1)
        if numeric_cols
        else np.zeros((len(df), 0), dtype=np.float32)
    )

    np.savez(
        cache_path,
        group_ids=group_ids,
        uids=uids,
        iids=iids,
        raw_iids=raw_iids,
        cids=cids,
        labels=labels,
        pos_ids=pos_ids,
        pos_lens=pos_lens,
        neg_ids=neg_ids,
        neg_lens=neg_lens,
        raw_pos_ids=raw_pos_ids,
        raw_neg_ids=raw_neg_ids,
        pos_cates=pos_cates,
        neg_cates=neg_cates,
        extra_cat_ids=extra_cat_arr,
        extra_seq_ids=extra_seq_ids,
        extra_seq_lens=extra_seq_lens,
        numeric_vals=numeric_vals,
    )
    if is_main_process():
        size_mb = os.path.getsize(cache_path + ".npz") / 1024 / 1024
        print(f"  缓存已保存: {cache_path}.npz ({size_mb:.1f} MB, {len(uids)} 条样本, 总耗时 {time.time() - t0:.1f}s)")

    return (
        group_ids, uids, iids, raw_iids, cids, labels,
        pos_ids, pos_lens, neg_ids, neg_lens, raw_pos_ids, raw_neg_ids,
        pos_cates, neg_cates, extra_cat_arr, extra_seq_ids, extra_seq_lens, numeric_vals,
    )


class CTRMapDataset(torch.utils.data.Dataset):
    """Map-style CTR dataset with BARS/FuxiCTR-compatible feature remapping."""

    def __init__(self, csv_path, item_vis_emb, cfg, max_samples=None):
        max_seq_len = cfg["data"].get("max_seq_len", 100)
        data_dir = os.path.dirname(csv_path)
        feature_maps = _build_or_load_feature_maps(cfg, data_dir)

        base = os.path.splitext(csv_path)[0]
        cache_version = cfg["data"].get("cache_version", "v1")
        suffix = f"_n{max_samples}" if max_samples else ""
        cache_path = f"{base}_cache_{cache_version}_seq{max_seq_len}{suffix}"
        cache_file = cache_path + ".npz"

        if os.path.exists(cache_file):
            if is_main_process():
                print(f"  加载缓存: {cache_file}")
            try:
                data = np.load(cache_file)
            except (OSError, EOFError, zipfile.BadZipFile):
                if is_main_process():
                    print(f"  缓存损坏，删除并重建: {cache_file}")
                    os.remove(cache_file)
                data = None
        else:
            data = None

        if data is not None:
            self.group_ids = data["group_ids"] if "group_ids" in data else data["uids"]
            self.uids = data["uids"]
            self.iids = data["iids"]
            self.raw_iids = data["raw_iids"] if "raw_iids" in data else self.iids
            self.cids = data["cids"] if "cids" in data else np.zeros_like(self.iids)
            self.labels = data["labels"]
            self.pos_ids = data["pos_ids"]
            self.pos_lens = data["pos_lens"]
            self.neg_ids = data["neg_ids"]
            self.neg_lens = data["neg_lens"]
            self.raw_pos_ids = data["raw_pos_ids"] if "raw_pos_ids" in data else self.pos_ids
            self.raw_neg_ids = data["raw_neg_ids"] if "raw_neg_ids" in data else self.neg_ids
            self.pos_cates = data["pos_cates"] if "pos_cates" in data else np.zeros_like(self.pos_ids)
            self.neg_cates = data["neg_cates"] if "neg_cates" in data else np.zeros_like(self.neg_ids)
            self.extra_cat_ids = data["extra_cat_ids"] if "extra_cat_ids" in data else np.zeros((len(self.uids), 0), dtype=np.int32)
            self.extra_seq_ids = data["extra_seq_ids"] if "extra_seq_ids" in data else np.zeros((len(self.uids), 0, max_seq_len), dtype=np.int32)
            self.extra_seq_lens = data["extra_seq_lens"] if "extra_seq_lens" in data else np.zeros((len(self.uids), 0), dtype=np.int32)
            self.numeric_vals = data["numeric_vals"] if "numeric_vals" in data else np.zeros((len(self.uids), 0), dtype=np.float32)
            if max_samples and len(self.uids) > max_samples:
                self._slice(max_samples)
            if is_main_process():
                print(f"  缓存加载完成: {len(self.uids)} 条样本")
        else:
            (
                self.group_ids, self.uids, self.iids, self.raw_iids, self.cids, self.labels,
                self.pos_ids, self.pos_lens, self.neg_ids, self.neg_lens,
                self.raw_pos_ids, self.raw_neg_ids, self.pos_cates, self.neg_cates,
                self.extra_cat_ids, self.extra_seq_ids, self.extra_seq_lens, self.numeric_vals,
            ) = _preprocess_csv_to_cache(csv_path, cache_path, cfg, feature_maps, max_samples=max_samples)

    def _slice(self, n):
        for name in (
            "group_ids", "uids", "iids", "raw_iids", "cids", "labels",
            "pos_ids", "pos_lens", "neg_ids", "neg_lens", "raw_pos_ids",
            "raw_neg_ids", "pos_cates", "neg_cates", "extra_cat_ids",
            "extra_seq_ids", "extra_seq_lens", "numeric_vals",
        ):
            setattr(self, name, getattr(self, name)[:n])

    def __len__(self):
        return len(self.uids)

    def __getitem__(self, idx):
        return (
            int(self.group_ids[idx]),
            int(self.uids[idx]),
            int(self.iids[idx]),
            int(self.raw_iids[idx]),
            int(self.cids[idx]),
            self.pos_ids[idx],
            int(self.pos_lens[idx]),
            self.neg_ids[idx],
            int(self.neg_lens[idx]),
            self.raw_pos_ids[idx],
            self.raw_neg_ids[idx],
            self.pos_cates[idx],
            self.neg_cates[idx],
            self.extra_cat_ids[idx],
            self.extra_seq_ids[idx],
            self.extra_seq_lens[idx],
            self.numeric_vals[idx],
            float(self.labels[idx]),
        )


def make_collate_fn(item_vis_emb):
    use_pretrained = item_vis_emb is not None

    def collate_fn(batch):
        (
            group_ids, uids, iids, raw_iids, cids, pos_ids_list, pos_lens,
            neg_ids_list, neg_lens, raw_pos_ids_list, raw_neg_ids_list,
            pos_cates_list, neg_cates_list, extra_cat_list, extra_seq_list,
            extra_seq_lens_list, numeric_list, labels,
        ) = zip(*batch)

        pos_ids_np = np.stack(pos_ids_list)
        neg_ids_np = np.stack(neg_ids_list)
        raw_pos_ids_np = np.stack(raw_pos_ids_list)
        raw_neg_ids_np = np.stack(raw_neg_ids_list)
        pos_cates_np = np.stack(pos_cates_list)
        neg_cates_np = np.stack(neg_cates_list)

        if use_pretrained:
            raw_iids_np = np.array(raw_iids, dtype=np.int32)
            item_vis_t = torch.from_numpy(item_vis_emb[raw_iids_np].copy())
            pos_vis_t = torch.from_numpy(item_vis_emb[raw_pos_ids_np].copy())
            neg_vis_t = torch.from_numpy(item_vis_emb[raw_neg_ids_np].copy())
        else:
            batch_size = len(uids)
            seq_len = pos_ids_np.shape[1]
            item_vis_t = torch.zeros(batch_size, 0, dtype=torch.float32)
            pos_vis_t = torch.zeros(batch_size, seq_len, 0, dtype=torch.float32)
            neg_vis_t = torch.zeros(batch_size, seq_len, 0, dtype=torch.float32)

        return (
            torch.tensor(group_ids, dtype=torch.long),
            torch.tensor(uids, dtype=torch.long),
            torch.tensor(iids, dtype=torch.long),
            item_vis_t,
            torch.from_numpy(pos_ids_np.astype(np.int64)),
            torch.tensor(pos_lens, dtype=torch.long),
            torch.from_numpy(neg_ids_np.astype(np.int64)),
            torch.tensor(neg_lens, dtype=torch.long),
            pos_vis_t,
            neg_vis_t,
            torch.tensor(cids, dtype=torch.long),
            torch.from_numpy(pos_cates_np.astype(np.int64)),
            torch.from_numpy(neg_cates_np.astype(np.int64)),
            torch.from_numpy(np.stack(extra_cat_list).astype(np.int64)),
            torch.from_numpy(np.stack(extra_seq_list).astype(np.int64)),
            torch.from_numpy(np.stack(extra_seq_lens_list).astype(np.int64)),
            torch.from_numpy(np.stack(numeric_list).astype(np.float32)),
            torch.tensor(labels, dtype=torch.float32),
        )

    return collate_fn
