"""
下载并解压 KuaiVideo_x1 数据集
来源: https://huggingface.co/datasets/reczoo/KuaiVideo_x1
"""

import os
import sys
import zipfile
import subprocess

DATASET_URL = "https://huggingface.co/datasets/reczoo/KuaiVideo_x1/resolve/main/KuaiVideo_x1.zip"
SAVE_DIR = os.path.dirname(os.path.abspath(__file__))
ZIP_FILE = os.path.join(SAVE_DIR, "KuaiVideo_x1.zip")
EXTRACT_DIR = os.path.join(SAVE_DIR, "KuaiVideo_x1")


def download():
    if os.path.exists(ZIP_FILE):
        print(f"[skip] {ZIP_FILE} 已存在，跳过下载")
        return

    print(f"[download] 正在从 HuggingFace 下载 KuaiVideo_x1.zip (~2.27GB) ...")
    print(f"  URL: {DATASET_URL}")
    print(f"  保存到: {ZIP_FILE}")

    # 优先用 wget，其次 curl
    try:
        subprocess.run(
            ["wget", "-O", ZIP_FILE, "--progress=bar:force", DATASET_URL],
            check=True,
        )
    except FileNotFoundError:
        # wget 不存在，用 curl
        subprocess.run(
            ["curl", "-L", "-o", ZIP_FILE, "--progress-bar", DATASET_URL],
            check=True,
        )

    print(f"[download] 下载完成: {ZIP_FILE}")


def extract():
    if os.path.exists(EXTRACT_DIR) and os.listdir(EXTRACT_DIR):
        print(f"[skip] {EXTRACT_DIR} 已存在且非空，跳过解压")
        return

    print(f"[extract] 正在解压 {ZIP_FILE} ...")
    with zipfile.ZipFile(ZIP_FILE, "r") as zf:
        # 检查 zip 内部是否自带 KuaiVideo_x1/ 顶层目录
        top_dirs = {name.split("/")[0] for name in zf.namelist() if "/" in name}
        if "KuaiVideo_x1" in top_dirs and len(top_dirs) == 1:
            # zip 内部已有 KuaiVideo_x1/ 目录，直接解压到项目根目录
            zf.extractall(SAVE_DIR)
        else:
            # zip 内部没有顶层目录，解压到 KuaiVideo_x1/ 里
            os.makedirs(EXTRACT_DIR, exist_ok=True)
            zf.extractall(EXTRACT_DIR)
    print(f"[extract] 解压完成 → {EXTRACT_DIR}")


def verify():
    expected = ["train.csv", "test.csv", "item_visual_emb_dim64.h5", "user_visual_emb_dim64.h5"]
    # 解压后的数据目录
    data_dir = os.path.join(SAVE_DIR, "KuaiVideo_x1")

    if not os.path.isdir(data_dir):
        print("[warn] 未找到解压后的数据目录 KuaiVideo_x1/")
        return

    print(f"\n[verify] 数据目录: {data_dir}")
    all_ok = True
    for fname in expected:
        fpath = os.path.join(data_dir, fname)
        if os.path.exists(fpath):
            size_mb = os.path.getsize(fpath) / 1024 / 1024
            print(f"  ✓ {fname} ({size_mb:.1f} MB)")
        else:
            print(f"  ✗ {fname} 缺失")
            all_ok = False

    if all_ok:
        print("\n[done] 数据准备完成，可以开始训练:")
        print(f"  python train_kuaivideo.py --config config/rankmixer_small.yaml")


if __name__ == "__main__":
    download()
    extract()
    verify()
