# RankMixer

Unofficial implementation of **RankMixer: Scaling Up Ranking Models in Industrial Recommenders** (ByteDance, [arXiv:2507.15551](https://arxiv.org/abs/2507.15551)).

## Architecture

RankMixer replaces diverse handcrafted feature-interaction modules with a unified, scalable architecture:

```
Input Features → Semantic Grouping → Chunking (d) → T Tokens → Proj(D)
                                                          ↓
                                                  ┌───────────────┐
                                                  │ RankMixer Block │ × L
                                                  │  ├─ Multi-head  │
                                                  │  │  Token Mixing│
                                                  │  └─ Per-token   │
                                                  │     FFN / MoE   │
                                                  └───────────────┘
                                                          ↓
                                                  Mean Pooling → Output
```

**Key components:**
- **Multi-head Token Mixing**: Parameter-free reshape + transpose (H=T), replaces self-attention
- **Per-token FFN**: Each token has independent FFN parameters (not shared)
- **Sparse-MoE variant**: ReLU Routing + DTSI (Dense-Train / Sparse-Infer), replaces Dense FFN

**Scaling directions** (orthogonal):
- `T` — number of feature tokens
- `D` — hidden dimension
- `L` — number of layers
- `E` — number of experts (MoE variant)

**Parameter formula**: `#Param ≈ 2kLTD²` (Dense), `FLOPs ≈ 4kLTD²`

## Dataset

[KuaiVideo_x1](https://huggingface.co/datasets/reczoo/KuaiVideo_x1) — CTR prediction on short-video recommendation.

| Split | Samples | Features |
|-------|---------|----------|
| Train | ~10.93M | user_id, item_id, is_click, user/item visual embeddings (64d) |
| Test  | ~2.73M  | same |

## Quick Start

### 1. Prepare Data

```bash
python prepare_data.py
```

Downloads `KuaiVideo_x1.zip` (~2.27GB) from HuggingFace and extracts it.

### 2. Train

```bash
# RankMixer-Small (Dense, ~150M params) — 对应论文 RankMixer-100M
python train_kuaivideo.py --config config/kuaivideo_small.yaml

# RankMixer-Middle (MoE, E=8) — Sparse-MoE 扩展实验
python train_kuaivideo.py --config config/kuaivideo_middle.yaml

# RankMixer-Large (Dense, ~1.2B params) — 对应论文 RankMixer-1B
python train_kuaivideo.py --config config/kuaivideo_large.yaml
```

## Model Configurations

| Config | T | D | L | FFN Type | E | ~Params | Paper Reference |
|--------|---|---|---|----------|---|---------|-----------------|
| `kuaivideo_small.yaml` | 16 | 768 | 2 | Dense | — | 150M | RankMixer-100M |
| `kuaivideo_middle.yaml` | 16 | 1024 | 2 | Sparse-MoE | 8 | — | MoE extension (Sec 4.5) |
| `kuaivideo_large.yaml` | 32 | 1536 | 2 | Dense | — | 1.2B | RankMixer-1B |

> **Note**: Dense and MoE are **alternatives** (not stacked). MoE replaces the Per-token FFN with Sparse Mixture-of-Experts using ReLU routing.

## Project Structure

```
rankmixer/
├── rankmixer.py              # Core model implementation
├── train_kuaivideo.py        # Training script (loads config from YAML)
├── prepare_data.py           # Download & extract dataset
├── config/
│   ├── kuaivideo_small.yaml  # RankMixer-100M (Dense)
│   ├── kuaivideo_middle.yaml # MoE extension experiment
│   └── kuaivideo_large.yaml  # RankMixer-1B (Dense)
└── KuaiVideo_x1/             # Dataset (after prepare_data.py)
    ├── train.csv
    ├── test.csv
    ├── user_visual_emb_dim64.h5
    └── item_visual_emb_dim64.h5
```

## Requirements

- Python 3.8+
- PyTorch
- h5py, pyyaml, scikit-learn, numpy

```bash
pip install torch h5py pyyaml scikit-learn numpy
```

## Reference

```bibtex
@article{rankmixer2025,
  title={RankMixer: Scaling Up Ranking Models in Industrial Recommenders},
  author={ByteDance},
  journal={arXiv preprint arXiv:2507.15551},
  year={2025}
}
```
