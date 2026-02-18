# RankMixer & TokenMixer-Large & HSTU

Unofficial implementation of:
- **RankMixer: Scaling Up Ranking Models in Industrial Recommenders** (ByteDance, [arXiv:2507.15551](https://arxiv.org/abs/2507.15551))
- **TokenMixer-Large: Scaling Up Large Ranking Models in Industrial Recommenders** (ByteDance, [arXiv:2602.06563](https://arxiv.org/abs/2602.06563))
- **Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations** (Meta, [arXiv:2402.17152](https://arxiv.org/abs/2402.17152))

## Architecture

### RankMixer (v1)

```
Input Features → Chunking (d) → T Tokens → Proj(D)
                                      ↓
                              ┌───────────────┐
                              │ RankMixer Block │ × L
                              │  ├─ Token Mixing│  (parameter-free)
                              │  └─ Per-token   │
                              │     FFN (GELU)  │
                              └───────────────┘
                                      ↓
                              Mean Pooling → Output
```

### TokenMixer-Large (v2, improved)

```
Input Features → Semantic Group Tokenizer → T Tokens + Global Token → Proj(D)
                                                      ↓
                                        ┌─────────────────────────┐
                                        │ TokenMixer-Large Block   │ × L
                                        │  ├─ Mixing (reshape)     │
                                        │  ├─ pSwiGLU + Pre-Norm   │
                                        │  ├─ Reverting (reshape)   │
                                        │  └─ pSwiGLU + Pre-Norm   │
                                        │  + Inter-layer Residual   │
                                        │  + Auxiliary Loss         │
                                        └─────────────────────────┘
                                                      ↓
                                          Global Token → Output
```

### HSTU (Meta, Generative Recommenders)

```
Input Features → Chunking (d) → T Tokens → Proj(D)
                                      ↓
                              ┌───────────────────┐
                              │ HSTU Layer         │ × L
                              │  ├─ f1: SiLU proj  │ → U, V, Q, K
                              │  ├─ Attention:      │
                              │  │   SiLU(QK^T+rab) │ (no Softmax!)
                              │  │   × V            │
                              │  └─ f2: Norm(·)⊙U  │ → FFN
                              └───────────────────┘
                                      ↓
                              Mean Pooling → Output
```

**Key differences across models:**

| Aspect | RankMixer | TokenMixer-Large | HSTU |
|--------|-----------|-----------------|------|
| Token interaction | Reshape mixing (param-free) | Mixing & Reverting | Multi-head attention (no Softmax) |
| FFN | Per-token GELU FFN | Per-token SwiGLU | Shared FFN + U gating |
| Normalization | LayerNorm, Post-Norm | RMSNorm, Pre-Norm | RMSNorm |
| Attention | None | None | SiLU(QK^T + rel_bias) ⊙ V |
| Position encoding | None | None | Learned relative bias |
| Output | Mean Pooling | Global Token | Mean Pooling |
| MoE | ReLU + DTSI | Top-k + Shared Expert | None |
| Origin | ByteDance | ByteDance | Meta |

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

### 2. Train RankMixer

```bash
# Small (Dense, ~150M) — 论文 RankMixer-100M
python train_kuaivideo.py --config config/kuaivideo_small.yaml

# Middle (MoE, E=8) — Sparse-MoE 扩展实验
python train_kuaivideo.py --config config/kuaivideo_middle.yaml

# Large (Dense, ~1.2B) — 论文 RankMixer-1B
python train_kuaivideo.py --config config/kuaivideo_large.yaml
```

### 3. Train TokenMixer-Large

```bash
# Small (Dense, L=4)
python train_kuaivideo.py --config config/tokenmixer_small.yaml

# Middle (MoE, L=4, E=8, top_k=2)
python train_kuaivideo.py --config config/tokenmixer_middle.yaml
```

### 4. Train HSTU

```bash
# Small (L=4, H=8, ~36M)
python train_kuaivideo.py --config config/hstu_small.yaml

# Middle (L=8, H=16, ~58M)
python train_kuaivideo.py --config config/hstu_middle.yaml

# Large (L=12, H=32, ~184M)
python train_kuaivideo.py --config config/hstu_large.yaml
```

## Model Configurations

### RankMixer

| Config | T | D | L | FFN Type | E | Paper Reference |
|--------|---|---|---|----------|---|-----------------|
| `kuaivideo_small.yaml` | 16 | 768 | 2 | Dense | — | RankMixer-100M |
| `kuaivideo_middle.yaml` | 16 | 1024 | 2 | Sparse-MoE | 8 | MoE extension |
| `kuaivideo_large.yaml` | 32 | 1536 | 2 | Dense | — | RankMixer-1B |

### TokenMixer-Large

| Config | T+1 | D | L | FFN Type | E | top_k |
|--------|-----|---|---|----------|---|-------|
| `tokenmixer_small.yaml` | 17 | 544 | 4 | Dense pSwiGLU | — | — |
| `tokenmixer_middle.yaml` | 17 | 544 | 4 | Sparse-Pertoken MoE | 8 | 2 |

### HSTU

| Config | T | D | L | Heads | Head dim | ~Params |
|--------|---|---|---|-------|----------|---------|
| `hstu_small.yaml` | 16 | 256 | 4 | 8 | 32 | 36M |
| `hstu_middle.yaml` | 16 | 512 | 8 | 16 | 32 | 58M |
| `hstu_large.yaml` | 32 | 1024 | 12 | 32 | 32 | 184M |

## Project Structure

```
rankmixer/
├── rankmixer.py              # RankMixer model (v1)
├── tokenmixer_large.py       # TokenMixer-Large model (v2)
├── hstu.py                   # HSTU model (Meta)
├── train_kuaivideo.py        # Unified training script (auto-selects model by config)
├── prepare_data.py           # Download & extract dataset
├── config/
│   ├── kuaivideo_small.yaml  # RankMixer configs
│   ├── kuaivideo_middle.yaml
│   ├── kuaivideo_large.yaml
│   ├── tokenmixer_small.yaml # TokenMixer-Large configs
│   ├── tokenmixer_middle.yaml
│   ├── hstu_small.yaml       # HSTU configs
│   ├── hstu_middle.yaml
│   └── hstu_large.yaml
└── KuaiVideo_x1/             # Dataset
```

## Requirements

```bash
pip install torch h5py pyyaml scikit-learn numpy
```

## References

```bibtex
@article{rankmixer2025,
  title={RankMixer: Scaling Up Ranking Models in Industrial Recommenders},
  author={ByteDance},
  journal={arXiv preprint arXiv:2507.15551},
  year={2025}
}

@article{tokenmixerlarge2025,
  title={TokenMixer-Large: Scaling Up Large Ranking Models in Industrial Recommenders},
  author={ByteDance},
  journal={arXiv preprint arXiv:2602.06563},
  year={2025}
}

@inproceedings{hstu2024,
  title={Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations},
  author={Zhai, Jiaqi and Liao, Lucy and Liu, Xing and Wang, Yueming and Li, Rui and Cao, Xuan and Gao, Leon and Gong, Zhaojie and Gu, Fangda and He, Jiayuan and Lu, Yinghai and Shi, Yu},
  booktitle={ICML},
  year={2024}
}
```
