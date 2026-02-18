# RankMixer & TokenMixer-Large

Unofficial implementation of:
- **RankMixer: Scaling Up Ranking Models in Industrial Recommenders** (ByteDance, [arXiv:2507.15551](https://arxiv.org/abs/2507.15551))
- **TokenMixer-Large: Scaling Up Large Ranking Models in Industrial Recommenders** (ByteDance, [arXiv:2602.06563](https://arxiv.org/abs/2602.06563))

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

**Key improvements in TokenMixer-Large over RankMixer:**

| Aspect | RankMixer | TokenMixer-Large |
|--------|-----------|-----------------|
| Block design | Token Mixing + FFN | Mixing & Reverting (aligned residuals) |
| FFN | Per-token GELU FFN | Per-token SwiGLU (gate/up/down) |
| Normalization | LayerNorm, Post-Norm | RMSNorm, Pre-Norm |
| Init | Standard | Down-matrix Small Init (σ=0.01) |
| Depth support | L=2 (shallow) | L=4+ with Inter-layer Residual + Aux Loss |
| Output | Mean Pooling | Global Token (like BERT [CLS]) |
| MoE routing | ReLU + DTSI | Top-k Softmax + Shared Expert + α scaling |
| MoE training | Dense-Train / Sparse-Infer | Sparse-Train / Sparse-Infer |

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

## Project Structure

```
rankmixer/
├── rankmixer.py              # RankMixer model (v1)
├── tokenmixer_large.py       # TokenMixer-Large model (v2)
├── train_kuaivideo.py        # Unified training script (auto-selects model by config)
├── prepare_data.py           # Download & extract dataset
├── config/
│   ├── kuaivideo_small.yaml  # RankMixer configs
│   ├── kuaivideo_middle.yaml
│   ├── kuaivideo_large.yaml
│   ├── tokenmixer_small.yaml # TokenMixer-Large configs
│   └── tokenmixer_middle.yaml
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
```
