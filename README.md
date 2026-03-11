# 6G CSI Foundation Model

**Large Model for Channel State Information Compression and Reconstruction in 6G Intelligent Communication**

A Transformer + BERT-style masked self-supervised pretraining framework for 6G Massive MIMO CSI compression and reconstruction.

---

## Project Structure

```
.
├── configs/
│   ├── pretrain.yaml          # BERT-style masked pretraining config
│   └── finetune.yaml          # Compression & reconstruction finetune config
├── data/                      # Dataset files (not tracked by git)
├── scripts/
│   ├── 0_generate_toy_dataset.py   # Generate synthetic CSI data
│   ├── 1_pretrain.py               # Run masked pretraining
│   └── 2_finetune_compression.py   # Run compression finetuning
├── src/
│   ├── data/
│   │   ├── csi_dataset.py     # Dataset: load .npz / .pt complex CSI
│   │   └── tokenizer.py       # CSI tokenizer + BERT masking
│   ├── metrics/
│   │   └── nmse.py            # NMSE metric for complex CSI
│   ├── models/
│   │   ├── transformer_block.py   # TransformerBlock + TransformerEncoder
│   │   ├── pretrain_model.py      # Masked CSI pretraining model
│   │   └── compression_model.py   # Encoder → bottleneck → decoder
│   └── train/
│       ├── pretrain_loop.py   # Pretraining loop (with optional AMP)
│       └── finetune_loop.py   # Finetuning loop (with optional AMP)
├── tests/
│   ├── test_shapes.py         # Shape verification for all modules
│   └── test_tokenization.py   # Tokenize/detokenize round-trip tests
├── requirements.txt
└── README.md
```

---

## CSI Data Format

- **Input shape per sample**: `[N_sc, N_r, N_t]` complex64
- **DataLoader batch**: `[B, N_sc, N_r, N_t]` complex64
- **Tokenisation**: Flatten `N_r × N_t` antenna pairs → `N_r*N_t` tokens, each with `N_sc*2` Re/Im features
- **After projection**: `[B, N_r*N_t, d_model]`

---

## Quick Start (End-to-End)

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Generate synthetic dataset

```bash
python scripts/0_generate_toy_dataset.py
# Creates data/csi_toy.npz and data/csi_toy.pt
# Default: 2000 samples, N_sc=32, N_r=4, N_t=4
```

Customise dimensions:
```bash
python scripts/0_generate_toy_dataset.py \
    --n_samples 5000 --n_sc 64 --n_r 8 --n_t 4
```

### 3. Pretrain (BERT-style masked CSI modelling)

```bash
python scripts/1_pretrain.py --config configs/pretrain.yaml
```

Checkpoints are saved to `checkpoints/pretrain/`.

### 4. Finetune for compression & reconstruction

```bash
# Without pretrained backbone
python scripts/2_finetune_compression.py --config configs/finetune.yaml

# With pretrained backbone
python scripts/2_finetune_compression.py \
    --config configs/finetune.yaml \
    --pretrained checkpoints/pretrain/pretrain_epoch0100.pt
```

Checkpoints are saved to `checkpoints/finetune/`.

---

## Architecture Overview

### Pretraining (BERT-style masked CSI modelling)

```
H [B, N_sc, N_r, N_t] (complex)
  └─ CSITokenizer (flatten + Re/Im + linear projection)
       └─ [B, N_r*N_t, d_model]
            └─ Random masking (mask_ratio % of tokens)
                 └─ TransformerEncoder (N_layers × Block)
                      └─ Reconstruction head (Linear)
                           └─ MSE loss on masked positions
```

### Downstream: Compression & Reconstruction

```
H [B, N_sc, N_r, N_t] (complex)
  └─ CSITokenizer → TransformerEncoder → mean-pool
                                        → Bottleneck Linear → z [B, latent_dim]
                                                              → Expand → TransformerDecoder
                                                                          → CSITokenizer.detokenize
                                                                               → Ĥ [B, N_sc, N_r, N_t]
                                                                                    └─ NMSE loss
```

**Compression ratio** is controlled by `latent_dim` in `configs/finetune.yaml`.

---

## Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_tokenization.py -v
python -m pytest tests/test_shapes.py -v
```

---

## Configuration

Edit `configs/pretrain.yaml` or `configs/finetune.yaml` to adjust:

| Parameter | Description |
|-----------|-------------|
| `csi.n_sc` | Number of subcarriers |
| `csi.n_r` | Receive antennas |
| `csi.n_t` | Transmit antennas |
| `model.d_model` | Transformer hidden dimension |
| `model.n_heads` | Number of attention heads |
| `model.n_layers` | Number of Transformer blocks |
| `model.latent_dim` | Bottleneck size (controls compression ratio) |
| `model.mask_ratio` | Fraction of tokens masked during pretraining |
| `train.use_amp` | Enable AMP mixed precision (GPU only) |
| `train.device` | `"cuda"` or `"cpu"` |

---

## Metrics

- **NMSE**: `E[||H - Ĥ||² / ||H||²]`
- **NMSE (dB)**: `10 · log₁₀(NMSE)`

---

## Dependencies

```
torch >= 2.0.0
numpy >= 1.24.0
pyyaml >= 6.0
tqdm >= 4.65.0
```
