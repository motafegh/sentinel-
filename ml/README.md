# ml — SENTINEL Machine Learning Core

Dual-path smart contract vulnerability detector. A three-phase 7-layer **Graph Attention Network** encodes AST/CFG structure with typed edge relations; a **LoRA-adapted CodeBERT** encodes source text across sliding windows. A **three-eye CrossAttentionFusion** (GNN eye, Transformer eye, Fused eye) produces per-class probabilities across **10 vulnerability classes**.

**Current architecture: v7** — `MODEL_VERSION = "v7"`, `FEATURE_SCHEMA_VERSION = "v7"`, `NODE_FEATURE_DIM = 11`

---

## Table of Contents

- [Setup](#setup)
- [System Overview](#system-overview)
- [Data Pipeline](#data-pipeline)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
  - [GNN Encoder (7-layer, three-phase GAT)](#gnn-encoder)
  - [Transformer Encoder (CodeBERT + LoRA)](#transformer-encoder)
  - [CrossAttentionFusion](#crossattentionfusion)
  - [Three-Eye Classifier](#three-eye-classifier)
  - [Node Feature Vector (v7 Schema, 11-dim)](#node-feature-vector)
  - [Edge Types](#edge-types)
- [Output Classes](#output-classes)
- [Training](#training)
- [Inference](#inference)
- [Tests](#tests)
- [Repository Layout](#repository-layout)

---

## Setup

```bash
cd ~/projects/sentinel
poetry install
source ml/.venv/bin/activate
export TRANSFORMERS_OFFLINE=1          # required — prevents HuggingFace network calls
export TRITON_CACHE_DIR=/tmp/triton_cache  # required on WSL2 — avoids p9io crash
```

**Key dependencies:**

| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | `2.5.1+cu124` | Training, inference |
| `torch-geometric` | `2.7.0` | GNN layers, graph batching |
| `torch-scatter` | `2.1.2+pt25cu124` | GPU scatter kernels for pooling |
| `transformers` | `4.x` | CodeBERT base model |
| `peft` | `0.x` | LoRA adapters on CodeBERT |
| `slither-analyzer` | `>=0.9.3` | Graph extraction from Solidity |

---

## System Overview

```
Input: Solidity contract (.sol)
        │
        ├─ graph_extractor.py ──► .pt graph (NODE_FEATURE_DIM=11, 8 edge types)
        │
        └─ retokenize_windowed.py ► .pt tokens ([4, 512] windows)
                │
        DualPathDataset (cached_dataset_deduped.pkl)
                │
        ┌───────┴────────────────────────┐
        │                                │
   GNNEncoder                  TransformerEncoder
   7-layer GAT                 CodeBERT + LoRA r=16
   3 phases, JK attention       12 layers, Q+V adapters
   hidden_dim=256               frozen base, BF16
        │                                │
        └───────── CrossAttentionFusion ──┘
                   bidirectional cross-attention
                   node↔token, attn_dim=256
                   output_dim=128 LOCKED
                │
         Three-Eye Classifier
         GNN eye [B,256] + TF eye [B,256] + Fused [B,128]
         → [B,384] → Linear(384,192) → Linear(192,10)
                │
         [B, 10] logits → sigmoid → thresholded predictions
```

---

## Data Pipeline

Run scripts in this order for a full re-extraction:

```bash
# 1. Re-extract graphs from BCCC corpus
poetry run python ml/scripts/reextract_graphs.py

# 2. Tokenize all contracts (windowed, [4,512])
poetry run python ml/scripts/retokenize_windowed.py

# 3. Build + dedup label CSV
poetry run python ml/scripts/build_multilabel_index.py
poetry run python ml/scripts/dedup_multilabel_index.py --relabel-timestamp

# 4. Inject DoS augmented pairs
poetry run python ml/scripts/inject_augmented.py

# 5. Remove noisy labels (structural precondition heuristics)
poetry run python ml/scripts/label_cleaner.py \
    --graphs-dir ml/data/graphs \
    --label-csv ml/data/processed/multilabel_index_deduped.csv

# 6. Build paired cache
poetry run python ml/scripts/create_cache.py

# 7. Generate splits (only if needed — current splits are valid)
poetry run python ml/scripts/create_splits.py
```

**Current data state (v7):**

| File | Count / Size | Contents |
|------|-------------|----------|
| `ml/data/graphs/` | 41,522 .pt | v7 graphs, 11-dim, FEATURE_SCHEMA_VERSION="v7" |
| `ml/data/tokens_windowed/` | 44,470 .pt | windowed tokens [4,512] |
| `ml/data/cached_dataset_deduped.pkl` | 2.28 GB | 41,577 paired (graph, tokens) |
| `ml/data/processed/multilabel_index_cleaned.csv` | 44,524 rows | cleaned labels (training target) |
| `ml/data/splits/deduped/` | 3 .npy | train=29,103 / val=6,236 / test=6,237 |

2,948 stems in the CSV have no matching graph (expected Slither extraction failures). The cache builder excludes them automatically.

---

## Dataset

`DualPathDataset` (`ml/src/datasets/dual_path_dataset.py`) loads pairs from the pre-built `.pkl` cache. `dual_path_collate_fn` batches graph data via PyG `Batch.from_data_list` and stacks token tensors.

**Label distribution (cleaned v7 — training targets):**

| Class | Total | Train |
|-------|-------|-------|
| IntegerUO | 13,797 | 9,613 |
| GasException | 4,957 | ~3,500 |
| ExternalBug | 3,009 | ~2,100 |
| Reentrancy | 3,886 | 2,775 |
| TOD | 3,028 | ~2,100 |
| MishandledException | 2,442 | ~1,700 |
| CallToUnknown | 2,873 | ~2,000 |
| UnusedReturn | 1,051 | ~730 |
| Timestamp | 538 | ~375 |
| DenialOfService | 372 | ~260 |

59.3% of rows are all-zero. Handled by `WeightedRandomSampler` (3× weight for any-vuln rows) and `AsymmetricLoss` (γ⁻=2.0).

---

## Model Architecture

### GNN Encoder

**File:** `ml/src/models/gnn_encoder.py`

7-layer Graph Attention Network, three phases, Jumping Knowledge (JK) attention aggregation:

```
Phase 1 — Structural + CONTAINS (layers 1+2)
  GAT over edge types 0–5, 8 heads, add_self_loops=True
  LayerNorm after phase

Phase 2 — CONTROL_FLOW directed (layers 3+4+5, 3 hops)
  conv3:  CF edges (type 6), 1 head — first hop
  conv3b: 2nd hop — CALL→TMP→WRITE (CEI pattern)
  conv3c: 3rd hop — ENTRY→CALL→TMP→WRITE
  LayerNorm after phase

Phase 3 — REVERSE_CONTAINS type-7 (layers 6+7)
  Reversed CONTAINS edges (flipped at runtime, never written to .pt files)
  LayerNorm after phase

JK attention aggregation over all 7 layer outputs → hidden_dim=256
Edge type embedding: Embedding(8, 64) concatenated per message
```

### Transformer Encoder

**File:** `ml/src/models/transformer_encoder.py`

`microsoft/codebert-base` (124M params) + LoRA:
- Base model frozen; LoRA r=16, α=32 on Q+V of all 12 layers
- Input: `[B, 4, 512]` — 4 sliding windows of 512 tokens each
- Each window processed independently → `WindowAttentionPooler` → `[B, 768]`
- BF16 precision; not compiled (HuggingFace control flow isolates cleanly)

### CrossAttentionFusion

**File:** `ml/src/models/fusion_layer.py`

Bidirectional cross-attention fusing graph and text modalities:

```
1. Project nodes [N,256] → [N,256]
2. LayerNorm(768) + project tokens [B,512,768] → [B,512,256]
3. _scatter_to_dense → static [B,1024,256]  (compile-safe; max_nodes=1024)
4. Node→Token cross-attention → enriched_nodes [B,1024,256]
5. Token→Node cross-attention → enriched_tokens [B,512,256]
6. Masked mean pooling of real nodes  → [B,256]
7. Masked mean pooling of real tokens → [B,256]
8. Concat [B,512] → Linear + ReLU → [B,128]
```

**output_dim=128 is LOCKED** — the ZKML proxy MLP depends on this shape.

`_scatter_to_dense` replaces PyG's `to_dense_batch` to eliminate the `GuardOnDataDependentSymNode` compile graph break (zero graph breaks confirmed in production).

### Three-Eye Classifier

**File:** `ml/src/models/sentinel_model.py`

```
GNN eye:   max_pool + mean_pool over FUNCTION nodes → [B,512] → Linear → [B,128]
TF eye:    pooled token embedding → Linear(768,128) → [B,128]
Fused eye: CrossAttentionFusion → [B,128]

Concat [B,384] → Linear(384,192) → GELU → Linear(192,10) → logits
Aux heads: one Linear(128,10) per eye for auxiliary loss (training only)
```

### Node Feature Vector

**v7 schema, 11 dimensions:**

| Dim | Feature | Notes |
|-----|---------|-------|
| [0] | `type_id / 12.0` | Node type (0–12 → 0.0–1.0) |
| [1] | `visibility` | 0.0=public/external, 0.5=internal, 1.0=private |
| [2] | `uses_block_globals` | 1.0 if reads block.timestamp/number/difficulty/etc. |
| [3] | `view` | 0/1 |
| [4] | `payable` | 0/1 |
| [5] | `complexity` | log1p(CFG_block_count) / log1p(100) |
| [6] | `loc` | log1p(lines) / log1p(1000) |
| [7] | `return_ignored` | 1.0 if call return value unused downstream |
| [8] | `call_target_typed` | 0=dynamic/unknown target, 1=typed interface |
| [9] | `has_loop` | 0/1 |
| [10] | `external_call_count` | log1p(count) / log1p(20) |

CFG nodes inherit dims [1,3,4,5,9] from their parent FUNCTION node. `in_unchecked` was removed in v7 (dead feature for 87.9% of dataset).

### Edge Types

| Type | Name | Description |
|------|------|-------------|
| 0 | CALLS | function → called function (internal) |
| 1 | READS | function → state variable |
| 2 | WRITES | function → state variable |
| 3 | EMITS | function → event |
| 4 | INHERITS | contract → parent contract |
| 5 | CONTAINS | contract/function → child node |
| 6 | CONTROL_FLOW | CFG block → CFG block |
| 7 | REVERSE_CONTAINS | flip of type 5, generated at runtime |

---

## Output Classes

10 vulnerability classes (fixed, append-only — changing order breaks all checkpoints):

| Index | Class | Description |
|-------|-------|-------------|
| 0 | CallToUnknown | Low-level calls to unknown addresses |
| 1 | DenialOfService | Gas griefing / unbounded loops |
| 2 | ExternalBug | External contract dependency bugs |
| 3 | GasException | Unchecked send/transfer failures |
| 4 | IntegerUO | Integer overflow/underflow (Solidity <0.8) |
| 5 | MishandledException | Unchecked return values from external calls |
| 6 | Reentrancy | Cross-function/cross-contract reentrancy |
| 7 | Timestamp | Block timestamp manipulation |
| 8 | TransactionOrderDependence | Front-running / TOD |
| 9 | UnusedReturn | Return value of internal functions ignored |

---

## Training

### Launch

```bash
source ml/.venv/bin/activate
TRANSFORMERS_OFFLINE=1 TRITON_CACHE_DIR=/tmp/triton_cache PYTHONPATH=. nohup \
    python ml/scripts/train.py \
    --run-name v7.0 --experiment-name sentinel-v7 \
    --epochs 100 --gradient-accumulation-steps 8 \
    > ml/logs/v7.0-launch.log 2>&1 &

# Monitor:
bash ml/scripts/monitor.sh
```

All v7 defaults are correct. No override flags needed beyond `--gradient-accumulation-steps 8`.

### TrainConfig Reference (v7)

| Parameter | Default | Notes |
|-----------|---------|-------|
| `batch_size` | `8` | 6.9 / 8.0 GB VRAM; 16 saturates |
| `gradient_accumulation_steps` | `8` | effective batch = 64 |
| `gnn_hidden_dim` | `256` | |
| `gnn_layers` | `7` | 2+3+2 three-phase |
| `lora_r` | `16` | LoRA rank on Q+V |
| `epochs` | `100` | |
| `patience` | `30` | early stopping |
| `loss_fn` | `"asl"` | AsymmetricLoss(γ⁻=2.0, γ⁺=1.0, clip=0.01) |
| `use_weighted_sampler` | `"positive"` | 3× for any-vuln rows |
| `dos_loss_weight` | `0.0` | DoS detached — data-starved |
| `pos_weight_min_samples` | `3000` | caps pos_weight for small classes |
| `num_workers` | `4` | fork, CoW cache — zero extra RAM |
| `use_compile` | `True` | submodule-level, skip transformer |
| `use_amp` | `True` | BF16; no GradScaler |

### torch.compile Strategy

Submodule-level (not whole-model): `gnn`, `fusion`, `classifier`, eye projectors, aux heads are compiled. `model.transformer` (CodeBERT+LoRA) is skipped — its HuggingFace control flow causes graph breaks that contaminate the GNN compile context when compiled together. `cache_size_limit=256` prevents dynamo falling back after 8 unique shapes.

### Post-Training Workflow

```bash
# Threshold tuning
poetry run python ml/scripts/tune_threshold.py --checkpoint ml/checkpoints/v7.0_best.pt

# Behavioral gate (≥80% detection, ≥80% specificity required)
poetry run python ml/scripts/manual_test.py --checkpoint ml/checkpoints/v7.0_best.pt

# Promote to production
poetry run python ml/scripts/promote_model.py --checkpoint ml/checkpoints/v7.0_best.pt
```

### v7 Targets

| Class | v5.2 F1 | v7 Target |
|-------|---------|-----------|
| IntegerUO | 0.732 | ≥ 0.75 |
| GasException | 0.407 | ≥ 0.45 |
| Reentrancy | 0.322 | ≥ 0.45 |
| MishandledException | 0.342 | ≥ 0.50 |
| UnusedReturn | 0.238 | ≥ 0.45 |
| Timestamp | 0.174 | ≥ 0.30 |
| DenialOfService | 0.329 | ≥ 0.35 |
| CallToUnknown | 0.284 | ≥ 0.35 |
| TOD | 0.283 | ≥ 0.30 |
| ExternalBug | 0.262 | ≥ 0.30 |
| **Macro avg** | **0.3422** | **≥ 0.45** |

---

## Inference

```bash
TRANSFORMERS_OFFLINE=1 PYTHONPATH=. uvicorn ml.src.inference.api:app --port 8001
```

**Endpoints:**
- `POST /predict` — `{"contract_source": "..."}` → per-class probabilities + detections
- `GET /health` — liveness
- `GET /metrics` — Prometheus

`predictor.py` detects checkpoint architecture via `_ARCH_TO_FUSION_DIM` / `_ARCH_TO_NODE_DIM`. `three_eye_v7` → fusion_dim=128, node_dim=11.

---

## Tests

```bash
cd ml && poetry run pytest tests/ -v
```

| File | Coverage |
|------|---------|
| `test_preprocessing.py` | Schema (NODE_FEATURE_DIM=11, 13 types), feature builders, CFG inheritance |
| `test_model.py` | Forward pass shapes, aux output, [B,10] output |
| `test_training.py` | TrainConfig, ASL loss, gradient flow |
| `test_cache.py` | Cache key, schema-version invalidation, atomic write |
| `test_dataset.py` | DualPathDataset loading, collate, batch shapes |

---

## Repository Layout

```
ml/
├── README.md                       ← this file
├── pyproject.toml
├── scripts/                        ← see scripts/README.md
│   ├── train.py                    ← training entry point
│   ├── label_cleaner.py
│   ├── create_cache.py
│   ├── reextract_graphs.py
│   ├── retokenize_windowed.py
│   ├── tune_threshold.py
│   ├── manual_test.py
│   ├── monitor.sh
│   └── archive/                    ← completed one-off scripts
├── src/
│   ├── models/
│   │   ├── sentinel_model.py       ← SentinelModel v7, three-eye
│   │   ├── gnn_encoder.py          ← 7-layer three-phase GAT
│   │   ├── transformer_encoder.py  ← CodeBERT + LoRA
│   │   └── fusion_layer.py         ← CrossAttentionFusion, compile-safe
│   ├── preprocessing/
│   │   ├── graph_schema.py         ← NODE_FEATURE_DIM=11, FEATURE_SCHEMA_VERSION="v7"
│   │   └── graph_extractor.py      ← Slither → v7 graph .pt files
│   ├── datasets/
│   │   └── dual_path_dataset.py
│   ├── training/
│   │   ├── trainer.py              ← TrainConfig, train(), BF16, submodule compile
│   │   └── losses.py               ← AsymmetricLoss
│   └── inference/
│       ├── api.py                  ← FastAPI :8001
│       ├── predictor.py
│       └── preprocess.py
├── data/                           ← NOT committed (.gitignore)
│   ├── graphs/                     ← 41,522 .pt graph files (v7, 11-dim)
│   ├── tokens_windowed/            ← 44,470 .pt token files ([4,512])
│   ├── processed/                  ← CSV label files
│   ├── splits/                     ← train/val/test .npy indices
│   └── cached_dataset_deduped.pkl  ← 2.28 GB paired cache
├── checkpoints/                    ← NOT committed
└── logs/                           ← NOT committed
```

---

## Key Invariants

- `NODE_FEATURE_DIM=11` and `NUM_CLASSES=10` are locked. Any change requires full re-extraction + retraining.
- `fusion_output_dim=128` is locked — the ZKML proxy MLP (M2) depends on it.
- Bump `FEATURE_SCHEMA_VERSION` in `graph_schema.py` after any schema change to invalidate inference caches.
- `weights_only=False` for all `.pt` files — PyG 2.7 metadata and PEFT LoRA objects are not safe-tensors serialisable.
- Always set `TRITON_CACHE_DIR=/tmp/triton_cache` on WSL2 before training.
