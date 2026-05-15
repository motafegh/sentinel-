# ml — SENTINEL Machine Learning Core

Dual-path smart contract vulnerability detector. A three-phase **Graph Attention Network** encodes AST/CFG structure with typed edge relations; a **LoRA-adapted CodeBERT** encodes source text. A **three-eye CrossAttentionFusion** (GNN eye, Transformer eye, Fused eye) produces per-class probabilities across **10 vulnerability classes**.

**Current architecture: v5.2** (`MODEL_VERSION = "v5.2"`, `FEATURE_SCHEMA_VERSION = "v3"`)

---

## Table of Contents

- [Setup](#setup)
- [System Overview](#system-overview)
- [Shared Preprocessing Layer](#shared-preprocessing-layer)
- [Data Preparation](#data-preparation)
- [Dataset — DualPathDataset](#dataset--dualpathdataset)
- [Model Architecture](#model-architecture)
  - [SentinelModel — Orchestration (v5)](#sentinelmodel--orchestration-v5)
  - [GNN Encoder (three-phase GAT)](#gnn-encoder-three-phase-gat)
  - [Transformer Encoder (CodeBERT + LoRA)](#transformer-encoder-codebert--lora)
  - [CrossAttention Fusion](#crossattentionFusion)
  - [Classifier and Auxiliary Heads](#classifier-and-auxiliary-heads)
  - [Node Feature Vector (v2 Schema)](#node-feature-vector-v2-schema)
  - [Edge Types (v3 Schema)](#edge-types-v3-schema)
- [Output Classes](#output-classes)
- [Training](#training)
  - [Run Training](#run-training)
  - [TrainConfig Reference](#trainconfig-reference)
  - [Loss Functions](#loss-functions)
  - [Per-Class Threshold Tuning](#per-class-threshold-tuning)
  - [Model Promotion](#model-promotion)
- [Inference API](#inference-api)
  - [Endpoints](#endpoints)
  - [HTTP Status Codes](#http-status-codes)
- [Predictor](#predictor)
- [ContractPreprocessor](#contractpreprocessor)
- [Inference Cache](#inference-cache)
- [Drift Detection](#drift-detection)
- [MLflow and Model Registry](#mlflow-and-model-registry)
- [DVC](#dvc)
- [Docker — Slither Environment](#docker--slither-environment)
- [Testing](#testing)
- [Scripts Reference](#scripts-reference)
- [File Reference](#file-reference)
- [Critical Constraints](#critical-constraints)

---

## Setup

```bash
# Python 3.12.1 required — strict pin in pyproject.toml
cd ml
poetry install

# TRANSFORMERS_OFFLINE must be exported at shell level before any Python import.
# HuggingFace checks it at import time — os.environ inside Python is too late.
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
```

Key runtime dependencies (from `pyproject.toml`):

| Package | Version | Role |
|---|---|---|
| `torch` | `^2.5.0` | Core deep learning |
| `torch-geometric` | `^2.6.0` | GNN layers, `Data`, `Batch` |
| `transformers` | `^4.45.0` | CodeBERT backbone |
| `peft` | `>=0.13.0,<0.16.0` | LoRA adapters (**hard requirement** — missing raises `RuntimeError` at import) |
| `fastapi` | `^0.115.0` | Inference HTTP API |
| `mlflow` | `^2.17.0` | Experiment tracking and model registry |
| `scipy` | `^1.13.0` | KS test for drift detection |
| `slither-analyzer` | `>=0.9.3` | Graph extraction (enforced at `graph_schema.py` import) |

Hardware: tested on RTX 3070 8 GB with AMP (BF16/TF32). CPU inference is supported; Slither graph extraction requires `solc` installed locally or via Docker.

---

## System Overview

```
BCCC-SCsVul-2024 corpus (~111K rows, 10 vulnerability folders + NonVulnerable)
         │
         ▼
  ast_extractor.py (offline, 11 workers)           tokenizer (CodeBERT)
  ─────────────────────────────────────            ────────────────────
  .sol → Slither AST/CFG → PyG Data                .sol → input_ids [512]
  graph.x [N, 12]  (v2 node features)              attention_mask [512]
  graph.edge_index [2, E]
  graph.edge_attr  [E]  (type ids 0-6)
         │                                               │
         ▼                                               ▼
  ml/data/graphs/<md5>.pt                  ml/data/tokens/<md5>.pt
         │                                               │
         └──────────────┬────────────────────────────────┘
                        ▼
               DualPathDataset + dual_path_collate_fn
                        │
                        ▼
                  SentinelModel v5.2
              ┌─────────┬──────────┐
              │ GNN eye │ TF eye   │ Fused eye
              │ [B,128] │ [B,128]  │ [B,128]
              └────────────────────┘
                   cat → [B, 384]
                   Linear(384, 10)
                        │
                   logits [B, 10]
                   Sigmoid → probs
                   per-class thresholds
                        │
                   vulnerability report
```

---

## Shared Preprocessing Layer

Graph construction logic is centralised in `ml/src/preprocessing/` — the **single source of truth** for both the offline batch extractor and online inference preprocessor. Any divergence between them would silently corrupt inference (wrong features, no error signal).

```
ml/src/preprocessing/
├── graph_schema.py      — NODE_TYPES, EDGE_TYPES, FEATURE_NAMES, FEATURE_SCHEMA_VERSION
└── graph_extractor.py   — extract_contract_graph(), typed exceptions, GraphExtractionConfig
```

### Schema Change Protocol

Any modification to `NODE_TYPES`, `VISIBILITY_MAP`, `EDGE_TYPES`, or `FEATURE_NAMES` (or the logic in `_build_node_features()`) requires **all four** of the following steps:

1. Rebuild all ~44K `.pt` graph files — `python ml/scripts/reextract_graphs.py`
2. Rebuild all token `.pt` files (only if tokenizer logic changed)
3. Retrain the model from scratch — `python ml/scripts/train.py`
4. Increment `FEATURE_SCHEMA_VERSION` in `graph_schema.py` to invalidate all inference caches

Skipping any step causes silent accuracy regression.

---

## Data Preparation

All commands from project root (`~/projects/sentinel`).

### Step 1 — Graph Extraction

```bash
# Offline batch pipeline — 11 workers, checkpoint/resume system
# Reads BCCC parquet, resolves solc version per group, writes <md5>.pt to ml/data/graphs/
PYTHONPATH=. poetry run python ml/src/data_extraction/ast_extractor.py

# Or re-extract an existing dataset with v2 schema
PYTHONPATH=. poetry run python ml/scripts/reextract_graphs.py
```

Output: `ml/data/graphs/<md5_hash>.pt` — one file per contract, containing:
- `graph.x` — `[N, 12]` float32 node feature matrix
- `graph.edge_index` — `[2, E]` int64
- `graph.edge_attr` — `[E]` int64 edge type ids (0–6)
- `graph.contract_path`, `graph.contract_hash`, `graph.y`

### Step 2 — Build Multi-Label Index

```bash
PYTHONPATH=. poetry run python ml/scripts/build_multilabel_index.py
```

Bridges the BCCC SHA256 hash (filename in SourceCodes/) and the internal MD5 hash (`.pt` stem) via `graph.contract_path`. Groups by SHA256, applies `max()` over Class columns (OR semantics). Output: `ml/data/processed/multilabel_index.csv` (~68K rows, pre-dedup — see Step 3).

### Step 3 — Deduplicate

```bash
PYTHONPATH=. poetry run python ml/scripts/dedup_multilabel_index.py
```

Output: `ml/data/processed/multilabel_index_deduped.csv` — used by `DualPathDataset` as default label source.

### Step 4 — Create Stratified Splits

```bash
PYTHONPATH=. poetry run python ml/scripts/create_splits.py
```

Stratified 70/15/15 split (seed=42). Preserves class ratio. Output in `ml/data/splits/deduped/`:
- `train_indices.npy`
- `val_indices.npy`
- `test_indices.npy`

### Step 5 — Validate Graph Dataset

```bash
PYTHONPATH=. poetry run python ml/scripts/validate_graph_dataset.py
```

Full integrity check: node feature shape `[N, 12]`, valid `edge_index`, label coverage against label CSV, paired hash alignment.

### Step 6 — Pre-flight GNN Gate (non-negotiable)

```bash
PYTHONPATH=. poetry run pytest ml/tests/test_cfg_embedding_separation.py -v
```

This test must pass before any training run. It verifies the three-phase GNN produces meaningfully different embeddings for call-before-write (reentrancy-vulnerable) vs write-before-call (CEI-safe) contracts on the `withdraw` function node. If it fails, fix the extractor or GNN before proceeding.

### CEI Augmentation (optional)

```bash
bash ml/scripts/run_augmentation.sh
```

Generates 50 synthetic CEI contract pairs (25 safe + 25 vulnerable) via `generate_cei_pairs.py` and `generate_safe_variants.py`, writes to `ml/data/augmented/`, then injects them into the training split via `inject_augmented.py`.

### Docker — Slither Environment

For environments without local Slither/solc:

```bash
docker build -f ml/docker/Dockerfile.slither -t sentinel-slither .
docker run -v $(pwd):/workspace sentinel-slither \
    python ml/src/data_extraction/ast_extractor.py
```

The image (`ubuntu:20.04`) pre-installs `slither-analyzer==0.10.0` and pre-compiled solc binaries for all versions from `0.4.2` to `0.8.20`.

---

## Dataset — DualPathDataset

`ml/src/datasets/dual_path_dataset.py`

Loads paired graph and token files for training and evaluation. The MD5 hash is the pairing key — `graphs/<md5>.pt` and `tokens/<md5>.pt` with the same stem belong to the same contract.

### Label Modes

| Mode | `label_csv` arg | Label source | Output shape |
|---|---|---|---|
| Binary | `None` | `graph.y` (scalar 0/1) | `[B]` long |
| Multi-label | `Path("ml/data/processed/multilabel_index_deduped.csv")` | 10-dim multi-hot float32 | `[B, 10]` float32 |

Multi-label class order: `[CallToUnknown, DenialOfService, ExternalBug, GasException, IntegerUO, MishandledException, Reentrancy, Timestamp, TransactionOrderDependence, UnusedReturn]`

### RAM Cache

Pass `cache_path=Path("ml/data/cached_dataset_deduped.pkl")` to `__init__` to use a pre-built pickle mapping each hash to its `(graph, token)` pair. Reduces per-epoch I/O from hours to minutes. Build the cache once with `ml/scripts/create_cache.py`.

### Collate Function

`dual_path_collate_fn` — PyG `Batch.from_data_list()` for graphs, stacks token tensors. Import alongside the dataset:

```python
from ml.src.datasets.dual_path_dataset import DualPathDataset, dual_path_collate_fn
```

---

## Model Architecture

### SentinelModel — Orchestration (v5)

`ml/src/models/sentinel_model.py`

Three independent 128-dim "eyes" are concatenated and classified jointly:

```
graphs [B]             input_ids [B, 512]
    │                      │
GNNEncoder             TransformerEncoder (CodeBERT + LoRA)
node_embs [N, 128]     token_embs [B, 512, 768]
    │         └─────────────┤
    │              CrossAttentionFusion
    │                  fused_eye [B, 128]
    │
    ├─ func-level pool → gnn_eye_proj    → gnn_eye [B, 128]
    │
    └─────────────── CLS token → transformer_eye_proj → transformer_eye [B, 128]

cat([gnn_eye, transformer_eye, fused_eye]) → [B, 384]
Linear(384, num_classes=10)               → logits [B, 10]
```

No Sigmoid inside the model — applied externally in `Predictor` and by `BCEWithLogitsLoss`.

**GNN eye** — function-level pool (FUNCTION/MODIFIER/FALLBACK/RECEIVE/CONSTRUCTOR nodes only). After Phase 3 reverse-CONTAINS aggregation these nodes carry CFG ordering signal. `global_max_pool` + `global_mean_pool` → cat → `Linear(256, 128) + ReLU + Dropout`.

**Transformer eye** — CLS token at position 0 (12-layer bidirectional summary) → `Linear(768, 128) + ReLU + Dropout`.

**Fused eye** — bidirectional cross-attention between node embeddings and all 512 token embeddings → `[B, 128]`.

**Ghost-graph fallback**: a contract with no function-level nodes (interface-only) would cause batch-size mismatch. For such graphs all nodes are included in the pool.

**Checkpoint format**: `{"model", "optimizer", "epoch", "best_f1", "config", "model_version"}`

### GNN Encoder (three-phase GAT)

`ml/src/models/gnn_encoder.py`

Four GAT layers arranged in three phases, with Jumping Knowledge connections (v5.2):

| Phase | Layers | Edge filter | `add_self_loops` | `heads` | Purpose |
|---|---|---|---|---|---|
| Phase 1 | 1 + 2 | types 0–5 (structural) | `True` | 8 (concat) | Propagate function properties; inter-function context |
| Phase 2 | 3 | type 6 (CONTROL_FLOW only) | **`False`** — critical | 1 (mean) | Enrich CFG nodes with execution-order signal |
| Phase 3 | 4 | type 7 (REVERSE_CONTAINS) | `False` | 1 (mean) | Aggregate CFG signal UP into function nodes |

**Phase 1-A1 — JK Connections** (`use_jk=True`, `jk_mode='attention'`): learned attention over all three phase outputs prevents Phase 1 structural signal from being over-smoothed by phases 2 and 3. Custom `_JKAttention(Linear(channels, 1))` — not PyG's LSTM-based JumpingKnowledge. JK tensors are consumed from `_live = []` **without** `.detach()` — do not change this.

**Phase 1-A2 — Per-Phase LayerNorm**: `ModuleList([LayerNorm(hidden_dim) for _ in range(3)])` applied after each phase's residual, before JK collection. Prevents Phase 1 magnitude (two layers + residual) from dominating JK attention softmax.

**Phase 1-A3 — REVERSE_CONTAINS edges**: type 7, runtime-only. Generated inside `GNNEncoder.forward()` by flipping CONTAINS(5) edges. Never written to graph `.pt` files. Gives Phase 3 a distinct learned direction embedding (fixes v5.0 limitation L2 where both directions shared the same CONTAINS(5) embedding).

**Default parameters**: `hidden_dim=128`, `heads=8`, `dropout=0.2`, `use_edge_attr=True`, `edge_emb_dim=32`, `num_layers=4`, `use_jk=True`, `jk_mode='attention'`. Total trainable parameters: ~91K.

`forward(return_intermediates=True)` returns `(x, batch, {"after_phase1", "after_phase2", "after_phase3"})` with detached tensors — used by `test_cfg_embedding_separation.py`.

### Transformer Encoder (CodeBERT + LoRA)

`ml/src/models/transformer_encoder.py`

| Component | Params | Status |
|---|---|---|
| CodeBERT backbone (`microsoft/codebert-base`) | 124,705,536 | **Frozen** |
| LoRA matrices (r=16, α=32, Q+V of all 12 layers) | ~295,296 | **Trainable** |

Returns all token embeddings `[B, 512, 768]` (not just CLS). CrossAttentionFusion needs all 512 tokens so each GNN node can query which tokens are relevant to it.

LoRA scale factor: `alpha/r = 32/16 = 2.0`. The `peft` library handles `requires_grad=False` on all backbone weights — do not wrap `self.bert()` in `torch.no_grad()` as it would also cut gradients to LoRA A/B matrices.

v5 default: `lora_r=16`, `lora_alpha=32`. (v4 used `r=8, alpha=16`.)

### CrossAttentionFusion

`ml/src/models/fusion_layer.py`

Bidirectional cross-attention enriches both modalities before pooling:

```
1. Project GNN nodes  [N, 128] → [N, 256]
2. Project tokens     [B, 512, 768] → [B, 512, 256]
3. Pad nodes          [N, 256] → [B, max_nodes, 256]   (to_dense_batch)
4. Node→Token cross-attention  (key_padding_mask = token PAD positions)
   → enriched_nodes  [B, max_nodes, 256]
   Zero-out padded node positions after attention
5. Token→Node cross-attention  (key_padding_mask = padded node positions)
   → enriched_tokens [B, 512, 256]
6. Masked mean pool enriched nodes  → [B, 256]
7. Masked mean pool enriched tokens → [B, 256]
8. cat → [B, 512] → Linear(512, 128) → fused_eye [B, 128]
```

`need_weights=False` on both `MultiheadAttention` calls — avoids materialising `[B, max_nodes, 512]` attention weight matrices (~12.6 MB per forward) and enables the fused CUDA efficient-attention kernel.

### Classifier and Auxiliary Heads

```python
# Main classifier
self.classifier = nn.Linear(3 * eye_dim, num_classes)   # 384 → 10

# Auxiliary heads — training only
self.aux_gnn         = nn.Linear(eye_dim, num_classes)  # 128 → 10
self.aux_transformer = nn.Linear(eye_dim, num_classes)
self.aux_fused       = nn.Linear(eye_dim, num_classes)
```

`forward(return_aux=True)` returns `(logits, {"gnn": ..., "transformer": ..., "fused": ...})`. Always `False` at inference — zero overhead.

Auxiliary loss (λ=0.3) keeps each eye's gradient alive even if the main classifier learns to weight one eye heavily. `aux_loss_weight` ramps linearly from 0 to λ over the first `aux_loss_warmup_epochs=3` epochs (Fix #33) to prevent aux heads from dominating early gradients.

### Node Feature Vector (v2 Schema)

`NODE_FEATURE_DIM = 12`, `FEATURE_SCHEMA_VERSION = "v3"`, `NUM_NODE_TYPES = 13`

| Index | Name | Description |
|---|---|---|
| 0 | `type_id` | `float(NODE_TYPES[kind]) / 12.0` — normalised to [0, 1] |
| 1 | `visibility` | `VISIBILITY_MAP` ordinal (0=private, 1=internal, 2=public/external) |
| 2 | `pure` | 1.0 if `Function.pure` |
| 3 | `view` | 1.0 if `Function.view` |
| 4 | `payable` | 1.0 if `Function.payable` |
| 5 | `complexity` | `float(len(func.nodes))` — CFG block count |
| 6 | `loc` | `float(len(source_mapping.lines))` |
| 7 | `return_ignored` | 0.0=captured / 1.0=discarded / -1.0=IR unavailable |
| 8 | `call_target_typed` | 0.0=raw address / 1.0=typed interface / -1.0=unavailable |
| 9 | `in_unchecked` | 1.0 if function contains an `unchecked{}` block |
| 10 | `has_loop` | 1.0 if function contains a loop |
| 11 | `external_call_count` | `log1p(n) / log1p(20)`, clamped [0, 1] |

**Node types** (id 0–12):

| Category | Types |
|---|---|
| Declaration-level (v1, stable) | `STATE_VAR=0`, `FUNCTION=1`, `MODIFIER=2`, `EVENT=3`, `FALLBACK=4`, `RECEIVE=5`, `CONSTRUCTOR=6`, `CONTRACT=7` |
| CFG subtypes (v2) | `CFG_NODE_CALL=8`, `CFG_NODE_WRITE=9`, `CFG_NODE_READ=10`, `CFG_NODE_CHECK=11`, `CFG_NODE_OTHER=12` |

`type_id` is normalised as `float(id) / 12.0` in the extractor. Recover with `(x[:, 0] * 12.0).round().long()`.

### Edge Types (v3 Schema)

`NUM_EDGE_TYPES = 8` (embedding table size in GNNEncoder)

| Id | Name | Written to disk | Description |
|---|---|---|---|
| 0 | `CALLS` | ✓ | function → internally-called function |
| 1 | `READS` | ✓ | function → state variable it reads |
| 2 | `WRITES` | ✓ | function → state variable it writes |
| 3 | `EMITS` | ✓ | function → event it emits |
| 4 | `INHERITS` | ✓ | contract → parent contract (linearised MRO) |
| 5 | `CONTAINS` | ✓ | function node → its CFG_NODE children |
| 6 | `CONTROL_FLOW` | ✓ | CFG_NODE → successor CFG_NODE (directed) |
| 7 | `REVERSE_CONTAINS` | **✗ runtime only** | CFG_NODE → parent function — generated in Phase 3 by flipping CONTAINS(5) |

---

## Output Classes

10-class multi-label output. Class index is stable across the codebase — defined as `CLASS_NAMES` in `trainer.py`.

| Index | Class |
|---|---|
| 0 | `CallToUnknown` |
| 1 | `DenialOfService` |
| 2 | `ExternalBug` |
| 3 | `GasException` |
| 4 | `IntegerUO` |
| 5 | `MishandledException` |
| 6 | `Reentrancy` |
| 7 | `Timestamp` |
| 8 | `TransactionOrderDependence` |
| 9 | `UnusedReturn` |

---

## Training

### Run Training

```bash
# Smoke run — 2 epochs, 10% data (run first to clear all Phase 4 gates)
TRANSFORMERS_OFFLINE=1 PYTHONPATH=. python ml/scripts/train.py \
    --run-name v5.2-smoke \
    --experiment-name sentinel-v5.2 \
    --epochs 2 \
    --smoke-subsample-fraction 0.1 \
    --gradient-accumulation-steps 4

# Full 60-epoch run (after smoke gates pass)
TRANSFORMERS_OFFLINE=1 PYTHONPATH=. python ml/scripts/train.py \
    --run-name v5.2-full \
    --experiment-name sentinel-v5.2 \
    --epochs 60 \
    --gradient-accumulation-steps 4

# Ablation — disable JK connections
TRANSFORMERS_OFFLINE=1 PYTHONPATH=. python ml/scripts/train.py \
    --run-name v5.2-no-jk \
    --no-jk \
    --epochs 60 \
    --gradient-accumulation-steps 4

# Resume from checkpoint (example — continue r3)
TRANSFORMERS_OFFLINE=1 PYTHONPATH=. python ml/scripts/train.py \
    --resume ml/checkpoints/v5.2-jk-20260515c-r3_best.pt \
    --no-resume-model-only \
    --run-name v5.2-jk-20260515c-r4 \
    --experiment-name sentinel-v5.2 \
    --epochs 60 \
    --gradient-accumulation-steps 4 \
    --early-stop-patience 20
```

**Phase 4 gates (smoke run)**: GNN gradient share ≥ 15% at step 100; JK all-phase attention weights > 5%; no NaN loss after step 50.

### TrainConfig Reference

All fields are in `ml/src/training/trainer.py::TrainConfig`. Key defaults:

| Field | Default | Notes |
|---|---|---|
| `num_classes` | `10` | Track 3 multi-label |
| `fusion_output_dim` | `128` | Width of each eye output |
| `gnn_hidden_dim` | `128` | GNN node embedding width |
| `gnn_layers` | `4` | Number of GAT layers (validated in `__post_init__`) |
| `gnn_heads` | `8` | Phase 1 attention heads |
| `gnn_dropout` | `0.2` | |
| `use_edge_attr` | `True` | Feed edge-type embeddings into GATConv |
| `gnn_edge_emb_dim` | `32` | `nn.Embedding(8, 32)` |
| `gnn_use_jk` | `True` | Phase 1-A1 JK connections |
| `gnn_jk_mode` | `'attention'` | Learned per-phase attention aggregation |
| `lora_r` | `16` | LoRA rank (v5; was 8 in v4) |
| `lora_alpha` | `32` | Effective scale = alpha/r = 2.0 |
| `lora_dropout` | `0.1` | |
| `lora_target_modules` | `["query", "value"]` | All 12 CodeBERT layers |
| `epochs` | `60` | |
| `batch_size` | `16` | Fix #28 — reduced for 8 GB GPU |
| `lr` | `2e-4` | Base LR for AdamW |
| `weight_decay` | `1e-2` | |
| `gnn_lr_multiplier` | `2.5` | GNN effective LR = lr × 2.5 |
| `lora_lr_multiplier` | `0.5` | LoRA effective LR = lr × 0.5 |
| `gradient_accumulation_steps` | `1` | Use 4 on 8 GB GPU (effective batch = 64) |
| `loss_fn` | `"bce"` | `"bce"` or `"focal"` |
| `focal_gamma` | `2.0` | |
| `focal_alpha` | `0.25` | |
| `aux_loss_weight` | `0.3` | λ for per-eye auxiliary loss |
| `aux_loss_warmup_epochs` | `3` | Ramp aux weight from 0 → λ linearly |
| `early_stop_patience` | `10` | Epochs without val F1 improvement |
| `grad_clip` | `1.0` | Max gradient norm |
| `warmup_pct` | `0.10` | OneCycleLR warmup fraction |
| `use_amp` | `True` | AMP (BF16) + TF32 matmuls |
| `num_workers` | `2` | DataLoader workers |
| `cache_path` | `"ml/data/cached_dataset_deduped.pkl"` | Pre-built RAM cache |
| `checkpoint_dir` | `"ml/checkpoints"` | |

**Optimizer**: `AdamW` with per-parameter-group LRs (GNN 2.5×, LoRA 0.5×, default for the rest).  
**Scheduler**: `OneCycleLR` with full epoch count — always created with `epochs=config.epochs` so checkpoint `state_dict` resumes correctly (Fix #32).  
**Pos-weight**: computed from training split only, sqrt-scaled (`raw_ratio ** 0.5`), applied to BCEWithLogitsLoss.

### Loss Functions

**BCEWithLogitsLoss** (default `loss_fn="bce"`): class-balanced `pos_weight` applied.

**FocalLoss** (`loss_fn="focal"`): `FocalLoss(gamma=2.0, alpha=0.25)`. Expects post-sigmoid probabilities — `_FocalFromLogits` wrapper in `trainer.py` applies sigmoid before calling `FocalLoss.forward()`. Explicit `.float()` cast guards against BF16 underflow (Fix #6).

**Auxiliary loss**: `total_loss = main_loss + λ_eff × (loss_gnn + loss_transformer + loss_fused)` where `λ_eff` ramps from 0 to `aux_loss_weight=0.3` over `aux_loss_warmup_epochs=3` epochs.

### Resume from Checkpoint

```bash
python ml/scripts/train.py --resume ml/checkpoints/<name>_best.pt
```

Architecture mismatch detection: if the checkpoint's `model_version` pre-dates v5.2, a warning is logged. `_parse_version()` handles tuple comparison.

### Per-Class Threshold Tuning

```bash
PYTHONPATH=. python ml/scripts/tune_threshold.py \
    --checkpoint ml/checkpoints/v5.2-jk-20260515c-r3_best.pt
```

Runs one forward pass over the val set, sweeps thresholds per class (default 0.05–0.95 in steps of 0.01), picks the threshold that maximises per-class F1, saves to `<checkpoint_stem>_thresholds.json`. The `Predictor` loads this file automatically.

### Hyperparameter Search

```bash
# Single-run wrapper (smoke or confirm regime)
PYTHONPATH=. python ml/scripts/auto_experiment.py \
    --regime smoke \
    --run-name auto-001 \
    --experiment-name sentinel-v5.2 \
    --loss-fn focal --gamma 2.0 --alpha 0.25

# Overnight sequential launcher (4 experiments)
nohup python ml/scripts/run_overnight_experiments.py \
    > ml/logs/overnight.log 2>&1 &
```

`auto_experiment.py` emits machine-readable score lines (`SENTINEL_SCORE=...`) and exits with code 0/1/2/3.

### Model Promotion

```bash
# Promote to Staging
PYTHONPATH=. python ml/scripts/promote_model.py \
    --checkpoint ml/checkpoints/v5.2-jk-20260515c-r3_best.pt \
    --stage Staging \
    --val-f1-macro 0.52 \
    --note "v5.2 full run; JK active"

# Promote to Production
PYTHONPATH=. python ml/scripts/promote_model.py \
    --checkpoint ml/checkpoints/v5.2-jk-20260515c-r3_best.pt \
    --stage Production \
    --val-f1-macro 0.52

# Dry run
python ml/scripts/promote_model.py --dry-run ...
```

Logs checkpoint as MLflow artifact, registers as `sentinel-vulnerability-detector` in the Model Registry, transitions to `Staging` or `Production`. Valid stages: `{"Staging", "Production"}`. Exit codes: 0 (success), 1 (not found / MLflow error).

---

## Inference API

`ml/src/inference/api.py` — FastAPI application. Predictor loads once at startup via the lifespan context manager.

```bash
SENTINEL_CHECKPOINT=ml/checkpoints/multilabel-v4-finetune-lr1e4_best.pt \
TRANSFORMERS_OFFLINE=1 \
uvicorn ml.src.inference.api:app --host 0.0.0.0 --port 8001
```

Environment variables:

| Variable | Default | Description |
|---|---|---|
| `SENTINEL_CHECKPOINT` | `ml/checkpoints/multilabel-v4-finetune-lr1e4_best.pt` | Checkpoint path |
| `SENTINEL_PREDICT_TIMEOUT` | `60` | Inference timeout in seconds |
| `SENTINEL_DRIFT_BASELINE` | `ml/data/drift_baseline.json` | Drift baseline path |
| `SENTINEL_DRIFT_CHECK_INTERVAL` | `50` | Run KS test every N requests |

### Endpoints

**`POST /predict`**

```json
// Request
{
  "source_code": "pragma solidity ^0.8.0; contract Foo { ... }"
}

// Response
{
  "vulnerabilities": [
    {
      "vulnerability_class": "Reentrancy",
      "probability": 0.87,
      "detected": true
    },
    ...
  ],
  "thresholds": [0.45, 0.50, ...],
  "num_nodes": 14,
  "num_edges": 22,
  "architecture": "three_eye_v5"
}
```

Source size enforced before preprocessing: `MAX_SOURCE_BYTES = 1 MB`.

**`GET /health`**

```json
{
  "status": "ok",
  "model_loaded": true,
  "architecture": "three_eye_v5",
  "thresholds_loaded": true
}
```

Prometheus metrics exposed at `/metrics`: `sentinel_model_loaded`, `sentinel_gpu_memory_bytes`, `sentinel_drift_alerts_total{stat=...}`.

### HTTP Status Codes

| Code | Trigger |
|---|---|
| 200 | Successful prediction |
| 400 | Missing `pragma solidity` / empty input |
| 413 | Source exceeds 1 MB or CUDA OOM |
| 500 | Preprocessing / runtime error |
| 503 | Predictor not loaded |
| 504 | Inference timeout |

---

## Predictor

`ml/src/inference/predictor.py`

Handles checkpoint loading, per-class threshold loading, and inference. Loaded once at API startup.

- Architecture is read from `checkpoint["config"]["architecture"]` — `_ARCH_TO_FUSION_DIM` allowlist; unknown architecture raises `ValueError` (Bug 4 fix).
- `fusion_output_dim`, `gnn_dropout`, `lora_target_modules` are read from checkpoint config (Fix #2).
- Per-class thresholds loaded from `<checkpoint_stem>_thresholds.json`. Falls back to user-supplied threshold (default 0.5) with per-class warning if file is missing.
- Warmup forward pass at startup uses a real 2-node 1-edge graph so GATConv message-passing is exercised (Fix #5 / Audit #5).
- `self.thresholds_loaded` exposed for `/health`.
- `_score()` emits `"vulnerability_class"` key (canonical schema — Bug 3 fix).

---

## ContractPreprocessor

`ml/src/inference/preprocess.py`

Converts one Solidity contract into `(graph, tokens)` for the model.

```python
from ml.src.inference.preprocess import ContractPreprocessor

preprocessor = ContractPreprocessor()

# From file path
graph, tokens = preprocessor.process(Path("contract.sol"))

# From raw source string (writes NamedTemporaryFile — solc requires a real path)
graph, tokens = preprocessor.process_source(source_code: str)

# Sliding-window for long contracts
windows: list[dict] = preprocessor.process_source_windowed(source_code: str)
```

Shape contract (must match training data — do not change without retraining):

| Tensor | Shape | dtype |
|---|---|---|
| `graph.x` | `[N, 12]` | float32 |
| `graph.edge_index` | `[2, E]` | int64 |
| `tokens["input_ids"]` | `[1, 512]` | long |
| `tokens["attention_mask"]` | `[1, 512]` | long |

`MAX_SOURCE_BYTES = 1 MB` — checked before Slither is invoked.

---

## Inference Cache

`ml/src/inference/cache.py`

Disk-backed content-addressed cache for `(graph, tokens)` pairs. Slither + tokenization takes 3–5 s; cached hits return in < 50 ms.

```python
from ml.src.inference.cache import InferenceCache

cache = InferenceCache(
    cache_dir=Path("ml/data/inference_cache"),
    ttl_seconds=86400,   # 24h default
)
```

Key format: `"{content_md5}_{FEATURE_SCHEMA_VERSION}"`. Bumping `FEATURE_SCHEMA_VERSION` in `graph_schema.py` automatically invalidates all cached entries. Atomic writes via tmp-file + rename — concurrent writers are safe (last-write-wins, both identical).

---

## Drift Detection

`ml/src/inference/drift_detector.py`

Kolmogorov–Smirnov test compares a rolling window of request stats against a pre-computed baseline. Fires a Prometheus counter (`sentinel_drift_alerts_total{stat=...}`) when `p < 0.05`.

```python
detector = DriftDetector(baseline_path="ml/data/drift_baseline.json")

# Per request
detector.update_stats({"num_nodes": 14, "num_edges": 22})

# Every SENTINEL_DRIFT_CHECK_INTERVAL requests (default 50)
detector.check()
```

**Baseline source**: do not use `ml/data/graphs/` (BCCC-2024 historical snapshot — will fire on every modern 2026 contract). Build from warm-up request data after ≥ 500 real requests:

```bash
python ml/scripts/compute_drift_baseline.py \
    --source warmup \
    --warmup-log ml/data/warmup_stats.jsonl \
    --output ml/data/drift_baseline.json
```

Minimum samples required before KS runs: `MIN_SAMPLES_FOR_KS = 30`.

---

## MLflow and Model Registry

Tracking URI defaults to project-local SQLite: `sqlite:///mlruns.db`. Override via `MLFLOW_TRACKING_URI`.

Each training run logs: all `TrainConfig` fields, per-epoch val metrics (macro F1, per-class F1, hamming loss), gradient norm, VRAM usage, JK attention weights per phase. Best checkpoint is logged as an artifact outside the epoch loop (optimisation #6).

Model Registry name: `sentinel-vulnerability-detector`. Use `promote_model.py` for all stage transitions — manual `.pt` copying leaves no audit trail.

---

## DVC

Data files too large for Git are tracked with DVC:

| DVC pointer | Tracks |
|---|---|
| `ml/data/graphs.dvc` | `ml/data/graphs/` (44,472 `.pt` graph files) |
| `ml/data/tokens.dvc` | `ml/data/tokens/` (44,472 `.pt` token files) |
| `ml/data/splits.dvc` | `ml/data/splits/` (train/val/test `.npy` index arrays) |
| `ml/checkpoints.dvc` | `ml/checkpoints/` (trained model checkpoints) |

Remote configured in `.dvc/config`. Pull data: `dvc pull`.

---

## Docker — Slither Environment

`ml/docker/Dockerfile.slither` — Ubuntu 20.04, `slither-analyzer==0.10.0`. Pre-installs solc binaries for versions `0.4.2` through `0.8.20` at build time (avoids GitHub API rate limits at runtime).

```bash
docker build -f ml/docker/Dockerfile.slither -t sentinel-slither .
docker run -v $(pwd):/workspace sentinel-slither \
    python ml/src/data_extraction/ast_extractor.py
```

---

## Testing

12 test modules, ~3,100 lines. All tests live in `ml/tests/`.

```bash
# Full suite
TRANSFORMERS_OFFLINE=1 PYTHONPATH=. pytest ml/tests/ -v

# Non-integration only (no Slither required)
TRANSFORMERS_OFFLINE=1 PYTHONPATH=. pytest ml/tests/ -v -m "not integration"

# Pre-flight GNN gate (non-negotiable before training)
TRANSFORMERS_OFFLINE=1 PYTHONPATH=. pytest ml/tests/test_cfg_embedding_separation.py -v
```

| Test file | Lines | What is tested |
|---|---|---|
| `test_model.py` | 379 | `SentinelModel` forward pass, `return_aux`, three-eye shapes, auxiliary heads |
| `test_gnn_encoder.py` | 304 | Per-phase output shapes, JK gradient flow, `return_intermediates`, edge-type filtering |
| `test_preprocessing.py` | 891 | Schema sanity (`NODE_FEATURE_DIM=12`, 13 node types), `_build_node_features()`, `_build_control_flow_edges()`, integration tests with Slither |
| `test_cfg_embedding_separation.py` | 282 | **Pre-flight gate**: call-before-write vs write-before-call produce different function embeddings |
| `test_fusion_layer.py` | 162 | `CrossAttentionFusion` output shapes, mask handling, gradient flow |
| `test_trainer.py` | 275 | `TrainConfig` validation, `compute_pos_weight()`, aux loss warmup, scheduler resume |
| `test_dataset.py` | 226 | `DualPathDataset` label modes, paired hash discovery, collate function |
| `test_api.py` | 210 | `/predict` and `/health` endpoints, error codes (400/413/500/503/504) |
| `test_preprocessing_api.py` → `test_api.py` | — | FastAPI integration via `TestClient(app)`, session-scoped fixture |
| `test_predictor.py` → `test_model.py` | — | Architecture detection, threshold loading |
| `test_drift_detector.py` | 123 | KS alert firing, `MIN_SAMPLES_FOR_KS` guard, Prometheus counter |
| `test_cache.py` | 95 | Cache key format, TTL expiry, atomic write, schema-version invalidation |
| `test_promote_model.py` | 134 | Stage validation, dry-run mode, MLflow registration |

`conftest.py` sets `TRANSFORMERS_OFFLINE=1` before any import and provides a session-scoped `TestClient` fixture (model loads once for all API tests).

Slither integration tests are marked `@pytest.mark.integration` and require `slither-analyzer>=0.9.3` and `solc` to be installed.

---

## Scripts Reference

All commands from project root. Run with `PYTHONPATH=. poetry run python ml/scripts/<script>.py`.

### Data Pipeline

| Script | Purpose |
|---|---|
| `build_multilabel_index.py` | BCCC CSV → `multilabel_index.csv` (SHA256↔MD5 bridge, max-OR labels) |
| `dedup_multilabel_index.py` | Deduplicate index → `multilabel_index_deduped.csv` |
| `create_label_index.py` | Scan graph `.pt` files → `label_index.csv` (binary mode) |
| `create_splits.py` | Stratified 70/15/15 split → `splits/deduped/*.npy` |
| `verify_splits.py` | Assert split integrity and class ratio preservation |
| `validate_graph_dataset.py` | Full integrity check: shape, edge validity, label coverage |
| `reextract_graphs.py` | Re-run graph extraction with current v2 schema on existing dataset |

### Augmentation

| Script | Purpose |
|---|---|
| `generate_cei_pairs.py` | Generate synthetic CEI-vulnerable Solidity pairs |
| `generate_safe_variants.py` | Generate safe (CEI-pattern) variants |
| `extract_augmented.py` | Extract graphs/tokens from `ml/data/augmented/*.sol` |
| `inject_augmented.py` | Inject augmented samples into the training split |
| `run_augmentation.sh` | Shell wrapper: generate → extract → inject in one step |

### Training

| Script | Purpose |
|---|---|
| `train.py` | Main training entry point (see [Run Training](#run-training)) |
| `auto_experiment.py` | Single-run wrapper: train → tune → emit machine-readable score |
| `run_overnight_experiments.py` | Sequential launcher for 4 hyperparameter experiments |

### Post-Training

| Script | Purpose |
|---|---|
| `tune_threshold.py` | Per-class threshold sweep on val set → `<name>_thresholds.json` |
| `promote_model.py` | Log checkpoint to MLflow, register, transition stage |
| `manual_test.py` | Interactive CLI for single-contract inference smoke test |

### Analysis and Utilities

| Script | Purpose |
|---|---|
| `analyse_truncation.py` | Token length distribution; count truncated contracts |
| `compute_drift_baseline.py` | Build `drift_baseline.json` from warm-up request logs |
| `compute_locked_hashes.py` | SHA256 manifest of locked architecture files |
| `create_cache.py` | Pre-build `cached_dataset_deduped.pkl` for RAM cache |

---

## File Reference

```
ml/
├── pyproject.toml              — Python 3.12.1, all dependencies
├── poetry.lock
├── __init__.py
├── DIAGRAMS.md                 — Mermaid diagrams (system lifecycle, architecture, data flow)
├── locked_files.sha256         — SHA256 manifest for architecture lock
├── checkpoints.dvc             — DVC pointer to ml/checkpoints/
│
├── docker/
│   └── Dockerfile.slither      — Ubuntu 20.04 + slither==0.10.0 + solc binaries
│
├── data/
│   ├── graphs.dvc              — 44,472 .pt graph files (tracked by DVC)
│   ├── tokens.dvc              — 44,472 .pt token files (tracked by DVC)
│   ├── splits.dvc              — train/val/test .npy arrays (tracked by DVC)
│   ├── augmented/              — 50 synthetic CEI contract pairs (.sol)
│   ├── tokens_orphaned/        — 24,148 legacy token files (no graph counterpart; not used in training)
│   └── processed/
│       └── multilabel_index_deduped.csv  — label source for DualPathDataset
│
├── src/
│   ├── preprocessing/
│   │   ├── graph_schema.py     — NODE_TYPES, EDGE_TYPES, FEATURE_NAMES (single source of truth)
│   │   └── graph_extractor.py  — extract_contract_graph(), GraphExtractionConfig
│   ├── data_extraction/
│   │   ├── ast_extractor.py    — offline batch orchestration (11 workers, checkpoint/resume)
│   │   └── tokenizer.py        — CodeBERT tokenization pipeline
│   ├── datasets/
│   │   └── dual_path_dataset.py  — DualPathDataset, dual_path_collate_fn
│   ├── models/
│   │   ├── sentinel_model.py   — SentinelModel v5.2 (three-eye orchestration)
│   │   ├── gnn_encoder.py      — GNNEncoder (three-phase GAT + JK + LayerNorm)
│   │   ├── transformer_encoder.py  — CodeBERT + LoRA
│   │   └── fusion_layer.py     — CrossAttentionFusion (bidirectional cross-attention)
│   ├── training/
│   │   ├── trainer.py          — TrainConfig, train(), CLASS_NAMES, MODEL_VERSION
│   │   └── focalloss.py        — FocalLoss (post-sigmoid, BF16 guard)
│   ├── inference/
│   │   ├── api.py              — FastAPI app (/predict, /health, Prometheus)
│   │   ├── predictor.py        — Predictor (checkpoint load, threshold load, warmup)
│   │   ├── preprocess.py       — ContractPreprocessor (graph + tokens for one contract)
│   │   ├── cache.py            — InferenceCache (disk-backed, content-addressed, TTL)
│   │   └── drift_detector.py   — DriftDetector (KS test, rolling window, Prometheus)
│   └── utils/
│       └── hash_utils.py       — MD5/SHA256 hashing utilities
│
├── scripts/
│   ├── train.py                — Training entry point (argparse → TrainConfig → train())
│   ├── tune_threshold.py       — Per-class threshold optimiser
│   ├── promote_model.py        — MLflow model registry promotion
│   ├── auto_experiment.py      — Single-run hyperparameter search wrapper
│   ├── run_overnight_experiments.py  — Sequential 4-experiment launcher
│   ├── build_multilabel_index.py
│   ├── dedup_multilabel_index.py
│   ├── create_splits.py
│   ├── verify_splits.py
│   ├── validate_graph_dataset.py
│   ├── reextract_graphs.py
│   ├── generate_cei_pairs.py
│   ├── generate_safe_variants.py
│   ├── extract_augmented.py
│   ├── inject_augmented.py
│   ├── run_augmentation.sh
│   ├── analyse_truncation.py
│   ├── compute_drift_baseline.py
│   ├── compute_locked_hashes.py
│   ├── create_cache.py
│   ├── manual_test.py
│   └── test_contracts/         — 20 hand-crafted .sol files for manual smoke tests
│
└── tests/
    ├── conftest.py             — TRANSFORMERS_OFFLINE, session-scoped TestClient
    ├── test_model.py           — SentinelModel + GNNEncoder unit tests
    ├── test_gnn_encoder.py     — Per-phase shapes, JK, intermediates
    ├── test_preprocessing.py   — Schema sanity + integration (requires Slither)
    ├── test_cfg_embedding_separation.py  — Pre-flight GNN gate (non-negotiable)
    ├── test_fusion_layer.py
    ├── test_trainer.py
    ├── test_dataset.py
    ├── test_api.py
    ├── test_drift_detector.py
    ├── test_cache.py
    └── test_promote_model.py
```

---

## Critical Constraints

These are structural invariants enforced by the codebase. Violating any causes silent accuracy regression or runtime errors.

**`FEATURE_SCHEMA_VERSION`** — must be bumped whenever `NODE_TYPES`, `EDGE_TYPES`, `FEATURE_NAMES`, or `_build_node_features()` logic changes. Invalidates all inference caches and requires a full dataset rebuild + retrain.

**`NODE_FEATURE_DIM = 12`** — `GNNEncoder.conv1.in_channels` is set to this constant at construction. Any graph `.pt` file with a different `x.shape[1]` will crash at the first forward pass.

**`NUM_EDGE_TYPES = 8`** — `GNNEncoder` embeds all 8 edge type IDs via `nn.Embedding(8, gnn_edge_emb_dim)`. `REVERSE_CONTAINS=7` is never written to disk; it is generated at runtime.

**`type_id` normalisation** — stored as `float(type_id) / 12.0` in graph `.pt` files. Recover with `(x[:, 0] * _MAX_TYPE_ID).round().long()`. Raw IDs 0–12 must never appear unnormalised in `graph.x`.

**`peft` library** — hard requirement. Missing `peft` raises `RuntimeError` at `TransformerEncoder` import time, not at model construction.

**`slither-analyzer >= 0.9.3`** — enforced at `graph_schema.py` import. Older Slither silently produces wrong `in_unchecked` features (`NodeType.STARTUNCHECKED` only available in ≥ 0.9.3).

**`TRANSFORMERS_OFFLINE=1`** — must be set before any Python import, not inside Python. HuggingFace reads it at import time.

**`add_self_loops=False` in Phase 2** — self-loops cancel directional signal in CONTROL_FLOW edges. Do not add them.

**No `torch.no_grad()` around `self.bert()`** — would cut gradients to LoRA A/B matrices. The `requires_grad=False` split is handled internally by `peft`.

**JK tensors collected without `.detach()`** — `_live = []` in `GNNEncoder.forward()` must not use `.detach()`. The diagnostic `_intermediates` dict uses `.detach().clone()` only for test inspection.

**`process_source()` requires a temp file** — Slither shells out to `solc`, which requires a real file path. Never pass raw source code directly to Slither.

**`graph_builder.py` must not be used in `preprocess.py`** — its one-hot encoding produces 17-dim features; the model expects 12-dim. Using it causes a mat-mul shape error.