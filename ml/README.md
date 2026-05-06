# M1 — ML Core

Dual-path smart contract vulnerability detector. A **Graph Attention Network (GNN)** encodes the contract's AST structure with typed edge relations; a **LoRA-fine-tuned CodeBERT** encodes its source text. A **bidirectional CrossAttentionFusion** merges both representations before a 10-class multi-label sigmoid classifier produces per-vulnerability probabilities.

> Visual Mermaid diagrams (system lifecycle, model architecture, dataset loading flow) → [`ml/DIAGRAMS.md`](DIAGRAMS.md)

---

## Table of Contents

- [Setup](#setup)
- [System Overview](#system-overview)
- [Shared Preprocessing Layer](#shared-preprocessing-layer)
  - [What Changes Require a Schema Rebuild](#what-changes-require-a-schema-rebuild)
- [Data Preparation](#data-preparation)
  - [Step 1 — Graph Extraction](#step-1--graph-extraction)
  - [Step 2 — Tokenisation](#step-2--tokenisation)
  - [Step 3 — Build Multi-Label Index](#step-3--build-multi-label-index)
  - [Step 4 — Create Stratified Splits](#step-4--create-stratified-splits)
  - [Step 5 — Validate Graph Dataset](#step-5--validate-graph-dataset)
  - [Docker — Slither Environment](#docker--slither-environment)
- [Dataset — DualPathDataset](#dataset--dualpathddataset)
  - [Label Modes](#label-modes)
  - [Paired Hash Discovery](#paired-hash-discovery)
  - [RAM Cache](#ram-cache)
  - [Collate Function](#collate-function)
- [Model Architecture](#model-architecture)
  - [GNN Encoder](#gnn-encoder)
  - [Transformer Encoder (CodeBERT + LoRA)](#transformer-encoder-codebert--lora)
  - [CrossAttention Fusion](#crossattention-fusion)
  - [Classifier](#classifier)
  - [Long-Contract Path — Sliding Window](#long-contract-path--sliding-window)
  - [Node Feature Vector](#node-feature-vector)
  - [Edge Types](#edge-types)
- [Output Classes](#output-classes)
- [Dataset](#dataset)
- [Training](#training)
  - [Run Training](#run-training)
  - [FocalLoss](#focalloss)
  - [Resume from Checkpoint](#resume-from-checkpoint)
  - [Per-Class Threshold Tuning](#per-class-threshold-tuning)
  - [Hyperparameter Search — Overnight Experiments](#hyperparameter-search--overnight-experiments)
  - [Recommended v4 Configuration](#recommended-v4-configuration)
- [Active Checkpoint](#active-checkpoint)
  - [Per-Class Thresholds and F1 (v3)](#per-class-thresholds-and-f1-v3)
  - [Retrain Evaluation Protocol](#retrain-evaluation-protocol)
- [Inference API](#inference-api)
  - [Endpoints](#endpoints)
  - [HTTP Status Codes](#http-status-codes)
- [Inference Cache](#inference-cache)
- [Drift Detection](#drift-detection)
- [MLflow and Model Registry](#mlflow-and-model-registry)
- [DVC](#dvc)
- [Testing](#testing)
- [File Reference](#file-reference)
- [Critical Constraints](#critical-constraints)
- [Known Limitations](#known-limitations)

---

## Setup

```bash
# Python 3.12.1 required (strict — pyproject.toml pin)
cd ml
poetry install

# TRANSFORMERS_OFFLINE must be set at shell level before any Python import
# HuggingFace checks this at import time — setting it inside Python is too late
export TRANSFORMERS_OFFLINE=1

# GPU: tested on RTX 3070 8 GB with AMP (BF16). CPU inference is supported
# but Slither graph extraction still requires solc installed locally or via Docker.
```

Key runtime dependencies: `torch ^2.5.0`, `peft >=0.13.0`, `torch-geometric ^2.6.0`,
`transformers ^4.45.0`, `fastapi ^0.115.0`, `scipy ^1.13.0`, `mlflow ^2.17.0`.

---

## System Overview

```
BCCC-SCsVul-2024  (.sol files + labels)
        │
        │   ┌──────────────────────────────────────────────────────────────┐
        │   │  SHARED PREPROCESSING LAYER (both pipelines import from here)│
        │   │                                                              │
        │   │  graph_schema.py  ──imports──►  graph_extractor.py           │
        │   │  NODE_TYPES, EDGE_TYPES,         extract_contract_graph()    │
        │   │  FEATURE_SCHEMA_VERSION,         GraphExtractionConfig       │
        │   │  NODE_FEATURE_DIM=8              typed exceptions            │
        │   │                                                              │
        │   └──────────────┬──────────────────────────┬────────────────────┘
        │                  │ used by (offline)         │ used by (online)
        │                  ▼                           ▼
        ├──► ast_extractor.py                  preprocess.py
        │    11 workers · solc version-pinned  ContractPreprocessor
        │    → graphs/  ~68K .pt               → graph + tokens per request
        │
        ├──► tokenizer.py
        │    CodeBERT tokenizer
        │    → tokens/  ~68K .pt
        │
        ├──► build_multilabel_index.py
        │    → multilabel_index.csv  (68,523 rows × 10 classes)
        │
        └──► create_splits.py
             → train/val/test_indices.npy  (47,966 / 10,278 / 10,279)
                          │
                          ▼
        ┌─────────────────────────────────────────────────────┐
        │  TRAINING   scripts/train.py + MLflow               │
        │                                                     │
        │  DualPathDataset                                    │
        │    graphs/ + tokens/ paired by MD5 stem             │
        │    binary mode:     label ← graph.y                 │
        │    multi-label mode: label ← multilabel_index.csv   │
        │             │                                       │
        │             ▼                                       │
        │  SentinelModel.forward(graphs, input_ids, attn_mask)│
        │    GNNEncoder      → node_embs [N, 64]              │
        │    TransformerEncoder → token_embs [B, 512, 768]    │
        │    CrossAttentionFusion → fused [B, 128]            │
        │    Classifier       → logits [B, 10]                │
        │             │                                       │
        │  BCEWithLogitsLoss  or  FocalLoss(gamma=2, α=0.25)  │
        │                                                     │
        │  → checkpoints/  best.pt + _thresholds.json         │
        └─────────────────────────────────────────────────────┘
                          │
             ┌────────────┴─────────────┐
             ▼                          ▼
     tune_threshold.py         promote_model.py
     0.05–0.95 per class       MLflow Staging → Production
     → *_thresholds.json
                          │
                          ▼
        ┌─────────────────────────────────────────────────────┐
        │  INFERENCE API   src/inference/api.py   port 8001   │
        │                                                     │
        │  POST /predict  { "source_code": "..." }            │
        │    │                                                │
        │    ├── InferenceCache  (md5 + FEATURE_SCHEMA_VERSION│
        │    │     hit → return cached (graph, tokens)        │
        │    │     miss → run ContractPreprocessor → cache    │
        │    │                                                │
        │    ├── ContractPreprocessor (preprocess.py)         │
        │    │     extract_contract_graph() ← graph_extractor │
        │    │     CodeBERT tokenizer                         │
        │    │                                                │
        │    ├── SentinelModel.forward() → logits [B, 10]     │
        │    │     sliding window if > 512 tokens             │
        │    │                                                │
        │    ├── sigmoid → probs, apply per-class thresholds  │
        │    └── DriftDetector (KS test every 50 requests)    │
        │                                                     │
        │  GET /health    GET /metrics (Prometheus)           │
        └─────────────────────────────────────────────────────┘
```

---

## Shared Preprocessing Layer

**Files:** `ml/src/preprocessing/graph_schema.py` and `ml/src/preprocessing/graph_extractor.py`

These two files are the most architecturally critical layer in the module. Before they existed, `ast_extractor.py` (offline) and `preprocess.py` (online inference) each contained identical, hand-duplicated node/edge feature logic. A missed sync caused silent inference accuracy regression — model receives features encoded differently from training, with no error message.

```
graph_schema.py  ── single source of truth for all schema constants
│
│   NODE_TYPES       = { CONTRACT:7, STATE_VAR:0, FUNCTION:1,
│                         MODIFIER:2, EVENT:3, FALLBACK:4,
│                         RECEIVE:5,  CONSTRUCTOR:6 }
│   EDGE_TYPES       = { CALLS:0, READS:1, WRITES:2, EMITS:3, INHERITS:4 }
│   VISIBILITY_MAP   = { public:0, external:0, internal:1, private:2 }
│   FEATURE_NAMES    = (type_id, visibility, pure, view, payable,
│                        reentrant, complexity, loc)
│   NODE_FEATURE_DIM = 8       ← GNNEncoder in_channels (locked in checkpoint)
│   NUM_EDGE_TYPES   = 5       ← GNNEncoder Embedding table width
│   FEATURE_SCHEMA_VERSION = "v1"  ← inference cache key suffix
│   assert len(FEATURE_NAMES) == NODE_FEATURE_DIM  ← compile-time guard
│
└── imported by ──► graph_extractor.py
                          │
                          ▼
graph_extractor.py  ── canonical Solidity → PyG graph implementation
│
│   GraphExtractionConfig  (dataclass)
│     multi_contract_policy   "first" | "by_name"
│     target_contract_name    used when policy="by_name"
│     include_edge_attr        attach edge_attr [E] to Data (default True)
│     solc_binary              override solc path (offline: version-pinned)
│     solc_version             for --allow-paths compat check (≥ 0.5.0)
│     allow_paths              --allow-paths for local import resolution
│
│   Exception hierarchy (typed for HTTP translation):
│     GraphExtractionError         base
│       SolcCompilationError       bad Solidity       → HTTP 400 (user error)
│       SlitherParseError          Slither/infra fail  → HTTP 500
│       EmptyGraphError            zero AST nodes      → HTTP 400
│
│   extract_contract_graph(sol_path, config) → Data
│     Never returns None. Always raises on failure.
│     Returns: x[N,8] · edge_index[2,E] · edge_attr[E] · contract_name
│     Does NOT set .contract_hash / .contract_path / .y — callers attach.
│
└── called by:
     ├── data_extraction/ast_extractor.py   offline batch, 11 workers
     └── src/inference/preprocess.py        online, one contract per request
```

### What Changes Require a Schema Rebuild

Any modification to `NODE_TYPES`, `VISIBILITY_MAP`, `EDGE_TYPES`, or `_build_node_features()` in `graph_extractor.py` requires **all four steps** — skipping any causes silent accuracy regression:

1. Rebuild ~68K graph `.pt` files: `python data_extraction/ast_extractor.py --force`
2. Rebuild token `.pt` files: `python data_extraction/tokenizer.py --force`
3. Retrain from scratch: `python scripts/train.py`
4. Increment `FEATURE_SCHEMA_VERSION` in `graph_schema.py` (invalidates inference cache)

---

## Data Preparation

> Only needed when the dataset changes or `FEATURE_SCHEMA_VERSION` is bumped.
> **Never regenerate splits** — all checkpoints share the same `val_indices.npy`.

```bash
cd ml
```

### Step 1 — Graph Extraction

```
contracts_metadata.parquet
        │
        ▼
ast_extractor.py  ── orchestration only ── (11 parallel workers)
        │
        ├── parse_solc_version()      resolve solc binary per version group
        ├── GraphExtractionConfig(    builds config per contract:
        │     solc_binary=...,          version-pinned binary
        │     solc_version=...,         for --allow-paths compat
        │     allow_paths=...,          project root for imports
        │     multi_contract_policy="first"
        │   )
        │
        ├── calls graph_extractor.extract_contract_graph(sol_path, config)
        │   ┌──────────────────────────────────────────────────────────┐
        │   │  imports from graph_schema.py:                           │
        │   │    NODE_TYPES, EDGE_TYPES, VISIBILITY_MAP                │
        │   │                                                          │
        │   │  Slither(sol_path, solc=..., solc_args=...)              │
        │   │    ↓ SolcCompilationError or SlitherParseError on fail   │
        │   │  _select_contract()  → first non-dependency contract     │
        │   │    ↓ EmptyGraphError if all declarations are deps        │
        │   │  _build_node_features(obj, type_id)  8-dim float32       │
        │   │    [type_id, visibility, pure, view, payable,            │
        │   │     reentrant, complexity, loc]                          │
        │   │  _build_edges()                                          │
        │   │    CALLS, READS, WRITES, EMITS, INHERITS                 │
        │   │    → edge_index [2,E] int64, edge_attr [E] int64         │
        │   │  returns Data(x, edge_index, edge_attr, contract_name)   │
        │   └──────────────────────────────────────────────────────────┘
        │
        ├── attaches caller-specific metadata:
        │     graph.contract_path  → path whose md5 stem = SHA256
        │     graph.y = 0          (multilabel_index.csv owns labels)
        │
        └── writes <md5_of_path>.pt ──► ml/data/graphs/
              graph.x           [N, 8]   node feature matrix
              graph.edge_index  [2, E]   COO connectivity
              graph.edge_attr   [E]      edge type IDs  (0–4)
              graph.contract_path        Path.stem = SHA256 (CSV bridge)
```

```bash
# Build with Docker (no local solc required — see Docker section below)
docker build -f docker/Dockerfile.slither -t sentinel-slither .

poetry run python data_extraction/ast_extractor.py \
  --input ml/data/processed/_cache/contracts_metadata.parquet \
  --output ml/data/graphs/
```

> **Two hash systems — never mix:**
> - **SHA256** = MD5 of `.sol` file *content* → BCCC filename, CSV column 2
> - **MD5** = MD5 of `.sol` file *path* → `.pt` filename (`hash_utils.get_contract_hash()`)
>
> Bridge: `graph.contract_path` inside each `.pt` → `Path(...).stem` = SHA256.

### Step 2 — Tokenisation

```
contracts_metadata.parquet
        │
        ▼
tokenizer.py  (microsoft/codebert-base tokenizer)
        │
        ├── reads source code string from parquet
        ├── md5 hash via hash_utils.get_contract_hash(path)
        ├── AutoTokenizer(max_length=512, padding, truncation)
        └── <md5_of_path>.pt ──► ml/data/tokens/
              tokens["input_ids"]       [512]  int64
              tokens["attention_mask"]  [512]  int64  (1=real, 0=PAD)
              tokens["schema_version"]         FEATURE_SCHEMA_VERSION string
```

```bash
poetry run python data_extraction/tokenizer.py \
  --input ml/data/processed/_cache/contracts_metadata.parquet \
  --output ml/data/tokens/
```

### Step 3 — Build Multi-Label Index

```bash
poetry run python scripts/build_multilabel_index.py
# Output: ml/data/processed/multilabel_index.csv
#         68,523 rows × columns: md5_stem + 10 class columns (0/1)
#         md5_stem is the pairing key for DualPathDataset._label_map
```

> `create_label_index.py` is **OBSOLETE** — it reads binary `graph.y` labels into a simple
> `hash → 0/1` CSV. It was superseded by `build_multilabel_index.py` which produces the
> full 10-column multi-label index from BCCC vulnerability annotations. Do not use.

### Step 4 — Create Stratified Splits

```bash
poetry run python scripts/create_splits.py
# Output: ml/data/splits/
#   train_indices.npy   47,966 samples
#   val_indices.npy     10,278 samples
#   test_indices.npy    10,279 samples
# Stratified on any_vulnerable = sum(class_cols) > 0
```

> **Do NOT regenerate.** All checkpoints (v1, v2, v3) were evaluated on the same
> `val_indices.npy`. Regenerating breaks experiment comparability.

### Step 5 — Validate Graph Dataset

```bash
poetry run python scripts/validate_graph_dataset.py [--graphs-dir ml/data/graphs]
# Checks per .pt file:
#   edge_attr is present
#   shape is [E]  (1-D int64, NOT the old [E, 1] shape)
#   all values in [0, NUM_EDGE_TYPES) = [0, 5)
#
# Exit 0 → safe to train
# Exit 1 → re-extract: python data_extraction/ast_extractor.py --force
```

### Docker — Slither Environment

`docker/Dockerfile.slither` builds a self-contained Ubuntu 20.04 image with:

- `slither-analyzer==0.10.0`
- **26 pre-bundled solc binaries** baked at image build time — no runtime GitHub API calls

| Version range | Binaries included |
|---------------|------------------|
| 0.8.x | 0.8.0, 0.8.20 |
| 0.7.x | 0.7.6 |
| 0.6.x | 0.6.12 |
| 0.5.x | 0.5.0, 0.5.1, 0.5.2, 0.5.4, 0.5.7, 0.5.8, 0.5.17 |
| 0.4.x | 0.4.2, 0.4.4, 0.4.8, 0.4.11, 0.4.12, 0.4.15, 0.4.16, 0.4.17, 0.4.18, 0.4.19, 0.4.20, 0.4.21, 0.4.23, 0.4.24, 0.4.25, 0.4.26 |

Default version: `0.8.20`. `solc-select` is included for runtime switching.

```bash
docker build -f docker/Dockerfile.slither -t sentinel-slither .
docker run --rm -v $(pwd):/workspace sentinel-slither \
  python data_extraction/ast_extractor.py --input ... --output ...
```

---

## Dataset — DualPathDataset

**File:** `ml/src/datasets/dual_path_dataset.py`

`DualPathDataset` loads paired graph and token `.pt` files for training. Each sample is one smart contract represented two ways — as a PyG graph (→ GNNEncoder) and as CodeBERT token tensors (→ TransformerEncoder).

### Label Modes

```
DualPathDataset(label_csv=None)         ← binary mode (default)
  label = graph.y                         scalar 0/1 long
  collate produces: labels [B] long

DualPathDataset(label_csv=Path(...))    ← multi-label mode (Track 3)
  label = multilabel_index.csv row         float32 [10]
  collate produces: labels [B, 10] float32

  _label_map: dict  md5_stem → float32[10]
    built once at __init__ from multilabel_index.csv
    columns: md5_stem + CallToUnknown ... UnusedReturn (10 classes)
```

### Paired Hash Discovery

```
__init__():
  graph_hashes  = {f.stem for f in graphs_dir/*.pt}
  token_hashes  = {f.stem for f in tokens_dir/*.pt}
  paired_hashes = graph_hashes ∩ token_hashes   (sorted — deterministic)

  unmatched graph files → WARNING (skipped silently)
  unmatched token files → WARNING (skipped silently)

  if indices provided:
    self.paired_hashes = [paired_hashes[i] for i in indices]
    → enforces train / val / test splits from .npy index files

  if validate=True:
    load self[0] eagerly → catches .pt format errors before training starts
```

**PyTorch 2.6+ safe-globals:** `Data`, `DataEdgeAttr`, `DataTensorAttr` are registered at module level so `weights_only=True` works on graph `.pt` files without disabling pickle security.

**Edge_attr shape guard** in `__getitem__`: old `.pt` files stored `edge_attr` as `[E, 1]`; `GNNEncoder` requires `[E]`. A `squeeze(-1)` guard normalises legacy files transparently — no re-extraction needed.

### RAM Cache

```
DualPathDataset(cache_path=Path("ml/data/cached_dataset.pkl"))

  if cache file exists:
    load pickle → self.cached_data: dict  md5_stem → (graph, tokens)
    integrity check:
      isinstance(cached_data, dict)
      paired_hashes[0] is present
      entry has graph.x and tokens["input_ids"]

  __getitem__: reads from dict instead of individual .pt files
  → reduces per-epoch I/O from hours to minutes on large datasets

  if cache file missing: WARNING + fallback to per-file reads
```

Build the cache once with `create_cache.py` (not in the repository yet; scaffold noted in the docstring).

> **Note:** `cache_path` is an explicit constructor argument (default `None`). The original code had a hardcoded absolute path that silently missed the cache on every machine except the author's. Callers must opt in deliberately.

### Collate Function

`dual_path_collate_fn` must be at module level (not a method) for DataLoader multiprocessing compatibility.

```
batch: List[(graph, tokens, label)]
          │
          ├── graphs → Batch.from_data_list()       → PyG Batch
          │     (variable-size graphs merged into one disconnected graph
          │      with a `batch` index tensor [N] mapping nodes to samples)
          │
          ├── tokens["input_ids"]      → torch.stack → [B, 512]
          │   tokens["attention_mask"] → torch.stack → [B, 512]
          │
          └── labels
                multi-label: stack [B, 10] float32  (kept as-is)
                binary:      stack [B, 1]  long → squeeze(1) → [B]

returns: (Batch, {input_ids, attention_mask}, labels)
```

**Usage in training:**

```python
from ml.src.datasets.dual_path_dataset import DualPathDataset, dual_path_collate_fn
from torch.utils.data import DataLoader
import numpy as np

train_idx = np.load("ml/data/splits/train_indices.npy")
dataset = DualPathDataset(
    graphs_dir  = "ml/data/graphs",
    tokens_dir  = "ml/data/tokens",
    indices     = train_idx.tolist(),
    label_csv   = Path("ml/data/processed/multilabel_index.csv"),
)
loader = DataLoader(
    dataset,
    batch_size      = 32,
    collate_fn      = dual_path_collate_fn,
    num_workers     = 2,
    persistent_workers = True,
    pin_memory      = True,
)
```

---

## Model Architecture

```
Solidity source string
        │
        ├────────────────────────────────────┐
        │                                    │
        ▼                                    ▼
┌───────────────────────┐      ┌─────────────────────────────────┐
│  GNN PATH             │      │  TRANSFORMER PATH               │
│                       │      │                                 │
│  graph_extractor.py   │      │  CodeBERT tokenizer             │
│   → x  [N, 8]         │      │   → input_ids    [B, 512]       │
│   → edge_index [2, E] │      │   → attn_mask    [B, 512]       │
│   → edge_attr  [E]    │      │                                 │
│                       │      │  TransformerEncoder             │
│  GNNEncoder           │      │   CodeBERT (124M frozen)        │
│   edge_emb [E, 16]    │      │   + LoRA r=8 α=16 (~295K train) │
│   conv1 [N,8]→[N,64]  │      │   Q+V, all 12 layers            │
│   conv2 [N,64]→[N,64] │      │                                 │
│   conv3 [N,64]→[N,64] │      │  Output: [B, 512, 768]          │
│                       │      │  (all token positions, not CLS) │
│  Output: [N, 64]      │      └───────────────┬─────────────────┘
│          [N] batch    │                      │
└──────────┬────────────┘                      │
           └─────────────────┬─────────────────┘
                             │
                             ▼
           ┌─────────────────────────────────────┐
           │  CrossAttentionFusion               │
           │                                     │
           │  node_proj  [N,64]  → [N,256]       │
           │  token_proj [B,512,768] → [B,512,256]│
           │  to_dense_batch → [B, max_n, 256]   │
           │                                     │
           │  Node→Token MHA                     │
           │    Q=nodes  [B,n,256]               │
           │    K=V=tokens [B,512,256]            │
           │    key_mask=PAD token positions      │
           │    → enriched_nodes [B,n,256]        │
           │    (pad positions zeroed after attn) │
           │                                     │
           │  Token→Node MHA                     │
           │    Q=tokens [B,512,256]              │
           │    K=V=nodes [B,n,256]               │
           │    key_mask=padded node positions    │
           │    → enriched_tokens [B,512,256]     │
           │                                     │
           │  Masked mean pool:                  │
           │    pooled_nodes  → [B, 256]          │
           │    pooled_tokens → [B, 256]          │
           │  concat [B,512] → Linear→ReLU→Drop  │
           │  → [B, 128]   ← LOCKED DIM          │
           └─────────────────┬───────────────────┘
                             │
                             ▼
           ┌─────────────────────────────────────┐
           │  Classifier                         │
           │  Linear(128, 10) → logits [B, 10]   │
           │  NO Sigmoid inside model            │
           └─────────────────┬───────────────────┘
                             │
                             ▼
           Predictor._score():
             probs = sigmoid(logits)        [B, 10]
             apply per-class threshold JSON
             vulnerabilities = [
               {vulnerability_class, probability}
               for prob ≥ threshold[class]
             ]
             label = "vulnerable" if any else "safe"
```

### GNN Encoder

**File:** `ml/src/models/gnn_encoder.py`

```
Input: x [N,8],  edge_index [2,E],  batch [N],  edge_attr [E] (int64)
           │                                              │
           │           edge_emb  Embedding(5,16)          │
           │           [E] → [E, 16]                      │
           │                    │                         │
           ▼                    ▼                         │
     GATConv conv1  (in=8, out=8, heads=8, edge_dim=16)
          → [N, 64]   (8 heads × 8 dims, concat=True)
          → ReLU → Dropout(0.2)

     GATConv conv2  (in=64, out=8, heads=8, edge_dim=16)
          → [N, 64]   2-hop structural context
          → ReLU → Dropout(0.2)

     GATConv conv3  (in=64, out=64, heads=1, edge_dim=16)
          → [N, 64]   3-hop context, final node embeddings
          (no activation — CrossAttentionFusion projects next)

Output: node_embs [N, 64],  batch [N]
        NOT pooled — pooling deferred to CrossAttentionFusion
```

> **Why edge_attr matters:** A `CALLS` edge and a `READS` edge are fundamentally different structural patterns. Reentrancy requires a `CALLS` edge back to the caller. Without typed edges, GATConv attention is purely node-feature-based and cannot distinguish these.
>
> **Graceful degradation:** `edge_attr=None` (legacy `.pt` files) → edge embeddings fall back to zero vectors. Old checkpoints still run.

### Transformer Encoder (CodeBERT + LoRA)

**File:** `ml/src/models/transformer_encoder.py`

```
Input: input_ids [B,512],  attention_mask [B,512]
           │
           ▼
  CodeBERT  microsoft/codebert-base
  ├── 12 attention layers, hidden_dim=768
  ├── 124,705,536 parameters — ALL FROZEN (requires_grad=False)
  └── LoRA matrices injected into query + value of every layer
        A [768, r=8]  and  B [r=8, 768]  per projection
        Forward: W_frozen @ x + (B @ A) @ x × (alpha/r = 2.0)
        Trainable: ~295,296 parameters across 12 layers × Q+V

Output: last_hidden_state [B, 512, 768]
        ALL 512 positions — CrossAttentionFusion attends to each one
```

> **Why LoRA:** Full fine-tune → OOM on 8 GB VRAM + catastrophic forgetting. Frozen → never adapts to vulnerability semantics. LoRA → ~295K trainable params steer attention toward security patterns without touching the frozen backbone.
>
> **Why no `torch.no_grad()` wrapper:** peft's `get_peft_model()` marks every original CodeBERT weight with `requires_grad=False`. Wrapping the whole `self.bert()` call in `no_grad()` would cut gradient flow to the LoRA A/B matrices inside the same pass — silently killing LoRA training.

### CrossAttention Fusion

**File:** `ml/src/models/fusion_layer.py`

```
node_embs [N,64]  +  token_embs [B,512,768]  +  attention_mask [B,512]
      │                       │
      ▼                       ▼
node_proj Linear(64→256)   token_proj Linear(768→256)
      │                       │
      ▼                       │
to_dense_batch()              │
  padded_nodes [B, n, 256]    │
  node_real_mask [B, n]       │
  node_padding_mask = ~mask   │
                              │
  token_padding_mask = (attention_mask == 0)
      │
      ├── Node→Token MHA   Q=nodes  K=V=tokens  key_mask=token_PAD
      │         → enriched_nodes [B, n, 256]
      │         zero-out padding node positions after attention
      │
      └── Token→Node MHA   Q=tokens  K=V=nodes  key_mask=node_padding
                → enriched_tokens [B, 512, 256]

Masked mean pool enriched_nodes  → pooled_nodes  [B, 256]
Masked mean pool enriched_tokens → pooled_tokens [B, 256]

cat([pooled_nodes, pooled_tokens]) → [B, 512]
Linear(512→128) → ReLU → Dropout → [B, 128]  ← LOCKED (ZKML depends on this)
```

### Classifier

```
[B, 128]  →  nn.Linear(128, 10)  →  raw logits [B, 10]

No Sigmoid inside the model.
  Training:  BCEWithLogitsLoss / FocalLoss handle sigmoid internally.
  Inference: Predictor._score() applies sigmoid(logits) externally.
```

### Long-Contract Path — Sliding Window

```
Source > 512 tokens?
        │
        ▼ YES
window_size = 512,  stride = 256,  max_windows = 8

GNN graph built ONCE from the full AST (no windowing on graph side)

For each window:
  SentinelModel.forward(graph, window_input_ids, window_mask) → probs [1,10]

Aggregate:
  final_probs[class] = max(window_probs[class])   per class

API response includes: "windows_used": N,  "truncated": false
```

### Node Feature Vector

| Index | Feature | Encoding |
|-------|---------|----------|
| 0 | `type_id` | CONTRACT=7, STATE_VAR=0, FUNCTION=1, MODIFIER=2, EVENT=3, FALLBACK=4, RECEIVE=5, CONSTRUCTOR=6 |
| 1 | `visibility` | public/external=0, internal=1, private=2 |
| 2 | `pure` | 0/1 |
| 3 | `view` | 0/1 |
| 4 | `payable` | 0/1 |
| 5 | `reentrant` | 0/1 (Slither `is_reentrant` flag) |
| 6 | `complexity` | float — CFG node count |
| 7 | `loc` | float — lines of source |

Insertion order: `CONTRACT → STATE_VARs → FUNCTIONs → MODIFIERs → EVENTs`
Non-function nodes receive `0.0` for features 2–5.

### Edge Types

| ID | Type | Meaning |
|----|------|---------|
| 0 | `CALLS` | function → internally-called function |
| 1 | `READS` | function → state variable it reads |
| 2 | `WRITES` | function → state variable it writes |
| 3 | `EMITS` | function → event it emits |
| 4 | `INHERITS` | contract → parent contract (linearised MRO) |

---

## Output Classes

Defined in `ml/src/training/trainer.py` as `CLASS_NAMES` — **single source of truth** for index order. Never insert in the middle; append new classes at index 10+.

| Index | Class |
|-------|-------|
| 0 | CallToUnknown |
| 1 | DenialOfService |
| 2 | ExternalBug |
| 3 | GasException |
| 4 | IntegerUO |
| 5 | MishandledException |
| 6 | Reentrancy |
| 7 | Timestamp |
| 8 | TransactionOrderDependence |
| 9 | UnusedReturn |

---

## Dataset

| Item | Value |
|------|-------|
| Source | BCCC-SCsVul-2024 |
| Graph `.pt` files | 68,523 (MD5 stem, `ml/data/graphs/`) — re-extracted 2026-05-03 |
| Token `.pt` files | 68,568 (MD5 stem, `ml/data/tokens/`) |
| Split — train | 47,966 samples |
| Split — val | 10,278 samples |
| Split — test | 10,279 samples |
| Vulnerable rate | 64.3% (64,099 vulnerable / 24,456 safe — stratified) |
| Label CSV | `ml/data/processed/multilabel_index.csv` — 68,523 rows × 10 classes |

---

## Training

### Run Training

```bash
cd ml
TRANSFORMERS_OFFLINE=1 poetry run python scripts/train.py \
  --run-name   multilabel-v3-fresh-60ep \
  --experiment sentinel-retrain-v3 \
  --label-csv  data/processed/multilabel_index.csv \
  --epochs     60 \
  --batch-size 32 \
  --patience   10
```

Key `TrainConfig` fields:

| Field | v3 value | Notes |
|-------|----------|-------|
| `architecture` | `"cross_attention_lora"` | Written into checkpoint config |
| `batch_size` | 32 | Safe on RTX 3070 8 GB with AMP |
| `lora_r` | 8 | LoRA rank (~295K trainable params) |
| `lora_alpha` | 16 | Effective scale = alpha/r = 2.0 |
| `loss_fn` | `"bce"` | v4 plan: switch to `"focal"` |
| `use_edge_attr` | True | Typed edge-relation embeddings |
| `gnn_edge_emb_dim` | 16 | Edge embedding dimension |
| `fusion_output_dim` | 128 | Fused representation size (**LOCKED**) |
| `grad_clip` | 1.0 | Prevents LoRA gradient spikes |
| `patience` | 10 | Early-stop on val F1-macro |

Speed optimisations: AMP/BF16, TF32 matmuls, `persistent_workers=True`, `zero_grad(set_to_none=True)`, MLflow artifact logged once at end (not per epoch).

MLflow tracks per run: all `TrainConfig` fields, `val_f1_macro`, `val_f1_micro`, `val_hamming`, `val_exact_match`, `focal_gamma`, `focal_alpha`, and `val_f1_{class}` × 10.

### FocalLoss

**File:** `ml/src/training/focalloss.py`

Use `--loss-fn focal` when class imbalance is the bottleneck (e.g. DenialOfService at 137 samples).

```
FocalLoss(gamma=2.0, alpha=0.25)

  BCE × alpha_t × (1 - pt)^gamma

  pt = model confidence on the correct class
       target=1 → pt = p      (confident positive prediction → small loss)
       target=0 → pt = 1 - p  (confident negative prediction → small loss)

  alpha_t = 0.25 for positive class  (vulnerable=64% majority → down-weighted)
          = 0.75 for negative class  (safe=36% minority      → up-weighted)

  gamma=2.0 — hard-example focus multiplier
    pt=0.9 (easy):   (1-0.9)^2 = 0.01 → 99% loss reduction
    pt=0.5 (hard):   (1-0.5)^2 = 0.25 → 75% loss reduction
    pt=0.1 (very hard): (1-0.1)^2 = 0.81 → 19% loss reduction
```

> **Important:** `FocalLoss` expects **post-sigmoid probabilities**, not raw logits. The `_FocalFromLogits` wrapper in `trainer.py` applies `sigmoid()` before forwarding to `FocalLoss.forward()`. Do not pass logits directly.
>
> **BF16 guard:** `predictions.float()` and `targets.float()` are cast at the top of `forward()`. Under AMP autocast, BF16 probabilities near 0 silently become exactly `0.0`, making `log(p) = -inf` and `loss = nan`. The explicit cast prevents this.
>
> **`alpha=0.25` rationale:** The class ratio is 1.8× (64% vs 36%), not 3× as the default weight ratio (0.25 vs 0.75) might suggest. See `run_overnight_experiments.py` experiment 1 which tests `alpha=0.35` to soften the penalty gap.

### Resume from Checkpoint

```bash
# Model-only resume (recommended when batch_size or any hyperparameter changed)
poetry run python scripts/train.py \
  --resume-from ml/checkpoints/multilabel-v3-fresh-60ep_best.pt

# Full resume (model + optimizer + scheduler + patience counter)
# Only use when batch_size is IDENTICAL to the checkpoint
poetry run python scripts/train.py \
  --resume-from ml/checkpoints/multilabel-v3-fresh-60ep_best.pt \
  --no-resume-model-only

# Full resume + reset optimizer (keeps model weights + epoch counter,
# discards stale Adam moments — use when batch_size changed)
poetry run python scripts/train.py \
  --resume-from ml/checkpoints/multilabel-v3-fresh-60ep_best.pt \
  --no-resume-model-only \
  --resume-reset-optimizer
```

Validates on resume: `num_classes` and `architecture` must match current `TrainConfig`.

A `{checkpoint}.state.json` **patience sidecar** is written after every epoch with the real `patience_counter`. On resume, the sidecar overrides the checkpoint's saved counter (which is always 0 at a new best). If absent, a warning is logged.

Checkpoint load pattern (`weights_only=False` required — LoRA state dict contains peft objects):

```python
raw = torch.load(path, weights_only=False)
state_dict = raw["model"] if "model" in raw else raw
```

### Per-Class Threshold Tuning

```bash
TRANSFORMERS_OFFLINE=1 ml/.venv/bin/python scripts/tune_threshold.py \
  --checkpoint ml/checkpoints/multilabel-v3-fresh-60ep_best.pt
# Grid: 0.05, 0.10, ..., 0.95 (19 values) per class on val split
# Writes: ml/checkpoints/multilabel-v3-fresh-60ep_best_thresholds.json
```

The thresholds JSON **must travel with** its checkpoint. Never deploy a checkpoint without its companion JSON.

### Hyperparameter Search — Overnight Experiments

**File:** `ml/scripts/run_overnight_experiments.py`

Sequential launcher for 4 overnight MLflow experiments. GPU trains one model at a time; experiments run in order with error isolation (a crash in one does not abort the rest).

```bash
# Start all 4 experiments
nohup TRANSFORMERS_OFFLINE=1 poetry run python scripts/run_overnight_experiments.py \
    > ml/logs/overnight.log 2>&1 &
echo "PID $!"

# Resume from experiment N after a crash (0-indexed from 1)
nohup poetry run python scripts/run_overnight_experiments.py \
    --start-from 3 > ml/logs/overnight_resume.log 2>&1 &

# Check progress in the morning
tail -50 ml/logs/overnight.log
mlflow ui --port 5000
```

Experiment matrix (each overrides only changed fields from `TrainConfig` defaults):

| # | Run name | Change | Hypothesis |
|---|----------|--------|------------|
| 1 | `run-alpha-tune` | `focal_alpha=0.35` | Soften 3× penalty gap closer to 1.8× actual class ratio |
| 2 | `run-more-epochs` | `epochs=40` | Baseline still improving at epoch 16 — extend to find plateau |
| 3 | `run-lr-lower` | `lr=3e-5, epochs=30` | Reduce F1 oscillation (~0.15 range between epochs) |
| 4 | `run-combined` | `focal_alpha=0.35, lr=3e-5, epochs=30` | Combine best changes from 1 and 3 |

**Reading order after overnight run:**
1. `run-lr-lower` → did the F1 curve smooth out?
2. `run-more-epochs` → what epoch did it peak?
3. `run-alpha-tune` → did F1-safe improve vs baseline?
4. `run-combined` → did combining both beat all singles?
5. For each: `val_recall_vulnerable` is the primary signal

### Recommended v4 Configuration

v3 plateaued at raw F1-macro 0.4715 from epoch ~54 — capacity ceiling under BCE with LoRA r=8.

```bash
TRANSFORMERS_OFFLINE=1 poetry run python scripts/train.py \
  --run-name    multilabel-v4-focal-lora16 \
  --experiment  sentinel-retrain-v4 \
  --epochs      60 \
  --batch-size  32 \
  --patience    10 \
  --loss-fn     focal \
  --focal-gamma 2.0 \
  --lora-r      16 \
  --lora-alpha  32
```

| Change | Rationale |
|--------|-----------|
| `--loss-fn focal --focal-gamma 2.0` | Down-weights easy negatives; forces attention on DenialOfService (137 support) |
| `--lora-r 16` | Doubles trainable params (~589K vs 295K) — addresses plateau |
| Weighted sampler for DenialOfService | 39× underrepresented vs IntegerUO |

---

## Active Checkpoint

```
── v3 (current best) ────────────────────────────────────────────────────────
File:        ml/checkpoints/multilabel-v3-fresh-60ep_best.pt
Thresholds:  ml/checkpoints/multilabel-v3-fresh-60ep_best_thresholds.json
Run:         multilabel-v3-fresh-60ep  (sentinel-retrain-v3)
Completed:   2026-05-05  |  60 epochs  |  batch_size=32
Best epoch:  ~52–53  |  plateau from ~ep 54
Raw F1-macro:   0.4715
Tuned F1-macro: 0.5069  ✅ (gate was > 0.4884)
Architecture:   cross_attention_lora  (LoRA r=8 α=16, edge_attr active)

── v2 (paused — superseded) ─────────────────────────────────────────────────
File:        ml/checkpoints/multilabel_crossattn_v2_best.pt
Status:      Stopped at epoch 43, batch-size mismatch. Superseded by v3.
Best raw F1: 0.4629 (epoch 37)

── baseline (pre-edge_attr) ─────────────────────────────────────────────────
File:        ml/checkpoints/multilabel_crossattn_best.pt
Val F1-macro: 0.4679  (epoch 34)
Architecture: cross_attention_lora  (trained WITHOUT edge_attr)
```

### Per-Class Thresholds and F1 (v3)

| Class | Threshold | F1 | Precision | Recall | Support |
|-------|-----------|----|-----------|--------|---------|
| CallToUnknown | 0.70 | 0.394 | 0.322 | 0.507 | 1,266 |
| DenialOfService | 0.95 | 0.400 | 0.318 | 0.540 | 137 |
| ExternalBug | 0.65 | 0.435 | 0.312 | 0.715 | 1,622 |
| GasException | 0.55 | 0.550 | 0.403 | 0.867 | 2,589 |
| IntegerUO | 0.50 | 0.821 | 0.759 | 0.896 | 5,343 |
| MishandledException | 0.60 | 0.492 | 0.365 | 0.754 | 2,207 |
| Reentrancy | 0.65 | 0.536 | 0.449 | 0.665 | 2,501 |
| Timestamp | 0.75 | 0.479 | 0.403 | 0.591 | 1,077 |
| TransactionOrderDependence | 0.60 | 0.477 | 0.342 | 0.787 | 1,800 |
| UnusedReturn | 0.70 | 0.486 | 0.395 | 0.631 | 1,716 |

### Retrain Evaluation Protocol

| Gate | Requirement |
|------|-------------|
| Graph dataset | `validate_graph_dataset.py` exits 0 |
| Held-out split | Use `ml/data/splits/val_indices.npy` — do NOT regenerate |
| **v4 success gate** | Tuned val F1-macro > **0.5069** on the same held-out split |
| Per-class floor | No class drops > 0.05 F1 from v3 tuned values above |
| Rollback rule | Tuned F1 < 0.5069 → revert to v3, adjust hyperparameters |
| MLflow experiment | `sentinel-retrain-v4` |

---

## Inference API

Start the server:

```bash
TRANSFORMERS_OFFLINE=1 \
SENTINEL_CHECKPOINT=ml/checkpoints/multilabel-v3-fresh-60ep_best.pt \
SENTINEL_THRESHOLDS=ml/checkpoints/multilabel-v3-fresh-60ep_best_thresholds.json \
ml/.venv/bin/uvicorn ml.src.inference.api:app --port 8001
```

Environment variables:

| Variable | Default | Purpose |
|----------|---------|---------|
| `SENTINEL_CHECKPOINT` | `ml/checkpoints/multilabel-v3-fresh-60ep_best.pt` | Checkpoint path |
| `SENTINEL_THRESHOLDS` | auto-detected (`{stem}_thresholds.json`) | Per-class threshold JSON |
| `SENTINEL_PREDICT_TIMEOUT` | `60` | Inference timeout (seconds) |
| `SENTINEL_DRIFT_BASELINE` | `ml/data/drift_baseline.json` | Drift detection baseline |
| `SENTINEL_DRIFT_CHECK_INTERVAL` | `50` | KS test every N requests |

### Endpoints

**`POST /predict`**

```json
// Request
{ "source_code": "<solidity source string>" }

// Response
{
  "label": "vulnerable",
  "vulnerabilities": [
    { "vulnerability_class": "Reentrancy", "probability": 0.8943 },
    { "vulnerability_class": "IntegerUO",  "probability": 0.7102 }
  ],
  "thresholds":   [0.70, 0.95, 0.65, 0.55, 0.50, 0.60, 0.65, 0.75, 0.60, 0.70],
  "truncated":    false,
  "windows_used": 1,
  "num_nodes":    12,
  "num_edges":    18
}
```

- `thresholds` — 10 per-class values in `CLASS_NAMES` index order
- `truncated: true` — single window was cut at 512 tokens
- `windows_used > 1` — sliding-window path was taken

**`GET /health`**

```json
{
  "status": "ok",
  "predictor_loaded": true,
  "checkpoint": "ml/checkpoints/multilabel-v3-fresh-60ep_best.pt",
  "architecture": "cross_attention_lora",
  "thresholds_loaded": true
}
```

**`GET /metrics`** — Prometheus endpoint.

| Metric | Type | Description |
|--------|------|-------------|
| `sentinel_model_loaded` | Gauge | 1 when predictor is loaded |
| `sentinel_gpu_memory_bytes` | Gauge | GPU memory allocated (bytes) |
| `sentinel_drift_alerts_total{stat}` | Counter | KS drift alerts by stat name |

### HTTP Status Codes

| Code | Meaning |
|------|---------|
| 200 | Success |
| 400 | Invalid or empty Solidity input |
| 413 | Source > 1 MB or GPU OOM |
| 503 | Predictor not yet loaded |
| 504 | Inference timeout |

---

## Inference Cache

**File:** `ml/src/inference/cache.py`

```
Cache key: "{content_md5}_{FEATURE_SCHEMA_VERSION}"
               │                    │
               │                    └── bumping version auto-invalidates
               │                        all stale entries without manual cleanup
               └── hash_utils.get_contract_hash_from_content(source)
                   (content-addressed: same code uploaded twice → cache hit)

Cache hit  → return (graph.pt, tokens.pt) directly, skip Slither (3–5 s saved)
Cache miss → run ContractPreprocessor, write result to disk
TTL        → configurable, default 86400 s (24 hours)
```

```python
from ml.src.inference.cache import InferenceCache
from ml.src.inference.preprocess import ContractPreprocessor

cache = InferenceCache(
    cache_dir="~/.cache/sentinel/preprocess",
    ttl_seconds=86400,
)
preprocessor = ContractPreprocessor(cache=cache)
```

---

## Drift Detection

**File:** `ml/src/inference/drift_detector.py`

```
Request stream
      │
      ▼
DriftDetector.update_stats({"num_nodes": N, "num_edges": E, ...})
      │
      ├── requests < N_WARMUP (500)?
      │     → suppress all alerts (warm-up mode)
      │
      └── requests ≥ 500  AND  request_count % DRIFT_CHECK_INTERVAL == 0?
            → KS test per feature stat against drift_baseline.json
                p < 0.05 → increment sentinel_drift_alerts_total{stat}
```

Build the baseline **after warm-up** (not from training data):

```bash
# After collecting ≥ 500 real audit requests:
python ml/scripts/compute_drift_baseline.py \
    --source      warmup \
    --warmup-log  ml/data/warmup_stats.jsonl \
    --output      ml/data/drift_baseline.json
```

> **Do not use `ml/data/graphs/` as the baseline.** The BCCC-2024 corpus is a historical
> snapshot — using it causes the KS test to fire on almost every modern 2026 contract.
> The baseline must come from real production traffic collected during warm-up.

---

## MLflow and Model Registry

```bash
mlflow ui --port 5000
# → http://localhost:5000
```

| Experiment | Status | Best Tuned F1 |
|------------|--------|---------------|
| `sentinel-multilabel` | complete | 0.4679 (epoch 34) |
| `sentinel-retrain-v2` | paused at epoch 43 (batch-size mismatch) | 0.4629 |
| `sentinel-retrain-v3` | complete | **0.5069** |

**Promote a checkpoint:**

```bash
python ml/scripts/promote_model.py \
    --checkpoint   ml/checkpoints/multilabel-v3-fresh-60ep_best.pt \
    --stage        Staging \
    --val-f1-macro 0.5069 \
    --note         "v3: edge_attr active; tuned F1-macro 0.5069"

# --dry-run           preview without writing to MLflow
# --stage Production  archives the previous Production version
```

---

## DVC

Large artifacts (graphs, tokens, splits, checkpoints) are DVC-tracked, not stored in git.

```bash
dvc pull   # download current data version
dvc push   # push new artifacts after retraining
```

| DVC pointer | Content |
|-------------|---------|
| `ml/data/graphs.dvc` | ~68K graph `.pt` files |
| `ml/data/tokens.dvc` | ~68K token `.pt` files |
| `ml/data/splits.dvc` | `train/val/test_indices.npy` |
| `ml/checkpoints.dvc` | All checkpoint `.pt` + threshold `.json` files |

---

## Testing

```bash
cd ml
poetry run pytest tests/ -v
```

`tests/conftest.py` sets `TRANSFORMERS_OFFLINE=1` before any import (HuggingFace checks this
at import time) and creates a **session-scoped** `TestClient` so the ~500 MB model loads once
for all API tests rather than once per test function.

All 10 test modules use synthetic data — no real contracts or checkpoints required.

| Test file | What it covers |
|-----------|----------------|
| `test_model.py` | `SentinelModel` forward pass, output shape, class count |
| `test_gnn_encoder.py` | `GNNEncoder`: edge_attr embedding, graceful degradation on None, head divisibility |
| `test_fusion_layer.py` | `CrossAttentionFusion`: output shape, masked pooling, device parity |
| `test_preprocessing.py` | `ContractPreprocessor`, `graph_extractor` typed exceptions |
| `test_dataset.py` | `DualPathDataset`: pairing logic, binary/multi-label modes, split loading, edge_attr guard, cache integrity, collation |
| `test_trainer.py` | `FocalLoss` forward (BF16 guard, alpha weighting), trainer utilities |
| `test_api.py` | `/predict` and `/health` endpoint contracts, error codes |
| `test_cache.py` | `InferenceCache`: miss writes, hit returns same object, TTL expiry, schema version invalidation |
| `test_drift_detector.py` | `DriftDetector`: warm-up suppression, KS fires on p < 0.05, buffer rolling |
| `test_promote_model.py` | `promote_model.py`: stage validation, dry-run no-op, MLflow tag writes |

---

## File Reference

```
ml/
├── src/
│   ├── models/
│   │   ├── sentinel_model.py        SentinelModel — top-level orchestrator
│   │   ├── gnn_encoder.py           GNNEncoder — 3-layer GAT + edge-type embeddings → [N,64]
│   │   ├── transformer_encoder.py   TransformerEncoder — CodeBERT + LoRA (r=8 default)
│   │   └── fusion_layer.py          CrossAttentionFusion — output_dim=128 (LOCKED)
│   │
│   ├── preprocessing/
│   │   ├── graph_schema.py          NODE_TYPES, EDGE_TYPES, FEATURE_NAMES,
│   │   │                            FEATURE_SCHEMA_VERSION, NODE_FEATURE_DIM=8,
│   │   │                            NUM_EDGE_TYPES=5
│   │   └── graph_extractor.py       extract_contract_graph() — Slither → PyG Data
│   │                                GraphExtractionConfig, typed exceptions
│   │
│   ├── inference/
│   │   ├── api.py                   FastAPI app — lifespan, /predict, /health, /metrics
│   │   ├── predictor.py             Predictor — checkpoint load, sigmoid, per-class thresholds
│   │   ├── preprocess.py            ContractPreprocessor — thin wrapper over graph_extractor
│   │   │                            process(), process_source(), process_source_windowed()
│   │   ├── cache.py                 InferenceCache — disk-backed content-addressed cache (T1-A)
│   │   └── drift_detector.py        DriftDetector — KS-based feature drift monitoring (T2-B)
│   │
│   ├── training/
│   │   ├── trainer.py               Trainer, TrainConfig, CLASS_NAMES, NUM_CLASSES
│   │   │                            AMP, early stopping, full/model-only resume,
│   │   │                            patience sidecar, MLflow logging
│   │   └── focalloss.py             FocalLoss — gamma=2.0, alpha=0.25, FP32 cast
│   │                                opt-in via loss_fn="focal"; expects post-sigmoid probs
│   │
│   ├── datasets/
│   │   └── dual_path_dataset.py     DualPathDataset — binary + multi-label modes,
│   │                                paired-hash discovery, RAM cache, edge_attr guard
│   │                                dual_path_collate_fn — module-level for multiprocessing
│   │
│   └── utils/
│       └── hash_utils.py            get_contract_hash(path)         MD5 of path string
│                                    get_contract_hash_from_content() MD5 of source text
│                                    validate_hash(), get_filename_from_hash()
│                                    get_filename_from_path(), extract_hash_from_filename()
│
├── data_extraction/
│   ├── ast_extractor.py             Offline batch Slither → PyG .pt (V4.3)
│   │                                orchestration only — graph logic in graph_extractor.py
│   └── tokenizer.py                 Offline CodeBERT tokenisation + schema version metadata
│
├── scripts/
│   ├── train.py                     Main training entry point (full/model-only/reset-optimizer)
│   ├── tune_threshold.py            Per-class threshold sweep (0.05–0.95 grid)
│   ├── run_overnight_experiments.py 4-experiment hyperparameter launcher
│   │                                --start-from N crash-resume; sequential GPU runs
│   ├── create_splits.py             Fixed stratified train/val/test split indices
│   ├── build_multilabel_index.py    Build multilabel_index.csv from BCCC labels
│   ├── validate_graph_dataset.py    Validate edge_attr shape [E] + value range in .pt files
│   ├── analyse_truncation.py        Measure token truncation stats across dataset
│   ├── promote_model.py             MLflow registry CLI — Staging / Production promotion
│   ├── compute_drift_baseline.py    Build drift_baseline.json from warmup logs
│   └── create_label_index.py        OBSOLETE — binary label_index.csv from graph.y
│                                    superseded by build_multilabel_index.py
│
├── docker/
│   └── Dockerfile.slither           Ubuntu 20.04 + slither-analyzer==0.10.0
│                                    26 pre-bundled solc binaries (0.4.2 – 0.8.20)
│                                    solc-select included for version switching
│
├── tests/
│   ├── conftest.py                  Sets TRANSFORMERS_OFFLINE=1 before imports
│   │                                Session-scoped TestClient (model loads once)
│   └── test_*.py                    10 test modules — all synthetic data
│
├── data/                            DVC-tracked (not in git)
│   ├── graphs/                      ~68K <md5>.pt PyG graph files
│   ├── tokens/                      ~68K <md5>.pt token tensor files
│   ├── splits/                      train/val/test_indices.npy
│   └── processed/
│       └── multilabel_index.csv     68,523 rows × md5_stem + 10 class columns
│
├── checkpoints/                     DVC-tracked (not in git)
│   ├── multilabel-v3-fresh-60ep_best.pt            ← active (v3, tuned F1 0.5069)
│   ├── multilabel-v3-fresh-60ep_best_thresholds.json
│   ├── multilabel_crossattn_v2_best.pt             ← paused v2 (superseded)
│   ├── multilabel_crossattn_best.pt                ← original baseline (pre-edge_attr)
│   └── multilabel_crossattn_best_thresholds.json
│
├── DIAGRAMS.md                      Mermaid visual diagrams (GitHub-rendered)
├── pyproject.toml                   Python 3.12.1, torch ^2.5.0, peft >=0.13.0
└── README.md                        This file
```

---

## Critical Constraints

| Constraint | Locked Value | Consequence of change |
|-----------|-------------|----------------------|
| `GNNEncoder in_channels` | **8** | Rebuild all 68K graph files + retrain |
| CodeBERT model | `microsoft/codebert-base` | Rebuild token files + retrain |
| `MAX_TOKEN_LENGTH` | **512** | Rebuild token files + retrain |
| Node feature order | fixed 8-dim (see table) | Rebuild graph files + retrain |
| `CrossAttentionFusion output_dim` | **128** | Rebuild ZKML circuit + redeploy verifier |
| `FEATURE_SCHEMA_VERSION` | **`"v1"`** | Bump alongside graph rebuild — invalidates inference cache |
| `CLASS_NAMES` order | indices **0–9 stable** | Silent wrong-class mapping across all consumers |
| `NUM_EDGE_TYPES` | **5** | Rebuild edge_emb layer + retrain |
| `weights_only=False` on checkpoint load | required | LoRA state dict contains peft-specific objects |
| `TRANSFORMERS_OFFLINE` | must be set at **shell level** | Cannot be set inside Python after `transformers` is imported |
| `edge_attr` shape | **`[E]` 1-D int64** | `[E,1]` crashes `nn.Embedding`; validate before training |
| `dual_path_collate_fn` | must be **module-level** | Lambda/method collate functions crash DataLoader multiprocessing |

---

## Known Limitations

**1. Multi-contract files**
Only the first non-dependency contract per `.sol` file is analysed. `GraphExtractionConfig.multi_contract_policy` scaffold exists (`"first"`, `"by_name"`); the `"all"` policy is not yet implemented. See `docs/ROADMAP.md` Move 9.

**2. DenialOfService class**
137 training samples — 39× fewer than IntegerUO. Even with threshold tuned to 0.95, F1 is 0.40. Weighted sampling and focal loss are the planned remediation for v4.

**3. Drift baseline not yet collected**
`DriftDetector` is code-complete but cannot be activated until the warm-up phase (first 500 real audit requests) has been run. Use `compute_drift_baseline.py --source warmup` after warm-up completes.

**4. Temp-file cleanup on SIGKILL (Move 8 partial)**
`preprocess.py`'s `process_source()` writes a `NamedTemporaryFile` because solc requires a real path. The `finally` block cleans up on normal exit and graceful signals, but a `SIGKILL` (e.g. OOM kill) leaves the temp file on disk. Low-priority hardening task — does not affect correctness. Tracked in `docs/ROADMAP.md`.

**5. M6 Integration API not built**
The `api/` directory for the public-facing audit endpoint does not exist. Auth/rate-limit design is required before building routes. See `docs/STATUS.md` and `docs/ROADMAP.md`.
