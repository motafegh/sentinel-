# SENTINEL ML Data Pipeline — Technical Reference

**Document:** ML-T3-DATA-PIPELINE  
**Status:** Active — v5.2 deduped dataset  
**Last updated:** 2026-05-14  

---

## Table of Contents

1. [Overview](#1-overview)
2. [Directory Layout](#2-directory-layout)
3. [Hash Identity and Pairing Key](#3-hash-identity-and-pairing-key)
4. [Graph Format](#4-graph-format)
5. [Token Format](#5-token-format)
6. [Label Index](#6-label-index-multilabel_index_deduped-csv)
7. [DualPathDataset](#7-dualpathdata-set)
8. [Collation](#8-collation)
9. [Split Construction](#9-split-construction)
10. [CEI Augmentation](#10-cei-augmentation)
11. [RAM Cache](#11-ram-cache)
12. [Positive-Weight Computation](#12-positive-weight-computation)
13. [Graph Extraction Pipeline](#13-graph-extraction-pipeline)
14. [Deduplication History](#14-deduplication-history)
15. [Ghost Graphs](#15-ghost-graphs)
16. [Serialization Safety](#16-serialization-safety)
17. [Data Pipeline End-to-End Flow](#17-data-pipeline-end-to-end-flow)

---

## 1. Overview

The SENTINEL training pipeline consumes **paired (graph, token, label)** triples for each Solidity smart contract. Each contract is represented two ways simultaneously:

- **Graph path**: a PyG `Data` object encoding the AST + CFG structure, consumed by `GNNEncoder`.
- **Token path**: a dict of CodeBERT token tensors, consumed by `TransformerEncoder`.

Both representations are keyed by the same MD5 hash of the source file's relative path, written as `<md5>.pt`. A contract is in the dataset only if both a graph file and a token file for its hash exist.

The active training dataset is the **deduped dataset**: 44,470 unique content-identified contracts split 70/15/15 across train/val/test.

---

## 2. Directory Layout

```
ml/data/
├── graphs/                          44,470 files — graph .pt files (PyG Data objects)
│   └── <md5_stem>.pt
├── tokens/                          44,470 files — token .pt files (dict: input_ids, attention_mask)
│   └── <md5_stem>.pt
├── tokens_orphaned/                 24,148 files — legacy tokens from the 68K dataset
│   └── <md5_stem>.pt                             no matching graph; safe to ignore
├── augmented/                       50 files — CEI augmentation .sol source contracts
│   ├── cei_vuln_001.sol             (Reentrancy=1)
│   ├── cei_safe_001.sol             (all classes=0)
│   └── ...
├── splits/
│   └── deduped/
│       ├── train_indices.npy        31,142 indices into multilabel_index_deduped.csv
│       ├── val_indices.npy           6,661 indices
│       └── test_indices.npy          6,667 indices
└── processed/
    └── multilabel_index_deduped.csv 44,470 rows × 11 columns (md5_stem + 10 class labels)

ml/data/
├── cached_dataset_deduped.pkl       ~1.4 GB RAM cache: dict md5_stem → (graph, tokens)
└── cached_dataset.pkl               STALE — built from the 68K leaky dataset; do not use
```

**Key invariant**: `graphs/`, `tokens/`, and `multilabel_index_deduped.csv` must all contain exactly the same set of MD5 stems. If a stem appears in only two of the three, `DualPathDataset` will exclude it from training (via the three-way intersection at init time).

---

## 3. Hash Identity and Pairing Key

**Source**: `ml/src/utils/hash_utils.py`

The MD5 stem is the **pairing key** that links a graph file to its corresponding token file and label row.

```python
def get_contract_hash(contract_path: Union[str, Path]) -> str:
    """Hash the full relative path string (not file content)."""
    path_string = str(contract_path)
    hash_object = hashlib.md5(path_string.encode('utf-8'))
    return hash_object.hexdigest()
```

**Important**: `get_contract_hash` hashes the **file path**, not the file content. A separate function `get_contract_hash_from_content` exists for content-based hashing. The deduplication step (Section 14) used content-based hashing to find duplicate contracts; the on-disk `.pt` filenames use path-based hashing.

**Deduplication note**: The BCCC dataset stores the same `.sol` file under multiple category directories, giving the same source different paths and therefore different path-MD5s. The deduplication step identified duplicates by comparing file content and collapsed them to one canonical row. The `tokens_orphaned/` directory holds the 24,148 token files whose path-MD5s no longer have corresponding graph files after deduplication.

---

## 4. Graph Format

**Written by**: `ml/src/preprocessing/graph_extractor.py`  
**Loaded by**: `DualPathDataset.__getitem__` with `weights_only=True`

Each file is a serialized PyG `Data` object.

### 4.1 Tensor Attributes

| Attribute | Shape | dtype | Description |
|---|---|---|---|
| `graph.x` | `[N, 12]` | float32 | Node feature matrix. N = number of nodes. 12 = `NODE_FEATURE_DIM`. |
| `graph.edge_index` | `[2, E]` | int64 | COO-format edge list. Row 0 = source, row 1 = destination. |
| `graph.edge_attr` | `[E]` | int64 | Edge type IDs, values in `[0, 6]` on disk. Shape must be 1-D. |
| `graph.num_nodes` | int | — | Total node count. |

**Critical shape constraint**: `graph.edge_attr` must be shape `[E]` (1-D), not `[E, 1]`. `nn.Embedding` in `GNNEncoder` crashes on 2-D input. `DualPathDataset.__getitem__` applies `squeeze(-1)` as a safety fix for any pre-refactor files that stored `[E, 1]`, but all freshly extracted graphs have the correct `[E]` shape.

### 4.2 Node Feature Layout (v3 schema, 12 dims)

| Index | Name | Range | Notes |
|---|---|---|---|
| 0 | `type_id` | `float(0–12) / 12.0` | Normalised node type ID. Raw 0–12 dominates dot product without normalisation. |
| 1 | `visibility` | `{0, 1, 2}` | VISIBILITY_MAP ordinal: public/external=0, internal=1, private=2 |
| 2 | `pure` | `{0.0, 1.0}` | `Function.pure` flag |
| 3 | `view` | `{0.0, 1.0}` | `Function.view` flag |
| 4 | `payable` | `{0.0, 1.0}` | `Function.payable` flag |
| 5 | `complexity` | float ≥ 0 | CFG block count (`len(func.nodes)`) |
| 6 | `loc` | float ≥ 0 | Lines of code from `source_mapping.lines` |
| 7 | `return_ignored` | `{0.0, 1.0, -1.0}` | 0.0=captured, 1.0=discarded, -1.0=IR unavailable |
| 8 | `call_target_typed` | `{0.0, 1.0, -1.0}` | 0.0=raw address, 1.0=typed interface, -1.0=source unavailable |
| 9 | `in_unchecked` | `{0.0, 1.0}` | 1.0 if inside `unchecked{}` block |
| 10 | `has_loop` | `{0.0, 1.0}` | 1.0 if function contains a loop |
| 11 | `external_call_count` | `[0.0, 1.0]` | `log1p(n) / log1p(20)`, clamped to `[0, 1]` |

Non-Function nodes receive `0.0` for features `[2:]` except `call_target_typed[8]` which defaults to `1.0` (safe default for non-function nodes). CFG_NODE `in_unchecked[9]` is always `0.0` and never inherited from the parent function.

### 4.3 Node Type Vocabulary (13 types)

| ID | Name | Category |
|---|---|---|
| 0 | `STATE_VAR` | Declaration |
| 1 | `FUNCTION` | Declaration |
| 2 | `MODIFIER` | Declaration |
| 3 | `EVENT` | Declaration |
| 4 | `FALLBACK` | Declaration |
| 5 | `RECEIVE` | Declaration |
| 6 | `CONSTRUCTOR` | Declaration |
| 7 | `CONTRACT` | Declaration |
| 8 | `CFG_NODE_CALL` | CFG (v2+) |
| 9 | `CFG_NODE_WRITE` | CFG (v2+) |
| 10 | `CFG_NODE_READ` | CFG (v2+) |
| 11 | `CFG_NODE_CHECK` | CFG (v2+) |
| 12 | `CFG_NODE_OTHER` | CFG (v2+) |

For a single IR node spanning multiple operations, the CFG type is assigned by priority: CALL (8) > WRITE (9) > READ (10) > CHECK (11) > OTHER (12).

### 4.4 Edge Type Vocabulary (8 types)

| ID | Name | Direction | Stored on disk | Notes |
|---|---|---|---|---|
| 0 | `CALLS` | function → called function | Yes | |
| 1 | `READS` | function → state variable | Yes | |
| 2 | `WRITES` | function → state variable | Yes | |
| 3 | `EMITS` | function → event | Yes | |
| 4 | `INHERITS` | contract → parent contract | Yes | |
| 5 | `CONTAINS` | function → CFG_NODE children | Yes | Added in v2 |
| 6 | `CONTROL_FLOW` | CFG_NODE → successor CFG_NODE | Yes | Added in v2, directed |
| 7 | `REVERSE_CONTAINS` | CFG_NODE → parent function | **No** | Runtime-only; generated in GNNEncoder Phase 3 by reversing CONTAINS(5) edges. Fixes v5.0 limitation L2 (shared embedding for both CONTAINS directions). |

Disk files contain edge type IDs `[0, 6]` only. ID 7 is generated at forward-pass time inside `GNNEncoder` and never written to `.pt` files. Incrementing `NUM_EDGE_TYPES` from 7 to 8 adds one row to the `nn.Embedding` table but requires no graph re-extraction.

### 4.5 Schema Version

`FEATURE_SCHEMA_VERSION = "v3"` is stored in the feature schema module and used as a cache-key suffix for inference. Bumping this string invalidates all cached graph/token pairs built under a previous schema.

**Schema history**:
- `v1` — 8 features, 5 edge types, 8 node types
- `v2` — 12 features, 7 edge types, 13 node types (CFG subtypes added)
- `v3` — same features/nodes as v2; adds `REVERSE_CONTAINS(7)` as runtime-only edge type; no disk change

---

## 5. Token Format

**Written by**: `ml/src/data_extraction/tokenizer.py`  
**Loaded by**: `DualPathDataset.__getitem__` with `weights_only=True`

Each file is a serialized Python dict:

```python
{
    "input_ids":      torch.Tensor  # shape [512], dtype int64
    "attention_mask": torch.Tensor  # shape [512], dtype int64 — 1=real token, 0=PAD
}
```

**Tokenizer**: `microsoft/codebert-base`, `max_length=512`, `truncation=True`, `padding="max_length"`.

**Environment requirement**: `TRANSFORMERS_OFFLINE=1` must be set at the shell level before importing the tokenizer. Failing to set this causes HuggingFace to attempt a remote model check on every import.

**Shape validation**: `DualPathDataset.__getitem__` raises `ValueError` if either tensor is not exactly shape `[512]`. Tokens truncate at 512 subword tokens; contracts longer than this lose their tail.

---

## 6. Label Index: `multilabel_index_deduped.csv`

**Path**: `ml/data/processed/multilabel_index_deduped.csv`  
**Shape**: 44,470 rows × 11 columns

```
md5_stem, CallToUnknown, DenialOfService, ExternalBug, GasException,
IntegerUO, MishandledException, Reentrancy, Timestamp,
TransactionOrderDependence, UnusedReturn
```

All label columns are `int` `{0, 1}`. Multi-label: a single contract can have multiple classes set to 1. When loaded by `DualPathDataset`, they become `float32 [10]` tensors.

### Class Index Mapping

| Index | Class Name | Full-dataset count | Notes |
|---|---|---|---|
| 0 | `CallToUnknown` | 3,610 | |
| 1 | `DenialOfService` | 377 | **Severely data-starved**; highest pos_weight |
| 2 | `ExternalBug` | 3,404 | |
| 3 | `GasException` | 5,597 | |
| 4 | `IntegerUO` | 15,529 | **Dominant class** |
| 5 | `MishandledException` | 4,709 | |
| 6 | `Reentrancy` | 5,025 | Includes 25 CEI-vulnerable augmentation contracts |
| 7 | `Timestamp` | 2,191 | |
| 8 | `TransactionOrderDependence` | 3,391 | |
| 9 | `UnusedReturn` | 3,037 | |

The class index ordering is alphabetical and stable. Changing the column order in the CSV would corrupt all trained checkpoints.

---

## 7. DualPathDataset

**Source**: `ml/src/datasets/dual_path_dataset.py`  
**Class**: `DualPathDataset(Dataset)`

### 7.1 Constructor Signature

```python
DualPathDataset(
    graphs_dir:  str,
    tokens_dir:  str,
    indices:     Optional[List[int]] = None,    # split indices into paired_hashes
    validate:    bool                = True,    # load sample[0] at init
    label_csv:   Optional[Path]      = None,    # multi-label mode if provided
    cache_path:  Optional[Path]      = None,    # RAM cache pkl; explicit opt-in
)
```

### 7.2 Initialization Sequence

1. If `label_csv` is provided, load the CSV and build `_label_map: Dict[str, Tensor[10]]`.
2. Glob `graphs_dir/*.pt` and `tokens_dir/*.pt` to get `graph_hashes` and `token_hashes`.
3. Compute `paired_hashes = graph_hashes ∩ token_hashes`. Log unpaired counts.
4. Sort `paired_hashes` for deterministic indexing across runs.
5. If `indices` is provided, filter `paired_hashes` to the indexed subset (enforces train/val/test split).
6. If `cache_path` is provided and the file exists, load the pickle dict into `self.cached_data`. Performs a spot-check: verifies the dict type, checks the first MD5 key is present, and validates the first `(graph, tokens)` entry has `graph.x` and `tokens["input_ids"]`.
7. If `validate=True`, calls `self[0]` to catch file-format issues before training begins.

**Unpaired token handling**: If token files exist with no matching graph (e.g., the 24K orphaned tokens are accidentally placed in `tokens/`), this is logged at `DEBUG` level only (downgraded from `WARNING` in Phase 2-B4). Under normal operation these files live in `tokens_orphaned/`.

### 7.3 `__getitem__` Return Value

```python
(graph: PyG Data, tokens: Dict[str, Tensor], label: Tensor[10] float32)
```

**Load priority**:
1. If `cached_data` is loaded and `hash_id` is in `cached_data`: read `(graph, tokens)` from the in-memory dict.
2. Otherwise (cache miss, or no cache): read from disk using `torch.load(..., weights_only=True)`.

**Edge attr fix**: If `graph.edge_attr.ndim > 1`, applies `squeeze(-1)` to collapse `[E, 1]` → `[E]`. This is a no-op on correctly-shaped tensors.

**Label extraction (multi-label mode)**: Looks up `hash_id` in `_label_map` and returns the pre-built `float32 [10]` tensor.

**Label extraction (binary mode)**: Returns `graph.y` as a `[1]` int64 tensor.

### 7.4 Validation Checks per Sample

| Check | Condition | Error |
|---|---|---|
| `input_ids` shape | Must be `[512]` | `ValueError` |
| `attention_mask` shape | Must be `[512]` | `ValueError` |
| Hash in label map | Hash must appear in `_label_map` | `KeyError` |
| `graph.x` present | Cache spot-check | `RuntimeError` |

---

## 8. Collation

**Function**: `dual_path_collate_fn` (module-level for DataLoader multiprocessing compatibility)

```python
dual_path_collate_fn(batch) -> (
    batched_graphs:  PyG Batch,
    batched_tokens:  Dict[str, Tensor[B, 512]],
    batched_labels:  Tensor[B, 10] float32
)
```

**Graph batching**: `Batch.from_data_list(graphs, exclude_keys=_EXCLUDE)`. The `exclude_keys` parameter strips stale metadata attributes (`contract_hash`, `contract_path`, `contract_name`, `node_metadata`, `num_edges`, `num_nodes`, `y`) that exist on the ~280 v5.0-era `.pt` files but not on freshly extracted ones. Without this, `Batch.from_data_list` raises when mixing old and new graphs.

```python
_EXCLUDE = ["contract_hash", "contract_path", "contract_name",
            "node_metadata", "num_edges", "num_nodes", "y"]
```

**Token stacking**: `torch.stack` over `[B]` lists of `[512]` tensors → `[B, 512]`.

**Label batching**: Multi-label produces `[B, 10]` float32. Binary produces `[B]` int64 (squeezes the `[B, 1]` stack).

The `batch` tensor added by `Batch.from_data_list` maps each node back to its sample index in `[0, B)`. This is used by `GNNEncoder` for pooling.

---

## 9. Split Construction

**Split files**: `ml/data/splits/deduped/*.npy`  
**Method**: Stratified multi-label split via scikit-learn, `random_state=42`

| Split | File | Size | % |
|---|---|---|---|
| train | `train_indices.npy` | 31,142 | 70% |
| val | `val_indices.npy` | 6,661 | 15% |
| test | `test_indices.npy` | 6,667 | 15% |

Indices are integer positions into the sorted `paired_hashes` list produced by `DualPathDataset.__init__`. The val and test splits are **pure organic data** — augmentation rows are injected only into train (see Section 10).

**No cross-split content leakage**: Verified by MD5 intersection check. All 10 classes are represented in both val and test.

**Split loading in trainer**:
```python
train_idx = np.load("ml/data/splits/deduped/train_indices.npy")
val_idx   = np.load("ml/data/splits/deduped/val_indices.npy")

train_ds = DualPathDataset(
    graphs_dir="ml/data/graphs",
    tokens_dir="ml/data/tokens",
    label_csv=Path("ml/data/processed/multilabel_index_deduped.csv"),
    indices=train_idx.tolist(),
    cache_path=Path("ml/data/cached_dataset_deduped.pkl"),
)
```

---

## 10. CEI Augmentation

**Source contracts**: `ml/data/augmented/` (50 files)  
**Generator script**: `ml/scripts/generate_cei_pairs.py`  
**Injection script**: `ml/scripts/inject_augmented.py`

### 10.1 Augmentation Set

25 structurally matched pairs:

| Pattern | Count | Labels |
|---|---|---|
| `cei_vuln_*.sol` | 25 | `Reentrancy=1`, all other classes=0 |
| `cei_safe_*.sol` | 25 | all classes=0 |

**Structural pairing**: Each vulnerable file has a matching safe file with identical contract structure but with the call/state-write ordering reversed. Vulnerable files execute the external call **before** the state write (classic reentrancy). Safe files execute the state write **before** the external call (CEI pattern).

All 50 files have corresponding `.pt` files in `graphs/` and `tokens/`.

### 10.2 `inject_augmented.py` Labeling Rules

```python
filename pattern → label column set to 1
────────────────────────────────────────
cei_vuln_*.sol   → Reentrancy
cei_safe_*.sol   → (none — all zeros)
dos_*.sol        → DenialOfService
```

This is extensible: new patterns can be added to the `_label_row()` function.

### 10.3 Injection Procedure (idempotent)

1. Scan `ml/data/augmented/*.sol`.
2. For each file: compute path-MD5. If already in `multilabel_index_deduped.csv`, skip.
3. Extract graph via `graph_extractor.py` → write to `ml/data/graphs/<md5>.pt`.
4. Tokenize via `tokenizer.py` → write to `ml/data/tokens/<md5>.pt`.
5. Append new row to `multilabel_index_deduped.csv`.
6. Append new row indices to `train_indices.npy` (train-only; val/test are untouched).

**Post-injection**: The RAM cache must be rebuilt (see Section 11). The CSV row count and train split size both increase.

**Dry run**: `python ml/scripts/inject_augmented.py --dry-run` prints what would be done without writing any files.

---

## 11. RAM Cache

**File**: `ml/data/cached_dataset_deduped.pkl`  
**Format**: `dict[str, Tuple[PyG Data, Dict[str, Tensor]]]` — maps `md5_stem → (graph, tokens)`  
**Size**: ~1.4 GB  
**Entries**: 44,470  
**Builder**: `ml/scripts/create_cache.py`

### 11.1 Purpose

Without the cache, each training epoch reads 44K pairs of `.pt` files from disk. On the RTX-3070 workstation, the cache cuts per-epoch I/O time by ~30%.

### 11.2 Build Command

```bash
source ml/.venv/bin/activate
PYTHONPATH=/home/motafeq/projects/sentinel \
python ml/scripts/create_cache.py \
    --graphs-dir ml/data/graphs \
    --tokens-dir ml/data/tokens \
    --label-csv  ml/data/processed/multilabel_index_deduped.csv \
    --output     ml/data/cached_dataset_deduped.pkl \
    --workers    8
```

The script intersects `label_csv` stems ∩ graph stems ∩ token stems to determine which pairs to cache, skipping orphans. Loading is parallelized across `--workers` threads.

### 11.3 When to Rebuild

The cache must be rebuilt whenever:

| Trigger | Reason |
|---|---|
| New contracts injected via `inject_augmented.py` | New MD5s not in current cache |
| Graph `.pt` files re-extracted (schema change) | Cached graph objects are stale |
| Token `.pt` files regenerated | Cached token tensors are stale |

`DualPathDataset` performs a spot-check at init: it verifies the first `paired_hashes[0]` key exists in `cached_data`. If the cache is stale (e.g., a new graph was injected but cache was not rebuilt), the first new MD5 will be a cache miss, and the loader falls back to disk for that sample without raising an error.

### 11.4 Cache Integrity Check (at dataset init)

```python
# Pseudo-code of DualPathDataset cache validation
assert isinstance(cached_data, dict)          # type check
assert paired_hashes[0] in cached_data         # spot-check key presence
g, t = cached_data[paired_hashes[0]]
assert hasattr(g, "x")                         # graph has features
assert "input_ids" in t                        # tokens present
```

---

## 12. Positive-Weight Computation

**Location**: computed inline in `ml/src/training/trainer.py` at training launch  
**Function**: `compute_pos_weight()`  
**Not saved to disk** — recomputed each run

```python
# Formula (sqrt-scaled to avoid extreme values for rare classes)
pos_weight[c] = sqrt(neg_count[c] / pos_count[c])
```

Computed from the **train split only** (not the full dataset). Class counts change when augmentation adds new positive examples.

| Class | Approximate pos_weight (pre-augmentation) |
|---|---|
| IntegerUO | ~1.4 (dominant class, ~15.5K positives) |
| Reentrancy | ~2.6 |
| DenialOfService | ~12.7 (377 positives, severely data-starved) |
| Timestamp | ~3.7 |

`DenialOfService` has the highest pos_weight. Without augmentation or oversampling, the model tends to predict all-zero for this class.

---

## 13. Graph Extraction Pipeline

**Source**: `ml/src/preprocessing/graph_extractor.py`  
**Schema constants**: `ml/src/preprocessing/graph_schema.py` (single source of truth)

### 13.1 Tool Requirements

- **Slither** ≥ 0.9.3 (required for `NodeType.STARTUNCHECKED` support for `in_unchecked` feature)
- **solc-select**: manages compiler versions; versions 0.4.0–0.8.31 all installed
- `graph_schema.py` asserts the Slither version at import time and raises immediately if too old

### 13.2 Per-Contract Extraction Steps

1. Detect Solidity version from `pragma solidity` statement.
2. Select matching `solc` binary via `solc-select`.
3. Run Slither on the contract.
4. For each Slither `Contract`: build declaration nodes (CONTRACT, FUNCTION, STATE_VAR, etc.).
5. For each `Function`: build CFG nodes (CFG_NODE_CALL, CFG_NODE_WRITE, etc.) and CONTAINS/CONTROL_FLOW edges.
6. Build cross-declaration edges: CALLS, READS, WRITES, EMITS, INHERITS.
7. Normalise `type_id`: `x[0] = float(type_id) / 12.0`.
8. Assemble `PyG Data(x=..., edge_index=..., edge_attr=...)` with `edge_attr` shape `[E]` int64.
9. Write to `ml/data/graphs/<md5>.pt`.

### 13.3 Output Graph Validation

Before training, `validate_graph_dataset.py` checks all `.pt` files for:
- `graph.x.shape[1] == 12` (correct `NODE_FEATURE_DIM`)
- `graph.edge_attr.ndim == 1` (1-D constraint for `nn.Embedding`)
- `graph.edge_attr.max() <= 6` (no REVERSE_CONTAINS on disk)

### 13.4 Uncompilable Contracts

~280 contracts (~0.6%) in the 44,470-file set are stale v5.0 graphs for genuinely uncompilable 0.4.x contracts with 0.5 syntax. These produce syntactically valid but possibly reduced graphs. The ghost graph gate (Section 15) is the downstream check.

---

## 14. Deduplication History

### 14.1 Problem

The original dataset (`multilabel_index.csv`, 68,523 rows) used path-based MD5s. The BCCC-SCsVul-2024 dataset stores the same `.sol` source file under multiple category directories (e.g., a contract appears under both `Reentrancy/` and `IntegerUO/`). Path-based hashing assigned each copy a different MD5, creating separate rows with different labels for identical source code. When splits were constructed from these rows, the same contract content appeared in both train and test, inflating validation metrics.

**Leakage extent**: 7,630 content groups (34.9%) spanned multiple splits.

### 14.2 Resolution

1. Computed content-based MD5 (`get_contract_hash_from_content`) for all 68K source files.
2. For files sharing the same content-MD5, merged their label rows with a logical OR across all 10 classes.
3. Deduplicated to **44,470** unique content-identified contracts.
4. Rebuilt splits from scratch (stratified, `random_state=42`, 70/15/15).
5. Verified zero MD5 overlap between train/val/test.

The ~24,148 token files with no corresponding graph after deduplication were moved to `ml/data/tokens_orphaned/`. They are not loaded during training.

### 14.3 Impact on Metrics

All F1 metrics from runs before 2026-05-14 on the original 68K dataset are **invalid** due to content leakage. The first clean baseline will come from the v5.2 full run on the deduped dataset.

---

## 15. Ghost Graphs

**Definition**: A graph where no FUNCTION/FALLBACK/RECEIVE/CONSTRUCTOR/MODIFIER nodes survived extraction (interface-only contracts, pure abstract contracts, or uncompilable contracts with partial Slither output).

**Count**: 66 graphs (0.1% of 44,470). Gate passed.

**Handling in `GNNEncoder`**: The Three-Eye Classifier's GNN eye pools only over function-level nodes. For ghost graphs with no such nodes, the pooling falls back to all-node pool. This prevents NaN from empty pooling but produces low-signal GNN embeddings for these contracts. The 0.1% rate is accepted as negligible.

---

## 16. Serialization Safety

### 16.1 `weights_only` Policy

| File type | `weights_only` value | Reason |
|---|---|---|
| Graph `.pt` files | `True` | Safe: only PyG `Data` tensors and metadata |
| Token `.pt` files | `True` | Safe: only plain Python dict with Tensors |
| Checkpoint `.pt` files | `False` | Required: LoRA state dict contains `peft` module objects that cannot be deserialized with the safe pickle engine |

### 16.2 Safe Globals Allowlist

Registered at module import time in `DualPathDataset` and `create_cache.py`:

```python
torch.serialization.add_safe_globals([
    Data,           # torch_geometric.data.Data
    DataEdgeAttr,   # torch_geometric.data.data.DataEdgeAttr
    DataTensorAttr, # torch_geometric.data.data.DataTensorAttr
    GlobalStorage,  # torch_geometric.data.storage.GlobalStorage
])
```

These classes appear in `.pt` graph files saved by PyG. Without registering them, `weights_only=True` raises `UnpicklingError`. If a future PyG release adds additional wrapper classes that cause `UnpicklingError`, add them to this list rather than reverting to `weights_only=False`.

---

## 17. Data Pipeline End-to-End Flow

```
Source .sol files
      │
      ▼
[graph_extractor.py]               [tokenizer.py]
  Slither + solc-select               CodeBERT tokenizer
  CFG + AST extraction                max_length=512, truncation
  NODE_FEATURE_DIM=12                 padding="max_length"
      │                                     │
      ▼                                     ▼
ml/data/graphs/<md5>.pt            ml/data/tokens/<md5>.pt
  PyG Data (x, edge_index,           dict {input_ids [512],
  edge_attr [E] int64)                attention_mask [512]}
      │                                     │
      └─────────────────┬───────────────────┘
                        │
                        ▼
              [create_cache.py]
              Three-way intersection:
              graphs ∩ tokens ∩ label_csv
                        │
                        ▼
            cached_dataset_deduped.pkl
            dict md5_stem → (graph, tokens)
                        │
                        ▼
           [DualPathDataset.__init__]
           paired_hashes = sorted(graphs ∩ tokens)
           filtered by split indices
                        │
              ┌─────────┼─────────┐
              ▼         ▼         ▼
           train      val      test
          (31,142)   (6,661)  (6,667)
              │
              ▼
     [dual_path_collate_fn]
     Batch.from_data_list(graphs)  →  PyG Batch
     torch.stack(token tensors)    →  [B, 512]
     torch.stack(label tensors)    →  [B, 10] float32
              │
              ▼
     DataLoader → trainer
     GNNEncoder ← batched_graphs (Batch + edge_attr [E_total] int64)
     TransformerEncoder ← batched_tokens {input_ids [B,512], ...}
     CrossAttentionFusion → [B, 128]
     ThreeEyeClassifier → logits [B, 10]
     sigmoid(logits) → per-class probs
```

### Augmentation Injection Path

```
ml/data/augmented/*.sol
      │
      ▼
[inject_augmented.py]
  extract graph → ml/data/graphs/<md5>.pt
  extract tokens → ml/data/tokens/<md5>.pt
  append row → multilabel_index_deduped.csv
  append indices → splits/deduped/train_indices.npy
      │
      ▼
[create_cache.py]  ← must re-run after injection
  rebuild cached_dataset_deduped.pkl
```

---

## Appendix: Locked Constants

| Constant | Value | Consequence of change |
|---|---|---|
| `FEATURE_SCHEMA_VERSION` | `"v3"` | Invalidates inference cache |
| `NODE_FEATURE_DIM` | `12` | Rebuild 44K graphs + retrain |
| `NUM_CLASSES` | `10` | Append-only; removing a class corrupts all checkpoints |
| `NUM_EDGE_TYPES` | `8` | Rebuild `nn.Embedding` table row; no re-extraction needed |
| `NUM_NODE_TYPES` | `13` | Rebuild graphs + retrain |
| `fusion_output_dim` | `128` | ZKML proxy depends on this; changing breaks the EZKL circuit |
| `MAX_TOKEN_LENGTH` | `512` | Retrain required; changes all token files |
| `type_id` normalisation | `float(id) / 12.0` | Silent feature distribution shift; retrain required |
| `edge_attr` shape | `[E]` 1-D int64 | `nn.Embedding` crashes on `[E, 1]` |
