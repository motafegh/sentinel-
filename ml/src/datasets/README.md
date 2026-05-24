# datasets — DualPathDataset

Paired graph + token dataset for SENTINEL training and evaluation.

---

## Files

| File | Contents |
|------|----------|
| `dual_path_dataset.py` | `DualPathDataset` class + `dual_path_collate_fn` |

---

## DualPathDataset

### What it does

Loads paired samples from a pre-built pickle cache (`cached_dataset_v8.pkl`). Each sample is one smart contract represented two ways:
- **Graph** — PyG `Data` object with v8 schema (11-dim node features, 11 edge types)
- **Tokens** — `[4, 512]` GraphCodeBERT token tensors from sliding-window tokenization

The cache stores 41,576 paired (graph, tokens) objects keyed by MD5 hash. Samples with no matching graph or no matching tokens are excluded at cache-build time.

### Label Modes

**Binary mode (label_csv=None, default):**
- Labels come from graph.y — scalar 0/1 long tensor
- Collate produces [B] long
- Used for binary training and inference with old checkpoints

**Multi-label mode (label_csv=Path(...)):**
- Labels come from multilabel_index.csv — float32 tensor [10]
- Each position is 0.0 or 1.0 for one of the 10 vulnerability classes
- Collate produces [B, 10] float32
- Used for Track 3 multi-label retrain

### Lazy loading from cache

The cache is memory-mapped: graph and token data are loaded in `__getitem__`, not all at once. Memory usage stays flat regardless of dataset size.

### Cache validation

When `cache_path` is provided, the dataset performs several validation checks:
- **Schema version validation:** Cache must contain `__schema_version__` key matching current `FEATURE_SCHEMA_VERSION`
- **Random integrity sampling:** 10 random hashes are sampled to verify cache entries exist and have correct structure
- **Type guard:** Validates cache is a dict and entries have required attributes (graph.x, tokens.input_ids)

### Instantiation

```python
import numpy as np
from ml.src.datasets.dual_path_dataset import DualPathDataset

val_indices = np.load("ml/data/splits/deduped/val_indices.npy")

# Multi-label mode (Track 3)
dataset = DualPathDataset(
    graphs_dir="ml/data/graphs",
    tokens_dir="ml/data/tokens_windowed",
    indices=val_indices.tolist(),
    label_csv="ml/data/processed/multilabel_index_cleaned.csv",
    cache_path="ml/data/cached_dataset_v8.pkl",
)

# Binary mode (legacy)
dataset = DualPathDataset(
    graphs_dir="ml/data/graphs",
    tokens_dir="ml/data/tokens_windowed",
    indices=val_indices.tolist(),
    cache_path="ml/data/cached_dataset_v8.pkl",
)
# indices=None → uses all 41,576 cached samples
```

### `__getitem__` return value

```python
graph, tokens, label = dataset[i]

# graph: PyG Data object
#   graph.x              [N, 11]   float32 — node features (v8 schema)
#   graph.edge_index     [2, E]    int64   — directed edges (COO)
#   graph.edge_attr      [E]       int64   — edge type, 1-D, values 0–10
#   graph.contract_hash  str               — MD5 hash

# tokens: dict
#   tokens["input_ids"]      [4, 512]  long  — GraphCodeBERT token IDs, 4 windows
#   tokens["attention_mask"] [4, 512]  long  — 1=real token, 0=padding
#   (Also accepts single-window [512] shape for backward compatibility)

# label: Tensor[10]  float  — multi-hot vulnerability label vector (multi-label mode)
#        Tensor[1]   long   — binary label (binary mode)
```

### Safe-globals allowlist

Graph files contain PyG `Data` objects. The following classes are registered via `torch.serialization.add_safe_globals()` to enable `weights_only=True`:
- `Data`, `DataEdgeAttr`, `DataTensorAttr`, `GlobalStorage`

This allows safe deserialization of PyG objects without disabling pickle security entirely. If future PyG releases add new wrapper classes, they must be added to this allowlist.

---

## dual_path_collate_fn

Module-level collate function required by `DataLoader`. Must be at module level (not a lambda or nested function) for multiprocessing compatibility.

### What it does

Merges a list of `(graph, tokens, label)` samples into batch tensors.

```python
batched_graphs = Batch.from_data_list(graphs)
# PyG merges variable-size graphs into one disconnected super-graph.
# batched_graphs.batch maps each node → its graph index [0,0,0,1,1,...]

batched_tokens = {
    "input_ids":      torch.stack([...]),   # [B, 4, 512]
    "attention_mask": torch.stack([...]),   # [B, 4, 512]
}

batched_labels = torch.stack(labels)   # [B, 10]  float
```

### Output shapes (what the training loop receives)

| Tensor | Shape | Dtype |
|--------|-------|-------|
| `batched_graphs.x` | `[N_total, 11]` | float32 |
| `batched_graphs.edge_index` | `[2, E_total]` | int64 |
| `batched_graphs.edge_attr` | `[E_total]` | int64 |
| `batched_graphs.batch` | `[N_total]` | int64 |
| `batched_tokens["input_ids"]` | `[B, 4, 512]` | int64 |
| `batched_tokens["attention_mask"]` | `[B, 4, 512]` | int64 |
| `batched_labels` | `[B, 10]` | float32 |

### Usage with DataLoader

```python
from torch.utils.data import DataLoader
from ml.src.datasets.dual_path_dataset import DualPathDataset, dual_path_collate_fn

loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True,
    num_workers=4,
    collate_fn=dual_path_collate_fn,
)

for graphs, tokens, labels in loader:
    # graphs: PyG Batch
    # tokens: {"input_ids": [B,4,512], "attention_mask": [B,4,512]}
    # labels: [B, 10] float
    ...
```

**WeightedRandomSampler:** trainer uses `use_weighted_sampler="positive"` to give 3× weight to rows with any positive label. Passed as `sampler=` to DataLoader (mutually exclusive with `shuffle=True`).

---

## Split Index Files

Pre-computed in `ml/data/splits/deduped/` by `ml/scripts/create_splits.py`.

| File | Samples | Description |
|------|---------|-------------|
| `train_indices.npy` | 29,103 | Positions into the sorted cache sample list |
| `val_indices.npy` | 6,236 | Same — used for checkpoint selection and threshold tuning |
| `test_indices.npy` | 6,237 | Same — held out; never used during training |

Indices are `int64` position arrays (not boolean masks). Stratified split (seed=42).

**Important:** load from `ml/data/splits/deduped/val_indices.npy` (numpy binary), not from any `.txt` files. The `.npy` files are the authoritative source.
