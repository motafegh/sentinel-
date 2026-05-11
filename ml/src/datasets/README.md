# datasets — DualPathDataset

Paired graph + token dataset for SENTINEL training and evaluation.

---

## Files

| File | Contents |
|---|---|
| `dual_path_dataset.py` | `DualPathDataset` class + `dual_path_collate_fn` |

---

## DualPathDataset

### What it does

Loads paired `.pt` files from `ml/data/graphs/` and `ml/data/tokens/`.
Each sample is one smart contract represented two ways:
- **Graph** — PyG `Data` object produced by `ASTExtractor` + `GraphBuilder`
- **Tokens** — dict of CodeBERT tensors produced by `tokenizer_v1_production.py`

Pairing is by MD5 hash: `{hash}.pt` in both directories = same contract.
13 unmatched token files exist and are silently dropped during `__init__`.

### Lazy loading

Files are read from disk in `__getitem__`, not in `__init__`.
Memory usage stays flat regardless of dataset size (68,555 samples × 2 files each).

### Instantiation

```python
import numpy as np
from ml.src.datasets.dual_path_dataset import DualPathDataset

val_indices = np.load("ml/data/splits/val_indices.npy")

dataset = DualPathDataset(
    graphs_dir="ml/data/graphs",
    tokens_dir="ml/data/tokens",
    indices=val_indices.tolist(),   # list[int] — positions into sorted hash list
)
# indices=None → uses all 68,555 paired samples
```

`validate=True` (default) loads `dataset[0]` during `__init__` to catch broken `.pt` files early.

### `__getitem__` return value

```python
graph, tokens, label = dataset[i]

# graph: PyG Data object
#   graph.x              [N, 8]   float32 — N node features
#   graph.edge_index     [2, E]   int64   — directed edges (COO)
#   graph.edge_attr      [E, 1]   int64   — edge type (may be None)
#   graph.contract_hash  str             — MD5 hash (metadata)
#   graph.y              [1]      long    — label baked in at preprocessing

# tokens: dict
#   tokens["input_ids"]      [512]  long  — CodeBERT token IDs
#   tokens["attention_mask"] [512]  long  — 1=real token, 0=padding

# label: Tensor[1]  long  — 0 (safe) or 1 (vulnerable)
#   Always shape [1] here; squeezed to scalar in collate_fn
```

**Label source:** `graph.y` (baked in at preprocessing time).
The CSV `contract_labels_correct.csv` is an audit reference only — not read during training.

### Safe-globals allowlist

Graph files contain PyG `Data` objects (custom classes).
`torch.load(..., weights_only=True)` blocks unknown classes by default in PyTorch 2.6+.
`dual_path_dataset.py` adds the required allowlist at module level:

```python
torch.serialization.add_safe_globals([Data, DataEdgeAttr, DataTensorAttr])
```

Token files contain only plain tensors — loaded with `weights_only=True` without the allowlist.

---

## dual_path_collate_fn

Module-level collate function required by `DataLoader`.
Must be at module level (not a lambda or nested function) for multiprocessing compatibility.

### What it does

Merges a list of `(graph, tokens, label)` samples into batch tensors.

```python
# Input: list of B samples from __getitem__

batched_graphs  = Batch.from_data_list(graphs)
# PyG merges variable-size graphs into one disconnected super-graph.
# batched_graphs.batch maps each node → its graph index: [0,0,0,1,1,2,...]

batched_tokens = {
    "input_ids":      torch.stack([...]),   # [B, 512]
    "attention_mask": torch.stack([...]),   # [B, 512]
}

batched_labels = torch.stack(labels).squeeze(1)   # [B, 1] → [B]
# squeeze(1) not bare squeeze() — bare squeeze() would collapse [1,1] → scalar when B=1
```

### Output shapes (what the training loop receives)

| Tensor | Shape | Dtype |
|---|---|---|
| `batched_graphs.x` | `[N_total, 8]` | float32 |
| `batched_graphs.edge_index` | `[2, E_total]` | int64 |
| `batched_graphs.batch` | `[N_total]` | int64 |
| `batched_tokens["input_ids"]` | `[B, 512]` | int64 |
| `batched_tokens["attention_mask"]` | `[B, 512]` | int64 |
| `batched_labels` | `[B]` | int64 |

### Usage with DataLoader

```python
from torch_geometric.loader import DataLoader
from ml.src.datasets.dual_path_dataset import DualPathDataset, dual_path_collate_fn

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,          # True for train, False for val/test
    collate_fn=dual_path_collate_fn,
)

for graphs, tokens, labels in loader:
    # graphs: PyG Batch
    # tokens: dict with "input_ids" [B,512] and "attention_mask" [B,512]
    # labels: [B] long
    ...
```

---

## Split Index Files

Pre-computed in `ml/data/splits/` by `ml/scripts/create_splits.py`.

| File | Samples | Description |
|---|---|---|
| `train_indices.npy` | 47,988 | Positions into the sorted paired-hash list |
| `val_indices.npy` | 10,283 | Same — used for checkpoint selection |
| `test_indices.npy` | 10,284 | Same — held out; never used during training |

Indices are `int64` position arrays (not boolean masks).
Stratified split (seed=42) — class ratio 64.3%/35.7% preserved in all three splits.
