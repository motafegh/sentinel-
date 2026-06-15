# datasets — SentinelDataset

> **Status:** ✅ Current — v9 schema, Stage 7B export-backed, verified 2026-06-14

Paired graph + token dataset for SENTINEL training and evaluation. Reads from v2 export artifacts (sharded .pt files + labels.parquet + manifest.json).

---

## Files

| File | Lines | Contents |
|------|-------|----------|
| `sentinel_dataset.py` | 193 | `SentinelDataset` class — PyTorch Dataset backed by v2 export artifact |
| `collate.py` | 51 | `sentinel_collate_fn` — batches 5-tuple items into training batches |

---

## SentinelDataset

### What it does

Loads paired samples from a v2 export artifact (output of `sentinel-data export`). Each sample is one smart contract represented as a 5-tuple:
- **graph** — PyG `Data` object with v9 schema (12-dim node features, 12 edge types)
- **tokens** — `[4, 512]` GraphCodeBERT token tensors from sliding-window tokenization
- **y** — `float32 Tensor[10]` multi-label vulnerability targets
- **contract_id** — `str` (sha256 of the contract)
- **confidence_tier** — `str | None` ("T0", "T1", "T2", or None for NonVulnerable)

The export artifact stores sharded .pt files (graphs and tokens), a labels.parquet file, and a manifest.json with schema version + artifact hash.

### Three construction gates (hard ValueError on failure)

1. **Format schema version** must be "v1"
2. **Graph schema version** must match `FEATURE_SCHEMA_VERSION`
3. **Artifact hash** must be intact (data-integrity check)

### LRU shard caching

Graph and token shards are LRU-cached (default 4 shards). Set `SENTINEL_SHARD_CACHE_SIZE` env var to override.

### Precomputed num_nodes

`__init__` precomputes `num_nodes_map: dict[str, int]` for every contract in the split. Fast path: reads from `shard_index["num_nodes"]` if present (new exports). Fallback: loads graph shards to count nodes.

### Instantiation

```python
from pathlib import Path
from ml.src.datasets.sentinel_dataset import SentinelDataset

dataset = SentinelDataset(
    split="val",
    export_dir=Path("data/exports/sentinel-v3-smartbugs-2026-06-13"),
)
```

### `__getitem__` return value

```python
graph, tokens, y, contract_id, confidence_tier = dataset[i]

# graph: PyG Data object
#   graph.x              [N, 12]  float32 — node features (v9 schema)
#   graph.edge_index     [2, E]   int64   — directed edges (COO)
#   graph.edge_attr      [E]      int64   — edge type, 1-D, values 0–11
#   graph.contract_hash  str              — MD5 hash

# tokens: dict
#   tokens["input_ids"]      [4, 512]  long  — GraphCodeBERT token IDs, 4 windows
#   tokens["attention_mask"] [4, 512]  long  — 1=real token, 0=padding

# y: Tensor[10]  float  — multi-hot vulnerability label vector
# contract_id: str       — sha256 hash of the contract
# confidence_tier: str | None — "T0", "T1", "T2", or None
```

### Safe-globals allowlist

Graph files contain PyG `Data` objects. The following classes are registered via `torch.serialization.add_safe_globals()` to enable `weights_only=True`:
- `Data`, `DataEdgeAttr`, `DataTensorAttr`, `GlobalStorage`

This allows safe deserialization of PyG objects without disabling pickle security entirely.

---

## sentinel_collate_fn

Module-level collate function required by `DataLoader`. Must be at module level (not a lambda or nested function) for multiprocessing compatibility.

### What it does

Merges a list of `(graph, tokens, y, contract_id, confidence_tier)` samples into batch tensors.

```python
graphs_batch = Batch.from_data_list(graphs)
# PyG merges variable-size graphs into one disconnected super-graph.
# batched_graphs.batch maps each node → its graph index [0,0,0,1,1,...]

tokens_batch = {
    "input_ids":      torch.stack([...]),   # [B, 4, 512]
    "attention_mask": torch.stack([...]),   # [B, 4, 512]
}

y_batch = torch.stack(labels)   # [B, 10]  float
```

### Output shapes (what the training loop receives)

| Tensor | Shape | Dtype |
|--------|-------|-------|
| `graphs_batch.x` | `[N_total, 12]` | float32 |
| `graphs_batch.edge_index` | `[2, E_total]` | int64 |
| `graphs_batch.edge_attr` | `[E_total]` | int64 |
| `graphs_batch.batch` | `[N_total]` | int64 |
| `tokens_batch["input_ids"]` | `[B, 4, 512]` | int64 |
| `tokens_batch["attention_mask"]` | `[B, 4, 512]` | int64 |
| `y_batch` | `[B, 10]` | float32 |

### Usage with DataLoader

```python
from torch.utils.data import DataLoader
from ml.src.datasets.sentinel_dataset import SentinelDataset
from ml.src.datasets.collate import sentinel_collate_fn

dataset = SentinelDataset(split="train", export_dir=export_path)
loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True,
    num_workers=4,
    collate_fn=sentinel_collate_fn,
)

for graphs, tokens, y_batch, contract_ids, tiers in loader:
    # graphs: PyG Batch
    # tokens: {"input_ids": [B,4,512], "attention_mask": [B,4,512]}
    # y_batch: [B, 10] float
    # contract_ids: list[str]
    # tiers: list[str | None]
    ...
```

**WeightedRandomSampler:** trainer uses `use_weighted_sampler="positive"` to give 3× weight to rows with any positive label. Passed as `sampler=` to DataLoader (mutually exclusive with `shuffle=True`).

---

## Export Artifact Structure

```
sentinel-v3-smartbugs-2026-06-13/
├── manifest.json          ← schema_version, graph_schema_version, artifact_hash
├── labels.parquet         ← contract_id, class_0..class_9, confidence_tier
├── graphs/
│   ├── graphs-00000.pt    ← PyG Batch shards
│   ├── graphs-00001.pt
│   └── ...
├── tokens/
│   ├── tokens-00000.pt    ← [N_shard, 4, 512] int64
│   ├── tokens-00001.pt
│   └── ...
└── shard_index.json       ← {contract_id: {shard, pos_in_shard, num_nodes}}
```

---

## Schema Constants

| Constant | Value | Source |
|----------|-------|--------|
| `NODE_FEATURE_DIM` | 12 | `sentinel_data.representation.graph_schema` |
| `NUM_EDGE_TYPES` | 12 | `sentinel_data.representation.graph_schema` |
| `NUM_CLASSES` | 10 | `sentinel_data.representation.graph_schema` |
| `FEATURE_SCHEMA_VERSION` | `"v9"` | `sentinel_data.representation.graph_schema` |
