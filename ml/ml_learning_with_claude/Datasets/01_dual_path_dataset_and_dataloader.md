# Datasets — Chunk 1: DualPathDataset & DataLoader

> **File:** `ml/src/datasets/dual_path_dataset.py`
> **What you'll learn:** PyTorch's `Dataset` and `DataLoader` API, the collate function pattern, content-addressed pairing, multi-label vs binary mode, RAM caching with schema validation, and production-grade integrity checks.
> **Time:** ~25 minutes
> **Interview relevance:** ML (training pipelines), MLOps (data loading, caching, validation)

---

## 1. The PyTorch Dataset Contract

Every dataset in PyTorch must implement two methods:

```python
class MyDataset(Dataset):
    def __len__(self) -> int:
        return number_of_samples
    
    def __getitem__(self, idx: int) -> Any:
        return one_sample
```

`DataLoader` calls `__len__` to know how many batches to create, and calls `__getitem__` repeatedly (possibly in parallel worker processes) to load individual samples. These are then collated into a batch by `collate_fn`.

**`DualPathDataset`** loads pairs of (graph, tokens) — two modalities for the same contract. Each pair is a sample. The `__getitem__` returns a 3-tuple: `(graph, tokens, label)`.

---

## 2. Content-Addressed Pairing

```python
graph_files = list(self.graphs_dir.glob("*.pt"))
token_files = list(self.tokens_dir.glob("*.pt"))

graph_hashes  = {f.stem for f in graph_files}    # set of MD5 stems
token_hashes  = {f.stem for f in token_files}    # set of MD5 stems
paired_hashes = graph_hashes & token_hashes       # SET INTERSECTION
```

**Set intersection (`&`)** finds hashes present in **both** directories. This is the pairing mechanism: a contract is only included if both its graph `.pt` and its token `.pt` exist.

Why might they not both exist?
- The graph extractor failed for a contract (compilation error) but the tokenizer succeeded
- The tokenizer failed but the graph extraction succeeded
- New contracts were added after one pipeline ran but not the other

Unpaired files are logged and skipped gracefully.

**Sort for determinism:**
```python
self.paired_hashes = sorted(paired_hashes)
```

Sets are unordered in Python. Without sorting, the ordering of samples could differ between runs, making index-based splits (train/val/test) non-reproducible. Sorted list = same order every time.

---

## 3. Binary vs Multi-label Mode

```python
class DualPathDataset(Dataset):
    def __init__(self, ..., label_csv=None):
```

**Binary mode** (`label_csv=None`): labels come from `graph.y` — a scalar long tensor `[0]` or `[1]`. Used for older binary vulnerability detection (safe/vulnerable).

**Multi-label mode** (`label_csv=Path("multilabel_index.csv")`): labels come from a CSV:
```
md5_stem,CallToUnknown,DenialOfService,ExternalBug,...,UnusedReturn
abc123,0,1,0,0,1,0,0,0,0,0
def456,1,0,0,0,0,0,1,0,0,0
```

```python
df = pd.read_csv(label_csv)
class_cols = [c for c in df.columns if c != "md5_stem"]
label_matrix = torch.tensor(df[class_cols].values.astype("float32"), dtype=torch.float32)
self._label_map = {stem: label_matrix[i] for i, stem in enumerate(stems)}
```

`_label_map` is a dict: `{md5_stem → float32 tensor [10]}`. Looking up `hash_id` in `__getitem__` gives the label vector for that contract.

**Why `float32` for multi-label?** Binary cross-entropy loss (used for multi-label) expects float targets. `long` (integer) is for cross-entropy (single-class). This is an easy-to-miss distinction in PyTorch.

---

## 4. Train/Val/Test Splits via Indices

```python
def __init__(self, ..., indices=None):
    ...
    if indices is not None:
        self.paired_hashes = [self.paired_hashes[i] for i in indices]
```

The dataset doesn't hardcode splits. Instead, it takes an `indices` list — positions into the full sorted paired-hash list. This way:
- `DualPathDataset(..., indices=train_indices)` → training set
- `DualPathDataset(..., indices=val_indices)` → validation set
- `DualPathDataset(..., indices=test_indices)` → test set

All three share the same source directories. The splits are determined by which indices you pass in, loaded from `.npy` files:
```python
train_indices = np.load("ml/data/splits/deduped/train.npy")
```

**Why index-based splits instead of separate directories?**
- No data duplication: all `.pt` files live in one place
- Splits can be changed without moving files
- Reproducible: same `.npy` files = same splits always

---

## 5. Lazy Loading — One File Per `__getitem__` Call

```python
def __getitem__(self, idx):
    hash_id = self.paired_hashes[idx]
    
    if self.cached_data is not None and hash_id in self.cached_data:
        graph, tokens = self.cached_data[hash_id]    # O(1) dict lookup
    else:
        graph_path = self.graphs_dir / f"{hash_id}.pt"
        token_path = self.tokens_dir / f"{hash_id}.pt"
        graph  = torch.load(graph_path, weights_only=True)
        tokens = torch.load(token_path, weights_only=True)
```

**Lazy loading**: files are loaded on demand during training, not all at once during `__init__`. This keeps memory usage constant regardless of dataset size. Loading all 41,576 graphs at startup would require several GB of RAM before training even begins.

**`weights_only=True`**: a security flag in PyTorch 2.6+. It prevents arbitrary Python code execution during deserialization (pickle can execute code — a security risk if loading untrusted files). The `add_safe_globals()` call at module level whitelists PyG's internal classes:

```python
torch.serialization.add_safe_globals([Data, DataEdgeAttr, DataTensorAttr, GlobalStorage])
```

Without this, `weights_only=True` would fail for PyG graph objects because they use these internal classes.

---

## 6. The RAM Cache — 2.2 GB In Memory

```python
if cache_path is not None:
    with open(cache_path, "rb") as f:
        self.cached_data = pickle.load(f)
    # self.cached_data: {md5_stem → (graph, tokens), ..., "__schema_version__": "v8"}
```

Loading 41,576 pairs from individual `.pt` files during training causes disk I/O bottlenecks. Each epoch = 41,576 × 2 file reads = ~83K file opens. With an HDD or slow NVMe, this dominates training time.

The solution: load everything into a single pickle file (`cached_dataset_v8.pkl`, 2.2 GB) once, then read from memory (O(1) dict lookup) for every subsequent epoch.

**Trade-off:**
- Without cache: constant disk I/O every epoch, works on any machine, no 2.2 GB RAM overhead
- With cache: 2.2 GB RAM required, but each epoch is drastically faster

**The cache is created by `create_cache.py`**:
```python
# create_cache.py (simplified)
cache = {}
for hash_id in paired_hashes:
    graph  = torch.load(f"graphs/{hash_id}.pt", ...)
    tokens = torch.load(f"tokens/{hash_id}.pt", ...)
    cache[hash_id] = (graph, tokens)
cache["__schema_version__"] = FEATURE_SCHEMA_VERSION
with open("cached_dataset_v8.pkl", "wb") as f:
    pickle.dump(cache, f)
```

---

## 7. Cache Integrity Validation

Three levels of cache validation protect against stale/corrupted caches:

### Level 1: Schema Version Check
```python
_cached_schema = self.cached_data.get("__schema_version__")
if _cached_schema != _FEATURE_SCHEMA_VERSION:
    raise RuntimeError(f"Cache schema mismatch: {_cached_schema!r} vs {_FEATURE_SCHEMA_VERSION!r}")
```

If you change the graph feature engineering and bump `FEATURE_SCHEMA_VERSION = "v9"`, but forget to rebuild the cache, this check catches it **at startup** rather than silently training on stale features.

### Level 2: Random Sample Integrity Check (Fix D1/H14)

```python
_sample_hashes = random.sample(self.paired_hashes, min(10, len(self.paired_hashes)))
for _spot in _sample_hashes:
    if _spot not in self.cached_data:
        raise RuntimeError(f"Cache is stale — hash {_spot!r} not found")
    _g, _t = self.cached_data[_spot]
    if not hasattr(_g, "x"):
        raise ValueError("cached graph missing 'x' attribute")
```

**Why random, not just check index 0?** The first hash in sorted order tends to be stable across builds (same contract, same MD5). A spot-check on only index 0 could pass even when 99% of the cache is stale. Random sampling of 10 hashes makes undetected staleness 10× less likely.

### Level 3: Eager Validation
```python
if validate and len(self.paired_hashes) > 0:
    _ = self[0]   # load the first sample
```

Calls `__getitem__` at init time. If the file format is wrong or a file is missing, you get an error **before training starts** rather than after an hour of warmup.

> 🎯 **INTERVIEW FOCUS:** "How do you validate that your dataset is correct before a long training run?" — Schema version checks, random integrity samples, eager validation on first item. Fail fast at startup, not after 12 hours.

---

## 8. The `edge_attr` Shape Fix

```python
# Fix #1: Guard against pre-refactor .pt files with edge_attr [E, 1]
if hasattr(graph, "edge_attr") and graph.edge_attr is not None:
    if graph.edge_attr.ndim > 1:
        graph.edge_attr = graph.edge_attr.squeeze(-1)
```

Old `.pt` files stored `edge_attr` as shape `[E, 1]`. New files use `[E]` (PyG convention). `nn.Embedding` requires 1D input — calling it with `[E, 1]` crashes.

`squeeze(-1)` removes the last dimension if it's size 1: `[E, 1]` → `[E]`. It's a no-op if already `[E]`. This forward-compatibility fix means both old and new files work without re-extraction.

---

## 9. The `dual_path_collate_fn` — Batching Heterogeneous Data

The default PyTorch `collate_fn` works for uniform tensors. But this dataset has three different data types per sample:
1. **Graph**: variable-size PyG `Data` objects (different N and E per contract)
2. **Tokens**: dict of `[W, 512]` tensors (uniform shape)
3. **Labels**: either `[10]` (multi-label) or `[1]` (binary)

```python
def dual_path_collate_fn(batch):
    graphs = [item[0] for item in batch]
    tokens = [item[1] for item in batch]
    labels = [item[2] for item in batch]
    
    # 1. Batch graphs: variable-size graphs → single disconnected mega-graph
    batched_graphs = Batch.from_data_list(graphs, exclude_keys=_EXCLUDE)
    
    # 2. Stack tokens: [B × [W,512]] → [B, W, 512]
    batched_tokens = {
        "input_ids":      torch.stack([t["input_ids"] for t in tokens]),
        "attention_mask": torch.stack([t["attention_mask"] for t in tokens]),
    }
    
    # 3. Stack labels appropriately
    stacked = torch.stack(labels)
    if first_label.dim() == 1 and first_label.shape[0] > 1:
        batched_labels = stacked         # [B, 10] float32 multi-label
    else:
        batched_labels = stacked.squeeze(1)  # [B] long binary
    
    return batched_graphs, batched_tokens, batched_labels
```

### `Batch.from_data_list()` — PyG's Graph Batching

This is the most interesting part. PyG's solution to batching variable-size graphs:

```
Graph A: N=5 nodes, E=8 edges
Graph B: N=3 nodes, E=4 edges
Graph C: N=7 nodes, E=12 edges

Batch:
  x:          [15, 11]  (5+3+7 nodes, 11 features each)
  edge_index: [2, 24]   (8+4+12 edges)
  batch:      [0,0,0,0,0, 1,1,1, 2,2,2,2,2,2,2]
               ← graph A ← graph B ← graph C →
```

The `batch` tensor is the key: it maps each node to its graph in the batch. PyG's message-passing and pooling operations use this tensor to handle the different-size graphs correctly.

**`exclude_keys=_EXCLUDE`:**
```python
_EXCLUDE = ["contract_hash", "contract_path", "contract_name", "node_metadata", "num_edges", "num_nodes", "y"]
```

Old `.pt` files have extra non-tensor fields. `Batch.from_data_list()` tries to batch everything — it would fail trying to combine lists of strings (`contract_path`). `exclude_keys` tells it to skip these fields.

> 🎯 **INTERVIEW FOCUS:** "How do you batch variable-size graphs in PyTorch?" — `Batch.from_data_list()` creates a single disconnected mega-graph. The `batch` index tensor maps each node to its original graph, enabling per-graph operations (pooling, reading out) in the GNN.

---

## 10. The `weights_only=False` Exception

```
Key Invariant: `weights_only` for graph `.pt` — `False`
"PyG 2.7 metadata not safe-tensors serialisable"
```

This invariant is documented in `README.md`. Despite the security improvement of `weights_only=True`, it currently fails for some PyG graph files because PyG 2.7 uses Python objects that haven't been added to the safe globals list. The `add_safe_globals()` call in `dual_path_dataset.py` adds the known safe classes, but if a new PyG version adds more, those would need to be added too.

**The safe path forward**: add new classes to the safe globals list rather than reverting to `weights_only=False`.

---

## 11. The DataLoader Setup (How the Dataset Is Used)

```python
from torch.utils.data import DataLoader, WeightedRandomSampler

train_ds = DualPathDataset(
    graphs_dir="ml/data/graphs",
    tokens_dir="ml/data/tokens_windowed",
    indices=np.load("ml/data/splits/deduped/train.npy"),
    label_csv=Path("ml/data/processed/multilabel_index_cleaned.csv"),
    cache_path=Path("ml/data/cached_dataset_v8.pkl"),
)

train_loader = DataLoader(
    train_ds,
    batch_size=8,
    shuffle=False,                           # WeightedRandomSampler handles ordering
    sampler=WeightedRandomSampler(...),      # 3× weight for vulnerable contracts
    collate_fn=dual_path_collate_fn,
    num_workers=4,                           # parallel data loading
    pin_memory=True,                         # faster CPU→GPU transfer
)
```

**`WeightedRandomSampler`**: the dataset has 59.3% all-zero labels (no vulnerability). Without resampling, the model would see mostly safe contracts and learn to always predict "safe." The sampler gives 3× weight to contracts with at least one vulnerability, increasing their probability of being selected.

**`num_workers=4`**: `__getitem__` runs in 4 parallel worker processes while the GPU runs the forward pass. The `collate_fn` must be a module-level function (not a lambda or nested function) because it's pickled and sent to worker processes.

**`pin_memory=True`**: uses pinned (page-locked) memory for tensors loaded by workers. GPU DMA transfers from pinned memory are faster than from regular memory.

---

## 12. Summary

| Concept | Implementation | Key insight |
|---------|---------------|------------|
| Dataset API | `__len__` + `__getitem__` | DataLoader drives these; implement nothing else |
| Content-addressed pairing | Set intersection on MD5 stems | Both files must exist; no manual CSV mapping needed |
| Binary vs multi-label mode | `label_csv` argument | `float32[10]` vs `long[1]` |
| Lazy loading | Load in `__getitem__`, not `__init__` | Constant memory regardless of dataset size |
| RAM cache | 2.2 GB pickle, schema-versioned | Eliminates per-epoch disk I/O bottleneck |
| Cache validation | Schema check + random sample + eager load | Fail fast before training, not mid-epoch |
| Graph batching | `Batch.from_data_list()` | Disconnected mega-graph + `batch` index tensor |
| Collate function | Module-level, handles mixed types | Required for DataLoader multiprocessing |
| Weighted sampling | 3× weight for vulnerable samples | Corrects class imbalance without oversampling |

---

## Interview Questions

1. **"Explain the PyTorch Dataset/DataLoader pattern."**
   → `Dataset` implements `__len__` (how many samples) and `__getitem__` (load one sample by index). `DataLoader` wraps a `Dataset`, handles batching via `collate_fn`, optional parallel loading via `num_workers`, and sampling via `sampler`.

2. **"How do you handle batching when samples have different sizes (e.g., graphs)?"**
   → For graphs: `Batch.from_data_list()` creates a single disconnected mega-graph where all node/edge tensors are concatenated and a `batch` index tensor maps each node to its original graph. For variable-length sequences: padding + attention masks.

3. **"Your training dataset is 41,576 files. How do you handle train/val/test splits without duplicating data?"**
   → Index-based splits: keep all files in one directory, load `indices` arrays from `.npy` files, pass `indices` to `DualPathDataset`. All three splits share the same source directories; no data is copied or moved.

4. **"How do you detect and handle class imbalance in a multi-label classification problem?"**
   → `WeightedRandomSampler` with higher weight for positive samples. During training, the sampler oversamples vulnerable contracts (59.3% are all-zero labels). Combine with `AsymmetricLoss` (different γ for positive vs negative targets) for additional imbalance handling in the loss function.

---

**Datasets module complete!** ✅

**Next module:** `Models/01_gnn_fundamentals_and_gat.md` — Graph Attention Networks from first principles.
