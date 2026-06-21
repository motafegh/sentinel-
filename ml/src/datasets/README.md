# ml/src/datasets — SENTINEL Dataset and Collation

PyTorch Dataset backed by v2 export artifacts and its collation function.

---

## Files

| File | Lines | Purpose |
|------|-------|---------|
| `sentinel_dataset.py` | 193 | `SentinelDataset` — loads (graph, tokens, labels) from v2 exports |
| `collate.py` | 51 | `sentinel_collate_fn` — batches SentinelDataset items |
| `__init__.py` | 0 | Empty |

---

## sentinel_dataset.py

### SentinelDataset

PyTorch `Dataset` that reads from a `chunk_export()` output directory.

**Returns 5-tuples per item:**
```python
(graph, tokens, y, contract_id, confidence_tier)
```
- `graph`: PyG Data — `x[n_nodes,12]`, `edge_index[2,E]`, `edge_attr[E]`
- `tokens`: dict — `"input_ids"[4,512]` int64, `"attention_mask"[4,512]` int64
- `y`: float32 Tensor[10] — multi-label targets
- `contract_id`: str (sha256 of the contract)
- `confidence_tier`: str | None ("T0", "T1", "T2", or None for NonVulnerable)

**Constructor validation (3 hard gates):**
1. Format schema version must be `"v1"`
2. Graph schema version must match `FEATURE_SCHEMA_VERSION`
3. Artifact hash must be intact (data-integrity check)

**Shard loading:** LRU-cached (default 4 shards; `SENTINEL_SHARD_CACHE_SIZE` env var).

**Properties:**
- `num_nodes_map`: dict mapping contract_id -> num_nodes (for weighted sampler)

---

## collate.py

### sentinel_collate_fn

Collates a list of SentinelDataset items into a training batch.

**Input:** list of `(graph, tokens, y, contract_id, confidence_tier)` tuples

**Output:** 5-tuple batch:
```python
(graphs_batch, tokens_batch, y_batch, contract_ids, confidence_tiers)
```
- `graphs_batch`: PyG `Batch` — merged graph batch
- `tokens_batch`: dict — `"input_ids"[B,4,512]`, `"attention_mask"[B,4,512]`
- `y_batch`: float32 Tensor `[B,10]`
- `contract_ids`: list[str]
- `confidence_tiers`: list[str | None]

Uses `Batch.from_data_list()` with `_EXCLUDE_KEYS` for non-tensor metadata.
