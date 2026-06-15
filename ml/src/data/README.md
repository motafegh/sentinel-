# ml/src/data — Data Storage Directory

> **Status:** ✅ Current — v9 schema, verified 2026-06-14

Storage location for processed ML data artifacts. This directory and its contents are **not committed to git** (see `.gitignore`).

## Structure

```
ml/data/
├── graphs/                      # PyG graph files — v9 schema, 12-dim
├── tokens_windowed/             # GraphCodeBERT token windows — [4, 512]
├── cached_dataset_v9.pkl        # paired cache — (graph, tokens) tuples
├── processed/
│   ├── multilabel_index.csv             # raw label index
│   ├── multilabel_index_deduped.csv     # after content-hash dedup
│   └── multilabel_index_cleaned.csv     # after structural label cleaning (training target)
└── splits/
    └── deduped/
        ├── train_indices.npy    # 29,103 positions
        ├── val_indices.npy      # 6,236 positions
        └── test_indices.npy     # 6,237 positions
```

## Data Files

### Graph Files (`graphs/*.pt`)

Each file is a PyG `Data` object:

| Field | Shape | Dtype | Notes |
|-------|-------|-------|-------|
| `graph.x` | `[N, 12]` | float32 | Node features, v9 schema |
| `graph.edge_index` | `[2, E]` | int64 | Directed edges, COO format |
| `graph.edge_attr` | `[E]` | int64 | Edge type indices 0–11 |
| `graph.contract_hash` | str | — | MD5 hash, matches token file name |

- **Count:** 41,576 files
- **Naming:** `<md5_hash>.pt`
- **Schema version:** `FEATURE_SCHEMA_VERSION = "v9"`
- **Loading:** `torch.load(path, weights_only=False)` — PyG 2.7 metadata blocks `weights_only=True`

### Token Files (`tokens_windowed/*.pt`)

Each file is a tensor of shape `[4, 512]`:
- 4 sliding windows of 512 GraphCodeBERT tokens
- Stride=256 between windows; code_budget=464 per window when K=48
- **Count:** 44,470 files (includes contracts without matching graphs)
- **Naming:** `<md5_hash>.pt` (matches graph file)

### Cached Dataset (`cached_dataset_v9.pkl`)

Pre-built paired cache:
- **Contents:** 41,576 `(graph, tokens, label)` tuples
- **Schema:** v9 — regenerate if `FEATURE_SCHEMA_VERSION` is bumped
- **Build:** `poetry run python ml/scripts/create_cache.py`

### Split Indices (`splits/deduped/*.npy`)

Stratified train/val/test split (seed=42). Stored as int64 position arrays into the cached dataset's sorted sample list.

| File | Samples |
|------|---------|
| `train_indices.npy` | 29,103 |
| `val_indices.npy` | 6,236 |
| `test_indices.npy` | 6,237 |

Always load from `.npy` files — never from `.txt` equivalents.

## Schema Version

**Current: v9**

| Constant | Value |
|----------|-------|
| `NODE_FEATURE_DIM` | 12 |
| `FEATURE_SCHEMA_VERSION` | `"v9"` |
| `NUM_EDGE_TYPES` | 12 |
| `NUM_CLASSES` | 10 |

Bump `FEATURE_SCHEMA_VERSION` in `graph_schema.py` for any schema change. This invalidates the cache (`create_cache.py` checks the version and refuses stale data).

## Regeneration

```bash
# Full re-extraction + cache rebuild:
poetry run python ml/scripts/reextract_graphs.py
poetry run python ml/scripts/retokenize_windowed.py
poetry run python ml/scripts/build_multilabel_index.py
poetry run python ml/scripts/dedup_multilabel_index.py --relabel-timestamp
poetry run python ml/scripts/inject_augmented.py
poetry run python ml/scripts/label_cleaner.py \
    --graphs-dir ml/data/graphs \
    --label-csv ml/data/processed/multilabel_index_deduped.csv
poetry run python ml/scripts/create_cache.py
# Splits are stable — regenerate only if the dataset composition changes:
# poetry run python ml/scripts/create_splits.py
```

## Storage Requirements

| Location | Size |
|----------|------|
| `graphs/` | ~2–3 GB |
| `tokens_windowed/` | ~1–2 GB |
| `cached_dataset_v9.pkl` | ~2.2 GB |
| `processed/` + `splits/` | < 100 MB |
| **Total** | **~5–8 GB** |
