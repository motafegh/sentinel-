> **Superseded v1 plan:** retained for history. Use [D1 v2](../D1_developer_handbook.md) and the implemented [DATA chapters](../../../handbook/03_data_pipeline.md).

# D1.2a — Data Module Doc

**Doc target:** `docs/handbook/02_data_module.md`
**Estimated time:** 0.75h
**Rule:** Every claim verified against source code.

---

## Source files to read before writing (8 files)

1. `data_module/sentinel_data/representation/graph_schema.py`
   - Extract: CLASS_NAMES (10 names, alphabetical order), NUM_CLASSES, NODE_FEATURE_DIM, NUM_NODE_TYPES, NUM_EDGE_TYPES, FEATURE_SCHEMA_VERSION
   - Verify exact values, not from memory

2. `data_module/data/exports/sentinel-v2-baseline-2026-06-12/manifest.json`
   - Extract: n_contracts, n_contracts_with_reps, n_shards, graph_schema_version, schema_version, label_class_columns, artifact_hash
   - Verify all field names and values

3. `data_module/data/splits/v3/split_manifest.json`
   - Extract: version, seed, strategy, ratios, contract_counts (train/val/test), class_distributions
   - Verify split counts

4. `ml/src/datasets/sentinel_dataset.py:57-193`
   - SentinelDataset.__init__: how it loads export, verifies hash, builds label lookup, filters by split
   - SentinelDataset.__getitem__: return signature (graph, tokens, y, contract_id, confidence_tier)
   - _load_graph_shard, _load_token_shard: sharded .pt loading with LRU cache

5. `ml/src/datasets/collate.py`
   - sentinel_collate_fn: 5-tuple input, batch output (graphs_batch, tokens_batch, y_batch, contract_ids, confidence_tiers)
   - _EXCLUDE_KEYS: which graph attributes Batch.from_data_list skips

6. `ml/src/preprocessing/graph_extractor.py`
   - Inference-time extraction: how .sol source becomes a PyG Data object
   - GraphExtractionConfig: include_edge_attr=True (default), solc_version, multi_contract_policy

7. `ml/src/data_extraction/windowed_tokenizer.py`
   - How source code is tokenized into 4 windows of 512 tokens
   - _PAD_TOKEN_ID, _TRAINING_MAX_WINDOWS

8. `data_module/config.yaml`
   - Export configuration parameters

---

## Sections to write

**1. TL;DR** (4 lines)
```
What: Converts Solidity source → PyG graphs + tokenized windows for ML training
Export: 22,356 contracts, 5 shards, schema v9, 10 vulnerability classes
Key file: data_module/sentinel_data/representation/graph_schema.py
Tests: cd data_module && .venv/bin/python -m pytest (569 passed)
```

**2. The v2 export** (~1 page)
- Directory structure: `manifest.json`, `graphs/` (5 shards), `tokens/` (5 shards), `labels.parquet`, `metadata.parquet`
- Manifest fields (verified from actual file): n_contracts=22356, n_shards=5, graph_schema_version="v9", schema_version="v1"
- Shard format: `graphs-{shard:05d}.pt`, `tokens-{shard:05d}.pt`
- How SentinelDataset loads it:
  - `sentinel_dataset.py:78` — SentinelDatasetExport reads manifest
  - `sentinel_dataset.py:84-90` — Gate 1: format schema version check
  - `sentinel_dataset.py:93-99` — Gate 2: graph schema version check
  - `sentinel_dataset.py:104-108` — Gate 3: artifact hash integrity
  - `sentinel_dataset.py:113-119` — label lookup from parquet
  - `sentinel_dataset.py:124-127` — contract list filtered by split + shard index

**3. Graph schema** (~1 page)
- Constants (verify each from `graph_schema.py`):
  - FEATURE_SCHEMA_VERSION = "v9"
  - NODE_FEATURE_DIM = 12
  - NUM_NODE_TYPES = 14
  - NUM_EDGE_TYPES = 12
  - NUM_CLASSES = 10
- CLASS_NAMES (alphabetical, verify exact order from source):
  - CallToUnknown, DenialOfService, ExternalBug, GasException, IntegerUO, MishandledException, Reentrancy, Timestamp, TransactionOrderDependence, UnusedReturn
- Node types (14): CONTRACT, FUNCTION, MODIFIER, FALLBACK, RECEIVE, CONSTRUCTOR, VARIABLE, etc.
- Edge types (12): CALLS, CONTAINS, HANDLES, etc.
- Node features (12 dims): what each dimension represents

**4. Splits** (~0.5 page)
- From `split_manifest.json` (verify exact numbers):
  - train: 18,596 contracts
  - val: 1,983 contracts
  - test: 1,914 contracts
  - ratios: [0.7, 0.15, 0.15]
  - strategy: stratified per source (dive, solidifi, smartbugs_curated)
- Note: not all contracts have representations (22,356 total, 21,523 with reps)
- SentinelDataset further filters: only contracts in shard_index are usable

**5. How to run a new export** (~0.5 page)
- The export command (find in data_module CLI or scripts)
- What changes when schema bumps (FEATURE_SCHEMA_VERSION)
- The artifact hash: why it matters (tamper detection)

**6. Deep reference**
- → `data_module/README.md`
- → source: `sentinel_dataset.py`, `collate.py`, `graph_schema.py`
- → `ml/src/preprocessing/graph_extractor.py` (inference-time extraction)

---

## Verification checklist
- [ ] CLASS_NAMES order matches `graph_schema.py` exactly (alphabetical)
- [ ] Split counts match `split_manifest.json` exactly
- [ ] SentinelDataset.__getitem__ return signature (5-tuple) matches `collate.py` input
- [ ] NODE_FEATURE_DIM=12, NUM_NODE_TYPES=14, NUM_EDGE_TYPES=12 verified from source
- [ ] Export manifest fields match actual `manifest.json` content
- [ ] Test command `cd data_module && .venv/bin/python -m pytest` produces 569 passed
