# 02 — Data Module

## TL;DR

```
What: Converts Solidity source → PyG graphs + tokenized windows for ML training
Export: 22,356 contracts, 5 shards, schema v9, 10 vulnerability classes
Key file: data_module/sentinel_data/representation/graph_schema.py
Tests: cd data_module && .venv/bin/python -m pytest (569 passed, 47 skipped)
```

---

## 1. The v2 export

The data module's output is a **chunked export directory** that the ML module's `SentinelDataset` reads.

**Directory structure** (`data_module/data/exports/sentinel-v2-baseline-2026-06-12/`):

```
sentinel-v2-baseline-2026-06-12/
  manifest.json              ← export metadata (schema, hash, counts)
  labels.parquet             ← per-contract multi-label ground truth (10 classes)
  metadata.parquet           ← per-contract metadata (source, compiler, etc.)
  graphs/
    _shard_index.json        ← contract_id → (shard, position) mapping
    graphs-00000.pt          ← PyG Data objects, sharded
    graphs-00001.pt
    graphs-00002.pt
    graphs-00003.pt
    graphs-00004.pt
  tokens/
    _shard_index.json
    tokens-00000.pt          ← [N, 4, 512] int64 tensors (tokenized windows)
    tokens-00001.pt
    tokens-00002.pt
    tokens-00003.pt
    tokens-00004.pt
```

**Manifest fields** (verified from `manifest.json`):

| Field | Value | Meaning |
|---|---|---|
| `schema_version` | `"v1"` | Export format version (Gate 1 checks this) |
| `graph_schema_version` | `"v9"` | Graph schema version (Gate 2 checks this) |
| `n_contracts` | 22,356 | Total contracts in export |
| `n_contracts_with_reps` | 21,523 | Contracts with valid graph + token representations |
| `n_shards` | 5 | Number of .pt shard files |
| `label_class_columns` | 10 class names | Matches CLASS_NAMES in graph_schema.py |
| `artifact_hash` | `45e2a2d4...` | SHA-256 of all shard files (tamper detection) |

**How SentinelDataset loads it** (`sentinel_dataset.py:57-158`):

1. `:78` — `SentinelDatasetExport(export_dir)` reads `manifest.json`
2. `:84-90` — **Gate 1**: `schema_version` must match expected format version (`"v1"`)
3. `:93-99` — **Gate 2**: `graph_schema_version` must match model's `FEATURE_SCHEMA_VERSION` (`"v9"`)
4. `:104-108` — **Gate 3**: `verify_artifact_hash()` — recomputes SHA-256 of all shards, must match `artifact_hash` in manifest. Detects corruption/tampering.
5. `:113-119` — Builds label lookup from `labels.parquet`: `{contract_id: (y_tensor[10], confidence_tier)}`
6. `:124-127` — Filters contract list by split (train/val/test) and shard index membership

**`__getitem__`** (`sentinel_dataset.py:172-193`):
- Returns 5-tuple: `(graph: Data, tokens: dict, y: Tensor[10], contract_id: str, confidence_tier: str|None)`
- Graph loaded from shard via `_load_graph_shard(path)` with LRU cache
- Tokens: `[4, 512]` int64 → attention_mask computed as `(input_ids != PAD_TOKEN_ID)`

---

## 2. Graph schema

All constants defined in `data_module/sentinel_data/representation/graph_schema.py` (re-exported via `ml/src/preprocessing/graph_schema.py`):

| Constant | Value | Meaning |
|---|---|---|
| `FEATURE_SCHEMA_VERSION` | `"v9"` | Current schema version |
| `NODE_FEATURE_DIM` | 12 | Dimensions per node feature vector |
| `NUM_NODE_TYPES` | 14 | Number of distinct node types |
| `NUM_EDGE_TYPES` | 12 | Number of distinct edge types |
| `NUM_CLASSES` | 10 | Number of vulnerability classes |

**CLASS_NAMES** (alphabetical order — this exact order is the on-chain index):

| Index | Class name |
|---|---|
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

**Node types** (14):

| ID | Type | What it represents |
|---|---|---|
| 0 | STATE_VAR | State variable declaration |
| 1 | FUNCTION | Function definition |
| 2 | MODIFIER | Modifier definition |
| 3 | EVENT | Event definition |
| 4 | FALLBACK | Fallback function |
| 5 | RECEIVE | Receive function |
| 6 | CONSTRUCTOR | Constructor |
| 7 | CONTRACT | Contract declaration |
| 8 | CFG_NODE_CALL | CFG node: function call |
| 9 | CFG_NODE_WRITE | CFG node: state write |
| 10 | CFG_NODE_READ | CFG node: state read |
| 11 | CFG_NODE_CHECK | CFG node: condition check |
| 12 | CFG_NODE_OTHER | CFG node: other |
| 13 | CFG_NODE_ARITH | CFG node: arithmetic |

**Edge types** (12):

| ID | Type | What it represents |
|---|---|---|
| 0 | CALLS | Function calls another function |
| 1 | READS | Function reads a state variable |
| 2 | WRITES | Function writes a state variable |
| 3 | EMITS | Function emits an event |
| 4 | INHERITS | Contract inherits from another |
| 5 | CONTAINS | Contract contains a definition |
| 6 | CONTROL_FLOW | CFG edge (sequential) |
| 7 | REVERSE_CONTAINS | Reverse of CONTAINS |
| 8 | CALL_ENTRY | External call entry point |
| 9 | RETURN_TO | Return edge (pairs with CALL_ENTRY) |
| 10 | DEF_USE | Definition-use chain |
| 11 | EXTERNAL_CALL | External contract call |

---

## 3. Splits

From `data_module/data/splits/v3/split_manifest.json`:

| Split | Contracts | Ratio |
|---|---|---|
| train | 18,596 | 0.70 |
| val | 1,983 | 0.15 |
| test | 1,914 | 0.15 |

- **Strategy**: stratified per source (dive, solidifi, smartbugs_curated)
- **Seed**: 42 (reproducible)
- **Note**: 22,356 total contracts, but only 21,523 have valid representations (some fail graph extraction). SentinelDataset further filters to contracts present in the shard index.
- After filtering: train=17,877, val=1,878 usable in SentinelDataset

---

## 4. Inference-time extraction

At inference time (not training), the ML module extracts graphs from raw `.sol` source:

**Graph extraction** (`ml/src/inference/preprocess.py:137-145`):
- `GraphExtractionConfig`: `include_edge_attr=True` (default), `solc_version` auto-detected from source, `multi_contract_policy='most_derived'`
- Uses Slither's AST → builds PyG `Data` object with `x` [N, 12], `edge_index` [2, E], `edge_attr` [E]

**Windowed tokenization** (`ml/src/data_extraction/windowed_tokenizer.py`):
- `WINDOW_SIZE = 512` (CodeBERT max sequence length)
- `MAX_WINDOWS = 4` (cap; contracts longer than 4×512 tokens are sub-sampled via linspace)
- Output: `[4, 512]` int64 tensor — always 4 windows regardless of contract length
- Short contracts: 1 real window + 3 zero-padded
- `attention_mask = (input_ids != PAD_TOKEN_ID)` — 0 for padded windows

---

## 5. How to run a new export

The data module's export pipeline converts raw `.sol` files into the sharded `.pt` format:

```bash
cd ~/projects/sentinel
source data_module/.venv/bin/activate
python -m sentinel_data.export --output-dir data_module/data/exports/new-export-$(date +%Y-%m-%d)
```

**When to re-export:**
- Schema bump (FEATURE_SCHEMA_VERSION changes) — breaks Gate 2 if not re-exported
- New contracts added to the corpus
- Label corrections

**What changes:**
- `manifest.json` (new artifact_hash, new n_contracts)
- All shard files (graphs + tokens regenerated)
- `labels.parquet` (if labels updated)

**What stays:**
- `splits/` directory (independent of export, controls train/val/test assignment)
- Existing models trained on the old export (they still work — Gate checks prevent loading mismatched exports)

---

## 6. Collate function

`ml/src/datasets/collate.py` — `sentinel_collate_fn`:

- Input: list of 5-tuples `(graph, tokens, y, contract_id, confidence_tier)`
- Output: `(graphs_batch, tokens_batch, y_batch, contract_ids, confidence_tiers)`
- `Batch.from_data_list(graphs, exclude_keys=_EXCLUDE_KEYS)` — merges PyG graphs into a batch
- `_EXCLUDE_KEYS = ["contract_hash", "contract_path", "contract_name", "node_metadata", "num_edges", "num_nodes", "y"]` — non-tensor metadata that Batch can't merge
- `tokens_batch`: `input_ids` [B, 4, 512], `attention_mask` [B, 4, 512] — stacked via `torch.stack`

---

## Deep reference

- → `data_module/README.md` — data module overview
- → source: `sentinel_dataset.py:57-193`, `collate.py`, `graph_schema.py`
- → `ml/src/preprocessing/graph_extractor.py` — inference-time Slither-based extraction
- → `ml/src/data_extraction/windowed_tokenizer.py` — CodeBERT tokenization with sliding windows
- → `data_module/data/exports/sentinel-v2-baseline-2026-06-12/manifest.json` — actual export manifest
