# ml/src/data_extraction — Raw Data Processing

Offline batch pipeline converting Solidity contracts into model-ready graph and token representations.

## Purpose

Handles the extraction and tokenization of raw Solidity source code into the v8 graph and token formats consumed by the SENTINEL dual-path model. Both scripts checkpoint/resume so they can be interrupted and restarted.

## Components

### `ast_extractor.py`
**Graph Extraction Pipeline**

Orchestration layer for batch graph extraction from Solidity contracts:
- Reads contract metadata from parquet files
- Resolves correct `solc` binary per Solidity version via `solc-select`
- Spawns parallel worker processes for extraction
- Extracts PyG graphs with v8 schema (11-dim node features, 11 edge types)
- Writes `<md5_hash>.pt` files to `ml/data/graphs/`

**Architecture note:** core graph construction logic lives in:
- `ml/src/preprocessing/graph_schema.py` — schema constants, NodeType IntEnum
- `ml/src/preprocessing/graph_extractor.py` — `extract_contract_graph()`, feature builders, edge constructors

This ensures offline extraction and online inference compute identical features.

### `tokenizer.py`
**GraphCodeBERT Tokenizer Pipeline**

Converts Solidity contracts to GraphCodeBERT token sequences:
- MD5 hash naming (matches graph files for pairing)
- Sliding window tokenization: 4 windows × 512 tokens, stride=256
- Checkpoint/resume for interrupted runs
- Parallel workers with `init_worker()` pattern (tokenizer loaded once per worker process)

**Multiprocessing efficiency:** The `init_worker()` function loads the 500MB GraphCodeBERT tokenizer once per worker process at startup. Without this pattern, the tokenizer would be loaded 68,568 times (once per contract), causing severe performance degradation.

**Error handling:** Failed contracts are tracked in `failed_contracts.json` with their MD5 hashes. The pipeline logs success rate warnings if below 95%.

**Statistics output:** Reports truncation rate (contracts exceeding 512 tokens) and success/failure counts at completion.

## Data Flow

```
Raw Solidity Contracts (.sol)
        │
        ├─► ast_extractor.py ──► ml/data/graphs/<hash>.pt
        │                         PyG Data, graph.x=[N,11], graph.edge_attr=[E] int64
        │
        └─► tokenizer.py ──────► ml/data/tokens_windowed/<hash>.pt
                                  tensor [4, 512] (4 windows, stride=256)
```

## Usage

### Extract Graphs
```bash
poetry run python ml/src/data_extraction/ast_extractor.py \
    --metadata-path ml/data/processed/metadata.parquet \
    --output-dir ml/data/graphs \
    --workers 11
```

### Tokenize Contracts
```bash
poetry run python ml/src/data_extraction/tokenizer.py \
    --metadata-path ml/data/processed/metadata.parquet \
    --output-dir ml/data/tokens_windowed \
    --workers 11
```

Or via the pipeline scripts (preferred — handles checkpoint/resume automatically):
```bash
poetry run python ml/scripts/reextract_graphs.py
poetry run python ml/scripts/retokenize_windowed.py
```

## Schema Reference

**Current Schema: v8**

| Constant | Value |
|----------|-------|
| `NODE_FEATURE_DIM` | 11 |
| `FEATURE_SCHEMA_VERSION` | `"v8"` |
| `NUM_EDGE_TYPES` | 11 |
| `edge_attr` shape | `[E]` 1-D int64 |

`edge_attr` is stored as 1-D int64 (PyG convention). Older extractions may have `[E, 1]` shape — the GNNEncoder's `Embedding(11, 64)` handles both via `.squeeze(-1)` normalization in the dataset loader.

## Current Data State

| Directory / File | Count | Notes |
|-----------------|-------|-------|
| `ml/data/graphs/` | 41,576 `.pt` | v8 schema, 11-dim |
| `ml/data/tokens_windowed/` | 44,470 `.pt` | [4, 512], stride=256 |

2,948 stems in the label CSV have no matching graph — expected Slither extraction failures. The cache builder excludes them automatically.

## Dependencies

- `slither-analyzer` ≥ 0.9.3 — static analysis and graph extraction
- `solc-select` — Solidity compiler version management
- `torch-geometric` — PyG graph structures
- `transformers` — GraphCodeBERT tokenizer (`microsoft/graphcodebert-base`)
- `pandas` — metadata parquet handling
