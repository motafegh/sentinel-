# ml/src/data_extraction — Raw Data Processing

Offline batch processing pipeline for converting Solidity contracts into model-ready representations.

## Purpose

This module handles the extraction and tokenization of raw Solidity source code into the graph and token representations used by the SENTINEL dual-path architecture.

## Components

### `ast_extractor.py`
**AST Extractor V4.3 — Offline Batch Pipeline**

Orchestration layer for graph extraction from Solidity contracts:
- Reads contract metadata from parquet files
- Resolves correct solc binary for each Solidity version
- Spawns 11 worker processes for parallel extraction
- Extracts PyTorch Geometric graphs with v7 schema (11-dim features)
- Writes `<md5_hash>.pt` files to `ml/data/graphs/`

**Key Features:**
- Checkpoint/resume system for large batches
- Version-pinned solc binary resolution
- Multiprocessing for speed
- Error handling with skip-and-log policy

**Architecture Note:**
Graph construction logic has been extracted to:
- `ml/src/preprocessing/graph_schema.py` — Schema definitions
- `ml/src/preprocessing/graph_extractor.py` — Core extraction logic

This ensures training/inference feature consistency.

### `tokenizer.py`
**CodeBERT Tokenizer V1 — Production Pipeline**

Converts Solidity contracts to CodeBERT token sequences:
- MD5 hash naming (matches graph files)
- Checkpoint/resume system
- Multiprocessing (11 workers)
- Sliding window tokenization (4 windows × 512 tokens)
- Batch processing for speed

**Key Features:**
- Handles long contracts via sliding windows
- Consistent hashing for graph-token pairing
- Error handling and logging
- Progress tracking with tqdm

## Data Flow

```
Raw Solidity Contracts (.sol)
        │
        ├─► ast_extractor.py ──► ml/data/graphs/<hash>.pt
        │                         (PyG graphs, 11-dim features)
        │
        └─► tokenizer.py ──────► ml/data/tokens_windowed/<hash>.pt
                                  (4×512 token windows)
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

## Schema Compatibility

**Current Schema: v7**
- NODE_FEATURE_DIM = 11
- FEATURE_SCHEMA_VERSION = "v7"
- 8 edge types (0-7)

**Important:** 
- Existing .pt files in `ml/data/graphs/` use edge_attr shape [E, 1]
- New files use edge_attr shape [E] (PyG 1-D convention)
- GNNEncoder ignores edge_attr, so both shapes are safe

## Dependencies

- `slither-analyzer` >= 0.9.3 — Static analysis and graph extraction
- `solc-select` — Solidity version management
- `torch-geometric` — PyG graph structures
- `transformers` — CodeBERT tokenizer
- `pandas` — Metadata handling

## Performance

**Graph Extraction:**
- ~41,522 contracts
- 11 workers
- Several hours (depends on hardware)

**Tokenization:**
- ~44,470 contracts
- 11 workers
- ~30-60 minutes

## Error Handling

Both scripts implement:
- Checkpoint/resume for interrupted runs
- Skip-and-log policy for failed extractions
- Progress tracking
- Comprehensive error logging

## Integration

These scripts are called by:
- `ml/scripts/reextract_graphs.py` — Full graph re-extraction
- `ml/scripts/retokenize_windowed.py` — Full tokenization
- Manual runs for individual contracts or subsets
