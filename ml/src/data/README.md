# ml/src/data — Data Storage Directory

Storage location for processed ML data artifacts.

## Purpose

This directory contains the actual data files used by the SENTINEL ML pipeline, including extracted graphs, tokenized contracts, and cached datasets.

## Structure

### `graphs/`
- **Contents**: PyTorch Geometric graph files (.pt format)
- **Schema**: v7 schema with 11-dimensional node features
- **Count**: 41,522 graph files
- **Naming**: `<md5_hash>.pt` (MD5 hash of contract path)
- **Features**:
  - 8 edge types (CALLS, READS, WRITES, EMITS, INHERITS, CONTAINS, CONTROL_FLOW, REVERSE_CONTAINS)
  - 11-dimensional node feature vectors
  - FEATURE_SCHEMA_VERSION = "v7"

## Data Files

### Graph Files (`graphs/*.pt`)
Each graph file contains:
- Node features: shape [N, 11] where N is number of nodes
- Edge indices: shape [2, E] where E is number of edges
- Edge types: shape [E] with values 0-7
- Graph metadata: contract hash, extraction timestamp

### Token Files (stored in `ml/data/tokens_windowed/`)
- **Location**: Parent directory `ml/data/tokens_windowed/`
- **Format**: PyTorch tensors (.pt)
- **Shape**: [4, 512] - 4 sliding windows of 512 tokens each
- **Count**: 44,470 token files
- **Naming**: `<md5_hash>.pt` (matches corresponding graph file)

### Cached Dataset (stored in parent directory)
- **Location**: `ml/data/cached_dataset_deduped.pkl`
- **Size**: 2.28 GB
- **Contents**: 41,577 paired (graph, tokens) samples
- **Purpose**: Fast loading for training/inference

## Data Pipeline

Data flows through these stages:
1. **Raw contracts** → `ml/src/data_extraction/ast_extractor.py` → `graphs/*.pt`
2. **Raw contracts** → `ml/src/data_extraction/tokenizer.py` → `tokens_windowed/*.pt`
3. **Graphs + tokens** → `ml/scripts/create_cache.py` → `cached_dataset_deduped.pkl`

## Schema Versioning

Current schema version: **v7**
- NODE_FEATURE_DIM = 11
- FEATURE_SCHEMA_VERSION = "v7"
- Any schema changes require bumping this version and re-extracting all graphs

## Important Notes

- This directory is NOT committed to git (see .gitignore)
- Graph files use MD5 hash naming for consistent pairing with token files
- All graph files follow the v7 schema (11-dim node features)
- Missing graphs (2,948 stems) are expected due to Slither extraction failures
- Cache builder automatically excludes samples without matching graphs

## Storage Requirements

- Graphs directory: ~2-3 GB
- Tokens directory: ~1-2 GB
- Cached dataset: 2.28 GB
- Total: ~5-7 GB

## Regeneration

To regenerate all data files:
```bash
# Re-extract graphs
poetry run python ml/scripts/reextract_graphs.py

# Re-tokenize contracts
poetry run python ml/scripts/retokenize_windowed.py

# Rebuild cache
poetry run python ml/scripts/create_cache.py
```
