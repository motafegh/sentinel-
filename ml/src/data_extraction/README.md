# ml/src/data_extraction — Offline Batch Data Extraction

**Purpose:** One-time batch pipelines that convert raw Solidity contracts into paired graph (.pt) and token (.pt) files for training. These scripts are NOT used during inference.

## Modules

### `ast_extractor.py` — Graph Extraction Orchestrator

**What it does:**
1. Reads contract paths from `ml/data/processed/contracts_metadata.parquet`
2. Groups contracts by Solidity version (e.g., "0.4.26", "0.8.19")
3. For each version group, resolves the pinned `solc` binary from `.venv/.solc-select/artifacts/solc-{version}/`
4. Spawns 11 worker processes to extract graphs in parallel
5. Writes `<md5_hash>.pt` files to `ml/data/graphs/`

**Key implementation details:**
- Uses `multiprocessing.Pool` with `initializer=init_worker` to load Slither once per worker (not per contract)
- Checkpoint system: saves progress every 500 contracts to `checkpoint.json` for resume capability
- Filters out contracts where `solc_binary` resolution fails
- Attaches metadata: `contract_hash`, `contract_path`, `y` (label tensor)

**Error handling:**
- Catches `GraphExtractionError` from `extract_contract_graph()` and skips failed contracts
- Logs failure rate; warns if >5% of functions fail CFG extraction
- Writes failed contract hashes to `failed_contracts.json`

**CLI flags:**
```bash
--input       # Parquet file with contract metadata (default: ml/data/processed/contracts_metadata.parquet)
--output      # Output directory for .pt files (default: ml/data/graphs)
--workers     # Number of parallel workers (default: mp.cpu_count() - 1)
--checkpoint-every  # Save checkpoint frequency (default: 500)
--test        # Process only first 100 contracts
--resume      # Skip already-processed contracts (reads checkpoint.json)
--force       # Delete checkpoint and reprocess all contracts
--verbose     # Enable detailed logging
```

### `tokenizer.py` — CodeBERT Tokenizer

**What it does:**
1. Reads contract paths from parquet file
2. Loads `microsoft/codebert-base` tokenizer (NOT graphcodebert-base for tokenization)
3. Tokenizes each contract with max_length=512, padding="max_length", truncation=True
4. Generates MD5 hash using `get_contract_hash()` (same as graph extractor)
5. Saves `<hash>.pt` files containing: `input_ids`, `attention_mask`, `contract_hash`, metadata

**Tokenization config:**
```python
MAX_LENGTH = 512
PADDING = "max_length"
TRUNCATION = True
DEFAULT_WORKERS = 11
CHECKPOINT_INTERVAL = 500
```

**Output format:**
```python
{
    'input_ids': torch.Tensor([512]),           # Token IDs
    'attention_mask': torch.Tensor([512]),      # 1=real, 0=pad
    'contract_hash': str,                        # MD5 hash
    'contract_path': str,                        # Source file path
    'num_tokens': int,                           # Actual tokens before padding
    'truncated': bool,                           # Was original > 512?
    'tokenizer_name': str,                       # "microsoft/codebert-base"
    'max_length': int,                           # 512
    'feature_schema_version': str,               # e.g., "v8"
}
```

**Multiprocessing pattern:**
- `init_worker()` loads the 500MB tokenizer ONCE per worker at startup
- Without this: 68,568 separate loads = hours of runtime
- With this: 11 loads = seconds

**CLI flags:**
```bash
--input             # Parquet file (default: ml/data/processed/contracts_metadata.parquet)
--output            # Output directory (default: ml/data/tokens)
--workers           # Parallel workers (default: 11)
--chunk-size        # Contracts per batch (default: 50)
--checkpoint-every  # Save frequency (default: 500)
--test              # First 100 contracts only
--resume            # Skip processed contracts
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  ast_extractor.py                            │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │ Read parquet │───►│ Group by ver │───►│ Pool(workers)│  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
│                                              │               │
│                                    ┌─────────▼─────────┐    │
│                                    │ extract_contract_ │    │
│                                    │ graph() from      │    │
│                                    │ preprocessing/    │    │
│                                    └─────────┬─────────┘    │
└──────────────────────────────────────────────┼──────────────┘
                                               │
                                    ┌──────────▼──────────┐
                                    │ ml/data/graphs/     │
                                    │ <hash>.pt files     │
                                    └─────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                    tokenizer.py                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │ Read parquet │───►│ Pool(workers)│───►│ tokenize_    │  │
│  └──────────────┘    └──────────────┘    │ single_      │  │
│                                           │ contract()   │  │
│                                           └──────┬───────┘  │
└──────────────────────────────────────────────────┼──────────┘
                                                   │
                                        ┌──────────▼──────────┐
                                        │ ml/data/tokens/     │
                                        │ <hash>.pt files     │
                                        └─────────────────────┘
```

## Data Pairing

Both scripts use `get_contract_hash(contract_path)` from `ml/src/utils/hash_utils.py`:
- Hashes the **full file path** (not content) for uniqueness
- Returns 32-character MD5 hex string
- Same hash → graph and token files pair correctly in dataset

## Current Output State

| Directory | Files | Schema | Notes |
|-----------|-------|--------|-------|
| `ml/data/graphs/` | 41,576 | v8 (11-dim nodes, 11 edge types) | From ~68K input contracts |
| `ml/data/tokens/` | 44,470 | CodeBERT, [512] per file | Single-window format |

**Note:** 2,948 contracts have graphs but no tokens (or vice versa) — these are excluded from the paired dataset.

## Dependencies

```toml
slither-analyzer = "^0.9.3"   # AST parsing, graph extraction
solc-select                   # Version-pinned solc binaries
torch-geometric               # PyG Data structures
transformers                  # CodeBERT tokenizer
pandas                        # Parquet file handling
multiprocessing               # Parallel processing (stdlib)
```

## Related Scripts

- `ml/scripts/reextract_graphs.py` — Wrapper with additional validation
- `ml/scripts/retokenize_windowed.py` — Converts single-window tokens to 4-window sliding format
- `ml/scripts/create_cache.py` — Builds `cached_dataset_v8.pkl` from paired graph+token files
