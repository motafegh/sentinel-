# `sentinel_data.export` — Stage 7: Producing the Consumable Artifact

> **Status: ✅ Fully implemented.** 4 writers (graphs, tokens, labels, metadata) + orchestrator (`chunker.py`) + consumer-facing API (`SentinelDatasetExport`) + format schema (`v1.yaml`). The CLI `export` subcommand is wired and functional.

## 1. Purpose

Export is the **seam** between the data module and the ML training module. It takes all the artifacts produced by Stages 1–6 and produces a **sharded export** that the ML module's `SentinelDataset` reads during training.

The format contract:

- **Byte-identical**: the same input produces the same output, every time
- **Versioned**: schema v1 is the contract; future bumps are new files (e.g. v2)
- **Hash-verified**: the ML module verifies the artifact hash on load (via `SentinelDatasetExport.verify_artifact_hash`)
- **Sharded for lazy loading**: default 5,000 contracts per shard; the trainer can `mmap` relevant shards instead of loading the whole corpus

The `sentinel-data export` CLI subcommand is the entry point. It reads from `data/representations/`, `data/labels/merged/`, `data/splits/v<N>/`, and `data/preprocessed/`, and writes to `data/exports/<version>/`.

## 2. Source map

| File | Lines | Role |
|------|-------|------|
| `__init__.py` | 27 | Re-exports all public symbols: `chunk_export`, `SentinelDatasetExport`, `ExportManifest`, 4 writers. |
| `chunker.py` | 237 | **Orchestrator** — `chunk_export()` runs the 4 writers in order, computes the SHA-256 artifact hash, writes `manifest.json` last (Fix A — avoids circular hash). |
| `export.py` | 173 | `SentinelDatasetExport` — consumer-facing read-only API. Wraps an export directory, provides manifest loading, hash verification, split-aware contract ID lookup. |
| `graph_writer.py` | 106 | Writes sharded PyG `Batch` objects (`graphs-{shard:05d}.pt`). Contract order driven by split JSONL (train → val → test). |
| `token_writer.py` | 95 | Writes sharded token tensors (`tokens-{shard:05d}.pt`) as `[N, 4, 512]` int64 input_ids. Order mirrors `graph_writer`. |
| `label_writer.py` | 150 | Writes `labels.parquet` — 14 columns: contract_id, source, split, class_0..class_9 (int8, locked taxonomy order), confidence_tier (nullable). |
| `metadata_writer.py` | 200 | Writes `metadata.parquet` — 14 columns: contract_id, source, split, solc_version, version_bucket, loc, n_functions, n_pos, primary_class, node_count, edge_count, has_unchecked_block, dedup_group_id, confidence_tier. |
| `format_schema/v1.yaml` | 494 | The export format contract: shard patterns, column definitions, dtypes, version pins. |

**Sub-total: ~995 lines** across 7 Python files + 1 YAML (494 lines).

## 3. Key concepts

### The 4 file types per shard

| File | Content | Format |
|------|---------|--------|
| `graphs-{shard:05d}.pt` | PyG `Batch` object — the GNN input | torch pickle |
| `tokens-{shard:05d}.pt` | `torch.Tensor [N, 4, 512]` — the GraphCodeBERT input | torch tensor |
| `labels.parquet` | Per-contract labels (10 classes × per-class columns) | parquet |
| `metadata.parquet` | Per-contract enrichment (solc, LoC, node/edge counts, etc.) | parquet |

Default shard size: **5,000 contracts per shard** (configurable via `--shard-size`).

### The orchestrator (`chunker.py`)

```python
def chunk_export(
    rep_root: Path,       # data/representations/
    preproc_root: Path,   # data/preprocessed/
    splits_dir: Path,     # data/splits/v{N}/
    output_dir: Path,     # data/exports/<version>/
    config_path: Path | None = None,
    shard_size: int = 5000,
    source_set: list[str] | None = None,
    skipped_sources: list[dict] | None = None,
    graph_schema_version: str = "v9",
) -> ExportManifest:
```

Execution order:
1. `write_labels_parquet()` → `labels.parquet`
2. `write_metadata_parquet()` → `metadata.parquet`
3. `write_graphs_shards()` → `graphs/` directory + shard index
4. `write_tokens_shards()` → `tokens/` directory + shard index
5. Build `full_shard_index` from graph shard map + split JSONL order
6. Compute `artifact_hash` over all data files (Fix A — excludes `manifest.json`)
7. Write `manifest.json` LAST (avoids circular hash dependency)

### Fix A: manifest.json written last

The `artifact_hash` in `manifest.json` is a SHA-256 over all 4 data file types (sorted by relative path). The `manifest.json` itself is excluded from the hash — otherwise the hash would need to be computed before writing the manifest, but the manifest contains the hash. Writing the manifest last breaks this circular dependency.

### The consumer-facing API (`export.py`)

```python
class SentinelDatasetExport:
    """Read-only view of a chunk_export() output directory."""
    
    def __init__(self, export_dir: Path) -> None: ...
    
    @property
    def graphs_dir(self) -> Path: ...
    @property
    def tokens_dir(self) -> Path: ...
    @property
    def labels_path(self) -> Path: ...
    @property
    def metadata_path(self) -> Path: ...
    
    def verify_artifact_hash(self) -> bool: ...
    def get_split_contract_ids(self, split: str) -> list[str]: ...
```

The ML-side `SentinelDataset` (post-seam-swap) wraps this class to implement `__len__` and `__getitem__` for PyTorch training.

### The 10-class taxonomy in labels.parquet

The `class_0..class_9` columns use the **labeling order** (CallToUnknown=0, DenialOfService=1, …, UnusedReturn=9). This matches `class_names()` from `sentinel_data.labeling.schema` and `CLASS_NAMES` from `sentinel_data.representation.graph_schema` (per ADR-0009, 2026-06-12 — the two are now aligned).

### The manifest (`ExportManifest`)

```python
@dataclass
class ExportManifest:
    schema_version: str              # "v1"
    graph_schema_version: str        # "v9"
    artifact_hash: str               # SHA-256 over data files
    hash_algorithm: str              # "sha256"
    shard_size: int                  # 5000 (default)
    n_contracts: int                 # total from split JSONL
    n_contracts_with_reps: int       # those with .pt representations
    n_shards: int                    # number of graph/token shards
    splits: dict[str, list[str]]     # {train: [sha256...], val: [...], test: [...]}
    shard_index: dict[str, dict]     # {sha256: {shard: int, pos_in_shard: int, num_nodes: int}}
    source_set: list[str]            # sources actually exported
    skipped_sources: list[dict]      # sources enabled but skipped
    preprocessing_config_hash: str   # SHA-256 of config.yaml
    label_class_columns: list[str]   # ["CallToUnknown", ..., "UnusedReturn"]
    created_at: str                  # ISO-8601 UTC timestamp
```

## 4. Public API

| Symbol | Source | Description |
|--------|--------|-------------|
| `chunk_export(...)` | `chunker.py:151-234` | Main entry point — runs all 4 writers + manifest |
| `ExportManifest` | `chunker.py:44-60` | Dataclass for `manifest.json` |
| `SentinelDatasetExport` | `export.py:21-173` | Consumer-facing read-only API |
| `write_labels_parquet(...)` | `label_writer.py:116-143` | Write `labels.parquet` |
| `write_metadata_parquet(...)` | `metadata_writer.py:157-194` | Write `metadata.parquet` |
| `write_graphs_shards(...)` | `graph_writer.py:47-103` | Write sharded PyG Batch `.pt` files |
| `write_tokens_shards(...)` | `token_writer.py:36-92` | Write sharded token `.pt` files |

## 5. Inputs → outputs

| Input | Where | What |
|-------|-------|------|
| `data/representations/<source>/<sha256>.pt` | Stage 2 | Graph representations (input to `graph_writer`) |
| `data/representations/<source>/<sha256>.tokens.pt` | Stage 2 | Token representations (input to `token_writer`) |
| `data/labels/merged/<sha256>.labels.json` | Stage 3 | Labels (input to `label_writer` via split JSONL) |
| `data/splits/v<N>/{train,val,test}.jsonl` | Stage 5 | Split assignments (drives shard ordering) |
| `data/preprocessed/<source>/<sha256>.meta.json` | Stage 1b | For `metadata_writer` (solc_version, version_bucket) |
| `data/preprocessed/<source>/<sha256>.sol` | Stage 1b | For `metadata_writer` (loc, n_functions computed from source) |

| Output | Where | What |
|--------|-------|------|
| `data/exports/<version>/labels.parquet` | `label_writer` | Labels table |
| `data/exports/<version>/metadata.parquet` | `metadata_writer` | Metadata table |
| `data/exports/<version>/graphs/graphs-{shard:05d}.pt` | `graph_writer` | PyG Batch shards |
| `data/exports/<version>/graphs/_shard_index.json` | `chunker` | Per-shard position index |
| `data/exports/<version>/tokens/tokens-{shard:05d}.pt` | `token_writer` | Token shards |
| `data/exports/<version>/tokens/_shard_index.json` | `chunker` | Same index (mirrored) |
| `data/exports/<version>/manifest.json` | `chunker` | Written last; contains artifact_hash |

## 6. Pipeline interactions

| Stage | Direction | What |
|-------|-----------|------|
| Stage 2 (representation) | ← | `graph_writer` reads `.pt`; `token_writer` reads `.tokens.pt` |
| Stage 3 (labeling) | ← | `label_writer` reads `*.labels.json` via split JSONL |
| Stage 5 (splitting) | ← | All 4 writers read split JSONL for contract ordering |
| Stage 1b (preprocessing) | ← | `metadata_writer` reads `.meta.json` + `.sol` for enrichment |
| Stage 5 (registry) | ↔ | `Catalog.add_dataset_version` records `artifact_hash` + `artifact_path` |
| `ml/` training (SentinelDataset) | → | `SentinelDataset` (post-seam-swap) reads via `SentinelDatasetExport` |

## 7. Tests

**Location:** `data_module/tests/test_export/` (if it exists — check with `ls tests/test_export/`).

**Planned test coverage:**
- Round-trip: write a shard, read it back via `SentinelDatasetExport`, verify byte-identical
- Hash verification: `verify_artifact_hash()` returns True for clean export, False after tampering
- Schema versioning: a v1 export is readable by a v1 reader; a v2 export is rejected
- Shard integrity: missing or corrupted shards raise a clear error
- Manifest completeness: all required fields present and correct

**Command:**
```bash
cd ~/projects/sentinel/data_module
poetry run pytest tests/test_export/ -v
```

## 8. See also

- Previous stage: `sentinel_data/analysis/README.md`
- Consumer API: `sentinel_data.export.export.SentinelDatasetExport`
- The seam-swap (Stage 7B): `sentinel_data/representation/README.md` §3 — thin-adapter wrappers will be deleted and `ml/` will import from `sentinel_data.representation.X` directly
- Format schema: `sentinel_data/export/format_schema/v1.yaml`
- The ML-side `SentinelDataset`: `ml/src/training/dual_path_dataset.py` (will be replaced by a thin wrapper reading from the export)
- The two ML-side bugs to fix during Stage 7B: `predictor.py` tier threshold + `EMITS` edge in `graph_extractor.py` — both in `ml/`, not `sentinel_data/`
- MEMORY.md (canonical v2 facts): `~/.claude/projects/-home-motafeq-projects-sentinel/memory/MEMORY.md`
