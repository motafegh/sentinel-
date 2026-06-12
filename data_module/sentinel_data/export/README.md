# `sentinel_data.export` — Stage 7: Producing the Consumable Artifact (STUB)

> **Status: ⚠️ STUB.** The `export/` package contains only `__init__.py` (10 lines). The 4 shard writers, `format_schema/v1.yaml`, the `SentinelDatasetExport` consumer-facing class, and the `chunker.py` orchestrator described in the previous README **do not exist on disk**. The CLI's `_run_export` (`cli.py:661-668`) is also a stub: "NOT IMPLEMENTED — implement in Stage 7". This is the **next stage to build** after Stage 5 (splitting + registry) is complete.

## 1. Purpose

Export is the **seam** between the data module and the ML training module. Its job is to take all the artifacts produced by Stages 1–6 and produce the **sharded export** that the ML module's `SentinelDataset` reads during training.

The format contract is:

- **Byte-identical**: the same input produces the same output, every time
- **Versioned**: schema v1 is the contract; future bumps are new files (e.g. v2)
- **Hash-verified**: the ML module verifies the hash on load (via `Catalog.verify_artifact_hash`)

When the seam swap happens (Stage 7 of the data pipeline), the old `ml/src/inference/dual_path_dataset.py` is replaced by a thin `sentinel_dataset.py` that reads from the v2 export directly.

> **This README describes the PLANNED design (per the previous README and the Stage 7 plan).** The actual code currently consists of a 10-line `__init__.py` and a 7-line CLI stub. Do not assume any of the files mentioned in §2 below exist; verify before importing.

## 2. Source map — PLANNED (NOT YET IMPLEMENTED)

| File | Status | Role |
|------|--------|------|
| `__init__.py` | ✅ 10 lines | Module docstring describing the planned 4-writer design. |
| `format_schema/v1.yaml` | ❌ NOT ON DISK | The export format contract (column order, dtypes, shard pattern). |
| `graph_writer.py` | ❌ NOT ON DISK | Writes sharded PyG `Batch` objects. |
| `token_writer.py` | ❌ NOT ON DISK | Writes sharded token tensors. |
| `label_writer.py` | ❌ NOT ON DISK | Writes `labels.parquet` with per-class columns. |
| `metadata_writer.py` | ❌ NOT ON DISK | Writes `metadata.parquet` with provenance. |
| `chunker.py` | ❌ NOT ON DISK | Orchestrates the 4 writers per shard. |
| `export.py` | ❌ NOT ON DISK | `SentinelDatasetExport` consumer-facing API. |

> The previous `export/README.md` (still in place) describes this planned structure. It is **mostly accurate as a design spec** but misleading as a code reference. The Stage 7 implementation will follow this design.

## 3. Key concepts — PLANNED

### The 4 file types per shard (per the design)

| File | Content | Format |
|------|---------|--------|
| `graphs-{shard:05d}.pt` | PyG `Batch` object — the GNN input | torch pickle |
| `tokens-{shard:05d}.pt` | torch.Tensor `[N, 4, 512]` — the GraphCodeBERT input | torch tensor |
| `labels.parquet` | Per-contract labels (10 classes × per-class columns) | parquet |
| `metadata.parquet` | Per-contract provenance (sha, source, solc_version, version_bucket, loc, n_functions, n_pos, primary_class, confidence_tier) | parquet |

Default shard size: **5,000 contracts per shard** (configurable). Sharded for **efficient lazy loading** — the trainer can `mmap` the relevant shards instead of loading the whole corpus.

### The format schema (planned `format_schema/v1.yaml`)

```yaml
shard_size: 5000

file_types:
  graphs_shard:
    pattern: "graphs-{shard:05d}.pt"
    content: "PyG Batch object (concatenated from per-graph Data)"
  tokens_shard:
    pattern: "tokens-{shard:05d}.pt"
    content: "torch.Tensor [N_shard, 4, 512] — N_shard contracts per shard"
  labels_parquet:
    filename: "labels.parquet"
    columns:
      - contract_id: string
      - source: string
      - class_0: int8   # CallToUnknown (labeling order — see warning below)
      - class_1: int8   # DenialOfService
      # ... (10 classes total, labeling order from taxonomy.yaml)
      - confidence_0: float32
      # ...
      - split: string  # "train" / "val" / "test"
  metadata_parquet:
    filename: "metadata.parquet"
    columns:
      - contract_id: string
      - source: string
      - solc_version: string
      - version_bucket: string  # "legacy" / "transitional" / "modern"
      - loc: int
      - n_functions: int
      - n_pos: int
      - primary_class: string
      - confidence_tier: string
```

> **⚠ The class column order in the planned `format_schema/v1.yaml` is the LABELING order** (CallToUnknown=0, …, UnusedReturn=9), NOT the representation order. The ML module must use `class_names()` from the labeling taxonomy to decode the columns, OR the schema must be updated to match the model order before Stage 7 ships. See `sentinel_data/labeling.schema/README.md` §3 for the two-taxonomy divergence.

### The CLI subcommand (stub)

`cli.py:661-668`:
```python
def _run_export(args):
    print(f"[export] {STAGE_DESCRIPTIONS['export']}")
    print(f"  config : {args.config}")
    if args.dry_run:
        print("  (dry-run — no files written)")
        return
    print("  NOT IMPLEMENTED — implement in Stage 7")
```

`STAGE_DESCRIPTIONS["export"]` (`cli.py:92`): `"Shard export to sentinel-ml; predictor tier-threshold fix; EMITS edge fix"` — note that this description also commits to **fixing two known bugs in the ML module** during Stage 7 (the predictor.py tier threshold and the EMITS edge in graph_extractor). Both are in `ml/src/`, not `sentinel_data/`.

### Why this stage is deferred (per MEMORY.md)

The v2 data module build prioritized Stages 0–4 (skeleton, ingestion+preprocess, representation, labeling, verification) over Stage 7 (export). Reasons:

- **Seam-swap is a 1-line change** — `sentinel_data.representation.X` thin-adapter re-exports from `ml.src.preprocessing.X` already work. The `ml/` training pipeline can import from the new path today.
- **Direct read of representations + labels + manifest** — the v9 training pipeline (`dual_path_dataset.py`) can read `data/representations/<source>/<sha256>.pt` + `data/labels/merged/<sha256>.labels.json` + `data/splits/v1/{train,val,test}.jsonl` directly, without going through an export stage. The sharded format is a v2.1+ optimization for lazy loading.
- **The 2 unfixed ML bugs** (predictor.py tier threshold + EMITS edge) are on the ml/ side, not the data/ side. The Stage 7 fix lives in `ml/src/`, not `sentinel_data/`.

So the `export/` package is a **placeholder** that exists to (a) reserve the directory structure for the eventual implementation, (b) document the planned design, and (c) be the natural home for the future `chunker.py` / `SentinelDatasetExport` work.

## 4. Public API

**There is no public API today.** The package has no Python code beyond the 10-line `__init__.py`.

When Stage 7 ships, the planned public API is:

```python
# sentinel_data.export
class SentinelDatasetExport: ...       # the consumer-facing class
def export_dataset_version(version_name, *, output_dir=None) -> str: ...   # returns artifact hash

# sentinel_data.export.format_schema
SCHEMA_VERSION = "v1"
SHARD_SIZE = 5000
```

## 5. Inputs → outputs (PLANNED)

| Input | Where | What |
|-------|-------|------|
| `data/representations/<source>/<sha256>.pt` | Stage 2 | The graph representations (input to graphs_writer) |
| `data/representations/<source>/<sha256>.tokens.pt` | Stage 2 | The token representations (input to token_writer) |
| `data/labels/merged/<sha256>.labels.json` | Stage 3 | The labels (input to label_writer) |
| `data/splits/v<N>/{train,val,test}.jsonl` | Stage 5 | The split assignment per contract (input to metadata_writer.split column) |
| `data/preprocessed/<source>/<sha256>.meta.json` | Stage 1b | For metadata_writer (solc_version, version_bucket, etc.) |
| `data/registry/catalog.db` | Stage 5 | The artifact hash is recorded as `DatasetVersion.artifact_hash` |
| `data/verification/verification_report_*.md` | Stage 4 | Linked from `DatasetVersion.verification_report_path` |

| Output | Where | What |
|--------|-------|------|
| `data/exports/<version_name>/graphs-{shard:05d}.pt` | `graph_writer` | PyG Batch shards |
| `data/exports/<version_name>/tokens-{shard:05d}.pt` | `token_writer` | Token shards |
| `data/exports/<version_name>/labels.parquet` | `label_writer` | Labels table |
| `data/exports/<version_name>/metadata.parquet` | `metadata_writer` | Metadata table |
| `data/registry/catalog.db` updated | `Catalog.add_dataset_version` | With `artifact_hash` and `artifact_path` |
| `data/changelog.md` updated | `update_changelog` | With the new export |

## 6. Pipeline interactions

| Stage | Direction | What |
|-------|-----------|------|
| Stage 2 (representation) | ← | `graph_writer` reads `<sha>.pt` and `<sha>.tokens.pt` |
| Stage 3 (labeling) | ← | `label_writer` reads `data/labels/merged/*.labels.json` |
| Stage 5 (splitting) | ← | `metadata_writer` reads `data/splits/v<N>/*.jsonl` for the `split` column |
| Stage 1b (preprocessing) | ← | `metadata_writer` reads `*.meta.json` for solc_version + version_bucket |
| Stage 5 (registry) | ↔ | `Catalog.add_dataset_version` records the artifact_hash + artifact_path |
| `ml/` training (SentinelDataset) | → | `SentinelDataset` (post-seam-swap) reads the shards via `mmap` |
| Stage 4 (verification) | ← | The verification report path is linked from the registered DatasetVersion |
| (Two ML-side bugs to fix) | ✗ | `predictor.py` tier threshold + `EMITS` edge in `graph_extractor.py` — both in `ml/`, not `sentinel_data/` |

## 7. Tests

**There are no tests today** (no `tests/test_export/` directory, since no code exists).

When Stage 7 ships, the planned test coverage is:

- Round-trip: write a shard, read it back via `SentinelDatasetExport`, verify byte-identical
- Hash verification: the artifact hash matches the registered hash in the catalog
- Schema versioning: a v1 export is readable by a v1 reader; a v2 export is rejected by a v1 reader
- Shard integrity: missing or corrupted shards raise a clear error

## 8. See also

- Previous stage: `sentinel_data/analysis/README.md`
- **CRITICAL**: The Stage 7 implementation must address the **two-taxonomy divergence** — the planned `format_schema/v1.yaml` uses the labeling class order, but the model's classifier head uses the representation class order. See `sentinel_data/labeling.schema/README.md` §3 and `sentinel_data/representation/README.md` §3. The fix is either (a) update the schema to use the representation order, or (b) explicitly document the column order in the labels.parquet as the labeling order and add a mapping table in the README. The current plan does not address this.
- The **seam-swap** (Stage 7 of the data pipeline, separate from the export stage): see `sentinel_data/representation/README.md` §3 — the thin-adapter re-exports will be deleted and the import in `ml/` will be rebinded to `sentinel_data.representation.X` directly.
- The two ML-side bugs to fix during Stage 7: `predictor.py` tier threshold + `EMITS` edge in `graph_extractor.py`. See `MEMORY.md` §"Critical Stage 2 discoveries" — EMITS was actually fixed in the codebase; the predictor.py tier threshold is still OPEN.
- Stage 7 plan (when written): `docs/proposal/Data_Module_Proposals/actionable_plans/07_stage_7_export.md` (placeholder path — file not yet created)
- The format schema concept: mirrors the v1 contract that `ml/data/processed/multilabel_index.csv` established for the old `dual_path_dataset.py`
- The current ML-side `SentinelDataset`: `ml/src/training/dual_path_dataset.py` (will be replaced by a thin wrapper reading from the export)
