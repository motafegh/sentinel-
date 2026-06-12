# Stage 7A — Export Module (LIVE PLAN)

**Started:** 2026-06-12
**Scope:** Sub-stage 7A from the Stage 7 plan. Data-module export module + CLI + predictor tier fix. Seam swap (7.6-7.8) deferred to 7B.
**Source plan:** `docs/proposal/Data_Module_Proposals/actionable_plans/08_stage_7_export_seam.md`
**Current commit:** `e5ed4d2` (Stage 6 bug fixes)
**Mode:** BUILD

---

## Pre-work (user runs in a separate terminal — not in this session's code)

- [ ] **USER: Complete Stage 2 representation on the remaining ~21,580 contracts**
  ```bash
  source ml/.venv/bin/activate
  cd data_module
  TRANSFORMERS_OFFLINE=1 sentinel-data represent --config config.yaml
  ```
  - 776 of 22,356 are done (3.5%)
  - Content-addressed cache skips the 776 already-done
  - Hours-to-days of compute time
  - **We cannot test the export end-to-end until this completes**

- [ ] **USER: Confirm what to do with the `ml/data/` legacy cache files** (`cached_dataset_v9.pkl`, `graphs/`, `tokens_windowed/`, `splits/deduped/*.npy`)
  - Per `ml/CLAUDE.md` (line 71) and the user's answer: `ml/data/` is **not** a source of truth; the data_module is.
  - Per 7-P9: archive to `ml/scripts/_legacy_data_pipeline/` (deferred to 7B)
  - For 7A: ignore `ml/data/` entirely; the export reads from `data_module/data/`

---

## Step 1 — Format schema spec (`format_schema/v1.yaml`)

- [ ] Create `data_module/sentinel_data/export/format_schema/v1.yaml`
  - Defines the 4 file types (graphs_shard, tokens_shard, labels_parquet, metadata_parquet)
  - Defines the manifest fields
  - Pin the schema version (v1)
  - Document the shard naming pattern: `graphs-{shard:05d}.pt`, `tokens-{shard:05d}.pt`
  - This is the contract — every other piece conforms to it

- [ ] Re-read `data_module/data/labels/merged/$(head -1)` for label field structure
  - sha256, sources, classes.{ClassName}.{value, tier, source}
  - 10 classes: CallToUnknown, DenialOfService, ExternalBug, GasException, IntegerUO, MishandledException, Reentrancy, Timestamp, TransactionOrderDependence, UnusedReturn

- [ ] Re-read `data_module/data/representations/dive/$(head -1).rep.json` for sidecar structure
  - sha256, source, original_path, schema_version, extractor_version, node_count, edge_count, window_count, compute_time_ms, cache_hit, pragma, solc_version

- [ ] Re-read `data_module/sentinel_data/representation/graph_schema.py` to lock the column order in `labels.parquet`
  - Class order: 0=CallToUnknown, 1=DenialOfService, 2=ExternalBug, 3=GasException, 4=IntegerUO, 5=MishandledException, 6=Reentrancy, 7=Timestamp, 8=TransactionOrderDependence, 9=UnusedReturn
  - Verify this is the same order used in `ml/src/preprocessing/graph_schema.py` and the merged labels

- [ ] Decide pyarrow version pin and minimal Parquet features (compression, dictionary encoding)
  - `pyarrow` is in the .venv; check exact version
  - Use `snappy` compression as the default (PyArrow default; widely supported)

---

## Step 2 — `label_writer.py` + `metadata_writer.py`

- [ ] Read `data_module/sentinel_data/labeling/schema/__init__.py` for `class_names()`, `class_index()` exports
- [ ] Confirm the `Contract` dataclass fields available in the split JSONL: `sha256, source, tier, classes, primary_class, n_pos, loc` (the latter defaulted to 0 at split time — see Fix #3 below)
- [ ] **Implement `data_module/sentinel_data/export/label_writer.py` reading from the split JSONL only — no re-read of merged label files**
  - Function: `write_labels_parquet(splits_dir: Path, output_path: Path) -> Path`
  - Reads `data/splits/v{N}/{train,val,test}.jsonl` (each line is an asdict-serialized `Contract`)
  - Iterates the 3 split files; emits one row per contract
  - Writes one row per contract:
    - `contract_id: str` (sha256)
    - `source: str`
    - `split: str` ("train" / "val" / "test")
    - `class_0..class_9: int8` (binary value per class, in `class_names()` order)
    - `confidence_tier: str` — **set to `tier` when `n_pos > 0`, else `None` (pyarrow null)** (NOT the raw `tier` field, which is "T0" for NonVulnerable contracts by splitter default — see Fix #2 below)
  - Use `pyarrow.Table` + `pq.write_table` with `snappy` compression
  - Schema enforced via `pyarrow.schema`
  - Returns the output path

- [ ] **Implement `data_module/sentinel_data/export/metadata_writer.py`** — also reads from split JSONL for per-contract base fields, then enriches from `.rep.json` + `.meta.json` + `.sol` sidecars
  - Function: `write_metadata_parquet(rep_root: Path, preproc_root: Path, splits_dir: Path, output_path: Path) -> Path`
  - Reads the per-contract split rows for the base fields (sha256, source, primary_class, n_pos, split, tier)
  - Enriches each row with metadata from 3 sources (joined by sha256):
    - `.rep.json` (node_count, edge_count, schema_version, solc_version)
    - `.meta.json` (from preprocessed/) (version_bucket, has_unchecked_block, contract_names, dedup_group_id)
    - `.sol` source: **computes `loc` and `n_functions` on-the-fly** — the split JSONL's `loc=0` is a default and is NOT the real LoC (the splitter never computed it; see Fix #3). Reuse `_loc()` and `_function_count()` from `data_module/sentinel_data/analysis/feature_dist.py` to avoid regex duplication.
  - Writes one row per contract:
    - `contract_id: str` (sha256)
    - `source: str`
    - `split: str` (mirror of label_writer for convenience)
    - `solc_version: str`
    - `version_bucket: str` (modern / transitional / ancient — from `meta.json.version_bucket`)
    - `loc: int32` (computed from .sol)
    - `n_functions: int32` (computed from .sol)
    - `n_pos: int8`
    - `primary_class: str`
    - `node_count: int32` (from .rep.json; null if missing)
    - `edge_count: int32` (from .rep.json; null if missing)
    - `has_unchecked_block: bool`
    - `dedup_group_id: str`
    - `confidence_tier: str` (same logic as label_writer: `tier if n_pos > 0 else None`)
  - Use `pyarrow.Table` + `pq.write_table` with `snappy` compression
  - Returns the output path

- [ ] Test: write to a tmp dir, read back with `pyarrow.parquet.read_table`, assert column names + dtypes
- [ ] Test: verify all 22,356 contracts appear in both files
- [ ] Test: verify per-class positive counts match `feature_dist.build_balance_table` (cross-check)
- [ ] Test: assert that `confidence_tier` is `None` for `n_pos == 0` and equals the split JSONL's `tier` for `n_pos > 0`

---

## Step 3 — `graph_writer.py` + `token_writer.py`

- [ ] Read `data_module/sentinel_data/representation/cache_manager.py` to understand the sidecar API
- [ ] Confirm the .pt files are PyG `Data` objects (graph) and torch.Tensor (tokens) — verify with a small load
- [ ] **Implement `data_module/sentinel_data/export/graph_writer.py`** — the writer walks the **split JSONL** (not the representations directory directly), so the shard ordering is consistent with `labels.parquet` and `metadata.parquet`. This guarantees every contract in a shard has a matching parquet row, and we don't accidentally include unlabeled or unsplit contracts that have orphan representations.
  - Function: `write_graphs_shards(rep_root: Path, splits_dir: Path, output_dir: Path, shard_size: int = 5000) -> list[Path]`
  - Reads `splits/v{N}/{train,val,test}.jsonl` and builds an ordered list of (contract_id, source, split) tuples — the canonical order
  - For each tuple, locates the PyG `Data` file at `rep_root/<source>/<sha>.pt`; loads via `torch.load(path, weights_only=False)` (or use a safe-globals allowlist as `dual_path_dataset.py:82-87` does)
  - Batches into `torch_geometric.data.Batch.from_data_list([...])`
  - Writes `graphs-{shard:05d}.pt` to `output_dir/`
  - Returns the list of shard paths
  - Tracks per-shard the contract_ids (for the shard index)
  - Skips contracts that have no representation (warn, don't fail; the parquet has all 22,356 rows, the .pt shards have only the ones with reps)

- [ ] **Implement `data_module/sentinel_data/export/token_writer.py`** — same split-JSONL-driven pattern
  - Function: `write_tokens_shards(rep_root: Path, splits_dir: Path, output_dir: Path, shard_size: int = 5000) -> list[Path]`
  - Reads `splits/v{N}/{train,val,test}.jsonl` for the canonical order
  - For each tuple, loads `rep_root/<source>/<sha>.tokens.pt`
  - Stacks tensors into a single `[N, 4, 512]` torch.Tensor per shard
  - Writes `tokens-{shard:05d}.pt` to `output_dir/`

- [ ] Test: shard count = ceil(N / 5000) for N contracts
- [ ] Test: every contract in a shard has a matching entry in `labels.parquet` (the reverse is not required: parquet has all 22,356, shards have only the ones with reps)
- [ ] Test: every contract in a shard has a matching entry in `metadata.parquet`
- [ ] Test: shard sizes are approximately uniform (allow ±1 contract for the last shard)
- [ ] Test: graph_writer and token_writer produce the **same contract_id list** in the same order (per-shard consistency)

---

## Step 4 — `chunker.py` + `SentinelDatasetExport`

- [ ] Implement `data_module/sentinel_data/export/chunker.py`
  - Function: `chunk_export(rep_root: Path, preproc_root: Path, splits_dir: Path, output_dir: Path, shard_size: int = 5000) -> ExportManifest`
  - Orchestrates the 4 writers **in this order** (critical for the hash — see note below):
    1. `write_labels_parquet(splits_dir, output_dir / "labels.parquet")`
    2. `write_metadata_parquet(rep_root, preproc_root, splits_dir, output_dir / "metadata.parquet")`
    3. `write_graphs_shards(rep_root, splits_dir, output_dir / "graphs", shard_size)`
    4. `write_tokens_shards(rep_root, splits_dir, output_dir / "tokens", shard_size)`
    5. **Compute `artifact_hash` over the data files only** (graphs/, tokens/, labels.parquet, metadata.parquet — see Fix A below)
    6. **Write `manifest.json` LAST** with the `artifact_hash` field populated
  - Builds the shard index: `dict[contract_id, int]` mapping to shard number
  - **Fix A — avoid circular hash**: `manifest.json` is itself inside the export directory and contains `artifact_hash`. Computing the hash over the entire directory would include manifest.json, which contains the hash we're computing — a chicken-and-egg problem. Solution: compute the hash over the **4 data file types only** (`graphs/*.pt`, `tokens/*.pt`, `labels.parquet`, `metadata.parquet`), then write `manifest.json` last with the hash baked in. `verify_artifact_hash()` recomputes over the same set of files (excludes `manifest.json`) and compares.
  - Manifest fields (per the plan):
    - `schema_version: "v1"`
    - `shard_size: int`
    - `n_contracts: int` (total across all 3 splits)
    - `n_shards: int` (ceil(n_with_reps / shard_size))
    - `splits: {"train": [...], "val": [...], "test": [...]}` (list of contract_ids per split)
    - `shard_index: {contract_id: shard_number}` (for lazy load; only contracts that have a .pt file)
    - `created_at: str` (ISO 8601)
    - `preprocessing_config_hash: str` (from `config.yaml`)
    - `graph_schema_version: "v9"` (the v9 schema gate per 7-P2)
    - `source_set: list[str]`
    - `label_class_columns: list[str]` (the 10 class names in order)
    - `artifact_hash: str` (SHA-256 of the 4 data file types only — **NOT** the whole directory; `manifest.json` is excluded from the hash input; renamed from `compute_hash` to avoid collision with the imported function `compute_hash` from `sentinel_data.registry.catalog`)

- [ ] Implement `data_module/sentinel_data/export/export.py`
  - Class: `SentinelDatasetExport`
  - Constructor: `SentinelDatasetExport(export_dir: Path)`
  - Properties:
    - `export_dir: Path`
    - `graphs_dir: Path` (`<export_dir>/graphs/`)
    - `tokens_dir: Path` (`<export_dir>/tokens/`)
    - `labels_path: Path` (`<export_dir>/labels.parquet`)
    - `metadata_path: Path` (`<export_dir>/metadata.parquet`)
    - `manifest_path: Path` (`<export_dir>/manifest.json`)
    - `shard_index: dict[str, int]`
    - `manifest: ExportManifest`
  - Helper function (module-level, not a method): `_hash_export_data(export_dir: Path) -> str`
    - Walks `graphs/`, `tokens/`, `labels.parquet`, `metadata.parquet` (NOT `manifest.json`)
    - For each file, computes SHA-256, prepends the relative path, hashes the concatenation
    - Returns the final hex digest
    - Used by both `chunk_export` (to populate `manifest.artifact_hash`) and `SentinelDatasetExport.verify_artifact_hash` (to verify)
  - Method: `verify_artifact_hash() -> bool`
    - Calls `_hash_export_data(self.export_dir)` to recompute
    - Compares to `self.manifest.artifact_hash` (the field renamed from `compute_hash` — see Step 4 note above)
    - Returns True if they match
    - Importing `compute_hash` from `sentinel_data.registry.catalog` would shadow the field name; use a local alias like `_compute_dir_hash` or `_hash_export_dir` to keep the import and the method readable
  - Method: `get_split_contract_ids(split: str) -> list[str]`
  - Method: `__repr__` for debugging

- [ ] **Modify `data_module/sentinel_data/export/__init__.py`** (Fix C — the 10-line stub stays empty otherwise)
  - Export the public API: `SentinelDatasetExport`, `ExportManifest`, `chunk_export`, `write_labels_parquet`, `write_metadata_parquet`, `write_graphs_shards`, `write_tokens_shards`
  - Keep the docstring with the format_schema reference
  - Re-export `SentinelDatasetExport` so callers can do `from sentinel_data.export import SentinelDatasetExport`

- [ ] Test: `SentinelDatasetExport` can be constructed from a test export dir
- [ ] Test: `verify_artifact_hash()` returns True on a fresh export, False if any **data file** (.pt shard, .parquet) is modified
- [ ] Test: **adding/editing `manifest.json` does NOT change the hash** (regression test for Fix A — proves manifest.json is excluded from the hash input)
- [ ] Test: `get_split_contract_ids("train")` returns the same 15,644 ids that `splits/v1/train.jsonl` has
- [ ] Test: `from sentinel_data.export import SentinelDatasetExport, chunk_export` works (Fix C — `__init__.py` exports)

---

## Step 5 — Wire `_run_export` CLI

- [ ] Read `data_module/sentinel_data/cli.py` lines 632-638 (the current stub) and lines 525-666 (argparse setup for the export stage) to understand what arguments are already wired
- [ ] Add argparse arguments to the `export` stage:
  - `--dataset-version <name>` (default: latest registered version from catalog; or None = use current build)
  - `--split-version <N>` (default 1)
  - `--output-dir <path>` (default: `data/exports/<dataset_version>/`)
  - `--shard-size <N>` (default 5000)
  - `--no-shard-graphs` (skip graph shards — for label-only exports)
  - `--no-shard-tokens` (skip token shards)
- [ ] Implement `_run_export(args)` in `data_module/sentinel_data/cli.py`
  - Read the catalog to find the dataset version (if `--dataset-version` given)
  - Resolve the labels, representations, preprocessed, splits dirs from the registered artifact_path
  - Call `chunk_export(...)` from Step 4
  - Print a summary: n_contracts, n_shards, output_path
  - If `--dataset-version` given, update the catalog's artifact_hash for that version
- [ ] Test: `sentinel-data export --help` shows the new flags
- [ ] Test: `sentinel-data export --dataset-version sentinel-v2-dryrun-2026-08 --dry-run` prints the plan

---

## Step 6 — Predictor tier bug fix (surgical, in-place, separate commit)

- [ ] Read `ml/src/inference/predictor.py:670-757` to find where `confirmed` and `suspicious` are computed
- [ ] Re-read `ml/src/inference/predictor.py:298-336` to see how `self.thresholds` is loaded (per-class)
- [ ] Identify the exact fix: where the tier logic uses `self.tier_confirmed_threshold` (scalar) instead of `self.thresholds[predicted_class_idx]` (per-class)
- [ ] Apply the fix:
  - For each predicted class, use `self.thresholds[class_idx].item()` as the confirmed threshold
  - For suspicious: keep a class-wide scalar (per 7-P7) OR add a `self.tier_suspicious_thresholds` per-class list (preferred for consistency)
  - Decision: read the actual code first, then decide whether suspicious also needs per-class
- [ ] **Create new** `ml/tests/test_predictor.py` — **Fix D**: this file does not exist yet (the existing ml/tests/ files are `test_preprocessing.py`, `test_model.py`, etc.; there's nothing to extend). The new file will contain the tier-threshold regression tests:
  - Create a mock predictor with `thresholds = [0.9, 0.5, 0.5, ...]`
  - Run a prediction where `predicted_class_idx=0` (high-tuned threshold class)
  - Assert that the "confirmed" tier triggers at prob=0.92 but NOT at prob=0.85 (would have triggered with the old hardcoded 0.55)
  - Reverse for a low-tuned-threshold class
  - Use a mock for the model forward pass; do not depend on GPU or on real model weights
- [ ] Run the new `ml/tests/test_predictor.py` to confirm it passes
- [ ] Run the broader `ml/tests/` suite to confirm no regression (e.g. `test_preprocessing.py` shouldn't be affected by the predictor change)
- [ ] Commit as a separate, focused commit (this is an ml/ change; not in the data_module's commit history)

---

## Step 7 — Tests for the export module

- [ ] Create `data_module/tests/test_export/` package (with `__init__.py`)
- [ ] Test file: `data_module/tests/test_export/test_label_writer.py`
  - 10 synthetic labels with varied class distributions
  - Write to tmp, read back, assert column dtypes
  - Assert that `confidence_tier` is correctly inferred
- [ ] Test file: `data_module/tests/test_export/test_metadata_writer.py`
  - 10 synthetic contracts with .rep.json + .meta.json + .sol sidecars
  - Write to tmp, read back
  - Assert that `loc`, `n_functions` are computed correctly
  - Assert that `version_bucket`, `dedup_group_id` come through from .meta.json
- [ ] Test file: `data_module/tests/test_export/test_graph_token_writer.py`
  - Synthesize 12 PyG `Data` objects + 12 torch.Tensor
  - Shard with size=5 → 3 shards (5+5+2)
  - Assert shard index maps contract_ids to correct shard numbers
- [ ] Test file: `data_module/tests/test_export/test_chunker.py`
  - End-to-end: 50 synthetic contracts → chunk_export → verify all 4 file types + manifest
  - Verify SHA-256 of the 4 data file types (NOT including manifest.json) matches `manifest.artifact_hash` (NOT `manifest.compute_hash` — see Fix #4)
  - **Verify manifest.json is written LAST** (regression for Fix A): after chunk_export returns, mtime of manifest.json is later than mtime of all data files
- [ ] Test file: `data_module/tests/test_export/test_export.py`
  - `SentinelDatasetExport` roundtrip
  - `verify_artifact_hash()` returns True / False correctly
  - Tampering test: modify one byte in a .pt file → hash mismatch
- [ ] Integration smoke (real data, but small) — **Fix B**: don't use the first 100 split rows; only 3 of the first 100 train rows have representations (the split JSONL is sorted alphabetically by sha256 and most early hashes are dive contracts without reps). The smoke test would silently produce mostly-empty shards and fail the count cross-check.
  - **Approach 1 (preferred)**: filter to contracts that have a `.pt` file in `rep_root` — there are 776 such contracts total in the v2 baseline (the run is currently 3.5% complete). The smoke test exercises the full pipeline on these.
  - **Approach 2 (fallback if Approach 1 yields < 10 contracts for some reason)**: filter to the `solidifi` source (~276 reps available) — solidifi has a much higher rep-coverage rate than dive.
  - Run the full `chunk_export` on the filtered set
  - Read back `labels.parquet`, verify the row count matches the number of filtered contracts
  - Read back `metadata.parquet`, verify the row count matches
  - Verify `manifest.splits` has the right per-split counts
  - Verify `SentinelDatasetExport.verify_artifact_hash() == True`
- [ ] Verify all tests pass: `source ml/.venv/bin/activate && python -m pytest data_module/tests/test_export/ -v`
- [ ] Verify full data_module test suite is still green: `source ml/.venv/bin/activate && python -m pytest data_module/tests/ -q`

---

## Step 8 — ADR-0008 (export + seam swap design)

- [ ] Read `docs/decisions/ADR-0007-analysis-design.md` (Stage 6's ADR) for the format. Note: the file is at `docs/decisions/ADR-0007-*.md`, NOT at `docs/ml/adr/`. The `docs/ml/adr/` directory is for ML-side ADRs and has a different file (`0007-representation-port-design.md`) with a different number. Don't conflate them.
- [ ] Author `docs/decisions/ADR-0008-export-and-seam-swap-design.md`
  - Title: "ADR-0008: Export + Seam Swap Design"
  - **Sub-stage 7A scope only** (this is the explicit scope clarification; the seam swap is deferred to 7B)
  - Design decisions captured:
    - **D-7.1** — Sharded export is the consumable format (4 file types: graphs_shard, tokens_shard, labels_parquet, metadata_parquet)
    - **D-7.3** — The new `SentinelDataset` is thin (~150 lines) — noted as future work (7B)
    - **D-7.4** — The ML module's `pyproject.toml` adds `sentinel-data` as a runtime dep — noted as future work (7B)
    - **NEW (IC-1)** — Format schema versioning: `format_schema/v{MAJOR}.{MINOR}.yaml`. v1.0 is the current contract. Future bumps (v1.1, v2.0) are new files. The `SentinelDatasetExport` loads the schema version from the manifest and refuses to load if the consumer's pinned version is older.
    - **NEW (IC-2)** — Schema-version gate: the manifest carries `graph_schema_version: "v9"` and the new `SentinelDataset` (in 7B) will refuse to load if the model's trained version doesn't match. This is the 7-P2 gate. Documented here for forward reference even though 7B builds the consumer.
    - **NEW (IC-3)** — Confidence tier as a first-class field: `labels.parquet` and `metadata.parquet` both have a `confidence_tier` column (T0/T1/T2/T3/None). The Stage 5 splits' `tier` field is the source. Per 7-P4, this enables per-class loss weighting in Run 11.
    - **NEW (IC-4)** — Predictor tier bug fix scope: the per-class `self.thresholds[predicted_class_idx]` fix is in-scope for 7A because it's a standalone surgical fix that doesn't depend on the seam swap. The fix is one line in `_format_result` plus a unit test.
    - **NEW (IC-5)** — Defer seam swap to 7B: the dual-path test (7.7) and file deletion (7.8) are NOT in 7A's scope. The reason: 7A produces a v2 export that the existing `dual_path_dataset.py` cannot load (different format), so swapping the loader without a fallback would break the active Run 9 training pipeline. 7B's dual-path test is the safety net; we cannot build that test in 7A because 7A doesn't have the new loader yet.
  - Implementation choices made during 7A
  - Operational consequences
  - References to all 7 source files + 5 test files
- [ ] Commit: `docs(stage7): ADR-0008 — export module + seam-swap scope clarification`

---

## Step 9 — Learning doc + checklist update

- [ ] Re-read `docs/proposal/Data_Module_Proposals/actionable_plans/learning_docs/stage_4_verification.md` and `stage_5_splitting_registry.md` for the format
- [ ] Rewrite `docs/proposal/Data_Module_Proposals/actionable_plans/learning_docs/stage_7_export_seam.md`:
  - Title: "Stage 7 — Export + Seam Swap (Sub-stage 7A complete; 7B deferred)"
  - Status: "✅ 7A COMPLETE (export module + CLI + predictor fix). 7B deferred to next session."
  - Standard 6-section format (Problem, Solution, Context, Verification, Got-it checklist, Read next)
  - "What was actually built" section with file/line/LoC counts
  - "What's deferred to 7B" section with explicit list
  - "Got it" checklist with 8-9 questions (including the predictor tier fix question)
- [ ] Update `docs/proposal/Data_Module_Proposals/actionable_plans/learning_docs/LEARNING_CHECKLIST.md`:
  - Mark `## Stage 7 — Export + Seam Swap` as `[x] COMPLETE (7A only — 7B deferred)`
  - Tick the 7 mastery checkboxes
  - Add a new sub-section: `### 7B — Seam Swap (deferred to next session)` with the open items
  - Update the "Current build state" line at the top
- [ ] Commit: `docs(stage7): rewrite learning doc + update checklist for 7A`

---

## Step 10 — Commit everything in focused commits

Order of commits (each focused, each testable, each with a meaningful message):

1. `feat(stage7): add format_schema/v1.yaml — the export format contract` (Step 1)
2. `feat(stage7): add label_writer + metadata_writer (parquet output)` (Step 2)
3. `feat(stage7): add graph_writer + token_writer (sharded .pt output)` (Step 3)
4. `feat(stage7): add chunker + SentinelDatasetExport consumer API` (Step 4)
5. `feat(stage7): wire _run_export CLI with --dataset-version + --split-version` (Step 5)
6. `test(stage7): add full test suite for the export module` (Step 7)
7. `fix(ml): predictor tier threshold uses per-class self.thresholds[cls_idx] (F8/F10)` (Step 6)
8. `docs(stage7): ADR-0008 — export + seam-swap scope (7A only, 7B deferred)` (Step 8)
9. `docs(stage7): rewrite learning doc + update checklist for 7A` (Step 9)

---

## Open questions / decisions to make during execution

- [ ] **Pyarrow version** — confirm what's in the .venv; document the pin
- [ ] **PyG `Batch` import path** — check `torch_geometric.data.Batch` is the right path; check the `from_data_list` constructor
- [ ] **torch.load weights_only** — do we need to register safe globals for PyG types? Check `ml/src/datasets/dual_path_dataset.py:82-87` for the pattern
- [ ] **Predictor tier fix: suspicious per-class or scalar?** — read the code, decide based on what the existing test expects
- [ ] ~~**Should the export include NonVulnerable contracts?**~~ — DECIDED: yes. All 22,356 split contracts (including NonVulnerable) get rows in `labels.parquet` and `metadata.parquet`. Only the .pt shards skip NonVulnerable (and any other contract with no representation) — see Fix #7. NonVulnerable contracts are needed for the NonVuln:positive ratio in the loss.
- [ ] **What happens to a contract that's in a split but has no representation?** — warn and skip (don't fail). The `chunk_export` will report skipped count.
- [ ] **Schema v9 manifest field** — should we use `"v9"` (string) or `9` (int)? String for forward compat (v9.1, v10).

---

## Risks / gotchas to watch for

- [ ] **PyG `Batch` constructor** — `from_data_list` may fail if contracts have inconsistent node/edge types. Check what the per-contract graphs look like first.
- [ ] **memap for tokens** — the .tokens.pt files are `[4, 512]` per contract; for 5,000 contracts that's `[20000, 512]` ≈ 40MB per shard. Manageable.
- [ ] **Hash determinism** — the manifest SHA-256 must be stable across runs. Sort the file list before hashing; use canonical JSON.
- [ ] **Catalog version mismatch** — the `--dataset-version` lookup must hit the catalog atomically. If the catalog is being written by a parallel `sentinel-data register` call, we may read stale state. Document this as a known limitation (DVC serializes stages anyway).
- [ ] **Predictor test isolation** — the new tier test must not depend on GPU or on the actual model weights. Use a mock or a synthetic threshold list.
- [ ] **The `confidence_tier` for NonVulnerable contracts** — **DECIDED: `tier if n_pos > 0 else None` (pyarrow null).** The splitter's default `tier="T0"` is overwritten to `None` for NonVulnerable contracts. This way the model code can filter on `confidence_tier.is_null()` to skip tier-based weighting for negatives. The merge rule in `pipeline.class.merge_rules` in `config.yaml` already encodes this; the export is consistent.

---

## Files I'll create (8 source + 2 modify-source + 5 test + 1 schema + 2 docs = 18 files)

Source:
1. `data_module/sentinel_data/export/format_schema/v1.yaml`
2. `data_module/sentinel_data/export/label_writer.py`
3. `data_module/sentinel_data/export/metadata_writer.py`
4. `data_module/sentinel_data/export/graph_writer.py`
5. `data_module/sentinel_data/export/token_writer.py`
6. `data_module/sentinel_data/export/chunker.py`
7. `data_module/sentinel_data/export/export.py`
8. (modify) `data_module/sentinel_data/export/__init__.py` — **Fix C**: export the public API (`SentinelDatasetExport`, `ExportManifest`, `chunk_export`, `write_labels_parquet`, `write_metadata_parquet`, `write_graphs_shards`, `write_tokens_shards`)
9. (modify) `data_module/sentinel_data/cli.py` `_run_export`

Tests:
9. `data_module/tests/test_export/__init__.py`
10. `data_module/tests/test_export/test_label_writer.py`
11. `data_module/tests/test_export/test_metadata_writer.py`
12. `data_module/tests/test_export/test_graph_token_writer.py`
13. `data_module/tests/test_export/test_chunker.py`
14. `data_module/tests/test_export/test_export.py`

Predictor fix (separate package):
15. (modify) `ml/src/inference/predictor.py`
16. (create new) `ml/tests/test_predictor.py` — **Fix D**: this file does not exist; must be created

Docs:
17. `docs/decisions/ADR-0008-export-and-seam-swap-design.md`
18. (modify) `docs/proposal/Data_Module_Proposals/actionable_plans/learning_docs/stage_7_export_seam.md`
19. (modify) `docs/proposal/Data_Module_Proposals/actionable_plans/learning_docs/LEARNING_CHECKLIST.md`

---

## What this plan does NOT do (deferred to 7B)

- [ ] Author `sentinel-ml/src/datasets/sentinel_dataset.py` (the new ~150-line loader)
- [ ] Author the dual-path test (old vs new loader byte-identical for 100 contracts)
- [ ] Delete `ml/src/datasets/dual_path_dataset.py`
- [ ] Delete `ml/src/preprocessing/{graph_extractor,graph_schema}.py` (note: `ml/data/README.md` still references them)
- [ ] Delete `ml/scripts/{reextract_graphs,retokenize_windowed,build_multilabel_index,create_splits,create_cache,validate_graph_dataset,archive_v8_data}.py`
- [ ] Archive the deleted scripts to `ml/scripts/_legacy_data_pipeline/` (per 7-P9)
- [ ] Update `ml/src/inference/preprocess.py` to import from `sentinel_data.representation`
- [ ] Update `ml/pyproject.toml`: add `sentinel-data = "^0.1.0"`, remove `solc-select`, `py-solc-ast`, `solc`; KEEP `slither-analyzer` (per 7-P8)
- [ ] Update `ml/README.md` to remove the "Data Pipeline" section
- [ ] Fix the EMITS edge bug if it exists (need to investigate whether the 12-edges symptom is data-side or code-side)
- [ ] Docker build verification (`python:3.12.1-bookworm` per 7-P10)
- [ ] 7 v2-readiness gates check (per 7-P11)
- [ ] Run 11 (Stage 8)

---

## Definition of done for 7A

- [ ] `format_schema/v1.yaml` exists and is the authoritative spec
- [ ] All 4 writers produce valid output for a small synthetic fixture
- [ ] `chunk_export` runs end-to-end on synthetic data
- [ ] `SentinelDatasetExport` roundtrips correctly; `verify_artifact_hash` works
- [ ] `sentinel-data export --dry-run` prints the plan
- [ ] `sentinel-data export` (on the real v2 baseline, after Stage 2 completes) produces a valid export
- [ ] All 7A tests pass (target: ~25-30 tests across the 5 test files)
- [ ] Full data_module test suite is still green (no Stage 6 regression)
- [ ] Predictor tier fix landed in a separate commit
- [ ] ADR-0008 written; learning doc + checklist updated
- [ ] All commits pushed; `LEARNING_CHECKLIST.md` says `7A COMPLETE, 7B DEFERRED`
- [ ] 7B is documented as the next session's work
