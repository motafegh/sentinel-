# ADR-0008: Export + Seam Swap Design

**Date:** 2026-06-12
**Stage:** 7A of 8 (sub-stage: export module only; 7B seam swap deferred)
**Status:** Accepted (Stage 7A implementation complete)
**Author:** SENTINEL data engineering
**Plan reference:** [`docs/proposal/Data_Module_Proposals/actionable_plans/08_stage_7_export_seam.md`](../proposal/Data_Module_Proposals/actionable_plans/08_stage_7_export_seam.md)
**Live plan:** [`data_module/temp/live_plans/stage_7a_export_module.md`](../../data_module/temp/live_plans/stage_7a_export_module.md)

---

## Context

Stage 7 is the final data-module stage. Its two responsibilities are distinct:

1. **7A (this ADR):** Produce the consumable export artifact — sharded `.pt` files + Parquet metadata — that the ML module's `SentinelDataset` (Stage 7B) will load. This is the "write side" of the seam.
2. **7B (deferred):** Replace the ML module's `DualPathDataset` + legacy data scripts with the new `SentinelDataset` loader that reads the v2 export. This is the "read side" of the seam.

The seam is the boundary between the data module (produces the corpus) and the ML module (consumes it). The v1 seam was raw `.pt` files read by `dual_path_dataset.py` using legacy paths. The v2 seam is the export directory produced by `chunk_export()`.

**Why split 7A and 7B?** Stage 7A produces a new format that `dual_path_dataset.py` cannot load. Deleting `dual_path_dataset.py` without first building and validating the new `SentinelDataset` (7B's dual-path test) would break the active training pipeline with no fallback. The split lets 7A land cleanly while 7B adds the new loader with byte-identical verification.

---

## Design Decisions

### D-7.1 — Sharded export is the consumable format (4 file types)

The export artifact has 4 file types per export directory:
- **`graphs/graphs-{shard:05d}.pt`** — PyG `Batch` objects, one shard of up to 5,000 contracts
- **`tokens/tokens-{shard:05d}.pt`** — `torch.Tensor [N, 4, 512]`, same ordering
- **`labels.parquet`** — one row per labeled contract (all 22,356 rows, even without reps)
- **`metadata.parquet`** — same coverage, enriched with `.rep.json` + `.meta.json` + `.sol`

The shard size (5,000) is configurable via `pipeline.export_shard_size` in `config.yaml`.

**Why not a single file?** A single `.pt` file holding all 22K contracts would require loading the entire corpus into RAM on every training run. Shards enable lazy loading: the `SentinelDatasetExport.shard_index` maps each `contract_id` to its shard number, and the ML module loads only the required shard for each batch.

**Why Parquet for labels/metadata?** Parquet enables column-projection queries without reading the full file, supports nullable types (the `confidence_tier` column is nullable for NonVulnerable contracts), and interoperates with pandas/polars for exploratory analysis. Snappy compression is the default (widely supported, fast decompression).

**Operational consequence:** the ML module (`SentinelDataset`, built in 7B) receives a 5-tuple per contract: `(graph, tokens, y, contract_id, confidence_tier)`. The `confidence_tier` column is new — it enables per-class loss weighting in Run 11 (the Stage 5 tier carries downstream into training).

### D-7.2 — Contract order is driven by the split JSONL, not the representations directory

Both `graph_writer` and `token_writer` walk `splits/v{N}/{train,val,test}.jsonl` in order (train → val → test, then line order within each file). They do **not** glob the representations directory.

**Why:** if we globbed `representations/`, the order would be filesystem-dependent (directory listing order varies by OS, filesystem, and inode allocation). The split JSONL is the canonical, deterministic ordering. This guarantees that shard row N in `graphs-{shard:05d}.pt` corresponds exactly to row N in `tokens-{shard:05d}.pt`, and that the `shard_index` maps each `contract_id` to the correct position.

**Consequence:** contracts in the split that have no representation are silently skipped (warn only). The `labels.parquet` and `metadata.parquet` still include all 22,356 split rows; the `.pt` shards include only the 21,523 contracts that have representations.

### D-7.3 — The new `SentinelDataset` is thin (~150 lines, built in 7B)

The export API (`SentinelDatasetExport`) is deliberately minimal: it loads the manifest, exposes the shard index, and provides `verify_artifact_hash()`. It does NOT implement `__len__` or `__getitem__` — that's the ML module's job in 7B.

**Why separate:** the data module is write-only at runtime; the ML module is read-only. Putting `__getitem__` logic in the data module would couple training hyperparameters (batch size, augmentation) to the data pipeline. The separation keeps both modules independent and lets 7B evolve the loader without touching the export format.

### D-7.4 — `ml/pyproject.toml` adds `sentinel-data` as a runtime dep (7B, deferred)

The ML module currently imports the data module's code via thin adapters (`sentinel_data.representation.graph_extractor` → `ml/src/preprocessing/graph_extractor.py`). In 7B, the adapter is replaced by a direct import; `ml/pyproject.toml` gets `sentinel-data = "^0.1.0"`. This is noted here for forward reference; it is not implemented in 7A.

### IC-1 — Format schema versioning: `format_schema/v{MAJOR}.{MINOR}.yaml`

The format spec lives at `data_module/sentinel_data/export/format_schema/v1.yaml`. This file is the contract. Future schema changes:
- **Additive (backward-compatible):** new file `v1.1.yaml`. The consumer can load v1 exports with a v1.1 consumer (old fields still present, new fields silently absent).
- **Breaking:** new file `v2.yaml`. The consumer refuses to load v1 exports (schema_version gate, per below).

The `manifest.json` carries `schema_version: "v1"`. The `SentinelDataset` (7B) checks this field on load and raises `ValueError` if the export's version is older than the consumer's pinned version.

### IC-2 — Schema-version gate (v9 graph schema + v1 format schema)

The manifest carries two version fields:
- `graph_schema_version: "v9"` — the graph feature schema (12-dim node features, 12 edge types). The `SentinelDataset` checks this against `ml/src/preprocessing/graph_schema.FEATURE_SCHEMA_VERSION` at load time and refuses to load on mismatch. This is the 7-P2 gate.
- `schema_version: "v1"` — the export format contract. Same gate logic.

Documented here for forward reference even though 7B builds the consumer-side check.

### IC-3 — Confidence tier as a first-class field (per 7-P4)

Both `labels.parquet` and `metadata.parquet` include a `confidence_tier` column. The logic (Fix #2): `tier if n_pos > 0 else None`. The split JSONL's `tier="T0"` default for NonVulnerable contracts is overridden to `null` (pyarrow null) at the parquet level, so the ML module can filter on `confidence_tier.is_null()` to skip tier-based weighting for negatives.

The Stage 5 tier (T0/T1/T2/T3 or "untiered") flows through unchanged for positive contracts. This enables per-class loss weighting in Run 11 — contracts labeled by tool agreement (T3) carry less weight than those labeled by manual review (T1).

### IC-4 — Predictor tier bug fix (F8/F10), in-scope for 7A

The `ml/src/inference/predictor.py` bug was using `self.tier_confirmed_threshold` (a scalar class constant, 0.55) for the CONFIRMED tier decision for every class, ignoring the per-class thresholds loaded from the companion JSON. The fix uses `self.thresholds[cls_idx].item()` per class. This is a surgical, standalone fix that does not depend on the seam swap, so it lands in 7A.

**Impact:** a class tuned to 0.90 (e.g. CallToUnknown) was incorrectly triggering CONFIRMED at prob=0.56. A class tuned to 0.35 (e.g. Reentrancy) was not triggering CONFIRMED at prob=0.36. Both are now correct.

Also: `tier_thresholds["confirmed"]` in the result dict now returns the full per-class threshold list (not the scalar), so the result dict is self-documenting.

### IC-5 — Seam swap deferred to 7B

The deletion of `dual_path_dataset.py`, `graph_extractor.py` (from `ml/`), and the 7 legacy data scripts is NOT in 7A's scope. The dual-path test (byte-identical check for 100 contracts) is the safety net for the swap; 7A cannot build that test because 7A doesn't have the new loader yet. 7B's scope:

- Author `sentinel-ml/src/datasets/sentinel_dataset.py` (~150 lines)
- Dual-path test (old vs new loader, byte-identical for 100 contracts)
- Delete `ml/src/datasets/dual_path_dataset.py`
- Delete `ml/src/preprocessing/graph_extractor.py` + `graph_schema.py` (from `ml/`)
- Archive legacy scripts to `ml/scripts/_legacy_data_pipeline/`
- Update `ml/pyproject.toml`
- Docker build verification
- 7 v2-readiness gates check

---

## Implementation choices made during 7A

### Fix A — Circular hash avoidance

The `artifact_hash` in `manifest.json` is the SHA-256 of the 4 data file types only (`graphs/`, `tokens/`, `labels.parquet`, `metadata.parquet` + shard index files). `manifest.json` is excluded. This avoids the chicken-and-egg: manifest.json contains `artifact_hash`, so it cannot be part of the file set whose hash it stores. `manifest.json` is written LAST after the hash is computed.

`_hash_export_data()` in `chunker.py` is the single implementation used by both `chunk_export` (to populate the field) and `SentinelDatasetExport.verify_artifact_hash()` (to verify). The test suite includes a regression test: modifying `manifest.json` does NOT change the hash.

### Fix B — Smoke test uses contracts with existing `.pt` files

The integration smoke test does not use "the first 100 rows of the split JSONL" — only 3 of the first 100 train rows have representations (the JSONL is sorted alphabetically by sha256, and most early hashes are dive contracts that were unrepresented in the 776-contract pre-run baseline). The smoke uses the full export path with the full 21,523-contract corpus.

### Fix C — `export/__init__.py` exports the public API

The `__init__.py` (previously a docstring stub) now exports: `SentinelDatasetExport`, `ExportManifest`, `chunk_export`, `write_labels_parquet`, `write_metadata_parquet`, `write_graphs_shards`, `write_tokens_shards`. Callers can do `from sentinel_data.export import SentinelDatasetExport`.

### Fix D — `ml/tests/test_predictor.py` is new (no existing file to extend)

The plan noted "or extend existing" for the predictor tests — there was no existing `test_predictor.py`. 6 regression tests were written from scratch.

---

## Files built in 7A

**Source (8 new + 2 modified):**
1. `data_module/sentinel_data/export/format_schema/v1.yaml` — format contract
2. `data_module/sentinel_data/export/label_writer.py` — `write_labels_parquet()`
3. `data_module/sentinel_data/export/metadata_writer.py` — `write_metadata_parquet()`
4. `data_module/sentinel_data/export/graph_writer.py` — `write_graphs_shards()`
5. `data_module/sentinel_data/export/token_writer.py` — `write_tokens_shards()`
6. `data_module/sentinel_data/export/chunker.py` — `chunk_export()` + `ExportManifest`
7. `data_module/sentinel_data/export/export.py` — `SentinelDatasetExport`
8. `data_module/sentinel_data/export/__init__.py` (modified) — public API exports
9. `data_module/sentinel_data/cli.py` (modified) — `_run_export` + argparse wiring

**Tests (5 new files, 27 tests):**
10. `data_module/tests/test_export/__init__.py`
11. `data_module/tests/test_export/test_label_writer.py` (5 tests)
12. `data_module/tests/test_export/test_metadata_writer.py` (4 tests)
13. `data_module/tests/test_export/test_graph_token_writer.py` (5 tests)
14. `data_module/tests/test_export/test_chunker.py` (5 tests)
15. `data_module/tests/test_export/test_export.py` (8 tests)

**Predictor fix (separate ML commit):**
16. `ml/src/inference/predictor.py` (modified) — per-class `thresholds[cls_idx]` for CONFIRMED tier
17. `ml/tests/test_predictor.py` (new) — 6 regression tests

**Docs:**
18. `docs/decisions/ADR-0008-export-and-seam-swap-design.md` (this file)

---

## Operational consequences

- `sentinel-data export --dry-run` now resolves sources, prints the plan, and exits without writing.
- `sentinel-data export` writes the full export to `data/exports/<dataset-version>/`.
- `sentinel-data export --dataset-version sentinel-v2-gold-2026-08` uses that name as the output directory label.
- The CLI skips sources with no preprocessed dir (same guard as `sentinel-data represent`).
- The full data_module test suite is 531 passed / 51 skipped after 7A.
- The ML test suite passes with the predictor fix (6 new + existing tests pass).

---

## What this ADR does NOT cover (deferred to 7B)

See IC-5 above for the complete 7B scope. Key items:
- `SentinelDataset.__getitem__` (the read-side consumer)
- Dual-path test (byte-identical for 100 contracts)
- Deletion of legacy ML data scripts
- Docker build verification
- 7 v2-readiness gates check
- Run 11 launch
