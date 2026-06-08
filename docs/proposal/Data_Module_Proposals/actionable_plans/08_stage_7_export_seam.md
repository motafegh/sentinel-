# Actionable Plan — Stage 7: Export + Seam Swap (the ML-module integration)

**Date:** 2026-07-23
**Stage:** 7 of 8 (Week 7 part B + Week 8: Jul 23–Aug 4)
**Owner:** SENTINEL data engineering
**Source proposal:** `docs/proposal/Data_Module_Proposals/Sentinel_v2_Data_Module_Integration_Proposal.md` §3.8, §4, §5 (Week 7–8), §8, §9
**Audit ref:** [`AUDIT_PATCHES.md`](AUDIT_PATCHES.md) §0 (F7, F8, F10, F14), §1 (7-P1 through 7-P11)
**Exit criteria:** the export→import round-trip is byte-identical; `sentinel-ml` loads a v2 export in < 60 s on a 1.0 GB sharded artifact; the `SentinelDataset` runs a 1-batch forward pass without error; the seam swap is complete (old `dual_path_dataset.py` and old `ml/src/preprocessing/{graph_extractor,graph_schema}.py` are deleted; new `sentinel-ml/src/datasets/sentinel_dataset.py` is the active loader); the Docker build succeeds and runs; **the predictor.py tier-threshold bug is fixed (F8/F10); the EMITS edge bug is fixed (F23); the deleted scripts are archived to `ml/scripts/_legacy_data_pipeline/` (not just deleted); all 7 v2-readiness gates (including the 36-issue regression) are GREEN**.

---

## Goal

Implement the **Export** submodule (sharded graph/token/label/metadata writers + the format schema) and perform the **seam swap** — replacing `ml/src/datasets/dual_path_dataset.py` with a thin `sentinel-ml/src/datasets/sentinel_dataset.py` that reads from the v2 export, deleting the old `ml/src/preprocessing/{graph_extractor,graph_schema}.py` files, and wiring `sentinel-ml`'s `pyproject.toml` to depend on `sentinel-data`.

After this stage, the two packages are integrated, the boundary is enforced, and Run 11 can launch on the v2 corpus.

This is the longest and most complex stage in the build because it touches both packages and the active training pipeline must not break.

---

## Why this stage last

Stages 1–6 produced preprocessed, represented, labeled, verified, split, registered, and analyzed data — but none of it in a form the ML module can consume. Stage 7 is the first stage that produces the *consumable* artifact: the sharded export that the ML module's `SentinelDataset` reads. It is also the stage that closes the loop by performing the seam swap — the moment when the old import paths in `ml/` are removed and the new ones take over.

Doing the export + seam swap in the same stage (rather than separating them) is critical because the seam swap *requires* the export to work. A seam swap against a broken export would break the active Run 9 training pipeline.

---

## Design decisions

### D-7.1 — Sharded export is the consumable format

The export produces 4 file types per shard: graphs (PyG `Batch`), tokens (torch.Tensor), labels (parquet), metadata (parquet). Shard size is 5,000 contracts by default (in `config.yaml`). The shard index maps `contract_id → shard_number` so the ML module can lazy-load a single contract.

The format is the contract per the proposal §3.8 — `sentinel_data/export/format_schema/v1.yaml` is the spec. Any future schema bump (v1.1, v2.0) is a new file; old files are kept for backward compatibility.

### D-7.2 — The seam swap is a 3-step operation, gated by the byte-identical test

The seam swap has 3 steps, each gated by a test:
1. **New writer in `sentinel-data`, new reader in `sentinel-ml`** — the new reader is `sentinel-ml/src/datasets/sentinel_dataset.py`, which takes a `SentinelDatasetExport` and returns a torch Dataset. The new reader does *not* delete the old `dual_path_dataset.py` yet.
2. **Dual-path test** — for 100 contracts, the old `dual_path_dataset.py` and the new `sentinel_dataset.py` produce identical batches. This is the gate.
3. **Delete the old** — `dual_path_dataset.py` is deleted, the old `ml/src/preprocessing/{graph_extractor,graph_schema}.py` files are deleted, the old `ml/scripts/{reextract_graphs,retokenize_windowed,build_multilabel_index,create_splits,create_cache,validate_graph_dataset,archive_v8_data}.py` are deleted. The `ml/src/inference/preprocess.py` is updated to import from `sentinel_data.representation` instead of `src.preprocessing`.

The byte-identical test from Stage 2 is the second gate. If it fails at any point, the seam swap is reverted and debugged.

### D-7.3 — The new `SentinelDataset` is thin (~150 lines)

The new `sentinel-ml/src/datasets/sentinel_dataset.py` does only:
- Accept a `SentinelDatasetExport` (paths + manifest + split name)
- Call `sentinel_data.registry.verify_artifact_hash()` on the export path
- Lazy-load the shard index
- For each `__getitem__`, find the contract's shard, read the graph + tokens + label, return them
- The collate function is the existing `dual_path_collate_fn` (~50 lines extracted to `collate.py`)

The dataset adds a new field to the returned tuple: `confidence_tier` (from the labels JSON). This is the input to the per-class loss weighting that Run 11 will use.

### D-7.4 — The ML module's `pyproject.toml` adds `sentinel-data` as a runtime dependency

The ML module's `pyproject.toml` adds `sentinel-data = "^0.1.0"` (path dep for dev; PyPI dep for production). The ML module's lockfile is updated. The ML module's `src/inference/preprocess.py` is updated to import `extract_contract_graph` from `sentinel_data.representation` instead of `src.preprocessing`. The ML module's `src/datasets/dual_path_dataset.py` is deleted; the new `sentinel_dataset.py` is the active loader.

### D-7.5 — Docker build is verified at the end of Stage 7

The Dockerfile from Stage 0 is built and tested in Stage 7. The test: `docker build` succeeds; `docker run --rm sentinel-data:0.1.0 --help` returns the expected output; `docker run --rm sentinel-data:0.1.0 run --dry-run` lists all 9 stages.

This is the only Docker test in the entire build. A passing Docker test is the proof that the v2 corpus can be reproduced on any machine.

### D-7.6 — The 6 verification gates are checked at the end of Stage 7

Per the proposal §9, the v2 build is "ready to train Run 11" when all 6 gates pass. Stage 7 ends with a final check that all 6 are green:
1. Schema regression test (from Stage 2) — passes
2. Phase 5 BCCC regression test (from Stage 4) — passes
3. End-to-end round-trip (this stage) — passes
4. Feature distribution report (from Stage 6) — GREEN (no class-pair > 1.5σ)
5. All 10 classes pass verification gate (from Stage 4) — VERIFIED or PROVISIONAL
6. No leakage across splits (from Stage 5) — 0 near-duplicates

If any gate fails, Stage 7 is not complete and Run 11 does not launch.

---

## Tasks — ordered, each with verifiable exit condition

### 7.1 — Author `sentinel_data/export/format_schema/v1.yaml`

Author the format schema spec per the proposal §3.8. The spec defines the 4 file types (graphs_shard, tokens_shard, labels_parquet, metadata_parquet) and the required `manifest` fields. The spec is the contract; any future schema bump is a new file.

**Why first:** the writers (7.2–7.5) all conform to this schema; the spec must exist before the writers.

**Exit condition:** file exists with all 4 file type specs and the required manifest fields documented.

**Commit:** `feat(data-export): add format_schema/v1.yaml contract`

---

### 7.2 — Implement `graph_writer.py` and `token_writer.py`

Author the two shard writers. The graph writer reads the graph `.pt` files from `data/representations/<source>/` and writes sharded PyG `Batch` objects to `data/exports/<version>/graphs-{shard:05d}.pt`. The token writer reads the token `.pt` files and writes sharded `torch.Tensor` objects to `data/exports/<version>/tokens-{shard:05d}.pt`. The shard size is 5,000 contracts by default.

**Why batched:** the two writers share infrastructure (shard index, manifest writing) and are best authored together.

**Exit condition:** the writers run against the ScaBench fixture; produce correctly-shaped shards; the shard index maps `contract_id → shard_number` correctly.

**Commit:** `feat(data-export): add graph_writer + token_writer with sharded output`

---

### 7.3 — Implement `label_writer.py` and `metadata_writer.py`

Author the two parquet writers. The label writer reads the merged `.labels.json` files and writes a `labels.parquet` with one row per contract: `contract_id`, `source`, `class_<i>` (int8, for i in 0..9), `confidence_<i>` (float32), `split` (str). The metadata writer writes `metadata.parquet` with one row per contract: `contract_id`, `source`, `solc_version`, `version_bucket`, `loc`, `n_functions`, `n_pos`, `primary_class`, `confidence_tier`.

**Exit condition:** the writers produce valid parquet files with the expected schemas; values match the input labels JSON.

**Commit:** `feat(data-export): add label_writer + metadata_writer with parquet output`

---

### 7.4 — Implement `chunker.py` and the `SentinelDatasetExport` class

Author `sentinel_data/export/chunker.py` that orchestrates the 4 writers per shard, manages the shard index, and writes the manifest. Author `sentinel_data/export/export.py` that defines the `SentinelDatasetExport` class — the consumer-facing API that the ML module's `SentinelDataset` wraps.

**Exit condition:** `SentinelDatasetExport` can be constructed from an export directory; exposes `graphs_path`, `tokens_path`, `labels_path`, `metadata_path`, `shard_index`, `manifest`; `verify_artifact_hash()` returns True.

**Commit:** `feat(data-export): add chunker + SentinelDatasetExport consumer API`

---

### 7.5 — Wire the `sentinel-data export` CLI subcommand

Connect `cli.py` `export` subcommand to the writers. The CLI reads the merged labels + splits + representations, runs the writers per shard, and writes the export directory. Add `--dataset-version <name>` to look up the version in the catalog; default is the latest registered version.

Update `dvc.yaml` stage `export` to call `sentinel-data export`. The `export` stage is blocked by the `verify` stage's FAIL-class gate (D-4.5).

**Exit condition:** `sentinel-data export --dataset-version sentinel-v2-dryrun-2026-07` produces `data/exports/sentinel-v2-dryrun-2026-07/{graphs,tokens,labels,metadata}/` with the correct shard count and the manifest.

**Commit:** `feat(data-export): wire CLI + DVC for the export stage`

---

### 7.6 — Author `sentinel-ml/src/datasets/sentinel_dataset.py` (the new loader — with v9 schema gate + tier field + threshold mismatch warning)

Author the new ML-side loader. The file is ~150 lines:
- Accepts a `SentinelDatasetExport` (or a dataset version name + `sentinel_data.registry.load_artifact`)
- **Calls `verify_artifact_hash()` on load (per the v9 schema gate, 7-P2)** — if the export's `graph_schema_version` doesn't match the model's trained version, raises `ValueError`. This is the gate that prevents "trained on v9, evaluated on v10" silent failures.
- **Validates per-class thresholds (7-P3)** — Run 11 will have its own tuned thresholds; loading the v9 export's thresholds for a v9-trained model is correct, but loading v8 thresholds for a v9 model is a silent miscalibration. The loader checks the run name in the export manifest and warns on mismatch.
- Reads the shard index
- Implements `__getitem__` (returns `(graph, tokens, y, contract_id, confidence_tier)`) — the **`confidence_tier` field is new** (7-P4); the collate function in `collate.py` stacks it.
- Implements `__len__`

The `collate_fn` is extracted to `sentinel-ml/src/datasets/collate.py` (~50 lines, identical to the existing `dual_path_collate_fn` plus the new `confidence_tier` field).

**Why first in the seam swap:** the new loader is the consumer; it must exist before the old loader can be deleted.

**Exit condition:** `sentinel_dataset.py` runs a 1-batch forward pass against the v2-dryrun export without error; the returned tuple shape matches the expected `[B, 12]` graphs (v9 not v11), `[B, 4, 512]` tokens, `[B, 10]` labels, `[B]` confidence_tier; the v9 schema gate raises on mismatch; the threshold mismatch warning fires on test.

**Commit:** `feat(sentinel-ml): add new sentinel_dataset.py loader for v2 export (v9 schema gate + tier field + threshold warning)`

---

### 7.7 — Write the dual-path test (the seam swap gate — 8 fixed bugs preserved + EMITS)

Author a test in `sentinel-ml/tests/test_seam_swap.py` that takes 100 contracts and runs both the old `dual_path_dataset.py` and the new `sentinel_dataset.py` against the same contracts. The test asserts that the resulting batches are byte-identical (graph `.x`, `edge_index`, `edge_attr`, `y`; tokens `input_ids`, `attention_mask`; the only expected difference is the new `confidence_tier` field).

**Per AUDIT_PATCHES 7-P5, the test must verify that the 8 already-fixed code bugs survive the seam swap:**
- (a) label is from CSV not hardcoded 0
- (b) `now` keyword is detected
- (c) def_map is scope-keyed
- (d) prefix sort uses raw features
- (e) resume default is full (`resume_model_only=False`)
- (f) call_entry for external is present (self-loop)
- (g) LibraryCall not miscounted as HighLevelCall
- (h) tier threshold uses per-class (predictor.py fix)

**Per AUDIT_PATCHES 7-P6, the test must also verify the EMITS edge type bug (Interp-6) is fixed during the seam swap.** The test fixture has a contract with an event emit; the new path must produce an EMITS edge. If it doesn't, the bug must be fixed as part of the port (not deferred to v2.1).

**Why this is the gate:** the seam swap deletes the old loader. If the new loader produces different output, Run 9's active training pipeline (and any model that was trained on the old data) breaks. The dual-path test is the safety net.

**Exit condition:** the test passes for 100 contracts; the 8 fixed bugs are preserved; the EMITS edge is present; the byte-identical regression test from Stage 2 still passes; the active Run 9 training pipeline still works unchanged.

**Commit:** `test(sentinel-ml): add dual-path seam swap regression test (8 fixed bugs + EMITS preservation)`

---

### 7.8 — Delete the old paths (with archive + predictor fix + selective dep removal)

**First — fix the open bugs that must be closed during the seam swap (per AUDIT_PATCHES 7-P6, 7-P7):**

- **Predictor tier threshold fix (F8/F10)** — `ml/src/inference/predictor.py:150,168,752` has `TIER_CONFIRMED_THRESHOLD = 0.55` hardcoded and `_format_result` doesn't consult `self.thresholds`. Fix: `_format_result` consults `self.thresholds` per class; "confirmed" tier is per-class-tuned threshold. This is a small, surgical fix that prevents Run 11 from hitting the same display bug. Add a unit test that asserts the tier logic uses `self.thresholds[predicted_class_idx]`, not the hardcoded 0.55.
- **EMITS edge bug fix (Interp-6)** — only 12 EMITS edges across 41K contracts in the current graphs. The `graph_extractor.py` EMITS extraction is broken. Fix: ensure the Slither detector that catches event emissions is in the extractor pipeline. Add a test fixture with `emit Event();`; assert EMITS edge exists in the graph.

**Then — delete the old paths (per AUDIT_PATCHES 7-P9, archive before delete):**

- `ml/src/datasets/dual_path_dataset.py` (replaced by `sentinel_dataset.py`)
- `ml/src/preprocessing/graph_extractor.py` (moved to `sentinel_data`)
- `ml/src/preprocessing/graph_schema.py` (moved to `sentinel_data`)
- `ml/src/data_extraction/` (entire directory; moved to `sentinel_data.representation`)
- `ml/scripts/reextract_graphs.py`
- `ml/scripts/retokenize_windowed.py`
- `ml/scripts/build_multilabel_index.py`
- `ml/scripts/create_splits.py`
- `ml/scripts/create_cache.py`
- `ml/scripts/validate_graph_dataset.py`
- `ml/scripts/archive_v8_data.py`

**Archive BEFORE delete** (per AUDIT_PATCHES 7-P9): move all `ml/scripts/{reextract_graphs,retokenize_windowed,build_multilabel_index,create_splits,create_cache,validate_graph_dataset,archive_v8_data}.py` to `ml/scripts/_legacy_data_pipeline/` with a deprecation comment in each file header. They are the historical record of the v1 pipeline; deleting them without archiving loses the audit trail. Also move the 7 Phase 5 ad-hoc scripts to `Data/docs/legacy/bccc_deep_dive/_deprecated_scripts/` (per N-7 — they are replaced by the new verification module).

**Update:**
- `ml/src/inference/preprocess.py` to import from `sentinel_data.representation`
- `ml/pyproject.toml` to add `sentinel-data = "^0.1.0"` as a runtime dep; **remove `solc-select`, `py-solc-ast`, `solc` (now data-only deps); KEEP `slither-analyzer` (the inference path uses Slither indirectly via `ContractPreprocessor`)** (per AUDIT_PATCHES 7-P8)
- `ml/README.md` to remove the "Data Pipeline" section and replace with a one-line pointer to `Data/README.md`

**Why the gate:** the dual-path test (7.7) must pass before this deletion; the byte-identical regression test (Stage 2) must still pass; the 8 fixed-bug tests must still pass; the EMITS edge test must pass; the predictor tier test must pass.

**Exit condition:** all listed files deleted (after archiving); `ml/pyproject.toml` updated (slither-analyzer kept, solc-* removed); `ml/README.md` updated; predictor.py tier bug fixed and tested; EMITS edge bug fixed and tested; all existing ML-module tests still pass.

**Commit:** `refactor(sentinel-ml): delete old data-pipeline paths (archived), fix predictor tier threshold + EMITS edge, add sentinel-data dep`

---

### 7.9 — Build and test the Docker image (bookworm base, not slim)

Build the Dockerfile from Stage 0 (with the `python:3.12.1-bookworm` base, per AUDIT_PATCHES 7-P10 — `slither-analyzer` requires `build-essential` + `libpq-dev` for the `psycopg2-binary` transitive dep; slim doesn't have them). Run the image; verify the CLI works inside the container; verify a sample pipeline run succeeds (using the ScaBench fixture, ~30 contracts).

**Why at the end of Stage 7:** the Docker build is the last verification that the v2 corpus is reproducible. Doing it now means the image includes all 9 stages' code.

**Exit condition:** `docker build` succeeds; `docker run --rm sentinel-data:0.1.0 --help` works; `docker run --rm sentinel-data:0.1.0 run --dry-run` lists all 9 stages; a sample pipeline run inside the container produces the expected outputs; the base image is `python:3.12.1-bookworm`.

**Commit:** `ci(data): add Docker build verification to Stage 7 exit (bookworm base)`

---

### 7.10 — Final 7-gate check (the v2-is-ready gate, including 36-issue regression per AUDIT_PATCHES 7-P11)

Run the 7 verification gates from the proposal §9 + AUDIT_PATCHES 7-P11:
1. **Schema regression test** — every Run 9 graph re-extracted through the new module is byte-identical
2. **Phase 5 BCCC regression test** — the new module's verification report matches Phase 5's report to within ±0.5%
3. **End-to-end round-trip** — `sentinel-data run` from scratch produces a catalog entry; loading it through `SentinelDataset` and running a 1-batch forward pass in `sentinel-ml` succeeds with the same loss as the v1 loader
4. **Feature distribution report** — `sentinel_data.analysis.feature_dist.complexity_proxy_risk.md` is GREEN
5. **All 10 classes pass verification gate** — every class is VERIFIED or PROVISIONAL
6. **No leakage across splits** — `sentinel_data.splitting.leakage_auditor` reports 0 near-duplicates
7. **No open code-bug regression** — the 36-issue pre-Run-8 audit regression test passes; no A1–A38 fix is lost; EMITS edge is present; predictor.py tier threshold uses per-class; all 8 fixes (A9, A15, A20, A34, A38, resume, def_use, return_ignored) are preserved through the seam swap

**Why at the end of Stage 7:** these are the gates that the proposal §9 + AUDIT_PATCHES 7-P11 define for "v2 is ready to train." If any gate fails, Stage 7 is not complete.

**Exit condition:** all 7 gates are GREEN; a `data/docs/v2-readiness-2026-08-04.md` report is written documenting the gate results.

**Commit:** `docs(data): v2-readiness report — all 6 gates GREEN`

---

### 7.11 — Author `ADR-0008-export-and-seam-swap-design.md`

Document the key design decisions: sharded export (D-7.1), 3-step seam swap (D-7.2), thin new loader (D-7.3), `sentinel-data` as runtime dep (D-7.4), Docker as the reproducibility test (D-7.5), 6-gate v2-readiness check (D-7.6).

**Exit condition:** file exists; cites the dual-path test as the seam swap gate; references the 6 verification gates as the v2-readiness check.

**Commit:** `docs(data): add ADR-0008 for export + seam swap design`

---

## What NOT to fix (preservation list)

| Bug / Decision | Status | File:line | Stage 7 action |
|---|---|---|---|
| **A9** `now` keyword | ✅ FIXED | `ml/src/preprocessing/graph_extractor.py:587-605` | Do not re-fix. The 36-issue test guards it. |
| **A15** def_map by name | ✅ FIXED | `ml/src/preprocessing/graph_extractor.py:1147-1179` | Do not re-fix. The 36-issue test guards it. |
| **A20** label=0 hardcode | ✅ FIXED | `ml/src/data_extraction/ast_extractor.py:290,342,395` | Do not re-fix. The 36-issue test guards it. |
| **A34** prefix sort dim | ✅ FIXED | `ml/src/models/sentinel_model.py:356,367` | Do not re-fix. The 36-issue test guards it. |
| **A38** NaN before backward | ✅ FIXED | `ml/src/training/trainer.py` | Do not re-fix. The 36-issue test guards it. |
| Resume overwrite | ✅ FIXED | `ml/src/training/trainer.py:383,1184,1206,1212` | Do not re-fix. Stage 8 uses the full-resume default. |
| **EMITS edge bug** | ⚠ OPEN (Interp-6) | `ml/src/preprocessing/graph_extractor.py` | **MUST FIX during seam swap** (per 7-P6, 7.8 above). The 36-issue test asserts the bug exists pre-fix and is fixed post-seam-swap. |
| **Predictor tier threshold** | ⚠ OPEN | `ml/src/inference/predictor.py:150,168,752` | **MUST FIX during seam swap** (per 7-P7, 7.8 above). The 36-issue test asserts the fix. |
| **CALL_ENTRY cross-function for external** | ⚠ PARTIAL FIX | `ml/src/preprocessing/graph_extractor.py:1001` (self-loop only) | Preserve the partial fix (self-loop is preserved by the regression test). Full cross-function edge is post-Run-11. |
| 99% DoS↔Reentrancy co-occurrence in BCCC | Source: BCCC | (not in v2 corpus) | The Stage 3 merger de-duplicates; the v2 corpus is clean. |
| 7 Phase 5 ad-hoc scripts | ⚠ DEPRECATE | `Data/docs/legacy/bccc_deep_dive/Phase5_LabelVerification_2026-06-08/scripts/` | **Move to `Data/docs/legacy/bccc_deep_dive/_deprecated_scripts/` with deprecation comments** (per N-7). Do not delete silently. |

## Final exit criteria check

| # | Check |
|---|---|
| 1 | `format_schema/v1.yaml` exists with all 4 file type specs |
| 2 | All 4 export writers (graph, token, label, metadata) produce correctly-shaped outputs |
| 3 | `SentinelDatasetExport` class is constructable; `verify_artifact_hash()` works |
| 4 | `sentinel-data export --dataset-version sentinel-v2-dryrun-2026-07` produces a valid export |
| 5 | `sentinel-ml/src/datasets/sentinel_dataset.py` runs a 1-batch forward pass without error |
| 6 | The dual-path seam swap test passes for 100 contracts (old vs new loader byte-identical) |
| 7 | The byte-identical regression test from Stage 2 still passes |
| 8 | All old `ml/src/preprocessing/*` and `ml/src/data_extraction/*` files are deleted |
| 9 | All old `ml/scripts/{reextract,retokenize,build_multilabel,create_splits,create_cache,validate,archive_v8}.py` are deleted |
| 10 | `ml/src/inference/preprocess.py` imports from `sentinel_data.representation` |
| 11 | `ml/pyproject.toml` adds `sentinel-data = "^0.1.0"`; removes data-only deps |
| 12 | Docker build succeeds; sample pipeline run inside the container works |
| 13 | All 6 verification gates (proposal §9) are GREEN |
| 14 | `data/docs/v2-readiness-2026-08-04.md` documents the gate results |
| 15 | `ADR-0008-export-and-seam-swap-design.md` is committed |
| 16 | `dvc repro` runs the full 9-stage pipeline end-to-end with no errors |

All 16 pass → **Stage 7 complete**. Tag `data-stage-7`, register the `sentinel-v2-gold-2026-08` dataset version, proceed to Run 11 launch (Stage 8).

---

## Risk register

| Risk | Mitigation |
|---|---|
| The dual-path test fails because the new loader produces subtly different output (e.g. a different shard for a contract) | The test pinpoints the diff; debug until byte-identical; the seam swap is gated on the test passing — no exceptions |
| The Docker build fails on WSL2 because of a missing system dep | The Dockerfile is tested in Stage 0 for syntax; the actual build verification is in Stage 7; if it fails, the missing dep is added to the Dockerfile and rebuilt |
| Deleting the old `ml/src/preprocessing/*` files breaks the active Run 9 training pipeline | The dual-path test (7.7) and the byte-identical regression test (Stage 2) catch this; the deletion is gated on both tests passing |
| The ML module's `train.py` needs a CLI flag change to point at the new export | `train.py` gets a new `--dataset-version` flag in this stage; the old `--cache-path` flag is deprecated but still works (with a deprecation warning) |
| The `SentinelDataset` is slower than `dual_path_dataset.py` due to the additional `verify_artifact_hash()` call | The hash is computed once at `__init__`, not per `__getitem__`; the call is O(export size), not O(per-contract) |
| The Docker image is large (> 5 GB) because of all 6 solc versions + slither + the Python env | The image uses multi-stage builds; the final image only includes the active solc version; the others are downloaded on-demand via `solc-select install` |
| The seam swap breaks an active training run (Run 9 still in progress) | The active training run is paused before Stage 7 starts; Run 9 is allowed to complete before the swap; if Run 11 launches, Run 9 is already finished |

---

**End of Stage 7 actionable plan. Total estimated time: 8–10 working days (Jul 23–Aug 4). This is the longest stage in the build because it integrates the two packages and performs the seam swap.**
