# Stage 7B — Seam Swap (LIVE PLAN)

**Started:** 2026-06-12 (after 7A complete)
**Scope:** Sub-stage 7B from the Stage 7 plan. Touches the `ml/` package: new `SentinelDataset` loader, dual-path regression test, file deletion + archival, pyproject.toml update, Docker build, 7-gate check. EMITS edge investigation.
**Prerequisite:** Sub-stage 7A complete (`data_module/temp/live_plans/stage_7a_export_module.md` all steps ✅)
**Source plan:** `docs/proposal/Data_Module_Proposals/actionable_plans/08_stage_7_export_seam.md` (7.6 through 7.10)
**Mode:** BUILD

---

## Pre-work (must be done before 7B starts)

- [ ] **7A complete**: `format_schema/v1.yaml`, 4 writers, `chunker.py`, `SentinelDatasetExport`, `_run_export` CLI, all 7A tests pass
- [ ] **Predictor tier fix merged**: `ml/src/inference/predictor.py` uses `self.thresholds[predicted_class_idx]`
- [ ] **Stage 2 representation complete on the v2 baseline**: ≥ 22,356 contracts have `.pt` and `.tokens.pt` files (currently 776 / 3.5% — must run the full `sentinel-data represent`)
- [ ] **At least one real `chunk_export` run on the v2 baseline**: produces a valid `data/exports/v1/` directory that the new `SentinelDataset` can load
- [ ] **Active Run 9 training pipeline is paused or completed** — the seam swap touches `ml/src/datasets/dual_path_dataset.py` and friends. Running a training job that imports from these files during the swap is undefined behavior
- [ ] **Confirm there's a test fixture of 100+ contracts with both graph and token files**, ideally a mix of solidifi (clean) and dive (messy) so the dual-path test exercises both

---

## Step 1 — Read everything (don't skip; the seam swap is risky)

- [ ] Read `ml/src/datasets/dual_path_dataset.py` end-to-end (already partially read; re-read with focus on the `__getitem__` and `dual_path_collate_fn` so the new loader mirrors them)
- [ ] Read `ml/src/datasets/dual_path_dataset.py:100-200` to see how `paired_hashes` is built and how `indices` is applied
- [ ] Read `ml/src/datasets/dual_path_dataset.py:200-403` to see the cache logic, the `__getitem__`, and the collate function
- [ ] Read `ml/src/inference/preprocess.py` to see what it currently imports from `ml/src/preprocessing/`
- [ ] Read `ml/pyproject.toml` to see the current dependency list (note `slither-analyzer`, `solc-select`, `py-solc-ast`, `solc` — which to keep, which to drop per 7-P8)
- [ ] Read `ml/src/preprocessing/graph_extractor.py` and `ml/src/preprocessing/graph_schema.py` (already read `graph_schema.py`; re-read with focus on the exports the new loader will use)
- [ ] Read `Data/docs/legacy/bccc_deep_dive/Phase5_LabelVerification_2026-06-08/scripts/` to enumerate the 7 Phase 5 ad-hoc scripts to move (per N-7, move to `Data/docs/legacy/bccc_deep_dive/_deprecated_scripts/`)
- [ ] Re-read `docs/proposal/Data_Module_Proposals/archive/AUDIT_PATCHES_applied_2026-06-08.md` lines 179-189 (7-P1 through 7-P11) and 261-276 (preservation list) — these are the rules of the seam swap
- [ ] Re-read the 7A `format_schema/v1.yaml` to know exactly what the new loader consumes
- [ ] Read the 36-issue pre-Run-8 audit (per 7-P11) at `Data/audit/05_tests_config_audit.md` or wherever the regression-test list lives
- [ ] Read the `Data/docs/v2-readiness-2026-08-04.md` template (if it exists) or the proposal §9 for the 7-gate format
- [ ] Read `Data/docker/Dockerfile.data` to know the current Dockerfile (which base image, which deps)

---

## Step 2 — EMITS edge bug investigation (the open bug per 7-P6)

- [ ] Run a quick analysis on the 776 contracts that have representations:
  - Count EMITS edges per contract; confirm whether the 12-edges-across-41K-contacts symptom is reproducible
  - Test fixture: a contract with `emit EventName(args);` — confirm whether `graph_extractor.py:1656-1670` actually fires
  - If it does fire in v9 but didn't in the v1 cache: it's a data-side issue, not a code bug. Document and move on (defer to v2.1 if it's a stale-cache issue)
  - If it does NOT fire: the Slither detector integration has a bug. Find the relevant Slither API call (likely `contract.events` enumeration) and fix
- [ ] Read `ml/src/preprocessing/graph_extractor.py:1650-1700` to see the event extraction code
- [ ] Read `ml/src/preprocessing/graph_extractor.py:1860-1920` to see the EventCall IR fallback
- [ ] If a code fix is needed: write a test fixture (a contract with one explicit `emit` and one implicit `EventName(args)` call) and a test that asserts the graph has at least one EMITS edge
- [ ] Commit the fix (or document the no-op) as a focused commit before the seam swap

---

## Step 3 — Author the new `sentinel-ml/src/datasets/sentinel_dataset.py` (~150 lines)

- [ ] Create `sentinel-ml/src/datasets/sentinel_dataset.py` with:
  - **Imports**:
    - `from sentinel_data.export import SentinelDatasetExport`
    - `from sentinel_data.registry import verify_artifact_hash` (per D-7.3)
    - PyTorch, PyG, pandas, logging
    - `from ml.src.preprocessing.graph_schema import FEATURE_SCHEMA_VERSION` (for the v9 schema gate, 7-P2)
  - **Class `SentinelDataset(torch.utils.data.Dataset)`**:
    - `__init__(self, dataset_version: str | None = None, export_dir: Path | None = None, split: str = "train", model_graph_schema_version: str | None = None, run_name: str | None = None)`:
      - Accept either a registered `dataset_version` (looks it up in the catalog) OR a direct `export_dir` path
      - Constructs the `SentinelDatasetExport(export_dir)`
      - **Calls `verify_artifact_hash()` on load** (per 7-P3, 7-P11) — raise `ValueError` on mismatch
      - **Schema-version gate (7-P2)**: compares `self.export.manifest.graph_schema_version` to the loaded `model_graph_schema_version` (default: read from `FEATURE_SCHEMA_VERSION`); raise `ValueError` on mismatch
      - **Threshold-mismatch warning (7-P3)**: if the export manifest has a `run_name` field and the caller provided a different `run_name`, log a warning that the per-class thresholds may be miscalibrated
      - Loads the shard index into memory
      - Loads `labels.parquet` into a pandas DataFrame (filtered to the requested split)
      - Builds a sorted list of contract_ids for the requested split
      - **Returns the 5-tuple (graph, tokens, y, contract_id, confidence_tier)** per 7-P4 (the new field; `dual_path_collate_fn` is updated to stack it)
    - `__len__(self) -> int`: returns the number of contracts in the requested split
    - `__getitem__(self, idx: int) -> tuple[Data, Tensor, Tensor, str, str]`:
      - Look up the contract_id for this index
      - Find the shard number via the shard index
      - Lazy-load the shard (cache the loaded shard; unload if cache exceeds N shards)
      - Find the contract's position within the shard
      - Read the graph, tokens, label, confidence_tier
      - Return the 5-tuple
  - **Helper function `_shard_cache_key(export_dir: Path, shard_num: int) -> Path`**:
    - Computes `<export_dir>/graphs/graphs-{shard:05d}.pt` (and parallel for tokens)
  - **Helper function `_lazy_load_shard(shard_num: int) -> tuple[Batch, Tensor, list[int]]`**:
    - Loads the .pt files (graphs, tokens) and the per-shard contract_id list (stored in the shard index)
  - **Shard cache** (LRU, configurable size via env var `SENTINEL_SHARD_CACHE_SIZE`, default 4 shards):
    - Tracks loaded shards; evicts least-recently-used when cache is full
    - Prevents per-`__getitem__` disk reads in the hot path
- [ ] Create `sentinel-ml/src/datasets/collate.py` (~50 lines, extracted from `dual_path_collate_fn`):
  - Stacks the 5-tuple from `__getitem__` into a batch: `[B, 12, ...]` graphs, `[B, 4, 512]` tokens, `[B, 10]` labels, `[B]` contract_ids, `[B]` confidence_tiers
  - Imports the new fields per 7-P4
- [ ] Re-export `SentinelDataset` from `sentinel-ml/src/datasets/__init__.py` (add to the existing imports)
- [ ] Test: import the new loader in a one-liner and confirm no ImportError
- [ ] Test: load the v2-dryrun export, run `for batch in loader: print(len(batch))` to confirm iteration works
- [ ] Test: 1-batch forward pass against the v2 export — assert the tuple shape matches `[B, 12]` graphs (v9 not v11), `[B, 4, 512]` tokens, `[B, 10]` labels, `[B]` contract_ids, `[B]` confidence_tiers
- [ ] Test: schema-version gate raises `ValueError` when `model_graph_schema_version="v8"` is passed against a v9 export
- [ ] Test: `verify_artifact_hash()` failure raises `ValueError`

---

## Step 4 — Author the dual-path test (`sentinel-ml/tests/test_seam_swap.py`)

- [ ] Create `sentinel-ml/tests/test_seam_swap.py` with:
  - **Test `test_byte_identical_for_100_contracts`**:
    - Loads 100 contracts (mix of sources if available) with both old `dual_path_dataset.py` and new `sentinel_dataset.py`
    - Iterates both, asserts each batch is byte-identical for:
      - `graph.x` (the 12-dim node features)
      - `graph.edge_index` (the COO edges)
      - `graph.edge_attr` (the edge types)
      - `y` (the labels)
      - `tokens.input_ids` (the [4, 512] token ids)
      - `tokens.attention_mask` (the [4, 512] attention mask)
    - The ONLY expected difference is the new `confidence_tier` field (the old loader doesn't return it)
  - **Test `test_8_fixed_bugs_preserved`** (per 7-P5):
    - Fixture contract with a known label: assert the label comes from the CSV, NOT hardcoded 0 (catches F2/A20)
    - Fixture contract using the `now` keyword: assert the timestamp detection still fires (catches F2/A9)
    - Fixture contract with multiple scopes: assert `def_map` is scope-keyed (catches F3/A15)
    - Fixture contract with 50+ functions: assert prefix sort uses raw features (catches F4/A34)
    - Test that the resume default is full (`resume_model_only=False`) (catches F5)
    - Fixture contract with an external call: assert `call_entry` self-loop is present (catches F6)
    - Fixture contract using a library call: assert `LibraryCall` is not miscounted as `HighLevelCall` (catches F7)
    - Fixture predictor with non-trivial thresholds: assert tier logic uses per-class `self.thresholds[cls_idx]` (catches F8/F10, regression for the 7A predictor fix)
  - **Test `test_emits_edge_present`** (per 7-P6):
    - Fixture contract with `emit Transfer(address to, uint256 value);`
    - Assert the graph has at least one EMITS edge (type 3)
    - If the assertion fails: the EMITS bug from Step 2 must be fixed before this test can pass
  - **Test `test_old_loader_unchanged`**:
    - Sanity check that `dual_path_dataset.py` is still functional at the time the test runs (proves we haven't accidentally broken the old code)
  - **Test `test_no_active_training_breaks`** (integration):
    - Run a 1-batch forward pass through the existing training pipeline (if it can be invoked headlessly) — if not feasible, document the manual run procedure
- [ ] Run the full `ml/tests/` suite to confirm no regression
- [ ] Commit as `test(sentinel-ml): add dual-path seam swap regression test (8 fixed bugs + EMITS preservation)`

---

## Step 5 — Apply the predictor tier follow-up (verify the 7A fix actually works in the new loader)

- [ ] Confirm the 7A predictor fix is in: `ml/src/inference/predictor.py` uses `self.thresholds[predicted_class_idx]`
- [ ] Add a unit test in the new `ml/tests/test_predictor.py` (created in 7A) that mocks the full inference path:
  - Mock model returns probabilities where class 0 has the highest prob (e.g. 0.92)
  - `self.thresholds[0] = 0.85` (high-tuned)
  - Assert the result marks class 0 as "confirmed" (prob > 0.85)
  - Assert the result does NOT mark class 0 as "confirmed" when prob=0.83 (would have triggered with old hardcoded 0.55)
- [ ] Run the predictor test; confirm it passes
- [ ] Commit (or amend the 7A commit if it wasn't pushed yet)

---

## Step 6 — Delete the old paths (the seam swap) — gated by Step 4

**CRITICAL: Do not delete anything until `test_byte_identical_for_100_contracts` passes.** This is the gate.

- [ ] **Archive BEFORE delete** (per 7-P9):
  - `mkdir -p ml/scripts/_legacy_data_pipeline/`
  - `git mv ml/scripts/reextract_graphs.py ml/scripts/_legacy_data_pipeline/`
  - `git mv ml/scripts/retokenize_windowed.py ml/scripts/_legacy_data_pipeline/`
  - `git mv ml/scripts/build_multilabel_index.py ml/scripts/_legacy_data_pipeline/`
  - `git mv ml/scripts/create_splits.py ml/scripts/_legacy_data_pipeline/`
  - `git mv ml/scripts/create_cache.py ml/scripts/_legacy_data_pipeline/`
  - `git mv ml/scripts/validate_graph_dataset.py ml/scripts/_legacy_data_pipeline/`
  - `git mv ml/scripts/archive_v8_data.py ml/scripts/_legacy_data_pipeline/`
  - Each archived file gets a deprecation comment at the top:
    ```python
    """
    DEPRECATED 2026-06-12 — moved to ml/scripts/_legacy_data_pipeline/ as part of
    the Stage 7 seam swap. Replaced by sentinel_data.representation.* and
    sentinel_data.splitting.*. This file is preserved for the audit trail;
    do not import from here.
    """
    ```
- [ ] **Delete the old ML paths**:
  - `git rm ml/src/datasets/dual_path_dataset.py` (replaced by `sentinel_dataset.py`)
  - `git rm ml/src/preprocessing/graph_extractor.py` (moved to `data_module/sentinel_data/representation/graph_extractor.py`)
  - `git rm ml/src/preprocessing/graph_schema.py` (moved to `data_module/sentinel_data/representation/graph_schema.py`)
  - `git rm -r ml/src/data_extraction/` (the entire directory; replaced by `data_module/sentinel_data/representation/`)
- [ ] **Move the 7 Phase 5 ad-hoc scripts** to `Data/docs/legacy/bccc_deep_dive/_deprecated_scripts/` (per N-7):
  - Enumerate scripts in `Data/docs/legacy/bccc_deep_dive/Phase5_LabelVerification_2026-06-08/scripts/`
  - `git mv` each to the new location
  - Add a `README.md` in the new dir explaining why they were moved
- [ ] **Update `ml/src/inference/preprocess.py`**:
  - Find imports from `ml.src.preprocessing` (or relative `..preprocessing`)
  - Replace with imports from `sentinel_data.representation` (e.g. `from sentinel_data.representation.graph_extractor import extract_contract_graph`)
  - The exact import mapping depends on the 7A re-export shape — check the actual `sentinel_data.representation` exports
- [ ] **Update `ml/pyproject.toml`** (per 7-P8):
  - **Add** `sentinel-data = "^0.1.0"` as a runtime dependency (path dep for dev: `sentinel-data = { path = "../data_module", develop = true }`)
  - **Keep** `slither-analyzer` (the inference path uses Slither indirectly via `ContractPreprocessor`)
  - **Remove** `solc-select` (the inference path uses system solc, not the version-pinned batch solc)
  - **Remove** `py-solc-ast` (data-only)
  - **Remove** `solc` (data-only)
  - Update the lockfile (`poetry lock` or equivalent)
- [ ] **Update `ml/README.md`**:
  - Remove the "Data Pipeline" section
  - Add a one-line pointer: "For data pipeline docs, see `Data/README.md`"
  - Add a section on the new `SentinelDataset` loader (path, basic usage)
- [ ] **Update `ml/data/README.md`** (which currently says "v8 schema, 11-dim"):
  - Rewrite to reflect the v2 export from `data_module/data/exports/`
  - Reference `data_module/temp/live_plans/stage_7a_export_module.md` (or wherever the schema is documented) for the format
  - Note that `ml/data/` is no longer the source of truth (per the user's earlier answer)
- [ ] Run `ml/tests/` after each deletion to confirm no regression
- [ ] Run the dual-path test after each deletion to confirm the gate still holds
- [ ] Commit: `refactor(sentinel-ml): delete old data-pipeline paths (archived), update preprocess.py + pyproject.toml + README`

---

## Step 7 — Docker build verification (per 7-P10, 7-P9)

- [ ] **Update the Dockerfile** at `Data/docker/Dockerfile.data`:
  - Base image: `python:3.12.1-bookworm` (NOT `slim` — `slither-analyzer` requires `build-essential` + `libpq-dev` for `psycopg2-binary` transitive dep)
  - Install all 6 solc versions (or use multi-stage build with on-demand `solc-select install`)
  - Copy the data_module source
  - Install the package (`pip install -e /data_module`)
  - Set the entrypoint to `sentinel-data`
- [ ] Build the image: `docker build -t sentinel-data:0.1.0 -f Data/docker/Dockerfile.data data_module/`
  - If the build fails on `psycopg2-binary` or `slither-analyzer` deps: add the missing system libs to the Dockerfile and retry
- [ ] Test: `docker run --rm sentinel-data:0.1.0 --help` returns the expected output (lists all 9 stages)
- [ ] Test: `docker run --rm sentinel-data:0.1.0 run --dry-run` lists all 9 stages
- [ ] Test (sample pipeline run): `docker run --rm -v $(pwd)/data_module/data:/data sentinel-data:0.1.0 run --from-stage preprocess` produces expected outputs
  - This is the proof that the v2 corpus is reproducible on any machine
- [ ] Commit: `ci(data): verify Docker build with python:3.12.1-bookworm base`
- [ ] Note: if Docker is not available in the build environment, document the manual verification steps and skip the test; the Dockerfile itself is the artifact

---

## Step 8 — 7 v2-readiness gates (the final go/no-go check, per 7-P11)

- [ ] **Gate 1: Schema regression test (from Stage 2)**
  - Run the Stage 2 byte-identical test: every Run 9 graph re-extracted through the new module is byte-identical
  - Status: should be GREEN (the schema constants haven't changed in the swap)
- [ ] **Gate 2: Phase 5 BCCC regression test (from Stage 4)**
  - Run the 21-test BCCC regression suite
  - Status: should be GREEN (the verification module didn't change in the swap)
- [ ] **Gate 3: End-to-end round-trip (the new one)**
  - `sentinel-data run` from scratch produces a catalog entry
  - Loading it through the new `SentinelDataset` and running a 1-batch forward pass in `sentinel-ml` succeeds with the same loss as the v1 loader
  - Status: should be GREEN (the dual-path test guarantees this)
- [ ] **Gate 4: Feature distribution report (from Stage 6)**
  - `data_module/data/analysis/<run_id>/complexity_proxy_risk.md` is GREEN
  - Status: should be GREEN (Stage 6 already verified this on the v2 baseline)
- [ ] **Gate 5: All 10 classes pass verification gate (from Stage 4)**
  - Every class is VERIFIED or PROVISIONAL
  - Status: should be GREEN (Stage 4 already verified this; 9 PROVISIONAL + 1 VERIFIED for v2)
- [ ] **Gate 6: No leakage across splits (from Stage 5)**
  - `sentinel_data.splitting.leakage_auditor` reports 0 near-duplicates
  - Status: should be GREEN (Stage 5 verified 0 dedup groups in v2 baseline)
- [ ] **Gate 7: No open code-bug regression (the new 7th gate per 7-P11)**
  - The 36-issue pre-Run-8 audit regression test passes
  - No A1–A38 fix is lost
  - EMITS edge is present (regression for Step 2 fix)
  - Predictor tier threshold uses per-class (regression for the 7A fix)
  - All 8 specific fixes (A9, A15, A20, A34, A38, resume, def_use, return_ignored) are preserved through the seam swap
  - Status: should be GREEN (the dual-path test in Step 4 covers most of these)
- [ ] Author `Data/docs/v2-readiness-2026-08-04.md` documenting the gate results (all 7 GREEN + the v2-readiness verdict)
- [ ] Commit: `docs(data): v2-readiness report — all 7 gates GREEN`

---

## Step 9 — ADR-0008 amendment

- [ ] Re-read `docs/decisions/ADR-0008-export-and-seam-swap-design.md` (written in 7A, scoped to 7A only)
- [ ] Append a "7B Amendment" section to the same file (or create `docs/decisions/ADR-0008b-seam-swap-7b.md` if you want a separate file — preference: append to keep the ADR self-contained)
  - **D-7.2 (now complete)**: The 3-step seam swap (new writer, dual-path test, delete old) — all 3 steps done
  - **D-7.3 (now complete)**: The new `SentinelDataset` is thin (~150 lines) — implemented
  - **D-7.4 (now complete)**: The ML module's `pyproject.toml` adds `sentinel-data` as a runtime dep — done per 7-P8
  - **D-7.5 (now complete)**: Docker build verified
  - **D-7.6 (now complete)**: All 6 (now 7) v2-readiness gates GREEN
  - **NEW (IC-6) — EMITS edge**: investigated; the v9 extractor has the event code in place (lines 1656-1670 and 1860-1920). The 12-edges-across-41K-contracts symptom is data-side (v1 cache was built before the EMITS fix landed). The dual-path test asserts EMITS is present on a fixture contract.
  - **NEW (IC-7) — Predictor tier fix landed in 7A**: the per-class `self.thresholds[predicted_class_idx]` fix was the highest-impact small change. It was bundled with 7A because it didn't depend on the seam swap.
  - **NEW (IC-8) — Shard cache**: the new `SentinelDataset` includes an LRU shard cache (configurable via `SENTINEL_SHARD_CACHE_SIZE` env var, default 4 shards). The dual_path_dataset.py re-loaded `.pt` files per `__getitem__`; the new loader amortizes this across batch construction.
- [ ] Commit: `docs(stage7): ADR-0008 amendment — 7B seam swap complete`

---

## Step 10 — Learning doc + checklist update for 7B

- [ ] Re-read `docs/proposal/Data_Module_Proposals/actionable_plans/learning_docs/stage_7_export_seam.md` (rewritten in 7A for 7A scope)
- [ ] Append a "7B Section" to the same file:
  - Status: "✅ COMPLETE (7A + 7B done; Stage 7 fully complete)"
  - "What was actually built" — add the new files (`sentinel-ml/src/datasets/sentinel_dataset.py`, `collate.py`, the dual-path test) and the deleted/archived files
  - "What was deleted" — the seam swap
  - "What was deferred to v2.1" — EMITS edge deep investigation (if it was data-side), the CALL_ENTRY cross-function edge (already deferred per the plan)
  - "Got it" checklist — add 4-5 new questions covering the seam swap
- [ ] Update `LEARNING_CHECKLIST.md`:
  - Mark `## Stage 7 — Export + Seam Swap` as `[x] COMPLETE` (drop the "7A only" qualifier)
  - Tick all the mastery checkboxes
  - Update the build state line at the top: "Stages 0-7 code-complete; Stage 8 (Run 11) not started"
  - Update the directory tree: `export/` is no longer a stub
- [ ] Commit: `docs(stage7): rewrite learning doc + update checklist for 7B`

---

## Step 11 — Commit everything in focused commits

Order of commits (each focused, each testable, each with a meaningful message):

1. `fix(ml): EMITS edge preservation (per 7-P6, regression test for the v9 event code)` (Step 2)
2. `feat(sentinel-ml): add sentinel_dataset.py new loader (~150 lines, v9 schema gate + tier field + threshold warning)` (Step 3)
3. `feat(sentinel-ml): add collate.py (extracted dual_path_collate_fn + new confidence_tier field)` (Step 3)
4. `test(sentinel-ml): add dual-path seam swap regression test (8 fixed bugs + EMITS preservation)` (Step 4)
5. `test(ml): add test_predictor.py (per-class tier threshold regression)` (Step 5; may already be in 7A)
6. `refactor(sentinel-ml): archive 7 legacy scripts to _legacy_data_pipeline/` (Step 6)
7. `refactor(sentinel-ml): delete old data-pipeline paths (dual_path_dataset, graph_extractor, graph_schema, data_extraction/)` (Step 6, gated on Step 4)
8. `refactor(sentinel-ml): update preprocess.py to import from sentinel_data.representation` (Step 6)
9. `chore(sentinel-ml): add sentinel-data dep to pyproject.toml; remove data-only deps (solc-select, py-solc-ast, solc); keep slither-analyzer` (Step 6, per 7-P8)
10. `docs(sentinel-ml): update README.md to point at Data/README.md and document SentinelDataset` (Step 6)
11. `ci(data): verify Docker build with python:3.12.1-bookworm base (per 7-P10)` (Step 7)
12. `docs(data): v2-readiness report — all 7 gates GREEN (per 7-P11)` (Step 8)
13. `docs(stage7): ADR-0008 amendment — 7B seam swap complete` (Step 9)
14. `docs(stage7): rewrite learning doc + update checklist for 7B` (Step 10)

---

## Open questions / decisions to make during execution

- [ ] **EMITS edge: code fix or data refresh?** — depends on Step 2 investigation. If the v9 extractor code is correct, no code change needed; the v1 cache just needs regeneration (v2.1 work).
- [ ] **Shard cache size**: default 4 shards × ~5000 contracts = 20K contracts. May need to tune based on training memory budget. Configurable via `SENTINEL_SHARD_CACHE_SIZE` env var.
- [ ] **The `confidence_tier` collate**: stack as a `Tensor[str]` (PyTorch doesn't natively support str tensors) or as a `list[str]`? The collate function needs to handle this. PyTorch's default collate doesn't handle strings; need a custom approach. Options: (a) return as a Python list in the batch dict, (b) encode as int8 indices and provide a reverse lookup, (c) skip stacking and return per-sample.
- [ ] **Predictor fix follow-up**: is the `suspicious` tier also per-class? The 7A fix may have only addressed `confirmed`. Re-read `predictor.py:670-710` and decide.
- [ ] **The 7 Phase 5 ad-hoc scripts**: are all 7 still in `Data/docs/legacy/bccc_deep_dive/Phase5_LabelVerification_2026-06-08/scripts/`? Enumerate first; some may have been moved already by earlier work.
- [ ] **The dual-path test fixture**: 100 contracts with both old + new paths producing identical output. Where do these contracts come from? Options: (a) pick 100 from `ml/data/graphs/*.pt` (the legacy v8 data), (b) pick 100 from the v2 export (but the old loader can't load the v2 export), (c) build a fixture that has both old + new files.
  - **Decision needed**: the dual-path test asserts old == new for the SAME contracts. The old loader reads `ml/data/graphs/*.pt` (MD5-keyed, v8 schema); the new loader reads `data_module/data/exports/v1/graphs/*.pt` (sha256-keyed, v9 schema). These are different file paths and different naming. The test must either: (a) generate a v1-style cache from the v2 data first, (b) use a separate fixture that has both forms, or (c) assert the test is "soft" — the new loader's output is correct, the old loader's output is no longer accessible, and we trust the dual-path test from Stage 2 as the byte-identical guarantee.
- [ ] **Docker availability in the build environment**: if Docker isn't available, the Docker gate is "documented, not executed". The Dockerfile itself is the artifact. The user can verify the build on their own machine.
- [ ] **The `v2-readiness-2026-08-04.md` filename**: the plan specifies this date. If the actual completion date is different, update.

---

## Risks / gotchas to watch for

- [ ] **The dual-path test may fail subtly** — even a single float-difference in `graph.x` will fail the test. The most common cause: a different feature normalization in the new path. Mitigation: start with the most-aligned features (label, edge_index, edge_attr) and add feature-level asserts one at a time.
- [ ] **The `dual_path_collate_fn` doesn't handle `confidence_tier` (str)** — when extracting to `collate.py`, the str field breaks PyTorch's default collate. Need a custom collate that returns strs as a Python list.
- [ ] **EMITS edge test may not find a contract with a known emit** — the test fixture needs to be hand-crafted or sourced from a known contract (e.g. a Solidity SafeMath or OpenZeppelin ERC20).
- [ ] **The `data_extraction/` directory in ml/src** — the plan says to delete the entire directory. If anything else in `ml/src/` imports from it (e.g. `ml/src/inference/api.py`?), the deletion will break imports. Grep for `from .data_extraction` and `from ..data_extraction` first.
- [ ] **Docker build may fail on the transitive `psycopg2-binary` dep** — the plan addresses this (bookworm has `libpq-dev`). If it still fails, check the slither-analyzer version pin.
- [ ] **The `archived` scripts may have other callers** — `git grep` for `from ml.scripts.reextract_graphs` etc. before deletion. If something else imports them, the deletion will break it.
- [ ] **Predictor test isolation** — the 7A test must not depend on GPU or real model weights. Use a mock model that returns synthetic probabilities.
- [ ] **The v2-readiness gate "36-issue regression"** — this is the Stage 2 byte-identical test. If the seam swap accidentally changes a feature, this gate fails. The dual-path test catches this before we get here.
- [ ] **Schema version "v9" string vs int** — the manifest stores `"v9"` (string). The `SentinelDataset` should compare strings, not int. Easy to mess up.
- [ ] **`confidence_tier` for None** — pyarrow writes `None` as null. The `SentinelDataset` should pass null through as `None` (Python) or a sentinel string. Decide: null vs "" vs "None".

---

## Files I'll create / modify (per package)

### `ml/src/` (ml-side new + modify)
1. (create) `ml/src/datasets/sentinel_dataset.py` (~150 lines, the new loader)
2. (create) `ml/src/datasets/collate.py` (~50 lines, extracted dual_path_collate_fn + new confidence_tier field)
3. (modify) `ml/src/datasets/__init__.py` (re-export `SentinelDataset`)
4. (delete) `ml/src/datasets/dual_path_dataset.py` (after Step 4 gate)
5. (delete) `ml/src/preprocessing/graph_extractor.py` (after Step 4 gate)
6. (delete) `ml/src/preprocessing/graph_schema.py` (after Step 4 gate)
7. (delete) `ml/src/preprocessing/__init__.py` (if empty after deletions)
8. (delete) `ml/src/data_extraction/` (entire dir, after Step 4 gate)
9. (modify) `ml/src/inference/preprocess.py` (import from `sentinel_data.representation`)
10. (modify) `ml/src/inference/predictor.py` (the 7A fix; if not already done)
11. (modify) `ml/src/inference/api.py` (if it imports from any of the deleted paths)

### `ml/tests/` (ml-side tests)
12. (create) `ml/tests/test_seam_swap.py` (the dual-path test)
13. (create) `ml/tests/test_predictor.py` (the per-class tier regression; may already be in 7A)

### `ml/scripts/` (archive)
14-20. (move to `ml/scripts/_legacy_data_pipeline/`):
  - `reextract_graphs.py`
  - `retokenize_windowed.py`
  - `build_multilabel_index.py`
  - `create_splits.py`
  - `create_cache.py`
  - `validate_graph_dataset.py`
  - `archive_v8_data.py`

### `ml/` (config + docs)
21. (modify) `ml/pyproject.toml` (add `sentinel-data` dep; remove `solc-select`, `py-solc-ast`, `solc`; keep `slither-analyzer` per 7-P8)
22. (modify) `ml/poetry.lock` (regenerate)
23. (modify) `ml/README.md` (remove "Data Pipeline" section; add `SentinelDataset` section)
24. (modify) `ml/data/README.md` (rewrite to reflect v2 export from `data_module/data/exports/`)

### `Data/docs/legacy/` (archive)
25-31. (move 7 Phase 5 ad-hoc scripts to `Data/docs/legacy/bccc_deep_dive/_deprecated_scripts/`)
32. (create) `Data/docs/legacy/bccc_deep_dive/_deprecated_scripts/README.md` (why they were moved)

### `Data/docker/` (build verification)
33. (modify) `Data/docker/Dockerfile.data` (bookworm base, all 6 solc versions, `sentinel-data` install)

### `Data/docs/` (v2-readiness)
34. (create) `Data/docs/v2-readiness-2026-08-04.md` (the 7-gate report)

### `docs/decisions/` (ADR amendment)
35. (modify) `docs/decisions/ADR-0008-export-and-seam-swap-design.md` (append 7B amendment)

### `docs/proposal/.../learning_docs/` (docs)
36. (modify) `docs/proposal/Data_Module_Proposals/actionable_plans/learning_docs/stage_7_export_seam.md` (append 7B section)
37. (modify) `docs/proposal/Data_Module_Proposals/actionable_plans/learning_docs/LEARNING_CHECKLIST.md` (mark Stage 7 fully complete)

---

## What 7B does NOT do (deferred to Stage 8 / Run 11)

- [ ] Stage 8: Run 11 launch — train the model on the v2 export, generate per-class F1, compare against BCCC baseline
- [ ] Stage 8: Run the model interpretability suite to check whether the L4 finding ("complexity dominates all 10 classes") is gone in v2
- [ ] v2.1: EMITS edge deep investigation (if it was data-side per Step 2)
- [ ] v2.1: CALL_ENTRY cross-function edge (per the plan's "what NOT to fix" list)
- [ ] v2.1: Per-class overrides for the NonVuln cap (D-5.8 IC)
- [ ] v2.1: DISL re-introduction (was deferred in v1)

---

## Definition of done for 7B

- [ ] All 11 of the user's review fixes for 7A are in (7A plan is final)
- [ ] 7A is committed and pushed (separate commit chain; 7B builds on top)
- [ ] `sentinel-ml/src/datasets/sentinel_dataset.py` runs a 1-batch forward pass against the v2 export without error
- [ ] The dual-path test passes for 100 contracts (old vs new loader byte-identical)
- [ ] The 8 fixed code bugs are preserved (7-P5)
- [ ] The EMITS edge is present (7-P6)
- [ ] All old `ml/src/preprocessing/*` and `ml/src/data_extraction/*` files are deleted (after archiving)
- [ ] All old `ml/scripts/{reextract,retokenize,build_multilabel,create_splits,create_cache,validate,archive_v8}.py` are archived
- [ ] `ml/src/inference/preprocess.py` imports from `sentinel_data.representation`
- [ ] `ml/pyproject.toml` adds `sentinel-data = "^0.1.0"`; removes data-only deps; keeps `slither-analyzer` (per 7-P8)
- [ ] `ml/README.md` and `ml/data/README.md` updated
- [ ] Docker build succeeds (if Docker is available in the env); or Dockerfile is verified syntactically
- [ ] All 7 v2-readiness gates are GREEN
- [ ] `Data/docs/v2-readiness-2026-08-04.md` documents the gate results
- [ ] ADR-0008 amendment (or new ADR-0008b) is committed
- [ ] Learning doc + checklist updated
- [ ] All 14 commits landed in focused, testable units
- [ ] Stage 7 fully complete; ready for Run 11 (Stage 8)
