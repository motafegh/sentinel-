# Stage 7 — Export + Seam Swap

**Date:** 2026-07-23
**Status:** NOT STARTED. Reading required before Stage 8.
**Reading time:** 30-40 minutes.
**Goal:** After this doc, you can answer all 7 items in `LEARNING_CHECKLIST.md` §"Stage 7" from memory.

---

## 1️⃣ The Problem

### What Stage 7 has to deliver

Stages 1–6 produced verified, analyzed data — but none of it in a form the ML module can consume. Stage 7 is the **integration stage**: it produces the sharded export that `sentinel-ml`'s `SentinelDataset` reads, and performs the **seam swap** — replacing the old import paths with the new ones.

This is the longest and most complex stage (8-10 days) because it touches both packages and the active training pipeline must not break.

### The 3 open bugs to close

| Bug | Where | Fix |
|---|---|---|
| **Predictor tier threshold** | `predictor.py:150,168,752` (hardcoded 0.55) | Use per-class tuned threshold from calibration JSON |
| **EMITS edge** | `graph_extractor.py` (Interp-6) | Ensure Slither detector for event emissions is in pipeline |
| **CALL_ENTRY cross-function** | `graph_extractor.py:1001` (self-loop only) | Full cross-function edge deferred to post-Run-11 |

---

## 2️⃣ The Solution

### Sharded export (D-7.1)

4 file types per shard:
- **graphs_shard** — PyG `Batch` objects (`.pt`)
- **tokens_shard** — `torch.Tensor` objects (`.pt`)
- **labels_parquet** — one row per contract: `contract_id`, `class_<i>` (int8), `confidence_<i>` (float32), `split`
- **metadata_parquet** — one row per contract: `contract_id`, `source`, `solc_version`, `version_bucket`, `confidence_tier`

Shard size: 5,000 contracts by default. The shard index maps `contract_id → shard_number` for lazy loading.

### The seam swap (D-7.2, 3-step operation)

**Step 1:** New writer in `sentinel-data`, new reader in `sentinel-ml` (`sentinel_dataset.py`).

**Step 2:** Dual-path test — for 100 contracts, old `dual_path_dataset.py` and new `sentinel_dataset.py` produce identical batches. This is the gate.

**Step 3:** Delete old paths:
- `ml/src/datasets/dual_path_dataset.py` (replaced by `sentinel_dataset.py`)
- `ml/src/preprocessing/graph_extractor.py` (moved to `sentinel_data`)
- `ml/src/preprocessing/graph_schema.py` (moved to `sentinel_data`)
- `ml/scripts/{reextract_graphs,retokenize_windowed,...}.py` (archived to `_legacy_data_pipeline/`)

### The new `SentinelDataset` (D-7.3, ~150 lines)

The new loader:
- Accepts a `SentinelDatasetExport` (paths + manifest + split name)
- Calls `verify_artifact_hash()` on load (v9 schema gate)
- Validates per-class thresholds (warns on mismatch)
- Lazy-loads shards
- Returns `(graph, tokens, y, contract_id, confidence_tier)` — the `confidence_tier` field is new

### The 7 v2-readiness gates (D-7.6)

1. Schema regression test — passes
2. Phase 5 BCCC regression — passes
3. End-to-end round-trip — passes
4. Feature distribution report — GREEN
5. All 10 classes verification gate — VERIFIED or PROVISIONAL
6. No leakage across splits — 0 near-duplicates
7. No open code-bug regression — 36-issue test passes, EMITS fixed, predictor fixed

### Docker build verification (D-7.5)

`docker build` succeeds; `docker run --rm sentinel-data:0.1.0 --help` works; sample pipeline run inside the container produces expected outputs. Base image: `python:3.12.1-bookworm` (not slim — slither-analyzer needs `build-essential` + `libpq-dev`).

---

## 3️⃣ The Broader Context

### What Stage 7 enables downstream

- **Stage 8 (Run 11)** — the model trains on the v2 export via the new `SentinelDataset`

### What breaks if Stage 7 is wrong

- Dual-path test fails → seam swap breaks Run 9 (or Run 11) training
- Predictor tier bug unfixed → Run 11's "Detected" column is wrong (hidden by hardcoded 0.55)
- EMITS edge unfixed → only 12 EMITS edges across 41K contracts (under-representation)
- Docker build fails → v2 corpus can't be reproduced on other machines
- Missing archive → historical pipeline scripts lost (audit trail broken)

---

## 4️⃣ Verification — Stage 7 exit criteria

| # | Check | Status |
|---|---|---|
| 1 | `format_schema/v1.yaml` exists | ⏳ |
| 2 | 4 export writers produce correct output | ⏳ |
| 3 | `SentinelDatasetExport` constructable + hash verified | ⏳ |
| 4 | Dual-path seam swap test passes (100 contracts) | ⏳ |
| 5 | Old `ml/src/preprocessing/*` deleted | ⏳ |
| 6 | `ml/pyproject.toml` adds `sentinel-data` dep | ⏳ |
| 7 | Docker build succeeds | ⏳ |
| 8 | All 7 v2-readiness gates GREEN | ⏳ |
| 9 | `predictor.py` tier bug fixed | ⏳ |
| 10 | EMITS edge bug fixed | ⏳ |

---

## 5️⃣ The "got it" checklist

1. **What are the 7 v2-readiness gates?** Schema regression, Phase 5 regression, round-trip, feature distribution, class verification, no leakage, no code-bug regression.

2. **What's the predictor tier threshold fix?** `predictor.py:150,168,752` uses hardcoded 0.55. Fix: use `self.thresholds[predicted_class_idx]` per class.

3. **What's the EMITS edge fix?** `emit Event()` should create EMITS edge type 10. Currently only 12 edges across 41K contracts. Fix: ensure Slither detector for event emissions is in the pipeline.

4. **What's the Slither transitive dep test?** Docker build must ensure `pip install sentinel-data` brings in slither-analyzer transitively.

5. **What's the seam swap atomicity?** The switch happens in a single commit. The dual-path test + byte-identical regression + 36-issue test prove no behavior change.

6. **What does the shard export look like?** `graphs-{shard:05d}.pt` + `tokens-{shard:05d}.pt` + `labels.parquet` + `metadata.parquet`. 5,000 contracts per shard.

7. **Why archive before delete?** `ml/scripts/_legacy_data_pipeline/` preserves the historical pipeline. Deleting without archiving loses the audit trail.

If you can answer all 7, Stage 7 is mastered.

---

## 6️⃣ What to read next

- **LEARNING_CHECKLIST.md** §"Stage 7"
- **08_stage_7_export_seam.md** — the design + intent document
- **Sentinel_v2_Data_Module_Integration_Proposal.md** §3.8 (export), §4 (integration), §9 (v2-readiness gates)

When you're ready, say **"Stage 7 is mastered — let's move to Stage 8."**
