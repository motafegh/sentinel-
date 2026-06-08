# Actionable Plan — Stage 8: Run 11 Launch ("v2-baseline")

**Date:** 2026-08-05
**Stage:** 8 of 8 (Run launch: 2026-08-05)
**Owner:** SENTINEL ML engineering
**Source proposal:** `docs/proposal/Data_Module_Proposals/Sentinel_v2_Data_Module_Integration_Proposal.md` §5 (Aug 5), §9
**Audit ref:** [`AUDIT_PATCHES.md`](AUDIT_PATCHES.md) §0 (F8, F10, F15, F16, F20, F21, F22), §1 (8-P1 through 8-P12)
**Exit criteria:** Run 11 training starts cleanly on the v2-gold-2026-08 catalog entry; first epoch's val F1 is logged; the run is registered in MLflow with `sqlite:///mlruns.db` backend; the launch command is **timestamped** to prevent the Run 9 silent-overwrite incident; the watcher is a **copy of `ml/scripts/run8_watcher.sh`** with a `F1 > 0.1` floor; **per-class precision/recall/F1 are reported separately** (not just F1-macro); the predictor.py tier-threshold fix is verified active.

---

## Goal

Launch Run 11 ("v2-baseline") as the first training run on the v2 corpus. The run uses the v1 model architecture (no schema changes per the deferred-schema decision) but trains on the verified, multi-source, deduplicated v2 export. The first epoch's metrics are the baseline against which all future v2 runs are compared.

This stage is the *end* of the v2 build, not a new development stage. Its purpose is to launch the run, confirm it starts cleanly, and document the first-epoch results.

---

## Why this stage last

Per the proposal §5 (Aug 5 entry), Run 11 launches the day after Stage 7 completes. The 6 verification gates from Stage 7 are the precondition; the launch is the postcondition. Doing the launch as a separate stage (rather than as a task in Stage 7) means there is a clear handoff: Stage 7 delivers "v2 is ready"; Stage 8 starts "v2 is being trained."

---

## Design decisions

### D-8.1 — Run 11 uses the v1 architecture (v9 schema), not a new one

Per the deferred-schema decision (proposal §7) + AUDIT_PATCHES F1, the v2 baseline uses the v1 model architecture with the **active v9 schema** (NODE_FEATURE_DIM=12, NUM_EDGE_TYPES=12, NUM_NODE_TYPES=14, `_MAX_TYPE_ID=13.0`). The schema is what the v9 model checkpoint uses; we don't apply new schema changes in v2 (those are v2.1).

This decision is the discipline that lets us attribute Run 11's results to the *data* change, not the *model* change. Run 11's F1 vs Run 9's F1 is the data-only delta.

### D-8.2 — Run 11 uses the same TrainConfig as Run 9, with critical corrections

The Run 9 TrainConfig is the Run 11 config, **with three critical corrections from the audit (per AUDIT_PATCHES F16, F20, F21, 8-P1, 8-P2, 8-P3, 8-P4, 8-P5):**
- batch_size=8, gradient_accumulation_steps=8
- gnn_prefix_k=48, **`gnn_prefix_warmup_epochs=5`** (NOT 15 — Run 8/9 reduced this from 15 to 5, and the change was active in those runs; using 15 reverts a working change per F16, 8-P1)
- gnn_prefix_proj_lr_mult=5.0
- phase2_edge_types=[6, 8, 9, 10]
- weighted_sampler=positive
- loss=asl
- use_compile=True, use_amp=True
- **MLflow backend: `sqlite:///mlruns.db`** (NOT `file:///` — corrupt experiments 1,2,3 per F20, 8-P2)
- **`--early-stop-patience 30`** (matching Run 7/8/9; do not use the 20-epoch default)
- **`--threshold-tune-interval 10`** (matching Run 7/8/9; per-class thresholds are re-derived at ep10, ep20, etc. because the v2 corpus has different per-class base rates per F15, 8-P7)
- **`--jk-entropy-reg-lambda 0.005`** (the Run 7/8/9 value; the Run 9 typo was 0.0075 per F21, 8-P5; the launch command must be programmatically extracted from the original config, not typed by hand)

The dataset input: `--cache-path ml/data/cached_dataset_v10.pkl` becomes **`--dataset-version sentinel-v2-gold-2026-08`** (the v2 catalog entry from Stage 5).

### D-8.3 — The launch is monitored but not babysat; per-class P/R is the headline

Run 9 took ~35 min/epoch on the RTX 3070. Run 11 with the v2 corpus (similar size, ~30K contracts after dedup) takes a similar time per epoch. The first epoch's metrics are checked in this stage; subsequent epochs are monitored by the watcher (D-8.5).

**Per AUDIT_PATCHES 8-P6, 8-P8, the first-epoch report must include per-class precision, recall, AND F1 (not just F1-macro).** The Run 8 audit found "Test-set precision degenerate: DoS predicts positive for 76.8% of all test rows" — that was a precision problem, not F1. Reporting precision/recall separately surfaces this. The predictor.py tier-threshold fix (Stage 7) means the "Detected" column in the report is correct.

### D-8.4 — Run 11 is registered in MLflow with timestamped run-name

The MLflow run name is **`v2-baseline-$(date +%Y%m%d-%H%M%S)`** with a timestamp suffix (per AUDIT_PATCHES 8-P3, F21). The timestamp suffix prevents the Run 9 silent-overwrite incident (where the same `--run-name` was used twice and the new ep1 overwrote the ep14 best.pt). The run tags include: `dataset_version=sentinel-v2-gold-2026-08`, `sources=[bastet, scabench, web3bugs, defihacklabs, zenodo_16910242, smartbugs_curated]`, `schema_version=v9`, `architecture=four_eye_v8`, `predictor_tier_fix=true` (per the Stage 7 fix). The tags make the run self-describing for future audit.

### D-8.5 — The Run 11 watcher is a copy of `ml/scripts/run8_watcher.sh` with F1>0.1 floor

**Per AUDIT_PATCHES F22, 8-P9, 8-P10:** the watcher is a **copy of `ml/scripts/run8_watcher.sh`** with two changes:
1. The run name in the trigger patterns is updated to the timestamped name
2. The trigger logic **filters out F1=0.0 saves** — the watcher requires F1 > 0.1 before emitting "new best" (a tiny threshold that filters out the "best_f1 reset to 0.0" path that produced the Run 9 incident). The Run 8 watcher's existing trigger words (★ New best F1-macro, Epoch N/100 | Loss=, JK attention weights, Training complete / Early stopping) are preserved.

The watcher also monitors for **per-class F1 collapse** (8-P12): any class going from > 0.1 to 0.0 in a single epoch is a sudden collapse that the Run 8 audit pattern would have caught. The watcher emits a WARNING.

### D-8.6 — The Run 11 launch is the first test of the v2 module's promise

If Run 11 trains cleanly and the first-epoch val F1 is in the expected range (>= Run 7's first-epoch F1, which was ~0.20; >= Run 9 ep1's F1 of 0.2395 per AUDIT_PATCHES 8-P11), the v2 module's promise is validated: a clean data pipeline produces a clean training run. If Run 11 fails to start or the first-epoch F1 is unexpectedly low, the failure is debugged before further training.

---

## Tasks — ordered, each with verifiable exit condition

### 8.1 — Verify Run 9 has completed (or is paused) — confirm best checkpoint is archived

Before Run 11 launches, the active Run 9 must be either complete or paused. Run 9 was at ep52 best (F1=0.2965) per MEMORY.md; the assumption is Run 9 finished naturally. The best checkpoint is `ml/checkpoints/GCB-P1-Run9-v11-20260606_best.pt` — **archived to `ml/checkpoints/_archive/run9-v11-20260606_best.pt`** (per AUDIT_PATCHES §3 "Deletions" — this is the Run 9 baseline against which Run 11 is compared; archiving prevents accidental overwrite).

**Exit condition:** `ps aux | grep train.py` returns empty; the Run 9 checkpoint is saved AND archived; MLflow shows Run 9 status = FINISHED.

**Commit:** `chore: confirm Run 9 finished + archive best checkpoint before Run 11 launch`

---

### 8.2 — Update `ml/scripts/train.py` to support `--dataset-version`

The new training script accepts `--dataset-version <name>` (the v2 catalog entry name) and translates it to the path of the v2 export. The old `--cache-path` flag is deprecated but still works with a deprecation warning. The deprecation period is 1 release (Run 12 may remove it).

**Why first in this stage:** the script change is small but the launch depends on it.

**Exit condition:** `ml/scripts/train.py --help` shows the new `--dataset-version` flag; `--dataset-version sentinel-v2-gold-2026-08` resolves to the correct export path.

**Commit:** `feat(sentinel-ml): add --dataset-version flag to train.py`

---

### 8.3 — Update `ml/scripts/train.py` to read v2 metadata fields

The new training script reads the `metadata.parquet` from the v2 export and uses the new `confidence_tier` field for per-class loss weighting. The default weighting is to weight T0/T1 contracts 2× and T4 contracts 0.5×; this is configurable via `--class-weight-by-tier`.

**Exit condition:** the script reads the metadata; the per-class weights are applied correctly; a 1-batch test run shows the expected weighting in the loss.

**Commit:** `feat(sentinel-ml): use v2 confidence_tier for per-class loss weighting`

---

### 8.4 — Author the Run 11 launch command (12-condition pre-launch checklist)

Author the launch command per the proposal §5 (Aug 5 entry) + AUDIT_PATCHES 8-P1, 8-P2, 8-P3, 8-P4, 8-P5, N-11. The command follows the Run 9 template but uses the new `--dataset-version` flag, the timestamped `--run-name`, the sqlite MLflow backend, and the corrected `--gnn-prefix-warmup-epochs=5`.

**Per AUDIT_PATCHES N-11, the launch has a 12-condition pre-launch checklist (the `Run 11 launch checklist` doc).** The 12 conditions:

1. Run 9 best checkpoint is archived to `ml/checkpoints/_archive/run9-v11-20260606_best.pt`
2. The v2 catalog entry `sentinel-v2-gold-2026-08` is registered and reachable via `sentinel_data.registry.load_artifact`
3. The 7 v2-readiness gates are GREEN (Stage 7's exit criteria)
4. The predictor.py tier-threshold fix is active and tested
5. The EMITS edge bug is fixed and tested
6. The Run 11 launch command is **programmatically extracted from the original config** (not typed by hand) — per F21's "Lesson"
7. `--gnn-prefix-warmup-epochs=5` (not 15)
8. MLflow backend is `sqlite:///mlruns.db`
9. `--run-name` is timestamped: `v2-baseline-$(date +%Y%m%d-%H%M%S)`
10. `--early-stop-patience 30` and `--threshold-tune-interval 10` are set
11. `--jk-entropy-reg-lambda 0.005` (the Run 7/8/9 value; not 0.0075)
12. The watcher is a copy of `ml/scripts/run8_watcher.sh` with the F1 > 0.1 floor

**Exit condition:** all 12 checklist items are verified; the launch command is documented in `docs/training/GCB-P1-Run11-v2-baseline-2026-08-05.md`; the command runs without error in a `--dry-run` mode (or with `--epochs 1` for a smoke test).

**Commit:** `docs(training): add Run 11 launch command + 12-condition pre-launch checklist`

---

### 8.5 — Launch Run 11 (with immediate post-launch sanity check + per-class P/R)

Launch Run 11 in the background; the first epoch takes ~35 min on the RTX 3070. Monitor the first epoch's metrics: val F1, **per-class precision/recall/F1** (not just F1-macro, per AUDIT_PATCHES 8-P8), loss curve, gnn_prefix activation at ep5 (not ep15, per D-8.2).

**Per AUDIT_PATCHES 8-P6, the post-launch immediate sanity check is:** at ep1, the per-class F1 distribution should be non-degenerate (no class with F1=0.0 across all 10 classes, no class with F1=1.0). The Run 8 audit found that the predictor's tier-threshold bug hid 14/19 hits; **the predictor.py tier-threshold fix (Stage 7) is verified active by checking that the "Detected" column in the report matches the per-class tuned threshold** (not the hardcoded 0.55).

**Per AUDIT_PATCHES 8-P7, the launch must run `tune_threshold.py` after the first ep10 milestone** (not at the end). The per-class thresholds derived at ep10 are likely close to the final; running at ep10 + ep20 + ep30 + ep40 gives a 4-point trajectory that catches threshold drift. The current `threshold-tune-interval=10` already does this; the task is to verify it's active.

**Exit condition:** Run 11 starts cleanly; first epoch's val F1 is logged; per-class P/R/F1 are reported (not just F1-macro); the run is registered in MLflow with the timestamped name + sqlite backend; ep10 threshold tune runs cleanly.

**Commit:** `chore(ml): launch Run 11 v2-baseline (per-class P/R, ep10 threshold tune)`

---

### 8.6 — Document the first-epoch results (per-class P/R + comparison to Run 9 ep1)

Write a short report (`docs/training/GCB-P1-Run11-v2-baseline-2026-08-05.md`) capturing (per AUDIT_PATCHES 8-P6, 8-P8, 8-P11, N-12):
- Launch date, command, dataset version, sources, architecture, timestamped run name
- First-epoch val F1 (expected: >= 0.20, matching Run 7's first-epoch F1; >= 0.2395 matching Run 9's ep1 per 8-P11)
- **Per-class first-epoch precision, recall, AND F1** (the "before" baseline for future v2 runs; the precision/recall split is critical per 8-P8 because Run 8's "DoS predicts positive for 76.8% of test rows" was a precision problem, not F1)
- **Comparison to Run 9 ep1** (per 8-P11 + N-12): table showing Run 9 ep1 F1, Run 11 ep1 F1, and the delta. The v2 module's promise is validated if Run 11 ep1 > Run 9 ep1.
- Any unexpected behavior (e.g. a class that fails to predict, an OOM, a schema mismatch)
- The expected training trajectory (based on Run 9's plateau at ep52, Run 11's plateau is expected around ep40–60)
- The predictor.py tier-threshold fix verification (per 8-P6): the "Detected" column matches per-class tuned threshold

**Exit condition:** report is committed; the first-epoch results are documented (per-class P/R, not just F1); the comparison to Run 9 is explicit; future runs have a clear baseline to compare against.

**Commit:** `docs(training): document Run 11 first-epoch results (per-class P/R + Run 9 comparison)`

---

### 8.7 — Set up the Run 11 watcher (copy of run8 + F1>0.1 floor + per-class collapse detection)

**Per AUDIT_PATCHES 8-P9, 8-P10, 8-P12, F22:** the Run 11 watcher is a **copy of `ml/scripts/run8_watcher.sh`** (the proven Run 8 infrastructure) with the following changes:
1. The run name in the trigger patterns is updated to the timestamped name (e.g. `v2-baseline-20260805-143022`)
2. **The trigger logic filters out F1=0.0 saves** — the watcher requires F1 > 0.1 before emitting "new best" (a tiny threshold that filters out the "best_f1 reset to 0.0" path that produced the Run 9 silent-overwrite incident)
3. **The watcher monitors for per-class F1 collapse** — any class going from > 0.1 to 0.0 in a single epoch is a sudden collapse (Run 8's "DoS predicts positive for 76.8% of test rows" pattern); the watcher emits a WARNING
4. The existing trigger words (★ New best F1-macro, Epoch N/100 | Loss=, JK attention weights, Training complete / Early stopping) are preserved

The watcher monitors the training log, emits alerts on NaN/VRAM issues, and kills the run after a configurable patience threshold (30 epochs, matching Run 7/8/9).

**Exit condition:** the watcher is running; the watcher emits a "Run 11 started" alert; the watcher is configured to kill after 30 epochs of no F1 improvement; the F1>0.1 floor and per-class collapse detection are active.

**Commit:** `chore(ml): set up Run 11 watcher (copy of run8 + F1>0.1 floor + per-class collapse)`

---

## What NOT to fix (preservation list)

| Bug / Decision | Status | File:line | Stage 8 action |
|---|---|---|---|
| **A9** `now` keyword | ✅ FIXED | `ml/src/preprocessing/graph_extractor.py:587-605` | Do not re-fix. The 36-issue test (Stage 2) guards it. |
| **A15** def_map by name | ✅ FIXED | `ml/src/preprocessing/graph_extractor.py:1147-1179` | Do not re-fix. The 36-issue test guards it. |
| **A20** label=0 hardcode | ✅ FIXED | `ml/src/data_extraction/ast_extractor.py:290,342,395` | Do not re-fix. The 36-issue test guards it. |
| **A34** prefix sort dim | ✅ FIXED | `ml/src/models/sentinel_model.py:356,367` | Do not re-fix. The 36-issue test guards it. |
| **A38** NaN before backward | ✅ FIXED | `ml/src/training/trainer.py` | Do not re-fix. The 36-issue test guards it. |
| Resume overwrite | ✅ FIXED (default = full) | `ml/src/training/trainer.py:383,1184,1206,1212` | Do not re-fix. The Run 11 launch uses full-resume default; the watcher F1>0.1 floor is the additional safety net. |
| **Predictor tier threshold** | ✅ FIXED in Stage 7 | `ml/src/inference/predictor.py:150,168,752` | Do not re-fix. Stage 7's fix is verified active at launch (per 8-P6). |
| **EMITS edge** | ✅ FIXED in Stage 7 | `ml/src/preprocessing/graph_extractor.py` | Do not re-fix. Stage 7's fix is verified active. |
| 99% DoS↔Reentrancy co-occurrence in BCCC | Source: BCCC | (not in v2 corpus) | The v2 corpus is clean. The Stage 3 merger de-duplicates; the v2 catalog is verified clean by the Stage 4 verification report. |
| Run 9 best F1=0.2586 lost to silent overwrite | N/A (incident) | `project_run9_resume.md` | The Run 9 best checkpoint is archived in 8.1; the watcher F1>0.1 floor prevents recurrence; the launch is timestamped. |
| Per-class thresholds carry over from Run 9 | ❌ WRONG | `ml/calibration/temperatures_run9.json` | Per AUDIT_PATCHES F15, the v2 corpus has different per-class base rates; thresholds must be re-derived via `tune_threshold.py` at ep10, ep20, ep30, ep40. The 8.5 task verifies this is active. |
| v8 schema | ❌ WRONG | (proposal §2) | The active schema is v9. The Run 11 launch uses the v9-trained model. |

---

## Final exit criteria check (12-condition pre-launch + 7 launch-time)

| # | Check |
|---|---|
| 1 | `ps aux | grep train.py` is empty (Run 9 finished); Run 9 best checkpoint is archived |
| 2 | `ml/scripts/train.py --help` shows the new `--dataset-version` flag |
| 3 | The Run 11 launch command is documented and runs in dry-run / 1-epoch smoke mode |
| 4 | **All 12 pre-launch checklist items are verified** (8.4 — Run 9 archived, v2 catalog reachable, 7 v2-readiness gates GREEN, predictor fix active, EMITS fixed, command programmatically extracted, gnn-prefix-warmup=5, sqlite MLflow, timestamped run-name, patience=30, lambda=0.005, watcher copied from run8) |
| 5 | Run 11 launches cleanly; first epoch's val F1 is logged |
| 6 | **Per-class precision, recall, F1 are reported** (not just F1-macro); the predictor.py tier fix is verified |
| 7 | The run is registered in MLflow with **timestamped name** + **sqlite backend** + v2 catalog reference (dataset_version=sentinel-v2-gold-2026-08) |
| 8 | The first-epoch results are documented in `docs/training/GCB-P1-Run11-v2-baseline-2026-08-05.md` with **comparison to Run 9 ep1** |
| 9 | The Run 11 watcher is running, configured for 30-epoch patience, with **F1>0.1 floor + per-class collapse detection** |
| 10 | ep10 `tune_threshold.py` run is verified active; per-class thresholds are re-derived (NOT carried over from Run 9) |
| 11 | The 7 v2-readiness gates remain GREEN at launch (Stage 7's exit criteria are not regressed by the launch) |
| 12 | The 36-issue pre-Run-8 audit regression test still passes (no A1–A38 fix is lost between Stage 7 and Stage 8) |

All 12 pass → **v2 build complete**. The build is now in the "training" phase, not the "build" phase. Future work is Run 11+ iterations, schema v2.1, and the v3 architecture (with PDG/call-graph/opcode builders deferred from Stage 2).

---

## Risk register

| Risk | Mitigation |
|---|---|
| Run 11's first-epoch val F1 is unexpectedly low (< 0.10) | The first epoch is noisy; check the loss curve and per-class F1 distribution; if loss is decreasing and per-class F1 has a reasonable spread, the run is healthy; the patience watcher will kill if no improvement |
| Run 11 OOMs on the v2 corpus because it's slightly larger than the v1 corpus | The VRAM gate test from `ml/scripts/vram_gate_test.py` is run before launch; the corpus size is checked against the `max_nodes=2048` budget; if OOM, the batch size is reduced |
| The v2 export's shard format has a bug that surfaces only under training load | The Stage 7 end-to-end round-trip test catches most bugs; the dual-path test catches the loader-level bugs; any remaining bug surfaces in the first epoch's metrics (which is why we document them) |
| The Run 11 MLflow run is not linked to the v2 catalog | The `train.py` script's MLflow tag is `dataset_version=sentinel-v2-gold-2026-08`; the run is queryable in MLflow by this tag |
| The first-epoch results are misread as the final results | The first-epoch report explicitly notes "first epoch, not converged" and points to the patience-watcher for convergence monitoring |

---

**End of Stage 8 actionable plan. Total estimated time: 1 working day (Aug 5) for the launch; the run itself takes ~35 min/epoch and runs in the background for several days.**

---

# Build Summary (post-audit-patch 2026-06-08)

| Stage | Dates | Days | Status |
|---|---|---|---|
| 0. Skeleton + Data/ restructure | Jun 9–15 | 4–5 | ⏳ Pending |
| 1. Ingestion + Preprocessing | Jun 16–22 | 5 | ⏳ Pending |
| 2. Representation (port from ml/, 36-issue regression) | Jun 23–29 | 5 | ⏳ Pending |
| 3. Labeling (12 crosswalks, 99% co-occurrence regression) | Jun 30–Jul 13 | 10–14 | ⏳ Pending (extended from 1 week to 2 per AUDIT_PATCHES §3) |
| 4. Verification (per-stage p5_s1→s6 regression, CEI check) | Jul 14–20 | 5 | ⏳ Pending |
| 5. Splitting + Registry (migrations + retirement chain) | Jul 21–27 | 5 | ⏳ Pending |
| 6. Analysis (directed+conditional co-occurrence, label drift) | Jul 28–29 | 2 | ⏳ Pending |
| 7. Export + Seam Swap (predictor fix, EMITS fix, archive) | Jul 30–Aug 4 | 6 | ⏳ Pending |
| 8. Run 11 launch (12-condition checklist, per-class P/R) | Aug 5 | 1 | ⏳ Pending |

**Total build window: 8 weeks (Jun 9 – Aug 5, 2026).**

After Stage 8, the v2 corpus is the new ground truth. The BCCC data is preserved at `Data/docs/legacy/bccc_deep_dive/` for reference but is no longer the primary training source. Run 11+ iterations train on the v2 corpus; the v3 architecture (with the 4 new representation builders from Stage 2) is the next major work item.
