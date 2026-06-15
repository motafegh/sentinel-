# Run 12 → Run 13 Handoff Plan (2026-06-14)

> **Trigger:** Run 12 is training (PID 230342, ep44, f1_tuned=0.6941). This is the post-training plan: what to do WHEN Run 12 finishes, and how to launch Run 13 cleanly.
>
> **Scope:** Run 12 → Run 13 transition. ~3 weeks of work. Run 13 has 4 fixes: drop GasException, extend L4 to also drop `loc`, strip Solidifi `bug_*` prefix, inject 658 BCCC ME contracts.

---

## TL;DR

| Step | What | When | Who |
|---|---|---|---|
| 1 | Run 12 completes or early-stops | When patience=30 hit (likely ep60-80) | Claude + Ali |
| 2 | Run 12 final validation | After step 1 | Claude |
| 3 | Apply 4 pre-Run-13 fixes | After step 2 | Claude |
| 4 | Build v4 export (with BCCC ME) | After step 3 | Claude |
| 5 | Launch Run 13 | After step 4 | Claude + Ali approval |
| 6 | Run 13 monitoring (same as Run 12) | While Run 13 trains | cron + Claude |

---

## Step 1: Run 12 Completion Detection

**How to know Run 12 is "done":**
- **Early stop:** patience=30 means 30 epochs without improvement. Run 12 best f1_tuned=0.6941 was at ep30. If ep60 doesn't beat it, training will stop.
- **Full epochs:** 100 epochs, ~25 min each = ~42 hours total. Currently 17.6 hours in (ep44), ETA ep100 = ~24 more hours.
- **Manual kill:** Ali can kill if results are good enough.

**Detection signals:**
- Process PID 230342 disappears from `ps aux | grep train.py`
- Last log line in `ml/logs/GCB-P1-Run12-v3dospatched-20260613.log` says "Early stop" or "Training complete"
- The `best.pt` checkpoint stops updating

**Cron will detect automatically** — `ml/scripts/check_run12_status.sh` already handles "process died" critical notifications.

---

## Step 2: Run 12 Final Validation (~2 hours)

### 2.1 — Reproducibility (per `ml/testing_specs/L_release_readiness.md` §L.1)

- [ ] Checkpoint `seed` matches `MEMORY.md` Training History
- [ ] `TRANSFORMERS_OFFLINE=1` was set during training
- [ ] All 100 epochs (or until early stop) ran without restart
- [ ] `epoch_summary.jsonl` has one line per epoch with all required fields

### 2.2 — Performance analysis

- [ ] Read `epoch_summary.jsonl` end-to-end, identify:
  - Best epoch and its f1_tuned
  - DoS_F1 trajectory (should climb 0.11 → 0.36+)
  - Per-class F1 at best epoch (verify ExternalBug 0.88, Reentrancy 0.78, etc.)
  - Alerts (the 18 TransactionOrderDependence F1-AUC divergence alerts)
- [ ] Compute final f1_tuned and per-class metrics
- [ ] Compare to Run 11 ep1 (0.3384) — Run 12 should be much better

### 2.3 — OOD + contamination check (per `ml/testing_specs/A_benchmark_runs.md`)

- [ ] Run `python -m ml.scripts.check_contamination` (Tier 1-4)
- [ ] Test on SmartBugs Curated (143 real-world contracts) — primary benchmark
- [ ] Test on SolidiFI (283 synthetic contracts)
- [ ] Compare: did Run 12 improve on real-world OOD vs Run 9 best (0.2965)?

### 2.4 — Run 12 final report (NEW DOC)

Create `docs/training/GCB-P1-Run12-v3dospatched-analysis-2026-06-XX.md` with:
- Config recap (link to project_run12_launch.md)
- Per-epoch metrics table
- Per-class F1, AUC-PR, AUC-ROC at best epoch
- Comparison table: Run 12 vs Run 11 vs Run 9 vs Run 7 (all honest)
- Hypothesis verification: which of H1-H4 (from launch plan) were confirmed
- Lessons learned (what worked, what didn't)
- Recommendations for Run 13 (validates the 4 fixes)

### 2.5 — Save artifacts (~15 min)

- [ ] Copy final `best.pt` to `ml/checkpoints/GCB-P1-Run12-v3dospatched-20260613_FINAL.pt` (immutable name)
- [ ] Save `state.json` (full state for resume if needed)
- [ ] Compress `ml/logs/GCB-P1-Run12-v3dospatched-20260613/` to `ml/logs/_archive/Run12_2026-06-XX.tar.gz`
- [ ] Save MLflow experiment `sentinel-v12` results
- [ ] Update `MEMORY.md` with Run 12 final state

### 2.6 — Promote checkpoint to "Staging" (~30 min)

Per `ml/testing_specs/F_new_run_checklist.md` §F.2 and `ml/testing_specs/L_release_readiness.md`:
- [ ] Smoke inference on a few contracts (per `C_diagnostic_checks.md` C.2)
- [ ] Verify inference path uses the right checkpoint
- [ ] Don't promote to Production yet (Run 13 might be better)

---

## Step 3: Apply 4 Pre-Run-13 Fixes (~1-2 days)

### Fix 1: Drop GasException (NUM_CLASSES=9) — 30 min

**Why:** No source has any GasException data. F1=0.0 by definition. Dropping the class:
- Removes a dead neuron (10% of output capacity freed)
- Boosts f1_macro by ~0.07 just from the math (no longer averaging in 0.0)
- Reflects reality: we don't have data for this class

**Files to change:**
- `data_module/sentinel_data/representation/graph_schema.py`:
  - Remove `'GasException'` from `CLASS_NAMES`
  - `NUM_CLASSES` auto-derives from len(CLASS_NAMES) = 9
- `data_module/sentinel_data/representation/graph_schema.py` line 222-234: assertion still passes (now 9 entries)
- `ml/src/models/sentinel_model.py`: classifier head `Linear(256, 10)` → `Linear(256, 9)` (automatic if uses `NUM_CLASSES`)
- `ml/scripts/train.py`: `--num-classes` default (automatic)
- `ml/src/inference/predictor.py`: `output_dim` (automatic)

**Validation:**
- `pytest ml/tests/` (all existing tests should pass since they don't test GasException specifically)
- Verify `graph_schema.py:CLASS_NAMES` has 9 entries

**Script:** `/mnt/c/Users/lenovo/AppData/Local/Temp/opencode/drop_gas_exception.py` (to be created)

### Fix 2: Extend L4 to also drop `loc` (dim 6) — 15 min

**Why:** Inconsistency. L4 drops `complexity` (dim 5) but not `loc` (dim 6). Both encode size. Dropping both is more consistent.

**File to change:** `ml/src/models/gnn_encoder.py:435-438`:
```python
if self.drop_complexity:
    x = x.clone()
    x[:, 5] = 0.0  # complexity
    x[:, 6] = 0.0  # loc  ← NEW
```

**Validation:**
- Forward pass on a sample (should still work)
- All 13 tests in `test_gnn_encoder.py` should pass
- Verify output shape unchanged (still 256-dim hidden)

**Script:** `/mnt/c/Users/lenovo/AppData/Local/Temp/opencode/extend_l4_mitigation.py` (to be created)

### Fix 3: Strip Solidifi `bug_*` prefix — 30 min

**Why:** 86% of Solidifi contracts have function names like `bug_intou4`, `bug_re_ent*` that LITERALLY encode the vulnerability type. The model can read these and learn a shortcut. Real contracts don't have these names.

**File to change:** `data_module/sentinel_data/preprocessing/solidifi_normalizer.py` (NEW, or add to existing preprocessor):
```python
import re
def strip_bug_prefix(source: str) -> str:
    """Remove 'bug_<type>_' prefix from function/variable names to prevent label leakage."""
    return re.sub(r'\bbug_(intou|re_ent|tmstmp|txorigin|unchk|unchk_send|leak|dos)\w*', r'renamed_\1', source)
```

**Apply at:** `data_module/sentinel_data/preprocessing/` (during the preprocess stage, before graph extraction)

**Validation:**
- Re-tokenize a sample Solidifi contract, verify no "bug_intou" tokens appear
- Re-run smoke test (no regression on non-Solidifi data since rename only affects Solidifi)

**Script:** `/mnt/c/Users/lenovo/AppData/Local/Temp/opencode/strip_solidifi_bug_prefix.py` (to be created)

### Fix 4: Inject 658 BCCC ME contracts — 1-2 days (BIG)

**Why:** v3 ME = 39 (overfit noise). BCCC has 5,154 ME contracts, 658 are 2-tool confirmed. Injecting them gives v3 ME 39 → ~525 (13.5x).

**Process (per `data_module/docs/architecture.md` 5-stage protocol):**
1. Load 658 BCCC ME contract IDs from `/tmp/bccc_me_injection_candidates.json` (547 + 85 + 26)
2. **Cross-corpus dedup** — compute SHA256 of BCCC source, check against v3. (Already done: 0% overlap found in audit.)
3. **Compile probe** — try to compile each with our solc versions. ~80% expected to pass (BCCC is 92% pre-0.6 Solidity).
4. **Generate v4 export**:
   - Source: DIVE + SolidiFI + SmartBugs + 658 BCCC ME
   - Ingest → preprocess → represent → label (with v9 schema, NUM_CLASSES=9) → split (70/15/15 with L1+L2+L3 dedup) → export
5. **Validate v4**:
   - 0% leakage (run leakage_auditor)
   - 0% DoS+Reentrancy overlap (still applied)
   - All 6 GREEN + 1 AMBER gates pass
   - Smoke test on 1 epoch (per `D_smoke_preflight.md`)
6. **Update v4 README** with new artifact_hash

**Files/scripts to create:**
- `data_module/sentinel_data/scripts/inject_bccc_me.py` (the injection orchestrator)
- `data_module/sentinel_data/scripts/compile_probe.py` (verify BCCC compiles with our solc)
- `data_module/docs/v4-readiness-2026-06-XX.md` (the v4 gate doc, similar to v2-readiness-2026-06-12.md)
- Update `docs/CHANGELOG.md` with v4 entry

**Effort estimate:** 1-2 days (most of it is the dedup + compile probe + re-export orchestration)

---

## Step 4: Build v4 Export (1-2 days)

After all 4 fixes:
1. **Drop GasException** → re-build graph_schema.py
2. **Extend L4** → re-build model (just a code change, no data impact)
3. **Strip Solidifi bug_*** → re-preprocess Solidifi contracts
4. **Inject BCCC ME** → re-export v4

**v4 export expectations:**
- Path: `data_module/data/exports/sentinel-v4-bcccme-2026-06-XX/`
- artifact_hash: new (compute on build)
- 22,493 + ~480 injectable BCCC ME = ~22,973 contracts
- 21,657 with reps + ~480 new = ~22,137
- Splits: roughly 70/15/15 of 22,973
- 9 classes (GasException removed)
- Per-class: MishandledException 39 + 480 = ~519 (was 39), all others unchanged

**Validation gates (per `architecture.md` "6 GREEN + 1 AMBER"):**
- 0% leakage across splits
- 0 DoS+Reentrancy overlap
- 598/27/0 data_module tests pass (was)
- All splits have 9-class label vectors
- SentinelDataset 5-tuple shape unchanged
- Calibration files compatible with 9 classes

---

## Step 5: Launch Run 13 (1 day)

**Config (proposed, same as Run 12 unless changed):**
```bash
cd /home/motafeq/projects/sentinel
TRANSFORMERS_OFFLINE=1 PYTHONPATH=. TRITON_CACHE_DIR=/tmp/triton_cache nohup ml/.venv/bin/python ml/scripts/train.py \
    --run-name GCB-P1-Run13-v4bcccme-202606XX \
    --experiment-name sentinel-v13 \
    --export-dir data_module/data/exports/sentinel-v4-bcccme-2026-06-XX \
    --appnp-alpha 0.2 \
    --gnn-prefix-k 48 \
    --weighted-sampler positive \
    --dos-loss-weight 1.0 \
    --epochs 100 \
    --early-stop-patience 30 \
    > ml/logs/run13_launch_2026-06-XX.log 2>&1 &
```

**Same monitoring as Run 12:**
- Cron: `ml/scripts/check_run12_status.sh` → needs updating to point to Run 13 paths (or generalize to any run)
- Status log: `ml/logs/run13_status_checks.log`
- Notification log: `ml/logs/run13_notifications.log`
- State file: `ml/logs/.run13_check_state`

**Hypotheses to test (from launch plan + Run 12 learnings):**
- H1: MishandledException F1 becomes meaningful (0.5-0.7, not 1.0 overfit)
- H2: Overall f1_macro improves by ~0.07 (from removing GasException)
- H3: ExternalBug F1 stable (no regression)
- H4: Timestamp/UnusedReturn F1 stable (no regression from loc drop)

---

## Step 6: Run 13 Monitoring (while training)

- Same cron + PowerShell setup as Run 12
- Watch for: ep10/20/30/40/50 f1_tuned, DoS_F1 trajectory, any alerts
- **Run 12 + Run 13 comparison at ep30:** is Run 13 ahead of Run 12? (expected: yes, due to BCCC ME injection)

---

## Risk Register

| Risk | Likelihood | Mitigation |
|---|---|---|
| BCCC ME contracts fail to compile (40-92% rate per class) | High | Run compile probe, accept ~80% pass rate (~390 injectable) |
| Solidifi bug_* stripping breaks other code | Low | Apply only to function/variable names, test on sample |
| v4 export fails a gate | Medium | Re-run pipeline, check labels.parquet for new 0-class |
| Run 13 is worse than Run 12 (data regression) | Low | Compare per-class, identify which fix caused regression |
| L4 extension (drop loc) hurts small contracts | Medium | Spot-check 5 small contracts, verify no regression |
| Aliy wants different config for Run 13 | High | Get approval BEFORE launching |

---

## Decision gates (where Ali sign-off is needed)

- [ ] After Step 1 (Run 12 complete): OK to proceed to Step 3?
- [ ] After Step 2 (Run 12 report): OK to write up Run 12 final?
- [ ] After Step 3 (4 fixes applied): OK to inject BCCC?
- [ ] After Step 4 (v4 export built): OK to launch Run 13?
- [ ] After Step 5 (Run 13 launched): ongoing monitoring
- [ ] After Run 13 early stop: promote to Production? Or Run 14?

---

## Files to be created (in addition to existing)

- `docs/training/GCB-P1-Run12-v3dospatched-analysis-2026-06-XX.md` (Run 12 final report)
- `data_module/docs/v4-readiness-2026-06-XX.md` (v4 gate doc, mirrors v2-readiness-2026-06-12.md)
- `data_module/temp/live_plans/post_training_process_2026-06-14.md` (the overall workflow, you are here)
- `data_module/temp/live_plans/run13_plan_2026-06-14.md` (the 4-fixes details, you are here)
- `data_module/sentinel_data/scripts/inject_bccc_me.py` (BCCC injection orchestrator)
- `data_module/sentinel_data/scripts/compile_probe.py` (BCCC compile probe)
- `data_module/sentinel_data/preprocessing/solidifi_normalizer.py` (bug_* stripper)
- `/mnt/c/Users/lenovo/AppData/Local/Temp/opencode/drop_gas_exception.py` (NUM_CLASSES=9)
- `/mnt/c/Users/lenovo/AppData/Local/Temp/opencode/extend_l4_mitigation.py` (drop loc)
- `/mnt/c/Users/lenovo/AppData/Local/Temp/opencode/strip_solidifi_bug_prefix.py` (Solidifi rename)
- Updated `docs/CHANGELOG.md` with Run 12 final + Run 13 plan
- New ADR: `0009-drop-gas-exception-and-extend-l4.md` (documenting the architectural decisions)
