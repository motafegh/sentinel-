# I — Regression Guard

> Always load `00_rules.md` before following this procedure.
> Apply Rule 2 (gate assertions + completion attestation) at every step.
>
> **Last revised: 2026-06-14** (post-Run-12 launch). **No gate changes** — all 4 Production gates in §I.2.2 and 6 Staging gates in §I.2.1 are still valid and the source-of-truth (`ml/scripts/promote_model.py`) is unchanged. Added cross-reference to `data_module/temp/live_plans/post_training_process_2026-06-14.md` which is the canonical end-to-end post-training workflow that calls into §I.3 steps.

---

## When This File Applies

- Before promoting any checkpoint to Staging or Production
- After a new run completes and you want to claim it beats the current best
- When comparing two checkpoints and deciding which to keep
- When a deployed model produces results that are worse than the previous version
- When `promote_model.py` exits with a non-zero code and you need to understand why

Always load alongside: `G_ablation_protocol.md` G.2.5 (metric comparison
procedure) and `C_diagnostic_checks.md` C.2 (behaviour checks required
before promotion).

**End-to-end workflow**: For the full post-training → promotion workflow
(reproducibility check, performance analysis, OOD/contamination check,
final report, save artifacts, MLflow promotion, monitoring handover),
see `data_module/temp/live_plans/post_training_process_2026-06-14.md`.
This file (I) is the per-gate detail; that doc is the runner.

---

## I.1 — Promotion Script

Read `ml/scripts/promote_model.py` docstring before any promotion. The script
is the canonical enforcement mechanism — this spec describes what the script
enforces and what you must verify before calling it.

**Model registry name** (hardcoded in the script): `sentinel-vulnerability-detector`

**Valid stages:** `Staging`, `Production`

**Default experiment:** `sentinel-retrain-v2` (override with `--experiment`)

---

## I.2 — Promotion Gates

These are the gates enforced by `promote_model.py`. They fire even in
`--dry-run` mode so you can verify them without a live MLflow write.

### I.2.1 — Staging Promotion Gates

| Gate | Enforced by script | What to verify manually before running |
|---|---|---|
| Checkpoint file exists | YES (`sys.exit(1)` if missing) | Confirm the `.pt` file is the correct run's best checkpoint, not a mid-run save |
| Thresholds JSON present | WARNING only (not a hard gate) | Always run `tune_threshold.py` before Staging promotion; uniform 0.5 fallback is not acceptable for evaluation |
| `val_f1_macro` provided | Argument required | Read the value from `epoch_summary.jsonl` `f1_macro_tuned` field, not from memory |
| **Behavioral probes JSON present + all PASS** | **YES** (`sys.exit(1)` if any probe FAIL) | Run `synthetic_probes.py` first; the JSON is the gate artifact. See C.2.4. |
| **Label quality JSON present + no FAIL** | **YES** (`sys.exit(1)` if any class rate > 0.50) | Run `label_quality.py` first. See F.1.0. |

No F1 regression gate is enforced for Staging. Staging is for evaluation,
not deployment. The F1 regression gate is Production-only.

### I.2.2 — Production Promotion Gates

| Gate | Enforced by script | What to verify manually |
|---|---|---|
| `val_f1_macro` strictly > current Production F1 | YES (`sys.exit(1)` on tie or decrease) | Run `--dry-run` first to confirm the numeric comparison before committing |
| Drift baseline exists at `--require-baseline` path | YES (`sys.exit(1)` if missing) | Confirm baseline `source` field is `'warmup'`, not `'training'` — script checks this too |
| Drift baseline was built from warmup traffic (not training data) | YES (`sys.exit(1)` if `source=='training'`) | Check baseline JSON `source` field manually before passing path |
| Thresholds JSON present | WARNING only | Never promote to Production without per-class tuned thresholds |
| **Behavioral probes JSON PASS rate == 100%** | **YES** (`sys.exit(1)` if any probe FAIL) | All 30+ probes must pass. See C.2.4. |
| **Label quality JSON: no FAIL** | **YES** | No class > 50% positive rate. See F.1.0. |

**The F1 tie-break rule:** `val_f1_macro` must be strictly greater than the
current Production model's recorded metric, not equal. The script uses `<=`
to block the promotion — a tie is treated as a downgrade because it provides
no evidence of improvement.

---

## I.3 — Pre-Promotion Checklist

Complete these steps before running `promote_model.py`. Steps I.3.1–I.3.4
are required for both Staging and Production. Steps I.3.5–I.3.6 are
Production-only.

### I.3.1 — Behaviour Checks (now automated via C.2.4)

**As of 2026-06-17, I.3.1 is ENFORCED by `promote_model.py` via the behavioral
probes gate (C.2.4).** The previous version of this spec listed these as
manual checks; the ExternalBug FP on Run 12 showed that manual checks are
not reliable — they were done but the FP was still promoted. Now they are
automated.

**The behavioral probes gate (C.2.4)** is the new I.3.1:
1. SmartBugs Curated smoke inference — replaced by synthetic probes (C.2.4)
   when SmartBugs Curated is contaminated
2. Known-positive contract round-trip — 10+ synthetic contracts with
   "should trigger" expected outcomes
3. Known-negative (NonVulnerable) contract check — 10+ synthetic contracts
   with "should NOT trigger" expected outcomes
4. FP probe check — 5+ edge-case contracts (gray area)

**See `C_diagnostic_checks.md` C.2.4 for the probe definitions and
`promote_model.py` for the gate enforcement.**

### I.3.2 — Calibration Files

Both files must exist and be dated after the checkpoint was saved:

- `<checkpoint_stem>_thresholds.json` — per-class tuned thresholds from
  `tune_threshold.py` (19-candidate sweep, classes in `CLASS_NAMES` index order)
- `<checkpoint_stem>_temperatures.json` — per-class temperature scaling factors
  from `calibrate_temperature.py` (or a global scalar if per-class was not run)

If either file is missing: run the script, then re-verify behaviour checks.
Do not promote without both files present.

### I.3.3 — Contamination Check

Run `check_contamination.py` between the training split and the benchmark
before any promoted model is compared to prior benchmarks. Read the script
docstring for its exact comparison mode (hash-based, not filename-based).

### I.3.4 — Smoke Suite

Run `D.1` from `D_smoke_preflight.md`. All smoke tests must pass before
promotion. A failed smoke test after a checkpoint is saved means the
checkpoint is not safe to promote.

### I.3.5 — Drift Baseline (Production only)

Verify `drift_baseline.json` at the path you will pass to `--require-baseline`:

```bash
python -c "import json; d=json.load(open('ml/data/drift_baseline.json')); \
    print(d.get('source', 'MISSING'))"
```

Expected output: `warmup`. If output is `training` or `MISSING`, rebuild:

```bash
python ml/scripts/compute_drift_baseline.py --source warmup
```

Only run after the API has collected enough warm-up production requests
(the minimum count is in the script's `N_WARMUP` constant — read before running).

### I.3.6 — F1 Regression Dry-Run (Production only)

Always run `--dry-run` first to confirm the F1 gate will pass without
committing to MLflow:

```bash
python ml/scripts/promote_model.py \
    --checkpoint ml/checkpoints/<checkpoint_name>.pt \
    --stage Production \
    --val-f1-macro <tuned_f1_from_epoch_summary> \
    --require-baseline ml/data/drift_baseline.json \
    --dry-run
```

Confirm the output line:
```
  F1 gate      : <new_f1> > Production <current_f1> ✓
```

If the gate line does not appear, the new F1 does not exceed the current
Production model. Do not promote.

---

## I.4 — Regression Detection After Promotion

A regression is any of the following after a new checkpoint is promoted:

| Signal | Threshold | Source |
|---|---|---|
| Val macro-F1 (tuned) drops | > 0.02 vs prior best | `epoch_summary.jsonl` `f1_macro_tuned` |
| Any CEI class F1 (Reentrancy=6, ExternalBug=2, TOD=8) drops | > 0.05 vs prior | Per-class F1 from benchmark run |
| Any minority class F1 (DoS=1, Timestamp=7, UnusedReturn=9) drops | > 0.05 vs prior | Per-class F1 from benchmark run |
| Known-positive contracts no longer predicted correctly | Any regression | Manual behaviour check |
| Hamming loss increases | > 0.03 vs prior | `val_metrics["hamming"]` from epoch summary |

Thresholds above are regression signal thresholds, not absolute quality gates.
A drop below threshold is a finding that requires investigation, not an
automatic rollback trigger. Use the decision criteria in I.5.

---

## I.5 — Rollback Decision Criteria

Rollback the Production model if ANY of the following are true:

1. Val macro-F1 (tuned) dropped > 0.05 vs prior Production (not just delta noise)
2. A known-positive contract from a CEI class is no longer predicted correctly
3. Known-negative (NonVulnerable) FP rate increased by > 10 percentage points
4. A KILL-level alert was recorded in `alerts.jsonl` that was not resolved before
   promotion (i.e., the checkpoint was saved after a `TrainingAbortError`)

Do NOT rollback for:
- A single minority class F1 drop below 0.05 (investigate first)
- Val macro-F1 drop of < 0.02 (within normal run variance)
- Brier score or ECE increase without accompanying F1 drop

### I.5.1 — Rollback Procedure

MLflow rollback means transitioning the previous version back to Production:

```python
from mlflow.tracking import MlflowClient
client = MlflowClient()
# List current versions to find the previous one
versions = client.search_model_versions("name='sentinel-vulnerability-detector'")
# Identify the prior Production version (now Archived)
client.transition_model_version_stage(
    name="sentinel-vulnerability-detector",
    version="<prior_version_number>",
    stage="Production",
    archive_existing_versions=True,
)
```

Read the MLflow model registry before running — confirm the prior version
number and that its run still has valid metrics recorded.

After rollback:
1. File a `BUG-` entry documenting the regression (H.5)
2. Update `MEMORY.md` Training History to reflect the rollback
3. Do not re-promote the regressed checkpoint

---

## I.6 — Checkpoint Companion Files

Every checkpoint that enters promotion consideration must have these files
in the same directory, with the same stem as the `.pt` file:

| File | Generated by | Required for |
|---|---|---|
| `<stem>.pt` | `trainer.py` (auto-saves best val epoch) | All promotions |
| `<stem>_thresholds.json` | `tune_threshold.py` | Staging + Production |
| `<stem>_temperatures.json` | `calibrate_temperature.py` | Production (recommended for Staging) |

If `_thresholds.json` is missing at Staging promotion, `promote_model.py`
prints a WARNING but does not block. For Production, the absence of tuned
thresholds means the deployed model uses uniform 0.5 for all classes, which
is not acceptable for security-critical classifications.

---

## I.7 — Completion Attestation

After completing a promotion or regression investigation, append to the
relevant run doc:

```
## Procedure Attestation — I_regression_guard — <ISO date>
Checkpoint: <path>
Target stage: Staging / Production
val_f1_macro_tuned (this run): N
val_f1_macro_tuned (prior Production / baseline): N
Delta: ±N
Steps completed:
  I.3.1 behaviour checks:      PASS/FAIL/SKIP (reason if skipped)
  I.3.2 calibration files:     PRESENT/MISSING
  I.3.3 contamination check:   PASS/FAIL/SKIP
  I.3.4 smoke suite:           PASS/FAIL/SKIP
  I.3.5 drift baseline:        source=warmup / source=training / SKIP (Staging)
  I.3.6 dry-run F1 gate:       PASS/FAIL / SKIP (Staging)
promotion result:   SUCCESS / BLOCKED (reason) / SKIPPED
Regression signal:  NONE / <describe signal>
Rollback required:  YES (reason) / NO
BUG filed:          BUG-<ID> / NO
Written to:         [path of this attestation]
```
