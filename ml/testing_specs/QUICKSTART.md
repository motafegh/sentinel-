# QUICKSTART — ml/testing_specs/

> 5-minute guide to using the testing spec suite.

---

## The single command: `ml-validate`

The framework CLI runs ALL 9 gates in one shot:

```bash
# Run all 9 gates against Run 12 (6 PASS, 3 FAIL, Exit 1)
python -m ml.testing_specs.framework.cli run \
    --config ml/testing_specs/framework/templates/sentinel_v2.yaml \
    --exit-on-fail
```

The config drives everything: which gates to run, what paths to check, what
thresholds to use. See `framework/templates/sentinel_v2.yaml` for the
SENTINEL-specific template.

---

## Or the 3 commands individually

If you want to run gates one at a time:

```bash
# 1. Pre-launch: check label quality
python ml/testing_specs/label_quality.py --exit-on-fail

# 2. Post-training: run behavioral probes
python ml/testing_specs/synthetic_probes.py \
    --checkpoint ml/checkpoints/Run_best.pt \
    --output ml/checkpoints/Run_behavioral_probes.json \
    --exit-on-fail

# 3. Promote: behavioral + label gates are now hard gates
python ml/scripts/promote_model.py \
    --checkpoint ml/checkpoints/Run_best.pt \
    --stage Staging \
    --val-f1-macro 0.7004
```

That's it. The framework handles the rest.

---

## Additional gates (Run 12 verified 6 PASS / 3 FAIL)

The full framework runs 9 gates in one CLI call:

| Gate | Tool | What it checks |
|---|---|---|
| `file_exists` | framework | Checkpoint + thresholds exist |
| `behavioral_probes` | synthetic_probes.py | All 40 probes pass |
| `label_quality` | label_quality.py | No class > 50% positive |
| `calibration_files` | framework | Thresholds JSON valid |
| `threshold_sensitivity` | threshold_sensitivity.py | No F1 gaming, no over/under-prediction |
| `cross_tool` | cross_tool.py | Model agrees with Slither+Aderyn |
| `reproducibility` | auto_reproducibility_check.py | Model state + git + lockfile match |
| `stale_checkpoints` | check_stale_checkpoints.py | No stale/orphan checkpoints |
| `promote_model` | promote_model.py | All of the above + val F1 > prior |

Each tool can be run independently with `--exit-on-fail` to use as a
stand-alone gate in CI / cron / pre-commit. Unit tests for the gate
classes are in `ml/tests/test_framework_gates.py` (36 tests, 91% coverage).

---

## What each command does

### 1. `label_quality.py` — Pre-launch

Checks the training labels for:
- Per-class positive rate (FAIL if any class > 50% positive)
- Per-source dominance (FAIL if a single source is > 80% of positives)
- Class co-occurrence (FLAG suspicious correlations)

**Why:** Run 12 had ExternalBug=74% positive because the DIVE crosswalk
maps "Access Control" → ExternalBug. This check would have caught it.

### 2. `synthetic_probes.py` — Behavioral gate

Runs 40 fixed synthetic contracts (3 per class + 10 adversarial edge cases)
through the model with hardcoded expected probability bounds. Probes can
end in 3 states:
- **PASS** — model gave the expected probability
- **FAIL** — model gave the wrong probability
- **INFRA_ERROR** — inference pipeline crashed (e.g., pragma-only source,
  now gracefully handled by `predictor.predict_source()` returning a
  structured `no_contracts_found` response — see BUG-6 fix 2026-06-18)

**Why:** C.2.1 (SmartBugs Curated smoke test) is contaminated (95.8% in v3).
C.2.4 (synthetic probes) doesn't need a clean benchmark — the probes ARE
the clean benchmark.

**Why:** C.2.1 (SmartBugs Curated smoke test) is contaminated (95.8% in v3).
C.2.4 (synthetic probes) doesn't need a clean benchmark — the probes ARE
the clean benchmark.

### 3. `promote_model.py` — Promotion

The script enforces:
- Checkpoint exists
- `val_f1_macro > current Production F1`
- **Behavioral probes JSON present and all PASS** (NEW)
- **Label quality JSON present and no FAIL** (NEW)
- Drift baseline (Production only)

The new behavioral gate means **a model with a broken class is now blocked
from promotion**. The Run 12 ExternalBug FP would have been caught.

---

## Common scenarios

### "I just trained a new model. How do I check it's good?"

```bash
# 1. Run synthetic probes
python ml/testing_specs/synthetic_probes.py \
    --checkpoint ml/checkpoints/Run13_best.pt \
    --output ml/checkpoints/Run13_behavioral_probes.json \
    --exit-on-fail

# 2. If passed, promote to Staging
python ml/scripts/promote_model.py \
    --checkpoint ml/checkpoints/Run13_best.pt \
    --stage Staging \
    --val-f1-macro <F1 from training log>
```

### "I want to see if my model is broken on a specific class"

```bash
# Run just the ExternalBug probes
python ml/testing_specs/synthetic_probes.py \
    --checkpoint ml/checkpoints/Run12_FINAL.pt \
    --probe ext_owner_only
```

### "I want to add a new probe for a new failure mode"

```bash
# 1. Edit ml/testing_specs/synthetic_probes.py
# 2. Add your probe to the PROBES list:
{
    "id": "my_new_probe",
    "class": "ExternalBug",
    "operator": "<",
    "threshold": 0.30,
    "description": "What this probe tests",
    "source": "<your .sol source code here>",
}

# 3. Run it
python ml/testing_specs/synthetic_probes.py --probe my_new_probe

# 4. Run the full suite to confirm everything still works
python ml/testing_specs/synthetic_probes.py --exit-on-fail
```

### "I want to check my training labels before launching"

```bash
python ml/testing_specs/label_quality.py \
    --labels data_module/data/exports/<my_export>/labels.parquet \
    --exit-on-fail
```

### "I'm promoting to Production and need all gates"

```bash
# All gates in one shot
python -m ml.testing_specs.framework.cli run \
    --config ml/testing_specs/framework/templates/sentinel_v2.yaml \
    --exit-on-fail

# Or run individually
python ml/testing_specs/label_quality.py
python ml/testing_specs/synthetic_probes.py --exit-on-fail
python ml/testing_specs/threshold_sensitivity.py \
    --checkpoint ml/checkpoints/Run_FINAL.pt \
    --benchmark data_module/benchmarks/<your_benchmark> \
    --exit-on-fail
python ml/testing_specs/cross_tool.py \
    --checkpoint ml/checkpoints/Run_FINAL.pt \
    --benchmark data_module/benchmarks/<your_benchmark> \
    --exit-on-fail
python ml/scripts/auto_reproducibility_check.py \
    --checkpoint ml/checkpoints/Run_FINAL.pt \
    --exit-on-fail
python ml/scripts/check_stale_checkpoints.py --max-age-days 30 --exit-on-stale
python ml/scripts/promote_model.py --stage Production --val-f1-macro <F1> --dry-run
# If dry-run passes, run again without --dry-run
```

### "I'm closing a session and want to make sure no findings are floating"

```bash
# Detects unwritten findings in recently-edited files
python ml/scripts/session_close.py --dry-run
```

---

## Files you'll interact with most

| File | When |
|---|---|
| `framework/cli.py` | Always — runs all 9 gates |
| `synthetic_probes.py` | After training, before promotion |
| `label_quality.py` | Before training (pre-launch) |
| `threshold_sensitivity.py` | After training, F1 gaming check |
| `cross_tool.py` | After training, agreement with static analysis |
| `auto_reproducibility_check.py` | Before any release (L.4.1) |
| `check_stale_checkpoints.py` | Weekly, operational hygiene |
| `session_close.py` | End of session, anti-floating-findings |
| `promote_model.py` | When promoting to Staging or Production |
| `framework/templates/sentinel_v2.yaml` | The SENTINEL gate config |
| `00_rules.md` | Always (the universal invariants) |
| `C_diagnostic_checks.md` | Post-run analysis (incl. C.2.4 synthetic probes) |
| `F_new_run_checklist.md` | Pre-launch + post-run checklist (incl. F.1.0 label quality) |
| `I_regression_guard.md` | Promotion gates reference (incl. I.2.1/2.2/3.1) |
| `L_release_readiness.md` | Release gate + session handoff (incl. L.4.1/L.5.1) |

---

## What if I find a bug in the testing suite?

1. File a `BUG-<ID>` entry in `ml/audit_docs/ISSUES.md` per
   `H_issue_triage.md` H.5
2. Add a probe to `synthetic_probes.py` if the bug is model-behavior
3. Update the spec file that should have caught it
4. Run `python -m ml.testing_specs.framework.cli run --exit-on-fail` to
   verify all 9 gates catch the bug
