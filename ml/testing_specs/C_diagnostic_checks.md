# C — Diagnostic Checks

> Always load `00_rules.md` before following this procedure.
> Apply Rule 2 (gate assertions + completion attestation) at every step.
> This section covers post-run analysis, not pre-launch gates (see F).
>
> **Last revised: 2026-06-14** (post-Run-12 launch). Added reference to alert codes fired in Run 12 (18 alerts: `[9.3.6b]` AUC-PR<0.1 and `[9.3.6c]` F1-AUC divergence — see `MEMORY.md` Current State). Updated path references to v3 export.

---

## When This File Applies

- After any training run completes or is interrupted
- After each epoch boundary during a long run when checking for early problems
- When interpreting benchmark results (load alongside `A_benchmark_runs.md`)
- When investigating a metric regression between runs

---

## C.1 — Training Log Analysis

The training pipeline writes three JSONL files to `ml/logs/<run-name>/`.
Read `ml/src/training/training_logger.py` to understand the exact schema
before parsing any log. Do not infer field names from memory.

### C.1.1 — Log Files and Locations

| File | Content |
|---|---|
| `step_metrics.jsonl` | Per-step loss, LR, grad norm, VRAM |
| `epoch_summary.jsonl` | One line per epoch — 37-field schema (Spec §8) |
| `alerts.jsonl` | KILL / WARN_SKIP / WARN events with timestamps |

Always check `alerts.jsonl` first. A KILL-level alert means `TrainingAbortError`
was raised and the checkpoint was **not saved** — do not use the last checkpoint
if a KILL event is present after the epoch it was written.

### C.1.2 — KILL-Level Alerts (Stop-Everything)

KILL codes and their meaning (from `training_logger.py`):

- `[9.1.1]` Loss is NaN/Inf — Adam state likely corrupted; restart from last clean checkpoint
- `[9.1.2]` NaN/Inf in a named parameter — same; do not trust any checkpoint after this step
- `[9.1.3]` Adam `exp_avg` or `exp_avg_sq` is NaN/Inf — permanently corrupted optimizer
  state; must restart from a clean checkpoint, not resume from the corrupted one

If any of these appear, do not analyse further metrics from that run. Confirm
which epoch the KILL fired and which checkpoint was last cleanly saved before it.

### C.1.3 — WARN-Level Alerts (Analyse Before Continuing)

Key WARN codes and what to check:

| Alert code | Threshold | What to do |
|---|---|---|
| `[9.2.1]` | all-zero label batch | Count occurrences per epoch; >5% of steps is a data pipeline problem |
| `[9.2.2]` | NaN/Inf in `graphs.x` | Identify the contract; check graph extractor for that file |
| `[9.3.1]` | VRAM > 7500 MB | Reduce `--batch-size` or `--fusion-max-nodes` before next run |
| `[9.3.2]` | grad norm > 100× rolling mean | Check `--grad-clip` value; check loss function parameters |
| `[9.3.3]` | `aux_weight_norm` < 1e-6 | Aux head disconnected from gradient flow; check `--aux-loss-weight` |
| `[9.3.4]` | JK entropy < 0.5 | JK collapsed to single phase; see C.1.4 |
| `[9.3.6b]` | AUC-PR < 0.1 per label | Near-random signal for that class; check label distribution |
| `[9.3.6c]` | AUC-ROC delta < −0.02 while F1 > 0 | F1–AUC divergence; threshold gaming |
| `[9.3.6d]` | Brier score > 0.4 per label | Severe miscalibration; run temperature scaling |
| `[2.7]` | loss > 5× rolling mean | Loss spike; one-off is acceptable, recurring is a problem |

### C.1.4 — JK Entropy Check

Read `check_jk_entropy()` in `training_logger.py` for the exact implementation.

The entropy threshold is **0.5 nats** (`JK_ENTROPY_MIN = 0.5`). Entropy below
this fires `[9.3.4]` and means JK attention collapsed to a single phase —
the model is effectively ignoring two of the three graph phases.

To read JK entropy from the epoch log:
```
jq 'select(.epoch != null) | {epoch, jk_weight_entropy}' epoch_summary.jsonl
```

Known failure modes:
- `jk_entropy_reg_lambda=0.01` caused uniform 33/33/33 collapse in Run 3
  (read `train.py --jk-entropy-reg-lambda` arg comment for details)
- Entropy collapsing towards 0 indicates one phase dominating; entropy
  locking at `ln(3) ≈ 1.099` indicates the regularizer is too strong

### C.1.5 — GNN Share Verification

The smoke suite (D.2) checks `gnn_share >= 15%` at epoch 0. For a full run,
read `gnn_share` from the `step_metrics.jsonl` every 100 steps (log interval).

A `gnn_share` that stays near zero for more than the first 5 epochs indicates
the GNN gradient is not flowing. Cross-reference with `[9.3.3]` aux head norm
— if both are near zero, the GNN encoder may be effectively frozen.

### C.1.6 — Per-Class F1 Convergence

Read `per_class_f1` from `epoch_summary.jsonl`. For each class:

- A class with F1 = 0.0 at every epoch is either all-zero in the val split
  (label imbalance) or the model never predicts it above `--eval-threshold`
- A class with F1 oscillating ±0.04 every epoch around `--eval-threshold`
  is the documented minority-class boundary-crossing problem — this is why
  `--eval-threshold` defaults to 0.35 rather than 0.50 (read `train.py` arg
  comment for the exact rationale)
- `macro_f1` improvement is meaningful only when more than one class improves;
  a single dominant class (Reentrancy) driving macro can mask stagnation elsewhere

### C.1.7 — AUC and Brier Epoch Trends

Read `auc_roc_macro`, `auc_pr_macro`, `brier_score_overall`, and `ece`
from `epoch_summary.jsonl`. These are computed per epoch by `training_logger.py`.

- `auc_roc_delta` and `auc_pr_delta` are epoch-over-epoch deltas (Spec §3B.12/13);
  consistent negative delta on a class while F1 improves is the `[9.3.6c]`
  divergence pattern — threshold gaming
- `ece > 0.10` after temperature scaling indicates a calibration problem;
  run `ml/scripts/calibrate_temperature.py` and check `ece_pre` vs `ece_post`
  in the calibration log entries (they appear as `"type": "calibration"` lines
  in `epoch_summary.jsonl`)

---

## C.2 — Model Behaviour Verification

### C.2.1 — Smoke Inference Check

After a training run, run inference on the SmartBugs Curated benchmark contracts
(see `A_benchmark_runs.md` A.2). This is a qualitative sanity check, not an
accuracy measurement.

Expected behaviour:
- Reentrancy contracts should score high on the Reentrancy class
- Clean (no-vulnerability) contracts should score low on all classes
- If all 10 class outputs are identical (or all < 0.01), the model has collapsed;
  check `gnn_share` and JK entropy from the training log

Read `ml/src/inference/predictor.py` before running inference to confirm:
- Which checkpoint path it loads (hardcoded vs `SENTINEL_CHECKPOINT` env var)
  - **Default: `ml/checkpoints/GCB-P1-Run4-no-asl-pw_best.pt` (historical). Always set
    `SENTINEL_CHECKPOINT` to the active best from `MEMORY.md` Current State.**
- Whether `drop_complexity_feature` is read from the checkpoint config
  (if `--drop-complexity-feature` was used at training, it must be applied
  at inference — read `train.py` arg comment: predictor reads from checkpoint config).
  - **Run 12+ default: `drop_complexity_feature=True` (was `False` pre-Run 12)**

### C.2.2 — False Positive Probe

Run a set of known-clean contracts through inference. A well-trained model
should produce low probabilities (< 0.3) for all classes on clean contracts.

If Reentrancy probability is high on clean contracts:
- Check whether `pos_weight_min_samples` is suppressing the Reentrancy weight
  as intended. Default 3000: Reentrancy (4498 training positives) should be
  capped at `pos_weight=1.0` (read `train.py --pos-weight-min-samples`)
- Check the `--weighted-sampler` strategy — `timestamp-size` gives 4×
  to Timestamp+ large contracts; confirm this is not over-representing them

### C.2.3 — Threshold Verification

After training, run `ml/scripts/tune_threshold.py` to compute per-class optimal
thresholds. Read the script before running to confirm:
- The `--threshold-tune-interval` during training already ran a sweep
  (every 10 validation epochs by default; check the `[A37]` log entries)
- The companion `<run_name>_best_thresholds.json` file exists alongside
  the checkpoint in `ml/checkpoints/`
  - **Historical convention**: file is created at the `ml/checkpoints/`
    directory. **v3+ convention** (post-seam-swap): companion files may
    live at the run's log dir. Verify from `MEMORY.md`.
- If the thresholds JSON is missing, `promote_model.py` will warn but
  proceed — the deployed model will use uniform 0.5 for all classes

### C.2.4 — Synthetic Behavioral Probes (the missing test)

**Why this exists:** Run 12 was promoted to Staging despite giving
ExternalBug=0.82 on a safe_storage-style contract. The validation spec
suite at the time had no automated test that would have caught this.
C.2.4 is the missing test.

**What it does:** Runs the model against 30+ fixed synthetic contracts
(3 per class) with hardcoded expected probability bounds. Each probe has:
- A source code string
- A target class
- An expected operator (`>` or `<`)
- An expected probability threshold

If the model's actual probability violates the bound, the probe FAILS.

**Run the probes:**

```bash
# Default: Run 12 FINAL checkpoint
python ml/testing_specs/synthetic_probes.py --exit-on-fail

# Explicit checkpoint
python ml/testing_specs/synthetic_probes.py \
    --checkpoint ml/checkpoints/Run12_best.pt \
    --output ml/checkpoints/Run12_behavioral_probes.json \
    --exit-on-fail

# HTTP API mode (tests the inference_server, not the local checkpoint)
python ml/testing_specs/synthetic_probes.py \
    --base-url http://localhost:8001 \
    --output ml/checkpoints/Run12_behavioral_probes.json

# Single probe (debugging)
python ml/testing_specs/synthetic_probes.py \
    --checkpoint ml/checkpoints/Run12_best.pt \
    --probe ext_owner_only
```

**Output JSON format (`<output>`):**

```json
{
  "summary": {
    "total": 30, "passed": 19, "failed": 11,
    "pass_rate": 0.633, "all_passed": false,
    "by_class": {"ExternalBug": {"passed": 0, "failed": 3}, ...}
  },
  "results": [
    {
      "probe_id": "ext_owner_only",
      "class_name": "ExternalBug",
      "operator": "<", "threshold": 0.30,
      "actual": 0.861, "passed": false,
      "description": "address public owner; constructor() ...",
      "duration_s": 0.5, "source_chars": 95
    },
    ...
  ]
}
```

**Probes per class (3 each):**

For each of the 10 SENTINEL classes:
1. **Should trigger** — a contract that genuinely has the vulnerability
2. **Should NOT trigger** — a contract that LOOKS similar but is safe
3. **Edge case** — a contract that may or may not trigger (gray area)

See `synthetic_probes.py` for the full list. **Editing probes:**
- Open `ml/testing_specs/synthetic_probes.py`
- Add a new entry to the `PROBES` list
- Each entry must have: `id`, `class`, `operator`, `threshold`, `description`, `source`
- Re-run `python ml/testing_specs/synthetic_probes.py --exit-on-fail`

**Baseline result on Run 12 (2026-06-17):**
- 19/30 probes pass
- 11/30 fail — most notably all 3 ExternalBug probes (the FP issue)
- 1/3 IntegerUO probes (safe_math_08 in 0.8 gives 0.768)
- 2/3 CallToUnknown, GasException, DoS, UnusedReturn, MishandledException, TOD

**This probe set is the regression suite.** Any model that regresses on
these will fail this gate. Any future class with a broken feature will
also fail this gate.

**Auto-promotion gate:** The behavioral probe results JSON
(`<stem>_behavioral_probes.json`) is required by `promote_model.py`
for promotion. See `I_regression_guard.md` I.2.2 for the gate logic.

---

## C.3 — Regression Investigation

Use this when a new run shows lower macro-F1 than the previous best.

1. Read both runs' `epoch_summary.jsonl` and compare `per_class_f1` side-by-side
   — identify which specific class(es) regressed
2. Confirm the two runs used the same `splits_dir` (different split files
   change the val set and make F1 numbers incomparable)
3. Confirm the two runs used the same `--eval-threshold`; a threshold change
   of ±0.05 can move macro-F1 by ±0.03 without any model quality change
4. Check whether the schema version or label version changed between runs;
   either invalidates a direct comparison
5. Confirm contamination status for both runs (see `A_benchmark_runs.md` A.1)
   — if contamination check was not run on either run, the comparison is unreliable

---

## C.4 — Completion Attestation

After completing this section, append to the relevant run doc:

```
## Procedure Attestation — C_diagnostic_checks — <ISO date>
Steps completed:
  C.1.1 log files located:               PASS/FAIL
  C.1.2 KILL alerts checked:             PASS — zero KILL events / FAIL — see <link>
  C.1.3 WARN alerts reviewed:            PASS/FAIL/count=N
  C.1.4 JK entropy range:                min=X max=X (PASS if always > 0.5)
  C.1.5 GNN share:                       PASS/FAIL
  C.1.6 per-class F1 convergence:        PASS/FAIL — stagnant classes: [list]
  C.1.7 AUC/Brier trends:                PASS/FAIL/UNVERIFIED
  C.2.1 smoke inference:                 PASS/FAIL/UNVERIFIED
  C.2.2 FP probe:                        PASS/FAIL/UNVERIFIED
  C.2.3 threshold verification:          DONE/NOT DONE
  C.3 regression investigation:          N/A / findings: [link]
Steps skipped:     [any skipped + explicit reason]
Unverified items:  [anything not confirmable]
New findings:      [link to audit doc entry, or "none"]
Written to:        [path of this attestation]
```
