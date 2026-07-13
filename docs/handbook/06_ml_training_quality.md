# 06 — ML training, quality, interpretability, and MLOps

**Read this when:** you need to retrain, calibrate, choose thresholds, evaluate, interpret, or promote a teacher checkpoint.

**Skip this if:** you are consuming an already approved checkpoint and only need [inference](05_ml_model_inference.md).

**Estimated reading time:** 15 minutes.

## 30-second summary

Training is a multi-label optimization workflow, not just `trainer.fit()`: load a hash-verified DATA export, optimize the four-eye model, tune per-class thresholds, calibrate probabilities, evaluate held-out and out-of-distribution evidence, run behavioral/interpretability checks, create a warm-up drift baseline, and promote the checkpoint plus companion artifacts through explicit gates.

## Just-enough mental model

```text
versioned DATA → train → checkpoint
                   ├→ per-class thresholds
                   ├→ calibration
                   ├→ held-out/OOD/probes
                   ├→ interpretability evidence
                   └→ drift baseline → promotion record
```

Loss optimizes parameters; thresholds convert probabilities into decisions; calibration makes probability magnitudes meaningful. These are related but separate artifacts.

## Actual runtime/source walkthrough

- [`trainer.py`](../../ml/src/training/trainer.py) — `ml/src/training/trainer.py::Trainer` owns phase control, optimization, validation, checkpointing, metrics, threshold tuning hooks, and structured logging.
- [`losses.py`](../../ml/src/training/losses.py) — `::AsymmetricLoss` handles multi-label imbalance; [`focalloss.py`](../../ml/src/training/focalloss.py) contains focal alternatives.
- [`training_logger.py`](../../ml/src/training/training_logger.py) emits machine-readable health, gradient, calibration, ranking, and abort signals.
- [`tune_threshold.py`](../../ml/scripts/tune_threshold.py) sweeps thresholds independently per class, selecting by F1 then recall then lower threshold.
- [`calibrate_temperature.py`](../../ml/scripts/calibrate_temperature.py) fits post-training temperature scaling from validation evidence.
- [`promote_model.py`](../../ml/scripts/promote_model.py) records MLflow promotion and blocks missing probes, label-quality evidence, or production drift baselines.
- [`compute_drift_baseline.py`](../../ml/scripts/compute_drift_baseline.py) builds monitoring evidence from warm-up traffic; using training data is explicitly unsuitable for production promotion.
- [`scripts/interpretability`](../../ml/scripts/interpretability) contains maintained experiments for JK weights, attention, edge/feature ablations, saliency, probes, counterfactuals, calibration, permutation importance, and architectural behavior.

Promotion is a policy decision over evidence. Copying a `.pt` file is not promotion because it omits the audit trail and companion artifacts.

## Interfaces, data shapes, and configuration

A promoted model set contains at least:

- checkpoint weights and saved architecture/config;
- exact DATA export/split identity and hashes;
- ordered ten-class vocabulary and v9 schema;
- per-class threshold JSON;
- calibration artifact/metrics;
- behavioral probes and held-out/OOD reports;
- checkpoint SHA-256;
- approved drift baseline and promotion record.

Training policies are materialized by [`train.py`](../../ml/scripts/train.py), checkpoint configuration, and [`mlops_config.json`](../../ml/mlops_config.json). Values such as loss parameters, phase schedule, smoothing, thresholds, and early stopping are decision numbers: change them only with a versioned before/after evaluation.

## Failure modes and current limitations

- A green unit suite does not establish model quality.
- Validation-tuned thresholds can overfit; preserve a final held-out decision set.
- Per-class imbalance makes aggregate accuracy misleading.
- Calibration can improve probability trust without improving ranking, and vice versa.
- Interpretability experiments explain selected behaviors; they are not formal causal guarantees.
- MLflow/local reports and checkpoints may be local-only, so a promotion record must name artifact acquisition.
- Run 12 is the current local operational checkpoint, but its availability and current failing ML tests are documented in [status](16_current_status.md).

## Common change recipe

For retraining:

1. Freeze and register DATA export/splits; record contamination/leakage results.
2. Pin config, commit, seed, environment, and initial checkpoint.
3. Train while preserving structured logs and aborts.
4. Tune thresholds and calibration on their intended partitions.
5. Evaluate per class, macro/micro metrics, OOD sets, and behavioral probes.
6. Compare interpretability and drift signals with the previous candidate.
7. Promote checkpoint and companions together; reject missing evidence.
8. If the fusion behavior changes materially, redistill and regenerate ZKML.

## Verification commands

```bash
export TMPDIR=/tmp TMP=/tmp TEMP=/tmp
ml/.venv/bin/python -m pytest ml/tests -q                              # module
ml/.venv/bin/python ml/scripts/tune_threshold.py --help                # smoke
ml/.venv/bin/python ml/scripts/promote_model.py --help                 # smoke
ml/.venv/bin/python ml/scripts/compute_drift_baseline.py --help        # smoke
```

Training, calibration against full datasets, GPU inference, and registry promotion are live/expensive operations. See [current status](16_current_status.md) for the baseline—not individual chapter counts.

## Optional deep references

- [`ml/testing_specs`](../../ml/testing_specs) — quality specifications and evidence
- [`ml/reports`](../../ml/reports) — run evidence; verify artifact/commit identity
- [`ml/scripts/interpretability`](../../ml/scripts/interpretability) — experiment implementations
- [Evaluation](13_evaluation.md)
- [Change playbooks](15_change_playbooks.md)

## Technical mastery layer

### Prerequisite knowledge

Know multi-label loss, class imbalance, validation/test separation, calibration, precision/recall/Fβ, and checkpoint state.

### Source map and reading order

Read `trainer.py::TrainConfig`, `::train_one_epoch`, `::evaluate`, and `::train`; then calibration, threshold, promotion, interpretability, and drift scripts. [T04](technical/04_ml_training_quality_mlops.md) connects these phases.

### Execution trace and worked example

Training produces main/auxiliary logits and a checkpoint. Held-out logits receive per-class temperature calibration, threshold sweeping selects operating points, and promotion requires behavioral/quality evidence. A better loss alone is not release evidence.

### Implementation practice

[L04](labs/04_training_calibration_promotion.md) computes a confusion table and calibration effect with synthetic tensors. Any retrain must preserve dataset/model/config hashes plus its calibration/threshold bundle for rollback.

### Review and ownership check

Can you explain which split fits weights, calibration, thresholds, and which split only evaluates?
