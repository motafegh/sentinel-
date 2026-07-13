# T04 — Training, calibration, quality, interpretability, and MLOps

## Learning outcome

You can follow a training run from configuration through promotion evidence, compute the purpose of each loss/calibration/threshold stage, and identify where interpretability and drift checks influence release decisions.

## Prerequisites

Read [ML training and quality](../06_ml_training_quality.md). Know BCE-with-logits, class imbalance, precision/recall/Fβ, validation leakage, and checkpoint state dictionaries.

## Source map and reading order

1. `ml/src/training/trainer.py::TrainConfig`, `::compute_pos_weight`, `::train_one_epoch`, `::evaluate`, and `::train`.
2. Loss modules under `ml/src/training/` and `SentinelModel.forward(return_aux=True)`.
3. `ml/scripts/calibrate_temperature.py::{fit_temperatures,compute_ece}`.
4. `ml/scripts/tune_threshold.py::{sweep_one_class,apply_thresholds,evaluate_overall}`.
5. `ml/scripts/promote_model.py::promote` and behavioral/label-quality checks.
6. `ml/src/inference/drift_detector.py` and `ml/scripts/interpretability/`.

## Entry point and complete call chain

`trainer.py::train` validates `TrainConfig`, builds datasets/loaders/model/optimizer/scheduler, restores resume state when requested, and iterates phases and epochs. `train_one_epoch` obtains main and auxiliary logits, combines configured losses, scales/backpropagates, clips/steps, and records diagnostics. `evaluate` gathers logits/labels and computes model-selection evidence. The selected checkpoint is then calibrated on held-out logits, thresholds are swept per class against the chosen objective, and promotion checks compare evidence, probes, and label quality before changing model stage.

## Important symbols and configuration

- `TrainConfig` owns architecture, data paths, precision, accumulation, loss weights, phase behavior, and reproducibility controls.
- `compute_pos_weight` compensates multi-label class imbalance; focal variants emphasize hard examples.
- Auxiliary losses keep individual eyes and the CEI-focused phase-2 path useful.
- Temperature scaling changes probability calibration, not class ranking.
- Per-class thresholds turn probabilities into operational decisions; they are release artifacts.
- Drift monitoring observes inference distributions; interpretability scripts test whether learned channels behave plausibly.

## Annotated source excerpt

Source: `ml/src/training/trainer.py::compute_pos_weight`

```python
def compute_pos_weight(
    dataset: SentinelDataset,
    num_classes: int,
    device: torch.device,
    *,
    max_weight: float = 50.0,
) -> torch.Tensor:
```

This boundary computes training-only imbalance weights from the selected dataset. It must not be recomputed from test labels, and its output is not an inference threshold.

## Worked example

For one class with 100 validation positives, a threshold yields `tp=72`, `fp=18`, `fn=28`: precision `0.80`, recall `0.72`. With β greater than one, missing positives is penalized more, so a slightly lower threshold may win even with extra false positives. Temperature `T>1` softens overconfident logits before the sweep. The chosen threshold and temperature must be stored with the checkpoint/model hash; otherwise the same weights produce a different product decision.

## Success trace

Run configuration and dataset identity are recorded; losses remain finite; gradients reach all intended eyes; held-out metrics and calibration artifacts are generated; thresholds are class-specific; behavioral probes and promotion gates pass; the promoted model hash matches inference health output.

## Failure trace

NaN/Inf loss, absent gradients, schema mismatch, or inconsistent resume state stops training. A threshold tuned on test data invalidates evaluation. Better aggregate F-score does not override a failed safety probe. Drift can warn without automatically proving model degradation; investigate input/artifact changes first.

## Design reasoning and rejected alternatives

Multi-stage quality exists because training loss is not an operational guarantee. Calibration separates confidence quality from ranking, and threshold tuning separates ranking from the desired precision/recall tradeoff. A single scalar threshold was rejected for heterogeneous vulnerability classes. Promotion is evidence-gated rather than “latest checkpoint wins.”

## Safe change walkthrough

To change a loss weight, add/adjust a synthetic unit test, run a short controlled experiment with fixed seed/data identity, compare per-class and auxiliary diagnostics, recalibrate, retune thresholds, run interpretability/probes, and create promotion evidence. Roll back by restoring the previous checkpoint plus its exact calibration/threshold bundle—not weights alone.

## Guided lab

Complete [L04 — training, calibration, threshold, and promotion](../labs/04_training_calibration_promotion.md).

## Tests and expected results

```bash
TMPDIR=/tmp TMP=/tmp TEMP=/tmp ml/.venv/bin/python -m pytest \
  ml/tests/test_trainer.py ml/tests/test_predictor.py ml/tests/test_promote_model.py -q
```

Expected current truth: synthetic loss/threshold behavior passes, while two promotion tests fail because their fixtures do not create the behavioral-probe JSON now required by promotion. The exact focused result is in [current status](../16_current_status.md). Full training, GPU, MLflow, or checkpoint checks are live/module work with explicit prerequisites.

## Review questions

Why are `pos_weight`, temperature, and threshold three different controls? Which split may tune each? What evidence travels with a checkpoint? When should drift trigger rollback versus investigation?

## Ownership checklist

- I can reconstruct a run from config, data hash, and checkpoint.
- I can explain every loss contribution and decision artifact.
- I keep test labels out of fitting/tuning.
- I can reject promotion even when one aggregate metric improves.
