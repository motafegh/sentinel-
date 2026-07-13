# L04 — Loss, calibration, threshold, and promotion evidence

## Learning objective

Compute decision behavior from small tensors and distinguish model fitting, probability calibration, threshold tuning, and promotion.

## Prerequisites

Read [T04](../technical/04_ml_training_quality_mlops.md). Use the ML environment.

## Source reading order

`ml/tests/test_trainer.py` → `trainer.py::{compute_pos_weight,evaluate}` → `calibrate_temperature.py` → `tune_threshold.py::sweep_one_class` → `promote_model.py::promote`.

## Setup and artifact requirements

Tier is module. Synthetic tests need no production checkpoint. Promotion inspection can use a copied/local checkpoint and must never change production stage.

## Initial observation

```bash
TMPDIR=/tmp TMP=/tmp TEMP=/tmp ml/.venv/bin/python -m pytest \
  ml/tests/test_trainer.py ml/tests/test_predictor.py ml/tests/test_promote_model.py -q
```

## Controlled edit

Add a test-only table of logits/labels for one class. Compute sigmoid, apply two thresholds, and assert the resulting TP/FP/FN and Fβ ordering. Add an assertion that temperature greater than one moves probabilities toward 0.5 without reversing logit order. Do not edit stored thresholds.

## Expected success output

The table reproduces the manual confusion matrix; lower threshold increases or preserves recall; temperature changes calibration; a promotion fixture rejects unknown stage or missing required evidence.

## Expected failure output

Using probabilities as logits or tuning against test labels should make the expected calculation/evaluation contract fail. An aggregate improvement must not bypass behavioral gate failure. The current observation command has two recorded promotion-test failures because their fixtures do not create the behavioral-probe JSON now required by `promote`; distinguish those baseline failures from the intentional exercise failure.

## Verification

Run the selected tests and `verify_handbook.py lab --check L04`.

## Reset and cleanup

Restore modified test files and delete only test-generated temporary evidence. Do not alter model registry stages.

## Completion rubric

Complete when you can explain the independent purpose and fitting split for weights, temperature, threshold, and promotion gates.

## Review questions

Which operation changes ranking? Why use per-class thresholds? What must roll back with a checkpoint?

## Classification

Module; safe preflight; controlled synthetic-test edit.
