# Phase 9 — Evidence-Qualified Evaluation, Calibration, Thresholds, and Policy

**Status:** WAITING FOR G8  
**Gate:** G9

## Objective

Measure what the DATA vNext model supports and define policy without overstating probability outputs.

## Discrimination

On confirmed outcomes:

- Average Precision/PR-AUC;
- precision-recall curves;
- ROC-AUC as supplementary;
- log loss;
- Brier score;
- class/source/era/size strata;
- cluster-aware confidence intervals.

## Calibration

- fit only on calibration role;
- assess on independent internal/acceptance roles;
- report reliability, slope/intercept, Brier, and ECE sensitivity;
- do not fit unsupported classes.

## Thresholds

- define costs and review budget;
- select only on threshold-fit role;
- report full curves and independent assessment;
- keep raw probability separate from policy.

## Uncertainty

- source-removal sensitivity;
- crosswalk-policy sensitivity;
- indeterminate-outcome bounds;
- duplicate-family sensitivity;
- small-support warnings.

## Policy

For each class choose:

- validated for defined use;
- provisional;
- training-only;
- unsupported for outcome claims;
- disabled pending evidence.

Audit terms such as `safe` and `confirmed_vulnerable`. Add abstention where evidence does not support a strong verdict.

## G9 pass criteria

Each class has a bounded evidence-based claim and operational policy.
