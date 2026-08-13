# R4 Phase 8 — Existing Architecture Retraining Execution Plan

**Date:** 2026-08-13  
**Branch:** `r4/phase8-existing-model-retraining`  
**Entry gate:** G7 PASS  
**Target gate:** G8

## Objective

Produce a reproducible checkpoint of the existing ten-output Four-Eye architecture trained against the exact G7-passed DATA vNext lineage, without recreating historical zero-as-negative semantics and without using unsupported threshold/calibration/acceptance roles.

## Immutable input authority

- dataset: `sentinel-r4-vnext-v1`
- export schema: `v2`
- graph schema: `v9`
- representation binding digest: `7637461f6643d398c7a0446412fedd8877914c7b9ed41309dab45f18ed96f420`
- frozen Phase-6 roles are authoritative
- class order remains the locked ten-class order
- `GasException` and `UnusedReturn` remain supervision-disabled
- historical v1 labels/splits/thresholds/calibration are not fallback inputs

## Phase-8 semantic rules

1. Unknown/not-reviewed/masked cells remain nullable and must fail closed if passed to an unmasked binary loss.
2. Supervised optimizer cells are only `effective_loss_mask=true` rows from `TRAIN_STRONG` and `TRAIN_WEAK` contracts.
3. `TRAIN_UNLABELED`, `EXCLUDED`, `INTERNAL_AUDIT`, threshold/calibration roles, and untouched acceptance are not optimizer or checkpoint-selection inputs.
4. `MODEL_SELECTION` is positive-only limited support. It may drive positive-loss/checkpoint diagnostics only; it does not authorize precision/F1/AUC/FPR/calibration claims.
5. Weak positive evidence is not confirmed truth. Phase 8 uses an explicit optimizer coefficient `weak_positive_weight=0.25` and binds that value into every checkpoint/run manifest. The value is a conservative optimization weight, not an evidence probability or DATA semantic.
6. Strong positive optimizer weight is `1.0`.
7. Historical label smoothing is disabled in the vNext baseline because softening a positive toward zero would introduce unsupported negative mass.
8. Historical class-specific negative/threshold heuristics are not silently reused. Any retained optimization hyperparameter must be explicit in the Phase-8 run config.
9. No threshold sweep occurs in Phase 8. A fixed `0.5` positive recall may be logged only as a diagnostic. Raw probabilities and threshold-free positive loss are primary.
10. No semi-supervised / PU-learning objective is introduced in this phase. The large `TRAIN_UNLABELED` population remains available for future explicitly authorized work but is not silently treated as negative.

## Checkpoint-selection metric

The primary model-selection signal is **masked positive negative-log-likelihood** over `MODEL_SELECTION` cells with `outcome_metric_mask=true`:

`mean(softplus(-logit))`

Lower is better. Supporting diagnostics:

- mean positive probability on the same authorized cells;
- fixed-threshold positive recall at `0.5`;
- per-class positive loss/recall only where metric cells exist.

No F1, Hamming, specificity, precision, AUC, Brier, ECE, threshold tuning, or calibration is interpreted as valid Phase-8 model-selection evidence because confirmed-negative support is absent.

## Implementation milestones

### P8.1 — Historical bundle inventory

- read the local Run12 checkpoint/config where feasible;
- record architecture/config differences from current source;
- do not resume optimizer state from Run12 into the repaired-data run;
- use the inventory to distinguish architecture identity from old data-specific heuristics.

### P8.2 — DATA vNext ML compatibility seam

Implement an explicit vNext dataset adapter over the existing per-contract graph/token representation files and `ml_targets.parquet`.

Required batch semantics:

- targets `[B,10]` with unknown cells represented as `NaN`;
- effective loss mask `[B,10]` bool;
- outcome metric mask `[B,10]` bool;
- training-strength code `[B,10]`;
- role, group ID, contract ID;
- graph/token tensors unchanged.

The adapter must require the G7 representation-bound publication and must have no v1 label fallback.

### P8.3 — Masked optimization/evaluation

- masked main loss;
- masked auxiliary losses;
- weak/strong optimizer weights;
- disabled classes remain mask-zero;
- group-balanced training sampler based on frozen group IDs, not historical labels;
- positive-only model-selection metrics;
- no threshold sweep.

### P8.4 — Artifact/run binding

Every Phase-8 run/checkpoint binds at minimum:

- source commit;
- architecture identifier/model version;
- G7 manifest SHA-256;
- G7 representation binding digest;
- policy/partition versions;
- role counts used;
- class order;
- random seed;
- weak-positive weight;
- optimizer/loss configuration;
- exact training/model-selection contract/group counts.

### P8.5 — Tests before GPU work

Required tests:

- nullable target cannot contribute without a mask;
- no target `0` is created by the adapter;
- training dataset contains only `TRAIN_STRONG`/`TRAIN_WEAK`;
- model-selection dataset contains only `MODEL_SELECTION`;
- disabled classes contribute zero optimizer cells;
- weak cells receive only the configured weak coefficient;
- group sampler does not use legacy label heuristics;
- model-selection metrics ignore non-authorized cells;
- v1 export passed to the vNext path fails explicitly;
- checkpoint/run binding changes if any bound DATA/config identity changes.

### P8.6 — Local smoke training

Only after P8.2–P8.5 are green:

- run a tiny deterministic local smoke training on the user's GPU;
- inspect loss/gradient/probability behavior;
- verify checkpoint bindings;
- do not interpret the smoke checkpoint as a quality result.

### P8.7 — Full repaired baseline training

Only after the smoke gate passes:

- launch the full existing-architecture run;
- preserve intermediate/best checkpoints and structured logs;
- select checkpoint only by the authorized positive-only model-selection signal;
- record any all-positive/other degeneracy as a result rather than repairing it with invented negatives.

## Explicit risks carried into Phase 8

- There are zero confirmed-negative training cells. Positive-only supervised learning can drive broad overprediction; that is a data-evidence limitation, not permission to synthesize negatives.
- `MODEL_SELECTION` is positive-only. Phase 8 cannot establish false-positive behavior or calibrated decision thresholds.
- `GasException` and `UnusedReturn` cannot receive supervised loss under policy v1.
- Phase 9/10 remain blocked from unsupported calibration/untouched-acceptance claims until new evidence exists.

## G8 pass condition

A reproducible existing-architecture checkpoint is produced from the exact G7 DATA vNext lineage, with correct masks/roles/weak-strength handling, checkpoint-bound configuration and seed, no acceptance leakage, and no historical-zero fallback.
