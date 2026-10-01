# ADR-R4-014 — Phase-8 positive-only control and evidence-limited evaluation boundary

Date: 2026-10-01  
Status: ACCEPTED  
Decision ID: R4-D-014

## Context

R4-D-013 closes the DATA/representation physical-lineage gap, but physical validity does not answer the remaining ML question: what objective and evaluation semantics are justified by the available labels.

M0 proved that the current optimizer, masks, role isolation, positive-only metrics, checkpoint/resume mechanics and model architecture can support a repaired lineage after integration, but the canonical runner is still wired to historical G7/v9.

M1 then established the evidence boundary:

- confirmed-negative cells remain zero;
- R4-GAP-007 candidate #1 is NOT_CONFIRMED;
- candidate #2 has primary support only and still requires a genuinely independent agreeing verifier;
- masked positive BCE directly rewards response on known positives but contains no direct penalty for high scores on unknown cells;
- positive-only NLL can measure positive fit but cannot establish discrimination, false-positive behavior, thresholds, calibration or production quality;
- PU/nnPU is not currently justified because the accepted positive labels are source/class-conditioned rather than SCAR-like, while no accepted SAR propensity model, class-prior estimate or identifiability contract exists.

## Decision

Adopt `masked_positive_bce_control_v1` as the Phase-8 **bounded control objective**, not as full-horizon model-quality authority.

Optimizer authority remains exactly:

- `TRAIN_STRONG` positive cells, target `1.0`, strength weight 1.0;
- `TRAIN_WEAK` positive cells, target `1.0`, weak-positive weight 0.25;
- every unknown/unreviewed cell masked from supervised loss;
- no confirmed-negative optimizer cells;
- no `TRAIN_UNLABELED` optimizer use;
- no PU/nnPU risk term;
- no pseudo-negatives or inferred zeros.

The existing executable loss composition remains the control contract:

- main masked positive BCE;
- GNN / transformer / fused auxiliary masked positive BCE heads;
- auxiliary weight 0.3 with 8-epoch warmup;
- Phase-2 auxiliary masked positive BCE weight 0.2;
- JK entropy regularization lambda 0.005;
- no label smoothing.

Architecture remains frozen as `four_eye_v8` / `v8.1`; the repaired integration must explicitly instantiate graph schema V10 and bind the accepted GraphCodeBERT revision.

## Evaluation interpretation

`MODEL_SELECTION` may produce only positive-fit diagnostics:

- positive NLL;
- mean positive probability;
- positive recall at fixed diagnostic threshold 0.5.

Lower positive NLL may identify the best **positive-fit** checkpoint within a bounded pilot. It must not be described as best classifier, best discrimination, production candidate, calibrated checkpoint, or promotion winner.

`INTERNAL_AUDIT` remains observational and must not influence optimization or checkpoint selection.

The following remain unsupported:

- discrimination / specificity / false-positive rate;
- precision, ordinary binary F1, AUROC or PR-AUC against true negatives;
- threshold fitting;
- calibration fitting;
- untouched acceptance;
- production checkpoint promotion.

## Execution authority

This decision authorizes:

1. M3 engineering to bind the executable training seam to logical V3 + R4-D-011 + R4-D-013;
2. M4 preflight and a separately bounded pilot after the M3 integration passes fail-closed tests.

This decision does **not** authorize the 100-epoch/full-horizon run.

The M4 pilot horizon must be fixed and bound before execution, must be materially below the full 100-epoch horizon, and exists to test runtime correctness, optimization dynamics, checkpoint/resume behavior and the limited positive-fit diagnostics. It does not create a model-quality claim.

## PU disposition

PU/nnPU remains a future research option, not implementation authority. A later proposal must first bind a target population, positive-label selection/propensity model, per-class prior or defensible estimator assumptions, leakage-safe unlabeled optimizer population, misspecification sensitivity, and evaluation evidence capable of falsifying the proposal.

## Confirmed-negative continuation

R4-GAP-007 remains open independently. Any later accepted negative remains evaluation-only unless another versioned decision grants optimizer authority.

One or a few accepted negatives do not automatically authorize thresholds/calibration or establish a stable FPR estimate.

## Consequences

- M2 is complete.
- M3 becomes the next technical-completion work package.
- The historical G7/v9 runner remains a reproducibility path and must not be silently repurposed.
- The R4-D-013 integration must fail closed on lineage/schema/selector/decision mismatches.
- Full training remains HOLD; G8 remains open.

## Rollback

Rollback is the current state: no R4-D-013 training integration and no full run. R4-D-011/R4-D-013 physical artifacts remain immutable regardless of ML objective decisions.

## Evidence

- `docs/plan/system-finalization/technical-completion/working/2026-10-01_ml_m0_training_seam_audit.md`;
- `docs/plan/system-finalization/technical-completion/working/2026-10-01_ml_m1_supervision_evaluation_evidence.md`;
- `docs/plan/ml-R4/EVIDENCE_GAP_REGISTER.md` R4-GAP-007;
- `docs/plan/ml-R4/reviews/R4-GAP-007/candidate2_primary_review_v1.json`;
- `ml/src/training/vnext_losses.py`;
- `ml/src/training/vnext_epoch.py`;
- machine decision `evidence/2026-10-01_phase8_objective_evaluation/decision.json`.
