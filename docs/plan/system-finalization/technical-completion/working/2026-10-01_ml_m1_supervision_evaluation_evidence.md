# ML M1 — supervision / evaluation evidence investigation

**Date:** 2026-10-01  
**Plan:** `02_ML_TRAINING_EVALUATION_COMPLETION_PLAN.md`  
**Authority:** accepted logical V3 + R4-D-011 graph parent + R4-D-013 guarded-token physical successor  
**State:** COMPLETE WITH R4-GAP-007 OPEN — evidence-backed M2 recommendation ready

## Objective

Determine what the current supervision can honestly optimize and evaluate, whether Positive–Unlabeled (PU) learning is presently defensible, and which evaluation roles remain supported or unsupported before any versioned objective decision.

This investigation does not create new labels, targets, thresholds, calibration authority, or training authorization.

## M1-A — confirmed-negative evidence

### Current durable state

Confirmed-negative support remains **zero**.

R4-GAP-007 remains the only approved current negative-evidence process:

- committed hardened V3 queue: 200 cells / 200 globally unique groups;
- 25 queued candidates per enabled class;
- every queued cell remains `UNKNOWN / PENDING_REVIEW / target=None`;
- queue membership is review reservation, not negative truth;
- source absence, historical zero, static-tool silence, and unlabeled state are explicitly insufficient for target `0`.

Candidate #1 is durably `NOT_CONFIRMED`.

Candidate #2 `r4neg-bfe90ef82e33a324d612256a5d4053c6` has a complete primary `CallToUnknown` review that supports a class-specific negative, but the authoritative state remains unchanged because a genuinely distinct independent reviewer has not yet agreed.

The validator correctly requires:

1. class-specific negative scope;
2. complete primary code/graph review;
3. direct primary evidence;
4. an independent verifier with a different reviewer identity;
5. explicit independent status `AGREES`;
6. nonempty independent evidence.

Even an accepted cell is `EVALUATION_ONLY_NOT_TRAINING_AUTHORITY`; it does not authorize optimizer target `0`, threshold fitting, or calibration fitting.

### M1-A disposition

Do not self-verify candidate #2 and do not manufacture a substitute negative.

R4-GAP-007 remains open in parallel. Its current absence of accepted negatives must be treated as a hard limitation in M2.

The existing 59-negative/class calculation for observing zero false positives while bounding FPR below 5% at 95% one-sided confidence is **planning-only**. It assumes a meaningful sample design and independence; it is not a claim that 59 queue reviews automatically create an acceptance-quality evaluation set.

## M1-B — positive-only learning limits

### What the current objective actually optimizes

The current authorized loss is masked positive BCE.

For every optimizer-authorized cell, the target is exactly `1.0`. Unknown cells are masked out. Strong positives receive weight 1.0; weak positives use the explicit configured weight.

For one positive cell with logit `z`:

`L+(z) = softplus(-z) = -log(sigmoid(z))`.

Its derivative is `sigmoid(z) - 1`, so optimization pushes authorized positive logits upward. There is no direct loss term penalizing a high logit on an unknown cell.

This is not a bug: it is the mathematically correct consequence of refusing to turn unknowns into negatives.

### What positive-only training can support

Under the current evidence it can support only bounded statements such as:

- optimizer/runtime mechanics function;
- known positive cells can receive increasing model response;
- strong/weak weighting behaves as configured;
- positive-only NLL/probability/recall diagnostics can be reproduced;
- a checkpoint can be compared on **positive fit only**.

### What it cannot support

It cannot establish:

- vulnerability-vs-non-vulnerability discrimination;
- false-positive rate or specificity;
- precision, ordinary binary F1, AUROC, or PR-AUC over true positives/negatives;
- trustworthy decision thresholds;
- calibration;
- a claim that higher positive probability is globally better;
- production readiness.

A degenerate model that scores many unknown/negative contracts highly can still obtain excellent positive-only NLL. Therefore `best_positive_nll` is a limited positive-fit checkpoint diagnostic, not a quality/promotion criterion.

### Full-horizon implication

A 100-epoch positive-only run could be computationally reproducible yet still leave the central discrimination question unanswered. Current evidence does not justify spending the full training horizon merely to produce such a checkpoint.

Positive-only BCE remains useful as the **control objective for bounded mechanics/dynamics experiments**, not as current full-run quality authority.

## M1-C — PU-learning investigation

### External methodological evidence

Relevant research reviewed:

1. Kiryo et al., *Positive-Unlabeled Learning with Non-Negative Risk Estimator*, NeurIPS 2017: non-negative PU (nnPU) addresses the negative empirical-risk/overfitting pathology of unbiased PU estimators for flexible models.
2. Bekker & Davis, *Learning from Positive and Unlabeled Data under the Selected At Random Assumption*, PMLR 2018: PU learning requires assumptions about class distribution and/or the positive-label selection mechanism; SCAR is strong, SAR requires an explicit feature-conditioned selection mechanism.
3. Coudray et al., *Risk Bounds for Positive-Unlabeled Learning Under the Selected At Random Assumption*, JMLR 2023: SAR risk analysis relies on the positive-label propensity; if unknown it must be estimated under additional assumptions/models.
4. He, Liang & Liu, *Identifying Labeling Mechanism in Positive–Unlabeled Learning under Unknown Class Prior*, PMLR/UAI 2026: mechanism misspecification can systematically bias/unstabilize PU learning; distinguishing SCAR from SAR is itself a statistical inference problem.

### SENTINEL assumption check

The accepted DATA policy is visibly **not SCAR-like**.

Positive supervision is created through source/class-specific mechanisms:

- SolidiFI injected class → strong positive for the injected class;
- approved SmartBugs Curated category → strong positive for its mapped class;
- non-target classes stay unknown;
- DIVE is mostly unlabeled structure;
- only DIVE Front Running→TOD is weak-positive;
- multiple sources/classes are masked/deferred/excluded.

Thus the probability that a true positive becomes labeled is deliberately dependent on source, class, benchmark construction, and evidence process. The labeled-positive pool is not established to be an iid random sample from the unknown full positive population.

A SAR formulation is not automatically invalid, but the repository currently does **not** establish:

- a validated propensity model `P(labeled | Y=positive, X)` or a defensible lower-dimensional sufficient attribute set;
- per-class positive class priors for the target population;
- identifiability assumptions needed to estimate those priors from this corpus;
- a common target-population sampling model that makes the curated/injected positive sources and predominantly DIVE unlabeled pool exchangeable after conditioning;
- an evaluation population capable of detecting PU misspecification.

nnPU does not solve these missing assumptions. Its non-negative correction addresses empirical-risk overfitting once a valid PU risk formulation exists; it does not make a biased labeling mechanism or unknown class prior disappear.

### M1-C disposition

**Do not authorize PU/nnPU as the M2 production training objective under current evidence.**

PU may remain a future bounded research candidate only if a later work item first specifies and tests:

1. target population per class;
2. positive-label selection mechanism;
3. class-prior or propensity estimation assumptions;
4. leakage-safe unlabeled optimizer population excluding reserved evaluation groups;
5. sensitivity to prior/propensity misspecification;
6. independent negative/evaluation evidence capable of falsifying the approach.

No PU implementation change is warranted now.

## M1-D — evaluation-role feasibility

| Role / claim | Current status | Evidence-honest use |
|---|---|---|
| TRAIN_STRONG / TRAIN_WEAK | SUPPORTED FOR POSITIVE-ONLY CONTROL | masked positive optimization only |
| MODEL_SELECTION | LIMITED | positive-NLL/probability/positive-recall diagnostics only; not discrimination/promotion |
| INTERNAL_AUDIT | LIMITED | positive-only observational audit; must remain outside checkpoint selection |
| Confirmed-negative evaluation | UNSUPPORTED CURRENTLY | zero accepted cells; R4-GAP-007 remains open |
| Discrimination metrics | UNSUPPORTED | no accepted negative population |
| THRESHOLD_FIT | UNSUPPORTED_EMPTY | do not fit |
| CALIBRATION_FIT | UNSUPPORTED_EMPTY | do not fit |
| UNTOUCHED_ACCEPTANCE | UNSUPPORTED_EMPTY_FROZEN | do not simulate or repurpose another role |
| Production checkpoint promotion | UNSUPPORTED | requires later evidence/governance |
| Full-horizon G8 training | NOT AUTHORIZED | current evidence does not justify launch |

If candidate #2 later passes independent verification, one evaluation-only `CallToUnknown` negative would permit inspection of that single case. It would not by itself establish an FPR, threshold, calibration curve, or optimizer negative authority.

## M1 recommendation to M2

M2 should adopt an explicit **evidence-limited hold** rather than choose a stronger objective by wishful inference:

1. keep the existing masked positive BCE semantics as the reproducible **positive-only control objective**;
2. allow it only for bounded preflight/pilot mechanics and training-dynamics investigation until later authority changes;
3. retain positive-NLL as a positive-fit diagnostic, not a model-quality or production-promotion criterion;
4. do not authorize PU/nnPU now;
5. do not authorize supervised negatives now;
6. preserve every unknown cell as unknown;
7. preserve R4-GAP-007 reserved groups outside any future unlabeled optimizer population;
8. keep threshold, calibration, untouched acceptance, discrimination and production claims unsupported;
9. keep the 100-epoch/full-horizon run on HOLD;
10. make any later expansion depend on new evidence through a versioned decision, not an implementation shortcut.

This recommendation permits M3 engineering of the accepted R4-D-013 lineage and M4 bounded pilot **only after M2 explicitly defines that limited scope**. It does not authorize the full repaired training run.

## Remaining evidence gap

R4-GAP-007 remains open independently. Candidate #2 needs genuinely independent review from the existing blind bundle. The outcome of that review may refine a future M2 successor, but M1 does not stall or fabricate evidence while waiting for it.

## M1 disposition

**M1 COMPLETE WITH EVIDENCE GAP OPEN.**

The current evidence is sufficient to decide what must *not* be claimed or optimized and to formulate a bounded M2 contract. It is insufficient to authorize full-horizon model training, negative supervision, PU risk optimization, threshold fitting, calibration, or model-quality promotion.
