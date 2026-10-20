# SENTINEL ML R4 — Trustworthy-Label Recovery and Retraining Program

This directory governs the R4 repair of SENTINEL's DATA/ML training and evaluation foundation.

## Current restart — 2026-10-20

Do **not** restart from Phase 0, old DATA construction, V2.5/V2.6 staging, M4 execution, or the now-closed `CallToUnknown` definition audit. Historical G0–G7 are complete. Phase 8 remains `IN_PROGRESS`; G8 is not passed and M5/full training is **NOT AUTHORIZED**.

Current boundary:

- R4-D-009 accepts logical V3 grouping/roles/publication;
- R4-D-011 accepts the exact immutable V10/V2.6 graph/control root with digest `d9f925588913e66476cfbc097bace7daa7e673295fe2a243760313d0bef5ebdd`;
- R4-D-013 accepts the exact immutable guarded-token successor with 22,540 identities (14,751 guarded / 7,789 historical fallback) and digest `9885d7b88a46aff4102741d63eeaa0bd6968f0857f7bca6a389662bfaa158881`;
- R4-D-014 accepts `masked_positive_bce_control_v1` only for bounded positive-only control/evaluation; MODEL_SELECTION is positive-fit diagnostic only;
- ML M0-M4 are complete;
- R4-D-015 records `PASS_BOUNDED_PILOT_FULL_TRAINING_STILL_HELD` after the fixed 8-epoch / 120-step M4-C pilot;
- R4-D-016 resolves `CallToUnknown` as the raw/dynamic low-level unknown-target class without changing class index/order or taxonomy schema v1;
- in scope under R4-D-016: `call`, `delegatecall`, legacy `callcode`, applicable `staticcall`, and `send`;
- not positive solely from destination uncertainty: `transfer`, ordinary typed high-level calls, in-scope library calls, contract creation, and selfdestruct/suicide destinations;
- confirmed-negative support remains zero;
- candidate #1 historical `NOT_CONFIRMED` rationale is definition-stale; its current queue state remains `UNKNOWN / PENDING_REVIEW / target None` and it needs a fresh primary review if reused;
- candidate #2 primary review remains semantically compatible but authoritative state remains `UNKNOWN / PENDING_REVIEW / target None` until admissible independent agreement exists;
- three recent external candidate #2 review attempts are diagnostic/non-authoritative because of prior-context contamination and/or insufficient neutral governing material;
- candidate #2 blind-review v1 is historical only and retired for future adjudication;
- the active R4-GAP-007 work is to build/verify the self-contained candidate #2 v2 blind bundle and obtain genuinely isolated independent review;
- threshold fitting, calibration fitting and untouched acceptance remain unsupported/empty;
- M5/full training, G8 and production promotion remain unauthorized;
- Run12 remains the historical operational baseline.

## Current read order

1. applicable repository/module `CLAUDE.md` files;
2. `PLAN_STATUS_MATRIX.md`;
3. `adrs/ADR-R4-016-call-to-unknown-semantic-clarification.md` and `evidence/2026-10-20_call_to_unknown_semantic_alignment/decision.json`;
4. `docs/plan/system-finalization/technical-completion/02_ML_TRAINING_EVALUATION_COMPLETION_PLAN.md`;
5. `runs/2026-10-17_PHASE8_m4_bounded_pilot_review_and_full_training_hold.md` and `adrs/ADR-R4-015-bounded-pilot-pass-full-training-hold.md`;
6. `EVIDENCE_GAP_REGISTER.md`, `RISK_AND_BLOCKER_REGISTER.md`, `CLAIM_STATUS_MATRIX.md`, and `DECISION_REGISTER.md`;
7. R4-D-011/R4-D-013/R4-D-014 records when exact physical/objective authority is needed;
8. hardened V3 evidence snapshot when logical/evaluation population detail is needed.

Historical execution records—including the 2026-10-03 M4 working record and the closed 2026-10-20 semantic-alignment working record—may contain truthful *then-current* stop lines. They are not current restart authority after R4-D-015/R4-D-016.

## Immediate work

The active task is **candidate #2 independent-review v2 publication and isolation**, not another class-definition debate.

Use the committed builder:

`docs/plan/ml-R4/scripts/p8_build_gap007_candidate2_independent_review_v2.py`

The v2 archive must be self-contained and contain only:

- complete candidate source;
- manifest and hashes;
- review task;
- neutral R4-D-016 class definition/exclusions;
- generic class-specific confirmed-negative policy.

It must contain no primary-review result or prior candidate verdict. The reviewer must have no candidate-specific prior conclusion in context; if it does, it must stop as procedurally contaminated.

Until admissible independent evidence returns:

- candidate #2 remains UNKNOWN/PENDING_REVIEW/target `None`;
- contaminated or under-specified review attempts remain diagnostic only;
- do not infer target `0`;
- do not launch M5/full training.

## What R4 is solving

R4 starts from an established working premise:

> Historical contract-vulnerability labels are materially untrustworthy, semantically misaligned, or misleading. Prior contract-level investigations already provide substantial evidence for this conclusion.

R4 recovers prior evidence, reconstructs how unreliable labels reached training, builds evidence-qualified DATA vNext, and trains/evaluates only when the evidence supports the intended claims.

## Primary deliverable

A versioned DATA/ML bundle containing:

1. source-native label provenance;
2. contract-class evidence states;
3. corrected labels where evidence supports correction;
4. unknown/conflicting/not-applicable masks rather than false negatives;
5. leakage-safe training/selection/calibration/acceptance roles only where supportable;
6. the existing model architecture trained only after objective/evaluation authority is credible;
7. evidence-qualified evaluation/policy artifacts only where evidence permits them;
8. migration and rollback artifacts.

## Binding execution constraints

- Reuse prior DIVE, BCCC, manual, tool-assisted and source-level work before creating new reviews.
- A new contract review requires a registered evidence gap.
- Do not repeat a previous review merely to obtain a cleaner-looking answer.
- Preserve historical labels/exports and all accepted R4-D-009/R4-D-011/R4-D-013 roots; create versioned replacements.
- Do not redesign the architecture during the normal R4 path without a separate decision.
- Historical zero does not mean confirmed negative.
- Tools are evidence, not ground truth.
- Do not use the same leakage group across incompatible training/evaluation roles.
- Every conclusion must link to retained evidence or be marked unsupported.
- Repository tests do not substitute for protected-local physical binding.
- Physical validity does not establish model discrimination quality.
- R4-D-016 is current `CallToUnknown` authority; do not silently broaden/narrow it to force an adjudication.
- Do not count reviewer contamination as independence.
- Do not invent negative targets, PU assumptions, thresholds, calibration or untouched acceptance populations.
- Do not launch M5 until a later explicit training-authorization decision exists.

## Authority order

When files conflict:

1. applicable repository/module instructions, interpreted together with current committed R4 governance;
2. executable source/config/tests for actual behavior;
3. accepted machine-readable policy/artifacts and ADR/decision authority;
4. `PLAN_STATUS_MATRIX.md`, R4 registers and current technical-completion plans/checkpoints;
5. canonical handbook/current-status documentation;
6. dated historical plans/reports.

An old dated report may preserve historical truth but may not override a later accepted decision or current status record.

## Directory map

```text
ml-R4/
├── START_HERE_AGENT.md
├── 00_MASTER_PLAN.md
├── KNOWN_PREMISE_AND_NON_DUPLICATION_POLICY.md
├── MODEL_ARCHITECTURE_FREEZE.md
├── LABEL_STATE_AND_DATASET_ROLE_POLICY.md
├── PLAN_STATUS_MATRIX.md
├── EXECUTION_LOG.md
├── ARTIFACT_INDEX.md
├── PREVIOUS_EVIDENCE_REGISTER.md
├── EVIDENCE_GAP_REGISTER.md
├── DECISION_REGISTER.md
├── RISK_AND_BLOCKER_REGISTER.md
├── CLAIM_STATUS_MATRIX.md
├── adrs/
├── phases/
├── workstreams/
├── templates/
├── manifests/
├── findings/
├── scripts/
├── decisions/
└── runs/
```
