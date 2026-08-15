# SENTINEL ML R4 — Trustworthy-Label Recovery and Retraining Program

This directory governs the R4 repair of SENTINEL's DATA/ML training and evaluation foundation.

## Current restart — 2026-08-15

Do **not** restart R4 from Phase 0. Historical G0–G7 are complete and retained as reproducibility evidence. The current boundary is Phase 8:

- repaired-v2 physical DATA is accepted for bounded research under R4-D-008 / ADR-R4-008;
- accepted repaired-v2 publication: 22,540 contracts / 225,400 contract×class rows;
- physical representations: 22,540 / 22,540 contracts and 67,620 / 67,620 files validated;
- all 899 effective supervised cells are target `1`; confirmed-negative support is zero;
- threshold fitting, calibration fitting, and untouched acceptance remain unsupported/empty;
- historical four-window token selection is not accepted as adequate; a target-aware candidate is promising but unpromoted;
- the 100-epoch Phase-8 run is **NOT AUTHORIZED** and G8 remains open.

For current work, read in this order:

1. applicable repository/module `CLAUDE.md` files;
2. `PLAN_STATUS_MATRIX.md`;
3. `DECISION_REGISTER.md` and `adrs/ADR-R4-008-repaired-v2-data-acceptance-and-phase8-no-launch.md`;
4. `RISK_AND_BLOCKER_REGISTER.md`;
5. `runs/2026-08-15_PHASE8_repaired_data_acceptance_and_launch_decision.md`;
6. the exact source/tests for the objective, selector, evaluation, or representation work being changed.

The next authorized work is an evidence-honest objective/evaluation contract (confirmed-negative evidence or a formally bounded positive-unlabeled design), versioned selector diagnostics, compatibility/grouping sensitivity checks, and a credible quality gate. Any changed objective/selector/representation/role lineage must be versioned and rebound before full-training authorization.

## What R4 is solving

R4 starts from an established working premise:

> The historical contract-vulnerability labels are materially untrustworthy, semantically misaligned, or misleading. Prior contract-level investigations already provide substantial evidence for this conclusion.

R4 does **not** begin by asking whether the labels might be wrong. It begins by recovering previous evidence, reconstructing how unreliable labels reached the training export, building a trustworthy evidence-qualified DATA vNext, and retraining/evaluating only when the evidence supports the intended claims.

## Primary deliverable

A versioned DATA/ML bundle containing:

1. source-native label provenance;
2. contract-class evidence states;
3. corrected labels where evidence supports correction;
4. unknown/conflicting/not-applicable masks rather than false negatives;
5. leakage-safe training, selection, calibration, and acceptance partitions where those roles are actually supportable;
6. the existing model architecture retrained only after the objective/evaluation contract is credible;
7. evidence-qualified evaluation and policy thresholds where evidence permits them;
8. migration and rollback artifacts.

## Binding execution constraints

- Reuse prior DIVE, BCCC, manual, tool-assisted, and source-level work before creating new reviews.
- A new contract review requires a registered evidence gap.
- Do not repeat a previous review merely to obtain a cleaner-looking artifact.
- Preserve historical labels/exports and accepted repaired-v2 evidence; create versioned replacements.
- Do not redesign the architecture during the normal R4 path without a separate decision.
- Historical zero does not automatically mean confirmed negative.
- Tools are evidence, not ground truth.
- Do not use the same leakage group for training, model selection, threshold fitting, calibration fitting, and final acceptance.
- Every conclusion must link to retained evidence or be marked unsupported.
- Repository tests do not substitute for protected local physical DATA binding.
- A physically valid dataset does not by itself prove model discrimination quality.

## Authority order

When files conflict, follow this order:

1. Repository/module agent instructions such as `CLAUDE.md`.
2. Executable source/config/tests for actual behavior.
3. `00_MASTER_PLAN.md` and accepted machine-readable policy/artifacts.
4. Approved decision records / ADRs, including R4-D-008 for repaired-v2 physical authority.
5. `PLAN_STATUS_MATRIX.md`, `RISK_AND_BLOCKER_REGISTER.md`, and active run/decision records.
6. Canonical handbook/current-status documentation.
7. Historical plans, bootstrap instructions, and reports.

An active phase or decision may refine execution detail but may not weaken the master evidence rules.

## Historical bootstrap

`START_HERE_AGENT.md` records the original Phase-0 bootstrap assignment. That assignment is complete and historical; it must not be replayed as the current task simply because the file remains in the repository.

## Directory map

```text
ml-R4/
├── START_HERE_AGENT.md          # historical bootstrap + current redirect
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
