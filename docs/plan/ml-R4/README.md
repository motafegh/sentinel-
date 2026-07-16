# SENTINEL ML R4 — Trustworthy-Label Recovery and Retraining Program

This directory governs the R4 repair of SENTINEL's DATA/ML training and evaluation foundation.

## What R4 is solving

R4 starts from an established working premise:

> The current contract-vulnerability labels are materially untrustworthy, semantically misaligned, or misleading. Prior contract-level investigations already provide substantial evidence for this conclusion.

R4 does **not** begin by asking whether the labels might be wrong. It begins by recovering previous evidence, reconstructing how unreliable labels reached the training export, building a trustworthy evidence-qualified DATA vNext, and retraining the existing model architecture.

## Primary deliverable

A versioned DATA/ML bundle containing:

1. source-native label provenance;
2. contract-class evidence states;
3. corrected labels where evidence supports correction;
4. unknown/conflicting/not-applicable masks rather than false negatives;
5. leakage-safe training, selection, calibration, and acceptance partitions;
6. the existing model architecture retrained on DATA vNext;
7. evidence-qualified evaluation and policy thresholds;
8. migration and rollback artifacts.

## Binding execution constraints

- Reuse prior DIVE, BCCC, manual, tool-assisted, and source-level work before creating new reviews.
- A new contract review requires a registered evidence gap.
- Do not repeat a previous review merely to obtain a cleaner-looking artifact.
- Preserve historical labels and exports; create versioned replacements.
- Do not redesign the architecture during the normal R4 path.
- Historical zero does not automatically mean confirmed negative.
- Tools are evidence, not ground truth.
- Do not use the same leakage group for training, model selection, threshold fitting, calibration fitting, and final acceptance.
- Every conclusion must link to retained evidence or be marked unsupported.

## Authority order

When files conflict, follow this order:

1. Repository-level agent instructions such as `AGENTS.md` or `CLAUDE.md`.
2. `00_MASTER_PLAN.md`.
3. Approved decision records in `decisions/`.
4. `KNOWN_PREMISE_AND_NON_DUPLICATION_POLICY.md`.
5. `MODEL_ARCHITECTURE_FREEZE.md`.
6. The active phase plan.
7. Operational registers and templates.
8. Historical plans and reports.

The phase plan may add implementation detail but may not weaken the master plan.

## Start

The AI implementation agent must read:

1. `START_HERE_AGENT.md`
2. `00_MASTER_PLAN.md`
3. `KNOWN_PREMISE_AND_NON_DUPLICATION_POLICY.md`
4. `MODEL_ARCHITECTURE_FREEZE.md`
5. `phases/01_PHASE_0_BASELINE_AND_EVIDENCE_LOCATION.md`

It must execute Phase 0 first and stop at Gate G0.

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
├── phases/
├── workstreams/
├── templates/
├── manifests/
├── findings/
├── scripts/
├── decisions/
└── runs/
```
