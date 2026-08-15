# START HERE — AI Implementation Agent Instruction

> **Historical bootstrap notice (2026-08-15):** the Phase-0 assignment below is retained for provenance and is **not the current restart instruction**. R4 has passed historical G0–G7, repaired-v2 physical DATA is accepted for bounded research under R4-D-008 / ADR-R4-008, Phase 8 remains `IN_PROGRESS`, and the 100-epoch run is not authorized. For current work, read `PLAN_STATUS_MATRIX.md`, `DECISION_REGISTER.md`, `RISK_AND_BLOCKER_REGISTER.md`, and `runs/2026-08-15_PHASE8_repaired_data_acceptance_and_launch_decision.md` before acting. Current next work is the evidence-honest objective/evaluation contract and versioned token-selector diagnostics—not Phase 0 replay.

You are implementing the SENTINEL ML R4 trustworthy-label recovery and retraining program.

## Governing objective

The primary defect is already known: the existing vulnerability labels are materially untrustworthy, semantically misaligned, or misleading.

Your job is **not** to perform an open-ended investigation into whether the labels are bad.

Your job is to:

1. preserve the exact current DATA/ML baseline;
2. recover and structure all previous contract-level evidence;
3. reconstruct the mechanisms that produced misleading labels;
4. create an evidence ledger with explicit positive, negative, unknown, not-applicable, conflicting, and not-reviewed states;
5. conduct only targeted gap-filling reviews;
6. build a versioned trustworthy DATA vNext;
7. retrain the existing model architecture;
8. evaluate, calibrate, threshold, and promote it using leakage-safe evidence-qualified populations.

## Historical Phase-0 read order

The following sequence documents the original bootstrap path. Do not execute it as current work unless explicitly reconstructing Phase-0 history:

1. every applicable repository-level instruction file;
2. `docs/plan/ml-R4/README.md`;
3. `docs/plan/ml-R4/00_MASTER_PLAN.md`;
4. `docs/plan/ml-R4/KNOWN_PREMISE_AND_NON_DUPLICATION_POLICY.md`;
5. `docs/plan/ml-R4/MODEL_ARCHITECTURE_FREEZE.md`;
6. `docs/plan/ml-R4/LABEL_STATE_AND_DATASET_ROLE_POLICY.md`;
7. `docs/plan/ml-R4/phases/01_PHASE_0_BASELINE_AND_EVIDENCE_LOCATION.md`;
8. operational registers and templates referenced by Phase 0.

## Historical first execution assignment

Originally: execute **Phase 0 only**, then stop and report Gate G0.

That assignment is complete and historical. Current agents must not repeat it merely because this file exists.

## Prohibited during historical Phase 0

Do not:

- manually re-review contracts;
- rerun a previous large audit;
- change labels or crosswalks;
- regenerate the active export;
- change splits;
- alter thresholds or calibration;
- retrain the model;
- redesign the architecture;
- edit active MLOps configuration;
- delete, clean, reset, or overwrite unrelated local work;
- claim that a missing historical artifact has been recovered by creating a new one.

## Required historical Phase 0 behavior

- Resolve the actual local branch, commit, worktree status, and applicable instructions.
- Work on an isolated branch/worktree if changes will be committed.
- Identify the exact active DATA export, split, checkpoint, threshold sidecar, calibration behavior, and MLOps binding.
- Locate prior DIVE, BCCC, manual, tool-assisted, source-audit, benchmark, and model-run evidence.
- Create an availability inventory. Use `UNAVAILABLE` when an artifact cannot be found.
- Record population counts without combining distinct populations into one corpus number.
- Hash files with SHA-256. For directories, create deterministic manifests.
- Record which active artifacts are protected from modification.
- Create only the outputs required by Phase 0.
- Keep all new R4 outputs under `docs/plan/ml-R4/` unless the phase explicitly approves another versioned audit-output path.

## Required historical Phase 0 outputs

- `manifests/baseline_manifest.json`
- `manifests/protected_artifacts.json`
- `manifests/availability_inventory.csv`
- `manifests/evidence_location_inventory.csv`
- `findings/01_baseline_and_evidence_location.md`
- deterministic helper scripts under `scripts/` as needed

Historical updates included:

- `EXECUTION_LOG.md`
- `ARTIFACT_INDEX.md`
- `PREVIOUS_EVIDENCE_REGISTER.md`
- `RISK_AND_BLOCKER_REGISTER.md`
- `PLAN_STATUS_MATRIX.md`

## Current restart boundary

Current Phase-8 work must instead preserve these facts:

- historical G7/v1 is immutable reproducibility evidence;
- `sentinel-r4-vnext-v2` is physically accepted for bounded research under R4-D-008;
- all 899 current effective loss cells are positive target `1` and zero confirmed negatives exist;
- threshold fit, calibration fit, and untouched acceptance remain unsupported/empty;
- the historical four-window selector is not accepted as adequate and the target-aware candidate is not yet promoted;
- no 100-epoch run is authorized;
- objective, selector, role, or representation changes must be versioned rather than reverse-editing accepted repaired-v2 evidence.

The single current restart document is `runs/2026-08-15_PHASE8_repaired_data_acceptance_and_launch_decision.md` together with ADR-R4-008 and the current registers.
