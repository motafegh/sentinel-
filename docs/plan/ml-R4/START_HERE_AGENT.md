# START HERE — AI Implementation Agent Instruction

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

## Read before acting

Read in this order:

1. every applicable repository-level instruction file;
2. `docs/plan/ml-R4/README.md`;
3. `docs/plan/ml-R4/00_MASTER_PLAN.md`;
4. `docs/plan/ml-R4/KNOWN_PREMISE_AND_NON_DUPLICATION_POLICY.md`;
5. `docs/plan/ml-R4/MODEL_ARCHITECTURE_FREEZE.md`;
6. `docs/plan/ml-R4/LABEL_STATE_AND_DATASET_ROLE_POLICY.md`;
7. `docs/plan/ml-R4/phases/01_PHASE_0_BASELINE_AND_EVIDENCE_LOCATION.md`;
8. operational registers and templates referenced by Phase 0.

## First execution assignment

Execute **Phase 0 only**.

Do not begin Phase 1 automatically. Stop and report the Gate G0 result.

## Prohibited during Phase 0

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

## Required Phase 0 behavior

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

## Required Phase 0 outputs

- `manifests/baseline_manifest.json`
- `manifests/protected_artifacts.json`
- `manifests/availability_inventory.csv`
- `manifests/evidence_location_inventory.csv`
- `findings/01_baseline_and_evidence_location.md`
- deterministic helper scripts under `scripts/` as needed

Update:

- `EXECUTION_LOG.md`
- `ARTIFACT_INDEX.md`
- `PREVIOUS_EVIDENCE_REGISTER.md`
- `RISK_AND_BLOCKER_REGISTER.md`
- `PLAN_STATUS_MATRIX.md`

## Final response after Phase 0

Report:

1. `G0 PASS`, `G0 FAIL`, or `G0 BLOCKED`;
2. repository branch, commit, and worktree status;
3. exact active DATA/ML bundle;
4. distinct population counts;
5. prior evidence locations and availability;
6. unresolved artifact identities or contradictions;
7. files created or updated;
8. commands and validation performed;
9. confirmation that no protected DATA/ML artifact changed;
10. the single next permitted action.

Stop after the report.
