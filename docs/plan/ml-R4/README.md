# SENTINEL ML R4 — Trustworthy-Label Recovery and Retraining Program

This directory governs the R4 repair of SENTINEL's DATA/ML training and evaluation foundation.

## Current restart — 2026-09-02

Do **not** restart R4 from Phase 0 or from the August 15/21/23 intermediate Phase-8 handoffs. Historical G0–G7 are complete and retained as reproducibility evidence. Phase 8 remains `IN_PROGRESS`; G8 is not passed and the 100-epoch run is **NOT AUTHORIZED**.

Current boundary:

- repaired-v2 physical DATA remains accepted immutable reproducibility evidence under R4-D-008;
- logical V3 grouping/roles/publication remains accepted under R4-D-009 and the hardened V3 evidence snapshot remains the accepted pre-pilot logical baseline;
- R4-D-010 prohibits the new full run from using graph schema v9 and requires a separately versioned V10 physical representation lineage;
- the former 26-contract V10 parse-only remediation is complete in the protected V2.4 diagnostic lineage;
- the later V2.5 bounded structural-drift investigation is complete **20/20**, with 8 exact node-index-invariant graph-equivalence identities and 12 deterministic persistent-storage `CFG_NODE_WRITE` corrections, zero unexplained drift, and no blockers;
- the V2.6 storage-collection mutator correction is implemented and bounded to persistent-storage `push`/`pop` receivers only;
- exact final runtime partition is 22,539 ordinary identities under Slither 0.10.0 plus one declared identity-bound exception under Slither 0.11.5;
- protected-local Stages A-D pass for all 22,540 identities with exact accepted-V9 token bytes and the required 22,539 + 1 runtime split;
- the V2.6 full candidate is bound by digest `d9f925588913e66476cfbc097bace7daa7e673295fe2a243760313d0bef5ebdd`;
- Stage E V4 passes: all 355 current raw non-parse-only drifts are re-proven as 349 persistent-storage WRITE corrections plus 6 exact index-equivalent graphs, with zero unexplained drift;
- R4-D-011 physically accepts only the exact V2.6 protected-local root and digest above; selector promotion and training remain separate and unauthorized;
- confirmed-negative support remains zero; candidate #1 is `NOT_CONFIRMED`; candidate #2 primary review still requires genuinely independent agreement;
- threshold fitting, calibration fitting, and untouched acceptance remain unsupported/empty;
- the target-aware token selector remains promising but unpromoted;
- full training remains unauthorized.

For current work, read in this order:

1. applicable repository/module `CLAUDE.md` files;
2. `PLAN_STATUS_MATRIX.md`;
3. `runs/2026-09-02_PHASE8_v10_v26_physical_acceptance_and_no_launch.md`;
4. `adrs/ADR-R4-011-v10-v26-physical-representation-acceptance.md`;
5. `runs/2026-08-30_PHASE8_v10_v25_full_population_structural_evidence_plan.md`;
6. `runs/2026-08-27_PHASE8_v10_v25_current_restart_checkpoint.md` for historical staging context;
7. `reviews/R4-GAP-008/2026-08-26_v10_v25_bounded_structural_closure.md`;
8. `DECISION_REGISTER.md` and `adrs/ADR-R4-010-versioned-external-call-representation-correction.md`;
9. `RISK_AND_BLOCKER_REGISTER.md`, `EVIDENCE_GAP_REGISTER.md`, and `CLAIM_STATUS_MATRIX.md`.

R4-B008 / R4-GAP-008 are closed only for the R4-D-011 root and digest. Preserve that accepted root and its evidence; any regenerated or changed root must fail closed and receive a new physical decision.

The negative-evidence, selector-promotion, objective/evaluation, calibration, and training-authorization tracks remain separate later gates. Do not combine them with V10 physical-candidate construction.

The current executable prerequisite is the read-only full-population historical-control selector verification in `runs/2026-09-02_PHASE8_selector_control_equivalence_plan.md`. It must not rewrite accepted tokens, promote the guarded selector, or launch training.

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
- Preserve historical labels/exports, accepted repaired-v2 evidence, accepted logical V3, the frozen V2.3 structural reference, and protected V2.4 diagnostic evidence; create versioned replacements.
- Do not redesign the architecture during the normal R4 path without a separate decision.
- Historical zero does not automatically mean confirmed negative.
- Tools are evidence, not ground truth.
- Do not use the same leakage group for training, model selection, threshold fitting, calibration fitting, and final acceptance.
- Every conclusion must link to retained evidence or be marked unsupported.
- Repository tests do not substitute for protected local physical DATA binding.
- A physically valid dataset does not by itself prove model discrimination quality.
- Historical bounded V2.5 structural success did not equal physical V10 acceptance; only R4-D-011 grants authority to the exact V2.6 root/digest.
- Do not launch training from V10 until the separate supervision/evaluation, selector, run-control, and training-governance gates pass.

## Authority order

When files conflict, follow this order:

1. Repository/module agent instructions such as `CLAUDE.md`.
2. Executable source/config/tests for actual behavior.
3. `00_MASTER_PLAN.md` and accepted machine-readable policy/artifacts.
4. Approved decision records / ADRs, including R4-D-008, R4-D-009, and R4-D-010.
5. `PLAN_STATUS_MATRIX.md`, `RISK_AND_BLOCKER_REGISTER.md`, and the current restart checkpoint/staging records.
6. Canonical handbook/current-status documentation.
7. Historical plans, bootstrap instructions, and dated intermediate reports.

An active phase or decision may refine execution detail but may not weaken the master evidence rules.

## Historical bootstrap and intermediate handoffs

`START_HERE_AGENT.md` retains the original Phase-0 bootstrap assignment for provenance but now redirects to the current Phase-8 boundary.

These dated V10 records are historical execution context, not current restart authority:

- `runs/2026-08-21_PHASE8_gap008_external_call_semantics_audit.md`;
- `runs/2026-08-21_PHASE8_v10_external_call_implementation_handoff.md`;
- `runs/2026-08-21_PHASE8_v10_implementation_and_local_regression.md`;
- `runs/2026-08-23_PHASE8_v10_parse_only_resolution_working_plan.md`;
- `runs/2026-08-23_PHASE8_v10_structural_drift_probe_handoff.md`.

Do not replay their completed blockers merely because the files remain in the repository.

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
