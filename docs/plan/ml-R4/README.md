# SENTINEL ML R4 — Trustworthy-Label Recovery and Retraining Program

This directory governs the R4 repair of SENTINEL's DATA/ML training and evaluation foundation.

## Current restart — 2026-08-27

Do **not** restart R4 from Phase 0 or from the August 15/21/23 intermediate Phase-8 handoffs. Historical G0–G7 are complete and retained as reproducibility evidence. Phase 8 remains `IN_PROGRESS`; G8 is not passed and the 100-epoch run is **NOT AUTHORIZED**.

Current boundary:

- repaired-v2 physical DATA remains accepted immutable reproducibility evidence under R4-D-008;
- logical V3 grouping/roles/publication remains accepted under R4-D-009 and the hardened V3 evidence snapshot remains the accepted pre-pilot logical baseline;
- R4-D-010 prohibits the new full run from using graph schema v9 and requires a separately versioned V10 physical representation lineage;
- the former 26-contract V10 parse-only remediation is complete in the protected V2.4 diagnostic lineage;
- the later V2.5 bounded structural-drift investigation is complete **20/20**, with 8 exact node-index-invariant graph-equivalence identities and 12 deterministic persistent-storage `CFG_NODE_WRITE` corrections, zero unexplained drift, and no blockers;
- the V2.5 evidence-chain/full-gate and heterogeneous-runtime staging preflights pass;
- exact final runtime partition is 22,539 ordinary identities under Slither 0.10.0 plus one declared identity-bound exception under Slither 0.11.5;
- the Stage-A driver/staging tests pass 9/9 and population partition is 22,540 = 22,539 + 1;
- **Stage A primary generation has not yet been executed**;
- physical V10 acceptance remains false;
- confirmed-negative support remains zero; candidate #1 is `NOT_CONFIRMED`; candidate #2 primary review still requires genuinely independent agreement;
- threshold fitting, calibration fitting, and untouched acceptance remain unsupported/empty;
- the target-aware token selector remains promising but unpromoted;
- full training remains unauthorized.

For current work, read in this order:

1. applicable repository/module `CLAUDE.md` files;
2. `PLAN_STATUS_MATRIX.md`;
3. `runs/2026-08-27_PHASE8_v10_v25_current_restart_checkpoint.md`;
4. `reviews/R4-GAP-008/2026-08-26_v10_v25_bounded_structural_closure.md`;
5. `runs/2026-08-26_PHASE8_v10_v25_full_candidate_staging.md`;
6. `DECISION_REGISTER.md` and `adrs/ADR-R4-010-versioned-external-call-representation-correction.md`;
7. `RISK_AND_BLOCKER_REGISTER.md`, `EVIDENCE_GAP_REGISTER.md`, and `CLAIM_STATUS_MATRIX.md`;
8. exact source/tests for the V10 physical-candidate stage being executed.

The next permitted R4-B008 action is **Stage A**: run `scripts/p8_generate_v10_v25_primary_attempt.py` under exact Slither 0.10.0 in a fresh non-canonical attempt root. It must produce exactly 22,539 ordinary V2.5 identities and defer the one declared runtime exception without invoking extraction for it. Do not claim those artifacts exist before the Stage-A report passes.

The negative-evidence, selector-promotion, objective/evaluation, calibration, and training-authorization tracks remain separate later gates. Do not combine them with V10 physical-candidate construction.

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
- Bounded V2.5 structural success does not equal physical V10 acceptance.
- Do not launch training from V10 until the staged full candidate, binding, V3 transition audit, explicit review, and separate training governance all pass.

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
