# ml/testing_specs — Validation/Audit Procedures

This directory contains reusable and historical validation procedures for ML runs, data integrity, diagnostics, schema migration, inference, calibration mechanics, and release review.

> **Authority notice:** these specs were largely authored around Run12/pre-R4 workflows. They are **procedures, not current DATA/evaluation authority**. For any new DATA-vNext retrain or release decision, current R4 policy/roles and canonical handbook state override older split, negative-label, threshold, calibration, and acceptance assumptions in individual spec files.

Always read [`00_rules.md`](00_rules.md) alongside a selected procedure.

## Current R4 constraints before using a spec

Stable `main` is through R4 G6. Before applying any ML validation procedure, verify:

- DATA semantic policy: `docs/plan/ml-R4/specs/data_vnext_policy_v1.json`;
- frozen roles: `docs/plan/ml-R4/manifests/p6_partition_manifest.json`;
- role support: `docs/plan/ml-R4/manifests/p6_role_support_table.json`;
- acceptance state: `docs/plan/ml-R4/manifests/p6_untouched_acceptance_manifest.json`;
- current status: `docs/handbook/16_current_status.md`.

Current first-baseline limitations:

```text
historical/unknown zero ≠ confirmed negative
GasException + UnusedReturn supervision disabled
MODEL_SELECTION = positive-only limited
THRESHOLD_FIT = UNSUPPORTED_EMPTY
CALIBRATION_FIT = UNSUPPORTED_EMPTY
UNTOUCHED_ACCEPTANCE = UNSUPPORTED_EMPTY_FROZEN
Run12 = historical operational baseline
Phase 8 retrain waits for G7
```

If a procedure assumes otherwise, treat that assumption as historical and adapt/update the procedure before using it for a current decision.

## Files

| File | Purpose / current use |
|---|---|
| `00_rules.md` | mandatory authority/evidence rules |
| `QUICKSTART.md` | supplementary navigation; verify assumptions against R4 |
| `MIGRATION.md` | historical migration mechanics |
| `A_benchmark_runs.md` | benchmark/run validation mechanics |
| `B_contract_deep_dive.md` | contract-level diagnostic review |
| `B_data_pipeline.md` | historical DATA-pipeline validation; R4 semantics override old binary assumptions |
| `C_diagnostic_checks.md` | diagnostics |
| `D_smoke_preflight.md` | smoke/preflight mechanics |
| `E_preprocessing_consistency.md` | preprocessing consistency |
| `F_new_run_checklist.md` | run setup checklist; must bind vNext artifacts/roles for future retrain |
| `G_ablation_protocol.md` | ablation mechanics |
| `H_issue_triage.md` | issue triage |
| `I_regression_guard.md` | regression mechanics |
| `J_schema_migration.md` | schema migration procedures |
| `K_inference_api.md` | inference API validation |
| `L_release_readiness.md` | historical release checklist; current promotion authority is R4 Phase 9/10 evidence/gates |

## Framework

`framework/` contains CLI/gates/reporters from the historical validation system. It may still be useful for implementation diagnostics, but a green framework gate cannot create a missing R4 evidence role.

Examples:

- running threshold checks does not create a trustworthy threshold-fit corpus;
- running calibration checks does not authorize calibration on unknown/exposed rows;
- passing old staging checks does not promote a vNext model;
- historical Run12 gate outcomes are historical evidence, not the current project state.

## Usage

1. read `00_rules.md`;
2. read current R4 policy/role/current-status artifacts;
3. select only the procedure relevant to the task;
4. identify and replace any historical assumption before execution;
5. record outputs against exact DATA/checkpoint/config/role identities.

```bash
cat ml/testing_specs/00_rules.md
python -m ml.testing_specs.framework.cli run   # historical framework mechanics only
```

For current ML training/evaluation policy, use [`../README.md`](../README.md), [`../../docs/handbook/06_ml_training_quality.md`](../../docs/handbook/06_ml_training_quality.md), and the active R4 phase plans.
