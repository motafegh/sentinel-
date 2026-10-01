# ML M3 — R4-D-013 guarded training-lineage integration

**Date:** 2026-10-01  
**Authorities:** R4-D-009 logical V3; R4-D-011 graph parent; R4-D-013 guarded-token physical acceptance; R4-D-014 bounded positive-only objective/evaluation boundary  
**State:** IN PROGRESS

## Design

M3 will add a new fail-closed repaired-training seam beside the historical G7/v9 runner. Historical `VNextTrainingDataset`, `build_run_binding()`, and `run_phase8_training()` remain unchanged for reproduction.

The new seam consists of:

1. `LogicalV3GuardedTrainingDataset`
   - V3 manifest/role semantics remain the logical authority;
   - tracked R4-D-009 logical acceptance must match the overlay manifest;
   - R4-D-013 physical acceptance must match the exact guarded root/digest;
   - active training/selection graph-token-sidecar bytes must match records in the accepted guarded candidate manifest;
   - loaded graph payloads must be V10/V2.6;
   - loaded token payloads must be `r4-v10-v26-guarded-tokens-v1`, request `target_aware_guarded_v1`, and use only guarded or exact historical-control effective selection.

2. guarded population validation
   - derive expected logical V3 frozen/active population counts from the committed R4-D-009 acceptance record, not historical G7 constants.

3. `build_guarded_run_binding()`
   - bind logical V3 manifest + acceptance;
   - bind R4-D-011 graph-parent decision/digest;
   - bind R4-D-013 physical acceptance, candidate manifest and guarded digest;
   - bind R4-D-014 objective/evaluation decision;
   - bind explicit V10 model schema, frozen architecture, GraphCodeBERT revision, roles/populations, optimizer config, seed and runtime;
   - retain `full_training_authorized=false` and bounded-pilot-only scope.

4. explicit V10 model factory
   - construct the unchanged `four_eye_v8` / `v8.1` architecture with `graph_schema_version='v10'`;
   - do not modify the historical v9 factory default.

5. checkpoint/inference provenance
   - existing Phase-8 checkpoints embed the complete run-binding payload and digest;
   - therefore guarded pilot checkpoints will automatically carry D9/D11/D13/D14 lineage once the new binding is used;
   - inference/export compatibility remains a later promotion responsibility and no live Run12 change occurs in M3.

## Stop line

M3 may build and test the integration seam. It may not launch the M4 pilot or the full training run.


## Implementation checkpoint — repository-safe complete

Implemented on canonical `main`:

- `ml/src/datasets/vnext_logical_v3_guarded_dataset.py`
  composes R4-D-009 logical authority with the exact R4-D-013 physical
  guarded root instead of rewriting the V3 publication;
- full D13 acceptance checks bind exact population, selector distribution,
  D11 parent, candidate manifest, source commit and active artifact hashes;
- `ml/src/training/vnext_guarded_run_control.py` derives current V3
  population authority from the accepted logical-V3 record and enforces a
  bounded horizon below 100 epochs;
- `ml/src/training/vnext_guarded_binding.py` binds D9 + D11 + D13 + D14,
  frozen V10 architecture, GraphCodeBERT revision, optimizer/scheduler/runtime
  identity and explicit no-training/no-PU/no-threshold limits;
- `build_phase8_v10_model()` constructs the frozen four-eye architecture with
  graph schema V10 while the historical factory remains unchanged;
- `ml/tests/test_vnext_phase8_guarded.py` proves accepted-lineage loading,
  mutation rejection, V3 population derivation, bounded-horizon enforcement
  and complete guarded run binding;
- `p8_preflight_guarded_training_m3.py` performs the protected-local M3 proof
  without any optimizer step.

Repository-safe verification:

- Phase-8 guarded compatibility compile/tests: PASS;
- repaired/Phase-8 repository compile: PASS;
- repaired/research regression suite: PASS;
- committed logical-V3 snapshot verification: PASS;
- frozen historical G6 verification: PASS;
- repository-repair workflow remains red only at inherited repository-wide
  `git diff --check` whitespace debt.

### Remaining M3 gate

Run the protected-local no-step preflight against the actual accepted V3 overlay
and R4-D-013 physical root. A passing script intentionally reports
`PASS_M3_GUARDED_INTEGRATION_REVIEW_REQUIRED` with
`m4_execution_authorized=false`. M3 closes and M4 becomes ready only after
that report is reviewed.
