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
