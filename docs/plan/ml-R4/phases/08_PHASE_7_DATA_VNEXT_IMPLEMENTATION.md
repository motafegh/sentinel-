# Phase 7 — DATA vNext Implementation

**Status:** PASSED — G7 PASS  
**Gate:** G7

## Objective

Implement the approved versioned data design while preserving all historical artifacts.

## Expected components

- canonical source registry;
- crosswalk vNext;
- evidence ledger snapshot/binding;
- label/mask export;
- role/partition registry;
- manifests;
- validation suite;
- compatibility loader or explicit version error;
- documentation.

## Implementation rules

- new schema/version only;
- no overwrite of historical exports;
- deterministic build;
- manifest every input;
- assert class order;
- assert no role leakage;
- assert unknown/conflicting outcomes are masked;
- assert provenance and evidence IDs;
- compare population changes to historical export.

## Phase-6 handoff

Phase 7 must consume `r4-vnext-roles-v1` exactly. Threshold/calibration/untouched-acceptance roles are intentionally empty/unsupported; implementation must preserve that limitation. GasException and UnusedReturn remain supervision-disabled. The 836 contracts in incomplete-representation groups remain excluded unless a future versioned plan explicitly rebuilds and re-freezes roles.

## Implementation approach

R4 changes label/role semantics, not the graph/token feature representation. Therefore DATA vNext v2 is implemented as a **versioned semantic overlay** over the immutable representation lineage rather than duplicating all graph/token `.pt` artifacts.

The v2 overlay will contain:

- one canonical contract×class `label_states.parquet`;
- one derived per-contract `ml_targets.parquet` carrying nullable targets, strength, effective loss masks, outcome-metric masks, outcome states, and frozen role;
- machine-readable source registry and crosswalk-policy snapshot;
- exact bindings to the Phase-3 ledger, Phase-5 policy, and Phase-6 partition manifests;
- representation requirements/binding manifest;
- deterministic output manifest and validation report.

The legacy v1 export/merger/label writer remain unchanged and are not valid v2 inputs.

## Local boundary

The semantic overlay can be generated and validated remotely from committed frozen inputs. Physical verification that every non-excluded contract has the expected local graph/token/sidecar representation files requires the protected/local `data_module/data/representations/` tree. G7 cannot pass until that final local representation-binding gate succeeds and its report is registered.

## G7 pass criteria

DATA vNext reproduces from frozen inputs, passes semantic/schema/hash/role validation, physically binds every required representation on the local protected tree, and is suitable for the approved Phase-6 roles without creating unsupported threshold/calibration/acceptance artifacts.

## G7 closeout

The additive v2 implementation is merged and locally representation-bound.

- dataset: `sentinel-r4-vnext-v1`
- contracts: 22,493
- contract×class rows: 224,930
- required/checked representations: 21,657
- required/checked physical files: 64,971
- missing files: 0
- mismatches: 0
- representation digest: `7637461f6643d398c7a0446412fedd8877914c7b9ed41309dab45f18ed96f420`
- manifest state: `VALIDATED_G7_CANDIDATE`
- final representation-required validation: PASS

Historical v1 artifacts were not mutated. Threshold/calibration/untouched-acceptance roles remain intentionally unsupported/empty.

**G7 PASS.** Phase 8 is authorized to retrain the existing frozen architecture using this exact v2 lineage.

