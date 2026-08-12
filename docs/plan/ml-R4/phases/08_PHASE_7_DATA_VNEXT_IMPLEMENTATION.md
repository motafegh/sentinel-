# Phase 7 — DATA vNext Implementation

**Status:** READY — G6 SATISFIED  
**Gate:** G7

## Objective

Implement the approved versioned data design while preserving all historical artifacts.

## Expected components

- canonical source registry;
- crosswalk vNext;
- evidence ledger snapshot;
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

## G7 pass criteria

DATA vNext reproduces from frozen inputs, passes validation, and is suitable for approved roles.


## Phase-6 handoff

Phase 7 must consume `r4-vnext-roles-v1` exactly. Threshold/calibration/untouched-acceptance roles are intentionally empty/unsupported; implementation must preserve that limitation. GasException and UnusedReturn remain supervision-disabled. The 836 contracts in incomplete-representation groups remain excluded unless a future versioned plan explicitly rebuilds and re-freezes roles.
