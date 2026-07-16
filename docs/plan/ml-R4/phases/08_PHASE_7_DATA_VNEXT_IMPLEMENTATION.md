# Phase 7 — DATA vNext Implementation

**Status:** WAITING FOR G6  
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
