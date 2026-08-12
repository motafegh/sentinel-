# ADR-R4-007 — DATA vNext v2 Implementation and G7 Publication Acceptance

**Status:** Accepted  
**Date:** 2026-08-12  
**Deciders:** Ali Rajabi (routine technical/governance approval delegated), GPT-5.6 Sol  
**Scope:** R4 Phase-7 implementation acceptance and Phase-8 training input authority

## Context

R4 Phases 0–6 reconstructed the historical label defect, froze `data-vnext-policy-v1`, and assigned leakage-safe roles in `r4-vnext-roles-v1`. Phase 7 then implemented the approved semantics as an additive v2 overlay rather than rewriting historical v1 artifacts or duplicating the existing graph/token tensors.

Remote CI proved deterministic semantic generation. The required local gate then physically verified all **21,657** non-excluded representation triplets (**64,971 files**) from the real protected representation tree with zero missing files and zero mismatches.

## Decision

Accept `sentinel-r4-vnext-v1` as the **G7-passed DATA vNext implementation** and the only authorized DATA input lineage for the first Phase-8 repaired-model retrain.

The accepted bundle includes:

- `label_states.parquet` — canonical 224,930-row contract×class semantic state;
- `ml_targets.parquet` — derived per-contract ten-class target/strength/mask/role projection;
- source/crosswalk/evidence/representation registries;
- deterministic semantic validation report;
- local physical representation-binding report;
- final representation-required G7 validation report;
- v2 format schema and fail-closed loader/validator/publication code.

The physical representation binding digest is:

`7637461f6643d398c7a0446412fedd8877914c7b9ed41309dab45f18ed96f420`

The accepted semantics remain intentionally asymmetric:

- positive targets: **1,007**;
- confirmed-negative targets: **0**;
- STRONG signals: **403**;
- WEAK signals: **604**;
- GasException and UnusedReturn supervision remain disabled;
- threshold fit and calibration fit remain unsupported/empty;
- untouched acceptance remains unsupported/empty/frozen.

## Phase-8 authority

Phase 8 may add only the compatibility required to train the frozen four-eye architecture from this exact v2 lineage. It may not:

- silently rebuild/rebalance Phase-6 roles;
- reinterpret unknown/masked cells as negatives;
- silently fall back to historical v1 labels;
- change class order or graph schema;
- manufacture threshold/calibration/acceptance populations;
- treat a different DATA export as equivalent without a new versioned decision.

Any numeric weak-loss weight is a Phase-8 training-config decision and must be checkpoint-bound.

## Historical compatibility and rollback

Historical v1 artifacts remain immutable and reproducible as historical evidence. Rollback is selection of a prior hash-bound compatible bundle, never reverse mutation of v2 or v1 files.

## Evidence

- implementation merge: `81d9c547d3610e2cfb12a5927a7a78b5693430c2`;
- local G7 evidence commit: `5bd9c19eb46cd804b34ac0c2cd598767f10c7fad`;
- local representation digest: `7637461f6643d398c7a0446412fedd8877914c7b9ed41309dab45f18ed96f420`;
- branch + PR G7 workflows: PASS;
- historical G3–G6 regression gates: PASS on the integration tree.
