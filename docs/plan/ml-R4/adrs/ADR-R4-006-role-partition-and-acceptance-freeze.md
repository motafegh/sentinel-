# ADR-R4-006 — Leakage-Safe Role Partition and Empty Acceptance Freeze

**Status:** Accepted  
**Date:** 2026-08-12  
**Deciders:** Ali Rajabi (technical/governance approval delegated), GPT-5.6 Sol  
**Scope:** R4 Phase-6 dataset role partition and acceptance support

## Context

Phase 5 established source/class training authority but intentionally did not assign dataset roles. Phase 6 must ensure leakage groups cannot cross incompatible purposes and must not promote exposed or semantically invalid evaluation data into threshold/calibration/acceptance roles.

The Phase-6 inventory found 13,509 leakage groups across 22,493 contracts:

- 350 strong-eligible groups before representation filtering;
- 492 weak-TOD groups;
- 12,667 unlabeled groups;
- no existing group crossed historical train/val/test splits or sources;
- no policy-approved confirmed-negative rows exist.

Historical auxiliary corpora do not solve the missing-negative/acceptance problem:

- hand-written contracts were explicitly used to validate model and agent behavior, so they are exposed;
- quickstart Tier-A historically maps SmartBugs `access_control` and SolidiFI `tx.origin` to `NonVulnerable`, contradicting current ExternalBug semantics;
- Tier-E defines safe candidates using BCCC `NonVulnerable` plus Slither/Aderyn silence, which is not class-specific confirmed-negative evidence;
- no committed Tier-E quickstart manifest exists;
- Web3Bugs is unavailable and BCCC/DeFiHackLabs remain deferred.

## Decision

### Leakage group identity

Use the strongest available grouping key in this order:

1. `project_group_id`;
2. `dedup_group_id`;
3. `contract_id` fallback.

Every active contract belongs to exactly one group and every group receives exactly one role.

### Representation fail-closed rule

If any contract in a leakage group lacks the required existing representation, the **whole group** is assigned `EXCLUDED` for the first baseline. This prevents represented and unrepresented members of one leakage family from entering incompatible roles.

This excludes 836 contracts / 835 groups, matching the historical no-representation population at group-safe granularity.

### Strong-positive group allocation

After representation filtering, every Phase-5-enabled class with strong source evidence must retain at least one group in each of:

- `TRAIN_STRONG`;
- `MODEL_SELECTION`;
- `INTERNAL_AUDIT`.

Strong groups are deterministically ranked from partition version + ledger hash + policy hash + group ID. The target allocation is approximately 70% train, 15% model selection, 15% internal audit, with class coverage overriding approximate percentages.

The frozen result is:

- `TRAIN_STRONG`: 238 groups / 275 contracts;
- `MODEL_SELECTION`: 51 / 56;
- `INTERNAL_AUDIT`: 51 / 62.

Model selection is explicitly **positive-only limited support**. It may support positive-loss/recall checkpoint diagnostics but not full discrimination metrics such as F1/AUC without trustworthy negatives.

### Weak and unlabeled groups

- represented DIVE TOD weak-positive groups with no strong signal → `TRAIN_WEAK`: 465 groups / 773 contracts;
- remaining represented active groups → `TRAIN_UNLABELED`: 11,869 groups / 20,491 contracts.

Row-level masking remains governed by `data-vnext-policy-v1`; group role never turns unlabeled cells into negatives.

### Threshold fit

`THRESHOLD_FIT = UNSUPPORTED_EMPTY`.

No class-specific confirmed-negative support exists. Threshold fitting must not reuse unknown/tool-silent/all-zero data as negatives.

### Calibration fit

`CALIBRATION_FIT = UNSUPPORTED_EMPTY`.

Reliable calibration is not justified by strong/weak positives plus unlabeled outcomes alone.

### Untouched acceptance

`UNTOUCHED_ACCEPTANCE = UNSUPPORTED_EMPTY_FROZEN`.

The acceptance manifest contains zero contracts and zero groups and is hash-frozen. No historical `test` label, exposed manual suite, semantically corrupted quickstart benchmark, BCCC/tool-silence set, unavailable source, or deferred source may be renamed untouched acceptance.

A later untouched-acceptance claim requires a separately protected, semantically trustworthy, unexposed corpus and a new versioned decision.

## Consequences

### Positive

- role leakage is prevented before implementation/training;
- scarce strong classes retain train/model-selection/internal-audit support;
- missing representations are handled at group level rather than silently filtered after split;
- threshold/calibration/acceptance limitations are explicit and cannot be hidden by metric code;
- later acquisition of a true acceptance corpus can be versioned without changing the current frozen partition.

### Negative

- the first repaired baseline cannot produce a legitimate untouched-acceptance claim;
- threshold fitting and calibration are unavailable under policy v1;
- model selection is positive-only and therefore limited;
- 836 contracts remain excluded because the current representation population is incomplete.

## Rejected alternatives

- **Reuse historical train/val/test roles:** rejected because historical splits were built before repaired label semantics and role separation.
- **Treat unknown/all-zero rows as negative support:** rejected by Phase 2/5 evidence.
- **Use manual Safe contracts as acceptance:** rejected because the suite was exposed to model/agent validation.
- **Use quickstart NonVulnerable contracts as negatives:** rejected because its builder contains invalid canonical mappings.
- **Use BCCC+two-tool silence as safe:** rejected because source label/tool silence is not confirmed absence.
- **Split only represented members of mixed representation groups:** rejected because it can leak one family across included/excluded roles.

## Implementation contract

The controlling artifacts are the Phase-6 role/contract manifests, support table, unsupported-role manifest, untouched-acceptance manifest, and partition manifest. Phase 7 must consume these roles exactly; it may not regenerate or rebalance them implicitly.
