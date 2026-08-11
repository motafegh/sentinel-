# ADR-R4-001 — Separate Outcome Truth from Training Signal

**Status:** Accepted  
**Date:** 2026-08-12  
**Deciders:** Ali Rajabi (technical/governance approval delegated), GPT-5.6 Sol  
**Scope:** R4 DATA vNext label-state schema

## Context

Historical SENTINEL flattened many semantically different states into a binary ten-cell vector. `0` represented explicit source zero, unknown, unsupported class, dropped category, source absence, mapped NonVulnerable, and post-export suppression. The ML dataset then supplied every zero to loss as a negative.

Phase 2 reconstructed that corruption mechanism. Phase 3 created a contract×class evidence ledger. Phase 4 showed that some source assertions may still carry useful weak signal without being trustworthy outcome labels.

A single `0/1` target therefore cannot represent both truth and training utility.

## Decision

DATA vNext uses `contract_id × class_index` as the semantic unit and separates four concepts:

1. **source-native claim** — what a source actually asserted or failed to assert;
2. **canonical outcome state** — what the available evidence justifies as truth;
3. **training signal** — whether an approved positive/negative signal may be used by training;
4. **training strength** — `STRONG`, `WEAK`, or `NONE`.

Canonical outcome states remain:

- `CONFIRMED_POSITIVE`
- `CONFIRMED_NEGATIVE`
- `UNKNOWN`
- `NOT_APPLICABLE`
- `CONFLICTING_EVIDENCE`
- `NOT_REVIEWED`
- `INVALID_RECORD`

Source-claim states explicitly include positive, explicit-zero, unknown, unsupported, dropped-category, out-of-taxonomy, unavailable, and no-assertion states.

`target_value` is nullable:

- `1` for an approved strong positive or explicitly authorized weak-positive training signal;
- `0` **only** when the canonical outcome is `CONFIRMED_NEGATIVE`;
- `null` when no authorized training target exists.

A weak positive may have:

```text
outcome_state      = UNKNOWN
training_signal    = POSITIVE
training_strength  = WEAK
target_value       = 1
outcome_metric_eligible = false
```

This does not convert the weak assertion into a confirmed outcome.

Unknown/not-reviewed/conflicting outcomes remain excluded from **outcome metrics**. Weak training is a distinct training pathway and does not relax the outcome-truth rule.

## Numeric weights

DATA vNext does **not** encode an arbitrary numeric reliability weight as semantic truth. Phase 8 must map `STRONG`/`WEAK` to numeric optimizer weights in explicit training configuration and bind that configuration to the checkpoint.

This prevents a hidden implementation default from becoming a data-policy decision.

## Consequences

### Positive

- unknown can no longer silently become negative;
- weak evidence can be used without pretending it is ground truth;
- future confirmed negatives have a precise representation;
- metrics and training authority are independently controlled;
- the 10-output model shape can remain frozen.

### Negative

- consumers must carry masks/strength in addition to `y`;
- the historical ASL path cannot consume vNext correctly without Phase-8 compatibility changes;
- some classes may have only positive/unlabeled evidence and require appropriate training handling.

## Rejected alternatives

- **Keep binary labels and add confidence tier:** rejected because confidence does not distinguish unknown from negative.
- **Use `-1` as unknown inside the target tensor:** rejected because it overloads numeric target semantics and is easy to mishandle in loss/metrics.
- **Convert weak labels to soft targets such as 0.6:** rejected because a sample precision estimate is not a calibrated per-contract probability.

## Implementation contract

The controlling machine-readable representation is `docs/plan/ml-R4/specs/data_vnext_policy_v1.json`. Phase 7 must represent these states explicitly; Phase 8 must fail closed if vNext masks/strength are absent.
