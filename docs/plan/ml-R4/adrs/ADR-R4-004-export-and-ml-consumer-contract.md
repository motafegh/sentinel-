# ADR-R4-004 — DATA vNext Export and ML Consumer Contract

**Status:** Accepted  
**Date:** 2026-08-12  
**Deciders:** Ali Rajabi (technical/governance approval delegated), GPT-5.6 Sol  
**Scope:** R4 DATA vNext export schema and existing-architecture compatibility

## Context

Historical export schema v1 contains one contract row with ten non-nullable integer class columns and one contract-level confidence tier. `SentinelDataset` turns those ten values directly into `y[10]`; collate carries no class mask or label strength; `AsymmetricLoss` treats every zero as a negative.

That interface cannot safely consume DATA vNext semantics.

Phase 8 explicitly allows export-schema compatibility, class masks, weak/strong handling, and metric masking while keeping the model architecture unchanged.

## Decision

### New format version

DATA vNext uses export format **v2**. Historical v1 remains immutable and must require an explicit v1 reader.

No v2 consumer may silently fall back to historical v1 zero semantics when required v2 fields are absent.

### Canonical semantic artifact

The canonical label-state artifact is a long contract×class table, conceptually `label_states.parquet`, with one row per `contract_id × class_index`.

Required semantic fields include:

- `contract_id`
- `class_index`
- `class_name`
- `historical_state`
- source-claim/provenance references
- `outcome_state`
- nullable `target_value`
- `training_signal`
- `training_strength`
- `loss_eligible`
- `outcome_metric_eligible`
- `role_eligibility`
- `policy_decision_id`
- `evidence_ids`
- `limitations`

### Derived ML projection

For efficient model loading, Phase 7 may create a per-contract ten-class projection derived mechanically from the canonical long table. For every class index it must preserve at least:

- nullable target value;
- training strength;
- loss-eligibility mask;
- outcome-metric-eligibility mask;
- canonical outcome state;
- policy-decision identity.

The derived projection is not an independent source of truth. Its manifest must bind the canonical semantic artifact hash.

### Consumer behavior

Phase 8 compatibility code must make absence of v2 semantics a hard failure when training on a v2 export.

Conceptually the batch interface expands from:

```text
(graph, tokens, y, contract_id, confidence_tier)
```

to a structure containing:

```text
graph
tokens
targets[10]
training_strength[10]
loss_mask[10]
outcome_metric_mask[10]
contract_id
lineage/policy identity
```

Exact Python container shape is an implementation decision only if it preserves all fields and tests; it is not a semantic decision.

### Weak-label numeric weight

DATA vNext carries `WEAK` categorically. It does not declare a universal numeric optimizer weight.

Phase 8 must:

- choose any numeric weak-loss weight explicitly in training config;
- ensure `WEAK` is never silently treated as `STRONG`;
- bind the chosen value to the checkpoint/config lineage;
- use only permitted training/model-selection roles to choose it.

### Metrics

A training signal does not automatically authorize outcome metrics. Weak signals always have outcome metrics masked. Strong confirmed outcomes are only metric-usable when Phase 6 assigns a compatible metric role.

### Class-disabled behavior

GasException and UnusedReturn output positions remain present. For policy v1 they must have no loss-eligible targets until re-enabled by a later ADR. Consumers must not fill their missing targets with zero.

## Consequences

- existing graph/token representation and 10-output architecture can remain unchanged;
- dataset/collate/loss code must change in Phase 8 to consume masks/strength;
- v2 is intentionally incompatible with naive v1 training code;
- historical reproducibility remains possible through explicit v1 readers.

## Rejected alternatives

- **Add only a single contract-level confidence tier:** insufficient for class-specific truth.
- **Use NaN targets without explicit state/strength:** too easy for downstream code to mishandle and loses provenance.
- **Overwrite historical `labels.parquet`:** rejected because it destroys lineage and rollback.

## Implementation contract

Phase 7 writes versioned v2 artifacts. Phase 8 performs the minimal consumer compatibility changes allowed by the R4 model-architecture freeze. Phase 5 itself changes neither path.
