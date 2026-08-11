# ADR-R4-003 — Crosswalk and Evidence Aggregation Semantics

**Status:** Accepted  
**Date:** 2026-08-12  
**Deciders:** Ali Rajabi (technical/governance approval delegated), GPT-5.6 Sol  
**Scope:** R4 DATA vNext source-native transformation and multi-source aggregation

## Context

The historical pipeline binarized each source before aggregation. By the time the merger ran, `0` no longer revealed whether a class was explicitly zero, unknown, unsupported, dropped, absent, mapped to NonVulnerable, or later suppressed. The merger then used positive precedence over those collapsed zeros.

Phase 2 established that this representation loss, rather than multi-source overlap, was the primary corruption mechanism.

## Decision

### Preserve source claims before crosswalk

Every source claim retained by DATA vNext records:

- source identity;
- source record/category;
- source-claim state;
- crosswalk action;
- canonical class if one exists;
- evidence/independence identifiers;
- limitations.

Crosswalk actions are semantic transforms, not implicit labels. At minimum the implementation must distinguish:

- `DIRECT`
- `SEMANTIC_COMPRESSION`
- `LOSSY_NO_CANONICAL_TARGET`
- `OUT_OF_TAXONOMY_NO_CANONICAL_TARGET`
- `UNSUPPORTED`
- `DROPPED_CATEGORY`
- `NO_ASSERTION`

No-target actions preserve provenance but yield no training target.

### Remove historical synthetic mappings

For DATA vNext:

- SmartBugs `bad_randomness` does not automatically map to Timestamp;
- SmartBugs `short_addresses` and `other` do not map to global NonVulnerable;
- DIVE Bad Randomness remains no-target;
- source/class absence never becomes negative.

### Aggregate evidence states, not binary cells

Canonical outcome aggregation is:

1. confirmed positive with no confirmed negative → `CONFIRMED_POSITIVE`;
2. confirmed negative with no confirmed positive → `CONFIRMED_NEGATIVE`;
3. confirmed positive + confirmed negative → `CONFLICTING_EVIDENCE`;
4. no confirmed outcome → `UNKNOWN` or `NOT_REVIEWED` according to evidence presence/review state.

Weak positive training evidence is orthogonal:

- an explicitly authorized weak-positive source may create `training_signal=POSITIVE`, `training_strength=WEAK`, `target_value=1`;
- this does not change the outcome to confirmed positive;
- weak evidence cannot override a confirmed contrary outcome.

No weak-negative source is authorized in policy v1.

### No implicit voting

- tool/source counts are not an independence model;
- correlated Slither/Aderyn/source agreement is not multiplied into authority;
- implementation may not resolve conflicts by majority vote unless a later ADR defines and validates independence/weighting.

### No global NonVulnerable synthesis

DATA vNext does not produce a global NonVulnerable class or all-zero safety claim from absence of canonical positives. Class-specific negatives require class-specific evidence.

## Consequences

### Positive

- every historical zero mechanism can remain distinguishable;
- new sources can be integrated without silently creating negatives;
- aggregation becomes auditable and deterministic;
- weak training can coexist with unresolved truth.

### Negative

- the merger becomes more stateful than the current binary helper;
- export consumers must understand nullable targets and explicit state.

## Rejected alternatives

- **Retain positive precedence over zeros:** rejected because zeros do not share one meaning.
- **Tier-based winner-takes-all:** rejected because source tier does not establish class-specific negative authority.
- **Three-tool consensus:** rejected as a generic aggregator because tool dependence and detector coverage are class-specific.

## Implementation contract

Phase 7 must implement aggregation from the machine-readable policy and evidence ledger. It may not infer semantic precedence from current `_SOURCE_PRECEDENCE`, historical `confidence_tier`, or input order.
