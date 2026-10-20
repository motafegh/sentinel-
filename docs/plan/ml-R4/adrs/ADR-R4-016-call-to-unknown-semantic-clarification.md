# ADR-R4-016 — Clarify `CallToUnknown` as the low-level unknown-target class

**Date:** 2026-10-20  
**Status:** ACCEPTED  
**Decision ID:** R4-D-016  
**Scope:** R4-GAP-007 class semantics, confirmed-negative review authority, and candidate-review compatibility

## Context

R4-GAP-007 candidate #2 independent-review preparation exposed a long-standing wording inconsistency in class index 0, `CallToUnknown`.

The locked taxonomy prose said:

> Low-level calls (`call` / `delegatecall` / `send`) whose return value is not checked, **or calls to unverified external addresses**.

Read literally, the second clause could include ordinary typed high-level calls or `transfer()` to a caller-controlled address.

However, the repository's operational semantics have consistently been narrower:

- the original SolidiFI crosswalk maps **Unchecked-Send** to `CallToUnknown` because it is a low-level `send()` pattern;
- the original DIVE crosswalk explicitly says `CallToUnknown` is reserved for raw low-level `call` / `send` / `delegatecall` patterns where the callee is unverified;
- SmartBugs Curated maps `unchecked_low_level_calls` to this class;
- the class-specific verification pattern defines `CallToUnknown` as a low-level call to an unknown/dynamic address;
- R4-D-010 separates `HIGH_LEVEL_CALL`, `LOW_LEVEL_CALL`, `ETHER_TRANSFER`, `ETHER_SEND`, `LIBRARY_CALL`, and `CONTRACT_CREATION`, and explicitly states that `CallToUnknown` graph corroboration may use low-level/send evidence while transfer/library edges are not vulnerability truth;
- the accepted V10 semantic checker uses only `LOW_LEVEL_CALL` and `ETHER_SEND` as coarse `CallToUnknown` signals.

The inconsistency is original rather than newly introduced: the broad taxonomy prose and narrow crosswalk interpretation were both committed on 2026-06-11. This is therefore a semantic-alignment defect between canonical prose and the system's source mappings/verification behavior.

## Decision

### 1. Authoritative semantic scope

`CallToUnknown` is the **raw/dynamic low-level unknown-target interaction class**.

A source construct is in scope when it uses the low-level external-call family against an unverified or dynamically selected callee, including:

- `call`;
- `delegatecall`;
- legacy `callcode`;
- `staticcall` where applicable;
- `send`.

A checked raw low-level call can still satisfy `CallToUnknown`: target/callee uncertainty is class-defining. Unchecked return/failure handling is additional positive evidence and may overlap with exception/unused-return concerns; it is not the sole defining condition.

### 2. Explicit exclusions

The following are **not `CallToUnknown` solely because the destination address is external, dynamic, caller-controlled, or otherwise unverified**:

- Solidity `transfer()`;
- ordinary typed high-level interface calls;
- same-file or otherwise in-scope library calls;
- contract creation (`new` / creation IR);
- `selfdestruct` / `suicide` destinations.

Those constructs may be relevant to other vulnerability classes, but they do not become `CallToUnknown` merely from destination uncertainty.

### 3. Taxonomy/version consequence

This is an authoritative semantic clarification of the existing class, not a new class identity or class-order change.

Therefore:

- `CallToUnknown` remains class index `0`;
- the 10-class set/order remains unchanged;
- existing model output dimensions remain unchanged;
- existing SolidiFI/SmartBugs positive mappings remain unchanged;
- DIVE still has no direct `CallToUnknown` positive category;
- taxonomy `schema_version: "1"` remains valid because no field schema, class set, class index, or positional model contract changes.

The taxonomy prose and class-specific verification pattern are updated to match this accepted scope. Any future expansion that intentionally makes typed high-level/transfer destinations positive by destination uncertainty alone would be a new material class-definition decision and must be versioned/governed separately.

### 4. Candidate #1 consequence

Candidate #1's historical primary `NOT_CONFIRMED` decision depended specifically on treating a caller-selected **typed high-level callback** as contradictory `CallToUnknown` evidence under the broad taxonomy phrase.

Under R4-D-016 that rationale is definition-stale.

The historical record remains preserved, but it no longer controls current `CallToUnknown` adjudication. Candidate #1 remains exactly what the immutable queue already says:

```text
state  = UNKNOWN / PENDING_REVIEW
target = None
```

If candidate #1 is used again, it requires a new primary review under R4-D-016. No target rollback is required because it was never accepted as a confirmed negative.

### 5. Candidate #2 consequence

Candidate #2's existing source-first primary review is semantically compatible with R4-D-016:

- it found no raw/low-level/send call family;
- its only relevant value interaction is `transfer()`;
- same-file SafeMath library calls were explicitly excluded;
- it concluded that `transfer()` does not satisfy the direct low-level unknown-target class intent.

The primary review may therefore be retained as primary evidence. Candidate #2 nevertheless remains `UNKNOWN / PENDING_REVIEW / target None` until an admissible independent agreeing verification exists.

### 6. Blind-review consequence

The v1 blind review bundle is retired for future independent adjudication because it omitted a self-contained governing class definition/negative-evidence policy and allowed reviewers to encounter contradictory repository prose.

A v2 bundle must contain, inside the bundle itself:

- complete candidate source;
- manifest and hashes;
- review task;
- the exact neutral R4-D-016 `CallToUnknown` definition and exclusions;
- the generic class-specific confirmed-negative evidence rule;
- no primary-review result, candidate-history conclusion, or other verdict leakage.

The reviewer must operate in a context with no candidate-specific prior conclusion available.

## Evidence

- `data_module/sentinel_data/labeling/schema/taxonomy.yaml` original/current history;
- original SolidiFI crosswalk (`Unchecked-Send -> CallToUnknown`);
- original DIVE crosswalk (`CallToUnknown` reserved for raw low-level patterns);
- `data_module/sentinel_data/labeling/crosswalks/smartbugs_curated.yaml`;
- `data_module/sentinel_data/verification/patterns/CallToUnknown.yaml`;
- `data_module/sentinel_data/representation/graph_schema_versions.py`;
- `data_module/sentinel_data/verification/semantic_checker.py`;
- ADR-R4-010 / R4-D-010;
- candidate #1 and candidate #2 primary-review records;
- `working/2026-10-20_ml_gap007_call_to_unknown_semantic_alignment.md`.

## Non-authorities

R4-D-016 does **not**:

- confirm candidate #1 or #2 as negative;
- create any target `0`;
- authorize optimizer use of confirmed negatives;
- establish false-positive rates, thresholds, calibration, or untouched acceptance;
- authorize M5/full training;
- pass G8;
- change accepted R4-D-011/R4-D-013 representation artifacts.

## Rollback

The previous broad taxonomy prose remains available in Git history. Rollback of this clarification would reopen R4-R027/R4-B009 and invalidate any independent review that relied on R4-D-016. It would not alter immutable queue targets because they remain `None` until explicit adjudication.
