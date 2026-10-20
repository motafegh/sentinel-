# ML / R4-GAP-007 — CallToUnknown semantic alignment checkpoint

**Date:** 2026-10-20  
**State:** ACTIVE BLOCKER — resolve neutral class semantics before any further candidate #2 independent adjudication  
**Applies to:** R4-GAP-007 / `CallToUnknown` confirmed-negative evaluation evidence  

## Current accepted boundary

The current accepted ML/R4 state is:

- logical authority remains R4-D-009 / `sentinel-r4-vnext-v3`;
- R4-D-011 is the immutable accepted V10/V2.6 graph/control root;
- R4-D-013 is the immutable accepted guarded-token successor;
- R4-D-014 limits the executable objective/evaluation regime to bounded positive-only control and positive-fit diagnostics;
- M0-M4 are complete;
- R4-D-015 records `PASS_BOUNDED_PILOT_FULL_TRAINING_STILL_HELD` after the 8-epoch / 120-step M4-C pilot;
- M5/full training is not authorized; G8 remains open;
- confirmed-negative targets remain zero;
- threshold-fit, calibration-fit and untouched-acceptance support remain unavailable.

## R4-GAP-007 authoritative candidate state

Candidate #1 remains durably `NOT_CONFIRMED`.

Candidate #2 remains:

```text
state  = UNKNOWN / PENDING_REVIEW
target = None
```

Its primary source-first review supports a class-specific negative, but this is not authoritative negative truth without admissible independent agreement.

No recent external review attempt changes that state.

## Independent-review attempts after M4 closure

Three review attempts supplied useful diagnostic information but are **not admissible as the required independent verification**:

1. **Repository-aware reviewer:** returned a substantive `CONFIRMED_NEGATIVE` analysis, but disclosed that `CLAUDE.md` exposed the direction of the prior candidate #2 primary result before adjudication. The result is supportive/diagnostic only, not independent verification.
2. **Strict bundle-only reviewer:** correctly stopped without a substantive verdict because the v1 blind ZIP did not contain the governing `CallToUnknown` definition or a sufficient generic negative-evidence rule. The session also disclosed candidate-specific prior context. No verdict is accepted.
3. **ZIP + neutral-material reviewer:** returned `NOT_CONFIRMED` under a literal reading of the supplied broad taxonomy clause, but disclosed candidate-specific prior context and is therefore procedurally contaminated. The result is not an adjudication. It is retained only as evidence that the class wording has a material interpretation ambiguity.

Do not count these attempts as agreeing votes, disagreeing votes, or candidate truth.

## Semantic inconsistency exposed

Current repository sources do not express `CallToUnknown` with one unambiguous boundary.

### Locked canonical taxonomy wording

`data_module/sentinel_data/labeling/schema/taxonomy.yaml` defines class 0 as:

> Low-level calls (`call` / `delegatecall` / `send`) whose return value is not checked, **or calls to unverified external addresses**. Includes `send()` without revert on failure and raw `address.call()` patterns.

Read literally, the second clause can be interpreted more broadly than raw low-level calls and can raise questions about operations such as `address.transfer()` to a caller-controlled address.

### Operational crosswalk wording

`data_module/sentinel_data/labeling/crosswalks/dive.yaml` states:

> `CallToUnknown` is reserved for raw low-level `call` / `send` / `delegatecall` patterns where the callee is unverified.

`data_module/sentinel_data/labeling/crosswalks/solidifi.yaml` likewise explains its `Unchecked-Send -> CallToUnknown` mapping in terms of unchecked low-level `send/call/delegatecall` behavior.

These operational mappings therefore express a narrower class boundary than the broad standalone phrase in the locked taxonomy.

## Why this blocks candidate #2 review

The candidate contains an address-level `transfer` to `msg.sender`. Whether that construct is outside or potentially inside `CallToUnknown` depends on which repository wording is treated as controlling.

A new reviewer must not be forced to invent that semantic precedence. Repeated blind reviews before the class boundary is resolved can legitimately produce different answers while each reviewer follows a different repository text.

Therefore:

- do **not** run another candidate #2 adjudication yet;
- do **not** change candidate #2 state/target;
- do **not** reinterpret any historical zero as negative;
- do **not** use any contaminated review as independent verification;
- do **not** launch M5/full training.

## Required next work

Perform a source/governance audit of the `CallToUnknown` definition itself, independent of candidate #2 outcome:

1. trace the provenance and authority of `taxonomy.yaml` and the relevant source crosswalk clarifications;
2. determine whether the narrower operational wording is merely an already-established interpretation of the locked taxonomy or whether adopting it would materially change the class definition;
3. inspect downstream consumers/evidence to determine the impact of either interpretation;
4. if the class definition must change materially, use the existing R4 class-definition governance/ADR + schema/versioning rules rather than silently editing wording;
5. after the neutral definition is resolved, build a self-contained `r4_gap007_candidate2_independent_review_v2` bundle containing the exact neutral class definition, generic negative-evidence rule, complete source, task and hashes, with no candidate-history leakage;
6. obtain a genuinely isolated independent review with no prior candidate-specific context.

Only after that review may candidate #2 be formally adjudicated.

## Current restart instruction

The active technical-completion action is **CallToUnknown semantic-definition alignment for R4-GAP-007**. M4 is complete; do not repeat M4 or begin M5 while this evidence blocker remains unresolved.
