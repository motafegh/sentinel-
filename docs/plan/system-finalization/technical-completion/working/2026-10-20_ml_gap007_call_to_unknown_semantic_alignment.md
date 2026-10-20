# ML / R4-GAP-007 — CallToUnknown semantic alignment checkpoint

**Date:** 2026-10-20  
**State:** COMPLETE — closed by R4-D-016  
**Applies to:** R4-GAP-007 / `CallToUnknown` confirmed-negative evaluation evidence  

## Accepted boundary

The accepted ML/R4 state remains:

- logical authority R4-D-009 / `sentinel-r4-vnext-v3`;
- immutable V10/V2.6 graph/control authority R4-D-011;
- immutable guarded-token authority R4-D-013;
- bounded positive-only objective/evaluation authority R4-D-014;
- M0-M4 complete;
- R4-D-015 = `PASS_BOUNDED_PILOT_FULL_TRAINING_STILL_HELD`;
- M5/full training unauthorized; G8 open;
- confirmed-negative targets remain zero;
- threshold-fit, calibration-fit and untouched-acceptance support remain unavailable.

## Problem audited

Candidate #2 independent-review preparation exposed conflicting `CallToUnknown` wording:

- `taxonomy.yaml` used a broad phrase that could be read to include any call to an unverified external address;
- original SolidiFI/DIVE crosswalks and the class-specific verification pattern operationalized the class as raw/low-level unknown-target behavior;
- accepted R4-D-010 and the V10 semantic checker separated transfer/high-level/library/creation interactions from `CallToUnknown` corroboration.

Three external candidate #2 review attempts were useful diagnostics but are not admissible independent verification because they were contaminated by candidate-specific prior context and/or the v1 bundle lacked self-contained governing material.

## Provenance findings

The inconsistency is original, not recent drift:

1. `taxonomy.yaml` was introduced on 2026-06-11 with the broad prose.
2. The SolidiFI crosswalk was committed minutes later and mapped `Unchecked-Send -> CallToUnknown` as a low-level-send concern.
3. The DIVE crosswalk, committed the same day, explicitly stated that `CallToUnknown` is reserved for raw low-level `call` / `send` / `delegatecall` patterns where the callee is unverified.
4. The Stage-4 class pattern subsequently defined the class as a low-level call to an unknown/dynamic address.
5. R4-D-010 later made call-kind separation explicit and directed `CallToUnknown` graph checking to low-level/send evidence while excluding transfer/library from vulnerability truth.
6. Current V10 `semantic_checker.py` and `graph_schema_versions.py` encode only `LOW_LEVEL_CALL` + `ETHER_SEND` as coarse `CallToUnknown` signals.

The narrow operational meaning therefore predates R4-GAP-007 candidate review and was not invented to produce a desired candidate #2 result.

## Decision — R4-D-016

R4-D-016 accepts `CallToUnknown` as the **raw/dynamic low-level unknown-target interaction class**.

Included call family:

- `call`;
- `delegatecall`;
- legacy `callcode`;
- `staticcall` where applicable;
- `send`.

A checked raw low-level call can still qualify because target uncertainty is class-defining; unchecked return/failure handling is additional positive evidence rather than the sole condition.

Explicit exclusions when destination uncertainty is the only reason:

- Solidity `transfer()`;
- ordinary typed high-level interface calls;
- same-file/in-scope library calls;
- contract creation;
- `selfdestruct` / `suicide` destinations.

The canonical taxonomy and class verification pattern are aligned to this definition. A focused V10 regression test locks low-level/send positive signals and transfer/high-level/library/creation exclusions.

## Versioning conclusion

No taxonomy schema bump is required:

- class index 0 remains `CallToUnknown`;
- the 10-class set/order is unchanged;
- model output shape is unchanged;
- source crosswalk mappings are unchanged;
- the clarification aligns canonical prose to long-standing operational semantics.

Future expansion of the class to make typed high-level or transfer destinations positive merely because the address is unverified would require a new governed semantic change.

## Candidate consequences

### Candidate #1

The historical `NOT_CONFIRMED` primary rationale depended on the broad clause to treat a caller-selected typed high-level callback as contradictory class-positive evidence.

Under R4-D-016 that rationale is definition-stale. The historical record remains preserved but does not control current semantics.

Candidate #1 remains the immutable queue state:

```text
state  = UNKNOWN / PENDING_REVIEW
target = None
```

If reused, candidate #1 requires a new primary review under R4-D-016.

### Candidate #2

The existing candidate #2 primary review is semantically compatible with R4-D-016 because it already:

- reviewed the complete source;
- excluded same-file SafeMath library calls;
- found no low-level/send family interaction;
- treated the sole `transfer()` interaction as outside direct `CallToUnknown` scope.

It remains:

```text
state  = UNKNOWN / PENDING_REVIEW
target = None
```

until a genuinely isolated independent reviewer agrees from sufficient self-contained evidence.

## Blind-review consequence

`r4_gap007_candidate2_independent_review_v1.zip` is historical only and retired for future adjudication.

The next artifact must be `r4_gap007_candidate2_independent_review_v2.zip`, containing only:

- complete source;
- manifest/hashes;
- review task;
- neutral R4-D-016 class definition/exclusions;
- generic class-specific confirmed-negative evidence rule.

It must contain no primary-review result or prior candidate conclusion, and the reviewer context must contain no candidate-specific history.

## Current restart instruction

Semantic alignment is complete. The active R4-GAP-007 action is now:

1. build/verify the self-contained candidate #2 v2 blind bundle;
2. provide it to a genuinely isolated reviewer with no repository/project history;
3. adjudicate candidate #2 only after that result returns.

Do not begin M5/full training, generate target `0`, or fit unsupported threshold/calibration roles while confirmed-negative evidence remains zero.
