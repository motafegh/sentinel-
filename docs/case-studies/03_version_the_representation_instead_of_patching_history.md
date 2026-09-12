# Case Study 03 — Version the representation instead of patching history

## 1. Decision in one sentence

When SENTINEL proved that graph schema v9 misrepresented important call semantics, it preserved v9 for reproducibility and created a new V10 lineage instead of silently patching accepted historical artifacts.

## 2. Problem

A source-reviewed negative candidate exposed a contradiction in the graph representation used by the repaired DATA/ML path.

For that contract, all 30 historical `EXTERNAL_CALL` edges represented same-file `SafeMath` library calls, while the actual Solidity `transfer` operation was not represented by the same call edge kind. The problem was larger than one sample: the graph vocabulary itself conflated semantically different operations and omitted substantial `Transfer` and `Send` behavior.

That made the issue relevant to model inputs, graph-based semantic checks, and any future training claim.

## 3. Evidence

The full-population R4-GAP-008 audit over 22,540 repaired graphs established several concrete facts:

- 217,490 historical type-11 edges existed;
- at least 11,702 were provable same-file declared-library edges;
- 9,013 of 13,025 transfer-containing graphs had no transfer-linked type-11 edge;
- 817 of 834 send-containing graphs had no send-linked type-11 edge.

The declared-library count was intentionally conservative, so it was treated as a proven lower bound rather than an estimate of every incorrect edge.

The defect was therefore a representation-semantics problem, not a reason to manufacture different labels.

## 4. Shortcut rejected

The project rejected two shortcuts:

1. patch the accepted v9 artifacts in place and continue as if lineage had not changed;
2. keep v9 eligible for the new full training run because reproducing a new representation was expensive.

Either choice would make later results difficult to audit because historical evidence and corrected semantics would share the same artifact identity.

## 5. Decision

R4-D-010 preserved v9 as immutable historical evidence but withdrew it from eligibility for the new repaired full-training lineage.

A new graph schema, V10, was required to distinguish typed/high-level calls, raw low-level calls, `Transfer`, `Send`, `LibraryCall`, and contract creation with explicit consumer semantics.

The decision also separated graph correction from token-selector changes so that multiple representation changes could not be hidden inside one comparison.

## 6. Implementation and validation

The V10 work was developed as a separately versioned candidate and subjected to source fixtures, population probing, runtime provenance checks, graph/sidecar binding, model-consumer compatibility checks, and full-population transition analysis.

The final accepted V2.6 candidate contains 22,540 identities and 67,620 files, uses the required 22,539 Slither-0.10 plus one identity-bound Slither-0.11.5 runtime split, and binds to digest `d9f925588913e66476cfbc097bace7daa7e673295fe2a243760313d0bef5ebdd`.

R4-D-011 accepts only that exact protected lineage.

## 7. Result

SENTINEL obtained a physically accepted V10 representation lineage without rewriting the historical evidence used to discover the defect.

This creates a clear audit trail:

- v9 explains the historical behavior;
- V10 records corrected semantics;
- the acceptance decision identifies exactly which bytes and digest are authoritative.

## 8. Remaining limitation

Physical representation acceptance is not model-quality evidence.

R4-D-011 explicitly does not authorize repaired training, threshold fitting, calibration, new label truth, selector changes, or production promotion. Phase 8 / G8 remains open.

## 9. Evidence trail

Primary references:

- `docs/plan/ml-R4/adrs/ADR-R4-010-versioned-external-call-representation-correction.md`;
- `docs/plan/ml-R4/adrs/ADR-R4-011-v10-v26-physical-representation-acceptance.md`;
- `docs/plan/ml-R4/runs/2026-08-21_PHASE8_gap008_external_call_semantics_audit.md`;
- `docs/plan/ml-R4/runs/2026-09-02_PHASE8_v10_v26_physical_acceptance_and_no_launch.md`.

Source, tests, machine-readable acceptance records, and protected evidence remain higher authority than this explanation.
