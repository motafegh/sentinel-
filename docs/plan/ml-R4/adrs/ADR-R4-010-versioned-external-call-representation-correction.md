# ADR-R4-010 — Version external-call representation correction before G8

**Date:** 2026-08-21
**Status:** ACCEPTED
**Decision ID:** R4-D-010
**Scope:** graph call-kind semantics, future physical representation lineage, and Phase-8 training eligibility

## Context

R4-GAP-007 candidate #2 exposed a contradiction in graph schema v9. The candidate's 30 `EXTERNAL_CALL` edges all represent same-file `SafeMath` `LibraryCall` operations, while its real Solidity `Transfer` operation has no type-11 edge.

The contradiction exists in executable source:

- `graph_extractor.py` creates type 11 from `node.high_level_calls` and `node.low_level_calls`; Slither includes `LibraryCall` in the high-level collection;
- the same path does not inspect `Transfer` or `Send` IR;
- `semantic_checker.py` treats any type-11 edge as a positive `CallToUnknown` and `ExternalBug` signal;
- the CallToUnknown verification pattern explicitly excludes library calls;
- historical BCCC review identifies transfer-only behavior as a major false-positive source.

The read-only R4-GAP-008 audit then inspected all 22,540 repaired-v2 graphs:

| Finding | Count | Rate / denominator |
|---|---:|---:|
| total v9 type-11 edges | 217,490 | all type-11 edges |
| provable same-file declared-library type-11 edges | 11,702 | 5.380% of type-11 edges |
| graphs with a provable declared-library type-11 edge | 1,489 | 6.606% of graphs |
| graphs whose type-11 edges are all provable library edges | 438 | 2.517% of 17,405 graphs with type 11 |
| raw low-level-call nodes receiving type 11 | 7,057 / 13,413 | 52.613% |
| send nodes receiving type 11 | 40 / 4,215 | 0.949% |
| transfer nodes receiving type 11 | 6,557 / 80,927 | 8.102% |
| transfer-containing graphs with no transfer-linked type 11 | 9,013 / 13,025 | 69.198% |
| send-containing graphs with no send-linked type 11 | 817 / 834 | 97.962% |

The declared-library classifier is intentionally conservative and can undercount imported/aliased/using-for library forms. The 11,702 figure is therefore a proven lower bound, not an estimate of every false edge.

## Decision

### 1. Preserve v9 as immutable historical/physical evidence

Repaired-v2 and logical V3 remain valid for byte/provenance reproducibility and for the evidence already derived from those exact bytes. Do not overwrite or relabel any accepted v9 artifact.

### 2. Withdraw v9 from future full-training eligibility

The 100-epoch Phase-8 run must not use graph schema v9 as its future physical input lineage. G8 remains open. This is an adequacy boundary, not deletion or retroactive corruption of historical evidence.

### 3. Require a versioned v10 candidate

Repository implementation must create a new candidate lineage rather than changing v9 under the same identity:

- graph schema candidate: `v10`;
- extractor candidate: `v2.3-r4-call-semantics`;
- representation root candidate: `representations-r4-v3-candidate`;
- preprocessing source remains accepted `sentinel-preprocessed-r4-v2`;
- existing v9 roots remain read-only.

The v10 call-kind vocabulary must distinguish:

- typed/high-level external call;
- raw low-level call;
- reverting value `Transfer`;
- boolean-returning `Send`;
- `LibraryCall`.

Implementation population probing additionally proved that Slither
`NewContract` is a call-family IR operation in this corpus. V10 must therefore
also distinguish contract creation instead of reporting or silently dropping
it. Contract creation is an external handoff for structural graph analysis,
but it is not a `CallToUnknown` positive signal.

`LibraryCall` must not be treated as an unknown external target. `Transfer` and `Send` must not silently disappear. Exact edge IDs are an implementation detail, but the schema registry and model consumer must bind them explicitly and must not clamp an unrecognized type into another meaning.

### 4. Correct semantic consumers

- `CallToUnknown` graph checking may use low-level/send call-kind evidence only as a coarse positive signal; it must not treat library or transfer edges as vulnerability truth.
- `ExternalBug` must no longer alias any-external-call presence. Until a source-backed class-specific v10 signal is defined, it is `NOT_EXTRACTABLE` from call-kind edges.
- Reentrancy structural analysis may use high-level, low-level, transfer, send, and contract-creation interaction kinds, but not library calls as external control handoff.
- Tool/graph signals remain corroboration; labels still require source/evidence authority.

### 5. Keep token-selector promotion separate

The v10 graph correction does not silently promote `target_aware_guarded_v1`. Initial v10 graph comparison must hold token tensors/selectors constant so graph-call semantics can be evaluated independently. Token-selector promotion remains a separate decision and evidence gate.

## Required acceptance evidence

Before any v10 physical lineage is accepted:

1. unit fixtures must distinguish library, typed high-level, low-level, transfer, and send IR;
2. candidate #1 and #2 must reproduce their source-reviewed call-kind inventories;
3. a full-population side-by-side v9→v10 audit must show no missing graph/source identities and quantify every call-kind transition;
4. all v10 graph/sidecar files must pass binding, deserialization, edge-range, and schema/extractor-version checks;
5. v9 source/graph/token/sidecar hashes must remain unchanged;
6. the model adapter must accept the v10 edge vocabulary without OOB clamping or checkpoint fallback;
7. a new physical binding digest and acceptance record must be created;
8. no full training, selector promotion, label change, threshold fitting, or calibration follows automatically.

## Consequences

R4-GAP-008 evidence is resolved by this decision, but implementation and local v10 generation remain open work. The implementation can be developed repository-remotely because it requires source/tests, while full-population generation and acceptance must run locally against ignored repaired-v2 data.

R4-GAP-007 may continue source-first candidate review. Candidate #2 primary review supports a class-specific negative but remains UNKNOWN/target `None` until a genuinely independent verifier agrees.

## Rollback

Rollback is selection of the immutable v9/repaired-v2 artifacts for historical reproduction only. Do not use rollback to authorize new full training. If the v10 candidate fails acceptance, keep G8 open and revise the versioned candidate; never patch accepted v9 bytes.

## Evidence

- `runs/2026-08-21_PHASE8_gap008_external_call_semantics_audit.md`;
- `reviews/R4-GAP-008/external_call_semantics_population_v1.json`;
- `reviews/R4-GAP-008/external_call_semantics_call_to_unknown_queue_v1.json`;
- `runs/2026-08-21_PHASE8_gap007_candidate2_primary_review.md`;
- `data_module/sentinel_data/representation/graph_extractor.py`;
- `data_module/sentinel_data/verification/semantic_checker.py`;
- `data_module/sentinel_data/verification/patterns/CallToUnknown.yaml`.
