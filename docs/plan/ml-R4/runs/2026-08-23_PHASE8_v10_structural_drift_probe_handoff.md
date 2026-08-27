# Phase-8 V10 structural-drift probe handoff

Date: 2026-08-23
Status: **CLOSED — 20/20 RESOLVED; HISTORICAL HANDOFF; SUPERSEDED AS RESTART AUTHORITY**
Scope: R4-B008 bounded structural-drift tranche only; no label, selector, objective, threshold, checkpoint, training, model-quality, or physical-acceptance authority

> **Current-state pointer (2026-08-27):** Do not resume the former 19/20 investigation from this file. The final bounded V2.5 result is 20/20 resolved with zero unexplained drift. Current restart authority is `runs/2026-08-27_PHASE8_v10_v25_current_restart_checkpoint.md`, followed by `runs/2026-08-26_PHASE8_v10_v25_full_candidate_staging.md`.

## Inherited historical boundary

This tranche began after the 26-contract parse-only remediation had already completed in the V2.4 compatibility lineage. The protected V2.4 diagnostic candidate had:

- 22,540 identities;
- exact accepted-V9 token bytes;
- zero parse-only outputs;
- zero unclassified call IR;
- required runtime split of 22,539 primary Slither-0.10 artifacts plus one identity-bound Slither-0.11.5 exception.

The complete transition audit v2 then reported 20 unexpected non-parse-only structural differences relative to the immutable V2.3 structural reference. Those 20 identities became this tranche's only root-cause scope.

Frozen structural reference:

`data_module/data/representations-r4-v3-candidate-v2.3-structural-reference-6087dc6d`

Binding digest:

`6087dc6d76d781efbefe0c4984458d291790c38b1c55d852f48fd796222b0260`

Protected V2.4 diagnostic candidate:

`data_module/data/representations-r4-v3-candidate`

Neither historical root may be overwritten during V2.5 construction.

## Diagnostic method established here

`docs/plan/ml-R4/scripts/p8_probe_v10_structural_drift.py` compares exact labelled directed multigraphs through unchanged edge type 10. Node labels include persisted `node_metadata` plus the exact feature row; edge direction, type, and multiplicity are preserved. Search exhaustion fails closed.

The bounded diagnostic established two distinct root-cause classes rather than permitting a blanket structural waiver:

1. node order/index instability where exact labelled graph semantics remain isomorphic;
2. CFG-node classification instability around storage mutations, requiring independent semantic evidence before any correction can be accepted.

## Versioned deterministic repair

The storage-classification root cause came from relying on Slither-derived alias resolution for state writes. Expression-level lvalue evidence is available earlier and can identify persistent storage through either:

- a direct `StateVariable` root; or
- a `LocalVariable` rooted in storage (`location = storage`, `is_storage = true`).

The versioned repair therefore advanced the extractor to:

`v2.5-r4-call-semantics-deterministic-cfg`

The V2.5 rule preserves CALL priority and existing Slither-confirmed WRITE classification, promotes only positively evidenced persistent-storage writes, excludes storage-reference declaration/rebinding false positives, and leaves historical v9 extraction inert.

Focused representation tests included negative controls for memory-member writes and bare storage-reference declaration/rebinding cases.

## Intermediate 19/20 checkpoint — historical only

An intermediate bounded verifier once reported:

- 8 `V25_NODE_ORDER_INDEX_EQUIVALENCE_REPRODUCED`;
- 11 `V25_DETERMINISTIC_STORAGE_WRITE_CORRECTION_PROVEN`;
- 1 blocked identity: `dive/83c9d2d26dc19eaa2aee29fa7aedb4f4e208429a96cc7a0ffee7491b9830630d`.

That state is **not current**. The blocked identity was already repeat-deterministic; its initial semantic evidence set simply omitted eight additional lower-class → WRITE nodes exposed after the first canonicalization pass.

The fail-closed blocker-evidence probe derived those eight nodes directly from the reproducibility report and proved all eight are persistent-storage writes under exact Slither 0.10.0. After merging that evidence, the identity contains 13 positively evidenced semantic WRITE targets total.

## Final closure result

Three fresh V2.5 generations of all 20 identities under exact Slither 0.10.0 completed with 20/20 full-analysis records each.

Final bounded verifier result:

- `unexpected_identities = 20`;
- `semantic_correction_identities = 12`;
- `index_equivalence_identities = 8`;
- `repeat_generations = 3`;
- `bounded_v25_reproducibility_passed = true`;
- `zero_unexplained_drift = true`;
- `blocking_identities = []`.

Final decision census:

| Decision | Identities |
|---|---:|
| `V25_DETERMINISTIC_STORAGE_WRITE_CORRECTION_PROVEN` | 12 |
| `V25_NODE_ORDER_INDEX_EQUIVALENCE_REPRODUCED` | 8 |

The 12 WRITE identities are not acceptance waivers: only independently evidenced persistent-storage mutation nodes are canonicalized to `CFG_NODE_WRITE`, after which exact node-index-invariant labelled multigraph equivalence is still required. The 8 index identities likewise require exact labelled graph equivalence rather than raw endpoint-index forgiveness.

Durable closure record:

`reviews/R4-GAP-008/2026-08-26_v10_v25_bounded_structural_closure.md`

Evidence-chain/full-population gate support:

- `scripts/p8_validate_v10_v25_evidence_chain.py`;
- `scripts/p8_audit_v10_transition_v3.py`.

## Current successor work

The bounded root-cause tranche is finished. Current R4-B008 work is the fresh full physical V2.5 candidate:

1. Stage A — generate 22,539 ordinary identities under exact Slither 0.10.0 and defer the one declared runtime exception without invoking it;
2. Stage B — fail-closed transfer of only validated primary triples to a fresh final-lineage root;
3. Stage C — fill exactly the declared exception under Slither 0.11.5;
4. Stage D — bind all 22,540 candidate identities and exact runtime/token/schema/extractor invariants;
5. Stage E — run the complete V3 transition audit and require the same exact 8+12 evidence classes with zero additional unexplained non-parse-only drift;
6. review the complete report before any separate physical-acceptance decision.

Current protocol:

`runs/2026-08-26_PHASE8_v10_v25_full_candidate_staging.md`

Current restart checkpoint:

`runs/2026-08-27_PHASE8_v10_v25_current_restart_checkpoint.md`

## Stop lines

- Do **not** resume the intermediate 19/20 state.
- Do not restart the 26-contract parse-only repair.
- Do not widen structural acceptance beyond the exact proven 8+12 classes.
- Do not overwrite accepted V9/repaired-v2, frozen V2.3 reference, or protected V2.4 diagnostic history.
- Do not treat bounded closure as physical acceptance.
- Do not authorize full training. Training remains a separate later gate.
