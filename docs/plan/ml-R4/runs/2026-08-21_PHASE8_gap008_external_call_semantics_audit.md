# Phase-8 R4-GAP-008 external-call semantics audit

**Date:** 2026-08-21
**Status:** COMPLETE — remediation decision accepted as R4-D-010
**Physical authority:** repaired-v2 / graph schema v9 remains immutable
**Training:** NOT AUTHORIZED

## Decision-useful answer so far

Candidate #2 of the R4-GAP-007 `CallToUnknown` pilot exposed a real semantic defect in the accepted v9 representation. Its graph contains 30 `EXTERNAL_CALL` edges; every one is attached to a same-file `SafeMath` library-call node. The contract's real `_customerAddress.transfer(_dividends)` interaction is present as a CFG call node but has no type-11 edge.

This is not a physical-integrity failure: the candidate source, metadata, graph, token tensor, and sidecar are present and hash-bound. It is a **representation semantic-integrity** problem. The current edge does not reliably mean what the class pattern and semantic checker assume it means.

No accepted artifact was modified. The full-population audit proved material impact, and ADR-R4-010 now requires a separately versioned v10/extractor candidate before G8.

## Trigger evidence

Candidate identity:

```text
candidate_id = r4neg-bfe90ef82e33a324d612256a5d4053c6
contract_id  = f7afe9fff9f6c117c6cd9dd4730c0f12e3cc3c8ab98797911de091e240051b93
group_id     = r4grp-dc843217924fe207d2a658ada327615a
class        = CallToUnknown / index 0
ordinal      = 2
```

Physical evidence:

- Solidity source: 343 raw / 344 normalized lines, pragma `^0.4.25`, compile status `ok_normalized_bytes`;
- source SHA-256 equals contract ID;
- graph: 205 nodes, 872 edges, one component, 30 type-11 edges;
- token input: `[4,512]`, 1,819 of 5,369 unique code tokens retained (`0.3387967964239151`);
- the Ether-transfer source site is absent from the selected token windows but present in the graph as a CFG call node;
- targeted Slither 0.11.5 / solc 0.4.25 found no raw/unchecked low-level call but reported two class-separate `reentrancy-unlimited-gas` findings involving the `transfer` path.

Exact graph-origin inspection found:

- 30/30 type-11 self-loops on nodes whose Slither IR operations are `LibraryCall` to `SafeMath`;
- 0 type-11 self-loops on `_customerAddress.transfer(_dividends)`;
- graph `has_cei_path=1`, so the misclassified library edges can also affect structural consumers beyond `CallToUnknown`.

## Source contradiction

The current sources disagree:

| Surface | Current meaning |
|---|---|
| `verification/patterns/CallToUnknown.yaml` | Same-file/OZ library calls are excluded; class intent is low-level calls to unknown/dynamic targets |
| `verification/semantic_checker.py` | Any type-11 edge is a positive CallToUnknown/ExternalBug signal |
| `representation/graph_extractor.py` | `node.high_level_calls` or `node.low_level_calls` creates type 11; comments explicitly include Slither `LibraryCall` as external |
| graph regression test | Documents library counting as known behavior rather than rejecting it |
| Slither IR for candidate #2 | Distinguishes `LibraryCall` from `Transfer` exactly, so a corrected implementation is technically possible |

The problem cannot be repaired safely by silently changing v9 behavior. The accepted repaired-v2 graphs and evidence bindings use extractor `v2.2-r4-repaired` and schema v9; a behavior change under those identities would break reproducibility.

## Preliminary queue-focused profile

A read-only first pass across the 25 queued `CallToUnknown` candidates found:

| Check | Count | Rate |
|---|---:|---:|
| candidates with at least one type-11 edge | 18 | 72% |
| candidates whose type-11 edges were all calls to a library declared in the same source | 1 | 4% |
| candidates containing raw low-level-call syntax | 6 | 24% |
| candidates containing `.transfer(...)` | 17 | 68% |
| transfer-containing candidates with no transfer-named type-11 edge | 12 | 48% of queue / 71% of transfer subset |
| candidates retaining less than 50% of code tokens | 13 | 52% |

The durable queue report is:

`reviews/R4-GAP-008/external_call_semantics_call_to_unknown_queue_v1.json`

Its SHA-256 is `04a1e0863dc8bef9ff828584b80e9d7e09e2d01dd43c7f49403deb4b74787e05`.

Name-based declared-library classification can undercount imported libraries, aliases, using-for syntax, and metadata-normalization variants. Counts are therefore proven lower bounds, not exhaustive false-positive estimates.

## Full-population result

The audit processed every repaired-v2 graph/source/sidecar binding:

| Finding | Count | Rate / denominator |
|---|---:|---:|
| graphs scanned | 22,540 | 100% of repaired-v2 |
| total type-11 edges | 217,490 | all type-11 edges |
| provable declared-library type-11 edges | 11,702 | 5.380% of type-11 edges |
| graphs with a provable library type-11 edge | 1,489 | 6.606% of graphs |
| graphs whose type-11 edges are all provable library edges | 438 | 2.517% of graphs with type 11 |
| raw low-level-call nodes receiving type 11 | 7,057 / 13,413 | 52.613% |
| send nodes receiving type 11 | 40 / 4,215 | 0.949% |
| transfer nodes receiving type 11 | 6,557 / 80,927 | 8.102% |
| transfer-containing graphs without a transfer-linked type 11 | 9,013 / 13,025 | 69.198% |
| send-containing graphs without a send-linked type 11 | 817 / 834 | 97.962% |
| graphs retaining less than 50% of code tokens | 12,653 | 56.136% |

Machine-readable population report:

`reviews/R4-GAP-008/external_call_semantics_population_v1.json`

SHA-256: `77f902608df4371271e085136d3990bdbd555fcbb80aefaa86f9af9d3ecbccd3`.

Both reports record repository HEAD
`9207843d609d8149bc7f33327f0fe0e3bcccbf31`, explicitly record that the
documentation worktree was dirty, and bind the exact audit implementation as
SHA-256 `e8b07188c986ff662b78e0e1c17c6eee896488315a07d5d4fe558655c3004211`.
The HEAD is the checked-out input baseline, not a false claim that the new audit
script already existed in that commit.

The audit status is `PASS_DIAGNOSTIC_ONLY`; all label, artifact-mutation, selector-promotion, and training authorizations are false.

## Completed audit procedure

1. Added deterministic read-only audit script `scripts/p8_audit_external_call_semantics.py`.
2. Added three focused classifier tests.
3. Audited the 25-candidate CallToUnknown queue stratum.
4. Audited all 22,540 repaired-v2 graphs and their source/sidecar bindings.
5. Saved queue and population machine-readable reports without altering accepted artifacts.
6. Accepted ADR-R4-010: v9 remains immutable historical evidence; a versioned v10/extractor candidate and full local acceptance are required before G8.

## Decision

This is a critical representation-adequacy blocker, not a reason to discard the source corpus or accepted provenance. R4-D-010 establishes:

- v9/repaired-v2 stays available for reproduction;
- no new full training may use v9;
- repository implementation may proceed on `v10` / extractor `v2.3-r4-call-semantics` / `representations-r4-v3-candidate`;
- implementation must distinguish high-level, low-level, transfer, send, and library calls;
- full generation and binding acceptance must run locally;
- selector promotion, confirmed-negative labels, and training remain separate gates.

## Stop lines

- Do not edit existing `.pt`, `.tokens.pt`, `.rep.json`, or accepted V3 snapshot files.
- Do not call a v9 type-11 edge vulnerability truth.
- Do not change v9/extractor `v2.2-r4-repaired` semantics in place.
- Do not launch training or promote the guarded selector.
- Do not accept candidate #2 as a confirmed negative until its primary review is complete and a genuinely independent verifier agrees.
