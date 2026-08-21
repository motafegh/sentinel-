# Phase-8 v10 external-call representation implementation handoff

**Date:** 2026-08-21
**Status:** READY FOR REPOSITORY IMPLEMENTATION; LOCAL GENERATION NOT STARTED
**Decision authority:** R4-D-010 / ADR-R4-010
**Blocking gate:** R4-B008 / G8
**Training:** NOT AUTHORIZED

## Outcome and boundary

The next useful repository tranche is a separately versioned graph-schema-v10
implementation that corrects call-kind semantics. This is source-and-test work
and can be performed by an assistant that only has the remote repository.

The remote tranche must not generate or accept the physical population, because
the accepted preprocessed Solidity, v9 graphs, token tensors, and sidecars are
ignored local data. Full v10 generation and acceptance therefore remain a later
protected-local tranche.

Candidate #2 independent negative review is a separate task. The assistant that
implements v10 must not be treated as an independent verifier if it has been
shown the primary conclusion or R4-GAP-008 reasoning. Use the blind bundle with
a genuinely distinct reviewer/context.

## Why this work is required

The R4-GAP-008 audit scanned all 22,540 repaired-v2 graph/source/sidecar
bindings. It found:

- 217,490 v9 type-11 edges;
- at least 11,702 type-11 edges that are provable same-file declared-library
  calls;
- only 7,057 / 13,413 raw-low-level nodes with type 11;
- only 40 / 4,215 send nodes with type 11;
- only 6,557 / 80,927 transfer nodes with type 11.

Candidate #2 is the concrete regression fixture: all 30 of its type-11 edges
represent `SafeMath` calls, while its actual Ether `Transfer` IR receives no
type-11 edge.

## Required reading order

1. `CLAUDE.md`;
2. `docs/plan/ml-R4/PLAN_STATUS_MATRIX.md`;
3. `docs/plan/ml-R4/runs/2026-08-21_PHASE8_gap008_external_call_semantics_audit.md`;
4. `docs/plan/ml-R4/adrs/ADR-R4-010-versioned-external-call-representation-correction.md`;
5. `data_module/sentinel_data/representation/graph_schema.py`;
6. `data_module/sentinel_data/representation/_schema_version_registry.json`;
7. the call-edge and CEI paths in
   `data_module/sentinel_data/representation/graph_extractor.py`;
8. `data_module/sentinel_data/verification/semantic_checker.py` and its class
   patterns;
9. `ml/src/datasets/vnext_repaired_dataset.py`,
   `ml/src/datasets/vnext_logical_v3_dataset.py`, and
   `ml/src/training/vnext_binding.py`;
10. GNN edge-embedding construction and inference/checkpoint compatibility
    paths.

## Repository implementation contract

### Preserve v9 rather than silently redefining it

- Existing v9 constants, accepted artifacts, digests, manifests, and historical
  checkpoint behavior must remain reproducible.
- Do not merely edit the global v9 `EXTERNAL_CALL=11` meaning and leave the
  schema/extractor identity unchanged.
- Introduce explicit version-aware schema selection. Historical v9 loading must
  continue to validate as v9; a v10 manifest must fail if interpreted through
  v9 constants, and vice versa.
- Register v10 in the schema registry with extractor identity
  `v2.3-r4-call-semantics` and a distinct edge vocabulary.

### Make call kinds explicit

The v10 graph must distinguish at least:

- typed/high-level cross-contract call;
- raw low-level call (`call`, `callcode`, `delegatecall`, `staticcall`);
- reverting Ether `Transfer`;
- boolean-returning Ether `Send`;
- `LibraryCall`.

The implementation should classify Slither IR operation types, not infer every
kind from display-name substrings. A library call must not become an
unknown-target signal. `Transfer` and `Send` must not disappear. If an IR object
cannot be classified safely, represent or report the uncertainty explicitly;
do not clamp it into an unrelated known edge ID.

### Correct downstream semantics

- `CallToUnknown`: low-level/send evidence may be coarse corroboration only;
  library and transfer edges are not positive truth.
- `ExternalBug`: do not alias presence of any external-interaction edge. Keep it
  `NOT_EXTRACTABLE` until a class-specific source-backed v10 signal exists.
- Reentrancy/CEI: high-level, low-level, transfer, and send can represent
  external control handoff; library calls cannot.
- Graph/tool checks remain corroboration, never label authority.

### Bind consumers without silent fallback

- Dataset and run-binding validation must require the selected manifest's exact
  graph-schema and extractor versions.
- The GNN edge embedding must cover every v10 edge ID deliberately.
- Out-of-range or unknown edge IDs must fail clearly; do not rely on inference
  resizing, modulo, clipping, or checkpoint fallback for the new training
  lineage.
- The first v10 comparison holds the existing token tensors and selector
  constant. Selector promotion is a separate decision.
- Do not change class order, label policy, roles, thresholds, calibration,
  training objective, or model architecture in this tranche.

## Required repository tests

At minimum, add deterministic tests that prove:

1. same-file and imported/`using for` library calls map to `LibraryCall`, not
   unknown external;
2. typed high-level calls map to their own kind;
3. `call`, `callcode`, `delegatecall`, and `staticcall` map to raw-low-level;
4. `transfer` and `send` map to distinct kinds;
5. CEI/reentrancy paths ignore library calls but include actual external
   handoffs;
6. `CallToUnknown` and `ExternalBug` follow R4-D-010;
7. v9 fixtures still deserialize/validate with unchanged semantics;
8. v10/v9 schema mismatch and out-of-range edge IDs fail closed;
9. model construction allocates the exact v10 edge vocabulary without loading
   historical learned state;
10. generation APIs require a new output root and refuse an accepted v9 root.

Use focused fixtures first, then the existing representation, semantic-checker,
dataset, binding, model, and inference regression suites affected by the change.

## Remote stop lines

The repository-only assistant must stop after source, tests, documentation, and
normal Git-safe validation. It must not:

- edit or regenerate `representations-r4-v2`;
- claim v10 physical acceptance from unit fixtures;
- fabricate population counts without the local corpus;
- change labels, roles, negative targets, selector authority, thresholds, or
  calibration;
- reuse Run12 weights/checkpoints or authorize training;
- perform candidate #2 independent verification from a context containing the
  primary result.

## Protected-local acceptance tranche after pull

After the implementation is reviewed and pulled locally:

1. verify source/tests and confirm v9 bytes/hashes remain unchanged;
2. generate v10 for bounded regression contracts including GAP-007 candidates
   #1 and #2;
3. reconcile their complete source-reviewed call inventories against v10;
4. generate all 22,540 graphs under
   `data_module/data/representations-r4-v3-candidate` while reusing the accepted
   preprocessing input and holding token tensors/selectors constant;
5. prove one-to-one source/graph/token/sidecar identity coverage;
6. run a deterministic v9→v10 transition audit for every graph and call kind;
7. validate schema/extractor sidecars, deserialization, edge ranges, graph
   invariants, and model-consumer compatibility;
8. create a new physical binding digest and local acceptance record;
9. keep G8 closed if any required check fails.

Even a passing v10 acceptance does not automatically promote the selector,
accept candidate #2, change the objective, fit thresholds/calibration, or launch
training.

## Candidate #2 independent-review material

Blind bundle:

`docs/plan/ml-R4/review_bundles/r4_gap007_candidate2_independent_review_v1.zip`

Expected archive SHA-256:

`2e7f48c9648097624406d167266a42a31055f222a0f468a0453b2f353b343f1a`

The archive contains only its manifest, the review task, and full Solidity
source. It excludes the primary verdict, Slither result, graph findings, and
labels.

## Prompt for the repository-only implementation assistant

```text
Read CLAUDE.md and follow the current DATA/ML authority order. Read
docs/plan/ml-R4/runs/2026-08-21_PHASE8_gap008_external_call_semantics_audit.md,
docs/plan/ml-R4/adrs/ADR-R4-010-versioned-external-call-representation-correction.md,
and docs/plan/ml-R4/runs/2026-08-21_PHASE8_v10_external_call_implementation_handoff.md.
Implement the bounded repository-only R4-D-010 v10 call-semantics tranche fully:
preserve v9 behavior and artifacts, introduce explicit version-aware v10 schema
and extractor identity, distinguish typed high-level/raw low-level/Transfer/Send/
LibraryCall IR kinds, correct semantic consumers and CEI behavior, bind datasets,
run metadata, model edge vocabulary, generation roots, and fail-closed checks,
and add focused plus affected regression tests. Do not regenerate ignored local
data, do not claim physical acceptance, do not change labels/roles/selectors/
thresholds/calibration/objective/architecture, do not reuse Run12 state, and do
not authorize training. Progressively document implementation evidence and leave
an exact protected-local generation/acceptance handoff. Do not perform candidate
#2 independent negative verification in this context.
```
