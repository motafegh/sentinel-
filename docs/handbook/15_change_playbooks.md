# 15 — Change playbooks

**Read this when:** you are about to change a schema, retrain, regenerate a circuit, upgrade contracts, or add a node/tool/class/source.

**Skip this if:** you are only reading or operating unchanged code.

**Estimated reading time:** 13 minutes.

## 30-second summary

SENTINEL changes have predictable blast radii. Start with the compatibility invariant, enumerate producers/consumers/artifacts/evidence, add mismatch tests, migrate in dependency order, and finish with module plus live verification. Never patch a downstream shape to accept an unversioned upstream semantic change.

## Just-enough mental model

```text
source policy/schema → DATA artifacts → teacher → proxy/circuit → verifier/registry → AGENTS/report/eval → operations/docs
```

The farther left a change begins, the larger its blast radius.

## Actual runtime/source walkthrough

### Schema, feature, edge, or class order

Update canonical DATA schema and version; regenerate representations/export/splits; update ML adapters/model; retrain/threshold/calibrate; redistill proxy; regenerate EZKL/verifier; migrate registry/interface if class layout changed; update AGENTS maps/eval/config; run end-to-end compatibility tests.

### Retrain teacher without shape change

Freeze DATA identity/config/seed; train and evaluate; tune thresholds/calibration; compute checkpoint hash and warm-up drift baseline; promote the complete artifact set; redistill proxy and reassess agreement because weights changed even if fusion width did not.

### Circuit regeneration

Pin teacher/proxy/EZKL; retrain proxy if needed; export ONNX; generate calibration; settings/compile/SRS/setup; prove/verify; regenerate Solidity verifier; test/deploy/swap; update artifacts and proof semantics review.

### UUPS/verifier upgrade

Preserve storage order/types; add implementation initializer only if versioned safely; run storage-layout and proxy tests; deploy verifier/implementation; authorize upgrade with controlled owner; verify events, old records, new submissions, pause controls, and rollback procedure.

### New LangGraph node

Classify evidence determinism; define `AuditState` fields/reducers and structured status; add timed node/routing/fan-in; update synthesis/evaluation/timeouts; test fast/deep/error paths; update graph metadata.

### New MCP tool

Define JSON Schema and bounded resources; implement registration+dispatcher; structured errors and health; add authentication/trust analysis if mutating; wire a graph node only deliberately; document port/tool family and live prerequisites.

### New vulnerability class

This is a full-system migration: label schema/order, datasets, model head/metrics/thresholds, proxy output/circuit, 138-signal layout and registry fixed array/constants, evidence mapping, UI/report/evaluation. Prefer a versioned V3 interface rather than mutating V2 semantics in place.

### New DATA or RAG source

Pin provenance/license; normalize stable identity; define labels/metadata; deduplicate and contamination-test; verify quality; version catalogs/index schemas; measure downstream effect; preserve removal/rebuild ability.

## Interfaces, data shapes, and configuration

Every change record should contain:

- motivation and measured baseline;
- compatibility version and migration type;
- affected source symbols, artifacts, routes/tools/contracts;
- before/after schemas and hashes;
- verification matrix: static, smoke, module, live;
- security/trust changes and rollback;
- updated status and known limitations.

Use [`handbook.toml`](_meta/handbook.toml) as a checklist of critical facts and owned paths, then confirm each item against executable source.

## Failure modes and current limitations

- Changing values without changing versions creates silent mixed artifacts.
- Reusing thresholds/calibration after retraining creates false decision semantics.
- Updating a verification key without the generated verifier/deployment creates proof incompatibility.
- Adding a tool without orchestration wiring creates a documented-but-unused service.
- Adding evidence without `tool_status` contaminates evaluation.
- Updating tests/docs but not local-only acquisition instructions makes fresh-clone verification impossible.

## Common change recipe

Universal recipe:

1. Bind baseline commit/artifacts/metrics.
2. Write the invariant and blast-radius checklist.
3. Add failure/mismatch tests.
4. Implement from producer to consumer.
5. Regenerate artifacts deterministically and record hashes.
6. Run static → smoke → module → live checks.
7. Record failures without weakening gates.
8. Update handbook metadata, relevant chapters, and status page.

## Verification commands

```bash
python3 docs/handbook/tools/verify_handbook.py static
python3 docs/handbook/tools/verify_handbook.py inventory
git diff --check
git diff --name-only
```

Then run every affected module/live command from [operations](14_operations.md).

## Optional deep references

- [Cross-module contracts](11_cross_module_contracts.md)
- [Security and trust](12_security_and_trust.md)
- [Evaluation](13_evaluation.md)
- [Reference](17_reference.md)

## Technical mastery layer

### Prerequisite knowledge

Know dependency graphs, migrations, artifact regeneration, staged rollout, rollback, and test pyramids.

### Source map and reading order

Choose the owning technical guide from [`technical/`](technical/), then inspect producer symbol, metadata `source_anchors`, consumers in [cross-module contracts](11_cross_module_contracts.md), focused tests, artifact matrix, operations, and status.

### Execution trace and worked example

A class addition touches taxonomy/class order, labels/export, model head/checkpoint/thresholds, proxy dimensions/circuit/keys/verifier, fixed Solidity arrays/constants, AGENTS evidence/evaluation, and every payload. It is a coordinated version migration, not a one-line enum edit.

### Implementation practice

Every recipe uses: characterize with a failing test → edit smallest producer → update consumers → regenerate/version artifacts → focused/module/live verification → rollout check → rollback check. The matching lab supplies a safe rehearsal; [L10](labs/10_end_to_end_capstone.md) is the cross-module rehearsal.

### Review and ownership check

Can you state blast radius and rollback bundle before editing? If not, return to source maps rather than starting implementation.
