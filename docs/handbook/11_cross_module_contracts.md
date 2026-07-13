# 11 — Cross-module contracts

**Read this when:** you need to know what must stay compatible across DATA, ML, ZKML, Contracts, and AGENTS.

**Skip this if:** you are reading a subsystem for orientation only; return here before changing it.

**Estimated reading time:** 14 minutes.

## 30-second summary

Most dangerous SENTINEL changes cross directories silently. The critical contracts are the ten-class order, v9 graph shape, token window shape, 128-value fusion seam, 138-signal proof layout, fixed-point encoding, model/artifact hashes, HTTP/MCP payloads, `AuditState`/report fields, and contract storage/interface order. Compatibility is established by shared values plus tests—not by similar names.

## Just-enough mental model

| Producer | Contract | Consumers |
|---|---|---|
| DATA | v9 graph, labels, tokens, split/export hash | ML training/inference |
| ML | ten probabilities, eye data, fusion[128], model hash | AGENTS, ZKML |
| ZKML | instances[128 inputs + 10 outputs], proof | Contracts/audit MCP |
| Contracts | V1/V2 records/events/query ABI | audit MCP/operations |
| AGENTS | evidence, dual verdicts, report/status | gateway/evaluation/users |

An interface is compatible only when meaning, ordering, version, and failure semantics all match.

## Actual runtime/source walkthrough

### DATA → ML

[`graph_schema.py`](../../data_module/sentinel_data/representation/graph_schema.py) owns class and feature order. [`SentinelDataset`](../../ml/src/datasets/sentinel_dataset.py) rejects export-format, graph-schema, or artifact-hash mismatch before samples reach the trainer. The graph is `x[N,12]`, `edge_index[2,E]`, `edge_attr[E]`; token IDs are `[4,512]`; labels are `[10]` in locked order.

### ML → AGENTS/ZKML

[`api.py`](../../ml/src/inference/api.py) publishes probabilities by class and checkpoint SHA-256. `/fusion-embedding` publishes exactly 128 floats. AGENTS converts ML payloads into evidence and routing signals; ZKML treats the fusion vector as its only input.

### ZKML → Contracts

[`settings.json`](../../zkml/ezkl/settings.json) declares public `[1,128]` inputs and public `[1,10]` outputs. [`AuditRegistry.sol`](../../contracts/src/AuditRegistry.sol) assumes outputs begin at index 128 and checks all ten. EZKL hex instances are decoded little-endian. Scale 13 means fixed-point values use 8192.

### AGENTS → Gateway/evaluation

[`AuditState`](../../agents/src/orchestration/state.py) is the graph-internal contract; [`synthesizer.py`](../../agents/src/orchestration/nodes/synthesizer.py) creates the report; [`models.py`](../../agents/src/api/models.py) defines gateway job/API models; [`pipeline_metrics.py`](../../agents/src/eval/pipeline_metrics.py) and [`gates.py`](../../agents/src/eval/gates.py) interpret verdict/report fields.

## Interfaces, data shapes, and configuration

### Compatibility registry

| Invariant | Current value | Primary source |
|---|---|---|
| graph schema | v9 | `data_module/.../graph_schema.py::FEATURE_SCHEMA_VERSION` |
| node/edge dimensions | 12 / 14 node types / 12 edge types | same |
| class order/count | locked list / 10 | same |
| token windows | 4 × 512 | `ml/src/data_extraction/windowed_tokenizer.py` |
| fusion width | 128 | `ml/src/models/sentinel_model.py` |
| proxy | 128→64→32→10, 10,666 params, v2.0 | `zkml/src/distillation/proxy_model.py` |
| proof signals | 128 inputs + 10 outputs | `zkml/ezkl/settings.json` |
| contract offsets | 128 and 10 | `contracts/src/AuditRegistry.sol` |
| teacher identity | SHA-256 checkpoint file | `ml/src/inference/predictor.py` |
| stored proof identity | keccak256 proof bytes | `contracts/src/AuditRegistry.sol` |

The metadata mirror in [`handbook.toml`](_meta/handbook.toml) exists to detect drift; it does not replace executable sources.

## Failure modes and current limitations

- A matching tensor length with a reordered class list is silent semantic corruption.
- “model hash” is not self-enforcing: consumers must propagate the same validated value.
- The direct submitter currently reports an ML-returned hash but submits its original argument.
- Proof hash names hide a SHA-256 versus keccak256 inconsistency across response and storage.
- `verdict_provable` and ZK proxy outputs are different contracts despite adjacent “proof” terminology.
- Empty findings without `tool_status.ran=true` are ambiguous and prohibited by Rule 5C.
- HTTP/MCP fallback or mock responses must retain provenance so evaluation does not mix them with live evidence.

## Common change recipe

Before changing any listed invariant:

1. Identify every producer, consumer, artifact, cache, test, deployment, and document from this table.
2. Decide whether the change is backward-compatible, versioned parallel support, or a hard migration.
3. Add mismatch tests before changing the producer.
4. Regenerate dependent DATA/model/circuit/verifier artifacts in order.
5. Validate semantic order, hashes, encodings, and failure behavior—not shapes alone.
6. Update `handbook.toml`, inventory output, operations evidence, and status.

## Verification commands

```bash
python3 docs/handbook/tools/verify_handbook.py static
python3 docs/handbook/tools/verify_handbook.py inventory
export TMPDIR=/tmp TMP=/tmp TEMP=/tmp
ml/.venv/bin/python -m pytest zkml/tests/test_e2e_integration.py -q
cd contracts && forge test
```

Current module counts are in [current status](16_current_status.md).

## Optional deep references

- [DATA artifacts](04_data_artifacts.md)
- [ML inference](05_ml_model_inference.md)
- [ZKML](07_zkml.md)
- [Contracts](08_contracts.md)
- [Change playbooks](15_change_playbooks.md)

## Technical mastery layer

### Prerequisite knowledge

Know producer/consumer compatibility, semantic versioning, tensor/JSON/ABI schemas, hashes, and artifact rollout.

### Source map and reading order

Follow the seam guides in order: [T02](technical/02_data_representation_export.md) DATA→ML, [T03](technical/03_ml_model_inference_internals.md) ML output, [T05](technical/05_zkml_proof_lifecycle.md) ML→proxy/verifier, [T06](technical/06_contracts_storage_upgrades.md) verifier→registry, and [T07](technical/07_agents_orchestration_evidence.md) report evidence.

### Execution trace and worked example

One contract carries `[N,12]`, `[4,512]`, `[10]`, four `[128]` eyes, fusion `[128]`, proxy `[10]`, public `[138]`, Solidity fixed array `[10]`, and report dictionaries. Class order, model/circuit versions, and hashes are semantic parts of each shape.

### Implementation practice

Write a compatibility record before any breaking change: producer symbol, old/new shape, all consumers, migration/regeneration, mixed-version behavior, rollout, rollback, and tests. Exercise the aggregate in [L10](labs/10_end_to_end_capstone.md).

### Review and ownership check

Can you predict the full blast radius of adding one class or graph feature without searching after the change?
