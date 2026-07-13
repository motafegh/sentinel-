# 04 — DATA artifacts and the ML seam

**Read this when:** you need to understand graph/token files, labels, splits, export hashes, or `SentinelDataset`.

**Skip this if:** you only need to run an audit; read [runtime flows](02_runtime_flows.md) instead.

**Estimated reading time:** 12 minutes.

## 30-second summary

DATA turns each Solidity contract into a v9 PyG graph, four 512-token windows, a ten-class multi-label vector, metadata, and a split assignment. Export shards are integrity-bound by a manifest hash and loaded by `SentinelDataset` through format, graph-schema, and artifact-hash gates. The large export and split directories in this checkout are ignored local artifacts, not fresh-clone assets.

## Just-enough mental model

```text
Solidity + labels
     ├─ graph: x[N,12], edge_index[2,E], edge_attr[E]
     ├─ tokens: input_ids[4,512]
     ├─ target: y[10]
     └─ identity: contract_id + metadata + split
                         ↓
              sharded DATA export
                         ↓
             SentinelDataset / collate
                         ↓
                    ML trainer
```

The graph schema and class order are compatibility contracts. Reordering either without versioning invalidates datasets, caches, checkpoints, proxy expectations, and on-chain class indices.

## Actual runtime/source walkthrough

1. [`graph_schema.py`](../../data_module/sentinel_data/representation/graph_schema.py) — `data_module/sentinel_data/representation/graph_schema.py::FEATURE_SCHEMA_VERSION` owns v9 constants, vocabularies, features, and class order.
2. [`orchestrator.py`](../../data_module/sentinel_data/representation/orchestrator.py) — `::represent_source` writes graph and token representations and records extractor/schema versions.
3. [`export.py`](../../data_module/sentinel_data/export/export.py) and its writer/chunker siblings build Parquet metadata/labels, graph/token shards, shard indexes, and `manifest.json`.
4. [`sentinel_dataset.py`](../../ml/src/datasets/sentinel_dataset.py) — `::SentinelDataset` opens the export, rejects format/schema/hash mismatches, intersects a chosen split with shard membership, and returns samples.
5. [`collate.py`](../../ml/src/datasets/collate.py) — `::sentinel_collate_fn` batches PyG objects, `[4,512]` token tensors, labels, IDs, and confidence tiers.

The local `sentinel-v2-baseline-2026-06-12` manifest reports 22,356 labeled contracts, 21,523 with both representations, and five shards. The local v3 split manifest covers a different 22,493-contract universe. This is not a contradiction: export membership is filtered by successful representation, while split membership describes the selected labeled universe. These numbers are local artifact facts, not promises that a clean clone contains the data.

## Interfaces, data shapes, and configuration

### Locked v9 schema

| Contract | Current value |
|---|---:|
| `FEATURE_SCHEMA_VERSION` | `v9` |
| `NODE_FEATURE_DIM` | 12 |
| Node types | 14, IDs 0–13 |
| Edge types | 12, IDs 0–11 |
| Vulnerability classes | 10 |
| Token shape per contract | `[4, 512]` |

Feature order is `type_id`, `visibility`, `uses_block_globals`, `view`, `payable`, `complexity`, `loc`, `return_ignored`, `call_target_typed`, `has_loop`, `external_call_count`, `in_unchecked_block`.

Class order is `CallToUnknown`, `DenialOfService`, `ExternalBug`, `GasException`, `IntegerUO`, `MishandledException`, `Reentrancy`, `Timestamp`, `TransactionOrderDependence`, `UnusedReturn`.

`SentinelDataset.__getitem__` returns `(graph, tokens, y, contract_id, confidence_tier)`. The collator returns batched equivalents. Its attention mask is derived from non-padding token IDs.

### Artifact classes

| Artifact | Classification | Fresh clone? | Acquisition |
|---|---|---:|---|
| Source/config/schema | tracked | yes | Git |
| DATA exports and splits | ignored local | no | regenerate or obtain an approved immutable snapshot |
| Run 12 checkpoint | DVC-managed local in this checkout | no | obtain the tracked pointer/remote access separately, then `dvc pull` |
| ML caches | regenerated | no | inference/preprocessing recreates them |

See the complete matrix in [reference](17_reference.md#artifact-matrix).

## Failure modes and current limitations

- A graph schema mismatch is a hard compatibility failure, not a warning.
- A missing export/split is expected in a fresh clone; commands must name the prerequisite.
- Corrupt or modified shards fail artifact-hash verification.
- A contract may have a split row but no usable representation and is then absent from `SentinelDataset`.
- `REVERSE_CONTAINS` is runtime-only and must not be serialized as an on-disk edge.
- Some source docstrings still mention older 11-dimensional schemas; executable constants and assertions are authoritative.

## Common change recipe

For any feature, node, edge, or class change:

1. Change the canonical DATA schema and bump `FEATURE_SCHEMA_VERSION`.
2. Update extractor assertions and version registry.
3. Regenerate all graph/token exports and their hashes.
4. Regenerate or validate split/label compatibility.
5. Update ML preprocessing, model input assumptions, and tests.
6. Retrain; do not reuse a checkpoint trained against the prior schema.
7. Revisit proxy/circuit and contract class-index compatibility.
8. Update [`handbook.toml`](_meta/handbook.toml) and run static validation.

## Verification commands

```bash
export TMPDIR=/tmp TMP=/tmp TEMP=/tmp
python3 docs/handbook/tools/verify_handbook.py static
python3 docs/handbook/tools/verify_handbook.py inventory
data_module/.venv/bin/python -m pytest data_module/tests/representation -q   # smoke/targeted
data_module/.venv/bin/python -m pytest data_module/tests -q                  # module
```

The current measured module result is recorded only in [current status](16_current_status.md).

## Optional deep references

- [`graph_schema.py`](../../data_module/sentinel_data/representation/graph_schema.py) — `::CLASS_NAMES`, `::FEATURE_NAMES`, `::NODE_TYPES`, `::EDGE_TYPES`
- [`windowed_tokenizer.py`](../../ml/src/data_extraction/windowed_tokenizer.py) — `::WindowedTokenizer`
- [`export.py`](../../data_module/sentinel_data/export/export.py) — `::SentinelDatasetExport` and artifact-hash verification
- [DATA pipeline](03_data_pipeline.md)
- [Cross-module contracts](11_cross_module_contracts.md)

## Technical mastery layer

### Prerequisite knowledge

Know PyTorch Geometric `Data`, tensor shapes, Parquet, sharding, and integrity hashes.

### Source map and reading order

Read `graph_schema.py` constants, representation extractors, export writers, `export.py::SentinelDatasetExport`, `ml/src/datasets/sentinel_dataset.py::SentinelDataset`, then `collate.py::sentinel_collate_fn`. See [T02](technical/02_data_representation_export.md).

### Execution trace and worked example

A contract hash maps through a split to `shard_index`, then graph `[N,12]`, token `[4,512]`, label `[10]`, and confidence tier. Dataset construction checks format schema, graph schema, and artifact hash before serving any sample.

### Implementation practice

[L02](labs/02_export_dataset_seam.md) uses a copied export to trigger hash/schema failures safely. Schema changes require version bump and dependent artifact regeneration; manifest hashes are never hand-edited.

### Review and ownership check

Can you prove row alignment and explain why format, graph, and hash failures are three different errors?
