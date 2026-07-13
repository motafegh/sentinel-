# T02 — Graph, token, export, and dataset seam

## Learning outcome

You can trace one represented contract into sharded export and through all three `SentinelDataset` admission gates, including concrete graph, token, label, split, and hash shapes.

## Prerequisites

Read [DATA artifacts](../04_data_artifacts.md). Know basic PyTorch tensors, PyTorch Geometric graphs, Parquet, and content hashes.

## Source map and reading order

1. `data_module/sentinel_data/representation/graph_schema.py` — v9 constants and canonical class order.
2. `representation/graph_extractor.py` and `::tokenizer.py` — graph and `[4,512]` windows.
3. `representation/orchestrator.py::_extract_one` — three-file representation output.
4. `export/{graph_writer,token_writer,label_writer,metadata_writer,chunker}.py`.
5. `export/export.py::SentinelDatasetExport` — manifest parsing, splits, hashes.
6. `ml/src/datasets/sentinel_dataset.py::SentinelDataset`, then `collate.py::sentinel_collate_fn`.

## Entry point and complete call chain

`represent_source` locates each `.meta.json`/`.sol` pair and delegates graph and token extraction. Export writers align records by contract hash, shard graphs/tokens, write labels and metadata, and construct `manifest.json` plus `shard_index`. `SentinelDataset.__init__` parses the export, checks format schema, graph schema, and artifact hash, loads labels, intersects the requested split with represented IDs, and prepares node-count metadata. `__getitem__` resolves a shard/position, reconstructs attention masks from token padding, and returns graph, token dict, ten-label tensor, contract ID, and optional confidence tier.

## Important symbols and configuration

- `FEATURE_SCHEMA_VERSION="v9"`, node feature dimension 12, 14 node types, 12 edge types.
- `CLASS_NAMES` fixes the ten-label order; numerical positions are an interface.
- Export format schema and graph schema are separate compatibility gates.
- `manifest.splits` contains ordered contract hashes; `shard_index` maps each hash to shard and position.
- Artifact hashing excludes `manifest.json` and `.hash_cache.json`; the latter accelerates unchanged rechecks.

## Annotated source excerpt

Source: `ml/src/datasets/sentinel_dataset.py::SentinelDataset.__init__`

```python
if self.export.manifest.schema_version != _EXPECTED_FORMAT_SCHEMA:
    raise ValueError("Export format schema version mismatch")
if self.export.manifest.graph_schema_version != expected_schema:
    raise ValueError("Graph schema version mismatch")
if not self.export.verify_artifact_hash():
    raise ValueError("Export artifact hash mismatch")
```

The three independent checks answer different questions: can this loader parse the container, can this model interpret the graph, and are the bytes the publisher committed to still present?

## Worked example

For contract hash `abc…`, `shard_index["abc…"]={"shard":2,"pos_in_shard":7,"num_nodes":43}`. Split `train` contains the hash. Graph shard 2 yields a PyG `Data` with `x=[43,12]`, `edge_index=[2,E]`, and edge types. Token shard 2 position 7 yields `input_ids=[4,512]`; the loader derives `attention_mask=[4,512]`. Labels Parquet yields `y=[10]`. A batch of three contracts collates graphs into one PyG batch, tokens to `[3,4,512]`, and labels to `[3,10]`.

## Success trace

Manifest parsing succeeds, the warm or cold hash path equals `artifact_hash`, the split filters only represented IDs, shard positions resolve, and collate preserves alignment across graph, token, label, ID, and confidence tier.

## Failure trace

Changing a shard byte causes the hash gate to raise. Changing `graph_schema_version` causes a schema error before training. A split ID without representation is intentionally filtered; unexpectedly large filtering indicates upstream representation loss and must be investigated. A missing manifest fails immediately.

## Design reasoning and rejected alternatives

Sharding avoids one-file-per-sample overhead while the index preserves random access. Reconstructing masks from a fixed padding ID avoids storing duplicate data. Separate format/graph versions prevent a container change from being confused with a semantic feature change. Loading corrupt bytes “with a warning” was rejected because training evidence would no longer identify its dataset.

## Safe change walkthrough

For a new graph feature, update the canonical ML schema, keep DATA thin adapters byte-identical, bump the graph version, regenerate representations and exports, update checkpoint/model expectations, and run adapter, schema, export, dataset, and model tests. For an export-only field, update manifest parsing/writers and format schema without pretending the graph semantics changed.

## Guided lab

Complete [L02 — export, hash, split, and dataset seam](../labs/02_export_dataset_seam.md).

## Tests and expected results

```bash
TMPDIR=/tmp TMP=/tmp TEMP=/tmp ml/.venv/bin/python -m pytest \
  ml/tests/test_sentinel_dataset.py -q
```

Expected with the local export present: normal loading and deliberate schema/hash gate tests pass. A fresh clone must report the ignored-local export as unavailable rather than silently substituting data.

## Review questions

What does each gate protect? Why is class order a compatibility surface? How does an ID move from split manifest to shard position? Which artifacts must change after one node feature changes?

## Ownership checklist

- I can state every per-sample and batched shape.
- I can distinguish format schema from graph schema.
- I can prove graph/token/label alignment by contract hash.
- I know which export artifacts are absent from a fresh clone.
