# T01 — DATA pipeline internals

## Learning outcome

You can explain what each of the ten DATA stages consumes and emits, identify the stage dispatcher, and change a stage without silently bypassing provenance, verification, leakage, or export gates.

## Prerequisites

Be comfortable with Python functions, JSON/YAML manifests, SHA-256, and train/validation/test splits. Read [DATA pipeline](../03_data_pipeline.md) first.

## Source map and reading order

1. `data_module/sentinel_data/cli.py::STAGES`, `::_handle_run`, and `::main` — dispatch and lifecycle order.
2. `data_module/sentinel_data/ingestion/ingest.py` and `::manifest.py` — connector output and acquisition evidence.
3. `data_module/sentinel_data/preprocessing/pipeline.py` — flatten, compile, normalize, segment, and deduplicate.
4. `data_module/sentinel_data/representation/orchestrator.py::represent_source` — graph/token materialization and cache rules.
5. `data_module/sentinel_data/labeling/merger.py`, `verification/gate.py`, and `splitting/splitters.py::apply_strategy`.
6. `registry/catalog.py::Catalog`, analysis modules, export writers, then `ingestion/freshness.py`.

## Entry point and complete call chain

`sentinel-data` enters `cli.py::main`, parses either a single stage or `run`, then `::_handle_run` walks `STAGES` and calls the `_STAGE_FN` handler. Freshness is a separate lifecycle command and is the tenth documented stage. The artifact chain is raw source plus manifest → preprocessed `.sol`/metadata → graph, tokens, and representation JSON → merged labels → verification report → deterministic split manifests → catalog/lineage record → analysis evidence → sharded export → freshness decision.

`label` is an important current gap: the libraries exist, but `cli.py::_run_label` still announces that CLI integration is not implemented. Do not describe the command as a completed automated label run.

## Important symbols and configuration

- `cli.py::STAGES` is the executable nine-stage `run` order; `freshness` is managed separately.
- `preprocessing/compiler.py` performs two-pass compiler handling; `deduplicator.py` applies similarity controls.
- `representation/orchestrator.py::_is_cached` binds cache reuse to schema and extractor versions.
- `splitting/splitters.py::{random_split,stratified_split,project_split,temporal_split}` are the four strategies.
- `registry/catalog.py::{DatasetVersion,SplitRecord,Catalog}` persist identity, hashes, lineage, retirement, and YAML mirrors.

## Annotated source excerpt

Source: `data_module/sentinel_data/cli.py::STAGES`

```python
STAGES: list[str] = [
    "ingest",
    "preprocess",
    "represent",
    "label",
    "verify",
    "split",
    "register",
    "analyze",
    "export",
]
```

This list is operational order, not proof that every handler is complete. The handbook adds freshness because it decides whether the acquisition lifecycle must restart.

## Worked example

Suppose connector `manual` acquires `Vault.sol`. Ingest computes its content hash and records source metadata. Preprocess produces a normalized, compiler-resolved contract and metadata keyed by SHA-256. Representation sees that metadata, builds v9 graph and `[4,512]` token artifacts, and writes a cache record. Labels become a ten-position multi-hot vector in canonical class order. Verification may corroborate or reject evidence. Split assignment uses contract identity and selected strategy, then leakage auditing checks related content across partitions. Export writes shards and a manifest whose artifact hash covers data files. `Catalog` registers the resulting version and lineage. Freshness later compares upstream state and recommends or triggers reacquisition.

## Success trace

The stage returns zero; expected companion files exist; manifest counts reconcile; verification and leakage reports are explicit; the catalog points to the same artifact hash that export computed; rerunning representation without `--force` reports cache hits.

## Failure trace

A missing preprocessed directory makes `represent_source` raise `FileNotFoundError` with the preceding command. A compiler or extraction failure increments failure counters rather than manufacturing a graph. A hash or schema mismatch must stop loading downstream. A label CLI message saying “NOT IMPLEMENTED” is a gap, not success.

## Design reasoning and rejected alternatives

Content hashes make identity independent of filenames. Manifests keep provenance beside the materialized artifact. Project/temporal splits exist because random-only splits can leak family or time information. The project rejects “best effort” downstream loading: silently accepting a mismatched schema or artifact would create irreproducible model results.

## Safe change walkthrough

To add a preprocessing field, first add a fixture assertion, update the producing transform and metadata schema, run preprocessing tests, then representation byte-identity/schema tests, leakage tests, export tests, and finally `SentinelDataset` gates. If bytes or schema change intentionally, bump the appropriate version and regenerate dependent artifacts; never edit an existing manifest hash by hand.

## Guided lab

Complete [L01 — DATA fixture and representation](../labs/01_data_fixture_representation.md), then [L02 — export and dataset seam](../labs/02_export_dataset_seam.md).

## Tests and expected results

```bash
TMPDIR=/tmp TMP=/tmp TEMP=/tmp data_module/.venv/bin/python -m pytest \
  data_module/tests/test_skeleton.py \
  data_module/tests/test_representation/test_orchestrator.py -q
```

Expected: the selected tests pass; dry-run lists the executable stages; representation tests prove cache, force, output-triplet, and metadata behavior. See [current status](../16_current_status.md) for volatile suite counts.

## Review questions

Why is freshness separate from `STAGES`? Where is leakage checked? Which identities are filenames versus hashes? What must be regenerated after a graph schema change?

## Ownership checklist

- I can locate every stage handler and its artifact.
- I distinguish implemented libraries from CLI integration.
- I can trace one contract’s hash and lineage.
- I stop on schema/hash/verification failures instead of weakening a gate.
