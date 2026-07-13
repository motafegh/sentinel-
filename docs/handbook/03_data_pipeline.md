# 03 — DATA pipeline

**Read this when:** you need to acquire, clean, label, verify, split, register, analyze, export, or freshness-check Solidity data.

**Skip this if:** your input is already an approved immutable export and you only need [artifact shapes](04_data_artifacts.md).

**Estimated reading time:** 15 minutes.

## 30-second summary

DATA is a ten-stage lifecycle, not a folder of ML-ready files: ingest, preprocess, represent, label, verify, split, register, analyze, export, and freshness. Each stage has a distinct trust job—provenance, compilation, feature extraction, ground truth, corroboration, leakage prevention, lineage, diagnostics, immutable packaging, or upstream-change detection. The CLI exposes all ten names, but its `label` command is currently a placeholder; labeling components exist and must be invoked through their source workflows until that adapter is completed.

## Just-enough mental model

```text
upstream sources
  → ingest → preprocess → represent → label → verify
  → split → register → analyze → export
          freshness watches upstream/tool drift ─┘
```

No later stage repairs an earlier trust failure. A well-trained model cannot compensate for leaked splits, a manifest cannot make wrong labels true, and a freshness report cannot replace pinned provenance.

## Actual runtime/source walkthrough

The stage registry lives in [`cli.py`](../../data_module/sentinel_data/cli.py) — `data_module/sentinel_data/cli.py::STAGES` and `::_STAGE_FN`.

| Stage | Source owner | What actually happens |
|---|---|---|
| 1. ingest | [`ingest.py`](../../data_module/sentinel_data/ingestion/ingest.py) `::ingest_all` | pulls enabled sources and writes SHA-256 provenance manifests |
| 2. preprocess | [`preprocess.py`](../../data_module/sentinel_data/preprocessing/preprocess.py) `::preprocess_all` | flatten, compile, deduplicate, normalize, segment, and compiler-bucket |
| 3. represent | [`orchestrator.py`](../../data_module/sentinel_data/representation/orchestrator.py) `::represent_source` | emits v9 graphs and windowed GraphCodeBERT tokens with cache identities |
| 4. label | [`merger.py`](../../data_module/sentinel_data/labeling/merger.py) and [`gate.py`](../../data_module/sentinel_data/labeling/gate.py) | applies source crosswalks, merges multi-label truth, and gates label quality |
| 5. verify | [`semantic_checker.py`](../../data_module/sentinel_data/verification/semantic_checker.py), tool runners, and `verification/gate.py` | checks AST semantics and corroborates labels with Slither/Aderyn evidence |
| 6. split | [`splitters.py`](../../data_module/sentinel_data/splitting/splitters.py) plus leakage/dedup/cap modules | builds deterministic train/validation/test sets and audits overlap |
| 7. register | [`catalog.py`](../../data_module/sentinel_data/registry/catalog.py) and [`lineage_tracker.py`](../../data_module/sentinel_data/registry/lineage_tracker.py) | records dataset versions, parentage, artifacts, and metrics in SQLite/YAML |
| 8. analyze | [`feature_dist.py`](../../data_module/sentinel_data/analysis/feature_dist.py), co-occurrence, drift, overlap, and probe modules | measures representation/label balance and change risk |
| 9. export | [`export.py`](../../data_module/sentinel_data/export/export.py) `::export_dataset` | creates Parquet tables, sharded tensors, indexes, manifest, and artifact hash |
| 10. freshness | [`freshness.py`](../../data_module/sentinel_data/ingestion/freshness.py) `::run_freshness_check` | compares pinned upstream/tool versions with current observations |

`sentinel-data run` walks the nine production stages in `STAGES`; `freshness` is a separate utility command and is also registered in `_STAGE_FN`. This is why the lifecycle has ten stages while the sequential build list has nine.

### Verification and lineage

Verification is not label generation. It produces evidence about whether labels and negatives are credible. Tool absence or compilation failure must be explicit in reports; an empty finding list must mean “ran and found none,” never “did not run.” Registration then binds a dataset identity to source manifests, transforms, artifacts, and parent versions so comparisons can explain what changed.

### Benchmarks

`data_module/benchmarks/` contains tracked benchmark definitions and contamination checks. Benchmark results are evidence about a specific artifact version; they are not interchangeable with the full corpus or a test-suite count.

## Interfaces, data shapes, and configuration

The main interface is:

```bash
cd data_module
.venv/bin/python -m sentinel_data.cli <stage> --config config.yaml
```

Use `--dry-run` where supported before writes. Source enablement, paths, preprocessing thresholds, split policies, and tool settings live in [`config.yaml`](../../data_module/config.yaml); never copy environment secrets into it.

Key boundaries:

- ingest output: immutable source files plus source manifest and content hashes;
- preprocess output: compiler-compatible normalized contracts plus dropped/error records;
- representation output: one graph/token identity per contract, versioned by schema and extractor;
- label output: ten-class multi-label records plus confidence/provenance;
- split output: contract IDs assigned deterministically with leakage evidence;
- export output: manifest, labels/metadata Parquet, graph/token shards, indexes, and an aggregate hash.

## Failure modes and current limitations

- `cli.py::_run_label` currently prints `NOT IMPLEMENTED`; the stage’s library components exist, but the one-command CLI seam is incomplete.
- Stage logs that say “skip” must be reviewed: some current paths predate Rule 5C’s structured-degradation standard.
- External tools and multiple Solidity compiler versions are operational prerequisites, not Python-only dependencies.
- Source deduplication, leakage auditing, and class caps are policy. Changing them requires before/after measurements.
- DATA exports, split manifests, and catalogs under `data_module/data/` are ignored local artifacts in this checkout.
- Freshness detects change; it does not automatically approve or ingest that change.

## Common change recipe

To add a source:

1. Add a pinned source configuration and ingestion adapter.
2. Record upstream identity and content hashes.
3. Establish label crosswalk semantics and negative policy.
4. Run preprocessing and representation on a small sample.
5. Run semantic/tool verification and contamination checks.
6. Rebuild splits; prove no family/duplicate leakage.
7. Register a new dataset version, analyze deltas, and export immutably.
8. Evaluate downstream ML changes before promotion.

For schema changes, follow the wider blast-radius playbook in [change playbooks](15_change_playbooks.md).

## Verification commands

```bash
export TMPDIR=/tmp TMP=/tmp TEMP=/tmp
data_module/.venv/bin/python -m sentinel_data.cli --help                 # smoke
data_module/.venv/bin/python -m sentinel_data.cli freshness --help       # smoke
data_module/.venv/bin/python -m pytest data_module/tests -q              # module
python3 docs/handbook/tools/verify_handbook.py inventory                 # documentation inventory
```

Commands that pull sources, invoke analyzers, or write a complete export are live/data-build operations. Current suite counts are only in [current status](16_current_status.md).

## Optional deep references

- [`data_module/sentinel_data`](../../data_module/sentinel_data) — executable pipeline packages
- [`data_module/benchmarks`](../../data_module/benchmarks) — benchmark manifests and contamination evidence
- [DATA artifacts](04_data_artifacts.md)
- [ML training and quality](06_ml_training_quality.md)
- [Evaluation](13_evaluation.md)

## Technical mastery layer

### Prerequisite knowledge

Know manifests, content hashes, compiler resolution, deduplication, stratified/project/temporal splits, and dataset leakage.

### Source map and reading order

Start at `data_module/sentinel_data/cli.py::{STAGES,_handle_run}` and follow each handler into ingestion, preprocessing, representation, labeling, verification, splitting, registry, analysis, export, then freshness. [T01](technical/01_data_pipeline_internals.md) gives the complete call/artifact chain.

### Execution trace and worked example

One acquired contract gains a SHA-256 manifest, normalized/compiler-resolved source, v9 graph/token representations, ten-class labels, verification evidence, deterministic split membership, catalog lineage, analysis reports, a sharded export position, and a freshness decision. The current label handler remains an explicit CLI integration gap.

### Implementation practice

Use [L01](labs/01_data_fixture_representation.md) to prove representation/cache behavior. A stage change begins with a fixture and ends only after downstream hashes/schema/leakage/export consumers are checked.

### Review and ownership check

Can you name every stage’s input/output and distinguish “library implemented” from “CLI path operational”?
