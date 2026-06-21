# SENTINEL Data Module (`sentinel-data`)

A standalone Python package (`sentinel-data`, Poetry-managed, Python ≥3.12,<3.13) that implements the full data pipeline for the SENTINEL smart contract vulnerability classifier — from raw source ingestion through labeled, verified, split, and exported dataset artifacts.

**Purpose**: Produce clean, leak-free, verifiably-labeled training data for the ML module (`ml/`). The data module owns all 9 pipeline stages; the ML module consumes the exported artifacts via `SentinelDataset`.

---

## Table of Contents

1. [Pipeline Architecture](#1-pipeline-architecture)
2. [Directory Structure](#2-directory-structure)
3. [Sources](#3-sources)
4. [Stage 1: Ingestion](#4-stage-1-ingestion)
5. [Stage 2: Preprocessing](#5-stage-2-preprocessing)
6. [Stage 3: Labeling](#6-stage-3-labeling)
7. [Stage 4: Representation](#7-stage-4-representation)
8. [Stage 5: Verification](#8-stage-5-verification)
9. [Stage 6: Splitting](#9-stage-6-splitting)
10. [Stage 7: Registry](#10-stage-7-registry)
11. [Stage 8: Analysis](#11-stage-8-analysis)
12. [Stage 9: Export](#12-stage-9-export)
13. [Configuration](#13-configuration)
14. [CLI Usage](#14-cli-usage)
15. [Dependency Map](#15-dependency-map)

---

## 1. Pipeline Architecture

The data module implements a 9-stage DVC-tracked pipeline (defined in `dvc.yaml`):

```
ingest → preprocess → label → represent → verify → split → register → analyze → export
```

Each stage reads from `data/<input>/` and writes to `data/<output>/`. The pipeline is orchestrated by the CLI (`sentinel-data`) and tracked by DVC for reproducibility.

**Key invariants**:
- Schema changes require bumping `FEATURE_SCHEMA_VERSION` (currently `"v9"`) and rebuilding all caches
- The 10-class taxonomy is LOCKED — changing class order invalidates existing checkpoints
- All graphs use SHA-256 content addressing (not MD5)
- The representation cache is keyed on `(sha256, schema_version, extractor_version)`

**Current corpus**: 22,493 contracts across 5 active sources, 0% leakage, splits 18,596/1,983/1,914 (train/val/test).

---

## 2. Directory Structure

```
data_module/
├── sentinel_data/               # Main Python package
│   ├── __init__.py              # Package docstring, pipeline architecture
│   ├── cli.py                   # CLI entry point (sentinel-data)
│   ├── ingestion/               # Stage 1: source ingestion
│   │   ├── ingest.py            # Orchestrates connector + manifest
│   │   ├── manifest.py          # Per-source pull record with SHA-256 verification
│   │   ├── label_folderize.py   # Label-aware folderization (DIVE, SmartBugs)
│   │   ├── freshness.py         # Source pin freshness checker
│   │   └── connectors/          # Source connectors
│   │       ├── base.py          # BaseConnector, SourceConfig, PullResult
│   │       ├── git_connector.py # Git clone + checkout
│   │       ├── manual_connector.py # Manual download / symlink
│   │       ├── huggingface_connector.py # HuggingFace datasets
│   │       ├── zenodo_connector.py # Zenodo records
│   │       └── etherscan_connector.py # Etherscan API (DISL)
│   ├── preprocessing/           # Stage 2: source normalization
│   │   ├── pipeline.py          # PreprocessingPipeline (5-step orchestration)
│   │   ├── compiler.py          # Two-pass solc compilation
│   │   ├── flattener.py         # Import flattening
│   │   ├── deduplicator.py      # Content-based dedup (SHA-256)
│   │   ├── normalizer.py        # Whitespace/format normalization
│   │   ├── segmenter.py         # Contract segmentation + version bucketing
│   │   └── parallel.py          # Multiprocessing variant
│   ├── labeling/                # Stage 3: label assignment
│   │   ├── schema/              # Canonical 10-class taxonomy
│   │   │   ├── __init__.py      # class_names(), class_index()
│   │   │   └── taxonomy.yaml    # Class definitions + descriptions
│   │   ├── parsers/             # Per-source label parsers
│   │   │   ├── solidifi.py      # SolidiFI ground-truth parser
│   │   │   ├── dive.py          # DIVE CSV label parser
│   │   │   └── smartbugs_curated.py # SmartBugs Curated parser
│   │   ├── crosswalks/          # Per-source label mapping YAMLs
│   │   ├── merger.py            # Multi-source label merger (tier precedence)
│   │   └── gate.py              # Go/No-Go minimum-viable-corpus gate
│   ├── representation/          # Stage 4: graph + token representation
│   │   ├── graph_schema.py      # CANONICAL schema (NODE_TYPES, EDGE_TYPES, FEATURE_NAMES)
│   │   ├── graph_extractor.py   # Solidity → PyG graph extraction (v8 schema)
│   │   ├── orchestrator.py      # Represent orchestrator (reads Stage 1, writes .pt)
│   │   ├── tokenizer.py         # Thin adapter over ml/ windowed tokenizer
│   │   ├── cache_manager.py     # Content-addressed representation cache
│   │   ├── versioner.py         # Schema + extractor version registry
│   │   ├── cfg_builder.py       # Standalone CFG artifact (opt-in)
│   │   ├── call_graph.py        # DEFERRED to v3.1
│   │   ├── pdg_builder.py       # DEFERRED to v3.1
│   │   └── opcode_extractor.py  # DEFERRED to v3.1
│   ├── verification/            # Stage 5: label verification
│   │   ├── gate.py              # Verification gate (VERIFIED/PROVISIONAL/FAIL)
│   │   ├── class_auditor.py     # Per-class stats + co-occurrence matrix
│   │   ├── semantic_checker.py  # Graph-feature-based label verification
│   │   ├── tool_validator.py    # Slither agreement rate per class
│   │   ├── fp_estimator.py      # Empirical false-positive rate (stratified)
│   │   ├── negative_checker.py  # NonVulnerable contamination check
│   │   ├── slither_runner.py    # Content-addressed Slither runner
│   │   ├── aeryn_runner.py      # Aderyn tool runner
│   │   └── patterns/            # Vulnerability pattern definitions
│   ├── splitting/               # Stage 6: train/val/test splits
│   │   ├── splitters.py         # 4 splitter strategies (random, stratified, project, temporal)
│   │   ├── dedup_enforcer.py    # Reassign straddling dedup groups
│   │   ├── leakage_auditor.py   # Post-split text similarity check
│   │   └── nonvulnerable_cap.py # NonVulnerable : positive ratio cap (3:1)
│   ├── registry/                # Stage 7: dataset versioning
│   │   ├── catalog.py           # SQLite + YAML mirror catalog
│   │   ├── lineage_tracker.py   # DAG lineage tracking
│   │   └── dataset_diff.py      # Dataset version diffing
│   ├── analysis/                # Stage 8: dataset analysis
│   │   ├── drift_monitor.py     # KS-test feature + label drift detection
│   │   ├── overlap_detector.py  # Pairwise Jaccard similarity between sources
│   │   ├── cooccurrence.py      # Class co-occurrence analysis
│   │   ├── feature_dist.py      # Feature distribution analysis
│   │   ├── balance_viz.py       # Class balance visualization
│   │   └── probe_dataset.py     # Dataset probing utilities
│   └── export/                  # Stage 9: consumer-facing export
│       ├── chunker.py           # Orchestrates all 4 writers → manifest.json
│       ├── export.py            # SentinelDatasetExport (read-only view)
│       ├── graph_writer.py      # Graph .pt sharding
│       ├── token_writer.py      # Token .pt sharding
│       ├── label_writer.py      # labels.parquet writer
│       ├── metadata_writer.py   # metadata.parquet writer
│       └── format_schema/       # Export format definitions
├── config.yaml                  # Single source of truth for all sources + pipeline settings
├── pyproject.toml               # Poetry build config (sentinel-data package)
├── dvc.yaml                     # 9-stage DVC DAG
├── benchmarks/                  # Benchmark datasets
├── tests/                       # Test suite
└── docs/                        # Documentation
```

---

## 3. Sources

Sources are defined in `config.yaml` under `sources_critical_path` and `sources_additive`. Each source has a confidence tier (T0–T4) that governs label precedence and verification thresholds.

### Critical-Path Sources

| Source | Tier | Connector | Contracts | Description |
|--------|------|-----------|-----------|-------------|
| **SolidiFI** | T0 (Gold) | git | 9,369 | ISSTA 2020, injected bugs with 100% ground-truth certainty |
| **DIVE** | T1 (Gold) | manual | 22,330 | Nature Sci. Data 2025, 8 DASP classes, multi-label, peer-reviewed |
| **Web3Bugs** | T1 (Gold) | git | ~3,500 | Contest-verified bugs from Code4rena/Sherlock/Immunefi |
| **SmartBugs Curated** | T3 (Structural) | manual | 143 | Hand-labeled contracts (DASP → 10 classes direct) |
| **DISL** | T4 (Bronze) | etherscan | 514,506 | Unlabeled Solidity files; NonVulnerable pool only, 3:1 cap |

### Deferred Sources

| Source | Status | Reason |
|--------|--------|--------|
| DeFiHackLabs | Deferred to v2.1 | Foundry project — requires forge-std clone |
| FORGE | Deferred to v2.2 | 50-entry agreement test required |
| Bastet | v2.1 additive | Replaces Code4rena scraper (legal risk) |
| ScrawlD | v2.1 additive | 5-tool majority voting |
| REKT | v2.1 additive | Verified real exploits |

---

## 4. Stage 1: Ingestion

**Entry point**: `sentinel_data.ingestion.ingest`

**What it does**:
1. Reads `config.yaml` to find enabled sources
2. For each source, selects the appropriate connector (git, manual, huggingface, etc.)
3. Pulls the source to `data/raw/<source>/repo/`
4. Writes `ingestion_manifest.json` with per-file SHA-256 fingerprints
5. For sources with separate label CSVs (DIVE, SmartBugs), runs label-aware folderization

**Key files**:
- `ingestion/ingest.py` — `ingest_source()`, `ingest_all()`
- `ingestion/manifest.py` — `IngestionManifest`, `FileRecord`, `verify()`
- `ingestion/label_folderize.py` — `folderize_by_labels()` for DIVE/SmartBugs
- `ingestion/freshness.py` — `run_freshness_check()` for source pin staleness

**Connectors** (`ingestion/connectors/`):
- `base.py` — `BaseConnector` (ABC), `SourceConfig`, `PullResult`
- `git_connector.py` — Clone + checkout to pinned commit
- `manual_connector.py` — Symlink/copy from staging path
- `huggingface_connector.py` — HuggingFace datasets API
- `zenodo_connector.py` — Zenodo record download
- `etherscan_connector.py` — Etherscan API (DISL)

**Output**: `data/raw/<source>/repo/` with `ingestion_manifest.json`

---

## 5. Stage 2: Preprocessing

**Entry point**: `sentinel_data.preprocessing.preprocess`

**What it does**: Runs a 5-step pipeline on each `.sol` file:

1. **Flatten** (`flattener.py`) — Resolve imports, produce single-file output
2. **Compile** (`compiler.py`) — Two-pass solc compilation (exact version, then nearest satisfying)
3. **Dedup** (`deduplicator.py`) — Content-based dedup via SHA-256 hash
4. **Normalize** (`normalizer.py`) — Whitespace/format normalization
5. **Segment + Bucket** (`segmenter.py`) — Contract segmentation, version bucketing (legacy/transitional/modern)

**Key files**:
- `preprocessing/preprocess.py` — `preprocess_source()`, supports `--sample N` and `--retry-failed`
- `preprocessing/pipeline.py` — `PreprocessingPipeline`, `ContractMeta` (sidecar), `PipelineResult`
- `preprocessing/compiler.py` — `compile_contract()` → `CompileResult` (two-pass solc)
- `preprocessing/parallel.py` — Multiprocessing variant for batch processing

**Output**: `data/preprocessed/<source>/<sha256>.{sol, meta.json}` + `dropped.csv` for failures

**Sidecar schema** (`ContractMeta`):
```json
{
  "sha256": "...",
  "source_name": "solidifi",
  "original_path": "repo/buggy_contracts/reentrancy/x.sol",
  "pragma": "^0.8.0",
  "solc_version": "0.8.19",
  "compile_status": "ok",
  "flatten_status": "flattened",
  "dedup_group_id": "abc123",
  "version_bucket": "modern",
  "has_unchecked_block": false
}
```

---

## 6. Stage 3: Labeling

**Entry point**: `sentinel_data.labeling.merger`

**What it does**: Assigns vulnerability labels to contracts using per-source parsers and crosswalks, then merges multi-source labels with tier-precedence conflict resolution.

**Key files**:
- `labeling/schema/__init__.py` — `class_names()` (locked 10-class order), `class_index()`
- `labeling/schema/taxonomy.yaml` — Canonical class definitions
- `labeling/parsers/solidifi.py` — SolidiFI ground-truth parser (T0)
- `labeling/parsers/dive.py` — DIVE CSV label parser (T1)
- `labeling/parsers/smartbugs_curated.py` — SmartBugs Curated parser (T3)
- `labeling/crosswalks/` — Per-source label mapping YAMLs (10 classes)
- `labeling/merger.py` — Multi-source merge with tier precedence (T0 > T1 > T2 > T3 > T4)
- `labeling/gate.py` — Go/No-Go minimum-viable-corpus gate

**Merge rules** (from `merger.py`):
- Tier precedence: T0 > T1 > T2 > T3 > T4 (lower index = higher confidence)
- Within a tier: positive wins over negative
- DoS+Reentrancy co-occurrence from a single T3/T4 source is flagged as suspect noise (threshold: 50%)

**Output**: `data/labels/<source>/<sha256>.labels.json` + `data/labels/merged/<sha256>.labels.json`

**Label schema**:
```json
{
  "sha256": "...",
  "sources": ["solidifi"],
  "classes": {
    "Reentrancy": {"value": 1, "tier": "T0", "source": "solidifi"},
    "DenialOfService": {"value": 0, "tier": null, "source": "solidifi"}
  },
  "n_pos": 1,
  "flags": []
}
```

---

## 7. Stage 4: Representation

**Entry point**: `sentinel_data.representation.orchestrator`

**What it does**: Converts preprocessed `.sol` files into PyG graph tensors and windowed GraphCodeBERT tokens.

**Key files**:
- `representation/graph_schema.py` — **CANONICAL** schema: `NODE_TYPES` (14), `EDGE_TYPES` (12), `FEATURE_NAMES` (12 dims), `CLASS_NAMES` (10)
- `representation/graph_extractor.py` — `extract_contract_graph()` → PyG `Data` object
- `representation/orchestrator.py` — `represent_source()` (reads Stage 1, writes .pt)
- `representation/tokenizer.py` — Thin adapter over `ml/src/data_extraction/windowed_tokenizer.py`
- `representation/cache_manager.py` — Content-addressed cache (sha256 + schema + extractor versions)
- `representation/versioner.py` — Schema + extractor version registry

**Graph shape contract** (v9 schema):
```
graph.x             [N, 12]  float32  node feature matrix (NODE_FEATURE_DIM=12)
graph.edge_index    [2, E]   int64    edge connectivity (COO format)
graph.edge_attr     [E]      int64    edge type IDs 0–11
graph.node_metadata list     of dicts (one per node: name, type, source_lines)
```

**Feature dimensions** (v9):
| Dim | Name | Description |
|-----|------|-------------|
| 0 | type_id | Node type / 13.0 (normalized) |
| 1 | visibility | 0=public, 0.5=internal, 1=private |
| 2 | uses_block_globals | block.timestamp/number/now detected |
| 3 | view | Function is view/pure |
| 4 | payable | Function is payable |
| 5 | complexity | log1p(CFG block count) / log1p(1000) |
| 6 | loc | log1p(line count) / log1p(1000) |
| 7 | return_ignored | External call return value discarded |
| 8 | call_target_typed | All calls via typed interfaces |
| 9 | has_loop | Function contains loop construct |
| 10 | external_call_count | log1p(count) / log1p(20) |
| 11 | in_unchecked_block | Inside unchecked{} (0.8+) or pre-0.8 era |

**Edge types** (12):
| ID | Name | Description |
|----|------|-------------|
| 0 | CALLS | Function calls |
| 1 | READS | State variable reads |
| 2 | WRITES | State variable writes |
| 3 | EMITS | Event emissions |
| 4 | INHERITS | Contract inheritance |
| 5 | CONTAINS | Contract contains function |
| 6 | CONTROL_FLOW | CFG edges |
| 7 | REVERSE_CONTAINS | Runtime-only (never on disk) |
| 8 | CALL_ENTRY | ICFG-Lite cross-function call |
| 9 | RETURN_TO | ICFG-Lite cross-function return |
| 10 | DEF_USE | Intra-function data-flow |
| 11 | EXTERNAL_CALL | Cross-contract call site marker |

**Cache key**: `(sha256, FEATURE_SCHEMA_VERSION, EXTRACTOR_VERSION)` — change any → cache invalidation.

**Output**: `data/representations/<source>/<sha256>.{pt, tokens.pt, rep.json}`

---

## 8. Stage 5: Verification

**Entry point**: `sentinel_data.verification.gate`

**What it does**: Produces per-class VERIFIED / PROVISIONAL / BEST-EFFORT / FAIL verdicts from multiple verification signals.

**Key files**:
- `verification/gate.py` — `run_gate()` → `GateResult` with per-class `ClassVerdict`
- `verification/class_auditor.py` — Per-class stats + 10×10 co-occurrence matrix
- `verification/semantic_checker.py` — Graph-feature-based label verification
- `verification/tool_validator.py` — Slither agreement rate per class
- `verification/fp_estimator.py` — Empirical false-positive rate (stratified by source+tier)
- `verification/negative_checker.py` — NonVulnerable contamination check
- `verification/slither_runner.py` — Content-addressed Slither runner with cache

**Gate rules** (from `gate.py`):
| Condition | Verdict |
|-----------|---------|
| FP rate > 30% | FAIL |
| T0 injection-verified, no semantic failures | VERIFIED |
| Semantic pass rate > 90%, no co-occurrence flag | VERIFIED |
| Semantic pass rate 60–90% | PROVISIONAL |
| Semantic pass rate 30–60% | BEST-EFFORT |
| Semantic pass rate < 30% | FAIL |
| Co-occurrence flag on high-noise source | BEST-EFFORT |

**Semantic check coverage** (from `semantic_checker.py`):
| Class | Feature Used |
|-------|-------------|
| Reentrancy | `has_cei_path` (EXTERNAL_CALL edge + WRITE reachable) |
| Timestamp | `feat[2]` uses_block_globals |
| IntegerUO | `feat[11]` unchecked_block OR pre-0.8 pragma |
| UnusedReturn | `feat[7]` return_ignored |
| MishandledException | `feat[7]` return_ignored (low-level call) |
| CallToUnknown | EXTERNAL_CALL edge (type 11) present |
| ExternalBug | EXTERNAL_CALL edge (weaker signal) |
| DenialOfService | NOT_EXTRACTABLE (no v9 feature) |
| GasException | NOT_EXTRACTABLE (no v9 feature) |
| TransactionOrderDependence | NOT_EXTRACTABLE (no v9 feature) |

**Output**: `data/analysis/<run_id>/verification_report.md`

---

## 9. Stage 6: Splitting

**Entry point**: `sentinel_data.splitting.splitters`

**What it does**: Produces leak-free, deterministic train/val/test splits using a two-pass approach: splitter → dedup_enforcer.

**Key files**:
- `splitting/splitters.py` — 4 strategies: `random_split`, `stratified_split`, `project_split`, `temporal_split`
- `splitting/dedup_enforcer.py` — Reassign straddling dedup groups (majority-wins rule)
- `splitting/leakage_auditor.py` — Post-split text shingle similarity check (safety net)
- `splitting/nonvulnerable_cap.py` — NonVulnerable : positive ratio cap (default 3:1)

**Splitter strategies**:
| Strategy | Description | Default For |
|----------|-------------|-------------|
| `stratified` | Per-class + per-source + per-tier distribution preserved (±2%) | Tool-derived sources |
| `project` | Each project goes entirely to one split | Audit datasets (Bastet, Web3Bugs) |
| `temporal` | Pre-2023 train/val, post-2023 test | Optional temporal evaluation |
| `random` | Random assignment, deterministic seed | Sanity testing only |

**Two-pass split**:
1. **Pass 1**: Splitter assigns contracts to splits per strategy
2. **Pass 2**: `dedup_enforcer` reassigns any near-dup group that straddles a split boundary (majority-wins, ties → train)

**NonVulnerable cap** (from `nonvulnerable_cap.py`):
- Default cap: 3:1 (NonVulnerable : positive ratio)
- Stratified by source to preserve per-source distribution
- Prevents the BCCC failure pattern (99% DoS↔Reentrancy co-occurrence from class imbalance)

**Output**: `data/splits/v{N}/{train,val,test}.jsonl` + `split_manifest.json`

---

## 10. Stage 7: Registry

**Entry point**: `sentinel_data.registry.catalog`

**What it does**: Tracks dataset provenance, artifacts, splits, and versioned snapshots in a SQLite database with YAML mirror.

**Key files**:
- `registry/catalog.py` — `Catalog` class (SQLite + YAML mirror)
- `registry/lineage_tracker.py` — DAG lineage tracking for artifacts
- `registry/dataset_diff.py` — Dataset version diffing

**Catalog tables** (6):
| Table | Purpose |
|-------|---------|
| `sources` | Source metadata (pin, tier, contract count) |
| `artifacts` | Content-addressed files (SHA-256, lineage DAG) |
| `splits` | Split configurations (version, seed, strategy) |
| `dataset_versions` | Named, immutable snapshots (append-only) |
| `dataset_version_retirements` | Retirement tracking |
| `schema_migrations` | Forward-only schema evolution |

**API**:
- `Catalog.add_source()`, `get_source()`, `list_sources()`
- `Catalog.add_artifact()`, `get_artifact()`
- `Catalog.add_split()`, `get_split()`
- `Catalog.add_dataset_version()`, `get_dataset_version()`, `list_dataset_versions()`
- `Catalog.load_artifact()` — ML module interface (hash-verified on load)
- `Catalog.verify_artifact_hash()` — SHA-256 verification
- `Catalog.write_yaml_mirror()` — Export for version control

**Output**: `data/registry/catalog.db` + `data/registry/catalog.yaml`

---

## 11. Stage 8: Analysis

**Entry point**: Various analysis modules

**What it does**: Produces dataset quality reports, drift detection, and overlap analysis.

**Key files**:
- `analysis/drift_monitor.py` — KS-test feature + label drift between versions
- `analysis/overlap_detector.py` — Pairwise Jaccard similarity (exact + near) between sources
- `analysis/cooccurrence.py` — Class co-occurrence analysis
- `analysis/feature_dist.py` — Feature distribution analysis
- `analysis/balance_viz.py` — Class balance visualization
- `analysis/probe_dataset.py` — Dataset probing utilities

**Drift monitor** (`drift_monitor.py`):
- Per-feature KS test (node_count, edge_count, loc, function_count, cyclomatic_complexity, call_depth)
- Per-class label distribution KS test (binary: count_positive / total)
- Outputs `data/analysis/<run_id>/drift_report.md`

**Overlap detector** (`overlap_detector.py`):
- Exact overlap: shared sha256s between sources
- Near overlap: shared dedup_groups (AST-similar, different sha256)
- Outputs `overlap_matrix.csv` + `overlap_heatmap.png`

**Output**: `data/analysis/<run_id>/{drift_report.md, overlap_matrix.csv, overlap_heatmap.png}`

---

## 12. Stage 9: Export

**Entry point**: `sentinel_data.export.chunker`

**What it does**: Produces the consumer-facing export artifact that the ML module consumes via `SentinelDataset`.

**Key files**:
- `export/chunker.py` — `chunk_export()` orchestrates all 4 writers
- `export/export.py` — `SentinelDatasetExport` (read-only view with hash verification)
- `export/graph_writer.py` — Graph .pt sharding (default 5000 contracts/shard)
- `export/token_writer.py` — Token .pt sharding
- `export/label_writer.py` — `labels.parquet` (10-class binary matrix)
- `export/metadata_writer.py` — `metadata.parquet` (per-contract metadata)

**Export layout**:
```
<output_dir>/
├── labels.parquet          # 10-class binary labels
├── metadata.parquet        # Per-contract metadata
├── graphs/
│   ├── graphs-{shard:05d}.pt   # Graph shards
│   └── _shard_index.json       # {sha: {shard, pos_in_shard, num_nodes}}
├── tokens/
│   ├── tokens-{shard:05d}.pt   # Token shards
│   └── _shard_index.json
├── manifest.json           # Written LAST (Fix A — avoids circular hash)
└── .hash_cache.json        # Warm-load fast path for hash verification
```

**Manifest fields**:
- `schema_version`: Export format version (currently "v1")
- `graph_schema_version`: Graph feature schema (currently "v9")
- `artifact_hash`: SHA-256 over all data files (excludes manifest.json)
- `n_contracts`: Total contracts in splits
- `n_contracts_with_reps`: Contracts with graph representations
- `n_shards`: Number of graph/token shards
- `splits`: `{train: [sha256...], val: [...], test: [...]}`
- `shard_index`: `{sha: {shard, pos_in_shard, num_nodes}}`
- `source_set`: List of sources actually exported
- `label_class_columns`: The 10 class names (locked order)

**Hash verification** (`SentinelDatasetExport.verify_artifact_hash()`):
- Warm path: stats each file, compares mtime+size to `.hash_cache.json` (milliseconds)
- Cold path: full SHA-256 over all data files (fallback)

**Output**: `data/exports/v{N}/` with manifest.json

---

## 13. Configuration

The single source of truth is `config.yaml` at the data_module root.

**Key sections**:
- `sources_critical_path`: Core sources (SolidiFI, DIVE, Web3Bugs, SmartBugs Curated, DISL)
- `sources_additive`: Deferred sources (Bastet, FORGE, ScrawlD, REKT, etc.)
- `pipeline.min_viable_corpus`: Gate thresholds (total ≥4000, major classes ≥300, minor ≥100)
- `pipeline.negative.positive_ratio_max`: NonVulnerable cap (default 3.0)
- `pipeline.dedup.ast_similarity_threshold`: Dedup threshold (default 0.85)

**Schema constants** (from `representation/graph_schema.py`):
- `FEATURE_SCHEMA_VERSION = "v9"`
- `NODE_FEATURE_DIM = 12`
- `NUM_NODE_TYPES = 14`
- `NUM_EDGE_TYPES = 12`
- `NUM_CLASSES = 10`

---

## 14. CLI Usage

The CLI entry point is `sentinel-data` (defined in `pyproject.toml`).

```bash
# Full pipeline
sentinel-data ingest --source solidifi
sentinel-data preprocess --source solidifi
sentinel-data label --source solidifi
sentinel-data represent --source solidifi
sentinel-data verify
sentinel-data split
sentinel-data register
sentinel-data analyze
sentinel-data export

# Dry run
sentinel-data ingest --source solidifi --dry-run

# Sample mode (fast iteration)
sentinel-data preprocess --source dive --sample 100

# Retry failed files
sentinel-data preprocess --source solidifi --retry-failed

# Force re-extraction
sentinel-data represent --source solidifi --force

# Analysis
sentinel-data analyze --drift --baseline-version v1
sentinel-data analyze --overlap
```

---

## 15. Dependency Map

### Module → Module Dependencies

```
sentinel_data/
├── cli.py → ingest, preprocess, label, represent, verify, split, register, analyze, export
├── ingestion/
│   ├── ingest.py → connectors, manifest
│   ├── manifest.py → (stdlib only)
│   ├── label_folderize.py → (stdlib only)
│   ├── freshness.py → ingest._all_sources
│   └── connectors/ → base.py (all connectors inherit BaseConnector)
├── preprocessing/
│   ├── preprocess.py → pipeline, ingestion.ingest._enabled_sources, label_folderize
│   ├── pipeline.py → compiler, deduplicator, flattener, normalizer, segmenter
│   ├── compiler.py → (subprocess: solc)
│   ├── flattener.py → _transitive_strip
│   ├── deduplicator.py → (hashlib)
│   ├── normalizer.py → (stdlib only)
│   └── segmenter.py → (stdlib only)
├── labeling/
│   ├── schema/__init__.py → taxonomy.yaml
│   ├── parsers/{solidifi,dive,smartbugs_curated}.py → schema.class_names
│   ├── merger.py → schema.class_names
│   └── gate.py → schema.class_names
├── representation/
│   ├── graph_schema.py → (stdlib only, canonical schema)
│   ├── graph_extractor.py → graph_schema (imports all constants)
│   ├── orchestrator.py → graph_extractor, tokenizer, cache_manager, versioner
│   ├── tokenizer.py → ml.src.data_extraction.windowed_tokenizer (thin adapter)
│   ├── cache_manager.py → (json, pathlib)
│   ├── versioner.py → graph_schema, orchestrator, cache_manager
│   └── cfg_builder.py → graph_extractor, orchestrator.EXTRACTOR_VERSION
├── verification/
│   ├── gate.py → class_auditor, semantic_checker, tool_validator, fp_estimator, negative_checker
│   ├── class_auditor.py → schema.class_names
│   ├── semantic_checker.py → schema.class_names, torch (for .pt loading)
│   ├── tool_validator.py → schema.class_names, slither_runner
│   ├── fp_estimator.py → schema.class_names, slither_runner
│   ├── negative_checker.py → schema.class_names, slither_runner
│   └── slither_runner.py → (subprocess: slither)
├── splitting/
│   ├── splitters.py → (stdlib: random, collections)
│   ├── dedup_enforcer.py → splitters.Contract, Splits
│   ├── leakage_auditor.py → splitters.Contract, Splits
│   └── nonvulnerable_cap.py → splitters.Contract, Splits
├── registry/
│   ├── catalog.py → (sqlite3, yaml, hashlib)
│   ├── lineage_tracker.py → catalog
│   └── dataset_diff.py → catalog
├── analysis/
│   ├── drift_monitor.py → schema.class_names, feature_dist
│   ├── overlap_detector.py → (json, csv, collections)
│   ├── cooccurrence.py → schema.class_names
│   ├── feature_dist.py → (json, pathlib)
│   ├── balance_viz.py → schema.class_names, matplotlib
│   └── probe_dataset.py → (json, pathlib)
└── export/
    ├── chunker.py → label_writer, metadata_writer, graph_writer, token_writer, schema.class_names
    ├── export.py → chunker.ExportManifest
    ├── graph_writer.py → (torch, pathlib)
    ├── token_writer.py → (torch, pathlib)
    ├── label_writer.py → (pyarrow, json)
    └── metadata_writer.py → (pyarrow, json)
```

### External Dependencies

| Package | Used By | Purpose |
|---------|---------|---------|
| `slither-analyzer` | graph_extractor, slither_runner | Solidity static analysis |
| `torch` | graph_extractor, orchestrator, semantic_checker | PyG graph tensors |
| `torch_geometric` | graph_extractor | PyG Data objects |
| `transformers` | tokenizer (via ml/) | GraphCodeBERT tokenization |
| `pyarrow` | export writers | Parquet file I/O |
| `pyyaml` | config, taxonomy, catalog | YAML parsing |
| `scipy` | drift_monitor | KS test |
| `matplotlib` | overlap_detector, balance_viz | Visualization |
| `networkx` | (optional) | Graph operations |

### ML Module → Data Module Interface

The ML module consumes data module artifacts via:
- `ml/src/datasets/sentinel_dataset.py` → `SentinelDatasetExport` (from `export/export.py`)
- `ml/src/preprocessing/graph_schema.py` → Re-export shim pointing to `sentinel_data.representation.graph_schema`
- `ml/scripts/train.py` → Reads `data/exports/v{N}/` via `SentinelDataset`

---

## Key Invariants

1. **Schema lock**: `CLASS_NAMES` order is frozen — changing it invalidates all checkpoints
2. **Content addressing**: All artifacts use SHA-256 (not MD5)
3. **Cache invalidation**: Representation cache keyed on `(sha256, schema_version, extractor_version)`
4. **Tier precedence**: T0 > T1 > T2 > T3 > T4 for label conflict resolution
5. **NonVulnerable cap**: Default 3:1 ratio prevents class imbalance
6. **Dedup enforcement**: Two-pass split ensures no near-dup group straddles split boundaries
7. **Manifest-last**: Export writes `manifest.json` last to avoid circular hash dependency
