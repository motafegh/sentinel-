# sentinel-data

> **Data engineering module for the SENTINEL smart-contract security oracle.**
> **Status (2026-06-12):** Stages 0–4 ✅ COMPLETE; Stage 5 (splitting + registry) ✅ COMPLETE; Stage 6 (analysis) ✅ COMPLETE; Stage 7 (export) ⏳ STUB. **Run 11 launch: 2026-08-18.**

Builds a **verified, multi-source, multi-label** Solidity contract dataset from 17 curated sources and exports it to `sentinel-ml` as versioned shards. The module exists because the BCCC corpus that trained Runs 1–9 had 89.4% Reentrancy false-positives and 86.9% CallToUnknown false-positives — the model was learning label noise, not vulnerability patterns. See `docs/legacy/bccc_deep_dive/` for the full Phase 1–5 investigation.

**One-way dependency:** `sentinel-data` → (used by) → `sentinel-ml`. This package never imports from `sentinel-ml`.

---

## Quick start

```bash
# Install (core deps, no GPU required)
cd data_module/
poetry install

# Verify the installation
poetry run sentinel-data --help

# Dry-run the full pipeline
poetry run sentinel-data run --dry-run

# Dry-run from a specific stage
poetry run sentinel-data run --dry-run --from-stage verify

# Run a single stage (dry)
poetry run sentinel-data ingest --source solidifi --dry-run

# Run a single stage (real)
poetry run sentinel-data verify

# Run all tests
poetry run pytest tests/ -v
```

---

## Module map

```
data_module/                              # THIS PACKAGE
├── pyproject.toml                         # sentinel-data package definition (poetry)
├── config.yaml                            # single source of truth for sources + pipeline settings
├── dvc.yaml                               # 9-stage DVC DAG
├── README.md                              # ← you are here
├── sentinel_data/                         # installable Python package (9 sub-packages)
│   ├── __init__.py                        # package docstring + __version__ = "0.1.0"
│   ├── cli.py                             # entry point: sentinel-data <stage> [--dry-run] (953 lines)
│   ├── ingestion/                         # Stage 1a — per-source connectors + SHA-256 manifests
│   │   ├── README.md
│   │   ├── ingest.py                      # orchestrator (ingest_source / ingest_all)
│   │   ├── manifest.py                    # IngestionManifest + FileRecord + verify_manifest
│   │   ├── freshness.py                   # pin staleness + slither-analyzer version check
│   │   ├── label_folderize.py             # per-class symlinks for CSV-labeled sources
│   │   └── connectors/                    # 5 connector types + 2 aliases (strategy pattern)
│   │       └── README.md
│   ├── preprocessing/                     # Stage 1b — flatten + 2-pass compile + 3-level dedup + normalize + segment
│   │   ├── README.md
│   │   ├── pipeline.py                    # PreprocessingPipeline + ContractMeta sidecar
│   │   ├── preprocess.py                  # CLI service (--sample N, --retry-failed)
│   │   ├── flattener.py                   # solc --flatten + 2-stage unresolved-import strip fallback
│   │   ├── compiler.py                    # 2-pass: exact pragma → nearest available
│   │   ├── deduplicator.py                # SHA-256 → address → (AST near-dup STUB)
│   │   ├── normalizer.py                  # strip comments, SPDX, whitespace
│   │   ├── segmenter.py                   # version bucket + has_unchecked_block
│   │   ├── parallel.py                    # multiprocessing wrapper
│   │   └── _transitive_strip.py           # helper for flattener's import-strip fallback
│   ├── representation/                    # Stage 2 — graph (.pt) + token extraction (v9 schema, thin-adapter)
│   │   ├── README.md
│   │   ├── graph_schema.py                # thin-adapter over ml/src/preprocessing/graph_schema.py
│   │   ├── graph_extractor.py             # thin-adapter over ml/src/preprocessing/graph_extractor.py
│   │   ├── tokenizer.py                   # thin-adapter over ml/src/data_extraction/windowed_tokenizer.py
│   │   ├── orchestrator.py                # v2 manifest-driven, SHA-256 from Stage 1
│   │   ├── cache_manager.py               # content-addressed cache (schema + extractor version)
│   │   ├── versioner.py                   # version registry (prevents Run 8 v8/v9 silent mix)
│   │   ├── cfg_builder.py                 # opt-in standalone CFG (--emit-cfg)
│   │   ├── call_graph.py                  # DEFERRED to v3.1
│   │   ├── opcode_extractor.py            # DEFERRED to v3.1
│   │   └── pdg_builder.py                 # DEFERRED to v3.1
│   ├── labeling/                          # Stage 3 — parsers + merger + Go/No-Go gate
│   │   ├── README.md
│   │   ├── merger.py                      # multi-source label merger (tier precedence + 99% co-occurrence)
│   │   ├── gate.py                        # minimum-viable-corpus Go/No-Go gate
│   │   ├── parsers/                       # one per source (solidifi, dive)
│   │   │   └── README.md
│   │   ├── schema/                        # canonical 10-class taxonomy
│   │   │   └── README.md
│   │   └── crosswalks/                    # source-specific class maps (TO BE CREATED)
│   ├── verification/                      # Stage 4 — AST semantic checks + Slither corroboration + gate
│   │   ├── README.md
│   │   ├── class_auditor.py               # 10×10 co-occurrence matrix + flagging
│   │   ├── semantic_checker.py            # graph-feature-based per-class checks
│   │   ├── tool_validator.py              # Slither per-class agreement
│   │   ├── fp_estimator.py                # stratified-by-(source,tier) FP rate
│   │   ├── negative_checker.py            # NonVulnerable contamination
│   │   ├── gate.py                        # per-class VERIFIED/PROVISIONAL/BEST-EFFORT/FAIL
│   │   ├── report_generator.py            # human-readable verification_report_<ts>.md
│   │   ├── slither_runner.py              # shared Slither with content-addressed cache
│   │   ├── probe_dataset.py               # 40+2 per class for model interpretability
│   │   ├── probe_trivials.py              # 10 trivial positives + 1 trivial negative
│   │   └── patterns/                      # 10 per-class pattern YAMLs (documentation only)
│   │       └── README.md
│   ├── splitting/                         # Stage 5a — train/val/test splits (4 strategies + dedup_enforcer + NonVuln cap)
│   │   ├── README.md
│   │   ├── splitters.py                   # Contract / Splits / SplitMetadata + 4 splitter functions
│   │   ├── dedup_enforcer.py              # BCCC-failure pattern fix
│   │   ├── leakage_auditor.py             # post-split text shingle safety net
│   │   └── nonvulnerable_cap.py           # 3:1 cap (friend review)
│   ├── registry/                          # Stage 5b — SQLite + YAML mirror + lineage
│   │   ├── README.md
│   │   ├── catalog.py                     # 4 base + 2 system tables
│   │   ├── dataset_diff.py                # per-class metric projection
│   │   └── lineage_tracker.py             # DAG of transformations + hash_artifact / verify_artifact
│   ├── analysis/                          # Stage 6 — 5 read-only exploratory tools
│   │   ├── README.md
│   │   ├── balance_viz.py                 # per-class / per-source / per-tier counts
│   │   ├── feature_dist.py                # ⭐ the Run-9-failure catcher (complexity_proxy_risk.md)
│   │   ├── cooccurrence.py                # directed + conditional matrices
│   │   ├── overlap_detector.py            # pairwise Jaccard between source datasets
│   │   ├── drift_monitor.py               # KS test for feature + label distribution
│   │   └── probe_dataset.py               # re-export (actual impl in verification/)
│   ├── export/                            # Stage 7 — STUB
│   │   ├── README.md
│   │   └── __init__.py                    # 10-line module docstring only
│   └── tests/                             # consolidated test overview
│       └── README.md
├── tests/                                 # pytest test suite (~94 tests + 2 integration)
│   ├── conftest.py                        # sys.path bootstrap (parallels cli.py:62-68)
│   ├── test_skeleton.py
│   ├── test_integration_solidifi.py
│   ├── test_integration_dive.py
│   ├── test_ingestion/                    # 3 test files
│   ├── test_preprocessing/                # 2 test files
│   ├── test_representation/               # 5 test files (no __init__.py — pytest rootdir)
│   ├── test_labeling/                     # 7 test files
│   ├── test_verification/                 # 12 test files
│   ├── test_splitting/                    # 1 test file
│   ├── test_registry/                     # 1 test file
│   └── test_analysis/                     # 1 test file
├── data/                                  # DVC-tracked pipeline outputs (not in git)
│   ├── raw/                               # ingested .sol files + manifests
│   ├── preprocessed/                      # flattened + compiled + deduped
│   ├── representations/                   # graph .pt + token files
│   ├── labels/                            # per-contract multi-label CSVs
│   ├── verification/                      # verification reports + dropped.csv
│   ├── splits/                            # train/val/test split manifests
│   ├── registry/                          # SQLite catalog
│   ├── analysis/                          # complexity_proxy_risk.md + co-occurrence + drift
│   └── exports/                           # sharded .pt export for sentinel-ml (after Stage 7)
├── docs/
│   ├── architecture.md                    # data-flow + DAG diagrams
│   ├── decisions/                         # ADRs (append-only)
│   └── legacy/bccc_deep_dive/             # frozen Phase 1–5 BCCC investigation
└── docker/
    └── Dockerfile.data                    # python:3.12.1-bookworm + 6 baseline solc versions
```

---

## Pipeline stages

| # | Stage | CLI subcommand | What it does | Status |
|---|-------|----------------|--------------|--------|
| 0 | (skeleton) | — | Package setup + DVC scaffold | ✅ |
| 1a | Ingest | `sentinel-data ingest [--source NAME]` | Pull raw `.sol` from all enabled sources + SHA-256 manifests | ✅ |
| 1b | Preprocess | `sentinel-data preprocess [--source NAME] [--workers N] [--sample N] [--retry-failed]` | Flatten + 2-pass compile + 3-level dedup + normalize + segment | ✅ |
| 2 | Represent | `sentinel-data represent [--source NAME] [--workers N] [--limit N] [--force] [--emit-cfg]` | Extract v9 graph (`.pt`) + windowed tokens (GraphCodeBERT, `[4,512]`) via thin-adapter | ✅ |
| 3 | Label | `sentinel-data label [--source NAME]` | Apply per-source crosswalk YAMLs + merge + 99% co-occurrence flag + Go/No-Go gate | ⚠️ CLI STUB (`cli.py:223-229`); merger runs from Python today |
| 4 | Verify | `sentinel-data verify [--strict] [--semantic-limit-per-class N] [--tool-limit-per-class N] [--negative-limit N] [--force-slither] [--skip-tool-validator] [--skip-fp-estimator] [--skip-negative-checker]` | 5 sub-checkers (semantic + tool_validator + fp_estimator + negative_checker + class_auditor) → per-class gate → `verification_report_<ts>.md` | ✅ |
| 5a | Split | `sentinel-data split [--version N] [--seed N] [--nonvuln-cap RATIO]` | 4 strategies + dedup_enforcer + NonVulnerable 3:1 cap → `data/splits/v<N>/{train,val,test}.jsonl` + `split_manifest.json` | ✅ |
| 5b | Register | `sentinel-data register --name NAME [--version N] [--sources SRC ...] [--verification-report PATH] [--retire-previous NAME]` | Register a dataset version in `data/registry/catalog.db` + YAML mirror | ✅ |
| 6 | Analyze | `sentinel-data analyze [--only TOOL] [--run-id ID] [--corpus VERSION] [--baseline-version VERSION]` | 5 read-only tools: `balance_viz`, `feature_dist` (⭐ the Run-9-failure catcher with `complexity_proxy_risk.md`), `cooccurrence`, `overlap_detector`, `drift_monitor` | ✅ |
| 7 | Export | `sentinel-data export` | Sharded export to sentinel-ml + 2 ML-side bug fixes (predictor.py tier threshold + EMITS edge) | ⏳ STUB |
| — | (utility) | `sentinel-data freshness` | Pin staleness + `slither-analyzer` version check → `data/analysis/freshness_report.md` | ✅ |
| — | (orchestrator) | `sentinel-data run [--from-stage STAGE]` | Walk multiple stages in sequence (calls the dispatch table in `cli.py:679-690`) | ✅ |

Run 11 launch: **2026-08-18.**

---

## Data sources (17 enabled + BCCC deferred)

See `config.yaml` for the full list with pins and connector types. Summary:

**Tier 1 (gold — human-curated or mathematically certain):**
ScaBench, SmartBugs Curated, SolidiFI, DIVE (Nature 2025), FORGE (ICSE 2026), Web3Bugs

**Tier 2 (silver — expert-audited or tool-majority):**
ScrawlD (3/5 tool majority), Code4rena audit reports, DeFi Hacks REKT, DeFiVulnLabs, EVMbench, Bastet

**Tier 3 (bronze — tool-generated, conservative):**
SmartBugs Wild ⚠ (97% FP rate — use with care), Slither-Audited (HF), OpenZeppelin Contracts, Ethernaut

**Tier 4 (unlabeled pretraining):**
DISL (514K contracts), ReentrancyStudy (230K)

**BCCC (deferred):** 89.4% Reentrancy FP. Legacy outputs at `docs/legacy/bccc_deep_dive/`. Verified v1.4 labels (24,021 contracts) may be re-introduced as a gold supplement in v2.1.

> **Sources list is also reflected in v2 design documents.** The 5 critical-path sources for Run 11 (per MEMORY.md §"Critical v2 facts") are: **DeFiHackLabs, SolidiFI, DIVE, SmartBugs Curated, Web3Bugs** + DISL as the NonVulnerable pool (3:1 cap).

---

## Schema version

The active graph schema is **v9** (verified 2026-06-08). See `sentinel_data/representation/graph_schema.py` for the full constant table.

| Constant | Value |
|----------|-------|
| `FEATURE_SCHEMA_VERSION` | `"v9"` |
| `NODE_FEATURE_DIM` | `12` |
| `NUM_NODE_TYPES` | `14` |
| `NUM_EDGE_TYPES` | `12` |
| `NUM_CLASSES` | `10` (LOCKED) |
| `EXTRACTOR_VERSION` | `"v2.1-windowed-gcb"` |

> **Single source of truth (post-Phase D, 2026-06-12, ADR-0009):** the canonical 10-class order is the LABELING order: `CallToUnknown=0, DenialOfService=1, ExternalBug=2, GasException=3, IntegerUO=4, MishandledException=5, Reentrancy=6, Timestamp=7, TransactionOrderDependence=8, UnusedReturn=9`. This order is used by:
> - `sentinel_data/representation/graph_schema.py` (canonical)
> - `ml/src/training/trainer.py:105-116` (defines its own copy, in sync)
> - `ml/src/preprocessing/graph_schema.py` (shim re-exports the canonical)
> - The v9 best checkpoint (`GCB-P1-Run9-v11-20260606_best.pt`, `class_names` field)
> - The v2 export (`labels.parquet` columns `class_0..class_9`)
> - `sentinel_data.labeling.schema.class_names()`
>
> The pre-Run-7 "representation order" (Reentrancy=0, ..., NonVulnerable=9) is **historical and no longer used in production**. Do not re-introduce it. The schema-dim gate test in `tests/test_representation/test_byte_identical_regression.py` and the new CLASS_NAMES assertions in `tests/test_representation/test_thin_adapter.py` guard against silent re-ordering.

---

## WSL2 caveats

All commands in this module are run inside WSL2. Use `wsl -- bash -c '...'` from the Windows host when scripting. The PowerShell host throws errors on inline WSL commands. Example:

```powershell
# From Windows PowerShell
wsl -- bash -c 'cd /home/motafeq/projects/sentinel/Data && poetry run sentinel-data run --dry-run'
```

The `cli.py:62-68` `sys.path` bootstrap makes the CLI work from any CWD (including Docker) by adding the repo root and `ml/` to `sys.path` before any imports.

---

## MLflow backend

The v2 build uses `sqlite:///mlruns.db` only. The `file:///` backend is **corrupt** (experiments 1, 2, 3 are in the file backend and return empty results). Any plan or script that logs an experiment must set:

```bash
export MLFLOW_TRACKING_URI=sqlite:///mlruns.db
```

This is also declared in `config.yaml` under `pipeline.mlflow.uri`.

---

## Architecture in 3 sentences

**The v2 data pipeline is a 5-stage transformation pipeline** (ingest → preprocess → represent → label → verify) followed by **3 cross-cutting support systems** (splitting, registry, analysis) and **one final stage** (export — currently a stub). **Every stage writes to a well-known path** under `data/` and is the canonical input of the next stage; **`sentinel-data` has a one-way dependency on `sentinel-ml`** (the data module never imports from training code). **The thin-adapter pattern** in `sentinel_data/representation` re-exports the v9 graph and tokenizer code from `ml/src/preprocessing/` and `ml/src/data_extraction/`, so bug fixes apply once and propagate automatically; the Stage 7 seam swap deletes the wrappers and rebinds the import.

For the full architecture (data-flow diagrams, DAGs, the BCCC failure retrospective), see `docs/architecture.md` and `docs/legacy/bccc_deep_dive/`.

---

## Design decisions

See `../../docs/decisions/` for the full ADR register (project-wide). The most important for this module:

- **ADR-0001:** Why `sentinel-data` is a separate package (the BCCC failure + one-way dependency rule)
- **ADR-0002:** Code bug state at build start (8 fixed bugs + 2 still-open — what the Stage 2 regression test guards)
- **ADR-0007:** Representation port design (the thin-adapter pattern + lazy import support)
- **ADR-0009:** Canonical 10-class vocabulary — labeling order as the single source of truth (resolves the two-taxonomy divergence)

---

## Contributing

1. Read `../../docs/decisions/ADR-0001-sentinel-data-skeleton.md` for the architectural context
2. Check `docs/proposal/Data_Module_Proposals/00_INDEX.md` for the current stage plan
3. `poetry install` and `poetry run pytest tests/ -v` must pass before any PR
4. The Stage 2 regression test (`tests/test_representation/test_byte_identical_regression.py`) is the merge gate for any extractor change
5. The BCCC regression test (`tests/test_verification/test_bccc_regression.py`) is the merge gate for any verification change
6. **The 10-class taxonomy is locked per ADR-0009** (labeling order). If you add a new class, bump `FEATURE_SCHEMA_VERSION` to v10, re-train from scratch, and re-export.

---

## Quick links

- **Per-subpackage READMEs**: 9 files at `sentinel_data/<subpackage>/README.md` + 4 sub-subfolder READMEs (`connectors`, `parsers`, `schema`, `patterns`)
- **Tests overview**: `tests/README.md`
- **Stage plans**: `docs/proposal/Data_Module_Proposals/actionable_plans/`
- **ADRs**: `../../docs/decisions/` (project-wide; see `INDEX.md` there)
- **MEMORY.md** (canonical v2 facts): `~/.claude/projects/-home-motafeq-projects-sentinel/memory/MEMORY.md`
- **Run 9 best checkpoint** (model-side, for context): `ml/checkpoints/GCB-P1-Run9-v11-20260606_best.pt` (ep52, fixed=0.2965, tuned=0.3081)
- **Run 10 plan**: train on v1.3 verified labels (in `data_module/docs/legacy/bccc_deep_dive/Phase5_LabelVerification_2026-06-08/outputs/`)
- **Run 11 launch**: 2026-08-18; sqlite mlflow; `--gnn-prefix-warmup-epochs=5`; `--jk-entropy-reg-lambda=0.005`; timestamped `--run-name`; watcher = copy of `ml/scripts/run8_watcher.sh` with F1>0.1 floor
