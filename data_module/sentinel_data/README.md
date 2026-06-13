# `sentinel_data` — SENTINEL Data Engineering Module

> **Status: 9 sub-packages; Stages 0–4 ✅ COMPLETE; Stage 5 (splitting + registry) ✅ COMPLETE; Stages 6 (analysis) ✅ COMPLETE; Stage 7 (export) ✅ COMPLETE.** The `label` CLI subcommand is a stub (merger runs from Python). Run 11 launch: 2026-08-18.

## 1. Purpose

`sentinel_data` is the **data engineering layer** of the SENTINEL smart-contract security oracle. It is a self-contained, reproducible pipeline that turns raw Solidity source code from multiple public corpora (SolidiFI, DIVE, DeFiHackLabs, SmartBugs Curated, BCCC) into model-ready training artifacts. The pipeline produces the dataset that the ML training module consumes in Runs 1–11.

The package is organized as a **5-stage transformation pipeline** (per the v2 architecture plan) plus 3 cross-cutting support systems and a CLI orchestrator. Each stage reads what the previous stage wrote, applies one transformation, and writes its outputs to a well-known location under `data/`:

```
raw ──► [ingest] ──► preprocessed ──► [preprocess] ──► representations ──► [label] ──►
labels/merged ──► [verify] ──► verification_report ──► [split] ──► splits/v1 ──►
[register] ──► catalog ──► [analyze] ──► analysis/<run_id> ──► [export] ──► data/exports/<version>/
```

> **Note on pipeline order**: the previous `__init__.py` claimed the order was `ingest → preprocess → label → represent → split`. **That is wrong.** The actual CLI order (per `cli.py:STAGES`) is `ingest → preprocess → represent → label → verify → split → register → analyze → export`. The representation stage MUST happen before labeling because the labels are stored separately from the representations and labeling only needs the `meta.json` sidecar, but the canonical order in `cli.py` is the source of truth. The old diagram above is corrected below in §3.

The **single user-facing surface** is the `sentinel-data` CLI (`sentinel_data/cli.py`). Every stage is exposed as a subcommand, plus a `run` subcommand that walks multiple stages in sequence.

## 2. Source map (top level)

| File | Lines | Role |
|------|-------|------|
| `__init__.py` | 57 | Package docstring + `__version__ = "0.1.0"`. The 5-stage conceptual model + 3 cross-cutting subpackages. ⚠ The diagram in this file is out of date with `cli.py` (the actual pipeline order is 9 stages, not 5). |
| `cli.py` | 953 | **The single user-facing surface.** `sentinel-data <stage> [--dry-run] [--config] [--source]` + `sentinel-data run [--from-stage STAGE]`. 9 stages + 1 utility (`freshness`). |
| `ingestion/` | 524 lines (4 files) | Stage 1a — pull raw contracts from heterogeneous source types. See `ingestion/README.md`. |
| `preprocessing/` | 1,192 lines (9 files) | Stage 1b — flatten, compile, dedup, normalize, segment. See `preprocessing/README.md`. |
| `representation/` | 1,302 lines (11 files) | Stage 2 — graph + token extraction (v9 schema, thin-adapter pattern). See `representation/README.md`. |
| `labeling/` | 386 lines (3 files + 2 sub-folders) | Stage 3 — parsers + merger + gate. See `labeling/README.md`. |
| `verification/` | 2,709 lines (10 files + 1 sub-folder) | Stage 4 — 5 sub-checkers + gate + report + slither_runner + probe. See `verification/README.md`. |
| `splitting/` | 759 lines (4 files) | Stage 5 — 4 splitters + dedup_enforcer + leakage_auditor + NonVuln cap. See `splitting/README.md`. |
| `registry/` | 800 lines (3 files) | Stage 5 — SQLite + YAML mirror + lineage + dataset_diff. See `registry/README.md`. |
| `analysis/` | 1,350 lines (6 files) | Stage 6 — 5 analysis tools (balance_viz, cooccurrence, drift_monitor, feature_dist, overlap_detector). See `analysis/README.md`. |
| `export/` | 695 lines (7 files + format_schema/) | Stage 7 — 4-writer sharded export (graphs, tokens, labels, metadata) + orchestrator + consumer-facing API. See `export/README.md`. |

**Sub-total: 9,726 lines of Python** across 60 files (excluding `__pycache__` and `__init__.py` files except the package-level one).

## 3. The actual pipeline order (corrected from the old `__init__.py`)

Per `cli.py:STAGES` (line 71-81):

```
0. (skeleton, this package) ──────────────────── ✅
1. ingest      ──► data/raw/<source>/             ✅
2. preprocess  ──► data/preprocessed/<source>/    ✅
3. represent   ──► data/representations/<source>/ ✅
 4. label       ──► data/labels/<source>/, data/labels/merged/  ✅ (CLI is stub; merger runs from Python)
 5. verify      ──► data/verification/verification_report_*.md   ✅
 6. split       ──► data/splits/v<N>/              ✅
 7. register    ──► data/registry/catalog.db       ✅
 8. analyze     ──► data/analysis/<run_id>/        ✅
 9. export      ──► data/exports/<version>/         ✅
```

Utility subcommand (always available):
- `freshness` — `data/analysis/freshness_report.md`

### Conceptual grouping (per `__init__.py:17-28`)

The package's own docstring groups the 9 stages into 5 "directed pipeline" stages + 3 "cross-cutting" support systems. This conceptual model is the *original* v2 design and is still accurate for the high-level data flow:

**5 stage subpackages (the directed pipeline):**
- `ingestion` — pull and stage raw contract corpora
- `preprocessing` — clean, normalize, deduplicate, segment
- `labeling` — map source labels to canonical taxonomy
- `representation` — build PyG graphs and CodeBERT tokens
- `splitting` — train/val/test splits with leakage audit

**3 cross-cutting subpackages:**
- `verification` — Go/No-Go gate that catches label noise
- `registry` — provenance, lineage, and dataset diffs
- `analysis` — post-hoc diagnostics and visualizations

**2 subpackages outside the conceptual model:**
- `export` — Stage 7 sharded export (post-Stage-7 design, now implemented)
- `cli` (in `cli.py` at the package root) — the orchestrator

## 4. Key concepts (cross-cutting)

### The 5-stage conceptual model (per `__init__.py`)

The original v2 architecture is summarized in `__init__.py:9-12`:

```python
raw ──► [ingest] ──► preprocessed ──► [preprocess] ──► labeled ──► [label] ──►
verified ──► [represent] ──► representations ──► [split] ──► splits
```

**But this diagram is out of date** — the `represent` step actually happens BEFORE the `label` step (the canonical CLI order puts `represent` at index 2 and `label` at index 3). The diagram is preserved as-is in `__init__.py` for historical context; trust the `cli.py:STAGES` order for the actual implementation.

### The 3 cross-cutting architectural contracts

1. **Architectural contract**: `sentinel-data` has a one-way dependency on nothing in the ML training code (`sentinel-ml`). Training consumes artifacts produced here; the training code never feeds back. This is what makes the data pipeline independently testable, versionable, and reusable across training experiments. (Per `__init__.py:30-34`.)

2. **Reproducibility contract**: every artifact is content-addressed by SHA-256 and stamped with `schema_version` + `extractor_version` in a JSON sidecar. Two runs with the same config produce byte-identical outputs.

3. **Thin-adapter contract**: the `representation` subpackage re-exports the PyG graph and CodeBERT token logic from the existing `ml/` package via ~10-line re-export files. Bug fixes in the graph or token extraction logic apply once (in `ml/`) and automatically propagate to the new path. Stage 7 of the data pipeline deletes the wrappers and re-binds `ml/` to import from this package directly. (Per `__init__.py:40-46`.)

### The `sentinel-data` CLI (`cli.py`)

Every stage is exposed as a subcommand. The CLI is **the single user-facing surface** — there's no Python API for "run the whole pipeline." A user wanting to run Stage 4 says `sentinel-data verify` (not `python -c "from sentinel_data.verification import run_audit"`).

`cli.py:_STAGE_FN` (line 679-690) is the dispatch table. Each stage function returns `None` on success, or an int exit code (currently only `verify` does this — non-zero on `--strict` mode FAIL).

### Versioning (per `__init__.py:49-56`)

This is `sentinel-data` v0.1.0. The package version is **separate from the artifact version** (schema_version="v9", extractor_version="v2.0-thin-adapter"). Changing the package implementation does NOT automatically invalidate cached representations unless the explicit version stamps in the sidecar change.

### The `sys.path` bootstrap in `cli.py:62-68`

```python
_HERE = Path(__file__).resolve()
_DATA_DIR = _HERE.parent.parent          # sentinel/Data/
_REPO_ROOT = _DATA_DIR.parent            # sentinel/
_ML_ROOT   = _REPO_ROOT / "ml"
for _p in (_REPO_ROOT, _ML_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
```

The CLI's **first non-comment code** adds the SENTINEL repo root and `ml/` to `sys.path`. This is required because the `representation` subpackage uses the **thin-adapter pattern**: `sentinel_data.representation` re-exports the graph and tokenizer code from `ml/src/preprocessing/` and `ml/src/data_extraction/`. Without these paths on `sys.path`, the re-exports fail with `ModuleNotFoundError` whenever the CLI is invoked from outside the repo root (e.g. via an installed entry-point, a Docker container with a different CWD, or a CI runner).

This is the **only** place in the production code where `sys.path` is manipulated; tests use `conftest.py` instead. The two strategies are intentionally parallel: the CLI must work without pytest, the test suite must work without the CLI.

### MLflow backend (per the root README)

The v2 build uses `sqlite:///mlruns.db` only. The `file:///` backend is corrupt (experiments 1, 2, 3 are in the file backend and return empty results). Any plan or script that logs an experiment must set:

```bash
export MLFLOW_TRACKING_URI=sqlite:///mlruns.db
```

This is also declared in `config.yaml` under `pipeline.mlflow.uri`.

### WSL2 caveats (per the root README)

All commands in this module are run inside WSL2. Use `wsl -- bash -c '...'` from the Windows host when scripting. The PowerShell host throws errors on inline WSL commands. Example:

```powershell
# From Windows PowerShell
wsl -- bash -c 'cd /home/motafeq/projects/sentinel/Data && poetry run sentinel-data run --dry-run'
```

## 5. The CLI surface

| Subcommand | Stage | Implementation | Returns exit code on strict? |
|------------|-------|----------------|-------------------------------|
| `sentinel-data run [--from-stage STAGE] [--config CONFIG] [--dry-run]` | All stages in order | `cli.py:_handle_run` | No |
| `sentinel-data ingest [--source NAME] [--config CONFIG] [--dry-run]` | 1a | `ingestion.ingest_source` / `ingest_all` | No |
| `sentinel-data preprocess [--source NAME] [--workers N] [--sample N] [--retry-failed] [--dry-run]` | 1b | `preprocessing.preprocess_source` / `preprocess_all` | No |
| `sentinel-data represent [--source NAME] [--workers N] [--limit N] [--force] [--emit-cfg] [--dry-run]` | 2 | `representation.orchestrator.represent_source` | No |
| `sentinel-data label [--source NAME] [--dry-run]` | 3 | **STUB** — "NOT IMPLEMENTED — implement in Stage 3" (`cli.py:223-229`). Merger runs from Python today. | No |
| `sentinel-data verify [--strict] [--semantic-limit-per-class N] [--tool-limit-per-class N] [--negative-limit N] [--force-slither] [--skip-tool-validator] [--skip-fp-estimator] [--skip-negative-checker] [--dry-run]` | 4 | `verification.class_auditor` + `semantic_checker` + `tool_validator` + `fp_estimator` + `negative_checker` + `gate` + `report_generator` | **Yes** — non-zero on FAIL with `--strict` |
| `sentinel-data split [--version N] [--seed N] [--nonvuln-cap RATIO] [--dry-run]` | 5 (splitting) | `cli.py:_run_split` — reads merged labels, runs stratified splitter, applies dedup_enforcer + NonVulnerable cap, writes train/val/test + manifest | No |
| `sentinel-data register --name NAME [--version N] [--sources SRC ...] [--verification-report PATH] [--retire-previous NAME]` | 5 (registry) | `cli.py:_run_register` — opens Catalog, registers DatasetVersion, writes YAML mirror | No |
| `sentinel-data analyze [--only TOOL] [--run-id ID] [--corpus VERSION] [--baseline-version VERSION] [--dry-run]` | 6 | `cli.py:_run_analyze` — runs 5 analysis tools (balance_viz, feature_dist, cooccurrence, overlap_detector, drift_monitor if `--baseline-version` given) | No |
| `sentinel-data export [--config CONFIG] [--split-version N] [--shard-size N] [--output-dir PATH] [--dry-run]` | 7 | `cli.py:_run_export` — calls `export.chunk_export()` to produce sharded graphs/tokens/labels/metadata + manifest | No |
| `sentinel-data freshness [--config CONFIG]` | (utility) | `ingestion.freshness.run_freshness_check` | No |

The 9 stages are ordered in `cli.py:STAGES` (line 71-81):

```python
STAGES: list[str] = [
    "ingest", "preprocess", "represent", "label", "verify",
    "split", "register", "analyze", "export",
]
```

## 6. The 9 stages — what each one does

| # | Stage | What it does | Input | Output | Status |
|---|-------|--------------|-------|--------|--------|
| 1 | **ingest** | Pull raw `.sol` contracts from all enabled sources with SHA-256 manifests | `config.yaml` `sources_critical_path` + `sources_additive` | `data/raw/<source>/repo/`, `data/raw/<source>/ingestion_manifest.json` | ✅ |
| 2 | **preprocess** | Flatten + 2-pass compile (exact → nearest) + 3-level dedup (SHA → address → AST) + normalize + segment + version-bucket | `data/raw/<source>/` | `data/preprocessed/<source>/<sha>.sol` + `<sha>.meta.json` + `dropped.csv` | ✅ |
| 3 | **represent** | Extract v9 graph (`.pt`) + windowed token files (GraphCodeBERT, `[4,512]`) via thin-adapter from `ml/src/preprocessing/` + `ml/src/data_extraction/windowed_tokenizer.py` | `data/preprocessed/<source>/` | `data/representations/<source>/<sha>.{pt, tokens.pt, rep.json}` + `_version_registry.json` | ✅ |
| 4 | **label** | Apply per-source crosswalk YAMLs; merge labels with tier precedence; flag 99% co-occurrence noise; Go/No-Go gate | `data/preprocessed/<source>/` + crosswalks | `data/labels/<source>/<sha>.labels.json` + `data/labels/merged/<sha>.labels.json` + gate verdict | ⚠️ CLI stub; merger runs from Python |
| 5 | **verify** | AST semantic checks (v9 features) + tool corroboration (Slither) + FP estimation (stratified) + negative checker (contamination) → per-class VERIFIED/PROVISIONAL/BEST-EFFORT/FAIL gate | `data/labels/merged/`, `data/representations/<source>/`, `data/preprocessed/<source>/` | `data/verification/verification_report_<ts>.md` | ✅ |
| 6 | **split** | Deterministic train/val/test splits (4 strategies, default stratified) + 2-pass with dedup_enforcer + NonVulnerable 3:1 cap (stratified by source) | `data/labels/merged/` | `data/splits/v<N>/{train,val,test}.jsonl` + `split_manifest.json` | ✅ |
| 7 | **register** | Register a dataset version in the SQLite catalog (4 base + 2 system tables) + YAML mirror | `data/splits/v<N>/` | `data/registry/catalog.db` + `data/registry/catalog.yaml` | ✅ |
| 8 | **analyze** | 5 read-only exploratory tools: balance_viz + feature_dist (the **Run-9-failure catcher** with `complexity_proxy_risk.md` headline) + cooccurrence + overlap_detector + drift_monitor (with `--baseline-version`) | `data/labels/merged/`, `data/representations/`, `data/preprocessed/` | `data/analysis/<run_id>/*.csv`, `*.png`, `complexity_proxy_risk.md`, `drift_report.md` | ✅ |
| 9 | **export** | Shard export to sentinel-ml: 4 writers (graphs as PyG Batch, tokens as torch.Tensor, labels/metadata as parquet) + orchestrator + manifest with SHA-256 artifact hash | `data/representations/`, `data/labels/merged/`, `data/splits/v<N>/`, `data/preprocessed/` | `data/exports/<version>/` (graphs/, tokens/, labels.parquet, metadata.parquet, manifest.json) | ✅ |

## 7. Pipeline interactions

| Stage | Direction | What |
|-------|-----------|------|
| `cli.py` | ↔ | Top-level orchestrator; `_STAGE_FN` dispatch table at line 679-690 |
| `config.yaml` | → | Source definitions, tier mappings, gate thresholds, cap values |
| `sentinel_data.ingestion` | → 1a | Pulls raw contracts + writes manifests |
| `sentinel_data.preprocessing` | → 1b | Reads raw + writes preprocessed |
| `sentinel_data.representation` | → 2 | Reads preprocessed + writes representations (via thin-adapter to `ml/`) |
| `sentinel_data.labeling` | → 3 | Reads preprocessed + writes labels |
| `sentinel_data.verification` | → 4 | Reads labels + representations + preprocessed + writes report |
| `sentinel_data.splitting` | → 5a | Reads merged labels + writes splits |
| `sentinel_data.registry` | → 5b | Reads split manifest + writes catalog |
| `sentinel_data.analysis` | → 6 | Reads labels + representations + preprocessed + writes analysis |
| `sentinel_data.export` | → 7 | Reads representations + labels + splits + preprocessed; writes sharded export (4 file types + manifest) |
| `ml/` (post-Stage-7) | ↔ | Reads from `sentinel_data.export` (or directly from `sentinel_data.representation` + `sentinel_data.labeling` + `sentinel_data.splitting` during the seam-swap transition) |

## 8. Tests

**Location:** `data_module/tests/` — 8 test packages mirroring the source layout:
- `test_ingestion/` — `test_connector.py`, `test_manifest.py`, `test_label_folderize.py`
- `test_preprocessing/` — `test_pipeline.py`, `test_retry_failed.py`
- `test_representation/` — `test_thin_adapter.py`, `test_byte_identical_regression.py`, `test_orchestrator.py`, `test_solidifi_fixes.py`, `test_13_issue_preservation.py` (no `__init__.py` — pytest still discovers)
- `test_labeling/` — `test_parser_solidifi.py`, `test_parser_dive.py`, `test_merger.py`, `test_gate.py`, `test_crosswalk_solidifi.py`, `test_crosswalk_dive.py`, `test_taxonomy.py`
- `test_verification/` — `test_class_auditor.py`, `test_semantic_checker.py`, `test_tool_validator.py`, `test_fp_estimator.py`, `test_negative_checker.py`, `test_gate.py`, `test_report_generator.py`, `test_patterns.py`, `test_probe_dataset.py`, `test_cli_verify.py`, `test_bccc_regression.py`, `test_smartbugs_recall.py`
- `test_splitting/` — `test_splitters.py`
- `test_registry/` — `test_catalog.py`
- `test_analysis/` — (no dedicated tests; covered by `tests/test_skeleton.py` and integration tests)

Plus top-level:
- `tests/test_skeleton.py` — import smoke test
- `tests/test_integration_solidifi.py` — full end-to-end on SolidiFI
- `tests/test_integration_dive.py` — full end-to-end on DIVE (22K files)
- `tests/conftest.py` — pytest fixtures (sys.path bootstrap mirrors `cli.py:62-68`)

**Command:**
```bash
cd ~/projects/sentinel/data_module
poetry run pytest tests/ -v
```

**Test counts (per MEMORY.md, 2026-06-11):**
- Stage 0: 27 tests
- Stage 1: 35 tests (cumulative 65/65)
- Stage 2: +20 tests (cumulative 85/85)
- Stage 3: +15 tests (cumulative 100/100) — actually 80/80 per MEMORY.md
- Stage 4: +11 tests (cumulative 91/91)
- **Total: ~91 unit tests + 2 integration tests + 1 skeleton test = ~94 tests**

> **Note:** The test counts above are from MEMORY.md; the actual current count should be re-verified with `pytest --collect-only`.

## 9. See also

- **The whole package is documented per-subpackage** — see the 9 sub-package READMEs linked in §2.
- **The root README**: `data_module/README.md`
- **The test architecture overview**: `data_module/tests/README.md`
- **Architecture + ADRs**: `docs/ml/adr/0007-representation-port-design.md` and other ADRs in `docs/decisions/`
- **The full Stage plans**: `docs/proposal/Data_Module_Proposals/actionable_plans/`
- **MEMORY.md** (canonical v2 facts, do not re-derive): `~/.claude/projects/-home-motafeq-projects-sentinel/memory/MEMORY.md`
- **ML-side counterparts** (for context): `ml/src/preprocessing/`, `ml/src/data_extraction/`, `ml/src/training/`
- **Run 11 launch plan (2026-08-18)**: `docs/training/run11_launch_plan.md` (to be created)
