# `sentinel_data.ingestion` — Stage 1a: Pulling Raw Contracts from the Wild

> **Status: ✅ Fully implemented for `git` + `manual` source types. 3 stubs (HuggingFace, Zenodo, Etherscan) raise `NotImplementedError` — see `connectors/` subfolder.**

## 1. Purpose

This is the **front door** of the SENTINEL v2 data pipeline. Its only job is to **pull raw `.sol` files from upstream sources, verify their integrity, and record exactly what was fetched and when.** Every contract that eventually becomes part of the training dataset first passes through this module.

The contracts arrive in different shapes — git repos, HuggingFace datasets, Zenodo records, Etherscan API responses, or manually-downloaded directories/zips. The connectors normalize them into a uniform layout with SHA-256 verification. The pipeline never sees the heterogeneity.

In machine learning, your model is only as good as your data. Without this stage, you can't answer "what exact version of BCCC did we train on?" six months from now. The BCCC corpus (the predecessor) had a 38.8% duplication rate across sources and no way to reconstruct which source version was used. This stage is the antidote.

## 2. Source map

| File | Lines | Role |
|------|-------|------|
| `__init__.py` | 6 | Module docstring. |
| `ingest.py` | 113 | **Orchestrator** — reads `config.yaml`, picks the right connector per source, calls `pull()`, writes the manifest. Entry points: `ingest_source(name, cfg, data_dir, dry_run)` + `ingest_all(cfg, data_dir, dry_run)`. |
| `manifest.py` | 111 | `IngestionManifest` dataclass + `FileRecord` per `.sol` file + `verify_manifest()` load-time check. SHA-256 of every file recorded. |
| `freshness.py` | 120 | `run_freshness_check(cfg, data_dir)` — compares pinned versions to upstream HEAD (via `git ls-remote`) and checks installed `slither-analyzer` against latest PyPI. Writes `data/analysis/freshness_report.md`. |
| `label_folderize.py` | 173 | `folderize_by_labels(repo_dir, labels_csv, id_column, class_columns, source_subdir="__source__")` — for sources that distribute labels in a separate CSV (DIVE, SmartBugs Wild, Bastet), creates per-class folder symlinks. |
| `connectors/` | (sub-folder) | Strategy pattern. 5 connector types + 2 aliases (audit_report, rekt_scraper). See `connectors/README.md`. |

**Sub-total: 524 lines** across the 4 Python files here (excluding `__init__.py` + `connectors/`).

## 3. Key concepts

### The 3 input sources for `config.yaml`

`ingest.py:15-21` merges 3 places where sources can be declared:

```python
def _all_sources(cfg: dict) -> dict[str, dict]:
    out = {}
    out.update(cfg.get("sources_critical_path") or {})
    out.update(cfg.get("sources_additive") or {})
    out.update(cfg.get("sources") or {})        # legacy v1 key
    return out

def _enabled_sources(cfg: dict) -> dict[str, dict]:
    return {k: v for k, v in _all_sources(cfg).items() if v.get("enabled")}
```

`sources_critical_path` is the v2 list of T0/T1/T2 sources that Run 11 ships with (DeFiHackLabs, SolidiFI, DIVE, SmartBugs Curated, Web3Bugs). `sources_additive` is for v2.1 deferred sources. `sources` is a legacy v1 key (kept for back-compat).

**An enabled source has `enabled: true` in its config entry.** A source with `enabled: false` (or absent) is preserved in config but skipped during ingestion. This lets you disable a noisy source without deleting its config.

### The `SourceConfig` dataclass (`connectors/base.py:12-24`)

A `dict` from `config.yaml` is parsed into a typed `SourceConfig` by `ingest.py:29-46` (`_source_config`). The dataclass has 9 standard fields + a catch-all `extra: dict` for source-specific config (e.g. `staging_path` for `manual`, `post_clone_cmd` for `git`).

### The 5-stage ingest call chain

1. `sentinel-data ingest --source <name>` (or no `--source` for all)
2. `cli.py:_run_ingest` parses args, loads `config.yaml`, calls `ingest.py:ingest_source` or `ingest_all`
3. `ingest_source` resolves `enabled` + builds `SourceConfig` + calls `connectors.get_connector(connector_type).pull(cfg, raw_dir)`
4. The connector materializes the source into `data/raw/<source>/repo/` and returns `PullResult(sol_files=[...])`
5. `ingest.py:80-91` wraps the result in an `IngestionManifest` with per-file SHA-256 and writes `data/raw/<source>/ingestion_manifest.json`

### The append-only manifest (`manifest.py:18-43`)

```python
@dataclass
class IngestionManifest:
    source: str
    connector: str
    url: str
    pin: str                           # commit / version / record — empty = HEAD at fetch time
    resolved_pin: str                  # the actual commit/hash resolved at fetch time
    fetched_at: str                    # ISO-8601
    duration_s: float
    contract_count: int
    files: list[FileRecord] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)
```

The manifest is **append-only** — past entries are never deleted, only new ones are added. A re-ingest **re-validates every SHA-256** (`manifest.verify()`) and fails loud if any file changed (upstream force-push, accidental edit, disk corruption). This is the audit trail that lets you answer "what version of DeFiHackLabs did Run 11 train on?" six months from now.

### The freshness check (`freshness.py`)

`run_freshness_check(cfg, data_dir)` compares each pinned source to upstream HEAD. For git sources it uses `git ls-remote <url> HEAD` to get the latest commit SHA. For the installed `slither-analyzer`, it queries PyPI's JSON API and compares to `importlib.metadata.version("slither-analyzer")`. Output: `data/analysis/freshness_report.md` with one line per source like:

```
- **defihacklabs**: STALE — pinned=abc123... upstream=def456...
- **solidifi**: OK
- **dive**: UNPINNED — upstream HEAD=abc...
```

The report is **informational, not blocking** — it triggers a human review process to decide whether to bump the pin. A stale Slither is an early-warning for the extractor-broke-silently failure mode (Slither API changes broke `graph_extractor.py` in Run 9 — see ADR-0002).

### Label folderization for CSV-based sources (`label_folderize.py:73-172`)

Some sources distribute labels in a separate CSV (DIVE, SmartBugs Wild, Bastet). `folderize_by_labels` materializes per-class folder symlinks in `data/raw/<source>/repo/<Class>/<id>.sol` so downstream code that assumes folder structure (crosswalk YAMLs, split-stage stratification) has a uniform interface.

**Layout** (revised 2026-06-10):
```
data/raw/<source>/repo/__source__/<id>.sol     ← canonical (real) files
data/raw/<source>/repo/<Class1>/<id>.sol       → ../../__source__/<id>.sol   (symlink)
data/raw/<source>/repo/<Class2>/<id>.sol       → ../../__source__/<id>.sol
```

Multi-label handling: a contract with 3 positive labels appears in 3 folders. For folderized sources like SolidiFI where the canonical location is `repo/buggy_contracts/`, set `source_subdir="buggy_contracts"`. For flat sources like DIVE, set `source_subdir="__source__"` and the function moves flat files into `__source__/` first.

The operation is **idempotent** — running twice produces the same result.

## 4. Public API

### `ingest_source(name, cfg, data_dir, dry_run=False)` — `ingest.py:49-92`

```python
def ingest_source(
    name: str,
    cfg: dict,
    data_dir: Path,
    dry_run: bool = False,
) -> IngestionManifest | None:
    """Pull one source and write its ingestion manifest.
    
    Returns the manifest on success, None on dry_run.
    Raises ConnectorError on failure (source not found, not enabled,
    connector error, etc.).
    """
```

### `ingest_all(cfg, data_dir, dry_run=False)` — `ingest.py:95-113`

```python
def ingest_all(cfg: dict, data_dir: Path, dry_run: bool = False) -> list[IngestionManifest]:
    """Ingest every enabled source and return their manifests.
    
    Prints progress to stdout for each source pulled.
    """
```

### `IngestionManifest` — `manifest.py:26-43`

```python
@dataclass
class IngestionManifest:
    source: str
    connector: str
    url: str
    pin: str
    resolved_pin: str
    fetched_at: str
    duration_s: float
    contract_count: int
    files: list[FileRecord]
    extra: dict[str, Any]
    
    def save(self, path: Path) -> None: ...
    @classmethod
    def load(cls, path: Path) -> "IngestionManifest": ...
    def verify(self, raw_dir: Path) -> tuple[bool, list[str]]: ...
```

### `verify_manifest(raw_dir, source)` — `manifest.py:108-111`

```python
def verify_manifest(raw_dir: Path, source: str) -> tuple[bool, list[str]]:
    """Load and verify a source's manifest against files on disk.
    
    Returns (ok, errors). ok=True means every SHA-256 still matches.
    """
```

### `run_freshness_check(cfg, data_dir)` — `freshness.py:17-62`

```python
def run_freshness_check(cfg: dict, data_dir: Path) -> str:
    """Generate a freshness report and write it to data/analysis/freshness_report.md.
    
    Returns the report content as a string.
    """
```

### `folderize_by_labels(repo_dir, labels_csv, id_column, class_columns, source_subdir="__source__")` — `label_folderize.py:73-172`

```python
def folderize_by_labels(
    repo_dir: Path,
    labels_csv: Path,
    id_column: str,
    class_columns: list[str],
    source_subdir: str = "__source__",
) -> FolderizationResult:
    """Create per-class symlinks in `repo_dir` based on a labels CSV.
    
    Returns FolderizationResult with counts.
    """
```

## 5. Inputs → outputs

| Input | Where | What |
|-------|-------|------|
| `config.yaml` `sources_critical_path` / `sources_additive` | `data/config.yaml` | Source definitions with `enabled`, `connector`, `url`/`pin`/`staging_path`, etc. |
| `data/raw/<source>/` (created) | `ingest.py:69` | Destination dir for the materialized source |

| Output | Where | What |
|--------|-------|------|
| `data/raw/<source>/repo/*.sol` | The materialized source | `.sol` files (canonical or symlinked). For multi-class folderization, also `<Class>/<id>.sol` symlinks. |
| `data/raw/<source>/ingestion_manifest.json` | `manifest.save()` | The per-source pull record with SHA-256 per file. |
| `data/raw/<source>/.post_clone_done` (git only) | `git_connector.py:41` | Idempotency marker for `post_clone_cmd`. |
| `data/analysis/freshness_report.md` | `freshness.py:61` | Pin staleness report (git pins + slither-analyzer version). |

## 6. Pipeline interactions

| Stage | Direction | What |
|-------|-----------|------|
| `config.yaml` | → | Source definitions parsed by `ingest.py:_source_config()` |
| `connectors/` (this subpackage) | → | Per-source pull logic — see `connectors/README.md` |
| Stage 1b (preprocessing) | → | Reads `data/raw/<source>/repo/*.sol` (via the manifest's `files` list) |
| `sentinel_data.preprocessing.preprocess` | → | Calls `_maybe_folderize` to invoke `label_folderize.folderize_by_labels` for sources with `labels_csv` set |
| Stage 4 (verification, manifest verify) | → | `manifest.verify_manifest()` is the load-time gate |

## 7. Tests

**Location:** `data_module/tests/test_ingestion/`
- `test_connector.py` — base class + registry + `find_sol_files` include/exclude
- `test_manifest.py` — SHA-256 verification + append-only semantics
- `test_label_folderize.py` — flat-vs-folderized layout

**Command:**
```bash
cd ~/projects/sentinel/data_module
poetry run pytest tests/test_ingestion/ -v
```

**Top-level integration tests** (also relevant):
- `tests/test_integration_solidifi.py` — full end-to-end on SolidiFI
- `tests/test_integration_dive.py` — full end-to-end on DIVE (22K files)

## 8. See also

- Connectors: `sentinel_data/ingestion/connectors/README.md`
- CLI entry: `sentinel_data/cli.py` (`_run_ingest` at line 111, `run_freshness_check` at line 670)
- Next stage: `sentinel_data.preprocessing/README.md`
- Stage 1 plan: `docs/proposal/Data_Module_Proposals/actionable_plans/02_stage_1_ingestion_preprocessing.md`
- One-way dependency rule: ADR-0001 in `docs/decisions/`
- Slither API drift context: ADR-0002 (Run 9 breakage + the freshness check's purpose)
