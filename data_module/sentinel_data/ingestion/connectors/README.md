# `sentinel_data.ingestion.connectors` — Strategy-Pattern Source Connectors

> **Status: 5 implemented + 2 stub-aliased, 3 hard-stubs.** Fully usable for git + manual sources; HF/Zenodo/Etherscan raise `NotImplementedError`.

## 1. Purpose

This subpackage implements the **strategy pattern** for ingesting raw Solidity contracts from heterogeneous source types (git repos, HuggingFace datasets, Zenodo records, Etherscan API, pre-downloaded directories/zips). Every data source declared in `config.yaml` maps to exactly one connector; the connector handles the type-specific pull, and downstream code (`ingest.py`, `manifest.py`) is identical for all sources.

The connector system is the **seam between "data shape" and "pipeline"** — adding a new corpus (e.g. a HuggingFace dataset) means dropping in one new `*_connector.py` file that satisfies `BaseConnector`, plus one entry in `_REGISTRY`. Nothing else changes.

## 2. Source map

| File | Lines | Role |
|------|-------|------|
| `__init__.py` | 38 | `_REGISTRY` (5 types + 2 aliases) + `get_connector()` factory. Public exports: `BaseConnector`, `ConnectorError`, `PullResult`, `SourceConfig`, `GitConnector`, `get_connector`. |
| `base.py` | 95 | Abstract `BaseConnector` + `SourceConfig` / `PullResult` dataclasses + `ConnectorError` exception + `find_sol_files()` static method. |
| `git_connector.py` | 89 | `GitConnector` — clones a repo (shallow or full) at a pinned commit. **Fully implemented.** |
| `manual_connector.py` | 189 | `ManualConnector` + `materialize_staging()` + `_extract_zip()` — handles pre-downloaded dirs/zips. **Fully implemented.** |
| `huggingface_connector.py` | 13 | `HuggingFaceConnector` — **STUB**, raises `NotImplementedError`. |
| `zenodo_connector.py` | 13 | `ZenodoConnector` — **STUB**, raises `NotImplementedError`. |
| `etherscan_connector.py` | 13 | `EtherscanConnector` — **STUB**, raises `NotImplementedError`. |

**Sub-total: 450 lines** (the existing `ingestion/README.md` says "~350" — corrected to 450 from `find`/line counts).

## 3. Key concepts

### The `SourceConfig` dataclass (`base.py:12-24`)

Every connector receives the same parsed config. The dataclass captures **all** the fields a connector might need — connectors ignore the ones they don't use. The `extra: dict[str, Any]` catch-all lets source-specific keys (`staging_path`, `post_clone_cmd`, …) flow through without schema changes.

| Field | Type | Used by |
|-------|------|---------|
| `name` | `str` | All (logging) |
| `connector` | `str` | All (registry lookup) |
| `url` | `str` | `GitConnector`, future Etherscan |
| `pin` | `str` | `GitConnector` (commit SHA), `ManualConnector` (version string) |
| `hf_dataset` | `str` | `HuggingFaceConnector` (when implemented) |
| `zenodo_record` | `str` | `ZenodoConnector` (when implemented) |
| `include_subdirs` / `exclude_subdirs` | `list[str]` | All (via `find_sol_files`) |
| `extra` | `dict` | Source-specific (e.g. `staging_path`, `post_clone_cmd`) |

### The 5+2 connector types (`__init__.py:10-18`)

```python
_REGISTRY = {
    "git":          GitConnector,        # ← the workhorse
    "huggingface":  HuggingFaceConnector,  # stub
    "zenodo":       ZenodoConnector,     # stub
    "etherscan":    EtherscanConnector,  # stub
    "manual":       ManualConnector,     # ← the other workhorse
    "audit_report": ManualConnector,     # alias
    "rekt_scraper": ManualConnector,     # alias
}
```

**Three source types are aliased to `ManualConnector` for future-proofing** — they share the "pre-downloaded data on disk" contract. Use the alias that best describes the source (e.g. for the audit-report corpus from `rekt_scraper`, set `connector: rekt_scraper` in `config.yaml` even though it routes to the same class as `manual`).

### `BaseConnector.find_sol_files()` (`base.py:64-94`)

Static method shared by all connectors. The **include/exclude** logic is a two-level filter:

1. **Include (allowlist)** — if `include_subdirs` is non-empty, only descend into those top-level subdirs of `root`. Used for repos whose root mixes contracts with tool output (e.g. SolidiFI's `results/` containing Mythril/Slither analyses — you only want `buggy_contracts/`).
2. **Exclude (blocklist)** — skip these top-level subdirs. Applied AFTER include.

Walk is `rglob("*.sol")` then filter — i.e. recursive. A path that doesn't match any include rule is walked from `root` itself.

### `GitConnector` clone strategy (`git_connector.py:59-69`)

```python
if cfg.pin:
    # Full clone so we can checkout an arbitrary commit (audit trail)
    _run(["git", "clone", "--quiet", cfg.url, str(repo_dir)])
    _run(["git", "checkout", cfg.pin], cwd=repo_dir)
else:
    # Shallow clone for speed (HEAD at fetch time)
    _run(["git", "clone", "--depth", "1", "--quiet", cfg.url, str(repo_dir)])
```

- **With `pin`**: full clone + `git checkout <pin>`. Required so the SHA is verifiable — a shallow clone of `HEAD` doesn't let you `checkout` an arbitrary commit.
- **Without `pin`**: shallow clone of `HEAD`. Faster, but the SHA is "whatever HEAD was at fetch time" (recorded in `resolved_pin`).

The connector is **idempotent** — if `repo_dir` already exists, it reuses the clone and just records the current commit as `resolved_pin`. Use `--force` semantics (a manual `rm -rf` of the raw dir) to re-pull.

### `ManualConnector` 3-mode staging (`manual_connector.py:103-158`)

`materialize_staging()` handles three cases based on what `staging_path` points to:

| Case | Detection | Behavior |
|------|-----------|----------|
| **ZIP file** | `staging.is_file() and suffix == ".zip"` | Extract (mode=`symlink` → extract to `__zip_extracted/`, then symlink; mode=`copy` → extract in place) |
| **Directory** | `staging.is_dir()` | Symlink (default) or copy entire dir to `repo_dir` |
| **Glob pattern** | Path contains `*?[` | Resolve to a single path; raise if 0 or >1 matches |

**Mode `symlink` (default)**: fast, no data movement, reversible. Doesn't work across filesystems. **Mode `copy`**: needed for Docker or cross-FS staging. For ZIPs, the `symlink` mode actually extracts the zip and symlinks the extracted dir (there's nothing to symlink to a zip).

`_extract_zip()` (`manual_connector.py:161-189`) explicitly skips `__MACOSX/` and `.DS_Store` — the DIVE zip has 44,687 entries but only 22,332 are `.sol`; the rest are macOS resource forks. Also defends against zip-slip (the classic "extract to `../../etc/passwd`" attack).

## 4. Public API

### `SourceConfig` — `base.py:12-24`

```python
@dataclass
class SourceConfig:
    name: str
    connector: str
    url: str = ""
    pin: str = ""
    hf_dataset: str = ""
    zenodo_record: str = ""
    description: str = ""
    include_subdirs: list[str] = field(default_factory=list)
    exclude_subdirs: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)
```

### `PullResult` — `base.py:27-36`

```python
@dataclass
class PullResult:
    source: str
    local_dir: Path
    resolved_pin: str
    sol_files: list[Path]
    fetched_at: str
    duration_s: float
    extra: dict[str, Any] = field(default_factory=dict)
```

### `ConnectorError` — `base.py:39-40`

```python
class ConnectorError(Exception):
    """Raised when a connector cannot complete a pull."""
```

### `BaseConnector` (abstract) — `base.py:43-94`

```python
class BaseConnector(ABC):
    def pull(self, cfg: SourceConfig, dest: Path) -> PullResult:
        """Template method — wraps _pull with timing + timestamp."""

    @abstractmethod
    def _pull(self, cfg: SourceConfig, dest: Path) -> PullResult:
        """Subclass implements the actual pull logic."""

    @staticmethod
    def find_sol_files(root: Path, include_subdirs=None, exclude_subdirs=None) -> list[Path]:
        """Walk `root` for *.sol, applying include/exclude filters."""
```

### `get_connector()` — `__init__.py:21-32`

```python
def get_connector(connector_type: str) -> BaseConnector:
    """Instantiate and return the connector for *connector_type*.
    Raises ConnectorError if the type is not in the registry.
    """
```

### `materialize_staging()` — `manual_connector.py:103-158`

```python
def materialize_staging(staging: Path, repo_dir: Path, extra: dict, source_name: str) -> None:
    """Materialize `staging` (dir/zip/glob) into `repo_dir`.
    Three cases (zip/dir/glob) with symlink|copy modes. See above.
    """
```

## 5. Inputs → outputs

| Input | Where | What |
|-------|-------|------|
| `cfg` (config dict) | `data/config.yaml` → `sentinel_data.ingestion.ingest._source_config()` | Parsed into `SourceConfig` |
| `cfg["sources_critical_path"][name]["enabled"]` | `config.yaml` | `True` to include, `False` to skip |
| `cfg["sources_critical_path"][name]["connector"]` | `config.yaml` | One of `git` / `huggingface` / `zenodo` / `etherscan` / `manual` / `audit_report` / `rekt_scraper` |
| `cfg["sources_critical_path"][name]["url"]` + `["pin"]` | `config.yaml` | For `git` connector |
| `cfg["sources_critical_path"][name]["extra"]["staging_path"]` | `config.yaml` | For `manual` connector |
| `dest` (Path) | `ingest.py:79` | Per-source raw dir, e.g. `data/raw/defihacklabs/` |

| Output | Where | What |
|--------|-------|------|
| `dest/repo/` | `git_connector.py:30` or `manual_connector.py:64` | The cloned/materialized source. The rest of the pipeline reads `.sol` files from here (or from subdirs as filtered by `find_sol_files`). |
| `dest/ingestion_manifest.json` | `ingest.py:91` | SHA-256 per file + pin + timestamp. See `manifest.py`. |
| `dest/.post_clone_done` (git only) | `git_connector.py:41` | Marker so `post_clone_cmd` is idempotent. |

## 6. Pipeline interactions

| Stage | Direction | What |
|-------|-----------|------|
| `config.yaml` | → | Source definitions parsed by `_source_config()` (`ingest.py:29-46`) |
| `sentinel_data.ingestion.ingest` | ↔ | Orchestrator that calls `get_connector()` and writes the manifest |
| `sentinel_data.ingestion.manifest` | → | `IngestionManifest` consumed by every downstream stage's `verify_manifest()` |
| Stage 2 (preprocess) | ← | Reads `data/raw/<source>/repo/`, filters `.sol` files via `find_sol_files` |

## 7. Tests

**Location:** `data_module/tests/test_ingestion/test_connector.py` (and `test_manifest.py`, `test_label_folderize.py`).

**Command:**
```bash
cd ~/projects/sentinel/data_module
poetry run pytest tests/test_ingestion/ -v
```

**Coverage:**
- `test_connector.py` — base class, registry, `find_sol_files` include/exclude behavior
- `test_manifest.py` — SHA-256 verification, append-only semantics
- `test_label_folderize.py` — the `__source__/` flat-vs-folderized layout

## 8. See also

- Parent: `sentinel_data.ingestion/README.md`
- Orchestrator: `sentinel_data.ingestion.ingest`
- Manifest: `sentinel_data.ingestion.manifest`
- Stage 1 plan: `docs/proposal/Data_Module_Proposals/actionable_plans/02_stage_1_ingestion_preprocessing.md`
- Connector design rationale: ADR-0001 (one-way dependency) + ADR-0002 (Slither API drift)
- DeFiHackLabs use case: `flattener.py` was added because 717/738 DeFiHackLabs PoCs import `forge-std/Test.sol` (not in the cloned repo) — see `flattener.py:9-13`
- DIVE integration: `manual_connector.py:24-29` for the DIVE zip staging rationale
