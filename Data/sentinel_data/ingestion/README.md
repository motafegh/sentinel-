# `sentinel_data.ingestion` — Pulling Raw Contracts from the Wild

## What This Module Does

The ingestion module is Stage 1 of the SENTINEL data pipeline. Its job is deceptively simple: **pull raw `.sol` files from upstream sources, verify their integrity, and record exactly what was fetched and when.**

Think of it as the "front door" of the pipeline. Every contract that eventually becomes part of the training dataset first passes through this module. The contracts arrive in different formats — git repos, HuggingFace datasets, Zenodo archives, or manual uploads — and this module normalizes them into a uniform directory structure with SHA-256 verification.

## Why This Matters

In machine learning, your model is only as good as your data. If you can't answer "what exact version of this dataset did we train on?" six months from now, you have a reproducibility problem. The ingestion module solves this by:

1. **Pinning every source to a specific commit/version** — no floating `HEAD` or `latest`
2. **Computing SHA-256 hashes for every downloaded file** — any tampering is detectable
3. **Writing an append-only ingestion manifest** — the audit trail is permanent

The BCCC dataset (the predecessor to this pipeline) had a 38.8% duplication rate across sources and no way to reconstruct which version of each source was used. This module prevents that class of failure.

## Architecture Overview

```
config.yaml (source definitions)
        │
        ▼
┌─────────────────────────────────────────┐
│           ingest.py (orchestrator)       │
│  Reads config → picks connector →        │
│  calls pull() → writes manifest          │
└──────────────┬──────────────────────────┘
               │
    ┌──────────┴──────────┐
    │                     │
    ▼                     ▼
┌─────────┐       ┌──────────────┐
│ connectors/ │       │ manifest.py  │
│ (strategy   │       │ (SHA-256     │
│  pattern)   │       │  verify)     │
└─────────────┘       └──────────────┘
```

### Key Files

| File | Purpose | Lines |
|------|---------|-------|
| `ingest.py` | Orchestrator — coordinates connector + manifest for one or all sources | 107 |
| `manifest.py` | Per-source pull record with SHA-256 per-file verification | 98 |
| `freshness.py` | Compares pinned versions to upstream HEAD; checks slither-analyzer version | 119 |
| `label_folderize.py` | Creates per-class folder symlinks from label CSVs (DIVE, SmartBugs Wild, Bastet) | 172 |
| `connectors/` | Strategy pattern — 5 connector types for different source formats | ~350 |

## The Connector System

The connector system follows the **Strategy pattern**. Every data source plugs into one of 5 connector types:

```python
# connectors/__init__.py
_REGISTRY = {
    "git": GitConnector,
    "huggingface": HuggingFaceConnector,
    "zenodo": ZenodoConnector,
    "etherscan": EtherscanConnector,
    "manual": ManualConnector,       # also handles "audit_report" and "rekt_scraper"
}
```

### `BaseConnector` (the abstract interface)

Every connector inherits from `BaseConnector`, which provides:
- A `pull(cfg, dest)` template method that handles timing and error wrapping
- An abstract `_pull(cfg, dest)` method that subclasses implement
- A `find_sol_files(root)` static method that walks a directory tree for `.sol` files

The `SourceConfig` dataclass defines what a connector needs:
- `name`, `connector`, `url`, `pin` (the basics)
- `hf_dataset`, `zenodo_record` (source-type-specific)
- `include_subdirs`, `exclude_subdirs` (directory filtering)
- `extra` (catch-all for source-specific config)

### The 5 Connector Types

| Connector | Status | What It Does |
|-----------|--------|--------------|
| `GitConnector` | **Fully implemented** | Clones a git repo at a pinned commit; supports shallow clones (no pin) and full clones (with pin); runs optional `post_clone_cmd` |
| `ManualConnector` | **Fully implemented** | Handles manually-downloaded data; supports directories, `.zip` files, and glob patterns; includes zip-slip defense |
| `HuggingFaceConnector` | **Stub** | Will load HuggingFace datasets; raises `NotImplementedError` |
| `ZenodoConnector` | **Stub** | Will download Zenodo records; raises `NotImplementedError` |
| `EtherscanConnector` | **Stub** | Will fetch contracts from Etherscan; raises `NotImplementedError` |

### How `GitConnector` Works

This is the most important connector — it handles the majority of sources (ScaBench, Web3Bugs, Bastet, DeFiHackLabs, SmartBugs, OpenZeppelin, Messi-Q, and more):

```python
class GitConnector(BaseConnector):
    def _pull(self, cfg, dest):
        # 1. Clone the repo (shallow if no pin, full if pin set)
        self._clone(cfg.url, dest, cfg.pin)
        # 2. Run optional post-clone command (e.g. checkout_sources.py)
        if cfg.extra.get("post_clone_cmd"):
            subprocess.run(cfg.extra["post_clone_cmd"], shell=True, cwd=dest)
        # 3. Find all .sol files
        sol_files = self.find_sol_files(dest, cfg.include_subdirs, cfg.exclude_subdirs)
        return PullResult(source=cfg.name, local_dir=dest, sol_files=sol_files, ...)
```

### How `ManualConnector` Works

For sources that require manual download (e.g. SolidiFI benchmark ZIP files):

```python
class ManualConnector(BaseConnector):
    def _pull(self, cfg, dest):
        staging = cfg.extra["staging_path"]  # where the user put the data
        materialize_staging(staging, dest, cfg.extra, cfg.name)
        # Handles: zip extraction, directory symlink/copy, glob resolution
```

The `materialize_staging` function handles 3 cases:
1. **ZIP files** — safe extraction with zip-slip defense and macOS metadata stripping
2. **Directories** — symlink or copy into the destination
3. **Glob patterns** — resolve matching files from a base directory

## The Ingestion Manifest

Every `sentinel-data ingest --source <name>` run produces an `ingestion_manifest.json`:

```json
{
  "source": "defihacklabs",
  "connector": "git",
  "url": "https://github.com/DeFiHackLabs/DeFiHackLabs",
  "pin": "abc123...",
  "resolved_pin": "abc123...",
  "fetched_at": "2026-06-09T10:30:00Z",
  "duration_s": 45.2,
  "contract_count": 342,
  "files": [
    {"path": "src/test/2024-01/flashloan_exp.sol", "sha256": "def456...", "size_bytes": 1234},
    ...
  ]
}
```

The manifest is **append-only** — past ingestions are never deleted, only new ones are added. This is the audit trail that lets you answer "what version of DeFiHackLabs did Run 11 train on?" six months from now.

### SHA-256 Verification

The `verify_manifest(raw_dir, source)` function re-checks every SHA-256 against the current files on disk. If any file's hash has changed (upstream force-push, accidental edit, disk corruption), the verification fails loud.

## The Freshness Checker

The `freshness.py` module compares each pinned source version against the upstream HEAD:

- **Git sources** — uses `git ls-remote` to get the latest commit SHA
- **Slither-analyzer** — checks PyPI for the latest version vs the installed version
- **HF/Zenodo** — will query their APIs (currently stubbed)

The output is a `freshness_report.md` that lists "behind by N commits/versions" per source. The report is informational, not blocking — it triggers a human review process to decide whether to bump the pin.

## Label-Aware Folderization

For sources that distribute labels in separate CSV files (DIVE, SmartBugs Wild, Bastet), the `label_folderize.py` module creates per-class folder symlinks. This is useful for:

- Sources where labels are in a CSV, not in folder names
- Multi-label contracts (a contract can appear in multiple class folders)
- Idempotent re-runs (symlinks are checked before creation)

The `FolderizationResult` dataclass tracks what happened:
- `contracts_seen` — total contracts processed
- `symlinks_created` — new symlinks created
- `classes_present` — which classes were found
- `multi_label` — contracts with multiple labels

## How to Use

```bash
# Ingest a single source
sentinel-data ingest --source defihacklabs

# Ingest all enabled sources
sentinel-data ingest

# Dry-run (print planned action without executing)
sentinel-data ingest --source scabench --dry-run

# Check source freshness
sentinel-data freshness
```

## Configuration

Sources are defined in `config.yaml` under `sources_critical_path` and `sources_additive`:

```yaml
sources_critical_path:
  defihacklabs:
    enabled: true
    connector: git
    url: https://github.com/DeFiHackLabs/DeFiHackLabs
    pin: <commit-sha>
    description: "DeFi exploit PoCs"
    include_subdirs: ["src/test"]
```

The `enabled: true` flag controls whether the source is processed in the pipeline. Disabled sources are preserved in the config but skipped during ingestion.

## What This Module Does NOT Do

- It does not parse or analyze contract contents (that's preprocessing)
- It does not extract representations (that's representation)
- It does not assign labels (that's labeling)
- It does not verify label correctness (that's verification)

The ingestion module is a **pure data movement layer**. Its only concern is getting the right files to the right place with the right hashes.

## Pipeline Position

```
Stage 0: Skeleton (package setup)
    ↓
Stage 1: Ingestion + Preprocessing ← YOU ARE HERE
    ↓
Stage 2: Representation
    ↓
Stage 3: Labeling
    ↓
Stage 4: Verification
    ↓
Stage 5: Splitting + Registry
    ↓
Stage 6: Analysis
    ↓
Stage 7: Export + Seam Swap
    ↓
Stage 8: Run 11 Launch
```
