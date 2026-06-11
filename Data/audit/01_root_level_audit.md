# Audit Report — Root-Level Files (Stage 0)

**Scope:** `cli.py`, `__init__.py`, `pyproject.toml`, `config.yaml`, `dvc.yaml`, `README.md`, `pytest.ini`, `docker/Dockerfile.data`, `docker/.dockerignore`
**Plan Reference:** `01_stage_0_skeleton.md` (D-0.1 through D-0.10)

---

## 1. `sentinel_data/__init__.py`

| Check | Status | Detail |
|-------|--------|--------|
| Package boundary (D-0.2) | PASS | No sentinel-ml reference |
| Version declared | PASS | `__version__ = "0.1.0"` matches pyproject.toml |
| Docstring | PASS | One-way dependency rule stated |

**No issues.**

---

## 2. `sentinel_data/cli.py` (413 lines)

| Check | Status | Line(s) | Detail |
|-------|--------|---------|--------|
| CLI surface (D-0.7) | PASS | — | `run --from-stage` + per-stage subcommands |
| `99%%` escape | PASS | 58 | Fixed correctly |
| `yaml.safe_load` | PASS | 70 | No `yaml.load()` usage |
| Lazy imports in stage fns | PASS | 83,103,132,249 | Only imports when stage is invoked |
| Type hints | PASS | — | All functions typed |
| `_handle_run` arg injection | **WARN** | 386-389 | Hardcodes `source=None, workers=1, sample=None, retry_failed=False` — `sentinel-data run --source solidifi` silently ignores `--source` |
| `freshness` not in STAGES | **WARN** | 42-52 | Utility command exists but not in `STAGES` list or `dvc.yaml` |
| Docstring stale | **WARN** | 3 | Says "all stage implementations are placeholders" but 3 of 9 are real |

### WARN-1: `_handle_run` drops stage-specific args

```python
# Line 386-389
args = argparse.Namespace(
    config=parsed.config,
    dry_run=parsed.dry_run,
    source=None,        # ← dropped from CLI
    workers=1,          # ← dropped from CLI
    sample=None,        # ← dropped from CLI
    retry_failed=False, # ← dropped from CLI
)
```

Running `sentinel-data run --source solidifi` silently ignores `--source`. The `run` subcommand cannot pass stage-specific flags to individual stages.

**Impact:** MEDIUM — users must run stages individually to use `--source`.

---

## 3. `pyproject.toml` (60 lines)

| Check | Status | Detail |
|-------|--------|--------|
| Package boundary (D-0.2) | PASS | No sentinel-ml in any dep group |
| Python version | PASS | `>=3.12,<3.13` |
| Entry point | PASS | `sentinel-data = "sentinel_data.cli:main"` |
| Dependency groups | PASS | Core / pipeline / ml / dev separation |
| Build system | PASS | poetry-core |

### WARN: Tsinghua PyPI mirror

Lines 53-56 set `pypi.tuna.tsinghua.edu.cn` as `priority = "primary"`. This is a regional mirror that may be slow or unavailable outside China. Should be documented or made configurable.

---

## 4. `config.yaml` (360 lines)

| Check | Status | Line(s) | Detail |
|-------|--------|---------|--------|
| Config-as-data (D-0.6) | PASS | — | Single source of truth |
| MLflow uri (F20) | PASS | 36 | `sqlite:///mlruns.db` |
| Solc baselines (F14) | PASS | 39-44 | 6 versions present |
| BCCC deferred | PASS | 351 | `deferred_sources.bccc` exists |
| Dedup threshold | PASS | 46 | `0.85` correct |
| Verification thresholds | PASS | 48-50 | Present |
| No sentinel-ml refs | PASS | — | Verified |
| FORGE URL | PASS | 213 | `shenyimings/FORGE-Artifacts` correct |
| scabench URL | PASS | 334 | `scabench-org/scabench` correct |

### FAIL-1: Hardcoded absolute paths

```
Line 125: staging_path: "/home/motafeq/projects/sentinel/Data/data/raw_staging/dive"
Line 140: labels_csv: "/home/motafeq/projects/sentinel/Data/data/raw_staging/dive_labels/DIVE_Labels.csv"
```

These are user-specific and break on any other machine, CI, or Docker. Should use relative paths (`data/raw_staging/dive`).

### FAIL-2: `defihacklabs` has `enabled: false`

DeFiHackLabs is a critical-path source (the highest-confidence Tier-1 source after ScaBench). The Stage 1 plan requires it enabled for the end-to-end test. This appears to be an oversight.

### WARN: Empty pins reduce reproducibility

| Source | Line | Pin |
|--------|------|-----|
| smartbugs_curated | 170 | `""` |
| web3bugs | 180 | `""` |
| disl | 189 | `""` |
| All additive sources | various | `""` |

The plan says pins are "placeholders" but for reproducibility, git-pinned sources should have commit SHAs. This is deferred to Stage 1 but should be tracked.

---

## 5. `dvc.yaml`

| Check | Status | Detail |
|-------|--------|--------|
| 9-stage DAG | PASS | All stages defined with deps/outs |
| Placeholder commands | PASS | Each stage has a placeholder `cmd:` |
| `.gitkeep` outputs | PASS | Each stage references `.gitkeep` |

**No issues.**

---

## 6. `README.md`

| Check | Status | Detail |
|-------|--------|--------|
| Module description | PASS | Clear purpose statement |
| Installation | PASS | `poetry install` instructions |
| Quickstart | PASS | `sentinel-data run --dry-run` |
| Directory map | PASS | Full tree |
| 9-stage descriptions | PASS | One-liner per stage |
| WSL2 caveats | PASS | `wsl -- bash -c '...'` pattern |
| MLflow backend | PASS | `sqlite:///mlruns.db` noted |
| Schema version | PASS | v9 documented |

**No issues.**

---

## 7. `docker/Dockerfile.data`

| Check | Status | Detail |
|-------|--------|--------|
| Base image | PASS | `python:3.12.1-bookworm` (not slim) |
| 6 solc baselines | PASS | Installed via `solc-select install` |
| Poetry install | PASS | Layer-cached |
| Entrypoint | PASS | `sentinel-data` |

**No issues.**

---

## 8. `pytest.ini`

| Check | Status | Detail |
|-------|--------|--------|
| testpaths | PASS | `tests` |
| addopts | PASS | `-v --tb=short` |

**No issues.**

---

## Summary

| Status | Count |
|--------|-------|
| PASS | 31 |
| WARN | 9 |
| FAIL | 2 |

**FAIL items:**
1. Hardcoded absolute paths in config.yaml (breaks portability)
2. `defihacklabs` disabled despite being critical-path source

**Top WARN items:**
1. `_handle_run` drops stage-specific args (MEDIUM impact)
2. Tsinghua PyPI mirror as primary (low impact for non-China users)
3. Empty pins on 20+ sources (reproducibility concern)
