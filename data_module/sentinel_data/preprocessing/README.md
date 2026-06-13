# `sentinel_data.preprocessing` — Stage 1b: From Raw Source to Clean Contracts

> **Status: ✅ Fully implemented** (9 files, ~1,190 lines). Serial + parallel pipelines both shipped. Two-pass compilation with pragma tolerance; three-level dedup (SHA-256 → address → AST near-dup stub); normalize; version-bucket segment.

## 1. Purpose

This is **the quality gate of the entire system**. It takes raw `.sol` files from Stage 1a and runs them through a **5-step transformation pipeline** that makes them ready for representation extraction. The output is a clean, deduplicated, normalized, version-bucketed contract with a `meta.json` sidecar recording every transformation.

The pipeline is **drop-not-fix for compile failures**: a contract that fails to compile is dropped, not passed through with a warning. A near-duplicate is merged, not stored twice. A comment-heavy file is stripped to its essential structure. The BCCC dataset (the predecessor) had a 38.8% duplication rate and no compile-time quality gate — contracts that couldn't compile were silently reclassified as `NonVulnerable`, conflating "clean" with "unparseable." This module prevents that.

## 2. Source map

| File | Lines | Role |
|------|-------|------|
| `__init__.py` | 5 | Module docstring + `__all__`. |
| `pipeline.py` | 246 | `PreprocessingPipeline` orchestrator + `ContractMeta` sidecar dataclass + `META_SCHEMA_VERSION = "1"`. **Serial** execution. |
| `preprocess.py` | 264 | `preprocess_source(name, cfg, data_dir, dry_run, n_workers, sample, retry_failed)` + `preprocess_all(...)` — the CLI service. Handles `--sample N`, `--retry-failed`, and `_maybe_folderize`. |
| `flattener.py` | 311 | `flatten_contract(sol_path) -> FlattenResult` — `solc --flatten` with two-stage fallback (recursive unresolved-import strip + inheritance-parent strip). |
| `compiler.py` | 191 | `compile_contract(sol_path) -> CompileResult` — two-pass solc (exact requested → nearest available). Resolves pragma `^ 0.4 .9` (spaced) and exact-version pragmas. |
| `deduplicator.py` | 85 | `Deduplicator.process(content, path) -> DedupRecord` — level 1 (SHA-256) + level 2 (Ethereum address). Level 3 (AST near-dup) is **stubbed** — see §3. |
| `normalizer.py` | 39 | `normalize(source) -> NormalizeResult` — strip SPDX / line + block comments / trailing whitespace / collapse blank lines. |
| `segmenter.py` | 68 | `segment_and_bucket(source, pragma) -> Segment` — version bucket (`legacy` < 0.6, `transitional` 0.6–0.7, `modern` ≥ 0.8) + `has_unchecked_block` detection. |
| `parallel.py` | 156 | `run_preprocess_parallel(pipeline, sol_files, raw_base, n_workers)` — `multiprocessing.Pool` wrapper with auto-tuned chunksize. |
| `_transitive_strip.py` | 104 | Helper for `flattener.py`: when the recursive strip modifies a transitive relative-imported file, writes a `.sentinel_stripped.sol` sibling and rewrites the top-level import. Auto-cleaned by `pipeline.py:_process_one`. |

**Sub-total: 1,498 lines** across 10 Python files.

## 3. Key concepts

### The 5-step pipeline (`pipeline.py:80-216`)

```
Raw .sol file
    │
    ▼
┌─────────────┐
│ 1. Flatten   │  Resolve import chains → single file (solc --flatten
└──────┬──────┘  with fallback to recursive unresolved-import strip)
       │
       ▼
┌─────────────┐
│ 2. Compile   │  Two-pass: exact pragma → nearest available
└──────┬──────┘  (drop on failure with reason recorded)
       │
       ▼
┌─────────────┐
│ 3. Dedup     │  SHA-256 → address-level
└──────┬──────┘  (AST near-dup @ 0.85 STUB — requires Slither)
       │
       ▼
┌─────────────┐
│ 4. Normalize │  Strip comments, SPDX, whitespace
└──────┬──────┘
       │
       ▼
┌──────────────────────┐
│ 5. Segment + Bucket  │  Version tag: legacy/transitional/modern
└──────────────────────┘  + has_unchecked_block detection
       │
       ▼
  Clean .sol + meta.json
```

### Drop-not-fix policy (`pipeline.py:165-173`)

Files that fail both compilation passes are dropped. The compile error and attempted versions are recorded in the meta.json sidecar AND a `dropped.csv` file. **This is a deliberate design choice** — an unparseable contract cannot be reliably represented as a graph. BCCC had 8,232 contracts that didn't compile cleanly; they were silently reclassified as `NonVulnerable`, which is not the same as clean.

Drop reasons (per `pipeline.py:166-185`):
- `compile_failed` — both compilation passes failed
- `duplicate` — exact SHA-256 or Ethereum-address match
- (No `flatten_failed` reason — flattener has 3 fallback paths and rarely drops)

### Two-pass compilation (`compiler.py:43-101`)

**Pass 1**: try the exact solc version from the pragma.
**Pass 2**: if Pass 1 fails, try the nearest available version from `~/.solc-select/artifacts/`.

This recovers contracts with unusual pragma syntax (spaced pragmas like `^ 0.4 .9`, exact-version pragmas like `0.4.25` compiled with the wrong version). The compiler strips whitespace before regex matching (AUDIT_PATCHES 1-P1, 1-P2).

`_extract_pragma()` (`compiler.py:106-113`) strips internal whitespace. `_satisfying_versions()` (`compiler.py:129-150`) returns the available versions that satisfy the constraint, newest first. The candidates are tried in order, and the last error message is the one returned on full failure (most informative).

### The flattener's two-stage fallback (`flattener.py:58-134`)

Solidity files often import from other files. The flattener resolves these chains:

1. **Try `solc --flatten`** with the version matching the file's pragma.
2. **If that fails, try stripping unresolvable imports** as a second fallback. The compile step will succeed if the only unresolvable imports are external dependencies (forge-std, hardhat, openzeppelin) and the rest of the file is self-contained.
3. **Recursion** — when a relative import DOES resolve on disk, we also process that file's imports (since the relative import will pull in its own transitive imports at compile time). This is what catches the DeFiHackLabs pattern where `interface.sol` and `basetest.sol` are shared relative files that themselves import `forge-std`.

The strip also handles `contract Foo is A, B, C {` — if a stripped import brought in named symbols (`Test` from forge-std), the inheritance list is also stripped. Otherwise the compile would still fail with "Identifier not found."

**The forge-std trick** — when a bare `import "forge-std/Test.sol";` is stripped, we conservatively assume the common forge-std symbols (`Test`, `console`, `console2`, `Vm`, `ScriptUtils`) are brought in. The compile will fail fast on any actually-undefined symbol we missed, so being too aggressive just costs a few extra dropped files, not silent corruption. (The Phase 5 retry recovered 717/738 DeFiHackLabs PoCs that were dropped before this fallback was added — see `flattener.py:9-13`.)

### Transitive strip sibling files (`_transitive_strip.py:31-104`)

When the recursive strip modifies a transitive relative-imported file (e.g. `interface.sol` that itself imported `forge-std/Test.sol`), the compiler must see the modified version. We can't safely modify the on-disk file (it might be used by other PoCs in the same source), so we write a sibling file with a `.sentinel_stripped.sol` suffix and rewrite the top-level import to point at it. The sibling is auto-cleaned by `pipeline.py:_process_one:159-163` after the compile step.

### Three-level dedup (`deduplicator.py:32-113`)

| Level | Method | What it catches | Status | 101:
|-------|--------|-----------------|--------|
| 1 | Exact SHA-256 | Whitespace/comment-only differences across sources | ✅ |
| 2 | Ethereum address | Same address appearing in multiple files | ✅ |
| 3 | AST near-dup (text-normalized) | Copy-paste-with-minor-edits patterns | ✅ (text-normalized hash; Slither-based clustering deferred to v2.1) |

The 0.85 threshold is intentional — BCCC's 38.8% duplication rate was at 0.85–0.95 similarity; 0.92 is too strict and misses the "minor edits" cases that caused the duplication.

### Normalization (`normalizer.py:25-38`)

Strips:
- SPDX license headers
- Block comments `/* ... */`
- Line comments `// ...`
- Trailing whitespace
- Multiple blank lines (collapsed to 1)

**Preserves `now`** (Solidity's alias for `block.timestamp`). The A9 fix in the graph extractor relies on detecting `now` in the source; stripping it would break that detection.

### Version bucketing (`segmenter.py:35-54`)

| Bucket | Solidity Version | Characteristics |
|--------|------------------|-----------------|
| `legacy` | < 0.6 | Pre-SafeMath era; integer overflow is implicit |
| `transitional` | 0.6–0.7 | Bridge era; limited `unchecked{}` support |
| `modern` | ≥ 0.8 | Full `unchecked{}` support; built-in overflow checks |

Also records `has_unchecked_block` for 0.8+ files (relevant for IntegerUO detection in Stage 4; this was `feat[11] / in_unchecked_block` in the v9 schema).

### Parallel processing (`parallel.py:80-156`)

For large sources (DIVE = 22,330 files, Web3Bugs = ~3,500, SmartBugs Wild = 47K), serial processing is 6+ hours. `run_preprocess_parallel` cuts this by N (worker count).

- Module-level picklable worker (lambdas and bound methods don't pickle)
- `mp.Pool(processes=n_workers).imap(worker, batch, chunksize=...)`
- chunksize auto-tuned: `max(1, total // (n_workers * 16))` (16 chunks/worker sweet spot)
- Defaults: `n_workers = min(os.cpu_count(), 8)` — cap at 8 even on bigger boxes because solc subprocesses are themselves multi-threaded
- Dedup is best-effort (not race-free) — duplicates are dropped, not corrupted, so a race here is acceptable

> Note: serial is the default. Pass `--workers N` to the CLI to use parallel.

### Incremental retry: `--retry-failed` (`preprocess.py:135-198`)

Build-system-style incremental: install solc 0.7.4 → run `sentinel-data preprocess --source solidifi --retry-failed` → 50 files move from dropped to preprocessed. No need to reprocess 22,000 files.

The retry mode **merges** with the existing preprocessed state: files that now succeed are added to `preprocessed/`, files that still fail stay in `dropped.csv` with their (possibly updated) error message.

### The `ContractMeta` sidecar (`pipeline.py:33-60`)

Every preprocessed file is accompanied by a `meta.json` with these 19 fields:

| Field | Type | Source step | 152:
|-------|------|-------------| 153:
| `sha256` | `str` | dedup |
| `source_name` | `str` | orchestrator |
| `original_path` | `str` | manifest |
| `pragma` | `str` | compile (raw, whitespace-stripped) |
| `solc_version` | `str` | compile (the version that succeeded) |
| `compile_status` | `str` | `"ok" \| "failed"` |
| `compile_error` | `str` | empty on success |
| `attempted_solc_versions` | `list[str]` | compile |
| `flatten_status` | `str` | flatten (`"flattened" \| "skipped_no_imports" \| "skipped_error" \| "stripped_unresolved_imports"`) |
| `dedup_group_id` | `str` | dedup |
| `is_duplicate` | `bool` | dedup |
| `duplicate_of` | `str` | dedup (sha256 of canonical) |
| `version_bucket` | `str` | segment |
| `has_unchecked_block` | `bool` | segment |
| `contract_names` | `list[str]` | segment |
| `n_raw_lines` | `int` | before normalize |
| `n_normalized_lines` | `int` | after normalize |
| `meta_schema_version` | `str` | `"1"` (LOCKED) |
| `extra` | `dict[str, Any]` | catch-all for future extensions |

This sidecar is the contract between preprocessing and every downstream stage. Representation reads `sha256` for cache keys. Labeling reads `source_name` for crosswalk lookup. Splitting reads `version_bucket` for stratification.

## 4. Public API

### `PreprocessingPipeline` — `pipeline.py:72-216`

```python
class PreprocessingPipeline:
    def __init__(self, source_name: str, out_dir: Path): ...
    def run(self, sol_files: list[Path], raw_base: Path) -> PipelineResult:
        """Process all sol_files through the 5-step pipeline.
        
        Writes .sol + .meta.json for successful files, dropped.csv for failures.
        """
```

### `preprocess_source(name, cfg, data_dir, dry_run=False, n_workers=1, sample=None, retry_failed=False)` — `preprocess.py:29-107`

```python
def preprocess_source(
    name: str, cfg: dict, data_dir: Path,
    dry_run: bool = False, n_workers: int = 1,
    sample: int | None = None, retry_failed: bool = False,
) -> None:
    """Run the 5-step pipeline for a single source.
    
    Supports full mode, --sample N for fast iteration, and --retry-failed
    to reprocess only previously-dropped files (incremental build).
    """
```

### `preprocess_all(cfg, data_dir, dry_run=False, n_workers=1, sample=None, retry_failed=False)` — `preprocess.py:247-264`

```python
def preprocess_all(cfg, data_dir, dry_run=False, n_workers=1, sample=None, retry_failed=False) -> None:
    """Run the preprocessing pipeline for every enabled source in config."""
```

### `flatten_contract(sol_path) -> FlattenResult` — `flattener.py:58-134`

```python
def flatten_contract(sol_path: Path) -> FlattenResult:
    """Flatten `sol_path`. Returns FlattenResult with content ready for next step.
    
    Three paths: solc --flatten, recursive unresolved-import strip, or skipped.
    """
```

### `compile_contract(sol_path) -> CompileResult` — `compiler.py:43-101`

```python
def compile_contract(sol_path: Path) -> CompileResult:
    """Try to compile `sol_path` with the appropriate solc version.
    
    Two-pass: exact requested, then nearest available. Does NOT raise.
    """
```

### `Deduplicator.process(content, path) -> DedupRecord` — `deduplicator.py:41-79`

```python
class Deduplicator:
    def process(self, content: str, path: Path) -> DedupRecord:
        """Check content against seen SHA-256 hashes and Ethereum addresses.
        Returns DedupRecord indicating whether this file is a duplicate.
        """
```

### `normalize(source) -> NormalizeResult` — `normalizer.py:25-38`

```python
def normalize(source: str) -> NormalizeResult:
    """Strip SPDX headers, line/block comments, trailing whitespace, and collapse blank lines."""
```

### `segment_and_bucket(source, pragma_raw) -> Segment` — `segmenter.py:35-54`

```python
def segment_and_bucket(source: str, pragma_raw: str) -> Segment:
    """Return one Segment representing the file.
    
    Multi-contract files kept whole (splitting would break import chains).
    Records all contract_names for Stage 4 label matching.
    """
```

### `run_preprocess_parallel(pipeline, sol_files, raw_base, n_workers=None)` — `parallel.py:80-156`

```python
def run_preprocess_parallel(
    pipeline: PreprocessingPipeline,
    sol_files: list[Path],
    raw_base: Path,
    n_workers: int | None = None,
) -> PipelineResult:
    """Run `pipeline._process_one` in parallel over `sol_files`.
    n_workers defaults to min(os.cpu_count(), 8)."""
```

## 5. Inputs → outputs

| Input | Where | What |
|-------|-------|------|
| `data/raw/<source>/repo/*.sol` | Stage 1a ingest output | Raw `.sol` files (canonical or symlinked) |
| `data/raw/<source>/ingestion_manifest.json` | Stage 1a | Per-source file list (respected for include/exclude scoping) |
| `data/labels/<source>.csv` (optional) | `_maybe_folderize` | Per-source labels CSV for folderization |
| `~/.solc-select/artifacts/solc-<v>/solc-<v>` | System | solc binaries (98 versions per `reference_solc.md`) |

| Output | Where | What |
|--------|-------|------|
| `data/preprocessed/<source>/<sha256>.sol` | `pipeline.py:194` | Cleaned, normalized, version-bucketed contract |
| `data/preprocessed/<source>/<sha256>.meta.json` | `pipeline.py:101` | 17-field sidecar (see §3) |
| `data/preprocessed/<source>/dropped.csv` | `pipeline.py:104` | Compile failures + duplicates (with attempted solc versions + error message) |

## 6. Pipeline interactions

| Stage | Direction | What |
|-------|-----------|------|
| Stage 1a (ingestion) | ← | Reads `data/raw/<source>/` + manifest |
| `sentinel_data.ingestion.label_folderize` | ← | Called by `_maybe_folderize` for sources with `labels_csv` set |
| Stage 2 (representation) | → | Reads `data/preprocessed/<source>/*.meta.json` (sha256 for cache keys, solc_version for binary resolution) |
| Stage 3 (labeling) | → | Reads `data/preprocessed/<source>/<sha256>.meta.json` (source_name for crosswalk lookup, original_path for folder extraction) |
| `sentinel_data.splitting.leakage_auditor` | → | Reads `data/preprocessed/<source>/*.sol` for text shingle similarity |

## 7. Tests

**Location:** `data_module/tests/test_preprocessing/`
- `test_pipeline.py` — full 5-step pipeline on small synthetic Solidity files
- `test_retry_failed.py` — `--retry-failed` merge semantics

**Command:**
```bash
cd ~/projects/sentinel/data_module
poetry run pytest tests/test_preprocessing/ -v
```

**Top-level integration tests** (also exercise this stage):
- `tests/test_integration_solidifi.py` — 283 contracts, end-to-end
- `tests/test_integration_dive.py` — 22,073 contracts, end-to-end

## 8. See also

- Previous stage: `sentinel_data/ingestion/README.md`
- Next stage: `sentinel_data/representation/README.md`
- CLI entry: `sentinel_data/cli.py` (`_run_preprocess` at line 131)
- Stage 1 plan: `docs/proposal/Data_Module_Proposals/actionable_plans/02_stage_1_ingestion_preprocessing.md`
- AUDIT_PATCHES 1-P1, 1-P2 (pragma tolerance fixes)
- A-1 fix (strip_comments) referenced in `tokenizer.py` representation adapter
- Drop-not-fix policy rationale: BCCC failure retrospective in `docs/legacy/bccc_deep_dive/`
