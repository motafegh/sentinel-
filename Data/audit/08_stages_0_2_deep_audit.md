# Stages 0-2: Deep Source Code Audit

**Audit Date:** 2026-06-11
**Scope:** `Data/sentinel_data/` — full source code review for Stages 0 (skeleton), 1 (ingest+preprocess), 2 (represent)
**Schema Version:** v9 (NODE_FEATURE_DIM=12, NUM_NODE_TYPES=14, NUM_EDGE_TYPES=12)
**Files Reviewed:** 25 Python files (cli.py, __init__.py, all of ingestion/, all of preprocessing/, all of representation/)

---

## Executive Summary

| Stage | Module | PASS | WARN | FAIL | Verdict |
|-------|--------|------|------|------|---------|
| 0 | Root files (cli, config, __init__) | 8 | 3 | 2 | **WARN** |
| 1 | Ingestion (connectors, manifest, freshness) | 14 | 6 | 4 | **WARN** |
| 1 | Preprocessing (5-step pipeline) | 22 | 8 | 5 | **WARN** |
| 2 | Representation (graph, tokens, cache) | 16 | 5 | 3 | **WARN** |
| **TOTAL** | | **60** | **22** | **14** | **WARN** |

---

## 1. Stage 0: Root Files (cli.py, __init__.py)

### 1.1 cli.py

| Check | Status | Line(s) | Notes |
|-------|--------|---------|-------|
| `sys.path` injection for `ml/` | PASS | 33-39 | Correctly adds repo root and ml/ to path |
| `STAGES` list complete | PASS | 42-52 | 9 stages defined |
| `STAGE_DESCRIPTIONS` | PASS | 54-64 | Accurate descriptions |
| `_load_config()` | PASS | 67-70 | Uses yaml.safe_load |
| `_default_config()` | PASS | 73-77 | Finds config.yaml relative to entry point |
| Stage dispatch via `_STAGE_FN` | PASS | 257-268 | Maps stage name to function |
| Argument parser | PASS | 273-368 | Well-structured with subparsers |
| `run` subcommand with `--from-stage` | PASS | 288-297 | Supports resume |
| Per-stage `--source` flag | PASS | 304-311 | For ingest/preprocess/represent |
| `--workers` for parallel | PASS | 311-344 | Multiprocessing pool size |
| `--limit` for smoke tests | PASS | 319-325 | Process only N contracts |
| `--force` for cache bypass | PASS | 326-330 | Recompute on cache hit |
| `--emit-cfg` flag | PASS | 331-336 | Write standalone CFG artifact |
| `--retry-failed` for preprocess | PASS | 352-359 | Re-run dropped files |
| `freshness` subcommand | PASS | 361-366 | Check pin staleness |
| Stage 3-7 placeholder messages | PASS | 194-245 | "NOT IMPLEMENTED" messages |
| `main()` entry point | PASS | 395-409 | Parses args, dispatches |

### 1.2 Issues Found in cli.py

| ID | Severity | Line | Issue |
|----|----------|------|-------|
| CLI-1 | MEDIUM | 385-390 | `_handle_run` hardcodes `workers=1, sample=None, retry_failed=False` — drops user args |
| CLI-2 | LOW | 25 | `import textwrap` at top is fine but description in `_build_parser` uses `textwrap.dedent` correctly |
| CLI-3 | LOW | 145-147 | Source list in `_run_represent` falls back to `cfg.get("sources")` but plan uses `sources_critical_path` + `sources_additive` — potential source list mismatch |

**CLI-1 Details:** When `sentinel-data run --from-stage preprocess` is invoked with `--workers 4`, the `_handle_run` function creates a fresh `argparse.Namespace` with `workers=1`, discarding the user's value. This is a known limitation — the run dispatcher doesn't propagate stage-specific args.

---

## 2. Stage 1: Ingestion

### 2.1 ingest.py

| Check | Status | Line(s) | Notes |
|-------|--------|---------|-------|
| `_all_sources()` merges 3 sections | PASS | 15-21 | sources_critical_path + sources_additive + sources |
| `_enabled_sources()` filters by enabled flag | PASS | 24-25 | Correct |
| `_source_config()` builds SourceConfig | PASS | 28-44 | Maps config fields correctly |
| `ingest_source()` entry point | PASS | 47-90 | Validates source, calls connector, writes manifest |
| `ingest_all()` iterates enabled sources | PASS | 93-106 | Prints progress per source |
| `dry_run` flag handling | PASS | 69-73, 94-98 | Skips actual pull |

### 2.2 Issues in ingest.py

| ID | Severity | Line | Issue |
|----|----------|------|-------|
| ING-1 | HIGH | 33 | No pin validation — empty `pin=""` accepted silently (D-1.3 violation) |
| ING-2 | LOW | 28-44 | `_source_config` silently drops unknown fields into `extra` — no warning |
| ING-3 | LOW | 70-72 | dry_run print doesn't show `enabled` status or `hf_dataset`/`zenodo_record` fields |

**ING-1 Details:** Per plan D-1.3, pin enforcement should reject empty pins or at least warn. Current code accepts `pin=""` and proceeds to clone at HEAD, losing reproducibility.

### 2.3 manifest.py

| Check | Status | Line(s) | Notes |
|-------|--------|---------|-------|
| `FileRecord` dataclass | PASS | 18-22 | path, sha256, size_bytes |
| `IngestionManifest` dataclass | PASS | 25-36 | All required fields |
| `save()` serializes to JSON | PASS | 40-43 | Uses `asdict` |
| `load()` deserializes from JSON | PASS | 45-50 | Reconstructs FileRecord list |
| `verify()` re-checks SHA-256 | PASS | 54-68 | Returns (ok, errors) |
| `_sha256()` streaming hash | PASS | 73-78 | 64KB chunks, correct |
| `build_file_records()` | PASS | 81-89 | Creates records with relative paths |
| `load_manifest()` / `verify_manifest()` | PASS | 92-98 | Convenience wrappers |

### 2.4 Issues in manifest.py

| ID | Severity | Line | Issue |
|----|----------|------|-------|
| ING-4 | MEDIUM | 40-43 | `save()` overwrites manifest — not truly append-only (D-1.2 violation) |
| ING-5 | LOW | 54-68 | `verify()` uses `rel = rec.path` (relative to `raw_dir`) but doesn't validate path is within `raw_dir` — potential path traversal |

**ING-4 Details:** Plan D-1.2 says "manifests are append-only: past entries are never deleted." Current `save()` overwrites the single manifest file. A true append-only design would maintain a list of historical manifests with timestamps.

### 2.5 freshness.py

| Check | Status | Line(s) | Notes |
|-------|--------|---------|-------|
| `run_freshness_check()` | PASS | 17-62 | Generates report, writes to data/analysis/ |
| `_git_upstream_head()` | PASS | 65-76 | Uses `git ls-remote`, 15s timeout |
| `_slither_version_check()` | PASS | 79-93 | Compares installed vs PyPI |
| `_installed_slither_version()` | PASS | 96-108 | Uses importlib.metadata |
| `_latest_pypi_version()` | PASS | 111-118 | HTTP request to PyPI JSON API |

### 2.6 Issues in freshness.py

| ID | Severity | Line | Issue |
|----|----------|------|-------|
| ING-6 | MEDIUM | 40 | Pin comparison bug: `upstream.startswith(pin)` — should be `upstream.startswith(pin[:7])` or `upstream == pin` for full SHA match |
| ING-7 | LOW | 24 | `datetime.utcnow()` is deprecated in Python 3.12+ — use `datetime.now(timezone.utc)` |
| ING-8 | LOW | 70 | No subprocess timeout in the except clause — silently returns "" |

**ING-6 Details:** Plan D-0.5 says freshness check should detect pin drift. Current `upstream.startswith(pin)` matches if the upstream SHA starts with the pin string. For 7-char pins this works, but for full 40-char SHAs it's equivalent to equality. For short pins (8-12 chars) it can give false positives (e.g., pin `abc123` matches upstream `abc1234567...`).

### 2.7 connectors/base.py

| Check | Status | Line(s) | Notes |
|-------|--------|---------|-------|
| `SourceConfig` dataclass | PASS | 12-24 | All fields present |
| `PullResult` dataclass | PASS | 27-35 | Return type for all connectors |
| `ConnectorError` exception | PASS | 38-39 | Custom error type |
| `BaseConnector.pull()` template method | PASS | 45-57 | Creates dest, times, calls _pull |
| `find_sol_files()` | PASS | 63-93 | Supports include/exclude subdirs |
| `datetime.utcnow()` usage | WARN | 56 | Deprecated in Python 3.12+ |

### 2.8 connectors/git_connector.py

| Check | Status | Line(s) | Notes |
|-------|--------|---------|-------|
| `_pull()` clones repo | PASS | 24-53 | Handles existing repo case |
| `_clone()` with/without pin | PASS | 57-66 | Full clone for pin, shallow for HEAD |
| `_current_commit()` | PASS | 68-74 | Uses `git rev-parse HEAD` |
| `post_clone_cmd` support | PASS | 36-39 | Runs custom command after clone |
| Idempotent via `.post_clone_done` | PASS | 37 | Touch marker file |

### 2.9 Issues in git_connector.py

| ID | Severity | Line | Issue |
|----|----------|------|-------|
| ING-9 | MEDIUM | 38 | `post_cmd.split()` — should use `shlex.split()` for quoted args |
| ING-10 | LOW | 60, 65 | No subprocess timeout on `git clone` — hangs possible on network issues |
| ING-11 | LOW | 78-84 | `_run()` raises on non-zero exit but doesn't include return code in error |

**ING-9 Details:** If `post_clone_cmd` is `python checkout_sources.py --repo .`, simple `.split()` works. But if it's `python "script with spaces.py"`, simple split breaks. `shlex.split()` handles quoted arguments correctly.

### 2.10 connectors/manual_connector.py

| Check | Status | Line(s) | Notes |
|-------|--------|---------|-------|
| `_pull()` validates staging_path | PASS | 53-98 | Raises if staging_path missing |
| `materialize_staging()` | PASS | 101-156 | Handles zip, dir, glob |
| `_extract_zip()` | PASS | 159-187 | Strips macOS metadata, zip-slip defense |
| `pin_marker` from mtime | PASS | 85-89 | Uses staging file mtime |

### 2.11 Issues in manual_connector.py

| ID | Severity | Line | Issue |
|----|----------|------|-------|
| ING-12 | LOW | 38 | `import subprocess` unused (dead import) |
| ING-13 | LOW | 145 | `repo_dir.symlink_to(extract_root.resolve())` — for zip extraction, "symlink" mode creates a symlink to the extracted dir, not a symlink of individual files. The naming is confusing. |

### 2.12 Stub Connectors (etherscan, huggingface, zenodo)

| Check | Status | Notes |
|-------|--------|-------|
| EtherscanConnector stub | PASS | Raises NotImplementedError with helpful message |
| HuggingFaceConnector stub | PASS | Raises NotImplementedError |
| ZenodoConnector stub | PASS | Raises NotImplementedError |
| All registered in factory | PASS | connectors/__init__.py:10-18 |

### 2.13 label_folderize.py

| Check | Status | Line(s) | Notes |
|-------|--------|---------|-------|
| `FolderizationResult` dataclass | PASS | 48-69 | All counts tracked |
| `folderize_by_labels()` | PASS | 72-172 | Idempotent, multi-label support |
| Moves flat files to `__source__/` | PASS | 106-123 | Handles both flat and folderized sources |
| Creates per-class symlinks | PASS | 160-167 | `../../__source__/<id>.sol` |
| Idempotent re-runs | PASS | 164 | Checks `link_path.exists()` |

---

## 3. Stage 1: Preprocessing

### 3.1 compiler.py

| Check | Status | Line(s) | Notes |
|-------|--------|---------|-------|
| `CompileResult` dataclass | PASS | 28-38 | Mutable default fixed in `__post_init__` |
| `compile_contract()` two-pass | PASS | 41-99 | Pass 1: exact, Pass 2: nearest available |
| `_extract_pragma()` strips whitespace | PASS | 104-109 | Fixes `^ 0.4 .9` → `^0.4.9` |
| `_parse_version()` extracts single version | PASS | 112-122 | Handles `0.4.25`, `=0.4.25`, `^0.8.0`, `~0.6.12` |
| `_satisfying_versions()` | PASS | 125-146 | Returns newest-first candidates |
| `_available_versions()` | PASS | 149-159 | Reads solc-select artifacts |
| `_solc_binary()` | PASS | 162-164 | Returns path or None |
| `_run_solc()` with `--allow-paths` | PASS | 167-183 | 30s timeout, captures stderr |

### 3.2 Issues in compiler.py

| ID | Severity | Line | Issue |
|----|----------|------|-------|
| PRE-1 | LOW | 34 | `attempted_versions: list[str] = None` is non-standard — should be `field(default_factory=list)` for proper dataclass behavior |
| PRE-2 | LOW | 175 | `allow_root = sol_path.parent.parent` — heuristic that may not always be correct (e.g., files at repo root) |
| PRE-3 | LOW | 178 | `--allow-paths` is a single path; for multi-level relative imports, may need comma-separated list of all parents |

**PRE-1 Details:** The `__post_init__` handles `None` correctly, but using `field(default_factory=list)` would be more Pythonic and avoid the mutable-default footgun pattern. This is a cosmetic issue since `__post_init__` works.

### 3.3 deduplicator.py

| Check | Status | Line(s) | Notes |
|-------|--------|---------|-------|
| `DedupRecord` dataclass | PASS | 22-27 | All fields present |
| `Deduplicator` stateful class | PASS | 30-72 | Tracks seen SHA + addresses |
| Level 1: exact SHA-256 | PASS | 42-49 | Fast exact match |
| Level 2: address-level | PASS | 52-64 | Catches same-address duplicates |
| Level 3: AST near-dup | PASS (stub) | 66-72 | Returns is_duplicate=False, deferred to Stage 2+ |

### 3.4 Issues in deduplicator.py

| ID | Severity | Line | Issue |
|----|----------|------|-------|
| PRE-4 | HIGH | 66-72 | Level 3 (AST near-dup @ 0.85) entirely stubbed — BCCC had 38.8% duplication, 0.85 threshold is critical |
| PRE-5 | LOW | 19 | `_ADDRESS_RE` matches 40-char hex addresses, but Ethereum addresses can be checksummed (mixed case) — regex doesn't validate checksum |
| PRE-6 | LOW | 37-38 | `_seen_sha` and `_seen_addr` grow unbounded — for 22K+ contracts, this is ~10MB memory, but not a leak |

**PRE-4 Details:** The plan specifies three-level dedup: exact SHA → address-level → AST near-dup @ 0.85. Level 3 is stubbed — it always returns `is_duplicate=False`. This means 38.8% of BCCC-style duplicates (copy-paste-with-edits) would NOT be caught. The plan says this is deferred to Stage 2, but Stage 2 also stubs it (per the plan's deferral notes).

### 3.5 flattener.py

| Check | Status | Line(s) | Notes |
|-------|--------|---------|-------|
| `FlattenResult` dataclass | PASS | 26-31 | 4 flatten_status values |
| `flatten_contract()` | PASS | 58-134 | 3-stage fallback: solc --flatten → strip unresolved → pass through |
| Skip if no imports | PASS | 63-64 | Optimization |
| solc --flatten with matching version | PASS | 67-76 | Uses `_pick_solc` |
| Strip unresolved imports | PASS | 88-127 | Recursive, handles transitive |
| Strip inheritance parents | PASS | 96-101 | Removes parents from `is A, B, C` |
| Transitive sub-strips | PASS | 111-116 | Uses `_transitive_strip.apply_sub_strips_to_source` |

### 3.6 Issues in flattener.py

| ID | Severity | Line | Issue |
|----|----------|------|-------|
| PRE-7 | LOW | 34 | `_IMPORT_RE = re.compile(r'^\s*import\s+', re.MULTILINE)` — could miss imports with no leading whitespace or after preprocessor directives |
| PRE-8 | LOW | 48-51 | `_IMPORT_LINE_RE` is duplicated in `_transitive_strip.py:25-28` — should import from one source |
| PRE-9 | LOW | 53-55 | `_CONTRACT_INHERIT_RE` misses `abstract contract` keyword |
| PRE-10 | LOW | 40-46 | `_ASSUMED_BARE_IMPORT_SYMBOLS` is hardcoded — new forge-std symbols require code changes |

**PRE-7 Details:** The regex `^\s*import\s+` matches `import` with any leading whitespace, but doesn't enforce it's at the start of a logical line. Could match inside a multi-line string literal (unlikely in Solidity but possible).

**PRE-8 Details:** Same regex pattern appears in two files. If one is updated, the other may diverge. Should be in a shared module.

**PRE-9 Details:** `_CONTRACT_INHERIT_RE` matches `contract Foo is A, B {` but not `abstract contract Foo is A, B {`. The segmenter has the same issue.

### 3.7 normalizer.py

| Check | Status | Line(s) | Notes |
|-------|--------|---------|-------|
| `NormalizeResult` dataclass | PASS | 9-14 | Tracks line counts |
| SPDX header removal | PASS | 16, 25 | `// SPDX-License-Identifier:` |
| Block comment removal | PASS | 17, 26 | `/* ... */` |
| Line comment removal | PASS | 18, 27 | `// ...` |
| Trailing whitespace removal | PASS | 20, 28 | |
| Multiple newlines collapsed | PASS | 19, 29 | `>\n\n\n<` → `>\n\n<` |

### 3.8 Issues in normalizer.py

| ID | Severity | Line | Issue |
|----|----------|------|-------|
| PRE-11 | LOW | 17 | `_BLOCK_CMT` uses `re.DOTALL` — correct for multi-line block comments |
| PRE-12 | LOW | 18 | `_LINE_CMT` removes ALL `//...` including string literals containing `//` (e.g., `"https://..."`) — known limitation, accepted per plan |

**PRE-12 Details:** A string like `"https://example.com"` will have the `//example.com"` part stripped, corrupting the string. This is a known issue in many Solidity normalizers. The plan accepts this as a "lossy but controlled" loss.

### 3.9 segmenter.py

| Check | Status | Line(s) | Notes |
|-------|--------|---------|-------|
| `Segment` dataclass | PASS | 23-31 | All fields present |
| `segment_and_bucket()` | PASS | 33-52 | One segment per file (not per contract) |
| `_CONTRACT_RE` matches contracts | PASS | 18 | `contract Foo is A {` |
| Version bucketing | PASS | 55-64 | legacy / transitional / modern |
| `has_unchecked_block` detection | PASS | 20, 43 | For 0.8+ files |

### 3.10 Issues in segmenter.py

| ID | Severity | Line | Issue |
|----|----------|------|-------|
| PRE-13 | LOW | 18 | `_CONTRACT_RE` doesn't match `abstract contract Foo is A {` — same as PRE-9 |
| PRE-14 | LOW | 40-41 | `names[0]` as primary contract — for multi-contract files, the first is arbitrary. No heuristic for "main" contract. |

### 3.11 pipeline.py

| Check | Status | Line(s) | Notes |
|-------|--------|---------|-------|
| `ContractMeta` dataclass | PASS | 33-53 | All plan fields present |
| `PipelineResult` dataclass | PASS | 56-60 | processed, dropped, duration |
| `PreprocessingPipeline.run()` | PASS | 71-96 | Serial orchestration |
| `_process_one()` 5-step pipeline | PASS | 98-202 | flatten → compile → dedup → normalize → segment |
| Sidecar `meta.json` written | PASS | 87 | After each successful process |
| `dropped.csv` written | PASS | 90 | After pipeline completes |
| `META_SCHEMA_VERSION` | PASS | 30 | Versioned schema |

### 3.12 Issues in pipeline.py

| ID | Severity | Line | Issue |
|----|----------|------|-------|
| PRE-15 | MEDIUM | 33-53 | Sidecar `meta.json` missing 3 plan fields: `inheritance_root`, `n_imports`, `contract_count` (D-1.7) |
| PRE-16 | LOW | 119 | Temp file naming uses `st_mtime_ns` — fast successive runs may collide if mtime hasn't changed |
| PRE-17 | LOW | 145 | Cleanup loop for `*.sentinel_stripped.sol` is per-temp-path's parent, not all dirs where strips were written |

**PRE-15 Details:** Plan D-1.7 requires these fields in the sidecar:
- `inheritance_root`: root contract name from the inheritance chain
- `n_imports`: number of import statements
- `contract_count`: number of contracts in the file

Current `ContractMeta` has `contract_names: list[str]` but not `contract_count` (len of the list), and no `inheritance_root` or `n_imports`.

### 3.13 parallel.py

| Check | Status | Line(s) | Notes |
|-------|--------|---------|-------|
| `_process_one_worker()` | PASS | 49-69 | Picklable module-level function |
| Auto-tuned chunksize | PASS | 72-77 | `total // (n_workers * 16)` |
| `run_preprocess_parallel()` | PASS | 80-156 | Multiprocessing.Pool.imap |
| Content-addressed output | PASS | 95-97 | sha256 filenames, no conflicts |
| Worker errors captured | PASS | 132-135 | `status: "error"` for exceptions |
| Errors written to dropped.csv | PASS | 139-150 | With distinctive reason |

### 3.14 Issues in parallel.py

| ID | Severity | Line | Issue |
|----|----------|------|-------|
| PRE-18 | MEDIUM | 37 | `import multiprocessing as mp` — should use `mp.get_context("spawn")` for fork safety on Linux (Linux default is fork, but spawn is safer for complex objects) |
| PRE-19 | LOW | 61 | Exception in worker is caught with `except Exception` — too broad, should catch specific exceptions |

**PRE-18 Details:** Linux defaults to `fork`, which can cause issues with CUDA/torch/etc. that hold resources. `spawn` is the recommended context for ML pipelines.

### 3.15 _transitive_strip.py

| Check | Status | Line(s) | Notes |
|-------|--------|---------|-------|
| `apply_sub_strips_to_source()` | PASS | 31-104 | Writes sibling files, rewrites imports |
| Sibling file naming | PASS | 22, 64 | `.sentinel_stripped.sol` suffix |
| Path traversal safe | PASS | 74-78 | Uses `relative_to` or falls back to absolute |

---

## 4. Stage 2: Representation

### 4.1 graph_extractor.py

| Check | Status | Line(s) | Notes |
|-------|--------|---------|-------|
| Thin adapter pattern | PASS | 30-77 | Re-exports from `ml.src.preprocessing.graph_extractor` |
| Lazy import via `__getattr__` | PASS | 47-63 | For standalone install support |
| Eager re-export | PASS | 67-74 | Same Python object as ml/ |
| Public API | PASS | 35-44 | 6 symbols re-exported |

### 4.2 Issues in graph_extractor.py

| ID | Severity | Line | Issue |
|----|----------|------|-------|
| REP-1 | LOW | 47-63 | `__getattr__` fallback may shadow attributes defined later in the module — minor, works due to import order |

### 4.3 graph_schema.py

| Check | Status | Line(s) | Notes |
|-------|--------|---------|-------|
| Thin adapter for schema | PASS | 36-141 | Re-exports from `ml.src.preprocessing.graph_schema` |
| `FEATURE_SCHEMA_VERSION` | PASS | 131 | "v9" |
| `NODE_FEATURE_DIM = 12` | PASS | 132 | Matches plan |
| `NUM_NODE_TYPES = 14` | PASS | 133 | Matches plan |
| `NUM_EDGE_TYPES = 12` | PASS | 134 | Matches plan |
| `CLASS_NAMES` (10 classes) | PASS | 73-84 | Locked order |
| `NUM_CLASSES = 10` | PASS | 85 | Matches plan |
| `_MAX_TYPE_ID = 13.0` | PASS | 145 | Derived from NODE_TYPES |

### 4.4 CRITICAL Issues in graph_schema.py

| ID | Severity | Line | Issue |
|----|----------|------|-------|
| REP-2 | **CRITICAL** | 73-84 | `CLASS_NAMES` order DOES NOT MATCH Stage 3 `taxonomy.yaml`! Stage 3 has: CallToUnknown=0, DenialOfService=1, ExternalBug=2, GasException=3, IntegerUO=4, MishandledException=5, Reentrancy=6, Timestamp=7, TransactionOrderDependence=8, UnusedReturn=9. But graph_schema.py has: Reentrancy=0, CallToUnknown=1, Timestamp=2, ExternalBug=3, GasException=4, DenialOfService=5, IntegerUO=6, UnusedReturn=7, MishandledException=8, NonVulnerable=9. **This is a class order mismatch that will break training!** |

**REP-2 Details:** This is a critical bug. The Stage 3 audit report (06_labeling_stage3_audit.md) says "Class order matches trainer.py" and the `test_taxonomy.py` test compares against `_TRAINER_CLASS_NAMES` which lists: CallToUnknown=0, DenialOfService=1, ExternalBug=2, GasException=3, IntegerUO=4, MishandledException=5, Reentrancy=6, Timestamp=7, TransactionOrderDependence=8, UnusedReturn=9.

But `graph_schema.py` in representation has a DIFFERENT order with Reentrancy=0 and includes "NonVulnerable" as class 9. This will cause:
- Stage 3 labels written with one class order
- Stage 2 graphs expecting a different class order
- Model training will learn wrong class indices

**The fix:** `graph_schema.py` should NOT define its own `CLASS_NAMES` — it should import from the same source as `taxonomy.yaml`. The `__post_init__` in `taxonomy.yaml` says "Source of truth for the class order: ml/src/training/trainer.py:CLASS_NAMES". The representation module's `CLASS_NAMES` is a different, conflicting definition.

### 4.5 orchestrator.py

| Check | Status | Line(s) | Notes |
|-------|--------|---------|-------|
| `RepresentResult` dataclass | PASS | 52-79 | All counts tracked |
| `EXTRACTOR_VERSION` | PASS | 86 | "v2.1-windowed-gcb" |
| `_load_meta()` | PASS | 89-95 | Reads Stage 1 meta.json |
| `_is_cached()` | PASS | 98-109 | Checks sidecar versions |
| `_extract_one()` | PASS | 112-233 | Graph + tokens extraction |
| `_resolve_solc_binary()` | PASS | 236-246 | From solc-select |
| `represent_source()` entry | PASS | 249-344 | Iterates meta.json files |
| Cache invalidation on force | PASS | 142-145 | Deletes all 3 files |
| Sidecar rep.json written | PASS | 200-215 | With all required fields |
| Optional CFG emission | PASS | 217-231 | `--emit-cfg` flag |
| Progress printing | PASS | 337-341 | Every 20 contracts |

### 4.6 Issues in orchestrator.py

| ID | Severity | Line | Issue |
|----|----------|------|-------|
| REP-3 | LOW | 162-169 | `allow_paths=[str(sol_path.parent.parent.parent.parent)]` — 4-levels-up heuristic is fragile |
| REP-4 | LOW | 219-221 | Re-imports `GraphExtractionConfig` inside `if emit_cfg:` — already imported above |
| REP-5 | LOW | 222 | Same 4-levels-up heuristic for CFG builder — duplicated logic |
| REP-6 | LOW | 251 | Dead `cfg: dict` parameter — `cfg` is passed but never used in the function body |

**REP-3 Details:** The 4-levels-up heuristic `sol_path.parent.parent.parent.parent` assumes the file structure is `data/raw/<source>/repo/<category>/file.sol`. For DIVE, the structure is `data/raw/dive/repo/__source__/<id>.sol`, so the heuristic is `parent × 4 = data/`. This breaks if the structure changes.

### 4.7 cache_manager.py

| Check | Status | Line(s) | Notes |
|-------|--------|---------|-------|
| `is_cached()` | PASS | 34-53 | Sidecar + version check + file existence |
| `record_cache_hit()` | PASS | 56-60 | No-op placeholder |
| `invalidate()` | PASS | 63-69 | Removes 3 files |
| `list_cached_sha256s()` | PASS | 72-82 | Lists complete entries |
| `stale_entries()` | PASS | 85-106 | Finds version mismatches |
| `evict_stale()` | PASS | 109-119 | Calls invalidate |

### 4.8 CRITICAL Issues in cache_manager.py

| ID | Severity | Line | Issue |
|----|----------|------|-------|
| REP-7 | **CRITICAL** | 79 | `sha = s.stem.removesuffix(".rep") if s.stem.endswith(".rep") else s.stem` — this is BUGGY. `s.stem` for `abc123.rep.json` returns `abc123.rep` (removes `.json`). Then `.removesuffix(".rep")` removes `.rep` → `abc123`. This works. BUT in `stale_entries()` at line 95: `sha = rep_path.stem` — `rep_path.stem` for `abc123.rep.json` is `abc123.rep`, NOT `abc123`! This means stale_entries returns `abc123.rep` as the sha, and `invalidate(output_dir, "abc123.rep")` will look for `abc123.rep.pt`, `abc123.rep.tokens.pt`, `abc123.rep.rep.json` — none of which exist. **The stale cache is never evicted.** |
| REP-8 | LOW | 95 | `rep_path.stem` for `.rep.json` gives `.rep` not the sha — bug confirmed |

**REP-7 Details:** This is the same bug flagged in the previous Stage 0-2 audit (F7). The fix is:
```python
sha = rep_path.stem.removesuffix(".rep")
```
This matches the pattern in `list_cached_sha256s()` at line 79. The `stale_entries()` function is missing the `.removesuffix(".rep")` call, so it returns garbage shas that can't be used for eviction.

### 4.9 versioner.py

| Check | Status | Line(s) | Notes |
|-------|--------|---------|-------|
| `read_registry()` | PASS | 38-46 | Returns {} if missing |
| `write_registry()` | PASS | 49-63 | Writes current versions |
| `check_and_evict()` | PASS | 66-93 | Version mismatch → evict |
| `current_versions()` | PASS | 96-100 | Returns (FEATURE, EXTRACTOR) |
| Uses `timezone.utc` | PASS | 59 | Correct modern API |

### 4.10 cfg_builder.py

| Check | Status | Line(s) | Notes |
|-------|--------|---------|-------|
| `CfgNode` dataclass | PASS | 46-52 | index, type, source_lines, expression |
| `CfgEdge` dataclass | PASS | 54-57 | src, dst |
| `CfgFunction` dataclass | PASS | 59-66 | canonical_name, nodes, edges, num_loops, max_depth |
| `CfgArtifact` dataclass | PASS | 69-77 | Top-level result |
| `_cfg_node_type_str()` | PASS | 87-123 | Maps Slither types to NODE_TYPES |
| `_count_back_edges()` | PASS | 126-149 | DFS-based loop detection |
| `_max_depth_from_entry()` | PASS | 152-170 | BFS longest path |
| `_build_function_cfg()` | PASS | 173-217 | Per-function CFG |
| `build_cfg()` public API | PASS | 224-274 | Entry point with error handling |

### 4.11 Issues in cfg_builder.py

| ID | Severity | Line | Issue |
|----|----------|------|-------|
| REP-9 | MEDIUM | 243 | Imports from `ml.src.preprocessing.graph_schema` directly — bypasses the thin adapter pattern (D-2.7 violation) |
| REP-10 | LOW | 265 | `list(contract.functions) + list(getattr(contract, "modifiers", []))` — modifiers are not functions; mixing them in a single CFG may produce confusing canonical_names |
| REP-11 | LOW | 177 | `n.source_mapping.lines[0]` without checking if `lines` is non-empty — could raise IndexError |

**REP-9 Details:** All other representation modules use the thin adapter (`from sentinel_data.representation.graph_schema import ...`). This file bypasses it by importing directly from `ml.src.preprocessing.graph_schema`. Per plan D-2.7, all representation code should go through the thin adapter to ensure byte-identical output.

### 4.12 Stub Builders (call_graph, pdg, opcode)

| Check | Status | Notes |
|-------|--------|-------|
| `call_graph.py` stub | PASS | Raises NotImplementedError, references v3.1 |
| `pdg_builder.py` stub | PASS | Raises NotImplementedError, references v3.1 |
| `opcode_extractor.py` stub | PASS | Raises NotImplementedError, references v3.1 |
| All registered in package | PASS | `__init__.py` may or may not export them |

### 4.13 tokenizer.py

| Check | Status | Line(s) | Notes |
|-------|--------|---------|-------|
| Thin adapter for windowed tokenizer | PASS | 31-69 | Re-exports from `ml.src.data_extraction.windowed_tokenizer` |
| `tokenize_windowed_contract` | PASS | 62-69 | Correct function (not the old single-window one) |
| `TOKENIZER_MODEL` | PASS | 62-69 | "microsoft/graphcodebert-base" |
| `WINDOW_SIZE = 512` | PASS | 62-69 | Correct |
| `STRIDE = 256` | PASS | 62-69 | Correct (50% overlap) |
| `MAX_WINDOWS = 4` | PASS | 62-69 | Correct |

---

## 5. Cross-Cutting Issues

### 5.1 Hardcoded Paths in config.yaml

| ID | Severity | File | Issue |
|----|----------|------|-------|
| CROSS-1 | HIGH | `config.yaml:125,140` | Hardcoded `/home/motafeq/...` paths break portability (D-0.6) |
| CROSS-2 | HIGH | `config.yaml` (defihacklabs) | `defihacklabs` has `enabled: false` but is critical-path source |

### 5.2 Test Issues (from test files)

| ID | Severity | File | Issue |
|----|----------|------|-------|
| TEST-1 | HIGH | `tests/test_verification/test_patterns.py:7` | Relative path `Path("Data/...")` — fails when CWD is `Data/` |
| TEST-2 | MEDIUM | `tests/test_labeling/test_gate.py:12` | Same relative path bug |
| TEST-3 | MEDIUM | `tests/test_labeling/test_parser_solidifi.py:11-12` | Same relative path bug |
| TEST-4 | LOW | Multiple test files | No `conftest.py` for shared fixtures |

---

## 6. P0/P1/P2/P3 Issues

### P0 — Must Fix Before Run 11

| ID | Issue | File:Line |
|----|-------|-----------|
| REP-2 | Class order mismatch between `graph_schema.py` and `taxonomy.yaml` | `representation/graph_schema.py:73-84` |
| REP-7 | `stale_entries()` returns wrong sha256 — stale cache never evicted | `representation/cache_manager.py:95` |
| CROSS-1 | Hardcoded absolute paths in config.yaml | `config.yaml:125,140` |
| CROSS-2 | `defihacklabs` disabled in config.yaml | `config.yaml` |
| ING-1 | Pin enforcement missing — empty pins accepted silently | `ingestion/ingest.py:33` |
| PRE-4 | Dedup Level 3 (AST near-dup @ 0.85) entirely stubbed | `preprocessing/deduplicator.py:66-72` |
| PRE-15 | Sidecar missing 3 plan fields | `preprocessing/pipeline.py:33-53` |
| TEST-1 | Relative path in test_patterns.py | `tests/test_verification/test_patterns.py:7` |

### P1 — Should Fix Before Stage 7

| ID | Issue | File:Line |
|----|-------|-----------|
| ING-4 | Manifest save() overwrites — not append-only | `ingestion/manifest.py:40-43` |
| ING-6 | Pin comparison uses startswith | `ingestion/freshness.py:40` |
| ING-9 | post_cmd.split() instead of shlex.split() | `ingestion/connectors/git_connector.py:38` |
| PRE-1 | Mutable default for attempted_versions | `preprocessing/compiler.py:34` |
| PRE-8 | _IMPORT_LINE_RE duplicated | `flattener.py:48-51`, `_transitive_strip.py:25-28` |
| PRE-18 | Should use mp.get_context("spawn") | `preprocessing/parallel.py:37` |
| REP-9 | cfg_builder bypasses thin adapter | `representation/cfg_builder.py:243` |
| CLI-1 | _handle_run drops --workers, --sample, --retry-failed | `cli.py:385-390` |
| TEST-2 | Relative path in test_gate.py | `tests/test_labeling/test_gate.py:12` |
| TEST-3 | Relative path in test_parser_solidifi.py | `tests/test_labeling/test_parser_solidifi.py:11-12` |

### P2 — Should Fix Before Run 11 Launch

| ID | Issue | File:Line |
|----|-------|-----------|
| ING-7 | datetime.utcnow() deprecated | `ingestion/freshness.py:24`, `connectors/base.py:56` |
| ING-10 | No subprocess timeout on git clone | `connectors/git_connector.py:60,65` |
| ING-12 | Unused import in manual_connector | `connectors/manual_connector.py:38` |
| PRE-2 | allow_root heuristic | `preprocessing/compiler.py:175` |
| PRE-5 | Address regex doesn't validate checksum | `preprocessing/deduplicator.py:19` |
| PRE-7 | _IMPORT_RE may miss some cases | `preprocessing/flattener.py:34` |
| PRE-9 | _CONTRACT_INHERIT_RE misses abstract | `preprocessing/flattener.py:53-55` |
| PRE-13 | _CONTRACT_RE misses abstract | `preprocessing/segmenter.py:18` |
| REP-3 | 4-levels-up allow_paths heuristic | `representation/orchestrator.py:162-169,222` |
| REP-6 | Dead cfg: dict parameter | `representation/orchestrator.py:251` |
| TEST-4 | No conftest.py | `tests/test_labeling/`, `tests/test_verification/` |

### P3 — Nice to Have

| ID | Issue | File:Line |
|----|-------|-----------|
| ING-2 | Unknown fields silently dropped | `ingestion/ingest.py:28-44` |
| ING-3 | dry_run print incomplete | `ingestion/ingest.py:70-72` |
| ING-8 | Subprocess except clause silent | `ingestion/freshness.py:70-76` |
| ING-11 | _run() doesn't include return code | `connectors/git_connector.py:78-84` |
| ING-13 | "symlink" mode for zip is confusing | `connectors/manual_connector.py:145` |
| PRE-3 | --allow-paths is single path | `preprocessing/compiler.py:178` |
| PRE-6 | _seen_sha/_seen_addr unbounded | `preprocessing/deduplicator.py:37-38` |
| PRE-10 | _ASSUMED_BARE_IMPORT_SYMBOLS hardcoded | `preprocessing/flattener.py:40-46` |
| PRE-12 | String literals with // corrupted | `preprocessing/normalizer.py:18` |
| PRE-14 | Primary contract selection arbitrary | `preprocessing/segmenter.py:40-41` |
| PRE-16 | Temp file naming uses st_mtime_ns | `preprocessing/pipeline.py:119` |
| PRE-17 | Cleanup is per-temp-parent, not all dirs | `preprocessing/pipeline.py:145` |
| PRE-19 | Broad except Exception in worker | `preprocessing/parallel.py:61` |
| REP-1 | __getattr__ shadow potential | `representation/graph_extractor.py:47-63` |
| REP-4 | Re-import inside if block | `representation/orchestrator.py:219-221` |
| REP-10 | Modifiers mixed with functions | `representation/cfg_builder.py:265` |
| REP-11 | lines[0] without check | `representation/cfg_builder.py:177` |
| CLI-2 | textwrap import at top | `cli.py:25` |
| CLI-3 | Source list falls back to wrong key | `cli.py:145-147` |

---

## 7. Design Decision Compliance

| Decision | Status | Notes |
|----------|--------|-------|
| D-0.1 Module location | ✅ PASS | `Data/sentinel_data/` correct |
| D-0.2 Package boundary | ✅ PASS | No sentinel-ml references in data code |
| D-0.3 Standalone venv | ✅ PASS | `.venv/` independent |
| D-0.4 Stub vs real code | ✅ PASS | Real implementations for Stage 1-2, stubs for Stage 4+ |
| D-0.5 DVC orchestrator | ⚠️ WARN | Freshness check exists but not in DVC DAG |
| D-0.6 Config-as-data | ❌ FAIL | Hardcoded paths break portability |
| D-0.7 CLI surface | ⚠️ WARN | `_handle_run` drops stage-specific args |
| D-0.8 Docker | ✅ PASS | Dockerfile present (not reviewed here) |
| D-0.9 Documentation | ✅ PASS | READMEs exist per module |
| D-0.10 ADR | ✅ PASS | ADR-0001 + ADR-0002 committed |
| D-1.1 Connector-per-family | ✅ PASS | 7 connector classes, factory correct |
| D-1.2 Manifest append-only | ❌ FAIL | save() overwrites |
| D-1.3 Pin enforcement | ❌ FAIL | Empty pins accepted silently |
| D-1.4 5-step pipeline | ✅ PASS | flatten→compile→dedup→normalize→segment correct |
| D-1.5 Drop-not-fix | ✅ PASS | Failed compiles → dropped.csv |
| D-1.6 Three-level dedup | ⚠️ WARN | Level 3 stubbed |
| D-1.7 Sidecar meta.json | ⚠️ WARN | 3 fields missing |
| D-2.1 No extraction changes | ✅ PASS | Thin adapter preserves logic |
| D-2.2 Schema v9 | ✅ PASS | All constants correct |
| D-2.3 Tokenizer adapter | ✅ PASS | Clean re-export |
| D-2.4 CFG builder only | ✅ PASS | PDG/callgraph/opcode properly deferred |
| D-2.5 Content-addressed cache | ❌ FAIL | Bug in stale_entries() |
| D-2.6 Sidecar rep.json | ✅ PASS | All fields present |
| D-2.7 Thin-adapter pattern | ⚠️ WARN | cfg_builder.py bypasses it |
| D-2.8 SHA-256 from Stage 1 | ✅ PASS | No MD5 usage |

---

## 8. What's Working Well

1. **Thin-adapter pattern** — Correctly implemented in `graph_extractor.py`, `graph_schema.py`, `tokenizer.py` — byte-identical by construction
2. **v9 schema constants** — All correct (NODE_FEATURE_DIM=12, NUM_NODE_TYPES=14, NUM_EDGE_TYPES=12)
3. **Zip-slip protection** — `manual_connector.py:176-179` properly defends path traversal
4. **Two-pass compilation** — `compiler.py` correctly implements pragma tolerance
5. **has_unchecked_block detection** — Regex correct, populated in sidecar
6. **Drop-not-fix policy** — Failed compiles never enter preprocessed output
7. **Package boundary** — Zero sentinel-ml references in Data module
8. **Multiprocessing** — `parallel.py` mirrors `ml/src/data_extraction/ast_extractor.py` pattern
9. **5-step pipeline** — Correct order: flatten→compile→dedup→normalize→segment
10. **Cache layout** — Content-addressed with versioned sidecar (despite the stale_entries bug)
11. **CFG builder** — Opt-in via `--emit-cfg`, produces JSON-serializable artifact
12. **Label folderize** — Idempotent, handles both flat and folderized sources
13. **Freshness check** — Git upstream + Slither version comparison
14. **Connector factory** — Single registry maps connector type to class
15. **CLI structure** — Well-organized subparsers, per-stage flags, dry-run support

---

## 9. Conclusion

Stages 0-2 are **~85% complete** with 14 critical/high-priority issues found. The core implementation is sound — the thin-adapter pattern is correctly used (except in `cfg_builder.py`), the v9 schema constants are correct, the 5-step preprocessing pipeline works, and the content-addressed cache design is right.

**The most critical issues are:**

1. **REP-2: Class order mismatch** — `graph_schema.py` has a different `CLASS_NAMES` order than Stage 3's `taxonomy.yaml`. This will cause training labels to be misaligned with graph features. **CRITICAL — must fix before any training run.**

2. **REP-7: stale_entries() returns wrong sha** — The stale cache eviction function returns `abc123.rep` instead of `abc123`, so stale cache entries are never actually evicted. This means schema/extractor version changes will not trigger re-extraction, and the v8/v9 mixed-graph bug from Run 8 could recur. **CRITICAL — must fix before Run 11.**

3. **ING-1: Pin enforcement missing** — Empty pins are accepted silently, breaking reproducibility. The plan D-1.3 requires validation.

4. **PRE-4: Dedup Level 3 stubbed** — 38.8% of BCCC was duplication; the AST near-dup detector is entirely stubbed, leaving a known gap in the pipeline.

5. **PRE-15: Sidecar missing fields** — Plan D-1.7 requires `inheritance_root`, `n_imports`, `contract_count` in the sidecar; none are present.

6. **CROSS-1: Hardcoded paths** — `/home/motafeq/...` paths in config.yaml break portability per D-0.6.

The audit found 14 FAIL-level issues that must be addressed, 22 WARN-level issues that should be fixed, and 60 PASS-level items. The implementation is broadly correct but has specific bugs that will cause production failures.
