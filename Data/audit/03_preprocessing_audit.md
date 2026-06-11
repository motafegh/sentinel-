# Audit Report — Preprocessing Module (Stage 1)

**Scope:** `pipeline.py`, `preprocess.py`, `flattener.py`, `compiler.py`, `deduplicator.py`, `normalizer.py`, `segmenter.py`, `parallel.py`, `_transitive_strip.py`
**Plan Reference:** `02_stage_1_ingest_preprocess.md` (D-1.4 through D-1.7)

---

## Executive Summary

| Category | PASS | WARN | FAIL |
|----------|------|------|------|
| `__init__.py` | 0 | 1 | 0 |
| `pipeline.py` | 7 | 2 | 0 |
| `preprocess.py` | 5 | 2 | 0 |
| `flattener.py` | 8 | 2 | 0 |
| `compiler.py` | 7 | 0 | 1 |
| `deduplicator.py` | 4 | 1 | 1 |
| `normalizer.py` | 7 | 1 | 0 |
| `segmenter.py` | 5 | 1 | 0 |
| `parallel.py` | 6 | 2 | 0 |
| `_transitive_strip.py` | 4 | 1 | 0 |
| Cross-cutting | 8 | 2 | 1 |
| **Total** | **61** | **15** | **3** |

---

## 1. `__init__.py`

| Check | Status | Detail |
|-------|--------|--------|
| Empty file | **WARN** | No public API exported. Consumers must use full module paths. Should export `PreprocessingPipeline`, `ContractMeta`, `PipelineResult`. |

---

## 2. `pipeline.py` (228 lines)

| Check | Status | Line(s) | Detail |
|-------|--------|---------|--------|
| 5-step pipeline order | PASS | 106-177 | flatten→compile→dedup→normalize→segment+bucket. Correct per D-1.4. |
| Drop-not-fix policy | PASS | 151-171 | Compile failures and duplicates → `drop_row` dicts → `dropped.csv`. Files never enter `preprocessed/`. |
| ContractMeta sidecar schema | **WARN** | 33-53 | See D-1.7 schema mismatch below |
| `_write_meta` | PASS | 205-207 | `dataclasses.asdict()` + JSON. Clean. |
| `_write_dropped` | PASS | 210-228 | Defensive field union across different drop-reason dicts. |
| Cleanup of temp files | PASS | 130-149 | `finally` block cleans `.sentinel_compile_*` and `.sentinel_stripped.sol` siblings. |
| `n_raw` line count | PASS | 104 | `source.count("\n") + 1` — correct for empty files. |
| Schema version | PASS | 30 | `META_SCHEMA_VERSION = "1"`. Present in every meta.json. |

### D-1.7 Sidecar Schema Mismatch

Plan specifies these fields; actual implementation diverges:

| Plan Field (D-1.7) | Implementation | Status |
|--------------------|----------------|--------|
| `sha256` | `sha256` | PASS |
| `source` | `source_name` | **WARN** — renamed, breaks plan contract |
| `original_path` | `original_path` | PASS |
| `contract_count` | `contract_names` (list) | **WARN** — different name + type (list vs int) |
| `version_bucket` | `version_bucket` | PASS |
| `inheritance_root` | — | **FAIL** — missing entirely |
| `dedup_group_id` | `dedup_group_id` | PASS |
| `parent_sha256` | `duplicate_of` | **WARN** — renamed |
| `pragma` | `pragma` | PASS |
| `solc_version` | `solc_version` | PASS |
| `compile_status` | `compile_status` | PASS |
| `compile_error` | `compile_error` | PASS |
| `n_imports` | — | **FAIL** — missing entirely |
| `n_normalized_lines` | `n_normalized_lines` | PASS |
| — | `flatten_status` | Extra field (not in D-1.7) |
| — | `is_duplicate` | Extra field (not in D-1.7) |
| — | `has_unchecked_block` | Extra field (not in D-1.7) |

**Impact:** MEDIUM — downstream stages (representation, labeling) may depend on field names. The `source`→`source_name` rename is the most dangerous.

---

## 3. `preprocess.py` (256 lines)

| Check | Status | Line(s) | Detail |
|-------|--------|---------|--------|
| CLI service entry point | PASS | 29-101 | `preprocess_source` + `preprocess_all`. Correctly reads manifest, supports `--sample`, `--retry-failed`. |
| Retry-failed merge | PASS | 129-191 | Rebuilds dropped.csv by joining old + new results. Correctly removes files that now succeed. |
| `_maybe_folderize` | PASS | 194-238 | Label-aware folderization. Idempotent. |
| `_load_dropped_files` | PASS | 104-126 | Reads dropped.csv, reconstructs paths, silently skips missing files. |
| `FileNotFoundError` handling | PASS | 255-256 | `preprocess_all` catches and prints SKIP. |

### WARN: Private import coupling

Line 25: `from sentinel_data.ingestion.ingest import _enabled_sources` — imports a private function from another module. Fragile coupling.

**Fix:** Promote `_enabled_sources` to public API or add a public accessor.

### WARN: `n_workers` parameter unclear

Line 34: `n_workers` parameter accepted but only used if `> 1`. The parameter name and docstring don't clarify that `1` means serial execution.

---

## 4. `flattener.py` (311 lines)

| Check | Status | Line(s) | Detail |
|-------|--------|---------|--------|
| `solc --flatten` primary path | PASS | 70-76 | Subprocess with timeout=30. |
| Fallback: strip unresolved imports | PASS | 88-127 | Two-stage fallback: flatten → strip. |
| Recursive transitive strip | PASS | 145-236 | `_strip_unresolved_imports_recursive` follows relative imports. Uses `_seen` set to prevent cycles. |
| Inheritance parent stripping | PASS | 268-292 | `_strip_unresolved_inheritance` removes parents from `is A, B, C`. |
| `_ASSUMED_BARE_IMPORT_SYMBOLS` | PASS | 40-46 | Conservatively assumes Test, console, console2, Vm, ScriptUtils. |
| `_IMPORT_LINE_RE` regex | PASS | 48-51 | Correctly handles `{A, B}`, `* as X`, bare imports. |
| `_CONTRACT_INHERIT_RE` | PASS | 53-55 | Correctly captures `contract X is A, B {`. |

### WARN: `_pick_solc` repeated imports

Lines 297-310: Imports `_available_versions`, `_parse_version`, `_satisfying_versions`, `_solc_binary` from compiler **inside the function body** (3 times). Should be top-level imports.

### WARN: Temp file naming collision risk

Line 119: Uses `st_mtime_ns` for uniqueness. If two files have the same stem and modification time (unlikely but possible in CI), the temp file would collide. Consider adding a UUID suffix.

---

## 5. `compiler.py` (183 lines)

| Check | Status | Line(s) | Detail |
|-------|--------|---------|--------|
| Two-pass compile | PASS | 63-99 | Pass 1: exact version. Pass 2: satisfying versions (newest first). Correct per D-1.4 + 1-P1/1-P2. |
| Pragma tolerance (spaced pragmas) | PASS | 108-109 | `re.sub(r"\s+", "", m.group(1))` strips whitespace. |
| Exact-version first | PASS | 68-75 | Tries requested version before falling back. |
| `_run_solc` with `--allow-paths` | PASS | 167-183 | Expands allowed paths to `sol_path.parent.parent`. |
| Timeout | PASS | 179 | 30s timeout on solc subprocess. |
| Error truncation | PASS | 97, 183 | Last 300/500 chars of stderr. |
| `_available_versions` | PASS | 149-159 | Reads `~/.solc-select/artifacts/`. Correctly sorts oldest-first. |
| `_satisfying_versions` ceiling | PASS | 125-146 | Handles `<X.Y.Z` ceiling for double-bound pragmas. |

### FAIL: Mutable default in `CompileResult`

Line 34: `attempted_versions: list[str] = None` — mutable default on a dataclass. While `None` + `__post_init__` works, the correct pattern is `field(default_factory=list)`. This is a latent bug if someone accesses the default before `__post_init__`.

**Fix:**
```python
from dataclasses import field
attempted_versions: list[str] = field(default_factory=list)
```

---

## 6. `deduplicator.py` (76 lines)

| Check | Status | Line(s) | Detail |
|-------|--------|---------|--------|
| Level 1: exact SHA-256 | PASS | 42-50 | Correct. |
| Level 2: address-level | PASS | 52-64 | Same Ethereum address → same dedup group. |
| Level 3: AST near-dup | **WARN** | 66-72 | Stubbed. `dedup_group_id=sha256`. |
| Stateful dedup in parallel | PASS | 30-37 | Documented race condition in `parallel.py`. |
| `_sha256` encoding | PASS | 75-76 | `content.encode()` defaults to UTF-8. |

### FAIL: Dedup threshold 0.85 not implemented

The plan's `ast_similarity_threshold=0.85` (1-P4) is not referenced anywhere in this module. The threshold config in `config.yaml` is never read. Level 3 is entirely absent.

**Impact:** HIGH — the 38.8% BCCC duplication rate was caused by near-duplicate contracts. Without Level 3, v2 sources with copy-paste-with-minor-edits patterns will have the same duplication problem.

**Note:** The plan says Level 3 is "deferred to Stage 2" in some places. If this is intentional, it should be clearly documented in the sidecar (`dedup_level: 2` or similar).

---

## 7. `normalizer.py` (35 lines)

| Check | Status | Line(s) | Detail |
|-------|--------|---------|--------|
| Strip SPDX headers | PASS | 16, 25 | `_SPDX_RE` correct. |
| Strip block comments | PASS | 18, 26 | `_BLOCK_CMT` with `re.DOTALL`. |
| Strip line comments | PASS | 17, 27 | `_LINE_CMT` correct. |
| Strip trailing whitespace | PASS | 20, 28 | |
| Collapse multiple newlines | PASS | 19, 29 | `\n{3,}` → `\n\n`. |
| `now` keyword preservation (A9) | PASS | — | `now` is not a comment/SPDX/whitespace. Survives normalization. |
| `n_lines_before`/`n_lines_after` | PASS | 24, 34 | `count('\n') + 1`. Correct. |

**No issues.** Clean, minimal implementation.

---

## 8. `segmenter.py` (64 lines)

| Check | Status | Line(s) | Detail |
|-------|--------|---------|--------|
| Version buckets | PASS | 55-64 | `legacy` (<0.6), `transitional` (0.6-0.7), `modern` (>=0.8). |
| `has_unchecked_block` detection | PASS | 20, 43 | `\bunchecked\s*\{` with word boundary. Correct. |
| `contract_names` extraction | PASS | 18, 40 | `\bcontract\s+(\w+)\s*(?:is\s+[^{]+)?\{`. |
| Unknown pragma → legacy | PASS | 57-58 | Conservative default. |

### WARN: `_CONTRACT_RE` misses `abstract contract`

Line 18: Regex doesn't match `abstract contract X is A {`. Modern Solidity uses `abstract contract` frequently. Would return `contract_names=[]` for abstract contracts.

**Fix:** Update regex to:
```python
r'\b(?:abstract\s+)?contract\s+(\w+)\s*(?:is\s+[^{]+)?\{'
```

---

## 9. `parallel.py` (156 lines)

| Check | Status | Line(s) | Detail |
|-------|--------|---------|--------|
| Module-level picklable worker | PASS | 49-69 | `_process_one_worker` is a plain function. |
| Fresh pipeline per worker | PASS | 56 | Each worker creates its own `PreprocessingPipeline`. |
| Meta written in parent | PASS | 126-128 | `_write_meta` called in parent process. |
| Error isolation | PASS | 59-63 | `try/except` wraps `_process_one`. |
| Chunksize auto-tuning | PASS | 72-77 | `max(1, total // (n_workers * 16))`. |
| Worker errors → dropped.csv | PASS | 139-150 | Worker exceptions appended with `reason=worker_exception`. |

### WARN: `mp` fork safety

Line 119: Uses default `mp` start method (fork on Linux). `solc` subprocesses after fork can be unsafe on some systems.

**Fix:** Use `mp.get_context("spawn")` or document the fork assumption.

---

## 10. `_transitive_strip.py` (104 lines)

| Check | Status | Line(s) | Detail |
|-------|--------|---------|--------|
| Sibling file writing | PASS | 58-67 | Writes `.sentinel_stripped.sol` next to original. |
| Import rewriting | PASS | 85-103 | Rewrites relative imports to point at sentinel-stripped siblings. |
| `relative_to` fallback | PASS | 73-78 | Falls back to absolute path if `relative_to` fails. |

### WARN: `_IMPORT_LINE_RE` duplication

Lines 25-28: Same regex as in `flattener.py` (line 48). DRY violation.

**Fix:** Extract to a shared `_regex.py` or import from one location.

---

## Cross-Cutting Checks

| Check | Status | Detail |
|-------|--------|--------|
| Pipeline order | PASS | flatten→compile→dedup→normalize→segment. All 5 steps correct. |
| Two-pass compile | PASS | Exact version first, then satisfying versions. |
| Dedup threshold 0.85 | **FAIL** | Not implemented. AST near-dup deferred. |
| Sidecar schema (D-1.7) | **WARN** | 3 missing fields, 2 renamed, 3 extra. |
| has_unchecked_block | PASS | Detected via `\bunchecked\s*\{`. |
| Drop-not-fix | PASS | Failed compiles → dropped.csv. |
| Error handling | PASS | All steps catch errors gracefully. |
| Type hints | PASS | All functions have return type hints. |
| A9 regression (now keyword) | PASS | Normalizer doesn't strip `now`. |
| `version_bucketer.py` | **WARN** | Plan references separate file; merged into `segmenter.py`. Acceptable divergence. |

---

## Actionable Fix Priority

| Priority | Finding | File:Line | Effort |
|----------|---------|-----------|--------|
| **P0** | Fix `CompileResult` mutable default | `compiler.py:34` | 1 line |
| **P0** | Document dedup Level 3 stub clearly | `deduplicator.py` | 5 lines |
| **P1** | Add missing sidecar fields | `pipeline.py:33-53` | 30 lines |
| **P1** | Fix sidecar field renames | `pipeline.py` | 10 lines |
| **P2** | Fix `_CONTRACT_RE` for `abstract contract` | `segmenter.py:18` | 1 line |
| **P2** | Move `_pick_solc` imports to top-level | `flattener.py:297-310` | 5 lines |
| **P2** | Extract shared `_IMPORT_LINE_RE` | `flattener.py:48`, `_transitive_strip.py:25` | 10 lines |
| **P2** | Add temp file UUID suffix | `flattener.py:119` | 1 line |
| **P3** | Export public API from `__init__.py` | `__init__.py` | 5 lines |
| **P3** | Promote `_enabled_sources` to public | `ingestion/ingest.py` | 2 lines |
| **P3** | Use `mp.get_context("spawn")` | `parallel.py:119` | 3 lines |
