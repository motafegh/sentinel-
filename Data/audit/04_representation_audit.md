# Audit Report — Representation Module (Stage 2)

**Scope:** `__init__.py`, `graph_schema.py`, `graph_extractor.py`, `tokenizer.py`, `orchestrator.py`, `cache_manager.py`, `versioner.py`, `cfg_builder.py`, deferred stubs, `_schema_constants.md`, `_schema_version_registry.json`
**Plan Reference:** `03_stage_2_representation.md` (D-2.1 through D-2.8)

---

## Executive Summary

| Category | PASS | WARN | FAIL |
|----------|------|------|------|
| Thin-adapter correctness | ✅ | | |
| v9 schema constants | ✅ | | |
| Cache/versioner logic | | ⚠️ | ❌ |
| Orchestrator | ✅ | ⚠️ | |
| CFG builder | ✅ | ⚠️ | |
| Deferred stubs | ✅ | | |
| Schema docs | ✅ | | |
| **Total** | **~40** | **4** | **1** |

**Overall: PASS with 1 critical bug in cache_manager.py**

---

## 1. `__init__.py` (64 lines)

| Check | Status | Detail |
|-------|--------|--------|
| Re-exports from graph_schema | PASS | All 12 schema symbols re-exported (lines 18-32) |
| Re-exports from graph_extractor | PASS | All 6 extractor symbols re-exported (lines 33-40) |
| `__all__` complete | PASS | 18 entries, matches imports |
| Missing tokenizer re-export | **WARN** | `tokenizer.py` not re-exported from `__init__.py`. Correct per D-2.7 but inconsistent if downstream expects single import surface. |
| Docstring accuracy | PASS | Correctly documents Stage 0→Stage 2→Stage 7 lifecycle |

**No bugs found.**

---

## 2. `graph_schema.py` (148 lines)

| Check | Status | Detail |
|-------|--------|--------|
| Thin-adapter pattern | PASS | Eager re-export from `ml.src.preprocessing.graph_schema` (line 130-141) |
| `_MAX_TYPE_ID` derivation | PASS | Correctly derives `float(max(NODE_TYPES.values()))` at line 145 |
| `CLASS_NAMES` definition | PASS | Correctly defined locally (lines 73-84) |
| `NUM_CLASSES` derivation | PASS | `len(CLASS_NAMES)` = 10 (line 85) |
| `__getattr__` lazy fallback | PASS | Graceful fallback for standalone PyPI install |
| `is` equality guarantee | PASS | Same Python object, two import names |

### LOW: `__getattr__` unreachable branch (lines 100-106)

`_LOCAL_ATTRS = frozenset(("_MAX_TYPE_ID", "CLASS_NAMES", "NUM_CLASSES"))`. These are all eagerly defined at module level. `__getattr__` is only called when normal `__dict__` lookup fails. Since these names are always in `__dict__`, the branch at line 100-106 is **dead code** — the `AttributeError` is never raised. Harmless but misleading.

**Fix:** Remove the `_LOCAL_ATTRS` check or add a comment that this is a defensive guard.

### MAINTENANCE TRAP: `CLASS_NAMES` not in `_LIVE_SCHEMA_ATTRS`

If someone removed the eager import block (lines 130-141), `CLASS_NAMES` is not in `_LIVE_SCHEMA_ATTRS` (line 38-58), so `__getattr__` would raise `AttributeError` instead of returning the local value. Not a bug today but a maintenance trap.

---

## 3. `graph_extractor.py` (77 lines)

| Check | Status | Detail |
|-------|--------|--------|
| Thin-adapter pattern | PASS | Clean eager re-export (lines 67-74) |
| Symbol list matches live module | PASS | All 6 public symbols exported |
| `__getattr__` fallback | PASS | Consistent pattern with graph_schema.py |
| Lazy import support | PASS | Standalone PyPI install supported |

**No bugs found.** Cleanest file in the module.

---

## 4. `tokenizer.py` (72 lines)

| Check | Status | Detail |
|-------|--------|--------|
| Thin-adapter pattern | PASS | Eager re-export from `ml.src.data_extraction.windowed_tokenizer` |
| Correct module target | PASS | Points to `windowed_tokenizer` (not wrong `tokenizer.py`) |
| Docstring documents fix | PASS | Lines 4, 12-14 document the wrong-module correction |
| Symbol list | PASS | `tokenize_windowed_contract`, `init_worker`, 4 constants |

**No bugs found.**

---

## 5. `orchestrator.py` (344 lines)

| Check | Status | Line(s) | Detail |
|-------|--------|---------|--------|
| Reads `.meta.json` from Stage 1 | PASS | 133-134 | Correctly reads `<sha256>.meta.json` |
| Content-addressed output | PASS | — | `<sha256>.{pt, tokens.pt, rep.json}` naming |
| Cache check (D-2.5) | PASS | 98-109 | `_is_cached()` checks sidecar versions |
| Sidecar format (D-2.6) | PASS | — | All required fields present |
| SHA-256 from Stage 1 (D-2.8) | PASS | — | `meta["sha256"]` used throughout |
| `--force` support | PASS | 143-145 | Deletes all 3 files |
| Dry-run mode | PASS | 294-299 | |
| Progress reporting | PASS | 337-341 | |
| Idempotent | PASS | — | Cache hits skip reprocessing |

### LOW: Dead parameter `cfg: dict` (line 251)

The `cfg: dict` parameter is accepted but **never used** anywhere in the function body. The function reads from `data_dir / "preprocessed"` and writes to `data_dir / "representations"` — neither path comes from `cfg`.

**Fix:** Remove `cfg: dict` from the signature, or add `# TODO: use cfg for source validation`.

### WARN: Lazy imports inside hot loop

Lines 125, 157-160, 176: `FEATURE_SCHEMA_VERSION`, `extract_contract_graph`, `GraphExtractionConfig`, and `tokenize_windowed_contract` are imported inside `_extract_one()` which is called per-contract. Python caches module imports, so this is O(1) after the first call, but it's an unnecessary pattern.

### WARN: Fragile `allow_paths`

Line 164: `allow_paths=[str(sol_path.parent.parent.parent.parent)]` — hard-codes a 4-level parent traversal. If the directory layout changes, this breaks silently.

---

## 6. `cache_manager.py` (119 lines)

| Check | Status | Line(s) | Detail |
|-------|--------|---------|--------|
| Content-addressed (D-2.5) | PASS | — | Key = `(sha256, schema_version, extractor_version)` |
| `is_cached()` checks 3 files | PASS | 48-53 | sidecar + .pt + .tokens.pt |
| `invalidate()` removes all 3 | PASS | 63-69 | |
| `evict_stale()` | PASS | — | Calls `stale_entries()` then `invalidate()` |

### FAIL: `stale_entries()` sha256 extraction bug (line 95, MEDIUM severity)

```python
# Line 95
sha = rep_path.stem
```

For `<sha256>.rep.json`, `Path("abc123.rep.json").stem` returns `"abc123.rep"` — **not** `"abc123"`. Compare with `list_cached_sha256s()` at line 79 which correctly uses `rep_path.stem.removesuffix(".rep")`.

**Current behavior:** `stale_entries()` returns sha256 strings with a `.rep` suffix, then `invalidate()` tries to delete `<sha>.rep.json` using that wrong name — **it silently fails to delete stale entries**.

**Impact:** HIGH — when schema/extractor versions change, `evict_stale()` calls `stale_entries()` which returns wrong sha256s, then `invalidate()` can't find the files. **Stale cache entries are never evicted.** This is the core safety mechanism against the "silent mix of versions" failure mode from Run 8.

**Fix:**
```python
# Line 95 — change:
sha = rep_path.stem
# to:
sha = rep_path.stem.removesuffix(".rep") if rep_path.stem.endswith(".rep") else rep_path.stem
```

Or use the same logic as `list_cached_sha256s()`:
```python
sha = rep_path.name.split(".")[0]
```

---

## 7. `versioner.py` (100 lines)

| Check | Status | Detail |
|-------|--------|--------|
| Registry format (D-2.5) | PASS | Schema + extractor + timestamp |
| `check_and_evict()` logic | PASS | Compares registry vs live, evicts if mismatched |
| `current_versions()` | PASS | Returns live values from thin adapters |
| `write_registry()` | PASS | Creates parent dir, writes JSON |

**No bugs found.** Clean and correct.

---

## 8. `cfg_builder.py` (274 lines)

| Check | Status | Detail |
|-------|--------|--------|
| Opt-in via `--emit-cfg` | PASS | Documented in orchestrator |
| Self-contained | PASS | No dependency on thin adapters for core logic |
| CfgArtifact JSON-serializable | PASS | `dataclasses.asdict()` at line 80 |
| Slither IR usage | PASS | Uses `func.nodes`, `node.sons` — same IR as graph_extractor |
| Loop detection (DFS) | PASS | `_count_back_edges()` correctly counts back-edges |
| Node type classification | PASS | Checks CALL/WRITE/READ/CHECK/ARITH with correct priority |

### WARN: Imports bypass thin adapter

Line 243: `from ml.src.preprocessing.graph_schema import FEATURE_SCHEMA_VERSION` — imports directly from `ml.src`, bypassing the thin adapter. Breaks the "single import path through thin adapter" principle.

**Fix:** Use `from sentinel_data.representation.graph_schema import FEATURE_SCHEMA_VERSION`.

### WARN: Docstring inconsistency

Docstring at line 11 says `build_cfg(sol_path, config) -> CfgArtifact | None` but the signature at line 229 returns `CfgArtifact` (never None). The `error` field handles failures instead.

---

## 9–11. Deferred Stubs

| File | Status | Detail |
|------|--------|--------|
| `call_graph.py` (30 lines) | PASS | Clear deferral docstring, `NotImplementedError`, planned API documented |
| `opcode_extractor.py` (31 lines) | PASS | Clear deferral docstring, `NotImplementedError`, planned API documented |
| `pdg_builder.py` (29 lines) | PASS | Clear deferral docstring, `NotImplementedError`, planned API documented |

**All properly deferred per AUDIT_PATCHES 2-P9.**

---

## 12. `_schema_constants.md` (99 lines)

| Check | Status | Detail |
|-------|--------|--------|
| v9 constants correct | PASS | All 6 values match live module |
| Feature vector layout | PASS | 12 features, indices 0-11, correct names |
| Node types | PASS | 14 types, ids 0-13 |
| Edge types | PASS | 12 types, ids 0-11 |
| Class order | PASS | 10 classes, indices 0-9 |

**No issues.** Accurate reference document.

---

## 13. `_schema_version_registry.json` (37 lines)

| Check | Status | Detail |
|-------|--------|--------|
| v9 values | PASS | All correct |
| v8 history | PASS | Correct previous values |
| Checkpoints documented | PASS | v9 and v8 checkpoint paths listed |

**No issues.**

---

## Cross-Cutting: Stage 0 Stub Bug Fix (D-2.2)

| Check | Status | Detail |
|-------|--------|--------|
| `NODE_TYPES: dict[str, int]` | PASS | Live module: `dict[str, int] = {"STATE_VAR": 0, ...}` |
| `EDGE_TYPES: dict[str, int]` | PASS | Live module: `dict[str, int] = {"CALLS": 0, ...}` |
| `FEATURE_NAMES: tuple[str, ...]` | PASS | Live module: `tuple[str, ...] = (...)` |
| Feature schema version = "v9" | PASS | Verified |
| `_MAX_TYPE_ID = 13.0` | PASS | Derived correctly |

**The Stage 0 dict-direction bug has been fixed.**

---

## Cross-Cutting: Thin-Adapter Pattern

| Check | Status | Detail |
|-------|--------|--------|
| `is` equality guarantee | PASS | Same Python object, different import name |
| `__getattr__` fallback | PASS | All 3 adapter files have it |
| Bug-fix-once principle | PASS | All logic lives in `ml/`, adapters just re-export |

---

## Cross-Cutting: Content-Addressed Cache (D-2.5)

| Check | Status | Detail |
|-------|--------|--------|
| Cache key = (sha256, schema_v, extractor_v) | PASS | Checked in `_is_cached()` and `is_cached()` |
| 3-file consistency | PASS | `.pt` + `.tokens.pt` + `.rep.json` all checked |
| Force bypass | PASS | `--force` deletes all 3 files |
| **Stale entry eviction** | **FAIL** | Bug in `stale_entries()` — stale entries never evicted |

---

## Summary of Findings

### Bugs (Fix Required)

| # | File | Line | Severity | Description |
|---|------|------|----------|-------------|
| **BUG 1** | `cache_manager.py` | 95 | **HIGH** | `stale_entries()` uses `rep_path.stem` which gives `"abc123.rep"` not `"abc123"`. Stale cache entries are **never evicted** on version bump. |
| BUG 2 | `orchestrator.py` | 251 | LOW | Dead parameter `cfg: dict` — accepted but never used |
| BUG 3 | `graph_schema.py` | 100-106 | LOW | `__getattr__` branch for `_LOCAL_ATTRS` is unreachable dead code |

### Warnings (Should Fix)

| # | File | Line | Description |
|---|------|------|-------------|
| WARN 1 | `cfg_builder.py` | 243 | Imports `FEATURE_SCHEMA_VERSION` from `ml.src` directly, bypassing thin adapter |
| WARN 2 | `cfg_builder.py` | 11 vs 229 | Docstring says `-> CfgArtifact | None` but returns `CfgArtifact` |
| WARN 3 | `orchestrator.py` | 164 | Hard-coded 4-level parent traversal for `allow_paths` — fragile |
| WARN 4 | `orchestrator.py` | 79 | `RepresentResult.extractor_version` default is `"v2.0-thin-adapter"` but actual is `"v2.1-windowed-gcb"` |

### No Issues (Clean Files)

- `graph_extractor.py` ✅
- `tokenizer.py` ✅
- `versioner.py` ✅
- `call_graph.py` ✅ (deferred correctly)
- `opcode_extractor.py` ✅ (deferred correctly)
- `pdg_builder.py` ✅ (deferred correctly)
- `_schema_constants.md` ✅
- `_schema_version_registry.json` ✅

---

## Recommended Immediate Action

**Fix BUG 1 (`cache_manager.py:95`)** — it silently breaks the versioner's cache invalidation, which is the core safety mechanism against the "silent mix of versions" failure mode from Run 8. This is a 2-line fix.
