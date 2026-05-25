# Audit Flags — Learning With Claude

All issues found during teaching. Never delete entries — only append.
Format: one flag per entry, with file, location, description, severity, and status.

Severity scale: **Low** (code smell, clarity issue) | **Medium** (silent bug risk, design gap) | **High** (correctness bug, data corruption risk)
Status: **Open** (not yet fixed) | **Noted** (acknowledged, fix deferred) | **Fixed** (resolved in codebase)

---

## A1 — `graph_schema.py` — Type ID normalization missing range guard
**File:** `ml/src/preprocessing/graph_schema.py`
**Location:** `NODE_FEATURE_DIM` docstring + `FEATURE_NAMES` description of `type_id`
**Issue:** `type_id` is documented as `float(NODE_TYPES[kind]) / 12.0`, implying [0,1] range.
But there is no assertion that `max(NODE_TYPES.values()) == 12`. If a 14th node type is added
(id=13), the normalization silently produces values > 1.0, violating the feature range contract
and destabilising GNN dot products. The existing `assert len(NODE_TYPES) == 13` catches count
but not the max value.
**Fix:** Add `assert max(NODE_TYPES.values()) == 12` in `graph_schema.py` import-time checks.
**Severity:** Medium
**Status:** Open
**Raised:** Session 1

---

## A2 — `hash_utils.py` — `validate_hash` accepts uppercase hex
**File:** `ml/src/utils/hash_utils.py`
**Location:** `validate_hash()` function
**Issue:** Uses `int(hash_string, 16)` to validate hex format, which accepts uppercase
characters (A-F). But `hashlib.md5().hexdigest()` always returns lowercase. The validator
is silently permissive — it validates hashes the system never produces. If a component
accidentally produces uppercase hashes, `validate_hash` says "valid" but the hash won't
match any `.pt` filename (which are always lowercase), causing silent lookup failure.
**Fix:** Replace with `re.match(r'^[0-9a-f]{32}$', hash_string)` to precisely match the
actual contract of `hexdigest()`.
**Severity:** Low
**Status:** Open
**Raised:** Session 1

---

## A3 — `graph_extractor.py` — `_MAX_TYPE_ID` is dynamic, contradicts schema
**File:** `ml/src/preprocessing/graph_extractor.py`
**Location:** Line 113: `_MAX_TYPE_ID = float(max(NODE_TYPES.values()))`
**Issue:** This is a dynamic computation — it changes automatically whenever `NODE_TYPES`
gains a new entry. This contradicts `graph_schema.py`'s documented `/12.0` normalization.
If a new node type is added without retraining the model:
  1. ALL existing type_id normalizations shift (e.g. CFG_NODE_OTHER: 1.0 → 0.923).
  2. The new type collides with the old max type's normalized value (both → 1.0).
The model was trained with the old normalizations and silently receives wrong features.
No crash, no warning.
**Fix:** Hardcode `_MAX_TYPE_ID: float = 12.0` with a comment pointing to `graph_schema.py`
and the schema change policy. Adding a new node type should require a conscious change here too.
**Severity:** Medium
**Status:** Open
**Raised:** Session 2, Chunk 1

---

## A4 — `graph_extractor.py` — `assert` used for production invariant check
**File:** `ml/src/preprocessing/graph_extractor.py`
**Location:** Line 1250: `assert len(node_metadata) == x.shape[0], (...)`
**Issue:** Python `assert` statements are silently removed when running with `python -O`
(optimization flag) or `PYTHONOPTIMIZE=1`, which is common in production batch deployments.
This check guards the critical invariant that `node_metadata` is index-aligned with `graph.x`.
If violated silently, node metadata lookups return data for the wrong node — e.g. the
cross-attention fusion routes the wrong transformer context to the wrong graph node.
**Fix:** Replace with an explicit check:
```python
if len(node_metadata) != x.shape[0]:
    raise SlitherParseError(
        f"node_metadata length {len(node_metadata)} ≠ x.shape[0] {x.shape[0]} "
        f"for '{contract.name}'. Bug in node construction."
    )
```
**Severity:** Medium
**Status:** Open
**Raised:** Session 2, Chunk 1

---

## A5 — `graph_extractor.py` — `except AttributeError` scope too broad in `_compute_return_ignored`
**File:** `ml/src/preprocessing/graph_extractor.py`
**Location:** `_compute_return_ignored()`, outer `except AttributeError` block
**Issue:** The broad `except AttributeError` catches any AttributeError from inside the entire
function body — including programming errors like refactored field names. A bug introduced during
refactoring would silently return `-1.0` (sentinel) instead of crashing, hiding the error.
**Fix:** Tighten the try scope to only wrap `func.nodes` access. Inner loop errors should propagate.
**Severity:** Low
**Status:** Open
**Raised:** Session 2, Chunk 2

---

## A6 — `graph_extractor.py` — `except Exception: pass` in `_compute_call_target_typed`
**File:** `ml/src/preprocessing/graph_extractor.py`
**Location:** `_compute_call_target_typed()`, line ~312
**Issue:** `except Exception: pass` is maximally broad — catches all exceptions including
`SystemExit` subclasses and hides Slither API changes. If Slither renames a field and
`recv_type.name` raises `AttributeError`, the code silently falls to the regex scan instead
of surfacing the API breakage. Should be `except (ImportError, AttributeError, TypeError)`.
**Severity:** Medium
**Status:** Open
**Raised:** Session 2, Chunk 2

---

## A7 — `graph_extractor.py` — `_compute_in_unchecked` is dead code
**File:** `ml/src/preprocessing/graph_extractor.py`
**Location:** Lines 331–360, `_compute_in_unchecked()` function
**Issue:** Function is marked DEPRECATED with comment "safe to delete after v8 extraction is
complete." v8 is the current schema. The function is never called anywhere in the file.
Dead code adds maintenance burden and confusion.
**Fix:** Delete the function entirely. Verify no tests, docstrings, or scripts reference it first.
**Severity:** Low
**Status:** Open
**Raised:** Session 2, Chunk 2

---

## A8 — `graph_extractor.py` — `is True` identity check misses truthy non-booleans
**File:** `ml/src/preprocessing/graph_extractor.py`
**Location:** `_compute_has_loop()`, fallback: `getattr(func, "is_loop_present", None) is True`
**Issue:** `is True` uses identity comparison, not truthiness. If `is_loop_present` returns
integer `1` or any truthy non-boolean, the check silently returns `False` — loop presence missed.
**Fix:** Replace with `bool(getattr(func, "is_loop_present", False))`.
**Severity:** Low
**Status:** Open
**Raised:** Session 2, Chunk 2

---

## A9 — `graph_extractor.py` — string-based type check fragile on Slither class rename
**File:** `ml/src/preprocessing/graph_extractor.py`
**Location:** `_compute_uses_block_globals()`: `type(rv).__name__ == "SolidityVariableComposed"`
**Issue:** If Slither renames the class (not just moves it), this check silently returns 0.0
for all timestamp-related variables. The `Timestamp` and `TOD` vulnerability classes lose
their primary direct signal with no warning. No crash, no log — pure silent miss.
**Fix:** Try the import first, fall back to string check only on ImportError:
```python
try:
    from slither.core.variables.variable import SolidityVariableComposed as _SVC
    _is_svc = lambda rv: isinstance(rv, _SVC)
except ImportError:
    _is_svc = lambda rv: type(rv).__name__ == "SolidityVariableComposed"
```
**Severity:** Medium
**Status:** Open
**Raised:** Session 2, Chunk 2
