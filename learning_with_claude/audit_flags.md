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
