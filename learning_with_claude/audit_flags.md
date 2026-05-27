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

---

## A10 — `graph_extractor.py` — `except Exception: pass` in `_cfg_node_type`
**File:** `ml/src/preprocessing/graph_extractor.py`
**Location:** `_cfg_node_type()`, lines 493–494: `except Exception: pass`
**Issue:** The entire priority-classification logic — all 4 priority branches — is wrapped in a
single `except Exception: pass`. If Slither renames `NodeType.IF`, moves `LowLevelCall`, or
changes `state_variables_written`, the exception is swallowed and every node silently becomes
`CFG_NODE_OTHER` (12). CALL nodes become OTHER: the GNN sees no external calls. WRITE nodes
become OTHER: reentrancy patterns vanish. Zero crash, zero log — silent total loss of CFG signal.
**Fix:** Move each import to module level with a clear ImportError guard. Inside the function,
catch only the specific exceptions that Slither attribute access can raise
(`AttributeError`, `TypeError`). Let unexpected exceptions propagate.
**Severity:** Medium
**Status:** Open
**Raised:** Session 3, Chunk 3

---

## A11 — `graph_extractor.py` — hardcoded parent feature indices in `_build_cfg_node_features`
**File:** `ml/src/preprocessing/graph_extractor.py`
**Location:** `_build_cfg_node_features()`, lines 543–547: `p[1]`, `p[3]`, `p[4]`, `p[5]`, `p[9]`
**Issue:** Parent feature dims are accessed by raw integer index. The comment names them
(visibility=1, view=3, payable=4, complexity=5, has_loop=9) but nothing enforces that contract.
If `graph_schema.py` reorders `FEATURE_NAMES` — e.g. inserting a new dim before index 3 — these
indices silently read the wrong feature (e.g. `p[3]` might now be `view` → `payable`, swapping
two values). The `len(p) > 9` guard for has_loop means if parent_features is shorter than 10
elements, has_loop falls back to 0.0 for every CFG node — all DoS loop detection degrades silently.
**Fix:** Use `FEATURE_NAMES.index("has_loop")` etc. to derive indices at parse time, or define
named constants `_F_VISIBILITY = 1` etc. in graph_schema.py alongside FEATURE_NAMES.
**Severity:** Medium
**Status:** Open
**Raised:** Session 3, Chunk 3

---

## A12 — `graph_extractor.py` — `n.node_id` accessed without `getattr` fallback in sort key
**File:** `ml/src/preprocessing/graph_extractor.py`
**Location:** `_build_control_flow_edges()`, line 611: `n.node_id`
**Issue:** The CFG sort key uses `n.node_id` directly (no `getattr` with default). If a Slither
version does not attach `node_id` to synthetic nodes (ENTRY_POINT, BEGIN_LOOP, etc.), this raises
`AttributeError` inside the `sorted()` call, aborting CFG construction for the entire function
with no edges produced — silently dropped with zero graph signal for that function.
**Fix:** `getattr(n, "node_id", 0)` in the sort key lambda.
**Severity:** Low
**Status:** Open
**Raised:** Session 3, Chunk 3

---

## A13 — `graph_extractor.py` — silently dropped CONTROL_FLOW edges not logged
**File:** `ml/src/preprocessing/graph_extractor.py`
**Location:** `_build_control_flow_edges()`, lines 639–641: `if successor in node_index_map`
**Issue:** When a CFG successor node is not in `node_index_map` (e.g. a cross-function edge or
Slither synthetic node not added in Pass 1), the edge is silently dropped. No log, no counter.
In a contract with complex control flow, entire branches may produce zero CONTROL_FLOW edges.
The GNN's Phase 2 (which uses CONTROL_FLOW(6) edges) operates on a structurally incomplete graph.
**Fix:** Log a debug/warning when a successor is not found:
`logger.debug("CFG edge dropped: successor %s not in node_index_map for func %s", successor, func.name)`
**Severity:** Low
**Status:** Open
**Raised:** Session 3, Chunk 3

---

## A14 — `graph_extractor.py` — RETURN_TO creates cartesian product including impossible revert→normal paths
**File:** `ml/src/preprocessing/graph_extractor.py`
**Location:** `_add_icfg_edges()`, lines 699–706: cartesian product of callee_terminals × call_site_sons
**Issue:** Every terminal of the callee (including revert/throw terminals) gets a RETURN_TO edge
to every successor of the call site (post-call code). A revert terminal unwinds the call stack and
never transfers control to post-call code — but the graph models this as a possible path. The GNN
can form spurious reentrancy patterns: a node that always reverts appears to "reach" the code after
the call. A full ICFG would distinguish normal-return terminals from exceptional-exit terminals.
**Fix:** Classify callee terminal nodes as normal-exit or exceptional-exit (revert/throw). Only
wire normal-exit terminals to call-site successors. This requires Slither to expose CFG node type
for THROW/REVERT nodes, which it does via `NodeType.THROW`.
**Severity:** Medium
**Status:** Open
**Raised:** Session 4, Chunk 4

---

## A15 — `graph_extractor.py` — DEF_USE def_map keyed by variable name, not object identity
**File:** `ml/src/preprocessing/graph_extractor.py`
**Location:** `_add_def_use_edges()`, line 752: `def_map.setdefault(lval.name, [])`
**Issue:** Two different `LocalVariable` objects with the same name (e.g. loop variable `i` in
nested scopes, or Solidity variable shadowing) are conflated in `def_map`. Reads of the inner
variable appear as uses of the outer variable's definitions, producing spurious DEF_USE edges
that model data flow that cannot occur. Slither's SSA partially mitigates this but `.name` does
not always include SSA suffixes depending on the Slither version.
**Fix:** Key `def_map` by the variable object itself (Python objects are hashable by identity)
rather than by `.name`: `def_map.setdefault(id(lval), []).append(node_idx)` and match on
`id(var)` in Pass 2 (requires building a secondary `var_id_to_name` lookup or using object keys).
**Severity:** Medium
**Status:** Open
**Raised:** Session 4, Chunk 4

---

## A16 — `graph_extractor.py` — `assert` for sentinel range in `_build_node_features`
**File:** `ml/src/preprocessing/graph_extractor.py`
**Location:** `_build_node_features()`, lines 856–857: `assert return_ignored in (-1.0, 0.0, 1.0)`
**Issue:** Same pattern as A4. Under `python -O` / `PYTHONOPTIMIZE=1`, these asserts are silently
removed. If a future change to `_compute_return_ignored` or `_compute_call_target_typed` returns
an out-of-range value, it passes through to the model's feature vector with no crash and no log.
**Fix:** Replace with explicit checks:
```python
if return_ignored not in (-1.0, 0.0, 1.0):
    raise SlitherParseError(f"return_ignored out of range: {return_ignored}")
if call_target_typed not in (-1.0, 0.0, 1.0):
    raise SlitherParseError(f"call_target_typed out of range: {call_target_typed}")
```
**Severity:** Low
**Status:** Open
**Raised:** Session 4, Chunk 4

---

## A17 — `graph_extractor.py` — Slither exception routing by string keyword matching
**File:** `ml/src/preprocessing/graph_extractor.py`
**Location:** `extract_contract_graph()`, lines 1059–1067
**Issue:** `Slither(...)` failures are routed to `SolcCompilationError` vs `SlitherParseError`
by checking if `str(exc).lower()` contains keywords like "compil", "syntax", "solc". This is
fragile in two directions: (1) A genuine Slither internal error whose message mentions "compiler"
gets misclassified as user fault → API returns HTTP 400 instead of HTTP 500. (2) A Solidity
syntax error whose message doesn't match keywords (non-English locale, future Slither version)
gets misclassified as system fault → API returns HTTP 500 for bad user input. Both mask the
real failure category.
**Fix:** Catch specific exception types from Slither's and crytic-compile's public exception
hierarchy: `slither.exceptions.SlitherError`, `crytic_compile.CryticCompileException` etc.
instead of pattern-matching the string representation.
**Severity:** Medium
**Status:** Open
**Raised:** Session 5, Chunk 5

---

## A18 — `graph_extractor.py` — `except Exception: pass` when building ICFG maps
**File:** `ml/src/preprocessing/graph_extractor.py`
**Location:** `extract_contract_graph()`, lines 1154–1167: inner `try/except Exception: pass`
**Issue:** The block that populates `_func_entry_map` and `_func_terminal_map` for a given
function is wrapped in a bare `except Exception: pass`. If the `NodeType` import fails or any
Slither attribute raises, the maps for this function are silently left empty. Then
`_add_icfg_edges` finds no entry point and no terminals for calls into this function,
producing zero CALL_ENTRY and RETURN_TO edges for all callers. The GNN's inter-procedural
reentrancy detection loses these edges with zero signal: no log, no counter, no warning.
**Fix:** At minimum, `logger.warning(...)` with the function name and exception. Ideally catch
only `ImportError, AttributeError` and let unexpected exceptions propagate.
**Severity:** Medium
**Status:** Open
**Raised:** Session 5, Chunk 5
