# Preprocessing — Chunk 3: Feature Engineering Deep Dive

> **File:** `ml/src/preprocessing/graph_extractor.py` (lines 218–872)
> **What you'll learn:** How each of the 11 features is computed — the implementation details, the bugs that were found and fixed, sentinel values, and why each design decision was made.
> **Time:** ~30 minutes
> **Interview relevance:** ML (feature engineering), AI (signal design), Blockchain (Solidity semantics)

---

## 1. The Feature Vector Builder: `_build_node_features()`

This function takes a Slither AST object and its `type_id` and returns a Python `list[float]` of exactly 11 elements.

```python
def _build_node_features(obj: Any, type_id: int) -> list:
    _is_function = hasattr(obj, "nodes") and hasattr(obj, "pure")
    
    # Compute features...
    
    return [
        float(type_id) / _MAX_TYPE_ID,  # [0]
        visibility,                      # [1]
        uses_block_globals,              # [2]
        view,                            # [3]
        payable,                         # [4]
        complexity,                      # [5]
        loc,                             # [6]
        return_ignored,                  # [7]
        call_target_typed,               # [8]
        has_loop,                        # [9]
        external_call_count,             # [10]
    ]
```

**Key design decision:** Non-function nodes (STATE_VAR, EVENT, MODIFIER) get `0.0` for all function-specific features (2–5, 7, 9–10), except:
- `call_target_typed` defaults to `1.0` (safe default: not applicable)
- `loc` is computed for all node types with `source_mapping`

**How to detect if it's a function:**
```python
_is_function = hasattr(obj, "nodes") and hasattr(obj, "pure")
```
A Slither `Function` object has a `nodes` list (CFG) and a `pure` boolean. State variables and events don't. This is duck-typing rather than `isinstance()` — more robust across Slither versions.

---

## 2. Feature [0]: type_id — Normalised Node Category

```python
float(type_id) / _MAX_TYPE_ID  # 12.0
```

`_MAX_TYPE_ID = float(max(NODE_TYPES.values()))` — derived from the schema, not hardcoded as `12.0`. If a 14th node type is added, this auto-updates.

Examples:
- STATE_VAR (0) → `0.0 / 12 = 0.0`
- FUNCTION (1) → `1/12 ≈ 0.083`
- CONTRACT (7) → `7/12 ≈ 0.583`
- CFG_NODE_OTHER (12) → `12/12 = 1.0`

**Why normalize?** The GNN also has a learnable `nn.Embedding` table for node types — but `type_id` is still included as a direct feature because embedding lookups only help if the GNN layers see the raw category signal too. Many GNN papers do both.

---

## 3. Feature [1]: visibility — Access Control Ordinal

```python
visibility = float(VISIBILITY_MAP.get(
    str(getattr(obj, "visibility", "public")), 0.0
))
```

- `getattr(obj, "visibility", "public")` — safe attribute access with default
- `str(...)` — converts any enum to string
- `.get(..., 0.0)` — unknown visibility → public (safe default)

**The BUG-3 story (v6):**
Original code used `{public:0, internal:1, private:2}`. A CONTRACT node representing a library had `visibility="private"` → feature value `2.0`. All other features are in [0,1]. This 2× outlier meant private functions had enormous dot products in early GNN layers, biasing attention. Fixed in v6 by remapping to `{public:0.0, internal:0.5, private:1.0}`.

---

## 4. Feature [2]: uses_block_globals — Timestamp/TOD Signal

```python
def _compute_uses_block_globals(func: Any) -> float:
    _BLOCK_GLOBALS = {"timestamp", "number", "difficulty", "basefee", "prevrandao"}
    for node in (getattr(func, "nodes", None) or []):
        for op in (getattr(node, "irs", None) or []):
            for rv in (getattr(op, "read", None) or []):
                if type(rv).__name__ == "SolidityVariableComposed":
                    name = getattr(rv, "name", "") or ""
                    part = name.split(".")[-1].lower()  # "block.timestamp" → "timestamp"
                    if part in _BLOCK_GLOBALS:
                        return 1.0
    return 0.0
```

**Why is this feature needed?**

`block.timestamp` is a Solidity global variable — it's NOT a state variable. Slither doesn't add it to `func.state_variables_read`. This means `block.timestamp` access creates **no READS edge** in the graph. Without this feature, Timestamp vulnerability contracts are **completely invisible** to the GNN.

This feature was added in v4, replacing the `pure` boolean (which was almost always 0 and provided zero signal).

**Implementation details:**
- `SolidityVariableComposed` is Slither's type for global variables like `block.timestamp`
- We use `type(rv).__name__ == "SolidityVariableComposed"` (string comparison) instead of `isinstance` because Slither's module structure can vary across versions — this is more robust
- `name.split(".")[-1]` handles both `"block.timestamp"` and `"block.number"` correctly

---

## 5. Feature [5]: complexity — Log-Normalized CFG Block Count

```python
try:
    _raw = float(len(obj.nodes)) if obj.nodes else 0.0
    complexity = min(math.log1p(_raw) / math.log1p(100), 1.0)
except Exception:
    complexity = 0.0
```

`obj.nodes` is the list of Slither CFG nodes for a function. The count of CFG nodes is a rough measure of function complexity.

**BUG-2 story (fixed in v5):**
Original code used `len(obj.nodes)` directly. A complex function with 150 CFG blocks would get `complexity=150`, while `payable` is `0` or `1`. The 150× imbalance overwhelmed all other features in dot products. After log-norm: `log1p(150)/log1p(100) = 1.0` (clamped).

---

## 6. Feature [6]: loc — Log-Normalized Lines of Code

```python
loc_raw = 0.0
sm = getattr(obj, "source_mapping", None)
if sm is not None and getattr(sm, "lines", None):
    loc_raw = float(len(sm.lines))
loc = min(math.log1p(loc_raw) / math.log1p(1000), 1.0)
```

`source_mapping.lines` is a list of line numbers covered by this AST node's source code.

**BUG-1 story (fixed in v5):**
The same bug as complexity — raw line count up to 2538 for CONTRACT nodes, while all other features in [0,1].

**Different denominator for loc vs complexity:**
- `complexity` clamped at `log1p(100)` — 100 CFG blocks is a "complex function"
- `loc` clamped at `log1p(1000)` — 1000 lines is "very large code"
These are domain knowledge choices, not magic numbers.

---

## 7. Feature [7]: return_ignored — The Most Complex Feature

This is the most sophisticated feature, with a two-pass algorithm:

```python
def _compute_return_ignored(func: Any) -> float:
    # Returns: 0.0 (captured), 1.0 (discarded), -1.0 (IR unavailable)
```

**Three possible values (sentinel pattern):**
- `0.0` — safe: all external call return values are read afterward
- `1.0` — vulnerable: at least one call's return value is discarded
- `-1.0` — unknown: Slither IR is unavailable (treated as "not confirmed safe")

**Why `-1.0` instead of assuming `0.0`?**
This is the **closed-world assumption guard**: if you don't know → don't assume safe. `-1.0` propagates uncertainty into the model rather than hiding it.

**The IMP-D1 fix (sequential scan instead of global set):**

**Old approach (buggy):**
```python
all_read_names = {v.name for node in func.nodes for op in node.irs for v in op.read}
for call_op in external_calls:
    if call_op.lvalue.name in all_read_names:
        return 0.0  # "captured"
```

**Problem:** If `tmp0` (the lvalue of a call) happens to share its name with a variable read **anywhere** in the function — even before the call — the code incorrectly concludes the return was captured. This was a **false negative**: real UnusedReturn cases were missed.

**New approach (IMP-D1 — sequential scan):**
```python
all_ops_ordered = [(node, op) for node in nodes for op in node.irs]

for call_idx, (_, op) in enumerate(all_ops_ordered):
    if not isinstance(op, (LowLevelCall, HighLevelCall, Send)):
        continue
    lval_name = op.lvalue.name
    # Only check reads AFTER this call in CFG order
    used_after = any(
        getattr(rv, "name", None) == lval_name
        for _, later_op in all_ops_ordered[call_idx + 1:]
        for rv in (getattr(later_op, "read", None) or [])
    )
    if not used_after:
        return 1.0  # return was discarded
```

This correctly handles the temporal ordering: a read must happen **after** the call to count as "captured."

**BUG-9 fix (added Send to the check):**
Slither classifies `addr.send(amount)` as a `Send` IR operation, NOT a `LowLevelCall`. Original code only checked `LowLevelCall` and `HighLevelCall`, missing all `.send()` return value discards. This was a major miss for the MishandledException class.

> 🎯 **INTERVIEW FOCUS:** "How would you detect if a call's return value is used?" — Build a sequential ordered list of all IR operations, find the call, then scan forward for any read of its lvalue name.

---

## 8. Feature [8]: call_target_typed — Raw Address vs Interface

```python
def _compute_call_target_typed(func: Any) -> float:
    # Returns: 1.0 (typed), 0.0 (raw address), -1.0 (unknown)
```

**What it measures:** Does this function call external contracts through a typed interface (`IToken(addr).transfer(...)`) or through a raw address (`address(addr).call(...)`)? Raw address calls are more dangerous — they bypass type checking and can call any function on any contract.

**Algorithm:**
1. Check `func.low_level_calls` — any low-level call → immediately `0.0` (raw)
2. Check `func.high_level_calls` — if any receiver has `ElementaryType("address")` type → `0.0`
3. If all calls are typed → `1.0`
4. Fallback: scan source code for `address(...).call` pattern

**The sentinel `-1.0` again:** If type resolution fails AND source is unavailable, return `-1.0` not `1.0`. Don't assume safe.

---

## 9. Feature [9]: has_loop

```python
def _compute_has_loop(func: Any) -> float:
    try:
        from slither.core.cfg.node import NodeType
        loop_types = {NodeType.IFLOOP, NodeType.STARTLOOP, NodeType.ENDLOOP}
        for node in (getattr(func, "nodes", None) or []):
            if getattr(node, "type", None) in loop_types:
                return 1.0
    except Exception:
        pass
    # Fallback: Slither convenience attribute
    if getattr(func, "is_loop_present", None) is True:
        return 1.0
    return 0.0
```

Three Slither node types signal a loop:
- `IFLOOP` — the loop condition check (e.g., `while (i < arr.length)`)
- `STARTLOOP` — loop entry point
- `ENDLOOP` — loop exit

Checking any one of them is sufficient. The fallback `is_loop_present` handles older Slither versions.

---

## 10. Feature [10]: external_call_count — Log-Normalized Count

```python
def _compute_external_call_count(func: Any) -> float:
    n  = len(list(func.high_level_calls or []))
    n += len(list(func.low_level_calls  or []))
    # Count Transfer/Send IR ops
    for node in (func.nodes or []):
        for op in (node.irs or []):
            if isinstance(op, (Transfer, Send)):
                n += 1
    return min(math.log1p(n) / math.log1p(20), 1.0)
```

**Why count Transfer/Send separately?**
Slither classifies `payable(addr).transfer(amount)` as a `Transfer` IR operation, not `high_level_calls`. The old code only counted `high_level_calls` and `low_level_calls`, giving `ext_calls=0.0` for the DoS pattern `distribute() { for each user: user.transfer(share); }`. This was **completely missing the DoS signal** for ETH-transfer loops.

**Log normalization:**
`log1p(n) / log1p(20)` means 1 call → 0.23, 5 calls → 0.60, 20+ calls → 1.0.

---

## 11. The CFG Node Feature Builder: `_build_cfg_node_features()`

CFG nodes (statements inside functions) have a simpler feature vector. Most features are inherited from their parent function via `parent_features`:

```python
def _build_cfg_node_features(slither_node, func, cfg_type, parent_features=None):
    p = parent_features or []  # parent FUNCTION's feature vector
    
    return [
        float(cfg_type) / _MAX_TYPE_ID,  # [0] own type (CFG_NODE_CALL, etc.)
        p[1] if len(p) > 1 else 0.0,    # [1] visibility — inherited
        0.0,                              # [2] uses_block_globals — not per-statement
        p[3] if len(p) > 3 else 0.0,    # [3] view — inherited
        p[4] if len(p) > 4 else 0.0,    # [4] payable — inherited
        p[5] if len(p) > 5 else 0.0,    # [5] complexity — inherited
        loc,                              # [6] own loc (this statement's source span)
        0.0,                              # [7] return_ignored — not per-statement
        1.0,                              # [8] call_target_typed — safe default
        p[9] if len(p) > 9 else 0.0,    # [9] has_loop — inherited
        0.0,                              # [10] external_call_count — not per-statement
    ]
```

**BUG-C3 story:**
Before this fix, all CFG nodes had `0.0` for dims 1,3,4,5,9 (visibility, view, payable, complexity, has_loop). This meant 9 out of 11 dimensions were zero for every statement node. The GNN had almost no signal for statement-level nodes, severely limiting its ability to detect CEI (Check-Effects-Interactions) patterns.

**Why inherit only these dims?**
- `visibility/view/payable/complexity/has_loop` are **function-level properties** that apply to all statements within the function
- `uses_block_globals` is explicitly **not** inherited — it's a contract-level concern, not a per-statement detail
- `return_ignored` is **not** per-statement (it's a function-level check)
- `loc` is computed per-statement (each statement has its own source span)

**CRITICAL note about `has_loop` inheritance:**
The parent FUNCTION might have `has_loop=1.0` because somewhere in its body there's a loop. ALL its CFG nodes inherit this `1.0`. But that's actually correct — if the function has a loop, every statement inside it exists in a potentially loopy context. The GNN learns from CONTROL_FLOW edges which specific statements are inside vs outside the loop.

---

## 12. The Graceful Degradation Pattern

Every feature computation function follows the same pattern:

```python
def _compute_something(func):
    try:
        # Use Slither API to compute feature
        from slither.slithir.operations import LowLevelCall
        ...
        return result
    except AttributeError:
        logger.warning("unavailable for %s", func.canonical_name)
        return -1.0  # or 0.0 depending on the feature's sentinel semantics
    except Exception:
        return 0.0
```

**Why catch exceptions here instead of letting them propagate?**
These functions run inside a worker process processing ~68,000 contracts. If one contract has a weird Slither AST layout and raises an unhandled exception, we don't want to crash the entire batch run. Log the warning, use a safe default, and move on.

**However:** For `RuntimeError` (Slither not installed), the code re-raises immediately:
```python
try:
    graph = extract_contract_graph(path, config)
except RuntimeError:
    raise  # Slither not installed — FATAL, don't continue
except GraphExtractionError as exc:
    return None  # expected extraction failure — skip
```

This is a **severity-based exception handling** pattern: recoverable failures are caught and logged; fatal infrastructure failures are re-raised immediately.

---

## 13. Summary

Feature engineering in SENTINEL is not about applying sklearn transformers — it's about **extracting semantic signals from Slither's IR** with careful error handling and normalisation:

| Feature | Signal | Key implementation detail |
|---------|--------|--------------------------|
| type_id | Node category | Normalized by max type ID |
| visibility | Access control | Ordinal float, not one-hot |
| uses_block_globals | Timestamp/TOD | Scans IR for SolidityVariableComposed |
| view/payable | Function modifiers | Direct attribute access |
| complexity | CFG complexity | log1p normalized |
| loc | Code length | log1p normalized, all node types |
| return_ignored | UnusedReturn/MishandledException | Sequential scan (IMP-D1) |
| call_target_typed | CallToUnknown | Type resolution + source fallback |
| has_loop | DoS/IntegerUO | Three loop node types |
| external_call_count | DoS/Reentrancy | Includes Transfer/Send IR ops |

---

## Interview Questions

1. **"How did you handle missing data in feature engineering?"**
   → Sentinel values (-1.0) for features where unavailability has semantic meaning (return_ignored, call_target_typed). Zero/safe defaults for features where unavailability means "not applicable."

2. **"Why use log normalization instead of min-max scaling for count features?"**
   → Count data is right-skewed (a few functions have 100+ CFG blocks, most have <20). Log normalization handles this naturally while preserving ordering. Min-max would compress most values near 0 and one outlier would dominate.

3. **"Walk me through how you'd detect the reentrancy pattern from the feature vector."**
   → Look for: `external_call_count > 0` (there are external calls), plus through CONTROL_FLOW edges: a CFG_NODE_CALL followed by CFG_NODE_WRITE in execution order. The `return_ignored` feature also captures the related MishandledException pattern.

---

**Next:** `04_cfg_extraction_and_graph_building.md` — how the CFG graph is built, ICFG edges, and DEF_USE data-flow edges.
