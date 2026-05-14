Good call — `preprocess.py` is the online mirror of `ast_extractor.py`, both wrapping `graph_extractor.py`. Including it in Group 2 gives us the complete extraction pipeline (offline batch + online inference + shared core). Let me read the full source files now.
Good — I have all three files. Now let me search for additional cross-file patterns that matter for this group:
Excellent data. Now I have the full picture for Group 2. Let me compile the adversarial audit.
---

# 🔴 GROUP 2 — ADVERSARIAL AUDIT: Graph Extraction Engine

**Files:** `graph_extractor.py` (890 lines), `ast_extractor.py` (438 lines), `preprocess.py` (533 lines)  
**Audit lens:** Hostile. Every line is a potential silent data corruptor. Every `except` block is a lie.

---

## FINDING 2.1 — `_add_node` Type-ID Roundtrip Bug: `actual_type_id` Always 0 for Function Subtypes

**Severity: CRITICAL**

In `graph_extractor.py` line 744:
```python
actual_type_id = int(x_list[-1][0])
```

After `_build_node_features()` returns, this reads back `x_list[-1][0]` to get the "actual" type_id (in case `_build_node_features` overrode it for CONSTRUCTOR/FALLBACK/RECEIVE). But `_build_node_features` normalises: `float(type_id) / 12.0` (line 563). So `x_list[-1][0]` is a normalised float, not the raw integer.

For CONSTRUCTOR (type_id=6): `x_list[-1][0] = 6.0 / 12.0 = 0.5`. Then `int(0.5) = 0`.  
For FALLBACK (type_id=4): `x_list[-1][0] = 4.0 / 12.0 = 0.333...`. Then `int(0.333) = 0`.  
For RECEIVE (type_id=5): `x_list[-1][0] = 5.0 / 12.0 = 0.416...`. Then `int(0.416) = 0`.  
For FUNCTION (type_id=1): `x_list[-1][0] = 1.0 / 12.0 = 0.083...`. Then `int(0.083) = 0`.

**Every function subtype gets `actual_type_id = 0`, which maps to `"STATE_VAR"` in `_type_name_map`.** The `node_metadata` type field is wrong for ALL function nodes — they're labelled "STATE_VAR" instead of "CONSTRUCTOR", "FALLBACK", "RECEIVE", or "FUNCTION".

**Impact:** The `node_metadata` type field is used by `test_cfg_embedding_separation.py` to locate function nodes:
```python
if type_id == NODE_TYPES["FUNCTION"]:  # line 80
```
Wait — actually that test reads `graph.x[i, 0]` directly (the normalised float) and compares against `NODE_TYPES["FUNCTION"]` which is an integer. So line 94:
```python
if round(graph.x[i, 0].item() * 12) == NODE_TYPES["FUNCTION"]
```
This is correct (round-trips through normalisation). But the `node_metadata` type field is still wrong, which means any tooling that relies on `node_metadata["type"]` to identify CONSTRUCTOR vs FUNCTION vs STATE_VAR nodes would be completely broken.

**The hostile read:** If a debugging or monitoring tool uses `node_metadata` to display which nodes are constructors vs state variables, it would show "STATE_VAR" for every function node. During the v5.0 behavioral test failure (0% specificity), such a tool would have given completely misleading diagnostic information.

**Recommendation:** Compute `actual_type_id` BEFORE normalisation, or reverse-normalise properly:
```python
actual_type_id = int(round(x_list[-1][0] * 12))  # reverse normalisation
```
Or better: store the raw type_id separately before calling `_build_node_features`.

---

## FINDING 2.2 — CFG Failure Count Not Propagated to Caller

**Severity: HIGH**

`_cfg_failure_count` (lines 764-810) tracks how many functions had CFG extraction failures. If the rate exceeds 5%, it logs an error. But this count is **never returned or attached to the graph object**. The caller (both `ast_extractor.py` and `preprocess.py`) receives a graph with no indication that some functions had their CONTAINS/CONTROL_FLOW edges omitted.

**The hostile read:** A contract where 60% of functions fail CFG extraction (due to Slither version mismatch) produces a graph that looks valid — it has nodes, it has edges — but is missing most of its CFG structure. The GNN processes this graph in Phase 2/3 with almost no CONTROL_FLOW edges, producing near-zero directional signal. The model makes a prediction on an impoverished graph, and the prediction is wrong. **No error is raised, no warning reaches the API caller.**

The v5.1 proposal mentions adding a `cfg_failure_rate` field to the graph for the trainer to use as a quality signal. But the current code doesn't propagate it at all.

**Recommendation:** Attach `_cfg_failure_count` and `_func_total` to the returned Data object:
```python
graph.cfg_failure_count = _cfg_failure_count
graph.cfg_func_total = _func_total
graph.cfg_failure_rate = _cfg_failure_count / max(_func_total, 1)
```
Then in `preprocess.py`, check the rate and log a warning to the API response. In the dataset, filter out graphs with failure_rate > 0.3 (or whatever threshold the v5.1 plan specifies).

---

## FINDING 2.3 — Broad `except Exception` Swallows in Feature Computation

**Severity: HIGH (data integrity)**

Five feature-computation functions use broad `except Exception` or `except` blocks that silently return default values:

| Function | Line | Fallback | What it hides |
|---|---|---|---|
| `_compute_return_ignored` | 202 | Returns `-1.0` (sentinel) | Any AttributeError during IR traversal |
| `_compute_call_target_typed` | 239 | Falls through to source scan | Any exception during type resolution |
| `_compute_in_unchecked` | 266 | Falls through to regex | AttributeError on NodeType |
| `_compute_has_loop` | 289 | Returns `0.0` | Any exception during node traversal |
| `_compute_external_call_count` | 311 | Returns `0.0` | Any exception during call counting |
| `_build_control_flow_edges` (caller) | 788 | `_cfg_failure_count += 1`, continues | ANY exception during CFG construction |

**The hostile read:** Imagine a Slither API change where `func.slithir_operations` raises a `TypeError` instead of returning a list. `_compute_return_ignored` catches it, logs a warning, returns `-1.0`. Every function in every contract gets `return_ignored = -1.0`. The model sees 100% sentinel values for this feature — but training proceeds "normally." The feature is dead weight, but no error surfaces.

Similarly, if `_compute_external_call_count` fails for every function (because `high_level_calls` changed API), every function gets `external_call_count = 0.0`. The model loses a key vulnerability signal (how many external calls a function makes) with **zero indication** that anything is wrong.

**Recommendation:** Count sentinel/default returns across the full dataset. In the trainer or dataset loader, assert that the sentinel rate for each feature is below a threshold (e.g., < 20%). If `_compute_return_ignored` returns `-1.0` for > 20% of functions, raise an error during data loading rather than training on garbage.

---

## FINDING 2.4 — `node_metadata`: Dead Weight in Production

**Severity: MEDIUM (performance / maintenance)**

`node_metadata` is attached to every returned `Data` object (line 884) and is always populated. Cross-file search shows **zero runtime consumers** — no model code, no inference code, no script reads it. Only tests read it.

For a 68K-contract dataset, each `node_metadata` entry is a dict with 3 keys (name, type, source_lines), and there are N entries per graph. This is:
- Unnecessary memory during training (Data objects are loaded, `node_metadata` is carried through the DataLoader but never accessed)
- Unnecessary disk space in .pt files
- Maintenance surface: any schema change to graph nodes requires updating node_metadata too

**The hostile read:** The `source_lines` field stores a list of Python ints (line numbers) for every node. For a contract with 200 nodes and average 5 lines per node, that's ~1000 integers stored per graph as a Python list inside a PyG Data object. During DataLoader batching, PyG doesn't know how to collate arbitrary Python lists — it either skips them or creates nested lists. This could cause subtle batching errors or excessive memory use.

**Recommendation:** Make `node_metadata` opt-in via `GraphExtractionConfig`:
```python
include_node_metadata: bool = False  # Only True for tests/debugging
```
In production (preprocess.py, ast_extractor.py), set it to False. Only set True in test configs.

---

## FINDING 2.5 — Duplicate `canonical_name` Causes Silent Function Drop

**Severity: HIGH**

In `_add_node` (line 734-752), duplicate detection uses `canonical_name` as the key:
```python
key: str = getattr(obj, "canonical_name", None) or obj.name
if key in node_map:
    return None
```

If a contract has two functions with the same `canonical_name` (which happens with overloaded Solidity functions like `transfer(address)` and `transfer(address,uint256)`), the second one is **silently dropped**. Its nodes are never added to the graph.

But then at line 768-774:
```python
for func in contract.functions:
    fn_idx = _add_node(func, NODE_TYPES["FUNCTION"])
    if fn_idx is None:
        # Duplicate function name — still need to find its index for CFG edges
        fn_key = getattr(func, "canonical_name", None) or func.name
        fn_idx = node_map.get(fn_key)
        if fn_idx is None:
            continue
```

When `_add_node` returns None for a duplicate, the code tries to find the first function's index and **attaches the second function's CFG children to the first function's node**. This means:
- Function A: `transfer(address)` — simple, no external calls
- Function B: `transfer(address,uint256)` — complex, reentrancy-vulnerable

Function B's CFG nodes (including CFG_NODE_CALL and CFG_NODE_WRITE) get attached to Function A's graph node via CONTAINS edges. The GNN sees Function A as having external calls and state writes, even though Function A is safe. **This creates false positives.**

Conversely, Function B's declaration-level edges (CALLS, READS, WRITES) are correctly built (line 847-864 iterates all `contract.functions`), but its CFG subgraph is orphaned from its declaration node.

**The hostile read:** The v5.0 behavioral test showed 0% specificity (every clean contract flagged as vulnerable). Overloaded functions merging their CFG subgraphs into the wrong parent node would contribute to this — safe functions appear to have dangerous CFG structures.

**Recommendation:** Handle overloaded functions properly. Slither's `canonical_name` for overloaded functions includes the parameter signature (e.g., `Contract.transfer(address)` vs `Contract.transfer(address,uint256)`). If they're truly colliding, disambiguate by appending the parameter count:
```python
key = getattr(obj, "canonical_name", None) or obj.name
if key in node_map:
    # Disambiguate overloaded functions
    key = f"{key}__{len(getattr(obj, 'parameters', []))}"
```

---

## FINDING 2.6 — `_compute_call_target_typed` Regex Fallback is Fragile

**Severity: MEDIUM**

Lines 252-254:
```python
raw_addr_pattern = re.compile(r"address\s*\(\s*(?!this\b)[^)]+\)\s*\.call")
if raw_addr_pattern.search(sm.content):
    return 0.0
```

This regex tries to detect `address(...).call(...)` patterns in the source. Several issues:

1. **`sm.content`** — The docstring says "source_mapping unavailable" returns -1.0. But when `sm.content` IS available, it contains the **function body only**, not the full contract. If the function is a one-liner like `addr.call{value: v}("")`, the regex matches `addr.call` but NOT `address(...)` — it looks for `address\s*\(` first. **Low-level calls via stored address variables (`addr.call()`) are missed.**

2. **Negative lookahead `(?!this\b)`** is meant to exclude `address(this).call()` (self-calls). But `address(this).transfer()` would also be excluded by the regex — and transfers are NOT calls. The regex is over-broad in its exclusion.

3. **`\.call`** matches `.call{value:...}` (Solidity 0.8+ syntax) but does NOT match `.delegatecall()` or `.staticcall()`, which are equally dangerous low-level patterns.

**The hostile read:** A contract using `addr.delegatecall(data)` (a classic proxy pattern vulnerability) would have `call_target_typed = 1.0` (the regex doesn't match, so it falls through to the "all calls typed" return). The model is told this function has no raw-address calls, when in fact it has one of the most dangerous call types.

**Recommendation:** Expand the regex to cover `.delegatecall`, `.staticcall`, and `assembly { call(...) }` patterns. Or better: rely entirely on Slither IR (which correctly identifies all call types) and remove the regex fallback, making the -1.0 sentinel the only fallback.

---

## FINDING 2.7 — `_build_cfg_node_features` Sets `call_target_typed = 1.0` for ALL CFG Nodes

**Severity: MEDIUM**

Line 409:
```python
1.0,              # [8]  call_target_typed — default safe (not applicable)
```

Every CFG_NODE gets `call_target_typed = 1.0`, including CFG_NODE_CALL nodes. But a CFG_NODE_CALL by definition contains an external call. If that call goes to a raw address, the parent function has `call_target_typed = 0.0` but the CFG_NODE_CALL child has `call_target_typed = 1.0`. This is semantically contradictory.

**The hostile read:** The GNN receives: parent FUNCTION node says "raw address call" (feature[8]=0.0), but the child CFG_NODE_CALL says "typed call" (feature[8]=1.0). During Phase 3 reverse-CONTAINS aggregation, the CFG node's "typed" signal partially cancels the function node's "raw address" signal. The model learns a weaker representation of the raw-address danger.

**Recommendation:** For CFG_NODE_CALL nodes, inherit `call_target_typed` from the parent function, or compute it from the specific IR operation in the node. At minimum, set it to `0.0` (not 1.0) for CFG_NODE_CALL — a call node with a "safe typed" default is semantically wrong.

---

## FINDING 2.8 — `visibility` Feature Computed for Non-Declaration Nodes

**Severity: LOW**

Line 522-524:
```python
visibility = float(VISIBILITY_MAP.get(
    str(getattr(obj, "visibility", "public")), 0
))
```

This computes `visibility` for every node, including STATE_VAR, CONTRACT, EVENT, and MODIFIER nodes. But:
- CONTRACT nodes don't have visibility → defaults to 0.0 (public)
- EVENT nodes don't have visibility → defaults to 0.0 (public)
- STATE_VAR nodes DO have visibility — this is correct and useful

However, for CFG nodes, `_build_cfg_node_features` hardcodes `visibility = 0.0` (line 402), which is correct (not applicable). But for declaration nodes like CONTRACT and EVENT, the feature vector contains a misleading `visibility = 0.0` that implies "public access" when it's actually "not applicable."

**The hostile read:** The GNN sees CONTRACT nodes with visibility=0 (same as public functions). During Phase 1 structural aggregation, the GNN might learn a spurious association between "public" and contract-level properties.

**Recommendation:** Low priority, but for correctness: set visibility to a "not applicable" value (e.g., -1.0 sentinel) for CONTRACT, EVENT, and MODIFIER nodes. Or at minimum, document in the schema that visibility=0 for these node types means "not applicable," not "public."

---

## FINDING 2.9 — `complexity` Feature Not Normalised

**Severity: MEDIUM**

Line 544:
```python
complexity = float(len(obj.nodes)) if obj.nodes else 0.0
```

This is the raw CFG node count. For a simple function, this might be 3-5. For a complex function, it could be 50-100. The feature is NOT normalised to [0,1] like `external_call_count` is (which uses `log1p(n)/log1p(20)`).

Meanwhile, `loc` (line 528) is also un-normalised:
```python
loc = float(len(sm.lines))
```

These two features can have values up to 100+, while most other features are in [0, 1] or [-1, 1]. This creates a feature scale imbalance that the GNN must compensate for through its learnable weights.

**The hostile read:** During training, the GNN's attention mechanism computes dot products between node features and learnable query/key vectors. Features with large ranges (complexity, loc) dominate the dot product, drowning out features with small ranges (pure, view, payable). The model may take many epochs to learn to down-weight these features, contributing to slow convergence or the gradient collapse observed in v5.0.

**Recommendation:** Normalise `complexity` and `loc` to [0,1] ranges. For complexity, use the same `log1p(n)/log1p(50)` formula (50 CFG nodes is very complex). For loc, use `log1p(n)/log1p(100)` (100 lines is a large function). This requires a schema version bump and full retrain.

---

## FINDING 2.10 — `ast_extractor.py` Import Side Effect: `sys.path.insert`

**Severity: MEDIUM**

Line 71:
```python
sys.path.insert(0, str(Path(__file__).parent.parent))
```

This mutates the global `sys.path` at import time. It's necessary for the script to work when run directly (`python ml/src/data_extraction/ast_extractor.py`), but if this module is ever imported by another module (e.g., in tests), it pollutes `sys.path` for the entire process.

More critically, line 79:
```python
from src.preprocessing.graph_schema import NODE_TYPES, VISIBILITY_MAP, EDGE_TYPES  # re-exported for any external callers
```

This imports `NODE_TYPES`, `VISIBILITY_MAP`, and `EDGE_TYPES` from `src.preprocessing.graph_schema` — but these are **never used** in the body of `ast_extractor.py`. The comment says "re-exported for any external callers" but the `__init__.py` files are empty, so there's no actual re-export mechanism. Any external caller would import from `graph_schema` directly, not through `ast_extractor`.

**The hostile read:** These unused imports create a false impression that `ast_extractor.py` directly uses these constants, making it harder to track the dependency graph. They also mean `ast_extractor.py` will crash at import time if `graph_schema.py` fails to import (e.g., due to the Slither version check) — even though `ast_extractor.py` doesn't actually need those constants (it calls `extract_contract_graph()` which handles everything internally).

**Recommendation:** Remove the unused imports on line 79. The `sys.path.insert` is unavoidable for CLI usage but should be guarded:
```python
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))
```

---

## FINDING 2.11 — `preprocess.py` Writes Temp File with Predictable Name

**Severity: LOW (security)**

Lines 271-278:
```python
tmp = tempfile.NamedTemporaryFile(
    suffix=".sol",
    prefix=_SENTINEL_TMP_PREFIX,
    mode="w",
    encoding="utf-8",
    delete=False,
)
```

The temp file is created with the fixed prefix `sentinel_prep_` and a random suffix (Python's `NamedTemporaryFile` adds random chars). The file contains the user's Solidity source code. It's written to the system temp directory.

**The hostile read:** On a multi-tenant system, another user could scan `/tmp/sentinel_prep_*.sol` and read submitted contract source code. For a vulnerability detection API, the submitted contracts may contain proprietary or sensitive code. The temp file is created with default permissions (typically 644 on Linux — world-readable).

**Recommendation:** Set `umask(0o600)` before creating the temp file, or use `os.open` with mode `0o600` to create the file with restricted permissions. This is a defence-in-depth measure.

---

## FINDING 2.12 — `_select_contract` Falls Back to Interface (Ghost Graph Risk)

**Severity: MEDIUM**

Lines 618-629:
```python
non_iface = [c for c in candidates if not c.is_interface]
if non_iface:
    return max(non_iface, key=lambda c: len(c.functions))

# All candidates are interfaces — fall back to the first one
logger.warning(...)
return candidates[0]
```

When ALL non-dependency contracts in a file are interfaces, the function falls back to returning the first interface. The warning is logged but extraction proceeds. An interface contract has:
- No function bodies → no CFG nodes, no CONTAINS/CONTROL_FLOW edges
- Empty function nodes → all features default to 0.0

The resulting graph is a "ghost graph" — a handful of declaration nodes with no edges. The v5.0 documentation specifically identifies ghost graphs as a cause of the 15% detection rate (the model predicts "safe" for everything because there's no vulnerability signal in an empty graph).

**The hostile read:** The fallback exists so extraction "still proceeds rather than silently failing." But proceeding with a ghost graph is worse than failing — it produces a valid-looking .pt file that gets included in the training set, diluting the signal with noise. For inference, it produces a prediction on an empty graph, which is meaningless.

**Recommendation:** Raise `EmptyGraphError` instead of falling back to an interface. The error message can explain that the file contains only interfaces. The caller can decide whether to return HTTP 400 (inference) or skip (batch). Ghost graphs should never enter the training set.

---

## FINDING 2.13 — CFG Node Ordering: `source_mapping.lines[0]` Can Be None

**Severity: MEDIUM**

Line 459:
```python
n.source_mapping.lines[0]
if n.source_mapping and n.source_mapping.lines else 0,
```

The sort key accesses `n.source_mapping.lines[0]` assuming it's an integer. But Slither's `source_mapping.lines` can be `None` or an empty list even when `source_mapping` is not None. The code handles this with the `else 0` clause. However, there's a more subtle issue: synthetic CFG nodes (ENTRY_POINT, EXPRESSION) can have the same source line as real nodes. Sorting by `(source_line, node_id)` puts them in a deterministic order, but the node_id is Slither's internal ID which can change across Slither versions.

**The hostile read:** If the same contract is extracted with two different Slither versions, and synthetic nodes get different node_ids, the CFG node order changes. This changes the graph structure (node indices shift), producing a different graph from the same source. The GNN would receive a structurally different input for the same contract across re-extraction runs, making results non-reproducible.

**Recommendation:** Add a secondary sort key that's more stable than `node_id`, such as the IR operation type or the string representation. Document that graph determinism is guaranteed only within the same Slither version.

---

## FINDING 2.14 — `preprocess.py` `process()` Doesn't Include `FEATURE_SCHEMA_VERSION` in Cache Key

**Severity: HIGH (repeated from Group 1, now with extraction context)**

As identified in Finding 1.7, `process()` (line 191) uses only `get_contract_hash(sol_path)` as the cache key, while `process_source()` (line 253) uses `f"{content_hash}_{FEATURE_SCHEMA_VERSION}"`. 

The extraction context makes this worse: `process()` calls `_extract_graph()` which calls `extract_contract_graph()`. If the graph extraction logic changes (e.g., the Phase 0 interface-selection fix), the cache key doesn't change, and stale pre-fix graphs are returned.

**Recommendation:** (Same as Finding 1.7) Include `FEATURE_SCHEMA_VERSION` in the `process()` cache key.

---

## FINDING 2.15 — `ast_extractor.py` Hardcodes `label=0` for All Batch Extraction

**Severity: MEDIUM**

Line 311:
```python
worker = partial(
    self.contract_to_pyg,
    solc_binary=solc_bin,
    solc_version=version,
    label=0,  # ← hardcoded
)
```

The batch pipeline sets `label=0` (safe) for every contract. The actual label comes from the dataset's parquet file, but it's never passed through the extraction pipeline. The label is later overwritten when the dataset loader builds the training set from the multilabel index.

However, the `data.y` tensor (line 235) is set to `torch.tensor([label], dtype=torch.long)` — a single integer. The multilabel dataset expects a multi-hot vector, not a single integer. So this `y` value is always overwritten by the dataset loader. But if someone ever loads a .pt file directly and uses `data.y`, they'd get `0` regardless of the contract's actual vulnerability.

**The hostile read:** If the dataset loader has a bug that fails to overwrite `data.y`, every training sample would have label=0, and the model would learn to predict "safe" for everything. This is exactly what happened in v5.0 (15% detection rate on behavioral tests).

**Recommendation:** Either (a) pass the actual label from the parquet file, or (b) set `data.y = -1` as a sentinel meaning "label not set, must be overwritten by dataset loader," and add an assertion in the dataset loader that `y != -1`.

---

## Summary Table

| # | Finding | Severity | Category |
|---|---------|----------|----------|
| 2.1 | `_add_node` type-id roundtrip: metadata always "STATE_VAR" | CRITICAL | Data correctness |
| 2.2 | CFG failure count not propagated to caller | HIGH | Observability |
| 2.3 | Broad `except Exception` swallows in feature computation | HIGH | Data integrity |
| 2.4 | `node_metadata` dead weight in production | MEDIUM | Performance |
| 2.5 | Duplicate `canonical_name` merges CFG into wrong parent | HIGH | Graph correctness |
| 2.6 | `call_target_typed` regex misses delegatecall/staticcall | MEDIUM | Feature quality |
| 2.7 | CFG_NODE_CALL gets `call_target_typed=1.0` (contradicts parent) | MEDIUM | Feature semantics |
| 2.8 | `visibility` computed for non-declaration nodes | LOW | Feature semantics |
| 2.9 | `complexity` and `loc` not normalised (scale imbalance) | MEDIUM | Feature engineering |
| 2.10 | `ast_extractor.py` unused imports + sys.path pollution | MEDIUM | Code quality |
| 2.11 | Temp file world-readable on multi-tenant systems | LOW | Security |
| 2.12 | Interface fallback creates ghost graphs | MEDIUM | Data quality |
| 2.13 | CFG node ordering non-deterministic across Slither versions | MEDIUM | Reproducibility |
| 2.14 | `process()` missing FEATURE_SCHEMA_VERSION in cache key | HIGH | Cache correctness |
| 2.15 | `ast_extractor.py` hardcodes label=0 | MEDIUM | Data correctness |

**1 CRITICAL, 4 HIGH, 7 MEDIUM, 3 LOW**

The **CRITICAL** finding (2.1 — type-id roundtrip bug) means `node_metadata` type labels are wrong for every function node. The **HIGH** findings (2.2, 2.3, 2.5, 2.14) should be addressed before v5.1 training begins — they affect data integrity, graph structure correctness, and cache correctness.

---

Ready for **Group 3: GNN Architecture & Signal Propagation** (`gnn_encoder.py`, `sentinel_model.py`) whenever you want to continue.