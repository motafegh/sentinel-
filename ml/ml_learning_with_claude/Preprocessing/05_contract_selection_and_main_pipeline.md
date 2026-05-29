# Preprocessing — Chunk 5: Contract Selection Heuristics & The Main Pipeline

> **Files:** `ml/src/preprocessing/graph_extractor.py` (lines 879–1329), `ml/src/preprocessing/__init__.py`
> **What you'll learn:** The `extract_contract_graph()` main function, the exception hierarchy, the `@dataclass` config pattern, contract selection heuristics, and how the full pipeline assembles a PyG graph.
> **Time:** ~25 minutes
> **Interview relevance:** ML (pipeline design), MLOps (error handling, typed exceptions), Blockchain (multi-contract files)

---

## 1. The Exception Hierarchy

```python
class GraphExtractionError(Exception):
    """Base for all extraction failures."""

class SolcCompilationError(GraphExtractionError):
    """Solidity syntax error / wrong pragma version.
    → User error → HTTP 400 (inference) / log+skip (batch)"""

class SlitherParseError(GraphExtractionError):
    """Slither failed internally after compilation.
    → Infrastructure error → HTTP 500 (inference) / log+skip (batch)"""

class EmptyGraphError(GraphExtractionError):
    """Contract produced zero analyzable AST nodes.
    → User error (only imports) → HTTP 400 / log+skip (batch)"""
```

**Why a typed exception hierarchy?**

The caller (`preprocess.py` for inference, `ast_extractor.py` for batch) needs to respond **differently** to different failures:

| Exception | HTTP response | Batch action |
|-----------|--------------|-------------|
| `SolcCompilationError` | 400 Bad Request ("fix your Solidity") | log + skip |
| `SlitherParseError` | 500 Server Error ("our infra failed") | log + skip |
| `EmptyGraphError` | 400 Bad Request ("submit self-contained file") | log + skip |

Without typed exceptions, the caller would have to parse exception messages with string matching — fragile and error-prone.

**Pattern:**
```python
try:
    graph = extract_contract_graph(path, config)
except SolcCompilationError:
    return {"error": "compilation_failed"}, 400
except SlitherParseError:
    return {"error": "internal_error"}, 500
except EmptyGraphError:
    return {"error": "empty_contract"}, 400
```

> 🎯 **INTERVIEW FOCUS:** "How do you design exception hierarchies for production systems?" — Base exception for all failures, typed subclasses for different error categories, each with clear ownership (user vs infrastructure).

---

## 2. The `@dataclass` Configuration Pattern

```python
@dataclass
class GraphExtractionConfig:
    multi_contract_policy: str = "most_derived"
    target_contract_name: str | None = None
    include_edge_attr: bool = True
    solc_binary: str | Path | None = None
    solc_version: str | None = None
    allow_paths: str | None = None
```

**`@dataclass`** is a Python decorator that auto-generates `__init__`, `__repr__`, and `__eq__` from class field annotations. This avoids writing boilerplate `def __init__(self, field1=default1, ...): self.field1 = field1`.

**Why a config object instead of function arguments?**

If `extract_contract_graph` had all these as individual arguments:
```python
def extract_contract_graph(sol_path, multi_contract_policy="most_derived", solc_binary=None, ...):
```
Adding a new config option would require changing the function signature everywhere it's called. With a config object, you just add a field with a default:
```python
config.new_option = False  # old callers don't need to change
```

This is the **Open/Closed Principle**: open for extension (add a field), closed for modification (existing call sites don't change).

---

## 3. The `_select_contract()` Heuristic — BUG-6 Story

```python
def _select_contract(sl, config):
    candidates = [c for c in sl.contracts if not c.is_from_dependency()]
    ...
```

**The problem:** BCCC dataset contracts often look like this:

```solidity
// StandardToken.sol
library SafeMath { ... }                  // utility library
contract StandardToken { ... }            // ERC20 base (many functions)
contract ERC20Token is StandardToken {    // VULNERABLE contract (few functions)
    function transferFrom(...) { /* reentrancy bug here */ }
}
```

If you select "the contract with the most functions," you select `StandardToken` (utility library). `ERC20Token` (the vulnerable one) gets skipped. **This is BUG-6.**

**Three heuristics compared:**

| Heuristic | Logic | Accuracy on BCCC |
|-----------|-------|-----------------|
| `most_funcs` | pick contract with most functions | **52.6% — worse than random** |
| `last` | pick the last non-interface contract | **87.4%** |
| `most_derived` | pick the contract that inherits from the most in-file contracts | **~92%+** |

**Why `most_derived` works:**

In BCCC, vulnerable contracts typically:
- Inherit from library/base contracts defined earlier in the same file
- Are the "specific implementation" that adds the vulnerable overrides

```python
def _derivation_score(c):
    inherited_in_file = sum(
        1 for parent in (c.inheritance or [])
        if parent.name in candidate_names  # only count in-file parents
    )
    source_idx = non_iface.index(c)  # tiebreak: later-defined wins
    return (inherited_in_file, source_idx)

best = max(non_iface, key=_derivation_score)
```

`ERC20Token` inherits from `StandardToken` (which is in `candidate_names`). Score = (1, last_position). `StandardToken` inherits from nothing in-file. Score = (0, middle_position). `ERC20Token` wins.

**Fallback chain:**
```
most_derived with inheritance → last-defined (no inheritance found) → first interface (all interfaces)
```

> 🎯 **INTERVIEW FOCUS:** "How did you identify that your data pipeline was selecting the wrong contracts?" — Audit script (`task16_wrong_contract_selection.py`) comparing predicted vs ground-truth vulnerable contract for 100 random files.

---

## 4. The Main `extract_contract_graph()` Function — Full Flow

```python
def extract_contract_graph(sol_path: Path, config=None) -> Data:
    # 1. Setup + Slither instantiation
    # 2. Contract selection (_select_contract)
    # 3. Add declaration nodes (fixed order)
    # 4. For each function: add FUNCTION node + CFG nodes + CONTAINS/CF edges
    # 5. _add_icfg_edges (CALL_ENTRY + RETURN_TO)
    # 6. _add_def_use_edges (DEF_USE)
    # 7. Add declaration-level edges (CALLS, READS, WRITES, EMITS, INHERITS)
    # 8. Assemble PyG Data object
    # 9. Validate: feature dimension, OOR values, metadata alignment
    # 10. Return
```

Let's walk through each step:

### Step 1: Slither Instantiation with Error Translation

```python
try:
    slither_kwargs = {"solc_args": solc_args, "detectors_to_run": []}
    if config.solc_binary:
        slither_kwargs["solc"] = str(config.solc_binary)
    sl = Slither(str(sol_path), **slither_kwargs)
except Exception as exc:
    exc_lower = str(exc).lower()
    if any(kw in exc_lower for kw in ("compil", "syntax", "invalid solidity", "parsing", "solc")):
        raise SolcCompilationError(...)
    raise SlitherParseError(...)
```

**Slither raises generic `Exception`** — it doesn't have its own typed exceptions. This code translates generic errors into our typed hierarchy by checking the error message content.

`detectors_to_run=[]` disables Slither's vulnerability detection (we don't need Slither to detect vulns — the ML model does that). This makes Slither run much faster.

### Step 3: Declaration Nodes — Fixed Insertion Order

```python
_add_node(contract, NODE_TYPES["CONTRACT"])
for parent in (contract.inheritance or []):
    _add_node(parent, NODE_TYPES["CONTRACT"])   # BUG-H8 fix: add parent contracts
for var in contract.state_variables:
    _add_node(var, NODE_TYPES["STATE_VAR"])
# then functions, modifiers, events...
```

**BUG-H8 fix:** INHERITS edges connect the main contract to its parent contracts. But those parent contracts can only be endpoints in the edge graph if they have nodes. Original code didn't add parent contract nodes → INHERITS edges were silently dropped (both endpoints must be in `node_map`).

### Step 4: EMITS Edge with Fallback — BUG-H7

```python
emitted: set[str] = set()

# Primary: use Slither's events_emitted property
if hasattr(func, "events_emitted"):
    for evt in func.events_emitted:
        emitted.add(evt.canonical_name or evt.name)

# Fallback: scan IR for EventCall objects (works for Solidity <0.4.21)
if not emitted:
    from slither.slithir.operations import EventCall
    for node in (func.nodes or []):
        for ir in (node.irs or []):
            if isinstance(ir, EventCall):
                emitted.add(ir.name)
```

**BUG-H7:** Old Solidity (pre-0.4.21) emits events without the `emit` keyword: `Transfer(from, to, value)`. Slither's `events_emitted` property is unreliable for these. Scanning IR ops for `EventCall` objects works for both old and new syntax.

### Step 9: Validation

```python
# Feature dimension check
if x.shape[1] != NODE_FEATURE_DIM:
    raise SlitherParseError(f"expected {NODE_FEATURE_DIM}, got {x.shape[1]}")

# Out-of-range check (BUG-L4)
oor_mask = (x < -1.0) | (x > 1.0)
if oor_mask.any():
    logger.warning("OOR features in '%s': %d cells...", contract.name, oor_mask.sum())

# Metadata alignment check
assert len(node_metadata) == x.shape[0]
```

Three levels of validation:
1. **Hard fail**: wrong feature dimension → `SlitherParseError` (code bug, must fix)
2. **Warning**: OOR values → log and continue (might be recoverable data quirk)
3. **Assert**: metadata alignment → crash immediately (invariant violation = code bug)

The choice between hard fail vs warning is deliberate:
- Shape mismatch means a code bug that would silently corrupt training → hard fail
- OOR values might be rare edge cases that don't corrupt the whole dataset → warn and continue

---

## 5. The `_add_edge` Helper and the `node_map`

```python
node_map: dict[str, int] = {}  # canonical_name → x_list index

def _add_edge(src_key: str, dst_key: str, etype: int) -> None:
    si = node_map.get(src_key)
    di = node_map.get(dst_key)
    if si is not None and di is not None:
        edges.append([si, di])
        edge_types.append(etype)
    # Silently skip if either endpoint not in graph
```

**Why silently skip missing endpoints?**
If a function calls an external contract's function, that function won't have a node in the graph (it's from an imported dependency). Trying to add a CALLS edge to it would fail. Silent skip is correct: we only graph what we know about, and the model learns from the local contract structure.

**`canonical_name` vs `name`:**
Slither's `canonical_name` includes the contract name: `"MyToken.withdraw"`. This disambiguates functions with the same name in different contracts. `name` alone would be `"withdraw"` — could be ambiguous.

---

## 6. The Final PyG Data Assembly

```python
if edges:
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_attr  = torch.tensor(edge_types, dtype=torch.long)
else:
    edge_index = torch.zeros((2, 0), dtype=torch.long)
    edge_attr  = torch.zeros(0, dtype=torch.long)

graph = Data(x=x, edge_index=edge_index)
if config.include_edge_attr:
    graph.edge_attr = edge_attr

graph.node_metadata  = node_metadata
graph.contract_name  = contract.name
graph.num_nodes      = int(x.shape[0])
graph.num_edges      = len(edges)

return graph
```

Key details:
- `torch.tensor(edges, dtype=torch.long).t().contiguous()` — `edges` is a `[[src, dst], ...]` list (shape `[E, 2]`). `.t()` transposes to `[2, E]` (PyG convention). `.contiguous()` ensures memory layout is contiguous (required for PyG operations).
- Handles the edge case of zero edges (`edges = []`) by creating empty tensors with the correct shape `[2, 0]`
- `config.include_edge_attr=True` by default, but can be disabled for ablation experiments

---

## 7. The `__init__.py` — Package Design

```python
from .graph_schema import (
    EDGE_TYPES, FEATURE_NAMES, FEATURE_SCHEMA_VERSION,
    NODE_FEATURE_DIM, NODE_TYPES, NUM_EDGE_TYPES, VISIBILITY_MAP,
)
from .graph_extractor import (
    EmptyGraphError, GraphExtractionConfig, GraphExtractionError,
    SlitherParseError, SolcCompilationError, extract_contract_graph,
)

__all__ = [
    "EDGE_TYPES", "FEATURE_NAMES", "FEATURE_SCHEMA_VERSION", ...
    "EmptyGraphError", "GraphExtractionConfig", ...
]
```

**Why explicit re-exports?**
The `__init__.py` creates a **clean public API** for the package. Users import from `ml.src.preprocessing` (not from the submodules):

```python
# Good — stable API, submodule can be refactored
from ml.src.preprocessing import extract_contract_graph, NODE_FEATURE_DIM

# Bad — coupled to internal structure
from ml.src.preprocessing.graph_extractor import extract_contract_graph
from ml.src.preprocessing.graph_schema import NODE_FEATURE_DIM
```

The `__all__` list is the **public contract**: these are the names that `from package import *` will import, and they signal "these are the officially supported exports."

---

## 8. Why `extract_contract_graph` Never Returns None

```
"Never returns None. Always raises GraphExtractionError on failure."
```

This is an important contract. The function is designed for **two caller contexts**:

**Inference (online):**
```python
# In preprocess.py:
graph = extract_contract_graph(path, config)  # raises on failure
# No None check needed — exception is caught at HTTP layer
```

**Batch (offline):**
```python
# In ast_extractor.py:
try:
    graph = extract_contract_graph(path, config)
except GraphExtractionError:
    return None  # batch decides to skip
```

The function raises → callers decide the skip policy. This is the **separation of concerns**: the extraction function doesn't know if it's running in a batch or API context. The caller decides how to handle failures.

Contrast with the old design where both callers had return-None checks scattered throughout.

---

## 9. The SENTINEL Shape Contract

At the top of `graph_extractor.py`:
```
SHAPE CONTRACT (must match training data):
  graph.x             [N, 11]  float32  node features
  graph.edge_index    [2, E]   int64    COO edges
  graph.edge_attr     [E]      int64    edge type IDs 0-10
  graph.node_metadata list     dicts, index-aligned with x
  graph.contract_name str
  graph.num_nodes     int
  graph.num_edges     int
```

This is a **documented interface contract**: callers of `extract_contract_graph` can rely on this structure. The training dataset, the GNN encoder, and the inference pipeline all depend on this being stable.

The comment "must match training data" is a reminder: if you change the shape, you need to rebuild your training data.

---

## 10. Summary — The Complete Pipeline

```
.sol file on disk
    ↓
Slither(sol_path)          [parse with version-appropriate solc]
    ↓
_select_contract()         [most_derived heuristic]
    ↓
_add_node() × N            [build feature vectors, maintain node_map]
    ↓
_build_control_flow_edges() × functions  [CFG nodes + CONTAINS + CF edges]
    ↓
_add_icfg_edges()          [CALL_ENTRY + RETURN_TO across functions]
    ↓
_add_def_use_edges()       [DEF_USE data-flow edges]
    ↓
_add_edge() × declarations [CALLS, READS, WRITES, EMITS, INHERITS]
    ↓
torch.tensor(x_list)       [build feature matrix [N, 11]]
    ↓
validate shape, OOR, metadata alignment
    ↓
Data(x, edge_index, edge_attr, node_metadata, ...)
```

---

## Interview Questions

1. **"How do you design an error handling strategy for a production ML preprocessing pipeline?"**
   → Typed exception hierarchy: base exception for catch-all, subclasses for user errors (HTTP 400) vs infrastructure errors (HTTP 500). Functions raise; callers decide the error policy (re-raise for API, skip for batch). Never return None from a function that can also return a valid result.

2. **"A Solidity file defines 3 contracts. How do you know which one contains the vulnerability?"**
   → `most_derived` heuristic: pick the contract that inherits from the most other non-interface contracts defined in the same file. The vulnerable implementation contract typically inherits from utility libraries defined earlier in the file.

3. **"Why is `edge_index.contiguous()` called after transpose?"**
   → PyG operations (GAT, scatter) require memory-contiguous tensors for CUDA kernels. `.t()` creates a view (non-contiguous), `.contiguous()` materializes it as a new tensor with the correct memory layout. Skipping this causes runtime errors in PyG CUDA operations.

---

**Next:** `06_batch_pipeline_ast_extractor.md` — multiprocessing, checkpoint/resume, and solc binary management.
