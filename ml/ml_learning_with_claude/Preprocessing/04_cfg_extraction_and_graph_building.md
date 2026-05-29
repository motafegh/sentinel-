# Preprocessing — Chunk 4: CFG Extraction, ICFG Edges, and DEF_USE Data-Flow

> **File:** `ml/src/preprocessing/graph_extractor.py` (lines 439–778)
> **What you'll learn:** How the Control Flow Graph (CFG) is built, ICFG-Lite for cross-function reasoning, DEF_USE for data-flow reasoning, and the critical two-pass algorithm pattern.
> **Time:** ~30 minutes
> **Interview relevance:** ML (graph construction), AI (GNN input design), Blockchain (Solidity CFG semantics)

---

## 1. What Is a Control Flow Graph (CFG)?

A **Control Flow Graph** (CFG) represents the execution order of statements inside a function. Each node is a statement (or a synthetic node like ENTRY_POINT), and edges represent "execution can go from statement A to statement B."

Example for a simple function:
```solidity
function withdraw(uint amount) public {
    require(balances[msg.sender] >= amount);   // Node 1: CHECK
    balances[msg.sender] -= amount;             // Node 2: WRITE
    msg.sender.call{value: amount}("");         // Node 3: CALL
}
```

CFG:
```
ENTRY_POINT → Node1(CHECK) → Node2(WRITE) → Node3(CALL) → EXIT
```

CONTROL_FLOW edges: ENTRY→1, 1→2, 2→3, 3→EXIT

Note: In this **correct** implementation, state is updated (WRITE) before the external call (CALL). The GNN would see: CFG_NODE_WRITE → CFG_NODE_CALL. Compare to reentrancy: CALL → WRITE.

---

## 2. The Two-Pass Algorithm in `_build_control_flow_edges()`

```python
def _build_control_flow_edges(func, func_node_idx, node_index_map, x_list, node_metadata, parent_features=None):
    # Sort for determinism
    cfg_nodes = sorted(
        func.nodes or [],
        key=lambda n: (
            n.source_mapping.lines[0] if n.source_mapping and n.source_mapping.lines else 0,
            n.node_id,
        ),
    )
    
    # PASS 1: Assign indices, build feature vectors
    for slither_node in cfg_nodes:
        cfg_type  = _cfg_node_type(slither_node)
        graph_idx = len(x_list)          # ← CRITICAL: use global list length
        node_index_map[slither_node] = graph_idx
        x_list.append(_build_cfg_node_features(...))
        contains_edges.append((func_node_idx, graph_idx))
    
    # PASS 2: Build CONTROL_FLOW edges (all nodes indexed in Pass 1)
    for slither_node in cfg_nodes:
        src_idx = node_index_map[slither_node]
        for successor in (slither_node.sons or []):
            if successor in node_index_map:
                control_flow_edges.append((src_idx, node_index_map[successor]))
    
    return contains_edges, control_flow_edges
```

**Why two passes?**

You can't build CONTROL_FLOW edges in one pass. To add an edge from node A to node B, you need the **graph index** of B. But B might not have been processed yet. The two-pass solution:
1. **Pass 1:** Assign graph indices to all CFG nodes and build their feature vectors
2. **Pass 2:** Now all indices exist, so build all edges using `node_index_map[successor]`

**The most critical line:**
```python
graph_idx = len(x_list)  # CORRECT index
```

`x_list` is the **single shared list** of all node feature vectors (declaration nodes + CFG nodes from all functions). Its length before appending is the next available global index. Do NOT use `len(node_index_map)` — that's a local dict for only this function's CFG nodes.

**Why sort CFG nodes?**
```python
sorted(func.nodes, key=lambda n: (source_line, node_id))
```
Without sorting, `func.nodes` might be in a different order across Slither versions or Python dict iteration orders. Sorted by source line + node_id guarantees **deterministic** graph construction — the same contract always produces the same graph.

---

## 3. CONTAINS Edges — Connecting Functions to Their Statements

```python
contains_edges.append((func_node_idx, graph_idx))
```

For each CFG node, we add a CONTAINS edge from its parent FUNCTION node to it. This creates a two-level graph structure:

```
CONTRACT (id=7)
  └─CONTAINS→ FUNCTION "withdraw" (id=1)
                └─CONTAINS→ CFG_NODE_CHECK (id=11)  [require]
                └─CONTAINS→ CFG_NODE_WRITE (id=9)   [balance update]
                └─CONTAINS→ CFG_NODE_CALL (id=8)    [external call]
```

In Phase 1 of the GNN, these CONTAINS edges allow the GNN to see "what statements are inside this function."

In Phase 3, REVERSE_CONTAINS edges (the reverse of CONTAINS) allow information to flow back up from statement nodes to their parent function.

---

## 4. ICFG-Lite: `_add_icfg_edges()`

**ICFG** = Interprocedural CFG. A regular CFG only shows control flow **within** a single function. An ICFG connects multiple functions' CFGs, showing what happens when function A calls function B.

```python
def _add_icfg_edges(contract, func_entry_map, func_terminal_map, func_cfg_maps, edges, edge_types):
    for func in contract.functions:
        for node in func.nodes:
            for callee in (node.internal_calls or []):
                callee_key = callee.canonical_name
                
                # CALL_ENTRY: calling node → callee's ENTRYPOINT
                callee_entry = func_entry_map.get(callee_key)
                if callee_entry is not None:
                    edges.append([caller_idx, callee_entry])
                    edge_types.append(EDGE_TYPES["CALL_ENTRY"])
                
                # RETURN_TO: callee terminals → call-site successors
                for terminal_idx in callee_terminals:
                    for son in node.sons:
                        edges.append([terminal_idx, son_idx])
                        edge_types.append(EDGE_TYPES["RETURN_TO"])
```

**What this enables:**
```
caller_function:
  ...
  call_site_node ──CALL_ENTRY──► callee ENTRYPOINT
  
callee_function:
  ENTRYPOINT
  ...
  terminal_node ──RETURN_TO──► call_site_successor
```

**Why does this help detect reentrancy?**

Reentrancy spans two functions: the `withdraw()` function calls an external contract which calls back into `withdraw()`. The ICFG-Lite edges (CALL_ENTRY/RETURN_TO) allow the GNN to reason about the sequence of events across function calls — the GNN can learn that "a CFG_NODE_WRITE after a CALL_ENTRY → CALL_ENTRY chain" is a reentrancy pattern.

**Three data structures accumulated during function processing:**
```python
_func_entry_map:    dict = {}   # canonical_name → graph_idx of ENTRYPOINT node
_func_terminal_map: dict = {}   # canonical_name → [graph_idx of terminal nodes]
_func_cfg_maps:     dict = {}   # canonical_name → {slither_node → graph_idx}
```

These are filled during the main function loop and used after the loop to add ICFG edges.

---

## 5. DEF_USE Edges: `_add_def_use_edges()`

**Data-flow analysis**: tracks where a value is defined (produced) and where it is used (consumed).

```python
def _add_def_use_edges(contract, func_cfg_maps, edges, edge_types):
    for func in contract.functions:
        # Pass 1: Build def_map — variable_name → [node_idx defining it]
        def_map = {}
        for node in func.nodes:
            for ir in (node.irs or []):
                lval = getattr(ir, "lvalue", None)
                if isinstance(lval, LocalVariable):
                    def_map.setdefault(lval.name, []).append(node_idx)
        
        # Pass 2: For each IR read, emit DEF_USE edge if var is in def_map
        for node in func.nodes:
            for ir in (node.irs or []):
                for var in (getattr(ir, "read", None) or []):
                    if var.name in def_map:
                        for def_idx in def_map[var.name]:
                            if def_idx != use_idx:
                                edges.append([def_idx, use_idx])
                                edge_types.append(EDGE_TYPES["DEF_USE"])
```

**Example of what DEF_USE captures:**

```solidity
function transfer(address to, uint amount) public {
    uint newBalance = balances[msg.sender] - amount;  // node A: defines newBalance
    require(newBalance >= 0);                          // node B: READS newBalance
    balances[msg.sender] = newBalance;                 // node C: READS newBalance
}
```

DEF_USE edges: A→B, A→C

This is crucial for **integer overflow** detection: the GNN can learn the pattern "value computed by arithmetic operation, then written to state without a bounds check." Without DEF_USE edges, the GNN can't connect the arithmetic node (A) to the write node (C).

**Why only LocalVariable, not StateVariable or TemporaryVariable?**
- `StateVariable` DEF_USE is already covered by READS/WRITES edges (declaration-level)
- `TemporaryVariable` is an intra-node SSA temporary — the def and use happen within the same IR operation, so there's no meaningful edge to add
- `LocalVariable` represents function-scoped variables that flow between statements — this is the interesting data-flow

**Deduplication:**
```python
seen_pairs: set = set()
pair = (def_idx, use_idx)
if pair not in seen_pairs:
    seen_pairs.add(pair)
    edges.append(...)
```
A variable might be read by multiple IR operations in the same CFG node. We add at most one DEF_USE edge per (def_node, use_node) pair.

---

## 6. `_cfg_node_type()` — Priority Classification

When Slither generates a CFG node, it might represent multiple operations. How do you decide which node type to assign?

```python
def _cfg_node_type(slither_node: Any) -> int:
    irs = list(slither_node.irs or [])
    
    # Priority 1: any external call (CALL)
    if any(isinstance(op, (LowLevelCall, HighLevelCall, Transfer, Send)) for op in irs):
        return NODE_TYPES["CFG_NODE_CALL"]
    
    # Priority 2: state write
    if state_variables_written or any(isinstance(op.lvalue, StateVariable) for op in irs):
        return NODE_TYPES["CFG_NODE_WRITE"]
    
    # Priority 3: state read
    if state_variables_read or any isinstance(v, StateVariable) ...:
        return NODE_TYPES["CFG_NODE_READ"]
    
    # Priority 4: control flow check
    if slither_node.type in {IF, IFLOOP, STARTLOOP, ENDLOOP, THROW}:
        return NODE_TYPES["CFG_NODE_CHECK"]
    
    return NODE_TYPES["CFG_NODE_OTHER"]
```

**Why does priority matter?**

Consider a node like: `if (balances[msg.sender] >= amount) { addr.call(...); }`. Slither might represent this as a single IR node with both a state read AND an external call. Assigning `CFG_NODE_CALL` (highest priority) is correct because the external call is the most vulnerability-relevant operation.

**The GNN learns vulnerability patterns from combinations of node types and edge types.** The reentrancy pattern is: `CFG_NODE_CALL --CONTROL_FLOW--> CFG_NODE_WRITE`. If we miscategorize the call node as `CFG_NODE_READ`, the GNN can't learn this pattern.

**Why include Transfer/Send?**
```python
if any(isinstance(op, (LowLevelCall, HighLevelCall, Transfer, Send)) for op in irs):
```
DoS vulnerabilities often use `addr.transfer(amount)` in loops. Transfer/Send are distinct Slither IR types that must be explicitly included. This was the same bug fix as in `_compute_external_call_count`.

---

## 7. Graph Node Ordering — Why It Matters

The main `extract_contract_graph()` function adds nodes in a **fixed insertion order**:

```python
# Fixed insertion order (must be stable across runs)
_add_node(contract, NODE_TYPES["CONTRACT"])        # 1. CONTRACT
for parent in contract.inheritance: _add_node(parent, NODE_TYPES["CONTRACT"])  # 2. Parent contracts
for var in contract.state_variables: _add_node(var, NODE_TYPES["STATE_VAR"])  # 3. State vars

for func in contract.functions:
    fn_idx = _add_node(func, NODE_TYPES["FUNCTION"])   # 4. FUNCTION
    # Immediately after: all CFG nodes for this function
    _build_control_flow_edges(func, fn_idx, ...)       # 4a. CFG nodes

for mod in contract.modifiers: _add_node(mod, NODE_TYPES["MODIFIER"])  # 5. Modifiers
for event in contract.events: _add_node(event, NODE_TYPES["EVENT"])    # 6. Events
```

**Why does order matter?**
Node indices (their positions in `x_list`) flow directly into `edge_index`. If the insertion order is different between two runs of the same contract, the graph topology is different. An index that was node 5 in one run might be node 8 in another run, making `edge_index[0][3] = 5` mean a different node.

**The `_add_node` deduplication guard:**
```python
def _add_node(obj, initial_type_id):
    key = obj.canonical_name or obj.name
    if key in node_map:
        return None  # Skip duplicate
    idx = len(x_list)
    node_map[key] = idx
    ...
    return idx
```

Slither's `contract.functions` can include inherited functions multiple times. The `node_map` dict deduplicates by canonical name — each function gets exactly one node.

---

## 8. The CFG Failure Rate Monitor

```python
_cfg_failure_count = 0
_func_total = len(contract.functions)

for func in contract.functions:
    try:
        contains_edges, control_flow_edges = _build_control_flow_edges(...)
    except Exception as exc:
        _cfg_failure_count += 1
        logger.warning("CFG extraction failed for %s: %s", func.canonical_name, exc)

if _cfg_failure_count > 0 and _func_total > 0:
    failure_rate = _cfg_failure_count / _func_total
    log_fn = logger.error if failure_rate > 0.05 else logger.debug
    log_fn("CFG extraction: %d/%d functions failed (%.0f%%)", ...)
```

**MLOps pattern:** Track failure rates, not just failures. A single function failing CFG extraction is benign (happens with synthetic compiler-generated functions). But if 10% of functions fail, there's a Slither version mismatch or corrupt source — escalate to `logger.error`.

The threshold `0.05` (5%) is a domain knowledge decision: SENTINEL contracts typically have <5% CFG failures.

---

## 9. ICFG Entry/Terminal Detection

```python
from slither.core.cfg.node import NodeType as _SNT

func_nodes = func.nodes or []
for _n in func_nodes:
    if _n.type == _SNT.ENTRYPOINT and _n in cfg_node_map:
        _func_entry_map[func_key] = cfg_node_map[_n]
        break

_func_terminal_map[func_key] = [
    cfg_node_map[_n]
    for _n in func_nodes
    if _n in cfg_node_map and not (getattr(_n, "sons", None) or [])
]
```

- **Entry node**: has type `ENTRYPOINT` (Slither always creates this as the first CFG node)
- **Terminal nodes**: nodes with no successors (`sons` is empty) — these are the "last statement before return"

The "no successors" approach for terminal detection is robust: it works even if Slither changes how it represents function exit points.

---

## 10. Putting It Together: The Graph for a Reentrancy Contract

For this vulnerable function:
```solidity
function withdraw(uint amount) public {
    require(balances[msg.sender] >= amount);  // CHECK
    msg.sender.call{value: amount}("");        // CALL (before state update!)
    balances[msg.sender] -= amount;            // WRITE (after call = bug!)
}
```

Graph nodes (simplified):
```
Node 0: CONTRACT
Node 1: STATE_VAR "balances"
Node 2: FUNCTION "withdraw"
Node 3: CFG_NODE_OTHER (ENTRYPOINT)
Node 4: CFG_NODE_CHECK (require)
Node 5: CFG_NODE_CALL  (msg.sender.call)   ← type_id=8
Node 6: CFG_NODE_WRITE (balances update)   ← type_id=9
```

Edges:
```
0→1: CONTAINS (contract owns state var)
0→2: CONTAINS? (no — CONTAINS is function→CFG, not contract→function)
2→3: CONTAINS (function → entry CFG node)
2→4: CONTAINS
2→5: CONTAINS
2→6: CONTAINS
3→4: CONTROL_FLOW (execution order)
4→5: CONTROL_FLOW ← call BEFORE write = reentrancy signal
5→6: CONTROL_FLOW
2→1: READS (function reads balances)
2→1: WRITES (function writes balances)
```

The GNN Phase 2 processes CONTROL_FLOW edges. The pattern `CFG_NODE_CALL --CF--> CFG_NODE_WRITE` is the reentrancy signature that the model must learn to detect.

---

## 11. Summary

| Component | What it does | Key insight |
|-----------|-------------|-------------|
| `_build_control_flow_edges` | CFG → graph nodes + edges | Two-pass: assign indices first, then add edges |
| `_cfg_node_type` | Classify CFG statements | Priority order: CALL > WRITE > READ > CHECK > OTHER |
| `_add_icfg_edges` | Cross-function CFG | Enables reentrancy pattern detection across function boundaries |
| `_add_def_use_edges` | Data-flow edges | Connects where a value is computed to where it's used |
| Deduplication | `node_map` dict | canonical_name → index, prevents duplicate nodes |
| Determinism | Sort by source_line + node_id | Same source = same graph, every run |
| Failure monitoring | failure_rate > 0.05 → error | Data quality monitoring at extraction time |

---

## Interview Questions

1. **"What is a Control Flow Graph and why is it useful for vulnerability detection?"**
   → CFG represents execution order as a directed graph. Edge direction encodes which statements run before others. Reentrancy = external call before state update = CFG_NODE_CALL → CFG_NODE_WRITE edge direction.

2. **"What is interprocedural analysis?"**
   → Analyzing code flow across function call boundaries. ICFG-Lite adds CALL_ENTRY edges from call sites to callee entry points and RETURN_TO edges from callee exits to call-site continuations. Enables detecting vulnerabilities that span multiple functions.

3. **"How do you handle cycles in a CFG (loops)?"**
   → The CFG naturally represents loops as directed cycles (IFLOOP node with a back edge). The GNN processes these with message passing — cyclic edges are handled by the message-passing framework (each node aggregates neighbor messages regardless of cycles). The `has_loop` feature explicitly flags the presence of loops.

4. **"Why is DEF_USE analysis important for integer overflow detection?"**
   → Integer overflow often involves: arithmetic operation (defines a value) → that value is written to state or used in a condition without bounds checking. DEF_USE edges directly connect the defining node to the using node, allowing the GNN to learn "arithmetic result used directly in state write = potential overflow."

---

**Next:** `05_contract_selection_and_main_pipeline.md` — the `extract_contract_graph()` function and the contract selection heuristics.
