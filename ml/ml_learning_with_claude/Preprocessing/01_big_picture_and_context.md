# Preprocessing — Chunk 1: The Big Picture & Context

> **What you'll learn:** What SENTINEL does, why graphs matter for code analysis, and just enough Solidity/Blockchain to understand everything else.
> **Time:** ~25 minutes
> **Interview relevance:** ML, AI, Blockchain

---

## 1. The Problem SENTINEL Solves

Smart contracts are programs deployed on a blockchain (primarily Ethereum). Once deployed, they **cannot be changed**. If there's a bug, it stays there — and bugs have cost hundreds of millions of dollars (DAO hack 2016: $60M, Poly Network 2021: $611M).

SENTINEL's job: **automatically detect vulnerability patterns in Solidity source code before deployment**.

This is a **classification problem**: given a contract's source code, predict which of 10 vulnerability classes are present. It's multi-label (a contract can have multiple vulnerabilities simultaneously).

---

## 2. Why Use a Graph? (Key Mental Model)

Code has inherent **structure** that plain text doesn't capture. Consider:

```solidity
function withdraw(uint amount) public {
    require(balances[msg.sender] >= amount);
    msg.sender.call{value: amount}("");  // ← external call FIRST
    balances[msg.sender] -= amount;      // ← state update SECOND ← REENTRANCY BUG!
}
```

A text model might not notice that the external call happens **before** the balance update. But a graph model sees:
- A `CONTROL_FLOW` edge: CFG node (external call) → CFG node (state write)
- The **direction** of that edge encodes the execution order

**This is why graphs are used**: they encode the **structural and semantic relationships** in code that flat text misses. Specifically:

| Graph Element | What it captures |
|--------------|-----------------|
| Node = FUNCTION | A function exists in the contract |
| Node = STATE_VAR | A state variable exists |
| Node = CFG_NODE_CALL | A statement that makes an external call |
| Node = CFG_NODE_WRITE | A statement that writes state |
| Edge = CONTROL_FLOW | Execution order between statements |
| Edge = CALLS | Function A calls function B |
| Edge = READS / WRITES | Function reads/writes a state variable |

---

## 3. Solidity Primer (Fast Forward — Know These Terms)

> ⚡ **FAST FORWARD:** You don't need to write Solidity. You just need to recognize these concepts when reading the code.

**Contract** — the top-level unit, like a class:
```solidity
contract MyToken {
    ...
}
```

**State variable** — stored permanently on the blockchain (expensive to read/write):
```solidity
mapping(address => uint) public balances;
```

**Function** — executable code. Can be `public`, `private`, `external`, `internal`. Can be `view` (read-only), `pure` (no state access), `payable` (receives ETH):
```solidity
function transfer(address to, uint amount) public {
    balances[msg.sender] -= amount;
    balances[to] += amount;
}
```

**Modifier** — a reusable function guard (like a decorator):
```solidity
modifier onlyOwner() {
    require(msg.sender == owner);
    _;  // ← "rest of function goes here"
}
```

**Event** — logged on the blockchain, can be monitored off-chain:
```solidity
event Transfer(address indexed from, address indexed to, uint value);
```

**Constructor** — runs once on deployment, sets initial state.

**Fallback / Receive** — special functions called when ETH is sent directly to the contract.

---

## 4. The 10 Vulnerability Classes — What They Mean

You need to understand these conceptually to understand why the feature engineering choices were made.

### Reentrancy (Class 6) — The Classic Bug
An external call is made before state is updated. The called contract can "re-enter" the caller, calling `withdraw()` again before the balance was decremented.

**Pattern:** `external_call → state_write` (bad order)
**What graph signal captures it:** CFG_NODE_CALL → CFG_NODE_WRITE with CONTROL_FLOW edge in that direction; plus `external_call_count > 0`

### Integer Overflow/Underflow (Class 4) — IntegerUO
In Solidity <0.8.0, arithmetic does NOT revert on overflow. `uint(0) - 1 = 2^256 - 1`. Attacker can cause `balances[msg.sender] = huge number` by triggering underflow.

**What graph signal captures it:** Functions that write state variables + high complexity (many CFG nodes with arithmetic)

### Denial of Service (Class 1)
A function loops over an array that an attacker can make arbitrarily large, consuming all gas and making the contract permanently unusable.

**Pattern:** Loop over user-controlled array + external calls inside the loop
**What graph signal captures it:** `has_loop=1.0` + `external_call_count > 0`

### Timestamp Manipulation (Class 7)
Contracts using `block.timestamp` for randomness or deadlines can be manipulated by miners who set the timestamp within a small range.

**What graph signal captures it:** `uses_block_globals=1.0` — a feature specifically added because Slither's normal state variable analysis MISSES `block.timestamp` (it's not a state variable).

### MishandledException (Class 5) / UnusedReturn (Class 9)
The return value of `.send()` or `.call()` is not checked. They return `false` on failure instead of reverting.

**What graph signal captures it:** `return_ignored=1.0` — the feature checks if the return value lvalue is ever read after the call in CFG order.

---

## 5. The Preprocessing Pipeline in One Paragraph

The preprocessing module (`ml/src/preprocessing/`) takes a Solidity `.sol` file and produces a **PyTorch Geometric (PyG) `Data` object** — a graph tensor. It uses **Slither** (a Solidity static analysis framework) to parse the contract into an AST, extract functions/variables/events/modifiers, and build a Control Flow Graph (CFG) for each function. Each node in the graph gets an **11-dimensional feature vector**. Edges are typed (11 types). The whole schema is versioned (`FEATURE_SCHEMA_VERSION = "v8"`).

---

## 6. PyTorch Geometric (PyG) — What You Need to Know

> 🎯 **INTERVIEW FOCUS:** "How are graphs represented in PyTorch?"

PyG represents graphs using **COO (Coordinate) format**:

```python
from torch_geometric.data import Data
import torch

# 4 nodes, 2 edges
x = torch.tensor([[feat1, feat2, ...],   # node 0 features
                  [feat1, feat2, ...],   # node 1 features
                  [feat1, feat2, ...],   # node 2 features
                  [feat1, feat2, ...]])  # node 3 features

# Edge from node 0→1 and node 2→3
edge_index = torch.tensor([[0, 2],   # source nodes
                           [1, 3]])  # destination nodes

# Edge type (typed edges)
edge_attr = torch.tensor([5, 6])  # CONTAINS=5, CONTROL_FLOW=6

graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
```

Key properties:
- `graph.x` — shape `[N, F]` where N=nodes, F=features per node
- `graph.edge_index` — shape `[2, E]` where E=edges
- `graph.edge_attr` — shape `[E]` with typed edge IDs
- `graph.num_nodes`, `graph.num_edges` — metadata

In SENTINEL: `N` = number of nodes in one contract's graph, `F = 11` (fixed), `E` = number of edges.

---

## 7. Slither — What It Does and Why It Matters

> ⚡ **FAST FORWARD:** Know what Slither provides. You don't need to know its API by heart.

**Slither** is a Python library that:
1. Calls the Solidity compiler (`solc`) to compile the contract
2. Parses the AST output into Python objects
3. Provides `Contract`, `Function`, `StateVariable`, `Event`, `Modifier` objects
4. For each function, provides a CFG: `func.nodes` (list of `Node` objects)
5. For each CFG node, provides IR operations: `node.irs` (list of IR operations like `HighLevelCall`, `LowLevelCall`, `Assignment`, etc.)

SENTINEL uses Slither to answer questions like:
- "Does this function call external contracts?" → `func.high_level_calls`
- "Does this function read block.timestamp?" → scan `node.irs` for `SolidityVariableComposed`
- "Does this function write state variables?" → `node.state_variables_written`

---

## 8. The Schema Version System — Why It Exists

> 🎯 **INTERVIEW FOCUS:** "How do you handle schema evolution in ML pipelines?"

The graph schema has evolved from v1 to v8. Each version adds/removes features or edge types. The problem: **if you change the schema without rebuilding your data**, the model receives features it was never trained on. This is a **silent accuracy bug** — no error message, just wrong predictions.

SENTINEL's solution:
1. A version constant: `FEATURE_SCHEMA_VERSION = "v8"` in `graph_schema.py`
2. Cache keys are: `"{file_md5}_{FEATURE_SCHEMA_VERSION}"` — changing the version **invalidates all cached graphs**
3. A **Change Policy**: any schema change requires (1) rebuilding all 41,576 graphs, (2) retraining from scratch

This is a classic **MLOps** pattern: **versioned artifacts + cache invalidation**.

---

## 9. The Single Source of Truth Pattern

Before the current design, both `ast_extractor.py` (offline batch) and `preprocess.py` (online inference) had **duplicate copies** of the feature engineering code. If you fixed a bug in one, you had to remember to fix the other. A missed sync meant inference would silently give wrong predictions.

The fix: extract everything into `graph_schema.py` and `graph_extractor.py`. Both pipelines **import** from these files. Now a single edit automatically fixes both.

```
graph_schema.py   ← constants (NODE_TYPES, EDGE_TYPES, FEATURE_NAMES)
       ↑
graph_extractor.py ← single extract_contract_graph() function
       ↑                            ↑
ast_extractor.py           preprocess.py
(offline batch)            (online inference)
```

> 🎯 **INTERVIEW FOCUS:** "How do you prevent train/inference skew?" This pattern is your answer. One function, imported in both places.

---

## 10. Summary — What You Now Know

- SENTINEL detects 10 Solidity vulnerability classes using a dual-path ML system
- Graphs capture structural code patterns that flat text misses
- The preprocessing module converts Solidity → PyG graph using Slither
- Graphs use COO format: `x[N,11]`, `edge_index[2,E]`, `edge_attr[E]`
- Schema versioning prevents silent train/inference skew
- Single source of truth (one module imported by both pipelines) prevents code duplication bugs

---

## Interview Questions to Practice

1. **"Why would you use a graph neural network for code analysis instead of just a language model?"**
   → Graphs capture execution order (CONTROL_FLOW edges), data flow (DEF_USE), and semantic relationships (READS/WRITES) that raw text doesn't encode structurally. A GNN can explicitly learn the pattern "external call before state write = reentrancy."

2. **"How do you prevent train/inference skew in production ML systems?"**
   → Single source of truth: one function (or module) that does preprocessing, imported by both the training pipeline and the inference API. Schema versioning with cache key invalidation.

3. **"A model is giving wrong predictions in production but no error is raised. What could cause this?"**
   → Feature engineering divergence: the inference pipeline is computing features differently from training (e.g., different normalization, different bug-fix applied to only one side). Schema version mismatch.

---

**Next:** `02_graph_schema_node_types_edges.md` — deep dive into the graph vocabulary.
