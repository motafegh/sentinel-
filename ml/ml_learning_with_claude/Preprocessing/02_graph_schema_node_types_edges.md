# Preprocessing — Chunk 2: Graph Schema — Node Types, Edge Types, and Constants

> **File:** `ml/src/preprocessing/graph_schema.py`
> **What you'll learn:** The full graph vocabulary — all 13 node types, all 11 edge types, the feature layout, Python patterns (IntEnum, dataclass, module-level asserts)
> **Time:** ~25 minutes
> **Interview relevance:** ML, AI, Blockchain

---

## 1. Why This File Is Special

`graph_schema.py` is the **single source of truth** for the entire SENTINEL system. It's imported by:
- `graph_extractor.py` — to build graphs
- `gnn_encoder.py` — to know how many edge types to embed
- `preprocess.py` — to validate inference inputs
- `drift_detector.py` — to know feature names for monitoring
- Every test file — to assert correctness

**Pattern:** Put all magic numbers in a central constants file. Never hardcode `11` (feature dim) or `13` (node types) anywhere else. Instead reference `NODE_FEATURE_DIM` and `len(NODE_TYPES)`.

---

## 2. The 13 Node Types

```python
NODE_TYPES: dict[str, int] = {
    # Declaration-level (stable since v1)
    "STATE_VAR":   0,
    "FUNCTION":    1,
    "MODIFIER":    2,
    "EVENT":       3,
    "FALLBACK":    4,
    "RECEIVE":     5,
    "CONSTRUCTOR": 6,
    "CONTRACT":    7,
    # CFG subtypes (added in v2)
    "CFG_NODE_CALL":   8,   # statement with external call
    "CFG_NODE_WRITE":  9,   # statement writing state variable
    "CFG_NODE_READ":   10,  # statement reading state variable
    "CFG_NODE_CHECK":  11,  # require/assert/if condition
    "CFG_NODE_OTHER":  12,  # everything else
}
```

**Why two categories?**

- **Declaration-level nodes (0–7):** One per declaration in the contract. These are "what exists" — the contract's API surface.
- **CFG nodes (8–12):** One per *statement* inside each function. These are "what happens" — the execution behavior.

**Priority rule for CFG nodes:** A single IR node might involve both a call AND a state write. Priority: CALL > WRITE > READ > CHECK > OTHER. External calls are the most vulnerability-relevant, so they win.

**Ordering constraint — NEVER INSERT IN THE MIDDLE:**
IDs must stay stable. If you add a new type at the end (id=13), existing datasets are safe. If you insert id=3 between EVENT and FALLBACK, you shift all IDs above 3 by 1, corrupting all saved `.pt` files.

---

## 3. The NodeType IntEnum — Python Pattern

```python
from enum import IntEnum

class NodeType(IntEnum):
    STATE_VAR      = NODE_TYPES["STATE_VAR"]    # 0
    FUNCTION       = NODE_TYPES["FUNCTION"]      # 1
    ...
    CFG_NODE_OTHER = NODE_TYPES["CFG_NODE_OTHER"] # 12
```

**Why IntEnum instead of just using the dict?**

IntEnum gives you:
1. **Named constants** — `NodeType.FUNCTION` is clearer than `1`
2. **Type safety** — you can't accidentally pass `"FUNCTION"` where an int is expected
3. **Derived from the dict** — values are read from `NODE_TYPES`, so they can't drift apart
4. **Set membership** — `NodeType.FUNCTION in {NodeType.FUNCTION, NodeType.MODIFIER}` works because IntEnum extends `int`

Usage in production code:
```python
# Good — readable, type-safe
STRUCTURAL_PREFIX_TYPES = frozenset({
    NodeType.FUNCTION, NodeType.MODIFIER, NodeType.CONSTRUCTOR,
    NodeType.FALLBACK, NodeType.RECEIVE,
})

# Bad — magic numbers, fragile
if node_type == 1 or node_type == 2 or node_type == 6:
    ...
```

**Why `frozenset` (not `set`)?**
`frozenset` is immutable and hashable. It can be used as a dict key or cached. It signals "this never changes." `set` is mutable and slightly slower for membership tests.

---

## 4. STRUCTURAL_PREFIX_TYPES — What It Is and Why

```python
STRUCTURAL_PREFIX_TYPES: frozenset[NodeType] = frozenset({
    NodeType.FUNCTION,
    NodeType.MODIFIER,
    NodeType.CONSTRUCTOR,
    NodeType.FALLBACK,
    NodeType.RECEIVE,
})
```

This is the set of node types used for **GNN prefix injection into the Transformer**. The model takes the top K=48 nodes of these types from the GNN output and prepends them as "soft prefix tokens" to the BERT input.

**Why not include CFG nodes?** After Phase 3 of the GNN, FUNCTION nodes already aggregate all their CFG children's signals via REVERSE_CONTAINS edges. Adding raw CFG nodes to the prefix would inflate the budget (K needs to be 100+ instead of 48) for little additional signal.

**Why K=48?** An audit of 41,576 training graphs showed P95 (95th percentile) of declaration node count = 47. So K=48 covers 95.5% of contracts without truncation.

> 🎯 **INTERVIEW FOCUS:** "How did you decide on K=48?" — Data-driven hyperparameter: measured the distribution, set K to cover the 95th percentile.

---

## 5. The 11 Edge Types

```python
EDGE_TYPES: dict[str, int] = {
    "CALLS":           0,  # function → internally-called function
    "READS":           1,  # function → state variable it reads
    "WRITES":          2,  # function → state variable it writes
    "EMITS":           3,  # function → event it emits
    "INHERITS":        4,  # contract → parent contract
    "CONTAINS":        5,  # function → its CFG_NODE children
    "CONTROL_FLOW":    6,  # CFG_NODE → successor CFG_NODE
    "REVERSE_CONTAINS":7,  # runtime-only: CFG_NODE → parent function
    "CALL_ENTRY":      8,  # calling CFG_NODE → callee ENTRYPOINT
    "RETURN_TO":       9,  # callee terminal → call-site successor
    "DEF_USE":         10, # defining node → reading node (data-flow)
}
NUM_EDGE_TYPES: int = 11
```

**Three categories of edges:**

### Semantic edges (0–4) — "What relationships exist"
These are graph-level, declaration-to-declaration relationships. They are independent of execution order.

| Edge | Example in Solidity |
|------|---------------------|
| CALLS | `withdraw()` calls `transfer()` |
| READS | `getBalance()` reads `balances[msg.sender]` |
| WRITES | `deposit()` writes `totalSupply` |
| EMITS | `transfer()` emits `Transfer` event |
| INHERITS | `ERC20Token` inherits from `StandardToken` |

### Intra-function structural (5–6) — "How a function is structured"
These turn declaration-level FUNCTION nodes into nodes with internal structure.

| Edge | Meaning |
|------|---------|
| CONTAINS | `transferFn` → each statement inside it |
| CONTROL_FLOW | `check_balance_stmt` → `call_external_stmt` → `write_balance_stmt` |

CONTROL_FLOW is the key reentrancy signal: the **direction** of the edge encodes execution order.

### Cross-function and data-flow (7–10) — "How execution flows across functions"
These are the most powerful for vulnerability detection:

| Edge | What it captures |
|------|----------------|
| REVERSE_CONTAINS | Aggregates CFG signal back up to function level (runtime only, never on disk) |
| CALL_ENTRY | Call site → entry of callee (inter-procedural CFG) |
| RETURN_TO | Callee exit → return point (closes inter-procedural loop) |
| DEF_USE | Variable defined in node A is read in node B (data flow) |

**CALL_ENTRY + RETURN_TO = ICFG-Lite (Interprocedural CFG).** This enables the GNN to reason about reentrancy patterns that span function boundaries.

**DEF_USE** captures things like: a variable computed by a call return value, then used in a state write. Key for integer overflow and return-value-ignored patterns.

---

## 6. The GNNEncoder Phase System

The 11 edge types aren't all fed to the GNN at once. They're organized into **3 phases**, each layer processing a different subset:

```
Phase 1 (layers 1+2):
  Edge types 0–5 (CALLS, READS, WRITES, EMITS, INHERITS, CONTAINS)
  "What exists and how it's structured"
  
Phase 2 (layers 3+4+5):
  Each layer gets a different subset:
    Layer 3: CONTROL_FLOW(6) only — execution order within functions
    Layer 4: CALL_ENTRY(8) + RETURN_TO(9) — cross-function control flow
    Layer 5: CF(6) + CALL_ENTRY(8) + RETURN_TO(9) — combined integration
  "How execution flows"

Phase 3 (layers 6+7+8):
  REVERSE_CONTAINS(7) — aggregates CFG signal up to functions
  Then CONTAINS(5) — pushes enriched function context back down
  "Bidirectional information exchange"
```

**Why separate phases?** Different vulnerability patterns need different information:
- Reentrancy needs CONTROL_FLOW to detect call-before-write
- Inheritance bugs need INHERITS edges  
- DoS needs CONTAINS + CONTROL_FLOW for loop detection

Mixing all edges in every layer would dilute the signal.

---

## 7. REVERSE_CONTAINS — Runtime-Only Edge (Important Detail)

```python
"REVERSE_CONTAINS":  7,  # runtime-only; CFG_NODE → parent function
```

This edge type is **never written to disk**. It's generated inside the GNNEncoder during Phase 3 by flipping CONTAINS(5) edges. Why?

If CONTAINS goes: `function → statement`, then REVERSE_CONTAINS goes: `statement → function`.

This allows the GNN to **aggregate** Phase 2 (execution-order) signals from all statement nodes back up to their parent function node. After Phase 3, each FUNCTION node carries a summary of its entire internal CFG.

Then the downward CONTAINS pass (conv4c in Phase 3) distributes this enriched FUNCTION context back to all child CFG nodes, giving every statement node awareness of what the full function does.

---

## 8. The 11-Dimension Feature Vector

```python
NODE_FEATURE_DIM: int = 11

FEATURE_NAMES: tuple[str, ...] = (
    "type_id",              # [0]  float(NODE_TYPE_ID)/12.0 → [0,1]
    "visibility",           # [1]  0.0=public/external, 0.5=internal, 1.0=private
    "uses_block_globals",   # [2]  1.0 if reads block.timestamp/number/etc.
    "view",                 # [3]  1.0 if Function.view (read-only)
    "payable",              # [4]  1.0 if Function.payable (receives ETH)
    "complexity",           # [5]  log1p(CFG_block_count)/log1p(100) → [0,1]
    "loc",                  # [6]  log1p(lines)/log1p(1000) → [0,1]
    "return_ignored",       # [7]  0.0=captured / 1.0=discarded / -1.0=unavailable
    "call_target_typed",    # [8]  0.0=raw addr / 1.0=typed / -1.0=unavailable
    "has_loop",             # [9]  1.0 if function contains a loop
    "external_call_count",  # [10] log1p(count)/log1p(20) → [0,1]
)
```

**Three groups of features:**

**Structural identity (0–1):**
- `type_id`: normalized node type (e.g., FUNCTION=1/12=0.083)
- `visibility`: access control level

**Function semantics (2–5):** Only for FUNCTION-type nodes; 0.0 for others
- `uses_block_globals`: does this function touch timestamp/block.number?
- `view`, `payable`: function modifier flags
- `complexity`: how many CFG blocks does this function have?

**Vulnerability signals (6–10):** The most interview-relevant
- `loc`: how long is this function?
- `return_ignored`: does this function discard call return values?
- `call_target_typed`: does this function call raw addresses?
- `has_loop`: does this function loop?
- `external_call_count`: how many external calls?

---

## 9. Why Log-Normalization?

```python
complexity = log1p(CFG_block_count) / log1p(100)  # [0,1]
loc        = log1p(lines) / log1p(1000)            # [0,1]
```

**The problem:** Raw values are unbounded. A CONTRACT node might have `loc=2538`. All other features are in [0,1]. If you put a raw `2538` into the feature vector alongside `payable=1.0`, the dot products in GNN attention are dominated by `loc`. The model effectively ignores all other features.

**Why `log1p` specifically?**
- `log1p(x) = log(1 + x)` — avoids `log(0)` since `log1p(0) = 0`
- Compresses large values: `log1p(100) ≈ 4.6`, `log1p(1000) ≈ 6.9`
- Preserves ordering (larger value = higher feature)
- Normalized by dividing by `log1p(max)` to put result in [0,1]

Example: `complexity` with 50 CFG blocks:
`log1p(50) / log1p(100) = 3.93 / 4.62 = 0.85` → reasonable [0,1] value

> 🎯 **INTERVIEW FOCUS:** "How do you handle features with very different scales?" — Log-normalization for naturally skewed count data. Min-max or z-score for normally distributed data.

---

## 10. The VISIBILITY_MAP

```python
VISIBILITY_MAP: dict[str, float] = {
    "public":   0.0,
    "external": 0.0,
    "internal": 0.5,
    "private":  1.0,
}
```

**Design decisions:**
1. **`public` and `external` both map to 0.0** — from a vulnerability perspective, both are accessible from outside the contract
2. **Ordinal, not one-hot** — the ordering `private > internal > public` is meaningful for vulnerability risk (private functions can't be attacked externally)
3. **Values in [0,1]** — stays in the same scale as all other features

**History:** Earlier versions used `{public:0, internal:1, private:2}`. The `private=2` value exceeded the `[0,1]` range, creating a 2× scale imbalance in GNN dot products for 17.7% of the dataset. This was **BUG-3**, fixed in schema v6 by switching to the float encoding above.

> 🎯 **INTERVIEW FOCUS:** "You found that 17.7% of your training data had a feature value outside the expected range. What do you do?" — Apply an in-place patch (`patch_graph_features.py`), bump the schema version to invalidate caches, document the bug and fix.

---

## 11. Module-Level Assertions — Defensive Programming

At the bottom of `graph_schema.py`:

```python
assert len(FEATURE_NAMES) == NODE_FEATURE_DIM, (
    f"FEATURE_NAMES has {len(FEATURE_NAMES)} entries but NODE_FEATURE_DIM={NODE_FEATURE_DIM}."
)
assert len(EDGE_TYPES) == NUM_EDGE_TYPES, ...
assert len(NODE_TYPES) == 13, ...
```

These assertions run **at import time**. If you add a feature to `FEATURE_NAMES` but forget to update `NODE_FEATURE_DIM`, the entire system fails loudly at startup — not silently during training.

**Pattern:** Use module-level asserts to catch invariant violations at import time rather than deep in a training loop.

---

## 12. Schema Version History — What to Know for Interviews

The schema evolved through 8 versions. Key version events to remember:

| Version | Key Change | Why |
|---------|-----------|-----|
| v1 | 8 features, 5 edge types | Original implementation |
| v2 | +CFG nodes, +CONTAINS/CF edges | Needed execution-order signal for reentrancy |
| v4 | `pure` → `uses_block_globals` | `pure` was almost always 0 (useless); block globals needed for Timestamp class |
| v5 | BUG FIX: log-normalize complexity, loc | Raw values dominated dot products |
| v5 | BUG FIX: `most_derived` contract selection | Old heuristic had 47.4% error rate |
| v6 | BUG FIX: visibility 0/1/2 → 0/0.5/1.0 | Value=2 violated [0,1] range for 17.7% of data |
| v7 | Drop `in_unchecked`, add ICFG/DEF_USE | in_unchecked was 0 for 87.9% of dataset (dead feature) |
| v8 | +CALL_ENTRY, +RETURN_TO, +DEF_USE on disk | Cross-function + data-flow signal for reentrancy/CEI |

> 🎯 **INTERVIEW FOCUS:** "How do you handle feature drift in your ML pipeline?" — Versioned schemas, cache key invalidation, rebuild+retrain when schema changes.

---

## 13. Summary

You now understand:
- 13 node types: 8 declaration-level (what exists) + 5 CFG subtypes (what happens)
- 11 edge types in 3 semantic categories: semantic / structural / cross-function+dataflow
- 11-dim feature vector with log-normalization
- `IntEnum` for type-safe named constants
- `frozenset` for immutable set membership
- Module-level assertions for invariant checking at import time
- Schema versioning with cache invalidation

---

## Interview Questions

1. **"What is COO format for graphs and why use it?"**
   → Coordinate format: `edge_index[2, E]` with source/destination node indices. Memory-efficient for sparse graphs — only stores actual edges, not all N×N possibilities.

2. **"Why separate node types for CFG statements vs declaration-level nodes?"**
   → Different initial embeddings for different semantic roles. A CFG_NODE_CALL starts with different representation than CFG_NODE_WRITE, giving the GNN a head start on their different vulnerability implications.

3. **"If you removed the `has_loop` feature, which vulnerability class would lose the most signal?"**
   → DenialOfService (unbounded loops + external calls inside loops). Also IntegerUO (overflow inside loops).

---

**Next:** `03_feature_engineering_deep_dive.md` — each of the 11 features explained at the implementation level.
