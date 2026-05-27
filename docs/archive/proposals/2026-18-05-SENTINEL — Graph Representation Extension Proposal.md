# SENTINEL — Graph Representation Extension Proposal
## v3 — Final Consolidated Version

**Project:** Sentinel ML Module — Smart Contract Vulnerability Detection
**Document type:** Full Engineering Design Proposal
**Schema baseline:** v7
**Codebase commits audited:** `011466693d9ba32ec1f34d848349e11613ebc8a4` (first audit); `0114666` + `700081c` (second audit); third-pass integration against live v7 training logs (Epochs 5–8)
**Date:** May 2026
**Status:** Ready for implementation

---

## Preface — Audit History

This document has been through two independent audits against live source code.
All claims are verified against actual function bodies, not comments or docstrings
(both of which contain stale content in the v7 codebase).

### First Audit Corrections (8 items)

| # | Draft error | Corrected fact | Verified at |
|---|-------------|----------------|-------------|
| 1 | `graph.y : [10] float32` multi-label stored in graph | `graph.y = torch.tensor([label], dtype=torch.long)` — scalar `[1]` int64 binary. Multi-label `[10]` float32 is assembled at training time by `DualPathDataset` from `multilabel_index.csv`, never stored in `.pt` files | `ast_extractor.py:235`; `dual_path_dataset.py:320–332` |
| 3 | `v.is_storage` used to detect `StateVariable` | Not a Slither API attribute — `hasattr(v, "is_storage")` is always `False`, making the branch silently dead. Correct idiom: `isinstance(v, StateVariable)` — already used in `_cfg_node_type()` | `graph_extractor.py:459,467` |
| 4 | `isinstance(ir_op, Binary)` catches arithmetic only | `Binary` covers comparisons, logical, bitwise operators too. Must filter by `BinaryType` to restrict to arithmetic ops | Slither IR API — not verifiable from codebase alone |
| 5 | `cfg_node_map` accessible as global for ICFG second pass | It is declared `cfg_node_map: dict = {}` *inside* the `for func in contract.functions` loop and discarded after each iteration. A global map must be explicitly accumulated | `graph_extractor.py:989` |
| 6 | `if called_func not in node_map` correct key check | `node_map` is keyed by `canonical_name` strings (`dict[str, int]` at line 934). `called_func` is a Slither `Function` object — this check is always `True`, silently skipping all callees | `graph_extractor.py:934,940` |
| 7 | `_build_node_features` docstring describes 12-dim v4 layout | Docstring is stale (line 632 says "12-dimensional feature vector (v4 schema)"). Function body correctly returns 11 dims. Pre-existing codebase bug | `graph_extractor.py:632,714–727` |
| 8 | `in_unchecked` described as removed | `_compute_in_unchecked()` is fully implemented and still called at line 699. Its result is assigned to `in_unchecked` but the return list does not include it — dead code residue from the v7 drop | `graph_extractor.py:699,714–727` |

### Third-Pass Integration (v3)

Third-pass audited and merged a comprehensive improvement proposal against live v7 training logs
and live source code. Items adopted: training state snapshot, speed optimization status,
structural comparison gate (§7.7), checkpoint naming risk (§7.8), and Part 10 Concerns.
Items rejected and documented: VRAM "4.8%" claim (actual 86%), S1 batch_size increase (OOM),
S2/S3 as new proposals (already applied), DoS 9-class reduction (blocked by ZKML).

### Second Audit Corrections (4 items — errors introduced by first audit)

| # | First-audit error | Corrected fact | Verified at |
|---|-------------------|----------------|-------------|
| A | "Live working dataset is ~44,470 unique contracts" | **Effective training set is ~41,576 graphs** (= 41,577 cache pairs). 44,470 was the windowed token count at a prior snapshot — tokens are not the training bottleneck. ~3,000 contracts have tokens but no matching graph and are silently excluded by the cache builder. The original "~41,522" figure was directionally correct for the graph-side count. | `ls ml/data/graphs/*.pt \| wc -l` = 41,576; cache = 41,577 pairs |
| B | `CFG_NODE_CHECK` triggered by "IF/IFLOOP/THROW" | `check_types` in `_cfg_node_type()` is `{SNT.IF, SNT.IFLOOP, SNT.STARTLOOP, SNT.ENDLOOP, SNT.THROW}` — **STARTLOOP and ENDLOOP are included**. The first audit's description was incomplete. | `graph_extractor.py:443` |
| C | Extension B "Category 3: Condition variables" listed as a *definition* category | `Condition` IR ops have no `lvalue` — they cannot define variables. Category 3 does not appear in `_is_targeted_definition()` and cannot. Condition nodes receive DEF_USE edges as **use sites** (when a tracked definition flows into them), not as definition sites. The category table was misleading; the implementation is correct. | `graph_extractor.py:_is_targeted_definition` |
| D | Extension A uses `caller_node.internal_calls` without caveats | The existing extractor uses `func.internal_calls` (function-level) at line 1077. Whether Slither's `Node` objects expose `internal_calls` at node-level is not confirmed by any usage in the current codebase. **Requires a 500-contract sample test before full implementation** — see Risk 7.6. | `graph_extractor.py:1077` |

---

## Part 1 — Current System Reference

This section is the authoritative ground-truth description of the v7 system,
verified line-by-line against actual function bodies (not docstrings or comments).

---

### 1.1 Graph Object Shape Contract

`extract_contract_graph()` in `graph_extractor.py` is the single source of
truth for both the offline batch pipeline (`ast_extractor.py`) and the online
inference API (`preprocess.py`). It returns a PyG `Data` object with the
following fields:

```
graph.x              [N, 11]  float32   node feature matrix
graph.edge_index     [2, E]   int64     directed edge pairs, COO format
graph.edge_attr      [E]      int64     edge type IDs 0–6 on disk
                                        (ID 7 REVERSE_CONTAINS is runtime-only,
                                         generated in GNNEncoder.forward(), never
                                         written to any .pt file)
graph.node_metadata  list[dict]         index-aligned with x
                                        keys: "name" (str), "type" (str), "source_lines" (list[int])
graph.contract_name  str
graph.num_nodes      int
graph.num_edges      int
```

`extract_contract_graph()` does NOT attach `.y`, `.contract_hash`, or
`.contract_path`. Each caller attaches its own values:

```python
# ast_extractor.py:235 — offline batch
graph.y = torch.tensor([label], dtype=torch.long)   # scalar [1] int64

# DualPathDataset — training time, multi-label mode
y_multilabel = multilabel_index[contract_hash]       # [10] float32, assembled at runtime
```

The `[10]` float32 multi-label representation is a training-time construct,
not a stored graph property.

---

### 1.2 Node Vocabulary — 13 Types (IDs 0–12)

```
ID   Name              Origin                          Role
──────────────────────────────────────────────────────────────────
 0   STATE_VAR         contract.state_variables        Storage slot
 1   FUNCTION          contract.functions              Entry point, owns CFG
 2   MODIFIER          contract.modifiers              Reusable guard
 3   EVENT             contract.events                 Log emission target
 4   FALLBACK          Function (is_fallback=True)     ETH fallback entry
 5   RECEIVE           Function (is_receive=True)      ETH receive entry
 6   CONSTRUCTOR       Function (is_constructor=True)  Initialization path
 7   CONTRACT          Slither.contracts               Top-level scope node
 8   CFG_NODE_CALL     FunctionNode w/ external call   Statement: call (priority 1)
 9   CFG_NODE_WRITE    FunctionNode w/ state write     Statement: write (priority 2)
10   CFG_NODE_READ     FunctionNode w/ state read      Statement: read  (priority 3)
11   CFG_NODE_CHECK    FunctionNode typed IF/IFLOOP/   Statement: guard (priority 4)
                       STARTLOOP/ENDLOOP/THROW
12   CFG_NODE_OTHER    FunctionNode (all others)       Statement: other (priority 5)
```

Priority rule in `_cfg_node_type()` (`graph_extractor.py:438–478`):
when a single Slither IR node contains multiple operation types, the
highest-priority type wins. A node that both writes state and makes an
external call is typed as `CFG_NODE_CALL` (8), not `CFG_NODE_WRITE` (9).

`check_types` (verified at line 443):
```python
check_types = {SNT.IF, SNT.IFLOOP, SNT.STARTLOOP, SNT.ENDLOOP, SNT.THROW}
```
Note: `ENDLOOP` is a convergence point (join node after a loop), not a
branch. It appears in `check_types` because Slither may assign it guard-like
semantics in its CFG. This is consistent with the live extractor.

---

### 1.3 Edge Vocabulary — 8 Types (IDs 0–7)

```
ID   Name               Direction                  Phase   On-disk
────────────────────────────────────────────────────────────────────
 0   CALLS              FUNCTION → FUNCTION        1       Yes
 1   READS              FUNCTION → STATE_VAR       1       Yes
 2   WRITES             FUNCTION → STATE_VAR       1       Yes
 3   EMITS              FUNCTION → EVENT           1       Yes
 4   INHERITS           CONTRACT → CONTRACT        1       Yes
 5   CONTAINS           FUNCTION → CFG_NODE        1       Yes
 6   CONTROL_FLOW       CFG_NODE → CFG_NODE        2       Yes
 7   REVERSE_CONTAINS   CFG_NODE → FUNCTION        3       NO — runtime only
```

`REVERSE_CONTAINS(7)` is generated inside `GNNEncoder.forward()`:
```python
rev_contains_ei = edge_index[:, contains_mask].flip(0)
rev_type_ids    = torch.full((n_rev,), 7, dtype=torch.long, ...)
rev_contains_ea = self.edge_embedding(rev_type_ids)
```
No `.pt` file on disk ever contains edge type ID 7. The `nn.Embedding` table
needs 8 rows so that index-7 lookups do not fail during the forward pass.

---

### 1.4 Node Feature Vector — v7 Schema (11 dims)

Verified against actual return list at `graph_extractor.py:714–727`.

```
Dim  Name                 Range        Description
──────────────────────────────────────────────────────────────────────────────
[0]  type_id              [0,1]        float(NODE_TYPES[kind]) / 12.0
[1]  visibility           {0,.5,1}     public/external=0.0, internal=0.5, private=1.0
[2]  uses_block_globals   {0,1}        reads block.timestamp/number/difficulty/basefee
[3]  view                 {0,1}        read-only function
[4]  payable              {0,1}        ether-accepting entry point
[5]  complexity           [0,1]        log1p(CFG block count) / log1p(100)
[6]  loc                  [0,1]        log1p(source lines) / log1p(1000)
[7]  return_ignored       {-1,0,1}     call return discarded (-1 = IR unavailable)
[8]  call_target_typed    {-1,0,1}     external call uses typed interface (not raw address)
[9]  has_loop             {0,1}        function contains IFLOOP/STARTLOOP/ENDLOOP
[10] external_call_count  [0,1]        log1p(n) / log1p(20); includes Transfer/Send
```

CFG nodes inherit dims [1, 3, 4, 5, 9] from their parent FUNCTION node
(BUG-C3 fix in `_build_cfg_node_features()`, lines 527–531).
Verified:
```python
visibility = p[1] if len(p) > 1 else 0.0
view       = p[3] if len(p) > 3 else 0.0
payable    = p[4] if len(p) > 4 else 0.0
complexity = p[5] if len(p) > 5 else 0.0
has_loop   = p[9] if len(p) > 9 else 0.0
```

Dim [2] `uses_block_globals` is hardcoded `0.0` for CFG nodes (line 536) —
it is a function-scope signal, not statement-scope.

`return_ignored` and `call_target_typed` are asserted in range at lines
711–712; `-1.0` is a sentinel meaning "IR unavailable for this node."

The dimension formerly at index [9] in v6, `in_unchecked`, was dropped in v7
(BUG-L2). Its computation function `_compute_in_unchecked()` still exists and
is still called at line 699, but its result is discarded — dead code (see Section 2).

---

### 1.5 GNN Encoder — v7 Architecture

**7-layer, 3-phase GATConv design.** All verified against `gnn_encoder.py`.

```
Phase 1 — Layers 1+2  (struct_mask: edge_attr <= 5)
  ─────────────────────────────────────────────────
  Edges: CALLS(0), READS(1), WRITES(2), EMITS(3), INHERITS(4), CONTAINS(5)
  8-head GATConv, concat=True  →  output dim = 256
  _head_dim = hidden_dim // heads = 256 // 8 = 32; concat → 8 × 32 = 256
  add_self_loops = True
  Layer 1 (conv1): [N, 11] → [N, 256]     (no residual — dims differ)
  Layer 2 (conv2): [N, 256] → [N, 256]    (residual from Layer 1)
  LayerNorm after Phase 1
  Purpose: propagate function-level properties DOWN into CFG children via
           CONTAINS; aggregate inter-function structural context.

Phase 2 — Layers 3+4+5  (cfg_mask: edge_attr == 6)
  ─────────────────────────────────────────────────
  Edges: CONTROL_FLOW(6) only
  1-head GATConv, concat=False  →  output dim = 256
  add_self_loops = False  ← CRITICAL: self-loops cancel directional signal
  Layer 3 (conv3):  first CF hop
  Layer 4 (conv3b): second CF hop — CALL signal reaches WRITE 2 hops away
  Layer 5 (conv3c): third CF hop  — ENTRY signal reaches WRITE 3 hops away
                                    covers full CEI: ENTRY→CHECK→CALL→TMP→WRITE
  All three layers use residual connections.
  LayerNorm after Phase 2
  Purpose: encode execution order within each function's CFG.

Phase 3 — Layers 6+7  (CONTAINS edges flipped + re-typed as REVERSE_CONTAINS(7))
  ─────────────────────────────────────────────────────────────────────────────
  Edges: REVERSE_CONTAINS(7) — runtime-generated, never on disk
  1-head GATConv, concat=False, add_self_loops=False
  Layer 6 (conv4):  first RC hop  — CFG → direct FUNCTION parent
  Layer 7 (conv4b): second RC hop — grandchild → grandparent propagation
  LayerNorm after Phase 3
  Purpose: aggregate Phase-2-enriched CFG embeddings UP into FUNCTION nodes.

JK Aggregation (_JKAttention)
  ─────────────────────────────
  Learned per-node attention over [Phase1_out, Phase2_out, Phase3_out].
  Prevents Phase 1 structural signal from being over-smoothed.
  self.jk.last_weights [3] registered buffer stores mean per-phase attention
  for diagnostics (verified at gnn_encoder.py:102).
  Output: [N, 256] weighted sum.

Edge Embedding
  ───────────────
  nn.Embedding(NUM_EDGE_TYPES=8, edge_emb_dim=64)
  Rows 0–6: on-disk edge types. Row 7: REVERSE_CONTAINS (runtime).
  All phases use the same embedding table.
```

---

---

## Part 1.5 — v7 Training State Snapshot (Epochs 5–8)

Current state at time of writing. All values from live training logs.

### Training Performance

| Metric | Value |
|--------|-------|
| Speed | ~1.90–1.98 batch/s |
| Time per epoch | ~35 minutes |
| F1-macro (best) | 0.2096 (Epoch 7, new best each epoch) |
| Loss trend | Decreasing: 0.1558 → 0.1536 (step-level) |
| Aux warmup | 8-epoch schedule, at 8/8 (0.2625) |
| VRAM utilization | **6.9 / 8.0 GiB (86%)** at batch_size=8 |

**VRAM note:** batch=8 uses 86% of available VRAM. batch=16 saturates at 7.9/8.0 GiB (near OOM).
This was fixed as Fix #28 in `trainer.py:186`. There is no headroom to increase micro-batch size.

### Per-Class F1 Snapshot (Epoch 8)

| Class | F1 | Status |
|-------|----|----|
| IntegerUO | ~0.49 | Best performer |
| TimestampDependency | ~0.24–0.27 | Moderate |
| Reentrancy | ~0.18–0.20 | Below expectation (BUG-H5 noise) |
| AccessControl | ~0.16–0.18 | Below expectation |
| MishandledException | ~0.13–0.15 | Weak |
| TransactionOrderDependence | ~0.14–0.15 | Weak |
| DenialOfService | ~0.019 | Structurally unlearnable — detached from loss (`dos_loss_weight=0.0`); predictions still made but random |

### Eye Loss Analysis

| Eye | Loss Range | Interpretation |
|-----|-----------|----------------|
| GNN eye | ~0.44 | Stable, learning structural patterns |
| Transformer eye | ~0.46 | Slightly higher; CodeBERT semantic processing |
| Fused eye | ~0.50–0.52 | **Highest** — fusion receiving impoverished GNN inputs due to structural blind spots |

The fused eye producing the highest loss is a **diagnostic canary** for the graph extensions.
ICFG and DEF_USE edges give the GNN richer cross-function paths, which in turn feeds the
fusion layer. If the fused eye loss does NOT decrease proportionally with other eyes in v8
training, this signals a deeper architectural problem in the fusion mechanism itself — not
just a data richness problem.

### JK Attention Weight Dynamics

| Phase | Epoch 5 | Epoch 8 | Trend |
|-------|---------|---------|-------|
| Phase 1 (structural) | 0.086 | 0.096 | Slowly increasing |
| Phase 2 (CFG) | ~0.33–0.35 | ~0.33–0.35 | Stable |
| Phase 3 (reverse-contains) | 0.561 | 0.572 | Slowly increasing |

Phase 3 dominates attention. Phase 2 stability suggests the CFG signal is being learned but
its information ceiling is bounded by intra-function-only `CONTROL_FLOW(6)` edges — which
is exactly what ICFG-Lite (Extension A) addresses.

### GNN Gradient Health

GNN gradient share: **61–68%** across epochs. The 2.5× LR multiplier for GNN parameters
is effectively preventing gradient collapse. This was a critical v7 fix and continues to work.

---

## Part 1.6 — Speed Optimization Status

Speed optimizations evaluated against live code. Status as of v3.

| Optimization | Status | Notes |
|-------------|--------|-------|
| S2: Fused AdamW (`fused=True`) | **APPLIED** | `trainer.py:1029` |
| S3: SDPA attention for CodeBERT | **APPLIED** | `transformer_encoder.py:131–143` — try flash_attention_2, except → sdpa |
| S1: Increase batch_size 8→16 | **BLOCKED** | Batch=16 saturates at 7.9/8.0 GiB (Fix #28). No VRAM headroom. |
| S4: Cap fusion tokens at 1024 | **Optional** | Fusion currently processes `[B, 2048, 768]`. Capping at 1024 saves ~1–3% (fusion is only 10–15% of batch time). Risk: information loss for long contracts. Defer to v8 ablation. |
| Flash Attention 2 | **Deferred** | Requires `nvcc` — not available in WSL2 (CUDA runtime only, no compiler). `nvcc` is at `/mnt/c/…/*.exe`, unusable from Linux. Revisit if cuda-toolkit-12-4 is installed. |

**What will NOT help — verified reasons:**

| Optimization | Why |
|-------------|-----|
| CUDA Graphs | GNN graph sizes vary per batch — dynamic shapes prevent capture |
| Gradient Checkpointing | VRAM is already at 86%; trades compute for memory — opposite of what is needed |
| More DataLoader workers | Data is RAM-cached; 4 workers is already sufficient |
| Mixed precision FP8 | Already using BF16 AMP; FP8 requires H100+ |
| `torch.compile` on CodeBERT | HuggingFace dynamic dispatch causes graph breaks; compile slows it down |
| Pre-compute frozen CodeBERT embeddings | LoRA is active (590K trainable params); embeddings change every step |

---

## Part 2 — Pre-Implementation Cleanup (Required Before v8)

Both items verified against live code. Neither affects training — safe to apply
while v7 training runs.

---

### 2.1 Dead code: `in_unchecked` result discarded

**File:** `graph_extractor.py`
**Location:** Line 699

```python
# CURRENT — dead assignment
in_unchecked = _compute_in_unchecked(obj)    # line 699: computed, result unused

# Return list at lines 714–727 does NOT include in_unchecked
return [
    float(type_id) / _MAX_TYPE_ID,
    visibility,
    uses_block_globals,
    view,
    payable,
    complexity,
    loc,
    return_ignored,
    call_target_typed,
    has_loop,
    external_call_count,
]
```

`_compute_in_unchecked()` itself (lines 318–344) is still fully implemented.
Residue from the v7 `in_unchecked` drop (BUG-L2).

**Fix:**
```python
# Delete line 699 entirely.
# Mark the function as deprecated:
def _compute_in_unchecked(func: Any) -> float:
    """DEPRECATED (v7 BUG-L2): in_unchecked feature dropped from schema.
    No longer called. Safe to delete in v8 cleanup.
    """
```

### 2.2 Stale docstring: `_build_node_features`

**File:** `graph_extractor.py`
**Location:** Lines 631–659

Docstring header reads:
```
"Compute the 12-dimensional feature vector (v4 schema) for one AST node."
```
and lists `[9] in_unchecked` in the feature layout. Function body correctly
returns 11 elements.

**Fix:** Update header:
```
"Compute the 11-dimensional feature vector (v7 schema) for one AST node."
```
Remove the `[9] in_unchecked` entry and renumber subsequent dims.

---

## Part 3 — Structural Blind Spots

---

### 3.1 Phase 2 stops at every call-site boundary

Phase 2 processes only `CONTROL_FLOW(6)` edges, which are intra-function.
Consider this reentrancy pattern:

```solidity
function withdraw(uint amount) external {
    require(balances[msg.sender] >= amount);         // CFG_NODE_CHECK
    (bool ok,) = msg.sender.call{value: amount}(""); // CFG_NODE_CALL
    balances[msg.sender] -= amount;                  // CFG_NODE_WRITE
}

receive() external payable {
    withdraw(1 ether);   // re-enters before WRITE completes
}
```

The graph correctly shows `CHECK → CALL → WRITE` via `CONTROL_FLOW` edges
inside `withdraw`. Phase 2's 3-hop pass encodes "call before write" signal
in that function.

What is invisible: the path
`withdraw.CALL → receive.ENTRY → receive.CALL → withdraw.WRITE`
— the actual execution cycle that constitutes reentrancy. The two functions
are connected only by a `CALLS(0)` edge at the FUNCTION level, which is a
Phase 1 structural edge carrying no execution-order signal. Phase 2 never
crosses it. The model therefore cannot observe the reentrancy cycle
topologically — it infers reentrancy risk from proxy scalars
(`external_call_count`, `return_ignored`) rather than from the execution
structure that causes the vulnerability.

Cases where the vulnerable WRITE is in a helper function called after the
external CALL are almost entirely invisible to the current model.

### 3.2 Value flow is scalar, not topological

`return_ignored [7]` and `external_call_count [10]` are handcrafted scalars
that approximate value-flow facts:

- `return_ignored` collapses all call-return flows to a single 0/1 per
  function. It cannot express "return value was used but under a wrong
  guard," or "return value flows into a state write through two assignments."
- For `IntegerOverflow`, the question is: does this arithmetic result flow
  into a state write without passing through a bounds check? This requires
  tracing a def-use chain — no such chain exists in the current graph.

### 3.3 Guard scope not encoded as edges

A `CFG_NODE_CHECK` node governs which statements execute. The CFG encodes
this implicitly via `CONTROL_FLOW` edges, but:

1. A WRITE 4 hops after a `require` requires 4 message-passing steps to
   receive the check's embedding, adding noise at each hop.
2. No edge type distinguishes "this WRITE is only reached when the require
   passed" from "this WRITE is on the require-failed branch."

The GNN must infer guard-scope from attention patterns over multi-hop paths.
This is learnable but represents unnecessary difficulty when guard scope is
an explicit, computable structural property.

---

## Part 4 — Proposed Extensions

Three extensions in priority order. All are additive — no existing node type
ID, edge type ID, or feature dimension is renamed, removed, or renumbered.

---

### Extension A — ICFG-Lite (Inter-Procedural CFG Edges)

**Priority:** 1
**Schema version:** v8
**New edge types:** `CALL_ENTRY(8)`, `RETURN_TO(9)`
**Requires full re-extraction:** Yes (~41,576 contracts)
**Requires model retrain from scratch:** Yes
**Estimated graph size change:** +15–30% edges

#### 4.A.1 What It Adds

Two new directed edge types connecting call-site CFG nodes to callee CFG bodies:

- `CALL_ENTRY(8)` — from a `CFG_NODE_CALL` to the first reachable CFG node
  of the internally-called function's body
- `RETURN_TO(9)` — from the last CFG node(s) of the callee body back to the
  post-call successor at the call site

With these edges in Phase 2's mask, the 3-hop directed message pass naturally
follows execution across function boundaries instead of stopping at the call site.

#### 4.A.2 Scope Constraints (All Mandatory)

| Constraint | Reason |
|-----------|--------|
| Internal calls only (`func.internal_calls`) | External callee CFGs are not in this graph |
| Same-contract only (callee in `node_map`) | Cross-contract requires a separate compilation unit |
| Depth = 1 (do not follow callee's own callees) | Prevents O(N²) edge explosion on deeply nested contracts |
| Recursion guard (skip if function pair already visited) | Prevents infinite cycles in mutually-recursive contracts |

#### 4.A.3 Required Extractor Refactor: Global CFG Node Map

**This is the most critical structural change.** The current extractor scopes
`cfg_node_map` per-function and discards it after each iteration
(`graph_extractor.py:989`):

```python
# CURRENT (scoped per function, map discarded each iteration)
for func in contract.functions:
    fn_idx = _add_node(func, NODE_TYPES["FUNCTION"])
    cfg_node_map: dict = {}   # ← created here, gone after this iteration
    contains_edges, control_flow_edges = _build_control_flow_edges(
        func, fn_idx, cfg_node_map, x_list, node_metadata,
        parent_features=x_list[fn_idx],
    )
    for src, dst in contains_edges:
        edges.append([src, dst]); edge_types.append(EDGE_TYPES["CONTAINS"])
    for src, dst in control_flow_edges:
        edges.append([src, dst]); edge_types.append(EDGE_TYPES["CONTROL_FLOW"])
```

Required change — accumulate into a global map across all function iterations:

```python
# v8: accumulate all CFG node mappings across all functions
global_cfg_node_map: dict = {}   # slither FunctionNode → graph_idx (ALL functions)

for func in contract.functions:
    fn_idx = _add_node(func, NODE_TYPES["FUNCTION"])
    func_cfg_map: dict = {}       # local map for this function only
    contains_edges, control_flow_edges = _build_control_flow_edges(
        func, fn_idx, func_cfg_map, x_list, node_metadata,
        parent_features=x_list[fn_idx],
    )
    global_cfg_node_map.update(func_cfg_map)   # ← accumulate globally
    for src, dst in contains_edges:
        edges.append([src, dst]); edge_types.append(EDGE_TYPES["CONTAINS"])
    for src, dst in control_flow_edges:
        edges.append([src, dst]); edge_types.append(EDGE_TYPES["CONTROL_FLOW"])

# Second pass: ICFG edges use the now-complete global_cfg_node_map
_add_icfg_edges(contract, node_map, global_cfg_node_map, edges, edge_types)
```

`_build_control_flow_edges` signature is unchanged — it still receives and
populates its own local map parameter. Only the caller changes.

#### 4.A.4 New Helper: `_add_icfg_edges()`

**Pre-implementation requirement:** Before full extraction, run on a
500-contract sample and confirm `node.internal_calls` is populated at the
CFG-node level for Slither 0.11.x. The existing extractor uses
`func.internal_calls` at the function level (line 1077); node-level access
is standard Slither API but is not exercised anywhere in the current
codebase. If node-level `internal_calls` is empty for Slither nodes, fall
back to cross-referencing IR ops (`isinstance(ir_op, InternalCall)`) to
identify call-site nodes. See Risk 7.6.

```python
def _add_icfg_edges(
    contract: Any,
    node_map: dict[str, int],          # canonical_name → graph_idx (declaration nodes)
    global_cfg_node_map: dict,         # slither FunctionNode → graph_idx (all CFG nodes)
    edges: list,
    edge_types: list,
) -> None:
    """
    Add CALL_ENTRY(8) and RETURN_TO(9) edges for internal same-contract calls.
    Depth-1 only. Recursion-guarded via visited_pairs set.

    node_map is keyed by canonical_name STRINGS (dict[str, int]).
    Always resolve callee keys via canonical_name — never use a Function
    object as a dict key (Function objects are not the key type).
    """
    visited_pairs: set[tuple[str, str]] = set()

    for func in contract.functions:
        caller_key = getattr(func, "canonical_name", None) or func.name

        for caller_node in (getattr(func, "nodes", None) or []):
            caller_cfg_idx = global_cfg_node_map.get(caller_node)
            if caller_cfg_idx is None:
                continue

            for called_func in (getattr(caller_node, "internal_calls", None) or []):
                # node_map is keyed by canonical_name STRINGS — resolve correctly.
                # Do NOT use called_func (a Function object) as the key.
                callee_key = getattr(called_func, "canonical_name", None) or called_func.name
                if callee_key not in node_map:
                    continue   # callee not in this contract's graph

                # Recursion guard: one ICFG edge pair per caller-callee function pair
                pair = (caller_key, callee_key)
                if pair in visited_pairs:
                    continue
                visited_pairs.add(pair)

                # CALL_ENTRY: call-site CFG node → callee's entry CFG node
                callee_entry_idx = _get_callee_entry_idx(called_func, global_cfg_node_map)
                if callee_entry_idx is not None:
                    edges.append([caller_cfg_idx, callee_entry_idx])
                    edge_types.append(EDGE_TYPES["CALL_ENTRY"])

                # RETURN_TO: callee's exit CFG node(s) → post-call successor
                successor_idx = _get_cfg_post_call_idx(caller_node, global_cfg_node_map)
                for exit_idx in _get_callee_exit_idxs(called_func, global_cfg_node_map):
                    if successor_idx is not None:
                        edges.append([exit_idx, successor_idx])
                        edge_types.append(EDGE_TYPES["RETURN_TO"])


def _get_callee_entry_idx(func: Any, global_cfg_node_map: dict) -> int | None:
    """Graph index of the first non-synthetic CFG node of func (by source order).
    Synthetic nodes (ENTRY_POINT etc.) with source line 0 sort first but are
    included in global_cfg_node_map and will be returned — this is correct
    behaviour, as the ENTRY_POINT is the true execution entry.
    """
    sorted_nodes = sorted(
        getattr(func, "nodes", None) or [],
        key=lambda n: (
            n.source_mapping.lines[0]
            if n.source_mapping and n.source_mapping.lines else 0,
            n.node_id,
        ),
    )
    for node in sorted_nodes:
        idx = global_cfg_node_map.get(node)
        if idx is not None:
            return idx
    return None


def _get_callee_exit_idxs(func: Any, global_cfg_node_map: dict) -> list[int]:
    """Graph indices of all CFG nodes in func that have no successors (sons == [])."""
    result = []
    for node in (getattr(func, "nodes", None) or []):
        if not (getattr(node, "sons", None) or []):
            idx = global_cfg_node_map.get(node)
            if idx is not None:
                result.append(idx)
    return result


def _get_cfg_post_call_idx(caller_node: Any, global_cfg_node_map: dict) -> int | None:
    """Graph index of the first CFG successor of the call-site node."""
    for successor in (getattr(caller_node, "sons", None) or []):
        idx = global_cfg_node_map.get(successor)
        if idx is not None:
            return idx
    return None
```

#### 4.A.5 GNN Encoder Changes

Extend Phase 2 `cfg_mask` to include the two new types:

```python
# gnn_encoder.py — CURRENT
cfg_mask = edge_attr == _CONTROL_FLOW   # type 6 only

# gnn_encoder.py — v8
_CALL_ENTRY = EDGE_TYPES["CALL_ENTRY"]   # 8
_RETURN_TO  = EDGE_TYPES["RETURN_TO"]    # 9

cfg_mask = (
    (edge_attr == _CONTROL_FLOW) |
    (edge_attr == _CALL_ENTRY)   |
    (edge_attr == _RETURN_TO)
)
```

`nn.Embedding` grows: `nn.Embedding(8, 64)` → `nn.Embedding(11, 64)` (v8
ships A+B together; see Part 6).
`add_self_loops=False` must remain in place for all Phase 2 convolutions.

No checkpoint compatibility. Model must be trained from scratch.

#### 4.A.6 Schema Constants

```python
# graph_schema.py — v8 additions
FEATURE_SCHEMA_VERSION = "v8"
NUM_EDGE_TYPES         = 11   # was 8; +3 for ICFG + DEF_USE (v8 ships A+B together)

EDGE_TYPES = {
    "CALLS":             0,   # unchanged
    "READS":             1,   # unchanged
    "WRITES":            2,   # unchanged
    "EMITS":             3,   # unchanged
    "INHERITS":          4,   # unchanged
    "CONTAINS":          5,   # unchanged
    "CONTROL_FLOW":      6,   # unchanged
    "REVERSE_CONTAINS":  7,   # unchanged (runtime-only)
    "CALL_ENTRY":        8,   # NEW — internal call site → callee entry CFG node
    "RETURN_TO":         9,   # NEW — callee exit CFG node → post-call successor
    "DEF_USE":           10,  # NEW — def site → use site (see Extension B)
}
```

The assertion guards in `graph_schema.py:391–402` enforce `NUM_EDGE_TYPES`
consistency at import time. Any mismatch causes an immediate `AssertionError`
on startup — confirmed present in v7 source.

#### 4.A.7 Expected Detection Impact

| Vulnerability class | Current evidence | Added evidence via ICFG | Expected change |
|---------------------|-----------------|------------------------|----------------|
| Reentrancy (cross-function) | CALL and WRITE in same CFG only | CALL → receive/fallback ENTRY → withdraw WRITE path visible in Phase 2 | Higher recall for re-entry where WRITE is in helper or callee |
| Authorization bypass (helper-mediated) | CALLS edge only at FUNCTION level | Execution path through helper reaches sensitive WRITE | New topological signal |
| DoS (call-in-loop via helper) | `has_loop` + `external_call_count` scalars | Loop body CFG → helper callee body connected | Better context for bounded loops |
| MishandledException (CEI-across-functions) | `return_ignored` scalar | RETURN_TO edge brings callee return into Phase 2 view | Partial improvement; DFG is the main fix |

---

### Extension B — Targeted Def-Use Graph (Intra-Function DFG)

**Priority:** 2
**Schema version:** v8 (ships with Extension A)
**New edge types:** `DEF_USE(10)`
**Requires full re-extraction:** Yes
**Requires model retrain from scratch:** Yes
**Estimated graph size change:** +20–40% edges additional

#### 4.B.1 What It Adds

One new directed edge type: `DEF_USE(10)` — from the CFG node where a targeted
variable is *defined* to every CFG node where that variable is *used*. This makes
value-flow between statements a first-class topological property instead of a
scalar feature approximation.

Scope is restricted to three high-value **definition** categories to avoid graph
explosion from a full DFG of every SSA temporary. Condition nodes are covered
as **use sites** (they receive DEF_USE edges from any tracked definition that
flows into them), not as definition sites — `Condition` IR ops have no `lvalue`
and cannot define variables:

| Category | IR types matched | Vulnerability relevance |
|----------|-----------------|------------------------|
| Call return values | `HighLevelCall`, `LowLevelCall`, `Send` | Ignored-return, mishandled exception |
| Arithmetic results | `Binary` where `op.type in _ARITHMETIC` | Integer overflow/underflow |
| State variable reads | `Assignment` RHS reading a `StateVariable` | How storage propagates after being read |

#### 4.B.2 Critical Slither API Requirements

**Do not use `v.is_storage`** — not a Slither API attribute.
`hasattr(v, "is_storage")` returns `False` for all Slither variable objects.
Correct idiom (already used in `_cfg_node_type()` at lines 459 and 467):

```python
from slither.core.variables.state_variable import StateVariable
isinstance(v, StateVariable)   # CORRECT
```

**Do not use `isinstance(ir_op, Binary)` alone** — Slither's `Binary` class
covers comparisons (`==`, `!=`, `<`, `>`), logical (`&&`, `||`), and bitwise
(`&`, `|`, `^`) in addition to arithmetic. Restrict to arithmetic:

```python
from slither.slithir.operations import Binary, BinaryType

_ARITHMETIC: set = {
    BinaryType.ADDITION,
    BinaryType.SUBTRACTION,
    BinaryType.MULTIPLICATION,
    BinaryType.DIVISION,
    BinaryType.MODULO,
    BinaryType.POWER,
}
if isinstance(ir_op, Binary) and ir_op.type in _ARITHMETIC:   # CORRECT
    ...
```

#### 4.B.3 New Helper: `_add_def_use_edges()`

Requires `global_cfg_node_map` — ships in the same v8 batch as Extension A.

```python
def _add_def_use_edges(
    func: Any,
    global_cfg_node_map: dict,   # slither FunctionNode → graph_idx
    edges: list,
    edge_types: list,
) -> None:
    """
    Add DEF_USE(10) edges for targeted variable categories within one function.
    Uses lval.name string keys — consistent with BUG-M1 fix in
    _compute_return_ignored() which uses the same name-based comparison.
    """
    from slither.core.variables.state_variable import StateVariable
    from slither.slithir.operations import (
        HighLevelCall, LowLevelCall, Send,
        Binary, BinaryType,
        Assignment,
    )

    _ARITHMETIC: set = {
        BinaryType.ADDITION,    BinaryType.SUBTRACTION,
        BinaryType.MULTIPLICATION, BinaryType.DIVISION,
        BinaryType.MODULO,     BinaryType.POWER,
    }

    # Pass 1: collect definition sites for targeted variable names
    # Key: lval.name (stable string, consistent with BUG-M1 fix)
    def_sites: dict[str, int] = {}

    for node in (getattr(func, "nodes", None) or []):
        node_idx = global_cfg_node_map.get(node)
        if node_idx is None:
            continue
        for ir_op in (getattr(node, "irs", None) or []):
            if not _is_targeted_definition(ir_op, StateVariable, HighLevelCall,
                                            LowLevelCall, Send, Binary,
                                            Assignment, _ARITHMETIC):
                continue
            lval = getattr(ir_op, "lvalue", None)
            if lval is not None:
                name = getattr(lval, "name", None)
                if name:
                    def_sites[name] = node_idx

    # Pass 2: find uses of defined variables and add DEF_USE edges.
    # Condition nodes are covered here as use sites — when a tracked
    # definition flows into a condition, it receives a DEF_USE edge.
    for node in (getattr(func, "nodes", None) or []):
        use_idx = global_cfg_node_map.get(node)
        if use_idx is None:
            continue
        for ir_op in (getattr(node, "irs", None) or []):
            for read_var in (getattr(ir_op, "read", None) or []):
                var_name = getattr(read_var, "name", None)
                if var_name and var_name in def_sites:
                    def_idx = def_sites[var_name]
                    if def_idx != use_idx:   # no self-loops
                        edges.append([def_idx, use_idx])
                        edge_types.append(EDGE_TYPES["DEF_USE"])


def _is_targeted_definition(
    ir_op, StateVariable, HighLevelCall, LowLevelCall, Send,
    Binary, Assignment, _ARITHMETIC
) -> bool:
    """True if this IR op defines a variable in one of the three targeted categories."""
    # Category 1: call return values
    if isinstance(ir_op, (HighLevelCall, LowLevelCall, Send)):
        return True
    # Category 2: arithmetic results — BinaryType filter is REQUIRED (not just Binary)
    if isinstance(ir_op, Binary) and ir_op.type in _ARITHMETIC:
        return True
    # Category 3: assignments reading a StateVariable
    # isinstance(v, StateVariable) is the CORRECT Slither idiom — NOT v.is_storage
    if isinstance(ir_op, Assignment):
        return any(
            isinstance(v, StateVariable)
            for v in (getattr(ir_op, "read", None) or [])
        )
    return False
```

Call site inside the main extraction loop (after accumulating `global_cfg_node_map`):

```python
for func in contract.functions:
    _add_def_use_edges(func, global_cfg_node_map, edges, edge_types)
for mod in contract.modifiers:
    _add_def_use_edges(mod, global_cfg_node_map, edges, edge_types)
```

#### 4.B.4 GNN Encoder Changes

Include `DEF_USE` in Phase 2 mask alongside `CONTROL_FLOW` and ICFG types:

```python
_DEF_USE = EDGE_TYPES["DEF_USE"]   # 10

cfg_mask = (
    (edge_attr == _CONTROL_FLOW) |
    (edge_attr == _CALL_ENTRY)   |
    (edge_attr == _RETURN_TO)    |
    (edge_attr == _DEF_USE)
)
```

`nn.Embedding` is `nn.Embedding(11, 64)` for v8 (covers all types 0–10).

#### 4.B.5 Scalar Feature Deprecation Path (v9 decision point)

After v8 training, run this ablation before deciding whether to deprecate
`return_ignored [7]`:

1. **Baseline:** v8 with DEF_USE edges + `return_ignored` feature
2. **Ablation:** zero out `return_ignored` in all graphs; retrain everything else equal
3. If per-class F1 delta for `MishandledException` and `UnusedReturn` is < 0.5pp,
   the feature is redundant — deprecate it in v9 (`NODE_FEATURE_DIM` drops to 10)
4. If delta ≥ 0.5pp, keep the scalar; it carries signal DEF_USE does not fully capture

Do not remove `return_ignored` before this ablation.

#### 4.B.6 Expected Detection Impact

| Vulnerability class | With Targeted DFG |
|---------------------|-------------------|
| MishandledException | Call result with no outgoing DEF_USE edges = structurally ignored, not just heuristically flagged by a scalar |
| IntegerOverflow/Underflow | Arithmetic result DEF_USE chain reaches STATE_VAR WRITE without passing through a CFG_NODE_CHECK |
| AccessControl (tx.origin) | `tx.origin` assignment flows through def-use chain to a `require` condition check |
| UnusedReturn (ERC20 `transfer`) | Same topology as MishandledException — replaces scalar with structural fact |

---

### Extension C — Control-Dependence Edges

**Priority:** 3
**Schema version:** v9
**New edge types:** `CONTROL_DEP(11)`
**Requires full re-extraction:** Yes
**Requires model retrain from scratch:** Yes
**Estimated graph size change:** +10–20% edges

#### 4.C.1 What It Adds

One new edge type: `CONTROL_DEP(11)` — from a `CFG_NODE_CHECK` node
directly to every statement whose execution it governs:

- A guarded WRITE is 1 hop from its guard via `CONTROL_DEP` instead of N
  hops via `CONTROL_FLOW` chains
- The edge type explicitly marks "guarded" vs "unguarded" relationships

#### 4.C.2 New Helper: `_add_control_dep_edges()`

The check_types set below uses `{SNT.IF, SNT.IFLOOP, SNT.STARTLOOP, SNT.THROW}`.
This is intentionally narrower than `_cfg_node_type()`'s set which also includes
`SNT.ENDLOOP`. ENDLOOP is a convergence join point — statements after it are
not control-dependent on the loop condition in the same sense as statements
inside the loop body. Including ENDLOOP would connect the loop condition to
all post-loop code, which is not meaningful control dependence.

```python
def _add_control_dep_edges(
    func: Any,
    global_cfg_node_map: dict,
    edges: list,
    edge_types: list,
) -> None:
    """
    Add CONTROL_DEP(11) edges from each CFG_NODE_CHECK to every CFG node
    that is control-dependent on it.

    Uses symmetric difference of reachable sets from each branch.
    Valid for small Solidity CFGs (typically 5–30 nodes per function).
    """
    try:
        from slither.core.cfg.node import NodeType as SNT
        _check_types = {SNT.IF, SNT.IFLOOP, SNT.STARTLOOP, SNT.THROW}
    except Exception:
        return

    def _cfg_reachable(start: Any) -> set:
        visited, stack = set(), [start]
        while stack:
            n = stack.pop()
            if n in visited:
                continue
            visited.add(n)
            stack.extend(getattr(n, "sons", None) or [])
        return visited

    for node in (getattr(func, "nodes", None) or []):
        if getattr(node, "type", None) not in _check_types:
            continue
        check_idx = global_cfg_node_map.get(node)
        if check_idx is None:
            continue
        sons = getattr(node, "sons", None) or []
        if len(sons) < 2:
            continue   # unconditional — no control dependence

        true_reach  = _cfg_reachable(sons[0])
        false_reach = _cfg_reachable(sons[1])
        dependent   = true_reach.symmetric_difference(false_reach)

        for dep_node in dependent:
            dep_idx = global_cfg_node_map.get(dep_node)
            if dep_idx is not None and dep_idx != check_idx:
                edges.append([check_idx, dep_idx])
                edge_types.append(EDGE_TYPES["CONTROL_DEP"])
```

#### 4.C.3 GNN Encoder Changes

`CONTROL_DEP` belongs in **Phase 1** (structural), not Phase 2 (execution-order).
Control dependence is a scope relationship — which statements are guarded by
a condition — not a sequential ordering of execution steps.

```python
# Phase 1 mask extended (v9)
_CONTROL_DEP = EDGE_TYPES["CONTROL_DEP"]   # 11
struct_mask  = (edge_attr <= _CONTAINS) | (edge_attr == _CONTROL_DEP)
```

`nn.Embedding` grows to `nn.Embedding(12, 64)`.

#### 4.C.4 Expected Detection Impact

| Vulnerability class | With Control-Dependence Edges |
|---------------------|-------------------------------|
| AccessControl | 1-hop CONTROL_DEP from `require(msg.sender == owner)` to every dependent WRITE; currently N hops |
| DoS (loop-guarded distribution) | Loop-condition CHECK directly connected to loop-body CALL nodes |
| TimestampDependency | CHECK reading `block.timestamp` directly connected to dependent branch — combined with `uses_block_globals [2]` feature |
| Reentrancy (guarded re-entry) | Distinguishes guarded CALL paths from unguarded ones — reduces false positives on correctly-guarded contracts |

---

## Part 5 — What Is Explicitly Rejected or Deferred

| Extension | Status | Reason |
|-----------|--------|--------|
| Full PDG | Deferred past v9 | Extensions A+B+C cover ~80% of PDG's practical value with far lower risk. Reconsider if v8+v9 validation F1 ≥ 0.82 and remaining FNs are value-flow-opaque |
| Full SSA / Φ-nodes | Deferred | Not in Slither's public API. Extension B's targeted DFG achieves the practical benefit |
| Full AST graph | Rejected | Adds thousands of nodes per contract, duplicating the Transformer eye (CodeBERT) that already handles syntactic patterns |
| Inter-contract call graph | Deferred | BCCC is predominantly single-file; cross-contract edges would be absent or incomplete for majority of training samples |
| Storage-slot-level heap graph | Deferred | Requires EVM bytecode analysis outside the current toolchain |

---

## Part 6 — Complete Schema Evolution

### v7 (current — training active)

```
FEATURE_SCHEMA_VERSION = "v7"
NODE_FEATURE_DIM = 11
NUM_NODE_TYPES   = 13
NUM_EDGE_TYPES   =  8

Phase 1 edges: 0,1,2,3,4,5       (struct_mask: edge_attr <= 5)
Phase 2 edges: 6                  (cfg_mask:    edge_attr == 6)
Phase 3 edges: 7 (runtime-only)   (contains_mask: edge_attr == 5, then flipped)

Effective training set: ~41,576 graph+token pairs (cache = 41,577 pairs)
```

### v8 (Extensions A + B — ship together)

```python
FEATURE_SCHEMA_VERSION  = "v8"
NODE_FEATURE_DIM        = 11      # UNCHANGED
NUM_NODE_TYPES          = 13      # UNCHANGED
NUM_EDGE_TYPES          = 11      # was 8: + CALL_ENTRY(8), RETURN_TO(9), DEF_USE(10)

EDGE_TYPES = {
    "CALLS":            0,   # UNCHANGED
    "READS":            1,   # UNCHANGED
    "WRITES":           2,   # UNCHANGED
    "EMITS":            3,   # UNCHANGED
    "INHERITS":         4,   # UNCHANGED
    "CONTAINS":         5,   # UNCHANGED
    "CONTROL_FLOW":     6,   # UNCHANGED
    "REVERSE_CONTAINS": 7,   # UNCHANGED (runtime-only, never on disk)
    "CALL_ENTRY":       8,   # NEW
    "RETURN_TO":        9,   # NEW
    "DEF_USE":          10,  # NEW
}

# GNNEncoder:
# nn.Embedding(8, 64)  →  nn.Embedding(11, 64)

Phase 1 edges: 0,1,2,3,4,5       (unchanged)
Phase 2 edges: 6,8,9,10          (+CALL_ENTRY, RETURN_TO, DEF_USE)
Phase 3 edges: 7 (runtime-only)   (unchanged)
```

### v9 (Extension C + conditional scalar deprecation)

```python
FEATURE_SCHEMA_VERSION  = "v9"
NUM_EDGE_TYPES          = 12   # + CONTROL_DEP(11)
EDGE_TYPES["CONTROL_DEP"] = 11

# IF ablation confirms return_ignored is redundant:
NODE_FEATURE_DIM        = 10   # drop return_ignored [7]; re-extraction required
# ELSE:
NODE_FEATURE_DIM        = 11   # keep unchanged

# GNNEncoder:
# nn.Embedding(11, 64)  →  nn.Embedding(12, 64)

Phase 1 edges: 0,1,2,3,4,5,11   (+CONTROL_DEP)
Phase 2 edges: 6,8,9,10          (unchanged from v8)
Phase 3 edges: 7 (runtime-only)  (unchanged)
```

---

## Part 7 — Risk Analysis

### 7.1 Graph size explosion (Medium)

ICFG and DFG edges both increase edge count. Primary mitigations:
depth-1 ICFG limit and 3-category DFG restriction.

**Mandatory validation gate before full re-extraction:**
Run on 2,000 randomly sampled contracts. Accept only if:
- P99 edge count per graph < 5,000
- No single graph exceeds 10,000 edges
- DataLoader batch fits in GPU memory at default batch size (8)

Tighten ICFG depth or DFG categories if gate fails.

### 7.2 Recursive call cycles (Low-Medium)

Solidity allows mutual recursion. The `visited_pairs` set in `_add_icfg_edges`
handles this: `pair=(A,B)` is marked before expanding B's callees, and `(B,A)`
prevents the reverse expansion. Result: one ICFG edge pair per caller-callee
function pair, regardless of how many call sites exist between them.

### 7.3 Phase 2 heterogeneity (Low-Medium)

Adding CALL_ENTRY, RETURN_TO, and DEF_USE to Phase 2 makes it heterogeneous.
The embedding table provides a distinct learned vector per type.

**Diagnostic after initial v8 training:**
Check `jk.last_weights` (registered buffer in `_JKAttention`).
- Phase 2 weight < 0.10 → heterogeneity collapsed its contribution
- Phase 2 weight > 0.80 → Phase 2 dominates; structural and RC phases ignored

If either condition holds: consider a dedicated Phase 2b sub-phase for ICFG
edges only.

### 7.4 Slither API stability (Low)

All new helpers use: `ir_op.read`, `ir_op.lvalue`, `Binary.type`,
`isinstance(v, StateVariable)`, `node.sons`. These are part of Slither's
stable public IR API, consistent across 0.9.x–0.11.x.

Any Slither version bump after v8 implementation must be validated on a
500-contract sample before full re-extraction.

### 7.5 Evaluation confounding (Managed)

Shipping A and B in the same v8 schema prevents perfect attribution from one
training run. Required ablation table:

| Run | Edges included | Purpose |
|-----|---------------|---------|
| v8-A | CALL_ENTRY + RETURN_TO only | Isolate ICFG contribution |
| v8-B | DEF_USE only | Isolate DFG contribution |
| v8-AB | All three new v8 types | Joint effect |
| v7 | Baseline | Reference |

All four runs against the same v8-extracted dataset. DEF_USE edges can be
zeroed out at training time by excluding edge type 10 from the mask without
re-extraction. Report per-class F1 delta from v7 for each run.

### 7.6 Node-level `internal_calls` API (Medium — NEW)

`_add_icfg_edges()` uses `caller_node.internal_calls` to find internally
called functions at the CFG-node level. The existing extractor uses
`func.internal_calls` at the function level (line 1077) — node-level access
is not exercised anywhere in the current codebase.

Slither's `Node` class does expose `internal_calls` per node in 0.9.x+,
but behaviour must be confirmed before full extraction.

**Validation protocol (mandatory — run before Phase 1):**
```python
# Quick sanity check on 10 contracts:
for contract in slither.contracts:
    for func in contract.functions:
        func_calls = set(f.canonical_name for f in func.internal_calls)
        node_calls = set()
        for node in func.nodes:
            for called in (node.internal_calls or []):
                node_calls.add(called.canonical_name)
        assert node_calls.issubset(func_calls), \
            f"node-level calls not subset of func-level: {node_calls - func_calls}"
```

If `node.internal_calls` is empty or unavailable, fall back to IR-level
detection:
```python
from slither.slithir.operations import InternalCall
called_funcs = [
    ir_op.function
    for ir_op in (getattr(caller_node, "irs", None) or [])
    if isinstance(ir_op, InternalCall)
]
```

### 7.7 `global_cfg_node_map` refactor has ripple effects (Medium)

The accumulation of `global_cfg_node_map` across all function iterations is the single
most critical code change in the entire proposal. It touches the main extraction loop and
changes the lifetime of a data structure that was previously per-function-scoped. While
conceptually simple (one dict, one `.update()` call, one second-pass function), any bug
here produces incorrect graphs silently for the entire dataset.

**Mandatory mitigation — structural comparison gate (Phase 1):**
Extract the same 2,000 sampled contracts with both the v7 extractor and the v8 extractor.
Verify:
- All v7 edges are present bit-for-bit in v8 output (CALLS, READS, WRITES, EMITS,
  INHERITS, CONTAINS, CONTROL_FLOW are additive — none may change or disappear)
- New edge types (CALL_ENTRY, RETURN_TO, DEF_USE) appear only in v8 output
- Node indices and node feature vectors are identical for all pre-existing nodes

This gate must pass before full re-extraction of 41,576 contracts.

### 7.8 Stale checkpoint naming (Low — Cosmetic)

`train.py` line 68: `--run-name` default is `"multilabel-v5-fresh"`. This is a leftover
from v5 and will produce confusing checkpoint filenames during v8 ablation runs.

**Fix before v8 training:** update the default to `"sentinel-v8"` (or pass `--run-name`
explicitly per ablation run: `v8-a`, `v8-b`, `v8-ab`). Not blocking for v7.

---

## Part 8 — Rollout Phases

### Phase 0 — Pre-implementation cleanup
**DONE (2026-05-18).** All items applied to `graph_extractor.py`. Non-breaking.

- [x] Delete line 699 in `graph_extractor.py`: `in_unchecked = _compute_in_unchecked(obj)` → replaced with discarded call + deprecation comment
- [x] Remove dead `in_unchecked = 0.0` default variable
- [x] Mark `_compute_in_unchecked()` as `# DEPRECATED (v7 BUG-L2) — safe to delete`
- [x] Update `_build_node_features` docstring: "11-dimensional feature vector (v7 schema)", feature table corrected

### Phase 1 — Extractor refactor + sample validation
**Start after v7 training completes.**

- [ ] Run 10-contract validation of `node.internal_calls` at node level (Risk 7.6)
- [ ] If node-level API fails: implement `InternalCall` IR fallback
- [ ] Refactor `extract_contract_graph()` to accumulate `global_cfg_node_map`
- [ ] Implement `_add_icfg_edges()` with recursion guard and depth-1 limit
- [ ] Implement `_add_def_use_edges()` with `isinstance(v, StateVariable)` and `BinaryType` filtering
- [ ] Run extractor on 2,000-contract sample
- [ ] Validate `edge_attr.max() <= 10` on all sample graphs
- [ ] Log per-edge-type counts — confirm CALL_ENTRY, RETURN_TO, DEF_USE are all non-zero
- [ ] Check P99 edge count against the 5,000-edge gate
- [ ] **Structural comparison gate:** extract the same 2,000 contracts with both v7 and v8 extractors; verify all v7 edges are present identically in v8 output (new edge types are additive — existing edges must be bit-for-bit unchanged). Any regression means the `global_cfg_node_map` refactor introduced a bug.

### Phase 2 — v8 full re-extraction (~41,576 contracts)

- [ ] Update `graph_schema.py`: `FEATURE_SCHEMA_VERSION="v8"`, `NUM_EDGE_TYPES=11`
- [ ] Cache invalidated automatically by the version bump (schema version in cache keys)
- [ ] Run full extraction
- [ ] Validate 100% of graphs have `edge_attr.max() <= 10`
- [ ] Verify non-zero per-edge-type counts across full dataset
- [ ] Re-run dataset statistics (mean/P99 node count, edge count, per-type distribution)

### Phase 3 — v8 training ablation

- [ ] Fix stale checkpoint name: update `train.py` `--run-name` default from `"multilabel-v5-fresh"` → `"sentinel-v8"` before starting any v8 run
- [ ] Apply speed optimization S4 (cap fusion tokens at 1024) if desired — single line in `fusion_layer.py`, measure actual batch/s impact
- [ ] Train v8-A (ICFG only), v8-B (DFG only), v8-AB (both)
- [ ] Compare per-class F1 delta from v7 baseline for all three runs
- [ ] Examine `jk.last_weights` per phase after each run — verify fused eye loss decreases relative to other eyes compared to v7 (canary metric)
- [ ] Run `return_ignored` scalar ablation on the best v8-AB checkpoint
- [ ] Review `early_stop_patience=30` after first v8 run — richer graphs may require longer patience before improvements stabilise

### Phase 4 — v9 (conditional on Phase 3 results)

- [ ] Implement `_add_control_dep_edges()`
- [ ] Decide on `return_ignored` deprecation based on ablation result — do NOT remove speculatively
- [ ] Update encoder: add CONTROL_DEP to Phase 1 `struct_mask`; grow embedding to 12 rows
- [ ] Update `graph_schema.py`: `FEATURE_SCHEMA_VERSION="v9"`, `NUM_EDGE_TYPES=12`
- [ ] Full re-extraction and retrain

---

## Part 9 — Architecture Preservation

All three extensions are purely additive. They preserve:

- The three-phase encoder design and its rationale
- JK attention aggregation and per-phase LayerNorm
- The offline/online single-source-of-truth principle (`graph_extractor.py`
  is the only authoritative implementation of the graph contract)
- The assertion guards in `graph_schema.py:391–402` that catch constant
  mismatches at import time
- The `REVERSE_CONTAINS` runtime-generation pattern (no change needed)
- The `add_self_loops=False` rule for all Phase 2 convolutions — CRITICAL,
  must not be changed when adding new Phase 2 edge types

The Phase 2 edge set changes from {6} to {6, 8, 9, 10} in v8.
The Phase 1 edge set changes from {0–5} to {0–5, 11} in v9.
Phase 3 is unchanged across both versions.

---

## Part 10 — Concerns, Caveats, and Open Questions

### 10.1 Label quality is the binding constraint

The graph extensions improve the model's ability to *observe* vulnerability patterns.
But two HIGH-severity label bugs set a hard F1 ceiling that no graph change can lift:

- **BUG-H4:** 48.2% of Timestamp=1 contracts have `uses_block_globals=0` — the model
  learns to associate "no timestamp signal" with timestamp vulnerability. Best-case
  Timestamp F1 is capped well below 1.0 regardless of graph richness.
- **BUG-H5:** ~14% of Reentrancy=1 contracts have no external calls — cannot be
  genuinely reentrancy-vulnerable by definition.

**These must be fixed as a parallel track** via `label_cleaner.py` (already built),
not sequenced after the graph extensions. Running the cleaner on v8 graphs as part
of Phase 2 is mandatory, not optional.

### 10.2 DoS class: 9-class reduction is BLOCKED by ZKML

The DenialOfService class (F1 ≈ 0.019) is structurally unlearnable: 98.1% of DoS
labels co-occur with Reentrancy; only 7 pure DoS samples exist in training.

Reducing to 9-class classification is **architecturally blocked**: the ZKML proxy MLP
is hardcoded to `Linear(128→64→32→10)` and the ZK circuit depends on that shape.
`trainer.py:280` explicitly documents this:
```
# NUM_CLASSES stays 10 (ZKML proxy MLP is hardcoded to 10 outputs — LOCKED).
```

Current approach (`dos_loss_weight=0.0`) is correct — DoS gradient is detached,
predictions are still made. Re-enable only if dedicated DoS data collection succeeds
and augments the class to ≥300 pure samples. Do not restructure the classifier.

### 10.3 Fused eye loss is a diagnostic canary

The fused eye consistently producing the highest loss (~0.50–0.52 vs ~0.44–0.46 for
GNN/TF eyes) is expected in v7 — the GNN receives impoverished intra-function-only
structural input, so the fusion layer cannot combine effectively.

**v8 success criterion:** after ICFG + DEF_USE edges are added, the fused eye loss
should decrease proportionally with or more than the GNN eye. If fused eye loss
remains the highest in v8 by a similar margin as v7, this signals a deeper architectural
issue in the `CrossAttentionFusion` mechanism itself — not just insufficient graph data.
Track this explicitly in the v8-AB training logs.

### 10.4 `return_ignored` deprecation must be evidence-based

The v9 decision point on deprecating `return_ignored [7]` requires a controlled ablation
on the best v8-AB checkpoint. **Do not remove it speculatively.** The scalar may carry
signal DEF_USE edges do not fully capture:
- Cross-function return value flow (DEF_USE is intra-function only in v8)
- Cases where IR is unavailable and the `-1.0` sentinel carries diagnostic value

The 0.5pp F1 threshold for MishandledException + UnusedReturn is the decision gate.

### 10.5 Early stopping patience may need adjustment for v8

`early_stop_patience=30` was calibrated for v7 training dynamics. With richer graphs
(denser edge connectivity, new Phase 2 types), the model may take longer to reach
stable optima per epoch. Review after the first complete v8 training run — if the
best F1 epoch is consistently near epoch 60–80 and still improving, consider
increasing patience to 40–50 for subsequent v8 ablation runs.

### 10.6 Speed optimization sequencing

S2 (fused AdamW) and S3 (SDPA) are already applied to the v7 training run and will
carry over to v8 automatically. S4 (cap fusion tokens at 1024) is optional and should
be measured in a controlled single-epoch comparison before committing. S1 (batch_size
increase) is permanently blocked by VRAM constraints on the RTX 3070 8 GB.

For Flash Attention 2: install `cuda-toolkit-12-4` via the NVIDIA apt repo for Ubuntu
24.04, then `pip install flash-attn --no-build-isolation`. This would provide ~15%
speedup on CodeBERT attention. Not blocking for v8.

---

## Appendix — Quick Reference: All Schema Constants

### v7 (current — training active)
```
FEATURE_SCHEMA_VERSION = "v7"
NODE_FEATURE_DIM = 11
NUM_NODE_TYPES   = 13
NUM_EDGE_TYPES   =  8

Phase 1 edges: 0,1,2,3,4,5       (struct_mask: edge_attr <= 5)
Phase 2 edges: 6                  (cfg_mask:    edge_attr == 6)
Phase 3 edges: 7 (runtime-only)   (contains_mask: edge_attr == 5, then flipped)

Effective training set: 41,576 graphs / 41,577 cache pairs
```

### v8 (proposed — Extensions A + B)
```
FEATURE_SCHEMA_VERSION = "v8"
NODE_FEATURE_DIM = 11             (unchanged)
NUM_NODE_TYPES   = 13             (unchanged)
NUM_EDGE_TYPES   = 11             (+3 new types: CALL_ENTRY, RETURN_TO, DEF_USE)

Phase 1 edges: 0,1,2,3,4,5       (unchanged)
Phase 2 edges: 6,8,9,10          (+CALL_ENTRY, RETURN_TO, DEF_USE)
Phase 3 edges: 7 (runtime-only)   (unchanged)

GNNEncoder: nn.Embedding(8,64) → nn.Embedding(11,64)
```

### v9 (proposed — Extension C)
```
FEATURE_SCHEMA_VERSION = "v9"
NODE_FEATURE_DIM = 11 or 10      (10 only if return_ignored ablation confirms redundancy)
NUM_NODE_TYPES   = 13             (unchanged)
NUM_EDGE_TYPES   = 12             (+CONTROL_DEP)

Phase 1 edges: 0,1,2,3,4,5,11   (+CONTROL_DEP)
Phase 2 edges: 6,8,9,10          (unchanged from v8)
Phase 3 edges: 7 (runtime-only)  (unchanged)

GNNEncoder: nn.Embedding(11,64) → nn.Embedding(12,64)
```
