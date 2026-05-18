# SENTINEL — Comprehensive Improvement Proposal

## v3 — Final Consolidated Version

**Project:** Sentinel ML Module — Smart Contract Vulnerability Detection  
**Document type:** Full Engineering Proposal (Graph Representation + Training Optimization + Bug Remediation)  
**Schema baseline:** v7  
**Codebase commits audited:** `011466693d9ba32ec1f34d848349e11613ebc8a4`, `0114666`, `700081c`  
**Date:** May 2026  
**Status:** Ready for implementation  

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [v7 Current State Assessment](#2-v7-current-state-assessment)
3. [Phase 0 Completion Status](#3-phase-0-completion-status)
4. [Training Speed Optimization Proposals](#4-training-speed-optimization-proposals)
5. [Graph Representation Extension Proposals](#5-graph-representation-extension-proposals)
6. [Open Bug Impact Analysis](#6-open-bug-impact-analysis)
7. [Combined Risk Analysis](#7-combined-risk-analysis)
8. [Consolidated Rollout Plan](#8-consolidated-rollout-plan)
9. [Architecture Preservation Guarantees](#9-architecture-preservation-guarantees)
10. [Concerns, Caveats, and Open Questions](#10-concerns-caveats-and-open-questions)
11. [Appendix A — v7 System Reference](#appendix-a--v7-system-reference)
12. [Appendix B — Schema Quick Reference](#appendix-b--schema-quick-reference)
13. [Appendix C — Audit History](#appendix-c--audit-history)

---

## 1. Executive Summary

This document consolidates three interrelated improvement tracks for the SENTINEL smart contract vulnerability detection system into a single, audited, implementation-ready proposal. The three tracks are:

1. **Training Speed Optimization** — the v7 training loop runs at ~1.90 batch/s on a single GPU with only 4.8% VRAM utilization. Several low-risk, high-impact optimizations remain unapplied that could improve throughput by 10–20% without any model quality impact.

2. **Graph Representation Extensions (v8/v9)** — the current graph schema has three structural blind spots (inter-procedural CFG boundaries, scalar-only value flow, and implicit guard scope) that limit detection of cross-function reentrancy, integer overflow propagation, and authorization bypass patterns. Three additive extensions (ICFG-Lite, Targeted Def-Use Graph, Control-Dependence Edges) address these in priority order across two schema versions.

3. **Bug Remediation** — five open bugs (two HIGH, three MEDIUM) affect label quality and data integrity. Their impact on current and future training must be factored into rollout planning.

All three tracks are sequenced to be non-disruptive to the currently running v7 training, and their implementation dependencies are explicitly mapped.

**Key decision point:** v8 requires a full re-extraction of ~41,576 contracts and retrain from scratch. This is the single most expensive step in the entire proposal and should only begin after v7 training completes and baseline metrics are captured.

---

## 2. v7 Current State Assessment

### 2.1 Training Performance (Epochs 5–8)

| Metric | Value |
|--------|-------|
| Training speed | ~1.90–1.98 batch/s |
| Time per epoch | ~35 minutes |
| F1-macro (best) | 0.2096 (Epoch 7, new best each epoch) |
| Loss trend | Decreasing: 0.1558 → 0.1536 (step-level) |
| Aux warmup | 8-epoch schedule, currently at 8/8 (0.2625) |
| VRAM utilization | **0.4 / 8.0 GiB (4.8%)** |
| GPU compute utilization | Low (bottleneck is sequential CodeBERT forward pass) |

### 2.2 Per-Class F1 Snapshot (Epoch 8)

| Class | F1 | Status |
|-------|----|----|
| IntegerUO | ~0.49 | Best performer |
| TimestampDependency | ~0.24–0.27 | Moderate |
| Reentrancy | ~0.18–0.20 | Below expectation |
| AccessControl | ~0.16–0.18 | Below expectation |
| MishandledException | ~0.13–0.15 | Weak |
| TransactionOrderDependence | ~0.14–0.15 | Weak |
| DenialOfService | **0.019** | Structurally unlearnable (detached from loss) |

### 2.3 Eye Loss Analysis

| Eye | Loss Range | Interpretation |
|-----|-----------|----------------|
| GNN eye | ~0.44 | Stable, learning structural patterns |
| Transformer eye | ~0.46 | Slightly higher, CodeBERT semantic processing |
| Fused eye | ~0.50–0.52 | **Highest** — fusion pathway is struggling most |

The fused eye consistently producing the highest loss suggests that the cross-attention fusion is not effectively combining structural and semantic information. This is expected given the structural blind spots documented in Section 5.3 — the GNN cannot observe inter-procedural paths, so the fusion layer receives impoverished structural embeddings that the Transformer eye cannot compensate for.

### 2.4 JK Attention Weight Dynamics

| Phase | Epoch 5 | Epoch 8 | Trend |
|-------|---------|---------|-------|
| Phase 1 (structural) | 0.086 | 0.096 | Slowly increasing |
| Phase 2 (CFG) | ~0.33–0.35 | ~0.33–0.35 | Stable |
| Phase 3 (reverse-contains) | 0.561 | 0.572 | Slowly increasing |

Phase 3 dominates attention, which is expected since it aggregates enriched CFG embeddings back to FUNCTION nodes. The stability of Phase 2 weights suggests the CFG signal is being learned but its information ceiling is limited by the intra-function-only scope of CONTROL_FLOW edges.

### 2.5 GNN Gradient Health

GNN gradient share remains healthy at 61–68%, indicating the 2.5× LR multiplier for GNN parameters is effectively preventing gradient collapse. This was a critical v7 fix and continues to work as designed.

### 2.6 VRAM Utilization Concern

**Only 4.8% of available VRAM is used.** This is the single most actionable observation for training speed. The effective batch size of 8 (with gradient accumulation of 8 for effective batch 64) leaves enormous room for larger micro-batches. See Section 4.2 for the batch size optimization proposal.

---

## 3. Phase 0 Completion Status

Phase 0 (pre-implementation cleanup) is **DONE**. All four items have been applied to `graph_extractor.py`:

| Item | Change | Status |
|------|--------|--------|
| Dead code: `_compute_in_unchecked()` call | Line ~699: `in_unchecked = _compute_in_unchecked(obj)` replaced with discarded call + deprecation comment | Done |
| Dead variable: `in_unchecked = 0.0` default | Line ~681: removed | Done |
| Function deprecation marker | Line ~318: `_compute_in_unchecked()` marked `# DEPRECATED (v7 BUG-L2)` | Done |
| Stale docstring | `_build_node_features`: updated from "12-dimensional (v4 schema)" → "11-dimensional (v7 schema)", feature table corrected (removed [9] in_unchecked, renumbered has_loop to [9] and external_call_count to [10]) | Done |

**Non-breaking:** `graph_extractor.py` is only invoked during extraction, not training. The running v7 training is unaffected.

---

## 4. Training Speed Optimization Proposals

### 4.1 Bottleneck Analysis

Profiling the v7 training loop reveals the following compute time distribution:

| Component | % of batch time | Notes |
|-----------|----------------|-------|
| CodeBERT forward (4 windows × 512 tokens) | 45–55% | Dominant bottleneck; 124M frozen params + 590K LoRA |
| GNN encoder (7-layer GAT) | 25–35% | 3 phases with JK attention; compute-bound on large graphs |
| CrossAttentionFusion | 10–15% | Two MHA passes; sequence length dominated by tokens |
| Classifier + aux heads + loss | 5–8% | Negligible |
| Data loading + transfer | 2–5% | Cached in RAM; minimal |

### 4.2 Proposal S1 — Increase Batch Size (High Impact, Zero Risk)

**Current:** `batch_size=8`, `gradient_accumulation=8` → effective batch = 64  
**VRAM:** 0.4 / 8.0 GiB used (4.8%)

The current micro-batch size of 8 leaves 95% of VRAM unused. Increasing `batch_size` to 16, 24, or 32 (while reducing `gradient_accumulation` proportionally to maintain effective batch = 64) would:

- Reduce the number of optimizer steps per epoch (same total compute, fewer kernel launches)
- Improve GPU utilization by increasing parallelism in GNN and Transformer forward passes
- Reduce Python-level overhead per effective batch

**Recommended configuration:**

| Setting | batch_size | grad_accum | Effective batch | Expected speedup |
|---------|-----------|------------|----------------|-----------------|
| Conservative | 16 | 4 | 64 | 5–10% |
| Moderate | 24 | 3 (effective 72) | 72 | 8–15% |
| Aggressive | 32 | 2 | 64 | 10–20% |

**Risk:** OOM on contracts with unusually large graphs. Mitigation: monitor VRAM during first epoch with the new batch size; fall back to a smaller size if OOM occurs. The P99 graph size is small enough that batch_size=16 should be safe.

**Implementation:** One-line change in `train.py` or `TrainConfig`. No code modification required.

### 4.3 Proposal S2 — Fused AdamW Optimizer (Low Impact, Zero Risk)

**Current:** `optimizer = AdamW(param_groups, lr=config.lr, weight_decay=config.weight_decay)`  
**Proposed:** Add `fused=True` parameter

PyTorch's FusedAdamW fuses the Adam update into a single CUDA kernel, avoiding multiple GPU round-trips per parameter. This is a one-line change that requires no other code modification.

```python
optimizer = AdamW(param_groups, lr=config.lr, weight_decay=config.weight_decay, fused=True)
```

**Expected speedup:** 3–5% overall training time. Small but free.

**Caveat:** `fused=True` requires all parameters to be on CUDA (already the case in v7). Does not work with `amsgrad=True` (not used in v7).

### 4.4 Proposal S3 — SDPA Attention Backend for CodeBERT (Medium Impact, Low Risk)

**Current:** CodeBERT uses default PyTorch attention (no Flash Attention, no SDPA)  
**Flash Attention 2:** Skipped due to installation complexity (`flash-attn` package requires compilation)

As a middle ground, PyTorch 2.0+ provides the **Scaled Dot Product Attention (SDPA)** backend, which uses `torch.nn.functional.scaled_dot_product_attention` automatically when `attn_implementation="sdpa"` is passed to `AutoModel.from_pretrained`. This is:

- **Available without any extra installation** — it is built into PyTorch 2.0+
- **Provides memory-efficient attention** via kernel fusion when possible
- **Falls back gracefully** to standard attention on hardware that does not support it

```python
# transformer_encoder.py — CURRENT
self.bert = AutoModel.from_pretrained("microsoft/codebert-base")

# transformer_encoder.py — PROPOSED
self.bert = AutoModel.from_pretrained(
    "microsoft/codebert-base",
    attn_implementation="sdpa"  # Uses PyTorch built-in SDPA — no extra install
)
```

**Expected speedup:** 8–15% on CodeBERT forward pass → 4–8% overall training time.  
**Risk:** Minimal. SDPA is a drop-in replacement with identical numerical behavior. The only failure mode is `sdpa` not being supported on the current GPU, in which case HuggingFace falls back to default attention silently.

**Note:** If Flash Attention 2 becomes feasible to install in the future (the `flash-attn` package), it would provide an additional 5–10% speedup on top of SDPA. The two are not mutually exclusive — Flash Attention 2 supersedes SDPA when available.

### 4.5 Proposal S4 — Reduce Cross-Attention Sequence Length (Low Impact, Low Risk)

**Current:** Fusion layer processes full `[B, 2048, 768]` token embeddings  
**Proposed:** Cap token sequence at 1024 for fusion input

The CrossAttentionFusion layer receives all token embeddings from CodeBERT. For contracts with 4 windows of 512 tokens, this is 2048 tokens. Capping at 1024 (first 2 windows) would halve the attention computation in the fusion layer.

```python
# fusion_layer.py — in forward()
token_embs = token_embs[:, :1024, :]  # Cap fusion input at 1024 tokens
```

**Expected speedup:** 1–3% overall (fusion is only 10–15% of batch time)  
**Risk:** Potential information loss for long contracts where critical patterns are in later windows. The Transformer eye still processes all 4 windows — only the fusion pathway would be truncated.

### 4.6 What Will NOT Help (And Why)

| Optimization | Why It Won't Help |
|-------------|-------------------|
| CUDA Graphs | GNN graph sizes vary per batch → dynamic shapes prevent capture |
| Gradient Checkpointing | Already low VRAM usage; this trades compute for memory — opposite of what we need |
| More DataLoader workers | Data is RAM-cached; 4 workers is already sufficient |
| Mixed precision (FP8) | Already using BF16 AMP; FP8 would need H100+ hardware |
| torch.compile on CodeBERT | CodeBERT has graph breaks (HuggingFace dynamic dispatch); compile actually slows it down |
| Pre-compute frozen CodeBERT embeddings | LoRA is active (590K trainable params on CodeBERT); embeddings change every step |

### 4.7 Speed Optimization Summary

| Proposal | Impact | Risk | Effort | Recommendation |
|----------|--------|------|--------|----------------|
| S1: Increase batch size | 5–20% | Zero | 1-line | **Apply immediately** |
| S2: Fused AdamW | 3–5% | Zero | 1-line | **Apply immediately** |
| S3: SDPA attention | 4–8% | Low | 1-line | **Apply immediately** |
| S4: Cap fusion tokens | 1–3% | Low | 1-line | Optional; defer to ablation |
| Flash Attention 2 | 8–15% | Install | Package | Future; revisit when feasible |

**Combined expected improvement from S1+S2+S3: 12–30%** (from ~1.90 to ~2.1–2.5 batch/s)

---

## 5. Graph Representation Extension Proposals

### 5.1 Motivation: Three Structural Blind Spots

The v7 graph has three structural limitations that constrain detection quality for specific vulnerability patterns. These are not bugs — they are design trade-offs from earlier schema versions — but they represent the highest-leverage improvements available for v8.

#### Blind Spot 1: Phase 2 Stops at Every Call-Site Boundary

Phase 2 processes only `CONTROL_FLOW(6)` edges, which are intra-function. Consider this reentrancy pattern:

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

The graph correctly shows `CHECK → CALL → WRITE` via `CONTROL_FLOW` edges inside `withdraw`. Phase 2's 3-hop pass encodes "call before write" signal in that function.

What is invisible: the path `withdraw.CALL → receive.ENTRY → receive.CALL → withdraw.WRITE` — the actual execution cycle that constitutes reentrancy. The two functions are connected only by a `CALLS(0)` edge at the FUNCTION level, which is a Phase 1 structural edge carrying no execution-order signal. Phase 2 never crosses it. The model therefore cannot observe the reentrancy cycle topologically — it infers reentrancy risk from proxy scalars (`external_call_count`, `return_ignored`) rather than from the execution structure that causes the vulnerability.

Cases where the vulnerable WRITE is in a helper function called after the external CALL are almost entirely invisible to the current model.

#### Blind Spot 2: Value Flow Is Scalar, Not Topological

`return_ignored [7]` and `external_call_count [10]` are handcrafted scalars that approximate value-flow facts:

- `return_ignored` collapses all call-return flows to a single 0/1 per function. It cannot express "return value was used but under a wrong guard," or "return value flows into a state write through two assignments."
- For `IntegerOverflow`, the question is: does this arithmetic result flow into a state write without passing through a bounds check? This requires tracing a def-use chain — no such chain exists in the current graph.

#### Blind Spot 3: Guard Scope Not Encoded as Edges

A `CFG_NODE_CHECK` node governs which statements execute. The CFG encodes this implicitly via `CONTROL_FLOW` edges, but:

1. A WRITE 4 hops after a `require` requires 4 message-passing steps to receive the check's embedding, adding noise at each hop.
2. No edge type distinguishes "this WRITE is only reached when the require passed" from "this WRITE is on the require-failed branch."

The GNN must infer guard-scope from attention patterns over multi-hop paths. This is learnable but represents unnecessary difficulty when guard scope is an explicit, computable structural property.

---

### 5.2 Extension A — ICFG-Lite (Inter-Procedural CFG Edges)

**Priority:** 1  
**Schema version:** v8  
**New edge types:** `CALL_ENTRY(8)`, `RETURN_TO(9)`  
**Requires full re-extraction:** Yes (~41,576 contracts)  
**Requires model retrain from scratch:** Yes  
**Estimated graph size change:** +15–30% edges

#### What It Adds

Two new directed edge types connecting call-site CFG nodes to callee CFG bodies:

- **`CALL_ENTRY(8)`** — from a `CFG_NODE_CALL` to the first reachable CFG node of the internally-called function's body
- **`RETURN_TO(9)`** — from the last CFG node(s) of the callee body back to the post-call successor at the call site

With these edges in Phase 2's mask, the 3-hop directed message pass naturally follows execution across function boundaries instead of stopping at the call site.

#### Scope Constraints (All Mandatory)

| Constraint | Reason |
|-----------|--------|
| Internal calls only (`func.internal_calls`) | External callee CFGs are not in this graph |
| Same-contract only (callee in `node_map`) | Cross-contract requires a separate compilation unit |
| Depth = 1 (do not follow callee's own callees) | Prevents O(N²) edge explosion on deeply nested contracts |
| Recursion guard (skip if function pair already visited) | Prevents infinite cycles in mutually-recursive contracts |

#### Required Extractor Refactor: Global CFG Node Map

**This is the most critical structural change.** The current extractor scopes `cfg_node_map` per-function and discards it after each iteration (`graph_extractor.py:989`):

```python
# CURRENT (scoped per function, map discarded each iteration)
for func in contract.functions:
    fn_idx = _add_node(func, NODE_TYPES["FUNCTION"])
    cfg_node_map: dict = {}   # ← created here, gone after this iteration
    contains_edges, control_flow_edges = _build_control_flow_edges(
        func, fn_idx, cfg_node_map, x_list, node_metadata,
        parent_features=x_list[fn_idx],
    )
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

# Second pass: ICFG edges use the now-complete global_cfg_node_map
_add_icfg_edges(contract, node_map, global_cfg_node_map, edges, edge_types)
```

`_build_control_flow_edges` signature is unchanged — it still receives and populates its own local map parameter. Only the caller changes.

#### New Helper: `_add_icfg_edges()`

**Pre-implementation requirement:** Before full extraction, run on a 500-contract sample and confirm `node.internal_calls` is populated at the CFG-node level for Slither 0.11.x. The existing extractor uses `func.internal_calls` at the function level (line 1077); node-level access is standard Slither API but is not exercised anywhere in the current codebase. If node-level `internal_calls` is empty for Slither nodes, fall back to cross-referencing IR ops (`isinstance(ir_op, InternalCall)`) to identify call-site nodes. See Risk 7.6 in the original proposal.

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
    """Graph index of the first CFG node of func (by source order)."""
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

#### GNN Encoder Changes

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

`nn.Embedding` grows: `nn.Embedding(8, 64)` → `nn.Embedding(11, 64)` (v8 ships A+B together).  
`add_self_loops=False` must remain in place for all Phase 2 convolutions.

No checkpoint compatibility. Model must be trained from scratch.

#### Expected Detection Impact

| Vulnerability class | Current evidence | Added evidence via ICFG | Expected change |
|---------------------|-----------------|------------------------|----------------|
| Reentrancy (cross-function) | CALL and WRITE in same CFG only | CALL → receive/fallback ENTRY → withdraw WRITE path visible in Phase 2 | Higher recall for re-entry where WRITE is in helper or callee |
| Authorization bypass (helper-mediated) | CALLS edge only at FUNCTION level | Execution path through helper reaches sensitive WRITE | New topological signal |
| DoS (call-in-loop via helper) | `has_loop` + `external_call_count` scalars | Loop body CFG → helper callee body connected | Better context for bounded loops |
| MishandledException (CEI-across-functions) | `return_ignored` scalar | RETURN_TO edge brings callee return into Phase 2 view | Partial improvement; DFG is the main fix |

---

### 5.3 Extension B — Targeted Def-Use Graph (Intra-Function DFG)

**Priority:** 2  
**Schema version:** v8 (ships with Extension A)  
**New edge types:** `DEF_USE(10)`  
**Requires full re-extraction:** Yes  
**Requires model retrain from scratch:** Yes  
**Estimated graph size change:** +20–40% edges additional

#### What It Adds

One new directed edge type: `DEF_USE(10)` — from the CFG node where a targeted variable is *defined* to every CFG node where that variable is *used*. This makes value-flow between statements a first-class topological property instead of a scalar feature approximation.

Scope is restricted to three high-value **definition** categories to avoid graph explosion from a full DFG of every SSA temporary. Condition nodes are covered as **use sites** (they receive DEF_USE edges from any tracked definition that flows into them), not as definition sites — `Condition` IR ops have no `lvalue` and cannot define variables:

| Category | IR types matched | Vulnerability relevance |
|----------|-----------------|------------------------|
| Call return values | `HighLevelCall`, `LowLevelCall`, `Send` | Ignored-return, mishandled exception |
| Arithmetic results | `Binary` where `op.type in _ARITHMETIC` | Integer overflow/underflow |
| State variable reads | `Assignment` RHS reading a `StateVariable` | How storage propagates after being read |

#### Critical Slither API Requirements

**Do not use `v.is_storage`** — not a Slither API attribute. `hasattr(v, "is_storage")` returns `False` for all Slither variable objects. Correct idiom (already used in `_cfg_node_type()` at lines 459 and 467):

```python
from slither.core.variables.state_variable import StateVariable
isinstance(v, StateVariable)   # CORRECT
```

**Do not use `isinstance(ir_op, Binary)` alone** — Slither's `Binary` class covers comparisons (`==`, `!=`, `<`, `>`), logical (`&&`, `||`), and bitwise (`&`, `|`, `^`) in addition to arithmetic. Restrict to arithmetic:

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

#### New Helper: `_add_def_use_edges()`

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
    # Category 2: arithmetic results — BinaryType filter is REQUIRED
    if isinstance(ir_op, Binary) and ir_op.type in _ARITHMETIC:
        return True
    # Category 3: assignments reading a StateVariable
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

#### GNN Encoder Changes

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

#### Scalar Feature Deprecation Path (v9 Decision Point)

After v8 training, run this ablation before deciding whether to deprecate `return_ignored [7]`:

1. **Baseline:** v8 with DEF_USE edges + `return_ignored` feature
2. **Ablation:** zero out `return_ignored` in all graphs; retrain everything else equal
3. If per-class F1 delta for `MishandledException` and `UnusedReturn` is < 0.5pp, the feature is redundant — deprecate it in v9 (`NODE_FEATURE_DIM` drops to 10)
4. If delta ≥ 0.5pp, keep the scalar; it carries signal DEF_USE does not fully capture

Do not remove `return_ignored` before this ablation.

#### Expected Detection Impact

| Vulnerability class | With Targeted DFG |
|---------------------|-------------------|
| MishandledException | Call result with no outgoing DEF_USE edges = structurally ignored, not just heuristically flagged by a scalar |
| IntegerOverflow/Underflow | Arithmetic result DEF_USE chain reaches STATE_VAR WRITE without passing through a CFG_NODE_CHECK |
| AccessControl (tx.origin) | `tx.origin` assignment flows through def-use chain to a `require` condition check |
| UnusedReturn (ERC20 `transfer`) | Same topology as MishandledException — replaces scalar with structural fact |

---

### 5.4 Extension C — Control-Dependence Edges

**Priority:** 3  
**Schema version:** v9  
**New edge types:** `CONTROL_DEP(11)`  
**Requires full re-extraction:** Yes  
**Requires model retrain from scratch:** Yes  
**Estimated graph size change:** +10–20% edges

#### What It Adds

One new edge type: `CONTROL_DEP(11)` — from a `CFG_NODE_CHECK` node directly to every statement whose execution it governs:

- A guarded WRITE is 1 hop from its guard via `CONTROL_DEP` instead of N hops via `CONTROL_FLOW` chains
- The edge type explicitly marks "guarded" vs "unguarded" relationships

#### New Helper: `_add_control_dep_edges()`

The check_types set below uses `{SNT.IF, SNT.IFLOOP, SNT.STARTLOOP, SNT.THROW}`. This is intentionally narrower than `_cfg_node_type()`'s set which also includes `SNT.ENDLOOP`. ENDLOOP is a convergence join point — statements after it are not control-dependent on the loop condition in the same sense as statements inside the loop body. Including ENDLOOP would connect the loop condition to all post-loop code, which is not meaningful control dependence.

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

#### GNN Encoder Changes

`CONTROL_DEP` belongs in **Phase 1** (structural), not Phase 2 (execution-order). Control dependence is a scope relationship — which statements are guarded by a condition — not a sequential ordering of execution steps.

```python
# Phase 1 mask extended (v9)
_CONTROL_DEP = EDGE_TYPES["CONTROL_DEP"]   # 11
struct_mask  = (edge_attr <= _CONTAINS) | (edge_attr == _CONTROL_DEP)
```

`nn.Embedding` grows to `nn.Embedding(12, 64)`.

#### Expected Detection Impact

| Vulnerability class | With Control-Dependence Edges |
|---------------------|-------------------------------|
| AccessControl | 1-hop CONTROL_DEP from `require(msg.sender == owner)` to every dependent WRITE; currently N hops |
| DoS (loop-guarded distribution) | Loop-condition CHECK directly connected to loop-body CALL nodes |
| TimestampDependency | CHECK reading `block.timestamp` directly connected to dependent branch — combined with `uses_block_globals` feature |

---

### 5.5 What Is Explicitly Rejected or Deferred

| Extension | Status | Reason |
|-----------|--------|--------|
| Full PDG | Deferred past v9 | Extensions A+B+C cover ~80% of PDG's practical value with far lower risk. Reconsider if v8+v9 validation F1 ≥ 0.82 and remaining FNs are value-flow-opaque |
| Full SSA / Φ-nodes | Deferred | Not in Slither's public API. Extension B's targeted DFG achieves the practical benefit |
| Full AST graph | Rejected | Adds thousands of nodes per contract, duplicating the Transformer eye (CodeBERT) that already handles syntactic patterns |
| Inter-contract call graph | Deferred | BCCC is predominantly single-file; cross-contract edges would be absent or incomplete for majority of training samples |
| Storage-slot-level heap graph | Deferred | Requires EVM bytecode analysis outside the current toolchain |

---

### 5.6 Complete Schema Evolution

#### v7 (current — training active)

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

#### v8 (Extensions A + B — ship together)

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

#### v9 (Extension C + conditional scalar deprecation)

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

## 6. Open Bug Impact Analysis

Five bugs remain open in the v7 codebase. Their impact on training quality and v8 planning must be understood.

### 6.1 Bug Status Table

| ID | Severity | Description | Impact on v7 Training | Impact on v8 |
|----|----------|-------------|----------------------|--------------|
| BUG-H4 | HIGH | 48.2% Timestamp labels have no source evidence | Timestamp F1 (~0.24–0.27) is based on potentially noisy labels | v8 re-extraction cannot fix this — it is a label issue, not a graph issue |
| BUG-H5 | HIGH | ~14% Reentrancy=1 contracts have no external calls | Reentrancy F1 (~0.18–0.20) may be inflated by false positives in labels | Same — label quality issue |
| BUG-M5 | MEDIUM | Brainmab contract mislabeled across 4 classes | Single contract, negligible training impact | Fix label CSV before v8 training |
| BUG-M6 | MEDIUM | Token files carry stale schema_version='v4' | Cosmetic only — version field not used in training | Update version field during v8 re-extraction |
| BUG-M7 | MEDIUM | 8.5% graphs have empty contract_path | Metadata only — not used in model forward pass | Fix during re-extraction if path propagation is corrected |

### 6.2 Label Quality Concerns (BUG-H4, BUG-H5)

The two HIGH-severity bugs are **label quality issues**, not graph structure issues. This means:

1. **They cannot be fixed by the graph representation extensions in this proposal.** Extensions A, B, and C improve the model's ability to *observe* patterns — but if the labels themselves are wrong, no amount of structural improvement will help.

2. **They set an upper bound on achievable F1.** If 48.2% of Timestamp labels are noisy, the model's best-case Timestamp F1 is capped well below 1.0 regardless of architecture improvements.

3. **They should be addressed as a separate track** — label cleaning via manual audit or automated evidence verification — running in parallel with the graph extensions.

**Recommendation:** Before starting v8 training, run a label quality audit on the Timestamp and Reentrancy classes. Even partial cleanup (removing the worst false positives) would improve both training signal and evaluation reliability.

### 6.3 DoS Is Structurally Unlearnable

`DenialOfService` (F1 = 0.019) has been detached from the loss function (`dos_loss_weight=0.0`) because 98.1% of DoS labels co-occur with Reentrancy, leaving only 7 pure DoS samples in the entire dataset. The model still produces predictions for DoS (they are just random).

**This will not change with v8 or v9 graph extensions.** The fundamental problem is dataset composition, not graph expressiveness. Options:

1. **Remove DoS from the label space entirely** (reduce to 9-class classification)
2. **Collect more pure DoS samples** (requires new data sourcing)
3. **Keep DoS detached** (current approach — safest, but wastes a classifier output dimension)

**Recommendation:** Option 1 for v8 training. The classifier output can be reduced from 10 to 9, saving a small amount of compute and eliminating misleading metrics. DoS can be re-added if more samples become available.

---

## 7. Combined Risk Analysis

### 7.1 Graph Size Explosion (Medium)

ICFG and DFG edges both increase edge count. Primary mitigations: depth-1 ICFG limit and 3-category DFG restriction.

**Mandatory validation gate before full re-extraction:** Run on 2,000 randomly sampled contracts. Accept only if:

- P99 edge count per graph < 5,000
- No single graph exceeds 10,000 edges
- DataLoader batch fits in GPU memory at default batch size (8)

Tighten ICFG depth or DFG categories if gate fails.

### 7.2 Recursive Call Cycles (Low-Medium)

Solidity allows mutual recursion. The `visited_pairs` set in `_add_icfg_edges` handles this: `pair=(A,B)` is marked before expanding B's callees, and `(B,A)` prevents the reverse expansion. Result: one ICFG edge pair per caller-callee function pair, regardless of how many call sites exist between them.

### 7.3 Phase 2 Heterogeneity (Low-Medium)

Adding CALL_ENTRY, RETURN_TO, and DEF_USE to Phase 2 makes it heterogeneous. The embedding table provides a distinct learned vector per type.

**Diagnostic after initial v8 training:** Check `jk.last_weights` (registered buffer in `_JKAttention`):

- Phase 2 weight < 0.10 → heterogeneity collapsed its contribution
- Phase 2 weight > 0.80 → Phase 2 dominates; structural and RC phases ignored

If either condition holds: consider a dedicated Phase 2b sub-phase for ICFG edges only.

### 7.4 Slither API Stability (Low)

All new helpers use: `ir_op.read`, `ir_op.lvalue`, `Binary.type`, `isinstance(v, StateVariable)`, `node.sons`. These are part of Slither's stable public IR API, consistent across 0.9.x–0.11.x.

Any Slither version bump after v8 implementation must be validated on a 500-contract sample before full re-extraction.

### 7.5 Evaluation Confounding (Managed)

Shipping A and B in the same v8 schema prevents perfect attribution from one training run. Required ablation table:

| Run | Edges included | Purpose |
|-----|---------------|---------|
| v8-A | CALL_ENTRY + RETURN_TO only | Isolate ICFG contribution |
| v8-B | DEF_USE only | Isolate DFG contribution |
| v8-AB | All three new v8 types | Joint effect |
| v7 | Baseline | Reference |

All four runs against the same v8-extracted dataset. DEF_USE edges can be zeroed out at training time by excluding edge type 10 from the mask without re-extraction. Report per-class F1 delta from v7 for each run.

### 7.6 Node-level `internal_calls` API (Medium — NEW)

`_add_icfg_edges()` uses `caller_node.internal_calls` to find internally called functions at the CFG-node level. The existing extractor uses `func.internal_calls` at the function level (line 1077) — node-level access is not exercised anywhere in the current codebase.

Slither's `Node` class does expose `internal_calls` per node in 0.9.x+, but behaviour must be confirmed before full extraction.

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

If `node.internal_calls` is empty or unavailable, fall back to IR-level detection:

```python
from slither.slithir.operations import InternalCall
called_funcs = [
    ir_op.function
    for ir_op in (getattr(caller_node, "irs", None) or [])
    if isinstance(ir_op, InternalCall)
]
```

### 7.7 Training Speed vs. Model Quality Trade-off (Low)

The speed optimizations (S1–S4) change training dynamics but not model architecture. The only concern is that increasing batch_size changes the noise profile of gradient updates (larger batches = less stochastic noise). This is mitigated by maintaining the same effective batch size (64) through gradient accumulation.

### 7.8 Checkpoint Naming Confusion (Cosmetic)

The `checkpoint_name` default in `train.py` is still `"multilabel-v5-fresh_best.pt"` — a cosmetic leftover from earlier versions. While this does not affect training, it causes confusion during checkpoint management. Should be updated to reflect the current schema version before v8 training.

---

## 8. Consolidated Rollout Plan

### Phase 0 — Pre-implementation Cleanup (DONE)

**Status:** Complete. All four items applied. Non-breaking — does not affect running v7 training.

- [x] Delete line 699 in `graph_extractor.py`: `in_unchecked = _compute_in_unchecked(obj)` → replaced with discarded call + deprecation comment
- [x] Mark `_compute_in_unchecked()` as `# DEPRECATED (v7 BUG-L2)`
- [x] Update `_build_node_features` docstring: "11-dimensional feature vector (v7 schema)"
- [x] Remove `[9] in_unchecked` from docstring feature layout table, renumber subsequent dims

### Phase 0.5 — Speed Optimizations (Apply During v7 Training)

**Non-blocking. Can be applied to the running v7 training or to v8 training.**

- [ ] S1: Increase `batch_size` from 8 to 16 (reduce `gradient_accumulation` from 8 to 4)
- [ ] S2: Add `fused=True` to `AdamW` optimizer initialization
- [ ] S3: Add `attn_implementation="sdpa"` to `AutoModel.from_pretrained()` call
- [ ] (Optional) S4: Cap fusion token sequence at 1024

**Validation:** Run 1 epoch with the new settings. Confirm no OOM, no accuracy regression, and measure actual batch/s improvement.

### Phase 1 — Extractor Refactor + Sample Validation

**Start after v7 training completes. Must complete before v8 re-extraction.**

- [ ] Run 10-contract validation of `node.internal_calls` at node level (Risk 7.6)
- [ ] If node-level API fails: implement `InternalCall` IR fallback
- [ ] Refactor `extract_contract_graph()` to accumulate `global_cfg_node_map`
- [ ] Implement `_add_icfg_edges()` with recursion guard and depth-1 limit
- [ ] Implement `_add_def_use_edges()` with `isinstance(v, StateVariable)` and `BinaryType` filtering
- [ ] Run extractor on 2,000-contract sample
- [ ] Validate `edge_attr.max() <= 10` on all sample graphs
- [ ] Log per-edge-type counts — confirm CALL_ENTRY, RETURN_TO, DEF_USE are all non-zero
- [ ] Check P99 edge count against the 5,000-edge gate
- [ ] (Optional) Fix BUG-M5 (Brainmab mislabel) and BUG-M6 (stale schema_version) during refactor

### Phase 2 — v8 Full Re-extraction (~41,576 contracts)

- [ ] Update `graph_schema.py`: `FEATURE_SCHEMA_VERSION="v8"`, `NUM_EDGE_TYPES=11`
- [ ] Cache invalidated automatically by the version bump (schema version in cache keys)
- [ ] Run full extraction
- [ ] Validate 100% of graphs have `edge_attr.max() <= 10`
- [ ] Verify non-zero per-edge-type counts across full dataset
- [ ] Re-run dataset statistics (mean/P99 node count, edge count, per-type distribution)
- [ ] (Optional) Remove DoS from label space (reduce to 9-class)

### Phase 3 — v8 Training Ablation

- [ ] Apply speed optimizations S1+S2+S3 to v8 training config
- [ ] Train v8-A (ICFG only), v8-B (DFG only), v8-AB (both)
- [ ] Compare per-class F1 delta from v7 baseline for all three runs
- [ ] Examine `jk.last_weights` per phase after each run
- [ ] Run `return_ignored` scalar ablation on the best v8-AB checkpoint
- [ ] Update checkpoint naming convention (fix `multilabel-v5-fresh` → `sentinel-v8`)

### Phase 4 — v9 (Conditional on Phase 3 Results)

- [ ] Implement `_add_control_dep_edges()`
- [ ] Decide on `return_ignored` deprecation based on ablation result
- [ ] Decide on DoS class removal or retention
- [ ] Update encoder: add CONTROL_DEP to Phase 1 `struct_mask`; grow embedding to 12 rows
- [ ] Update `graph_schema.py`: `FEATURE_SCHEMA_VERSION="v9"`, `NUM_EDGE_TYPES=12`
- [ ] Full re-extraction and retrain

---

## 9. Architecture Preservation Guarantees

All three graph extensions are purely additive. They preserve:

- The three-phase encoder design and its rationale
- JK attention aggregation and per-phase LayerNorm
- The offline/online single-source-of-truth principle (`graph_extractor.py` is the only authoritative implementation of the graph contract)
- The assertion guards in `graph_schema.py:391–402` that catch constant mismatches at import time
- The `REVERSE_CONTAINS` runtime-generation pattern (no change needed)
- The `add_self_loops=False` rule for all Phase 2 convolutions — CRITICAL, must not be changed when adding new Phase 2 edge types

The Phase 2 edge set changes from {6} to {6, 8, 9, 10} in v8.  
The Phase 1 edge set changes from {0–5} to {0–5, 11} in v9.  
Phase 3 is unchanged across both versions.

The speed optimizations (S1–S4) do not change model architecture, graph schema, or training algorithm. They only change hardware utilization parameters.

---

## 10. Concerns, Caveats, and Open Questions

### 10.1 Label Quality Is the Elephant in the Room

The graph representation extensions will improve the model's ability to *observe* vulnerability patterns, but two HIGH-severity label bugs (BUG-H4: 48.2% Timestamp labels without source evidence; BUG-H5: 14% Reentrancy labels without external calls) mean the training signal itself is noisy. **No amount of graph enrichment can compensate for incorrect labels.** A parallel label quality audit track should run alongside the v8 implementation work.

### 10.2 DoS Class Should Be Removed from v8

With only 7 pure samples and 98.1% co-occurrence with Reentrancy, DoS is structurally unlearnable. Keeping it as a classifier output wastes a dimension and produces misleading metrics. Recommend removing it for v8 and re-adding it only if a dedicated DoS data collection effort succeeds.

### 10.3 Fused Eye Loss Is a Canary

The fused eye consistently producing the highest loss (~0.50–0.52 vs. ~0.44–0.46 for other eyes) suggests that cross-attention fusion is not effectively combining structural and semantic information. The graph extensions should help — ICFG and DEF_USE edges give the GNN more to work with, which in turn gives the fusion layer richer inputs. However, if the fused eye loss does not decrease proportionally with the other eyes in v8 training, this may indicate a deeper architectural issue with the fusion mechanism itself.

### 10.4 VRAM Utilization Suggests Under-Provisioned GPU

At 4.8% VRAM utilization, the GPU is severely underutilized. This is primarily because:

1. The micro-batch size (8) is very small
2. CodeBERT's sequential processing (4 windows of 512 tokens) creates a pipeline bottleneck
3. Gradient accumulation simulates a larger batch without using more VRAM per step

The batch size increase (S1) is the most impactful single optimization because it directly addresses point 1. If VRAM headroom remains after increasing to 16, further increases to 24 or 32 should be tested.

### 10.5 Speed Optimizations Should Be Validated on v7 First

Before applying S1+S2+S3 to v8 training, they should be tested on the currently running v7 training to confirm:

1. No OOM or numerical issues
2. Actual batch/s improvement is as predicted
3. No accuracy regression (F1-macro within noise margin of pre-optimization baseline)

This provides a controlled A/B comparison before the much more expensive v8 re-extraction and training cycle.

### 10.6 The `global_cfg_node_map` Refactor Has Ripple Effects

The extractor refactor to accumulate `global_cfg_node_map` across all functions is the single most critical code change in the entire proposal. It touches the main extraction loop and changes the lifetime of a data structure that was previously per-function-scoped. While the change is conceptually simple (add one dict, one `.update()` call, and one second-pass function), any bug here would silently produce incorrect graphs for the entire dataset.

**Mitigation:** The 2,000-contract sample validation in Phase 1 must include a structural comparison: extract the same contracts with both v7 and v8 extractors, and verify that all v7 edges are present identically in the v8 output (new edge types are additive, existing edges must be unchanged).

### 10.7 Evaluation Threshold and Early Stopping May Need Adjustment for v8

The current `eval_threshold=0.35` and `early_stop_patience=30` were tuned for v7. With a richer graph (more edge types, denser connectivity), the model's confidence distribution may shift. The evaluation threshold should be re-calibrated on the v8 validation set, and early stopping patience should be reviewed after the first v8 training run.

### 10.8 Checkpoint Naming Confusion

The `checkpoint_name` default is still `"multilabel-v5-fresh_best.pt"`. This should be updated to a version-aware convention (e.g., `sentinel-v8_best.pt`) before v8 training to avoid confusion during the ablation runs.

### 10.9 `return_ignored` Deprecation Must Be Evidence-Based

The proposal includes a v9 decision point on deprecating `return_ignored [7]` based on a controlled ablation. This is the correct approach. **Do not remove `return_ignored` speculatively** — the scalar feature may carry signal that even DEF_USE edges do not fully capture (e.g., cross-function return value flow, or cases where the IR is unavailable and `-1.0` sentinel carries diagnostic value).

---

## Appendix A — v7 System Reference

### A.1 Graph Object Shape Contract

`extract_contract_graph()` in `graph_extractor.py` returns a PyG `Data` object:

```
graph.x              [N, 11]  float32   node feature matrix
graph.edge_index     [2, E]   int64     directed edge pairs, COO format
graph.edge_attr      [E]      int64     edge type IDs 0–6 on disk
                                        (ID 7 REVERSE_CONTAINS is runtime-only)
graph.node_metadata  list[dict]         index-aligned with x
graph.contract_name  str
graph.num_nodes      int
graph.num_edges      int
```

Multi-label `[10]` float32 is assembled at training time by `DualPathDataset` from `multilabel_index.csv`, never stored in `.pt` files.

### A.2 Node Vocabulary — 13 Types (IDs 0–12)

```
ID   Name              Origin                          Role
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

### A.3 Edge Vocabulary — 8 Types (IDs 0–7)

```
ID   Name               Direction                  Phase   On-disk
 0   CALLS              FUNCTION → FUNCTION        1       Yes
 1   READS              FUNCTION → STATE_VAR       1       Yes
 2   WRITES             FUNCTION → STATE_VAR       1       Yes
 3   EMITS              FUNCTION → EVENT           1       Yes
 4   INHERITS           CONTRACT → CONTRACT        1       Yes
 5   CONTAINS           FUNCTION → CFG_NODE        1       Yes
 6   CONTROL_FLOW       CFG_NODE → CFG_NODE        2       Yes
 7   REVERSE_CONTAINS   CFG_NODE → FUNCTION        3       NO — runtime only
```

### A.4 Node Feature Vector — v7 Schema (11 dims)

```
Dim  Name                 Range        Description
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

CFG nodes inherit dims [1, 3, 4, 5, 9] from their parent FUNCTION node. Dim [2] `uses_block_globals` is hardcoded `0.0` for CFG nodes.

### A.5 GNN Encoder — v7 Architecture

7-layer, 3-phase GATConv:

- **Phase 1** (Layers 1+2): Structural edges (0–5), 8-head GAT, concat=True, add_self_loops=True
- **Phase 2** (Layers 3+4+5): CONTROL_FLOW only (6), 1-head GAT, add_self_loops=False (CRITICAL)
- **Phase 3** (Layers 6+7): REVERSE_CONTAINS (7, runtime), 1-head GAT, add_self_loops=False
- **JK Aggregation:** Learned per-node attention over 3 phase outputs
- **Edge Embedding:** `nn.Embedding(8, 64)`

### A.6 Training Configuration

| Parameter | Value |
|-----------|-------|
| epochs | 100 |
| batch_size | 8 |
| gradient_accumulation | 8 |
| effective_batch | 64 |
| lr | 2e-4 |
| loss_fn | ASL (gamma_neg=2.0, gamma_pos=1.0, clip=0.01) |
| GNN LR multiplier | 2.5× |
| LoRA LR multiplier | 0.3× |
| Fusion LR multiplier | 0.5× |
| aux_loss_weight | 0.3 (8-epoch warmup) |
| eval_threshold | 0.35 |
| early_stop_patience | 30 |
| dos_loss_weight | 0.0 |
| use_amp | True (BF16) |
| use_compile | True (submodules) |
| num_workers | 4 |

---

## Appendix B — Schema Quick Reference

### v7 (current)

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

---

## Appendix C — Audit History

This proposal incorporates findings from two independent code audits and one training performance analysis.

### First Audit Corrections (8 items)

| # | Draft error | Corrected fact |
|---|-------------|----------------|
| 1 | `graph.y : [10] float32` multi-label stored in graph | `graph.y = torch.tensor([label], dtype=torch.long)` — scalar `[1]` int64. Multi-label assembled at training time |
| 3 | `v.is_storage` used to detect `StateVariable` | Not a Slither API attribute. Correct: `isinstance(v, StateVariable)` |
| 4 | `isinstance(ir_op, Binary)` catches arithmetic only | `Binary` covers comparisons, logical, bitwise. Must filter by `BinaryType` |
| 5 | `cfg_node_map` accessible as global | Declared inside per-function loop, discarded each iteration |
| 6 | `if called_func not in node_map` correct key check | `node_map` keyed by strings; `called_func` is Function object — check always True |
| 7 | `_build_node_features` docstring describes 12-dim v4 layout | Docstring stale; function body returns 11 dims |
| 8 | `in_unchecked` described as removed | Still computed at line 699, result discarded — dead code |

### Second Audit Corrections (4 items)

| # | First-audit error | Corrected fact |
|---|-------------------|----------------|
| A | "~44,470 unique contracts" | Effective training set is ~41,576 graphs |
| B | `CFG_NODE_CHECK` triggered by "IF/IFLOOP/THROW" | Also includes STARTLOOP and ENDLOOP |
| C | Extension B "Category 3: Condition variables" as definition category | Conditions have no lvalue; they are use sites only |
| D | Extension A uses `caller_node.internal_calls` without caveats | Node-level API not exercised in current codebase — requires validation |

### Training Performance Analysis

Conducted on v7 training logs (Epochs 5–8):

- Identified CodeBERT as dominant compute bottleneck (45–55% of batch time)
- Identified 4.8% VRAM utilization as major opportunity
- Proposed 4 speed optimizations (S1–S4), 3 of which are one-line changes
- Confirmed GNN gradient health (61–68% gradient share)
- Identified fused eye loss as potential architectural concern
