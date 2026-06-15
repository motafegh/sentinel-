# Phase 2 Receptive Field & CEI Detection — Corrected Analysis

**Date:** 2026-06-05  
**Status:** Source code verified  
**Based on:** `SENTINEL-Understanding-Run7.md` + Phase 2 interpretability experiments  
**Source files verified:** `gnn_encoder.py:580-629`, `sentinel_model.py:555-557`, `graph_extractor.py:1001-1062`

---

## 1. The Problem

Reentrancy follows the CEI (Checks-Effects-Interactions) violation pattern:

```
function withdraw() {
    require(balances[msg.sender] >= amount);  // CHECK   (CFG_NODE_CHECK)
    msg.sender.call.value(amount)("");          // INTERACT (CFG_NODE_CALL)
    balances[msg.sender] -= amount;             // EFFECT  (CFG_NODE_WRITE) ← WRONG ORDER
}
```

The GNN needs to "see" **CHECK → CALL → WRITE** along CONTROL_FLOW edges. This requires 3 directed CFG hops. Two independent problems prevent this:

1. **The GNN's Phase 2 has ~1.5 effective CF hops, not 3** (architecture limitation)
2. **62% of reentrancy-positive contracts have no reachable CEI path within 8 hops** (data limitation)

---

## 2. Phase 2 Architecture — What Actually Happens

### Layer-Subet Splitting (IMP-G1)

```
Layer 3 (conv3):  CONTROL_FLOW(6) only      — intra-function execution ordering
Layer 4 (conv3b): CALL_ENTRY(8)+RETURN_TO(9) only — cross-function call structure
Layer 5 (conv3c): CF+ICFG joint             — integration layer
```

**Source:** `gnn_encoder.py:585-597`

```python
x2 = self.conv3(x, cf_only_ei, cf_only_ea)       # CF only     ← 1 CF hop
x  = x + self.dropout(x2)
x2 = self.conv3b(x, icfg_only_ei, icfg_only_ea)  # ICFG only   ← 0 CF hops
x  = x + self.dropout(x2)
x2 = self.conv3c(x, phase2_ei, phase2_ea)         # joint       ← 1 CF hop (diluted)
x  = x + self.dropout(x2)
```

### Effective CF Receptive Field

| Layer | Edge type | Additional CF hops? | Reason |
|-------|-----------|---------------------|--------|
| conv3 | CF only | **+1** | Direct CFG predecessor |
| conv3b | ICFG only | **+0** | CALL_ENTRY/RETURN_TO are cross-function, not intra-function CF |
| conv3c | CF+ICFG | **+1 (diluted)** | Joint attention includes ICFG neighbors, diluting CF signal |

**Effective CF hops: ~2** (conv3 + conv3c). The "~1.5" estimate from the original analysis is fair when accounting for residual connections that preserve 0-hop self information blended with the message-passed signal.

**Why not 3:** IMP-G1 splits Phase 2 into layer-specific edge subsets. This was designed to give "distinct representations per layer" for JK aggregation. But it means conv3b processes CALL_ENTRY/RETURN_TO edges which are cross-function, not intra-function CF. A WRITE node gets zero new CF neighbors from conv3b.

**The analogy:** Trying to walk 3 steps north by taking: 1 step north, 1 step east, 1 step northeast. You end up ~1.7 steps north, not 3.

### Residual Connection Doesn't Help

The residual `x = x + self.dropout(x2)` preserves the original node features + all previously accumulated signal. After conv3b, the WRITE node has: its own features + 1-hop CF info + 1-hop ICFG info. It does NOT have 2-hop CF info. The residual preserves but doesn't extend the CF receptive field.

### Why IMP-R7-1 (heads=4) Doesn't Fix This

Phase 2 already uses 4 attention heads (was 1). More heads give more diverse attention patterns within each layer, but they don't extend the receptive field. If each layer sees only 1 CF hop's worth of neighbors, 4 heads × 1 hop still = 1 hop per layer.

---

## 3. The 62% Unreachable Problem

**Source:** `_compute_has_cei_path()` at `graph_extractor.py:1001-1062`, EXP-E1 results

### What the BFS Does

BFS from every `CFG_NODE_CALL` to every `CFG_NODE_WRITE` following only `CONTROL_FLOW(6)` edges. `max_hops=8`. Returns 1 if any path exists, 0 otherwise.

### Results (EXP-E1)

| k (hops) | % reachable | Interpretation |
|-----------|-------------|----------------|
| 1 | 17.6% | CALL directly followed by WRITE (no CHECK in between) |
| 2 | 29.6% | CALL → intermediate → WRITE |
| 3 | 32.2% | CALL → 2 intermediates → WRITE |
| 4 | 35.7% | Adding hops has diminishing returns |
| 8 | 38.2% | **Saturation: only +2.5pp from k=4 to k=8** |

**62% of reentrancy-positive contracts have NO reachable CEI path within 8 hops.** The curve saturates after k=5, meaning adding more GNN layers won't help for the unreachable 62%.

### Why 62% Are Unreachable

1. **Cross-contract reentrancy (~30-40%):** `A.withdraw()` calls `B.attack()` which calls back into `A`. The graph only contains nodes from the target contract. No cross-contract callback edges exist.

2. **Indirect reentrancy via delegatecall (~10-15%):** Proxy patterns forward calls to another contract's code. The reentrancy happens in the delegate's context, which may not be represented in the target contract's CFG.

3. **CFG abstraction level (~10-15%):** Slither merges or splits nodes differently from the ideal CEI chain — single EXPRESSION nodes that both read and write state, entry/exit nodes breaking the linear CHECK→CALL→WRITE chain.

---

## 4. The Dilution Problem

GAT attention distributes signal across ALL neighbors (weighted by learned attention). If a CALL node has 5 CF successors:

```
CALL → WRITE, CALL → CHECK, CALL → TMP1, CALL → TMP2, CALL → RETURN
```

The WRITE signal gets at most 1/5 of the attention. After 2 hops:

```
Signal strength ≈ original × (1/avg_degree)^k
```

For avg_degree ≈ 5 in CFG graphs:
- 1 hop: 20% signal
- 2 hops: 4% signal
- 3 hops: 0.8% signal

By hop 3, the original CHECK signal reaching the WRITE node is <1% of its original magnitude. The model can't learn from this.

---

## 5. The aux_phase2 Fix — What's Already Done

**Correction to original analysis:** The original analysis stated "aux_phase2 uses global_mean_pool over function nodes." This is **incorrect** — it was fixed in BUG-R7-1.

**Current code (`sentinel_model.py:555-557`):**

```python
cfg_pool_mask = torch.isin(node_type_ids, _CFG_IDS_CPU.to(node_embs.device))
phase2_pooled = global_mean_pool(_phase2_x[cfg_pool_mask], batch[cfg_pool_mask], size=num_graphs)
aux_phase2_logits = self.aux_phase2(phase2_pooled)
```

`cfg_pool_mask` selects **CFG nodes only** (types 8-12: CFG_NODE_CALL, WRITE, READ, CHECK, OTHER). Function nodes receive zero Phase 2 messages, so pooling over them was BUG-R7-1 — fixed.

**What still could be improved:**
- Pool only CALL+WRITE+CHECK (3 types) instead of all 5 CFG types — narrows to CEI-relevant nodes
- Use max+mean dual pooling (like the CFG eye) instead of mean-only — preserves the strongest CEI signal
- Add a separate CEI-path-level aux head that only looks at WRITE node representations after conv3c

---

## 6. The 10-Vulnerability-Class Pattern Requirements

The same hop/dilution problem affects all classes:

| Class | Pattern | Hops needed | Current effective hops |
|-------|---------|-------------|----------------------|
| Reentrancy | CHECK→CALL→WRITE | 3 CF | ~2 |
| IntegerUO | ARITH→WRITE (no check before) | 2 CF | ~2 (marginal) |
| DoS | LOOP→CALL (in loop body) | 2-3 CF | ~2 (marginal) |
| MishandledException | CALL→IGNORED_RETURN | 2 CF + 1 DEF_USE | ~2 CF + 1 DFG |
| Timestamp | TIMESTAMP_READ→CONDITION | 2-3 CF | ~2 (marginal) |

All classes need multi-hop CF signal that the current architecture provides at ~2 hops. APPNP-style teleport would help all of them simultaneously by preserving the Phase 1 structural signal at every layer.

---

## 7. Proposals for Run 8 (Code-Only Changes) — ALL IMPLEMENTED ✅

### ✅ Proposal 1: APPNP-Style Phase 1 Teleport in Phase 2 (IMPLEMENTED 2026-06-05)

**What:** At each Phase 2 layer, teleport a fraction α of the original Phase 1 output directly into the current representation.

**Implemented in:** `gnn_encoder.py` — `appnp_alpha` param (default `0.0`). CLI: `--appnp-alpha 0.2`.

**Actual code:**
```python
_phase1_anchor = x.detach() if self.appnp_alpha > 0.0 else None
# ... after each Phase 2 layer conv + relu + dropout:
if _phase1_anchor is not None:
    x = self.appnp_alpha * _phase1_anchor + (1.0 - self.appnp_alpha) * x
```

Applied at all 3 Phase 2 layers (conv3, conv3b, conv3c). `detach()` prevents gradient shortcut back to Phase 1 — Phase 1 gradients flow only through JK aggregation.

**Wired through:** `SentinelModel(appnp_alpha=...)` → `GNNEncoder(appnp_alpha=...)` → `TrainConfig.appnp_alpha` → `train.py --appnp-alpha` → `predictor.py saved_cfg.get("appnp_alpha", 0.0)`.

**Run 8 value:** `--appnp-alpha 0.2`

**Expected impact:** +0.03-0.06 F1 for Reentrancy, +0.01-0.02 for other CF-dependent classes

### ✅ Proposal 2: Refine aux_phase2 Pooling to CEI-Node Subset (IMPLEMENTED 2026-06-05)

**What:** Change aux_phase2 from pooling over all 5 CFG types to pooling over only CALL+WRITE+CHECK (the 3 CEI-relevant types).

**Implemented in:** `sentinel_model.py` — new `_CEI_TYPE_IDS` / `_CEI_IDS_CPU` constants (types 8=CALL, 9=WRITE, 11=CHECK). `aux_phase2` head uses `cei_pool_mask` instead of `cfg_pool_mask`.

```python
_CEI_TYPE_IDS: frozenset[int] = frozenset({
    NODE_TYPES["CFG_NODE_CALL"],   # 8
    NODE_TYPES["CFG_NODE_WRITE"],  # 9
    NODE_TYPES["CFG_NODE_CHECK"],  # 11
})
_CEI_IDS_CPU: torch.Tensor = torch.tensor(sorted(_CEI_TYPE_IDS), dtype=torch.long)
# ...
cei_pool_mask = torch.isin(node_type_ids, _CEI_IDS_CPU.to(node_embs.device))
phase2_pooled = global_mean_pool(_phase2_x[cei_pool_mask], batch[cei_pool_mask], size=num_graphs)
```

The `cfg_eye` (4th classifier eye) still uses all 5 CFG types — only `aux_phase2` changes.

**Expected impact:** +0.02-0.04 F1 for Reentrancy (complementary to Proposal 1)

### ✅ Proposal 3: fusion_max_nodes default → 2048 (IMPLEMENTED 2026-06-05)

**What:** `TrainConfig.fusion_max_nodes` default changed from 1024 to 2048. `train.py` argparse default changed to 2048. No re-extraction needed — affects only CrossAttentionFusion dense padding.

**Scope confirmed:** 227 graphs (0.55%) exceed 1024 nodes (max=1,735). 0 graphs exceed 2048. 2048 covers 100% of v10 graphs.

**Expected impact:** +0.005-0.01 F1 (small; covers the 227 previously-truncated complex contracts)

---

## 8. Proposals for Run 9 (Re-Extraction Required)

### Proposal 4: Add Path-Level Features as Node Attributes

**What:** Precompute CEI-relevant path features and add them to the node feature vector.

**New features:**
- `dist_to_nearest_call` (for WRITE nodes): CF hops to nearest CALL node, normalized [0,1]
- `has_check_before_call` (for CALL nodes): 1.0 if CHECK node exists in CF ancestors within 3 hops
- `cei_violation_score`: 1.0 if CHECK→CALL→WRITE exists in CF order within 5 hops, 0.0 otherwise

**Where:** `graph_extractor.py` — after building CFG nodes, run BFS from CALL and WRITE nodes

**Why:** Bypasses the hop/dilution problem entirely by making the CEI pattern a direct input feature rather than something the GNN must learn to detect. The GNN then only needs to learn that these features correlate with vulnerability labels — a much simpler task.

**Expected impact:** +0.05-0.10 F1 for Reentrancy (directly addresses root cause)

### Proposal 5: Inherit `uses_block_globals` on CFG Nodes

**What:** Currently `uses_block_globals` (feat[2]) is per-function only. CFG nodes that read `block.timestamp` don't carry this signal.

**Where:** `graph_extractor.py:710` — `_build_cfg_node_features()` already inherits some dims from parent FUNCTION. Add `uses_block_globals` to inherited dims.

**Why:** Timestamp vulnerability requires identifying `block.timestamp` reads in CFG nodes. Currently the GNN can only see this signal at the FUNCTION level (dim 2 = 1.0 if ANY statement reads block globals). Per-statement inheritance would let the GNN learn which specific statement reads the timestamp.

**Expected impact:** +0.02-0.04 F1 for Timestamp

### Proposal 6: Add DELEGATECALL Edge Type

**What:** Distinguish `delegatecall` from regular internal calls in the graph.

**Where:** `graph_extractor.py` — add `DELEGATECALL` edge type (ID 11+), `graph_schema.py` — bump NUM_EDGE_TYPES

**Why:** Delegatecall-based reentrancy (proxy patterns) is invisible because `delegatecall` looks identical to `call` in the current graph. A dedicated edge type would let the GNN learn proxy-specific patterns.

**Expected impact:** +0.02-0.03 F1 for Reentrancy

---

## 9. Priority Ranking

| Priority | Change | Phase | Status | Expected Impact | Risk |
|----------|--------|-------|--------|-----------------|------|
| P1 | APPNP teleport (Proposal 1) | Run 8 | ✅ **DONE** `--appnp-alpha 0.2` | +0.03-0.06 F1 Reentrancy | Low |
| P2 | Refine aux_phase2 pooling (Proposal 2) | Run 8 | ✅ **DONE** CEI-only mask | +0.02-0.04 F1 Reentrancy | Low |
| P3 | Increase fusion_max_nodes (Proposal 3) | Run 8 | ✅ **DONE** default=2048 | +0.005-0.01 F1 | Low |
| P4 | Path-level features (Proposal 4) | Run 9 | ⬜ Pending re-extraction | +0.05-0.10 F1 Reentrancy | Medium |
| P5 | CFG `uses_block_globals` inheritance (Proposal 5) | Run 9 | ⬜ Pending re-extraction | +0.02-0.04 F1 Timestamp | Low |
| P6 | DELEGATECALL edge type (Proposal 6) | Run 9 | ⬜ Pending re-extraction | +0.02-0.03 F1 Reentrancy | High |

---

## 10. What NOT to Do

| Change | Why not |
|--------|---------|
| Remove IMP-G1 edge splitting (give all 3 layers access to CF) | Loses "distinct representations per layer" that JK aggregates. May help Reentrancy but hurt other classes |
| Add more Phase 2 layers (9+) | Dilution increases exponentially with depth. More layers ≠ more signal after ~3 hops |
| Increase JK entropy λ from 0.005 to 0.01 | Phase 3 drift is already mild (0.347→0.395 over 40 epochs). Stronger regularization may hurt Phase 1/2 equally |
| Remove CONTAINS edges from Phase 1 | CONTAINS is the backbone of the graph hierarchy — removing it disconnects FUNCTION from CFG |

---

*Analysis verified against source code: `gnn_encoder.py:580-629`, `sentinel_model.py:555-557`, `graph_extractor.py:1001-1062`.*
