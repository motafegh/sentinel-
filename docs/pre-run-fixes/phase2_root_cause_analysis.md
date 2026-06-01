# Phase 2 — Complete Root Cause Analysis

**Date:** 2026-06-01  
**Author:** Interpretability audit (Phases A–G)  
**Status:** Pre-Run-5 reference — all issues confirmed against source code  
**Scope:** Why Phase 2 (GNN layers L3+L4+L5, CFG/ICFG/DEF_USE edges) fails to contribute meaningful signal to final predictions despite receiving 72–91% of Phase 1 gradient norm.

---

## Executive Summary

Phase 2 receives substantial gradient signal (EXP-B1: 72–91% of Phase 1) but contributes negligible inference signal (EXP-L2: embedding ablation drop = 1.11×10⁻⁶ vs structural = 0.0121). Seven confirmed root causes explain this. They are not independent: they compound. Fixing any single one is insufficient — the full set must be addressed together in Run 5 and beyond.

---

## Confirmed Root Cause 1 — FUNCTION Nodes Get Identity Transform from Phase 2

**Source:** `ml/src/models/gnn_encoder.py` lines 543–551

```python
# Phase 2: CF-only layer
cf_only_ei = edge_mask(cf_only)
x = x + dropout(relu(conv3(x, cf_only_ei, cf_ea[cf_only])))
```

CF edges only connect CFG nodes to CFG nodes. FUNCTION nodes have zero CF edges. PyG's `GATConv` returns zero vectors for nodes with no incident edges. The residual connection then computes:

```
x_function = x_function + dropout(relu(0)) = x_function
```

FUNCTION nodes pass through all three Phase 2 layers completely unchanged. Since JK aggregation pools over all node types and graph-level readout is performed over FUNCTION-type nodes (EXP-A1: 100% of graphs), Phase 2's contribution to the final graph embedding is structurally suppressed.

**Evidence:** EXP-A1 confirms 100% of graphs use FUNCTION-like nodes for pooling. EXP-L2 structural ablation confirms Phase 2 edge removal has near-zero effect on most class predictions (exceptions: Timestamp suppression via CONTROL_FLOW is large).

---

## Confirmed Root Cause 2 — aux_phase2_loss_weight Was 0.0 Throughout All of Run 4

**Source:** `ml/src/training/trainer.py`

- `TrainConfig.aux_phase2_loss_weight: float = 0.10` — **first introduced in commit 9310046 (pre-Run-5)**
- `train_epoch()` function default: `aux_phase2_loss_weight=0.0` (backward-compat default)
- During Run 4, this parameter did not exist in `TrainConfig`. It was added specifically as a Run 5 fix.

This means there was **no auxiliary supervision signal** pushing Phase 2 representations toward class-discriminative structure. The only gradient Phase 2 received came through the full forward path — already dominated by Phase 1 (which sees the same input x before Phase 2 modifies it, due to residual structure).

**Evidence:** Git log — commit e022018 "Run4 pre-flight" has no `aux_phase2_loss_weight`. Commit 9310046 "pre-Run-5" is the first occurrence.

---

## Confirmed Root Cause 3 — Phase 2 Has 8× Lower Attention Head Capacity Than Phase 1

**Source:** `ml/src/models/gnn_encoder.py` lines 249–277

```python
# Phase 1: conv1, conv2 — 8 heads, concat=True → 256-dim output
GATConv(in_channels, 32, heads=8, concat=True, ...)   # → 256

# Phase 2: conv3, conv3b, conv3c — 1 head, concat=False → 256-dim output
GATConv(256, 256, heads=1, concat=False, ...)          # → 256
```

Phase 1 uses 8 attention heads and concatenates → 8 distinct attention patterns per node pair. Phase 2 uses 1 head → single scalar attention weight. In the limit, Phase 2 attention reduces to a weighted average with a single learned weighting — far less expressive for learning conditional routing (e.g., "which CFG predecessor matters for this vulnerability class?").

This is compounded by EXP-L3's finding: all Phase 2 GAT attention weights = 1.0 (uniform). The 1-head architecture has not learned selective attention within CFG.

**Evidence:** EXP-L3 — all attention weights in conv3/conv3b = 1.0 (uniform), confirming the single-head architecture has not differentiated message routing.

---

## Confirmed Root Cause 4 — JK Entropy Regularizer Pushes Phase 2 Weight Below 1/3

**Source:** `ml/src/models/gnn_encoder.py` lines 92–131 (JumpingKnowledge class)

```python
self.attn = nn.Linear(channels, 1, bias=False)  # single scalar per phase
# softmax over K=3 phases
scores = self.attn(xs).squeeze(-1)   # [N, 3]
weights = F.softmax(scores, dim=-1)  # [N, 3]
```

The JK attention is a single linear layer per node, scoring each of the 3 phase embeddings. The entropy regularizer (`λ × (log(3) − H)`) pushes toward uniform weights (H = log(3)).

However, for FUNCTION nodes: Phase 1 embedding ≠ Phase 2 embedding (Phase 2 is identity for FUNCTION nodes, so Phase 2 embedding = Phase 1 embedding). With two identical inputs and one different (Phase 3), the single linear layer cannot easily assign Phase 2 a weight different from Phase 1. The entropy regularizer then pushes both toward 1/3, meaning **Phase 2 is assigned 1/3 weight by pressure, not by learned utility.**

EXP-B3 result: Universal Phase3 > Phase1 > Phase2 ordering. No class selectively upweights Phase 2. Std 0.01–0.03 (stable — weights frozen at regularizer equilibrium).

**Evidence:** EXP-B3 — Phase 2 JK weight universally lowest across all classes. EXP-L1 — Phase 2 JK weight = 0.322 (lowest of 3 phases, Phase 3 = 0.346).

---

## Confirmed Root Cause 5 — DEF_USE Edges Get Only 1 Hop (Single Layer, Layer 5 Only)

**Source:** `ml/src/models/gnn_encoder.py` lines 466–484 (edge mask construction)

```python
# phase2_ei includes: CF + CALL_ENTRY + RETURN_TO + DEF_USE
# But DEF_USE is only present in conv3c (layer 5, the joint CF+ICFG layer)
# conv3 (layer 3): CF-only
# conv3b (layer 4): CALL_ENTRY + RETURN_TO (ICFG-only)
# conv3c (layer 5): CF + ICFG + DEF_USE (joint)
```

DEF_USE edges encode data flow (variable definition → use). Meaningful data-flow reasoning requires chaining: A defines → B uses → C uses, etc. With only one hop via DEF_USE (Layer 5), the model cannot propagate information through def-use chains longer than 1 step.

EXP-E1 result: DEF_USE added to Phase 2 edge set, but k=8 reachability is only 38.2% for FUNCTION nodes — meaning 61.8% of FUNCTION nodes cannot receive ANY Phase 2 message even with 8 hops, because they require going through intermediate CFG nodes that are never reached.

**Evidence:** EXP-E1 — Phase 2 k=8 CEI reachability 38.2%. Graph architecture limits how much def-use chain information can propagate upward to FUNCTION nodes.

---

## Confirmed Root Cause 6 — Phase 3 Does Phase 2's Job (REVERSE_CONTAINS Lifts CFG Information)

**Source:** `ml/src/models/gnn_encoder.py` lines 580–592 (Phase 3 forward)

```python
# Phase 3: REVERSE_CONTAINS (CFG → FUNCTION) + CONTAINS (FUNCTION → CFG)
x = x + dropout(relu(conv5(x, rev_contains_ei, rev_contains_ea)))  # upward
x = x + dropout(relu(conv5b(x, contains_ei, contains_ea)))          # downward
```

REVERSE_CONTAINS edges (built at runtime: gnn_encoder.py:483) connect each CFG node to its parent FUNCTION node. Phase 3 Layer 6 lifts CFG-level information (whatever Phase 2 computed) up to the FUNCTION level. Phase 3 Layer 7 then pushes FUNCTION information back down to CFG.

This means **Phase 3 is the actual aggregation point where CFG signal reaches FUNCTION nodes.** Phase 2's work is only useful insofar as it modifies CFG node embeddings before Phase 3 lifts them. If Phase 2 does little to CFG embeddings (Root Cause 1: FUNCTION nodes are identity), Phase 3 simply lifts Phase 1 embeddings.

**Evidence:** EXP-L2 — Phase 3 edges not ablated in that experiment. EXP-B3 — Phase 3 has highest JK weight (0.346) despite receiving lowest gradient (EXP-B1), consistent with Phase 3 being the "delivery" layer that carries Phase 2 work to the classifier.

---

## Confirmed Root Cause 7 — Suppression Encoded in Learned Weights (Reentrancy Inversion)

**Source:** `ml/interpretability_results/exp_l2/exp_l2_ablation_delta.json`

```json
"delta_pos_structural": {
  "Reentrancy": {
    "CONTROL_FLOW": 0.020328,   // removing CF INCREASES Reentrancy prediction
    "CALL_ENTRY":   0.010085,
    "RETURN_TO":    0.018186
  }
}
```

All Phase 2 structural ablation deltas for Reentrancy are **positive** — removing Phase 2 edges INCREASES the Reentrancy prediction score. This means Phase 2 is actively suppressing Reentrancy predictions during inference.

The mechanism: Contracts with dense CFG (many CF edges, complex control flow) are statistically NOT Reentrancy-vulnerable in the training distribution — they are large, well-engineered contracts. Phase 2 learned "dense CFG → not Reentrancy" and applies this as suppression. This is a valid correlation in the data, but it means Phase 2 encodes a **negative** shortcut rather than positive CFG-based Reentrancy evidence.

This is the deepest issue — even where Phase 2 is active, it may be working against vulnerability detection rather than for it.

**Evidence:** EXP-L2 structural ablation. EXP-S3: cfg_call_count Cohen's d for Timestamp (not Reentrancy directly). EXP-L7: Reentrancy calibration poor. EXP-L6: CEI counterfactual safe > vuln delta = −0.0071 (model predictions go wrong direction).

---

## Summary Table

| # | Root Cause | Scope | Fix in Run 5? |
|---|-----------|-------|---------------|
| 1 | FUNCTION nodes get identity transform from Phase 2 | Architecture | Partial (aux loss may not fix pooling mismatch) |
| 2 | aux_phase2_loss_weight = 0.0 throughout Run 4 | Training | YES — aux_phase2_loss_weight=0.10 in Run 5 |
| 3 | 8× head capacity gap (Phase 2 heads=1 vs Phase 1 heads=8) | Architecture | NO — not changed for Run 5 |
| 4 | JK entropy regularizer pushes Phase 2 to 1/3 by default | Training | PARTIAL — λ=0.005 unchanged; aux loss may shift equilibrium |
| 5 | DEF_USE gets only 1 hop (Layer 5 only) | Architecture | NO — not changed for Run 5 |
| 6 | Phase 3 does Phase 2's job via REVERSE_CONTAINS | Architecture | NO — by design; aux loss targets CEI pooling instead |
| 7 | Phase 2 learned suppression (Reentrancy inversion) | Training | UNKNOWN — Run 5 aux loss may rebalance |

---

## Run 5 Intervention Coverage

Run 5 addresses Root Cause 2 directly (`aux_phase2_loss_weight=0.10`) and Root Cause 7 indirectly (by providing explicit CEI-level supervision, the suppression signal may be counteracted). Root Causes 1, 3, 5, 6 require architectural changes beyond Run 5 scope.

**Expected Run 5 outcome:** Phase 2 gradient norms should increase for Reentrancy and TOD. JK Phase 2 weight may shift above 0.322. Whether this translates to F1 improvement depends on whether the CEI auxiliary signal is strong enough to overcome the pooling mismatch (Root Cause 1).

**Monitoring:** Log per-phase gradient norms each epoch (EXP-B1 method). Log JK weight distribution each epoch (EXP-B3 method). If Phase 2 JK weight rises above 0.35 by epoch 10, the aux loss is taking effect.
