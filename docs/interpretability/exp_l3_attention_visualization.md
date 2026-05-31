# EXP-L3 — GAT conv3 Attention Visualization

**Layer:** 3 — Behavioral Interpretability  **Priority:** P2  **Status:** PASS  
**Date run:** 2026-05-30  **Checkpoint:** GCB-P1-Run4 (ep32, F1=0.3362)

## Hypothesis

The conv3 layer (Phase 2, CONTROL_FLOW-only pass) should assign disproportionately high attention to CONTROL_FLOW and CALL_ENTRY edges. For reentrancy contracts specifically, high-attention edges should include CFG_NODE_CALL → CFG_NODE_WRITE transitions, reflecting the CEI (Check-Effects-Interactions) vulnerability pattern.

## Method

11 hand-crafted test contracts (6 vulnerability types, paired vulnerable/safe where available) are extracted via Slither into graphs. The conv3 layer's GAT attention weights are extracted for Phase 2 (CONTROL_FLOW-only subgraph). The top-20 attention edges per contract are ranked, and the fraction that are CONTROL_FLOW or CALL_ENTRY edges is computed. Pass criterion: ≥30% of top-20 edges are CF/CALL_ENTRY. A diagnostic also checks whether CFG_NODE_CALL → CFG_NODE_WRITE appears in the top edges (the CEI pattern).

Note: integer_uo_vulnerable.sol was skipped (requires solc 0.7.x; installed is 0.8.0).

## Results

| Contract | CF edges total | CF fraction in top edges | CFG_CALL→WRITE present | Result |
|----------|---------------|--------------------------|------------------------|--------|
| inheritance_propagation | 7 | **100%** | No | PASS |
| inheritance_safe | 7 | **100%** | No | PASS |
| integer_uo_safe | 4 | **100%** | No | PASS |
| reentrancy_safe | 6 | **100%** | No | PASS |
| reentrancy_vulnerable | 6 | **100%** | No | PASS |
| timestamp_safe | 3 | **100%** | No | PASS |
| timestamp_vulnerable | 3 | **100%** | No | PASS |
| tod_safe | 6 | **100%** | No | PASS |
| tod_vulnerable | 5 | **100%** | No | PASS |
| unused_return_safe | 4 | **100%** | No | PASS |
| unused_return_vulnerable | 2 | **100%** | No | PASS |

Mean CF fraction — reentrancy contracts: **1.00**  
Mean CF fraction — all other contracts: **1.00**  
CFG_NODE_CALL → CFG_NODE_WRITE in top edges: **0/11 contracts**

**Pass criteria:** ≥30% CF/CALL_ENTRY in top edges for ≥ most contracts  
**Overall: PASS** — 11/11 contracts at 100% CF fraction.

## Key Findings

- All top-attention edges in conv3 are CONTROL_FLOW edges (attn=1.0 for all), because conv3 operates on the CONTROL_FLOW-only Phase 2 subgraph — it physically cannot attend to non-CF edges.
- The 100% CF fraction result is structurally guaranteed by the architecture: conv3 receives only CF edges in its edge_index. This is not a learned behavior but a hardcoded architectural constraint.
- CFG_NODE_CALL → CFG_NODE_WRITE transitions (the CEI reentrancy pattern) do NOT appear in the top-attention edges for either reentrancy contract. This is the negative signal: even when CF edges are the only option, the GAT doesn't concentrate attention on the semantically critical call→write sequence.
- Reentrancy-vulnerable and reentrancy-safe contracts show nearly identical attention patterns: both attend to CFG_NODE_OTHER→CFG_NODE_CALL and CFG_NODE_CALL→CFG_NODE_OTHER edges but not to the critical CALL→WRITE boundary.
- The diagnostic notes: "CF attention NOT higher for reentrancy — model may not be using CEI structure."

## Architecture Implications

The conv3 CONTROL_FLOW pass produces 100% CF-fraction top-attention as a mathematical necessity (it processes only CF edges), so the PASS result is not evidence of learned semantic attention. The more informative finding is the absence of the CEI pattern (CALL→WRITE) in the top-attention edges, which is consistent with the EXP-L2 result (CFG edge ablation has near-zero impact). Conv3 is processing CF edges uniformly without concentrating on semantically relevant transitions. Combined with EXP-L1 (Phase 2 has the lowest JK weight) and EXP-L2 (CFG ablation delta ≈ 0), this confirms that the Phase 2 CFG layers are not carrying meaningful vulnerability signal.

## Caveats

- The 100% CF fraction is an artifact of architecture (conv3 uses CF-only edge_index), not evidence of learned attention quality. The true measure of CF learning is the uniformity of attention weights within the CF subgraph and the CEI-pattern check.
- All attention weights in this experiment are 1.0 — GAT has converged to uniform attention across CF edges, suggesting no discriminative edge selection within the CFG.
- integer_uo_vulnerable.sol was excluded due to the same solc-select version issue as EXP-L6.
- Test contracts are small (3–16 nodes), which may not represent the complexity of production contracts in the training corpus.
- PNG visualizations are saved at `ml/logs/interpretability/*_attn.png` for manual inspection.
