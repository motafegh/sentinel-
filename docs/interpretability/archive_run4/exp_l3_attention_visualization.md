# EXP-L3 — GAT conv3 Attention Visualization

**Layer:** 3 — Behavioral Interpretability  **Priority:** P2  **Status:** ARCHITECTURAL N/A (audit 2026-06-01)
**Date run:** 2026-05-30  **Checkpoint:** GCB-P1-Run4 (ep32, F1=0.3362)

> **Audit note (2026-06-01):** The original PASS status was retracted. The "100% CF fraction" result is mathematically guaranteed by architecture — conv3 receives only CONTROL_FLOW edges in its `edge_index` and physically cannot attend to any other edge type. Reporting this as a PASS is a false positive. The true finding, buried in the Caveats section, is the meaningful one: **all GAT attention weights = 1.0 (uniform)** — the model has learned no selective attention within the CFG. That is the headline finding for this experiment.

## Hypothesis

The conv3 layer (Phase 2, CONTROL_FLOW-only pass) should assign disproportionately high attention to CONTROL_FLOW and CALL_ENTRY edges. For reentrancy contracts specifically, high-attention edges should include CFG_NODE_CALL → CFG_NODE_WRITE transitions, reflecting the CEI (Check-Effects-Interactions) vulnerability pattern.

## Method

11 hand-crafted test contracts (6 vulnerability types, paired vulnerable/safe where available) are extracted via Slither into graphs. The conv3 layer's GAT attention weights are extracted for Phase 2 (CONTROL_FLOW-only subgraph). The top-20 attention edges per contract are ranked, and the fraction that are CONTROL_FLOW or CALL_ENTRY edges is computed. Pass criterion: ≥30% of top-20 edges are CF/CALL_ENTRY. A diagnostic also checks whether CFG_NODE_CALL → CFG_NODE_WRITE appears in the top edges (the CEI pattern).

> **Method note (audit 2026-06-01):** The pass criterion ("≥30% of top-20 edges are CF") is trivially satisfied when conv3 is wired to a CF-only subgraph. The criterion should have been formulated to test within-CF selectivity (e.g., "do attention weights concentrate on CEI-relevant CFG_CALL→CFG_WRITE transitions?"). The 100% result answers a question that the architecture already answers by construction.

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
**Original verdict: PASS** — 11/11 contracts at 100% CF fraction.  
**Revised verdict (audit 2026-06-01): ARCHITECTURAL N/A** — see audit note at top of doc.

## Key Findings

> **Real headline finding:** All GAT attention weights in conv3 = **1.0 (uniform)**. The model has learned no selective attention within the control-flow graph. Every CF edge receives identical attention regardless of semantic importance.

- The 100% CF fraction in top-attention is mathematically guaranteed: conv3's `edge_index` contains only CONTROL_FLOW edges. The pass criterion tests something the architecture enforces by construction, not something the model learned.
- **CEI pattern absent (0/11 contracts):** CFG_NODE_CALL → CFG_NODE_WRITE transitions do not appear in top-attention edges for any contract, including reentrancy-vulnerable ones. Even when CF edges are the only option, the model does not concentrate attention on the semantically critical call→write boundary.
- Reentrancy-vulnerable and reentrancy-safe contracts produce identical attention patterns — both attend uniformly across all CF edges with no difference attributable to the vulnerability.
- Uniform attention (all weights = 1.0) means conv3 is effectively performing mean aggregation over all CF neighbours, not a learned selective aggregation. Consistent with EXP-L2 (CFG structural ablation Δ ≈ 0) and EXP-L1 (Phase 2 JK weight lowest across all classes).

## Architecture Implications

The conv3 CONTROL_FLOW pass produces 100% CF-fraction top-attention as a mathematical necessity (it processes only CF edges), so the PASS result is not evidence of learned semantic attention. The more informative finding is the absence of the CEI pattern (CALL→WRITE) in the top-attention edges, which is consistent with the EXP-L2 result (CFG edge ablation has near-zero impact). Conv3 is processing CF edges uniformly without concentrating on semantically relevant transitions. Combined with EXP-L1 (Phase 2 has the lowest JK weight) and EXP-L2 (CFG ablation delta ≈ 0), this confirms that the Phase 2 CFG layers are not carrying meaningful vulnerability signal.

## Caveats

- The 100% CF fraction is an artifact of architecture (conv3 uses CF-only edge_index), not evidence of learned attention quality. The true measure of CF learning is the uniformity of attention weights within the CF subgraph and the CEI-pattern check.
- All attention weights in this experiment are 1.0 — GAT has converged to uniform attention across CF edges, suggesting no discriminative edge selection within the CFG.
- integer_uo_vulnerable.sol was excluded due to the same solc-select version issue as EXP-L6.
- Test contracts are small (3–16 nodes), which may not represent the complexity of production contracts in the training corpus.
- PNG visualizations are saved at `ml/logs/interpretability/*_attn.png` for manual inspection.
