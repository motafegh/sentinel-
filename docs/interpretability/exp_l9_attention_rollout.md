# EXP-L9 — Attention Rollout Attribution

**Layer:** 3 — Behavioral Interpretability  **Priority:** P2  **Status:** PASS  
**Date run:** 2026-05-30  **Checkpoint:** GCB-P1-Run4 (ep32, F1=0.3362)

## Hypothesis

Attention rollout (propagating attention weights through GNN layers to produce node-level attribution scores) should identify CFG_NODE_CALL and CFG_NODE_WRITE nodes as the top-attributed nodes for reentrancy contracts. These node types represent the external call and state-write operations that define the CEI vulnerability pattern.

## Method

3 test contracts are processed (reentrancy_vulnerable, reentrancy_safe, inheritance_propagation). For each contract, rollout attribution is computed by multiplying Phase 2 (conv3 CF-only) and Phase 3 (conv4 REVERSE_CONTAINS) attention matrices in sequence. The resulting per-node attribution vector is ranked. Pass criterion: for reentrancy contracts, ≥2 of the top-10 nodes by rollout score are CFG_NODE_CALL (type 8) or CFG_NODE_WRITE (type 9).

Note: This is a partial rollout (2 of 8 layers) since full 8-layer rollout requires additional model instrumentation.

## Results

| Contract | Nodes | Pool nodes | CALL/WRITE in top-10 | Result |
|----------|-------|------------|----------------------|--------|
| reentrancy_vulnerable | 12 | 2 | **3/10** | PASS |
| reentrancy_safe | 12 | 2 | **3/10** | PASS |
| inheritance_propagation | 16 | 3 | **2/10** | PASS |

### Top-10 node attribution — reentrancy_vulnerable
| Rank | Node | Type | Score |
|------|------|------|-------|
| 1 | 5 | FUNCTION | 0.250 |
| 2 | 2 | FUNCTION | 0.250 |
| 3 | 4 | CFG_NODE_WRITE | 0.183 |
| 4 | 8 | CONTRACT | 0.074 |
| 5 | 3 | CONTRACT | 0.067 |
| 6 | 9 | CFG_NODE_CALL | 0.050 |
| 7 | 11 | CFG_NODE_WRITE | 0.045 |
| 8 | 10 | CONTRACT | 0.044 |
| 9 | 7 | CFG_NODE_OTHER | 0.026 |
| 10 | 6 | CONTRACT | 0.010 |

### Top-10 node attribution — reentrancy_safe
| Rank | Node | Type | Score |
|------|------|------|-------|
| 1 | 5 | FUNCTION | 0.250 |
| 2 | 2 | FUNCTION | 0.250 |
| 3 | 4 | CFG_NODE_WRITE | 0.183 |
| 4 | 8 | CFG_NODE_WRITE | 0.074 |
| 5 | 3 | CONTRACT | 0.067 |
| 6 | 9 | CONTRACT | 0.049 |
| 7 | 11 | CONTRACT | 0.047 |
| 8 | 10 | CFG_NODE_CALL | 0.045 |
| 9 | 7 | CFG_NODE_OTHER | 0.026 |
| 10 | 6 | CONTRACT | 0.009 |

**Pass criteria:** Reentrancy contracts with ≥2 CALL/WRITE in top-10: 2/2  
**Overall: PASS** — Both reentrancy contracts have 3 CALL/WRITE nodes in top-10.

## Key Findings

- Both reentrancy_vulnerable and reentrancy_safe produce identical top-2 rankings (FUNCTION nodes 5 and 2, each with score=0.250). This means rollout cannot distinguish the vulnerable from the safe version — the attribution is structurally identical between the pair.
- FUNCTION nodes dominate the top ranks for both reentrancy contracts (positions 1–2, score=0.25 each). These are not the semantically critical nodes for CEI detection.
- CFG_NODE_WRITE and CFG_NODE_CALL nodes do appear in positions 3–8, satisfying the 2/10 criterion, but their scores (0.045–0.183) are substantially below the FUNCTION nodes.
- The inheritance_propagation contract (no reentrancy) also achieves PASS (2/10 CALL/WRITE nodes), suggesting the criterion does not distinguish reentrancy-specific attribution from general graph structure.
- All three contracts have the same pool nodes count relative to their size (approximately 1 pool node per 6 nodes), and the rollout scores for pool nodes are near-zero, concentrating attribution in non-pool CFG and FUNCTION nodes.

## Architecture Implications

The rollout PASS is technically satisfied but the diagnostic pattern is unfavorable: identical top attribution for vulnerable vs. safe reentrancy contracts means rollout provides no discriminative signal for the CEI pattern. FUNCTION nodes absorb the majority of rollout weight due to their position as natural aggregation hubs in the REVERSE_CONTAINS hierarchy, which connects to the EXP-L1 finding (Phase 3 REVERSE_CONTAINS dominant). The rollout is propagating the structural hierarchy signal rather than semantic CFG patterns. A true vulnerability-aware rollout would show higher attribution scores for CFG_NODE_CALL→CFG_NODE_WRITE sequences in the vulnerable contract vs. the safe one; this is not observed here.

## Caveats

- Rollout covers only Phase 2 (conv3) and Phase 3 (conv4/conv4b/conv4c), not all 8 GNN layers. Full 8-layer rollout would require hooking into conv1, conv2, conv3b, conv3c as well.
- Identical attribution for vulnerable and safe reentrancy is a strong negative signal, but these specific test contracts are very small (12 nodes) and minimally different, which may exaggerate the similarity.
- The pass criterion (≥2 CALL/WRITE in top-10) is a weak baseline that does not require any differentiation between vulnerable and safe contracts.
- Results saved at `ml/logs/interpretability/exp_l9_rollout/l9_results.json` and `attention_rollout_report.txt`.
