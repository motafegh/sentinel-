# EXP-L9 — Attention Rollout Attribution

**Layer:** 3 — Behavioral Interpretability
**Priority:** P2
**Status:** FAIL
**Date run:** 2026-05-31
**Checkpoint:** GCB-P1-Run4 (ep32, F1=0.3362)
**Script:** `ml/scripts/interpretability/exp_l9_attention_rollout.py`
**Output:** `ml/logs/interpretability/exp_l9_rollout/`

---

## Hypothesis

Attention rollout should assign higher mean attribution scores to CFG_NODE_CALL (type 8) and CFG_NODE_WRITE (type 9) nodes in reentrancy-vulnerable contracts than in reentrancy-safe contracts. The relative attribution rank of CALL/WRITE nodes should discriminate vulnerable from safe.

## Method

3 test contracts are processed (reentrancy_vulnerable, reentrancy_safe, inheritance_propagation). For each contract, rollout attribution is computed by multiplying Phase 2 (conv3 CF-only) and Phase 3 (conv4 REVERSE_CONTAINS) attention matrices in sequence. The resulting per-node attribution vector is ranked.

**Revised pass criterion (COMPLETENESS audit INCOMPLETE-8):** PASS iff vulnerable contract's mean CALL/WRITE attribution score > safe contract's mean CALL/WRITE attribution score. The original criterion (≥2 CALL/WRITE in top-10) was non-discriminative — both contracts satisfied it identically.

Note: This is a partial rollout (2 of 8 layers) since full 8-layer rollout requires additional model instrumentation.

## Results

| Contract | Nodes | CALL/WRITE in top-10 | Mean CW attribution |
|----------|-------|----------------------|---------------------|
| reentrancy_vulnerable | 12 | 3 | 0.09038 |
| reentrancy_safe | 12 | 3 | **0.09692** |
| delta (vulnerable − safe) | — | 0 | **−0.00654** |

**Result: FAIL.** Safe contract has higher mean CALL/WRITE attribution than vulnerable.

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

## Retraction of Original PASS

The original (2026-05-30) report recorded status PASS under the criterion "≥2 CALL/WRITE nodes in top-10 for reentrancy contracts." This status is **retracted**. Both reentrancy_vulnerable and reentrancy_safe each had exactly 3 CALL/WRITE nodes in their top-10 — the criterion was trivially satisfied by both and provided no discriminative signal. The original PASS was an artifact of a non-discriminative threshold.

With the corrected relative-rank criterion, the result is FAIL: the safe contract has marginally higher mean CALL/WRITE attribution (0.09692 vs 0.09038, delta=−0.00654).

## Key Findings

1. **Rollout cannot distinguish vulnerable from safe reentrancy contracts.** The top-2 attributed nodes in both contracts are identical (FUNCTION nodes 5 and 2, score=0.250 each). Attribution scores and node rankings are structurally near-identical across the pair.

2. **FUNCTION nodes absorb most rollout weight.** FUNCTION nodes occupy the top-2 positions in both contracts due to their role as aggregation hubs in the REVERSE_CONTAINS hierarchy (Phase 3). This is consistent with EXP-L1 (Phase 3 has highest JK weight=0.346) — rollout propagates the structural hierarchy signal rather than semantic CFG patterns.

3. **Consistent with EXP-L6 counterfactual.** The model does not structurally attend more to CALL/WRITE nodes in vulnerable reentrancy vs safe reentrancy contracts. Both the rollout (L9) and the counterfactual experiment (L6) confirm the model does not isolate CEI node sequences as the primary discriminator.

4. **Test contracts are small and structurally similar.** These 12-node test contracts may exaggerate attribution similarity. Results should be validated on larger, more structurally distinct contract pairs.

## Pass/Fail Analysis

- Relative CALL/WRITE attribution (vulnerable vs safe): delta = −0.00654 → FAIL.
- The previous PASS (original criterion) is retracted.

## Caveats

- Partial rollout (2 of 8 layers) — full 8-layer rollout would require hooking into conv1, conv2, conv3b, conv3c and may produce different attribution profiles.
- Only 3 test contracts. Conclusions should be validated on a larger held-out set.
- The 12-node test contracts are minimally different, which may exaggerate rollout similarity between vulnerable and safe.

## Recommended Next Steps

1. Implement full 8-layer rollout to evaluate whether earlier-layer attention changes the vulnerable/safe attribution gap.
2. Test on larger, structurally distinct contract pairs where vulnerable contracts have clearly more CALL/WRITE nodes than safe ones.
3. Cross-reference with EXP-L4 gradient saliency to check whether the gradient signal — which does not depend on the rollout path — shows a similar null result.
4. Consider integrated gradients as an alternative attribution method that does not rely on attention weights and may be more sensitive to CEI patterns.
