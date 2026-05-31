# EXP-B3: JK Weight Distribution Per Class

**Layer:** 3 — Learning
**Priority:** B3
**Status:** COMPLETE
**Date:** 2026-05-31
**Checkpoint:** GCB-P1-Run4 (ep32, F1=0.3362)
**Script:** `ml/scripts/interpretability/exp_b3_jk_weight_distribution.py`
**Output:** `ml/logs/interpretability/b3_jk_weight_distribution.json`

---

## Purpose

Measure JumpingKnowledge (JK) attention weight distribution per class — mean and std per phase across 470 contracts. Diagnoses whether Phase 2 weights are consistently lower across all classes, and whether any class selectively upweights Phase 2 relative to Phase 1 or Phase 3.

## Method

For 470 sampled val-split contracts, a forward pass is run and the JK attention weights over the three phase outputs are extracted. Contracts are grouped by vulnerability label. For each class, mean and std of per-phase JK weights are computed over all contracts containing that class label. Reported values are softmax-normalised attention weights summing to 1.0 per contract.

## Results

### JK Weight Mean (std) Per Phase Per Class

| Class | Phase1 | Phase2 | Phase3 |
|-------|--------|--------|--------|
| CallToUnknown | 0.332 (0.013) | 0.322 (0.012) | 0.346 (0.023) |
| DenialOfService | 0.334 (0.012) | 0.319 (0.011) | 0.347 (0.022) |
| ExternalBug | 0.333 (0.012) | 0.323 (0.009) | 0.343 (0.020) |
| GasException | 0.333 (0.017) | 0.324 (0.012) | 0.343 (0.026) |
| IntegerUO | 0.336 (0.013) | 0.322 (0.010) | 0.342 (0.021) |
| MishandledException | 0.333 (0.014) | 0.321 (0.011) | 0.346 (0.024) |
| Reentrancy | 0.334 (0.013) | 0.322 (0.010) | 0.344 (0.022) |
| Timestamp | 0.335 (0.015) | 0.323 (0.013) | 0.342 (0.025) |
| TOD | 0.333 (0.013) | 0.320 (0.010) | 0.347 (0.022) |
| UnusedReturn | 0.334 (0.014) | 0.321 (0.011) | 0.345 (0.023) |

### Phase Ordering Summary

All 10 classes show the same ordering: **Phase 3 > Phase 1 > Phase 2**.

| Statistic | Phase1 | Phase2 | Phase3 |
|-----------|--------|--------|--------|
| Grand mean | 0.334 | 0.322 | 0.345 |
| Range across classes | 0.332–0.336 | 0.319–0.324 | 0.342–0.347 |

Phase 2 is consistently the lowest by ~1.2pp vs Phase 1 (grand means: 0.334 vs 0.322). Phase 3 leads by ~1.1pp vs Phase 1 (0.334 vs 0.345).

## Key Findings

1. **Universal Phase 3 > Phase 1 > Phase 2 ordering.** No class deviates from this pattern. The model has learned to slightly upweight Phase 3 (REVERSE_CONTAINS hierarchy) and slightly downweight Phase 2 (CFG/ICFG), consistently across all vulnerability types.

2. **No class selectively upweights Phase 2.** Despite the architectural intent that Phase 2 captures CFG-level patterns relevant to Reentrancy and TOD, neither class shows elevated Phase 2 weights. The JK weights are class-agnostic in their phase preference.

3. **Small standard deviations.** Phase weight std values of 0.01–0.03 indicate stable, low-variance JK allocations per class. The model is not dynamically routing attention based on contract-specific CFG content.

4. **JK weights are near-uniform.** With values 0.322–0.345, the distribution is close to uniform (0.333 each). The JK attention module has learned only a mild phase preference, consistent with EXP-L1 reporting 99.98% of max entropy for JK weights.

5. **Consistent with EXP-B1 gradient norms.** Phase 1 > Phase 2 > Phase 3 in gradient norms (EXP-B1), but Phase 3 > Phase 1 > Phase 2 in JK attention weights. Phase 3 is upweighted at inference despite receiving the least gradient during training — suggesting Phase 3 provides a complementary representation that the JK module learned to rely on even though it required less loss signal to learn.

## Pass/Fail Analysis

No binary pass criterion defined (diagnostic experiment). The finding confirms that Phase 2 JK weights are universally low and class-agnostic — no vulnerability class has discovered a Phase 2 upweighting strategy.

## Recommended Next Steps

1. Re-run after Run 5 training (with `aux_phase2_loss_weight=0.10`) to check whether the Phase 2 auxiliary loss shifts JK weights towards Phase 2 for relevant classes (Reentrancy, TOD).
2. Monitor JK weight distribution as a training metric — a shift towards Phase 2 in Run 5 would confirm the auxiliary loss is working as intended.
3. Cross-reference with EXP-L5 Phase 2 probing — if Run 5 increases Phase 2 probing F1 for Reentrancy, it should correlate with JK Phase 2 weight increase.
