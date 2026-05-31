# EXP-B1: Phase 2 Gradient Norm

**Layer:** 3 — Learning
**Priority:** B1
**Status:** COMPLETE
**Date:** 2026-05-31
**Checkpoint:** GCB-P1-Run4 (ep32, F1=0.3362)
**Script:** `ml/scripts/interpretability/exp_b1_phase2_gradient_norm.py`
**Output:** `ml/logs/interpretability/b1_phase2_gradient_norm.json`

---

## Purpose

Measure gradient norm at each GNN phase LayerNorm output during a supervised forward pass. Diagnoses whether Phase 2 (CFG/ICFG layers, L3+L4+L5) receives meaningful loss signal or is effectively ignored by backpropagation.

## Method

For each of 10 vulnerability classes, a mini-batch of positive contracts is assembled and a forward+backward pass is run. Gradient norms are recorded at the LayerNorm output of each phase: Phase 1 (L1+L2 norm), Phase 2 (L3+L4+L5 norm), Phase 3 (L6+L7+L8 norm). Mean gradient norm per phase per class is reported.

## Results

### Mean Gradient Norm Per Phase Per Class

| Class | Phase1 | Phase2 | Phase3 | P1>P2? | P2>P3? |
|-------|--------|--------|--------|--------|--------|
| CallToUnknown | 0.1165 | 0.0961 | 0.0918 | ✓ | ✓ |
| DenialOfService | 0.1102 | 0.0790 | 0.0656 | ✓ | ✓ |
| ExternalBug | 0.0820 | 0.0601 | 0.0502 | ✓ | ✓ |
| GasException | 0.0708 | 0.0565 | 0.0426 | ✓ | ✓ |
| IntegerUO | 0.1340 | 0.1007 | 0.0794 | ✓ | ✓ |
| MishandledException | 0.0681 | 0.0501 | 0.0405 | ✓ | ✓ |
| Reentrancy | 0.1261 | 0.0900 | 0.0770 | ✓ | ✓ |
| Timestamp | 0.2692 | 0.2332 | 0.1864 | ✓ | ✓ |
| TOD | 0.0674 | 0.0521 | 0.0398 | ✓ | ✓ |
| UnusedReturn | 0.1172 | 0.0859 | 0.0716 | ✓ | ✓ |

Phase 1 > Phase 2 > Phase 3 holds without exception for all 10 classes.

### Phase 2 / Phase 1 Gradient Ratio

| Class | Phase2/Phase1 |
|-------|---------------|
| CallToUnknown | 0.825 |
| DenialOfService | 0.717 |
| ExternalBug | 0.733 |
| GasException | 0.798 |
| IntegerUO | 0.751 |
| MishandledException | 0.736 |
| Reentrancy | 0.714 |
| Timestamp | 0.867 |
| TOD | 0.774 |
| UnusedReturn | 0.733 |

Phase 2 receives approximately 71–87% of the Phase 1 gradient signal, depending on class.

## Key Findings

1. **No phase is gradient-starved.** Phase 2 receives 71–87% of Phase 1 gradient norm — substantial signal, not near-zero. The concern that Phase 2 might be entirely bypassed during learning is not supported.

2. **Monotonic decreasing pattern.** Phase 1 > Phase 2 > Phase 3 for every class. Earlier phases receive stronger gradient signal, consistent with Phase 1 learning the dominant structural pattern first and later phases refining it.

3. **Timestamp dominates.** Timestamp has the highest absolute gradient norms across all phases (Phase1=0.2692, Phase2=0.2332, Phase3=0.1864), consistent with it being the class with the strongest model performance (ep32 F1=0.329) and a strong size-based shortcut (EXP-S3).

4. **Phase 2 gradient is meaningful but secondary.** The ~75–80% ratio suggests Phase 2 participates in learning but does not drive the loss as strongly as Phase 1. This is coherent with EXP-L1 (Phase 2 JK weight=0.322, lowest of three phases) and EXP-L5 (Phase 2 probing adds no linear signal for most classes).

5. **Phase 3 receives least gradient.** Despite having the highest JK attention weight (EXP-L1: 0.346), Phase 3 receives the lowest gradient norm. This reflects the JK aggregation architecture: Phase 3 is upweighted at inference but receives reduced gradient due to its position at the end of the forward path.

## Pass/Fail Analysis

No binary pass criterion was defined for this experiment (diagnostic). All phases receive non-trivial gradient signal — the null hypothesis (Phase 2 is gradient-starved) is rejected.

## Recommended Next Steps

1. Cross-reference Phase 2 gradient ratios with EXP-L5 Phase 2 probing results — classes with higher Phase2/Phase1 ratios (Timestamp=0.867, CallToUnknown=0.825) may show more Phase 2 benefit in probing after a longer training run.
2. Monitor Phase 2 gradient ratio across training epochs in Run 5 to check whether the auxiliary Phase 2 loss (`aux_phase2_loss_weight=0.10`) increases Phase 2 gradient norms.
3. Consider logging phase gradient norms as a training metric to detect gradient collapse early.
