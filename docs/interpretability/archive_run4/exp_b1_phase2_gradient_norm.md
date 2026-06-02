# EXP-B1: Phase 2 Gradient Norm

**Layer:** 3 — Learning
**Priority:** B1
**Status:** COMPLETE (rerun 2026-06-01 with corrected gradient method)
**Date:** 2026-05-31 (original) / 2026-06-01 (corrected rerun)
**Checkpoint:** GCB-P1-Run4 (ep32, F1=0.3362)
**Script:** `ml/scripts/interpretability/exp_b1_phase2_gradient_norm.py`
**Output:** `ml/interpretability_results/exp_b1/`

> **Method correction (2026-06-01 audit):** Original run backpropagated through the raw logit `logits[0, class_idx]`. This is non-standard — training uses `BCEWithLogitsLoss` which applies sigmoid before computing gradient. The script was corrected to use `F.binary_cross_entropy_with_logits(logits[0, class_idx].unsqueeze(0), target=1)`. Absolute gradient magnitudes are ~3.5× smaller after the fix (sigmoid derivative dampens the gradient at high logit values), but the relative ordering across phases is unchanged. Results below are from the corrected run.

---

## Purpose

Measure gradient norm at each GNN phase LayerNorm output during a supervised forward pass. Diagnoses whether Phase 2 (CFG/ICFG layers, L3+L4+L5) receives meaningful loss signal or is effectively ignored by backpropagation.

## Method

For each of 10 vulnerability classes, a mini-batch of positive contracts is assembled and a forward+backward pass is run. Gradient norms are recorded at the LayerNorm output of each phase: Phase 1 (L1+L2 norm), Phase 2 (L3+L4+L5 norm), Phase 3 (L6+L7+L8 norm). Mean gradient norm per phase per class is reported.

## Results

### Mean Gradient Norm Per Phase Per Class

| Class | Phase1 | Phase2 | Phase3 | P2/P1 ratio |
|-------|--------|--------|--------|-------------|
| CallToUnknown | 0.051893 | 0.041190 | 0.035657 | 79.4% |
| DenialOfService | 0.034882 | 0.025193 | 0.020473 | 72.2% |
| ExternalBug | 0.044038 | 0.033292 | 0.028976 | 75.6% |
| GasException | 0.041313 | 0.032616 | 0.025441 | 78.9% |
| IntegerUO | 0.039092 | 0.028423 | 0.022379 | 72.7% |
| MishandledException | 0.045841 | 0.036230 | 0.031026 | 79.0% |
| Reentrancy | 0.050614 | 0.037378 | 0.035040 | 73.8% |
| Timestamp | 0.094683 | 0.086430 | 0.073734 | 91.3% |
| TransactionOrderDependence | 0.033661 | 0.025190 | 0.020505 | 74.8% |
| UnusedReturn | 0.080223 | 0.061586 | 0.052428 | 76.8% |

Phase 1 > Phase 2 > Phase 3 holds without exception for all 10 classes. Phase 2 receives 72–91% of Phase 1 gradient signal (corrected run; original raw-logit run showed 71–87% — ordering unchanged).

## Key Findings

1. **No phase is gradient-starved.** Phase 2 receives 71–87% of Phase 1 gradient norm — substantial signal, not near-zero. The concern that Phase 2 might be entirely bypassed during learning is not supported.

2. **Monotonic decreasing pattern.** Phase 1 > Phase 2 > Phase 3 for every class. Earlier phases receive stronger gradient signal, consistent with Phase 1 learning the dominant structural pattern first and later phases refining it.

3. **Timestamp dominates.** Timestamp has the highest absolute gradient norms across all phases (Phase1=0.0947, Phase2=0.0864, Phase3=0.0737) and the highest P2/P1 ratio (91.3%), consistent with it being the class with the strongest model performance and a strong size-based shortcut (EXP-S3).

4. **Phase 2 gradient is meaningful but secondary.** The 72–91% ratio range (mean ~78%) suggests Phase 2 participates in learning but does not drive the loss as strongly as Phase 1. Coherent with EXP-L1 (Phase 2 JK weight=0.322, lowest of three phases) and EXP-L5 (Phase 2 probing adds no linear signal for most classes).

5. **Phase 3 receives least gradient.** Despite having the highest JK attention weight (EXP-L1: 0.346), Phase 3 receives the lowest gradient norm. This reflects the JK aggregation architecture: Phase 3 is upweighted at inference but receives reduced gradient due to its position at the end of the forward path.

## Pass/Fail Analysis

No binary pass criterion was defined for this experiment (diagnostic). All phases receive non-trivial gradient signal — the null hypothesis (Phase 2 is gradient-starved) is rejected.

## Recommended Next Steps

1. Cross-reference Phase 2 gradient ratios with EXP-L5 Phase 2 probing results — classes with higher Phase2/Phase1 ratios (Timestamp=0.867, CallToUnknown=0.825) may show more Phase 2 benefit in probing after a longer training run.
2. Monitor Phase 2 gradient ratio across training epochs in Run 5 to check whether the auxiliary Phase 2 loss (`aux_phase2_loss_weight=0.10`) increases Phase 2 gradient norms.
3. Consider logging phase gradient norms as a training metric to detect gradient collapse early.
