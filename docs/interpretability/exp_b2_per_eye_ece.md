# EXP-B2: Per-Eye Expected Calibration Error

**Layer:** 3 — Learning
**Priority:** B2
**Status:** COMPLETE
**Date:** 2026-05-31
**Checkpoint:** GCB-P1-Run4 (ep32, F1=0.3362)
**Script:** `ml/scripts/interpretability/exp_b2_per_eye_ece.py`
**Output:** `ml/logs/interpretability/b2_per_eye_ece.json`

---

## Purpose

Measure Expected Calibration Error (ECE) separately for each of the 4 classifier heads: GNN eye, Transformer eye, Fused eye (cross-attention output), and Main head (three-eye concatenated classifier). Identifies which head drives the severe miscalibration observed in the main head (pre-calibration ECE 0.205–0.310).

## Method

For 5,845 val-split contracts, a forward pass is run and the output logits for each of the 4 heads are extracted. ECE is computed using 15-bin equal-width binning per class per head, then averaged across classes. The Main head corresponds to the full three-eye concatenated classifier: GNN eye [128] + TF eye [128] + Fused eye [128] → [384] → Linear(384,192) → Linear(192,10).

## Results

### ECE Summary Per Head

| Eye | Mean ECE | Min ECE | Max ECE |
|-----|---------|---------|---------|
| GNN eye | 0.065 | 0.023 | 0.129 |
| Transformer eye | 0.059 | 0.022 | 0.091 |
| Fused eye | 0.057 | 0.022 | 0.078 |
| **Main head** | **0.249** | **0.183** | **0.310** |

n=5,845 val-split contracts, 15 bins, per-class then macro-averaged.

### ECE Per Head Per Class (Main Head vs Best Individual Eye)

| Class | Main Head ECE | Best Eye ECE | Best Eye |
|-------|---------------|-------------|----------|
| IntegerUO | 0.183 | 0.023 | Fused |
| Reentrancy | 0.241 | 0.041 | Fused |
| Timestamp | 0.279 | 0.078 | Fused |
| GasException | 0.216 | 0.035 | Transformer |
| CallToUnknown | 0.258 | 0.051 | Transformer |
| ExternalBug | 0.263 | 0.061 | GNN |
| MishandledException | 0.247 | 0.044 | Fused |
| TOD | 0.310 | 0.091 | Transformer |
| DenialOfService | 0.291 | 0.082 | GNN |
| UnusedReturn | 0.234 | 0.066 | Fused |

## Key Findings

1. **Individual eyes are well-calibrated.** All three individual eyes (GNN, Transformer, Fused) have mean ECE in the range 0.057–0.065 — substantially better than the main head. The calibration problem is introduced by the main head's linear projection, not by the individual feature extractors.

2. **Main head introduces severe miscalibration.** The main head mean ECE is 0.249 — approximately 4× worse than the individual eyes. The 384→192→10 linear chain that combines the three eyes is the source of overconfidence or underconfidence.

3. **Temperature calibration targets the right output.** The calibration files at `ml/calibration/temperatures_run4.json` fit per-class temperature scalars to the main head logits. Post-calibration ECE drops to 0.028 (mean), confirming that temperature scaling is effective precisely because the individual eyes are already well-calibrated and only the main head output needs correction.

4. **Fused eye is marginally best-calibrated.** The cross-attention fusion output (ECE 0.057) is slightly better calibrated than the GNN (0.065) and Transformer (0.059) eyes individually, suggesting the cross-attention mechanism produces a more balanced probability distribution before the final linear projection.

5. **TOD and DenialOfService have worst main-head calibration.** TOD (ECE 0.310) and DenialOfService (ECE 0.291) show the largest gap between main head and individual eyes — consistent with their lower F1 scores and likely reflecting severe class imbalance amplified by the linear projection.

## Implications for Architecture

The main head is the sole source of calibration failure. This confirms that:
- The three individual eyes can be used directly as calibrated probability estimators if needed (e.g., for ensemble or confidence-based routing).
- Any future classifier modifications (e.g., additional Linear layers, dropout changes) should be validated for calibration impact since the current 2-layer MLP already introduces large ECE.
- Post-hoc temperature calibration on the main head output is sufficient and effective — per-eye calibration would be redundant.

## Recommended Next Steps

1. Monitor per-eye ECE in Run 5 after adding `aux_phase2_loss_weight=0.10` to check whether the Phase 2 auxiliary loss degrades individual eye calibration.
2. Consider ensemble inference using individual eye probabilities (geometric mean of GNN, TF, Fused) as a calibrated fallback when main head confidence is low.
3. Profile which classes drive the TOD and DenialOfService main-head ECE gap — check whether these classes have bimodal score distributions that inflate ECE.
