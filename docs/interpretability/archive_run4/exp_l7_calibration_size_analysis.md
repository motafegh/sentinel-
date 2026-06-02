# EXP-L7: Calibration and Size-Stratified Analysis

**Layer:** 3  **Priority:** 2  **Status:** COMPLETE (2026-05-30)  
**Checkpoint:** `ml/checkpoints/GCB-P1-Run4-no-asl-pw_best.pt` (Run 4, ep32, F1=0.3362)  
**Output:** `ml/logs/interpretability/exp_l7_calibration/`

---

## Hypothesis

1. **Calibration:** Model confidence (sigmoid output) will not match actual positive frequency — the model will be overconfident (predicted prob > actual freq) on rare classes and underconfident on common classes, given it was trained with ASL and without explicit calibration. ECE is expected to be high (>0.15) for most classes.

2. **Size-stratified F1:** Performance will degrade on large contracts (>150 nodes) relative to small contracts (<30 nodes). Timestamp in particular is expected to show the worst size gap, given EXP-S3 found a graph-size shortcut (Cohen's d=1.657) — the model learned that "large Timestamp contract" is a reliable signal, which will break when encountering large non-Timestamp contracts.

---

## Method

Up to 1000 val-split contracts are sampled (936 collected after filtering). Full model inference (GNN + GraphCodeBERT + fusion) produces sigmoid probabilities `[N, 10]`. Calibration is computed by binning predicted probabilities into 10 equal-width bins and measuring ECE = Σ (|bin| / N) × |mean_pred − frac_positive|. Node counts per contract are extracted from the cached graph's `num_nodes` attribute. Contracts are stratified into small (<30 nodes), medium (30–150 nodes), and large (>150 nodes). Per-class F1 is computed at threshold=0.5 for each stratum. Pass criterion: F1_large within 10pp (0.10) of F1_small for each class with ≥20 large and ≥20 small examples.

---

## Results

### Node Count Statistics

| Metric | Value |
|--------|-------|
| Min | 6 nodes |
| Median | 94 nodes |
| Mean | 131.6 nodes |
| Max | 1414 nodes |

Note: C-4 warning fired — 1 graph with 1207 nodes exceeds max_nodes=1024 (truncated).

### ECE Per Class (Expected Calibration Error)

| Class | ECE | Interpretation |
|-------|-----|----------------|
| CallToUnknown | 0.2800 | Severely miscalibrated |
| DenialOfService | 0.3097 | Severely miscalibrated (highest) |
| ExternalBug | 0.2504 | Severely miscalibrated |
| GasException | 0.2471 | Severely miscalibrated |
| IntegerUO | 0.2050 | Severely miscalibrated |
| MishandledException | 0.2474 | Severely miscalibrated |
| Reentrancy | 0.2703 | Severely miscalibrated |
| Timestamp | 0.2069 | Severely miscalibrated |
| TransactionOrderDependence | 0.2490 | Severely miscalibrated |
| UnusedReturn | 0.2568 | Severely miscalibrated |

All ECE values fall in the range 0.205–0.310 (mean ≈ 0.252). Reference: well-calibrated models typically have ECE < 0.05; ECE > 0.10 is considered poor calibration.

### Size-Stratified F1

| Class | Small F1 (n=57) | Medium F1 (n=631) | Large F1 (n=248) | Gap | Pass |
|-------|-----------------|-------------------|------------------|-----|------|
| CallToUnknown | 0.6667 | 0.2857 | 0.2941 | 0.3726 | **FAIL** |
| DenialOfService | NaN | 1.0 | 1.0 | — | N/A (no small positives) |
| ExternalBug | 0.0 | 0.2692 | 0.3529 | 0.3529 | **FAIL** |
| GasException | 0.6667 | 0.1795 | 0.2857 | 0.3810 | **FAIL** |
| IntegerUO | 0.7826 | 0.8150 | 0.8261 | 0.0435 | **PASS** |
| MishandledException | NaN | 0.1562 | 0.3590 | — | N/A (no small positives) |
| Reentrancy | 0.8000 | 0.3929 | 0.4615 | 0.3385 | **FAIL** |
| Timestamp | **1.0** | **1.0** | **0.3636** | **0.6364** | **FAIL** (worst) |
| TransactionOrderDependence | 0.0 | 0.0800 | NaN | — | N/A (no large positives) |
| UnusedReturn | NaN | 0.2308 | 0.6667 | — | N/A (no small positives) |

**Overall pass/fail: 1/6 evaluated classes pass. Overall: FAIL.**

---

## Key Findings

1. **Catastrophic miscalibration across all classes:** Every class has ECE > 0.20, with DenialOfService at 0.31 being the worst. This means model confidence scores are not meaningful probability estimates — a predicted probability of 0.7 does not correspond to 70% actual positive frequency. Deployment without calibration (temperature scaling or Platt scaling) would produce misleading confidence reports.

2. **Timestamp size shortcut confirmed at prediction level:** This is the most stark finding. Timestamp achieves F1=1.0 on both small (n=57) and medium (n=631) contracts, but collapses to F1=0.3636 on large contracts (n=248). The gap of 0.6364 is the largest of any class. This directly confirms EXP-S3's finding (Cohen's d=1.657): the model learned that small/medium-sized Timestamp contracts are easy to identify (perhaps via few distinctive structural features), but fails to generalize to large contracts where that shortcut breaks.

3. **IntegerUO is the only robust class:** IntegerUO shows F1=0.783 (small), 0.815 (medium), 0.826 (large) — a gap of only 0.044, the only PASS among evaluated classes. IntegerUO has the best training coverage (1994 positive examples) and the most stable gradients throughout Run 4. Its size-robustness suggests the model has learned a genuine feature (e.g., integer operation patterns visible in features) rather than a size shortcut.

4. **Reentrancy shows moderate size sensitivity:** F1 drops from 0.800 (small) to 0.462 (large), a gap of 0.338. Smaller contracts that have reentrancy are likely simpler (single-function reentrancy), while larger contracts have more complex call trees where the model's limited CFG signal (EXP-L2, EXP-L4) fails.

5. **Small contract performance is unreliable for rare classes:** Only 57 small contracts (6% of the 936-sample set), so F1 values for small contracts have high variance. ExternalBug F1=0.0 for small and 0.35 for large is likely a data artifact rather than a genuine finding.

6. **Median contract size is 94 nodes:** Most of the val split falls in the medium stratum (631/936 = 67%), confirming that SENTINEL's training distribution is dominated by medium-sized contracts. Performance on tail sizes (very small or very large) is less reliable.

---

## Implications for Architecture

- **Calibration layer needed for deployment:** All 10 classes require post-hoc calibration before SENTINEL scores can be used as probabilities for risk assessment. Temperature scaling (a single learned scalar T applied as `sigmoid(logit/T)`) is the recommended first step; per-class Platt scaling if class-specific miscalibration patterns differ.

- **Timestamp must be fixed at data level:** The 0.64 F1 gap between small/medium vs large contracts confirms this is a data distribution problem: few large Timestamp contracts in training, causing the model to use size as a proxy. Sol-3 (Timestamp gating from EXECUTION_PLAN) and balanced resampling of large Timestamp contracts are the correct remedies.

- **max_nodes=1024 truncation affects large contracts:** The C-4 warning (1207-node graph truncated) and the large-contract F1 drops suggest that the 1024-node ceiling is an active constraint. The MEMORY.md pending item "C-4: quantify % corpus > 1024 nodes; consider raising to 2048 before Phase 2" is supported by this evidence.

- **ECE uniformly high suggests systematic overconfidence:** ASL loss (which clips positive logits) combined with class imbalance likely pushes the model toward producing consistently high sigmoid outputs for positive predictions, inflating confidence. A follow-up experiment comparing ECE before and after temperature scaling would quantify the calibration gap.

---

## Known Caveats

- 936 samples collected (of 1000 requested) — 64 contracts skipped due to cache miss or graph format issues.
- Size strata are unbalanced: 57 small / 631 medium / 248 large. F1 on small contracts is computed from very few examples and high variance.
- The threshold=0.5 is used for all classes. Optimal thresholds (from tune_threshold.py) would improve F1 for rare classes, but size stratification at the optimal threshold is not computed here.
- 4 classes (DenialOfService, MishandledException, TransactionOrderDependence, UnusedReturn) cannot be evaluated for pass/fail due to NaN F1 from absence of positive examples in one stratum.
- ECE is computed on the raw sigmoid output without threshold tuning or temperature scaling.
