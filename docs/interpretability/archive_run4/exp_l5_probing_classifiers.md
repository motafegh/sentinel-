# EXP-L5: Probing Classifiers Per GNN Phase

**Layer:** 3 — Learning
**Priority:** P1
**Status:** FAIL
**Run date:** 2026-05-31
**Checkpoint:** `ml/checkpoints/GCB-P1-Run4-no-asl-pw_best.pt` (Run 4, ep32, F1=0.3362)
**Script:** `ml/scripts/interpretability/exp_l5_probing_classifiers.py`
**Output:** `ml/logs/interpretability/exp_l5_probing_classifiers/`

---

## Hypothesis

If Phase 2 (CFG/ICFG/DFG layers) adds genuinely new vulnerability signal beyond Phase 1 (structural + CONTAINS), then linear probes trained on Phase 2 embeddings should outperform Phase 1 probes. Specifically, Reentrancy probes should improve by at least 3 percentage points from Phase 1 to Phase 2, since reentrancy is a CFG-level pattern (external call followed by state write).

---

## Method

For each of 500 sampled val-split contracts, GNNEncoder.forward is called with `return_intermediates=True` to obtain per-phase node embeddings at three checkpoints: after Phase 1 (L1+L2), after Phase 2 (L3+L4+L5), and after Phase 3 (L6+L7+L8). Each embedding `[N, 256]` is pooled over function-level nodes (FUNCTION, MODIFIER, FALLBACK, RECEIVE, CONSTRUCTOR; types 1,2,4,5,6) using `torch.cat([func_embs.max(0).values, func_embs.mean(0)], dim=0)` to produce a graph-level vector `[512]`. A logistic regression probe (C=1.0, max_iter=500, lbfgs) is trained on 80% of samples and evaluated on 20%, stratified by class where possible.

**Fix applied (COMPLETENESS audit INCOMPLETE-5):** The original run used mean-only pooling producing `[256]` vectors, which does not match the model's actual pooling scheme. This re-run corrects to max+mean concatenation producing `[512]` vectors. This change substantially affects several classes.

---

## Results

### Probing Classifier F1 Table (3 Phases × 10 Classes)

| Class | Phase1 F1 | Phase2 F1 | Phase3 F1 | Δ P2-P1 | Δ P3-P1 |
|-------|-----------|-----------|-----------|---------|---------|
| CallToUnknown | 0.0500 | 0.0526 | 0.0909 | +0.0026 | +0.0409 |
| DenialOfService | 0.0000 | 0.0000 | 0.0000 | +0.0000 | +0.0000 |
| ExternalBug | 0.0625 | 0.0606 | 0.1714 | -0.0019 | +0.1089 |
| GasException | 0.1644 | 0.1600 | 0.1000 | -0.0044 | -0.0644 |
| IntegerUO | 0.4190 | 0.4486 | 0.5273 | +0.0296 | **+0.1083** |
| MishandledException | 0.0000 | 0.0000 | 0.1667 | +0.0000 | +0.1667 |
| Reentrancy | 0.1702 | 0.1633 | 0.2069 | -0.0069 | +0.0367 |
| Timestamp | 0.0000 | 0.0000 | 0.5000 | +0.0000 | +0.5000 |
| TransactionOrderDependence | 0.0000 | 0.0000 | 0.0541 | +0.0000 | +0.0541 |
| UnusedReturn | 0.0000 | 0.0000 | 0.0000 | +0.0000 | +0.0000 |

### Pass/Fail Check

| Check | Value | Threshold | Result |
|-------|-------|-----------|--------|
| Reentrancy: Phase2 F1 > Phase1 F1 + 3pp | Phase2=0.1633, Phase1=0.1702, Δ=-0.0069 | +0.03 | **FAIL** |

---

## Impact of Pooling Fix

The corrected max+mean [512] pooling substantially changed results for several classes compared to the prior mean-only [256] run:

| Class | Old Phase1 F1 (mean-only) | New Phase1 F1 (max+mean) | Change |
|-------|--------------------------|--------------------------|--------|
| IntegerUO | 0.1143 | **0.4190** | +0.3047 |
| GasException | 0.0000 | **0.1644** | +0.1644 |
| Reentrancy | 0.0000 | **0.1702** | +0.1702 |
| CallToUnknown | 0.1818 | 0.0500 | -0.1318 |

The large gains for IntegerUO and GasException confirm the COMPLETENESS audit finding (INCOMPLETE-5): mean-only pooling was discarding the max-pooled signal that captures peak activation in the most active function nodes — critical for detecting integer operations and gas-intensive patterns. The original probing results for these classes were significantly understated.

---

## Key Findings

1. **Pooling choice is critical:** Max+mean pooling produces substantially different (and more informative) probing results than mean-only. IntegerUO Phase2 F1=0.449 (vs 0.114 with mean-only) confirms that max pooling captures function-level extremes that correlate with integer overflow patterns.

2. **Phase 2 still does not improve Reentrancy:** With the corrected pooling, Reentrancy Phase2 F1 (0.1633) is slightly below Phase1 (0.1702). The CFG/ICFG layers do not create a more linearly separable reentrancy representation. This is consistent with EXP-L1 (Phase 2 lowest JK weight) and EXP-L2 (CFG ablation near-zero effect).

3. **Phase 3 shows consistent improvement for most classes:** Phase 3 improves on Phase 1 for 7 out of 10 classes. IntegerUO (+0.1083), MishandledException (+0.1667), and Timestamp (+0.5000) show the largest gains. Phase 3 (REVERSE_CONTAINS hierarchy) introduces a linearly separable component for these classes.

4. **GNN embeddings remain partially non-linearly encoded:** DenialOfService, UnusedReturn remain at F1=0 across all phases. These classes may require non-linear probes or richer pooling.

5. **Timestamp Phase3 F1=0.500 with small positives:** Treat with caution — only ~3 positive training samples in 500-contract sample. Likely reflects probe memorization.

---

## Caveats

- 500 contracts produces severe class imbalance for rare classes (Timestamp, DenialOfService). Probe results for these classes are unreliable.
- The duplicate `--n-contracts` argument bug in the original script was fixed before running (replaced with `parser.set_defaults`).
- AUROC values for classes with <10 positive test examples should be interpreted cautiously.

---

## Recommended Next Steps

1. Re-run with n_contracts=2000 using the train split for more reliable probe statistics on rare classes.
2. Investigate whether a non-linear probe (MLP) can decode UnusedReturn and DenialOfService, which remain at F1=0 with logistic regression.
3. Consider per-class CFG-level pooling to capture intra-function CFG topology for the Reentrancy probe.
4. Cross-reference IntegerUO Phase3 gain (+0.1083) with EXP-B3 JK weights to confirm Phase 3 dominance for this class.
