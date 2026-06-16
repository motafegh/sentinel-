# Deep Dive: Conformal Prediction for SENTINEL

**Date:** 2026-06-16
**Module:** `ml`
**Phase:** Run 14 preparation
**Tier:** 1 (highest priority)
**Proposal Reference:** §5.1 of `2026-06-15_ml_Run13_prep_proposal_next_gen_ml_methods.md`

---

## 1. Full Method Explanation

### 1.1 What Is Conformal Prediction?

Conformal prediction (CP) is a framework for constructing prediction sets with **finite-sample coverage guarantees**. Given a significance level α (e.g., 0.05), CP produces a set C(X) such that:

```
P(Y ∈ C(X)) ≥ 1 - α
```

This means: for 95% of test samples, the true label will be in the prediction set. The guarantee is **distribution-free** — it holds regardless of the underlying data distribution, model, or loss function.

### 1.2 Split Conformal Prediction (SCP)

The most practical variant for deployed models. Uses a held-out calibration set:

1. **Calibration phase:** Run the trained model on a held-out calibration set. For each sample, compute a non-conformity score — typically `1 - p(y_true)` (the model's confidence in the true label).

2. **Threshold computation:** Find the (1-α) quantile of all non-conformity scores. This is the threshold q above which predictions are "non-conforming."

3. **Inference:** For a new sample, include all classes whose non-conformity score ≤ q. This produces a prediction set.

### 1.3 Key Properties

- **No retraining required** — post-hoc on existing model
- **Distribution-free** — works for any model, any data distribution
- **Finite-sample guarantee** — holds exactly for finite calibration sets (not asymptotic)
- **Valid for multi-label** — each class independently checked against threshold
- **Calibration set size matters** — larger = tighter sets; 500+ samples recommended

### 1.4 Non-Conformity Scores for Multi-Label

For multi-label classification (SENTINEL's use case), common non-conformity scores:

- **Margin-based:** `1 - p(y_true)` for each true class, take max
- **Inverse probability:** `1 - p(y_c)` for each class c, include c if score ≤ q
- **Normalized:** Divide by model calibration to account for varying difficulty

### 1.5 Relationship to Existing SENTINEL Components

SENTINEL already has:
- **Per-class thresholds** (tuned post-hoc via grid search in `tune_threshold.py`)
- **3-tier suspicion output** (CONFIRMED ≥ 0.55, SUSPICIOUS ≥ 0.25, NOTEWORTHY < 0.25)

CP **complements** these by adding:
- A **formal guarantee** on coverage (not just heuristic thresholds)
- **Adaptive prediction sets** — harder samples get larger sets
- **Calibration-aware thresholds** — derived from actual model behavior, not grid search

---

## 2. SENTINEL-Specific Audit

### 2.1 Current State of Uncertainty in SENTINEL

From `predictor.py:660-760` (`_format_result`):
- Output is raw sigmoid probabilities with per-class thresholds
- No distinction between epistemic (model uncertainty) and aleatoric (data ambiguity)
- A p=0.55 prediction could be a confident boundary case or a wild guess on OOD input

From `api.py:167-192` (`PredictResponse`):
- Three-tier system: CONFIRMED/SUSPICIOUS/NOTEWORTHY
- Thresholds are static, per-class, loaded from JSON
- No coverage guarantee — a class with threshold 0.55 could have 50% or 90% true positive rate

### 2.2 Where CP Fits in the Inference Pipeline

```
Source Code → Preprocessor → Model Forward Pass → Sigmoid → Per-Class Probs
                                                                    ↓
                                                    Current: Per-class thresholds → tiers
                                                    + CP: Prediction sets with coverage
```

CP operates on the **output of the sigmoid**, after the model forward pass. It does NOT modify the model, training, or preprocessing. This is its key advantage: zero-risk integration.

### 2.3 Calibration Set Requirements

SENTINEL has 1,983 validation samples (v3 split). CP needs a **held-out calibration set** — this could be:
- The existing validation split (1,983 samples) — used for both calibration and threshold tuning
- A dedicated calibration split carved from validation (e.g., 500 for calibration, 1,483 for threshold tuning)
- The 66 honest OOD contracts — but too small for reliable calibration (need 500+)

**Recommendation:** Use the full validation split for calibration. The per-class thresholds are already tuned on this data; CP adds a layer of formal guarantees on top.

### 2.4 Integration Points

**predictor.py** — `_score_windowed` method (line 583):
```python
# After sigmoid computation (line 638):
probs = torch.sigmoid(logits.float()).squeeze(0)   # [num_classes]
# CP: compute prediction set from probs using calibration threshold
pred_set = set(i for i, p in enumerate(probs) if 1 - p <= q_threshold)
```

**predictor.py** — `_format_result` method (line 660):
```python
# Add to result dict:
"prediction_set": pred_set,  # set of class indices in the CP set
"cp_coverage": 1 - alpha,     # coverage guarantee (e.g., 0.95)
"cp_threshold": q_threshold,  # calibrated threshold
```

**api.py** — `PredictResponse` (line 167):
```python
# Add fields:
prediction_set: list[str] = Field(default_factory=list, description="CP prediction set — classes with formal 95% coverage guarantee")
cp_coverage: float = Field(default=0.95, description="CP coverage guarantee")
```

### 2.5 Critical Considerations

1. **Calibration set contamination:** The 17.4% contamination rate in SmartBugs Wild means the calibration set may contain in-distribution samples. This doesn't break the guarantee (CP is valid for any distribution), but it may produce tighter sets than warranted for OOD inputs.

2. **Multi-label vs multi-class:** CP for multi-label requires per-class non-conformity scores. Each class is independently calibrated. The prediction set is the union of all classes that pass the threshold.

3. **Coverage vs. set size trade-off:** Lower α (e.g., 0.01) → larger prediction sets but higher coverage. Higher α (e.g., 0.10) → smaller sets but lower coverage. For security-critical applications, α=0.05 (95% coverage) is standard.

4. **Class imbalance effect:** Rare classes (DoS: 243 positives) will have wider calibration distributions → larger prediction sets → less informative. CP honestly reflects this: "I'm less certain about rare classes."

5. **OOD detection:** CP prediction sets for OOD contracts will be larger (more classes included) than for in-distribution contracts. This is a feature, not a bug — it honestly signals uncertainty.

---

## 3. Implementation Plan

### 3.1 Phase A: Core CP Implementation (~1 day)

**Files to modify:**
- `ml/src/inference/predictor.py` — add CP calibration and prediction set computation
- `ml/src/inference/api.py` — extend PredictResponse schema

**Steps:**
1. Create `ml/src/inference/conformal.py` with `ConformalPredictor` class
2. Implement calibration: `fit(calibration_loader, model)` → compute per-class thresholds
3. Implement prediction: `predict(probs)` → prediction set
4. Integrate into `predictor.py` `_score_windowed` method
5. Extend `PredictResponse` in `api.py`

### 3.2 Phase B: Calibration Script (~0.5 day)

**New file:** `ml/scripts/audit/calibrate_conformal.py`

1. Load Run 12 checkpoint
2. Run inference on validation split (1,983 samples)
3. Compute per-class non-conformity scores
4. Find (1-α) quantile thresholds
5. Save to `ml/calibration/run12/conformal_thresholds.json`

### 3.3 Phase C: Validation (~0.5 day)

**Verification:**
1. Run CP on 66 honest OOD contracts
2. Verify coverage ≥ 95% (prediction sets contain true labels ≥ 95% of the time)
3. Measure average prediction set size
4. Compare with per-class threshold coverage
5. Document results in audit report

### 3.4 Expected Outcomes

- **Coverage guarantee:** 95% of test samples have true label in prediction set
- **Set size:** Average 1.5-3 classes per prediction (for 9-class problem)
- **Latency:** +0.1ms per prediction (negligible — just threshold comparison)
- **No retraining:** Zero GPU cost for integration

### 3.5 Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Calibration set too small | Low (1,983 samples) | Medium | Use full validation split |
| Contamination inflates coverage | Low | Low | CP is valid for any distribution |
| Rare classes have wide sets | Expected | Low | Honestly reflects uncertainty |
| OOD contracts get large sets | Expected | Low | Feature, not bug — signals uncertainty |

---

## 4. References

- **CONFIDE** (2026): `arXiv:2604.08885` — Conformal prediction for fine-tuned transformers
- **ECP** (2024): `PMLR 230:466-489` — Evidential conformal prediction with Dirichlet non-conformity
- **Romano et al.** (2020): "Conformalized Quantile Regression" — foundational SCP paper
- **Lei & Wasserman** (2014): "Distribution-Free Predictive Inference For Regression" — theoretical foundation
