# Run 9 Calibration Report
**Checkpoint:** `ml/checkpoints/GCB-P1-Run9-v11-20260606_best.pt` (ep52, fixed F1=0.2965)  
**Date:** 2026-06-08  
**Val set:** `ml/data/splits/deduped/val_indices.npy` — 6,236 samples  
**Schema:** v9 (NODE_FEATURE_DIM=12, NUM_EDGE_TYPES=12)

---

## 1. Threshold Tuning (`GCB-P1-Run9-v11-20260606_thresholds.json`)

Grid swept 0.05–0.95 step 0.025 (37 points per class) on val probabilities.

| Class | Threshold | F1 | Precision | Recall | Support |
|-------|-----------|-----|-----------|--------|---------|
| CallToUnknown | 0.325 | 0.2450 | 0.1564 | 0.5650 | 469 |
| DenialOfService | 0.375 | 0.2128 | 0.2632 | 0.1786 | 56 |
| ExternalBug | 0.325 | 0.2489 | 0.1685 | 0.4758 | 475 |
| GasException | 0.325 | 0.3471 | 0.2414 | 0.6180 | 746 |
| **IntegerUO** | **0.325** | **0.6277** | **0.5560** | **0.7205** | 2025 |
| MishandledException | 0.325 | 0.2864 | 0.1921 | 0.5629 | 604 |
| Reentrancy | 0.350 | 0.3019 | 0.2111 | 0.5299 | 670 |
| Timestamp | 0.325 | 0.2222 | 0.1705 | 0.3188 | 138 |
| TransactionOrderDependence | 0.300 | 0.2284 | 0.1412 | 0.5969 | 459 |
| UnusedReturn | 0.325 | 0.2248 | 0.1512 | 0.4381 | 404 |

**Overall (val):** F1-macro=0.2945 · F1-micro=0.3527 · Hamming=0.2126 · Exact-match=41.6%

**Notes:**
- All thresholds cluster at 0.30–0.375 — model needs aggressive thresholding; default 0.5 would miss most positives
- DenialOfService support=56 (only 0.9% of val) — weak precision (0.263) but that's the best achievable
- Thresholds tuned on *raw* (pre-temperature) probabilities — use these OR temperature scaling, not both

---

## 2. Temperature Calibration (`temperatures_run9.json`, `temperatures_run9_stats.json`)

Per-class LBFGS temperature fitting on val logits. All temperatures < 1.0 → model is **overconfident** (ASL training pushes probs high, consistent with this).

| Class | T | ECE before | ECE after | Status |
|-------|---|-----------|-----------|--------|
| CallToUnknown | 0.2697 | 0.2546 | 0.0356 | ✓ |
| DenialOfService | 0.1481 | 0.3154 | 0.0010 | ✓ |
| ExternalBug | 0.2703 | 0.2427 | 0.0160 | ✓ |
| GasException | 0.2823 | 0.2312 | 0.0366 | ✓ |
| IntegerUO | 0.5214 | 0.1183 | 0.0536 | ⚠ slightly >0.05 |
| MishandledException | 0.2655 | 0.2417 | 0.0254 | ✓ |
| Reentrancy | 0.2675 | 0.2537 | 0.0466 | ✓ |
| Timestamp | 0.3512 | 0.2182 | 0.0372 | ✓ |
| TransactionOrderDependence | 0.2662 | 0.2478 | 0.0254 | ✓ |
| UnusedReturn | 0.2666 | 0.2480 | 0.0127 | ✓ |

**Mean ECE:** 0.2372 → 0.0290 (**87.8% reduction**)  
9/10 classes meet the <0.05 ECE target. IntegerUO at 0.054 is marginal.

---

## 3. Script Fixes Applied (2026-06-08)

Both scripts had alignment bugs with Run 9 that were fixed before running:

| File | Fix |
|------|-----|
| `tune_threshold.py` | `load_model_from_checkpoint`: added `drop_complexity_feature`, `appnp_alpha`; fixed `fusion_max_nodes` default 1024→2048; updated defaults for label-csv, splits-dir, cache |
| `interpretability/utils.py` | `load_model`: added `drop_complexity_feature`, `appnp_alpha`, `fusion_max_nodes`; updated `add_common_args` defaults to v9 paths; fixed `_MAX_TYPE_ID` comment |

These were critical — without them the model ran with `drop_complexity=False` (feat[5] not zeroed), `appnp_alpha=0.0` (no teleport), and `max_nodes=1024` (node truncation), all differing from training.

---

## 4. Usage Notes

- **For max F1:** use `GCB-P1-Run9-v11-20260606_thresholds.json` (no temperature scaling)
- **For calibrated probabilities:** apply `temperatures_run9.json` to raw logits first, then use 0.5 threshold
- **Do NOT combine:** the decision thresholds were tuned on raw probs; applying temperatures changes the prob distribution, making those thresholds incorrect
