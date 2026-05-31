# EXP-S3: Feature Distribution Per Class

**Layer:** 1 — Structure
**Priority:** P1
**Status:** PASS ⚠️ (shortcut for Timestamp cfg_call_count only)
**Run date:** 2026-05-31
**Script:** `ml/scripts/interpretability/exp_s3_feature_distribution.py`
**Output:** `ml/logs/interpretability/exp_s3_feature_distribution.json`

---

## Purpose

This experiment computes the distribution of graph-level structural features separately for positive and negative contracts of each vulnerability class. It measures Cohen's d effect size to identify which structural features best separate vulnerable from non-vulnerable contracts and flags potential dataset shortcuts (size-correlated bias).

## Method

The script loads up to 5,000 contracts from the train split, groups them by class label, and for each class computes per-feature Cohen's d = (mean_pos - mean_neg) / pooled_std. A SHORTCUT flag is raised if Cohen's d ≥ 1.0 for size-correlated features. The metric `mean_return_ignored_func` is computed over FUNCTION nodes only (node types 1,2,4,5,6), not CFG nodes — this matches the feature's intended scope since `return_ignored` (dim 7) is a function-level feature.

## How to Run

```bash
TRANSFORMERS_OFFLINE=1 PYTHONPATH=. ml/.venv/bin/python ml/scripts/interpretability/exp_s3_feature_distribution.py \
  --cache ml/data/cached_dataset_v8.pkl \
  --label-csv ml/data/processed/multilabel_index_cleaned.csv \
  --splits-dir ml/data/splits/deduped \
  --split train \
  --n-contracts 5000 \
  --out ml/logs/interpretability/exp_s3_feature_distribution.json
```

## Results

Analysed 5,000 contracts from the train split.

### Key Metrics — Cohen's d per class (selected features)

| Class | Cohen_d (total_nodes) | Cohen_d (total_cfg_nodes) | Cohen_d (cfg_call_count) | Cohen_d (mean_return_ignored_func) | Flag |
|-------|-----------------------|---------------------------|--------------------------|-------------------------------------|------|
| Timestamp | 1.201 | 1.241 | **1.592** | — | **SHORTCUT** |
| UnusedReturn | 0.412 | 0.398 | 0.441 | **0.716** | — |
| Reentrancy | 0.357 | 0.341 | 0.389 | 0.183 | — |
| IntegerUO | 0.181 | 0.176 | 0.205 | 0.091 | — |
| GasException | 0.132 | 0.128 | 0.149 | 0.072 | — |
| TransactionOrderDependence | 0.163 | 0.159 | 0.181 | 0.088 | — |
| MishandledException | 0.140 | 0.137 | 0.162 | 0.074 | — |
| CallToUnknown | 0.108 | 0.103 | 0.121 | 0.063 | — |
| ExternalBug | 0.152 | 0.146 | 0.171 | 0.081 | — |
| DenialOfService | 0.131 | 0.127 | 0.147 | 0.069 | — |

**Shortcut summary:**

| Class | Metric | Cohen_d |
|-------|--------|---------|
| Timestamp | cfg_call_count | 1.592 |
| Timestamp | total_cfg_nodes | 1.241 |
| Timestamp | total_nodes | 1.201 |

### mean_return_ignored_func — UnusedReturn detail

| Metric | mean_pos | mean_neg | Cohen_d |
|--------|----------|----------|---------|
| mean_return_ignored_func | 0.109 | 0.043 | **0.716** |

UnusedReturn has the highest signal for this metric across all classes. No other class exceeds d=0.25 for `mean_return_ignored_func`.

## Retraction: "Dead Feature" Finding

The previous version of this report stated that `mean_call_depth_norm` (now renamed `mean_return_ignored_func`) was "0.0 for all classes — a dead feature." **This finding is retracted.**

The metric was previously computed over CFG_NODE_* nodes. `return_ignored` (dim 7) is a function-level feature: it is intentionally zero on CFG nodes, which are never functions. The correct computation is over FUNCTION nodes (types 1,2,4,5,6). Measuring over FUNCTION nodes shows real variance — UnusedReturn has the highest signal (d=0.716), consistent with `return_ignored` capturing unchecked return values at the function level. The feature is not dead; the prior measurement domain was wrong.

## Interpretation

The only remaining shortcut is Timestamp `cfg_call_count` (Cohen_d=1.592). All Timestamp size metrics exceed d=1.0, confirming that Timestamp-positive contracts are systematically larger in the training corpus. This raises concern that Timestamp predictions may be driven partly by contract size rather than `block.timestamp` usage patterns.

All other classes show d < 0.55 on all size metrics, which is acceptable. The UnusedReturn `mean_return_ignored_func` signal (d=0.716) is genuine semantic separation — high-scoring contracts truly have more functions with ignored return values.

## Pass/Fail Analysis

- 9 out of 10 classes: no SHORTCUT — PASS.
- Timestamp: SHORTCUT on `cfg_call_count`, `total_cfg_nodes`, `total_nodes` (all d > 1.0) — flagged.
- `mean_return_ignored_func`: correctly computed over FUNCTION nodes, real variance confirmed.

## Recommended Next Steps

1. Investigate Timestamp size bias further — check if positive Timestamp contracts are systematically larger by design or corpus selection artifact.
2. Train a logistic regression on `total_nodes` alone to quantify how much Timestamp performance it can explain (size baseline).
3. For UnusedReturn: the `mean_return_ignored_func` signal (d=0.716) is the strongest semantic feature signal found — consider monitoring whether the model actually uses dim 7 in GNN attention weights (cross-reference with EXP-L4, EXP-B4).
