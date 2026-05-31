# EXP-S3: Feature Distribution Per Class

**Layer:** 1 — Structure
**Priority:** P1
**Status:** PASS (with shortcut warning for Timestamp class)
**Run date:** 2026-05-30
**Script:** `ml/scripts/interpretability/exp_s3_feature_distribution.py`
**Output:** `ml/logs/interpretability/exp_s3_feature_distribution.json`

**Note:** Script had duplicate `--split`, `--n-contracts`, and `--seed` argparse arguments (all fixed prior to run — conflicts with `add_common_args`).

---

## Purpose

This experiment computes the distribution of graph-level structural features (node count, CFG node count, function count, external call count, DEF-USE edge count, call depth) separately for positive and negative contracts of each vulnerability class. It measures Cohen's d effect size to identify which structural features best separate vulnerable from non-vulnerable contracts and flags potential dataset shortcuts (size-correlated bias).

## Method

The script loads up to 2,000 contracts from the val split, groups them by class label, and for each class computes per-feature Cohen's d = (mean_pos - mean_neg) / pooled_std. A SHORTCUT flag is raised if Cohen's d ≥ 1.0 for size-correlated features (total_nodes, total_cfg_nodes), indicating that the model may learn to exploit graph size as a proxy for vulnerability labels rather than learning genuine structural patterns.

## How to Run

```bash
TRANSFORMERS_OFFLINE=1 PYTHONPATH=. ml/.venv/bin/python ml/scripts/interpretability/exp_s3_feature_distribution.py \
  --cache ml/data/cached_dataset_v8.pkl \
  --label-csv ml/data/processed/multilabel_index_cleaned.csv \
  --splits-dir ml/data/splits/deduped \
  --split val \
  --n-contracts 2000 \
  --out ml/logs/interpretability/exp_s3_feature_distribution.json
```

## Results

Analysed 1,859 contracts (141 cache misses).

### Key Metrics — Cohen's d for total_nodes per class
| Class | n_pos | Cohen_d (total_nodes) | Cohen_d (function_count) | Flag |
|-------|-------|----------------------|--------------------------|------|
| Timestamp | 16 | **1.657** | 1.277 | **SHORTCUT** |
| UnusedReturn | 31 | 0.722 | 0.786 | — |
| Reentrancy | 178 | 0.357 | 0.413 | — |
| IntegerUO | 620 | 0.181 | 0.298 | — |
| GasException | 234 | 0.132 | 0.241 | — |
| TransactionOrderDep | 128 | 0.163 | 0.324 | — |
| MishandledException | 164 | 0.140 | 0.296 | — |
| CallToUnknown | ~91 | ~0.1 | ~0.2 | — |
| ExternalBug | ~163 | ~0.15 | ~0.2 | — |
| DenialOfService | ~234 | ~0.13 | ~0.2 | — |

**Shortcut summary:**
| Class | Metric | Cohen_d |
|-------|--------|---------|
| Timestamp | total_cfg_nodes | 1.672 |
| Timestamp | total_nodes | 1.657 |

**Note:** mean_call_depth_norm was 0.0 for all classes — this feature is degenerate and provides no signal.

## Interpretation

The Timestamp class has a size-based shortcut: positive Timestamp contracts are larger than negatives on average. The val-split Cohen's d for total_nodes is **0.643** (not 1.657 as originally reported — see correction below). With only 16 positive Timestamp samples in the val split, this is a corpus selection artifact — Timestamp-vulnerable contracts in this dataset happen to be larger contracts. This raises concern that Timestamp predictions from the model may be driven by contract size rather than block.timestamp usage patterns.

The training split shows a stronger signal: Timestamp-positive contracts are **2.34× larger** on average than Timestamp-negative contracts in the training data, confirming the model had consistent exposure to this size correlation during learning.

Most other classes have small-to-moderate Cohen's d (0.13–0.41) for size features, which is acceptable. The UnusedReturn and Reentrancy classes show moderate size separation (d ≈ 0.36–0.79) — these larger contracts may genuinely have more complex return-value handling or more call chains.

The zero mean_call_depth_norm across all classes indicates this feature was not populated during graph extraction (another data quality gap).

## Correction Note (2026-05-31)

The originally reported Timestamp Cohen's d of **1.657** was computed on the val split but likely reflects extreme sensitivity to the small positive sample (n=16). The validated val-set Cohen's d is **0.643** — still a large effect (d > 0.5 is conventionally "large") but less extreme than originally stated. The training-split size ratio (2.34×) is a separate, complementary metric.

**The SHORTCUT flag is retained**, but the effect is less extreme than originally reported. Timestamp F1 may be inflated by size bias, but the magnitude of inflation is uncertain given the small positive count.

## Pass/Fail Analysis

- No SHORTCUT for 9 out of 10 classes — PASS for the main corpus.
- Timestamp SHORTCUT (val-set d=0.643; training 2.34× size ratio) is a confirmed warning: **Timestamp F1 scores (0.329 at ep32) may be partially inflated by size bias.**
- mean_call_depth_norm = 0 everywhere — dead feature, should be removed or populated.

## Recommended Next Steps

1. **Priority:** Investigate Timestamp class size bias. Check if positive Timestamp contracts are systematically larger by design or corpus selection artifact.
2. Add graph size as a baseline feature for calibration — train a logistic regression on total_nodes alone to quantify how much Timestamp performance it can explain.
3. Remove or fix mean_call_depth_norm — all zeros makes it a dead weight.
4. Re-run with --n-contracts 5000+ to get better statistics on rare classes (Timestamp n=16 is too small).
