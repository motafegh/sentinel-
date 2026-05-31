# EXP-A3: JK Entropy Logging

**Layer:** 3 — Learning
**Priority:** P1
**Status:** PASS
**Run date:** 2026-05-30
**Script:** `ml/scripts/interpretability/exp_a3_jk_entropy_logging.py`
**Output:** `ml/logs/interpretability/exp_a3_jk_entropy.json`

---

## Purpose

This experiment parses training logs to track the JK (Jumping Knowledge) attention weights across the three GNN phases over the course of training. It validates that the JK aggregator maintains healthy diversity across phases rather than collapsing to a single-phase mode, which would indicate wasted capacity.

## Method

The script parses the Run 4 training log (`graphcodebert-p1-run4-20260525.log`) to extract per-epoch JK phase weight means (P1, P2, P3) and computes the Shannon entropy H = -Σ wᵢ log(wᵢ) over the three phases. A PASS is issued when entropy exceeds 0.50 (max is log(3) ≈ 1.099). It also generates an entropy trend plot and a recommended MLflow logging snippet for real-time monitoring.

## How to Run

```bash
TRANSFORMERS_OFFLINE=1 PYTHONPATH=. ml/.venv/bin/python ml/scripts/interpretability/exp_a3_jk_entropy_logging.py \
  --log-file ml/logs/graphcodebert-p1-run4-20260525.log \
  --out ml/logs/interpretability/exp_a3_jk_entropy.json
```

## Results

Parsed 47 epoch entries from Run 4 training log.

### Key Metrics
| Metric | Value | Pass Threshold | Status |
|--------|-------|---------------|--------|
| Last epoch entropy (ep48) | 1.0953 | > 0.50 | PASS |
| Mean entropy (all epochs) | 1.0973 | > 0.50 | PASS |
| Min entropy (ep37) | 1.0935 | > 0.50 | PASS |
| Max entropy (ep1) | 1.0986 | — | — |

**Phase weight summary (across all 47 epochs):**
- P1 mean: ~0.320 (range 0.303–0.356)
- P2 mean: ~0.321 (range 0.313–0.345)
- P3 mean: ~0.359 (range 0.309–0.381) — slight upward drift in later epochs

Entropy is extremely close to maximum (log(3) = 1.0986) throughout all epochs, confirming near-uniform phase weighting.

## Interpretation

The JK entropy remains near-maximum (1.0935–1.0986 vs theoretical max 1.099) across all 47 training epochs, meaning all three GNN phases contribute almost equally to the final node embeddings. There is no collapse to a single phase — a failure mode seen in Run 2 (Phase 3 dominating at 86.6%). The slight P3 bias in later epochs (P3 reaching 0.381 at ep37 vs ~0.31 for P1) is consistent with Phase 3's CONTAINS/REVERSE_CONTAINS edges being most informative for contract-level hierarchy reasoning. The entropy regularizer (λ=0.005) is working as designed.

## Pass/Fail Analysis

All criteria passed:
- Entropy is consistently near-maximum, confirming the JK aggregator uses all three phases.
- No single-phase collapse occurred (which would manifest as H approaching 0).
- Phase 3 slight bias is architecturally expected (contract-level aggregation).

## Recommended Next Steps

- Add the suggested MLflow logging snippet to `trainer.py` to enable real-time JK monitoring during future runs.
- If a future run shows H < 0.8 for consecutive epochs, increase λ from 0.005.
- Cross-reference with exp_l1 (JK weight analysis) once a checkpoint is available to correlate phase weights with per-class attribution.
