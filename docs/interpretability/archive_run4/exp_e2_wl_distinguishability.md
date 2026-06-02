# EXP-E2: WL Distinguishability

**Layer:** 2 — Expressivity
**Priority:** P0
**Status:** PASS
**Run date:** 2026-05-30
**Script:** `ml/scripts/interpretability/exp_e2_wl_distinguishability.py`
**Output:** `ml/logs/interpretability/exp_e2_wl_distinguishability.json`

---

## Purpose

This experiment tests whether the Weisfeiler-Lehman (WL) graph isomorphism test can distinguish positive contracts (vulnerable) from negative contracts (clean) for each vulnerability class. If WL cannot distinguish the graphs after k rounds, no 1-WL-equivalent GNN (including the SENTINEL GNN) can either — making this a hard theoretical lower bound on GNN expressivity for the task.

## Method

For each of 4 classes (Reentrancy, IntegerUO, Timestamp, CallToUnknown), the script samples matched positive-negative pairs and runs typed-directed WL hashing for 1–8 rounds. A pair is "distinguishable" if the WL hash differs between positive and negative. The percentage of non-distinguishable pairs (collision rate) is reported per round. Pass criterion: collision rate < some threshold (the experiment uses collision rate to judge — lower is better, meaning WL can distinguish more pairs).

## How to Run

```bash
TRANSFORMERS_OFFLINE=1 PYTHONPATH=. ml/.venv/bin/python ml/scripts/interpretability/exp_e2_wl_distinguishability.py \
  --cache ml/data/cached_dataset_v8.pkl \
  --label-csv ml/data/processed/multilabel_index_cleaned.csv \
  --splits-dir ml/data/splits/deduped \
  --split val \
  --n-contracts 50 \
  --out ml/logs/interpretability/exp_e2_wl_distinguishability.json
```

## Results

Formed matched positive-negative pairs per class: Reentrancy=45, IntegerUO=44, Timestamp=44, CallToUnknown=45.

### Key Metrics — WL collision rate (%) at each round
| Class | r1 | r2 | r3 | r4 | r5 | r6 | r7 | r8 | Status |
|-------|----|----|----|----|----|----|----|----|--------|
| Reentrancy | 11.1 | 11.1 | 11.1 | 11.1 | 11.1 | 11.1 | 11.1 | 11.1 | PASS |
| IntegerUO | 4.5 | 4.5 | 4.5 | 4.5 | 4.5 | 4.5 | 4.5 | 4.5 | PASS |
| Timestamp | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | PASS |
| CallToUnknown | 6.7 | 6.7 | 6.7 | 6.7 | 6.7 | 6.7 | 6.7 | 6.7 | PASS |

All 4 classes passed.

## Interpretation

WL hashing can distinguish positive from negative contracts for all tested classes. The Timestamp class has a 0% collision rate — every positive-negative pair is WL-distinguishable even at r=1. Combined with the exp_s3 finding that Timestamp contracts are significantly larger, this perfect distinguishability may be driven by graph size differences rather than genuine vulnerability-specific structural patterns.

The Reentrancy class has the highest collision rate (11.1%), meaning ~1 in 9 positive-negative pairs have identical WL hashes — the GNN cannot theoretically distinguish these pairs no matter how deep it goes. These ~5 non-distinguishable reentrancy pairs represent a hard upper bound on reentrancy recall.

Importantly, the WL collision rate stabilizes after r=1 (no change from r1 to r8), suggesting the graphs are fully determined at their 1-hop neighborhood structure — additional GNN layers don't bring new expressivity for the WL measure on these graphs.

## Pass/Fail Analysis

All 4 classes passed. This confirms:
- The SENTINEL graph representation contains sufficient structural diversity that WL can distinguish vulnerable from non-vulnerable contracts.
- The GNN's 8-layer depth is theoretically sufficient for the distinguishable pairs.
- Theoretical maximum precision is bounded by the non-colliding fraction (100% - collision_rate).

## Recommended Next Steps

- Run with more classes (DenialOfService, ExternalBug, etc.) to get full coverage.
- Investigate the 11.1% non-distinguishable Reentrancy pairs — do they correspond to model prediction failures?
- Cross-reference Timestamp 0% collision rate with the exp_s3 size-shortcut finding — check if WL distinguishability is driven by node count alone.
- Test with higher n-contracts (200+) for statistical stability.
