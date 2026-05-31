# EXP-E4: Direction Sensitivity

**Layer:** 2 — Expressivity
**Priority:** P1
**Status:** FAIL
**Run date:** 2026-05-30
**Script:** `ml/scripts/interpretability/exp_e4_direction_sensitivity.py`
**Output:** `ml/logs/interpretability/exp_e4_direction_sensitivity.json`

---

## Purpose

This experiment tests whether using directed versus undirected graph edges improves the GNN's ability to distinguish reentrancy-positive from reentrancy-negative contracts via WL hashing. If directed edges provide strictly more discriminative information, the difference in WL collision rates (directed - undirected) should be positive and large. This validates the architectural choice to use directed GAT convolutions in SENTINEL.

## Method

For 45 matched positive-negative contract pairs, the script runs two WL hash trajectories: (1) directed WL using the true edge directions, and (2) undirected WL using symmetrized edges (bidirectional). For each of 8 WL rounds, it computes the fraction of pairs that are WL-distinguishable under each setting. Pass criterion: directed distinguishability exceeds undirected by ≥10% at round 8.

## How to Run

```bash
TRANSFORMERS_OFFLINE=1 PYTHONPATH=. ml/.venv/bin/python ml/scripts/interpretability/exp_e4_direction_sensitivity.py \
  --cache ml/data/cached_dataset_v8.pkl \
  --label-csv ml/data/processed/multilabel_index_cleaned.csv \
  --splits-dir ml/data/splits/deduped \
  --split val \
  --n-contracts 50 \
  --out ml/logs/interpretability/exp_e4_direction_sensitivity.json
```

## Results

45 matched reentrancy positive-negative pairs analysed.

### Key Metrics
| Metric | Value | Pass Threshold | Status |
|--------|-------|---------------|--------|
| Directed WL (all rounds) | 88.9% | — | — |
| Undirected WL (all rounds) | 88.9% | — | — |
| Directed - Undirected at r=8 | 0.0% | ≥10% | FAIL |

The WL curves are completely identical across all 8 rounds for both directed and undirected settings.

## Interpretation

The zero difference between directed and undirected WL distinguishability is a significant finding. It means that reversing all edge directions (making the graph undirected) produces exactly the same WL hash behavior for reentrancy contracts. This could indicate: (1) the key structural features for reentrancy distinction are already in node features (type labels) rather than edge directions, (2) the graph is sufficiently symmetric that direction adds no new information, or (3) the CALL_ENTRY/RETURN_TO edges that encode directionality are absent in 24% of reentrancy positives (exp_s4 finding), diluting the directional signal.

The 88.9% base distinguishability (identical for both settings) is high, meaning WL can distinguish nearly all positive-negative pairs regardless of edge direction. The zero direction gain combined with the exp_s2 finding that REVERSE_CONTAINS edges are entirely absent from the corpus is a strong signal that edge direction is currently under-utilized in the SENTINEL architecture.

## Pass/Fail Analysis

- Failed: directed and undirected WL are identical (0% difference).
- This challenges the architectural assumption that directed GAT convolutions are critical for reentrancy detection — at least at the WL structural level.
- However, this does not mean directed edges are useless: the GAT architecture can learn directional attention weights even when WL structural hashes are equivalent.

## Recommended Next Steps

1. After IMP-D1 re-extraction (adding REVERSE_CONTAINS and more ICFG edges), re-run to check if directionality gains increase.
2. Cross-reference with attention rollout (exp_l9) to check whether the trained model's attention weights actually favor directed edges.
3. Consider whether adding DELEGATECALL edge types (directional) would increase the directed-undirected gap.
4. The 88.9% WL distinguishability (regardless of direction) is the key positive finding — the structural content is there, direction just doesn't add extra discriminative power at the WL level.
