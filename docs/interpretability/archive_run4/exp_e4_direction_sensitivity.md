# EXP-E4: Direction Sensitivity

**Layer:** 2 — Expressivity
**Priority:** P1
**Status:** FAIL
**Run date:** 2026-05-31
**Script:** `ml/scripts/interpretability/exp_e4_direction_sensitivity.py`
**Output:** `ml/logs/interpretability/exp_e4_direction_sensitivity.json`

---

## Purpose

This experiment tests whether using directed versus undirected graph edges improves the GNN's ability to distinguish reentrancy-positive from reentrancy-negative contracts via WL hashing. This validates the architectural choice to use directed GAT convolutions in SENTINEL. The original run (2026-05-30) tested CONTROL_FLOW only; this re-run extends to all 4 Phase 2 edge types (COMPLETENESS audit fix).

## Method

For 92 matched reentrancy positive-negative pairs, the script runs two WL hash trajectories: (1) directed WL using the true edge directions, and (2) undirected WL using symmetrized edges (bidirectional). The test is repeated for each of the 4 Phase 2 edge types independently. Pass criterion: directed distinguishability exceeds undirected by ≥10% at round 8 for at least one edge type.

**Fix applied (COMPLETENESS audit):** The original run tested CONTROL_FLOW direction only. This re-run adds DEF_USE (id=10), CALL_ENTRY (id=8), and RETURN_TO (id=9).

## How to Run

```bash
TRANSFORMERS_OFFLINE=1 PYTHONPATH=. ml/.venv/bin/python ml/scripts/interpretability/exp_e4_direction_sensitivity.py \
  --cache ml/data/cached_dataset_v8.pkl \
  --label-csv ml/data/processed/multilabel_index_cleaned.csv \
  --splits-dir ml/data/splits/deduped \
  --split val \
  --n-contracts 100 \
  --out ml/logs/interpretability/exp_e4_direction_sensitivity.json
```

## Results

92 matched reentrancy positive-negative pairs analysed.

### Key Metrics — Directed vs Undirected WL Distinguishability

| Edge Type | ID | Directed | Undirected | Diff | Status |
|-----------|----|----------|------------|------|--------|
| CONTROL_FLOW | 6 | 89.1% | 89.1% | 0.0% | FAIL |
| DEF_USE | 10 | 89.1% | 89.1% | 0.0% | FAIL |
| CALL_ENTRY | 8 | 89.1% | 89.1% | 0.0% | FAIL |
| RETURN_TO | 9 | 89.1% | 89.1% | 0.0% | FAIL |

The WL curves are completely identical across all 8 rounds for both directed and undirected settings, for all 4 edge types tested.

## Interpretation

The original finding — "direction adds no discriminative power" — was based on CONTROL_FLOW only. This re-run confirms the same result for all four Phase 2 edge types: DEF_USE, CALL_ENTRY, and RETURN_TO each produce zero difference between directed and undirected WL distinguishability.

The 89.1% base WL distinguishability is unchanged — structural content is high regardless of direction. This means the key discriminative information for reentrancy is carried in node feature distributions (types, counts) rather than in the specific direction of edges. Making any of the 4 Phase 2 edge types undirected does not reduce the ability to tell reentrancy positives from negatives.

This does not mean directed edges are architecturally useless: GAT convolutions can learn directional attention weights even when WL structural hashes are equivalent. However, at the structural topology level, direction is not adding measurable signal.

## Pass/Fail Analysis

- All 4 edge types: directed and undirected WL are identical (0% difference).
- Pass criterion (≥10% directed advantage at r=8 for any edge type): FAIL.
- The finding generalises the original CONTROL_FLOW result to the full Phase 2 edge set.

## Recommended Next Steps

1. After IMP-D1 re-extraction (adding more ICFG edges), re-run to check whether additional edge coverage changes the directed-undirected gap.
2. Cross-reference with EXP-L9 attention rollout to check whether the trained model's GAT attention weights favour directed patterns even where WL does not distinguish them.
3. Consider whether adding DELEGATECALL edge types (directional cross-contract calls) would increase the directed-undirected gap for reentrancy.
4. The 89.1% WL distinguishability regardless of direction remains the key positive finding — structural content is sufficient for discrimination even without directional signal.
