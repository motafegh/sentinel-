# EXP-E1: K-Hop Receptive Field

**Layer:** 2 — Expressivity
**Priority:** P0
**Status:** FAIL
**Run date:** 2026-05-31
**Script:** `ml/scripts/interpretability/exp_e1_receptive_field.py`
**Output:** `ml/logs/interpretability/exp_e1_receptive_field.json`

---

## Purpose

This experiment measures the k-hop receptive field of the GNN architecture by checking what fraction of structurally relevant nodes can reach each other within k hops for k=1–8. It validates whether SENTINEL's 8-layer GNN is theoretically deep enough to propagate information across the graph structures in the corpus. Three analyses are run: (1) CEI reachability for reentrancy, (2) Phase 3 FUNCTION→CFG aggregation coverage, (3) CALLS connectivity.

## Method

For each analysis, the script builds the adjacency structure from cached graphs using the relevant edge types for each phase, then computes the fraction of target node pairs reachable within k hops using BFS/sparse matrix powers. PHASE2_EDGE_TYPES now includes DEF_USE (id=10), which was absent in the original run (COMPLETENESS audit fix). Pass criteria: ≥50% of reentrancy-positive contracts have CEI-chain nodes mutually reachable within k=8 hops (A1); ≥80% FUNCTION→CFG coverage via CONTAINS (A2); CALLS connectivity (A3).

**Fix applied (COMPLETENESS audit):** DEF_USE (id=10) was absent from PHASE2_EDGE_TYPES in the original run. This re-run corrects the edge set.

## How to Run

```bash
TRANSFORMERS_OFFLINE=1 PYTHONPATH=. ml/.venv/bin/python ml/scripts/interpretability/exp_e1_receptive_field.py \
  --cache ml/data/cached_dataset_v8.pkl \
  --label-csv ml/data/processed/multilabel_index_cleaned.csv \
  --splits-dir ml/data/splits/deduped \
  --split val \
  --n-contracts 200 \
  --out ml/logs/interpretability/exp_e1_receptive_field.json
```

## Results

199 reentrancy positives, 187 negatives, 7,692 FUNCTION nodes sampled.

### Key Metrics

| Analysis | Metric | Value | Pass Threshold | Status |
|----------|--------|-------|----------------|--------|
| A1 | CEI reachability at k=8 (reentrancy pos) | **38.2%** | ≥50% | FAIL |
| A1 | CEI reachability at k=8 (reentrancy neg) | 27.3% | — | — |
| A2 | FUNCTION→CFG coverage via CONTAINS | **85.5%** | ≥80% | PASS |
| A3 | CALLS connectivity | 22.6% | — | — |

### A1 Detailed — CEI reachability per k

| k | Positive rate | Negative rate |
|---|---------------|---------------|
| 1 | 17.6% | 17.1% |
| 2 | 29.6% | 23.0% |
| 3 | 32.2% | 23.0% |
| 4 | 35.7% | 25.7% |
| 5 | 36.2% | 26.2% |
| 6 | 36.2% | 26.7% |
| 7 | 37.7% | 26.7% |
| 8 | 38.2% | 27.3% |

## DEF_USE Correction Note

DEF_USE (id=10) was absent from PHASE2_EDGE_TYPES in the original run (COMPLETENESS audit finding INCOMPLETE-2). Re-running with the corrected edge set shows k=8 positive rate of 38.2% vs prior 37.7% — a minor improvement of +0.5pp. This confirms DEF_USE adds limited additional reachability for Reentrancy CEI chains. The structural core of the A1 failure is not the missing edge type but the genuine graph topology gap.

## Interpretation

A1 still fails at 38.2% (threshold 50%). The reachability rate saturates between k=5 and k=8 (+2.0pp over last 3 hops), indicating the graph diameter for unreachable CEI pairs exceeds what additional layers can fix. A positive-negative gap of +10.9pp (38.2% vs 27.3%) confirms that CEI path structure does differentiate reentrancy contracts structurally, but coverage is insufficient.

A2 now passes at 85.5% (threshold 80%) — FUNCTION→CFG coverage via CONTAINS is adequate.

A3 CALLS connectivity at 22.6% is informational only; no pass threshold was set.

## Pass/Fail Analysis

- A1: FAIL — 38.2% CEI reachability, well below the 50% threshold. Root cause: ~62% of reentrancy positives lack mutually reachable CEI-chain nodes within 8 hops, likely using indirect patterns (cross-contract reentrancy, delegatecall).
- A2: PASS — 85.5% FUNCTION→CFG coverage via CONTAINS exceeds the 80% threshold.
- A3: 22.6% CALLS connectivity — informational.

Overall status: FAIL (A1 is the primary criterion).

## Recommended Next Steps

1. Re-run after IMP-D1 re-extraction to check whether additional ICFG edges improve A1 reachability.
2. Investigate the 62% reentrancy positives without reachable CEI paths — check for cross-contract or delegatecall reentrancy patterns.
3. Cross-reference A1 gap (+10.9pp) with model recall — do contracts with CEI-reachable paths have higher model confidence?
4. Consider adding DELEGATECALL edge types to the phase 2 edge set to improve cross-contract reentrancy reachability.
