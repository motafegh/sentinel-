# EXP-E1: K-Hop Receptive Field

**Layer:** 2 — Expressivity
**Priority:** P0
**Status:** FAIL
**Run date:** 2026-05-30
**Script:** `ml/scripts/interpretability/exp_e1_receptive_field.py`
**Output:** `ml/logs/interpretability/exp_e1_receptive_field.json`

---

## Purpose

This experiment measures the k-hop receptive field of the GNN architecture by checking what fraction of structurally relevant nodes can reach each other within k hops for k=1–8. It validates whether SENTINEL's 8-layer GNN is theoretically deep enough to propagate information across the graph structures in the corpus. Three analyses are run: (1) CEI reachability for reentrancy, (2) Phase 3 FUNCTION→CFG aggregation coverage, (3) Phase 1 CONTRACT→FUNCTION coverage.

## Method

For each analysis, the script builds the adjacency structure from cached graphs (using the relevant edge types for each phase), then computes the fraction of target node pairs reachable within k hops using BFS/sparse matrix powers. Pass criteria: ≥50% of reentrancy-positive contracts have CEI-chain nodes mutually reachable within k=8 hops (A1); ≥70% of FUNCTION nodes can reach ≥50% of their CFG descendants within k=8 (A2); CONTRACT nodes reach FUNCTION nodes within k=2 (A3).

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

### Key Metrics
| Analysis | Metric | Value | Pass Threshold | Status |
|----------|--------|-------|---------------|--------|
| A1 | CEI reachability at k=8 (reentrancy pos) | 37.7% | ≥50% | FAIL |
| A1 | CEI reachability at k=8 (reentrancy neg) | 26.7% | — | — |
| A2 | FUNCTION nodes reaching ≥50% CFG | 0.1% | ≥70% | FAIL |
| A3 | CONTRACT nodes reaching FUNCTION at k=2 | 0.0% | — | FAIL |

**A1 Detailed (CEI reachability per k):**
| k | Positive rate | Negative rate |
|---|--------------|--------------|
| 1 | 17.1% | 17.1% |
| 2 | 29.1% | 23.0% |
| 3 | 32.2% | 23.0% |
| 4 | 35.7% | 25.7% |
| 8 | 37.7% | 26.7% |

**Total samples:** 199 reentrancy positives, 187 negatives, 7,692 FUNCTION nodes sampled, 1,474 CONTRACT nodes sampled.

## Interpretation

All three analyses failed, and the reasons align with the exp_a2 and exp_s2 findings: the current cached graphs lack CFG block nodes and REVERSE_CONTAINS edges. With zero CFG nodes (confirmed by exp_a2) and REVERSE_CONTAINS absent (confirmed by exp_s2), analyses A2 and A3 trivially fail — the required edge types for aggregation don't exist.

Analysis A1 (CEI reachability) using Phase 2 edges shows some discrimination: positive reentrancy contracts have 37.7% CEI reachability vs 26.7% for negatives, a +11 percentage point gap. However, this falls short of the 50% pass threshold, and the rate saturates by k=4 (no improvement from k=4 to k=8), suggesting the Phase 2 edges don't form long enough paths in the current graphs.

The A1 positive-negative gap (+11pp) is genuinely informative — it confirms CEI-relevant path structure differentiates reentrancy contracts even without CFG nodes, but 37.7% coverage means 62% of reentrancy positives are not reachable by the expected path type.

## Pass/Fail Analysis

All three criteria failed, but the root causes differ:
- A2 and A3 failures: data issue — CFG nodes and REVERSE_CONTAINS edges missing (pending IMP-D1).
- A1 failure: genuine gap — only 37.7% of reentrancy positives have CEI-reachable paths, even with 8 GNN layers.

## Recommended Next Steps

1. Re-run after IMP-D1 re-extraction — A2 and A3 will pass once CFG nodes and REVERSE_CONTAINS edges are added.
2. Investigate the 62% reentrancy positives without CEI-reachable paths — check if they use indirect reentrancy patterns (cross-contract, delegatecall).
3. Consider increasing GNN depth beyond 8 layers if re-extraction still shows <50% CEI reachability.
4. Cross-reference A1 gap (+11pp positive vs negative) with actual model recall — is there a correlation between CEI reachability and model prediction confidence?
