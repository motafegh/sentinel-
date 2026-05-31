# EXP-A1: Pooling Node-Type Audit

**Layer:** 1 — Structure
**Priority:** P0
**Status:** PASS (with minor JSON serialization error — results intact)
**Run date:** 2026-05-30
**Script:** `ml/scripts/interpretability/exp_a1_pooling_audit.py`
**Output:** `ml/logs/interpretability/exp_a1_pooling_audit.json` (truncated due to numpy bool_ serialization error)

---

## Purpose

This experiment audits how many graphs in the SENTINEL corpus contain at least one FUNCTION-like node (node types: FUNCTION=1, MODIFIER=2, FALLBACK=4, RECEIVE=5, CONSTRUCTOR=6). The GNN pooling strategy preferentially aggregates over FUNCTION-like nodes; if a graph contains none, it falls back to mean-pooling all nodes. This audit validates whether the fallback is ever triggered in practice.

## Method

The script loads the cached dataset and val-split indices, then for each of the 2,000 sampled graphs, counts FUNCTION-like nodes by checking node feature dimension 0 (node type index). It computes the fraction of graphs triggering the fallback and the distribution of FUNCTION-like node counts per graph. Pass criteria: ≥95% of graphs have at least one FUNCTION-like node, and <5% trigger the all-nodes fallback.

## How to Run

```bash
TRANSFORMERS_OFFLINE=1 PYTHONPATH=. ml/.venv/bin/python ml/scripts/interpretability/exp_a1_pooling_audit.py \
  --cache ml/data/cached_dataset_v8.pkl \
  --label-csv ml/data/processed/multilabel_index_cleaned.csv \
  --splits-dir ml/data/splits/deduped \
  --n-contracts 2000 \
  --out ml/logs/interpretability/exp_a1_pooling_audit.json
```

**Note:** Script exits with a `TypeError: Object of type bool_ is not JSON serializable` error after printing results — a numpy type serialization bug. The results table was printed before the error and all key metrics were captured. The JSON was partially written with the numeric fields intact.

## Results

Audited 1,859 graphs (141 cache misses from 2,000 requested).

### Key Metrics
| Metric | Value | Pass Threshold | Status |
|--------|-------|---------------|--------|
| Graphs with ≥1 FUNCTION-like node | 100.00% | ≥95% | PASS |
| Graphs triggering fallback | 0.00% | <5% | PASS |
| Mean FUNCTION-like nodes per graph | 20.20 | — | — |
| Mean fraction FUNCTION-like | 17.73% | — | — |

**FUNCTION-like node histogram:**
| Count range | Graphs | % |
|-------------|--------|---|
| 0 | 0 | 0.0% |
| 1 | 16 | 0.9% |
| 2–5 | 161 | 8.7% |
| 6–10 | 358 | 19.3% |
| >10 | 1,324 | 71.2% |

**FUNCTION-like node type breakdown (total across all graphs):**
| Type | Name | Count |
|------|------|-------|
| 1 | FUNCTION | 31,243 |
| 2 | MODIFIER | 2,778 |
| 4 | FALLBACK | 844 |
| 5 | RECEIVE | 0 |
| 6 | CONSTRUCTOR | 2,690 |

## Interpretation

Every single graph in the audited corpus has at least one FUNCTION-like node — the fallback to all-node pooling is never triggered. This validates a key architectural assumption: the GNN's attentive contract-level readout (which pools over FUNCTION nodes) will always have at least one representative node to aggregate. The average contract has ~20 function-like nodes, with 71% having more than 10 — providing rich pooling targets. The RECEIVE node type (type 5) has zero occurrences in this split, suggesting it is either absent from the corpus or mapped to a different type.

## Pass/Fail Analysis

Both pass criteria passed with perfect scores (100% / 0%). This confirms:
- The corpus contains structurally valid Solidity contracts with real function boundaries.
- No degenerate graphs will silently fall back to all-node pooling, which would produce noisy embeddings.
- The zero RECEIVE occurrences warrant checking if RECEIVE functions are under-represented in the dataset (may matter for payable vulnerability detection).

## Recommended Next Steps

- Fix the numpy bool_ JSON serialization bug in the script (`json_data["criterion1_pass"] = bool(...)` instead of np.bool_).
- Investigate zero RECEIVE (type 5) node count — these are contract-level payment receivers and may be relevant for reentrancy/DoS labels.
- Cross-reference with exp_a2 to ensure FUNCTION nodes carry correct feature values.
