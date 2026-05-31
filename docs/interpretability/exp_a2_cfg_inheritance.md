# EXP-A2: CFG Feature Inheritance

**Layer:** 1 — Structure
**Priority:** P1
**Status:** FAIL
**Run date:** 2026-05-30
**Script:** `ml/scripts/interpretability/exp_a2_cfg_inheritance.py`
**Output:** `ml/logs/interpretability/exp_a2_cfg_inheritance.json`

---

## Purpose

This experiment checks whether CFG (Control Flow Graph) block nodes in SENTINEL graphs correctly inherit feature values (visibility, has_loop, payable, is_modifier, has_state_write) from their parent FUNCTION nodes. In the v8 graph schema, CFG blocks should propagate their parent function's attributes through CONTAINS edges, enabling Phase 3 of the GNN to reason about function-level properties at block granularity.

## Method

The script loads 500 contracts from the val split and identifies CFG block nodes (node type 7 = STATEMENT / CFG_BLOCK). For each such node, it finds its FUNCTION parent via CONTAINS edges and checks whether the inherited feature dimensions match between parent and child. It then reports consistency rates and flags the BUG-C3 warning if payable inheritance is below 90%.

## How to Run

```bash
TRANSFORMERS_OFFLINE=1 PYTHONPATH=. ml/.venv/bin/python ml/scripts/interpretability/exp_a2_cfg_inheritance.py \
  --cache ml/data/cached_dataset_v8.pkl \
  --label-csv ml/data/processed/multilabel_index_cleaned.csv \
  --splits-dir ml/data/splits/deduped \
  --n-contracts 500 \
  --out ml/logs/interpretability/exp_a2_cfg_inheritance.json
```

## Results

Analysed 470 graphs (30 cache misses from 500 requested).

### Key Metrics
| Metric | Value | Pass Threshold | Status |
|--------|-------|---------------|--------|
| Graphs with ≥1 CFG→FUNCTION parent relationship | 0 (0.0%) | — | FAIL |
| Payable CFG inheritance rate | N/A (no CFG nodes) | ≥90% | FAIL |
| Overall dim[4] consistency rate | 0.0% | ≥80% | FAIL |

**All inheritance statistics showed 0 total CFG nodes across all 470 graphs.**

## Interpretation

The complete absence of CFG-to-FUNCTION parent relationships (0 graphs, 0 CFG nodes found) indicates one of two scenarios: (1) the current cached dataset (`cached_dataset_v8.pkl`) was generated before BUG-C3 was applied, meaning CFG nodes were not included in the graph extractor at extraction time, or (2) CFG nodes exist but use a different node type index than the script expects. The script explicitly warns: "BUG-C3 may not be applied. Re-run graph extractor before trusting payable-based vulnerability detection." This is consistent with the MEMORY.md note that IMP-D1 re-extraction (reextract_graphs.py) is pending.

## Pass/Fail Analysis

All criteria failed due to zero CFG coverage:
- The corpus lacks CFG block nodes in the current cache, meaning Phase 2/3 of the GNN (which uses ICFG-related edges) has no intra-function control flow structure to operate on.
- This is a data quality issue, not an architectural design flaw — the extractor needs to be re-run with BUG-C3 applied.
- Until re-extraction is complete, CFG-level feature inheritance cannot be validated.

## Recommended Next Steps

1. **Priority action:** Run `reextract_graphs.py` to rebuild all 41K graphs with BUG-C3 applied (IMP-D1 item from MEMORY.md).
2. After re-extraction, re-run this experiment to verify inheritance consistency ≥80%.
3. Cross-reference node type mapping in `graph_schema.py` to confirm which node type index represents CFG blocks.
4. Until re-extraction, treat any payable-dependent vulnerability predictions (especially reentrancy) as potentially unreliable.
