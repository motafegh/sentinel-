# EXP-S2: Edge Enrichment Ratio

**Layer:** 1 — Structure
**Priority:** P0
**Status:** FAIL (enrichment ratios below expected thresholds for key edge types)
**Run date:** 2026-05-30
**Script:** `ml/scripts/interpretability/exp_s2_edge_enrichment.py`
**Output:** `ml/logs/interpretability/exp_s2_edge_enrichment.json`

**Note:** Script had a duplicate `--split` argparse argument bug (fixed prior to run).

---

## Purpose

This experiment measures the per-class enrichment ratio of each edge type, defined as: (fraction of vulnerable contracts containing ≥1 edge of type T) / (baseline fraction across all contracts). An enrichment ratio > 1.0 means that edge type appears more frequently in contracts with that vulnerability. This validates whether the graph structure contains discriminative structural signals for each vulnerability class.

## Method

The script loads the full val split (6,236 contracts), groups them by vulnerability label, and for each of the 11 edge types computes the baseline presence fraction and per-class presence fraction. Enrichment = class_fraction / baseline_fraction. Named checks assert that structurally motivated edge-class pairs exceed specific thresholds (e.g., CONTROL_FLOW should enrich for Reentrancy ≥1.3×, RETURN_TO should enrich for TransactionOrderDep ≥1.2×).

## How to Run

```bash
TRANSFORMERS_OFFLINE=1 PYTHONPATH=. ml/.venv/bin/python ml/scripts/interpretability/exp_s2_edge_enrichment.py \
  --cache ml/data/cached_dataset_v8.pkl \
  --label-csv ml/data/processed/multilabel_index_cleaned.csv \
  --splits-dir ml/data/splits/deduped \
  --split val \
  --out ml/logs/interpretability/exp_s2_edge_enrichment.json
```

## Results

Analysed 5,845 graphs from val split (391 stems not found in cache — excluded).

### Key Metrics
| Metric | Value | Pass Threshold | Status |
|--------|-------|---------------|--------|
| CONTROL_FLOW enrichment for Reentrancy | 1.004 | ≥1.3 | FAIL |
| CONTROL_FLOW enrichment for IntegerUO | 1.000 | ≥1.3 | FAIL |
| CALL_ENTRY enrichment for Reentrancy | 1.065 | ≥1.3 | FAIL |
| CALL_ENTRY enrichment for ExternalBug | 1.229 | ≥1.3 | FAIL |
| RETURN_TO enrichment for Reentrancy | 1.076 | ≥1.3 | FAIL |
| DEF_USE enrichment for IntegerUO | 1.009 | ≥1.3 | FAIL |
| DEF_USE enrichment for UnusedReturn | 1.175 | ≥1.3 | FAIL |
| READS enrichment for TransactionOrderDep | 0.986 | ≥1.2 | FAIL |

**Full enrichment matrix (select notable values):**
| Edge Type | Baseline% | Best class enrichment |
|-----------|-----------|----------------------|
| EMITS (3) | 0.1% | UnusedReturn=15.5×, CallToUnknown=4.65× |
| RETURN_TO (9) | 55.4% | Timestamp=1.578×, UnusedReturn=1.532× |
| CALL_ENTRY (8) | 63.4% | Timestamp=1.414×, UnusedReturn=1.403× |
| CALLS (0) | 75.4% | UnusedReturn=1.306×, Timestamp=1.299× |
| CONTROL_FLOW (6) | 99.6% | Reentrancy=1.004× (near-baseline) |
| REVERSE_CONTAINS (7) | 0.0% | No contracts have this edge |

## Interpretation

All 8 named enrichment checks failed. The most critical finding is that CONTROL_FLOW edges (edge type 6) appear in 99.6% of all contracts, meaning they provide almost zero discriminative signal (ratio ≈ 1.0). The enrichment thresholds of 1.3× were too optimistic given the near-universal baseline presence of CFG edges. Conversely, EMITS (event emit) edges show extreme enrichment for UnusedReturn (15.5×) and CallToUnknown (4.65×) — these rare edges appear to be highly class-specific.

The REVERSE_CONTAINS edge (type 7) has 0% baseline presence — it is entirely absent from the corpus, which aligns with the exp_a2 finding that CFG-related structures were not extracted. RETURN_TO and CALL_ENTRY edges show meaningful Timestamp enrichment (1.41–1.58×), confirming that inter-procedural call structure correlates with timestamp-dependent vulnerability.

## Pass/Fail Analysis

The formal criteria failed, but the enrichment matrix reveals meaningful signals:
- High-enrichment edges for Timestamp: RETURN_TO (1.578×), CALL_ENTRY (1.414×), CALLS (1.299×)
- The EMITS edge is extremely rare (0.1% baseline) but highly discriminative — worth investigating
- CONTROL_FLOW and CONTAINS are ubiquitous (99.6%) and thus provide no discriminative signal
- The thresholds (1.3× for CFG edges) should be recalibrated to account for baseline prevalence

## Recommended Next Steps

1. Recalibrate enrichment thresholds using mutual information or chi-squared test instead of raw ratio (rare edges with high ratio may be spurious).
2. Investigate EMITS edge — 15.5× enrichment for UnusedReturn suggests event-emit patterns strongly correlate with return value mishandling.
3. After IMP-D1 re-extraction (which should add ICFG edges), re-run this experiment to check if CFG-related enrichment improves.
4. REVERSE_CONTAINS absence confirms IMP-D1 is needed before ICFG-dependent experiments.
