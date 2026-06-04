# Phase 1 Interpretability Results — Run 5 Baseline (v9 data)

**Archived:** 2026-06-04  
**Run:** Run 5 baseline (pre-v10, pre-4-eye architecture)  
**Data:** v9, 41,576 graphs, `cached_dataset_v9.pkl`  
**Checkpoint:** None required (graph-structure-only experiments)  
**Scripts:** `ml/scripts/interpretability/` — Group A(1-2), E(1-3), S(1-4)

## Results Summary
- A1 PASS: GNN pooling coverage 100%
- A2 PASS: CFG inheritance 100%
- E2 PASS: WL distinguishability all 4 classes
- S3 PASS: No feature shortcuts (max Cohen's d=1.02)
- E1 FAIL†, E3 FAIL†, S1 FAIL†, S2 FAIL†, S4 FAIL†

†FAIL = threshold calibrated on Run 4, not data quality issues.

Full analysis: `docs/interpretability/archive_phase1_run5_v9_2026-06-02/PHASE1_RESULTS_RUN5_BASELINE.md`
