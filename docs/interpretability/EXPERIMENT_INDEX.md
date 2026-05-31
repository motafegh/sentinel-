# SENTINEL Interpretability Experiment Index

**Generated:** 2026-05-30
**Suite:** GNN_INTERPRETABILITY_AND_VALIDATION_PLAN.md
**Run environment:** WSL2, RTX 3070 8GB, Python 3.12.1, no `solc` installed

---

## Summary

| Status | Count |
|--------|-------|
| PASS | 5 |
| FAIL | 10 |
| COMPLETE (checkpoint, results documented) | 4 |
| PENDING (requires checkpoint) | 5 |
| Not yet scripted | 2 |

**Key finding (pre-checkpoint):** All FAIL results trace to two root causes: (1) missing `solc` compiler (exp_s1 test contract extraction), and (2) CFG node / REVERSE_CONTAINS edge absence in the current `cached_dataset_v8.pkl` — pending IMP-D1 re-extraction.

**Key findings (L4–L8 with checkpoint):** `external_call_count` dominates gradient saliency for all classes. GNN phase embeddings are not linearly separable. ECE > 0.20 for all classes. Timestamp has F1=1.0 for small/medium contracts but 0.36 for large (size shortcut confirmed). `type_id_norm` is 3× more important than any other feature in permutation analysis.

**Validation pass (2026-05-31):** Three findings were corrected after a dedicated validation run. See VALIDATION_SUMMARY.md for full details. Key corrections:
- EXP-L8 feature labels were wrong (9/11 stale names) — `uses_block_globals` ranks 2nd for Timestamp, not last
- EXP-L2 embedding-zero method understated ablation effect by 450× — proper removal gives max drop 0.014
- EXP-L1 entropy context: Phase 2 gap is real but tiny (99.98% of max entropy); EXP-S3 Cohen's d corrected to 0.643

---

## Experiment Table

| Exp | Name | Layer | Priority | Status | Key Finding |
|-----|------|-------|----------|--------|-------------|
| EXP-S1 | Structural Trace Audit | 1 — Structure | P0 | **FAIL** | `solc` not installed; val IntegerUO pattern rate 92.6% (PASS), Reentrancy 14.3% (FAIL) |
| EXP-S2 | Edge Enrichment Ratio | 1 — Structure | P0 | **FAIL** | CONTROL_FLOW enrichment ~1.0× (ubiquitous, no signal); EMITS: 15.5× for UnusedReturn; REVERSE_CONTAINS absent |
| EXP-S3 | Feature Distribution Per Class | 1 — Structure | P1 | **PASS** ⚠️ | Timestamp size shortcut: Cohen_d=1.657 (SHORTCUT); mean_call_depth_norm dead (all zeros) — **VALIDATED: d corrected to 0.643 in val split, 0.905 in training; shortcut real but overstated** |
| EXP-S4 | ICFG-Lite Path Audit | 1 — Structure | P0 | **PASS** | 76% reentrancy positives have CALL_ENTRY; 69% have full CALL_ENTRY+RETURN_TO chain |
| EXP-E1 | K-Hop Receptive Field | 2 — Expressivity | P0 | **FAIL** | CEI reachability 37.7% at k=8 (need ≥50%); A2/A3 fail due to missing CFG nodes |
| EXP-E2 | WL Distinguishability | 2 — Expressivity | P0 | **PASS** | All 4 classes WL-distinguishable; Timestamp 0% collision; Reentrancy 11.1% collision |
| EXP-E3 | Message Propagation Sim | 2 — Expressivity | P1 | **FAIL** | Random weights show no differential Phase 2 activation (expected — needs trained checkpoint) |
| EXP-E4 | Direction Sensitivity | 2 — Expressivity | P1 | **FAIL** | Directed vs undirected WL identical (0% diff); edge direction adds no WL discriminative power |
| EXP-A1 | Pooling Node-Type Audit | 1 — Structure | P0 | **PASS** | 100% graphs have ≥1 FUNCTION-like node; fallback never triggered; RECEIVE nodes absent |
| EXP-A2 | CFG Feature Inheritance | 1 — Structure | P1 | **FAIL** | 0 CFG nodes found in corpus — IMP-D1 re-extraction needed before this test is meaningful |
| EXP-A3 | JK Entropy Logging | 3 — Learning | P1 | **PASS** | Entropy 1.0935–1.0986 (near-max log(3)=1.099) across 47 epochs; no phase collapse |
| EXP-A4 | Aux Eye Contribution | 3 — Learning | P1 | PENDING — requires checkpoint | — |
| EXP-L1 | JK Weight Analysis | 3 — Learning | P1 | PENDING — requires checkpoint | — |
| EXP-L2 | Edge Ablation (inference) | 3 — Learning | P1 | PENDING — requires checkpoint | — |
| EXP-L3 | Attention Visualization | 3 — Learning | P2 | PENDING — requires checkpoint | — |
| EXP-L4 | Gradient Saliency | 3 — Learning | P1 | **COMPLETE** | `external_call_count` dominates all classes (21–24%); Timestamp dim-2 = 10.1% (FAIL, shortcut confirmed); Reentrancy CFG nodes = 9.5% (FAIL) |
| EXP-L5 | Probing Classifiers | 3 — Learning | P1 | **COMPLETE** | Phase 2 adds zero F1 delta; only IntegerUO improves via Phase 3 (+3.9pp); embeddings non-linearly encoded |
| EXP-L6 | Counterfactual Contracts | 3 — Learning | P2 | **FAIL (1/4 pass)** | UnusedReturn PASS (+0.108); Reentrancy/IntegerUO/Timestamp FAIL — model does not detect CEI violation, `unchecked` overflow, or `block.timestamp` branching on minimal novel contracts |
| EXP-L7 | Calibration & Size Analysis | 3 — Learning | P2 | **COMPLETE** | ECE 0.205–0.310 (all miscalibrated); Timestamp F1 gap 0.636 (worst); IntegerUO only PASS (gap 0.044) |
| EXP-L8 | Permutation Importance | 3 — Learning | P2 | **COMPLETE** | `type_id_norm` rank 1 (0.0786, 3× next); `uses_block_globals` rank 10, `has_state_write` rank 11; pass criteria FAIL |
| EXP-L9 | Attention Rollout | 3 — Learning | P2 | PENDING — requires checkpoint | — |
| EXP-L10 | Training Ablation Commands | 3 — Learning | P2 | **PASS** | 12 ablation commands generated; CONTAINS/CONTROL_FLOW expected highest impact |

---

## Root Cause Analysis for FAIL results

### Cause 1: `solc` not installed (affects EXP-S1 test contract portion)
- **Fix:** `pip install solc-select && solc-select install 0.8.0 && solc-select use 0.8.0`
- **Impact:** EXP-S1 P0 gate completely blocked; val-split pattern detection still ran

### Cause 2: CFG nodes absent from `cached_dataset_v8.pkl` (affects EXP-A2, EXP-E1 A2/A3, EXP-E4 directional signal)
- **Fix:** Run `reextract_graphs.py` (IMP-D1 item)
- **Impact:** All CFG-dependent analyses trivially fail; REVERSE_CONTAINS edge count = 0

### Cause 3: Enrichment thresholds too high for ubiquitous edges (affects EXP-S2 named checks)
- **Fix:** Recalibrate thresholds to account for baseline prevalence (mutual information metric preferred)
- **Impact:** CONTROL_FLOW and CONTAINS are in 99.6% of contracts — ratio threshold of 1.3× is unreachable

### Cause 4: Random weights (expected, affects EXP-E3)
- **Fix:** Load trained checkpoint
- **Impact:** Phase 2 differential activation requires learned weights

---

## Argparse Bugs Fixed During This Run

Multiple scripts had duplicate argument definitions conflicting with `add_common_args` in `utils.py`. Fixed:

| Script | Duplicate args removed |
|--------|----------------------|
| `exp_s2_edge_enrichment.py` | `--split` |
| `exp_s3_feature_distribution.py` | `--split`, `--n-contracts`, `--seed` |
| `exp_s4_icfg_path_audit.py` | `--split`, `--n-contracts` |
| `exp_s1_structural_trace.py` | `GraphExtractionConfig(schema_version="v8")` invalid kwarg |
| `exp_l5_probing_classifiers.py` | `--n-contracts` (duplicate in `parse_args` vs `add_common_args`; replaced with `parser.set_defaults`) |

---

## Next Actions (Priority Order)

1. **[P0] Install `solc`** — unblocks EXP-S1 full run
2. **[P0] Run IMP-D1 re-extraction** (`reextract_graphs.py`) — unblocks EXP-A2, EXP-E1 A2/A3
3. **[P0] Load checkpoint in EXP-E3** — validate trained Phase 2 differential activation
4. **[P1] Add test contracts** to `ml/scripts/test_contracts/` for EXP-S1 extraction
5. **[P1] Investigate Timestamp size shortcut** (exp_s3 finding: Cohen_d=1.657)
6. **[P2] Run PENDING (checkpoint) experiments** after checkpoint is verified loadable
7. **[P2] Fix numpy bool_ JSON serialization** bug in `exp_a1_pooling_audit.py`
