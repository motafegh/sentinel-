# SENTINEL Interpretability Experiment Index

**Generated:** 2026-05-30  **Updated:** 2026-05-31
**Suite:** GNN_INTERPRETABILITY_AND_VALIDATION_PLAN.md
**Run environment:** WSL2, RTX 3070 8GB, Python 3.12.1, solc 0.4.0–0.8.35 available

---

## Summary

| Status | Count |
|--------|-------|
| PASS | 5 |
| FAIL | 10 |
| COMPLETE (checkpoint, results documented) | 4 |
| PENDING (requires checkpoint or data) | 5 |
| Not yet scripted | 2 |
| Phase B (new measurement scripts, not yet run) | 4 |

**Key finding (pre-checkpoint):** All FAIL results trace to two root causes: (1) missing `solc` compiler (exp_s1 test contract extraction — now fixed), and (2) REVERSE_CONTAINS edge absent from stored graphs (runtime-only by design, not a data gap).

**Key findings (L4–L8 with checkpoint):** `external_call_count` dominates gradient saliency for all classes. GNN phase embeddings are not linearly separable. ECE > 0.20 for all classes. Timestamp has F1=1.0 for small/medium contracts but 0.36 for large (size shortcut confirmed). `type_id_norm` is 3× more important than any other feature in permutation analysis.

**Validation pass (2026-05-31):** Three findings were corrected after a dedicated validation run. See VALIDATION_SUMMARY.md for full details. Key corrections:
- EXP-L8 feature labels were wrong (9/11 stale names) — `uses_block_globals` ranks 2nd for Timestamp, not last
- EXP-L2 embedding-zero method understated ablation effect by 450× — proper removal gives max drop 0.014
- EXP-L1 entropy context: Phase 2 gap is real but tiny (99.98% of max entropy); EXP-S3 Cohen's d corrected to 0.643

**Script fixes (2026-05-31):**
- EXP-L4: FEATURE_NAMES was stale hardcoded pre-v8 labels (same bug as L8) — fixed to import from graph_schema
- EXP-A2: `_CONTAINS_EDGE=0` (CALLS) hardcoded — fixed to `EDGE_TYPES["CONTAINS"]` (=5)
- EXP-E1: Analysis 2 used REVERSE_CONTAINS (runtime-only, never in .pt files) → rewrote as FUNCTION→CFG coverage via CONTAINS; Analysis 3 checked CONTRACT→FUNCTION (no such edge) → rewrote as CALLS connectivity
- EXP-L3: Previously only hooked conv3 (CF-only) — extended to also hook conv3b (CALL_ENTRY+RETURN_TO) in same forward pass

---

## Experiment Table

| Exp | Name | Layer | Priority | Status | Key Finding |
|-----|------|-------|----------|--------|-------------|
| EXP-S1 | Structural Trace Audit | 1 — Structure | P0 | **FAIL** | `solc` not installed; val IntegerUO pattern rate 92.6% (PASS), Reentrancy 14.3% (FAIL) |
| EXP-S2 | Edge Enrichment Ratio | 1 — Structure | P0 | **FAIL** | CONTROL_FLOW enrichment ~1.0× (ubiquitous, no signal); EMITS: 15.5× for UnusedReturn; REVERSE_CONTAINS absent |
| EXP-S3 | Feature Distribution Per Class | 1 — Structure | P1 | **PASS** ⚠️ | Timestamp size shortcut: Cohen_d=1.657 (SHORTCUT); mean_call_depth_norm dead (all zeros) — **VALIDATED: d corrected to 0.643 in val split, 0.905 in training; shortcut real but overstated** |
| EXP-S4 | ICFG-Lite Path Audit | 1 — Structure | P0 | **PASS** | 76% reentrancy positives have CALL_ENTRY; 69% have full CALL_ENTRY+RETURN_TO chain |
| EXP-E1 | K-Hop Receptive Field | 2 — Expressivity | P0 | **PARTIAL** | A1: CEI reachability 37.7% at k=8 (FAIL, need ≥50%); A2/A3 redesigned (2026-05-31): A2=FUNCTION→CFG coverage via CONTAINS, A3=CALLS connectivity — rerun needed |
| EXP-E2 | WL Distinguishability | 2 — Expressivity | P0 | **PASS** | All 4 classes WL-distinguishable; Timestamp 0% collision; Reentrancy 11.1% collision |
| EXP-E3 | Message Propagation Sim | 2 — Expressivity | P1 | **FAIL** | Random weights show no differential Phase 2 activation (expected — needs trained checkpoint) |
| EXP-E4 | Direction Sensitivity | 2 — Expressivity | P1 | **FAIL** | Directed vs undirected WL identical (0% diff); edge direction adds no WL discriminative power |
| EXP-A1 | Pooling Node-Type Audit | 1 — Structure | P0 | **PASS** | 100% graphs have ≥1 FUNCTION-like node; fallback never triggered; RECEIVE nodes absent |
| EXP-A2 | CFG Feature Inheritance | 1 — Structure | P1 | **FAIL** *(bug fixed)* | Bug fixed 2026-05-31: `_CONTAINS_EDGE=0` was CALLS not CONTAINS (=5). Still FAIL: 0 CFG nodes found — IMP-D1 re-extraction needed |
| EXP-A3 | JK Entropy Logging | 3 — Learning | P1 | **PASS** | Entropy 1.0935–1.0986 (near-max log(3)=1.099) across 47 epochs; no phase collapse |
| EXP-A4 | Aux Eye Contribution | 3 — Learning | P1 | PENDING — requires checkpoint | — |
| EXP-L1 | JK Weight Analysis | 3 — Learning | P1 | PENDING — requires checkpoint | — |
| EXP-L2 | Edge Ablation (inference) | 3 — Learning | P1 | PENDING — requires checkpoint | — |
| EXP-L3 | Attention Visualization | 3 — Learning | P2 | READY (script complete) | Now hooks conv3 (CF) + conv3b (CALL_ENTRY+RETURN_TO) simultaneously; requires checkpoint to run |
| EXP-L4 | Gradient Saliency | 3 — Learning | P1 | **COMPLETE** *(FEATURE_NAMES fixed)* | Bug fixed 2026-05-31: stale pre-v8 feature labels on dims 3–9. Rerun needed for corrected per-dim attribution. Previous conclusion (`external_call_count` dominates, Timestamp dim-2=10.1%) likely accurate but dims 3–9 labels wrong |
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

## Phase B — New Measurement Scripts (2026-05-31, not yet run)

| ID | Script | Purpose | Prerequisite |
|----|--------|---------|-------------|
| B1 | `exp_b1_phase2_gradient_norm.py` | Phase 2 gradient norm at each LayerNorm output — does Phase 2 receive loss signal? | checkpoint |
| B2 | `exp_b2_per_eye_ece.py` | Per-eye ECE (GNN / Transformer / Fused separately) — which eye drives miscalibration? | checkpoint; prerequisite for temperature scaling (Interp-1) |
| B3 | `exp_b3_jk_weight_distribution.py` | JK weight std + histogram per class — does model selectively use Phase 2 for specific classes? | checkpoint |
| B4 | `exp_b4_unusedreturn_saliency.py` | UnusedReturn top-scored contracts gradient saliency — structural signal or size shortcut? | checkpoint |

**Run order:** B2 must complete before Interp-1 (temperature scaling implementation).

---

## Next Actions (Priority Order, 2026-05-31)

1. **[A-now] Rerun EXP-L4** with fixed FEATURE_NAMES to get correct gradient saliency per dim
2. **[A-now] Rerun EXP-E1** with redesigned Analysis 2/3 to get meaningful FUNCTION→CFG coverage
3. **[A-now] Run EXP-L3** (conv3+conv3b hooks ready) — requires checkpoint
4. **[B-now] Run B1–B4 Phase B scripts** — all require checkpoint only (no data re-extraction)
5. **[C-after-B2] Implement Interp-1** temperature scaling after B2 reveals which eye is miscalibrated
6. **[D] Run IMP-D1 re-extraction** (`reextract_graphs.py`) — unblocks EXP-A2, EXP-E1 A1 full re-run
7. **[D] Data quality fixes** (Sol-1/2/3, IMP-D2) before Run 5
