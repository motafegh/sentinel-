# SENTINEL Interpretability Experiment Index

**Generated:** 2026-05-30  **Updated:** 2026-06-01 (audit fixes: B1 gradient method, L2 structural ablation, L3 reclassified, L4 rerun)
**Suite:** GNN_INTERPRETABILITY_AND_VALIDATION_PLAN.md
**Run environment:** WSL2, RTX 3070 8GB, Python 3.12.1, solc 0.4.0–0.8.35 available

---

## Summary

| Status | Count |
|--------|-------|
| PASS | 5 |
| FAIL | 14 |
| COMPLETE | 10 (A4, L1–L8 + B1–B4, now including L3 ARCH-N/A and L4 rerun) |
| PENDING | 0 |
| Phase B | 4 (B1–B4, all run 2026-05-31/06-01) |

**Key finding (pre-checkpoint):** All FAIL results trace to two root causes: (1) missing `solc` compiler (exp_s1 test contract extraction — now fixed), and (2) REVERSE_CONTAINS edge absent from stored graphs (runtime-only by design, not a data gap).

**Key findings (L4–L8 with checkpoint):** `external_call_count` dominates gradient saliency for all classes. GNN phase embeddings are not linearly separable. ECE > 0.20 for all classes. Timestamp has F1=1.0 for small/medium contracts but 0.36 for large (size shortcut confirmed). `type_id_norm` is 3× more important than any other feature in permutation analysis.

**Validation pass (2026-05-31):** Three findings were corrected after a dedicated validation run. See VALIDATION_SUMMARY.md for full details. Key corrections:
- EXP-L8 feature labels were wrong (9/11 stale names) — `uses_block_globals` ranks 2nd for Timestamp, not last
- EXP-L2 embedding-zero method understated ablation effect by 450× — proper removal gives max drop 0.014
- EXP-L1 entropy context: Phase 2 gap is real but tiny (99.98% of max entropy); EXP-S3 Cohen's d corrected to 0.643

**Audit fixes (2026-06-01):** Four additional corrections applied:
- EXP-B1: gradient backward changed from raw logit to BCEWithLogitsLoss (matches training loss); P2/P1 ratios updated to 72–91%
- EXP-L2: structural ablation added to measured results; embedding/structural ratio corrected from 450× to 10,944×; new finding: Phase 2 edges SUPPRESS Reentrancy predictions (positive delta on removal)
- EXP-L3: false PASS retracted and reclassified ARCHITECTURAL N/A — 100% CF fraction was architecturally guaranteed; real finding is uniform attention (all weights = 1.0)
- EXP-L4: FEATURE_NAMES fix applied (import from graph_schema); rerun 2026-06-01 with correct labels

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
| EXP-S3 | Feature Distribution Per Class | 1 — Structure | P1 | **PASS** ⚠️ | Timestamp cfg_call_count d=1.592 (SHORTCUT); return_ignored d=0.716 for UnusedReturn. "Dead feature" finding RETRACTED — was CFG node artifact |
| EXP-S4 | ICFG-Lite Path Audit | 1 — Structure | P0 | **PASS** | 76% reentrancy positives have CALL_ENTRY; 69% have full CALL_ENTRY+RETURN_TO chain |
| EXP-E1 | K-Hop Receptive Field | 2 — Expressivity | P0 | **FAIL** | A1: 38.2% at k=8 (DEF_USE added, was 37.7%); A2 PASS=85.5% CONTAINS coverage; A3=22.6% CALLS connectivity |
| EXP-E2 | WL Distinguishability | 2 — Expressivity | P0 | **PASS** | All 4 classes WL-distinguishable; Timestamp 0% collision; Reentrancy 11.1% collision |
| EXP-E3 | Message Propagation Sim | 2 — Expressivity | P1 | **FAIL** | Random weights show no differential Phase 2 activation (expected — needs trained checkpoint) |
| EXP-E4 | Direction Sensitivity | 2 — Expressivity | P1 | **FAIL** | All 4 edge types (CF/DEF_USE/CALL_ENTRY/RETURN_TO): directed=undirected=89.1%, diff=0%. Direction adds no power for any Phase 2 edge type |
| EXP-A1 | Pooling Node-Type Audit | 1 — Structure | P0 | **PASS** | 100% graphs have ≥1 FUNCTION-like node; fallback never triggered; RECEIVE nodes absent |
| EXP-A2 | CFG Feature Inheritance | 1 — Structure | P1 | **FAIL** *(bug fixed)* | Bug fixed 2026-05-31: `_CONTAINS_EDGE=0` was CALLS not CONTAINS (=5). Still FAIL: 0 CFG nodes found — IMP-D1 re-extraction needed |
| EXP-A3 | JK Entropy Logging | 3 — Learning | P1 | **PASS** | Entropy 1.0935–1.0986 (near-max log(3)=1.099) across 47 epochs; no phase collapse |
| EXP-A4 | Aux Eye Contribution | 3 — Learning | P1 | PENDING — requires checkpoint | — |
| EXP-L1 | JK Weight Analysis | 3 — Learning | P1 | PENDING — requires checkpoint | — |
| EXP-L2 | Edge Ablation (inference) | 3 — Learning | P1 | **COMPLETE** *(structural ablation added 2026-06-01)* | Embedding ablation: CFG combined drop=1.11e-6. Structural ablation: CFG combined drop=0.0121 (10,944× larger). Reentrancy structural deltas are POSITIVE (Phase 2 edges suppress Reentrancy). CONTROL_FLOW structural Δ Timestamp=+0.163 (very large suppression). |
| EXP-L3 | Attention Visualization | 3 — Learning | P2 | **ARCHITECTURAL N/A** *(audit 2026-06-01)* | 100% CF fraction in top edges is architecturally guaranteed (conv3 wired to CF-only subgraph). Real finding: all GAT attention weights = 1.0 (uniform) — no selective attention learned within CFG. |
| EXP-L4 | Gradient Saliency | 3 — Learning | P1 | **COMPLETE** *(rerun 2026-06-01, correct feature names)* | `external_call_count` dominates all 10 classes (21–24%); `complexity` rank 2 universally. Timestamp `uses_block_globals`=10.0% FAIL (threshold ≥20%). Reentrancy CFG_NODE_CALL+WRITE=8.9% FAIL. Global sensitivity artifact confirmed. |
| EXP-L5 | Probing Classifiers | 3 — Learning | P1 | **FAIL** *(pooling fixed)* | Max+mean [512] pooling fix: IntegerUO Phase1 now 0.419 (was 0.114 mean-only); Reentrancy Phase2 still -0.0069 vs Phase1 → FAIL |
| EXP-L6 | Counterfactual Contracts | 3 — Learning | P2 | **FAIL (1/4 pass)** | UnusedReturn PASS (+0.122); CEI safe>vuln (−0.0071); IntegerUO safe>vuln (−0.0642); Timestamp tied (0.0000). Model blind to structural vulnerability semantics |
| EXP-L7 | Calibration & Size Analysis | 3 — Learning | P2 | **COMPLETE** | ECE 0.205–0.310 (all miscalibrated); Timestamp F1 gap 0.636 (worst); IntegerUO only PASS (gap 0.044) |
| EXP-L8 | Permutation Importance | 3 — Learning | P2 | **COMPLETE** | `type_id_norm` rank 1 (0.0786, 3× next); `uses_block_globals` rank 10, `has_state_write` rank 11; pass criteria FAIL |
| EXP-L9 | Attention Rollout | 3 — Learning | P2 | **FAIL** *(criterion fixed)* | Relative-rank criterion: safe CW=0.09692 > vuln CW=0.09038 (delta=−0.00654). Original PASS retracted — prior criterion non-discriminative |
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
| ID | Script | Status | Key Finding |
|----|--------|--------|-------------|
| B1 | `exp_b1_phase2_gradient_norm.py` | **COMPLETE** *(method corrected 2026-06-01)* | Phase 1 > Phase 2 > Phase 3 for every class. Phase 2 = 72–91% of Phase 1 (corrected run using BCEWithLogitsLoss; was 75–86% with raw logit). Timestamp has highest absolute norms and highest P2/P1 ratio (91.3%). |
| B2 | `exp_b2_per_eye_ece.py` | **COMPLETE** | GNN/TF/Fused eyes ECE 0.057–0.065 (good). Main head ECE 0.249 (severe). Temperature scaling targets main head only. |
| B3 | `exp_b3_jk_weight_distribution.py` | **COMPLETE** | Universal Phase3 > Phase1 > Phase2 ordering. No class selectively upweights Phase 2. Std 0.01–0.03 (stable). |
| B4 | `exp_b4_unusedreturn_saliency.py` | **COMPLETE** | external_call_count + complexity dominate both high/low UnusedReturn scorers. return_ignored ranks 4th with 2.3% relative difference — size shortcut confirmed. |

**All B scripts complete.** Temperature scaling fitted: `ml/calibration/temperatures_run4.json` (ECE 0.249 → 0.028).

---

## Status as of 2026-06-01 (audit fixes complete)

All P1–P5 script fixes applied. B1–B4 complete. Temperature calibration fitted. Audit fixes (B1 gradient method, L2 structural ablation, L3 reclassification, L4 rerun) applied 2026-06-01.

**No pending experiments.** All 21 experiments (+ B1–B4) resolved.

**Deferred (post-Run 5):**
- IMP-D1 re-extraction (`reextract_graphs.py`) — raises max_nodes to 2048, unblocks EXP-A2
- DEF_USE chain length distribution (MISSING-3)
- STATE_VAR multi-function sharing for TOD (MISSING-4)
- L10 full training ablation (requires `--ablate-edge-type` in train.py)
