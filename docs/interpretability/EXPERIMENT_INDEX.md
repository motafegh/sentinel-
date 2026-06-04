# SENTINEL Interpretability Experiment Index — Run 7

**Status:** Phase 1 COMPLETE (v9 baseline) · Phase 2 Tier 1+2 COMPLETE (Run 7 ep39 checkpoint)
**Checkpoint:** `ml/checkpoints/GCB-P1-Run7-v10-20260603_best.pt` (ep39, F1=0.3074)
**Data:** v10, 41,576 graphs, `ml/data/cached_dataset_v10.pkl`
**Splits:** `ml/data/splits/v10_deduped/` — train=29,103 / val=6,236 / test=6,237
**Phase 2 results:** `ml/interpretability_results/phase2_run7_ep39_v10_2026-06-04/`

> Phase 1 results (v9 baseline, Run 5): `ml/interpretability_results/archive_phase1_run5_v9_2026-06-02/`
> Understanding doc: `docs/interpretability/SENTINEL-Understanding-Run7.md`

---

## Run Command Template

```bash
source ml/.venv/bin/activate
CKPT=ml/checkpoints/GCB-P1-Run7-v10-20260603_best.pt
OUT=ml/interpretability_results/phase2_run7_ep39_v10_2026-06-04

TRANSFORMERS_OFFLINE=1 PYTHONPATH=. python ml/scripts/interpretability/<script>.py \
  --checkpoint $CKPT \
  --out $OUT/<exp_id>/exp_<id>_<name>.json
```

---

## Experiment Suite

### Group A — Architecture Validation
| ID | Script | Needs ckpt | Phase 1 (v9) | Phase 2 (Run7) | Key finding |
|---|---|---|---|---|---|
| A1 | exp_a1_pooling_audit.py | No | ✅ PASS | — | Pooling coverage 100% |
| A2 | exp_a2_cfg_inheritance.py | No | ✅ PASS | — | CFG inheritance 100% |
| A3 | exp_a3_jk_entropy_logging.py | Yes | — | ⚠️ FAIL | Phase3 JK drift from ep1; never corrected |
| A4 | exp_a4_aux_contribution.py | Yes | — | ⚠️ FAIL | GNN useful solo for IntegerUO only (1/10 classes) |

### Group B — Training Diagnostics
| ID | Script | Needs ckpt | Phase 1 (v9) | Phase 2 (Run7) | Key finding |
|---|---|---|---|---|---|
| B1 | exp_b1_phase2_gradient_norm.py | Yes | — | ✅ PASS | Ph2/Ph1 ratio 0.78–0.92; ISSUE-1 fix confirmed |
| B2 | exp_b2_per_eye_ece.py | Yes | — | ✅ PASS | Per-eye ECE 0.04; ensemble ECE 0.23 (5.8× worse) |
| B3 | exp_b3_jk_weight_distribution.py | Yes | — | ✅ PASS | Phase3 drift global (spread=0.009), not class-selective |
| B4 | exp_b4_unusedreturn_saliency.py | Yes | — | ⚠️ FAIL | `return_ignored` rank-5 for both confident/non-confident UnusedReturn |

### Group E — Graph Expressivity
| ID | Script | Needs ckpt | Phase 1 (v9) | Phase 2 (Run7) | Key finding |
|---|---|---|---|---|---|
| E1 | exp_e1_receptive_field.py | No | ⚠️ FAIL† | — | — |
| E2 | exp_e2_wl_distinguishability.py | No | ✅ PASS | — | Graphs WL-distinguishable |
| E3 | exp_e3_message_propagation_sim.py | No | ⚠️ FAIL† | — | — |
| E4 | exp_e4_direction_sensitivity.py | Yes | — | ⏳ not run | — |

### Group L — Model Behaviour
| ID | Script | Needs ckpt | Phase 1 (v9) | Phase 2 (Run7) | Key finding |
|---|---|---|---|---|---|
| L1 | exp_l1_jk_weight_analysis.py | Yes | — | ⚠️ FAIL | All classes Phase3-dominant; entropy 99.5% max; 0/4 hypotheses |
| L2 | exp_l2_edge_ablation.py | Yes | — | ⚠️ FAIL | All edge deltas negligible (max 0.013); model ignores topology |
| L3 | exp_l3_attention_visualization.py | Yes | — | ⏳ not run | — |
| L4 | exp_l4_gradient_saliency.py | Yes | — | ⚠️ FAIL | `complexity` dominates all 10 classes uniformly (34–36%) |
| L5 | exp_l5_probing_classifiers.py | Yes | — | ⏳ not run | — |
| L6 | exp_l6_counterfactual_contracts.py | Yes | — | ⏳ not run | — |
| L7 | exp_l7_calibration_size_analysis.py | Yes | — | ⏳ not run | — |
| L8 | exp_l8_permutation_importance.py | Yes | — | ⏳ not run | — |
| L9 | exp_l9_attention_rollout.py | Yes | — | ⏳ not run | — |
| L10 | exp_l10_training_ablation.py | Yes | — | ⏳ not run | — |

### Group S — Structural Analysis
| ID | Script | Needs ckpt | Phase 1 (v9) | Phase 2 (Run7) | Key finding |
|---|---|---|---|---|---|
| S1 | exp_s1_structural_trace.py | No | ⚠️ FAIL† | — | — |
| S2 | exp_s2_edge_enrichment.py | No | ⚠️ FAIL† | — | — |
| S3 | exp_s3_feature_distribution.py | No | ✅ PASS | — | No shortcuts, max Cohen's d=1.02 |
| S4 | exp_s4_icfg_path_audit.py | No | ⚠️ FAIL† | — | — |

†Phase 1 FAIL = threshold calibrated on Run 4; not a data quality issue. Recalibrate for Run 7 if needed.

---

## Phase 2 Summary — What We Learned

**Confirmed working (PASS):**
- B1: ISSUE-1 data fix delivered real Phase 2 gradient signal (5–8× improvement over Run 4)
- B2: Individual eyes well-calibrated; miscalibration is in the final ensemble aggregation layers
- B3: JK Phase3 drift is uniform across all classes — not a sign of class-selective failure

**Confirmed broken (FAIL):**
- A3/B3/L1 together: JK routing is non-functional. Phase3 drifted globally, attention is near-uniform (99.5% max entropy). The 3-phase GNN produces no per-class specialisation.
- L4/B4 together: Model learned `complexity` as a universal proxy. All class-specific features (`return_ignored`, `external_call_count`, `uses_block_globals`) show zero discriminative elevation.
- L2: Model ignores edge topology entirely. Removing any single edge type barely changes predictions. The GNN is acting on node features, not relational structure.
- A4: GNN auxiliary heads non-discriminative for 9/10 classes when used in isolation.

**Root cause of F1 ceilings (0.234 UnusedReturn, 0.164 Timestamp/DoS):**
The model learned a complexity-based proxy classifier. Without DEF_USE edges AND active training pressure to use class-specific node features, it cannot detect UnusedReturn or Timestamp patterns specifically. Simply adding RC5 edges will not help unless the model is simultaneously forced to use them.

---

## Remaining Tier 2/3 Experiments (not run)

| ID | Tier | Value now | Reason lower priority |
|----|------|-----------|----------------------|
| E4 | 2 | Medium | CF edge direction — mostly answered by L2 (negligible edge impact) |
| L5 | 2 | High | Linear probing — still useful to quantify GNN vs TF class info separately |
| L8 | 2 | Medium | Permutation importance — mostly answered by L4 (complexity dominates) |
| L3 | 3 | Low | Attention heatmaps — visualisation only |
| L7 | 3 | Low | Calibration vs contract size — BUG-C4 complicates interpretation |
| L6 | 3 | Low | Counterfactual contracts — expensive, low priority |
| L9 | 3 | Low | Attention rollout — secondary |
| L10 | 3 | Low | Training ablation — generates run commands, not results |
