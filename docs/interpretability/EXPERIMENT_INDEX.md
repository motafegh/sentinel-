# SENTINEL Interpretability Experiment Index — Run 7

**Status:** Phase 1 COMPLETE (v9 baseline) · Phase 2 IN PROGRESS (Run 7 ep39 checkpoint)  
**Checkpoint:** `ml/checkpoints/GCB-P1-Run7-v10-20260603_best.pt` (ep39, F1=0.3074)  
**Data:** v10, 41,576 graphs, `ml/data/cached_dataset_v10.pkl`  
**Splits:** `ml/data/splits/v10_deduped/` — train=29,103 / val=6,236 / test=6,237  
**Phase 2 results:** `ml/interpretability_results/phase2_run7_ep39_v10_2026-06-04/`

> Phase 1 results (v9 baseline, Run 5): `ml/interpretability_results/archive_phase1_run5_v9_2026-06-02/`  
> Previous Run 4 results: `docs/interpretability/archive_run4/`  
> Understanding doc: `docs/interpretability/SENTINEL-Understanding-Run7.md`

---

## Run Command Template

```bash
source ml/.venv/bin/activate
CKPT=ml/checkpoints/GCB-P1-Run7-v10-20260603_best.pt
OUT=ml/interpretability_results/phase2_run7_ep39_v10_2026-06-04

TRANSFORMERS_OFFLINE=1 PYTHONPATH=. python ml/scripts/interpretability/<script>.py \
  --checkpoint $CKPT \
  --out $OUT/<exp_id>/
```

---

## Experiment Suite

### Group A — Architecture Validation
| ID | Script | Needs ckpt | Phase 1 (v9) | Phase 2 (Run7) |
|---|---|---|---|---|
| A1 | exp_a1_pooling_audit.py | No | ✅ PASS | — |
| A2 | exp_a2_cfg_inheritance.py | No | ✅ PASS | — |
| A3 | exp_a3_jk_entropy_logging.py | Yes | — | ⏳ |
| A4 | exp_a4_aux_contribution.py | Yes | — | ⏳ |

### Group B — Training Diagnostics
| ID | Script | Needs ckpt | Phase 1 (v9) | Phase 2 (Run7) |
|---|---|---|---|---|
| B1 | exp_b1_phase2_gradient_norm.py | Yes | — | ⏳ |
| B2 | exp_b2_per_eye_ece.py | Yes | — | ⏳ |
| B3 | exp_b3_jk_weight_distribution.py | Yes | — | ⏳ |
| B4 | exp_b4_unusedreturn_saliency.py | Yes | — | ⏳ |

### Group E — Graph Expressivity
| ID | Script | Needs ckpt | Phase 1 (v9) | Phase 2 (Run7) |
|---|---|---|---|---|
| E1 | exp_e1_receptive_field.py | No | ⚠️ FAIL† | — |
| E2 | exp_e2_wl_distinguishability.py | No | ✅ PASS | — |
| E3 | exp_e3_message_propagation_sim.py | No | ⚠️ FAIL† | — |
| E4 | exp_e4_direction_sensitivity.py | Yes | — | ⏳ |

### Group L — Model Behaviour
| ID | Script | Needs ckpt | Phase 1 (v9) | Phase 2 (Run7) |
|---|---|---|---|---|
| L1 | exp_l1_jk_weight_analysis.py | Yes | — | ⏳ |
| L2 | exp_l2_edge_ablation.py | Yes | — | ⏳ |
| L3 | exp_l3_attention_visualization.py | Yes | — | ⏳ |
| L4 | exp_l4_gradient_saliency.py | Yes | — | ⏳ |
| L5 | exp_l5_probing_classifiers.py | Yes | — | ⏳ |
| L6 | exp_l6_counterfactual_contracts.py | Yes | — | ⏳ |
| L7 | exp_l7_calibration_size_analysis.py | Yes | — | ⏳ |
| L8 | exp_l8_permutation_importance.py | Yes | — | ⏳ |
| L9 | exp_l9_attention_rollout.py | Yes | — | ⏳ |
| L10 | exp_l10_training_ablation.py | Yes | — | ⏳ |

### Group S — Structural Analysis
| ID | Script | Needs ckpt | Phase 1 (v9) | Phase 2 (Run7) |
|---|---|---|---|---|
| S1 | exp_s1_structural_trace.py | No | ⚠️ FAIL† | — |
| S2 | exp_s2_edge_enrichment.py | No | ⚠️ FAIL† | — |
| S3 | exp_s3_feature_distribution.py | No | ✅ PASS | — |
| S4 | exp_s4_icfg_path_audit.py | No | ⚠️ FAIL† | — |

†FAIL = threshold calibrated on Run 4, not a data quality issue.

---

## Priority Order for Phase 2

**Tier 1 — Run first (validates architecture + training fixes):**
- B1 (Ph2/Ph1 gradient ratio — confirms ISSUE-1 fix)
- B3 (JK weight distribution — characterises Phase 3 drift)
- L1 (JK weights per class)
- A3 (JK entropy from logs)
- A4 (aux head contribution)
- B2 (per-eye ECE)

**Tier 2 — Run next (model behaviour analysis):**
- L4 (gradient saliency)
- L2 (edge ablation)
- B4 (UnusedReturn saliency)
- E4 (direction sensitivity)
- L5 (probing classifiers)
- L8 (permutation importance)

**Tier 3 — Lower priority (expensive or secondary):**
- L3 (attention heatmaps)
- L7 (calibration vs size)
- L6 (counterfactual contracts)
- L9 (attention rollout)
- L10 (training ablation — generates commands, not results)
