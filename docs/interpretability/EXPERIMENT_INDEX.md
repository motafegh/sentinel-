# SENTINEL Interpretability Experiment Index — Run 5

**Status:** Pending Run 5 checkpoint  
**Data:** v9 (41,576 graphs, graphcodebert-base tokens, `cached_dataset_v9.pkl`)  
**Splits:** `ml/data/splits/v9_deduped/` — train=29,103 / val=6,236 / test=6,237  
**Checkpoint:** `ml/checkpoints/sentinel_best.pt` (populate after Run 5)

> Previous Run 4 results archived at `docs/interpretability/archive_run4/`

---

## Experiment Suite (25 scripts)

All scripts in `ml/scripts/interpretability/`. Run from project root with:
```bash
source ml/.venv/bin/activate
TRANSFORMERS_OFFLINE=1 PYTHONPATH=. python ml/scripts/interpretability/<script>.py \
  --checkpoint ml/checkpoints/sentinel_best.pt \
  --out ml/interpretability_results/<exp_id>/
```

### Group A — Architecture Validation
| ID | Script | What it measures | Needs checkpoint |
|---|---|---|---|
| A1 | exp_a1_pooling_audit.py | GNN per-phase pooling correctness | No |
| A2 | exp_a2_cfg_inheritance.py | CFG/inheritance edge coverage in v9 graphs | No |
| A3 | exp_a3_jk_entropy_logging.py | JK attention entropy distribution | Yes |
| A4 | exp_a4_aux_contribution.py | Aux head (Phase2 + CEI) contribution to loss | Yes |

### Group B — Training Diagnostics
| ID | Script | What it measures | Needs checkpoint |
|---|---|---|---|
| B1 | exp_b1_phase2_gradient_norm.py | Phase2/Phase1 grad norm ratio per class | Yes |
| B2 | exp_b2_per_eye_ece.py | Per-eye calibration (GNN / TF / Fused ECE) | Yes |
| B3 | exp_b3_jk_weight_distribution.py | JK attention weight distribution per phase | Yes |
| B4 | exp_b4_unusedreturn_saliency.py | UnusedReturn gradient saliency | Yes |

### Group E — Graph Expressivity
| ID | Script | What it measures | Needs checkpoint |
|---|---|---|---|
| E1 | exp_e1_receptive_field.py | k-hop reachability for Phase2 edge types | No |
| E2 | exp_e2_wl_distinguishability.py | WL graph distinguishability | No |
| E3 | exp_e3_message_propagation_sim.py | Message propagation simulation | No |
| E4 | exp_e4_direction_sensitivity.py | Directional edge sensitivity | Yes |

### Group L — Model Behaviour
| ID | Script | What it measures | Needs checkpoint |
|---|---|---|---|
| L1 | exp_l1_jk_weight_analysis.py | JK weight analysis per layer | Yes |
| L2 | exp_l2_edge_ablation.py | Edge type ablation (F1 delta per edge type) | Yes |
| L3 | exp_l3_attention_visualization.py | Cross-attention heatmaps | Yes |
| L4 | exp_l4_gradient_saliency.py | Feature gradient saliency per class | Yes |
| L5 | exp_l5_probing_classifiers.py | Linear probing on GNN/TF embeddings | Yes |
| L6 | exp_l6_counterfactual_contracts.py | Counterfactual contract perturbations | Yes |
| L7 | exp_l7_calibration_size_analysis.py | Calibration vs contract size (Timestamp strata) | Yes |
| L8 | exp_l8_permutation_importance.py | Feature permutation importance | Yes |
| L9 | exp_l9_attention_rollout.py | Attention rollout through layers | Yes |
| L10 | exp_l10_training_ablation.py | Training ablation (edge types vs F1) | Yes |

### Group S — Structural Analysis
| ID | Script | What it measures | Needs checkpoint |
|---|---|---|---|
| S1 | exp_s1_structural_trace.py | Structural pattern tracing | No |
| S2 | exp_s2_edge_enrichment.py | Edge type enrichment per vuln class | No |
| S3 | exp_s3_feature_distribution.py | Node feature distribution across classes | No |
| S4 | exp_s4_icfg_path_audit.py | ICFG path audit for CEI detection | No |

---

## Validation Scripts
| Script | Purpose |
|---|---|
| val_finding1_jk_weights.py | Validate JK weight findings |
| val_finding2_proper_ablation.py | Validate edge ablation methodology |
| val_finding4_timestamp_size.py | Validate Timestamp size stratification |

---

## Priority Order for Run 5

Run graph-only scripts first (no checkpoint needed) to validate v9 data quality,
then model-dependent scripts once Run 5 checkpoint is available.

**Phase 1 — v9 data validation (run now):** A1, A2, E1, E2, E3, S1, S2, S3, S4  
**Phase 2 — model diagnostics (post Run 5):** A3, A4, B1–B4, E4, L1–L10
