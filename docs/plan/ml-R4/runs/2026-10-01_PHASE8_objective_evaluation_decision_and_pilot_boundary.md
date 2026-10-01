# Phase-8 objective/evaluation decision and bounded-pilot boundary

Date: 2026-10-01  
Decision: R4-D-014  
Outcome: `ACCEPT_BOUNDED_POSITIVE_ONLY_CONTROL_HOLD_FULL_TRAINING`

## Result

M0 and M1 are complete. The evidence does not support negative supervision, PU/nnPU risk optimization, discrimination metrics, threshold fitting, calibration, untouched acceptance, checkpoint promotion, or the full 100-epoch run.

R4-D-014 therefore freezes the existing masked-positive loss semantics as a **control objective** for the next engineering stages only.

## Authorized next scope

- M3: integrate the exact accepted logical V3 + R4-D-011 graph parent + R4-D-013 guarded-token lineage into a new fail-closed training seam.
- M4: after M3 verification, run a separately specified bounded pilot for mechanics/dynamics/recovery evidence.

## Not authorized

- full 100-epoch training;
- PU/nnPU;
- target-zero optimizer supervision;
- threshold or calibration fitting;
- untouched acceptance use;
- production promotion;
- inference replacement;
- ZKML rebinding.

## Evaluation semantics

`best_positive_nll` remains a positive-fit diagnostic only. A pilot checkpoint selected by that metric is not evidence of general classifier quality.

Confirmed-negative evidence stays under R4-GAP-007. Candidate #2 remains pending genuinely independent verification; no target changes follow from this decision.

Machine-readable authority:

`evidence/2026-10-01_phase8_objective_evaluation/decision.json`.
