# ML M0 — current training-seam audit

**Date:** 2026-10-01  
**Plan:** `02_ML_TRAINING_EVALUATION_COMPLETION_PLAN.md`  
**Authority:** R4-D-013 accepted guarded-token physical lineage; R4-D-011 immutable V10 graph/control parent  
**State:** IN PROGRESS — source tracing only; no full-training authorization

## Objective

Trace the executable repaired Phase-8 training seam end to end and identify
exactly what already matches the accepted R4-D-013 physical lineage versus what
still assumes an older representation/binding boundary.

The seam under audit is:

`accepted DATA/roles`
→ `VNextTrainingDataset`
→ `collate`
→ `group-aware sampling`
→ `model input`
→ `masked loss`
→ `model-selection diagnostics`
→ `run binding / run control`
→ `checkpoint metadata / resume`
→ `inference-consumer compatibility`.

## Non-negotiable current boundaries

- R4-D-013 accepted token lineage:
  `r4-v10-v26-guarded-tokens-v1`.
- Accepted guarded binding digest:
  `9885d7b88a46aff4102741d63eeaa0bd6968f0857f7bca6a389662bfaa158881`.
- R4-D-011 graph parent remains immutable:
  `d9f925588913e66476cfbc097bace7daa7e673295fe2a243760313d0bef5ebdd`.
- Logical role/group authority remains accepted V3.
- Token tensor contract remains `[4,512]`.
- Architecture remains `four_eye_v8` / `v8.1`.
- Confirmed negatives remain zero.
- Run12 learned state is historical only.
- G8 remains open; full training is unauthorized.

## Audit questions

1. Which exact publication/representation roots does the current dataset adapter
   consume or default to?
2. Does the adapter verify the R4-D-013 token lineage/digest, or merely load
   files by path?
3. Are V3 roles enforced exactly, including exclusion of `INTERNAL_AUDIT`
   from model selection and optimization?
4. What tensor/metadata boundary does collate expose to the model?
5. What exact cells can reach optimization, and can any target other than
   `1.0` reach the current loss?
6. What metrics decide checkpoint/model selection today, and what claims do
   those metrics actually support?
7. What does the run binding hash/bind today: dataset manifest, representation
   digest, selector, source commit, objective config, architecture, seed,
   optimizer/scheduler, runtime precision?
8. Does resume/checkpoint validation fail closed on lineage/config mismatch?
9. What does the current runner still require before it can consume R4-D-013?
10. What metadata will a future repaired checkpoint need so inference and later
    ZKML work can distinguish it from Run12?

## Stop line

Do not alter objective semantics, create negative targets, authorize training,
fit thresholds/calibration, or launch a long run during M0. Any source change
must be limited to a clearly evidenced seam correction and separately validated.
