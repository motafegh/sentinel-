# SENTINEL ML Module

`ml/` owns the four-eye teacher architecture, historical Run12 inference/training compatibility, repaired-training preparation, evaluation utilities, interpretation, and model lineage.

> **Current authority:** Run12 is the **historical operational baseline**, not the repaired model. New training/evaluation work is governed by the current R4 DATA/representation/evidence decisions under [`docs/plan/ml-R4`](../docs/plan/ml-R4) and the canonical [ML handbook](../docs/handbook/05_ml_model_inference.md).

## Current state

Historical R4 G0–G7 remain PASSED. **Phase 8 is IN_PROGRESS; G8 and full training remain unauthorized.** No repaired R4 teacher has been trained or promoted.

The model architecture remains frozen while the DATA/representation/evaluation contract is repaired:

```text
graph x[N,12] + token tensor [4,512]
      ↓
GNN eye          128
Transformer eye  128
Fusion eye       128
CFG eye          128
      ↓ concatenate 512
classifier → 10 logits
```

Current boundaries:

- class count/order remains the locked ten-class order;
- fusion embedding remains exactly 128 values;
- current served checkpoint remains historical Run12 lineage;
- Run12 thresholds/calibration remain historical companions only;
- R4-D-010 makes historical graph schema v9 ineligible for a new full training run;
- R4-D-011 accepts the exact **V10 V2.6** physical representation lineage for controlled research/possible later training eligibility, but does not authorize training;
- R4-D-012 promotes `target_aware_guarded_v1` only for construction/evaluation of a fresh successor token/representation lineage that still requires separate physical acceptance;
- confirmed negatives remain zero;
- threshold fitting, calibration fitting, untouched acceptance, model-quality promotion, and the 100-epoch/full training run remain unsupported or unauthorized.

## Why Run12 is historical now

Run12 was trained before R4 established that many historical binary `0` cells represented unknown, unsupported, absent, dropped, or otherwise unresolved states rather than confirmed negatives.

The architecture itself remains useful, so R4 intentionally keeps the broad model contract frozen while repairing DATA semantics, leakage grouping, physical representation semantics, token selection, and evaluation authority first. The purpose of any later Phase-8 retrain is to measure the architecture against defensible current evidence—not to reuse Run12's old label assumptions or quality claims.

## Current repaired-training boundary

A repaired consumer must preserve, per class:

- nullable authorized target;
- training strength (`STRONG`, `WEAK`, `NONE`);
- effective loss mask;
- outcome/metric mask;
- leakage-safe dataset role;
- DATA/policy/representation lineage identity.

Unknown/masked cells must never be filled with `0` to satisfy a historical loss interface.

Current evidence roles remain deliberately constrained:

- optimizer supervision is positive-only under the accepted policy/roles;
- `MODEL_SELECTION` is limited and positive-only;
- confirmed negatives remain zero;
- `THRESHOLD_FIT = UNSUPPORTED_EMPTY`;
- `CALIBRATION_FIT = UNSUPPORTED_EMPTY`;
- `UNTOUCHED_ACCEPTANCE = UNSUPPORTED_EMPTY_FROZEN`.

GasException and UnusedReturn remain model output positions but are supervision-disabled in policy v1.

### Physical representation sequence before training

```text
historical v9 / Run12 compatibility
→ R4-D-011 accepted V10 V2.6 physical representation
→ R4-D-012 fresh guarded-selector successor candidate
→ binding + transition evidence + separate physical acceptance
→ objective/evaluation design and credible metric-role support
→ explicit training authorization, if all gates are satisfied
```

Do not mutate the R4-D-011 root or treat R4-D-012 selector promotion as training permission.

## Current inference API

The current API continues to support historical Run12 runtime continuity:

- `GET /health`
- `POST /predict`
- `POST /hotspots`
- `POST /fusion-embedding`

`/fusion-embedding` returns the 128-value teacher fusion vector used by the retained proxy/ZKML boundary. It does not prove source execution or the AGENTS verdict.

## Historical training utilities

Existing trainer/loss/threshold/calibration/promotion scripts remain useful as implementation mechanics and historical reproduction tools. Their existence does **not** authorize old evidence assumptions for a repaired model.

In particular:

- historical ASL consumed all binary zeros as negatives;
- historical threshold fitting cannot simply be repeated without an authorized threshold-fit population;
- historical temperature calibration cannot be reused with a new checkpoint;
- Run12 drift/threshold/calibration artifacts do not become compatible merely because a new model keeps ten outputs;
- a physically accepted representation does not establish model discrimination or production quality.

## Main source areas

```text
ml/src/models/            four-eye architecture
ml/src/inference/         historical Run12/current serving boundary
ml/src/training/          Phase-8-compatible training mechanics
ml/src/datasets/          R4-aware dataset/loader seams plus historical compatibility
ml/src/preprocessing/     inference preprocessing
ml/scripts/               training/evaluation/promotion utilities
```

## Verification

```bash
export TMPDIR=/tmp TMP=/tmp TEMP=/tmp
ml/.venv/bin/python -m pytest ml/tests -q
python3 docs/handbook/tools/verify_handbook.py static
python3 docs/plan/ml-R4/scripts/p6_validate_frozen_partitions.py
```

A passing ML suite proves implementation behavior in its declared scope, not repaired-model quality. Any later quality claim must bind to the exact DATA/representation artifact, training config, checkpoint, and authorized evidence population.

## Permanent rules for the repaired retrain

- Do not overwrite Run12; retain it as historical baseline/rollback evidence.
- Do not reuse Run12 thresholds/calibration automatically.
- Do not manufacture negative targets from unknown data.
- Do not treat weak labels as strong without explicit checkpoint-bound config.
- Do not compute discrimination metrics that require trusted negatives when the evidence population lacks them.
- Do not use historical v9 as the new-full-training graph lineage.
- Do not rewrite R4-D-011 tokens in place; R4-D-012 requires a new versioned candidate.
- Keep the 10-output/fusion-128 architecture frozen unless an explicit later architecture decision unfreezes it.
- Redistill/regenerate ZKML only after a repaired teacher candidate is actually selected.
- Do not launch the full training run without an explicit later authorization record.

For current detail, see [ML model/inference](../docs/handbook/05_ml_model_inference.md), [training/quality](../docs/handbook/06_ml_training_quality.md), [evaluation](../docs/handbook/13_evaluation.md), and [current status](../docs/handbook/16_current_status.md).
