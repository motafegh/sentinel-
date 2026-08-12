# SENTINEL ML Module

`ml/` owns the four-eye teacher architecture, historical Run12 inference/training compatibility, future DATA-vNext-aware retraining, evaluation utilities, interpretation, and model lineage.

> **Current authority:** Run12 is the **historical operational baseline**, not the repaired model. New training/evaluation work is governed by the R4 DATA vNext policy/roles under [`docs/plan/ml-R4`](../docs/plan/ml-R4) and the canonical [ML handbook](../docs/handbook/05_ml_model_inference.md).

## Current state

The model architecture remains frozen for the first repaired retrain:

```text
v9 graph + [4,512] tokens
      ↓
GNN eye          128
Transformer eye  128
Fusion eye       128
CFG eye          128
      ↓ concatenate 512
classifier → 10 logits
```

Important compatibility facts:

- graph schema: `v9`;
- node feature dimension: 12;
- class count/order: locked ten-class order;
- fusion embedding: exactly 128 values;
- current served checkpoint: historical Run12 lineage;
- Run12 thresholds/calibration: historical companions only;
- no repaired DATA-vNext teacher has been trained/promoted yet.

R4 Phase 7 must pass before Phase 8 retraining begins.

## Why Run12 is historical now

Run12 was trained before R4 established that many historical binary `0` cells represented unknown, unsupported, absent, dropped, or otherwise unresolved states rather than confirmed negatives.

The architecture itself is still valuable, so R4 intentionally keeps it frozen and changes the DATA/training contract first. The purpose of Phase 8 is to measure what the same broad architecture can learn from defensible DATA vNext semantics before redesigning the network.

## DATA vNext training boundary

The future vNext consumer must carry, per class:

- nullable authorized target;
- training strength (`STRONG`, `WEAK`, `NONE`);
- effective loss mask;
- outcome/metric mask;
- frozen dataset role;
- DATA/policy lineage identity.

Unknown/masked cells must never be filled with `0` to satisfy the historical loss interface.

Current first-baseline evidence roles are deliberately constrained:

- `TRAIN_STRONG`: supported;
- `TRAIN_WEAK`: DIVE TOD weak-positive only;
- `TRAIN_UNLABELED`: supported;
- `MODEL_SELECTION`: positive-only limited;
- `THRESHOLD_FIT = UNSUPPORTED_EMPTY`;
- `CALIBRATION_FIT = UNSUPPORTED_EMPTY`;
- `UNTOUCHED_ACCEPTANCE = UNSUPPORTED_EMPTY_FROZEN`.

GasException and UnusedReturn remain model output positions but are supervision-disabled in policy v1.

## Current inference API

The current API continues to support historical Run12 runtime continuity:

- `GET /health`
- `POST /predict`
- `POST /hotspots`
- `POST /fusion-embedding`

`/fusion-embedding` returns the 128-value teacher fusion vector used by the retained proxy/ZKML boundary. It does not prove source execution or the AGENTS verdict.

## Historical training utilities

Existing trainer/loss/threshold/calibration/promotion scripts remain useful as implementation mechanics and historical reproduction tools. Their existence does **not** authorize old evidence assumptions for the repaired model.

In particular:

- historical ASL consumed all binary zeros as negatives;
- historical threshold fitting cannot simply be repeated without an authorized threshold-fit role;
- historical temperature calibration cannot be reused with a new checkpoint;
- Run12 drift/threshold/calibration artifacts do not become compatible merely because a new model keeps ten outputs.

## Main source areas

```text
ml/src/models/            four-eye architecture
ml/src/inference/         Run12/current serving boundary
ml/src/training/          historical trainer/loss mechanics; Phase-8 adaptation target
ml/src/datasets/          historical v1 dataset/collate seam today
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

A passing ML suite proves implementation behavior, not repaired-model quality. Phase 8/9 must bind quality claims to the exact DATA vNext artifact, training config, and authorized evidence role.

## Permanent rules for the repaired retrain

- Do not overwrite Run12; retain it as historical baseline/rollback evidence.
- Do not reuse Run12 thresholds/calibration automatically.
- Do not manufacture negative targets from unknown data.
- Do not treat weak labels as strong without explicit checkpoint-bound config.
- Do not compute discrimination metrics that require trusted negatives when the role lacks them.
- Keep the 10-output/fusion-128 architecture frozen unless an explicit later architecture decision unfreezes it.
- Redistill/regenerate ZKML only after a repaired teacher candidate is actually selected.

For current detail, see [ML model/inference](../docs/handbook/05_ml_model_inference.md), [training/quality](../docs/handbook/06_ml_training_quality.md), [evaluation](../docs/handbook/13_evaluation.md), and [current status](../docs/handbook/16_current_status.md).
