# ml/CLAUDE.md

Instructions for AI-assisted work inside `ml/`.

Read this file once at session start, then inspect the actual source/tests and current R4 artifacts needed for the task.

## Current ML meaning

SENTINEL uses a four-eye ten-output model over v9 graph/token representations. The **current served Run12 checkpoint is a historical operational baseline** trained before R4 repaired DATA label semantics.

Do not treat Run12 weights, thresholds, calibration, old validation splits, or historical binary `y[10]` targets as the repaired model/evidence contract.

Current DATA/ML authority for new work:

1. executable ML/DATA source;
2. `docs/plan/ml-R4/specs/data_vnext_policy_v1.json`;
3. `docs/plan/ml-R4/manifests/p6_partition_manifest.json` and support/acceptance manifests;
4. relevant R4 ADR/decision/risk records;
5. canonical handbook pages 04–06/13/16.

Stable `main` is through R4 G6. Phase 7 DATA vNext implementation remains candidate until local representation binding/G7. Phase 8 retraining is not authorized until G7.

## Key source locations

| Concern | Path |
|---|---|
| model architecture | `ml/src/models/sentinel_model.py` |
| canonical graph/class schema | `data_module/sentinel_data/representation/graph_schema.py` |
| historical trainer | `ml/src/training/trainer.py` |
| historical loss | `ml/src/training/losses.py` |
| inference API | `ml/src/inference/api.py` |
| predictor/checkpoint/threshold compatibility | `ml/src/inference/predictor.py` |
| inference preprocessing | `ml/src/inference/preprocess.py` |
| historical dataset/collate seam | `ml/src/datasets/` |
| R4 DATA policy | `docs/plan/ml-R4/specs/data_vnext_policy_v1.json` |
| R4 frozen roles | `docs/plan/ml-R4/manifests/p6_partition_manifest.json` |
| R4 Phase 8 plan | `docs/plan/ml-R4/phases/09_PHASE_8_EXISTING_MODEL_RETRAINING.md` |
| R4 Phase 9 plan | `docs/plan/ml-R4/phases/10_PHASE_9_EVALUATION_CALIBRATION_AND_POLICY.md` |
| ML ADR history | `docs/ml/adr/` |
| historical/audit findings | `ml/audit_docs/` and dated ML docs |

Read source before asserting constants/fields. Do not use remembered values when they can be checked.

## Permanent R4 constraints

### Unknown is not negative

The historical consumer path treated every binary `0` as a supervised negative. That behavior is **not valid for DATA vNext**.

Future vNext-aware training must carry:

- authorized nullable target;
- training strength (`STRONG`, `WEAK`, `NONE`);
- effective loss mask;
- outcome/metric mask;
- frozen dataset role;
- DATA/policy/config lineage.

Never fill an unknown/masked/disabled vNext cell with zero just to satisfy an old loss API.

### Current class support

The ten-output order remains locked. GasException and UnusedReturn are supervision-disabled under `data-vnext-policy-v1` until later evidence-backed policy changes.

DIVE Front Running→TransactionOrderDependence is weak-positive only. Weak does not become strong or metric-grade evidence automatically.

### Evaluation roles

Current first-baseline role limitations:

- training strong: supported;
- training weak: supported for DIVE TOD only;
- training unlabeled: supported;
- model selection: positive-only limited;
- threshold fit: `UNSUPPORTED_EMPTY`;
- calibration fit: `UNSUPPORTED_EMPTY`;
- untouched acceptance: `UNSUPPORTED_EMPTY_FROZEN`.

Do not run a historical utility and then infer that the corresponding evidence role exists. A threshold/calibration script executing successfully is not authorization to fit policy on unknown/exposed data.

### Run12 compatibility

Preserve Run12/checkpoint companions for reproducibility and comparison. Do not overwrite them with repaired artifacts.

A new checkpoint that keeps the same architecture is still semantically new and must bind:

- exact DATA vNext artifact/policy/roles;
- training config and seed/initialization;
- strong/weak numeric optimization handling;
- checkpoint hash;
- checkpoint-selection evidence/limitations;
- any future authorized threshold/calibration artifacts.

Do not automatically reuse Run12 thresholds/calibration/drift/proxy-agreement evidence.

## Before modifying source

1. Determine whether the task touches historical Run12 compatibility or the repaired vNext path.
2. Read the exact source/tests.
3. If DATA semantics/evidence roles are involved, read the current R4 policy/manifest/ADR before coding.
4. Preserve the architecture freeze unless an explicit later R4 decision unfreezes it.
5. Add failure tests for missing masks/strength/roles where relevant.
6. Do not weaken DATA semantics to minimize ML changes; adapt the consumer correctly.

## Training / evaluation specs

`ml/testing_specs/` and older calibration/promotion documents are useful **historical/supplementary mechanics**, not stronger current authority than R4.

Use them when they match the current task, but verify assumptions against R4 first—especially:

- split/role names;
- binary-negative semantics;
- threshold fitting;
- calibration fitting;
- checkpoint promotion;
- acceptance/test claims.

If a testing spec conflicts with current R4 policy/roles, update/supersede the spec or treat it as historical. Do not bend R4 evidence to satisfy the old spec.

## Coding conventions

- type hints on ML source interfaces;
- import canonical schema constants; do not duplicate class/graph constants;
- training config belongs in explicit versioned configuration/dataclasses;
- new metrics/log fields should use structured logging;
- tests live under `ml/tests/` and should mirror the relevant source area;
- keep scripts single-purpose;
- decision numbers require measured evidence and explicit config;
- no silent failures/skips/defaults that can contaminate evaluation.

## Validation discipline

Use focused tests first, then the relevant module/R4 gate.

```bash
export TMPDIR=/tmp TMP=/tmp TEMP=/tmp
ml/.venv/bin/python -m pytest ml/tests -q
python3 docs/plan/ml-R4/scripts/p6_validate_frozen_partitions.py
python3 docs/handbook/tools/verify_handbook.py static
```

Phase-7/8/9-specific gates become authoritative only on branches where those artifacts/implementations exist.

## Session handoff

If work changes a checkpoint, DATA/ML semantic decision, training/evaluation policy, or important bug:

- update the appropriate committed R4/ADR/register when it is durable project state;
- preserve run/config/artifact hashes;
- retain historical artifacts rather than overwriting them;
- update canonical handbook/current status if the project-state boundary changed.

Do not leave a consequential model/data decision only in conversation or a private scratchpad.
