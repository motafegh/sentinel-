# 06 — ML training, quality, interpretability, and MLOps

**Read this when:** you need to understand repaired Phase-8 training mechanics, checkpoint selection, evaluation/calibration limits, or future teacher promotion.

**Skip this if:** you only need current Run12 inference; read [ML model and inference](05_ml_model_inference.md).

**Estimated reading time:** 15 minutes.

## 30-second summary

SENTINEL now has repaired Phase-8 training **mechanics**, but it does not yet have authorization for the full repaired training run or a promoted repaired checkpoint. The existing four-eye architecture remains frozen. Current training code carries explicit vNext targets/masks/strength, group-aware sampling/binding, model-selection logic, checkpoint metadata, and Phase-8 configuration; however, the DATA/representation/evaluation gates still control whether that machinery may consume a candidate at full scale.

The next eligible training lineage must use accepted logical V3 semantics/roles, the D-011 V10 V2.6 physical graph authority, and a separately accepted D-012 guarded-selector successor token lineage. Run12 weights/optimizer/threshold/calibration state remain historical only. Confirmed negatives remain zero, so ordinary binary threshold/calibration/false-positive claims are not currently supported.

## Just-enough mental model

```text
accepted semantic policy + logical V3 role/group authority
        +
accepted D-011 V10 graph lineage
        +
D-012 guarded-selector successor (still needs physical acceptance)
        ↓
exact run binding / Phase-8 settings
        ↓
masked + strength-aware optimization mechanics
        ↓
limited positive-only model-selection evidence
        ↓
checkpoint candidate
        ↓
NO automatic threshold/calibration/model-quality promotion
        ↓
later evaluation/promotion only if new evidence authorizes it
```

Loss optimization, checkpoint selection, threshold fitting, calibration fitting, untouched acceptance, and final promotion are separate evidence responsibilities.

## Actual runtime/source walkthrough

### Historical training stack

The large historical trainer remains useful for Run12 mechanics/reproduction:

- [`trainer.py`](../../ml/src/training/trainer.py) — historical optimization/validation/checkpointing;
- [`losses.py`](../../ml/src/training/losses.py) — historical multi-label loss behavior;
- threshold/calibration/promotion scripts — historical model-quality infrastructure.

Its key semantic limitation is the legacy binary `y[10]` contract, where historical zeros could reach loss/evaluation as negatives. That behavior must not be reused for repaired training truth.

### Current Phase-8 training mechanics

The repaired path has dedicated vNext components under `ml/src/training/`, including run binding, group-aware sampling, vNext losses/epoch mechanics, checkpoint metadata, model construction/parameter groups, and Phase-8 settings.

Those components are designed around explicit repaired semantics rather than weakening DATA to fit the legacy trainer. A run must preserve:

- nullable/authorized target state;
- training strength (`STRONG`, `WEAK`, `NONE`);
- effective loss mask;
- outcome/model-selection eligibility;
- accepted logical group/role identity;
- exact representation/token lineage;
- policy/class-order identity;
- run configuration, seed, source commit, and artifact binding.

Numeric weak-positive weighting is an optimizer configuration, not DATA truth.

### Current DATA/representation prerequisite

The training mechanics do not choose their own DATA authority. Current R4 ordering is:

1. D-009 logical V3 grouping/roles remain the accepted logical authority.
2. D-010 makes v9 ineligible for a new full run.
3. D-011 accepts the exact V10 V2.6 physical graph/control-token root.
4. D-012 promotes `target_aware_guarded_v1` only for a **fresh successor candidate**.
5. That successor still requires generation, binding, review, and physical acceptance.
6. Only a later explicit decision may authorize the full repaired training run against the accepted successor.

D-011 itself grants neither selector promotion nor training authority.

### Current model-selection/evaluation limitations

The repaired path has positive-only limited model-selection/internal-audit evidence. That can support bounded positive-selection diagnostics but not ordinary binary discrimination claims.

Current policy/evidence still records:

- `THRESHOLD_FIT = UNSUPPORTED_EMPTY`;
- `CALIBRATION_FIT = UNSUPPORTED_EMPTY`;
- `UNTOUCHED_ACCEPTANCE = UNSUPPORTED_EMPTY_FROZEN`;
- confirmed negatives = zero.

Therefore running historical threshold/calibration utilities would test code mechanics, not create legitimate evidence authority.

### Run12 separation

Run12 remains the historical operational baseline served today. For repaired training:

- do not reuse learned Run12 weights as repaired truth unless an explicit initialization decision permits a narrowly stated use;
- do not reuse Run12 optimizer/scheduler state;
- do not reuse Run12 thresholds/calibration as repaired policy;
- do not treat historical model-selection metrics as corpus-equivalent evidence;
- preserve Run12 artifacts as rollback/comparison history.

The architecture can remain shape-compatible while checkpoint meaning changes completely.

## Interfaces, data shapes, and configuration

A repaired checkpoint lineage must bind at least:

- `data-vnext-policy-v1` semantic identity;
- accepted logical V3 grouping/role identity;
- exact accepted graph/token candidate identity and digest;
- locked ten-class order;
- four-eye architecture/config identity;
- training config, seed, source commit, runtime precision, optimizer/scheduler behavior;
- explicit numeric handling for `WEAK` positives;
- checkpoint-selection evidence and its limitations;
- checkpoint SHA-256 / run binding;
- any later threshold/calibration artifact together with the specific evidence role that authorized it.

GasException and UnusedReturn remain output positions but supervision-disabled under policy v1 unless later evidence-backed policy changes that status.

## Failure modes and current limitations

- Reusing legacy `y[10]` zeros recreates the original label corruption.
- Treating `WEAK` as `STRONG` without explicit config silently changes source authority.
- Using historical G6 role identity instead of accepted logical V3 can reintroduce superseded grouping assumptions.
- Training on v9 for the new full run violates D-010.
- Training directly on D-011 while claiming D-012 guarded selection has been applied violates the selector decision.
- Launching the full run before successor physical acceptance/explicit authorization bypasses R4 gate authority.
- Using positive-only model-selection data to claim F1/AUC/false-positive rate as if negatives were trusted is invalid.
- Running threshold or temperature-calibration code does not create an authorized fitting population.
- No untouched acceptance corpus exists for the repaired path today.

## Common change recipe

For repaired training work:

1. verify the exact current R4 restart/acceptance authority first;
2. bind the dataset to accepted logical roles and exact physical representation/token artifacts;
3. preserve target/state/strength/masks without binary fallback;
4. choose and record explicit optimizer/scheduler/precision/seed behavior;
5. use only optimizer-authorized roles/cells for gradient updates;
6. use model-selection evidence only for claims it can actually support;
7. preserve raw run/checkpoint lineage and compare to Run12 with corpus/semantic caveats;
8. do not fit thresholds/calibration or claim untouched acceptance without an authorized role;
9. redistill/regenerate ZKML only after a repaired teacher candidate is actually selected/promoted;
10. require explicit later governance before a full training launch or model-quality promotion.

## Verification commands

```bash
export TMPDIR=/tmp TMP=/tmp TEMP=/tmp
ml/.venv/bin/python -m pytest ml/tests -q
python3 docs/handbook/tools/verify_handbook.py static
python3 docs/plan/ml-R4/scripts/p6_validate_frozen_partitions.py
```

Historical G6 validation remains useful compatibility evidence. Current D-011/D-012 and any future successor/full-run authority require their own current R4 binding/acceptance records.

## Optional deep references

- [Architecture](01_architecture.md)
- [DATA artifacts / ML seam](04_data_artifacts.md)
- [ML model and inference](05_ml_model_inference.md)
- [Evaluation](13_evaluation.md)
- [Current status](16_current_status.md)
- [R4 Phase 8 plan](../plan/ml-R4/phases/09_PHASE_8_EXISTING_MODEL_RETRAINING.md)
- [R4 Phase 9 plan](../plan/ml-R4/phases/10_PHASE_9_EVALUATION_CALIBRATION_AND_POLICY.md)

## Technical mastery layer

### Prerequisite knowledge

Know multi-label/partial-label optimization, masks, class imbalance, positive-only evaluation limits, checkpoint selection, calibration/thresholding, run binding, and strict dataset-role separation.

### Source map and reading order

Read the current R4 status/D-011/D-012 decisions first, then `ml/src/datasets/vnext_dataset.py`, `ml/src/training/vnext_*`, group sampler/parameter-group/config code, and only afterward the historical trainer/threshold/calibration utilities for compatibility context.

### Execution trace and worked example

A strong positive cell can enter the optimizer with full configured weight, while an authorized `WEAK` positive enters with the run's explicitly bound weak weight and an unknown cell contributes no loss. Model selection can compare positive-selection behavior on its authorized holdout, but it cannot infer false-positive rate without trustworthy negatives. Even a mechanically successful checkpoint remains only a candidate until the surrounding DATA/evaluation/promotion authority exists.

### Implementation practice

Make state/strength/masks and run binding explicit data structures, then assert them in tests before optimizing. Treat missing representation identity, role authority, or evaluation support as hard/degraded states—not as opportunities to substitute historical defaults.

### Review and ownership check

Can you separate (a) training mechanics that now exist, (b) exact DATA/representation artifacts they are allowed to consume, (c) evidence that may select a checkpoint, and (d) still-missing evidence needed for thresholds/calibration/untouched/model-quality promotion?