# 06 — ML training, quality, interpretability, and MLOps

**Read this when:** you need to retrain, evaluate, calibrate, choose thresholds, interpret, or promote a teacher checkpoint.

**Skip this if:** you only need current Run12 inference; read [ML model and inference](05_ml_model_inference.md).

**Estimated reading time:** 15 minutes.

## 30-second summary

The next teacher retrain must use DATA vNext semantics, not the historical binary target contract. R4 keeps the existing four-eye architecture frozen but changes the training evidence: strong positives, weak positives, unlabeled/masked cells, disabled classes, and frozen leakage-safe roles must be handled explicitly. Current training scripts for ASL, threshold sweeping, temperature calibration, and promotion remain historical tooling; their existence does not authorize old validation assumptions for the repaired baseline.

## Just-enough mental model

```text
DATA vNext semantic state + r4-vnext-roles-v1
        ↓
Phase-8 trainer compatibility
(target + strength + masks + role)
        ↓
retrain same four-eye architecture
        ↓
positive-only limited model-selection evidence
        ↓
Phase 9 evaluation policy
(no trustworthy threshold/calibration fit unless new evidence exists)
```

Loss optimization, checkpoint selection, threshold fitting, calibration fitting, and untouched acceptance are separate evidence roles. R4 deliberately refuses to reuse one set for all of them.

## Actual runtime/source walkthrough

### Historical training stack

Existing source remains useful as mechanics/history:

- [`trainer.py`](../../ml/src/training/trainer.py) — optimization, validation, checkpointing;
- [`losses.py`](../../ml/src/training/losses.py) — historical multi-label ASL behavior;
- `tune_threshold.py` — historical per-class threshold search;
- `calibrate_temperature.py` — historical temperature-fitting utility;
- promotion/drift/interpretability scripts — historical model-quality infrastructure.

The critical historical limitation is that the legacy dataset/collate/loss seam treated every binary `0` as supervised negative evidence.

### R4 retraining contract

Phase 8 must adapt consumers—not weaken DATA vNext—to carry:

- nullable/authorized target;
- training strength (`STRONG`, `WEAK`, `NONE`);
- effective loss mask;
- outcome/metric mask;
- frozen Phase-6 role;
- policy/DATA lineage identity.

Weak numeric optimizer weight is **not** DATA truth. Phase 8 must choose and bind any numeric `WEAK` weighting explicitly in training config. Unknown/masked cells must never become zeros to satisfy an old loss API.

### Current role limitations

The first repaired baseline has:

- `TRAIN_STRONG` support across the eight enabled supervised classes;
- `TRAIN_WEAK` for authorized DIVE TOD weak positives;
- a large `TRAIN_UNLABELED` population;
- positive-only limited `MODEL_SELECTION` and `INTERNAL_AUDIT` strong holdouts;
- `THRESHOLD_FIT = UNSUPPORTED_EMPTY`;
- `CALIBRATION_FIT = UNSUPPORTED_EMPTY`;
- `UNTOUCHED_ACCEPTANCE = UNSUPPORTED_EMPTY_FROZEN`.

Therefore Phase 8/9 must not report ordinary full binary validation quality by silently using unknowns as negatives.

## Interfaces, data shapes, and configuration

A repaired checkpoint lineage must bind at least:

- exact DATA vNext manifest/policy/schema/role identities;
- frozen class order/v9 representation schema;
- training config and seed/environment;
- numeric strong/weak loss handling;
- architecture/checkpoint SHA-256;
- checkpoint-selection evidence and its role limitations;
- any later threshold/calibration artifact together with the evidence role that justified it;
- drift/probe/interpretability evidence used for release decisions.

GasException and UnusedReturn remain output positions but are supervision-disabled in vNext policy v1. Training/metric code must mask them accordingly unless a later evidence-backed policy re-enables them.

## Failure modes and current limitations

- Reusing legacy `y[10]` zeros recreates the original corruption.
- Treating `WEAK` as `STRONG` without explicit config silently changes source authority.
- Using positive-only model-selection data to compute F1/AUC/false-positive rate as if negatives were trusted is invalid.
- Running `tune_threshold.py` does not create a legitimate threshold-fit corpus.
- Running temperature calibration on an unauthorized role does not create trustworthy calibration.
- Run12 threshold/calibration companions remain historical and cannot be promoted with a vNext retrain.
- No untouched acceptance corpus exists for the first repaired baseline, so Phase 10 cannot make an untouched-acceptance promotion claim unless new separately protected evidence is added.

## Common change recipe

For the repaired retrain:

1. finish/verify G7 DATA vNext and physical representation binding;
2. make dataset/collate/loss/metrics v2-aware without altering vNext semantics;
3. choose explicit strong/weak optimization behavior and bind config;
4. train the frozen architecture from a declared initialization strategy;
5. use only `TRAIN_*` roles for optimization;
6. use `MODEL_SELECTION` only for the limited positive diagnostics it can support;
7. do not fit thresholds/calibration until an authorized role exists;
8. preserve raw per-class/role evidence and compare against Run12 without pretending corpus equivalence;
9. redistill/regenerate ZKML only after a teacher candidate is actually selected.

## Verification commands

```bash
export TMPDIR=/tmp TMP=/tmp TEMP=/tmp
ml/.venv/bin/python -m pytest ml/tests -q
python3 docs/plan/ml-R4/scripts/p6_validate_frozen_partitions.py
```

Historical threshold/calibration script smoke tests remain useful implementation checks, but are not evidence that Phase-9 fitting is currently authorized.

## Optional deep references

- [DATA artifacts](04_data_artifacts.md)
- [Evaluation](13_evaluation.md)
- [Current status](16_current_status.md)
- [R4 Phase 8 plan](../plan/ml-R4/phases/09_PHASE_8_EXISTING_MODEL_RETRAINING.md)
- [R4 Phase 9 plan](../plan/ml-R4/phases/10_PHASE_9_EVALUATION_CALIBRATION_AND_POLICY.md)

## Technical mastery layer

### Prerequisite knowledge

Know multi-label optimization, positive-unlabeled/partial-label problems, masks, class imbalance, calibration, thresholding, and strict dataset-role separation.

### Source map and reading order

Read historical trainer/loss code to understand the current consumer seam, then read the accepted R4 vNext policy, role manifests, Phase-8 plan, and only then design compatibility changes. Do not infer future training semantics from old threshold/calibration scripts.

### Execution trace and worked example

A SolidiFI Reentrancy row can contribute `target=1, STRONG` in a training role. An authorized DIVE TOD row can contribute `target=1, WEAK`. A DIVE DoS historical positive can remain `target=null, NONE`. The trainer must optimize only authorized cells and preserve the distinction in checkpoint lineage.

### Implementation practice

Make masks/strength explicit tensors and assert them in tests before adapting loss. Add tests where unknown/disabled cells would catastrophically change gradients if accidentally treated as zero.

### Review and ownership check

Can you name which evidence may fit weights, select checkpoints, fit thresholds, fit calibration, and provide untouched acceptance—and explain why three of those roles are intentionally empty today?
