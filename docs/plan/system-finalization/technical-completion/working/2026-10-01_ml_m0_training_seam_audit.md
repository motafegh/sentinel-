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

## Findings

### 1. Canonical full-run entry point is still historical G7/v9

`ml/src/training/vnext_runner.py` currently constructs
`VNextTrainingDataset`, imports `CANONICAL_G7_BINDING_DIGEST`, and calls
`build_run_binding()`.

Those contracts explicitly require:

- `sentinel-r4-vnext-v1`;
- status `VALIDATED_G7_CANDIDATE`;
- graph schema `v9`;
- G7 binding digest
  `7637461f6643d398c7a0446412fedd8877914c7b9ed41309dab45f18ed96f420`.

Therefore the canonical durable runner cannot currently truthfully consume the
R4-D-013 guarded lineage. Supplying the new representation directory by path
would not make the run R4-D-013-bound.

### 2. Logical-V3/V10 seams exist, but they predate R4-D-013

`LogicalV3TrainingDataset` correctly binds logical V3 grouping/roles and keeps
confirmed negatives at zero.

`LogicalV3V10TrainingDataset` and `build_v10_run_binding()` provide a
fail-closed future-V10 seam, but they still require the older
`V10_REPRESENTATION_ROOT_NAME = representations-r4-v3-candidate` and an older
V10 physical-acceptance/training-authorization manifest contract.

R4-D-013 instead accepts:

- root basename
  `representations-r4-v10-v26-guarded-v1-candidate`;
- token lineage `r4-v10-v26-guarded-tokens-v1`;
- guarded binding digest
  `9885d7b88a46aff4102741d63eeaa0bd6968f0857f7bca6a389662bfaa158881`;
- selector policy `target_aware_guarded_v1` with exact
  `historical_linspace_v1` fallback;
- R4-D-011 as immutable graph parent.

The old V10 adapter therefore cannot be reused unchanged as the R4-D-013
training adapter.

### 3. Role and supervision isolation are sound

The common supervision conversion fails closed on target `0` or any
non-`1.0` authorized target.

Legal Phase-8 dataset roles are only:

- `TRAIN_STRONG`;
- `TRAIN_WEAK`;
- `MODEL_SELECTION`.

Training and model-selection roles cannot be mixed in one dataset.
`INTERNAL_AUDIT` is not an allowed dataset role and therefore cannot enter the
current optimizer/model-selection loaders through this adapter.

Group-level no-signal siblings remain visible in frozen role counts but are
excluded from active optimizer/metric populations. Current population
assertions remain:

- TRAIN_STRONG frozen/active: 275 / 275;
- TRAIN_WEAK frozen/active: 773 / 577;
- skipped TRAIN_WEAK no-signal siblings: 196;
- active training contracts/groups: 852 / 703;
- MODEL_SELECTION contracts/groups: 56 / 51.

### 4. Collate and optimization semantics are explicit

`vnext_collate_fn` carries targets, effective-loss mask, outcome-metric mask,
and strength codes separately.

`masked_bce_positive_loss()` re-validates that every optimizer-authorized cell
is finite target `1.0`. Unknown cells cannot influence the loss. Strong and
weak positives are weighted explicitly.

The current optimizer objective remains `masked_positive_bce`. No target-zero,
pseudo-negative, threshold-tuning or calibration path is enabled.

### 5. Model-selection semantics are intentionally limited

`evaluate_positive_selection()` evaluates only cells authorized by
`outcome_metric_mask`.

The durable runner selects `best_positive_nll` by lower positive-only NLL.
The fixed 0.5 threshold is diagnostic only. This supports a positive-fit
checkpoint-selection diagnostic, not discrimination, specificity/FPR,
calibration, or a deployable threshold claim.

### 6. V10 model support exists, but the factory is wired to v9

`GNNEncoder` is schema-aware and can construct the 17-edge-type V10 embedding
and route V10 call-kind edges correctly when initialized with
`graph_schema_version="v10"`.

However `build_phase8_model()` currently passes only `FROZEN_ARCHITECTURE`,
and that configuration does not include `graph_schema_version`.
`SentinelModel` therefore defaults to `v9`.

This is an integration/configuration gap, not evidence for an architecture
change.

### 7. Run binding and resume mechanics are strong but bind the wrong lineage

The current run binding already captures:

- clean tracked source commit;
- frozen architecture/model version/config;
- class order;
- publication manifest hash;
- representation digest;
- policy/partition hashes;
- train/model-selection population counts;
- seed and weak-positive weight;
- optimizer/scheduler/runtime precision settings;
- exact GraphCodeBERT revision;
- Python/Torch/dependency runtime versions;
- explicit negative/threshold/calibration/untouched limits.

Checkpoint/resume fail closed on the complete run-binding payload and digest,
restore optimizer/scheduler/RNG state, and do not expose model-only partial
resume.

For R4-D-013, the future binding must additionally and explicitly bind the
accepted guarded token lineage/selector, R4-D-013 decision/evidence, and
R4-D-011 graph-parent identity instead of the G7/v9 representation digest.

### 8. Durable Phase-8 checkpoint format is not directly consumable by live inference

The Phase-8 checkpoint payload stores `model_state_dict`,
`run_binding`, settings, optimizer/scheduler/RNG state, and selection records.

The current live `Predictor` expects the historical trainer format containing
`model` plus `config`, then separately loads threshold JSON.

Therefore a repaired checkpoint cannot be promoted by merely pointing
`SENTINEL_CHECKPOINT` at a Phase-8 `final.pt`. A later promotion/export seam
must explicitly translate or teach inference the Phase-8 checkpoint contract
and preserve lineage/model metadata.

### 9. Live inference preprocessing is still historical-control token selection

`ml/src/inference/preprocess.py::_tokenize_sliding_window()` explicitly uses
rounded linspace window subsampling. It does not implement the accepted guarded
selection seam.

The live service still runs the historical Run12 checkpoint, so this is not a
current production regression. It is, however, a mandatory future repaired
checkpoint compatibility requirement: training and inference window-selection
semantics cannot silently diverge.

### 10. Test/CI coverage is historical at the canonical runner boundary

The main Phase-8 compatibility tests exercise the G7/v9 dataset/runner and
generic checkpoint mechanics. Separate tests prove the old V10 adapter's
physical-acceptance + training-authorization stop lines, but there is no
end-to-end R4-D-013 dataset → V10 model → run-binding test yet.

That coverage belongs with the later M3 lineage integration after M2 fixes the
objective/evaluation decision.

## M0 disposition

**M0 COMPLETE.**

The executable training mechanics do not need to be redesigned during M0.
The major required later integration work is now explicit:

1. introduce an R4-D-013-aware logical-V3/V10/guarded dataset authority seam;
2. bind R4-D-013 token lineage/digest/selector and R4-D-011 graph parent;
3. construct the frozen model explicitly with graph schema V10;
4. route the canonical runner/micro-smoke to the accepted lineage only after
   M2 grants the relevant objective/evaluation/run authority;
5. add exact R4-D-013 integration tests;
6. define a future Phase-8 checkpoint → inference promotion/export contract;
7. reconcile future online preprocessing with the accepted token-selection
   semantics before a repaired checkpoint can become operational.

These are not authorization to launch or silently retrofit the current
historical runner now. The plan reserves final training-lineage integration for
M3 after M2.

## Stop line

Do not alter objective semantics, create negative targets, authorize training,
fit thresholds/calibration, or launch a long run during M0/M1. Preserve the
current historical runner for reproducibility until the versioned successor
integration is explicitly built under M3.
