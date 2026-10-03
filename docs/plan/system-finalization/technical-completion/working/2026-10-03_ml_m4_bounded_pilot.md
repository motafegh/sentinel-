# ML M4 — guarded bounded pilot

**Date:** 2026-10-03  
**Authorities:** R4-D-009 logical V3; R4-D-011 graph parent; R4-D-013 guarded-token physical acceptance; R4-D-014 bounded positive-only objective/evaluation boundary  
**Entry:** M3 protected-local no-step preflight reviewed PASS  
**State:** IN PROGRESS — contract fixed; M4-A mechanics smoke next

## Purpose

M4 proves that the accepted guarded Phase-8 seam can execute optimization,
selection diagnostics, checkpoint/recovery and bounded early dynamics safely.

M4 does **not** establish classifier quality and does **not** authorize the
100-epoch/full-horizon run.

## Frozen authority

The M4 seam must bind exactly:

- logical publication: R4-D-009 / `sentinel-r4-vnext-v3`;
- graph parent: R4-D-011 / digest
  `d9f925588913e66476cfbc097bace7daa7e673295fe2a243760313d0bef5ebdd`;
- guarded tokens: R4-D-013 / digest
  `9885d7b88a46aff4102741d63eeaa0bd6968f0857f7bca6a389662bfaa158881`;
- objective/evaluation: R4-D-014 /
  `masked_positive_bce_control_v1`;
- architecture: `four_eye_v8` / `v8.1`, graph schema V10;
- token tensor: `[4,512]`;
- GraphCodeBERT revision:
  `2b0488a7bb0eefc7041f1bb2cad1ab26b0da269d`;
- seed: `20260813`;
- weak-positive weight: `0.25`;
- batch size: `8`;
- gradient accumulation: `8`;
- BF16 CUDA autocast.

No Run12 learned state may be loaded.

## M4-A — one-optimizer-step protected-local CUDA smoke

Execute on the real accepted populations:

- TRAIN_STRONG + TRAIN_WEAK guarded dataset;
- MODEL_SELECTION guarded dataset;
- deterministic group sampler;
- exactly 8 train micro-batches = exactly 1 optimizer step;
- exactly 1 model-selection batch;
- BF16 autocast;
- finite total/main/aux/phase2 losses;
- finite gradients and clipped gradient norm through the existing epoch helper;
- scheduler advances exactly once;
- selection emits only positive-fit diagnostics;
- measure CUDA allocated/reserved peak;
- bind the same D9/D11/D13/D14 guarded run contract;
- write a fresh protected-local report;
- write **no durable training checkpoint** and grant no later authority by itself.

**M4-A exit:** explicit review PASS or FAIL. The 8-epoch pilot remains blocked
until M4-A passes.

## M4-B — checkpoint/resume/recovery proof

After M4-A PASS, prove the durable guarded runner can:

1. save a self-describing checkpoint with the exact guarded run binding;
2. reject a changed binding/config/lineage;
3. restore model, optimizer, scheduler and all RNG state;
4. resume at the exact next epoch/step;
5. preserve deterministic sampler epoch behavior;
6. leave no partial checkpoint promoted after a failed write;
7. preserve historical G7/v9 runner and Run12 state untouched.

Repository-safe synthetic/focused tests come first. Protected-local CUDA
recovery execution follows only after those tests pass.

## M4-C — fixed bounded dynamics pilot

Only after M4-A and M4-B pass.

Pilot horizon is fixed now at **8 epochs**.

Rationale:

- 8 is materially below the 100-epoch full horizon;
- it reaches the existing `aux_loss_warmup_epochs=8` boundary;
- it yields 8 independent positive-fit selection observations;
- it is sufficient to expose obvious non-finite behavior, runaway/saturation
  trends, scheduler/step-accounting errors and early optimization instability;
- it does not cross the 15-epoch GNN-prefix warmup and therefore cannot be
  misrepresented as a full training-quality study.

Using the M3-measured full population and unchanged batch/accumulation contract,
planning arithmetic is:

- 117 train loader batches per epoch;
- 15 optimizer steps per epoch;
- 120 planned optimizer steps across 8 epochs.

The pilot must persist:

- complete run binding;
- per-epoch losses and learning rates;
- positive-only MODEL_SELECTION diagnostics;
- checkpoint/index identities;
- runtime and GPU-memory telemetry;
- resume/failure evidence;
- final/best-positive-NLL checkpoint hashes.

Interpretation boundary:

- `best_positive_nll` means best positive fit **inside this pilot only**;
- no discrimination/FPR/specificity/precision/AUROC/PR-AUC claim;
- no threshold/calibration fitting;
- no untouched acceptance;
- no production/inference/ZKML promotion.

## M4-D — explicit launch/no-launch review

After M4-C, write an explicit decision record.

Allowed outcomes:

- `PASS_BOUNDED_PILOT_FULL_TRAINING_STILL_HELD`;
- `REVISE_M4_AND_REPEAT`;
- `STOP_ML_EXECUTION`.

A passing M4 does **not** automatically authorize M5. Any full-horizon launch
requires a separate explicit governance decision after reviewing M4 evidence
and all still-open blockers, including R4-GAP-007.

## M4-A implementation checkpoint

Implemented on canonical `main`:

- `docs/plan/ml-R4/scripts/p8_smoke_guarded_training_m4a.py`;
- exact real guarded populations and D9/D11/D13/D14 run binding;
- exactly 8 train micro-batches under accumulation=8;
- exactly one optimizer/scheduler step;
- exactly one MODEL_SELECTION batch;
- BF16 CUDA autocast required;
- finite loss/gradient checks inherited from `train_masked_epoch`;
- CUDA peak allocation/reservation telemetry;
- no checkpoint load and no durable checkpoint write;
- output status `PASS_M4A_GUARDED_CUDA_SMOKE_REVIEW_REQUIRED`;
- all later execution flags remain false pending review.

The Phase-8 compatibility workflow compiles the M4-A driver. Protected-local
execution remains the next action after repository-safe verification.

## Current stop line

Implement and repository-test M4-A only. Do not execute the 8-epoch pilot yet.
Do not add a generic full-training switch. Do not weaken
`full_training_authorized=false`.
