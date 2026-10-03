# ML M4 — guarded bounded pilot

**Date:** 2026-10-03  
**Authorities:** R4-D-009 logical V3; R4-D-011 graph parent; R4-D-013 guarded-token physical acceptance; R4-D-014 bounded positive-only objective/evaluation boundary  
**Entry:** M3 protected-local no-step preflight reviewed PASS  
**State:** IN PROGRESS — M4-A reviewed PASS; M4-B checkpoint/resume/recovery proof next

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

## M4-A reviewed closure — 2026-10-03

The repaired protected-local rerun from source commit
`cbc50e913458ce4df48888cb838d1655e53a47b8` passed with report status
`PASS_M4A_GUARDED_CUDA_SMOKE_REVIEW_REQUIRED`.

Bound local report:

- path:
  `data_module/data/r4-m4a-guarded-smoke-2026-10-03-b.json`;
- SHA-256:
  `3e6fee24e565b4647cb2ce0252d34f4d2fd3a3e890a448e25750194809cf4f6c`;
- tracked worktree at evidence binding: clean.

Observed mechanics:

- exactly 8 train micro-batches;
- exactly 1 optimizer step;
- scheduler advanced 0 → 1;
- BF16 CUDA execution on NVIDIA GeForce RTX 3070 Laptop GPU;
- finite total/main/aux/phase2 losses;
- one MODEL_SELECTION batch with 8 positive metric cells;
- no historical checkpoint loaded;
- no durable checkpoint written;
- full-training, M4-B and M4-C authority flags remained false;
- peak CUDA allocation approximately 5.57 GiB.

The first attempt's `contract_names` PyG batching failure was repaired at the
ML collate boundary without changing protected graph/token bytes. The successful
rerun exercises that repaired source.

### New bounded-pilot observation: fusion node truncation

The successful smoke encountered a 2,759-node graph while frozen
`fusion_max_nodes=2048`. The GNN still consumes the full sparse graph, but the
dense cross-attention fusion projection intentionally drops nodes above the
2,048-node cap.

This does not invalidate M4-A mechanics because the cap is part of the frozen
`four_eye_v8/v8.1` architecture, but it is now explicit M4 evidence and must
remain visible during M4-C interpretation. It is not evidence of full-graph
fusion coverage or model quality.

**M4-A decision:** PASS.

## Current stop line

Proceed to M4-B repository-safe checkpoint/resume/recovery proof, then its
protected-local bounded recovery execution. Do not execute M4-C yet. Do not add
a generic full-training switch. Do not weaken
`full_training_authorized=false`.


## M4-A first protected-local execution — failed before optimization

The first protected-local M4-A execution from source commit
`5ce5adc4ec77046af18812199d8ad8b9dbf865f9` reached the real guarded
TRAIN population and V10 model construction, then failed while PyG collated the
first training batch:

`KeyError: 'contract_names'`.

No optimizer step completed and no durable checkpoint/report was promoted.

Root cause:

- accepted V10 file-union graphs intentionally carry
  `graph.contract_names`;
- accepted single-contract graphs intentionally do not;
- the historical shared collate exclusion list already excluded
  `contract_name` and `node_metadata`, but not the newer V10 provenance
  fields;
- PyG therefore treated `contract_names` as batchable graph data and required
  it on every graph in a mixed batch;
- the model does not consume `contract_names` or the other V10
  extraction/provenance metadata.

Repair:

- extend the collate provenance exclusion contract to include V10-only
  non-model metadata, including `contract_names`,
  `graph_schema_version`, extractor identity and call-audit metadata;
- preserve structural/model tensors `x`, `edge_index`, `edge_attr` and
  ordinary PyG batch assignment unchanged;
- add a regression test that batches one file-union V10 graph with one
  single-contract V10 graph.

This is an ML batching-boundary repair only. R4-D-011/R4-D-013 protected-local
artifacts remain immutable and byte-unchanged. M4-A remains unpassed until the
repaired source is repository-verified and the protected-local smoke is rerun.
