# Phase-8 M4 bounded-pilot review and full-training hold

Date: 2026-10-17  
Decision ID: `R4-D-015`  
Decision: `PASS_BOUNDED_PILOT_FULL_TRAINING_STILL_HELD`  
G8: OPEN  
M5/full training: NOT AUTHORIZED

## Outcome

M4 passes as a bounded mechanics/recovery/dynamics program under the exact
R4-D-009 / R4-D-011 / R4-D-013 / R4-D-014 seam.

The bounded pilot establishes that the accepted guarded Phase-8 training seam
can execute optimization, positive-only model-selection diagnostics, durable
checkpointing/recovery and an eight-epoch dynamics horizon without non-finite
or obvious runtime-instability failure.

It does **not** establish classifier quality, discrimination, false-positive
behavior, threshold quality, calibration, untouched acceptance, production
readiness or full-training authority.

Full training therefore remains held.

## M4-A

Reviewed PASS.

- protected-local report SHA-256:
  `3e6fee24e565b4647cb2ce0252d34f4d2fd3a3e890a448e25750194809cf4f6c`;
- exactly 8 micro-batches → 1 optimizer step;
- one positive-only MODEL_SELECTION batch;
- finite losses;
- no historical checkpoint;
- no durable checkpoint;
- no later-stage authority granted.

The initial mixed-V10 PyG `contract_names` failure was repaired at the collate
boundary without mutating accepted R4-D-011/R4-D-013 artifacts.

## M4-B

Reviewed PASS.

- protected-local report SHA-256:
  `2d218324a1c74dc1b6747dd77ce7c751789a213bafca24a255db11a48294ab38`;
- run binding:
  `07037c62e0279f81959599b764e4dd2b4044100c51f4c256af78a35c2227d5f1`;
- epoch 1 completed at 15 optimizer steps and durably checkpointed;
- changed guarded-lineage payload was rejected fail closed;
- epoch-2 deterministic sampler reconstruction repeated identically;
- exact `latest.pt` resume completed epoch 2 at 30 total optimizer steps;
- model / optimizer / scheduler / RNG restoration completed;
- final/latest/best-positive-NLL checkpoint identities were durably indexed;
- simulated atomic-promotion failure is repository-tested to leave no partial
  promoted checkpoint.

The first protected-local M4-B attempt exposed a CUDA/CPU RNG-state restoration
defect. The central checkpoint layer now normalizes serialized RNG tensors to
CPU uint8 before PyTorch RNG restoration, with strict type/dtype validation.
The successful `-b` run proves the repaired CUDA resume path.

## M4-C

Reviewed PASS as bounded positive-fit dynamics evidence.

Protected-local report:

`data_module/data/r4-m4c-guarded-pilot-2026-10-17-a.json`

SHA-256:

`7293cf921e9cc5a09626a164f539cef1e800db860b47f10a5946d82be6d6409f`

Source commit:

`a39ef444c9a0204bdccf0ef978a91fa6a16f8075`

Run binding:

`c3caec1feb1862b9db18df111b1a266a1c842daf77b6842b3cac08489f796731`

Execution:

- 8 epochs;
- 15 optimizer steps per epoch;
- 120 / 120 planned optimizer steps executed;
- BF16 CUDA;
- peak allocated CUDA memory ~6209.51 MiB;
- runtime ~568.90 s;
- no historical checkpoint loaded;
- no threshold/calibration fitting;
- no untouched-acceptance use.

Positive-fit trajectory:

| Metric | Epoch 1 | Epoch 8 | Interpretation |
|---|---:|---:|---|
| positive NLL | 0.3874338865 | 0.0010884432 | strong positive-fit improvement only |
| mean positive probability | 0.6831893921 | 0.9989205003 | near-saturated positive response |
| positive recall @ 0.5 | 0.9859154820 | 1.0 | positive diagnostic only |
| main loss | 0.5605032584 | 0.0010866913 | optimizer fit to admitted positive objective |
| total loss | 0.7761455858 | 0.0526033528 | finite/stable bounded optimization |

Best positive-NLL checkpoint is epoch 8. This is the best **positive-fit**
checkpoint inside M4-C only; it is not the best classifier or a promotion
winner.

Checkpoint SHA-256 identities:

- best positive NLL:
  `f1d7a6d43f4a5f43a83abf343a372efc553e7d08c6b865d31e88d26adadeb854`;
- final:
  `2dbc9c765f7fce28bc2b2a4d756fd602a7396712d349aae0ccf5650df7bc9ba6`;
- latest:
  `7363104706f6a35ab53c2a2a3aa5d5c055199df3eda13a57e7f8d0f59c1b3a5e`;
- epoch-4 milestone:
  `3f71b0ae6c2ff14062b85afd017140381eaa30b928d7ef64b9c06885f413ce8b`.

## Architecture observation retained

Frozen `fusion_max_nodes=2048` remains an explicit limitation. M4-A observed a
2,759-node graph. Sparse GNN processing remains full-graph, but dense
cross-attention fusion is capped at 2,048 nodes. M4 does not establish full-node
fusion coverage.

This is carried as R4-R026 and does not invalidate the bounded pilot.

## Why full training remains held

R4-GAP-007 remains unresolved:

- confirmed-negative cells: 0;
- candidate #1: `NOT_CONFIRMED`;
- candidate #2: primary support only; genuinely independent verification still
  pending;
- accepted negative evidence remains evaluation-only unless a later decision
  grants optimizer authority.

R4-D-014 therefore remains controlling: positive-only masked BCE can establish
positive fit but cannot establish discrimination or false-positive behavior.
The near-saturated positive response observed in M4-C makes that distinction
more important, not less.

The following remain unsupported:

- specificity / false-positive rate;
- precision / AUROC / PR-AUC against confirmed negatives;
- threshold fitting;
- calibration fitting;
- untouched acceptance;
- production/inference/ZKML promotion.

Therefore M4 passes, but M5 is not authorized.

## Decision

`PASS_BOUNDED_PILOT_FULL_TRAINING_STILL_HELD`

This means:

- M4 is complete;
- bounded execution mechanics/recovery/dynamics are accepted evidence;
- G8 remains open;
- no full-horizon training run may start;
- M4-C checkpoints remain research/bounded-pilot evidence only;
- no checkpoint is promoted to inference or downstream ZKML;
- R4-D-014 positive-only objective remains bounded-control authority only.

## Next boundary

Return to R4-GAP-007 evidence work, beginning with the already-prepared blind
independent verification of candidate #2 from the committed hardened queue.

If new confirmed-negative evidence is accepted, reassess the evaluation and
full-training design under a new explicit decision. One or a few confirmed
negatives do not automatically establish stable FPR estimates, threshold or
calibration support, or M5 authority.
