# ADR-R4-015 — Bounded pilot pass with full-training hold

Date: 2026-10-17  
Status: ACCEPTED  
Decision ID: R4-D-015

## Context

R4-D-014 authorizes only a bounded positive-only control objective and a
separately fixed M4 pilot. It explicitly does not authorize the 100-epoch/full
training horizon because confirmed-negative evaluation evidence is absent and
positive-only diagnostics cannot establish discrimination or false-positive
behavior.

M4 is now complete:

- M4-A proved one real guarded optimizer step and positive-only selection on the
  exact accepted R4-D-009/R4-D-011/R4-D-013 seam;
- M4-B proved durable checkpointing, fail-closed binding validation,
  deterministic sampler reconstruction, exact next-epoch resume, model /
  optimizer / scheduler / RNG restoration, and bounded completion;
- M4-C completed the fixed eight-epoch dynamics horizon with 120/120 planned
  optimizer steps, finite losses, durable checkpoints and strongly improving
  positive-fit diagnostics.

The M4-C positive-fit trajectory reached mean positive probability ~0.9989 and
positive NLL ~0.00109 at epoch 8. This is evidence that the admitted positive
objective is readily optimized. It is not evidence about unknown/negative-cell
behavior.

R4-GAP-007 remains unresolved with zero confirmed-negative cells. Candidate #1
is NOT_CONFIRMED. Candidate #2 has primary support only and still awaits a
genuinely independent agreeing verifier.

## Decision

Accept M4 as a successful bounded mechanics/recovery/dynamics program.

The accepted M4-D outcome is:

`PASS_BOUNDED_PILOT_FULL_TRAINING_STILL_HELD`.

M4 evidence is sufficient to establish:

- the guarded V10/V3 training seam is executable under the frozen architecture;
- optimizer/scheduler accounting is coherent at the bounded horizon;
- BF16 CUDA execution is stable for the bounded pilot;
- checkpoint/resume and RNG restoration work under the guarded binding;
- the bounded positive-only objective strongly fits admitted positive cells;
- M4 does not exhibit a non-finite or obvious execution-instability failure.

M4 evidence is **not** sufficient to establish:

- discrimination;
- false-positive behavior or specificity;
- precision, AUROC or PR-AUC against confirmed negatives;
- threshold or calibration quality;
- untouched acceptance;
- production/inference/ZKML readiness;
- superiority over Run12 as an operational classifier;
- authority for the full Phase-8 horizon.

Therefore:

- M4 is COMPLETE;
- G8 remains OPEN;
- M5 execution remains NOT AUTHORIZED;
- full training remains NOT AUTHORIZED;
- M4-C checkpoints remain bounded research evidence only;
- R4-D-014 remains the objective/evaluation authority until another explicit
  decision changes it.

## Why the hold is required

The positive-only objective directly rewards high scores on admitted positive
cells and provides no accepted supervised penalty on unknown cells. The M4-C
near-saturated positive response is therefore compatible with both a useful
classifier and a degenerate high-score-on-everything classifier. Current
accepted evidence cannot distinguish those possibilities.

Proceeding to a much longer positive-only horizon would consume resources and
increase learned-state commitment without resolving the evidence question that
blocks interpretation. The next technically smallest action is therefore not
M5 execution; it is to improve the evaluation evidence boundary.

## Next boundary

Continue R4-GAP-007 from the committed hardened review queue, beginning with
genuinely independent verification of candidate #2 from its blind bundle.

If new confirmed-negative evidence is accepted, perform a fresh explicit review
of objective/evaluation/full-training authority. A small number of negatives may
improve qualitative observability but does not automatically justify stable FPR
estimates, threshold fitting, calibration fitting or M5 execution.

## Architecture limitation retained

The frozen four-eye architecture uses `fusion_max_nodes=2048`. Sparse GNN
processing remains full-graph, while dense cross-attention fusion is capped at
2,048 nodes. M4 observed a 2,759-node graph. This limitation remains explicit as
R4-R026 and is not resolved or hidden by this decision.

## Evidence

- M4-A report SHA-256:
  `3e6fee24e565b4647cb2ce0252d34f4d2fd3a3e890a448e25750194809cf4f6c`;
- M4-B report SHA-256:
  `2d218324a1c74dc1b6747dd77ce7c751789a213bafca24a255db11a48294ab38`;
- M4-C report SHA-256:
  `7293cf921e9cc5a09626a164f539cef1e800db860b47f10a5946d82be6d6409f`;
- M4-C run binding:
  `c3caec1feb1862b9db18df111b1a266a1c842daf77b6842b3cac08489f796731`;
- `runs/2026-10-17_PHASE8_m4_bounded_pilot_review_and_full_training_hold.md`;
- machine decision:
  `evidence/2026-10-17_phase8_m4_bounded_pilot/decision.json`;
- R4-GAP-007 in `EVIDENCE_GAP_REGISTER.md`;
- ADR-R4-014.

## Rollback

No operational promotion occurred. Rollback is simply to preserve the current
accepted DATA/representation authorities and treat all M4 checkpoints as bounded
research artifacts. Historical Run12 operational state remains untouched.
