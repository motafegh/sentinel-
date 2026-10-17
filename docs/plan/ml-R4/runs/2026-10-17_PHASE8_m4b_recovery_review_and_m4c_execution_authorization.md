# Phase-8 M4-B recovery review and M4-C execution authorization

Date: 2026-10-17  
Scope: bounded Phase-8 pilot only  
Authorities: R4-D-009, R4-D-011, R4-D-013, R4-D-014  
M4-B decision: PASS  
M4-C execution: AUTHORIZED  
M5/full training: NOT AUTHORIZED  
G8: OPEN

## M4-B reviewed evidence

The successful protected-local M4-B recovery proof is:

- report: `data_module/data/r4-m4b-guarded-recovery-2026-10-03-b.json`;
- report SHA-256: `2d218324a1c74dc1b6747dd77ce7c751789a213bafca24a255db11a48294ab38`;
- source commit: `ce6a2e5d85b6ae1c14080ff48d8c3c33c9c75b0d`;
- guarded run-binding digest: `07037c62e0279f81959599b764e4dd2b4044100c51f4c256af78a35c2227d5f1`.

The proof establishes the required M4-B properties:

1. Epoch 1 completed with exactly 15 optimizer steps and promoted a durable `latest.pt`.
2. A deliberately changed R4-D-013 guarded-lineage payload was rejected even while the claimed top-level digest string was held constant.
3. Resume restored model, optimizer, scheduler and RNG state successfully on CUDA after the RNG-device normalization repair.
4. Resume continued at exact epoch 2 and finished at exactly 30 total optimizer steps.
5. Deterministic epoch-2 group-sampler reconstruction repeated exactly for all 932 active train groups; sequence SHA-256 `b7e47b26a9b1ac3a19be4b3601043fdfe31af37f7e1af6061e1f62d0fa7046ce`.
6. Repository-safe tests prove simulated checkpoint promotion failure leaves neither promoted nor temporary checkpoint residue.
7. Historical G7/v9 runner and Run12 state remain separate and untouched.

Bound checkpoint identities:

- epoch-1 pause latest: `56a0ea198fce459c7b0a613f04bceb9c7b5cb5ef4da458400fc7c584d10089e5`;
- epoch-1 milestone: `dfa2c9dbffd744c5fdd287cc14a69948373e10a36e796c00430c1b514d0d9e69`;
- final epoch-2: `4f115c565e27dc86d6f446a12e991be20b0650608f37c2ea08f76bbff9262086`;
- latest after resume: `71a13988383e58e22ac366264167bcde97234b1ef840a7b094e24c1043dfad61`;
- best positive-NLL: `756fd2f6401721fc0347bc6fa85df70fe6035a704dd136b62644cab452c957a1`.

Focused repository verification passed 17 tests. The fixed M4-C driver also compiles and the Phase-8 compatibility workflow run `39048606424` completed successfully, including compile and compatibility tests.

## M4-C execution boundary

M4-C is now authorized under the already accepted R4-D-014 bounded-pilot authority.

The horizon is fixed to exactly 8 epochs with:

- batch size 8;
- gradient accumulation 8;
- 117 train loader batches per epoch;
- 15 optimizer steps per epoch;
- 120 optimizer steps total;
- BF16 CUDA autocast;
- exact R4-D-009 logical V3 + R4-D-011 V10 graph parent + R4-D-013 guarded-token physical lineage + R4-D-014 positive-only objective/evaluation contract;
- no Run12 learned-state loading.

M4-C must persist per-epoch finite losses, learning rates, positive-only MODEL_SELECTION diagnostics, run/checkpoint identities, runtime and CUDA memory telemetry, and final/best-positive-NLL checkpoint hashes.

The known `fusion_max_nodes=2048` architectural cap remains an explicit interpretation limitation. Sparse GNN processing remains full-graph, while dense fusion is bounded to at most 2,048 nodes. M4-C cannot be interpreted as proof of full-node fusion coverage.

## Interpretation and authority limits

Even a clean M4-C PASS establishes only bounded optimization/recovery/dynamics evidence.

It does not establish:

- discrimination or false-positive control;
- precision, specificity, AUROC or PR-AUC;
- threshold quality;
- calibration quality;
- untouched acceptance performance;
- production/inference readiness;
- ZKML promotion readiness;
- full-horizon training authority.

After M4-C, M4-D must explicitly review the evidence. The allowed next decisions remain:

- `PASS_BOUNDED_PILOT_FULL_TRAINING_STILL_HELD`;
- `REVISE_M4_AND_REPEAT`;
- `STOP_ML_EXECUTION`.

M5/full training remains unauthorized until a separate later governance decision explicitly grants it.

Machine-readable review: `evidence/2026-10-17_m4b_recovery_review/review.json`.
