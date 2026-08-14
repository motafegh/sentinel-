# R4 Plan Status Matrix

**Scope:** canonical R4 execution status on `main`. Phase 8 implementation has been adopted into `main`; the full retrain remains in progress and on a real-data launch hold until source, preprocessing, graph-target, token-coverage, and grouping repairs are versioned/re-frozen and G8 is closed.

| Phase | File | Status | Entry condition | Exit gate | Notes |
|---|---|---|---|---|---|
| 0 | `phases/01_PHASE_0_BASELINE_AND_EVIDENCE_LOCATION.md` | PASSED | Master plan adopted | G0 | Phase 0 complete; G0 PASS |
| 1 | `phases/02_PHASE_1_PREVIOUS_EVIDENCE_RECOVERY.md` | PASSED | G0 | G1 | Phase 1 complete; G1 PASS |
| 2 | `phases/03_PHASE_2_LABEL_CORRUPTION_RECONSTRUCTION.md` | PASSED | G1 | G2 | Phase 2 complete; G2 PASS — historical positive/zero origins reconstructed at category level |
| 3 | `phases/04_PHASE_3_EVIDENCE_LEDGER.md` | PASSED | G2 | G3 | Full 22,493-contract / 224,930-row ledger materialized and validated; G3 PASS |
| 4 | `phases/05_PHASE_4_TARGETED_GAP_ADJUDICATION.md` | PASSED | G3 | G4 | R4-GAP-002 resolved by checksum-bound 100-contract blind semantic review; four DIVE strata masked/excluded, TOD limited to TRAIN_WEAK; absent/provisional non-active sources explicitly masked/deferred; G4 PASS |
| 5 | `phases/06_PHASE_5_DATA_VNEXT_POLICY_AND_DESIGN.md` | PASSED | G4 | G5 | data-vnext-policy-v1 + contract-class schema + five accepted ADRs validated; eight classes enabled, GasException/UnusedReturn supervision disabled; no blanket negatives; G5 PASS |
| 6 | `phases/07_PHASE_6_PARTITIONS_AND_ACCEPTANCE_FREEZE.md` | PASSED | G5 | G6 | r4-vnext-roles-v1 covers 22,493 contracts/13,509 groups exactly once; 836 incomplete-representation contracts excluded; threshold/calibration/untouched acceptance frozen unsupported/empty; G6 PASS |
| 7 | `phases/08_PHASE_7_DATA_VNEXT_IMPLEMENTATION.md` | PASSED | G6 | G7 | sentinel-r4-vnext-v1 implemented and locally bound to 21,657 representations / 64,971 files with zero mismatches; G7 PASS |
| 8 | `phases/09_PHASE_8_EXISTING_MODEL_RETRAINING.md` | IN_PROGRESS | G7 | G8 | Runner/provenance and bounded GPU execution pass. Full launch is held after the completed audit found material source loss, post-compile normalization corruption, graph-target mismatch, token omission, and incomplete normalized grouping. A new repaired/re-frozen DATA lineage precedes the full retrain. |
| 9 | `phases/10_PHASE_9_EVALUATION_CALIBRATION_AND_POLICY.md` | WAITING | G8 | G9 | Independent roles |
| 10 | `phases/11_PHASE_10_ACCEPTANCE_PROMOTION_AND_ROLLBACK.md` | WAITING | G9 | G10 | Final decision |

## Canonical `main` baseline

Canonical `main` is now the active Phase-8 execution line:

| Phase | File | Status | Meaning |
|---:|---|---|---|
| 7 | `phases/08_PHASE_7_DATA_VNEXT_IMPLEMENTATION.md` | PASSED | G7 is canonical on `main`. |
| 8 | `phases/09_PHASE_8_EXISTING_MODEL_RETRAINING.md` | IN_PROGRESS | Phase-8 implementation is canonical on `main`; full launch is on DATA/representation hold and G8 remains open until a repaired lineage is re-frozen and the fixed-horizon retrain/evidence review complete. |

No Phase-8 implementation branch has higher authority than `main` after this adoption. Historical branch references remain useful only for provenance.

## Status vocabulary

- `READY`
- `IN_PROGRESS`
- `BLOCKED`
- `FAILED`
- `PASSED`
- `WAITING`
- `SUPERSEDED`
