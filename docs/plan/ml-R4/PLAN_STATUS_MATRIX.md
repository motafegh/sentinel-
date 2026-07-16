# R4 Plan Status Matrix

| Phase | File | Status | Entry condition | Exit gate | Notes |
|---|---|---|---|---|---|
| 0 | `phases/01_PHASE_0_BASELINE_AND_EVIDENCE_LOCATION.md` | PASSED | Master plan adopted | G0 | Phase 0 complete; G0 PASS |
| 1 | `phases/02_PHASE_1_PREVIOUS_EVIDENCE_RECOVERY.md` | WAITING | G0 | G1 | Reuse prior work |
| 2 | `phases/03_PHASE_2_LABEL_CORRUPTION_RECONSTRUCTION.md` | WAITING | G1 | G2 | No new broad audit |
| 3 | `phases/04_PHASE_3_EVIDENCE_LEDGER.md` | WAITING | G2 | G3 | Sidecar first |
| 4 | `phases/05_PHASE_4_TARGETED_GAP_ADJUDICATION.md` | WAITING | G3 | G4 | Gap ID mandatory |
| 5 | `phases/06_PHASE_5_DATA_VNEXT_POLICY_AND_DESIGN.md` | WAITING | G4 | G5 | ADRs required |
| 6 | `phases/07_PHASE_6_PARTITIONS_AND_ACCEPTANCE_FREEZE.md` | WAITING | G5 | G6 | Freeze acceptance |
| 7 | `phases/08_PHASE_7_DATA_VNEXT_IMPLEMENTATION.md` | WAITING | G6 | G7 | Versioned artifacts |
| 8 | `phases/09_PHASE_8_EXISTING_MODEL_RETRAINING.md` | WAITING | G7 | G8 | Architecture frozen |
| 9 | `phases/10_PHASE_9_EVALUATION_CALIBRATION_AND_POLICY.md` | WAITING | G8 | G9 | Independent roles |
| 10 | `phases/11_PHASE_10_ACCEPTANCE_PROMOTION_AND_ROLLBACK.md` | WAITING | G9 | G10 | Final decision |

## Status vocabulary

- `READY`
- `IN_PROGRESS`
- `BLOCKED`
- `FAILED`
- `PASSED`
- `WAITING`
- `SUPERSEDED`
