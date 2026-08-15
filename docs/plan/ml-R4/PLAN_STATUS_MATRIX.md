# R4 Plan Status Matrix

**Scope:** canonical R4 execution status on `main`. Historical G0–G7 evidence remains valid for the immutable `sentinel-r4-vnext-v1` lineage. Phase 8 is still `IN_PROGRESS`: the local gate re-audit corrections are implemented, but the new physical repaired-v2 corpus/publication has not yet been rebuilt, accepted, or smoke-tested on the local GPU. The 100-epoch retrain is not authorized.

| Phase | File | Status | Entry condition | Exit gate | Notes |
|---|---|---|---|---|---|
| 0 | `phases/01_PHASE_0_BASELINE_AND_EVIDENCE_LOCATION.md` | PASSED | Master plan adopted | G0 | Phase 0 complete; G0 PASS |
| 1 | `phases/02_PHASE_1_PREVIOUS_EVIDENCE_RECOVERY.md` | PASSED | G0 | G1 | Phase 1 complete; G1 PASS |
| 2 | `phases/03_PHASE_2_LABEL_CORRUPTION_RECONSTRUCTION.md` | PASSED | G1 | G2 | Phase 2 complete; G2 PASS — historical positive/zero origins reconstructed at category level |
| 3 | `phases/04_PHASE_3_EVIDENCE_LEDGER.md` | PASSED | G2 | G3 | Full 22,493-contract / 224,930-row historical ledger materialized and validated; G3 PASS |
| 4 | `phases/05_PHASE_4_TARGETED_GAP_ADJUDICATION.md` | PASSED | G3 | G4 | R4-GAP-002 resolved by checksum-bound 100-contract blind semantic review; four DIVE strata masked/excluded, TOD limited to TRAIN_WEAK; absent/provisional non-active sources explicitly masked/deferred; G4 PASS |
| 5 | `phases/06_PHASE_5_DATA_VNEXT_POLICY_AND_DESIGN.md` | PASSED | G4 | G5 | data-vnext-policy-v1 + contract-class schema + five accepted ADRs validated; eight classes enabled, GasException/UnusedReturn supervision disabled; no blanket negatives; G5 PASS |
| 6 | `phases/07_PHASE_6_PARTITIONS_AND_ACCEPTANCE_FREEZE.md` | PASSED | G5 | G6 | Historical `r4-vnext-roles-v1` covers 22,493 contracts/13,509 groups exactly once; threshold/calibration/untouched acceptance frozen unsupported/empty; G6 PASS. Repaired-v2 roles are a separate local rebuild lineage and do not overwrite this artifact. |
| 7 | `phases/08_PHASE_7_DATA_VNEXT_IMPLEMENTATION.md` | PASSED | G6 | G7 | Historical `sentinel-r4-vnext-v1` is bound to 21,657 representations / 64,971 files with zero mismatches; G7 PASS remains valid evidence for that immutable lineage. Later Phase-8 audit findings prevent using it for the full retrain. |
| 8 | `phases/09_PHASE_8_EXISTING_MODEL_RETRAINING.md` | IN_PROGRESS | G7 | G8 | Repository repair plus local gate corrections are implemented: full-manifest raw/completeness gates, evidence-preserving file-level graph union, token coverage, repaired grouping/source claims, ledger-bound publication, physical payload validation, role coverage, exact-state acceptance, dynamic ML adapter, and bounded GPU-smoke seam. Physical `sentinel-r4-vnext-v2` rebuild/acceptance and bounded repaired-data GPU smoke remain local-only prerequisites. Full 100-epoch training is not authorized. |
| 9 | `phases/10_PHASE_9_EVALUATION_CALIBRATION_AND_POLICY.md` | WAITING | G8 | G9 | Independent roles; current threshold/calibration support remains unavailable. |
| 10 | `phases/11_PHASE_10_ACCEPTANCE_PROMOTION_AND_ROLLBACK.md` | WAITING | G9 | G10 | Final decision; untouched acceptance remains unsupported/empty/frozen. |

## Canonical `main` Phase-8 boundary

`main` is the only active repository execution line. The Phase-8 state is deliberately split into three evidence layers:

| Layer | State | Meaning |
|---|---|---|
| Historical v1 / G7 | PASSED / immutable | `sentinel-r4-vnext-v1`, `r4-vnext-roles-v1`, graph schema v9 and binding digest `7637461f6643d398c7a0446412fedd8877914c7b9ed41309dab45f18ed96f420` remain reproducible historical evidence. |
| Repository repaired-v2 implementation | COMPLETE | New versioned source/tests/CI and local rebuild/acceptance interfaces exist on `main`; historical artifacts were not overwritten. |
| Physical repaired-v2 evidence | PENDING LOCAL | Git-ignored raw corpus, historical solc binaries, repaired preprocessed/representation trees, generated parquet/publication artifacts, token-coverage experiment and local GPU are required. No repaired-v2 population counts/binding digest are claimed yet. |

Required repaired lineage identifiers are:

- preprocessing: `sentinel-preprocessed-r4-v2`;
- provenance: `r4-provenance-v1`;
- evidence ledger: `evidence-ledger-r4-v2`;
- grouping: `r4-leakage-groups-v2`;
- role partition: `r4-vnext-roles-v2`;
- DATA publication: `sentinel-r4-vnext-v2`;
- representation extractor: `v2.2-r4-repaired`;
- graph schema remains `v9`;
- token tensor contract remains `[4, 512]`.

The exact restart/execution contract is the latest durable Phase-8 local-data rebuild handoff under `runs/`, amended by `2026-08-15_PHASE8_local_gate_reaudit_and_corrections.md`. No old pretraining handoff may be used to launch the 100-epoch job while this local repair/acceptance boundary remains open.

## Status vocabulary

- `READY`
- `IN_PROGRESS`
- `BLOCKED`
- `FAILED`
- `PASSED`
- `WAITING`
- `SUPERSEDED`
