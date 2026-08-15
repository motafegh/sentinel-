# R4 Plan Status Matrix

**Scope:** canonical R4 execution status on `main`. Historical G0–G7 evidence remains valid for the immutable `sentinel-r4-vnext-v1` lineage. Phase 8 is still `IN_PROGRESS`. Repaired-v2 physical source/representation evidence remains accepted under R4-D-008, but the 2026-08-15 full-population grouping audit demonstrated that `r4-leakage-groups-v2` over-connects unrelated contracts through arbitrary same-source address literals. R4-D-009 therefore makes the corrected logical V3 lineage the active candidate for future role/evaluation/training work. Repository tooling is implemented; local V3 generation, acceptance, and regenerated research evidence are pending. The 100-epoch retrain is not authorized.

| Phase | File | Status | Entry condition | Exit gate | Notes |
|---|---|---|---|---|---|
| 0 | `phases/01_PHASE_0_BASELINE_AND_EVIDENCE_LOCATION.md` | PASSED | Master plan adopted | G0 | Phase 0 complete; G0 PASS |
| 1 | `phases/02_PHASE_1_PREVIOUS_EVIDENCE_RECOVERY.md` | PASSED | G0 | G1 | Phase 1 complete; G1 PASS |
| 2 | `phases/03_PHASE_2_LABEL_CORRUPTION_RECONSTRUCTION.md` | PASSED | G1 | G2 | Phase 2 complete; G2 PASS — historical positive/zero origins reconstructed at category level |
| 3 | `phases/04_PHASE_3_EVIDENCE_LEDGER.md` | PASSED | G2 | G3 | Full 22,493-contract / 224,930-row historical ledger materialized and validated; G3 PASS |
| 4 | `phases/05_PHASE_4_TARGETED_GAP_ADJUDICATION.md` | PASSED | G3 | G4 | R4-GAP-002 resolved by checksum-bound 100-contract blind semantic review; four DIVE strata masked/excluded, TOD limited to TRAIN_WEAK; absent/provisional non-active sources explicitly masked/deferred; G4 PASS |
| 5 | `phases/06_PHASE_5_DATA_VNEXT_POLICY_AND_DESIGN.md` | PASSED | G4 | G5 | data-vnext-policy-v1 + contract-class schema + accepted ADRs validated; eight classes enabled, GasException/UnusedReturn supervision disabled; no blanket negatives; G5 PASS |
| 6 | `phases/07_PHASE_6_PARTITIONS_AND_ACCEPTANCE_FREEZE.md` | PASSED | G5 | G6 | Historical `r4-vnext-roles-v1` covers 22,493 contracts/13,509 groups exactly once; threshold/calibration/untouched acceptance frozen unsupported/empty; G6 PASS. Later V2/V3 Phase-8 role lineages do not overwrite this artifact. |
| 7 | `phases/08_PHASE_7_DATA_VNEXT_IMPLEMENTATION.md` | PASSED | G6 | G7 | Historical `sentinel-r4-vnext-v1` is bound to 21,657 representations / 64,971 files with zero mismatches; G7 PASS remains valid evidence for that immutable lineage. Later Phase-8 audit findings prevent using it for the full retrain. |
| 8 | `phases/09_PHASE_8_EXISTING_MODEL_RETRAINING.md` | IN_PROGRESS | G7 | G8 | R4-D-008 accepts repaired-v2 physical DATA (22,540 contracts / 67,620 files; binding digest `16dd4a3f...`) but R4-D-009 supersedes V2 grouping/roles for future research after a 10,327-contract address-connected group demonstrated over-grouping. V3 logical grouping/partition/publication/binding and evidence-regeneration tooling are implemented repository-side. Local V3 acceptance is pending. G8 remains open; no confirmed-negative evaluation population exists and selector promotion remains pending. Full 100-epoch training is not authorized. |
| 9 | `phases/10_PHASE_9_EVALUATION_CALIBRATION_AND_POLICY.md` | WAITING | G8 | G9 | Independent roles; current threshold/calibration support remains unavailable. |
| 10 | `phases/11_PHASE_10_ACCEPTANCE_PROMOTION_AND_ROLLBACK.md` | WAITING | G9 | G10 | Final decision; untouched acceptance remains unsupported/empty/frozen. |

## Canonical `main` Phase-8 boundary

`main` is the only active repository execution line. The Phase-8 state is deliberately split into evidence layers:

| Layer | State | Meaning |
|---|---|---|
| Historical v1 / G7 | PASSED / immutable | `sentinel-r4-vnext-v1`, `r4-vnext-roles-v1`, graph schema v9 and binding digest `7637461f6643d398c7a0446412fedd8877914c7b9ed41309dab45f18ed96f420` remain reproducible historical evidence. |
| Repaired-v2 physical source/representations | ACCEPTED / immutable evidence | R4-D-008 accepts 22,540 repaired contracts and 67,620 graph/token/sidecar files with zero missing/invalid; physical binding digest `16dd4a3f98c34e52e5c411b39268361881efede07e8f3f52d0c060dd1c5bb6dd`. This physical evidence remains reusable. |
| Repaired-v2 grouping/roles | SUPERSEDED FOR FUTURE RESEARCH | V2 grouping used arbitrary same-source address literals as union authority. Full-population audit found a 10,327-contract DIVE component driven by common addresses. V2 grouping/roles and V2 role-dependent research outputs remain historical evidence but are not future training/evaluation authority. |
| Logical V3 repository implementation | COMPLETE / LOCAL ACCEPTANCE PENDING | R4-D-009 introduces `r4-leakage-groups-v3`, `r4-vnext-roles-v3`, `sentinel-r4-vnext-v3`; address literals are diagnostic-only. V3 reuses repaired-v2 physical preprocessing/representations and role-independent evidence semantics. Local grouping → publication → same-byte binding → acceptance must pass before V3 is accepted. |
| Phase-8 research regeneration | WAITING ON LOCAL V3 | The confirmed-negative queue, selector population comparison, representation-sensitivity role sets, and CUDA selector comparison must be regenerated under V3. The V3 CUDA launcher requires worst-case probes instead of silently skipping them. |
| Full training / G8 | HOLD | Confirmed-negative evaluation evidence, threshold/calibration, and untouched acceptance remain absent; selector not promoted; no full-run horizon is authorized. |

## Version boundaries

Historical/physical roots that remain immutable:

- historical G7 publication: `sentinel-r4-vnext-v1`;
- repaired preprocessing: `sentinel-preprocessed-r4-v2`;
- repaired provenance/source claims: `r4-provenance-v1`;
- role-independent repaired evidence ledger: `evidence-ledger-r4-v2`;
- repaired representations: `representations-r4-v2` / extractor `v2.2-r4-repaired`;
- graph schema: `v9`;
- token tensor contract: `[4,512]`.

Historical/superseded logical V2 identifiers:

- grouping: `r4-leakage-groups-v2`;
- role partition: `r4-vnext-roles-v2`;
- publication: `sentinel-r4-vnext-v2`.

Active logical V3 candidate identifiers:

- grouping: `r4-leakage-groups-v3`;
- role partition: `r4-vnext-roles-v3`;
- DATA publication: `sentinel-r4-vnext-v3`;
- logical build: `r4-logical-lineage-v3`.

## Current restart boundary

From a clean/synchronized `main`, execute:

`runs/2026-08-15_PHASE8_logical_v3_grouping_repair_handoff.md`.

The required order is corrected grouping, V3 role/publication freeze, same-byte physical representation rebinding, V2→V3 acceptance audit, then regeneration of representation sensitivity, selector population evidence, V3 confirmed-negative pilot queue, and identical-initialization CUDA comparison with mandatory worst-case probes.

Do **not** manually adjudicate the old V2 confirmed-negative queue. Do **not** promote the selector or design/freeze a PU objective before V3 evidence is reviewed. Do **not** launch the 100-epoch job.

The completed repaired-v2 rebuild and R4-D-008 remain the physical-data reproducibility root. R4-D-009 / `adrs/ADR-R4-009-logical-v3-leakage-grouping-correction.md` governs the current logical correction.

## Status vocabulary

- `READY`
- `IN_PROGRESS`
- `BLOCKED`
- `FAILED`
- `PASSED`
- `WAITING`
- `SUPERSEDED`
