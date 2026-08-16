# R4 Plan Status Matrix

**Scope:** canonical R4 execution status on `main`. Historical G0–G7 evidence remains valid for the immutable `sentinel-r4-vnext-v1` lineage. Phase 8 is still `IN_PROGRESS`. Repaired-v2 physical source/representation evidence remains accepted under R4-D-008. The corrected logical V3 lineage has now passed protected local grouping/publication/binding/acceptance and is accepted under R4-D-009 for future Phase-8 role/evaluation/model research. V3 representation sensitivity, selector population comparison, confirmed-negative pilot-queue generation, and identical-initialization CUDA selector comparison with mandatory worst-case probes have also completed. The guarded selector is evidence-ready for a separate promotion decision but is not yet promoted. Confirmed-negative adjudication has not started, G8 remains open, and the 100-epoch retrain is not authorized.

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
| 8 | `phases/09_PHASE_8_EXISTING_MODEL_RETRAINING.md` | IN_PROGRESS | G7 | G8 | R4-D-008 accepts repaired-v2 physical DATA (22,540 contracts / 67,620 files; digest `16dd4a3f...`). R4-D-009 now accepts corrected V3 logical grouping/roles after local validation: 22,394 groups, max group 7, zero address-authority edges, unchanged semantic counts, exact same physical digest. V3 research regeneration completed: 932 positive-only effective loss cells, 143 positive-only model-selection cells; 200-cell negative pilot queue remains UNKNOWN/PENDING_REVIEW; guarded selector CPU evidence shows 476 improved / 261 equal / 0 regressed over 737 over-cap records; CUDA comparison completed 4/4 worst-case probes under identical initialization. Selector promotion, confirmed-negative adjudication/evaluation design, objective selection, and full training authorization remain pending. |
| 9 | `phases/10_PHASE_9_EVALUATION_CALIBRATION_AND_POLICY.md` | WAITING | G8 | G9 | Independent roles; current threshold/calibration support remains unavailable. |
| 10 | `phases/11_PHASE_10_ACCEPTANCE_PROMOTION_AND_ROLLBACK.md` | WAITING | G9 | G10 | Final decision; untouched acceptance remains unsupported/empty/frozen. |

## Canonical `main` Phase-8 boundary

`main` is the only active repository execution line. The Phase-8 state is deliberately split into evidence layers:

| Layer | State | Meaning |
|---|---|---|
| Historical v1 / G7 | PASSED / immutable | `sentinel-r4-vnext-v1`, `r4-vnext-roles-v1`, graph schema v9 and binding digest `7637461f6643d398c7a0446412fedd8877914c7b9ed41309dab45f18ed96f420` remain reproducible historical evidence. |
| Repaired-v2 physical source/representations | ACCEPTED / immutable evidence | R4-D-008 accepts 22,540 repaired contracts and 67,620 graph/token/sidecar files with zero missing/invalid; physical binding digest `16dd4a3f98c34e52e5c411b39268361881efede07e8f3f52d0c060dd1c5bb6dd`. This physical evidence remains reusable. |
| Repaired-v2 grouping/roles | SUPERSEDED FOR FUTURE RESEARCH | V2 grouping used arbitrary same-source address literals as union authority. Full-population audit found a 10,327-contract DIVE component driven by common addresses. V2 grouping/roles and V2 role-dependent research outputs remain historical evidence but are not future training/evaluation authority. |
| Logical V3 grouping/roles/publication | ACCEPTED | R4-D-009 accepts `r4-leakage-groups-v3`, `r4-vnext-roles-v3`, `sentinel-r4-vnext-v3`: 22,394 groups, max group size 7, 146 normalized-code edges, zero address-authority edges. Semantic counts are unchanged and V3 binds the same 67,620 physical files to the exact repaired-v2 digest. |
| Phase-8 research regeneration | COMPLETE THROUGH BOUNDED CUDA | V3 representation sensitivity, selector CPU comparison, V3 confirmed-negative pilot queue, and identical-initialization CUDA comparison are complete. All 4 requested worst-case probes completed. Guarded selector evidence supports moving to a separate promotion ADR, not silent promotion. |
| Confirmed-negative evaluation | APPROVED GAP / REVIEW NOT STARTED | R4-GAP-007 authorizes the V3 pilot: 200 PENDING_REVIEW cells, 25 per enabled class, all target `None`, all from `TRAIN_UNLABELED`, no negative-truth claim. Accepted negatives would initially be evaluation-only. |
| Full training / G8 | HOLD | All current supervised/model-selection cells remain positive-only; confirmed negatives, threshold/calibration, untouched acceptance, objective decision, and selector promotion remain unresolved. No full-run horizon is authorized. |

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

Current accepted logical V3 identifiers:

- grouping: `r4-leakage-groups-v3`;
- role partition: `r4-vnext-roles-v3`;
- DATA publication: `sentinel-r4-vnext-v3`;
- logical build: `r4-logical-lineage-v3`.

## Current restart boundary

Read first:

`runs/2026-08-16_PHASE8_logical_v3_acceptance_and_research_checkpoint.md`.

The earlier `runs/2026-08-15_PHASE8_logical_v3_grouping_repair_handoff.md` is now the completed execution procedure/history for the V3 repair sequence.

Current next controlled steps are:

1. run the Git-safe final V3 evidence snapshot helper and commit only its sanitized evidence directory;
2. if promoting `target_aware_guarded_v1`, create a separate selector-promotion ADR/decision and new extractor/representation lineage rather than mutating repaired-v2 bound tokens;
3. conduct R4-GAP-007 negative adjudication only under the explicit class-specific + independent-review contract;
4. decide objective/evaluation design after negative-evidence yield is known;
5. reconsider a full training horizon only after selector/objective/evaluation authority is versioned and bound.

Do **not** manually adjudicate the old V2 confirmed-negative queue. Do **not** treat the new V3 pilot queue as negative truth. Do **not** silently promote the selector, invent pseudo-negatives, or launch the 100-epoch job.

The completed repaired-v2 rebuild and R4-D-008 remain the physical-data reproducibility root. R4-D-009 governs the accepted logical V3 correction. R4-GAP-007 governs any upcoming confirmed-negative review.

## Status vocabulary

- `READY`
- `IN_PROGRESS`
- `BLOCKED`
- `FAILED`
- `PASSED`
- `WAITING`
- `SUPERSEDED`
