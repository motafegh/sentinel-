# R4 Plan Status Matrix

**Scope:** canonical R4 execution status on `main`. Historical G0–G7 evidence remains valid for the immutable `sentinel-r4-vnext-v1` lineage. Phase 8 is still `IN_PROGRESS`. Repaired-v2 physical source/representation evidence remains accepted under R4-D-008. Corrected logical V3 remains accepted under R4-D-009. A 2026-08-16 protected-local audit found reporting and evidence-lineage hardening defects in the post-acceptance research tranche. Repository fixes are now implemented; the affected local reports must be regenerated before the final Git-safe V3 evidence snapshot. G8 remains open and the 100-epoch retrain is not authorized.

| Phase | File | Status | Entry condition | Exit gate | Notes |
|---|---|---|---|---|---|
| 0 | `phases/01_PHASE_0_BASELINE_AND_EVIDENCE_LOCATION.md` | PASSED | Master plan adopted | G0 | Phase 0 complete; G0 PASS |
| 1 | `phases/02_PHASE_1_PREVIOUS_EVIDENCE_RECOVERY.md` | PASSED | G0 | G1 | Phase 1 complete; G1 PASS |
| 2 | `phases/03_PHASE_2_LABEL_CORRUPTION_RECONSTRUCTION.md` | PASSED | G1 | G2 | Historical positive/zero origins reconstructed; G2 PASS |
| 3 | `phases/04_PHASE_3_EVIDENCE_LEDGER.md` | PASSED | G2 | G3 | Historical 22,493-contract / 224,930-row ledger materialized and validated; G3 PASS |
| 4 | `phases/05_PHASE_4_TARGETED_GAP_ADJUDICATION.md` | PASSED | G3 | G4 | R4-GAP-002 resolved; DIVE role decisions bounded; G4 PASS |
| 5 | `phases/06_PHASE_5_DATA_VNEXT_POLICY_AND_DESIGN.md` | PASSED | G4 | G5 | `data-vnext-policy-v1`; eight classes enabled, GasException/UnusedReturn disabled; no blanket negatives; G5 PASS |
| 6 | `phases/07_PHASE_6_PARTITIONS_AND_ACCEPTANCE_FREEZE.md` | PASSED | G5 | G6 | Historical `r4-vnext-roles-v1` frozen; threshold/calibration/untouched acceptance unsupported/empty; G6 PASS |
| 7 | `phases/08_PHASE_7_DATA_VNEXT_IMPLEMENTATION.md` | PASSED | G6 | G7 | Historical `sentinel-r4-vnext-v1` / 21,657 representations / 64,971 files passed G7; immutable historical evidence |
| 8 | `phases/09_PHASE_8_EXISTING_MODEL_RETRAINING.md` | IN_PROGRESS | G7 | G8 | Repaired-v2 physical DATA accepted; logical V3 accepted. Post-acceptance audit found: combined outcome-metric population had been mislabeled as MODEL_SELECTION; snapshot lacked cross-report coherence checks; sensitivity lacked immutable lineage; queue global group uniqueness was not enforced; explicit source family IDs were not source-namespaced. Repository hardening is implemented. Protected local acceptance/sensitivity/selector/queue/GPU reports must now be regenerated coherently before final snapshot. Confirmed negatives remain zero and full training is unauthorized. |
| 9 | `phases/10_PHASE_9_EVALUATION_CALIBRATION_AND_POLICY.md` | WAITING | G8 | G9 | Current threshold/calibration support remains unavailable |
| 10 | `phases/11_PHASE_10_ACCEPTANCE_PROMOTION_AND_ROLLBACK.md` | WAITING | G9 | G10 | Untouched acceptance remains unsupported/empty/frozen |

## Canonical `main` Phase-8 boundary

| Layer | State | Meaning |
|---|---|---|
| Historical v1 / G7 | PASSED / immutable | `sentinel-r4-vnext-v1`, `r4-vnext-roles-v1`, graph schema v9 remain reproducible historical evidence |
| Repaired-v2 physical source/representations | ACCEPTED / immutable evidence | 22,540 contracts; 67,620 graph/token/sidecar files; zero missing/invalid; physical binding digest `16dd4a3f98c34e52e5c411b39268361881efede07e8f3f52d0c060dd1c5bb6dd` |
| Repaired-v2 grouping/roles | SUPERSEDED FOR FUTURE RESEARCH | V2 address-literal grouping produced a 10,327-contract DIVE component; keep only as historical logical evidence |
| Logical V3 grouping/roles/publication | ACCEPTED | 22,394 groups; max group size 7; 146 normalized-code edges; zero address-authority edges; semantic counts and physical binding unchanged |
| V3 evidence implementation | HARDENED IN REPOSITORY | MODEL_SELECTION/INTERNAL_AUDIT reporting separated; snapshot coherence fail-closed; sensitivity/selector/queue reports lineage-bound; queue globally group-unique; explicit source-family IDs source-namespaced |
| Protected local V3 research reports | REGENERATION REQUIRED | Earlier results remain useful pre-hardening observations but are not the final durable evidence package. Re-run acceptance, sensitivity, CPU selector, negative queue, and GPU comparison under the hardened source commit |
| Confirmed-negative evaluation | APPROVED GAP / REVIEW NOT STARTED | R4-GAP-007 remains active. Do not adjudicate the pre-hardening queue; regenerate and inspect the globally unique queue first |
| Full training / G8 | HOLD | No confirmed negatives; no threshold/calibration/untouched acceptance; selector unpromoted; objective/evaluation design unresolved; no full-run horizon authorized |

## Corrected outcome-population terminology

The V3 publication intentionally permits `outcome_metric_mask_*` for both `MODEL_SELECTION` and `INTERNAL_AUDIT` roles. Therefore:

- **143 contracts / 142 unique groups** is the previously observed **combined outcome-metric/audit population**, not the MODEL_SELECTION population;
- the protected-local audit observed active `MODEL_SELECTION = 71 contracts / 71 groups`;
- the protected-local audit observed active `INTERNAL_AUDIT = 72 contracts / 71 groups`;
- frozen role assignment remains `MODEL_SELECTION = 73 contracts / 71 groups` and `INTERNAL_AUDIT = 73 contracts / 71 groups`.

These active counts must be reproduced by the hardened acceptance rerun before they are treated as final snapshot evidence. The ML adapter itself permits `MODEL_SELECTION` for model selection and does not load `INTERNAL_AUDIT`; the defect was reporting/research-population labeling, not trainer leakage.

## Version boundaries

Historical/physical roots that remain immutable:

- historical G7 publication: `sentinel-r4-vnext-v1`;
- repaired preprocessing: `sentinel-preprocessed-r4-v2`;
- repaired provenance/source claims: `r4-provenance-v1`;
- role-independent evidence ledger: `evidence-ledger-r4-v2`;
- repaired representations: `representations-r4-v2` / extractor `v2.2-r4-repaired`;
- graph schema: `v9`;
- token tensor contract: `[4,512]`.

Historical/superseded logical V2 identifiers:

- grouping: `r4-leakage-groups-v2`;
- partition: `r4-vnext-roles-v2`;
- publication: `sentinel-r4-vnext-v2`.

Accepted logical V3 identifiers:

- grouping: `r4-leakage-groups-v3`;
- partition: `r4-vnext-roles-v3`;
- publication: `sentinel-r4-vnext-v3`;
- logical build: `r4-logical-lineage-v3`.

V3 family authority after hardening:

- normalized-code identity remains global;
- exact artifact identity remains global by artifact identity/hash;
- explicit source-native family/project identifiers are keyed as `<source>:<field>:<value>`;
- Ethereum address literals remain diagnostic-only and create zero union edges.

The accepted current V3 artifact had zero explicit-family edges, so source-namespacing hardening does not invalidate the accepted V3 grouping population.

## Current restart boundary

Read first:

`runs/2026-08-16_PHASE8_v3_evidence_hardening_handoff.md`

The earlier `runs/2026-08-16_PHASE8_logical_v3_acceptance_and_research_checkpoint.md` remains a pre-hardening historical checkpoint. Its instruction to create the final snapshot next is superseded.

Current execution order:

1. synchronize local `main` to the hardened repository head;
2. re-run corrected V3 acceptance reporting;
3. regenerate lineage-bound representation sensitivity;
4. regenerate lineage-bound CPU selector evidence;
5. regenerate the globally group-unique V3 confirmed-negative queue;
6. re-run the V3 CUDA selector comparison using the newly bound sensitivity report;
7. run the final snapshot helper only when all upstream reports agree; it must print `coherence=PASS`;
8. inspect and commit only the sanitized final evidence snapshot;
9. then prioritize R4-GAP-007 confirmed-negative adjudication and separately consider selector promotion.

Before selector promotion, add/execute a full-population check that the historical control selector reproduces the currently bound representation token tensors exactly. Coverage/CUDA evidence alone is not sufficient to change the bound extractor policy.

Do **not** manually adjudicate the V2 queue or the pre-hardening V3 queue. Do **not** infer target `0`, silently promote the selector, invent pseudo-negatives, reuse Run12 state, or launch the 100-epoch job.

## Status vocabulary

- `READY`
- `IN_PROGRESS`
- `BLOCKED`
- `FAILED`
- `PASSED`
- `WAITING`
- `SUPERSEDED`
