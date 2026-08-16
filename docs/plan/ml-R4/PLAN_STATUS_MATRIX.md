# R4 Plan Status Matrix

**Scope:** canonical R4 execution status on `main`. Historical G0–G7 evidence remains valid for the immutable `sentinel-r4-vnext-v1` lineage. Phase 8 is still `IN_PROGRESS`. Repaired-v2 physical source/representation evidence remains accepted under R4-D-008. Corrected logical V3 remains accepted under R4-D-009. The 2026-08-16 evidence-hardening/regeneration tranche is now complete: the hardened acceptance, sensitivity, CPU selector, confirmed-negative queue, CUDA comparison, and final coherence-gated Git-safe snapshot were regenerated and committed at `44fbb9c1d2033be8002fe404d650cf09f08b0f29`. G8 remains open and the 100-epoch retrain is not authorized.

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
| 8 | `phases/09_PHASE_8_EXISTING_MODEL_RETRAINING.md` | IN_PROGRESS | G7 | G8 | Repaired-v2 physical DATA accepted; logical V3 accepted; post-acceptance evidence-hardening defects fixed; hardened protected-local reports regenerated coherently; final Git-safe V3 snapshot committed with `coherence=PASS`. Confirmed negatives remain zero. Primary next track is R4-GAP-007 pilot adjudication; selector promotion remains a separate decision and full training is unauthorized. |
| 9 | `phases/10_PHASE_9_EVALUATION_CALIBRATION_AND_POLICY.md` | WAITING | G8 | G9 | Current threshold/calibration support remains unavailable |
| 10 | `phases/11_PHASE_10_ACCEPTANCE_PROMOTION_AND_ROLLBACK.md` | WAITING | G9 | G10 | Untouched acceptance remains unsupported/empty/frozen |

## Canonical `main` Phase-8 boundary

| Layer | State | Meaning |
|---|---|---|
| Historical v1 / G7 | PASSED / immutable | `sentinel-r4-vnext-v1`, `r4-vnext-roles-v1`, graph schema v9 remain reproducible historical evidence |
| Repaired-v2 physical source/representations | ACCEPTED / immutable evidence | 22,540 contracts; 67,620 graph/token/sidecar files; zero missing/invalid; physical binding digest `16dd4a3f98c34e52e5c411b39268361881efede07e8f3f52d0c060dd1c5bb6dd` |
| Repaired-v2 grouping/roles | SUPERSEDED FOR FUTURE RESEARCH | V2 address-literal grouping produced a 10,327-contract DIVE component; keep only as historical logical evidence |
| Logical V3 grouping/roles/publication | ACCEPTED | 22,394 groups; max group size 7; 146 normalized-code edges; zero address-authority edges; semantic counts and physical binding unchanged |
| V3 evidence implementation | HARDENED | MODEL_SELECTION/INTERNAL_AUDIT reporting separated; snapshot coherence fail-closed; sensitivity/selector/queue reports lineage-bound; queue globally group-unique; explicit source-family IDs source-namespaced |
| Hardened protected-local V3 research | REGENERATED / DURABLE SNAPSHOT COMMITTED | Acceptance, sensitivity, CPU selector, globally unique queue, and CUDA comparison all regenerated from source commit `83bd566b9...`; final snapshot at `docs/plan/ml-R4/evidence/2026-08-15_phase8_logical_v3/` passed coherence and SHA verification and was committed at `44fbb9c1d...` |
| Confirmed-negative evaluation | APPROVED GAP / PILOT READY | R4-GAP-007 remains active. The committed hardened queue contains 200 PENDING_REVIEW cells across 200 globally unique groups; queue membership is not negative truth |
| Selector promotion | NOT AUTHORIZED | Guarded candidate retains promising CPU/CUDA evidence but needs full-population bound-token control-equivalence evidence plus a separate versioned promotion decision |
| Full training / G8 | HOLD | No confirmed negatives; no threshold/calibration/untouched acceptance; selector unpromoted; objective/evaluation design unresolved; no full-run horizon authorized |

## Corrected outcome-population terminology

The V3 publication intentionally permits `outcome_metric_mask_*` for both `MODEL_SELECTION` and `INTERNAL_AUDIT` roles. The hardened acceptance rerun now durably establishes:

- **143 contracts / 142 unique groups** = combined outcome-metric population across `MODEL_SELECTION` + `INTERNAL_AUDIT`, not the MODEL_SELECTION population;
- active `MODEL_SELECTION = 71 contracts / 71 groups`;
- active `INTERNAL_AUDIT = 72 contracts / 71 groups`;
- frozen role assignment remains `MODEL_SELECTION = 73 contracts / 71 groups` and `INTERNAL_AUDIT = 73 contracts / 71 groups`.

The ML adapter itself permits `MODEL_SELECTION` for model selection and does not load `INTERNAL_AUDIT`; the defect was reporting/research-population labeling, not trainer leakage.

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

The accepted current V3 artifact had zero explicit-family edges, so source-namespacing hardening did not invalidate the accepted V3 grouping population.

## Durable hardened evidence

Final evidence root:

`docs/plan/ml-R4/evidence/2026-08-15_phase8_logical_v3/`

Closeout/restart record:

`runs/2026-08-16_PHASE8_v3_hardened_evidence_snapshot_closeout.md`

Key durable facts:

- hardened source commit for regenerated reports: `83bd566b9c4f4f653e530c2c0f5c990858dd759d`;
- final snapshot commit: `44fbb9c1d2033be8002fe404d650cf09f08b0f29`;
- final snapshot: `coherence=PASS`;
- all JSON files listed in `SHA256SUMS.txt` verified `OK` before commit;
- CPU selector: 1,018 analyzed / 737 over-cap / 476 improved / 261 equal / 0 regressed;
- confirmed-negative queue: 200 cells / 200 globally unique groups / all PENDING_REVIEW / target None / negative_truth_claim=false;
- CUDA selector comparison: identical initialization true / 4 of 4 worst-case probes / no Run12 state / no checkpoint / no promotion or training authorization.

## Current restart boundary

Read first:

`runs/2026-08-16_PHASE8_v3_hardened_evidence_snapshot_closeout.md`

Historical context only:

- `runs/2026-08-16_PHASE8_v3_evidence_hardening_handoff.md` = the now-completed regeneration procedure;
- `runs/2026-08-16_PHASE8_logical_v3_acceptance_and_research_checkpoint.md` = pre-hardening research checkpoint.

Current execution order:

1. treat the committed coherent V3 snapshot as the durable research-evidence boundary;
2. begin the R4-GAP-007 confirmed-negative pilot using only the committed hardened queue;
3. keep candidate state UNKNOWN/PENDING_REVIEW until class-specific primary review plus independent agreeing verification establishes a confirmed negative;
4. keep accepted negatives evaluation-only unless a later versioned policy grants optimizer authority;
5. separately design/execute the full-population control-selector → currently bound token-tensor equivalence check required before any guarded-selector promotion ADR;
6. revisit objective/evaluation/training authorization only after the new evidence supports it.

Do **not** manually adjudicate the obsolete V2 queue or the pre-hardening V3 queue. Do **not** infer target `0`, silently promote the selector, invent pseudo-negatives, reuse Run12 state, fit unsupported threshold/calibration roles, or launch the 100-epoch job.

## Status vocabulary

- `READY`
- `IN_PROGRESS`
- `BLOCKED`
- `FAILED`
- `PASSED`
- `WAITING`
- `SUPERSEDED`
