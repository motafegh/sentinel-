# R4 Plan Status Matrix

**Scope:** canonical R4 execution status on `main`. Historical G0–G7 evidence remains valid for the immutable `sentinel-r4-vnext-v1` lineage. Phase 8 remains `IN_PROGRESS`; G8 is not passed and full training is unauthorized. Repaired-v2 physical source/representation evidence remains accepted under R4-D-008, corrected logical V3 remains accepted under R4-D-009, and the exact protected-local V10 V2.6 representation root is physically accepted under R4-D-011 with binding digest `d9f925588913e66476cfbc097bace7daa7e673295fe2a243760313d0bef5ebdd`. The hardened 2026-08-16 Git-safe snapshot remains the accepted pre-pilot logical evidence boundary at `44fbb9c1d2033be8002fe404d650cf09f08b0f29`. R4-GAP-007 is separately in progress: candidate #1 is `NOT_CONFIRMED`; candidate #2 primary review supports `CONFIRMED_NEGATIVE` but still requires genuinely independent agreement, so confirmed negatives remain zero. R4-D-010 preserves v9 for historical reproduction while prohibiting it for the new full run. V2.6 Stages A-D and the complete V4 audit pass for 22,540 identities; all 355 current structural drifts are reconciled as 349 persistent-storage WRITE corrections plus 6 exact index-equivalent graphs with zero unexplained drift. Physical V10 acceptance is complete, but selector, objective/evaluation, threshold/calibration, untouched-acceptance, and explicit training authorization remain open.

| Phase | File | Status | Entry condition | Exit gate | Notes |
|---:|---|---|---|---|---|
| 0 | `phases/01_PHASE_0_BASELINE_AND_EVIDENCE_LOCATION.md` | PASSED | Master plan adopted | G0 | Phase 0 complete; G0 PASS |
| 1 | `phases/02_PHASE_1_PREVIOUS_EVIDENCE_RECOVERY.md` | PASSED | G0 | G1 | Phase 1 complete; G1 PASS |
| 2 | `phases/03_PHASE_2_LABEL_CORRUPTION_RECONSTRUCTION.md` | PASSED | G1 | G2 | Historical positive/zero origins reconstructed; G2 PASS |
| 3 | `phases/04_PHASE_3_EVIDENCE_LEDGER.md` | PASSED | G2 | G3 | Historical 22,493-contract / 224,930-row ledger materialized and validated; G3 PASS |
| 4 | `phases/05_PHASE_4_TARGETED_GAP_ADJUDICATION.md` | PASSED | G3 | G4 | R4-GAP-002 resolved; DIVE role decisions bounded; G4 PASS |
| 5 | `phases/06_PHASE_5_DATA_VNEXT_POLICY_AND_DESIGN.md` | PASSED | G4 | G5 | `data-vnext-policy-v1`; eight classes enabled, GasException/UnusedReturn disabled; no blanket negatives; G5 PASS |
| 6 | `phases/07_PHASE_6_PARTITIONS_AND_ACCEPTANCE_FREEZE.md` | PASSED | G5 | G6 | Historical `r4-vnext-roles-v1` frozen; threshold/calibration/untouched acceptance unsupported/empty; G6 PASS |
| 7 | `phases/08_PHASE_7_DATA_VNEXT_IMPLEMENTATION.md` | PASSED | G6 | G7 | Historical `sentinel-r4-vnext-v1` / 21,657 representations / 64,971 files passed G7; immutable historical evidence |
| 8 | `phases/09_PHASE_8_EXISTING_MODEL_RETRAINING.md` | IN_PROGRESS | G7 | G8 | Repaired-v2 physical DATA, logical V3, and exact V10 V2.6 physical representations are accepted evidence. R4-D-010 keeps v9 ineligible; R4-D-011 accepts V2.6 digest `d9f925...`. Selector promotion, negative-evidence, objective/evaluation, and launch authority remain separate; full training is unauthorized. |
| 9 | `phases/10_PHASE_9_EVALUATION_CALIBRATION_AND_POLICY.md` | WAITING | G8 | G9 | Current threshold/calibration support remains unavailable |
| 10 | `phases/11_PHASE_10_ACCEPTANCE_PROMOTION_AND_ROLLBACK.md` | WAITING | G9 | G10 | Untouched acceptance remains unsupported/empty/frozen |

## Canonical `main` Phase-8 boundary

| Layer | State | Meaning |
|---|---|---|
| Historical v1 / G7 | PASSED / immutable | `sentinel-r4-vnext-v1`, `r4-vnext-roles-v1`, graph schema v9 remain reproducible historical evidence |
| Repaired-v2 physical source/representations | ACCEPTED / immutable historical and reproducibility evidence | 22,540 contracts; 67,620 graph/token/sidecar files; zero missing/invalid; physical binding digest `16dd4a3f98c34e52e5c411b39268361881efede07e8f3f52d0c060dd1c5bb6dd`; v9 is not eligible for a new full run under R4-D-010 |
| Repaired-v2 grouping/roles | SUPERSEDED FOR FUTURE RESEARCH | V2 address-literal grouping produced a 10,327-contract DIVE component; keep only as historical logical evidence |
| Logical V3 grouping/roles/publication | ACCEPTED | 22,394 groups; max group size 7; 146 normalized-code edges; zero address-authority edges; semantic counts and physical binding unchanged |
| V3 evidence implementation | HARDENED | MODEL_SELECTION/INTERNAL_AUDIT reporting separated; snapshot coherence fail-closed; sensitivity/selector/queue reports lineage-bound; queue globally group-unique; explicit source-family IDs source-namespaced |
| Hardened protected-local V3 research | REGENERATED / DURABLE SNAPSHOT COMMITTED | Acceptance, sensitivity, CPU selector, globally unique queue, and CUDA comparison all regenerated from source commit `83bd566b9...`; final snapshot at `docs/plan/ml-R4/evidence/2026-08-15_phase8_logical_v3/` passed coherence and SHA verification and was committed at `44fbb9c1d...` |
| Confirmed-negative evaluation | IN_PROGRESS | Candidate #1 is `NOT_CONFIRMED`. Candidate #2 primary review supports `CONFIRMED_NEGATIVE`, but authoritative truth remains UNKNOWN / PENDING_REVIEW / target `None` until a genuinely independent reviewer agrees. Accepted negatives, if any, remain evaluation-only. |
| Historical V10 V2.3/V2.4 diagnostics | PRESERVED / NOT PHYSICALLY ACCEPTED | V2.3 is the frozen structural reference. The later protected V2.4 candidate completed the 26-contract parse-only repair: 22,540 identities, exact accepted-V9 token bytes, zero parse-only artifacts, zero unclassified call IR, and the required 22,539 Slither-0.10 + 1 Slither-0.11.5 runtime split. These are diagnostic/history roots, not the current future-training candidate. |
| V10 V2.5 bounded structural correction | PASSED / 20 OF 20 CLOSED | Extractor `v2.5-r4-call-semantics-deterministic-cfg`; three fresh 20-identity generations under exact Slither 0.10.0; 8 exact node-index-invariant graph-equivalence identities + 12 deterministic persistent-storage WRITE corrections; bounded verifier passed with zero unexplained drift and no blockers. |
| Historical V10 V2.5 full-candidate gate | BLOCKED AT STAGE E / PRESERVED | Protected-local Stages A-D passed, but Stage E found 311 raw non-parse-only drifts and left 298 outside the approved bounded evidence classes. This historical failure motivated V2.6 and is not current physical authority. |
| V10 V2.6 physical representation | ACCEPTED / IMMUTABLE LOCAL | R4-D-011 accepts the exact 22,540-identity root and digest `d9f925588913e66476cfbc097bace7daa7e673295fe2a243760313d0bef5ebdd`. V4 re-proves all 355 current drifts as 349 WRITE corrections plus 6 index-equivalent graphs with zero unexplained drift. Runtime split is 22,539 Slither-0.10 + one Slither-0.11.5 exception; all tokens are byte-identical to accepted V9. |
| Selector promotion | NOT AUTHORIZED | Guarded candidate retains promising CPU/CUDA evidence but needs full-population bound-token control-equivalence evidence plus a separate versioned promotion decision |
| Full training / G8 | HOLD | V10 V2.6 physical representations are accepted, but there are no confirmed negatives; threshold/calibration/untouched acceptance remain unsupported; selector is unpromoted; objective/evaluation design and explicit launch authority remain unresolved. |

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
- graph schema: `v9` (immutable accepted historical/physical evidence; not future-full-training eligible);
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

Current required but not-yet-accepted future physical candidate identifiers:

- graph schema: `v10`;
- extractor: `v2.5-r4-call-semantics-deterministic-cfg`;
- final candidate basename: `representations-r4-v3-candidate`;
- preprocessing input remains accepted `sentinel-preprocessed-r4-v2`;
- primary runtime: Slither 0.10.0;
- declared identity-bound exception: `dive/caa35c1a5906269bbe5e70de780d105c2968ece4fc038d7f7208efee681aeec9` under Slither 0.11.5.

V3 family authority after hardening:

- normalized-code identity remains global;
- exact artifact identity remains global by artifact identity/hash;
- explicit source-native family/project identifiers are keyed as `<source>:<field>:<value>`;
- Ethereum address literals remain diagnostic-only and create zero union edges.

The accepted current V3 artifact had zero explicit-family edges, so source-namespacing hardening did not invalidate the accepted V3 grouping population.

## Durable current evidence

Accepted V3 evidence root:

`docs/plan/ml-R4/evidence/2026-08-15_phase8_logical_v3/`

Current exact restart checkpoint:

`runs/2026-08-27_PHASE8_v10_v25_current_restart_checkpoint.md`

Bounded structural closure:

`reviews/R4-GAP-008/2026-08-26_v10_v25_bounded_structural_closure.md`

Current full-candidate construction protocol:

`runs/2026-08-26_PHASE8_v10_v25_full_candidate_staging.md`

Governing representation decision:

`adrs/ADR-R4-010-versioned-external-call-representation-correction.md`

Candidate #1 primary-review closeout:

`runs/2026-08-21_PHASE8_gap007_candidate1_primary_review.md`

Candidate #2 primary-review state:

`runs/2026-08-21_PHASE8_gap007_candidate2_primary_review.md`

Key durable V3 facts remain:

- hardened source commit for regenerated reports: `83bd566b9c4f4f653e530c2c0f5c990858dd759d`;
- final snapshot commit: `44fbb9c1d2033be8002fe404d650cf09f08b0f29`;
- final snapshot: `coherence=PASS`;
- all JSON files listed in `SHA256SUMS.txt` verified `OK` before commit;
- CPU selector: 1,018 analyzed / 737 over-cap / 476 improved / 261 equal / 0 regressed;
- confirmed-negative queue: 200 cells / 200 globally unique groups / all PENDING_REVIEW / target None / negative_truth_claim=false;
- CUDA selector comparison: identical initialization true / 4 of 4 worst-case probes / no Run12 state / no checkpoint / no promotion or training authorization.

## Current restart boundary

Read first:

`runs/2026-08-27_PHASE8_v10_v25_current_restart_checkpoint.md`

Then read:

- `reviews/R4-GAP-008/2026-08-26_v10_v25_bounded_structural_closure.md`;
- `runs/2026-08-26_PHASE8_v10_v25_full_candidate_staging.md`;
- `adrs/ADR-R4-010-versioned-external-call-representation-correction.md`;
- `runs/2026-08-16_PHASE8_v3_hardened_evidence_snapshot_closeout.md`;
- `runs/2026-08-21_PHASE8_gap007_candidate2_primary_review.md` for the separate confirmed-negative track.

Historical execution context only, not current restart authority:

- `runs/2026-08-21_PHASE8_gap008_external_call_semantics_audit.md`;
- `runs/2026-08-21_PHASE8_v10_external_call_implementation_handoff.md`;
- `runs/2026-08-21_PHASE8_v10_implementation_and_local_regression.md`;
- `runs/2026-08-23_PHASE8_v10_parse_only_resolution_working_plan.md`;
- `runs/2026-08-23_PHASE8_v10_structural_drift_probe_handoff.md`;
- `runs/2026-08-16_PHASE8_v3_evidence_hardening_handoff.md`;
- `runs/2026-08-16_PHASE8_logical_v3_acceptance_and_research_checkpoint.md`.

Current execution order:

1. preserve v9/repaired-v2, accepted logical V3, frozen V2.3 structural reference, and protected V2.4 diagnostic candidate as immutable evidence/history;
2. preserve the R4-D-011 accepted V2.6 root, its Stage A-D reports, 355-case evidence, V4 review, and exact hashes; do not regenerate or mutate them;
3. keep candidate #2 UNKNOWN/target `None` unless independent agreement completes its dual review; any accepted negative remains evaluation-only;
4. separately design/execute control-selector → bound-token equivalence before any guarded-selector promotion ADR;
5. revisit objective/evaluation design, including possible PU learning, only after new evidence supports a versioned decision;
6. define credible threshold/calibration/untouched-acceptance support rather than inventing populations;
7. reconsider training authorization only after all remaining evidence/design gates are satisfied and record it explicitly.

Do **not** manually adjudicate the obsolete V2 queue or the pre-hardening V3 queue. Do **not** patch v9 in place, train from v9, infer target `0`, self-verify candidate #2, silently promote the selector, invent pseudo-negatives, reuse Run12 state, fit unsupported threshold/calibration roles, implement PU as an ungoverned shortcut, overwrite protected V10 diagnostic/reference roots, or launch the 100-epoch job.

## Status vocabulary

- `READY`
- `IN_PROGRESS`
- `BLOCKED`
- `FAILED`
- `PASSED`
- `WAITING`
- `SUPERSEDED`
