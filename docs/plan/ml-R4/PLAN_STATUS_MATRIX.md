# R4 Plan Status Matrix

**Scope:** canonical R4 execution status on `main`. Historical G0–G7 evidence remains valid for immutable `sentinel-r4-vnext-v1`. Phase 8 remains `IN_PROGRESS`; G8 is not passed and full training is unauthorized. R4-D-009 is the accepted logical V3 authority; R4-D-011 accepts the exact V10/V2.6 graph/control root; R4-D-013 accepts the exact guarded-token successor; R4-D-014 limits ML execution to the bounded positive-only objective/evaluation contract; R4-D-015 closes M4 with `PASS_BOUNDED_PILOT_FULL_TRAINING_STILL_HELD`; and R4-D-016 resolves the `CallToUnknown` semantic-definition inconsistency. M0–M4 are complete. The active Phase-8 evidence boundary is R4-GAP-007: confirmed negatives remain zero and candidate #2 remains `UNKNOWN / PENDING_REVIEW / target None`. The next action is to build/verify the self-contained candidate #2 blind-review v2 package and obtain genuinely isolated independent review. M5/full training remains HOLD.

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
| 8 | `phases/09_PHASE_8_EXISTING_MODEL_RETRAINING.md` | IN_PROGRESS | G7 | G8 | R4-D-011 V10 graphs and R4-D-013 guarded tokens are accepted. R4-D-014 defines bounded positive-only control/evaluation. R4-D-015 closes M4 but keeps M5/full training on HOLD. R4-D-016 resolves `CallToUnknown` semantics; R4-GAP-007 independent negative evidence remains incomplete. |
| 9 | `phases/10_PHASE_9_EVALUATION_CALIBRATION_AND_POLICY.md` | WAITING | G8 | G9 | Threshold/calibration support remains unavailable |
| 10 | `phases/11_PHASE_10_ACCEPTANCE_PROMOTION_AND_ROLLBACK.md` | WAITING | G9 | G10 | Untouched acceptance remains unsupported/empty/frozen |

## Canonical `main` Phase-8 boundary

| Layer | State | Meaning |
|---|---|---|
| Historical v1 / G7 | PASSED / immutable | `sentinel-r4-vnext-v1`, `r4-vnext-roles-v1`, graph schema v9 remain reproducible historical evidence |
| Repaired-v2 physical source/representations | ACCEPTED / immutable historical and reproducibility evidence | 22,540 contracts; 67,620 graph/token/sidecar files; physical binding digest `16dd4a3f...`; v9 is not eligible for a new full run under R4-D-010 |
| Logical V3 grouping/roles/publication | ACCEPTED | R4-D-009: 22,394 groups, max group size 7, 146 normalized-code edges, zero address-authority edges |
| Hardened V3 evidence snapshot | ACCEPTED / immutable snapshot | `docs/plan/ml-R4/evidence/2026-08-15_phase8_logical_v3/`, snapshot commit `44fbb9c1d...`, `coherence=PASS` |
| `CallToUnknown` semantic authority | ACCEPTED / R4-D-016 | Raw/dynamic low-level unknown-target class. `call` / `delegatecall` / legacy `callcode` / `staticcall` / `send` are in scope. `transfer`, ordinary typed high-level calls, in-scope library calls, contract creation and selfdestruct/suicide destinations are not positive solely because their destination is unverified. Class index/order and taxonomy schema v1 remain unchanged. |
| Confirmed-negative evaluation | IN_PROGRESS / INDEPENDENT REVIEW PENDING | Confirmed negatives remain zero. Candidate #1's historical `NOT_CONFIRMED` rationale is definition-stale under R4-D-016; its queue state remains UNKNOWN/PENDING_REVIEW/target `None` and it needs a new primary review if reused. Candidate #2 primary remains semantically compatible but is still UNKNOWN/PENDING_REVIEW/target `None`; self-contained v2 blind review is next. Three prior external attempts are diagnostic only. |
| Historical V10 V2.3/V2.4 diagnostics | PRESERVED / NOT PHYSICALLY ACCEPTED | V2.3 structural reference and V2.4 parse-only repair remain diagnostic/history roots |
| Historical V10 V2.5 full-candidate evidence | PRESERVED / historical | Bounded evidence closed its original 20 cases; historical Stage E failure motivated V2.6 and is not current physical authority |
| V10 V2.6 physical representation | ACCEPTED / IMMUTABLE LOCAL | R4-D-011 accepts exact 22,540-identity root, digest `d9f925588913e66476cfbc097bace7daa7e673295fe2a243760313d0bef5ebdd`, 349 WRITE corrections + 6 index-equivalent graphs, zero unexplained drift |
| Guarded-token physical successor | ACCEPTED / IMMUTABLE LOCAL | R4-D-013 accepts `r4-v10-v26-guarded-tokens-v1`, digest `9885d7b88a46aff4102741d63eeaa0bd6968f0857f7bca6a389662bfaa158881`, 14,751 guarded + 7,789 historical fallback |
| Objective/evaluation boundary | ACCEPTED / LIMITED | R4-D-014 retains masked positive BCE as bounded control only; MODEL_SELECTION is positive-fit diagnostic only; no negative/PU, threshold, calibration, untouched acceptance or promotion authority |
| M3 guarded integration | COMPLETE | Exact D9/D11/D13/D14 binding, V10 17-edge model, 932 active train groups, 71 active selection groups; no-step preflight passed |
| M4 bounded pilot | COMPLETE | R4-D-015 accepts M4-A/B/C mechanics, recovery and 8-epoch/120-step positive-fit dynamics. Outcome: `PASS_BOUNDED_PILOT_FULL_TRAINING_STILL_HELD`. |
| Full training / G8 | HOLD | M4 passed, but confirmed-negative evidence remains zero; candidate #2 admissible independent review is unresolved; threshold/calibration/untouched roles remain unsupported; no M5 launch authority exists |

## Corrected outcome-population terminology

The V3 publication intentionally permits `outcome_metric_mask_*` for both `MODEL_SELECTION` and `INTERNAL_AUDIT` roles. The hardened acceptance establishes:

- **143 contracts / 142 unique groups** = combined outcome-metric population across `MODEL_SELECTION` + `INTERNAL_AUDIT`, not MODEL_SELECTION alone;
- active `MODEL_SELECTION = 71 contracts / 71 groups`;
- active `INTERNAL_AUDIT = 72 contracts / 71 groups`;
- frozen role assignment remains `MODEL_SELECTION = 73 contracts / 71 groups` and `INTERNAL_AUDIT = 73 contracts / 71 groups`.

The ML adapter permits `MODEL_SELECTION` for model-selection loading and does not load `INTERNAL_AUDIT`; the old defect was reporting/research-population labeling, not trainer leakage.

## Version boundaries

Historical/physical roots that remain immutable:

- historical G7 publication: `sentinel-r4-vnext-v1`;
- repaired preprocessing: `sentinel-preprocessed-r4-v2`;
- repaired provenance/source claims: `r4-provenance-v1`;
- role-independent evidence ledger: `evidence-ledger-r4-v2`;
- repaired representations: `representations-r4-v2` / extractor `v2.2-r4-repaired`;
- historical graph schema v9;
- token tensor contract `[4,512]`.

Accepted logical V3 identifiers:

- grouping: `r4-leakage-groups-v3`;
- partition: `r4-vnext-roles-v3`;
- publication: `sentinel-r4-vnext-v3`;
- logical build: `r4-logical-lineage-v3`.

Accepted R4-D-011 V10 physical parent:

- graph schema `v10`;
- extractor `v2.6-r4-call-semantics-deterministic-cfg-mutators`;
- root basename `representations-r4-v3-candidate` within the exact protected-local root recorded by R4-D-011;
- preprocessing input `sentinel-preprocessed-r4-v2`;
- primary runtime Slither 0.10.0;
- one identity-bound Slither 0.11.5 exception;
- historical-control token bytes retained as R4-D-011 control.

Accepted R4-D-013 guarded-token successor:

- graph parent remains exact R4-D-011 V10/V2.6;
- selector `target_aware_guarded_v1`;
- control `historical_linspace_v1`;
- token tensor `[4,512]`;
- lineage `r4-v10-v26-guarded-tokens-v1`;
- digest `9885d7b88a46aff4102741d63eeaa0bd6968f0857f7bca6a389662bfaa158881`;
- 22,540 identities = 14,751 guarded + 7,789 exact historical fallback.

V3 family authority remains:

- normalized-code identity global;
- exact artifact identity global;
- explicit source-native family/project identifiers source-namespaced;
- Ethereum address literals diagnostic-only and create zero union edges.

## Durable current evidence

Accepted V3 evidence root:

`docs/plan/ml-R4/evidence/2026-08-15_phase8_logical_v3/`

Current V10 physical authority:

- `runs/2026-09-02_PHASE8_v10_v26_physical_acceptance_and_no_launch.md`;
- `adrs/ADR-R4-011-v10-v26-physical-representation-acceptance.md`.

Current guarded-token physical authority:

- `runs/2026-09-29_PHASE8_guarded_token_physical_acceptance_and_no_launch.md`;
- `adrs/ADR-R4-013-guarded-token-physical-representation-acceptance.md`.

Current ML objective/evaluation authority:

- `runs/2026-10-01_PHASE8_objective_evaluation_decision_and_pilot_boundary.md`;
- `adrs/ADR-R4-014-phase8-positive-only-control-and-evaluation-hold.md`.

Current M4 closure:

- `runs/2026-10-17_PHASE8_m4_bounded_pilot_review_and_full_training_hold.md`;
- `adrs/ADR-R4-015-bounded-pilot-pass-full-training-hold.md`;
- `evidence/2026-10-17_phase8_m4_bounded_pilot/decision.json`.

Current `CallToUnknown` semantic authority:

- `adrs/ADR-R4-016-call-to-unknown-semantic-clarification.md`;
- `evidence/2026-10-20_call_to_unknown_semantic_alignment/decision.json`;
- `docs/plan/system-finalization/technical-completion/working/2026-10-20_ml_gap007_call_to_unknown_semantic_alignment.md` (closed working evidence).

Candidate #2 primary review remains primary evidence only, not independent truth:

`runs/2026-08-21_PHASE8_gap007_candidate2_primary_review.md`.

Blind-review v1 remains historical only and is retired for future adjudication. Current bundle builder:

`docs/plan/ml-R4/scripts/p8_build_gap007_candidate2_independent_review_v2.py`.

## Current restart boundary

For current DATA/ML technical-completion work, read in this order:

1. applicable repository/module instructions;
2. this `PLAN_STATUS_MATRIX.md`;
3. R4-D-016 ADR + machine decision;
4. `docs/plan/system-finalization/technical-completion/02_ML_TRAINING_EVALUATION_COMPLETION_PLAN.md`;
5. R4-D-015 run/ADR/machine decision for M4 closure;
6. `EVIDENCE_GAP_REGISTER.md`, `RISK_AND_BLOCKER_REGISTER.md`, `CLAIM_STATUS_MATRIX.md`, `DECISION_REGISTER.md`;
7. R4-D-011/R4-D-013/R4-D-014 records when their exact authority is needed;
8. the accepted V3 snapshot when logical/evidence population detail is needed.

Historical execution context only, not current restart authority:

- the in-progress stop lines inside `working/2026-10-03_ml_m4_bounded_pilot.md`;
- V2.5 staging/restart records whose blockers were closed by R4-D-011;
- earlier selector research reports superseded by R4-D-012/R4-D-013;
- pre-hardening V3 queues;
- candidate #1's historical `NOT_CONFIRMED` rationale when interpreting current R4-D-016 semantics;
- candidate #2 blind-review bundle v1 for future adjudication.

## Current execution order

1. preserve all accepted R4-D-009/R4-D-011/R4-D-013 physical/logical roots and historical evidence;
2. preserve R4-D-014/R4-D-015 boundaries: no M5/full training, threshold/calibration, promotion, pseudo-negatives or Run12-state reuse;
3. preserve R4-D-016 `CallToUnknown` semantics;
4. keep candidate #2 `UNKNOWN / PENDING_REVIEW / target None`;
5. build and verify the self-contained `r4_gap007_candidate2_independent_review_v2.zip` using the committed deterministic builder;
6. provide only that v2 archive to a genuinely isolated reviewer with no candidate-specific prior context;
7. adjudicate candidate #2 only after the independent result returns;
8. only after new accepted evidence exists, reassess objective/evaluation/full-training authority explicitly;
9. do not infer that one/few confirmed negatives automatically support stable FPR estimation, threshold fitting or calibration.

Do **not** manually adjudicate obsolete queues, patch v9 in place, infer target `0`, self-verify candidate #2, count contaminated review attempts as independent evidence, broaden/narrow R4-D-016 silently, invent pseudo-negatives, reuse Run12 learned state, fit unsupported threshold/calibration roles, overwrite protected V10/guarded-token roots, or launch the 100-epoch job.

## Status vocabulary

- `READY`
- `IN_PROGRESS`
- `BLOCKED`
- `FAILED`
- `PASSED`
- `WAITING`
- `SUPERSEDED`
