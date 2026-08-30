# R4 Claim Status Matrix

This matrix describes the **current evidence-qualified claim boundary**, not the historical binary-label surface. R4-D-008 accepts repaired-v2 physical DATA as immutable reproducibility evidence; R4-D-009 accepts corrected logical V3 grouping/roles for current Phase-8 research. R4-D-010 prohibits using v9 for the new full training run and requires a separately versioned V10 call-semantics candidate plus local acceptance. The current future-candidate extractor is V2.5. Its bounded 20-identity structural evidence is closed, and the fresh full 22,540-identity candidate is mechanically built and bound locally, but Stage E rejects physical acceptance because 298 additional full-population drifts lack approved evidence. The hardened V3 research tranche remains durably snapshotted with `coherence=PASS` at commit `44fbb9c1d2033be8002fe404d650cf09f08b0f29`. None of these facts establishes general model discrimination, calibration, thresholds, or production acceptance.

| Index | Class | DATA vNext supervision | Current positive authority | Confirmed-negative support | Discrimination | Calibration / threshold | Current claim status | Key limitation |
|---:|---|---|---|---|---|---|---|---|
| 0 | CallToUnknown | ENABLED | STRONG: SolidiFI Unchecked-Send; SmartBugs unchecked_low_level_calls | NONE — candidate #1 `NOT_CONFIRMED`; candidate #2 primary supports a negative but independent verification is pending | UNSUPPORTED | UNSUPPORTED_EMPTY | TRAINING_ONLY | No accepted negative; v9 call edges are semantically inadequate and the required V10 V2.5 physical lineage is not yet accepted |
| 1 | DenialOfService | ENABLED | STRONG: SmartBugs denial_of_service | NONE — V3 pilot review ready | UNSUPPORTED | UNSUPPORTED_EMPTY | TRAINING_ONLY | DIVE DoS is masked; no confirmed negatives yet |
| 2 | ExternalBug | ENABLED | STRONG: SolidiFI tx.origin; SmartBugs access_control | NONE — V3 pilot review ready | UNSUPPORTED | UNSUPPORTED_EMPTY | TRAINING_ONLY | DIVE Access Control is masked; class scope remains broader than any one source category |
| 3 | GasException | SUPERVISION_DISABLED_PENDING_EVIDENCE | none | NONE | UNSUPPORTED | UNSUPPORTED_EMPTY | DISABLED_PENDING_EVIDENCE | No active approved class-specific positive authority |
| 4 | IntegerUO | ENABLED | STRONG: SolidiFI Overflow-Underflow; SmartBugs arithmetic | NONE — V3 pilot review ready | UNSUPPORTED | UNSUPPORTED_EMPTY | TRAINING_ONLY | DIVE Arithmetic is masked; no confirmed negatives yet |
| 5 | MishandledException | ENABLED | STRONG: SolidiFI Unhandled-Exceptions | NONE — V3 pilot review ready | UNSUPPORTED | UNSUPPORTED_EMPTY | TRAINING_ONLY | Positive evidence only; no discrimination-grade negative population |
| 6 | Reentrancy | ENABLED | STRONG: SolidiFI Re-entrancy; SmartBugs reentrancy | NONE — V3 pilot review ready | UNSUPPORTED | UNSUPPORTED_EMPTY | TRAINING_ONLY | DIVE Reentrancy is masked; no confirmed negatives yet |
| 7 | Timestamp | ENABLED | STRONG: SolidiFI Timestamp-Dependency; SmartBugs time_manipulation | NONE — V3 pilot review ready | UNSUPPORTED | UNSUPPORTED_EMPTY | TRAINING_ONLY | SmartBugs bad_randomness→Timestamp is superseded/no-target; DIVE time manipulation is masked |
| 8 | TransactionOrderDependence | ENABLED | STRONG: SolidiFI TOD; SmartBugs front_running; WEAK training-only: DIVE Front Running | NONE — V3 pilot review ready | UNSUPPORTED | UNSUPPORTED_EMPTY | TRAINING_ONLY | DIVE contribution is weak and barred from outcome metrics/model selection; no confirmed negatives yet |
| 9 | UnusedReturn | SUPERVISION_DISABLED_PENDING_EVIDENCE | none | NONE | UNSUPPORTED | UNSUPPORTED_EMPTY | DISABLED_PENDING_EVIDENCE | DIVE Unchecked Return Values failed the Phase-4 authority threshold and no other active source directly supports the class |

## Global current claim boundary

Physical DATA:

- repaired-v2 physical DATA: **ACCEPTED_IMMUTABLE_REPRODUCIBILITY_EVIDENCE** under R4-D-008 / ADR-R4-008;
- contracts / contract×class rows / physical files: 22,540 / 225,400 / 67,620;
- physical binding digest: `16dd4a3f98c34e52e5c411b39268361881efede07e8f3f52d0c060dd1c5bb6dd`;
- graph schema v9 is **NOT ELIGIBLE FOR A NEW FULL TRAINING RUN** under R4-D-010;
- V10 `v2.3-r4-call-semantics` is preserved as the frozen structural-reference diagnostic lineage;
- the protected V2.4 compatibility candidate completed the former 26-contract parse-only remediation and binds all 22,540 identities with exact accepted-V9 token bytes, zero parse-only artifacts, zero unclassified call IR, and the required 22,539 primary + 1 identity-bound runtime split; it remains diagnostic history, not physical acceptance;
- the current future-candidate extractor is `v2.5-r4-call-semantics-deterministic-cfg`. Three fresh bounded generations closed all 20 previously unexpected structural-drift identities: 8 exact node-index-invariant labelled graph-equivalence identities and 12 deterministic persistent-storage `CFG_NODE_WRITE` corrections, with `zero_unexplained_drift = true` and no blockers;
- protected-local Stages A-D pass for 22,540 identities, exact accepted-V9 token bytes, zero missing/extra/invalid artifacts, and the exact 22,539 ordinary primary + one declared runtime-exception split; binding digest `17c5f334c75015fdaf89b1a9f77522af5185f2485c24df4e1e64917dc944f021`;
- Stage E returns `PASS_BASE_MECHANICS_WITH_STRUCTURAL_EVIDENCE_BLOCKER`: 311 raw non-parse-only drifts, 298 outside the approved old bounded classes, `physical_acceptance=false`, and `training_authorized=false`;
- physical V10 acceptance remains false pending duplicate-safe repeated full-population evidence, a new versioned audit with zero unexplained drift, explicit review, and a separate physical-acceptance decision.

Logical authority:

- logical V3 grouping/roles/publication: **ACCEPTED** under R4-D-009 / ADR-R4-009;
- V3 groups: 22,394; max group size: 7; address-authority edges: 0;
- target/strength semantic counts unchanged from repaired-v2.

Current V3 supervised/evaluation surface:

- effective loss cells: **932**, every authorized target `1`;
- active `MODEL_SELECTION`: **71 contracts / 71 groups**;
- active `INTERNAL_AUDIT`: **72 contracts / 71 groups**;
- combined outcome-metric population: **143 contracts / 142 unique groups**;
- total outcome-metric cells recorded by the V3 publication: **143**; this must not be mislabeled as MODEL_SELECTION-only evidence;
- confirmed-negative targets: **0**;
- `THRESHOLD_FIT = UNSUPPORTED_EMPTY`;
- `CALIBRATION_FIT = UNSUPPORTED_EMPTY`;
- `UNTOUCHED_ACCEPTANCE = UNSUPPORTED_EMPTY_FROZEN`;
- no repaired/V3 full training checkpoint exists;
- no class is currently validated for general vulnerability discrimination, specificity/FPR, calibration, or production outcome claims.

## Hardened V3 negative-evidence pilot state

R4-GAP-007 is in progress and confirmed-negative support remains `NONE` for every enabled class. Candidate #1 was fully reviewed as `NOT_CONFIRMED`. Candidate #2's source-first primary review supports `CONFIRMED_NEGATIVE`, but its authoritative queue truth remains UNKNOWN / PENDING_REVIEW / target `None` until a genuinely independent reviewer agrees.

The committed hardened pilot queue contains:

- 200 candidate cells;
- 25 candidates for each of the eight enabled classes;
- 200 globally unique reserved V3 leakage groups;
- `group_uniqueness_scope=GLOBAL_ACROSS_ENABLED_CLASSES`;
- all candidates `PENDING_REVIEW`;
- all current targets `None`;
- all source roles `TRAIN_UNLABELED`;
- `negative_truth_claim=false`;
- manifest/source lineage validated by the final coherence snapshot.

Queue membership or one primary opinion is not confirmed-negative support. A class remains `NONE` in the matrix until explicit class-specific adjudication and genuinely independent verification accept negative evidence. Any accepted negative is initially `EVALUATION_ONLY_NOT_TRAINING_AUTHORITY`.

Candidate #2 also exposed the representation-quality defect that led to R4-GAP-008 and R4-D-010. The population audit found at least 11,702 provable same-file declared-library calls among 217,490 v9 type-11 edges, while type 11 was attached to only 7,057 / 13,413 raw-low-level nodes, 40 / 4,215 send nodes, and 6,557 / 80,927 transfer nodes. These are historical representation diagnostics, not label truth. The remediation semantics are now bounded and proven; physical V2.5 acceptance remains pending.

## Selector/model-execution evidence boundary

`target_aware_guarded_v1` has completed the hardened corrected-V3 CPU coverage comparison and bounded identical-initialization CUDA comparison with all four required worst-case probes.

Durable CPU result:

- 1,018 records analyzed;
- 737 over-cap;
- 476 guarded improvements;
- 261 equal;
- 0 regressions;
- 0 failures.

Durable CUDA result:

- status `LOGICAL_V3_BOUNDED_RESEARCH_COMPLETE`;
- identical initialization true;
- 4/4 required worst-case probes completed;
- no Run12 weights;
- no checkpoint;
- selector promotion false;
- full-training authorization false.

This supports a separate selector-promotion decision, but **the selector is not yet promoted** and does not improve the model-quality claim boundary by itself. The positive-only CUDA result cannot establish false-positive discrimination.

Before selector promotion, the project still requires full-population verification that the historical control selector reproduces the currently bound representation token tensors exactly for the relevant population.

## Durable current evidence

Final coherent V3 snapshot:

`docs/plan/ml-R4/evidence/2026-08-15_phase8_logical_v3/`

Current V10 restart checkpoint:

`runs/2026-08-27_PHASE8_v10_v25_current_restart_checkpoint.md`.

Bounded V2.5 structural closure:

`reviews/R4-GAP-008/2026-08-26_v10_v25_bounded_structural_closure.md`.

Current full-candidate staging protocol:

`runs/2026-08-26_PHASE8_v10_v25_full_candidate_staging.md`.

Current candidate-review record for the separate R4-GAP-007 track:

`runs/2026-08-21_PHASE8_gap007_candidate2_primary_review.md`.

The accepted pre-pilot baseline remains `runs/2026-08-16_PHASE8_v3_hardened_evidence_snapshot_closeout.md`. The August 21 V10 implementation records and August 23 parse-only/structural-drift handoffs are historical execution context rather than the current restart boundary.

## Status vocabulary

- `VALIDATED_FOR_DEFINED_USE`
- `PROVISIONAL`
- `TRAINING_ONLY`
- `UNSUPPORTED_FOR_OUTCOME_CLAIMS`
- `DISABLED_PENDING_EVIDENCE`
