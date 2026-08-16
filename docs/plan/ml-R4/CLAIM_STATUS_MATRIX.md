# R4 Claim Status Matrix

This matrix describes the **current evidence-qualified claim boundary**, not the historical binary-label surface. R4-D-008 accepts repaired-v2 physical DATA; R4-D-009 accepts corrected logical V3 grouping/roles for current Phase-8 research. Neither decision establishes general model discrimination, calibration, thresholds, or production acceptance.

| Index | Class | DATA vNext supervision | Current positive authority | Confirmed-negative support | Discrimination | Calibration / threshold | Current claim status | Key limitation |
|---:|---|---|---|---|---|---|---|---|
| 0 | CallToUnknown | ENABLED | STRONG: SolidiFI Unchecked-Send; SmartBugs unchecked_low_level_calls | NONE — V3 pilot review pending | UNSUPPORTED | UNSUPPORTED_EMPTY | TRAINING_ONLY | Positive evidence only; false-positive behavior not yet identifiable |
| 1 | DenialOfService | ENABLED | STRONG: SmartBugs denial_of_service | NONE — V3 pilot review pending | UNSUPPORTED | UNSUPPORTED_EMPTY | TRAINING_ONLY | DIVE DoS is masked; no confirmed negatives yet |
| 2 | ExternalBug | ENABLED | STRONG: SolidiFI tx.origin; SmartBugs access_control | NONE — V3 pilot review pending | UNSUPPORTED | UNSUPPORTED_EMPTY | TRAINING_ONLY | DIVE Access Control is masked; class scope remains broader than any one source category |
| 3 | GasException | SUPERVISION_DISABLED_PENDING_EVIDENCE | none | NONE | UNSUPPORTED | UNSUPPORTED_EMPTY | DISABLED_PENDING_EVIDENCE | No active approved class-specific positive authority |
| 4 | IntegerUO | ENABLED | STRONG: SolidiFI Overflow-Underflow; SmartBugs arithmetic | NONE — V3 pilot review pending | UNSUPPORTED | UNSUPPORTED_EMPTY | TRAINING_ONLY | DIVE Arithmetic is masked; no confirmed negatives yet |
| 5 | MishandledException | ENABLED | STRONG: SolidiFI Unhandled-Exceptions | NONE — V3 pilot review pending | UNSUPPORTED | UNSUPPORTED_EMPTY | TRAINING_ONLY | Positive evidence only; no discrimination-grade negative population |
| 6 | Reentrancy | ENABLED | STRONG: SolidiFI Re-entrancy; SmartBugs reentrancy | NONE — V3 pilot review pending | UNSUPPORTED | UNSUPPORTED_EMPTY | TRAINING_ONLY | DIVE Reentrancy is masked; no confirmed negatives yet |
| 7 | Timestamp | ENABLED | STRONG: SolidiFI Timestamp-Dependency; SmartBugs time_manipulation | NONE — V3 pilot review pending | UNSUPPORTED | UNSUPPORTED_EMPTY | TRAINING_ONLY | SmartBugs bad_randomness→Timestamp is superseded/no-target; DIVE time manipulation is masked |
| 8 | TransactionOrderDependence | ENABLED | STRONG: SolidiFI TOD; SmartBugs front_running; WEAK training-only: DIVE Front Running | NONE — V3 pilot review pending | UNSUPPORTED | UNSUPPORTED_EMPTY | TRAINING_ONLY | DIVE contribution is weak and barred from outcome metrics/model selection; no confirmed negatives yet |
| 9 | UnusedReturn | SUPERVISION_DISABLED_PENDING_EVIDENCE | none | NONE | UNSUPPORTED | UNSUPPORTED_EMPTY | DISABLED_PENDING_EVIDENCE | DIVE Unchecked Return Values failed the Phase-4 authority threshold and no other active source directly supports the class |

## Global current claim boundary

Physical DATA:

- repaired-v2 physical DATA: **ACCEPTED_FOR_REUSE/BOUNDED_RESEARCH** under R4-D-008 / ADR-R4-008;
- contracts / contract×class rows / physical files: 22,540 / 225,400 / 67,620;
- physical binding digest: `16dd4a3f98c34e52e5c411b39268361881efede07e8f3f52d0c060dd1c5bb6dd`.

Logical authority:

- logical V3 grouping/roles/publication: **ACCEPTED** under R4-D-009 / ADR-R4-009;
- V3 groups: 22,394; max group size: 7; address-authority edges: 0;
- target/strength semantic counts unchanged from repaired-v2.

Current V3 supervised/evaluation surface:

- effective loss cells: **932**, every authorized target `1`;
- model-selection outcome cells: **143**, positive-only;
- confirmed-negative targets: **0**;
- `THRESHOLD_FIT = UNSUPPORTED_EMPTY`;
- `CALIBRATION_FIT = UNSUPPORTED_EMPTY`;
- `UNTOUCHED_ACCEPTANCE = UNSUPPORTED_EMPTY_FROZEN`;
- no repaired/V3 full training checkpoint exists;
- no class is currently validated for general vulnerability discrimination, specificity/FPR, calibration, or production outcome claims.

## V3 negative-evidence pilot state

R4-GAP-007 is APPROVED but adjudication has not started.

The generated pilot queue contains:

- 200 candidate cells;
- 25 candidates for each of the eight enabled classes;
- 200 reserved V3 leakage groups;
- all candidates `PENDING_REVIEW`;
- all current targets `None`;
- all source roles `TRAIN_UNLABELED`;
- `negative_truth_claim=false`.

Queue membership is not confirmed-negative support. A class remains `NONE` in the matrix until explicit class-specific adjudication and independent verification accept negative evidence.

## Selector/model-execution evidence boundary

`target_aware_guarded_v1` has completed corrected-V3 CPU coverage comparison and bounded identical-initialization CUDA comparison with all four required worst-case probes. This supports a separate selector-promotion decision, but **the selector is not yet promoted** and does not improve the model-quality claim boundary by itself.

The positive-only CUDA result cannot establish false-positive discrimination.

Detailed current checkpoint:

`runs/2026-08-16_PHASE8_logical_v3_acceptance_and_research_checkpoint.md`.

## Status vocabulary

- `VALIDATED_FOR_DEFINED_USE`
- `PROVISIONAL`
- `TRAINING_ONLY`
- `UNSUPPORTED_FOR_OUTCOME_CLAIMS`
- `DISABLED_PENDING_EVIDENCE`
