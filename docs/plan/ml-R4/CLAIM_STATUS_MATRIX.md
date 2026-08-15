# R4 Claim Status Matrix

This matrix describes the **current evidence-qualified claim boundary**, not the historical binary-label surface. R4-D-008 accepts repaired-v2 DATA for bounded research; it does not establish model discrimination, calibration, thresholds, or production acceptance.

| Index | Class | DATA vNext supervision | Current positive authority | Confirmed-negative support | Discrimination | Calibration / threshold | Current claim status | Key limitation |
|---:|---|---|---|---|---|---|---|---|
| 0 | CallToUnknown | ENABLED | STRONG: SolidiFI Unchecked-Send; SmartBugs unchecked_low_level_calls | NONE | UNSUPPORTED | UNSUPPORTED_EMPTY | TRAINING_ONLY | Positive evidence only; false-positive behavior not identifiable |
| 1 | DenialOfService | ENABLED | STRONG: SmartBugs denial_of_service | NONE | UNSUPPORTED | UNSUPPORTED_EMPTY | TRAINING_ONLY | DIVE DoS is masked; no confirmed negatives |
| 2 | ExternalBug | ENABLED | STRONG: SolidiFI tx.origin; SmartBugs access_control | NONE | UNSUPPORTED | UNSUPPORTED_EMPTY | TRAINING_ONLY | DIVE Access Control is masked; class scope remains broader than any one source category |
| 3 | GasException | SUPERVISION_DISABLED_PENDING_EVIDENCE | none | NONE | UNSUPPORTED | UNSUPPORTED_EMPTY | DISABLED_PENDING_EVIDENCE | No active approved class-specific positive authority |
| 4 | IntegerUO | ENABLED | STRONG: SolidiFI Overflow-Underflow; SmartBugs arithmetic | NONE | UNSUPPORTED | UNSUPPORTED_EMPTY | TRAINING_ONLY | DIVE Arithmetic is masked; no confirmed negatives |
| 5 | MishandledException | ENABLED | STRONG: SolidiFI Unhandled-Exceptions | NONE | UNSUPPORTED | UNSUPPORTED_EMPTY | TRAINING_ONLY | Positive evidence only; no discrimination-grade negative population |
| 6 | Reentrancy | ENABLED | STRONG: SolidiFI Re-entrancy; SmartBugs reentrancy | NONE | UNSUPPORTED | UNSUPPORTED_EMPTY | TRAINING_ONLY | DIVE Reentrancy is masked; no confirmed negatives |
| 7 | Timestamp | ENABLED | STRONG: SolidiFI Timestamp-Dependency; SmartBugs time_manipulation | NONE | UNSUPPORTED | UNSUPPORTED_EMPTY | TRAINING_ONLY | SmartBugs bad_randomness→Timestamp is superseded/no-target; DIVE time manipulation is masked |
| 8 | TransactionOrderDependence | ENABLED | STRONG: SolidiFI TOD; SmartBugs front_running; WEAK training-only: DIVE Front Running | NONE | UNSUPPORTED | UNSUPPORTED_EMPTY | TRAINING_ONLY | DIVE contribution is weak and barred from outcome metrics/model selection; no confirmed negatives |
| 9 | UnusedReturn | SUPERVISION_DISABLED_PENDING_EVIDENCE | none | NONE | UNSUPPORTED | UNSUPPORTED_EMPTY | DISABLED_PENDING_EVIDENCE | DIVE Unchecked Return Values failed the Phase-4 authority threshold and no other active source directly supports the class |

## Global repaired-v2 claim boundary

- repaired-v2 physical DATA: **ACCEPTED_FOR_BOUNDED_RESEARCH** under R4-D-008 / ADR-R4-008;
- effective training cells: **899**, every target `1`;
- confirmed-negative targets: **0**;
- MODEL_SELECTION: positive-only limited;
- `THRESHOLD_FIT = UNSUPPORTED_EMPTY`;
- `CALIBRATION_FIT = UNSUPPORTED_EMPTY`;
- `UNTOUCHED_ACCEPTANCE = UNSUPPORTED_EMPTY_FROZEN`;
- no repaired-v2 full training checkpoint exists;
- no class is currently validated for general vulnerability discrimination or production outcome claims.

## Status vocabulary

- `VALIDATED_FOR_DEFINED_USE`
- `PROVISIONAL`
- `TRAINING_ONLY`
- `UNSUPPORTED_FOR_OUTCOME_CLAIMS`
- `DISABLED_PENDING_EVIDENCE`
