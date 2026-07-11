# SENTINEL Agents Pipeline Evaluation Report

**Contracts evaluated:** 79  

**Macro-F1:** 0.2877  |  **Macro-Fbeta:** 0.3654  |  **Micro-F1:** 0.3240

## Gate assertions

| Gate | Description | Passed | Detail |
|---|---|---|---|
| `WS1a_silent_safe_on_flagged` | No consensus-flagged class ends SAFE (debate cannot clear consensus verdict) | FAIL | 67 violation(s): 01_approve_frontrun/IntegerUO (consensus=DISPUTED conf=0.00 → final=SAFE); 01_approve_frontrun/TransactionOrderDependence (consensus=DISPUTED conf=0.32 → final=SAFE); 01_cei_violation_erc721/TransactionOrderDependence (consensus=DISPUTED conf=0.32 → final=SAFE); 01_checks_effects_interactions/DenialOfService (consensus=DISPUTED conf=0.36 → final=SAFE); 01_checks_effects_interactions/GasException (consensus=DISPUTED conf=0.39 → final=SAFE) |
| `WS1b_inconclusive_on_timeout` | edge_debate_timeout emits INCONCLUSIVE (LLM-on mode only) | PASS | N/A in --no-llm mode |
| `WS1c_no_missing_consensus_votes` | No consensus vote is missing from final verdicts (Finding #15) | PASS | 0 missing vote(s) |
| `WS1d_confidence_1_0_not_downgraded` | No consensus confidence=1.0 class ends in SAFE (Finding #14 worst case) | PASS | 0 violation(s) |
| `WS1e_no_vulnerable_label_with_safe_verdict` | No contract has overall_label=vulnerable + overall_verdict=SAFE (Finding #19) | PASS | 0 violation(s) |
| `WS2_false_positives_on_safe` | Zero false-positive verdicts on the safe subset | FAIL | 5 FP(s) on 5 safe contract(s): 01_checks_effects_interactions: ['CallToUnknown', 'ExternalBug', 'IntegerUO', 'MishandledException']; 02_pull_over_push_payment: ['CallToUnknown', 'ExternalBug', 'IntegerUO', 'MishandledException', 'Reentrancy']; 03_openzeppelin_managed: ['ExternalBug', 'IntegerUO', 'MishandledException']; 04_pausable_circuit_breaker: ['ExternalBug', 'IntegerUO', 'MishandledException']; 05_emergency_stop_controlled: ['ExternalBug', 'IntegerUO', 'MishandledException', 'Timestamp'] |
| `WS3_long_contract_bug_detected` | edge_long_contract_truncated Reentrancy flagged (bug is past 2000-char cutoff) | FAIL | edge_long_contract_truncated not in corpus |
| `D4_eye_predictions_present` | eye_predictions field present in all ml_results | FAIL | eye_predictions present in 57/79 reports |
| `macro_f1_vs_baseline` | macro_F1 >= baseline (no baseline given — informational) | PASS | macro_F1 = 0.2877 (no baseline to compare) |

**Overall: GATE FAILURE**

## Per-class metrics

| Class | Support | TP | FP | FN | TN | Precision | Recall | F1 | F-beta |
|---|---|---|---|---|---|---|---|---|---|
| MishandledException | 19 | 6 | 23 | 13 | 37 | 0.2069 | 0.3158 | 0.2500 | 0.2857 |
| CallToUnknown | 17 | 13 | 22 | 4 | 40 | 0.3714 | 0.7647 | 0.5000 | 0.6311 |
| Reentrancy | 14 | 11 | 28 | 3 | 37 | 0.2821 | 0.7857 | 0.4151 | 0.5789 |
| ExternalBug | 11 | 10 | 41 | 1 | 27 | 0.1961 | 0.9091 | 0.3226 | 0.5263 |
| Timestamp | 11 | 7 | 12 | 4 | 56 | 0.3684 | 0.6364 | 0.4667 | 0.5556 |
| DenialOfService | 10 | 1 | 4 | 9 | 65 | 0.2000 | 0.1000 | 0.1333 | 0.1111 |
| GasException | 9 | 2 | 7 | 7 | 63 | 0.2222 | 0.2222 | 0.2222 | 0.2222 |
| IntegerUO | 8 | 4 | 31 | 4 | 40 | 0.1143 | 0.5000 | 0.1860 | 0.2985 |
| TransactionOrderDependence | 8 | 0 | 8 | 8 | 63 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| UnusedReturn | 8 | 4 | 9 | 4 | 62 | 0.3077 | 0.5000 | 0.3810 | 0.4444 |

## Contract-level accuracy

- **Loose**: 39/79 = 49.37%
- **Exact**: 0/79 = 0.00%

