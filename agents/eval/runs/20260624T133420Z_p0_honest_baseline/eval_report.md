# SENTINEL Agents Pipeline Evaluation Report

**Contracts evaluated:** 83  

**Macro-F1:** 0.1958  |  **Macro-Fbeta:** 0.2515  |  **Micro-F1:** 0.2724

## Gate assertions

| Gate | Description | Passed | Detail |
|---|---|---|---|
| `WS1a_silent_safe_on_flagged` | No consensus-flagged class ends SAFE (debate cannot clear consensus verdict) | PASS | 0 violation(s) |
| `WS1b_inconclusive_on_timeout` | edge_debate_timeout emits INCONCLUSIVE (LLM-on mode only) | FAIL | edge_debate_timeout present=False, INCONCLUSIVE emitted=False |
| `WS1c_no_missing_consensus_votes` | No consensus vote is missing from final verdicts (Finding #15) | PASS | 0 missing vote(s) |
| `WS1d_confidence_1_0_not_downgraded` | No consensus confidence=1.0 class ends in SAFE (Finding #14 worst case) | PASS | 0 violation(s) |
| `WS1e_no_vulnerable_label_with_safe_verdict` | No contract has overall_label=vulnerable + overall_verdict=SAFE (Finding #19) | PASS | 0 violation(s) |
| `WS2_false_positives_on_safe` | Zero false-positive verdicts on the safe subset | FAIL | 5 FP(s) on 5 safe contract(s): 01_checks_effects_interactions: ['UnusedReturn']; 02_pull_over_push_payment: ['ExternalBug', 'Reentrancy']; 03_openzeppelin_managed: ['CallToUnknown', 'ExternalBug', 'IntegerUO', 'Reentrancy']; 04_pausable_circuit_breaker: ['CallToUnknown', 'ExternalBug']; 05_emergency_stop_controlled: ['ExternalBug', 'Timestamp'] |
| `WS3_long_contract_bug_detected` | edge_long_contract_truncated Reentrancy flagged (bug is past 2000-char cutoff) | FAIL | edge_long_contract_truncated not in corpus |
| `D4_eye_predictions_present` | eye_predictions field present in all ml_results | FAIL | eye_predictions present in 61/83 reports |
| `macro_f1_vs_baseline` | macro_F1 >= baseline (no baseline given — informational) | PASS | macro_F1 = 0.1958 (no baseline to compare) |

**Overall: GATE FAILURE**

## Per-class metrics

| Class | Support | TP | FP | FN | TN | Precision | Recall | F1 | F-beta |
|---|---|---|---|---|---|---|---|---|---|
| MishandledException | 19 | 0 | 0 | 19 | 64 | nan | 0.0000 | 0.0000 | 0.0000 |
| CallToUnknown | 17 | 1 | 3 | 16 | 63 | 0.2500 | 0.0588 | 0.0952 | 0.0694 |
| Reentrancy | 15 | 11 | 31 | 4 | 37 | 0.2619 | 0.7333 | 0.3860 | 0.5392 |
| ExternalBug | 13 | 11 | 38 | 2 | 32 | 0.2245 | 0.8462 | 0.3548 | 0.5446 |
| Timestamp | 12 | 8 | 12 | 4 | 59 | 0.4000 | 0.6667 | 0.5000 | 0.5882 |
| DenialOfService | 10 | 0 | 4 | 10 | 69 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| UnusedReturn | 10 | 2 | 15 | 8 | 58 | 0.1176 | 0.2000 | 0.1481 | 0.1754 |
| GasException | 9 | 1 | 1 | 8 | 73 | 0.5000 | 0.1111 | 0.1818 | 0.1316 |
| IntegerUO | 9 | 7 | 32 | 2 | 42 | 0.1795 | 0.7778 | 0.2917 | 0.4667 |
| TransactionOrderDependence | 9 | 0 | 1 | 9 | 73 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Contract-level accuracy

- **Loose**: 38/83 = 45.78%
- **Exact**: 1/83 = 1.20%

