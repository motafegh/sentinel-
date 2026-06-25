# SENTINEL Agents Pipeline Evaluation Report

**Contracts evaluated:** 79  

**Macro-F1:** 0.1998  |  **Macro-Fbeta:** 0.2246  |  **Micro-F1:** 0.2661

## Gate assertions

| Gate | Description | Passed | Detail |
|---|---|---|---|
| `WS1a_silent_safe_on_flagged` | No consensus-flagged class ends SAFE (debate cannot clear consensus verdict) | PASS | 0 violation(s) |
| `WS1b_inconclusive_on_timeout` | edge_debate_timeout emits INCONCLUSIVE (LLM-on mode only) | PASS | N/A in --no-llm mode |
| `WS1c_no_missing_consensus_votes` | No consensus vote is missing from final verdicts (Finding #15) | PASS | 0 missing vote(s) |
| `WS1d_confidence_1_0_not_downgraded` | No consensus confidence=1.0 class ends in SAFE (Finding #14 worst case) | PASS | 0 violation(s) |
| `WS1e_no_vulnerable_label_with_safe_verdict` | No contract has overall_label=vulnerable + overall_verdict=SAFE (Finding #19) | PASS | 0 violation(s) |
| `WS2_false_positives_on_safe` | Zero false-positive verdicts on the safe subset | FAIL | 5 FP(s) on 5 safe contract(s): 01_checks_effects_interactions: ['ExternalBug']; 02_pull_over_push_payment: ['ExternalBug', 'Reentrancy']; 03_openzeppelin_managed: ['ExternalBug']; 04_pausable_circuit_breaker: ['ExternalBug']; 05_emergency_stop_controlled: ['ExternalBug', 'Timestamp'] |
| `WS3_long_contract_bug_detected` | edge_long_contract_truncated Reentrancy flagged (bug is past 2000-char cutoff) | FAIL | edge_long_contract_truncated not in corpus |
| `D4_eye_predictions_present` | eye_predictions field present in all ml_results | FAIL | eye_predictions present in 57/79 reports |
| `macro_f1_vs_baseline` | macro_F1 >= baseline (0.1958) | PASS | macro_F1 = 0.1998 (delta +0.0041 vs baseline 0.1958) |

**Overall: GATE FAILURE**

## Per-class metrics

| Class | Support | TP | FP | FN | TN | Precision | Recall | F1 | F-beta |
|---|---|---|---|---|---|---|---|---|---|
| MishandledException | 19 | 0 | 4 | 19 | 56 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| CallToUnknown | 17 | 1 | 4 | 16 | 58 | 0.2000 | 0.0588 | 0.0909 | 0.0685 |
| Reentrancy | 14 | 11 | 25 | 3 | 40 | 0.3056 | 0.7857 | 0.4400 | 0.5978 |
| ExternalBug | 11 | 8 | 30 | 3 | 38 | 0.2105 | 0.7273 | 0.3265 | 0.4878 |
| Timestamp | 11 | 7 | 12 | 4 | 56 | 0.3684 | 0.6364 | 0.4667 | 0.5556 |
| DenialOfService | 10 | 0 | 1 | 10 | 68 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| GasException | 9 | 2 | 2 | 7 | 68 | 0.5000 | 0.2222 | 0.3077 | 0.2500 |
| IntegerUO | 8 | 0 | 5 | 8 | 66 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| TransactionOrderDependence | 8 | 1 | 3 | 7 | 68 | 0.2500 | 0.1250 | 0.1667 | 0.1389 |
| UnusedReturn | 8 | 1 | 1 | 7 | 70 | 0.5000 | 0.1250 | 0.2000 | 0.1471 |

## Contract-level accuracy

- **Loose**: 29/79 = 36.71%
- **Exact**: 0/79 = 0.00%

## Baseline comparison

| Metric | Baseline | Current | Delta |
|---|---|---|---|
| macro_F1 | 0.1958 | 0.1998 | +0.0041 |
| macro_Fbeta | 0.2515 | 0.2246 | -0.0270 |
