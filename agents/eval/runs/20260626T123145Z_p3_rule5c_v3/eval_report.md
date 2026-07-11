# SENTINEL Agents Pipeline Evaluation Report

**Contracts evaluated:** 83  

**Macro-F1:** 0.3008  |  **Macro-Fbeta:** 0.3821  |  **Micro-F1:** 0.3299

## Gate assertions

| Gate | Description | Passed | Detail |
|---|---|---|---|
| `WS1a_silent_safe_on_flagged` | No consensus-flagged class ends SAFE (debate cannot clear consensus verdict) | FAIL | 71 violation(s): 01_approve_frontrun/CallToUnknown (consensus=DISPUTED conf=0.38 → final=SAFE); 01_approve_frontrun/IntegerUO (consensus=DISPUTED conf=0.00 → final=SAFE); 01_approve_frontrun/TransactionOrderDependence (consensus=DISPUTED conf=0.32 → final=SAFE); 01_bccc_reentrancy_injected_erc20/DenialOfService (consensus=DISPUTED conf=0.00 → final=SAFE); 01_cei_violation_erc721/TransactionOrderDependence (consensus=DISPUTED conf=0.32 → final=SAFE) |
| `WS1b_inconclusive_on_timeout` | edge_debate_timeout emits INCONCLUSIVE (LLM-on mode only) | FAIL | edge_debate_timeout present=False, INCONCLUSIVE emitted=False |
| `WS1c_no_missing_consensus_votes` | No consensus vote is missing from final verdicts (Finding #15) | PASS | 0 missing vote(s) |
| `WS1d_confidence_1_0_not_downgraded` | No consensus confidence=1.0 class ends in SAFE (Finding #14 worst case) | PASS | 0 violation(s) |
| `WS1e_no_vulnerable_label_with_safe_verdict` | No contract has overall_label=vulnerable + overall_verdict=SAFE (Finding #19) | FAIL | 1 violation(s): 08_bccc_unusedreturn_injected_batch (label=confirmed_vulnerable, verdict=SAFE) |
| `WS2_false_positives_on_safe` | Zero false-positive verdicts on the safe subset | FAIL | 5 FP(s) on 5 safe contract(s): 01_checks_effects_interactions: ['CallToUnknown', 'ExternalBug', 'IntegerUO', 'MishandledException']; 02_pull_over_push_payment: ['CallToUnknown', 'ExternalBug', 'IntegerUO', 'MishandledException', 'Reentrancy']; 03_openzeppelin_managed: ['ExternalBug', 'IntegerUO', 'MishandledException']; 04_pausable_circuit_breaker: ['ExternalBug', 'IntegerUO', 'MishandledException', 'Reentrancy']; 05_emergency_stop_controlled: ['ExternalBug', 'IntegerUO', 'MishandledException', 'Timestamp'] |
| `WS3_long_contract_bug_detected` | edge_long_contract_truncated Reentrancy flagged (bug is past 2000-char cutoff) | FAIL | edge_long_contract_truncated not in corpus |
| `D4_eye_predictions_present` | eye_predictions field present in all ml_results | FAIL | eye_predictions present in 61/83 reports |
| `macro_f1_vs_baseline` | macro_F1 >= baseline (0.2765) | PASS | macro_F1 = 0.3008 (delta +0.0243 vs baseline 0.2765) |

**Overall: GATE FAILURE**

## Per-class metrics

| Class | Support | TP | FP | FN | TN | Precision | Recall | F1 | F-beta |
|---|---|---|---|---|---|---|---|---|---|
| MishandledException | 19 | 6 | 23 | 13 | 41 | 0.2069 | 0.3158 | 0.2500 | 0.2857 |
| CallToUnknown | 17 | 13 | 22 | 4 | 44 | 0.3714 | 0.7647 | 0.5000 | 0.6311 |
| Reentrancy | 15 | 12 | 32 | 3 | 36 | 0.2727 | 0.8000 | 0.4068 | 0.5769 |
| ExternalBug | 13 | 11 | 43 | 2 | 27 | 0.2037 | 0.8462 | 0.3284 | 0.5189 |
| Timestamp | 12 | 8 | 12 | 4 | 59 | 0.4000 | 0.6667 | 0.5000 | 0.5882 |
| DenialOfService | 10 | 1 | 9 | 9 | 64 | 0.1000 | 0.1000 | 0.1000 | 0.1000 |
| UnusedReturn | 10 | 4 | 10 | 6 | 63 | 0.2857 | 0.4000 | 0.3333 | 0.3704 |
| GasException | 9 | 4 | 8 | 5 | 66 | 0.3333 | 0.4444 | 0.3810 | 0.4167 |
| IntegerUO | 9 | 5 | 34 | 4 | 40 | 0.1282 | 0.5556 | 0.2083 | 0.3333 |
| TransactionOrderDependence | 9 | 0 | 8 | 9 | 66 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Contract-level accuracy

- **Loose**: 45/83 = 54.22%
- **Exact**: 1/83 = 1.20%

## Baseline comparison

| Metric | Baseline | Current | Delta |
|---|---|---|---|
| macro_F1 | 0.2765 | 0.3008 | +0.0243 |
| macro_Fbeta | 0.3580 | 0.3821 | +0.0241 |
