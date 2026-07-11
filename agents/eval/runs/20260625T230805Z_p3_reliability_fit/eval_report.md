# SENTINEL Agents Pipeline Evaluation Report

**Contracts evaluated:** 83  

**Macro-F1:** 0.2765  |  **Macro-Fbeta:** 0.3580  |  **Micro-F1:** 0.3123

## Gate assertions

| Gate | Description | Passed | Detail |
|---|---|---|---|
| `WS1a_silent_safe_on_flagged` | No consensus-flagged class ends SAFE (debate cannot clear consensus verdict) | FAIL | 74 violation(s): 01_approve_frontrun/CallToUnknown (consensus=DISPUTED conf=0.38 → final=SAFE); 01_approve_frontrun/IntegerUO (consensus=DISPUTED conf=0.00 → final=SAFE); 01_approve_frontrun/TransactionOrderDependence (consensus=DISPUTED conf=0.32 → final=SAFE); 01_bccc_reentrancy_injected_erc20/DenialOfService (consensus=DISPUTED conf=0.00 → final=SAFE); 01_cei_violation_erc721/TransactionOrderDependence (consensus=DISPUTED conf=0.32 → final=SAFE) |
| `WS1b_inconclusive_on_timeout` | edge_debate_timeout emits INCONCLUSIVE (LLM-on mode only) | FAIL | edge_debate_timeout present=False, INCONCLUSIVE emitted=False |
| `WS1c_no_missing_consensus_votes` | No consensus vote is missing from final verdicts (Finding #15) | PASS | 0 missing vote(s) |
| `WS1d_confidence_1_0_not_downgraded` | No consensus confidence=1.0 class ends in SAFE (Finding #14 worst case) | PASS | 0 violation(s) |
| `WS1e_no_vulnerable_label_with_safe_verdict` | No contract has overall_label=vulnerable + overall_verdict=SAFE (Finding #19) | FAIL | 1 violation(s): 08_bccc_unusedreturn_injected_batch (label=confirmed_vulnerable, verdict=SAFE) |
| `WS2_false_positives_on_safe` | Zero false-positive verdicts on the safe subset | FAIL | 5 FP(s) on 5 safe contract(s): 01_checks_effects_interactions: ['CallToUnknown', 'ExternalBug', 'IntegerUO', 'MishandledException']; 02_pull_over_push_payment: ['CallToUnknown', 'ExternalBug', 'IntegerUO', 'MishandledException', 'Reentrancy']; 03_openzeppelin_managed: ['ExternalBug', 'IntegerUO', 'MishandledException', 'Reentrancy']; 04_pausable_circuit_breaker: ['ExternalBug', 'IntegerUO', 'MishandledException', 'Reentrancy']; 05_emergency_stop_controlled: ['ExternalBug', 'IntegerUO', 'MishandledException', 'Reentrancy', 'Timestamp'] |
| `WS3_long_contract_bug_detected` | edge_long_contract_truncated Reentrancy flagged (bug is past 2000-char cutoff) | FAIL | edge_long_contract_truncated not in corpus |
| `D4_eye_predictions_present` | eye_predictions field present in all ml_results | FAIL | eye_predictions present in 61/83 reports |
| `macro_f1_vs_baseline` | macro_F1 >= baseline (0.1998) | PASS | macro_F1 = 0.2765 (delta +0.0767 vs baseline 0.1998) |

**Overall: GATE FAILURE**

## Per-class metrics

| Class | Support | TP | FP | FN | TN | Precision | Recall | F1 | F-beta |
|---|---|---|---|---|---|---|---|---|---|
| MishandledException | 19 | 6 | 23 | 13 | 41 | 0.2069 | 0.3158 | 0.2500 | 0.2857 |
| CallToUnknown | 17 | 13 | 22 | 4 | 44 | 0.3714 | 0.7647 | 0.5000 | 0.6311 |
| Reentrancy | 15 | 12 | 37 | 3 | 31 | 0.2449 | 0.8000 | 0.3750 | 0.5505 |
| ExternalBug | 13 | 10 | 43 | 3 | 27 | 0.1887 | 0.7692 | 0.3030 | 0.4762 |
| Timestamp | 12 | 8 | 14 | 4 | 57 | 0.3636 | 0.6667 | 0.4706 | 0.5714 |
| DenialOfService | 10 | 1 | 7 | 9 | 66 | 0.1250 | 0.1000 | 0.1111 | 0.1042 |
| UnusedReturn | 10 | 4 | 11 | 6 | 62 | 0.2667 | 0.4000 | 0.3200 | 0.3636 |
| GasException | 9 | 2 | 9 | 7 | 65 | 0.1818 | 0.2222 | 0.2000 | 0.2128 |
| IntegerUO | 9 | 6 | 36 | 3 | 38 | 0.1429 | 0.6667 | 0.2353 | 0.3846 |
| TransactionOrderDependence | 9 | 0 | 10 | 9 | 64 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Contract-level accuracy

- **Loose**: 43/83 = 51.81%
- **Exact**: 1/83 = 1.20%

## Baseline comparison

| Metric | Baseline | Current | Delta |
|---|---|---|---|
| macro_F1 | 0.1998 | 0.2765 | +0.0767 |
| macro_Fbeta | 0.2246 | 0.3580 | +0.1334 |
