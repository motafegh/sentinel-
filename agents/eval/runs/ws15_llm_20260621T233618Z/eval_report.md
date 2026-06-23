# SENTINEL Agents Pipeline Evaluation Report

**Contracts evaluated:** 16  
**Positive verdict set:** ['CONFIRMED', 'LIKELY']  
**Macro-F1:** 0.3756  |  **Micro-F1:** 0.3939

## Gate assertions

| Gate | Description | Passed | Detail |
|---|---|---|---|
| `WS1a_silent_safe_on_flagged` | No consensus-flagged class ends SAFE (debate cannot clear consensus verdict) | PASS | 0 violation(s) |
| `WS1b_inconclusive_on_timeout` | edge_debate_timeout emits INCONCLUSIVE (LLM-on mode only) | FAIL | edge_debate_timeout present=False, INCONCLUSIVE emitted=False |
| `WS1c_no_missing_consensus_votes` | No consensus vote is missing from final verdicts (Finding #15) | PASS | 0 missing vote(s) |
| `WS1d_confidence_1_0_not_downgraded` | No consensus confidence=1.0 class ends in SAFE (Finding #14 worst case) | PASS | 0 violation(s) |
| `WS1e_no_vulnerable_label_with_safe_verdict` | No contract has overall_label=vulnerable + overall_verdict=SAFE (Finding #19) | PASS | 0 violation(s) |
| `WS2_false_positives_on_safe` | Zero false-positive verdicts on the safe subset | PASS | 0 FP(s) on 0 safe contract(s) |
| `WS3_long_contract_bug_detected` | edge_long_contract_truncated Reentrancy flagged (bug is past 2000-char cutoff) | FAIL | edge_long_contract_truncated not in corpus |
| `macro_f1_vs_baseline` | macro_F1 >= baseline (0.2455) | PASS | macro_F1 = 0.3756 (delta +0.1302 vs baseline 0.2455) |

**Overall: GATE FAILURE**

## Per-class metrics

| Class | Support | TP | FP | FN | TN | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|---|
| CallToUnknown | 7 | 7 | 4 | 0 | 5 | 0.6364 | 1.0000 | 0.7778 |
| DenialOfService | 7 | 3 | 0 | 4 | 9 | 1.0000 | 0.4286 | 0.6000 |
| ExternalBug | 2 | 1 | 10 | 1 | 4 | 0.0909 | 0.5000 | 0.1538 |
| GasException | 1 | 0 | 2 | 1 | 13 | 0.0000 | 0.0000 | 0.0000 |
| Reentrancy | 1 | 1 | 7 | 0 | 8 | 0.1250 | 1.0000 | 0.2222 |
| Timestamp | 1 | 1 | 2 | 0 | 13 | 0.3333 | 1.0000 | 0.5000 |
| IntegerUO | 0 | 0 | 1 | 0 | 15 | 0.0000 | nan | 0.0000 |
| MishandledException | 0 | 0 | 1 | 0 | 15 | 0.0000 | nan | 0.0000 |
| TransactionOrderDependence | 0 | 0 | 5 | 0 | 11 | 0.0000 | nan | 0.0000 |
| UnusedReturn | 0 | 0 | 2 | 0 | 14 | 0.0000 | nan | 0.0000 |

## Contract-level accuracy

- **Loose** (safe→no flag OR vuln→≥1 correct flag): 11/16 = 68.75%
- **Strict exact-match** (predicted set == label set): 0/16 = 0.00%

## Per-contract detail

| Contract | GT | Labels | Predicted | TP | FP | FN | Correct |
|---|---|---|---|---|---|---|---|
| 01_flash_loan_oracle_manipulation | vulnerable | ExternalBug | Reentrancy | — | Reentrancy | ExternalBug | ✗ |
| 01_proxy_delegatecall | vulnerable | CallToUnknown | CallToUnknown,ExternalBug,TransactionOrderDependence | CallToUnknown | ExternalBug,TransactionOrderDependence | — | ✓ |
| 01_unbounded_refund | vulnerable | DenialOfService | CallToUnknown,Reentrancy | — | CallToUnknown,Reentrancy | DenialOfService | ✗ |
| 02_access_control_bypass | vulnerable | ExternalBug | ExternalBug,TransactionOrderDependence | ExternalBug | TransactionOrderDependence | — | ✓ |
| 02_dynamic_dispatch | vulnerable | CallToUnknown | CallToUnknown,ExternalBug | CallToUnknown | ExternalBug | — | ✓ |
| 02_push_payment_failure | vulnerable | DenialOfService | CallToUnknown,DenialOfService,ExternalBug,GasException,Reentrancy | DenialOfService | CallToUnknown,ExternalBug,GasException,Reentrancy | — | ✓ |
| 03_dynamic_loop_gas_bomb | vulnerable | DenialOfService | CallToUnknown,DenialOfService,ExternalBug,GasException,Reentrancy | DenialOfService | CallToUnknown,ExternalBug,GasException,Reentrancy | — | ✓ |
| 03_low_level_forwarder | vulnerable | CallToUnknown | CallToUnknown,ExternalBug,Reentrancy,TransactionOrderDependence | CallToUnknown | ExternalBug,Reentrancy,TransactionOrderDependence | — | ✓ |
| 04_opaque_contract_factory | vulnerable | CallToUnknown | CallToUnknown,ExternalBug,TransactionOrderDependence,UnusedReturn | CallToUnknown | ExternalBug,TransactionOrderDependence,UnusedReturn | — | ✓ |
| 04_storage_growth_dos | vulnerable | DenialOfService | CallToUnknown,DenialOfService,Timestamp | DenialOfService | CallToUnknown,Timestamp | — | ✓ |
| 05_unexpected_revert_dos | vulnerable | DenialOfService | ExternalBug,Reentrancy,UnusedReturn | — | ExternalBug,Reentrancy,UnusedReturn | DenialOfService | ✗ |
| 05_upgrade_proxy_selfdestruct | vulnerable | CallToUnknown | CallToUnknown,ExternalBug,Reentrancy,Timestamp | CallToUnknown | ExternalBug,Reentrancy,Timestamp | — | ✓ |
| 06_tricky_call_in_fallback | vulnerable | CallToUnknown | CallToUnknown,ExternalBug,TransactionOrderDependence | CallToUnknown | ExternalBug,TransactionOrderDependence | — | ✓ |
| 06_tricky_dos_in_constructor | vulnerable | DenialOfService | — | — | — | DenialOfService | ✗ |
| 07_multivuln_call_reentrancy | vulnerable | CallToUnknown,Reentrancy,Timestamp | CallToUnknown,ExternalBug,IntegerUO,MishandledException,Reentrancy,Timestamp | CallToUnknown,Reentrancy,Timestamp | ExternalBug,IntegerUO,MishandledException | — | ✓ |
| 07_multivuln_dos_exception | vulnerable | DenialOfService,GasException | — | — | — | DenialOfService,GasException | ✗ |

## Baseline comparison

| Metric | Baseline | Current | Delta |
|---|---|---|---|
| macro_F1 | 0.2455 | 0.3756 | +0.1302 |
| micro_F1 | 0.2736 | 0.3939 | +0.1203 |
| contract_accuracy_loose | 0.4091 | 0.6875 | +0.2784 |
