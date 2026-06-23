# SENTINEL Agents Pipeline Evaluation Report

**Contracts evaluated:** 88  
**Positive verdict set:** ['CONFIRMED', 'LIKELY']  
**Macro-F1:** 0.0000  |  **Micro-F1:** 0.0000

## Gate assertions

| Gate | Description | Passed | Detail |
|---|---|---|---|
| `WS1a_silent_safe_on_flagged` | No consensus-flagged class ends SAFE (debate cannot clear consensus verdict) | PASS | 0 violation(s) |
| `WS1b_inconclusive_on_timeout` | edge_debate_timeout emits INCONCLUSIVE (LLM-on mode only) | PASS | N/A in --no-llm mode (no debate to time out) |
| `WS1c_no_missing_consensus_votes` | No consensus vote is missing from final verdicts (Finding #15) | PASS | 0 missing vote(s) |
| `WS1d_confidence_1_0_not_downgraded` | No consensus confidence=1.0 class ends in SAFE (Finding #14 worst case) | PASS | 0 violation(s) |
| `WS1e_no_vulnerable_label_with_safe_verdict` | No contract has overall_label=vulnerable + overall_verdict=SAFE (Finding #19) | PASS | 0 violation(s) |
| `WS2_false_positives_on_safe` | Zero false-positive verdicts on the safe subset | PASS | 0 FP(s) on 6 safe contract(s) |
| `WS3_long_contract_bug_detected` | edge_long_contract_truncated Reentrancy flagged (bug is past 2000-char cutoff) | FAIL | Reentrancy verdict = MISSING (positive set = ['CONFIRMED', 'LIKELY']) |
| `macro_f1_vs_baseline` | macro_F1 >= baseline (0.2455) | FAIL | macro_F1 = 0.0000 (delta -0.2455 vs baseline 0.2455) |

**Overall: GATE FAILURE**

## Per-class metrics

| Class | Support | TP | FP | FN | TN | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|---|
| MishandledException | 19 | 0 | 0 | 19 | 69 | nan | 0.0000 | 0.0000 |
| Reentrancy | 18 | 0 | 0 | 18 | 70 | nan | 0.0000 | 0.0000 |
| CallToUnknown | 17 | 0 | 0 | 17 | 71 | nan | 0.0000 | 0.0000 |
| Timestamp | 14 | 0 | 0 | 14 | 74 | nan | 0.0000 | 0.0000 |
| ExternalBug | 13 | 0 | 0 | 13 | 75 | nan | 0.0000 | 0.0000 |
| DenialOfService | 10 | 0 | 0 | 10 | 78 | nan | 0.0000 | 0.0000 |
| UnusedReturn | 10 | 0 | 0 | 10 | 78 | nan | 0.0000 | 0.0000 |
| GasException | 9 | 0 | 0 | 9 | 79 | nan | 0.0000 | 0.0000 |
| IntegerUO | 9 | 0 | 0 | 9 | 79 | nan | 0.0000 | 0.0000 |
| TransactionOrderDependence | 9 | 0 | 0 | 9 | 79 | nan | 0.0000 | 0.0000 |

## Contract-level accuracy

- **Loose** (safe→no flag OR vuln→≥1 correct flag): 6/88 = 6.82%
- **Strict exact-match** (predicted set == label set): 6/88 = 6.82%

## Per-contract detail

| Contract | GT | Labels | Predicted | TP | FP | FN | Correct |
|---|---|---|---|---|---|---|---|
| 01_approve_frontrun | vulnerable | TransactionOrderDependence | — | — | — | TransactionOrderDependence | ✗ |
| 01_batch_payout_swallow | vulnerable | MishandledException | — | — | — | MishandledException | ✗ |
| 01_bccc_reentrancy_injected_erc20 | vulnerable | Reentrancy,IntegerUO | — | — | — | IntegerUO,Reentrancy | ✗ |
| 01_cei_violation_erc721 | vulnerable | Reentrancy | — | — | — | Reentrancy | ✗ |
| 01_checks_effects_interactions | safe | — | — | — | — | — | ✓ |
| 01_erc20_underflow | vulnerable | IntegerUO | — | — | — | IntegerUO | ✗ |
| 01_flash_loan_oracle_manipulation | vulnerable | ExternalBug,MishandledException | — | — | — | ExternalBug,MishandledException | ✗ |
| 01_massive_storage_loop | vulnerable | GasException | — | — | — | GasException | ✗ |
| 01_multi_asset_transfer | vulnerable | UnusedReturn | — | — | — | UnusedReturn | ✗ |
| 01_proxy_delegatecall | vulnerable | CallToUnknown | — | — | — | CallToUnknown | ✗ |
| 01_unbounded_refund | vulnerable | DenialOfService,Reentrancy | — | — | — | DenialOfService,Reentrancy | ✗ |
| 01_vesting_schedule | vulnerable | Timestamp | — | — | — | Timestamp | ✗ |
| 02_access_control_bypass | vulnerable | ExternalBug,CallToUnknown | — | — | — | CallToUnknown,ExternalBug | ✗ |
| 02_auction_deadline | vulnerable | Timestamp,Reentrancy | — | — | — | Reentrancy,Timestamp | ✗ |
| 02_bccc_dos_injected_loop | vulnerable | DenialOfService,Timestamp | — | — | — | DenialOfService,Timestamp | ✗ |
| 02_calldata_expansion_dos | vulnerable | GasException | — | — | — | GasException | ✗ |
| 02_cross_function_reentrancy | vulnerable | Reentrancy | — | — | — | Reentrancy | ✗ |
| 02_delegatecall_muted | vulnerable | MishandledException,CallToUnknown | — | — | — | CallToUnknown,MishandledException | ✗ |
| 02_dynamic_dispatch | vulnerable | CallToUnknown | — | — | — | CallToUnknown | ✗ |
| 02_liquidation_ignore | vulnerable | UnusedReturn | — | — | — | UnusedReturn | ✗ |
| 02_mempool_sniping | vulnerable | TransactionOrderDependence | — | — | — | TransactionOrderDependence | ✗ |
| 02_pull_over_push_payment | safe | — | — | — | — | — | ✓ |
| 02_push_payment_failure | vulnerable | DenialOfService,Reentrancy | — | — | — | DenialOfService,Reentrancy | ✗ |
| 02_unchecked_auction_bid | vulnerable | IntegerUO | — | — | — | IntegerUO | ✗ |
| 03_bccc_calltounknown_injected_delegate | vulnerable | CallToUnknown,MishandledException | — | — | — | CallToUnknown,MishandledException | ✗ |
| 03_delegatecall_injection | vulnerable | ExternalBug,CallToUnknown | — | — | — | CallToUnknown,ExternalBug | ✗ |
| 03_dynamic_loop_gas_bomb | vulnerable | DenialOfService | — | — | — | DenialOfService | ✗ |
| 03_failed_batch_approve | vulnerable | UnusedReturn | — | — | — | UnusedReturn | ✗ |
| 03_low_level_forwarder | vulnerable | CallToUnknown | — | — | — | CallToUnknown | ✗ |
| 03_mev_arbitrage | vulnerable | TransactionOrderDependence | — | — | — | TransactionOrderDependence | ✗ |
| 03_multi_call_hub | vulnerable | MishandledException | — | — | — | MishandledException | ✗ |
| 03_nested_dynamic_array | vulnerable | GasException | — | — | — | GasException | ✗ |
| 03_openzeppelin_managed | safe | — | — | — | — | — | ✓ |
| 03_randomness_seed | vulnerable | Timestamp | — | — | — | Timestamp | ✗ |
| 03_read_only_reentrancy | vulnerable | Reentrancy | — | — | — | Reentrancy | ✗ |
| 03_safe_math_bypass | vulnerable | IntegerUO | — | — | — | IntegerUO | ✗ |
| 04_bccc_gas_injected_nested | vulnerable | GasException,DenialOfService | — | — | — | DenialOfService,GasException | ✗ |
| 04_erc777_callback_reentrancy | vulnerable | Reentrancy,MishandledException | — | — | — | MishandledException,Reentrancy | ✗ |
| 04_ico_phase_gate | vulnerable | Timestamp | — | — | — | Timestamp | ✗ |
| 04_nested_call_chain | vulnerable | UnusedReturn,MishandledException,Reentrancy | — | — | — | MishandledException,Reentrancy,UnusedReturn | ✗ |
| 04_opaque_contract_factory | vulnerable | CallToUnknown,ExternalBug | — | — | — | CallToUnknown,ExternalBug | ✗ |
| 04_pausable_circuit_breaker | safe | — | — | — | — | — | ✓ |
| 04_permit_sig_frontrun | vulnerable | TransactionOrderDependence | — | — | — | TransactionOrderDependence | ✗ |
| 04_sig_replay_attack | vulnerable | ExternalBug,MishandledException | — | — | — | ExternalBug,MishandledException | ✗ |
| 04_storage_growth_dos | vulnerable | DenialOfService | — | — | — | DenialOfService | ✗ |
| 04_time_calc_overflow | vulnerable | IntegerUO | — | — | — | IntegerUO | ✗ |
| 04_transfer_stipend_exhaustion | vulnerable | GasException | — | — | — | GasException | ✗ |
| 04_withdrawal_fails_silent | vulnerable | MishandledException | — | — | — | MishandledException | ✗ |
| 05_approval_race_swallow | vulnerable | MishandledException | — | — | — | MishandledException | ✗ |
| 05_batch_transfer_wrapping | vulnerable | IntegerUO | — | — | — | IntegerUO | ✗ |
| 05_bccc_externalbug_injected_flashloan | vulnerable | ExternalBug,UnusedReturn | — | — | — | ExternalBug,UnusedReturn | ✗ |
| 05_dutch_auction_race | vulnerable | TransactionOrderDependence | — | — | — | TransactionOrderDependence | ✗ |
| 05_emergency_stop_controlled | safe | — | — | — | — | — | ✓ |
| 05_eth_bank_multi_withdraw | vulnerable | Reentrancy | — | — | — | Reentrancy | ✗ |
| 05_logic_contract_selfdestruct | vulnerable | ExternalBug,CallToUnknown | — | — | — | CallToUnknown,ExternalBug | ✗ |
| 05_staticcall_gas_bomb | vulnerable | GasException | — | — | — | GasException | ✗ |
| 05_time_lock_governance | vulnerable | Timestamp | — | — | — | Timestamp | ✗ |
| 05_unexpected_revert_dos | vulnerable | DenialOfService,MishandledException | — | — | — | DenialOfService,MishandledException | ✗ |
| 05_upgrade_proxy_selfdestruct | vulnerable | CallToUnknown,ExternalBug | — | — | — | CallToUnknown,ExternalBug | ✗ |
| 05_wrapped_eth_deposit | vulnerable | UnusedReturn | — | — | — | UnusedReturn | ✗ |
| 06_bccc_timestamp_injected_vesting | vulnerable | Timestamp,TransactionOrderDependence | — | — | — | Timestamp,TransactionOrderDependence | ✗ |
| 06_tricky_call_in_fallback | vulnerable | CallToUnknown,ExternalBug | — | — | — | CallToUnknown,ExternalBug | ✗ |
| 06_tricky_dos_in_constructor | vulnerable | DenialOfService | — | — | — | DenialOfService | ✗ |
| 06_tricky_externalbug_callback_chain | vulnerable | ExternalBug,MishandledException | — | — | — | ExternalBug,MishandledException | ✗ |
| 06_tricky_gas_in_loop_condition | vulnerable | GasException | — | — | — | GasException | ✗ |
| 06_tricky_mishandled_internal_chain | vulnerable | MishandledException | — | — | — | MishandledException | ✗ |
| 06_tricky_overflow_in_interest_rate | vulnerable | IntegerUO | — | — | — | IntegerUO | ✗ |
| 06_tricky_reentrancy_in_modifier | vulnerable | Reentrancy | — | — | — | Reentrancy | ✗ |
| 06_tricky_timestamp_in_pricing | vulnerable | Timestamp,MishandledException | — | — | — | MishandledException,Timestamp | ✗ |
| 06_tricky_tod_mempool_sniping | vulnerable | TransactionOrderDependence | — | — | — | TransactionOrderDependence | ✗ |
| 06_tricky_unused_return_in_callchain | vulnerable | UnusedReturn,CallToUnknown,MishandledException | — | — | — | CallToUnknown,MishandledException,UnusedReturn | ✗ |
| 07_bccc_mishandled_injected_multicall | vulnerable | MishandledException,CallToUnknown | — | — | — | CallToUnknown,MishandledException | ✗ |
| 07_multivuln_call_reentrancy | vulnerable | CallToUnknown,Reentrancy,Timestamp | — | — | — | CallToUnknown,Reentrancy,Timestamp | ✗ |
| 07_multivuln_dos_exception | vulnerable | DenialOfService,GasException | — | — | — | DenialOfService,GasException | ✗ |
| 07_multivuln_externalbug_mishandled | vulnerable | ExternalBug,MishandledException,GasException | — | — | — | ExternalBug,GasException,MishandledException | ✗ |
| 07_multivuln_overflow_unused_return | vulnerable | IntegerUO,UnusedReturn,MishandledException | — | — | — | IntegerUO,MishandledException,UnusedReturn | ✗ |
| 07_multivuln_reentrancy_tod | vulnerable | Reentrancy,TransactionOrderDependence | — | — | — | Reentrancy,TransactionOrderDependence | ✗ |
| 07_multivuln_timestamp_call | vulnerable | Timestamp,CallToUnknown,MishandledException | — | — | — | CallToUnknown,MishandledException,Timestamp | ✗ |
| 08_bccc_unusedreturn_injected_batch | vulnerable | UnusedReturn,ExternalBug | — | — | — | ExternalBug,UnusedReturn | ✗ |
| 09_bccc_tod_injected_approve | vulnerable | TransactionOrderDependence,IntegerUO | — | — | — | IntegerUO,TransactionOrderDependence | ✗ |
| 10_bccc_weakaccess_injected_ownable | vulnerable | CallToUnknown,Timestamp | — | — | — | CallToUnknown,Timestamp | ✗ |
| 11_bccc_multivuln_oracle_borrow | vulnerable | ExternalBug,Reentrancy,Timestamp,CallToUnknown | — | — | — | CallToUnknown,ExternalBug,Reentrancy,Timestamp | ✗ |
| 12_bccc_dos_reentrancy_combo | vulnerable | DenialOfService,Reentrancy,UnusedReturn | — | — | — | DenialOfService,Reentrancy,UnusedReturn | ✗ |
| edge_borderline_no_corroboration | vulnerable | Timestamp | — | — | — | Timestamp | ✗ |
| edge_debate_timeout | vulnerable | Reentrancy | — | — | — | Reentrancy | ✗ |
| edge_long_contract_truncated | vulnerable | Reentrancy | — | — | — | Reentrancy | ✗ |
| edge_multi_bug | vulnerable | Reentrancy,Timestamp | — | — | — | Reentrancy,Timestamp | ✗ |
| edge_safe_rag_resembles_exploit | safe | — | — | — | — | — | ✓ |

## Baseline comparison

| Metric | Baseline | Current | Delta |
|---|---|---|---|
| macro_F1 | 0.2455 | 0.0000 | -0.2455 |
| micro_F1 | 0.2736 | 0.0000 | -0.2736 |
| contract_accuracy_loose | 0.4091 | 0.0682 | -0.3409 |
