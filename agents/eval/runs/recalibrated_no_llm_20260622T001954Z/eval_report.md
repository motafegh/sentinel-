# SENTINEL Agents Pipeline Evaluation Report

**Contracts evaluated:** 88  
**Positive verdict set:** ['CONFIRMED', 'LIKELY']  
**Macro-F1:** 0.2841  |  **Micro-F1:** 0.3294

## Gate assertions

| Gate | Description | Passed | Detail |
|---|---|---|---|
| `WS1a_silent_safe_on_flagged` | No consensus-flagged class ends SAFE (debate cannot clear consensus verdict) | PASS | 0 violation(s) |
| `WS1b_inconclusive_on_timeout` | edge_debate_timeout emits INCONCLUSIVE (LLM-on mode only) | PASS | N/A in --no-llm mode (no debate to time out) |
| `WS1c_no_missing_consensus_votes` | No consensus vote is missing from final verdicts (Finding #15) | PASS | 0 missing vote(s) |
| `WS1d_confidence_1_0_not_downgraded` | No consensus confidence=1.0 class ends in SAFE (Finding #14 worst case) | PASS | 0 violation(s) |
| `WS1e_no_vulnerable_label_with_safe_verdict` | No contract has overall_label=vulnerable + overall_verdict=SAFE (Finding #19) | PASS | 0 violation(s) |
| `WS2_false_positives_on_safe` | Zero false-positive verdicts on the safe subset | FAIL | 6 FP(s) on 6 safe contract(s): 01_checks_effects_interactions: ['CallToUnknown', 'ExternalBug', 'IntegerUO']; 02_pull_over_push_payment: ['CallToUnknown', 'ExternalBug', 'IntegerUO', 'Reentrancy']; 03_openzeppelin_managed: ['ExternalBug', 'IntegerUO']; 04_pausable_circuit_breaker: ['ExternalBug', 'IntegerUO']; 05_emergency_stop_controlled: ['ExternalBug', 'IntegerUO', 'Timestamp'] |
| `WS3_long_contract_bug_detected` | edge_long_contract_truncated Reentrancy flagged (bug is past 2000-char cutoff) | FAIL | Reentrancy verdict = DISPUTED (positive set = ['CONFIRMED', 'LIKELY']) |
| `macro_f1_vs_baseline` | macro_F1 >= baseline (0.2455) | PASS | macro_F1 = 0.2841 (delta +0.0386 vs baseline 0.2455) |

**Overall: GATE FAILURE**

## Per-class metrics

| Class | Support | TP | FP | FN | TN | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|---|
| MishandledException | 19 | 4 | 1 | 15 | 68 | 0.8000 | 0.2105 | 0.3333 |
| Reentrancy | 18 | 13 | 24 | 5 | 46 | 0.3514 | 0.7222 | 0.4727 |
| CallToUnknown | 17 | 10 | 23 | 7 | 48 | 0.3030 | 0.5882 | 0.4000 |
| Timestamp | 14 | 9 | 12 | 5 | 62 | 0.4286 | 0.6429 | 0.5143 |
| ExternalBug | 13 | 10 | 41 | 3 | 34 | 0.1961 | 0.7692 | 0.3125 |
| DenialOfService | 10 | 1 | 3 | 9 | 75 | 0.2500 | 0.1000 | 0.1429 |
| UnusedReturn | 10 | 4 | 10 | 6 | 68 | 0.2857 | 0.4000 | 0.3333 |
| GasException | 9 | 1 | 3 | 8 | 76 | 0.2500 | 0.1111 | 0.1538 |
| IntegerUO | 9 | 4 | 32 | 5 | 47 | 0.1111 | 0.4444 | 0.1778 |
| TransactionOrderDependence | 9 | 0 | 7 | 9 | 72 | 0.0000 | 0.0000 | 0.0000 |

## Contract-level accuracy

- **Loose** (safe→no flag OR vuln→≥1 correct flag): 42/88 = 47.73%
- **Strict exact-match** (predicted set == label set): 2/88 = 2.27%

## Per-contract detail

| Contract | GT | Labels | Predicted | TP | FP | FN | Correct |
|---|---|---|---|---|---|---|---|
| 01_approve_frontrun | vulnerable | TransactionOrderDependence | — | — | — | TransactionOrderDependence | ✗ |
| 01_batch_payout_swallow | vulnerable | MishandledException | — | — | — | MishandledException | ✗ |
| 01_bccc_reentrancy_injected_erc20 | vulnerable | Reentrancy,IntegerUO | — | — | — | IntegerUO,Reentrancy | ✗ |
| 01_cei_violation_erc721 | vulnerable | Reentrancy | CallToUnknown,ExternalBug,IntegerUO,Reentrancy | Reentrancy | CallToUnknown,ExternalBug,IntegerUO | — | ✓ |
| 01_checks_effects_interactions | safe | — | CallToUnknown,ExternalBug,IntegerUO | — | CallToUnknown,ExternalBug,IntegerUO | — | ✗ |
| 01_erc20_underflow | vulnerable | IntegerUO | — | — | — | IntegerUO | ✗ |
| 01_flash_loan_oracle_manipulation | vulnerable | ExternalBug,MishandledException | ExternalBug,IntegerUO,Reentrancy,UnusedReturn | ExternalBug | IntegerUO,Reentrancy,UnusedReturn | MishandledException | ✓ |
| 01_massive_storage_loop | vulnerable | GasException | Timestamp | — | Timestamp | GasException | ✗ |
| 01_multi_asset_transfer | vulnerable | UnusedReturn | — | — | — | UnusedReturn | ✗ |
| 01_proxy_delegatecall | vulnerable | CallToUnknown | CallToUnknown,ExternalBug,TransactionOrderDependence | CallToUnknown | ExternalBug,TransactionOrderDependence | — | ✓ |
| 01_unbounded_refund | vulnerable | DenialOfService,Reentrancy | CallToUnknown,ExternalBug,Reentrancy | Reentrancy | CallToUnknown,ExternalBug | DenialOfService | ✓ |
| 01_vesting_schedule | vulnerable | Timestamp | ExternalBug,IntegerUO,Reentrancy,Timestamp,UnusedReturn | Timestamp | ExternalBug,IntegerUO,Reentrancy,UnusedReturn | — | ✓ |
| 02_access_control_bypass | vulnerable | ExternalBug,CallToUnknown | ExternalBug,IntegerUO,TransactionOrderDependence | ExternalBug | IntegerUO,TransactionOrderDependence | CallToUnknown | ✓ |
| 02_auction_deadline | vulnerable | Timestamp,Reentrancy | ExternalBug,Reentrancy,Timestamp | Reentrancy,Timestamp | ExternalBug | — | ✓ |
| 02_bccc_dos_injected_loop | vulnerable | DenialOfService,Timestamp | — | — | — | DenialOfService,Timestamp | ✗ |
| 02_calldata_expansion_dos | vulnerable | GasException | Timestamp | — | Timestamp | GasException | ✗ |
| 02_cross_function_reentrancy | vulnerable | Reentrancy | CallToUnknown,ExternalBug,IntegerUO,Reentrancy,Timestamp | Reentrancy | CallToUnknown,ExternalBug,IntegerUO,Timestamp | — | ✓ |
| 02_delegatecall_muted | vulnerable | MishandledException,CallToUnknown | CallToUnknown,ExternalBug,TransactionOrderDependence | CallToUnknown | ExternalBug,TransactionOrderDependence | MishandledException | ✓ |
| 02_dynamic_dispatch | vulnerable | CallToUnknown | CallToUnknown,ExternalBug,IntegerUO,UnusedReturn | CallToUnknown | ExternalBug,IntegerUO,UnusedReturn | — | ✓ |
| 02_liquidation_ignore | vulnerable | UnusedReturn | DenialOfService,ExternalBug,GasException,Reentrancy,UnusedReturn | UnusedReturn | DenialOfService,ExternalBug,GasException,Reentrancy | — | ✓ |
| 02_mempool_sniping | vulnerable | TransactionOrderDependence | CallToUnknown,ExternalBug,IntegerUO,Reentrancy | — | CallToUnknown,ExternalBug,IntegerUO,Reentrancy | TransactionOrderDependence | ✗ |
| 02_pull_over_push_payment | safe | — | CallToUnknown,ExternalBug,IntegerUO,Reentrancy | — | CallToUnknown,ExternalBug,IntegerUO,Reentrancy | — | ✗ |
| 02_push_payment_failure | vulnerable | DenialOfService,Reentrancy | CallToUnknown,ExternalBug,IntegerUO,Reentrancy | Reentrancy | CallToUnknown,ExternalBug,IntegerUO | DenialOfService | ✓ |
| 02_unchecked_auction_bid | vulnerable | IntegerUO | — | — | — | IntegerUO | ✗ |
| 03_bccc_calltounknown_injected_delegate | vulnerable | CallToUnknown,MishandledException | — | — | — | CallToUnknown,MishandledException | ✗ |
| 03_delegatecall_injection | vulnerable | ExternalBug,CallToUnknown | CallToUnknown,ExternalBug,IntegerUO,Reentrancy,Timestamp,TransactionOrderDependence | CallToUnknown,ExternalBug | IntegerUO,Reentrancy,Timestamp,TransactionOrderDependence | — | ✓ |
| 03_dynamic_loop_gas_bomb | vulnerable | DenialOfService | CallToUnknown,DenialOfService,ExternalBug,GasException,Reentrancy | DenialOfService | CallToUnknown,ExternalBug,GasException,Reentrancy | — | ✓ |
| 03_failed_batch_approve | vulnerable | UnusedReturn | — | — | — | UnusedReturn | ✗ |
| 03_low_level_forwarder | vulnerable | CallToUnknown | CallToUnknown,ExternalBug,IntegerUO,Reentrancy,TransactionOrderDependence | CallToUnknown | ExternalBug,IntegerUO,Reentrancy,TransactionOrderDependence | — | ✓ |
| 03_mev_arbitrage | vulnerable | TransactionOrderDependence | — | — | — | TransactionOrderDependence | ✗ |
| 03_multi_call_hub | vulnerable | MishandledException | CallToUnknown,ExternalBug,IntegerUO,MishandledException,Reentrancy | MishandledException | CallToUnknown,ExternalBug,IntegerUO,Reentrancy | — | ✓ |
| 03_nested_dynamic_array | vulnerable | GasException | — | — | — | GasException | ✗ |
| 03_openzeppelin_managed | safe | — | ExternalBug,IntegerUO | — | ExternalBug,IntegerUO | — | ✗ |
| 03_randomness_seed | vulnerable | Timestamp | CallToUnknown,ExternalBug,Reentrancy,Timestamp | Timestamp | CallToUnknown,ExternalBug,Reentrancy | — | ✓ |
| 03_read_only_reentrancy | vulnerable | Reentrancy | — | — | — | Reentrancy | ✗ |
| 03_safe_math_bypass | vulnerable | IntegerUO | CallToUnknown,ExternalBug,IntegerUO,Reentrancy | IntegerUO | CallToUnknown,ExternalBug,Reentrancy | — | ✓ |
| 04_bccc_gas_injected_nested | vulnerable | GasException,DenialOfService | — | — | — | DenialOfService,GasException | ✗ |
| 04_erc777_callback_reentrancy | vulnerable | Reentrancy,MishandledException | ExternalBug,IntegerUO,Reentrancy,UnusedReturn | Reentrancy | ExternalBug,IntegerUO,UnusedReturn | MishandledException | ✓ |
| 04_ico_phase_gate | vulnerable | Timestamp | — | — | — | Timestamp | ✗ |
| 04_nested_call_chain | vulnerable | UnusedReturn,MishandledException,Reentrancy | CallToUnknown,ExternalBug,IntegerUO,MishandledException,Reentrancy,UnusedReturn | MishandledException,Reentrancy,UnusedReturn | CallToUnknown,ExternalBug,IntegerUO | — | ✓ |
| 04_opaque_contract_factory | vulnerable | CallToUnknown,ExternalBug | CallToUnknown,ExternalBug,TransactionOrderDependence | CallToUnknown,ExternalBug | TransactionOrderDependence | — | ✓ |
| 04_pausable_circuit_breaker | safe | — | ExternalBug,IntegerUO | — | ExternalBug,IntegerUO | — | ✗ |
| 04_permit_sig_frontrun | vulnerable | TransactionOrderDependence | — | — | — | TransactionOrderDependence | ✗ |
| 04_sig_replay_attack | vulnerable | ExternalBug,MishandledException | CallToUnknown,ExternalBug,IntegerUO,Reentrancy,UnusedReturn | ExternalBug | CallToUnknown,IntegerUO,Reentrancy,UnusedReturn | MishandledException | ✓ |
| 04_storage_growth_dos | vulnerable | DenialOfService | CallToUnknown,ExternalBug,Timestamp | — | CallToUnknown,ExternalBug,Timestamp | DenialOfService | ✗ |
| 04_time_calc_overflow | vulnerable | IntegerUO | CallToUnknown,ExternalBug,IntegerUO,Reentrancy,Timestamp | IntegerUO | CallToUnknown,ExternalBug,Reentrancy,Timestamp | — | ✓ |
| 04_transfer_stipend_exhaustion | vulnerable | GasException | DenialOfService,ExternalBug,GasException,IntegerUO | GasException | DenialOfService,ExternalBug,IntegerUO | — | ✓ |
| 04_withdrawal_fails_silent | vulnerable | MishandledException | — | — | — | MishandledException | ✗ |
| 05_approval_race_swallow | vulnerable | MishandledException | ExternalBug,IntegerUO,Reentrancy,Timestamp,UnusedReturn | — | ExternalBug,IntegerUO,Reentrancy,Timestamp,UnusedReturn | MishandledException | ✗ |
| 05_batch_transfer_wrapping | vulnerable | IntegerUO | CallToUnknown,ExternalBug | — | CallToUnknown,ExternalBug | IntegerUO | ✗ |
| 05_bccc_externalbug_injected_flashloan | vulnerable | ExternalBug,UnusedReturn | — | — | — | ExternalBug,UnusedReturn | ✗ |
| 05_dutch_auction_race | vulnerable | TransactionOrderDependence | — | — | — | TransactionOrderDependence | ✗ |
| 05_emergency_stop_controlled | safe | — | ExternalBug,IntegerUO,Timestamp | — | ExternalBug,IntegerUO,Timestamp | — | ✗ |
| 05_eth_bank_multi_withdraw | vulnerable | Reentrancy | CallToUnknown,ExternalBug,IntegerUO,Reentrancy | Reentrancy | CallToUnknown,ExternalBug,IntegerUO | — | ✓ |
| 05_logic_contract_selfdestruct | vulnerable | ExternalBug,CallToUnknown | ExternalBug,IntegerUO,Reentrancy | ExternalBug | IntegerUO,Reentrancy | CallToUnknown | ✓ |
| 05_staticcall_gas_bomb | vulnerable | GasException | — | — | — | GasException | ✗ |
| 05_time_lock_governance | vulnerable | Timestamp | ExternalBug,IntegerUO,Reentrancy,Timestamp | Timestamp | ExternalBug,IntegerUO,Reentrancy | — | ✓ |
| 05_unexpected_revert_dos | vulnerable | DenialOfService,MishandledException | ExternalBug,IntegerUO,Reentrancy,UnusedReturn | — | ExternalBug,IntegerUO,Reentrancy,UnusedReturn | DenialOfService,MishandledException | ✗ |
| 05_upgrade_proxy_selfdestruct | vulnerable | CallToUnknown,ExternalBug | ExternalBug,Timestamp | ExternalBug | Timestamp | CallToUnknown | ✓ |
| 05_wrapped_eth_deposit | vulnerable | UnusedReturn | — | — | — | UnusedReturn | ✗ |
| 06_bccc_timestamp_injected_vesting | vulnerable | Timestamp,TransactionOrderDependence | — | — | — | Timestamp,TransactionOrderDependence | ✗ |
| 06_tricky_call_in_fallback | vulnerable | CallToUnknown,ExternalBug | CallToUnknown,ExternalBug,TransactionOrderDependence | CallToUnknown,ExternalBug | TransactionOrderDependence | — | ✓ |
| 06_tricky_dos_in_constructor | vulnerable | DenialOfService | — | — | — | DenialOfService | ✗ |
| 06_tricky_externalbug_callback_chain | vulnerable | ExternalBug,MishandledException | CallToUnknown,ExternalBug,IntegerUO,MishandledException | ExternalBug,MishandledException | CallToUnknown,IntegerUO | — | ✓ |
| 06_tricky_gas_in_loop_condition | vulnerable | GasException | Timestamp | — | Timestamp | GasException | ✗ |
| 06_tricky_mishandled_internal_chain | vulnerable | MishandledException | — | — | — | MishandledException | ✗ |
| 06_tricky_overflow_in_interest_rate | vulnerable | IntegerUO | CallToUnknown,ExternalBug,IntegerUO,Timestamp | IntegerUO | CallToUnknown,ExternalBug,Timestamp | — | ✓ |
| 06_tricky_reentrancy_in_modifier | vulnerable | Reentrancy | CallToUnknown,ExternalBug,IntegerUO,Reentrancy | Reentrancy | CallToUnknown,ExternalBug,IntegerUO | — | ✓ |
| 06_tricky_timestamp_in_pricing | vulnerable | Timestamp,MishandledException | ExternalBug,IntegerUO,Reentrancy,Timestamp,UnusedReturn | Timestamp | ExternalBug,IntegerUO,Reentrancy,UnusedReturn | MishandledException | ✓ |
| 06_tricky_tod_mempool_sniping | vulnerable | TransactionOrderDependence | CallToUnknown,ExternalBug,Reentrancy | — | CallToUnknown,ExternalBug,Reentrancy | TransactionOrderDependence | ✗ |
| 06_tricky_unused_return_in_callchain | vulnerable | UnusedReturn,CallToUnknown,MishandledException | CallToUnknown,ExternalBug,Reentrancy,Timestamp,UnusedReturn | CallToUnknown,UnusedReturn | ExternalBug,Reentrancy,Timestamp | MishandledException | ✓ |
| 07_bccc_mishandled_injected_multicall | vulnerable | MishandledException,CallToUnknown | — | — | — | CallToUnknown,MishandledException | ✗ |
| 07_multivuln_call_reentrancy | vulnerable | CallToUnknown,Reentrancy,Timestamp | CallToUnknown,ExternalBug,IntegerUO,MishandledException,Reentrancy,Timestamp | CallToUnknown,Reentrancy,Timestamp | ExternalBug,IntegerUO,MishandledException | — | ✓ |
| 07_multivuln_dos_exception | vulnerable | DenialOfService,GasException | — | — | — | DenialOfService,GasException | ✗ |
| 07_multivuln_externalbug_mishandled | vulnerable | ExternalBug,MishandledException,GasException | ExternalBug,Reentrancy,UnusedReturn | ExternalBug | Reentrancy,UnusedReturn | GasException,MishandledException | ✓ |
| 07_multivuln_overflow_unused_return | vulnerable | IntegerUO,UnusedReturn,MishandledException | ExternalBug,IntegerUO,Reentrancy,UnusedReturn | IntegerUO,UnusedReturn | ExternalBug,Reentrancy | MishandledException | ✓ |
| 07_multivuln_reentrancy_tod | vulnerable | Reentrancy,TransactionOrderDependence | CallToUnknown,DenialOfService,ExternalBug,GasException,IntegerUO,Reentrancy | Reentrancy | CallToUnknown,DenialOfService,ExternalBug,GasException,IntegerUO | TransactionOrderDependence | ✓ |
| 07_multivuln_timestamp_call | vulnerable | Timestamp,CallToUnknown,MishandledException | CallToUnknown,ExternalBug,IntegerUO,MishandledException,Reentrancy,Timestamp | CallToUnknown,MishandledException,Timestamp | ExternalBug,IntegerUO,Reentrancy | — | ✓ |
| 08_bccc_unusedreturn_injected_batch | vulnerable | UnusedReturn,ExternalBug | — | — | — | ExternalBug,UnusedReturn | ✗ |
| 09_bccc_tod_injected_approve | vulnerable | TransactionOrderDependence,IntegerUO | — | — | — | IntegerUO,TransactionOrderDependence | ✗ |
| 10_bccc_weakaccess_injected_ownable | vulnerable | CallToUnknown,Timestamp | — | — | — | CallToUnknown,Timestamp | ✗ |
| 11_bccc_multivuln_oracle_borrow | vulnerable | ExternalBug,Reentrancy,Timestamp,CallToUnknown | — | — | — | CallToUnknown,ExternalBug,Reentrancy,Timestamp | ✗ |
| 12_bccc_dos_reentrancy_combo | vulnerable | DenialOfService,Reentrancy,UnusedReturn | — | — | — | DenialOfService,Reentrancy,UnusedReturn | ✗ |
| edge_borderline_no_corroboration | vulnerable | Timestamp | Timestamp | Timestamp | — | — | ✓ |
| edge_debate_timeout | vulnerable | Reentrancy | Reentrancy,UnusedReturn | Reentrancy | UnusedReturn | — | ✓ |
| edge_long_contract_truncated | vulnerable | Reentrancy | — | — | — | Reentrancy | ✗ |
| edge_multi_bug | vulnerable | Reentrancy,Timestamp | Reentrancy,Timestamp | Reentrancy,Timestamp | — | — | ✓ |
| edge_safe_rag_resembles_exploit | safe | — | CallToUnknown,ExternalBug,IntegerUO,Reentrancy | — | CallToUnknown,ExternalBug,IntegerUO,Reentrancy | — | ✗ |

## Baseline comparison

| Metric | Baseline | Current | Delta |
|---|---|---|---|
| macro_F1 | 0.2455 | 0.2841 | +0.0386 |
| micro_F1 | 0.2736 | 0.3294 | +0.0558 |
| contract_accuracy_loose | 0.4091 | 0.4773 | +0.0682 |
