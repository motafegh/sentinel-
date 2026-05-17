# Task 16: Wrong Contract Selection Audit
**Multi-contract files analyzed:** 500  **Files with matching graph .pt:** 135  **Files skipped (no graph/load error):** 365  **Files with no contract_name attr:** 0

## Heuristic Comparison
| Heuristic | Correct | Wrong | Accuracy | 95% CI | Wrong Rate | 95% CI (wrong) |
|-----------|---------|-------|----------|--------|------------|----------------|
| Most Functions | 71 | 64 | 52.6% | [44.2%, 60.8%] | 47.4% | [39.2%, 55.8%] |
| Last Contract | 118 | 17 | 87.4% | [80.8%, 92.0%] | 12.6% | [8.0%, 19.2%] |

## Per-Class Breakdown (Most Functions Heuristic)
| Vulnerability Class | Correct | Wrong | Wrong Rate |
|---------------------|---------|-------|------------|
| CallToUnknown | 7 | 9 | 56.2% |
| DenialOfService | 1 | 0 | 0.0% |
| ExternalBug | 6 | 6 | 50.0% |
| GasException | 10 | 7 | 41.2% |
| IntegerUO | 26 | 24 | 48.0% |
| MishandledException | 10 | 8 | 44.4% |
| Reentrancy | 10 | 9 | 47.4% |
| Timestamp | 6 | 3 | 33.3% |
| TransactionOrderDependence | 6 | 6 | 50.0% |
| UnusedReturn | 6 | 6 | 50.0% |

## Per-Class Breakdown (Last Contract Heuristic)
| Vulnerability Class | Correct | Wrong | Wrong Rate |
|---------------------|---------|-------|------------|
| CallToUnknown | 11 | 5 | 31.2% |
| DenialOfService | 1 | 0 | 0.0% |
| ExternalBug | 11 | 1 | 8.3% |
| GasException | 13 | 4 | 23.5% |
| IntegerUO | 42 | 8 | 16.0% |
| MishandledException | 17 | 1 | 5.6% |
| Reentrancy | 16 | 3 | 15.8% |
| Timestamp | 9 | 0 | 0.0% |
| TransactionOrderDependence | 12 | 0 | 0.0% |
| UnusedReturn | 10 | 2 | 16.7% |

## Breakdown by Number of Contracts in File
| # Contracts | Total Files | MF Correct | MF Wrong | MF Wrong Rate | LC Correct | LC Wrong | LC Wrong Rate |
|-------------|-------------|------------|----------|---------------|------------|----------|---------------|
| 2 | 17 | 14 | 3 | 17.6% | 17 | 0 | 0.0% |
| 3 | 39 | 20 | 19 | 48.7% | 36 | 3 | 7.7% |
| 4 | 7 | 6 | 1 | 14.3% | 7 | 0 | 0.0% |
| 5 | 18 | 14 | 4 | 22.2% | 17 | 1 | 5.6% |
| 6 | 9 | 3 | 6 | 66.7% | 7 | 2 | 22.2% |
| 7 | 10 | 5 | 5 | 50.0% | 9 | 1 | 10.0% |
| 8 | 13 | 6 | 7 | 53.8% | 10 | 3 | 23.1% |
| 9 | 6 | 1 | 5 | 83.3% | 4 | 2 | 33.3% |
| 10 | 7 | 2 | 5 | 71.4% | 6 | 1 | 14.3% |
| 11 | 3 | 0 | 3 | 100.0% | 2 | 1 | 33.3% |
| 12 | 2 | 0 | 2 | 100.0% | 1 | 1 | 50.0% |
| 13 | 3 | 0 | 3 | 100.0% | 1 | 2 | 66.7% |
| 20 | 1 | 0 | 1 | 100.0% | 1 | 0 | 0.0% |

## Wrong Selection Examples (Most Functions, first 20)
- md5=`e36e63b0e030...` actual=`LTE` predicted=`Crowdsale` (n_contracts=8)- md5=`487876d15840...` actual=`BlockCoreOne` predicted=`ERC721` (n_contracts=6)- md5=`b058f7596ae6...` actual=`BEZOP` predicted=`StandardToken` (n_contracts=7)- md5=`e548a4adec19...` actual=`BTPCoin` predicted=`TokenERC20` (n_contracts=3)- md5=`a155f683efca...` actual=`MLPPToken` predicted=`StandardToken` (n_contracts=5)- md5=`10f91b164da3...` actual=`ERC20Token` predicted=`StandardToken` (n_contracts=3)- md5=`37b3ca04d483...` actual=`PumaPayToken` predicted=`PumaPayPullPayment` (n_contracts=8)- md5=`12088e9bcdef...` actual=`FANBASE` predicted=`StandardToken` (n_contracts=3)- md5=`03263b7704e6...` actual=`CryptoTreasure` predicted=`usingOraclize` (n_contracts=7)- md5=`be07808d84ed...` actual=`Park` predicted=`StandardToken` (n_contracts=7)- md5=`57b7d95d87c5...` actual=`ElecTokenSmartContract` predicted=`ElecApprover` (n_contracts=9)- md5=`ae6bacda23f2...` actual=`ERC20Token` predicted=`StandardToken` (n_contracts=3)- md5=`177e67f40cdc...` actual=`Reval` predicted=`StandardToken` (n_contracts=7)- md5=`2f12901d4f86...` actual=`LinkToken` predicted=`StandardToken` (n_contracts=8)- md5=`1a389490f95e...` actual=`ImmlaToken` predicted=`ImmlaDistribution` (n_contracts=11)- md5=`24d38aec318a...` actual=`FNAToken` predicted=`StandardToken` (n_contracts=3)- md5=`e0ec5afe028a...` actual=`GangnamToken` predicted=`StandardToken` (n_contracts=3)- md5=`081450419b79...` actual=`MOVICoin` predicted=`StandardToken` (n_contracts=3)- md5=`6cd07611b548...` actual=`DGD` predicted=`StandardToken` (n_contracts=6)- md5=`67f5fa5f80f8...` actual=`ERC20Token` predicted=`StandardToken` (n_contracts=3)
... and 44 more

## Wrong Selection Examples (Last Contract, first 20)
- md5=`e36e63b0e030...` actual=`LTE` predicted=`Crowdsale` (n_contracts=8)- md5=`c30f5a630ae3...` actual=`IBattleboardData` predicted=`ManageBattleboards` (n_contracts=8)- md5=`2f882f56e7ea...` actual=`AOIonInterface` predicted=`AOTreasury` (n_contracts=6)- md5=`eb6f3df9f44d...` actual=`ERC20Interface` predicted=`DeDeContract` (n_contracts=3)- md5=`37b3ca04d483...` actual=`PumaPayToken` predicted=`PumaPayPullPayment` (n_contracts=8)- md5=`57b7d95d87c5...` actual=`ElecTokenSmartContract` predicted=`ElecSaleSmartContract` (n_contracts=9)- md5=`1a389490f95e...` actual=`ImmlaToken` predicted=`ImmlaDistribution` (n_contracts=11)- md5=`0db66636f537...` actual=`RicoToken` predicted=`PreSale` (n_contracts=12)- md5=`3e73b3bad1f6...` actual=`owned` predicted=`MyAdvancedToken` (n_contracts=3)- md5=`09547e848334...` actual=`StoxSmartToken` predicted=`StoxSmartTokenSale` (n_contracts=13)- md5=`27f174af3fe1...` actual=`GOLD` predicted=`ERC223` (n_contracts=13)- md5=`8c0758681c03...` actual=`RDOToken` predicted=`RDOCrowdsale` (n_contracts=6)- md5=`0c0d10726170...` actual=`ZOMToken` predicted=`Reward` (n_contracts=7)- md5=`d7b3acc8522a...` actual=`MiniMeTokenI` predicted=`ReferalsTokenHolder` (n_contracts=5)- md5=`a6235029d0f8...` actual=`PLCRVoting` predicted=`CivilParameterizer` (n_contracts=3)- md5=`0a84fea648d8...` actual=`Smartcop` predicted=`Smartcop_Locker` (n_contracts=9)- md5=`ad856be39a0f...` actual=`GVToken` predicted=`ICO` (n_contracts=10)

## Conclusion
The **Last Contract** heuristic outperforms Most Functions (87.4% vs 52.6% accuracy). 
Overall wrong-selection rate: **Most Functions=47.4%**, **Last Contract=12.6%**. 
Files with more contracts tend to have higher wrong-selection rates, suggesting that contract selection is a meaningful source of label noise for multi-contract files.