# Task 19: Timestamp Label Quality Audit
**Total Timestamp=1 contracts:** 2191  **Successfully analyzed:** 2191  **No source file:** 0  **No graph file:** 0  **Load errors:** 0

## Signal vs Feature Classification
| Category | Description | Count | Percentage |
|----------|-------------|-------|------------|
| (a) | Signal in source AND feature fires | 623 | 28.4% |
| (b) | Signal in source but feature doesn't fire | 491 | 22.4% |
| (c) | No signal AND feature doesn't fire (mislabelled?) | 1056 | 48.2% |
| (d) | No signal but feature fires | 21 | 1.0% |

### Potential Mislabel Rate: **48.2%** ⚠️ **EXCEEDS 30% THRESHOLD** — significant label quality concern!

## Timestamp Signal Patterns Found in Source
| Pattern | Count |
|---------|-------|
| block.timestamp | 647 |
| block.number | 222 |
| now | 570 |
| block.difficulty | 74 |
| blockhash | 89 |

## Category (b) Analysis: Signal in Source but Feature Doesn't Fire
**Total (b) cases:** 491  
**Due to wrong contract selection:** 202  
**Other reasons:** 289

### Wrong Contract Selection Cases
- stem=`000cab1510e0...` graph_contract=`hodlToken` block_globals_contract=`Pausable`- stem=`00201d36e298...` graph_contract=`FairEthereumDivs` block_globals_contract=`FairEthereumDivs` reason=unknown (contract matches or no multi-contract)- stem=`0029638f52fe...` graph_contract=`DailyLucky` block_globals_contract=`DailyLucky` reason=unknown (contract matches or no multi-contract)- stem=`0031f0404cfe...` graph_contract=`EnhancedToken` block_globals_contract=`Recoverable`- stem=`003264b79d4d...` graph_contract=`LetsbetToken` block_globals_contract=`DutchAuction`- stem=`00c88216767d...` graph_contract=`ETH_8` block_globals_contract=`ETH_8` reason=unknown (contract matches or no multi-contract)- stem=`010e924246e1...` graph_contract=`PirateLottery` block_globals_contract=`PirateLottery` reason=unknown (contract matches or no multi-contract)- stem=`0142adf22104...` graph_contract=`KOIOSToken` block_globals_contract=`KOIOSTokenSale`- stem=`016aba6228a8...` graph_contract=`TPCToken` block_globals_contract=`TPCToken` reason=unknown (contract matches or no multi-contract)- stem=`01813bdb4485...` graph_contract=`OREO` block_globals_contract=`OREO` reason=unknown (contract matches or no multi-contract)- stem=`019996eb4c7b...` graph_contract=`DtktSale` block_globals_contract=`DtktSale` reason=unknown (contract matches or no multi-contract)- stem=`019d0a47b9da...` graph_contract=`LoligoToken` block_globals_contract=`Presale`- stem=`01c61fcafd6a...` graph_contract=`PuzzleBID` block_globals_contract=`PuzzleBID` reason=unknown (contract matches or no multi-contract)- stem=`0244ca5b27a7...` graph_contract=`DoubleProfit` block_globals_contract=`DoubleProfit` reason=unknown (contract matches or no multi-contract)- stem=`02a24f6178af...` graph_contract=`SynergisProxyDeposit` block_globals_contract=`SynergisProxyDeposit` reason=unknown (contract matches or no multi-contract)- stem=`03e05de56c1c...` graph_contract=`StreamityContract` block_globals_contract=`StreamityCrowdsale`- stem=`03f297897e98...` graph_contract=`YellowBetterToken` block_globals_contract=`TokenSale`- stem=`04435bb9aeb0...` graph_contract=`Lambda` block_globals_contract=`LambdaLock`- stem=`04beebe6771f...` graph_contract=`Token` block_globals_contract=`PackSale`- stem=`04fca2f89165...` graph_contract=`BlockvToken` block_globals_contract=`PoolAllocations`

### Possible Reasons for (b) Failures (Beyond Wrong Contract)
1. **Slither IR omission**: `block.timestamp` reads via `SolidityVariableComposed` may not appear in IR for some code paths2. **Inline assembly**: `block.timestamp` accessed via assembly blocks where Slither doesn't track IR operations3. **Indirect access**: timestamp read through a library or inherited contract that the feature computation doesn't trace into4. **Compilation failure**: If Slither fell back to partial parsing, IR may be incomplete

## Category (c) Examples: No Signal in Source (Potential Mislabels)
**Count:** 1056
- stem=`00027685d5b6...` file=`4ac4a54daec2eb0c16ddb27ea980271d457bb1cab66dce8385c52b2502a359e4.sol` contract=`MOFONGO`- stem=`00178bc2fcf5...` file=`4b53cb0c00cbdb61ac13d5b6fe5f976b8695a8120ff7638e9123200caae590da.sol` contract=`NEWTOKEN`- stem=`002d58b35cbd...` file=`7ac8bd60bed00cf90241a2fd73c99d4d494db8ca4b318450932cf46cfc99b83d.sol` contract=`FavorCoin`- stem=`003110982c5c...` file=`9784a975ca93273c962d3a13f66e2048a61fc4ea17435408316fd5a8db533065.sol` contract=`Beercoin`- stem=`005623d5c73a...` file=`442222bb085c77de544c6badee0c484dd34a8d06bd97f5c01a0a24e8b8dde20a.sol` contract=`UWTToken`- stem=`0056815b97d3...` file=`15766aa4047c112191d8df951feaac868f8025eba71157f8fd1a7512081bd671.sol` contract=`Mono`- stem=`00570f419377...` file=`6cc32cf07883832280061aef7a4bd29f101ba3672c4e841f769d1832574313ae.sol` contract=`Gracointoken`- stem=`006011e5c695...` file=`2670f780233960baf8f3214ed9bb866732f04427e595bed9db0c20d03e15f89a.sol` contract=`CardsRead`- stem=`00630bdeb8cd...` file=`3956657b80ad02f2899634eeb9e5d6bc6aee136955e2778ba7ee8fefebb8e27f.sol` contract=`Rucid`- stem=`00650d7c24ef...` file=`548384cc518137f5a9b13947405905d6b26b5a933df50ad8b300298e8ec7407b.sol` contract=`ShoppingOfPeople`- stem=`0069d12fd2e9...` file=`09324f6a9f18821a9d7b0efec6711943bcac3266a6d477d27e07fa0577a39bbe.sol` contract=`InvestLTCS`- stem=`0073c4d44137...` file=`5c4579aee75d158f1dc3d88ebcc5d97a80ffe3cc5f1d7b8dccf90558d41eac78.sol` contract=`HighBank`- stem=`007be2a67d02...` file=`48e4b331c23837b1155adb4efa55670e9ff60678e4ac5110e0731b6ca065a0d8.sol` contract=`Bitsave`- stem=`00a324a9cc7e...` file=`350bd868880924db53dcd1ee9101a840236834aa83e09ad7977fb2b2aea3653b.sol` contract=`barnecoin`- stem=`00c3304855aa...` file=`5f4416538130e7508539051143934d0f6ff74c04ba8e05647d43ee6dea97e38c.sol` contract=`NexyZero`- stem=`00e2acd47342...` file=`47459ed61e6ea53fc6410e79eb7abf8dd37aa276e9f102b6fe4dd124f3d95a0b.sol` contract=`MyAdvancedToken`- stem=`00f38752a816...` file=`bf6862e9e4e2b77538de1b10c83269272bd7f47d77139ff2d4be27d1da2880e3.sol` contract=`Token`- stem=`00f6f2521f7b...` file=`42424c340d58a169930f3cb1e71ba06ce0d71387bc7b2355a65f8f723a1ba3f8.sol` contract=`CargoForwarderExpressToken`- stem=`0104164aed2f...` file=`606d101372c450422f64d912c231db06371f912cd6141b904585b02e2aa53c02.sol` contract=`EtmTest1Token`- stem=`010b0a6fac2d...` file=`b892d762ad79bbc929b56bdd12158461795d461a526eab5ca08670ad4c7073f1.sol` contract=`MainToken`
... and 10 more

## Category (d) Analysis: No Source Signal but Feature Fires
**Count:** 21
These are likely **true positives** where:- The timestamp signal is in an inherited contract not in the main file- The source regex missed an obfuscated/indirect reference- Slither detected block global access through IR analysis (more thorough than grep)

## Summary
- **Feature recall** (among source-verified Timestamp): 55.9%- **Potential mislabel rate** (category c): 48.2%
- ⚠️ **CRITICAL**: Over 30% of Timestamp=1 labels may be incorrect. Consider re-labelling or excluding these contracts from training.
- Wrong contract selection accounts for 202 of 491 feature misses (41%)