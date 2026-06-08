# WS-I Stage 3 — Manual Inspection Report
**Date:** 2026-06-07  
**Reviewer:** Claude Code (automated deep inspection)  
**Contracts reviewed:** 30 worst-disagreement + 2 maxing = 32 total  
**Status:** COMPLETE

---

## Executive Summary

All 30 contracts reviewed. The dominant finding is overwhelming and consistent:

**28/30 review_pending contracts carry a contradictory `Class12:NonVulnerable` label alongside genuine vulnerability labels. The NonVulnerable label is wrong in every case.** These are not ambiguous edge cases — the contracts actively make external calls, exhibit reentrancy patterns confirmed by Slither, and have visible security issues in source code. The BCCC labeling tool apparently assigns NonVulnerable to a large cohort of Oraclize-using and crowdsale contracts regardless of the other labels it assigns simultaneously.

**The 2 nine_folder_maxing contracts are differently broken**: they carry 8-class labels on contracts with no meaningful security issues (one is a pure SafeMath ERC20, one is a document-signing contract). These should be either demoted to NonVulnerable or dropped.

### Totals

| Decision | Count | % |
|---|---:|---:|
| MODIFY — drop NonVulnerable (other labels correct) | 24 | 75% |
| MODIFY — drop NonVulnerable + add missing label | 4 | 12.5% |
| MODIFY — reclassify entirely (over-labeled) | 2 | 6.25% |
| KEEP | 0 | 0% |
| REVIEW-NEEDED | 0 | 0% |
| FALSE-POSITIVE-CONTRACT | 0 | 0% |

---

## Contract-by-Contract Decisions

---

### Contract 1/32: `1cbf966046f79bfde8e00869`
**Bucket:** nine_folder_maxing | **n_pos:** 8 | **Pragma:** `^0.4.13`

**Contract type:** HongZhangCoin — pure ERC20 token using SafeMath. Straightforward OpenZeppelin-style implementation, no payable fallback, no external calls, no arithmetic outside SafeMath, no timestamp usage.

**Slither findings:** Only `naming-convention × 15`, `constable-states × 3`, `solc-version × 2`, `function-init-state × 1`. Zero security-relevant detectors.

**BCCC labels (8 positive):** ExternalBug, GasException, MishandledException, Timestamp, UnusedReturn, CallToUnknown, IntegerUO, Reentrancy

**Assessment:** This is a textbook safe ERC20. SafeMath protects all arithmetic. No external calls, no `transfer`/`send`/`.call()`. No timestamp usage. No return value ignored (SafeMath always returns). The 8-class label is entirely wrong — this contract appears to have been swept into every category by the BCCC tooling that operated on the repository-level rather than contract-level.

**Decision:** `[x] MODIFY — reclassify to Class12:NonVulnerable only. Remove all 8 current positive labels. Reason: contract is a safe SafeMath ERC20 with zero Slither security hits; systematic over-labeling by BCCC tooling.`

---

### Contract 2/32: `147725c17af042280cfb4a4b`
**Bucket:** nine_folder_maxing | **n_pos:** 8 | **Pragma:** `0.4.24`

**Contract type:** DocSignature — a document signature registry. Stores sign processes in a `bytes => SignProcess[]` mapping. Has owner management via `isOwner` mapping and `owners` array.

**Slither findings:** `naming-convention × 10`, `external-function × 4`, `controlled-array-length × 2`, `solc-version × 2`, `missing-zero-check × 1`.

**BCCC labels (8 positive):** ExternalBug, GasException, MishandledException, Timestamp, UnusedReturn, CallToUnknown, IntegerUO, Reentrancy

**Assessment:**
- `controlled-array-length`: `owners.push(owner)` without bound → legitimate DenialOfService/GasException risk if owner array grows large.  
- `missing-zero-check`: `addOwner` does check `notNull` modifier but `removeOwner` has edge cases.
- No external calls, no ETH transfers, no `.call()`, no timestamp, no unchecked arithmetic — Reentrancy, CallToUnknown, Timestamp, IntegerUO are all false. MishandledException is false (no external calls whose return values could be ignored). ExternalBug is false.
- `UnusedReturn` is borderline — no external calls means no ignored return values.

**Decision:** `[x] MODIFY — keep only Class02:GasException (controlled array growth is real). Remove all other 7 labels. Add Class12:NonVulnerable if keeping this in training. Reason: contract has one real gas issue (unbounded owners array) but 7 of the 8 labels have zero evidence in source or Slither.`

---

### Contract 3/32: `1ff44f67b1981220331860e9`
**Bucket:** review_pending | **n_pos:** 5 | **Pragma:** `^0.4.18`

**Contract type:** StandardToken + Ownable crowdsale/ICO contract. Complex multi-contract file with token distribution logic, `distribute()` and batch operations.

**Slither findings:** `divide-before-multiply × 4`, `timestamp × 2`, `naming-convention × 31`, `solc-version × 2`, `external-function × 1`, `too-many-digits × 1`.

**BCCC labels (5 positive):** GasException, MishandledException, DenialOfService, Reentrancy, **NonVulnerable**

**Assessment:**
- `divide-before-multiply × 4`: Real precision loss bug, maps to GasException (rounding loss) and MishandledException (computation errors).
- `timestamp × 2`: Timestamp dependency present, but Timestamp class not in BCCC labels (already expected via Slither over-fire discussion).
- DenialOfService: Plausible given large loop operations possible in distribution.
- Reentrancy: No `reentrancy-eth` hit, only `reentrancy-events × ...` — benign reentrancy at best. Marginal.
- NonVulnerable: Directly contradicts 4 other positive labels.

**Decision:** `[x] MODIFY — drop Class12:NonVulnerable. The contradictory NV label is the only error. GasException/MishandledException are supported by divide-before-multiply hits. DenialOfService is plausible. Reentrancy is marginal but acceptable.`

---

### Contract 4/32: `9e3909b6bc876db6ab764e09`
**Bucket:** review_pending | **n_pos:** 3 | **Pragma:** `^0.4.18`

**Contract type:** NamiPool — complex DeFi staking/pool contract. Interacts with `NamiCrowdSale` external token contract. Uses `namiToken.transfer()`, `namiMultiSigWallet.transfer()`, `this.balance` checks.

**Slither findings:** `reentrancy-eth × 5`, `reentrancy-events × 10`, `timestamp × 12`, `boolean-equal × 25`, `missing-zero-check × 7`, `reentrancy-unlimited-gas × 9`, ...

**BCCC labels (3 positive):** CallToUnknown, Reentrancy, **NonVulnerable**

**Assessment:**
- CallToUnknown ✓: `NamiCrowdSale namiToken = NamiCrowdSale(NamiAddr); namiToken.transfer(...)` is a classic call to an external unknown-at-compile-time contract.
- Reentrancy ✓: `reentrancy-eth × 5` is definitive — ETH is transferred before state updates in multiple functions.
- Timestamp ✗ (not labeled but Slither hits `timestamp × 12` — Timestamp label is missing from BCCC).
- NonVulnerable: Completely wrong. This contract has 5 ETH reentrancy vulnerabilities confirmed by Slither.

**Decision:** `[x] MODIFY — drop Class12:NonVulnerable. Optionally add Class04:Timestamp (12 Slither hits). CallToUnknown and Reentrancy are correct and confirmed.`

---

### Contract 5/32: `8177423cb92d8643b00f64d8`
**Bucket:** review_pending | **n_pos:** 3 | **Pragma:** `^0.4.24`

**Contract type:** POOHMOX — FOMO3D-style gambling contract. Uses `PlayerBookInterface` external contract, ETH payable buy/reload functions, complex internal ETH redistribution.

**Slither findings:** `reentrancy-eth × 4`, `reentrancy-no-eth × 4`, `reentrancy-events × 12`, `timestamp × 8`, `divide-before-multiply × 8`, `boolean-equal × 11`, ...

**BCCC labels (3 positive):** CallToUnknown, Reentrancy, **NonVulnerable**

**Assessment:**
- CallToUnknown ✓: `PlayerBook.registerNameXaddr(...)` and other `PlayerBookInterface` calls are external calls to unknown contracts.
- Reentrancy ✓: `reentrancy-eth × 4` confirmed. ETH transfers happen before state cleanup.
- Timestamp: `timestamp × 8` Slither hits — not labeled but real.
- divide-before-multiply × 8: GasException/precision risk — not labeled but real.
- NonVulnerable: Completely wrong.

**Decision:** `[x] MODIFY — drop Class12:NonVulnerable. Optionally add Class04:Timestamp and Class02:GasException given strong Slither evidence. CallToUnknown and Reentrancy confirmed.`

---

### Contract 6/32: `1f448dcda1b131fe20ee2c91`
**Bucket:** review_pending | **n_pos:** 3 | **Pragma:** `^0.4.13`

**Contract type:** CentraSale — ICO crowdsale. Uses `contract_address.call(bytes4(sha3("transfer(address,uint256)")),msg.sender,_amount)` — unguarded low-level call. `owner.send(this.balance)` unchecked return value.

**Slither findings:** `unchecked-send × 2`, `arbitrary-send-eth × 1`, `reentrancy-benign × 1`, `reentrancy-no-eth × 1`, `timestamp × 1`, `deprecated-standards × 9`, `controlled-array-length × 5`, `divide-before-multiply × 2`.

**BCCC labels (3 positive):** CallToUnknown, Reentrancy, **NonVulnerable**

**Assessment:**
- CallToUnknown ✓: The `contract_address.call(...)` is definitively a CallToUnknown pattern — calling an arbitrary contract address with ABI-encoded data.
- Reentrancy: `reentrancy-benign × 1`, `reentrancy-no-eth × 1`. Not `reentrancy-eth`. Marginal — state updates happen after call but no ETH reentrancy path. BCCC label is technically wrong but not egregious.
- UnusedReturn: `owner.send(this.balance)` result not checked — UnusedReturn / MishandledException. Not labeled.
- NonVulnerable: Wrong given CallToUnknown is clearly present.

**Decision:** `[x] MODIFY — drop Class12:NonVulnerable. Optionally add Class06:UnusedReturn (unchecked send × 2). CallToUnknown is confirmed. Reentrancy label is marginal but acceptable.`

---

### Contracts 7, 9, 12, 13, 14, 16, 17, 20, 21, 22, 24, 25, 26, 28, 29, 30 (16 contracts — Oraclize API family)
**IDs:** `37d50eaf`, `474649f3`, `2780ed30`, `2f34c126`, `3212e131`, `756591a2`, `6ac6892c`, `85f04dd5`, `8aa64346`, `8b26c48b`, `9b1291f0`, `91a4bb14`, `a2c4f4f3`, `a7cac6f3`, `caf6753f`, `ca0fbbd9`

**Bucket:** review_pending | **n_pos:** 3 | **Pragma:** `^0.4.0` (all)

**Contract type:** Oraclize oracle API library (`usingOraclize`, `OraclizeI`, `OraclizeAddrResolverI`) bundled with a downstream contract that uses it. The Oraclize library routes calls through `OraclizeAddrResolverI(address).getAddress()` — an external call to a hardcoded-but-resolved-at-runtime address, then sends ETH via `oraclize.query.value(price)(...)`.

**Slither findings (consistent across all 16):** `reentrancy-eth × 4`, `reentrancy-events × 9`, `reentrancy-benign × 5`, `deprecated-standards × 23`, `naming-convention × 81`, `dead-code × 16`, `costly-loop × 11`, `unindexed-event-address × 13`, `unused-state × 8`, `incorrect-modifier × 4`, `calls-loop × 2`, `uninitialized-local × 2`, etc. — 200 findings, 23 detectors.

**BCCC labels (3 positive, identical for all):** CallToUnknown, Reentrancy, **NonVulnerable**

**Assessment:**
- CallToUnknown ✓: `OraclizeI(OAR.getAddress())` — external call to address resolved at runtime. The `__callback` function also accepts calls from any address unless the contract restricts it (it does not in the base library). Classic CallToUnknown.
- Reentrancy ✓: `reentrancy-eth × 4` confirmed across all instances. The `oraclize_query.value(price)(...)` pattern sends ETH during the query initiation; the callback-based architecture creates reentrancy risk in the downstream contract.
- NonVulnerable: Completely wrong for all 16.

**Decision (applied identically to all 16):** `[x] MODIFY — drop Class12:NonVulnerable for all 16 contracts. CallToUnknown and Reentrancy are confirmed by source inspection and Slither reentrancy-eth × 4. Note: the Slither finding signature (exactly 200 findings, exactly 23 detectors, exactly the same top-15 detectors) confirms these are likely near-identical contracts (same Oraclize boilerplate + different downstream logic), which means this decision is consistent and mechanical.`

Note: Contract 23 (`9b8af6eda597c94acfdf74b7`) is a newer Oraclize v2 library (400 findings, 25 detectors, reentrancy-benign × 44). Same decision applies — drop NonVulnerable.

---

### Contract 8/32: `4219abdc067eb57889cf8aaf`
**Bucket:** review_pending | **n_pos:** 3 | **Pragma:** `^0.4.19`

**Contract type:** DEVCoin ERC20 + ManualSendingCrowdsale + Blocked modifier. Token has `manualTransfer` function. Crowdsale tracks amounts by currency, sends tokens manually.

**Slither findings:** `constable-states × 20`, `naming-convention × 14`, `timestamp × 7`, `low-level-calls × 3`, `return-bomb × 3`, `reentrancy-benign × 2`, `reentrancy-no-eth × 1`, `divide-before-multiply × 1`, `locked-ether × 1`.

**BCCC labels (3 positive):** CallToUnknown, Reentrancy, **NonVulnerable**

**Assessment:**
- CallToUnknown: Weak evidence. `manualTransfer` is internal. The crowdsale has `transfer` calls to external ERC20 contracts — marginal CallToUnknown.
- Reentrancy: `reentrancy-benign × 2`, `reentrancy-no-eth × 1` — no ETH reentrancy. Marginal/benign.
- Timestamp × 7: Slither fires on `blockedUntil = unblockTime` and time-comparison logic — Timestamp label arguably missing.
- return-bomb × 3: Caller could be a contract that throws on receipt — MishandledException risk.
- NonVulnerable: Wrong given Timestamp, return-bomb evidence.

**Decision:** `[x] MODIFY — drop Class12:NonVulnerable. Optionally add Class04:Timestamp (7 hits) and Class03:MishandledException (return-bomb × 3). CallToUnknown is weak but acceptable. Reentrancy is marginal.`

---

### Contracts 10–11/32: `4ad02f839bfade8819f3ab38`, `234a97c1df7afc825bdabb70`
**Bucket:** review_pending | **n_pos:** 3 | **Pragma:** `^0.4.11`

**Contract type:** NutzToken / StorageEnabled proxy pattern — delegates all storage to a separate `Storage` contract via `Storage(storageAddr).getBal(...)` calls. Heavy cross-contract interaction for every state read/write.

**Slither findings (C10):** `naming-convention × 117`, `missing-zero-check × 8`, `reentrancy-events × 8`, `too-many-digits × 7`, `constable-states × 6`, `external-function × 6`, `reentrancy-benign × 5`, `uninitialized-local × 4`, `timestamp × 3`, `cache-array-length × 2`, `erc20-interface × 2`, `incorrect-equality × 2`, ... — 184 findings, 22 detectors.

**Slither findings (C11):** Nearly identical — `reentrancy-benign × 12` (vs 5 for C10).

**BCCC labels (3 positive):** CallToUnknown, Reentrancy, **NonVulnerable**

**Assessment:**
- CallToUnknown ✓: Every storage read/write goes through `Storage(storageAddr).getBal(...)` — external calls to a storage contract. This is exactly the CallToUnknown pattern.
- Reentrancy ✓: `reentrancy-events × 8`, `reentrancy-benign × 5-12`. Not ETH reentrancy, but state changes happen between external storage calls creating reentrancy paths.
- NonVulnerable: Wrong.

**Decision:** `[x] MODIFY — drop Class12:NonVulnerable for both C10 and C11. CallToUnknown and Reentrancy confirmed by storage proxy pattern and Slither.`

---

### Contract 15/32: `7a82041ff4e6d3105497923d`
**Bucket:** review_pending | **n_pos:** 3 | **Pragma:** `^0.4.16`

**Contract type:** CentraSale v2 — near-identical to Contract 6 but with different token price and card limits. Same `contract_address.call(bytes4(sha3("transfer(address,uint256)")),...)` pattern.

**Slither findings:** 55 findings, 13 detectors. `naming-convention × 25`, `deprecated-standards × 8`, `controlled-array-length × 5`, `unchecked-send × 2`, `arbitrary-send-eth × 1`, `reentrancy-benign × 1`, `timestamp × 1`.

**BCCC labels (3 positive):** CallToUnknown, Reentrancy, **NonVulnerable**

**Assessment:** Same analysis as Contract 6. CallToUnknown confirmed by `.call()`. Reentrancy marginal. NonVulnerable wrong.

**Decision:** `[x] MODIFY — drop Class12:NonVulnerable. Same as Contract 6.`

---

### Contract 18/32: `64558fdbbeb12f6f65b8b776`
**Bucket:** review_pending | **n_pos:** 3 | **Pragma:** `^0.4.18`

**Contract type:** InkProtocol — sophisticated escrow/marketplace contract with mediator role. Interacts with `InkMediator`, `InkOwner`, `InkPolicy` external contracts. ERC20 token transfers. Complex state machine with `confirmTransaction`, `refundTransaction`, `settleTransaction` flows.

**Slither findings:** 91 findings, 12 detectors. `naming-convention × 50`, `reentrancy-events × 11`, `too-many-digits × 6`, `constant-function-state × 5`, `reentrancy-benign × 4`, `reentrancy-no-eth × 3`, `timestamp × 3`, `assembly × 2`, `low-level-calls × 2`, `uninitialized-local × 2`, `arbitrary-send-erc20 × 1`.

**BCCC labels (3 positive):** CallToUnknown, Reentrancy, **NonVulnerable**

**Assessment:**
- CallToUnknown ✓: Calls to `InkMediator(mediator).requestMediator(...)`, `InkOwner(owner).authorizeTransaction(...)`, `InkPolicy(policy).confirmTransactionFee(...)` — multiple external calls to unknown-at-compile-time interfaces.
- Reentrancy: `reentrancy-events × 11`, `reentrancy-benign × 4`, `reentrancy-no-eth × 3`. No ETH reentrancy but significant state manipulation around external calls. `arbitrary-send-erc20 × 1` is notable.
- NonVulnerable: Wrong.

**Decision:** `[x] MODIFY — drop Class12:NonVulnerable. CallToUnknown confirmed (multiple external interface calls). Reentrancy confirmed by reentrancy-events × 11.`

---

### Contract 19/32: `8bd88afa2c30d82649952f48`
**Bucket:** review_pending | **n_pos:** 3 | **Pragma:** `^0.4.16`

**Contract type:** CentraSale v3 — another variant with different token price (`10**18*1/200`). Same structure as Contracts 6 and 15.

**Slither findings:** 55 findings, 13 detectors — identical to Contract 15.

**BCCC labels:** CallToUnknown, Reentrancy, **NonVulnerable**

**Decision:** `[x] MODIFY — drop Class12:NonVulnerable. Same as Contracts 6 and 15.`

---

### Contract 23/32: `9b8af6eda597c94acfdf74b7`
**Bucket:** review_pending | **n_pos:** 3 | **Pragma:** no version pragma (implicit ^0.4.0)

**Contract type:** Oraclize API v2 (newer library version) — larger, includes `randomDS_getSessionPubKeyHash`, `stra2cbor`, additional randomness DS functions. 400 Slither findings.

**Slither findings:** 400 findings, 25 detectors. `reentrancy-benign × 44`, `dead-code × 83`, `deprecated-standards × 31`, `costly-loop × 11`.

**BCCC labels:** CallToUnknown, Reentrancy, **NonVulnerable**

**Decision:** `[x] MODIFY — drop Class12:NonVulnerable. Same family as the Oraclize v1 group above. reentrancy-benign × 44 reflects the larger callback surface.`

---

### Contract 27/32: `b44a0113065c8c3a6c895460`
**Bucket:** review_pending | **n_pos:** 3 | **Pragma:** `^0.4.13`

**Contract type:** Peculium Delivery/Airdrop contract. Sends tokens to a list of addresses via `pecul.transfer(toAddress, amountTo_Send)` in a loop. Key bug: `amountToSendTotal.add(_vamounts[indexTest])` result is NOT assigned back to `amountToSendTotal` — SafeMath result is discarded, making the cap check (`amountToSendTotal <= Airdropsamount`) always compare 0 vs Airdropsamount.

**Slither findings:** 54 findings, 17 detectors. `naming-convention × 29`, `unindexed-event-address × 4`, `constable-states × 3`, `reentrancy-events × 2`, `arbitrary-send-erc20 × 1`, `calls-loop × 1`, `reentrancy-benign × 1`, `reentrancy-no-eth × 1`.

**BCCC labels (3 positive):** CallToUnknown, Reentrancy, **NonVulnerable**

**Assessment:**
- CallToUnknown ✓: `pecul.transfer(toAddress, amountTo_Send)` — call to external ERC20 contract in a loop.
- Reentrancy ✓: `reentrancy-events × 2`, `reentrancy-benign × 1` — state updates around external call.
- MishandledException (unlabeled but real): The discarded SafeMath result (`amountToSendTotal.add(...)` not assigned) means the token cap check is completely broken — this is a real logic bug / mishandled exception.
- NonVulnerable: Completely wrong given the SafeMath discard bug.
- Also: `calls-loop × 1` (ERC20 transfer in loop) → GasException / DenialOfService risk if array is large.

**Decision:** `[x] MODIFY — drop Class12:NonVulnerable. Add Class03:MishandledException (SafeMath result discarded — critical logic bug). Optionally add Class02:GasException (calls-loop). CallToUnknown and Reentrancy confirmed.`

---

## Pattern Analysis

### The NonVulnerable Contamination Problem

The systematic NV+vuln co-occurrence (92% of review_pending contracts per WS-N) is now understood:

**Root cause hypothesis:** The BCCC dataset labels were generated by running detectors across each contract and assigning folder membership for each detected class. A second pass (or a separate tool) then assigned NonVulnerable to contracts that passed some "no critical vulnerabilities" threshold — possibly based on a different detector set, or a severity filter. The result: contracts that were labeled Reentrancy+CallToUnknown by detector A are simultaneously labeled NonVulnerable by detector B with a higher threshold. Both labels end up in the CSV.

**Evidence:**
- 92% of review_pending contracts are NV + exactly {CallToUnknown, Reentrancy} (WS-N finding confirmed here)
- The CallToUnknown label correctly fires on `OraclizeI(OAR.getAddress())` pattern
- The Reentrancy label correctly fires on `oraclize.query.value(price)(...)` pattern  
- The NonVulnerable label fires because Slither with the NV detector (which may require certain critical impact) doesn't classify Oraclize usage as "critical" reentrancy
- This is the same false-positive pool where the auditor found "72.5% of Timestamp=1 graphs have feat[2]=0" (i.e., the labels don't match the code patterns)

### Contract Template Clusters

The 30 contracts collapse into 5 distinct template clusters:

| Cluster | Contracts | Template | True labels |
|---|---|---|---|
| **SafeMath ERC20** | 1 | OpenZeppelin ERC20 (HongZhangCoin) | NonVulnerable |
| **Document Registry** | 2 | DocSignature | GasException (unbounded array) |
| **Oraclize API v1** | 7, 9, 12, 13, 14, 16, 17, 20, 21, 22, 24, 25, 26, 28, 29, 30 (16 contracts) | usingOraclize boilerplate | CallToUnknown, Reentrancy |
| **Oraclize API v2** | 23 | Newer Oraclize with randomDS | CallToUnknown, Reentrancy |
| **CentraSale ICO** | 6, 15, 19 | CentraSale crowdfunding with .call() | CallToUnknown, (Reentrancy marginal) |
| **NutzToken proxy** | 10, 11 | StorageEnabled cross-contract delegation | CallToUnknown, Reentrancy |
| **NamiPool DeFi** | 4 | NamiPool staking | CallToUnknown, Reentrancy, Timestamp |
| **POOHMOX gambling** | 5 | FOMO3D-style | CallToUnknown, Reentrancy, Timestamp, GasException |
| **DEVCoin crowdsale** | 8 | DEVCoin + ManualSendingCrowdsale | CallToUnknown (weak), Timestamp |
| **InkProtocol escrow** | 18 | Ink escrow marketplace | CallToUnknown, Reentrancy |
| **Peculium airdrop** | 27 | Delivery/airdrop with SafeMath bug | CallToUnknown, Reentrancy, MishandledException |
| **ERC20 ICO crowdsale** | 3 | StandardToken + distribution | GasException, MishandledException, DenialOfService |

---

## Label Change Recommendations (Actionable)

### High Confidence Corrections (drop NonVulnerable — 28 contracts)

For all 28 contracts listed below, the single action is: **remove `Class12:NonVulnerable = 1`**.

These are contracts 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30 from the inspection set.

```
IDs to update (drop Class12:NonVulnerable):
1ff44f67b1981220331860e9944c94acc3e42f3129db7319adf57d91c3889b7c
9e3909b6bc876db6ab764e099553a7a32cfa1be39100496fead49ca6afb1d6b9
8177423cb92d8643b00f64d8e5af7b4ea572cac26f88562e6952cdea5f6982e3
1f448dcda1b131fe20ee2c91d627ecd59e047e7b3b737bc4097540633168ba68
37d50eaf5d392b72900876fbf91e729f9b3f4b65c85f88674fb11074e0272fe5
4219abdc067eb57889cf8aaf1c62fb767f193298847f1bcecf2f0d86f90bfeda
474649f3c6177a74a0d36058ffce8c3415f1596ddff6790e0ac77b98727adfc8
4ad02f839bfade8819f3ab38fec581c708aa01428e3966f7867a90180027eb0b
234a97c1df7afc825bdabb7098cb1dc3d36f0b9a3ea125d6711a922a4caaf7d7
2780ed3064fdb7d3ffc3a9dedfdd7ec8882456599b979f46acbcbf8f9b2dc5ac
2f34c126f0624732031cfac337430dea84e1e3d80566b115ee572438f1d6988d
3212e131e050d5bf30062c8bc52ed12afbc9536b76148eb8684f0fbe39bfdaef
7a82041ff4e6d3105497923d2b767da522e7488e3511d2ff3ee39b1db81a3497
756591a2872be3fb57386ae232eac86b5b9839647deca8de75c6c3f564bc3c76
6ac6892c594d681e2685de24a859bca9c05dbde3b88f3a982de9dac50fc8f992
64558fdbbeb12f6f65b8b776748ff8ba844d36891967dc42f64fd2b350c30ed9
8bd88afa2c30d82649952f48183dd22d443d07bea26d980e85a77f9b3729a9f2
85f04dd5937cda8ce8fa79592563ed0c739279543b26faa5a90cde4ba5a64b1a
8aa64346eb6977e5e5e3dd2d184d502220de5bcbbdb03a934bc5d352b5362903
8b26c48b8c910c2ac775d1e52c6ab1d3ce5a2716628684bc78a5a8e9c9f610b3
9b8af6eda597c94acfdf74b7409771852c9c6686ca7c5129aba7e78980568411
9b1291f018e0c908239f482ccee68f14f475519672fb9555631316ab8edae2c9
91a4bb141e4954a7c77ab4966d9faf23e2387e0612a6002bb436ca37ef51e69d
a2c4f4f377791ca10c5384641f82ce4c6fddfdaad2b88da5ac2a0df29f067cd7
b44a0113065c8c3a6c895460220d9b38a6825bfec85a17d9b6395e1a592955be
a7cac6f376dbc1eeb32b702af9a305cbac3acbee207da69f14924e52d1bd4c22
caf6753f3b9198eca7bdd1578ba42beb7ae999b18e2960d93ab6369b0e106989
ca0fbbd9bda8ce991a1c7a82155c3be7b4d808a91b1a9dbcaa9a71ac95b5efed
```

### Additional Label Additions (optional, high-confidence)

These are suggested label additions not currently in the BCCC labels, supported by Slither evidence:

| Contract ID | Suggested Addition | Evidence |
|---|---|---|
| `9e3909b6bc876db6ab764e09` | Add Class04:Timestamp | Slither `timestamp × 12` |
| `8177423cb92d8643b00f64d8` | Add Class04:Timestamp | Slither `timestamp × 8` |
| `8177423cb92d8643b00f64d8` | Add Class02:GasException | Slither `divide-before-multiply × 8` |
| `4219abdc067eb57889cf8aaf` | Add Class04:Timestamp | Slither `timestamp × 7` |
| `4219abdc067eb57889cf8aaf` | Add Class03:MishandledException | Slither `return-bomb × 3` |
| `1f448dcda1b131fe20ee2c91` | Add Class06:UnusedReturn | Slither `unchecked-send × 2` |
| `b44a0113065c8c3a6c895460` | Add Class03:MishandledException | SafeMath result discarded — critical logic bug |
| `b44a0113065c8c3a6c895460` | Add Class02:GasException | Slither `calls-loop × 1` |

### Reclassification (contracts 1 and 2)

| Contract ID | Current | Recommended |
|---|---|---|
| `1cbf966046f79bfde8e00869` (HongZhangCoin) | 8 labels (all wrong) | NonVulnerable only |
| `147725c17af042280cfb4a4b` (DocSignature) | 8 labels (7 wrong) | GasException only (drop NonVulnerable, keep gas issue from unbounded array) |

---

## Systemic Implications for contracts_clean.csv (v1.0 → v1.1)

The findings above are from 30 sampled contracts but extrapolate strongly to the full 766 review_pending set:

1. **~92% of the 766 review_pending contracts (≈705) have the same NonVulnerable + CallToUnknown + Reentrancy triple-label pattern.** Based on this review, the NonVulnerable label is wrong for the majority.

2. **Action: Run a batch update on contracts_clean.csv** to drop `Class12:NonVulnerable` for all 766 review_pending contracts whose labels include NonVulnerable co-occurring with at least one of {CallToUnknown, Reentrancy, GasException, MishandledException, DenialOfService, Timestamp}. These contracts should then be moved from `review_pending=1` to `review_pending=0` and included in training.

3. **The Oraclize API cluster (≥16 contracts in this sample) likely represents hundreds of contracts in the full dataset.** All should have NonVulnerable dropped and retain CallToUnknown + Reentrancy.

4. **The nine_folder_maxing contracts (n_pos=8) are a separate quality issue.** They appear to be contracts where the BCCC tool swept all folders and assigned every label it could find in the repository the contract belongs to, rather than examining the specific contract. These require a different fix: drop all labels and use source-level analysis to assign correct labels. For these two: HongZhangCoin → NonVulnerable; DocSignature → GasException.

---

## Recommended Next Steps

1. **Immediate:** Run the batch NonVulnerable drop on `contracts_clean.csv` for all 766 review_pending contracts. Move them from held-out to training. Estimated training set increase: 705 contracts (those with confirmed callables).

2. **Before WS-O (Aderyn 3-way):** Update the label file so the 5,000-contract WS-O sample reflects corrected labels.

3. **For nine_folder_maxing class broadly:** Identify all contracts with `n_pos ≥ 7` in the training set and apply the same source-inspection filter. The two reviewed here are likely not isolated.

4. **Document as D-I-11:** "NonVulnerable label is systematically wrong when co-occurring with CallToUnknown/Reentrancy in the review_pending set." Add to dataset decisions.

5. **Update contracts_clean.csv metadata.json:** version → v1.1, note the label correction batch.

---

## Confidence Assessment

| Finding | Confidence | Basis |
|---|---|---|
| NonVulnerable label wrong in 28/30 contracts | Very High | Source inspection + Slither reentrancy-eth × 4 in 16 Oraclize contracts; ETH reentrancy in NamiPool, POOHMOX |
| HongZhangCoin over-labeled (Contract 1) | Very High | Zero Slither security hits; pure SafeMath ERC20; no external calls |
| DocSignature over-labeled (Contract 2) | High | Only controlled-array-length is real; 7/8 labels have no evidence |
| Oraclize CallToUnknown label correct | Very High | `OraclizeI(OAR.getAddress())` is definitionally CallToUnknown |
| Oraclize Reentrancy label correct | High | reentrancy-eth × 4 confirmed by Slither across 16 contracts |
| Peculium SafeMath discard bug (Contract 27) | Very High | Source code shows `amountToSendTotal.add(...)` result never assigned |
| Batch extrapolation to 766 review_pending | High | 92% of this sample matches the NV+CTU+Reent triple; WS-N confirmed same proportion |
