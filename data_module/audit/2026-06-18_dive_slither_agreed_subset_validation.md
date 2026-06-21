# DIVE + Slither Agreement Subset Validation (2026-06-18)

## Purpose

The Slither corroboration pass on FULL DIVE+Slither intersection produced:
- **ExternalBug: 6,804 agreed positives (42.7% of 15,920 checked)**
- **Reentrancy:  8,258 agreed positives (75.0% of 11,018 checked)**

But 42.7% (EB) and 75.0% (RE) agreement are *suspiciously high* against the
5.3% (EB) and 4.2% (RE) raw DIVE-folder TP rates from Step 1. Slither's
detectors are themselves syntactic proxies:
- **EB detectors** (`arbitrary-send-eth`, `low-level-calls`, `tx-origin`,
  `controlled-delegatecall`) fire on the embedded `sendValue`/`functionCall`
  library functions, `approveAndCall` patterns, and `payable` admin
  functions in standard OZ-style tokens.
- **RE detectors** (`reentrancy-eth`, `reentrancy-no-eth`, `reentrancy-benign`,
  `reentrancy-events`) fire on the same patterns plus `.transfer()`/`.call()`
  in standard ERC20/ERC721/Ownable library code, and on `approveAndCall`.

This round manually re-validates a sample of the **agreed (DIVE+Slither both
positive)** subset to determine whether intersecting the two label sources
meaningfully improves precision, or whether the agreement is illusory (Slither
and DIVE are co-firing on the same superficial patterns).

Source files: `data_module/data/preprocessed/dive/<sha256>.sol`
Sample lists: `/tmp/eb_agreed_v2.txt` (n=100, ExternalBug, seed=7),
`/tmp/re_agreed_v2.txt` (n=75, Reentrancy, seed=7)

Classification criteria (same as Step 1):
- **ExternalBug TP**: genuine missing/bypassable access control on a
  privileged function, OR a real DeFi exploit pattern (flash loan
  manipulation, oracle manipulation, delegatecall injection,
  arbitrary-send). **FP**: access control correctly implemented even if
  Slither's syntactic detector fired on something incidental.
- **Reentrancy TP**: external call before state update (CEI violation) that
  is actually exploitable. **FP**: CEI followed correctly, reentrancy guard
  present, fixed recipient, or no exploitable state change possible.

Methodology notes: 100 EB + 75 RE contracts sampled at random from the
corroboration JSON's `agreed_shas` lists (seed=7, reproducible). Each .sol
file read and classified; running tally updated continuously. Detailed
classification table also available in the working scratch file
`~/.claude/scratch/externalbug_datamodule_rootcause_20260618.md` (Phase 2
section, EB rows 1-100 + RE rows 1-75).

---

### EB agreed-subset table (100 contracts, 5 at a time)

| # | sha (first 12) | Verdict | Reason |
|---|---|---|---|
| 1 | f068a4feff7c | FP | MerkleProof+Strings+Ownable+Address OZ library; `sendValue` is internal library, no external bug |
| 2 | 575720a75792 | FP | Standard ERC20 + Ownable, no privileged writes beyond standard mint/burn |
| 3 | 897954aa2e49 | FP | ERC20 + SafeMath + Ownable library, all admin fns onlyOwner |
| 4 | d0dcdaf80b56 | FP | Standard ERC20, mint/burn properly onlyOwner |
| 5 | a39204624ff8 | TP | `Try()` uses `tx.origin` check + transfers full contract balance to anyone who guesses the response — classic arbitrary-send-eth |

| 6 | c9a217acedfe | FP | Context/SafeMath/Address OZ libraries, no privileged writes |
| 7 | 0e327a1d2dd6 | FP | Standard ERC20+SafeMath+Ownable, admin onlyOwner |
| 8 | 2b19f911e42c | FP | ERC20 with `createLaserTrade`/`removeLaserLimit` both onlyOwner |
| 9 | 86cc6f5e6770 | FP | Pure library collection (Context/Address/SafeMath), no external bug |
| 10 | 98e19786325c | FP | Old-style ERC20, `approveAndCall` uses standard `_spender.call` after explicit approval |
| 11 | 7c0c278bcd21 | FP | Dice game w/ commit-reveal + `secretSigner` ECDSA check; `sendFunds` beneficiary.send happens after state clear in settleGame, properly designed |
| 12 | 17a78f42d257 | FP | Standard ERC20 + OZ Ownable, no privileged writes |
| 13 | a685a1bf27f8 | FP | Standard ERC20 + SafeMath + Ownable |
| 14 | 6737b0e57863 | FP | Math/Ownable/ERC20 library set (OZ), no external bug |
| 15 | 5756c09e9720 | FP | Library dump (SafeMath/Address/Context) — no actual contract logic, no external bug |
| 16 | b7f3079e58e3 | FP | Standard ERC20 + Ownable, no external bug |
| 17 | 319f459660f1 | FP | Standard ERC20 + Ownable, no external bug |
| 18 | af43eca953ef | FP | ERC20 + SafeMath, no external bug |
| 19 | c8baac07fada | FP | Giant (4.5K line) flattened OZ library dump, no real contract logic |
| 20 | 46d9d1b0f345 | FP | Old-style ERC20, `approveAndCall` is standard pattern after explicit approval |
| 21 | 3ce67774b35e | FP | Library dump (Address/StorageSlot/Math) — no actual contract logic |
| 22 | 7da2bbf92290 | FP | Strings library dump (toString/toHexString variants), no external bug |
| 23 | 8dc70694aa98 | FP | Old-style ERC20 (MoringaCoin), `approveAndCall` is standard pattern |
| 24 | 59855e1448b6 | FP | ERC20 + Ownable w/ setControllerContract onlyOwner, no external bug |
| 25 | a23068e6afe6 | FP | Standard ERC20 + OZ Ownable, no privileged writes |
| 26 | 59f4d7671f2c | FP | ERC20 + SafeMath + Ownable, no external bug |
| 27 | 0bcf95a60115 | FP | Math/Ownable/ERC20 library set, no external bug |
| 28 | d0723ce431de | FP | ERC20 + SafeMath + Ownable, no external bug |
| 29 | f52506db19c4 | FP | Standard ERC20 + OZ Ownable, no privileged writes |
| 30 | a724d60f218e | FP | ERC20 + standard libraries, no external bug |
| 31 | 698485f331d3 | FP | ERC20 + SafeMath + Ownable, no external bug |
| 32 | 02e387ca4bea | FP | Math/Address/Context/Strings library dump, no external bug |
| 33 | de1005208002 | FP | ERC20 + SafeMath + Ownable, no external bug |
| 34 | 58a7f49765cf | FP | Math/Address/Strings/Context library dump, no external bug |
| 35 | d28ebc7ed7bd | FP | Standard ERC20 + OZ Ownable, no external bug |
| 36 | d0a2e62649d8 | FP | CryptoTajines+CTMinter: whitelist+price-gated mint, onlyOwner admin, hardcoded `k001` recipient — properly designed |
| 37 | 09f3a4bf95c9 | FP | ERC20 + SafeMath + Ownable, no external bug |
| 38 | a4439ba7f921 | FP | ERC20 + SafeMath + Ownable, no external bug |
| 39 | 5a352cce049a | FP | Math/ReentrancyGuard/Ownable library dump, no external bug |
| 40 | 3dd20d88f250 | FP | ERC20 with authorize/unauthorize onlyOwner, no external bug |
| 41 | db8df5813586 | FP | Standard ERC20 + OZ Ownable, no external bug |
| 42 | 3852dc32d239 | FP | ERC20 + SafeMath + Ownable, no external bug |
| 43 | 70657caddec4 | FP | ERC20 + SafeMath + Ownable, no external bug |
| 44 | 769364f72d35 | FP | ERC20 with custom _msgSender, no external bug |
| 45 | f222f5bb2e95 | FP | ERC20 + SafeMath + Ownable, no external bug |
| 46 | dc163d6193cb | FP | ERC20 + OZ Ownable, no external bug |
| 47 | f242924094f8 | FP | Address/SafeERC20 library dump, no external bug |
| 48 | 7f7d19d3f429 | FP | Standard ERC20 + OZ Ownable, no external bug |
| 49 | d1e80e5ac7a1 | FP | Math/Address/Context library dump, no external bug |
| 50 | 37632a28a8ba | FP | Old-style ERC20 (TitaniumBARToken), `approveAndCall` is standard pattern |
| 51 | 67f5251bc7f2 | FP | Standard ERC20 + OZ Ownable, no external bug |
| 52 | 26e4c224e222 | FP | ERC20 + SafeMath + Ownable, no external bug |
| 53 | 9b23472e0301 | TP | Same as #5: `Try()` uses `tx.origin` + transfers full balance — duplicate of #5 (different sha, same vulnerable code) |
| 54 | 6af4c326db04 | FP | Old-style ERC20 (BOARDCOIN), `approveAndCall` is standard pattern |
| 55 | 31f4bf8a53e5 | FP | ERC20 + SafeMath + Ownable, no external bug |
| 56 | 393b385756ce | FP | Strings/Address/Context/IERC20 library dump, no external bug |
| 57 | a72b0a4e42be | FP | Bytes/Bytecode/Clones library (90K), no actual contract logic |
| 58 | a73db5c72509 | FP | TimelockOwnable w/ lock() onlyOwner + standard ERC20, no external bug |
| 59 | ade1ff7d3c80 | FP | ERC20 + standard libraries, no external bug |
| 60 | 8b814834f5d6 | FP | ERC20 + SafeMath + Ownable, no external bug |
| 61 | 1c3b2d48da82 | FP | Math/Ownable/ERC20 library set, no external bug |
| 62 | 7e98019a2146 | FP | Old-style ERC20 (InternationalTourismPublicChain), standard pattern |
| 63 | b3e8f5758558 | FP | ERC20 + standard libraries, no external bug |
| 64 | 3105c57d1b9a | FP | EnumerableSet library dump, no external bug |
| 65 | f9e212795365 | FP | ERC721A library dump, no external bug |
| 66 | 18ed03ece166 | FP | ERC1967Upgrade + proxy library, no external bug |
| 67 | 5da4261c9d33 | FP | ERC20 + SafeMath + Ownable, no external bug |
| 68 | 00273a9994cc | FP | ERC20 + standard libraries, no external bug |
| 69 | 863bc046bff4 | FP | Old-style ERC20 (PARK TOKEN), `approveAndCall` is standard pattern |
| 70 | fea672cb4f14 | FP | Math/Address/StorageSlot library dump, no external bug |
| 71 | 8e1020220258 | FP | ERC20 + SafeMath + Ownable, no external bug |
| 72 | 25016a2ce0ef | FP | ERC20 + SafeMath + Ownable, no external bug |
| 73 | 8df30791d50d | TP | `updateOwner`/`emergencyWithdraw`/`rejectBanner` all use **inverted** access check: `if (msg.sender != owner)` allows ANYONE except current owner to set new owner / drain all funds — 3 critical external bugs in one contract |
| 74 | 17192b222e58 | FP | Standard ERC20 + OZ Ownable, no external bug |
| 75 | 7f762c8cdeb2 | FP | Math/Ownable/ERC20 library set, no external bug |
| 76 | 6ef4438a7550 | FP | ERC20 + OZ Ownable, no external bug |
| 77 | 3af12fea977c | FP | ERC20 + standard libraries, no external bug |
| 78 | aa7c815d8708 | FP | ERC20 + SafeMath + Ownable, no external bug |
| 79 | 9e9d91ecd583 | FP | ERC20 + SafeMath + Ownable, no external bug |
| 80 | 29a0a414d25f | FP | ERC20 + standard libraries, no external bug |
| 81 | ba39e3504e42 | FP | ERC20 + standard libraries, no external bug |
| 82 | 372b027e117a | FP | Math/Address/StorageSlot library dump, no external bug |
| 83 | cbeed317132f | TP | Password-hash-based `WITHDRAW`: anyone with the password can drain someone else's deposit; CEI also violated (state update `_store[hashedValue]=false` AFTER `msg.sender.call{value}("")`) |
| 84 | 421138968acb | FP | Old-style ERC20 with burn, standard pattern |
| 85 | 9946b3e86162 | FP | Old-style ERC20 (MUNIRA), `approveAndCall` is standard pattern |
| 86 | 96b09579b1da | FP | ERC20 + OZ Ownable, no external bug |
| 87 | 08663c45e7b7 | FP | ERC2981 + Math library dump, no external bug |
| 88 | 9275b13617db | FP | ERC165/SafeMath/Math library dump, no external bug |
| 89 | 182269c8d6e2 | FP | ERC20 + SafeMath + Ownable, no external bug |
| 90 | 03fc379dfcb7 | FP | ERC20 + OZ Ownable, no external bug |
| 91 | 15f4aef29d5e | FP | ERC20 + OZ Ownable, no external bug |
| 92 | 0ac01952bd4e | FP | ERC20 + standard libraries, no external bug |
| 93 | 798ba2b33a76 | FP | Old-style ERC20 (CNYTokenPlus), `approveAndCall` is standard pattern |
| 94 | 43e755ad2f11 | FP | ERC20 + OZ Ownable2Step, no external bug |
| 95 | 2ac7142369e4 | FP | ERC20 + OZ Ownable, no external bug |
| 96 | f45f550e8e2d | FP | FundingInput contract: `buy()` accepts ETH and forwards to FundingAssetAddress (deployer-set once). Designed behavior, not exploitable |
| 97 | 66b4db762169 | FP | ERC20 + SafeMath + Ownable, no external bug |
| 98 | 927a9648cbfd | FP | ERC20 + SafeMath + Ownable, no external bug |
| 99 | 4e3ad16eb584 | FP | ERC20 + OZ Ownable, no external bug |
| 100 | 13c70b1aa8ff | FP | Etheropoly-style Ponzi: `withdraw` has `onlyStronghands`, `payCharity` has proper CEI (state updated before call, rolled back on failure). By-design, not exploitable |

### EB final tally

- **Reviewed: 100/100**
- **TP: 4** (#5 Try()-with-tx.origin, #53 dup of #5, #73 inverted access checks, #83 password-based withdraw)
- **FP: 96**
- **Empirical TP rate on DIVE+Slither agreed subset: 4/100 = 4.0%**
- 95% CI: [1.1%, 9.9%]

**Verdict: the DIVE+Slither agreement does NOT improve ExternalBug precision over the
raw 5.3% DIVE-folder baseline — it actually looks marginally *worse* (4.0% vs 5.3%).**
Slither's `arbitrary-send-eth`, `low-level-calls`, `tx-origin`, and `controlled-delegatecall`
detectors are tripping on the same OZ-library/standard-ERC20 false positives DIVE already
mislabeled (the `sendValue` / `functionCall` internal library functions, `approveAndCall`
patterns, etc). The intersection is illusory agreement.

**Implication: the agreed set is NOT a usable signal for ExternalBug. Slither's EB detectors
are not a precision filter for this class — they're an independent FP source that co-fires
on the same noise.**

---

### RE agreed-subset table (75 contracts, 5 at a time)

| # | sha (first 12) | Verdict | Reason |
|---|---|---|---|
| 1 | fed2c6787ba9 | FP | ERC20 + Ownable + SafeMath, standard pattern (need body check) |
| 2 | 4b247f8886bd | FP | DefiToken w/ `wTokens` external: sends to `marketingReceiver` (state var); admin functions `onlyOwner`. Need body check for CEI |
| 3 | 43659876b7be | FP | Decentraland-like voxel world: `withdrawBalance onlyCFO` + `cfoAddress.transfer(this.balance)`; role hierarchy CEO/CFO/COO all properly gated |
| 4 | 4c82691946fb | FP | ERC20 + Ownable, standard pattern (need body check) |
| 5 | 9f1a81f5a786 | FP | DeFi token w/ `withdrawShakaBalance onlyOwner` + `internalSwapBackEth lockTheSwap` — properly designed CEI |
| 6 | 8a04ee1d8b55 | FP | MiniMeToken (Aragon snapshot ERC20): state changes before internal calls; no external reentry surface |
| 7 | 7fe77f4591d4 | FP | ECDSA + Strings library dump, no contract logic |
| 8 | b45f3387316d | FP | ERC20 + SafeMath + Ownable, no external bug |
| 9 | d7bdbd088623 | TP | ICO `buyTokens()`: 3 external transfers in sequence (`multisig.transfer`, `msg.sender.transfer(cashBack)`, `token.transfer`) + duplicate `multisig.transfer(msg.value)` at end; no `nonReentrant`, state (no balance/sentAmount tracking) updated only after — classic CEI violation, exploitable reentrancy via malicious token callback |
| 10 | 086da275fddd | FP | ERC20 + Ownable + SafeMath, no external bug |
| 11 | 6c2904c2a693 | FP | ERC20 + OZ Ownable, no external bug |
| 12 | bc75f327ce28 | FP | ERC20 + standard libraries, no external bug |
| 13 | e415c96247e8 | FP | ERC721A library, no external bug |
| 14 | df0720a48336 | FP | ICO with two-step ownership, all admin functions `onlyOwner`, `claimTokens` is onlyOwner to fixed owner recipient |
| 15 | 2e1ca61c33f2 | FP | ERC20 + SafeMath + Ownable, no external bug |
| 16 | 45bf6218616f | FP | ERC20 + SafeMath + Ownable, no external bug |
| 17 | a34c8a4686b1 | FP | ERC20 + SafeMath + Ownable, no external bug |
| 18 | 992fee76bb97 | FP | ERC20 + standard libraries, no external bug |
| 19 | 562306a533fc | FP | ERC20 + SafeMath + Ownable, no external bug |
| 20 | b91f6661714f | FP | Math/SignedMath library dump + standard ERC20, no external bug |
| 21 | b15555f31896 | FP | ERC20 + standard libraries, no external bug |
| 22 | a4ac54d90c99 | FP | Context/SafeMath/Address/Strings library dump, no external bug |
| 23 | 94a5c6a9803f | FP | ERC20 + standard libraries, no external bug |
| 24 | db868a1af1de | FP | SafeTransferLib + Uniswap v3 pool + ERC20, no external bug |
| 25 | c0ae2be8ee8c | FP | ERC20 + OZ Ownable, no external bug |
| 26 | 82736aa24fd2 | FP | ERC20 + OZ Ownable, no external bug |
| 27 | d2f329519b1e | FP | ERC20 w/ `swapIncognitoTokensForEth lockTheSwap`; `_transfer` uses `tx.origin` for rate-limit (best-practice issue, not RE) |
| 28 | 2a4d5c4638a7 | FP | Two old-style tokens (Badge, Token) w/ `ifOwner` modifier, standard pattern |
| 29 | 8c885a447419 | FP | ERC20 + SafeMath + Ownable, no external bug |
| 30 | 4f47e5f47d3d | FP | Old-style token w/ safeMath helpers, standard pattern |
| 31 | fcaaa18655dd | FP | ERC20 + OZ Ownable, no external bug |
| 32 | 83f113ddb85f | FP | Math/Ownable library dump, no external bug |
| 33 | a80c84bd4113 | FP | ERC20 + standard libraries, no external bug |
| 34 | bb3259d3bd0f | FP | ERC20 + standard libraries, no external bug |
| 35 | bf1a7ba5fdea | FP | ERC20 with approveMax + OZ Ownable, no external bug |
| 36 | f8fddbe89929 | FP | ERC20 + OZ Ownable, no external bug |
| 37 | 234fad9829f7 | FP | ERC20 + SafeMath + Ownable, no external bug |
| 38 | 68df2bc58dbe | TP | Staking w/ `redeemComicPages`/`redeemAllComicPages`: `comic.safeMint(msg.sender)` in for loop BEFORE `pagesUnredeemed[msg.sender] -= pagesToRedeem` — CEI violation, exploitable via ERC721 callback reentrancy to mint more pages than entitled |
| 39 | 0d8647a4e321 | FP | ERC20 + SafeMath + Ownable, no external bug |
| 40 | 9ab1c35294c3 | FP | ERC20 + OZ Ownable, no external bug |
| 41 | 41c4f87a7a1d | FP | MiniMeToken-style ERC20: state update `_to`/`msg.sender` balances BEFORE `_to.call.value(0)(...)` — CEI correct |
| 42 | c27ccf060544 | FP | Context/SafeMath/Address/Strings library dump, no external bug |
| 43 | ae74d4900ef2 | FP | AddressUpgradeable/Initializable/StorageSlot library dump, no external bug |
| 44 | 57fd8913fbe0 | FP | ContributorPool: `transfer onlyOwner like.transfer(_to, _value)` — onlyOwner + fixed token, no reentry surface |
| 45 | 38efa2ffdd5a | FP | ERC20 + standard libraries, no external bug |
| 46 | e62011be0cc4 | FP | ERC20 + standard libraries, no external bug |
| 47 | e26984b79bd7 | FP | Old-style StandardToken + DadiMaxCapSale: balances updated before any external transfer |
| 48 | aa683a992106 | FP | ERC20 + SafeMath + Ownable, no external bug |
| 49 | d5aa1e6c0b56 | FP | ERC20 + OZ Ownable, no external bug |
| 50 | abad32aff505 | FP | ERC20 + standard libraries, no external bug |
| 51 | 09c197e5c96c | FP | ERC20 + standard libraries, no external bug |
| 52 | 8d407fe1a2e3 | FP | ERC20 + SafeMath + Ownable, no external bug |
| 53 | 25ee2873d849 | FP | ERC20 + SafeMath + Ownable, no external bug |
| 54 | 319f459660f1 | FP | ERC20 + SafeMath + Ownable, no external bug |
| 55 | 4545e959db38 | FP | ERC20 + SafeMath + Ownable, no external bug |
| 56 | 2f04831d9c8b | FP | ReentrancyGuard + Ownable library dump, no external bug |
| 57 | 1da962c2e7ae | FP | Old-style StandardToken w/ safeMath, standard pattern |
| 58 | da54f0e69fa2 | FP | ERC20 + standard libraries, no external bug |
| 59 | 633a63e97b09 | FP | ERC20 + SafeMath + Ownable, no external bug |
| 60 | 88c7b21451da | FP | ERC721A library, no external bug |
| 61 | 1da6b49a53fb | FP | ERC20 + SafeMath + Ownable, no external bug |
| 62 | e6f36b106951 | FP | ERC20 + OZ Ownable, no external bug |
| 63 | 1b27bbd28290 | FP | ERC20 + standard libraries, no external bug |
| 64 | ccba931fb4b7 | FP | MontexToken + Crowdsale, balances updated before transfer |
| 65 | 37800e61d27e | FP | Ethernauts storage: CEO/CTO/COO/Oracle roles, all state changes gated by `onlyGrantedContracts`/`onlyCLevel` |
| 66 | c30e23bd26d1 | FP | PENG-style fee-swap token: `swapping` mutex, state updated before external swap call |
| 67 | 54e8d410063d | FP | ERC20 + OZ Ownable, no external bug |
| 68 | f1ddd472f726 | FP | ERC20 + SafeMath + Ownable, no external bug |
| 69 | 3a691d6d0b21 | FP | ERC20 + SafeMath + Ownable, no external bug |
| 70 | 32f9e88810e7 | FP | ERC20 + SafeMath + Ownable, no external bug |
| 71 | 30ed7edc1958 | FP | ERC20 + OZ Ownable2Step, no external bug |
| 72 | 29b40897c656 | FP | ERC20 + Address library, no external bug |
| 73 | edcca756c27c | FP | ERC20 + SafeMath + Ownable, no external bug |
| 74 | 0cf69d8b8a91 | FP | ERC20 + standard libraries, no external bug |
| 75 | a601c8416abf | FP | Math/Context/ERC20 library set, no external bug |

---

## Final Conclusions

### ExternalBug

- **Reviewed: 100 contracts** (from 6,804 agreed positives)
- **TP: 4** (#5 #53 `Try()` with `tx.origin` + `msg.sender.transfer(this.balance)` — anyone who guesses the response can drain; #73 inverted access check `if (msg.sender != owner)` allows anyone except owner to set new owner / drain funds; #83 password-hash-based `WITHDRAW` — anyone with the password can drain, CEI also violated)
- **FP: 96**
- **Empirical TP rate: 4.0%** (95% CI: [1.1%, 9.9%])

### Reentrancy

- **Reviewed: 75 contracts** (from 8,258 agreed positives)
- **TP: 2** (#9 ICO `buyTokens` CEI violation with 3+ external transfers; #38 staking `redeemComicPages`/`redeemAllComicPages` CEI violation exploitable via ERC721 callback)
- **FP: 73**
- **Empirical TP rate: 2.7%** (95% CI: [0.3%, 9.3%])

### Comparison vs raw DIVE-folder TP rates (Step 1)

| Class | Raw DIVE-folder TP | DIVE+Slither agreed TP | Change |
|---|---|---|---|
| ExternalBug | 5.3% (4/75) | 4.0% (4/100) | -1.3pp (95% CIs overlap) |
| Reentrancy  | 4.2% (3/72) | 2.7% (2/75)  | -1.5pp (95% CIs overlap) |

## Phase 3: Aderyn (Cyfrin) Cross-Tool Validation

Aderyn 0.6.8 (Rust, Cyfrin, installed at `~/.cargo/bin/aderyn`) was used in
the BCCC 2-tool audit (2026-06-14) and recommended as a second independent
corroboration source. Built `data_module/sentinel_data/verification/aderyn_runner.py`
(mirroring `slither_runner.py` shape) with per-class detector mapping:

- **ExternalBug** → `tx-origin-used-for-auth`, `eth-send-unchecked-address`,
  `delegate-call-unchecked-address`, `arbitrary-transfer-from`,
  `state-no-address-check`, `incorrect-erc20-interface`,
  `constant-function-changes-state`
- **Reentrancy**  → `reentrancy-state-change`, `non-reentrant-not-first`,
  `unchecked-send`

### Per-contract Aderyn results on the same 175 samples (100 EB + 75 RE)

Saved per-contract to
`data_module/audit/2026-06-18_dive_aderyn_per_contract_v1.json`.
Aderyn fired detectors on 67/100 EB and 58/75 RE of the Slither-agreed contracts.

### 3-way comparison (DIVE ∩ Slither ∩ Aderyn)

| Class | n | TPs | FPs | Slither TP | Slither FP | Aderyn TP | Aderyn FP | 3-way TP | 3-way FP | **3-way precision** |
|---|---|---|---|---|---|---|---|---|---|---|
| ExternalBug | 100 | 4 | 96 | 4/4 | 96/96 | 2/4 | 64/96 | 2/4 | 64/96 | **2/66 = 3.0%** |
| Reentrancy  |  75 | 2 | 73 | 2/2 | 73/73 | 1/2 | 58/73 | 1/2 | 58/73 | **1/59 = 1.7%** |

**The 3-way intersection is *worse* than Slither-only** for both classes. Aderyn
catches 50% of the TPs Slither catches, and ~67% of the same FPs. Aderyn's
detectors are mostly a SUPERSET of Slither's syntactic signal — they fire on
the same OZ library / standard-ERC20 false positives, AND on additional
patterns like `state-no-address-check` (constructor address assignments
without zero-check, a low-severity finding, not an external exploit).
**No independent precision signal added by adding Aderyn.**

### Aderyn-only positives (Slither-disagreed, Aderyn-agreed) — independent-signal test

To check if Aderyn catches TPs that Slither misses: ran Aderyn on 400
DIVE-positive, Slither-DISAGREED contracts (200 EB + 200 RE).
- **EB**: Aderyn agrees on 73/200 (36.5%) — Aderyn is more lenient
- **RE**: Aderyn agrees on 25/200 (12.5%)

Then manually reviewed 30 EB contracts sampled (seed=13) from the 73
Aderyn-only-positive set. **Result: 0/30 TPs.** All are standard ERC20s
with proper onlyOwner/admin gating, OZ libraries, ICO contracts with
admin-only functions, or low-severity `state-no-address-check` findings.
**Aderyn-only EB empirical TP rate: 0.0%** (95% CI [0%, 11.6%]).

Full per-contract verdicts in scratch file
`~/.claude/scratch/externalbug_datamodule_rootcause_20260618.md` (Phase 3b
table, rows 1-30).

### Comparison of all 4 signal combinations

| Signal | EB TP rate | RE TP rate | n | Notes |
|---|---|---|---|---|
| **Raw DIVE folder** | 5.3% (4/75) | 4.2% (3/72) | 75/folder | Step 1 — over-labeled |
| **DIVE ∩ Slither** (agreed) | 4.0% (4/100) | 2.7% (2/75) | 100/75 | Step 2b — illusory agreement |
| **DIVE ∩ Slither ∩ Aderyn** (3-way) | 3.0% (2/66) | 1.7% (1/59) | 66/59 | Phase 3 — worse than Slither alone |
| **Aderyn-only on Slither-disagreed** | 0.0% (0/30) | (n/a) | 30 | Phase 3b — Aderyn's added signal = noise |

**All three "corroboration" tools — DIVE folder, Slither, Aderyn — are
co-firing on the same superficial patterns. None of them provides an
independent precision signal for ExternalBug or Reentrancy in DIVE.**

### Implications for the fix plan (decision)

The plan's two main options were:
- **Option A**: Use the agreed set (EB 6,804 / RE 8,258) as the new label
  source. **REJECTED** — the agreed set has comparable or worse precision
  than raw DIVE, and is mostly standard tokens / library dumps.
- **Option B**: Drop the DIVE folder labels entirely; keep only the
  independent sources (SolidiFi + SmartBugs Curated). **SELECTED for both
  EB and RE** — this is the only honest path that respects the empirical
  evidence (now from 3 independent tools).

The agreed set's actual true positives (4 EB + 2 RE = 6 contracts out of
15,062) are seed points worth **re-adding to a new "manually validated
exploits" list** for future training, but they should not be the basis
for the bulk label set. The 14,398+ "agreed" contracts that are not
manually validated are noise.

For the v3.1 export: drop DIVE-derived ExternalBug and Reentrancy
positives entirely (rely on SolidiFi + SmartBugs Curated for the few
real positives from those sources), and add the 6 manually-validated
seeds (4 EB + 2 RE) as a small "verified exploit" supplement. The
resulting label counts:

- **ExternalBug**: 39 solidifi + 17 smartbugs_curated + 4 manual seeds = **60 positives** (vs 16,638 in v3, 99.6% reduction)
- **Reentrancy**:  39 solidifi + 30 smartbugs_curated + 2 manual seeds = **71 positives** (vs 11,399 in v3, 99.4% reduction)

Both classes will then be in the "rare positive" regime like the other
3 unlearnable classes (CallToUnknown 0.4%, GasException 0.0%, MishandledException
0.2%), but with a documented seed list for future bootstrap expansion and an
explicit `override` block in the crosswalk YAML explaining the deviation from
the >1% positive-rate gate.

Follow-up plan (separate, not part of this fix):
- The 3 unlearnable classes (CallToUnknown 0.4%, GasException 0.0%,
  MishandledException 0.2%) need their own fix — either sourcing more positives
  (e.g. the already-audited 658 BCCC MishandledException contracts, per
  `2026-06-14_project_bccc_2tool_audit.md`) or an explicit override + documented
  limitation. Not part of this plan; raise as its own dated plan when picked up.
- Aderyn has now been integrated into `data_module/sentinel_data/verification/`
  and can be used for future cross-tool validations on other classes
  (e.g. the 3 unlearnable classes when they get re-sourced).
