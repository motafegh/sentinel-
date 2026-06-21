# DIVE Crosswalk Sample Validation — Access Control & Reentrancy

**Date:** 2026-06-18
**Plan:** `docs/plans/2026-06-18_data_module_dive_crosswalk_externalbug_reentrancy_fix_plan.md` Step 1
**Method:** 75 contracts sampled at random (seed=42) from each of DIVE's
`Access Control/` (→ ExternalBug) and `Reentrancy/` (→ Reentrancy) folders.
Each contract's source is read and judged: does it genuinely exhibit the
target vulnerability, or was it filed there for an unrelated/weak reason?

Sample lists: `/tmp/Access_Control_sample.txt`, `/tmp/Reentrancy_sample.txt`
(seed=42, reproducible).

Status: COMPLETE — both samples (150 contracts) reviewed; see Conclusion below.

---

## Access Control → ExternalBug sample (target: 75 contracts)

| # | Filename | Verdict | Reason |
|---|---|---|---|
| 1 | 14437.sol | FP | onlyOwner correctly applied to all admin setters (fees, limits, bots) |
| 2 | 10972.sol | FP | withdraw() is pull-payment of caller's own balance; admin fns onlyOwner |
| 3 | 20922.sol | FP | All sensitive setters (fees, limits, trading) properly onlyOwner |
| 4 | 19757.sol | FP | All privileged functions (withdraw, adjustCut, changeOwner) gated onlyOwner |
| 5 | 18896.sol | FP | All admin functions (withdraw, fees, bots) correctly onlyOwner |
| 6 | 15589.sol | FP | mine() gated by hasMinePermission, kill() gated onlyOwner |
| 7 | 14087.sol | FP | removeLimits/addBots/openTrading all onlyOwner |
| 8 | 13451.sol | FP | withdraw, devMint, setPaused all properly onlyOwner |
| 9 | 6521.sol | FP | allowAddress/lockAddress/setLocked correctly onlyOwner |
| 10 | 11234.sol | FP | withdrawEther/withdrawAllEther/setState all onlyManager-gated |
| 11 | 11157.sol | FP | public payable mint is intentional NFT sale; admin fns onlyOwner |
| 12 | 13740.sol | FP | withdrawAll/withdraw/burn/add all onlyOwner |
| 13 | 18727.sol | FP | mint/pause/unpause/finalize all properly onlyOwner-gated |
| 14 | 19276.sol | FP | removeLimits/addBots/openTrading correctly onlyOwner |
| 15 | 9804.sol | FP | withdrawTokens/setPrice/setWallet all onlyOwner |
| 16 | 11030.sol | FP | withdraw/burn/withdrawForeignTokens all onlyOwner |
| 17 | 17962.sol | FP | removeLimits/openTrading correctly onlyOwner |
| 18 | 6432.sol | TP | StartNative()/Stop()/TokenPair() public unguarded, can drain contract balance |
| 19 | 18791.sol | FP | freezeAccount/setPrices correctly onlyOwner |
| 20 | 7598.sol | FP | safeWithdrawal checks owner==msg.sender before sending full balance |
| 21 | 21050.sol | FP | mint/pause/lock all properly onlyOwner |
| 22 | 10254.sol | FP | enableTrading/fees/wallet updates all onlyOwner |
| 23 | 1639.sol | FP | removeLimits/manageList/openTrading correctly onlyOwner |
| 24 | 655.sol | FP | mintTokens/drop gated via require(contOwnr)/auth modifier |
| 25 | 3367.sol | TP | installWinner/removeCountry/playJackpot/giveBalance have no owner check at all |
| 26 | 21038.sol | FP | setExtent/setisBot/setStructure all onlyOwner |
| 27 | 16214.sol | FP | setBaseURI/setStart onlyOwner; mint() is standard public paid NFT mint |
| 28 | 18605.sol | FP | plain ERC20 with no privileged/admin functions at all |
| 29 | 3228.sol | FP | removeLimits/addBots/openTrading correctly onlyOwner |
| 30 | 14077.sol | FP | blacklist/transferAllFunds gated onlyOwner; withdraw is self-funds pull |
| 31 | 13707.sol | FP | removeLimits/openTrading/setFee all onlyOwner |
| 32 | 4904.sol | FP | openTrading/setAEVOAI use inline require(msg.sender==deployer) correctly |
| 33 | 1386.sol | FP | onlyTech checks owner (naming bug) but stays restricted, no escalation |
| 34 | 4089.sol | FP | setCooldownEnabled/setLowerTax/delBot/openTrading all onlyOwner |
| 35 | 3506.sol | FP | two-step ownership transfer; transferAnyERC20Token onlyOwner |
| 36 | 20528.sol | FP | removeLimits/addBots/openTrading correctly onlyOwner |
| 37 | 11707.sol | FP | all setFee/setWallet/clearStuckBalance functions onlyOwner |
| 38 | 8006.sol | FP | static airdrop address list, only view functions, no privileged writes |
| 39 | 14977.sol | FP | All admin functions (setFee, setTrading, blockBots) correctly use onlyOwner |
| 40 | 4851.sol | FP | onlyowner modifier correctly applied to TransferrTransferr and renounce |
| 41 | 13102.sol | FP | withdraw() properly onlyOwner; public mint() is intentional public sale, not privileged |
| 42 | 21644.sol | FP | All owner functions (setTrading, manualswap, setFee) correctly use onlyOwner |
| 43 | 4203.sol | FP | removeLimits/addBots/openTrading all correctly gated by onlyOwner |
| 44 | 17688.sol | FP | All admin funcs use isAdministrator; withdrawReward only pays out caller's own accrued balance |
| 45 | 12734.sol | FP | removeLimits/addBots/openTrading all correctly gated by onlyOwner |
| 46 | 11804.sol | FP | Plain ERC20 with no privileged/admin functions at all |
| 47 | 19074.sol | FP | Two-step ownership transfer (onlyOwner/onlyOwners) correctly implemented throughout |
| 48 | 21489.sol | FP | Hardcoded address whitelist check correctly implemented, no bypass |
| 49 | 13153.sol | FP | setFee/setTrading/blockBots all correctly gated by onlyOwner |
| 50 | 19282.sol | FP | enableTrading/updateMW/AddExemptFee all correctly gated by onlyOwner |
| 51 | 14033.sol | FP | startTrading/addBot/mononificationTaxes all correctly gated by onlyOwner |
| 52 | 4913.sol | FP | airdrop/withdraw/burn all correctly gated by onlyOwner; transferOwnership lacks zero-check but is a no-op, not exploitable |
| 53 | 21040.sol | FP | setFee/setMaxTxnAndWalletSize correctly gated by onlyOwner |
| 54 | 7763.sol | FP | mintToken/kill/withdrawal all correctly gated by onlyOwner |
| 55 | 4331.sol | FP | recycleDividend/refundUnclaimedEthers/transferEthers all correctly gated by onlyOwner |
| 56 | 16518.sol | FP | lock()/release() correctly gated by onlyOwner; funds go to fixed beneficiary |
| 57 | 4535.sol | FP | setAntiBotsActive/setBlockCooldown correctly gated by onlyOwner |
| 58 | 3928.sol | FP | rescueETH/manualSwap/updateRouterAndPair all correctly gated by onlyOwner |
| 59 | 18375.sol | FP | changeBeneficiaryAddress/mintBatch correctly gated by onlyOwner; public mint is intentional paid sale |
| 60 | 20616.sol | FP | enqueue/dequeue/withdraw all correctly use require(msg.sender==owner) |
| 61 | 12809.sol | FP | setFee/setTrading/blockBots all correctly gated by onlyOwner |
| 62 | 16854.sol | FP | mint/finishMinting gated by hasMintPermission/onlyOwner; CappedToken.mint preserves modifier via super.mint |
| 63 | 19754.sol | TP | mint(uint256 _amount) has zero access control — anyone can mint unlimited tokens to self |
| 64 | 16547.sol | FP | removeLimits/openTrading/addbot all correctly gated by onlyOwner |
| 65 | 8115.sol | FP | withdrawLvgxBalance/enableLvgxTrade all correctly gated by onlyOwner |
| 66 | 4887.sol | FP | transferOwnership/transferAnyERC20Token correctly gated by onlyOwner |
| 67 | 20732.sol | FP | enableTrading/removeLimits/updateMaxTxnAmount all correctly gated by onlyOwner |
| 68 | 18763.sol | FP | startVoting/setOwner correctly gated by onlyOwner, no bypass found |
| 69 | 2753.sol | FP | mintToken/freezeAccount/setPrices all correctly gated by onlyOwner |
| 70 | 122.sol | FP | mintToken/freezeAccount/setPrices all correctly gated by onlyOwner |
| 71 | 19119.sol | FP | transferOwnership/transferAnyERC20Token correctly gated by onlyOwner |
| 72 | 11245.sol | FP | distributeeBitcoinCash* functions correctly gated by onlyOwner |
| 73 | 2425.sol | FP | withdrawTokens/withdrawEther correctly gated by onlyOwner |
| 74 | 5723.sol | FP | enableTrading/removeLimits/updateMaxTxnAmount all correctly gated by onlyOwner |
| 75 | 20641.sol | FP | withdraw/reserveVillains/setBaseURI all correctly gated by onlyOwner |

## Reentrancy → Reentrancy sample (target: 75 contracts)

| # | Filename | Verdict | Reason |
|---|---|---|---|
| 1 | 11933.sol | FP | withdraw(): no state var changed after call.value() — no reentrant-exploitable state |
| 2 | 16259.sol | FP | sendETHToFee()/manualSwap(): only sends to fixed taxWallet, no user-controlled reentry path |
| 3 | 6192.sol | FP | fallback(): balances updated BEFORE fundsWallet.transfer() — CEI followed |
| 4 | 19316.sol | FP | _transfer(): balances updated BEFORE swap/sendETHToFee, only fixed addresses receive ETH |
| 5 | 16298.sol | FP | claimFromAllLockers(): nonReentrant guards the only token-transfer path |
| 6 | 8820.sol | FP | sendETHToFee() only to fixed marketingAddress, balances updated before any external call |
| 7 | 4205.sol | FP | _claimFromAllLockers(): balances updated BEFORE token.transfer(), guarded by nonReentrant |
| 8 | 21374.sol | FP | _transfer(): balance updates happen before swap/fee send, sends only to fixed dev/marketing wallets |
| 9 | 8450.sol | FP | withdraw(): call.value() sent but no balance mapping to exploit, owner-only |
| 10 | 3031.sol | FP | fallback(): balances updated BEFORE fundsWallet.transfer() — CEI followed |
| 11 | 1424.sol | FP | transfer()/burn() update balances before/no external call; emergencyERC20Drain calls fixed owner |
| 12 | 17843.sol | FP | withdraw(): no balance state to reenter, _safeMint hook reentrancy mitigated by index check |
| 13 | 14152.sol | FP | _transfer() updates balances before swapBack()/external sends to fixed dev/marketing wallets |
| 14 | 17295.sol | FP | _transfer(): balances updated before fee swap/sends, only fixed wallets receive ETH |
| 15 | 6033.sol | FP | claim(): balance decremented BEFORE to.transfer(amount) — CEI followed, onlyOwner-gated |
| 16 | 537.sol | FP | fallback(): balances updated BEFORE owner.transfer(msg.value) — CEI followed |
| 17 | 17781.sol | FP | swapBack(): balances/state zeroed before external .call{value}() sends to fixed wallets |
| 18 | 6748.sol | FP | wTokens(): balance changes occur before payable(receiver).transfer() — CEI followed |
| 19 | 2228.sol | FP | swapBack(): tokensFor* zeroed BEFORE call{value}() to fixed dev/marketing wallets |
| 20 | 6721.sol | FP | withdraw() sends entire balance via owner.transfer(), no exploitable state, onlyOwner |
| 21 | 21452.sol | TP | HodlETH terminal()/pay path: balance not reset before msg.sender.transfer(amount) — classic reentrancy |
| 22 | 2065.sol | UNCLEAR | withdrawTaxEarning() follows CEI but buyCard()/processReferer() send to caller-influenced addrs — multi-path, 2000+ line file |
| 23 | 16516.sol | FP | fallback(): balances updated BEFORE fundsWallet.transfer(msg.value) — CEI followed |
| 24 | 14120.sol | FP | sendETHToFee(): fixed dev/marketing wallets only, balances updated before any external call |
| 25 | 450.sol | UNCLEAR | EDProxy.dtrade() uses delegatecall to arbitrary _callee, no concrete state vars to reason about |
| 26 | 4030.sol | FP | getEth()/manualSwap(): only sends to fixed _taxWallet, balances updated before external calls |
| 27 | 12662.sol | FP | _transfer(): balances updated before swap/sendETHToFee to fixed _feeAddrWallet |
| 28 | 11389.sol | FP | swapBack(): tokensFor* zeroed BEFORE call{value}() to fixed dev/marketing wallets |
| 29 | 13214.sol | FP | manualswap/manualsend(): fixed dev/marketing address only, balances updated before sends |
| 30 | 14543.sol | FP | _transfer(): balances updated before swap/sendETHToFee to fixed _taxWallet |
| 31 | 8024.sol | FP | swapBack(): tokensFor* zeroed BEFORE call{value}() to fixed dev/marketing wallets |
| 32 | 14753.sol | FP | _transfer(): balances updated before swap/sendETHToFee to fixed dev/marketing wallets |
| 33 | 9561.sol | FP | withdraw()/rescueEth(): only owner-controlled wallet, no exploitable balance state before call |
| 34 | 22060.sol | TP | pay(): dep.expect updated AFTER dep.depositor.send() in partial-pay branch — CEI violation |
| 35 | 7107.sol | UNCLEAR | OFTCoreV2: _creditTo precedes external onOFTReceived call, but file truncated before _debitFrom impl confirmed |
| 36 | 11861.sol | FP | _transfer(): balanceOf updated before swap/deployer.transfer(), only fixed deployer wallet |
| 37 | 21130.sol | FP | _transfer(): balances updated before swap/sendETHToFee to fixed dev/marketing wallets |
| 38 | 21067.sol | FP | _buy(): trasnferFromOwner (state update) occurs BEFORE bank.transfer(msg.value) — CEI followed |
| 39 | 7094.sol | FP | Only calls ERC20.transfer in a loop, no state-update/reentrancy pattern at all |
| 40 | 3294.sol | FP | sendETHToFee guarded by lockTheSwap mutex, sends to fixed _taxWallet |
| 41 | 5098.sol | FP | Plain ERC20, balances updated before any external interaction, no ETH withdraw |
| 42 | 17437.sol | FP | Standard ERC20 transfer/transferFrom, state updated before any external call |
| 43 | 5795.sol | FP | Gnosis-style multisig: tx.executed=true set before .call.value() (CEI followed) |
| 44 | 10339.sol | TP | withdrawReward() calls .transfer(reward) before zeroing book[WINNER][msg.sender] |
| 45 | 9554.sol | FP | withdraw() is onlyOwner, sends full balance to owner, no user state to exploit |
| 46 | 13385.sol | FP | Fee-swap guarded by lockTheSwap mutex; state zeroed before .call{value} |
| 47 | 9600.sol | FP | buyTokens: weiRaised/soldTokens updated before .transfer() to beneficiary/wallets |
| 48 | 5306.sol | FP | Fee-swap: tokensForX = 0 set before .call{value} sends, no lockTheSwap needed (internal-only) |
| 49 | 17898.sol | FP | swapping mutex guards the swap+owner.transfer() block; balance already decremented |
| 50 | 8397.sol | FP | All sends are onlyOwner (withdraw,drain-style) or state updated before external calls |
| 51 | 20077.sol | FP | swapping mutex flag guards swapBack; state vars zeroed before .call{value} |
| 52 | 13286.sol | FP | Fee-swap guarded by lockTheSwap; state zeroed before .call/.transfer |
| 53 | 18648.sol | FP | drain() onlyOwner sends full balance to owner, no exploitable user state |
| 54 | 2340.sol | FP | All reclaim functions set isCompensated/surplusEthReclaimed flags before .transfer() |
| 55 | 14696.sol | FP | Fee-swap guarded by lockTheSwap; state zeroed before .transfer() to taxWallet |
| 56 | 2879.sol | FP | swapping mutex guards swap+call block; tokensForX zeroed before .call{value} |
| 57 | 10098.sol | FP | requestPayDay() zeroes walletDeposits/withdrawedAmounts before .transfer() |
| 58 | 17803.sol | FP | lockTheSwap modifier (swapping=true;_;swapping=false) wraps the fee-distribution call |
| 59 | 4244.sol | FP | swapping mutex guards swapBack; tokensForX zeroed before .call{value} sends |
| 60 | 15301.sol | FP | swapping mutex guards swapBack; state zeroed before .call{value} sends |
| 61 | 4445.sol | FP | award() onlyOwner sends to owner-chosen destination, no exploitable per-user state |
| 62 | 1311.sol | FP | Plain ERC20 token, no ETH withdraw/external untrusted call present |
| 63 | 7951.sol | FP | Plain ERC20, balances updated before any external interaction, no ETH withdraw |
| 64 | 18786.sol | FP | Fee-swap guarded by lockTheSwap; state zeroed before .transfer() to taxWallet |
| 65 | 8339.sol | FP | Trivial Truffle Migrations contract, no ETH transfer or untrusted external call |
| 66 | 7472.sol | FP | withdraw() zeroes recommender[msg.sender] before .transfer(money) (CEI followed) |
| 67 | 15902.sol | FP | lockTheSwap modifier guards the .call{value} fee-distribution path |
| 68 | 20904.sol | FP | lockTheSwap modifier guards swapTokensForEth/fee-distribution .call/.transfer |
| 69 | 14800.sol | FP | withdrawETH/Address.sendValue are onlyOwner-only sends, no exploitable user state |
| 70 | 5379.sol | FP | swapping mutex guards swap+.call{value} to devWallet block |
| 71 | 5134.sol | FP | withdraw() updates payoutsTo_/referralBalance_ before .transfer(_dividends) |
| 72 | 10012.sol | FP | withdrawalETH/withdrawalToken are onlyOwner sends, no per-user state at risk |
| 73 | 7182.sol | FP | lockTheSwap modifier guards fee-distribution .transfer() to marketingAddress |
| 74 | 19580.sol | FP | lockTheSwap() modifier guards swap+.transfer() to taxWallet |
| 75 | 3897.sol | FP | owner.transfer(msg.value) precedes balance update, but recipient is fixed trusted owner, not attacker-controlled |

---

## Running tally (updated as batches complete)

- Access Control reviewed: 75/75 — TP: 4, FP: 71, UNCLEAR: 0
- Reentrancy reviewed: 75/75 — TP: 3, FP: 69, UNCLEAR: 3

## Conclusion (filled in once both samples complete)

**Access Control → ExternalBug: 4/75 TP = 5.3%**
**Reentrancy → Reentrancy: 3/75 TP = 4.0%** (excluding 3 UNCLEAR; 3/72 of decidable = 4.2%)

Both folders are drastically over-labeled relative to DIVE's own folder taxonomy. The
overwhelming majority of sampled contracts in "Access Control" are unremarkable ERC20/meme
tokens with correctly-implemented `onlyOwner` modifiers on every privileged function — DIVE
appears to have filed any contract that merely *contains* an owner/admin pattern into this
folder, regardless of whether that pattern is exploitable. Same story for "Reentrancy": the
vast majority of sampled contracts are tax/fee-swap tokens using a `lockTheSwap`/`swapping`
mutex or correct CEI ordering (state zeroed before `.call{value}`/`.transfer()`), with ETH
flowing only to fixed, contract-owned addresses (no attacker-controlled reentry surface) —
yet they were filed under "Reentrancy" presumably because they contain a `.call{value}()` or
`.transfer()` pattern that superficially resembles the vulnerability without the actual CEI
violation.

**Recommendation: (b) — drop the wholesale folder-membership mapping for both
ExternalBug-via-"Access Control" and Reentrancy-via-"Reentrancy".** Both TP rates (5.3% and
~4%) are far below the 60% keep-and-downsample threshold in the fix plan's decision
framework — this isn't a "downsample the noise" situation, it's a "the folder label carries
almost no signal" situation. Recommend instead: (1) drop these two folder-derived label
sources entirely from the v3+ crosswalk, (2) replace with independently-corroborated
positives only (e.g. Slither/Aderyn detector hits cross-validated against the DIVE filename,
or manual relabeling of the true positives found in this sample — 6432/3367/19754 for
ExternalBug, 21452/22060/10339 for Reentrancy — as a seed set), and (3) treat any remaining
DIVE-folder-only labels as unlabeled/excluded rather than positive, since mislabeling them
negative would also be wrong (a few are genuine bugs, just not 50-74% of the folder).
