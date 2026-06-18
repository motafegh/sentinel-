# Method 0 — TP/FP/BORDERLINE Criteria v1 (Frozen)

**Status:** FROZEN — 2026-06-19
**Version:** v1
**Governing protocol:** `docs/plan/data_module/CLAUDE.md` §4

**Provenance:** Criteria proposed by AI, confirmed by Ali (2026-06-19). Practical floor ≥10% TP confirmed. Developed through progressive batches: 10 + 20 + 12 targeted = 42 contracts reviewed. 1 TP per class found (consistent with tool-derived label source per M1).

---

## 1. ExternalBug (Access Control)

### Bar: REACHABLE

A privileged function (owner/admin/role-gated) is callable by an unauthorized caller — missing or misconfigured access control on a reachable execution path.

**What counts as privileged:** Functions that modify contract state variables, transfer assets (ETH/tokens), change critical parameters (fees, addresses, thresholds), or execute administrative operations (pause/unpause, upgrade, destroy).

**What counts as authorized:** The function body or its modifiers include an explicit check against the caller's identity or role (`onlyOwner`, `require(msg.sender == owner)`, `onlyRole(...)`, `auth`, `onlyAdmin`, `onlyWallet`). Factory patterns where `msg.sender == factory_contract` count as authorized.

**What does NOT count:** Pattern-matching on Ownable library inheritance alone (the contract HAS auth), standard ERC20/ERC721 interface functions (which don't need auth), or functions that operate only on the caller's own state (e.g., user withdraws their own funds).

### Worked examples

#### TP-1: cid=8822 — Unprotected proxy (13 lines)
```solidity
contract MyProxyContract {
    function ClaimReward(address targetContract, address operator, bool approved) public {
        IERC721(targetContract).setApprovalForAll(operator, approved);
    }
}
```
**Verdict: TP.** `ClaimReward` is `public` with NO access control modifier. ANY caller can invoke `setApprovalForAll` on ANY target ERC721 contract through this proxy. This is a privileged operation (approval management) with no auth.

#### TP-2: None found in 42-contract review — consistent with tool-derived label source (M1)
DIVE Access Control labels overwhelmingly flag contracts where every privileged function IS properly guarded (Ownable tokens, ERC721A NFTs with proper auth, factory-pattern contracts with `msg.sender == factory` checks). The class definition issue identified in MEMORY.md is confirmed: DIVE's "Access Control" folder predominantly contains contracts that USE access control, not contracts that LACK it. A contract with proper `onlyOwner` is NOT a true positive for ExternalBug.

#### FP-1: cid=21797 — Standard OZ Ownable token
OffsetOracle inherits OpenZeppelin Ownable v4.7.0. `adjustPrice(uint256)` is `public onlyOwner`. `getPrice()` is public view (no state change). Every state-changing function is behind `onlyOwner`. **Verdict: FP.** No missing access control — DIVE flagged Ownable library presence.

#### FP-2: cid=19142 — Custom Owned token with 2-step transfer
DDXPToken inherits `Owned` with `onlyOwner` modifier. `transferAnyERC20Token` is `public onlyOwner`. `approveAndCall` is a standard ERC20 extension (not privileged). **Verdict: FP.** All privileged functions guarded. DIVE flagged Owned pattern.

#### BORDERLINE: None found in 42-contract review
The REACHABLE bar produces a clean binary for access control — a function either has auth or it doesn't. Edge cases (role-based auth, factory patterns, multi-sig wallets) are resolved by checking for ANY explicit caller-identity check. No ambiguous cases found.

---

## 2. Reentrancy

### Bar: REACHABLE

An external call is made before state updates (CEI violation) on a reachable path where re-entry could alter contract behavior through the same or a different function.

**What counts as external call:** `.call{value:}()`, `.delegatecall()`, `.staticcall()`, ERC20/ERC721/interface method calls to addresses not under the contract's control, `.transfer()`/`.send()` (2300 gas limited — usually safe, still flagged if combined with state changes after).

**What counts as state change:** Assignment to storage variables, `mapping` writes, token balance transfers (internal), changes to any persistent contract state.

**What counts as reachable path:** The external call and subsequent state change are in the same execution flow. If the external call is in a `public`/`external` function or any function reachable from one, the path is reachable.

**What mitigates to BORDERLINE:**
1. A re-entrancy lock is present AND covers ALL external-call-containing paths (not just one)
2. The external call target is a trusted/known contract (Uniswap router, deployed proxy, verified protocol)
3. The state change after the external call is protected by a guard that limits damage (e.g., only affects the caller's own balance)
4. The CEI is in a constructor (non-exploitable — contract not yet deployed)

**What does NOT mitigate:** A lock that only prevents re-entry through the SAME function but not through OTHER functions (e.g., `swapLock` that prevents re-entering `swap` but not `transfer`).

### Worked examples

#### TP-1: cid=5900 — MultiSig CEI violation
```solidity
function executeTransaction(uint transactionId) public notExecuted(transactionId) {
    if (isConfirmed(transactionId)) {
        Transaction tx = transactions[transactionId];  // storage pointer
        if (tx.destination.call.value(tx.value)(tx.data)) {  // EXTERNAL CALL
            tx.executed = true;  // STATE CHANGE AFTER CALL
            Execution(transactionId);
        } else {
            ExecutionFailure(transactionId);
            tx.executed = false;
        }
    }
}
```
**Verdict: TP.** Classic CEI violation. `tx.destination.call.value(tx.value)(tx.data)` is an external call to an arbitrary destination. `tx.executed = true` is a storage write AFTER the external call. An attacker-controlled `tx.destination` can re-enter `executeTransaction` with the same transaction ID (since `notExecuted` still passes — `executed` hasn't been set). No re-entrancy lock is present. Slither confirmed: `reentrancy-eth` fired on this exact function.

#### TP-2: None found in additional 30 contracts — rare in DIVE single-label RE
The only confirmed CEI-without-mitigation pattern in DIVE is the MultiSig (0.4.11 pre-lock era). All 0.8.x meme tokens with CEI have partial locks (BORDERLINE). This is consistent with the Solidity ecosystem's evolution: post-2018, re-entrancy guards became standard.

#### FP-1: cid=4794 — approveAndCall (CEI respected)
```solidity
function approveAndCall(address _spender, uint256 _value, bytes _extraData) public returns (bool) {
    tokenRecipient spender = tokenRecipient(_spender);
    if (approve(_spender, _value)) {  // STATE CHANGE (allowance update)
        spender.receiveApproval(msg.sender, _value, this, _extraData);  // EXTERNAL CALL AFTER
        return true;
    }
}
```
**Verdict: FP.** The `approve()` call (state change) happens BEFORE `receiveApproval()` (external call). CEI is respected. The external call is to a known ERC223 callback interface.

#### FP-2: cid=5208 — COE swap (state before call)
```solidity
function swap(uint amt) public {
    swapLimit -= amt;  // STATE CHANGE FIRST
    burn(amt);          // STATE CHANGE FIRST
    if (amt.mul(ethSwapRate) > 0) {
        msg.sender.transfer(amt.mul(ethSwapRate).div(offset));  // EXTERNAL AFTER
    }
    if (amt.mul(swapRates[FUTX]) > 0) {
        ERC20(FUTX).transfer(msg.sender, ...);  // EXTERNAL AFTER
    }
}
```
**Verdict: FP.** All state changes (`swapLimit -= amt`, `burn(amt)`) happen BEFORE external calls. CEI respected. Slither flagged `reentrancy-eth` but this is a false positive — the `.transfer()` has 2300 gas and state is already updated.

#### BORDERLINE-1: cid=20724 — Meme token with Uniswap CEI + lockTheSwap
```solidity
function _transfer(address from, address to, uint256 amount) private {
    // ...
    if (canSwap && !inSwap && ...) {
        swapTokensForEth(contractTokenBalance);  // EXTERNAL: Uniswap router
        sendETHToFee(address(this).balance);       // EXTERNAL: .transfer() to wallets
    }
    _tokenTransfer(from, to, amount, takeFee);  // STATE CHANGE AFTER
}
// With modifier lockTheSwap { inSwap = true; _; inSwap = false; }
```
**Verdict: BORDERLINE.** CEI structurally present — external swap before state update. Partial mitigation: `lockTheSwap` prevents re-entering `swapTokensForEth` but NOT `_transfer` itself. The Uniswap router is a trusted contract (not attacker-controlled). Slither confirmed `reentrancy-eth`. Exploit would require the Uniswap router or fee wallets to re-enter `_transfer`, which is unlikely but not impossible. This is the dominant RE pattern in DIVE (5/10 contracts with RE signals).

#### BORDERLINE-2: cid=2948 — DSAuth setAuthority
```solidity
function setAuthority(DSAuthority authority_) public auth {
    authority = authority_;  // STATE CHANGE
}
modifier auth {
    require(isAuthorized(msg.sender, msg.sig));  // calls authority.canCall() EXTERNAL
    _;
}
```
**Verdict: BORDERLINE.** The `auth()` modifier calls `authority.canCall(src, this, sig)` BEFORE the function body. If the current authority is attacker-controlled, `canCall()` could re-enter `setAuthority` during the check. The CEI is: external call in modifier → state change in function body. Mitigation: requires authority to already be compromised. Exploit requires two-step attack: (1) become authority, (2) exploit CEI.

---

## 3. BORDERLINE bucket definition

A contract is BORDERLINE when the vulnerability pattern is structurally present but mitigating factors create genuine uncertainty about exploitability. Applicable factors:

- Re-entrancy lock is present but may not cover all entry points
- External call target is a trusted/known contract (not arbitrary/caller-controlled)
- The CEI is in a constructor (non-exploitable at deployment time)
- The exploited behavior would benefit only the contract owner (self-inflicted)
- The function operates only on the caller's own state/balance

Borderlines are counted and reported separately — NEVER silently folded into TP or FP. A borderline-heavy stratum indicates the criteria need sharper BORDERLINE/TP boundary definition (bump to criteria v2 if >30% of verdicts are borderline).

---

## 4. Decision procedure (pre-committed, from README §8)

**KEEP** a stratum only if its TP Wilson 95% CI is statistically distinguishable from and above the control-arm CI (non-overlapping, stratum higher).

**DROP** if the stratum's CI overlaps the null (indistinguishable from noise).

**ENLARGE the sample** if CIs are too wide to separate.

**Practical floor (confirmed by Ali, 2026-06-19):** Even if statistically significant, a stratum with TP rate < 10% is DROPPED — training on >90% noise is not practically useful.

KEEP is only valid if:
- Method 8 confirmed parser faithfulness (✓ for all 7 DIVE classes)
- Method 2 confirmed folder↔CSV identity (✓ 0 mismatches)

---

## 5. What this criteria does NOT cover

- Timestamp (Timestamp), IntegerUO (Arithmetic), DenialOfService, TransactionOrderDependence, UnusedReturn, CallToUnknown, GasException, MishandledException — these classes are out of scope for Phase 1.
- The EXPLOITABLE bar — the REACHABLE bar is a deliberate middle ground per M1 findings (DIVE labels are tool-derived, not expert-audited). An exploitable bar would produce near-zero TP rates for tool-derived labels.
- Per-contract economic impact assessment — this criteria judges code patterns, not financial damage.

### Record of review

- **42 contracts reviewed** over 3 progressive batches (seed=20260618, seed=20260618_2, seed=20260619_3)
- **EB:** 1 TP, 0 BORDERLINE, 29 FP, 4 control FP (out of 22 EB-labeled + 8 zero-label contracts in batches 1-3)
- **RE:** 1 TP, 9 BORDERLINE, 21 FP, 4 control FP (out of 22 RE-labeled + 8 zero-label contracts in batches 1-3)
- **Tool hints:** Slither (version-matched solc) + Aderyn 0.6.8 (--stdout) run on every contract before manual review
- **Blind second review:** NOT YET DONE — required by CLAUDE.md §5 before Method 4. The 42 pre-judged contracts and this criteria doc form the replication package.

### Criteria version history
- **v1 (2026-06-19):** Initial freeze. REACHABLE bar for EB and RE. BORDERLINE bucket defined. Practical floor ≥10% TP.
