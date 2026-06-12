# Class09: DenialOfService — Ground Truth Definition
**Phase 5 Stage 5.0 | Written:** 2026-06-08
**SENTINEL class index:** 8 (`Class09:DenialOfService`)
**BCCC folder:** `SourceCodes/DenialOfService/` (12,394 contracts)
**Phase 4 FP rate:** 56% in small sample (18 reviewed: 0 KEEP, 10 DROP, 8 UNCERTAIN)
**Tool coverage:** Poor — slither F1=0.013 (calls-loop only); aderyn has NO detector for this class

---

## 1. Canonical Definition

**Reference:** SWC-113 (DoS with Failed Call), SWC-128 (DoS with Block Gas Limit), DASP-5 (Denial of Service)

DenialOfService covers patterns where an attacker can **permanently or indefinitely block a critical contract function** — making it impossible for legitimate users to perform essential operations (withdrawals, refunds, state transitions).

**Boundary with GasException:** DenialOfService = the function is blocked (fails or stalls). GasException = the function runs out of gas due to loop size. A function CAN be both (loop that both exhausts gas AND can be forced to revert on one iteration). When in doubt: if the primary mechanism is loop size → GasException; if the primary mechanism is a forced external call failure → DenialOfService.

---

## 2. Inclusion Criteria

A contract is DenialOfService-positive if it contains a pattern where an attacker can **cause a key function to become permanently inoperable**:

**A. Push-pattern with forced revert (SWC-113):**
```solidity
function refundAll() public {
    for (uint i = 0; i < refunds.length; i++) {
        require(refunds[i].addr.send(refunds[i].amount));  // VULNERABLE
        // If one .send() fails, require() reverts entire loop
        // Attacker becomes one refund recipient and makes their address reject ETH
    }
}
```
Attacker deploys a contract with no `receive()` or a `receive()` that always reverts → causes `require(.send())` to fail → entire refund loop is permanently stuck.

**B. Owner-key DoS — owner can never act:**
```solidity
address public owner;
// owner set to address(0) or unreachable multisig — all onlyOwner functions blocked
```
Included only if the lockout is *exploitable* (e.g., constructor bug sets owner to `address(0)`; attacker can front-run constructor).

**C. State machine lock:**
```solidity
function finalize() public {
    require(msg.sender == lastBidder);  // last bidder controls when auction ends
    // Attacker wins auction with a contract that never calls finalize()
}
```
Attacker controls a required actor and refuses to act, permanently blocking state transition.

**D. Unbounded loop with forced per-iteration revert (combines A + GasException):**
Array grows via public function; one bad entry causes `require(call(...))` to revert → only the bad-entry variant qualifies for DoS (the pure gas-limit variant is GasException).

---

## 3. Exclusion Criteria

| Pattern | Reason excluded |
|---|---|
| `.transfer()` / `.send()` to a single known address (not in a loop over user-provided addresses) | Single failure doesn't permanently block — just one tx reverts |
| `revert()` / `require()` used as normal input validation | Correct defensive programming, not a vulnerability |
| Loop with fixed-size array (no user can grow it) | Not DoS-exploitable |
| Owner pauses/unpauses contract (intentional pausability) | By design; not a vulnerability |
| Ownable contract where owner can be transferred | Transfer available; not permanently locked |
| ERC-20 transfers using `.transfer()` on a single recipient | `.transfer()` to a single address is not a loop DoS pattern |
| Standard ERC-20/ERC-721 with no cross-contract calls in critical paths | BCCC mislabeling — DROP |

---

## 4. Pre-0.8 vs. Post-0.8 Distinctions

- `.send()` and `.call()` behaviour is version-agnostic in terms of the DoS pattern.
- Pre-0.8: `.transfer()` forwards 2300 gas stipend and reverts on failure — makes push-pattern particularly dangerous.
- Post-0.8: same `.transfer()` semantics; pattern unchanged.
- Pre-0.5: `throw` was used instead of `revert()`; functionally equivalent for DoS patterns.

---

## 5. Edge Cases

| Scenario | Verdict | Notes |
|---|---|---|
| Pull-payment pattern (users withdraw their own funds, not pushed) | DROP | Pull pattern is the correct fix for DoS; not vulnerable |
| Loop over addresses array, uses `.call()` without `require()` (silent failure) | DROP | Failure doesn't block the loop; just skips that recipient |
| `require(success)` on `.call()` inside a loop over user-controlled addresses | KEEP | Forced-revert DoS pattern confirmed |
| Contract that iterates over addresses but exits loop on failure (try/catch) | DROP | Failure is handled; loop continues |

---

## 6. Verification Methods

| Method | Detector / Pattern | Expected signal |
|---|---|---|
| **M3 Slither** | `calls-loop`, `reentrancy-unlimited-gas` | Very conservative; F1=0.013 on BCCC sample |
| **M2 Regex** | `\.send\(` or `\.call\(` inside `for\|while`, combined with `require\(` | High recall, many FPs |
| **M9 Manual** | Check: is there a loop over user-controlled addresses? Does one failure abort the whole loop? | Gold standard |

**No aderyn detector for this class.** M3 slither F1=0.013 is effectively noise on BCCC — slither is not reliable for this class. **Primary strategy: M2 regex pre-filter → M9 manual on all positives.**

---

## 7. Gate Criteria

- Stage 5.2: **Pre-expected FAIL** (slither F1=0.013, no aderyn, 56% FP rate). All 12,394 contracts proceed to Stage 5.3.
- Stage 5.3: T2 structural analysis — check for `for`/`while` loops containing `.send(`/`.call(` combined with `require(`
- Stage 5.4: ≥ 20 manual reviews across Tiers A–D; derive extrapolation rule for "push-pattern with require-on-send"
- Extrapolation rule must cover: "DROP if no loop over user-controlled addresses with require-on-call" as minimum
