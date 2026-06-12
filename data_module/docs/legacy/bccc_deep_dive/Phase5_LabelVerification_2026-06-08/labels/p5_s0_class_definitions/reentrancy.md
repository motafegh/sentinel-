# Class11: Reentrancy — Ground Truth Definition
**Phase 5 Stage 5.0 | Written:** 2026-06-08
**SENTINEL class index:** 10 (`Class11:Reentrancy`)
**BCCC folder:** `SourceCodes/Reentrancy/` (17,698 contracts — largest class)
**Phase 4 FP rate:** 89.4% (500-contract manual audit — the most thoroughly documented finding)
**Tool coverage:** Moderate — slither F1=0.229; aderyn F1=0.169

---

## Phase 4 Anchor Data (use in Stage 5.4 — do NOT re-sample)

500 contracts randomly audited (CRITICAL_FINDINGS.md):
| Pattern | Count | % | Verdict |
|---|---|---|---|
| `.call.value()` with state-after — true reentrancy | 53 | 10.6% | **KEEP** |
| `.transfer()` only — reverts on failure, no re-entry | 205 | 41.0% | **DROP** |
| `.send()` only — 2300 gas, limited re-entry risk | 71 | 14.2% | **DROP** (borderline; include at analyst discretion if state written after) |
| No external call at all | 171 | 34.2% | **DROP** |

→ Stage 5.4: write extrapolation rules from this anchor. No new Reentrancy sampling needed.

---

## 1. Canonical Definition

**Reference:** SWC-107 (Reentrancy), DASP-1 (Reentrancy)

Reentrancy occurs when a contract makes an **external call that forwards control to an unknown contract** (or a contract that the attacker controls), AND the calling contract has **not yet updated its own state** by the time the external call returns, allowing the called contract to re-enter and drain funds or manipulate state.

**The DAO hack (2016) is the canonical example:** ETH withdrawn via `.call.value()` before `balances[msg.sender] = 0` was reached.

### Strict definition (Phase 5 ground truth):

A contract is Reentrancy-positive if and only if:
1. It contains an external call that **forwards variable gas** (the attacker can execute arbitrary code in the callee), AND
2. **State that determines the amount or eligibility** of the call has not been finalized before the call executes (checks-effects-interactions violated), AND
3. Re-entering the calling contract would yield a different (and attacker-beneficial) outcome on the second call.

---

## 2. Inclusion Criteria

**A. Pre-0.8 Solidity — `.call.value()` pattern:**
```solidity
msg.sender.call.value(amount)();          // VULNERABLE if balance not zeroed first
addr.call.value(amount)("");              // equivalent
(bool ok,) = addr.call.value(amount)();  // equivalent
```
Pattern regex: `\.call\.value\s*\([^)]*\)\s*\(`

**B. Post-0.8 Solidity — `.call{value:}` pattern:**
```solidity
(bool ok,) = addr.call{value: amount}("");   // VULNERABLE if balance not zeroed first
```
Pattern regex: `\.call\s*\{[^}]*value\s*:`

**C. Cross-function reentrancy** (harder to detect automatically):
```solidity
function withdraw() external {
    uint bal = balances[msg.sender];
    msg.sender.call.value(bal)();           // state updated in a DIFFERENT function
    // balances[msg.sender] never zeroed here — zeroed in settle()
}
```
Treat as KEEP if the state variable controlling value is not updated before the call within the same execution path.

---

## 3. Exclusion Criteria

| Pattern | Reason excluded |
|---|---|
| `.transfer(amount)` | Forwards only 2300 gas stipend — insufficient to execute state-changing code in callee. **The #1 false positive in BCCC (41% of sample).** |
| `.send(amount)` | Same 2300 gas limit as `.transfer()`. DROP unless analyst specifically identifies a `receive()` that fits in 2300 gas and drains funds (borderline; treat as DROP by default). |
| `addr.call(data)` without ETH value (no `{value:}` or `.value()`) | Forwards no ETH; reentrancy cannot drain funds. DROP unless contract has other valuable state that can be manipulated. |
| Checks-effects-interactions pattern properly applied (`balances[msg.sender] = 0` BEFORE `.call.value()`) | Not exploitable — even if re-entered, balance is already 0. |
| `ReentrancyGuard` modifier (`nonReentrant`) | Properly defended. DROP. |
| No external call of any kind | BCCC mislabeling — DROP (34.2% of Phase 4 sample). |

---

## 4. Pre-0.8 vs. Post-0.8 Distinctions

- Pre-0.8 (92% of BCCC dataset): `.call.value(amount)()` syntax; `transfer()` / `send()` common.
- Post-0.8: `.call{value: amount}("")` syntax; `transfer()` still works.
- Post-0.8: integer overflow in balance variables no longer possible (checked by default) — this makes false reentrancy from arithmetic even less likely.
- Regex verification **MUST use both patterns** or silently misses 92% of the dataset.

---

## 5. Edge Cases

| Scenario | Verdict | Notes |
|---|---|---|
| `.call.value()` used with CEI pattern applied correctly | DROP | Not exploitable |
| `.call.value()` in constructor only | DROP | Constructor runs once at deploy; no attacker re-entry |
| `address.call.value(0)()` — zero ETH call | DROP | No funds to drain via reentrancy |
| `.call.value()` with `require(success)` before state update | KEEP | require() doesn't protect — call still executes before require checks |
| Proxy pattern with `delegatecall` (state in proxy, logic in impl) | KEEP if logic contract has `.call.value()` without CEI | Proxy reentrancy is an active attack vector |
| `nonReentrant` on withdraw but NOT on a related function that shares state | KEEP | Cross-function reentrancy still possible |

---

## 6. Verification Methods

| Method | Detector / Pattern | Expected signal |
|---|---|---|
| **M2 Regex (pre-0.8)** | `\.call\.value\s*\([^)]*\)\s*\(` | High recall for pre-0.8 true reentrancy |
| **M2 Regex (post-0.8)** | `\.call\s*\{[^}]*value\s*:` | High recall for post-0.8 true reentrancy |
| **M3 Slither** | `reentrancy-eth`, `reentrancy-no-eth`, `reentrancy-benign` | High precision; F1=0.229 on BCCC (because 89% of BCCC are FP — slither correctly rejects them) |
| **M4 Aderyn** | `reentrancy-state-change` | F1=0.169; use as corroborating signal |
| **M9 Manual** | Check: `.call.value()` present? CEI violated? Re-entry changes outcome? | Gold standard; use existing 500-contract anchor |

**Primary verification:** M2 regex (both patterns) to classify contracts as "has call.value" vs "no call.value". Contracts with NO regex match → automated DROP (very high confidence). Contracts with match → slither + manual sample.

---

## 7. Gate Criteria

- Stage 5.2: run M2 regex (both pre-0.8 + post-0.8 patterns) on all 17,698 contracts → contracts with no match → DROP (confidence 0.85)
- Expected outcome: ~88% DROP, ~12% provisional KEEP (matches the 10.6% true reentrancy rate + some ambiguous cases)
- Stage 5.4: use existing 500-contract audit as anchor for rule derivation — **no new Reentrancy sampling**
- Extrapolation rules minimum:
  1. "DROP if no `.call.value()` (pre-0.8) AND no `.call{value:}` (post-0.8) pattern found"
  2. "DROP if only `.transfer()` and/or `.send()` used (no `.call.value()`)"
  3. "KEEP-provisional if `.call.value()` found AND slither `reentrancy-eth` fires"
