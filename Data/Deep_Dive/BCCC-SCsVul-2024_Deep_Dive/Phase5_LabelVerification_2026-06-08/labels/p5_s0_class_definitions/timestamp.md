# Class04: Timestamp — Ground Truth Definition
**Phase 5 Stage 5.0 | Written:** 2026-06-08
**SENTINEL class index:** 3 (`Class04:Timestamp`)
**BCCC folder:** `SourceCodes/Timestamp/` (2,674 contracts)
**Phase 4 FP rate:** 50% (moderate noise; confirmed by CRITICAL_FINDINGS.md)
**Tool coverage:** Moderate — slither F1=0.129 (timestamp detector); aderyn has no detector

---

## ⚠️ NOT a Clean Class

Timestamp has 50% FP rate. It is NOT grouped with IntegerUO/UnusedReturn/MishandledException. It requires Stage 5.2 automated verification and may proceed to Stage 5.3.

---

## 1. Canonical Definition

**Reference:** SWC-116 (Block values as a proxy for time), DASP-8 (Bad Randomness — overlapping)

Timestamp vulnerability exists when `block.timestamp` (or the deprecated alias `now`) is used to **determine the outcome of a critical, security-sensitive decision** that an attacker can influence by manipulating the timestamp within the ±15-second miner tolerance (pre-merge) or is used as the **sole source of randomness**.

The core problem: miners (and post-merge validators) can adjust `block.timestamp` by up to ~15 seconds in either direction. This manipulation window is sufficient to change the outcome of timestamp-based decisions.

---

## 2. Inclusion Criteria

`block.timestamp` or `now` is used in a context where:

**A. Randomness / lottery:**
```solidity
uint winner = block.timestamp % participants.length;   // VULNERABLE
// or
uint random = uint(keccak256(abi.encodePacked(block.timestamp, msg.sender)));
```
Attacker (miner or colluding validator) times their tx to win the lottery.

**B. Access control or critical gating:**
```solidity
require(block.timestamp >= unlockTime);  // VULNERABLE if unlockTime is close to now
// Miner can satisfy this condition slightly early for their own tx
```
Only vulnerable if the window matters (e.g., financial gain from early unlock). A +/-15s window on a 1-year lockup is negligible → DROP.

**C. Determining a winner / refund amount based on timestamp:**
```solidity
if (block.timestamp - lastBid < 60) {
    winner = msg.sender;   // VULNERABLE — miner adjusts to win
}
```

**Key test:** Would a 15-second manipulation of `block.timestamp` give an attacker a meaningful advantage (financial gain, access before others, lottery win)? If yes → INCLUDE.

---

## 3. Exclusion Criteria

| Pattern | Reason excluded |
|---|---|
| `block.timestamp` used only for event logging (`emit Deposit(msg.sender, amount, block.timestamp)`) | No security impact from timestamp manipulation |
| `block.timestamp` stored for record-keeping only (no branch/condition based on it) | No exploitable decision path |
| Long-duration lockup (`require(block.timestamp >= deployTime + 365 days)`) | 15-second miner window is negligible vs. 1 year |
| `block.timestamp` in `require` where the condition is already safely past (e.g., historical check in frozen state) | Not exploitable at current chain state |
| Any use of `block.timestamp` in informational/view functions | No state change; no financial impact |
| Plain ERC-20 with `block.timestamp` only in a `lastTransfer` tracking mapping | No decision branch; BCCC mislabeling — DROP |

---

## 4. Pre-0.8 vs. Post-0.8 Distinctions

- `now` keyword: deprecated in Solidity 0.7.0, removed in 0.8.0. Use `block.timestamp` post-0.7.
- Regex must match both: `block\.timestamp` and `\bnow\b` (the latter only in pre-0.8 contracts).
- The vulnerability semantics are identical across versions.

---

## 5. Edge Cases

| Scenario | Verdict | Notes |
|---|---|---|
| `block.timestamp` used as part of keccak hash for "random" seed | KEEP | Miner-influenceable entropy |
| `require(block.timestamp > expiry)` with expiry set 1 hour in future | KEEP | 15s window can matter for MEV near expiry |
| `require(block.timestamp > expiry)` with expiry set 30 days in future | DROP | 15s window negligible vs. 30-day span |
| Timestamp used for rate-limiting (e.g., one action per day per user) | DROP | 15s manipulation doesn't help attacker bypass a 24h window |
| PRNG using `blockhash(block.number - 1) ^ block.timestamp` | KEEP | Both values are miner-influenceable |

---

## 6. Verification Methods

| Method | Detector / Pattern | Expected signal |
|---|---|---|
| **M2 Regex** | `block\.timestamp`, `\bnow\b` | Very high recall (captures all uses); ~50% FP (usage ≠ vulnerability) |
| **M3 Slither** | `timestamp` detector | Conservative; F1=0.129 on BCCC sample; flags uses in branches |
| **M9 Manual** | Check: is timestamp in a branch that controls money/randomness/access with a ≤15s-sensitive outcome? | Gold standard |

**Primary verification:** M3 Slither (already run on 15% sample) + M2 regex context filter. For contracts where `block.timestamp` is only in non-branch contexts → automated DROP. For contracts where slither fires → KEEP as provisional. For slither-negative/BCCC-positive → M9 sample.

**Aderyn:** No `block-timestamp-dependency` detector that maps to this class. Do not use aderyn as evidence for Timestamp.

---

## 7. Gate Criteria

- Stage 5.2: expect 50–80% agreement (slither is conservative but BCCC has 50% FP) → **likely provisional (Stage 5.3 for edge cases)**
- Stage 5.3: focus on contracts where `block.timestamp` appears only in non-branch context (automated DROP candidate)
- Stage 5.4: ≥ 20 manual reviews
- Extrapolation rule must cover: "DROP if `block.timestamp` / `now` appears only in event emissions or record-keeping mappings with no downstream branch"
