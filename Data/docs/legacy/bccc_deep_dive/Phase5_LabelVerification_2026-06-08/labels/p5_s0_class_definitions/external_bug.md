# Class01: ExternalBug — Ground Truth Definition
**Phase 5 Stage 5.0 | Written:** 2026-06-08
**SENTINEL class index:** 0 (`Class01:ExternalBug`)
**BCCC folder:** `SourceCodes/ExternalBug/` (3,604 contracts)
**Phase 4 FP rate:** 100% in tiny sample (1/1 false positive confirmed)
**Tool coverage:** Partial — slither F1=0.101, aderyn F1=0.145

---

## ⚠️ WARNING: Ambiguous Class

ExternalBug is a BCCC catch-all with no direct DASP-10 or SWC equivalent. Manual inspection of the dataset (including Phase 4 Stage 1 review and CRITICAL_FINDINGS.md) found a SafeMath library labeled as ExternalBug with no selfdestruct, no tx.origin, no delegatecall, no low-level calls whatsoever.

**If the inclusion criteria below cannot be verified against at least 20 manually reviewed contracts in Stage 5.4, this class should be declared "unverifiable" and excluded from SENTINEL training.**

---

## 1. Canonical Definition

ExternalBug covers vulnerabilities arising from dangerous interactions with external entities or addresses. Based on cross-referencing the BCCC paper and SWC registry, it encompasses:

| Sub-type | SWC | DASP | Description |
|---|---|---|---|
| Suicidal contract | SWC-106 | DASP-9 (Denial of Service) | `selfdestruct(addr)` reachable by any non-owner address |
| tx.origin authentication | SWC-115 | DASP-4 (Access Control) | Authorization checks using `tx.origin` instead of `msg.sender` |
| Unauthorized delegatecall | SWC-112 | DASP-6 (Short Address) | `delegatecall` to an address controlled by caller |
| Signature replay | SWC-121 | — | `ecrecover` result used without nonce/chain-id, enabling replay across txs |

---

## 2. Inclusion Criteria

A contract is ExternalBug-positive if it contains **at least one** of the following patterns AND that pattern creates an exploitable vulnerability (i.e., reachable by an unprivileged caller):

**A. Suicidal / selfdestruct:**
```
selfdestruct(addr)   // or suicide(addr) in pre-0.5 Solidity
```
- Reachable by any address (not just `owner`) via `msg.sender == owner` check
- OR no access control at all on the function containing `selfdestruct`

**B. tx.origin authentication:**
```
require(tx.origin == owner);   // or if (tx.origin == ...) { ... }
```
- Must be in a function that gates a privileged action (not in a payable fallback as a spam filter)

**C. Unauthorized delegatecall:**
```
addr.delegatecall(data)    // where addr comes from msg.sender, calldata, or storage written by msg.sender
```
- Fixed library addresses are EXCLUDED

**D. Signature replay (ecrecover without nonce):**
```
ecrecover(hash, v, r, s)
```
- No nonce, no chain-id, no expiry — signature can be replayed

---

## 3. Exclusion Criteria

| Pattern | Reason excluded |
|---|---|
| `selfdestruct` inside `onlyOwner` modifier (properly enforced) | Not exploitable by unprivileged caller |
| `tx.origin` used only in event logging | Not an auth bypass |
| `delegatecall` to hardcoded library address | Fixed address — not attacker-controlled |
| `address.call(...)` without delegatecall | That is CallToUnknown, not ExternalBug |
| SafeMath library (pure arithmetic, no external interactions) | No vulnerability; the Phase 4 FP was this exact case |
| Plain ERC-20/ERC-721 with no selfdestruct/tx.origin | BCCC mislabeling — DROP |
| `ecrecover` with on-chain nonce or expiry check | Replay protected — not vulnerable |

---

## 4. Pre-0.8 vs. Post-0.8 Distinctions

- `suicide()` was deprecated in Solidity 0.5.0 in favour of `selfdestruct()`. Both are equivalent and both qualify.
- `delegatecall` syntax unchanged across versions.
- `tx.origin` usage unchanged across versions.
- Pre-0.8: `ecrecover` without SafeMath-protected nonces very common. Post-0.8: less common but still possible.

---

## 5. Edge Cases

| Scenario | Verdict | Notes |
|---|---|---|
| `selfdestruct` in constructor only | DROP | Only fires at deploy time; no persistent vulnerability |
| `selfdestruct` protected by `require(block.timestamp > deadline)` (no sender check) | KEEP | Any caller can trigger after deadline |
| `tx.origin` used as `require(tx.origin != address(0))` (null check) | DROP | Not an auth bypass |
| Contract that IS a library (no state, no selfdestruct) labeled ExternalBug | DROP | BCCC mislabeling; confirmed FP pattern |

---

## 6. Verification Methods

| Method | Detector / Pattern | Expected signal |
|---|---|---|
| **M2 Regex** | `selfdestruct\|suicide\(`, `tx\.origin`, `\.delegatecall\(`, `ecrecover\(` | High recall, moderate precision |
| **M3 Slither** | `suicidal`, `tx-origin`, `controlled-delegatecall` | High precision on suicidal+tx.origin; miss on replay |
| **M4 Aderyn** | `selfdestruct`, `centralization-risk` | Partial |
| **M9 Manual** | Read source; apply inclusion criteria above | Gold standard |

**Primary verification: M2 + M3**. M9 for all contracts where M2 fires but M3 does not (high-FP zone).

---

## 7. Gate Criteria

- Stage 5.2 gate: agreement ≥ 80% between M2+M3 combined verdict and BCCC label → provisionally verified
- Given 100% FP in Phase 4 sample, **expect < 80% → this class will proceed to Stage 5.3**
- Stage 5.4 minimum: 20 manually reviewed contracts (across Tiers A–D)
- If Stage 5.4 cannot produce clear extrapolation rules (e.g., BCCC labeled everything with `call()` as ExternalBug regardless of vulnerability), declare class "unverifiable" and document for exclusion from SENTINEL training
