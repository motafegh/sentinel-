# Class08: CallToUnknown — Ground Truth Definition
**Phase 5 Stage 5.0 | Written:** 2026-06-08
**SENTINEL class index:** 7 (`Class08:CallToUnknown`)
**BCCC folder:** `SourceCodes/CallToUnknown/` (11,131 contracts)
**Phase 4 FP rate:** 91% (11 contracts reviewed: 1 KEEP, 10 DROP)
**Tool coverage:** Partial — slither F1=0.245; aderyn F1=0.140

---

## 1. Canonical Definition

**Reference:** SWC-107 (overlapping with reentrancy), DASP-7 (Bad Randomness is different; closest is DASP-2 "Access Control" for delegatecall). Also referred to as "low-level calls" in slither documentation.

CallToUnknown covers vulnerabilities where a contract makes a **low-level call to an address that is externally controlled or unverified**, transferring execution to an unknown (and potentially malicious) contract.

This differs from Reentrancy: Reentrancy is about the attacker re-entering the calling contract. CallToUnknown is about the calling contract handing control to an unknown address, which may behave maliciously (steal forwarded ETH, corrupt state via delegatecall, or front-run).

---

## 2. Inclusion Criteria

**A. `.call()` to an address provided by or controlled by `msg.sender` or `tx.origin`:**
```solidity
function execute(address target, bytes calldata data) external {
    target.call(data);   // VULNERABLE — caller controls target
}
```

**B. `.delegatecall()` to an address from calldata, storage written by user, or constructor arg:**
```solidity
function upgrade(address newImpl) external onlyOwner {
    impl = newImpl;
}
function fallback() external {
    impl.delegatecall(msg.data);   // delegatecall to owner-set impl
}
```
(Included if the address can be set to an attacker contract — e.g., no ownership validation, or ownership transferable to attacker)

**C. `.staticcall()` to externally-controlled address (lower risk but included per BCCC):**
Only include if the return value is used to make a security-critical decision with no validation.

**D. Low-level call to hardcoded-ish address that is actually user-supplied storage:**
```solidity
address public oracle;   // set by anyone via setOracle(addr)
function getPrice() public returns (uint) {
    (bool ok, bytes memory r) = oracle.call(...);  // VULNERABLE
}
```

---

## 3. Exclusion Criteria

| Pattern | Reason excluded |
|---|---|
| `.call()` to `address(this)` | Calling self — not an unknown address |
| `.call()` to a hardcoded address literal or known contract | Known contract — not "unknown" |
| `.call()` to an address in a library where the library is trusted (not user-supplied) | Trusted source |
| Standard ERC-20 `transfer()` / `approve()` — no low-level calls | High-level call, not low-level; BCCC mislabeling — **DROP** |
| `address.transfer(amount)` or `address.send(amount)` | High-level wrappers; do not forward arbitrary call data; not CallToUnknown |
| Contract with no `.call()`, `.delegatecall()`, `.staticcall()` whatsoever | BCCC mislabeling — **DROP** (91% FP in Phase 4; many labeled contracts have zero external calls) |
| Internal function calls (`this.someFunction()`) | Internal — not an unknown external call |

---

## 4. Pre-0.8 vs. Post-0.8 Distinctions

- Pre-0.8: `addr.call(data)` / `addr.call.value(amount)(data)` — return value is a `bool`.
- Post-0.8: `addr.call{value:}(data)` — same vulnerability; syntax change only.
- `delegatecall` syntax unchanged across all versions.
- Pre-0.5: `callcode` (deprecated delegatecall predecessor) also qualifies.

---

## 5. Edge Cases

| Scenario | Verdict | Notes |
|---|---|---|
| Contract calls `.call()` but checks return value and reverts on failure | DROP if call target is known; KEEP if target is user-controlled | The check-on-return doesn't prevent malicious callee behaviour |
| Proxy pattern: `delegatecall` to `implementation` address stored in upgradeable slot | KEEP if implementation can be set by attacker | UUPS/Transparent proxy by owner only → DROP |
| `.call()` to msg.sender (e.g., in a callback notification pattern) | KEEP | `msg.sender` could be an attacker contract |
| `address(token).call(...)` where `token` is an ERC-20 passed by user | KEEP | User-supplied ERC-20 address → unknown contract |

---

## 6. Verification Methods

| Method | Detector / Pattern | Expected signal |
|---|---|---|
| **M2 Regex** | `\.call\s*\(`, `\.delegatecall\s*\(`, `\.staticcall\s*\(` | Very high recall; many FPs (includes `.call()` to known addresses) |
| **M2 Regex (controlled)** | Combine regex match with check for user-supplied address context | Harder to automate; requires AST context |
| **M3 Slither** | `controlled-delegatecall`, `low-level-calls` | Moderate precision; F1=0.245 |
| **M4 Aderyn** | `centralization-risk`, `unsafe-erc20` | Partial coverage |
| **M9 Manual** | Is the call target externally controlled or fixed? | Gold standard; necessary for all regex-positive/slither-negative contracts |

**Primary strategy:** M2 regex to identify contracts with any low-level call → M3 slither to classify → M9 for slither-negative/BCCC-positive (the 91% FP zone). Contracts with no regex match for any of `.call(`, `.delegatecall(`, `.staticcall(` → automated DROP.

---

## 7. Gate Criteria

- Stage 5.2 automated DROP: contracts with zero regex matches for `.call(`, `.delegatecall(`, `.staticcall(` → confidence 0.90 DROP
- Expected gate outcome: 50–70% of contracts can be auto-dropped (those with no low-level calls at all, confirmed as the #1 FP pattern)
- Stage 5.3: structural analysis for remaining contracts — is the call target user-supplied or fixed?
- Stage 5.4: ≥ 20 manual reviews
- Extrapolation rules minimum:
  1. "DROP if no `.call()`, `.delegatecall()`, `.staticcall()` in source"
  2. "DROP if all low-level calls go to hardcoded addresses or `address(this)`"
  3. "KEEP-provisional if `.call()` target comes from function argument, storage set by non-owner, or `msg.sender`"
