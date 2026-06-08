# Class06: UnusedReturn — Ground Truth Definition
**Phase 5 Stage 5.0 | Written:** 2026-06-08
**SENTINEL class index:** 5 (`Class06:UnusedReturn`)
**BCCC folder:** `SourceCodes/UnusedReturn/` (3,229 contracts)
**Phase 4 FP rate:** 0% — CLEAN class, confirmed by manual review + 2 tools
**Tool coverage:** Slither F1=0.118 (unused-return); aderyn F1=0.000

---

## ⚠️ Clean Class — Expected to pass at Stage 5.1

---

## 1. Canonical Definition

**Reference:** SWC-104 (Unchecked Call Return Value) — partial overlap with MishandledException

UnusedReturn covers contracts where the **return value of an external call (or internal function call) is silently ignored**, causing the contract to proceed as if the call succeeded when it may have failed.

**Boundary with MishandledException:** Both involve ignored return values, but:
- **UnusedReturn** = any function return value not checked (broader — applies to high-level and low-level calls, library functions, ERC-20 return values)
- **MishandledException** = specifically `.call()`, `.send()` low-level return values not checked (focus on low-level calls and exception propagation)

In practice, BCCC may have blurred this boundary. Use the definitions below and flag any contracts that appear to match MishandledException instead.

---

## 2. Inclusion Criteria

**A. ERC-20 `transfer()` / `transferFrom()` return value not checked:**
```solidity
// pre-EIP-20 tokens return bool; some revert, some return false silently
token.transfer(recipient, amount);        // VULNERABLE — false not checked
token.transferFrom(from, to, amount);     // VULNERABLE — same
```
Key pattern: `token.transfer(...)` or `token.transferFrom(...)` called as a statement (not in `require()`, `if`, or assigned to a variable).

**B. `.send()` return value not checked:**
```solidity
msg.sender.send(amount);    // VULNERABLE — false on failure, silently ignored
```
Note: `.send()` is also MishandledException. If slither labels it as `unchecked-send`, map to both classes.

**C. Function with `returns (bool)` whose return value is discarded:**
```solidity
// externalContract.doSomething() returns (bool success)
externalContract.doSomething();   // VULNERABLE if failure silent
```

---

## 3. Exclusion Criteria

| Pattern | Reason excluded |
|---|---|
| `require(token.transfer(...))` | Return value IS checked |
| `bool ok = token.transfer(...); require(ok)` | Explicit check |
| `token.transfer(...)` where token is known to revert on failure (post-EIP-20 compliant) | Safe — revert on failure is acceptable |
| `transfer()` call using native `address.transfer()` (not ERC-20) | Native `transfer()` always reverts on failure — no return value to check |
| Internal function calls within same contract | Only checking external calls is in scope |

---

## 4. Pre-0.8 vs. Post-0.8 Distinctions

- Pre-0.8: many ERC-20 tokens returned `false` on failure instead of reverting. Checking return value was critical.
- Post-0.8 + OpenZeppelin SafeERC20: `safeTransfer()` wraps with `require(success)`. But raw `token.transfer()` still returns `bool`.
- The vulnerability is version-agnostic; more prevalent in pre-0.8 ecosystem.

---

## 5. Verification Methods

| Method | Detector | Notes |
|---|---|---|
| **M3 Slither** | `unused-return` | Reliable; F1=0.118 (tool gap, not label gap — slither misses some patterns) |
| **M2 Regex** | `\.transfer\s*\(` or `\.transferFrom\s*\(` as statement (not inside `require\|if\|=`) | High recall; false positives for native transfer |
| **M9 Manual** | Spot-check: is the return value assigned or wrapped in require? | For edge cases only |

---

## 6. Gate Criteria

- **Expected to PASS at Stage 5.1** (0% FP rate confirmed)
- No Stage 5.2–5.4 work planned unless Stage 5.1 shows unexpected noise
