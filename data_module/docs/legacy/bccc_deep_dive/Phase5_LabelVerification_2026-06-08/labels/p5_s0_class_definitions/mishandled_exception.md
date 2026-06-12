# Class03: MishandledException — Ground Truth Definition
**Phase 5 Stage 5.0 | Written:** 2026-06-08
**SENTINEL class index:** 2 (`Class03:MishandledException`)
**BCCC folder:** `SourceCodes/MishandledException/` (5,154 contracts)
**Phase 4 FP rate:** 0% — CLEAN class, confirmed by manual review + 2 tools
**Tool coverage:** Slither F1=0.158 (unchecked-transfer, unchecked-send); aderyn F1=0.173

---

## ⚠️ Clean Class — Expected to pass at Stage 5.1

---

## 1. Canonical Definition

**Reference:** SWC-104 (Unchecked Call Return Value), SWC-113 partial

MishandledException covers contracts where the **return value of a low-level call (`.call()`, `.send()`, `.delegatecall()`) is not checked**, allowing silent failures to go undetected and the contract to continue executing as if the call succeeded.

**Boundary with UnusedReturn:** MishandledException is specifically about low-level calls and exception propagation. UnusedReturn is broader (includes ERC-20 `transfer()` return values and any function return). When a contract has `.send()` with unchecked return, it qualifies for both — this is acceptable; the classes overlap.

---

## 2. Inclusion Criteria

**A. `.send()` return value not checked:**
```solidity
msg.sender.send(amount);     // VULNERABLE — send() returns bool; false on failure silently ignored
addr.send(10 ether);         // VULNERABLE
```
Pattern: `.send(` as statement, not wrapped in `require(` or assigned to a variable.

**B. `.call()` return value not checked:**
```solidity
addr.call(data);                         // VULNERABLE
addr.call.value(amount)("");             // VULNERABLE (pre-0.8)
(bool ok,) = addr.call{value:}("");     // NOT vulnerable — ok is captured (but might still not check ok)
addr.call{value: amount}(data);         // VULNERABLE if return not assigned
```

**C. `.delegatecall()` return value not checked:**
```solidity
impl.delegatecall(msg.data);   // VULNERABLE if return not checked
```

---

## 3. Exclusion Criteria

| Pattern | Reason excluded |
|---|---|
| `require(addr.send(amount))` | Return value IS checked |
| `(bool ok,) = addr.call(...); require(ok);` | Explicit check |
| `address.transfer(amount)` | Native transfer always reverts on failure — no return value to check |
| `try/catch` around the call | Exception IS handled |
| Pre-0.5 `throw` keyword on failure check | Old-style exception handling, but IS handled |

---

## 4. Pre-0.8 vs. Post-0.8 Distinctions

- Pre-0.8: `.send()` and `.call()` unchecked return is extremely common (pre-OpenZeppelin era).
- Post-0.8: same pattern; less common but still present in legacy code.
- Pre-0.5: no `require()` — only `throw`. `if (!addr.send(amount)) throw;` IS a check.

---

## 5. Verification Methods

| Method | Detector | Notes |
|---|---|---|
| **M3 Slither** | `unchecked-transfer`, `unchecked-send`, `unchecked-lowlevel` | Reliable; F1=0.158 (tool gap vs. label gap — labels are clean) |
| **M4 Aderyn** | `unchecked-return`, `uninitialized-local-variable` | F1=0.173; second corroborating tool |
| **M2 Regex** | `\.send\s*\(` or `\.call\s*\(` as statement | High recall; filter by not-in-require context |
| **M9 Manual** | Spot-check: is return value captured and checked? | For edge cases only |

---

## 6. Gate Criteria

- **Expected to PASS at Stage 5.1** (0% FP rate confirmed, 2 tool corroboration)
- No Stage 5.2–5.4 work planned unless Stage 5.1 shows unexpected noise
