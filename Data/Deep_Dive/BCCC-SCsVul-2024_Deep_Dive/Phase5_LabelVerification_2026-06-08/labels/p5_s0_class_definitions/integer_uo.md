# Class10: IntegerUO — Ground Truth Definition
**Phase 5 Stage 5.0 | Written:** 2026-06-08
**SENTINEL class index:** 9 (`Class10:IntegerUO`)
**BCCC folder:** `SourceCodes/IntegerUO/` (16,740 contracts)
**Phase 4 FP rate:** 0% — CLEAN class, confirmed by manual review + 2 tools
**Tool coverage:** Slither F1=0.236 (limited — slither cannot detect pre-0.8 overflow), aderyn F1=0.000

---

## ⚠️ Clean Class — Expected to pass at Stage 5.1

This class has 0% FP rate confirmed by Phase 4 manual review. It is expected to pass the Stage 5.1 gate using existing evidence. No new automated runs or manual reviews planned unless Stage 5.1 shows unexpected noise.

---

## 1. Canonical Definition

**Reference:** SWC-101 (Integer Overflow and Underflow)

Integer overflow/underflow occurs when arithmetic operations produce a result outside the valid range of the integer type, causing the value to **wrap around silently** (in pre-0.8 Solidity).

- **Overflow:** `uint256 x = type(uint256).max; x + 1 == 0` (wraps to 0)
- **Underflow:** `uint256 x = 0; x - 1 == type(uint256).max` (wraps to max)

Post-0.8: the compiler inserts overflow checks by default — unchecked arithmetic must be explicitly declared with `unchecked {}`. Overflow in post-0.8 without `unchecked {}` causes a `Panic(0x11)` revert (not a vulnerability).

---

## 2. Inclusion Criteria

**A. Pre-0.8 Solidity arithmetic without SafeMath:**
```solidity
// pragma solidity ^0.4.x or ^0.5.x or ^0.6.x or ^0.7.x
uint256 public totalSupply;
function mint(address to, uint256 amount) public {
    totalSupply += amount;        // VULNERABLE — overflow to 0
    balances[to] += amount;
}
```
Key signal: pragma < 0.8.0 AND arithmetic on integers AND no SafeMath library usage.

**B. Post-0.8 `unchecked {}` block with arithmetic that affects security-critical state:**
```solidity
// pragma solidity ^0.8.x
function deduct(uint256 amount) external {
    unchecked {
        balances[msg.sender] -= amount;   // VULNERABLE if no prior balance check
    }
}
```
Only include if the unchecked arithmetic is reachable without a prior `require(balances[msg.sender] >= amount)` check.

---

## 3. Exclusion Criteria

| Pattern | Reason excluded |
|---|---|
| Pre-0.8 with SafeMath imported and used on all arithmetic | SafeMath wraps with `require(c >= a)` — overflow reverts rather than wraps |
| Post-0.8 without `unchecked {}` | Compiler inserts checks — overflow reverts automatically |
| Overflow in loop counter only (no financial state change) | Low-impact; does not directly enable theft or loss |
| `assert(c >= a)` / `require(c >= a)` wrapping every arithmetic op | Manually protected |
| Division / modulo operations (cannot overflow by definition for unsigned integers) | Not an integer overflow |

---

## 4. Pre-0.8 vs. Post-0.8 Distinctions

- **Pre-0.8 (vast majority of BCCC = 92% pre-0.6):** All arithmetic vulnerable without SafeMath. Slither `divide-before-multiply` fires but does not cover add/sub/mul overflow directly.
- **Post-0.8:** Only `unchecked {}` blocks are vulnerable. Slither cannot detect pre-0.8 overflow (confirmed Phase 4, F1=0.000 for aderyn). This explains why tool agreement is low despite 0% FP — the tools have coverage gaps, not the labels.

---

## 5. Verification Methods

| Method | Detector | Notes |
|---|---|---|
| **M2 Regex** | `pragma solidity [^0-9]*0\.[0-7]\.` (pre-0.8) AND no SafeMath import | High recall for pre-0.8 IntegerUO |
| **M3 Slither** | `divide-before-multiply`, `tautology` | Partial coverage only |
| **M9 Manual** | Spot-check: is SafeMath used? Is `unchecked {}` present? | Gold standard; only needed for edge cases |

---

## 6. Gate Criteria

- **Expected to PASS at Stage 5.1** using existing evidence (0% FP rate, confirmed by manual review in Phase 4)
- If Stage 5.1 shows > 5% unexpected noise in a subpopulation, investigate that subpopulation specifically (e.g., contracts using SafeMath but with arithmetic outside SafeMath calls)
- No Stage 5.2–5.4 work planned unless Stage 5.1 gate fails
