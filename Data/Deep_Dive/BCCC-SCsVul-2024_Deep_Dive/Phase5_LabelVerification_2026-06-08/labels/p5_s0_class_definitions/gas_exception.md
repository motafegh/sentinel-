# Class02: GasException — Ground Truth Definition
**Phase 5 Stage 5.0 | Written:** 2026-06-08
**SENTINEL class index:** 1 (`Class02:GasException`)
**BCCC folder:** `SourceCodes/GasException/` (6,879 contracts)
**Phase 4 FP rate:** 67% in small sample (9 reviewed: 0 KEEP, 6 DROP, 3 UNCERTAIN)
**Tool coverage:** Partial — slither F1=0.131 (costly-loop only); aderyn has NO detector for this class

---

## 1. Canonical Definition

**Reference:** SWC-128 (DoS With Block Gas Limit)

GasException covers patterns where a transaction can **consume more gas than the block gas limit** or cause an **out-of-gas revert** due to unbounded iteration over state structures whose size is controlled (directly or indirectly) by external callers.

This is a **DoS via gas exhaustion** vulnerability — an attacker grows the state until no transaction can process it within a block's gas budget.

**Boundary with DenialOfService:** GasException = gas exhaustion from loops. DenialOfService = blocking a function via other means (failed external call, state corruption). See `denial_of_service.md` for that boundary.

---

## 2. Inclusion Criteria

A contract is GasException-positive if it contains a loop (`for`, `while`, `do-while`) that:

**A. Iterates over an array whose length grows with user input:**
```solidity
address[] public investors;

function addInvestor(address a) public {
    investors.push(a);        // unbounded — any caller adds
}

function payAll() public {
    for (uint i = 0; i < investors.length; i++) {   // VULNERABLE
        investors[i].transfer(share);
    }
}
```

**B. Iterates over a mapping by iterating an index array that grows with user input** (same pattern via index array)

**C. Performs expensive per-iteration work** (external calls, storage writes, keccak256 in loop) **AND** loop bound is either unbounded or controlled by a non-owner caller

**Key test:** Could a malicious user cause `investors.length` (or equivalent) to grow large enough to make the loop exceed the block gas limit? If yes → INCLUDE.

---

## 3. Exclusion Criteria

| Pattern | Reason excluded |
|---|---|
| Loop over constant/fixed-size array (hardcoded length) | Bounded; gas cost is deterministic |
| Loop where only `owner` or an `onlyOwner` function can grow the array | Owner cannot be the attacker |
| Loop inside a `view`/`pure` function (no state write) | No state change; gas cost for caller only, does not affect others |
| Loop whose bound comes from a hardcoded `constant` or `immutable` | Fixed gas cost |
| Single arithmetic operation (no loop) | High gas but not a gas exception from unbounded iteration |
| ERC-20 token with no loops at all | BCCC mislabeling — DROP |

---

## 4. Pre-0.8 vs. Post-0.8 Distinctions

- Pattern is version-agnostic — unbounded loops exist in all Solidity versions.
- Pre-0.8: missing SafeMath means integer overflow in loop counter is also possible (but that is IntegerUO, not GasException).
- Post-0.8: overflow in loop counter causes revert, not wrap-around (doesn't change the gas exhaustion risk).
- Gas costs changed significantly between versions (EIP-2929, EIP-1884) but the structural pattern is the same.

---

## 5. Edge Cases

| Scenario | Verdict | Notes |
|---|---|---|
| `payAll()` loops over array but has `require(investors.length < 100)` | DROP | Length is capped — cannot grow indefinitely |
| Loop over user-controlled array with length check BUT check is bypassable via another function | KEEP | Check is ineffective |
| Nested loops (O(n²) complexity), array grown by owner only | DROP | Owner-controlled; not an attacker-exploitable path |
| `transfer()` in loop (gas limit per call) | KEEP | Each `.transfer()` forwards limited gas, but many iterations still exhausts |
| `mapping` iterated via helper array that grows via public function | KEEP | Same pattern as A above |

---

## 6. Verification Methods

| Method | Detector / Pattern | Expected signal |
|---|---|---|
| **M3 Slither** | `costly-loop` | Good precision; misses some patterns |
| **M2 Regex** | `for\s*\(`, `while\s*\(` combined with `.push(` or dynamic array size | High recall, many FPs; needs manual filter |
| **M9 Manual** | Read loop body + array growth mechanism | Gold standard |

**No aderyn detector for this class** — M3 slither is the only automated tool. M9 required for all disputed contracts.

**Primary verification strategy:** slither `costly-loop` on all 6,879 contracts → treat positives as provisional; treat BCCC-positive/slither-negative contracts as the dispute bucket for M9 sampling.

---

## 7. Gate Criteria

- Stage 5.2: expect < 80% agreement (slither `costly-loop` is conservative; 67% BCCC FP rate confirmed). **Pre-expected Stage 5.3.**
- Stage 5.3: T2 structural analysis (count `for`/`while` loops + check if array grows from public functions)
- Stage 5.4: ≥ 20 manual reviews across Tiers A–D
- Extrapolation rule must cover at minimum: "DROP if no loop in source" and "DROP if loop bound is a constant or owner-set variable"
