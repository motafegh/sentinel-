# Echidna 2.3.2 — Usage Reference for SENTINEL Data-Module Investigation

**Status:** INSTALLED & VERIFIED — 2026-06-19
**Referenced from:** `docs/plan/data_module/CLAUDE.md` §8

---

## 1. What Echidna is

Echidna is a property-based fuzzer for EVM bytecode by Trail of Bits. Unlike Slither (static analysis) and Aderyn (AST pattern matching), Echidna **generates random transaction sequences and checks user-defined invariants** to find bugs. It's part of the symbolic execution & fuzzing family — fundamentally different methodology from static analysis.

It works by:
1. Compiling the contract (via `crytic_compile`)
2. Running Slither to extract ABI and assertions
3. Generating random call sequences against the contract
4. Checking assertions (`assert()`), custom properties (`echidna_` functions), or test-mode violations

---

## 2. Our installation

| Detail | Value |
|---|---|
| Binary | `/home/motafeq/.local/bin/echidna` |
| Version | 2.3.2 |
| Size | 32MB (statically linked ELF) |
| Source | Pre-built binary from GitHub releases (no Rust toolchain required) |
| Backend | `crytic_compile` + Slither ABI extraction + EVM fuzzing engine |

---

## 3. How to run

### Basic invocation

```bash
export PATH="/home/motafeq/.local/bin:/home/motafeq/projects/sentinel/.venv/bin:$PATH"

# Assertion mode: fuzz for assert() failures
echidna <contract.sol> --contract <ContractName> --test-mode assertion --test-limit 5000

# Property mode: check echidna_* functions (default)
echidna <contract.sol> --contract <ContractName> --test-limit 5000

# With config file
echidna <contract.sol> --contract <ContractName> --config config.yaml
```

### Key flags

| Flag | Purpose |
|---|---|
| `--contract NAME` | Contract to test (required) |
| `--test-mode assertion` | Check for `assert()` failure violations |
| `--test-limit N` | Max test sequences (default 50000) |
| `--timeout SECONDS` | Time limit |
| `--seq-len N` | Transaction sequence length (default 100) |
| `--config FILE` | YAML config file for advanced settings |
| `--seed N` | Reproducible seed |

---

## 4. What Echidna detects (and what it doesn't)

### Detected (confirmed via test)

| Bug type | How Echidna finds it | Example |
|---|---|---|
| **Integer overflow** | `--test-mode assertion` with `assert(counter >= x)` in `unchecked` block | Confirmed: falsified in 0.6s |
| **Assertion violations** | Any `assert(condition)` in the code that can be violated through call sequences | Standard use case |
| **Custom property violations** | Functions named `echidna_*` returning `bool` | `echidna_no_underflow()` etc. |

### NOT detected (for DIVE contracts)

| Bug type | Why Echidna misses it |
|---|---|
| **Missing access control** | No assertion to violate. A function that should be `onlyOwner` but isn't produces no assertion failure. Echidna just calls it — the call succeeds, property passes. |
| **Reentrancy (CEI)** | Requires custom attacker contracts with re-entrant fallbacks AND explicit balance invariants. DIVE contracts have neither. Without an `assert(address(this).balance == sum(balances))` invariant, the CEI produces no detectable state inconsistency from Echidna's perspective. |
| **Any vulnerability without an assertion** | Echidna is NOT a static analyzer. It doesn't inspect code patterns. It executes code and checks assertions. No assertion = nothing to check. |

---

## 5. Comparison with Slither and Aderyn for SENTINEL Phase 1

| Capability | Slither 0.11.5 | Aderyn 0.6.8 | Echidna 2.3.2 |
|---|---|---|---|
| Method | Static analysis (pattern matching) | Static analysis (AST detectors) | Property-based fuzzing (execution) |
| Requires assertions? | No — analyzes code structure | No — analyzes AST patterns | **Yes** — checks `assert()` or `echidna_*` functions |
| Detects CEI re-entrancy? | ✅ (via `reentrancy-eth` detector) | ⚠️ (only public/external, constructor CEIs often) | ❌ Without custom attacker contracts + invariants |
| Detects missing access control? | ❌ No detector | ❌ Only flags Ownable presence | ❌ No assertion to violate |
| Detects integer overflows? | ❌ | ❌ | ✅ (if `unchecked` + assertion) |
| Requires per-contract setup? | No — drop-in | No — drop-in | **Yes** — needs assertions or custom test contracts |
| Speed | ~2-5s per contract | ~2-3s per contract | <1s per contract (but needs assertions to find anything) |

---

## 6. Decision: NOT for Phase 1 investigation

**Echidna adds zero additional signal for the two vulnerability classes we are investigating:**

1. **ExternalBug (access control):** There is no `assert()` that fails when a missing-auth function is called. The call succeeds normally. Echidna has no way to know it shouldn't have.

2. **Reentrancy (CEI):** CEI violations require attacker contracts to exploit. A simple `asset()` won't detect them. The DIVE contracts we reviewed (42 contracts) have NO assertions about balances or state invariants. Echidna would find nothing on every single one.

**When Echidna WOULD be useful:** Targeted fuzzing of contracts that DO have assertions — foundry test suites, production protocols with invariants, or contracts where we've manually added assertions to test specific hypotheses. For the corpus-scale label-quality investigation (Phase 1), Slither and Aderyn are the appropriate tools.

---

## 7. References

- GitHub: https://github.com/crytic/echidna
- Docs: https://secure-contracts.com/program-analysis/echidna
- Our binary: v2.3.2 pre-built from GitHub releases
