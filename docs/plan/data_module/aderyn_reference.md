# Aderyn 0.6.8 — Usage Reference for SENTINEL Data-Module Investigation

**Status:** ACTIVE since 2026-06-18
**Referenced from:** `docs/plan/data_module/CLAUDE.md` §8

---

## 1. What Aderyn is

Aderyn is a Rust-based Solidity static analyzer by Cyfrin. It compiles Solidity code via `foundry-compilers`, extracts ASTs via `solidity-ast-rs`, and runs 88 detectors (High + Low) against the AST. Output is a structured Markdown or JSON report.

**How it works** (from official docs):
1. Search for all Solidity files within the directory structure
2. Compile the Solidity files and load their ASTs into a `WorkspaceContext`
3. For each available detector, call `detect()` and pass in the `WorkspaceContext`

---

## 2. Our installation

| Detail | Value |
|---|---|
| Binary | `/home/motafeq/.cargo/bin/aderyn` |
| Version | 0.6.8 |
| Size | 24MB |
| Installed | 22 Jan 2026 |
| Backend | `foundry-compilers` (auto-downloads solc versions) + `solidity-ast-rs` (AST parser) |

---

## 3. How to run

### Basic invocation (directory input — NOT single file)

Aderyn requires a **directory** as `[ROOT]`. The CLI docs claim single-file mode works, but v0.6.8 returns `Not a directory` error. Use a temp directory:

```bash
td=$(mktemp -d) && cp <contract.sol> "$td/" && aderyn "$td" 2>&1; rm -rf "$td"
```

### Flags we use

| Flag | Purpose |
|---|---|
| `--stdout` | Print report to stdout instead of writing `report.md` |

**Do NOT use `--highs-only`** — Low-severity findings (`centralization-risk`, `unsafe-erc20-operation`, `block-timestamp-deadline`, `non-reentrant-not-first`) provide useful context before manual review (e.g., confirming Ownable patterns, identifying ERC20 call sites, spotting re-entrancy guards).

### Typical invocation for DIVE contract analysis

```bash
export PATH="/home/motafeq/projects/sentinel/.venv/bin:/home/motafeq/.cargo/bin:$PATH"
td=$(mktemp -d)
cp /path/to/contract.sol "$td/"
aderyn --stdout "$td" 2>&1
rm -rf "$td"
```

### Common failures

| Failure | Cause | Response |
|---|---|---|
| `Error making context: Not a directory` | Single file passed as ROOT | Use temp dir |
| `Error("data did not match any variant of untagged enum BlockOrStatement", line: 1, column: N)` | Parser failure on specific 0.4.x syntax (`throw`, early 0.4.11 patterns) | Fall back to manual review. Rare — 2/10 0.4.x contracts tested failed. |
| Report generated but 0 issues | Contract is clean by Aderyn's detectors | Proceed to manual review |

---

## 4. High-severity detectors (relevant to SENTINEL Phase 1)

### `reentrancy_state_change` — Reentrancy: State change after external call

**Detector logic** (from source):
1. Iterates all `public`/`external` implemented functions
2. Builds a Control Flow Graph (CFG) per function body
3. Finds external call sites via `is_extcallish()`
4. For each external call, recursively searches CFG for state changes that follow it
5. If state change follows external call → flag it

**Scope limitation:** ONLY analyzes `public`/`external` functions. `private`/`internal` functions are NOT checked. This means:
- CEI in `_transfer()` (private) → NOT detected
- CEI in `constructor()` → detected (constructors are checked)

**What we found on DIVE contracts:** Detected CEIs in 5/14 contracts — ALL were constructor CEIs (Uniswap pair creation before state writes). Constructor CEIs are non-exploitable (contract not yet deployed). The meme-token `_transfer` → `swapTokensForEth` CEI pattern was missed because the external call chain goes through `private` functions.

**How to use as a hint:** A `Reentrancy: State change after external call` finding tells you the contract has an external-call-before-state pattern. Check whether the flagged location is:
- **Constructor:** Non-exploitable CEI. Clue that the contract uses Uniswap setup. Look for similar patterns in `_transfer`/other functions.
- **Regular public/external function:** Potentially exploitable. Prioritize manual review of that function.

### `unchecked-low-level-call` — Unchecked Low level calls

Detects raw `.call()`, `.delegatecall()`, `.staticcall()` without checking the return value. Can indicate a re-entrancy vector if combined with state changes.

### Other High detectors (not relevant to EB/RE)

`abi-encode-packed-hash-collision`, `arbitrary-transfer-from`, `contract-locks-ether`, `tx-origin-used-for-auth`, `unsafe-casting`, `weak-randomness`, `selfdestruct`, etc. — not relevant to ExternalBug or Reentrancy.

---

## 5. Low-severity detectors (context for manual review)

| Detector | What it tells you |
|---|---|
| `centralization-risk` | Contract has Ownable pattern with privileged functions. For EB: confirms Ownable presence — then manually check if ALL privileged functions are actually guarded by `onlyOwner`. |
| `unsafe-erc20-operation` | Uses `.transfer()` or other low-level ERC20 calls. May indicate an external call site — check for CEI pattern around it. |
| `block-timestamp-deadline` | Uses `block.timestamp` for swap deadline. Clue that contract does token swaps (Uniswap) — check `_transfer`/swap functions for CEI. |
| `non-reentrant-not-first` | `nonReentrant` modifier is not the first modifier. Tells you the contract HAS re-entrancy protection — check placement and coverage. |
| `state-change-without-event` | State changes without events. Clue for finding state-changing functions that may be part of a CEI chain. |
| `unused-state-variable` | Unused variables — may indicate dead code that could hide vulnerabilities. |

---

## 6. Blind spots (what Aderyn does NOT detect)

| Vulnerability class | Why Aderyn misses it |
|---|---|
| **Missing access control** | No detector for "function should be onlyOwner but isn't." `centralization-risk` only flags Ownable *presence*, not auth *absence*. |
| **CEI in private functions** | `reentrancy_state_change` only analyzes `public`/`external` functions. The DIVE meme-token CEI pattern (`_transfer` → `swapTokensForEth`) is invisible. |
| **Pre-0.8 CEI with throw** | Rare parser failures on specific 0.4.x syntax prevent AST/CFG construction. |
| **Interface-call CEI** | `is_extcallish()` may not recognize interface method calls as external (Uniswap router calls through state variables). |

---

## 7. Integration with SENTINEL investigation workflow

For every contract in a manual review batch (Method 0, Methods 3-6):

1. **Run Aderyn first** (temp dir + `--stdout` — full report, not `--highs-only`)
2. **Check `Issue Summary` counts**: `High=N, Low=M` gives overall signal.
3. **Check for `Reentrancy: State change after external call`**: If found, note location (constructor vs function). Constructor CEI = non-exploitable but confirms Uniswap setup pattern. Function CEI = potentially exploitable, prioritize manual review.
4. **Check Low findings for context**: `centralization-risk` → Ownable present; `unsafe-erc20-operation` → ERC20 calls present; `block-timestamp-deadline` → swap logic present; `non-reentrant-not-first` → re-entrancy guard present.
5. **Use as clues, not verdicts**: Aderyn findings tell you WHERE to look during manual review, never WHAT to conclude.
6. **Document in finding file**: Record Issue Summary counts (High=N, Low=M), whether `reentrancy_state_change` fired and at what location, and any Low findings relevant to EB/RE. Note if parser failed.

---

## 8. References

- Official docs: https://cyfrin.gitbook.io/cyfrin-docs
- Detector list: https://cyfrin.gitbook.io/cyfrin-docs/project-configuration/list-of-supported-detectors
- CLI options: https://cyfrin.gitbook.io/cyfrin-docs/aderyn/cli-options
- GitHub repo: https://github.com/Cyfrin/aderyn
- `reentrancy_state_change` source: `aderyn_core/src/detect/high/reentrancy_state_change.rs`
