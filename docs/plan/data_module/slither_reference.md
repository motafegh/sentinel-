# Slither 0.11.5 — Usage Reference for SENTINEL Data-Module Investigation

**Status:** ACTIVE since 2026-06-18
**Referenced from:** `docs/plan/data_module/CLAUDE.md` §8

---

## 1. What Slither is

Slither is a Python-based Solidity static analysis framework by Trail of Bits. It compiles Solidity via `crytic_compile`, extracts an internal representation (IR), and runs 100 detectors (High, Medium, Low, Informational, Optimization) against the IR. Output is printed to stderr (detector findings) and stdout (AST/IR JSON, if requested).

---

## 2. Our installation

| Detail | Value |
|---|---|
| Binary | `/home/motafeq/projects/sentinel/.venv/bin/slither` |
| Version | 0.11.5 |
| Python | 3.12 (in `.venv`) |
| Backend | `crytic_compile` → calls `solc` |
| solc management | `solc-select` (symlink at `.venv/bin/solc`) |

---

## 3. How to run — solc version matching (CRITICAL)

Slither's `crytic_compile` resolves `solc` from PATH. The contract's `pragma solidity` directive must match the `solc` version, or Slither returns a `InvalidCompilation` error.

### Version matching procedure

```bash
export PATH="/home/motafeq/projects/sentinel/.venv/bin:/home/motafeq/.cargo/bin:$PATH"

# For ^0.8.x contracts (use 0.8.19 — covers ^0.8.0 through ^0.8.19):
rm .venv/bin/solc && ln -s ~/.solc-select/artifacts/solc-0.8.19/solc-0.8.19 .venv/bin/solc

# For ^0.4.x contracts (use 0.4.24 — covers most 0.4.x):
rm .venv/bin/solc && ln -s ~/.solc-select/artifacts/solc-0.4.24/solc-0.4.24 .venv/bin/solc

# For exact-pragma contracts (e.g., =0.8.16):
# Find the nearest installed version via: solc-select versions
```

**Important:** `solc-select use <version>` does NOT reliably update the symlink used by Slither. The manual `rm && ln -s` above is the tested approach.

### Compilation failures

| Error | Cause | Fix |
|---|---|---|
| `Source file requires different compiler version (current compiler is 0.X.X)` | solc version doesn't match pragma | Switch solc via symlink |
| `pragma solidity 0.8.10 >=0.8.10 >=0.8.0 <0.9.0` (complex ranges) | Some contracts have multi-range pragmas | Try 0.8.19; if still fails, try the exact lowest version |
| `0.4.25` with specific formatting | Rare formatting issues | Try `solc-0.4.24` or `solc-0.4.25` if installed |

---

## 4. Running Slither

### Basic invocation

```bash
slither <contract.sol> 2>&1
```

Detector findings go to **stderr** (starts with `INFO:Detectors:`). AST JSON goes to **stdout** (can be ignored for hints).

### Extracting findings

```bash
# Get detector names and their findings
slither <contract.sol> 2>&1 | grep -E 'Detector:|INFO:Detectors' | head -20

# Get full finding details (function names, line numbers)
slither <contract.sol> 2>/dev/null
# (stdout is AST JSON, stderr has the findings)
```

### Typical invocation for DIVE contract analysis

```bash
export PATH="/home/motafeq/projects/sentinel/.venv/bin:/home/motafeq/.cargo/bin:$PATH"
# First: switch solc version (see §3)
rm .venv/bin/solc && ln -s ~/.solc-select/artifacts/solc-0.8.19/solc-0.8.19 .venv/bin/solc
# Then run:
slither /path/to/contract.sol 2>&1 | grep -E 'Detector:|reentrancy|arbitrary|unchecked|suicidal|shadowing|timestamp|assembly|pragma|locked-ether|missing-zero|controlled' -i
```

---

## 5. Relevant detectors for SENTINEL Phase 1 (EB/RE)

### Reentrancy detectors (High/Medium)

| Detector | Severity | What it detects | SENTINEL relevance |
|---|---|---|---|
| `reentrancy-eth` | High | Reentrancy enabling ETH theft. External call → ETH transfer → state change. | **Primary RE detector.** Flags our TP (MultiSig cid=5900) and BORDERLINE contracts (meme tokens). |
| `reentrancy-no-eth` | Medium | Reentrancy without direct ETH theft. External call → token transfer → state change. | Secondary RE detector. |
| `reentrancy-benign` | Low | Benign reentrancy (non-exploitable patterns). | Filter out: if only this fires, the CEI is probably safe. |
| `reentrancy-events` | Low | Reentrancy causing out-of-order events. | Minor. |
| `reentrancy-unlimited-gas` | Informational | Reentrancy via `.send()`/`.transfer()` (2300 gas limited — usually safe). | Confirm: if only this fires, the `.transfer()` gas limit makes it non-exploitable. |
| `reentrancy-balance` | High/Medium | Reentrancy leading to outdated balance checks. | Watch for balance-based checks. |

### Access-control-related detectors

| Detector | Severity | What it detects | SENTINEL relevance |
|---|---|---|---|
| `suicidal` | High | Public `selfdestruct`/`suicide` without access control. | Narrow — only selfdestruct. Not general missing-auth. |
| `unprotected-upgrade` | High | Unprotected upgradeable proxy contracts. | Narrow — only upgradeable proxies. |
| `arbitrary-send-eth` | High/Medium | Functions sending ETH to arbitrary destinations without restrictions. | May indicate missing withdrawal auth. |
| `arbitrary-send-erc20` | High | `transferFrom` with arbitrary `from` parameter. | May indicate missing token transfer auth. |
| `controlled-delegatecall` | High/Medium | `delegatecall` destination controlled by tainted input. | May indicate missing auth on delegatecall. |
| `events-access` | Low | Missing events on access-controlled operations. | Minor — about events, not functions. |
| `missing-zero-check` | Low | Missing zero-address validation in function parameters. | Clue for functions that manage ownership/roles. |

### Other useful detectors

| Detector | Severity | What it tells you |
|---|---|---|
| `solc-version` | Informational | Solidity version with known bugs. Clue: which version the contract was compiled with. |
| `locked-ether` | Medium | Contract can receive ETH but has no withdrawal function. Clue: funds may be locked. |
| `unchecked-transfer` | High/Medium | ERC20 `.transfer()` return value not checked. May indicate token interaction pattern. |
| `encode-packed-collision` | High | `abi.encodePacked()` collision risk with dynamic types. |
| `uninitialized-state` | High | State variables never initialized. |
| `divide-before-multiply` | Medium | Arithmetic precision loss. |

### Confirmed blind spot

**No detector for "privileged function missing access control modifier."** Slither's 100 detectors include narrow access-control cases (selfdestruct, upgrade, delegatecall, ETH send) but do NOT include a general detector for "function writes state / transfers assets but is not behind `onlyOwner` or equivalent." This is confirmed by reviewing the full `slither --list-detectors` output (all 100 entries). 

---

## 6. Output interpretation

### What "0 findings" means

If Slither compiles successfully and reports only `INFO:Detectors:` for informational findings (solc-version, naming-convention, etc.) with no High or Medium detectors firing:
- The contract has no reentrancy patterns Slither can detect
- The contract has no suicidal/unprotected-upgrade/arbitrary-send patterns
- BUT: the contract may still have missing access control (Slither blind spot)

### What reentrancy findings mean

```
INFO:Detectors:
Detector: reentrancy-eth
Reentrancy in <Contract>.<Function>(...) (<filepath>#<start>-<end>):
  - <external call> (<filepath>#<line>)
  - <state change> (<filepath>#<line>)
```

Each finding lists the specific external call and state change in order. **This is the most valuable hint Slither provides** — it directly points at the lines involved in the CEI violation.

### What to do with each finding

| Finding | Action |
|---|---|
| `reentrancy-eth` fires | **Check the flagged function manually.** Look for protection: `nonReentrant` modifier, `lock`/`inSwap` guards, CEI-compliant ordering. If protected AND the external call target is trusted (Uniswap router, known protocol) → BORDERLINE. If unprotected AND the call target is arbitrary → TP. |
| `reentrancy-no-eth` fires | Same manual check as above. |
| `reentrancy-benign` fires | Usually non-exploitable (`.transfer()` gas limit, known safe patterns). Confirm manually. |
| `unchecked-transfer` fires | Check whether return value matters. For trusted token contracts, usually benign. |
| `locked-ether` fires | May indicate a payable fallback without withdrawal. Check for `onlyOwner withdraw()`. |
| Only informational findings (solc-version, naming, etc.) | Contract is clean by Slither's detectors. Proceed to manual review (especially for EB). |

---

## 7. Integration with SENTINEL investigation workflow

For every contract in a manual review batch (Method 0, Methods 3-6):

1. **Check pragma** → set correct solc version via symlink (§3)
2. **Run Slither** → capture findings from stderr (§4)
3. **Extract detectors**: grep for `Detector:` lines, note which fired
4. **For RE**: prioritize contracts where `reentrancy-eth` or `reentrancy-no-eth` fired. Check the flagged lines during manual review.
5. **For EB**: Slither has no direct EB detector. Check `arbitrary-send-eth`, `suicidal`, `unprotected-upgrade`, `controlled-delegatecall` as indirect clues — if any fire, check the flagged function's access control. Otherwise, full manual review is required.
6. **Document in finding file**: Record which Slither detectors fired, which functions they flagged, and whether the hints were useful.

---

## 8. Comparison with Aderyn for SENTINEL Phase 1

| Capability | Slither 0.11.5 | Aderyn 0.6.8 |
|---|---|---|
| Reentrancy (CEI) detection | **Strong.** Finds CEI in all functions (public, private, internal). Catches meme-token `_transfer` CEI and MultiSig `.call{value:}` CEI. | **Limited.** Only analyzes `public`/`external` functions. Finds constructor CEIs (non-exploitable). Misses private-function CEI. |
| Access control detection | **Narrow.** Individual detectors for selfdestruct, upgrade, delegatecall, ETH send. No general "missing auth" detector. | **Information-only.** `centralization-risk` flags Ownable PRESENCE, not auth ABSENCE. |
| Pre-0.8 support | **Works** with solc version matching. | **Mostly works.** 8/10 0.4.x contracts compile, 2 have parser failures. |
| Output format | Detector names + function names + line numbers in stderr. | Structured Markdown/JSON report with severity, title, description, code snippets. |
| Hints for manual review | **Best for RE:** direct line-level CEI trace. | **Best for context:** constructor CEI flags, centralization-risk for Ownable status, block-timestamp for swap logic presence. |

**Recommendation: run both.** Slither for precise RE hints, Aderyn for contextual clues (Ownable presence, swap logic). Neither replaces manual review for access control.

---

## 9. References

- Official docs: https://github.com/crytic/slither
- Detector documentation: https://github.com/crytic/slither/wiki/Detector-Documentation
- Full detector list: `slither --list-detectors`
- crytic_compile: https://github.com/crytic/crytic-compile
