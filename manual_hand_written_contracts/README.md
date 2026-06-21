# SENTINEL Manual Hand-Written Contracts

Validation suite for the SENTINEL dual-path GNN+GraphCodeBERT multi-label vulnerability classifier and the AGENTS LangGraph orchestration module.

**83 contracts · 8,599 lines · 11 classes + Safe + cross-class injected**

---

## Purpose

These contracts exist to:

1. **Validate ML model predictions** — ground-truth labels via `// expect:` headers
2. **Test agents pipeline** — end-to-end audit through ML→Slither→Aderyn→RAG→LLM debate
3. **Probe edge cases** — tricky/buried vulnerabilities, multi-label overlaps, realistic DeFi patterns
4. **Serve as injection platform** — the `bccc_injected/` folder takes real BCCC base contracts and replaces their misleading labels with properly injected vulnerabilities

---

## Directory Layout

```
manual_hand_written_contracts/
├── CallToUnknown/            # 7 contracts
├── DenialOfService/          # 7 contracts
├── ExternalBug/              # 7 contracts
├── GasException/             # 6 contracts
├── IntegerUO/                # 7 contracts
├── MishandledException/      # 6 contracts
├── Reentrancy/               # 7 contracts
├── Timestamp/                # 7 contracts
├── TransactionOrderDependence/  # 6 contracts
├── UnusedReturn/             # 6 contracts
├── Safe/                     # 5 contracts (no expected vulns)
├── bccc_injected/            # 12 contracts (BCCC-derived)
└── README.md
```

---

## Label Scheme

Every contract begins with `// expect:` followed by a comma-separated list of classes (or empty for Safe):

```solidity
// expect: Reentrancy
// expect: Reentrancy,Timestamp,UnusedReturn
// expect:                      (Safe contract — no vulnerabilities)
```

### The 10 SENTINEL classes (locked order)

| # | Class | Severity | Description |
|---|-------|----------|-------------|
| 0 | CallToUnknown | high | Unchecked low-level calls (call/delegatecall/send) to unknown addresses |
| 1 | DenialOfService | high | Unbounded loops, push-over-pull failures, permanent bricking |
| 2 | ExternalBug | critical | Oracle manipulation, access control bypass, delegatecall injection, logic bugs |
| 3 | GasException | medium | Out-of-gas patterns: large loops, calldata expansion, stipend exhaustion |
| 4 | IntegerUO | high | Overflow/underflow in unchecked blocks or missing SafeMath |
| 5 | MishandledException | medium | Swallowed return values from external calls |
| 6 | Reentrancy | critical | CEI violation: external call before state update |
| 7 | Timestamp | low | Miner-manipulable block.timestamp in security logic |
| 8 | TransactionOrderDependence | medium | Front-running: approve race, mempool sniping, MEV |
| 9 | UnusedReturn | low | Any discarded return value (broader than MishandledException) |

---

## Contract Generations

### Generation 1 — Standard Class Contracts (55 files)

5 contracts per class folder. Each demonstrates the core vulnerability pattern clearly:

- **CallToUnknown**: proxy delegatecall, dynamic dispatch, low-level forwarder, opaque factory, upgrade proxy selfdestruct
- **DenialOfService**: unbounded refund, push-payment failure, dynamic loop gas bomb, storage growth, unexpected revert
- **ExternalBug**: flash loan oracle manipulation, access control bypass (tx.origin), delegatecall injection, sig replay, logic contract selfdestruct
- **GasException**: massive storage loop, calldata expansion DoS, nested dynamic arrays, transfer stipend, staticcall gas bomb
- **IntegerUO**: ERC20 underflow, unchecked auction, safe math bypass, time calc overflow, batch transfer wrapping
- **MishandledException**: batch payout swallow, delegatecall muted, multi-call hub, withdrawal silent fails, approval race swallow
- **Reentrancy**: CEI violation ERC721, cross-function reentrancy, read-only reentrancy, ERC777 callback, multi-withdraw bank
- **Timestamp**: vesting schedule, auction deadline, randomness seed, ICO phase gate, timelock governance
- **TransactionOrderDependence**: approve front-run, mempool sniping, MEV arbitrage, permit front-run, dutch auction race
- **UnusedReturn**: multi-asset transfer, liquidation ignore, failed batch approve, nested call chain, WETH deposit
- **Safe**: checks-effects-interactions, pull-over-push, OZ-managed, pausable circuit breaker, rate-limited vault

### Generation 2 — Tricky/Buried Vulnerabilities (10 files)

Harder-to-detect contracts. The vulnerability is obfuscated — buried deep in a function body, hidden in a modifier, or disguised by surrounding safe code.

| File | Class(es) | Trick |
|------|-----------|-------|
| `Reentrancy/06_tricky_reentrancy_in_modifier.sol` | Reentrancy | Reentrancy lives in a `collectFee()` modifier — state update in function body looks safe, but the modifier transfers ETH *before* `_` |
| `IntegerUO/06_tricky_overflow_in_interest_rate.sol` | IntegerUO | `principal * rate * elapsed` wraps before `/ denominator` — buried deep inside `_calculateInterest()` |
| `CallToUnknown/06_tricky_call_in_fallback.sol` | CallToUnknown | `fallback()` does `msg.sender.delegatecall(data)` — the vulnerability is in the fallback, not any named function |
| `DenialOfService/06_tricky_dos_in_constructor.sol` | DenialOfService | Constructor pushes 10000 entries — `computeStats()` then loops over them with O(n²) uniqueness check |
| `Timestamp/06_tricky_timestamp_in_pricing.sol` | Timestamp | `block.timestamp` is the 5th operand in a 12-term dynamic pricing formula — not obvious |
| `TransactionOrderDependence/06_tricky_tod_mempool_sniping.sol` | TOD | FIFO queue looks fair but depends entirely on mempool ordering — front-runnable |
| `ExternalBug/06_tricky_externalbug_callback_chain.sol` | ExternalBug, MishandledException | `delegatecall` to user-registered callback target + return value silently ignored |
| `MishandledException/06_tricky_mishandled_internal_chain.sol` | MishandledException | `_executeTransfer()` ignores return — sandwiched between safe `_validateSenderBalance()` and `_updateState()` |
| `GasException/06_tricky_gas_in_loop_condition.sol` | GasException | `for (i=0; i < candidates.length; i++)` — length not cached, pushes during execution make it infinite |
| `UnusedReturn/06_tricky_unused_return_in_callchain.sol` | UnusedReturn, CallToUnknown | 5-step call chain; steps 1-4 check returns, step 5 ignores it + user-supplied router address |

### Generation 3 — Multi-Vulnerability (6 files)

Contracts with 2-3 co-occurring vulnerability classes. Tests multi-label detection.

| File | Classes | Pattern |
|------|---------|---------|
| `Reentrancy/07_multivuln_reentrancy_tod.sol` | Reentrancy, TOD | CEI violation + approve race in different functions sharing the same balance mapping |
| `IntegerUO/07_multivuln_overflow_unused_return.sol` | IntegerUO, UnusedReturn | Fee calc overflow + token transfer return ignored in same function |
| `Timestamp/07_multivuln_timestamp_call.sol` | Timestamp, CallToUnknown | Timelock uses block.timestamp + executer calls arbitrary targets |
| `DenialOfService/07_multivuln_dos_exception.sol` | DenialOfService, GasException | Unbounded ledger array + storage expansion costs compound |
| `ExternalBug/07_multivuln_externalbug_mishandled.sol` | ExternalBug, MishandledException, GasException | Oracle in loop, returns ignored, gas can be exhausted by malicious feed |
| `CallToUnknown/07_multivuln_call_reentrancy.sol` | CallToUnknown, Reentrancy, Timestamp | Plugin system with delegatecall + CEI violation + timestamp-based selection |

### Generation 4 — BCCC-Derived Injected (12 files)

BCCC-SCsVul-2024 contracts have **unreliable folder labels** (41% of files appear in multiple folders, labels often wrong). These contracts take a real BCCC base and inject proper verified vulnerabilities.

All use `pragma solidity ^0.4.24` to match the BCCC source era.

| File | Injected Classes | BCCC Source Pattern |
|------|-----------------|-------------------|
| `01_bccc_reentrancy_injected_erc20.sol` | Reentrancy, IntegerUO | Standard ERC20 + reentrancy in `approveAndCall` + unchecked `burnFrom` |
| `02_bccc_dos_injected_loop.sol` | DenialOfService, Timestamp | Crowdsale + unbounded refund loop + timestamp in bonus calc |
| `03_bccc_calltounknown_injected_delegate.sol` | CallToUnknown, MishandledException | Owned contract + delegatecall in ownership transfer + ignored return |
| `04_bccc_gas_injected_nested.sol` | GasException, DenialOfService | Dividend token + O(n²) loop in `distributeDividends` |
| `05_bccc_externalbug_injected_flashloan.sol` | ExternalBug, UnusedReturn | Lending pool + spot price oracle + ignored transfer returns |
| `06_bccc_timestamp_injected_vesting.sol` | Timestamp, TOD | Vesting + miner-manipulable time calc + front-runnable permit |
| `07_bccc_mishandled_injected_multicall.sol` | MishandledException, CallToUnknown | Multisig wallet + all call return values ignored |
| `08_bccc_unusedreturn_injected_batch.sol` | UnusedReturn, ExternalBug | Batch operator + every transfer return ignored + unrestricted sweep |
| `09_bccc_tod_injected_approve.sol` | TOD, IntegerUO | ERC20 + approve race + unchecked allowance math |
| `10_bccc_weakaccess_injected_ownable.sol` | CallToUnknown, Timestamp | RBAC + owner delegatecall + timestamp-based role expiry |
| `11_bccc_multivuln_oracle_borrow.sol` | ExternalBug, Reentrancy, Timestamp, CallToUnknown | 4-class aggregator: oracle + CEI + timestamp + delegatecall callback |
| `12_bccc_dos_reentrancy_combo.sol` | DenialOfService, Reentrancy, UnusedReturn | Staking pool: O(n²) distribution + CEI violation + ignored returns |

---

## Usage

### With the ML inference server

```bash
poetry run python -c "
from pathlib import Path
from ml.src.inference.predictor import Predictor

predictor = Predictor()
contract = Path('manual_hand_written_contracts/Reentrancy/01_cei_violation_erc721.sol').read_text()
result = predictor.predict(contract)
print(result.class_probabilities)
"
```

### With the agents audit graph

```bash
python agents/scripts/run_real_audit.py \
  --contract manual_hand_written_contracts/ExternalBug/01_flash_loan_oracle_manipulation.sol \
  --output report.json
```

### With Slither directly

```bash
slither manual_hand_written_contracts/CallToUnknown/01_proxy_delegatecall.sol
```

### With the round-trip smoke test

```bash
python ml/scripts/smoke/run_round_trip.py \
  --contracts-dir manual_hand_written_contracts
```

---

## Design Principles

1. **`// expect:` is ground truth** — always matches the actual vulnerability pattern, never the folder name
2. **Compilable with `solc >= 0.8.0`** (except bccc_injected/ which uses `^0.4.24` to match source era)
3. **Slither-parsable** — all contracts avoid forge-specific imports and exotic syntax
4. **Realistic** — patterns based on real-world DeFi exploits: DAO hack, Parity multi-sig, Flash Loan attacks, CREAM/Alpha exploits
5. **Multi-label** — many contracts have 2-3 co-occurring classes to test multi-label detection
6. **Tricky variants** — test the model's ability to detect non-obvious vulnerability patterns (modifiers, fallbacks, internal chains, constructors)
7. **BCCC-derived** — enhances noisy real-world data with precise injected vulnerabilities

---

## Related BCCC Dataset Warning

The `BCCC-SCsVul-2024/SourceCodes/` at project root contains 111,897 `.sol` files but:

- **41% of unique files appear in multiple folders** — folder labels are denormalized duplicates
- **~5% are concatenated multi-pragma files** — not compilable
- **~50% of WeakAccessMod/ files** contain numeric-only data, not Solidity source
- **Labels are unreliable** — most "Reentrancy"-folder files are plain ERC20 tokens with no reentrancy

Always verify against the CSV labels (`BCCC-SCsVul-2024.csv` columns 242-253). The `bccc_injected/` contracts in this repo provide fixed, verified labels for a subset of those patterns.
