# Phase D Execution Plan — Economic Security + On-Chain Integration

**Duration:** 4-5 weeks  
**Effort:** Very High (complex DeFi logic, new paradigm)  
**Tests to add:** 20-25  
**Outcome:** Economic simulator, proof generation, on-chain submission

---

## Quick Reference

```
D.1 ItyFuzz integration (hybrid fuzzer + symbolic execution)
D.2 Anvil fork management (mainnet state snapshot)
D.3 Game-theoretic attack estimator (payoff modeling)
D.4 Economic simulator node (attack simulation)
D.5 ZKML proof generation (circuit integration)
D.6 On-chain submission (AuditRegistry integration)
D.7 (Optional) Echidna property fuzzing
D.8 (Optional) Severity/impact estimator
```

---

## D.1: ItyFuzz Integration (8-10 days)

**Files to create:**
- `agents/src/tools/ityfuzz_wrapper.py` (NEW, ~150 lines)

**What to do:**

1. **Install + configure ItyFuzz:**
   - Binary: https://github.com/ConsenSys/ityfuzz
   - RPC endpoint: mainnet fork URL
   - Timeout: 30-60s per contract

2. **Create wrapper:**
   ```python
   class ItyFuzzWrapper:
       def __init__(self, rpc_url: str, timeout: int = 60):
           self.rpc = rpc_url
           self.timeout = timeout
       
       async def fuzz(self, bytecode: str, abi: list) -> list[dict]:
           # Run ItyFuzz
           # Parse findings: {exploit_path, value_extracted, calls}
           # Return list of exploit findings
   ```

3. **Handle RPC connections:**
   - Mainnet fork via Anvil (see D.2)
   - State snapshot + replay
   - Cleanup after fuzzing

4. **Test:**
   - Subprocess invocation
   - Output parsing
   - Timeout handling
   - 8 test cases

**Success criteria:**
- ✅ ItyFuzz subprocess working
- ✅ Findings parsed correctly
- ✅ 8 tests PASS

---

## D.2: Anvil Fork Management (7-10 days)

**Files to create:**
- `agents/src/tools/anvil_fork.py` (NEW, ~120 lines)

**What to do:**

1. **Spawn Anvil fork:**
   ```python
   class AnvilFork:
       def __init__(self, mainnet_rpc: str, fork_block: int = "latest"):
           self.mainnet_rpc = mainnet_rpc
           self.fork_block = fork_block
       
       async def start(self) -> str:
           # Spawn: anvil --fork-url $RPC --fork-block-number $BLOCK
           # Return fork RPC endpoint
       
       async def stop(self):
           # Kill Anvil process
   ```

2. **State management:**
   - Snapshot state before fuzzing
   - Replay on failure
   - Cleanup after completion

3. **Test:**
   - Fork startup
   - RPC connectivity
   - State snapshot + replay
   - 6 test cases

**Success criteria:**
- ✅ Fork starting correctly
- ✅ RPC endpoint accessible
- ✅ 6 tests PASS

---

## D.3: Game-Theoretic Attack Estimator (10-12 days)

**Files to create:**
- `agents/src/econ/attack_simulator.py` (NEW, ~200 lines)

**What to do:**

1. **Implement attack models:**
   ```python
   class FlashLoanAttack:
       def cost_to_execute(contract_state) -> float:
           # Flash loan fee (typically 0.05%)
           # Return: estimated fee
       
       def profit_potential(contract_state) -> float:
           # Can we liquidate positions? Drain reserves?
           # Return: estimated profit
   
   class OracleManipulationAttack:
       def cost_to_execute(contract_state) -> float:
           # Price feed manipulation cost
       
       def profit_potential(contract_state) -> float:
           # Liquidation gain, arbitrage gain
   
   class MEVAttack:
       def sandwich_attack_profit(txs: list) -> float:
           # Front-run + back-run profit estimate
   ```

2. **Payoff modeling:**
   - Who benefits? (attacker, victims)
   - Feasibility: is it profitable?
   - Detectability: hard to see or obvious?

3. **Test:**
   - Cost calculations
   - Profit estimates
   - Edge cases (zero liquidity, safe guards)
   - 8 test cases

**Success criteria:**
- ✅ Attack models implemented
- ✅ Payoff calculations correct
- ✅ 8 tests PASS

---

## D.4: Economic Simulator Node (10-12 days)

**Files to update:**
- `agents/src/orchestration/nodes.py` (add ~150 lines)
- `agents/src/orchestration/graph.py` (add edges)

**What to do:**

1. **Create node:**
   ```python
   async def economic_simulator(state: AuditState) -> dict[str, Any]:
       # Trigger: contract imports DeFi interfaces
       if not has_defi_calls(state["external_call_summary"]):
           return {}
       
       # Start Anvil fork
       fork = await AnvilFork(...).start()
       
       # Run attack simulators
       attacks = []
       attacks.append(await simulate_flash_loan_attack(...))
       attacks.append(await simulate_oracle_attack(...))
       attacks.append(await simulate_mev_attack(...))
       
       await fork.stop()
       
       return {"econ_scenarios": attacks}
   ```

2. **Detect DeFi interactions:**
   - Look for external_call_summary
   - Match against known DeFi interfaces (Uniswap, Aave, Chainlink, etc.)

3. **Wire into graph:**
   - Add as Tier 2 node
   - After cross_validator (can run in parallel with Phase C)

4. **Test:**
   - Attack simulation logic
   - Fork management
   - Scenario aggregation
   - 8 test cases

**Success criteria:**
- ✅ Node integrated
- ✅ Attack simulations working
- ✅ econ_scenarios populated
- ✅ 8 tests PASS

---

## D.5: ZKML Proof Generation (8-10 days)

**Files to create:**
- `agents/src/zkml/proof_generator.py` (NEW, ~100 lines)

**What to do:**

1. **Read ZKML pipeline:**
   - File: `zkml/src/ezkl/run_proof.py`
   - Existing implementation: extract features → proxy → circuit → proof
   - Circuit config: `zkml/ezkl/settings.json`

2. **Create wrapper:**
   ```python
   async def generate_proof(audit_findings: dict) -> dict:
       # Extract features from audit findings
       # Feed to EZKL circuit (via subprocess)
       # Parse proof.json + public_signals
       # Return: {proof, public_signals, class_scores}
   ```

3. **Feature extraction:**
   - From verdicts + confidences
   - Map to circuit inputs (65 public signals)

4. **Test:**
   - Circuit invocation
   - Feature mapping
   - Proof validation
   - 6 test cases

**Success criteria:**
- ✅ Circuit invocation working
- ✅ Proofs generated
- ✅ 6 tests PASS

---

## D.6: On-Chain Submission (7-10 days)

**Files to create:**
- `agents/src/blockchain/submission.py` (NEW, ~150 lines)
- `agents/src/blockchain/contract_abi.py` (NEW, ~50 lines - AuditRegistry ABI)

**What to do:**

1. **Create Web3.py integration:**
   ```python
   from web3 import Web3
   
   async def submit_audit_on_chain(proof: dict, findings: dict, contract_address: str):
       # Connect to network (e.g., Sepolia testnet)
       # Call AuditRegistry.submitAudit(proof, findings, contract_address)
       # Handle gas management + nonce
       # Return: tx_hash, tx_receipt
   ```

2. **Implement AuditRegistry interface:**
   - Read ABI from: `contracts/src/IZKMLVerifier.sol` (interface)
   - Call: `submitAudit(bytes calldata proof, bytes calldata public_signals, bytes calldata findings)`
   - Handle: gas estimation, nonce sequencing

3. **Wire into graph:**
   - Add node: `submit_audit` (after generate_proof)
   - Make non-fatal (audit still succeeds even if on-chain fails)

4. **Test:**
   - Web3 connection (local network)
   - Transaction building
   - Gas estimation
   - 8 test cases

**Success criteria:**
- ✅ On-chain submission working (testnet)
- ✅ Transactions recorded
- ✅ 8 tests PASS

---

## D.7 (Optional): Echidna Property Fuzzing (10-12 days)

**Files to create:**
- `agents/src/tools/echidna_wrapper.py` (NEW, ~100 lines)

**What to do:**

1. **Generate property assertions:**
   - "balance can only increase via deposit()"
   - "owner can only be changed by self"
   - "total_supply ≥ 0"

2. **Run Echidna:**
   - Generate Echidna test harness
   - Run fuzzer (30-60s)
   - Parse failures

3. **Aggregate findings:**
   - Combine with other Tier 2 results

**Tests:** 8 test cases

---

## D.8 (Optional): Severity/Impact Estimator (8-10 days)

**Files to create:**
- `agents/src/econ/impact_estimator.py` (NEW, ~120 lines)

**What to do:**

1. **Financial impact modeling:**
   - Value at risk in contract
   - Potential loss if vulnerability exploited
   - Likelihood × Impact = Risk score

2. **Integration:**
   - Input: verdicts, contract state
   - Output: risk_score for each vulnerability

**Tests:** 6 test cases

---

## Phase D Summary

After Phase D:

✅ **ItyFuzz integration** (advanced fuzzing)  
✅ **Anvil fork management** (state simulation)  
✅ **Game-theoretic attack estimator** (payoff modeling)  
✅ **Economic simulator node** (DeFi attack detection)  
✅ **ZKML proof generation** (circuit integration)  
✅ **On-chain submission** (AuditRegistry integration)  
✅ **Optional: Echidna fuzzing** (property testing)  
✅ **Optional: Impact estimator** (financial modeling)  
✅ **65-85 cumulative tests** (total: ~305 PASS)

**THIRD NEW PARADIGM LIVE:**
- **Economic Security** (game theory + DeFi mechanics)

**System now COMPLETE:**
- Audit pipeline: ✅
- Formal verification: ✅
- Bytecode analysis: ✅
- Economic analysis: ✅
- On-chain integration: ✅

---

## Testing Checklist

- [ ] D.1: ItyFuzz — 8 tests PASS
- [ ] D.2: Anvil — 6 tests PASS
- [ ] D.3: Attack estimator — 8 tests PASS
- [ ] D.4: Economic simulator — 8 tests PASS
- [ ] D.5: Proof generation — 6 tests PASS
- [ ] D.6: On-chain submission — 8 tests PASS
- [ ] D.7 (opt): Echidna — 8 tests PASS
- [ ] D.8 (opt): Impact estimator — 6 tests PASS
- [ ] Full: `poetry run pytest agents/tests/ -q` → ~305 PASS
- [ ] Manual: End-to-end audit → proof → on-chain submission

---

## References

- Proposal: `AGENTS_EXTENDED_CAPABILITY_FINAL_PROPOSAL.md` §3, §5
- Master plan: `00_MASTER_EXECUTION_PLAN.md`
- ItyFuzz: https://github.com/ConsenSys/ityfuzz
- Foundry Anvil: https://book.getfoundry.sh/
- Web3.py: https://web3py.readthedocs.io/
- AuditRegistry: `contracts/src/AuditRegistry.sol`
- ZKML: `zkml/src/ezkl/run_proof.py`
