# Phase B Execution Plan — Symbolic Execution + Bytecode Analysis

**Duration:** 3-4 weeks  
**Effort:** High (new complex tools)  
**Tests to add:** 25-30  
**Outcome:** Halmos integration, Gigahorse decompiler, taint analysis, access control, CVE matching

---

## Quick Reference

```
B.1 Halmos integration (symbolic execution wrapper)
B.2 Z3 assertion generator (constraint templates)
B.3 Symbolic verifier node (proof generation)
B.4 Gigahorse decompiler wrapper (bytecode → CFG)
B.5 Bytecode analyzer node (control flow extraction)
B.6 Dataflow/taint analysis (input → storage tracking)
B.7 Access control analysis (role inference)
B.8 Call graph reachability (privilege escalation paths)
B.9 Historical vulnerability matching (CVE lookup)
```

---

## B.1: Halmos Integration (7-10 days)

**Files to create:**
- `agents/src/tools/halmos_wrapper.py` (NEW, ~100 lines)

**What to do:**

1. **Read Halmos docs:**
   - Tool: https://github.com/a16z/halmos (v0.3+)
   - Output format: JSON with properties + counterexamples

2. **Create wrapper:**
   ```python
   class HalmosWrapper:
       def __init__(self, timeout_seconds=10):
           self.timeout = timeout_seconds
       
       async def run(self, contract_path: str, invariants: list[str]) -> dict:
           # Subprocess call to halmos
           # Parse JSON output
           # Return: {passed: bool, counterexample: optional}
   ```

3. **Handle errors:**
   - Timeout: return empty findings
   - Syntax error: log + continue
   - Missing Halmos: graceful fallback

4. **Test:**
   - Mock Halmos subprocess
   - Test JSON parsing
   - Test timeout handling
   - 8 test cases

**Success criteria:**
- ✅ Wrapper callable via `await halmos.run(path, invariants)`
- ✅ Returns structured output
- ✅ Handles timeouts + errors
- ✅ 8 tests PASS

---

## B.2: Z3 Assertion Generator (5-7 days)

**Files to create:**
- `agents/src/tools/z3_assertions.py` (NEW, ~80 lines)

**What to do:**

1. **Create assertion templates:**
   ```python
   TEMPLATES = {
       'balance_non_negative': 'assert all(balance >= 0)',
       'total_supply_positive': 'assert total_supply > 0',
       'no_overflow': 'assert x < 2**256',
       'access_control': 'assert caller == admin || caller == owner',
   }
   ```

2. **Implement instantiation:**
   - Map contract variables to templates
   - Generate Z3 assertions
   - Return list of SMT queries

3. **Test:**
   - Template matching
   - Variable extraction
   - Assertion generation
   - 6 test cases

**Success criteria:**
- ✅ Templates work for common patterns
- ✅ Z3 assertions valid syntax
- ✅ 6 tests PASS

---

## B.3: Symbolic Verifier Node (7-10 days)

**Files to update:**
- `agents/src/orchestration/nodes.py` (add ~100 lines)
- `agents/src/orchestration/graph.py` (add edges)

**What to do:**

1. **Create node function:**
   ```python
   async def symbolic_verifier(state: AuditState) -> dict[str, Any]:
       # Trigger: Tier 2 (high-risk classes only)
       ml_result = state["ml_result"]
       high_risk = ["Reentrancy", "ExternalBug", "IntegerUO"]
       
       if not any(ml_result["probabilities"].get(c, 0) >= 0.50 for c in high_risk):
           return {}  # skip if no high-risk flags
       
       # Run Halmos + Z3
       halmos_findings = await halmos.run(...)
       z3_assertions = z3_gen.instantiate(contract, ...)
       
       # Aggregate
       return {"symbolic_findings": [...]}
   ```

2. **Wire into graph:**
   - Add edge: `consensus_engine` → `symbolic_verifier` (parallel with other Tier 2)
   - Or sequential: `cross_validator` → `symbolic_verifier`

3. **Test:**
   - Mock Halmos output
   - Trigger routing
   - Finding aggregation
   - 8 test cases

**Success criteria:**
- ✅ Node integrated into graph
- ✅ symbolic_findings populated
- ✅ 8 tests PASS

---

## B.4: Gigahorse Decompiler Wrapper (7-10 days)

**Files to create:**
- `agents/src/tools/gigahorse_wrapper.py` (NEW, ~120 lines)

**What to do:**

1. **Setup Gigahorse:**
   - Install: https://github.com/nevillegrech/gigahorse-toolchain
   - Input: compiled bytecode (EVM binary)
   - Output: Datalog facts (.facts files)

2. **Create wrapper:**
   ```python
   class GigahorseWrapper:
       async def decompile(self, bytecode: str) -> dict:
           # Run Gigahorse on bytecode
           # Parse .facts output
           # Extract: CFG, call graph, storage patterns
           # Return: {cfg, call_graph, functions, storage}
   ```

3. **Parse output:**
   - CFG edges: function A → B
   - Call graph: who calls whom
   - Storage: which slots accessed

4. **Test:**
   - Mock decompiler output
   - Parse validation
   - Edge extraction
   - 8 test cases

**Success criteria:**
- ✅ Gigahorse invocation working
- ✅ Output parsing correct
- ✅ CFG + call graph extracted
- ✅ 8 tests PASS

---

## B.5: Bytecode Analyzer Node (7-10 days)

**Files to update:**
- `agents/src/orchestration/nodes.py` (add ~100 lines)
- `agents/src/orchestration/graph.py` (add edges)

**What to do:**

1. **Create node:**
   ```python
   async def bytecode_analyzer(state: AuditState) -> dict[str, Any]:
       contract_code = state["contract_code"]
       
       # Trigger: obfuscated code or proxy pattern detected
       if not is_obfuscated(contract_code) and not uses_proxy(contract_code):
           return {}
       
       # Decompile bytecode
       bytecode = compile_to_bytecode(contract_code)
       analysis = await gigahorse.decompile(bytecode)
       
       return {"bytecode_analysis": analysis}
   ```

2. **Detect obfuscation/proxies:**
   - Obfuscated: high entropy, few readable strings
   - Proxy: delegatecall to fixed address

3. **Wire into graph:**
   - Add edge: `consensus_engine` → `bytecode_analyzer` (Tier 2)

4. **Test:**
   - Obfuscation detection
   - Proxy pattern detection
   - Analysis aggregation
   - 8 test cases

**Success criteria:**
- ✅ Node integrated
- ✅ bytecode_analysis populated
- ✅ 8 tests PASS

---

## B.6: Dataflow/Taint Analysis (5-7 days)

**Files to create:**
- `agents/src/tools/taint_analyzer.py` (NEW, ~90 lines)

**What to do:**

1. **Implement taint tracking:**
   - Mark untrusted sources: user input, external calls
   - Track data flow through contract
   - Mark sinks: storage writes, transfers

2. **Create node:**
   ```python
   async def taint_analyzer(state: AuditState) -> dict[str, Any]:
       # Taint propagation
       # Output: {taint_flows: [{source, sink, path}, ...]}
   ```

3. **Test:**
   - Simple data flows
   - Multi-hop paths
   - Edge cases (loops, conditions)
   - 5 test cases

**Success criteria:**
- ✅ Taint flows identified
- ✅ Paths traced correctly
- ✅ 5 tests PASS

---

## B.7: Access Control Analysis (5-7 days)

**Files to create:**
- `agents/src/tools/access_control_analyzer.py` (NEW, ~100 lines)

**What to do:**

1. **Extract roles from code:**
   - Modifiers: onlyOwner, onlyAdmin, etc.
   - State variables: owner, admin, roles mapping
   - Build role → functions matrix

2. **Analyze privilege:**
   - Who can call what
   - Escalation paths (public → private)

3. **Create node:**
   ```python
   async def permission_graph_node(state: AuditState) -> dict[str, Any]:
       # Role extraction + analysis
       # Return: {permission_graph: {role: [functions]}}
   ```

4. **Test:**
   - Role extraction
   - Permission matrix
   - Escalation detection
   - 5 test cases

**Success criteria:**
- ✅ Roles identified
- ✅ Permission matrix correct
- ✅ 5 tests PASS

---

## B.8: Call Graph Reachability (5-7 days)

**Files to create:**
- `agents/src/tools/reachability_analyzer.py` (NEW, ~80 lines)

**What to do:**

1. **Build call graph:**
   - From bytecode analysis (B.4) + source analysis
   - Edges: function A → B

2. **Compute reachability:**
   - BFS/DFS from public functions
   - Find paths to internal functions

3. **Create node:**
   ```python
   async def reachability_analyzer(state: AuditState) -> dict[str, Any]:
       # Reachability analysis
       # Return: {reachable_pairs: [(public, private), ...]}
   ```

4. **Test:**
   - Path finding
   - Unreachable functions
   - Cycle handling
   - 5 test cases

**Success criteria:**
- ✅ Paths computed correctly
- ✅ 5 tests PASS

---

## B.9: Historical Vulnerability Matching (4-5 days)

**Files to create:**
- `agents/src/tools/cve_matcher.py` (NEW, ~70 lines)

**What to do:**

1. **Query RAG for patterns:**
   - "Does this pattern match CVE-XXXX?"
   - Retrieve similar findings from expanded corpus (Phase A.5)

2. **Create node:**
   ```python
   async def cve_matcher(state: AuditState) -> dict[str, Any]:
       # Query RAG for pattern matches
       # Return: {cve_matches: [{cve, confidence, description}, ...]}
   ```

3. **Test:**
   - Similarity search
   - Match ranking
   - 4 test cases

**Success criteria:**
- ✅ CVE matches found
- ✅ 4 tests PASS

---

## Phase B Summary

After Phase B:

✅ **Halmos integration** (symbolic execution)  
✅ **Z3 assertions** (constraint solving)  
✅ **Symbolic verifier node** (proof generation)  
✅ **Gigahorse decompiler** (bytecode analysis)  
✅ **Bytecode analyzer node** (CFG + call graph)  
✅ **Dataflow tracking** (taint analysis)  
✅ **Access control analysis** (role inference)  
✅ **Reachability analysis** (privilege escalation paths)  
✅ **CVE matching** (historical lookups)  
✅ **60+ new tests** (total: ~310 PASS)

**Two new paradigms live in agents:**
1. **Symbolic Execution** (formal verification)
2. **Bytecode Analysis** (deep inspection)

**Ready to proceed to Phase C** (infrastructure) or Phase D (economic security)

---

## Testing Checklist

- [ ] B.1: Halmos wrapper — 8 tests PASS
- [ ] B.2: Z3 assertions — 6 tests PASS
- [ ] B.3: Symbolic verifier — 8 tests PASS
- [ ] B.4: Gigahorse wrapper — 8 tests PASS
- [ ] B.5: Bytecode analyzer — 8 tests PASS
- [ ] B.6: Taint analyzer — 5 tests PASS
- [ ] B.7: Access control — 5 tests PASS
- [ ] B.8: Reachability — 5 tests PASS
- [ ] B.9: CVE matcher — 4 tests PASS
- [ ] Full: `poetry run pytest agents/tests/ -q` → ~310 PASS
- [ ] Smoke test with obfuscated + proxy contracts

---

## References

- Proposal: `AGENTS_EXTENDED_CAPABILITY_FINAL_PROPOSAL.md` §3, §5
- Master plan: `00_MASTER_EXECUTION_PLAN.md`
- Phase A output: symbolic_findings, bytecode_analysis, taint_flows, permission_graph fields in AuditState
- Halmos docs: https://github.com/a16z/halmos
- Gigahorse: https://github.com/nevillegrech/gigahorse-toolchain
- Z3 Python API: https://github.com/Z3Prover/z3/tree/master/src/api/python
