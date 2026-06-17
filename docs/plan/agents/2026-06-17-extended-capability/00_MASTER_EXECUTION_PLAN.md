# Master Execution Plan — Agents Extended Capability Proposal

**Proposal:** `docs/proposal/agent_proposal/AGENTS_EXTENDED_CAPABILITY_FINAL_PROPOSAL.md`  
**Status:** Ready for implementation  
**Total effort:** 11-16 weeks across 4 phases

---

## Overview: What Gets Built

```
PHASE A (2-3 weeks) — Graph Enhancement & Foundations
  ├─ Reflection agent
  ├─ Multi-LLM debate upgrade (Prosecutor/Defender/Judge)
  ├─ RAG expansion (Code4rena, Sherlock, Solodit, Immunefi, SWC)
  ├─ Tool consensus voting
  ├─ Staged confidence tracking
  ├─ Metric attribution (LIME-style)
  └─ Hotspot attribution visualization

PHASE B (3-4 weeks) — New Paradigms: Symbolic Execution + Bytecode Analysis
  ├─ Symbolic verifier node (Halmos + Z3)
  ├─ Bytecode analyzer node (Gigahorse + CFG)
  ├─ Dataflow/taint analysis
  ├─ Access control analysis
  ├─ Call graph reachability
  └─ Historical vulnerability matching

PHASE C (2-3 weeks) — Production Infrastructure
  ├─ FastAPI gateway (job queue, polling)
  ├─ Pipeline evaluation framework (metrics)
  ├─ Prompt injection guards
  └─ System monitoring (health checks, alerts)

PHASE D (4-5 weeks) — Economic Security + On-Chain Integration
  ├─ Economic simulator (ItyFuzz + Anvil fork + game theory)
  ├─ ZKML proof generation
  ├─ On-chain submission (AuditRegistry.submitAudit)
  └─ (Optional: Echidna fuzzing, severity estimator, invariant verifier)
```

---

## Critical Dependencies & Sequencing

```
Phase A ──→ Phase B ──→ Phase C ──→ Phase D
  (must)       (must)      (can)      (can)
               parallel             parallel
               if Phase A           if B+C
               completes            complete
```

**Blocking relationships:**
- Phase B **requires** Phase A complete (schema extensions, consensus voting used by bytecode analyzer)
- Phase C can run **parallel with B** (independent infrastructure)
- Phase D **requires** B+C complete (uses probe results, needs gateway)

**Non-blocking relationships:**
- RAG expansion (Phase A) can proceed in parallel with anything
- Tool consensus (Phase A) is used by Phase B but can be stubbed in B and swapped in later

---

## Prerequisite Verification Checklist

Before starting **any phase**, verify:

- [ ] **Halmos installed** — `halmos --version` (a16z v0.3+)
- [ ] **Z3 Python bindings** — `python -c "import z3; print(z3.get_version_string())"`
- [ ] **Gigahorse available** — `which gigahorse` or local Docker image
- [ ] **ItyFuzz binary** — check if available / needs installation
- [ ] **Anvil (Foundry)** — `anvil --version` (for Phase D)
- [ ] **LM Studio concurrency tested** — 3 concurrent requests ✅ (already verified)
- [ ] **Aderyn working** — `poetry run python -c "from aderyn import Aderyn"` in agents env
- [ ] **Run 12 checkpoint available** — `ml/checkpoints/GCB-P1-Run12-v3dospatched-20260613_FINAL.pt` exists
- [ ] **Tests passing baseline** — `poetry run pytest agents/tests/ -q` → 219+ PASS

---

## Phase A — Detailed Sequencing

**Duration:** 2-3 weeks  
**Effort:** Medium (mostly prompt engineering + integration)  
**Tests to add:** 15-20

### A.1: Graph Cleanup (3 days)
- Remove module-level `audit_graph = build_graph()` side effect
- Replace with lazy init in graph.py:206
- Update imports in all test files
- Verify 219+ tests still pass

### A.2: AuditState Schema Extension (2 days)
- Add 8 new optional fields (line 50+):
  - `symbolic_findings`, `bytecode_analysis`, `taint_flows`, `permission_graph`
  - `reflection_notes` (Phase A)
  - `econ_scenarios` (already exists, verify)
- Update schema docstring
- Verify TypedDict compatibility

### A.3: Reflection Agent Implementation (5-7 days)
- New node: `agents/src/orchestration/nodes.py:1450+`
- Takes final_report + routing_decisions + verdicts + state
- Calls LLM with structured prompt (see AGENTS_EXTENDED_CAPABILITY_FINAL_PROPOSAL.md §3)
- Outputs `reflection_notes` field
- Write 5 tests (mock LLM, validate output schema)

### A.4: Multi-LLM Debate Upgrade (5-7 days)
- Modify cross_validator node (lines 941-1096)
- Split into three sequential calls: Prosecutor → Defender → Judge
- Each call uses `get_strong_llm()` with role-specific system prompt
- Aggregate verdicts using Judge output
- Write 8 tests (mock each role, verify debate logic)
- **Note:** Fallback to single cross_validator if LLM unavailable

### A.5: RAG Corpus Expansion (5-7 days)
- Create new fetchers in `agents/src/rag/fetchers/`:
  - `code4rena_fetcher.py` (50K findings)
  - `sherlock_fetcher.py` (oracle + MEV findings)
  - `solodit_fetcher.py` (aggregated database)
  - `immunefi_fetcher.py` (bounty disclosures)
  - `swc_registry_fetcher.py` (118 weakness types)
- Update `build_index.py` to use all fetchers
- Re-index FAISS + BM25 (expect 60K+ docs)
- Write 3 integration tests (verify fetcher output format)
- **Estimate:** 3-4 hours to fetch + parse each source

### A.6: Tool Consensus Voting (4-5 days)
- New node: `agents/src/orchestration/nodes.py:1650+`
- Runs after static_analysis (Slither + Aderyn results available)
- Weighted voting: per-class accuracy from Run 12 benchmarks
- Outputs confidence level (high/medium/low)
- Write 6 tests (voting logic, edge cases)

### A.7: Staged Confidence Tracking (4-5 days)
- Integrate into cross_validator (line 980+)
- Start with ML confidence, update per tool match
- Final confidence = product of probabilities (Bayesian style)
- Store in state + final_report
- Write 5 tests (confidence updating, bounds checking)

### A.8: Metric Attribution (5-7 days)
- New node: `agents/src/orchestration/nodes.py:1750+` (post-synthesizer)
- LIME-style: "60% ML, 30% Slither, 10% RAG"
- Compute feature importance per verdict
- Output to final_report
- Write 6 tests (attribution logic, percentages sum to 100)

### A.9: Hotspot Attribution Visualization (6-8 days)
- New node: `agents/src/orchestration/nodes.py:1850+`
- Generates interactive HTML
- Plot: code + AST graph, highlight suspicious nodes
- Clickable functions → shows subgraph
- Use Graphviz or D3.js for rendering
- Write 4 tests (HTML validity, data structure)

---

## Phase B — Detailed Sequencing

**Duration:** 3-4 weeks  
**Effort:** High (new tools, complex logic)  
**Tests to add:** 25-30

### B.1: Halmos Integration (7-10 days)
- Wrapper in `agents/src/tools/halmos_wrapper.py`
- Subprocess invocation with timeout (5-10s)
- Parse Halmos output JSON
- Extract invariants + counterexamples
- Handle timeouts gracefully (return no findings)
- Write 8 tests (mock Halmos, parse edge cases)

### B.2: Z3 Assertion Generator (5-7 days)
- Template library in `agents/src/tools/z3_assertions.py`
- Pre-built assertions for common patterns:
  - "balance >= 0", "total_supply > 0", "no arithmetic overflow"
- Instantiate per contract
- Write 6 tests (template matching, assertion generation)

### B.3: Symbolic Verifier Node (7-10 days)
- New node: `agents/src/orchestration/nodes.py:2000+`
- Trigger: Tier 2 (when ML flags high-risk classes)
- Call Halmos on scoped bytecode
- Aggregate results into `symbolic_findings`
- Write 8 tests (routing, finding aggregation)

### B.4: Gigahorse Decompiler Wrapper (7-10 days)
- Wrapper in `agents/src/tools/gigahorse_wrapper.py`
- Subprocess invocation (timeout: 15-30s)
- Parse CFG + call graph output
- Handle Docker invocation if needed
- Write 8 tests (mock decompiler, output parsing)

### B.5: Bytecode Analyzer Node (7-10 days)
- New node: `agents/src/orchestration/nodes.py:2150+`
- Trigger: When contract is obfuscated OR uses proxies
- Extract CFG, call graph, inferred logic
- Populate `bytecode_analysis` field
- Write 8 tests (CFG validation, logic inference)

### B.6: Dataflow/Taint Analysis (5-7 days)
- New node: `agents/src/orchestration/nodes.py:2300+`
- Track data flows: input → storage write
- Mark untrusted sources
- Populate `taint_flows` field
- Write 5 tests (path tracing, taint propagation)

### B.7: Access Control Analysis (5-7 days)
- New node: `agents/src/orchestration/nodes.py:2400+`
- Extract roles (admin, user, etc.)
- Build permission matrix
- Populate `permission_graph` field
- Write 5 tests (role inference, matrix validation)

### B.8: Call Graph Reachability (5-7 days)
- New node: `agents/src/orchestration/nodes.py:2500+`
- Find paths from public → private functions
- Detect privilege escalation paths
- Output to routing_decisions
- Write 5 tests (reachability algorithms, edge cases)

### B.9: Historical Vulnerability Matching (4-5 days)
- New node: `agents/src/orchestration/nodes.py:2600+`
- Query RAG for similar patterns
- Match against CVE database
- Output `cve_matches` to state
- Write 4 tests (similarity search, matching logic)

---

## Phase C — Detailed Sequencing

**Duration:** 2-3 weeks  
**Effort:** Medium-High (infrastructure, not algorithms)  
**Tests to add:** 15-20

### C.1: FastAPI Gateway Setup (5-7 days)
- New file: `agents/src/api/gateway.py`
- Endpoints:
  - `POST /audit` (submit contract)
  - `GET /audit/{job_id}` (poll status)
  - `GET /audit/{job_id}/report` (fetch results)
- Job queue: use Celery or APScheduler
- Write 8 tests (job lifecycle, polling, error handling)

### C.2: Pipeline Evaluation Framework (6-8 days)
- New file: `agents/src/eval/pipeline_metrics.py`
- Metrics: precision, recall, F1, AUC-PR per class
- Benchmark: 100-200 contracts with ground truth
- Compare verdicts to actual vulnerabilities
- Write 6 tests (metric computation, edge cases)

### C.3: Prompt Injection Guards (3-5 days)
- New file: `agents/src/security/input_sanitizer.py`
- Strip comments from Solidity source
- Detect prompt injection patterns
- Validate source syntax before passing to LLM
- Write 5 tests (injection detection, sanitization)

### C.4: System Monitoring (5-7 days)
- New file: `agents/src/monitoring/health_checks.py`
- MCP server liveness checks (8010, 8011, 8012, 8013)
- RAG index health (check doc count, latency)
- Pipeline latency per node (percentiles)
- Write 6 tests (health check logic, alert thresholds)

---

## Phase D — Detailed Sequencing

**Duration:** 4-5 weeks  
**Effort:** Very High (complex DeFi logic, new paradigm)  
**Tests to add:** 20-25

### D.1: ItyFuzz Integration (8-10 days)
- Wrapper in `agents/src/tools/ityfuzz_wrapper.py`
- Setup RPC endpoint (mainnet fork)
- Configure fuzzer parameters
- Execute fuzzer (timeout: 30-60s)
- Parse findings + exploit traces
- Write 8 tests (setup, execution, parsing)

### D.2: Anvil Fork Management (7-10 days)
- New file: `agents/src/tools/anvil_fork.py`
- Spawn Anvil fork process
- Snapshot + replay state
- Handle process lifecycle
- Write 6 tests (fork operations, state consistency)

### D.3: Game-Theoretic Attack Estimator (10-12 days)
- New file: `agents/src/econ/attack_simulator.py`
- Flash loan profit calculation
- Oracle manipulation cost modeling
- MEV extraction payoff (sandwich attack)
- Governance attack cost (voting threshold breach)
- Write 8 tests (payoff models, edge cases)

### D.4: Economic Simulator Node (10-12 days)
- New node: `agents/src/orchestration/nodes.py:2700+`
- Trigger: Contract imports DeFi interfaces
- Run attack simulators on forked state
- Aggregate into `econ_scenarios` field
- Handle timeouts gracefully
- Write 8 tests (simulator routing, result aggregation)

### D.5: ZKML Proof Generation (8-10 days)
- Wrapper in `agents/src/zkml/proof_generator.py`
- Call existing EZKL circuit (zkml/src/ezkl/run_proof.py)
- Extract audit findings → features
- Run circuit, get proof + public signals
- Write 6 tests (circuit integration, proof validation)

### D.6: On-Chain Submission (7-10 days)
- New node: `agents/src/orchestration/nodes.py:2850+`
- Web3.py integration
- Call AuditRegistry.submitAudit(proof, findings)
- Handle gas + nonce management
- Write 8 tests (submission logic, contract interaction mocking)

### D.7: (Optional) Echidna Property Fuzzing (10-12 days)
- Wrapper in `agents/src/tools/echidna_wrapper.py`
- Template property generation
- Echidna execution
- Test result aggregation
- Write 8 tests

### D.8: (Optional) Severity/Impact Estimator (8-10 days)
- New file: `agents/src/econ/severity_estimator.py`
- Financial impact calculation
- Risk modeling
- Write 6 tests

---

## Testing & Validation Strategy

### Baseline (Before Phase A)
```bash
# Ensure baseline passes
cd agents && poetry run pytest tests/ -q
# Expected: 219+ PASS
```

### Per-Phase Testing
- **Phase A:** Add 15-20 tests → Total: ~235 PASS
- **Phase B:** Add 25-30 tests → Total: ~260 PASS
- **Phase C:** Add 15-20 tests → Total: ~280 PASS
- **Phase D:** Add 20-25 tests → Total: ~305 PASS

### Integration Testing
- After each phase, run full test suite
- Smoke test with real contract (e.g., Uniswap V2)
- Check all nodes execute without error

### Manual Testing
- Phase A: Debate quality (manual review of Prosecutor/Defender/Judge outputs)
- Phase B: Halmos + Bytecode (check findings against known vulnerabilities)
- Phase C: Gateway (test job lifecycle end-to-end)
- Phase D: Economic sim (simulate known MEV exploits, verify detection)

---

## Effort Estimation Summary

| Phase | Duration | Dev Time | Testing | Total |
|-------|----------|----------|---------|-------|
| A | 2-3 wk | 30-35 hrs | 8-10 hrs | 40-45 hrs |
| B | 3-4 wk | 45-55 hrs | 15-18 hrs | 60-73 hrs |
| C | 2-3 wk | 25-30 hrs | 10-12 hrs | 35-42 hrs |
| D | 4-5 wk | 50-65 hrs | 15-20 hrs | 65-85 hrs |
| **Total** | **11-16 wk** | **150-185 hrs** | **48-60 hrs** | **200-245 hrs** |

---

## Success Criteria (Per Phase)

**Phase A:** 
- All 15-20 new tests PASS
- 219+ baseline tests still PASS
- Reflection + debate nodes integrated into graph
- RAG index rebuilt with 60K+ docs

**Phase B:**
- Halmos + Gigahorse integration complete
- All 25-30 new tests PASS
- `symbolic_verifier` + `bytecode_analyzer` nodes routing correctly
- Tested on sample vulnerable contracts

**Phase C:**
- Gateway `/audit` + `/audit/{job_id}` endpoints working
- Job queue persisting + polling working
- All 15-20 new tests PASS
- Monitoring alerts firing correctly

**Phase D:**
- ItyFuzz fuzzing producing findings
- Economic simulator detecting known attacks
- Proof generation + submission working
- All 20-25 new tests PASS
- End-to-end audit pipeline complete

---

## References

- **Proposal:** `docs/proposal/agent_proposal/AGENTS_EXTENDED_CAPABILITY_FINAL_PROPOSAL.md`
- **Current graph:** `agents/src/orchestration/graph.py`
- **AuditState:** `agents/src/orchestration/state.py`
- **Current nodes:** `agents/src/orchestration/nodes.py`
- **ML predictor:** `ml/src/inference/predictor.py`
- **ZKML circuit:** `zkml/src/ezkl/run_proof.py`
- **Contracts:** `contracts/src/AuditRegistry.sol`

---

## Next Steps

1. ✅ Verify prerequisites (Halmos, Z3, Gigahorse, ItyFuzz available)
2. ✅ Review this master plan with team
3. → **Start Phase A** (see `01_PHASE_A_EXECUTION_PLAN.md`)
