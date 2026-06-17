# Agents Extended Capability — Implementation Plan Index

**Start here.** This folder contains the complete, actionable implementation plan for the AGENTS extended capability proposal.

---

## Files in This Folder

| File | Purpose | Start Here |
|------|---------|-----------|
| **README.md** | This index (you are here) | ✓ |
| **00_MASTER_EXECUTION_PLAN.md** | Overview, sequencing, dependencies | ✓ First read this |
| **01_PHASE_A_EXECUTION_PLAN.md** | Detailed Phase A tasks (2-3 weeks) | After Master plan |
| **02_PHASE_B_EXECUTION_PLAN.md** | Detailed Phase B tasks (3-4 weeks) | After Phase A |
| **03_PHASE_C_EXECUTION_PLAN.md** | Detailed Phase C tasks (2-3 weeks) | Parallel with B |
| **04_PHASE_D_EXECUTION_PLAN.md** | Detailed Phase D tasks (4-5 weeks) | After B+C |

---

## Quick Start

1. **Read Master Plan** (20 min)
   - File: `00_MASTER_EXECUTION_PLAN.md`
   - Understand: phases, sequencing, dependencies, effort estimates

2. **Verify Prerequisites** (30 min)
   - Checklist in Master plan §"Prerequisite Verification"
   - Ensure: Halmos, Z3, Gigahorse, ItyFuzz, Anvil installed

3. **Start Phase A** (2-3 weeks)
   - File: `01_PHASE_A_EXECUTION_PLAN.md`
   - Follow step-by-step A.1 through A.9
   - Expected: 35-45 new tests, ~250 PASS total

4. **Proceed to Phases B, C, D**
   - Each phase builds on prior phases
   - Phases B and C can run in parallel
   - Phase D requires B+C complete

---

## What Gets Built (Phases A-D)

### Phase A: Graph Enhancement (2-3 weeks)
```
Reflection agent + Multi-LLM debate + RAG expansion + 
Consensus voting + Confidence tracking + Metric attribution + 
Hotspot visualization
```
**Impact:** Better verdict quality, explainability, broader knowledge base

### Phase B: Symbolic Execution + Bytecode Analysis (3-4 weeks)
```
Halmos integration + Z3 assertions + Symbolic verifier +
Gigahorse decompiler + Bytecode analyzer + Dataflow analysis +
Access control analysis + Call graph reachability + CVE matching
```
**Impact:** Formal verification, deep code inspection, security hardening

### Phase C: Production Infrastructure (2-3 weeks)
```
FastAPI gateway + Evaluation framework + 
Prompt injection guards + System monitoring
```
**Impact:** User-facing API, metrics, security, observability

### Phase D: Economic Security + On-Chain (4-5 weeks)
```
ItyFuzz integration + Anvil fork + Attack estimators +
Economic simulator + ZKML proofs + On-chain submission +
(Optional: Echidna fuzzing, impact estimator)
```
**Impact:** DeFi attack detection, proof verification, on-chain integration

---

## Testing & Validation

### Per-Phase Test Targets

| Phase | Tests to Add | Cumulative Total |
|-------|--------------|------------------|
| A | 15-20 | ~250 |
| B | 25-30 | ~310 |
| C | 15-20 | ~325 |
| D | 20-25 | ~305 |

### Success Criteria (All Phases)

- ✅ All new tests PASS
- ✅ All baseline tests still PASS (219+)
- ✅ Smoke test with real contracts (Uniswap, etc.)
- ✅ Manual verification of new features
- ✅ No regressions in existing nodes

---

## Key References

### Proposal Document
- **Location:** `docs/proposal/agent_proposal/AGENTS_EXTENDED_CAPABILITY_FINAL_PROPOSAL.md`
- **What:** Strategic overview, paradigm descriptions, learning outcomes
- **When to read:** Before starting implementation

### Current Codebase
- **Graph:** `agents/src/orchestration/graph.py`
- **Nodes:** `agents/src/orchestration/nodes.py` (1415 lines)
- **State:** `agents/src/orchestration/state.py` (160 lines)
- **LLM client:** `agents/src/llm/client.py` (use `get_strong_llm()`)
- **RAG:** `agents/src/rag/` (retriever, index builder)
- **Tests:** `agents/tests/` (219+ existing tests)

### External Tools
- **Halmos:** https://github.com/a16z/halmos
- **Z3:** https://github.com/Z3Prover/z3
- **Gigahorse:** https://github.com/nevillegrech/gigahorse-toolchain
- **ItyFuzz:** https://github.com/ConsenSys/ityfuzz
- **Foundry Anvil:** https://book.getfoundry.sh/
- **FastAPI:** https://fastapi.tiangolo.com/
- **Web3.py:** https://web3py.readthedocs.io/
- **ZKML:** `zkml/src/ezkl/run_proof.py` (existing circuit)
- **AuditRegistry:** `contracts/src/AuditRegistry.sol` (on-chain target)

---

## Implementation Checklist

### Before Starting
- [ ] Read Master plan (00_MASTER_EXECUTION_PLAN.md)
- [ ] Verify all prerequisites installed
- [ ] Confirm baseline tests pass: `poetry run pytest agents/tests/ -q` → 219+ PASS
- [ ] Create scratch file: `~/.claude/scratch/agents_implementation_<YYYYMMDD>.md`

### Phase A
- [ ] A.1 Graph cleanup
- [ ] A.2 AuditState schema
- [ ] A.3 Reflection agent
- [ ] A.4 Multi-LLM debate
- [ ] A.5 RAG expansion
- [ ] A.6 Consensus voting
- [ ] A.7 Confidence tracking
- [ ] A.8 Metric attribution
- [ ] A.9 Visualization
- [ ] Phase A tests: ~250 PASS

### Phase B (can start after A.1-A.2 complete)
- [ ] B.1 Halmos wrapper
- [ ] B.2 Z3 assertions
- [ ] B.3 Symbolic verifier node
- [ ] B.4 Gigahorse wrapper
- [ ] B.5 Bytecode analyzer node
- [ ] B.6 Dataflow analysis
- [ ] B.7 Access control analysis
- [ ] B.8 Reachability analysis
- [ ] B.9 CVE matching
- [ ] Phase B tests: ~310 PASS

### Phase C (can start in parallel with B)
- [ ] C.1 FastAPI gateway
- [ ] C.2 Evaluation framework
- [ ] C.3 Prompt guards
- [ ] C.4 Monitoring
- [ ] Phase C tests: ~325 PASS

### Phase D (after B+C complete)
- [ ] D.1 ItyFuzz wrapper
- [ ] D.2 Anvil fork management
- [ ] D.3 Attack estimator
- [ ] D.4 Economic simulator node
- [ ] D.5 ZKML proof generation
- [ ] D.6 On-chain submission
- [ ] D.7 (optional) Echidna
- [ ] D.8 (optional) Impact estimator
- [ ] Phase D tests: ~305 PASS

### Post-Implementation
- [ ] All tests pass
- [ ] Smoke test with real contracts
- [ ] Documentation updated
- [ ] Memory.md + project notes updated

---

## Effort Estimation

| Phase | Dev Time | Testing | Total | Duration |
|-------|----------|---------|-------|----------|
| A | 30-35 hrs | 8-10 hrs | 40-45 hrs | 2-3 wks |
| B | 45-55 hrs | 15-18 hrs | 60-73 hrs | 3-4 wks |
| C | 25-30 hrs | 10-12 hrs | 35-42 hrs | 2-3 wks |
| D | 50-65 hrs | 15-20 hrs | 65-85 hrs | 4-5 wks |
| **Total** | **150-185 hrs** | **48-60 hrs** | **200-245 hrs** | **11-16 wks** |

---

## How to Use These Plans

### For Implementation
1. Open Master plan → understand overall flow
2. Open Phase N plan → follow step-by-step instructions
3. Each plan has concrete file names, line numbers, code snippets
4. Each plan has test cases and success criteria

### For Project Tracking
1. Use checklist above to track progress
2. Check off each task (A.1, A.2, etc.) as complete
3. After each phase, verify test count matches expected total
4. Update scratch file daily with findings/blockers

### For Documentation
1. Each plan file references source code locations
2. Each plan file lists files to create/modify
3. Cross-reference with `AGENTS_EXTENDED_CAPABILITY_FINAL_PROPOSAL.md` for why

---

## Next Steps

**Ready to start?**

1. ✅ Read: `00_MASTER_EXECUTION_PLAN.md` (20 min)
2. ✅ Verify prerequisites (30 min)
3. ✅ Start Phase A: `01_PHASE_A_EXECUTION_PLAN.md`

**Questions?**

- Refer to proposal: `AGENTS_EXTENDED_CAPABILITY_FINAL_PROPOSAL.md`
- Check master plan for effort/dependencies
- Refer to code references for file locations

---

## Key Stats

- **Total effort:** 200-245 hours
- **Total duration:** 11-16 weeks (4 phases)
- **New tests:** 75-95
- **Final test count:** ~305 PASS
- **New nodes:** 9 (reflection, debate upgrade, consensus, confidence, attribution, symbolic_verifier, bytecode_analyzer, economic_simulator, submit_audit)
- **New paradigms:** 3 (formal verification, bytecode analysis, economic security)
- **New agent types:** Supporting tools (taint, access control, reachability, CVE matching, etc.)

---

**Created:** 2026-06-17  
**Version:** 1.0  
**Status:** Ready for implementation
