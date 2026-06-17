# AGENTS Module — Planning Index

**Status:** 2026-06-17 Complete  
**TOD Bug Fix:** ✅ Done (Commit 8c50fb8d7)  
**Proposal:** ✅ Finalized (additive design, 3 paradigms, 8 tools)  
**Planning:** ✅ Complete (11 detailed documents, 4,200+ lines)

---

## Quick Navigation

### 📋 What Should I Read First?

**If you just got here:**
1. Start: `~/projects/sentinel/docs/proposal/agent_proposal/AGENTS_EXTENDED_CAPABILITY_FINAL_PROPOSAL.md` (strategic overview)
2. Then: `~/.claude/projects/-home-motafeq-projects-sentinel/memory/2026-06-17_project_agents_state.md` (detailed state)

**If you want to implement Phase A-D:**
1. Read: `2026-06-17-extended-capability/README.md` (5 min)
2. Then: `2026-06-17-extended-capability/00_MASTER_EXECUTION_PLAN.md` (20 min)
3. Then: `2026-06-17-extended-capability/01_PHASE_A_EXECUTION_PLAN.md` (detailed steps)

**If you want to test the current system first:**
1. Read: `2026-06-17-agents-real-e2e-test/README.md` (5 min)
2. Then: `2026-06-17-agents-real-e2e-test/00_MASTER_TEST_PLAN.md` (20 min)
3. Then: `2026-06-17-agents-real-e2e-test/01_SETUP_PLAN.md` (follow setup steps)

---

## 📁 Folder Structure

```
docs/plan/agents/
├─ INDEX.md (this file)
├─ 2026-06-17-extended-capability/
│  ├─ README.md (quick start)
│  ├─ 00_MASTER_EXECUTION_PLAN.md (phasing + dependencies)
│  ├─ 01_PHASE_A_EXECUTION_PLAN.md (2-3 weeks, ~40-45 hrs)
│  ├─ 02_PHASE_B_EXECUTION_PLAN.md (3-4 weeks, ~60-73 hrs)
│  ├─ 03_PHASE_C_EXECUTION_PLAN.md (2-3 weeks, ~35-42 hrs)
│  └─ 04_PHASE_D_EXECUTION_PLAN.md (4-5 weeks, ~65-85 hrs)
│
└─ 2026-06-17-agents-real-e2e-test/
   ├─ README.md (purpose + success criteria)
   ├─ 00_MASTER_TEST_PLAN.md (strategy + prerequisites)
   ├─ 01_SETUP_PLAN.md (LM Studio + ML API + MCP servers)
   ├─ 02_EXECUTION_PLAN.md (run audits on test contracts)
   └─ 03_ANALYSIS_PLAN.md (analyze results + extract baselines)
```

---

## 🚀 What Happened (2026-06-17)

### 1. TOD Class Name Bug Fixed
- **Issue:** Routing tables used "TOD" but ML model outputs "TransactionOrderDependence"
- **Impact:** All TransactionOrderDependence contracts routed to wrong path (silent failure)
- **Fix:** Updated 6 files, 53+ tests PASS
- **Commit:** `8c50fb8d7`

### 2. Extended Capability Proposal Finalized
- **Design:** Additive (keep all 9 working nodes, add 3 paradigms + 8 tools)
- **Paradigms:** Formal verification, bytecode analysis, economic security
- **Effort:** ~200-245 hours over 11-16 weeks
- **File:** `docs/proposal/agent_proposal/AGENTS_EXTENDED_CAPABILITY_FINAL_PROPOSAL.md`

### 3. Implementation Plans Written
- **Phase A (2-3 wks):** Graph cleanup + reflection + debate + RAG expansion + consensus
- **Phase B (3-4 wks):** Halmos + Z3 + bytecode analysis + dataflow + access control + CVE matching
- **Phase C (2-3 wks):** FastAPI gateway + evaluation framework + monitoring
- **Phase D (4-5 wks):** ItyFuzz + economic simulator + ZKML + on-chain submission
- **Files:** 5 documents in `2026-06-17-extended-capability/` (2,328 lines total)

### 4. Real E2E Testing Plan Written
- **Purpose:** Validate 9-node graph before Phase A implementation
- **Duration:** 5-8 hours (setup 2-3h, test 3-4h, analyze 2-3h)
- **Components:** LM Studio, ML API, 4 MCP servers, 3 test contracts
- **Deliverables:** Performance baselines, 3 audit reports, bug list, confidence score
- **Files:** 5 documents in `2026-06-17-agents-real-e2e-test/` (1,885 lines total)

---

## 📚 All Planning Documents

### Extended Capability Implementation (4,200+ lines total)

| File | Lines | Content | Time |
|------|-------|---------|------|
| README.md | 247 | Index + quick start + timeline | 5 min |
| 00_MASTER_EXECUTION_PLAN.md | 420 | Phases A-D overview, sequencing, dependencies | 20 min |
| 01_PHASE_A_EXECUTION_PLAN.md | 638 | 9 detailed steps, code refs, tests (40-45 hrs) | — |
| 02_PHASE_B_EXECUTION_PLAN.md | 408 | 9 detailed steps, code refs, tests (60-73 hrs) | — |
| 03_PHASE_C_EXECUTION_PLAN.md | 236 | 4 detailed steps, code refs, tests (35-42 hrs) | — |
| 04_PHASE_D_EXECUTION_PLAN.md | 379 | 6+2 optional steps, code refs, tests (65-85 hrs) | — |

### Real E2E Testing Plan (1,885 lines total)

| File | Lines | Content | Time |
|------|-------|---------|------|
| README.md | 201 | Purpose + timeline + contracts | 5 min |
| 00_MASTER_TEST_PLAN.md | 272 | Strategy + prerequisites + measurements | 20 min |
| 01_SETUP_PLAN.md | 369 | LM Studio + ML API + MCP servers + checks (2-3 hrs) | — |
| 02_EXECUTION_PLAN.md | 373 | Test harness script + 3 contracts (1-2 hrs each) | — |
| 03_ANALYSIS_PLAN.md | 670 | JSON parsing + quality assessment + baselines (2-3 hrs) | — |

---

## 🎯 Key Decision Points

### Option A: Start Real E2E Testing Now
✅ **Recommended first step:**
- Validates current system works at scale
- Identifies hidden bugs before Phase A
- Establishes performance baselines
- Builds confidence in architecture
- **Duration:** 5-8 hours (one day)
- **Blocker for Phase A?** Only if bugs found (unlikely)

**Next:** Follow `2026-06-17-agents-real-e2e-test/01_SETUP_PLAN.md`

### Option B: Skip Testing, Go Straight to Phase A
❌ **Not recommended, but possible:**
- Saves 5-8 hours of testing time
- **Risk:** Phase A starts with hidden bugs
- **Cost of discovering bug in Phase A:** +2-4 hours fixing
- **Recommendation:** Do the 5-8 hour test first, save 4-8 hours debugging later

---

## 📊 Proposed Workflow

### Week 1 (Real E2E Testing)
- Day 1: Read proposal + master plans (30 min)
- Day 2: Setup (LM Studio, ML API, MCP servers) — 2-3 hours
- Day 3-4: Execute audits on 3 test contracts — 3-4 hours total
- Day 5: Analyze results + document baselines — 2-3 hours
- **Outcome:** Validated system + performance baselines + confidence score

### Week 2-3 (Phase A)
- Follow `01_PHASE_A_EXECUTION_PLAN.md` step-by-step
- Add reflection agent + multi-LLM debate + RAG expansion + consensus
- Write 15-20 new tests
- **Outcome:** ~250 PASS tests, improved verdict quality + explainability

### Week 4-7 (Phase B)
- Follow `02_PHASE_B_EXECUTION_PLAN.md`
- Add Halmos + Z3 + bytecode analysis + dataflow + access control + CVE matching
- Write 25-30 new tests
- **Outcome:** ~310 PASS tests, formal verification + bytecode-level inspection

### Week 5-7 (Phase C — Parallel with Phase B)
- Follow `03_PHASE_C_EXECUTION_PLAN.md`
- Add FastAPI gateway + evaluation framework + monitoring
- Write 15-20 new tests
- **Outcome:** ~325 PASS tests, production-ready infrastructure

### Week 8-12 (Phase D)
- Follow `04_PHASE_D_EXECUTION_PLAN.md`
- Add ItyFuzz + economic simulator + ZKML + on-chain submission
- Write 20-25 new tests
- **Outcome:** ~305 PASS tests, end-to-end audit proofs + on-chain integration

---

## 📖 Supporting References

### In SENTINEL Memory (Auto-Memory)
- **Location:** `~/.claude/projects/-home-motafeq-projects-sentinel/memory/`
- **Main Index:** `MEMORY.md` (agents section added)
- **Detailed State:** `2026-06-17_project_agents_state.md` (1,200+ lines, comprehensive)

### In SENTINEL Codebase
- **Proposal:** `docs/proposal/agent_proposal/AGENTS_EXTENDED_CAPABILITY_FINAL_PROPOSAL.md` (543 lines)
- **Bug Fix Commit:** `8c50fb8d7` (TOD class name fix)
- **Agent Code:** `agents/src/orchestration/{graph.py, nodes.py, state.py}`
- **Tests:** `agents/tests/` (219+ baseline PASS)
- **Old State Doc:** `agents/AGENTS_STATE_AND_REDESIGN_2026-06-14.md` (reference only)

---

## ✅ Success Criteria

### Real E2E Testing (Before Phase A)
- ✅ All 3 contracts audit without crashes
- ✅ Total time < 2 min per contract (120s threshold)
- ✅ LLM reasoning coherent (Prosecutor/Defender/Judge)
- ✅ Verdict accuracy > 70% on known contracts
- ✅ RAG precision > 50%
- ✅ Memory peak < 6GB
- ✅ No timeouts (MCP, LLM, ML)

### Phase A (Graph Enhancement)
- ✅ All new tests PASS + all 219 baseline tests still PASS
- ✅ ~250 PASS tests total
- ✅ Reflection agent working
- ✅ Multi-LLM debate implemented
- ✅ RAG corpus expanded
- ✅ Consensus voting + confidence tracking working
- ✅ Hotspot visualization implemented

### Phase B (Formal Verification + Bytecode)
- ✅ ~310 PASS tests total
- ✅ Halmos test generation working
- ✅ Z3 assertions verified
- ✅ Gigahorse bytecode analysis working
- ✅ Dataflow + access control + CVE matching implemented

### Phase C (Production)
- ✅ ~325 PASS tests total
- ✅ FastAPI gateway up + working
- ✅ Evaluation framework live
- ✅ Prompt injection guards active
- ✅ Monitoring + alerting live

### Phase D (Economic + On-Chain)
- ✅ ~305 PASS tests total
- ✅ ItyFuzz attack simulation working
- ✅ Economic simulator node working
- ✅ ZKML proof generation working
- ✅ On-chain audit submission working
- ✅ AuditRegistry integration complete

---

## 🔗 Cross-References

### From MEMORY.md
```
[[2026-06-17_project_agents_state]] — Detailed state (1,200 lines)
  • Bug fix details
  • Proposal summary
  • Plan overview
  • Architecture facts
  • Design decisions
```

### From Proposal
```
docs/proposal/agent_proposal/AGENTS_EXTENDED_CAPABILITY_FINAL_PROPOSAL.md
  • 3 paradigm descriptions
  • 8 supporting tools
  • Learning outcomes
  • Career value assessment
  • Total effort estimation
```

### From Plans
```
2026-06-17-extended-capability/ (implementation)
  • Phase A-D phased approach
  • ~200-245 hours total
  • Additive architecture
  • 4-phase sequencing

2026-06-17-agents-real-e2e-test/ (validation)
  • 9-node graph validation
  • Real LLM + MCP servers
  • Performance baselines
  • Bug discovery
```

---

## 🎓 What You'll Learn

### Phase A
- Multi-agent reasoning (reflection + debate)
- Consensus algorithms
- Confidence calibration
- Metric attribution (explainability)

### Phase B
- Property-based testing (Halmos)
- SMT solvers (Z3)
- Bytecode semantics (EVM)
- Dataflow analysis
- Access control matrices

### Phase C
- FastAPI production patterns
- Evaluation frameworks
- Security hardening (prompt injection)
- System observability

### Phase D
- Fuzzing (ItyFuzz)
- Economic game theory
- ZK proofs (EZKL)
- Smart contract verification

---

## 📞 Quick Answers

**Q: Should I do real E2E testing first?**  
A: Yes. 5-8 hours now saves 4-8 hours debugging later. Start with `2026-06-17-agents-real-e2e-test/01_SETUP_PLAN.md`.

**Q: Can I skip Phase C and go straight to Phase D?**  
A: No. Phase D depends on FastAPI gateway (Phase C). But B and C can run in parallel.

**Q: How long is Phase A?**  
A: 2-3 weeks, ~40-45 hours. Expect ~250 PASS tests after completion.

**Q: What if real E2E testing finds bugs?**  
A: Fix them before Phase A. Document in scratch file. Usually < 4 hours per bug.

**Q: Are all 9 existing nodes working?**  
A: Yes, 219+ baseline tests PASS. TOD bug fixed (Commit 8c50fb8d7).

**Q: Can I run Phase A and B in parallel?**  
A: Partially. A.1-A.2 must complete first (graph prep), then B can start.

---

## 📝 Version Info

**Created:** 2026-06-17  
**Status:** Ready for execution  
**Total Planning Effort:** 11 detailed documents, 4,200+ lines  
**Next Step:** Read `2026-06-17-agents-real-e2e-test/README.md` → Execute setup plan

---

**Ready to start?** Open `2026-06-17-agents-real-e2e-test/README.md` →
