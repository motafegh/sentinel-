# Agents Module Real End-to-End Testing Plan

**Purpose:** Test the agents module against real contracts with real LLM calls, real MCP servers, real ML inference — not mocks or stubs.

**Goal:** Validate that the 9-node graph works end-to-end, establish performance baselines, find hidden bugs, and build confidence before Phase A implementation.

**Effort:** 5-8 hours (one full day)

---

## Files in This Folder

| File | Purpose | Duration |
|------|---------|----------|
| **README.md** | This index | 5 min |
| **00_MASTER_TEST_PLAN.md** | Overview, prerequisites, strategy | 20 min read |
| **01_SETUP_PLAN.md** | Step-by-step environment setup | 2-3 hours execution |
| **02_EXECUTION_PLAN.md** | How to run real audits | 1-2 hours per contract |
| **03_ANALYSIS_PLAN.md** | Analyzing results + debugging | 2-3 hours |

---

## Quick Start (TL;DR)

1. **Read Master Plan** (20 min) — understand what we're testing and why
2. **Run Setup** (2-3 hours) — launch LM Studio, ML API, MCP servers
3. **Execute Tests** (1-2 hours) — run audits on 3 test contracts
4. **Analyze Results** (2-3 hours) — document findings, identify bugs

**Total: 5-8 hours**

---

## What Gets Tested

✅ **Full 9-node graph execution:**
```
ml_assessment → quick_screen → evidence_router → 
  [rag_research, static_analysis, graph_explain] → 
  audit_check → cross_validator → synthesizer
```

✅ **Real LLM reasoning** (cross_validator uses LM Studio)

✅ **Real MCP servers** (4 servers, not mocks)

✅ **Real ML inference** (Run 12 checkpoint)

✅ **Real RAG retrieval** (DeFiHackLabs corpus)

✅ **Real output** (audit reports with verdicts)

---

## Test Contracts

**3 contracts to audit:**

1. **Uniswap V2 Router** (known safe-ish)
   - Complex, realistic, no critical vulns expected
   - Baseline: what does a safe contract audit look like?

2. **Simple vulnerable ERC20** (intentionally buggy)
   - Reentrancy + access control issues
   - Baseline: can we detect intentional vulns?

3. **Mixed contract** (realistic DeFi)
   - Some safe patterns, some suspicious patterns
   - Baseline: how do we handle real-world code?

---

## What You'll Measure

**Performance (timing per node):**
- ml_assessment latency
- static_analysis latency (Slither + Aderyn)
- rag_research latency
- cross_validator latency (LLM calls)
- synthesizer latency
- Total end-to-end time

**Quality (reasoning + verdicts):**
- LLM reasoning quality (prosecutor/defender/judge output)
- Verdict consistency (does ML + Slither + RAG agree?)
- False positive rate (safe contracts flagged as vulnerable?)
- False negative rate (known vulns missed?)

**Stability (crashes, timeouts, OOM):**
- Any nodes crash?
- Any MCP servers hang?
- Memory footprint per contract
- CPU usage

---

## Success Criteria

✅ **Graph execution:** All 9 nodes execute without crashes  
✅ **LLM quality:** Prosecutor/defender/judge reasoning makes sense  
✅ **Verdict stability:** Results are consistent and defensible  
✅ **Performance:** End-to-end audit < 5 minutes for typical contract  
✅ **RAG quality:** Retrieved patterns are relevant to findings  
✅ **No hidden bugs:** Integration issues discovered and logged  

---

## How to Use This Plan

**For technical execution:**
1. `00_MASTER_TEST_PLAN.md` — understand strategy
2. `01_SETUP_PLAN.md` — follow setup checklist
3. `02_EXECUTION_PLAN.md` — run audits step-by-step
4. `03_ANALYSIS_PLAN.md` — analyze + debug

**For tracking:**
- Use setup checklist in 01_SETUP_PLAN.md
- Use test execution checklist in 02_EXECUTION_PLAN.md
- Use analysis checklist in 03_ANALYSIS_PLAN.md
- Document findings in scratch file (see 03_ANALYSIS_PLAN.md)

**For troubleshooting:**
- Each plan has "Common Issues" section
- Debugging strategies included
- Fallback options provided

---

## Deliverables (What You'll Have After This)

✅ **Performance baselines** — latency per node (data)

✅ **Test report** — markdown document with findings

✅ **Real audit outputs** — 3 actual audit reports to review

✅ **Bug list** — any issues found with fixes

✅ **Confidence score** — "the system works / needs X fixes"

✅ **Learning document** — what we discovered about the system

---

## Why Do This Before Phase A?

1. **Find bugs early** — better to debug now than in Phase A
2. **Validate assumptions** — latency, LLM quality, RAG relevance
3. **Establish baselines** — know performance before adding complexity
4. **Build confidence** — see the system actually work end-to-end
5. **Real output** — get actual audit reports to review
6. **Phase A ready** — understand MCP failure modes before redesigning them (Phase C)

**Risk:** Find 2-3 bugs → fix them → Phase A starts cleaner  
**Benefit:** Avoid finding those bugs in Phase A, B, C later

---

## Next Steps

1. ✅ Read this README (you're here)
2. → Read `00_MASTER_TEST_PLAN.md` (20 min)
3. → Follow `01_SETUP_PLAN.md` (2-3 hours, one-time)
4. → Execute `02_EXECUTION_PLAN.md` (1-2 hours per contract)
5. → Analyze `03_ANALYSIS_PLAN.md` (2-3 hours)

**Ready?** Open `00_MASTER_TEST_PLAN.md` →

---

## Key Files Referenced

- **Agents graph:** `agents/src/orchestration/graph.py`
- **Nodes:** `agents/src/orchestration/nodes.py`
- **MCP servers:** `agents/src/mcp/servers/`
- **ML API:** `ml/src/inference/api.py`
- **LLM client:** `agents/src/llm/client.py`
- **RAG:** `agents/src/rag/` + `agents/data/index/`

---

## Timeline (One Day)

```
08:00 - 08:20  Read master plan
08:20 - 10:30  Run setup (LM Studio, ML API, MCP servers)
10:30 - 12:00  Test contract 1 (Uniswap)
12:00 - 13:30  Lunch + review results
13:30 - 15:00  Test contract 2 (vulnerable ERC20)
15:00 - 16:30  Test contract 3 (mixed DeFi)
16:30 - 18:00  Analysis + documentation
18:00 - END    Review findings, write report
```

**Or spread across 2-3 days if preferred.**

---

## Changelog

- **v1.1 (2026-06-17):** Plan-vs-code audit applied. Fixed 5 discrepancies (LM Studio URL, model names, AUDIT_MOCK, RAG fetchers, test contracts). Status: **docs-only**, execution deferred. See `CHANGELOG.md`.
- **v1.0 (2026-06-17):** Initial plan. 5 docs, ~1,885 lines.

---

**Created:** 2026-06-17
**Version:** 1.1
**Status:** 📚 **Docs-only — execution deferred.** Plan validated against source code; ready to execute when Ali commits the time window. CHANGELOG records the 5 fixes.
