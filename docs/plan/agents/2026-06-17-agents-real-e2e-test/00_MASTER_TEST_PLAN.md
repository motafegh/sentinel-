# Master Test Plan — Agents Real E2E Testing

**Duration:** 5-8 hours (one day)  
**Scope:** Test 9-node agents graph with real contracts, real LLM, real MCP servers  
**Goal:** Validate end-to-end execution, establish baselines, find bugs before Phase A  

---

## What We're Testing

The complete agents audit pipeline:

```
┌─ INPUT: Solidity contract source code
│
├─ ml_assessment         → Run 12 model (ML inference API :8001, MLOps standard)
├─ quick_screen          → Slither + Aderyn (static analysis)
├─ evidence_router       → Route to deep or fast path
│
├─ DEEP PATH (parallel):
│  ├─ rag_research       → RAG server :8011 (DeFiHackLabs corpus)
│  ├─ static_analysis    → Scoped Slither + Aderyn
│  ├─ graph_explain      → Graph inspector :8013 (hotspots)
│  │
│  ├─ audit_check        → Audit server :8012 (on-chain history)
│  ├─ cross_validator    → LLM calls (LM Studio) → Prosecutor/Defender/Judge
│  └─ synthesizer        → Final report assembly
│
└─ OUTPUT: Audit report with verdicts, confidence, evidence
```

**This tests:**
- Graph orchestration (LangGraph)
- MCP communication (all 4 servers)
- ML inference (Run 12 checkpoint)
- Static analysis (Slither + Aderyn)
- RAG retrieval (DeFiHackLabs)
- LLM reasoning (cross_validator)
- Report assembly

---

## Why Real Testing Matters

**Unit tests cover logic.** Real testing covers:

❌ **Not caught by tests:**
- MCP servers actually communicate?
- LM Studio timeout handling?
- RAG retrieval actually helps or just noise?
- Memory footprint on large contracts?
- Does synthesizer crash with 50+ findings?
- Are LLM verdicts coherent?
- Does graph handle concurrent parallel nodes?

✅ **Caught by real testing:**
- All of the above + integration issues
- Hidden assumptions (e.g., "RAG always returns docs")
- Performance real-world numbers
- Failure modes nobody anticipated

---

## Test Strategy

### Phase 1: Setup (2-3 hours)

Start infrastructure components in order:

1. **LM Studio** (local machine)
   - Load a chat model (e.g., Qwen 2.5 7B)
   - Available at :1234

2. **ML Inference API** (background)
   - Load Run 12 checkpoint
   - Available at :8001 (MLOps standard, matches agents inference_server)

3. **MCP Servers** (background, 4 of them)
   - inference_server (:8010) → wraps ML API
   - rag_server (:8011) → DeFiHackLabs RAG
   - audit_server (:8012) → mock AuditRegistry
   - graph_inspector_server (:8013) → GNN hotspots

4. **Verify connectivity**
   - All servers responding?
   - Graph can invoke agents?

### Phase 2: Test Contracts (3-4 hours, 1-2 hrs per contract)

Run 3 different contracts to see different patterns:

**Test 1: Known safe contract** (Uniswap V2 Router)
- What: Complex but generally secure
- Expected: Minimal vulns, all detected by tools
- Verifies: Can we audit production-grade code?
- Time: 1-2 hours

**Test 2: Intentionally vulnerable** (simple ERC20 with bugs)
- What: Obvious reentrancy + access control
- Expected: All vulns caught, high confidence
- Verifies: Can we detect known patterns?
- Time: 1-2 hours

**Test 3: Mixed realistic DeFi** (multi-contract pattern)
- What: Some safe, some suspicious
- Expected: Good recall, reasonable precision
- Verifies: Real-world quality?
- Time: 1-2 hours

### Phase 3: Analysis (2-3 hours)

For each contract, analyze:

1. **Latency breakdown**
   ```
   ml_assessment: 15s
   quick_screen: 8s
   rag_research: 6s
   static_analysis: 12s
   cross_validator: 20s
   synthesizer: 2s
   ─────────────────────
   Total: 63s
   ```

2. **Verdict quality**
   - Did prosecutor/defender arguments make sense?
   - Did judge render reasonable verdict?
   - Any hallucinations?

3. **RAG quality**
   - Were retrieved docs relevant?
   - Did RAG help or just noise?

4. **Bug log**
   - Any crashes?
   - Any timeouts?
   - Any MCP errors?

---

## Prerequisites Checklist

Before starting, verify you have:

**Hardware:**
- [ ] 16GB+ RAM (for LM Studio + ML API)
- [ ] 10GB+ disk (for models + indices)
- [ ] GPU preferred (not required, but slow without)

**Software:**
- [ ] LM Studio installed (or OpenAI API key)
  - Download from: https://lmstudio.ai/
  - Or use OpenAI API
- [ ] poetry installed (for ML API + agents)
- [ ] Slither + Aderyn in agents/.venv
- [ ] Run 12 checkpoint exists: `ml/checkpoints/GCB-P1-Run12-v3dospatched-20260613_FINAL.pt`
- [ ] RAG index exists: `agents/data/index/`

**Access:**
- [ ] Can start background processes (terminals)
- [ ] Can monitor memory/CPU (top, htop)
- [ ] Can read logs

---

## Key Measurements

### Performance Metrics

For each contract, measure:

| Metric | How | Note |
|--------|-----|------|
| ml_assessment latency | Time from input to ML result | Should be 10-30s |
| static_analysis latency | Slither + Aderyn runtime | Should be 5-20s |
| rag_research latency | Query to results | Should be 2-5s |
| cross_validator latency | LLM calls (3 of them) | Should be 15-30s |
| Total end-to-end | Start to final report | Should be < 5 min |
| Memory peak | Max RAM used | Should be < 8GB |
| RAG relevance | % retrieved docs relevant | Should be > 70% |

### Quality Metrics

| Metric | How | Note |
|--------|-----|------|
| Verdict coherence | Do P/D/J output make sense? | Qualitative |
| False positive rate | Safe contracts flagged? | Count for test 1 |
| False negative rate | Known vulns missed? | Count for test 2 |
| Confidence calibration | Do confidence scores match accuracy? | Qualitative |

---

## Failure Scenarios & Recovery

**If LM Studio doesn't respond:**
- Fallback: Mock LLM responses (pre-written verdicts)
- Lesson: Need LLM timeout + retry logic

**If MCP server crashes:**
- Fallback: Replace with mock
- Lesson: Need server health checks (Phase C)

**If ML API OOM:**
- Fallback: Reduce batch size or use smaller model
- Lesson: Need memory management tuning

**If RAG returns empty:**
- Fallback: Skip RAG, continue with other tools
- Lesson: Corpus needs expansion (Phase A)

**If total time > 10 minutes:**
- Fallback: Still OK, document baseline
- Lesson: Need optimization (Phase C consideration)

---

## Success Criteria

✅ **Execution:** Graph runs without crashes on all 3 contracts  
✅ **LLM quality:** Prosecutor/Defender/Judge output reads well  
✅ **Verdict coherence:** Verdicts match evidence  
✅ **Performance:** End-to-end < 5 minutes per contract  
✅ **Stability:** No timeouts, no OOM  
✅ **Bug discovery:** Log any issues found  

---

## Deliverables

After completing this test, you'll have:

✅ **Performance baselines** (latency per node)  
✅ **3 real audit reports** (for portfolio review)  
✅ **Bug list** (if any found)  
✅ **RAG quality assessment** (is corpus helping?)  
✅ **LLM reasoning samples** (processor/defender/judge)  
✅ **Confidence in architecture** ("ready for Phase A" or "need fixes first")  

---

## Timeline

**Hour 0-1:** Setup (install, launch LM Studio)  
**Hour 1-3:** Start ML API, MCP servers, verify connectivity  
**Hour 3-5:** Test contract 1 (Uniswap)  
**Hour 5-7:** Test contract 2 (vulnerable)  
**Hour 7-8:** Test contract 3 (mixed) + analysis  
**Hour 8+:** Documentation + findings

---

## Next Steps

1. ✓ Read this master plan (done)
2. → Follow `01_SETUP_PLAN.md` (2-3 hours)
3. → Execute `02_EXECUTION_PLAN.md` (per contract)
4. → Analyze `03_ANALYSIS_PLAN.md` (findings)

---

## Key Files

- **Graph:** `agents/src/orchestration/graph.py`
- **Nodes:** `agents/src/orchestration/nodes.py`
- **ML API:** `ml/src/inference/api.py`
- **MCP servers:** `agents/src/mcp/servers/*.py`
- **Test script:** Will be created in `02_EXECUTION_PLAN.md`

---

**Ready to start?** Open `01_SETUP_PLAN.md` →
