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

## LLM Model Selection (Reference)

When loading models in LM Studio, use these exact IDs (verified against `agents/src/llm/client.py:51-58`):

| Role | Model ID | Used by | Why |
|---|---|---|---|
| `MODEL_FAST` | `gemma-4-e2b-it` | Tool selection, simple routing | Small, fast, low-latency |
| `MODEL_STRONG` | `qwen3.5-9b-ud` | `cross_validator` (P/D/J debate), report gen | Reasoning quality on long context |
| `MODEL_CODER` | `qwen2.5-coder-7b-instruct` | Solidity code review | Trained on 80+ languages incl. Solidity |
| `MODEL_EMBED` | `text-embedding-nomic-embed-text-v1.5` | RAG embeddings | 8K context, top of MTEB |

Verify loaded models with:
```bash
curl -s $LM_STUDIO_BASE_URL/models | jq '.data[].id'
```
Expect all 4 IDs above. Missing any → load before starting.

→ You now know: The plan originally said "Qwen 2.5 7B or Mistral-7B" but the actual code (`client.py:51-58`) has 4 specific model IDs. Loading the wrong models = silent fallback or 404 from LM Studio. Always cross-check with the `curl` above.

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
- [ ] **Slither + Aderyn in `agents/.venv`** — verify with:
  ```bash
  cd ~/projects/sentinel/agents
  poetry run slither --version   # expect 0.11.x
  poetry run aderyn --version   # expect ≥ 0.4.21 (pre-0.4.21 has known errors, see Run 12 eval)
  ```
- [ ] Run 12 checkpoint exists: `ml/checkpoints/GCB-P1-Run12-v3dospatched-20260613_FINAL.pt`
- [ ] RAG index exists: `agents/data/index/`
- [ ] `agents/.env` has: `LM_STUDIO_BASE_URL`, `AUDIT_MOCK=true`, `LM_STUDIO_API_KEY="lm-studio"`

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

---

## Learning Outcomes (Plan Onboarding)

→ You now know: This E2E test exists because no real E2E test exists — `tests/test_smoke_e2e.py` mocks all MCP calls (per its own docstring), and `scripts/smoke_*.py` only test individual servers. So this run is the first time the 9-node graph executes against real LLM + real MCP + real ML.

→ You now know: The 9 nodes (`nodes.py:162,291,336,409,493,676,842,941,1103`) and 4 MCP ports (8010/8011/8012/8013) and ML API port 8001 are all verified against code. The 5 v1.1 fixes are documented in `CHANGELOG.md`.

→ You now know: The plan's critical premise (5-8 hours saves 4-8 hours of debugging in Phase A) is valid because Phase A's first 3 steps (graph cleanup, reflection, debate) WILL interact with cross_validator + audit_check + rag_research — and any of those 3 having a real bug is a Phase A blocker.
