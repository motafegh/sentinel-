# Phase A Execution Plan — Graph Enhancement & Foundations

**Duration:** 2-3 weeks  
**Effort:** Medium  
**Tests to add:** 15-20  
**Outcome:** Reflection, debate, consensus voting, confidence tracking, RAG expansion

---

## Quick Reference: What Gets Done

```
A.1 Graph cleanup (remove side effects)
A.2 AuditState schema extension (8 new fields)
A.3 Reflection agent (self-critique post-synthesizer)
A.4 Multi-LLM debate (Prosecutor/Defender/Judge)
A.5 RAG expansion (Code4rena, Sherlock, Solodit, Immunefi, SWC)
A.6 Tool consensus voting (weighted per-class voting)
A.7 Staged confidence tracking (Bayesian updating)
A.8 Metric attribution (LIME-style breakdown)
A.9 Hotspot attribution visualization (interactive HTML)
```

---

## A.1: Graph Cleanup (3 days)

**Why:** Remove module-level side effects, enable lazy node loading

**What to do:**

1. **Read current state:**
   - File: `agents/src/orchestration/graph.py`
   - Find line: `audit_graph = build_graph()` (should be at end of file, ~210+)
   - This is a module-level side effect (runs at import time)

2. **Remove side effect:**
   - Delete the line `audit_graph = build_graph()`
   - Delete any module-level calls to `build_graph()`

3. **Update imports:**
   - Files that import `audit_graph`: search `grep -r "from.*graph import audit_graph"` in agents/
   - Change: `from agents.src.orchestration.graph import audit_graph` 
   - To: `from agents.src.orchestration.graph import build_graph; audit_graph = build_graph(use_checkpointer=True)`
   - Or better: pass builder function, call at startup

4. **Update tests:**
   - File: `agents/tests/` (all test files)
   - Some tests call `build_graph(use_checkpointer=False)` already ✅
   - Some may import `audit_graph` → update those
   - Check: `grep -r "audit_graph" agents/tests/`

5. **Verify:**
   - Run: `cd agents && poetry run pytest tests/ -q`
   - Expected: 219+ PASS (no regression)

**Files to change:**
- `agents/src/orchestration/graph.py` — remove module-level `audit_graph = ...`
- Any files importing it — search and update

**Success criteria:**
- ✅ `build_graph()` function exists and works
- ✅ No module-level side effects
- ✅ 219+ tests still PASS
- ✅ Tests can control checkpointer via parameter

---

## A.2: AuditState Schema Extension (2 days)

**Why:** Add fields for Phase A/B nodes before they implement them

**What to do:**

1. **Open file:**
   - `agents/src/orchestration/state.py`
   - Current size: ~160 lines
   - All fields use `total=False` (all optional) ✅

2. **Add Phase A fields:**
   - After line 150 (final_report field), add:
   ```python
   # Phase A — reflection & analysis
   reflection_notes: str | None
   # Structured self-critique from reflection agent
   
   taint_flows: list[dict[str, Any]]
   # Dataflow: {source, sink, path} from taint analyzer
   
   permission_graph: dict[str, list[str]]
   # Access control: {role: [callable_functions]}
   
   consensus_verdict: dict[str, dict]
   # Tool voting: {class: {ml_confidence, slither_match, aderyn_match, final_confidence}}
   ```

3. **Add Phase B fields:**
   ```python
   # Phase B — symbolic & bytecode
   symbolic_findings: list[dict[str, Any]]
   # Halmos: {invariant, proven, counterexample_if_false}
   
   bytecode_analysis: dict[str, Any]
   # Gigahorse: {cfg, call_graph, inferred_logic, storage_layout}
   ```

4. **Verify existing field:**
   - Line 115-117: `econ_scenarios` already exists ✅
   - Just ensure comment is clear

5. **Test:**
   - Update any tests that validate state schema
   - File: `agents/tests/test_orchestration.py` or similar
   - Verify TypedDict accepts new fields

**Files to change:**
- `agents/src/orchestration/state.py` — add ~20 lines

**Success criteria:**
- ✅ All new fields present in AuditState
- ✅ All marked as optional (total=False) ✅
- ✅ Schema tests PASS

---

## A.3: Reflection Agent (5-7 days)

**Why:** Self-critique of verdicts — catch internal inconsistencies

**What to do:**

1. **Create node file:**
   - File: `agents/src/orchestration/nodes.py`
   - Add function after `synthesizer` (line 1415+):
   ```python
   async def reflection(state: AuditState) -> dict[str, Any]:
       """
       Self-critique pass: verify audit consistency.
       
       Checks:
       - Unused evidence? (collected but not cited)
       - Contradictions? (SAFE verdict but Slither found bug)
       - Uncertain verdicts? (flag confidence < 0.7)
       - Failure modes? (what could make this audit wrong?)
       """
   ```

2. **Implement logic:**
   - Read: `state["final_report"]`, `state["verdicts"]`, `state["static_findings"]`, `state["rag_results"]`
   - Call `get_strong_llm()` with reflection prompt:
   ```
   Given:
     - ML signals: {class: prob}
     - Static findings: [detectors found]
     - RAG results: [matched patterns]
     - Verdicts: {class: CONFIRMED|LIKELY|DISPUTED|SAFE}
   
   Critique:
     1. Are signals consistent with verdicts?
     2. What evidence is unused?
     3. Which verdicts are uncertain?
     4. What could be wrong?
   ```

3. **Parse LLM output:**
   - Extract structured JSON or markdown
   - Store in `state["reflection_notes"]`

4. **Update graph routing:**
   - File: `agents/src/orchestration/graph.py` (line ~200)
   - Add: `graph.add_edge("synthesizer", "reflection")`
   - Add: `graph.add_edge("reflection", END)`

5. **Write tests:**
   - Mock LLM response
   - Verify reflection_notes is set
   - Test 5 cases: all signals agree, signals conflict, high confidence, low confidence, mixed

**Files to change:**
- `agents/src/orchestration/nodes.py` — add ~60 lines (reflection function)
- `agents/src/orchestration/graph.py` — add 2 lines (edges)

**Test file to create:**
- `agents/tests/test_reflection.py` — 5-6 test cases

**Success criteria:**
- ✅ Reflection node runs after synthesizer
- ✅ reflection_notes populated correctly
- ✅ 5-6 tests PASS
- ✅ 219+ baseline tests still PASS

---

## A.4: Multi-LLM Debate Upgrade (5-7 days)

**Why:** Three roles improve verdict quality vs single LLM pass

**What to do:**

1. **Read current cross_validator:**
   - File: `agents/src/orchestration/nodes.py:941-1096`
   - Currently makes 1 LLM call
   - Returns verdicts dict

2. **Refactor to three roles:**
   ```python
   async def cross_validator(state: AuditState) -> dict[str, Any]:
       # 1. Prosecutor: "Why IS this vulnerable?"
       prosecutor = await get_strong_llm().apredict(
           f"You are a security prosecutor. Given evidence, argue why "
           f"this contract HAS these vulnerabilities. Be specific. "
           f"Evidence: {json.dumps(evidence_bundle)}"
       )
       
       # 2. Defender: "Why is it NOT vulnerable?"
       defender = await get_strong_llm().apredict(
           f"You are a skeptical defender. Given the prosecutor's case, "
           f"argue why these findings are false positives or low-severity. "
           f"Evidence: {json.dumps(evidence_bundle)}"
       )
       
       # 3. Judge: "Render verdict"
       judge = await get_strong_llm().apredict(
           f"You are a judge. You've heard the prosecutor and defender. "
           f"Render per-class verdicts: CONFIRMED | LIKELY | DISPUTED | SAFE. "
           f"Prosecutor: {prosecutor}\nDefender: {defender}"
       )
   ```

3. **Parse judge output:**
   - Extract structured JSON with verdicts
   - Return same format as before (verdicts dict)

4. **Timeout handling:**
   - If any LLM call times out, fall back to rule-based verdicts
   - Log warning

5. **Update graph:**
   - No graph changes needed (same node, different internals)

6. **Write tests:**
   - Mock all three LLM calls
   - Test: prosecutor + defender + judge produce reasonable verdicts
   - Test: timeout fallback
   - Test: format parsing
   - 6-8 test cases

**Files to change:**
- `agents/src/orchestration/nodes.py` (lines 941-1096) — refactor cross_validator
- No graph changes

**Test file to update:**
- `agents/tests/test_cross_validator.py` — add 6-8 cases

**Success criteria:**
- ✅ Three LLM calls working
- ✅ Verdicts parsed correctly
- ✅ Fallback to rule-based works
- ✅ 8+ tests PASS
- ✅ 219+ baseline tests still PASS

---

## A.5: RAG Corpus Expansion (5-7 days)

**Why:** 60K+ findings vs 726 DeFiHackLabs gives much better retrieval

**What to do:**

1. **Create fetchers:**
   - Directory: `agents/src/rag/fetchers/`
   - Create 5 new files:

   **A.5a: Code4rena Fetcher** (2-3 days)
   - Source: Code4rena API or JSON export
   - Parse findings by: vulnerability type, severity, affected contract
   - Output: List of dicts with {title, description, severity, type, contract_code}
   - File: `code4rena_fetcher.py` (~50 lines)

   **A.5b: Sherlock Fetcher** (1-2 days)
   - Source: Sherlock audit reports + API
   - Focus on: oracle manipulation, MEV, state management
   - File: `sherlock_fetcher.py` (~40 lines)

   **A.5c: Solodit Fetcher** (1-2 days)
   - Source: Solodit aggregation (web scrape or API)
   - Parse by: vuln type, timestamp, severity
   - File: `solodit_fetcher.py` (~40 lines)

   **A.5d: Immunefi Fetcher** (1-2 days)
   - Source: Immunefi bounty disclosures
   - Extract: root cause, impact, fix
   - File: `immunefi_fetcher.py` (~40 lines)

   **A.5e: SWC Registry** (1 day)
   - Source: SWC (Solidity Weakness Classification) JSON
   - Parse 118 weakness types
   - File: `swc_registry_fetcher.py` (~30 lines)

2. **Update index builder:**
   - File: `agents/src/rag/build_index.py`
   - Call all 5 new fetchers + existing DeFiHackLabs
   - Combine into single corpus
   - Re-index FAISS + BM25

3. **Test fetchers:**
   - Each fetcher has unit test (mock data)
   - Verify output format consistency
   - 5-6 test cases total

4. **Re-build index:**
   - Run: `poetry run python agents/src/rag/build_index.py`
   - Expect: ~60K docs
   - Verify FAISS + BM25 indices created
   - Estimate time: 30-60 minutes

**Files to change:**
- `agents/src/rag/fetchers/code4rena_fetcher.py` — NEW (50 lines)
- `agents/src/rag/fetchers/sherlock_fetcher.py` — NEW (40 lines)
- `agents/src/rag/fetchers/solodit_fetcher.py` — NEW (40 lines)
- `agents/src/rag/fetchers/immunefi_fetcher.py` — NEW (40 lines)
- `agents/src/rag/fetchers/swc_registry_fetcher.py` — NEW (30 lines)
- `agents/src/rag/build_index.py` — UPDATE (add fetcher calls)

**Test file to create:**
- `agents/tests/test_rag_fetchers.py` — 5-6 test cases

**Success criteria:**
- ✅ All 5 fetchers working
- ✅ Corpus built with ~60K docs
- ✅ FAISS + BM25 indices created
- ✅ Retrieval quality improved (spot-check query results)

---

## A.6: Tool Consensus Voting (4-5 days)

**Why:** When Slither + Aderyn + ML disagree, weighted vote decides

**What to do:**

1. **Create consensus module:**
   - File: `agents/src/orchestration/consensus.py` (NEW, ~100 lines)
   - Function: `consensus_vote(ml_probs, slither_matches, aderyn_matches, class_name) → (verdict, confidence)`

2. **Implement voting logic:**
   ```python
   def consensus_vote(ml_prob, slither_found, aderyn_found, class_name):
       # Per-class accuracy from Run 12 benchmarks
       # Example: Reentrancy: ML=0.78, Slither=0.82, Aderyn=0.65
       weights = ACCURACY_WEIGHTS[class_name]  # from calibration data
       
       signals = {
           'ml': 1 if ml_prob >= 0.50 else 0,
           'slither': 1 if slither_found else 0,
           'aderyn': 1 if aderyn_found else 0
       }
       
       # Weighted sum
       score = sum(signals[k] * weights[k] for k in signals)
       confidence = score / sum(weights.values())
       
       verdict = 'CONFIRMED' if confidence >= 0.7 else 'LIKELY' if confidence >= 0.5 else 'DISPUTED'
       return verdict, confidence
   ```

3. **Add consensus node to orchestration:**
   - File: `agents/src/orchestration/nodes.py`
   - New function: `consensus_engine` (after audit_check)
   - Takes: ml_result, static_findings, route through consensus_vote
   - Output: `state["consensus_verdict"]` dict

4. **Wire into graph:**
   - File: `agents/src/orchestration/graph.py`
   - Add: `graph.add_edge("audit_check", "consensus_engine")`
   - Add: `graph.add_edge("consensus_engine", "cross_validator")`
   - Remove: `graph.add_edge("audit_check", "cross_validator")`

5. **Write tests:**
   - Test voting logic: all agree, disagree, mixed
   - Test edge cases: 0 signals, 1 signal
   - Test confidence bounds: 0-1 range
   - 6 test cases

**Files to change:**
- `agents/src/orchestration/consensus.py` — NEW (100 lines)
- `agents/src/orchestration/nodes.py` — add consensus_engine (~40 lines)
- `agents/src/orchestration/graph.py` — update edges

**Test file to create:**
- `agents/tests/test_consensus_voting.py` — 6 test cases

**Success criteria:**
- ✅ consensus_vote logic correct
- ✅ Confidence in [0, 1] range
- ✅ 6 tests PASS
- ✅ Node integrates into graph

---

## A.7: Staged Confidence Tracking (4-5 days)

**Why:** Track uncertainty growth/shrinkage through evidence pipeline

**What to do:**

1. **Create confidence tracker:**
   - File: `agents/src/orchestration/confidence.py` (NEW, ~80 lines)
   - Track per-class confidence through pipeline

2. **Implement Bayesian updating:**
   ```python
   # Start with ML confidence
   conf = ml_prob
   
   # Update with Slither evidence
   if slither_found:
       conf = conf * 1.1  # boost if Slither agrees
   else:
       conf = conf * 0.9  # reduce if Slither disagrees
   
   # Update with RAG
   if rag_score >= 0.7:
       conf = conf * 1.05  # minor boost from RAG
   
   # Clamp to [0, 1]
   conf = max(0, min(1, conf))
   ```

3. **Integrate into orchestration:**
   - Update: `consensus_engine` (A.6) to output confidence
   - Update: `cross_validator` to use + update confidence
   - Store in state: track confidence per class over pipeline

4. **Expose in final_report:**
   - Add field: `final_report["confidence_by_class"]` = {class: confidence}
   - Example: {"Reentrancy": 0.92, "IntegerUO": 0.54, ...}

5. **Write tests:**
   - Test Bayesian updating logic
   - Test bounds [0, 1]
   - Test pipeline flow: ML → Slither → RAG → verdict
   - 5-6 test cases

**Files to change:**
- `agents/src/orchestration/confidence.py` — NEW (80 lines)
- `agents/src/orchestration/nodes.py` — update consensus_engine + cross_validator to use confidence tracker
- `agents/src/orchestration/state.py` — ensure final_report includes confidence field

**Test file to create:**
- `agents/tests/test_confidence_tracking.py` — 5-6 test cases

**Success criteria:**
- ✅ Confidence tracking working end-to-end
- ✅ Confidence values in [0, 1]
- ✅ final_report includes confidence_by_class
- ✅ 5-6 tests PASS

---

## A.8: Metric Attribution (5-7 days)

**Why:** Explain verdicts — "60% from ML, 30% from Slither, 10% from RAG"

**What to do:**

1. **Create attribution module:**
   - File: `agents/src/orchestration/attribution.py` (NEW, ~120 lines)
   - Function: `attribute_verdict(ml_prob, slither_match, rag_score) → {ml_pct, slither_pct, rag_pct}`

2. **Implement LIME-style attribution:**
   ```python
   def attribute_verdict(ml_prob, slither_match, rag_score):
       # Normalize contributions
       ml_contrib = ml_prob
       slither_contrib = 1.0 if slither_match else 0.0
       rag_contrib = max(0, rag_score - 0.3)  # only if notable
       
       total = ml_contrib + slither_contrib + rag_contrib
       
       return {
           'ml_pct': round(100 * ml_contrib / total, 1),
           'slither_pct': round(100 * slither_contrib / total, 1),
           'rag_pct': round(100 * rag_contrib / total, 1)
       }
   ```

3. **Add attribution node:**
   - File: `agents/src/orchestration/nodes.py`
   - New function: `explainer` (after synthesizer)
   - Takes: verdicts + evidence sources
   - Output: `state["metric_attribution"]` dict

4. **Wire into graph:**
   - File: `agents/src/orchestration/graph.py`
   - Add: `graph.add_edge("reflection", "explainer")`
   - Add: `graph.add_edge("explainer", END)`

5. **Add to final_report:**
   - Include attribution breakdown per verdict
   - Example:
   ```json
   {
       "Reentrancy": {
           "verdict": "CONFIRMED",
           "confidence": 0.92,
           "attribution": {
               "ml_pct": 60,
               "slither_pct": 30,
               "rag_pct": 10
           }
       }
   }
   ```

6. **Write tests:**
   - Test attribution logic
   - Verify percentages sum to 100%
   - Test edge cases (0 evidence, 1 source)
   - 6 test cases

**Files to change:**
- `agents/src/orchestration/attribution.py` — NEW (120 lines)
- `agents/src/orchestration/nodes.py` — add explainer (~50 lines)
- `agents/src/orchestration/graph.py` — add edges

**Test file to create:**
- `agents/tests/test_metric_attribution.py` — 6 test cases

**Success criteria:**
- ✅ Attribution percentages sum to 100%
- ✅ Node integrates into graph
- ✅ final_report includes attribution data
- ✅ 6 tests PASS

---

## A.9: Hotspot Attribution Visualization (6-8 days)

**Why:** Interactive HTML shows which code lines/nodes matter

**What to do:**

1. **Create visualizer module:**
   - File: `agents/src/orchestration/visualizer.py` (NEW, ~200 lines)
   - Function: `generate_hotspot_html(state) → html_string`

2. **Implement HTML generation:**
   - Input: contract_code, graph_explanations, verdicts, ml_hotspots
   - Output: Interactive HTML with:
     - Left side: source code with line highlighting
     - Right side: AST/CFG graph with node highlighting
     - Clickable functions → show subgraph

3. **Use visualization library:**
   - Graphviz (for CFG) or D3.js (for interactivity)
   - Or simple HTML table + CSS highlighting

4. **Add visualizer node:**
   - File: `agents/src/orchestration/nodes.py`
   - New function: `visualizer` (last node before END)
   - Takes: all state
   - Output: state["hotspot_visualization"] (HTML string)
   - Also write to file: `agents/data/reports/{contract_address}/hotspot.html`

5. **Wire into graph:**
   - File: `agents/src/orchestration/graph.py`
   - Add: `graph.add_edge("explainer", "visualizer")`
   - Add: `graph.add_edge("visualizer", END)`

6. **Write tests:**
   - Mock state with sample contract
   - Verify HTML is valid (parse with BeautifulSoup)
   - Check for required elements (code, graph, highlighting)
   - Test CSS classes applied correctly
   - 4-5 test cases

**Files to change:**
- `agents/src/orchestration/visualizer.py` — NEW (200 lines)
- `agents/src/orchestration/nodes.py` — add visualizer (~60 lines)
- `agents/src/orchestration/graph.py` — add edges

**Test file to create:**
- `agents/tests/test_visualizer.py` — 4-5 test cases

**Success criteria:**
- ✅ HTML generated successfully
- ✅ HTML is valid (no parsing errors)
- ✅ Contains code + graph visualization
- ✅ Nodes highlighted appropriately
- ✅ 4-5 tests PASS

---

## Phase A Summary

After Phase A, you will have:

✅ **Clean graph** (no side effects)  
✅ **Extended schema** (8 new fields)  
✅ **Reflection agent** (self-critique)  
✅ **Multi-LLM debate** (Prosecutor/Defender/Judge)  
✅ **RAG corpus** (60K+ findings)  
✅ **Consensus voting** (weighted per-class)  
✅ **Confidence tracking** (Bayesian updating)  
✅ **Metric attribution** (LIME-style)  
✅ **Visualization** (interactive hotspots)  
✅ **35-45 new tests** (total: ~250 PASS)

**Ready to proceed to Phase B** (symbolic execution + bytecode analysis)

---

## Testing Checklist

- [ ] A.1: Graph cleanup — 219+ tests PASS
- [ ] A.2: Schema extension — new fields accept data
- [ ] A.3: Reflection agent — 5-6 tests PASS
- [ ] A.4: Multi-LLM debate — 8 tests PASS
- [ ] A.5: RAG expansion — fetchers work, index built (~60K docs)
- [ ] A.6: Consensus voting — 6 tests PASS
- [ ] A.7: Confidence tracking — 5-6 tests PASS
- [ ] A.8: Metric attribution — 6 tests PASS
- [ ] A.9: Visualization — 4-5 tests PASS
- [ ] Full integration: `poetry run pytest agents/tests/ -q` → ~250 PASS
- [ ] Smoke test with real contract (e.g., Uniswap V2)

---

## References

- Proposal: `docs/proposal/agent_proposal/AGENTS_EXTENDED_CAPABILITY_FINAL_PROPOSAL.md` §3-4
- Master plan: `docs/plan/agents/2026-06-17-extended-capability/00_MASTER_EXECUTION_PLAN.md`
- Current graph: `agents/src/orchestration/graph.py`
- Current nodes: `agents/src/orchestration/nodes.py`
- LLM client: `agents/src/llm/client.py` (use `get_strong_llm()`)
- RAG retriever: `agents/src/rag/retriever.py`
- Tests baseline: `agents/tests/test_smoke_e2e.py`, `test_routing_phase0.py`
