# SENTINEL Agents Module — Final Architecture & Actionable Plan

**Date:** 2026-05-20  
**Status:** Design locked; Phase 0 ready to implement  
**Supersedes:** `agents-module-suggestions.md`, `agents-modules-proposal-2.md`

---

## 1. Core Philosophy

The ML module produces a probability vector. A developer needs a verdict.

The gap between `{reentrancy: 0.87}` and "this function is exploitable because it sends ETH before updating state, matching the DAO pattern — here is the fix" is too large for rule-based code to cross. That is the agents layer's only job: **reconcile multiple independent evidence sources into a reasoned, traceable verdict.**

Three principles that constrain every design decision:

1. **ML is triage, not the judge.** Probabilities tell the agent *where to look*, not *what the verdict is*. Every downstream tool either confirms, refutes, or adds nuance.
2. **Verdicts must be traceable.** Every conclusion in the final report must cite which tools produced which evidence. A report that says "CONFIRMED" without showing the evidence chain is not auditing — it is marketing.
3. **Cost must be proportional to risk.** A contract where all probabilities are below 0.35 and Slither finds nothing does not justify an LLM call. An ambiguous high-TVL contract justifies deep analysis. Design tiers explicitly; never apply one depth to everything.

---

## 2. Architecture Decision

**LangGraph StateGraph — extended, not replaced.**

The existing 5-node graph (`ml_assessment → [rag_research ‖ static_analysis] → audit_check → synthesizer`) has the right skeleton. The decision is to extend it to 8 nodes by adding `evidence_router`, `cross_validator`, and keeping `graph_explain` as a Phase 1 addition.

**Why not separate agent processes (Proposal 1's model):**
- The Orchestrator/Triage/Analysis/Explanation/Fix split adds inter-process message passing overhead with no gain at current scale.
- LangGraph nodes share `AuditState` in a single process — the same cognitive separation happens through node boundaries without the complexity of separate agent loops.
- The `cross_validator` node achieves what Proposal 1 calls the "Debate pattern" in a single structured LLM call rather than two adversarial agent runs.

**Why LangGraph's `Send` API for fan-out:**  
Currently `_route_after_ml` returns `["rag_research", "static_analysis"]` — this already fans out correctly. The `Send` API upgrade allows arbitrary-count fan-out (e.g. one `Send` per flagged vulnerability class) when Phase 2 routing is implemented.

---

## 3. Final Graph Topology

```
START
  │
  ▼
ml_assessment              ← unchanged; add hotspot output in Phase 1
  │
  ▼
evidence_router            ← NEW (Phase 0); replaces _is_high_risk + _route_after_ml
  │
  ├─ [Reentrancy|Integer|Timestamp|External|Mishandled|Unused]
  │     → rag_research     ← enhanced: per-class scoped queries
  │     → static_analysis  ← already implemented; scope detectors per class (Phase 0)
  │
  ├─ [TOD|DoS]
  │     → rag_research
  │     → static_analysis
  │     → econ_assessment  ← Phase 3
  │
  └─ [all probs < 0.35 AND no static signal]
        → synthesizer      ← fast path; no LLM call
  │
  ▼ (fan-in — waits for all activated branches)
cross_validator            ← NEW (Phase 2); LLM reconciles all evidence
  │
  ▼
audit_check                ← unchanged; add similar-contract lookup (Phase 1)
  │
  ▼
synthesizer                ← extended: uses verdict from cross_validator
  │
  ▼
END


Phase 1 addition (between ml_assessment and evidence_router):

ml_assessment → graph_explain → evidence_router
                    ↑
            graph_inspector_server :8013
            (GNN attention → subgraph JSON + feature descriptions)
```

---

## 4. AuditState Schema (Final)

```python
class AuditState(TypedDict, total=False):
    # ── Input ────────────────────────────────────────────────────────────────
    contract_code:    str       # raw Solidity UTF-8
    contract_address: str       # on-chain address

    # ── ML evidence ──────────────────────────────────────────────────────────
    ml_result: dict             # existing schema: label, vulnerabilities, threshold,
                                # truncated, num_nodes, num_edges
    ml_hotspots: list[dict]     # Phase 1 — [{class, fn_name, lines, node_ids, score}]
                                # GNN attention + CodeBERT gradient × attention per class

    # ── Routing trace ────────────────────────────────────────────────────────
    routing_decisions: list[str]  # ["Reentrancy prob=0.87 → static+rag", ...]

    # ── Graph explanation (Phase 1) ──────────────────────────────────────────
    graph_explanations: dict    # {class: {subgraph_json, feature_descriptions, node_ids}}

    # ── Static analysis ──────────────────────────────────────────────────────
    static_findings: list[dict] # existing: [{tool, detector, impact, confidence,
                                #             description, lines}]

    # ── RAG evidence ─────────────────────────────────────────────────────────
    rag_results: list[dict]     # existing: ranked chunks with content, source, score, class

    # ── Economic simulation (Phase 3) ────────────────────────────────────────
    econ_scenarios: list[dict]  # [{name, inputs, outcome, exploitable, description}]

    # ── Cross-validation (Phase 2) ───────────────────────────────────────────
    confirmations:  dict        # {class: [tool, ...]}  — tools that agree with ML
    contradictions: dict        # {class: [tool, ...]}  — tools that disagree
    verdicts:       dict        # {class: "CONFIRMED"|"LIKELY"|"DISPUTED"|"SAFE"}

    # ── On-chain history ─────────────────────────────────────────────────────
    audit_history: list[dict]   # existing: prior AuditRegistry records

    # ── Final output ─────────────────────────────────────────────────────────
    final_report: dict          # extended: see §6
    narrative:    str | None    # LLM narrative (None when LLM offline)
    error:        str | None    # first non-fatal error from any node
```

---

## 5. Routing — Per-Class Matrix (replaces `_is_high_risk`)

```python
# agents/src/orchestration/routing.py

# Per-class probability thresholds for activating deep analysis.
# Lower than the inference threshold (0.50) — we want to investigate
# borderline cases, not just high-confidence ones.
DEEP_THRESHOLDS: dict[str, float] = {
    "Reentrancy":          0.35,
    "IntegerUO":           0.35,
    "GasException":        0.40,
    "Timestamp":           0.35,
    "TOD":                 0.35,
    "ExternalBug":         0.40,
    "CallToUnknown":       0.40,
    "MishandledException": 0.40,
    "UnusedReturn":        0.45,
    "DenialOfService":     0.30,   # rare class — lower threshold
}

# Which tools to activate per flagged class.
# static_analysis always runs when any class triggers (Slither is fast).
# rag_research runs for classes with documented exploit history.
# econ_assessment only for classes with economic exploitability (Phase 3).
ROUTING_RULES: dict[str, list[str]] = {
    "Reentrancy":          ["static_analysis", "rag_research"],
    "IntegerUO":           ["static_analysis", "rag_research"],
    "GasException":        ["static_analysis"],
    "Timestamp":           ["static_analysis", "rag_research"],
    "TOD":                 ["static_analysis", "rag_research", "econ_assessment"],
    "ExternalBug":         ["static_analysis", "rag_research"],
    "CallToUnknown":       ["static_analysis", "rag_research"],
    "MishandledException": ["static_analysis"],
    "UnusedReturn":        ["static_analysis"],
    "DenialOfService":     ["static_analysis", "rag_research", "econ_assessment"],
}

def compute_active_tools(ml_result: dict) -> list[str]:
    active: set[str] = set()
    for vuln in ml_result.get("vulnerabilities", []):
        cls  = vuln["vulnerability_class"]
        prob = vuln.get("probability", 0.0)
        if prob >= DEEP_THRESHOLDS.get(cls, 0.40):
            active.update(ROUTING_RULES.get(cls, []))
    return list(active)   # empty → fast path to synthesizer
```

**Routing decisions are always logged to `AuditState.routing_decisions`** so the final report can explain why each tool ran.

---

## 6. Verdict Model (what the report actually says)

The current report outputs `overall_label: "vulnerable"|"safe"` and a text `recommendation`. This is not auditable — "vulnerable" without evidence is just a restatement of the ML output.

**Final report adds per-class verdicts:**

```python
@dataclass
class VulnerabilityAssessment:
    vulnerability_class: str
    ml_probability:      float
    verdict:             Literal["CONFIRMED", "LIKELY", "DISPUTED", "SAFE"]
    # CONFIRMED  — ML + at least one other tool agree
    # LIKELY     — ML only, no static signal, but RAG finds precedent
    # DISPUTED   — ML flagged, but static analysis found nothing AND RAG has no match
    # SAFE       — below threshold on all tools

    evidence_sources:    list[str]   # ["ml:0.87", "slither:reentrancy-eth", "rag:TheDAO"]
    location:            str         # "function withdraw() lines 45-72" (from hotspots/Slither)
    severity:            Literal["CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO"]
    explanation:         str         # 1-2 sentences from synthesizer LLM or rule-based fallback

# Overall report
@dataclass
class AuditReport:
    contract_address:        str
    timestamp:               str
    overall_verdict:         Literal["CONFIRMED", "LIKELY", "DISPUTED", "SAFE"]
    vulnerability_findings:  list[VulnerabilityAssessment]
    overall_risk_score:      float      # max(ml_probability) across CONFIRMED+LIKELY
    tools_activated:         list[str]
    routing_trace:           list[str]
    audit_duration_ms:       int
    recommended_actions:     list[str]
    narrative:               str | None  # LLM summary
```

**Verdict logic in `cross_validator`:**

```
CONFIRMED  ← ML prob ≥ 0.50 AND (slither found same class OR rag_score ≥ 0.80)
LIKELY     ← ML prob ≥ 0.50 AND slither clean AND rag_score 0.50–0.79
DISPUTED   ← ML prob ≥ 0.50 AND slither clean AND rag_score < 0.50
SAFE       ← ML prob < threshold (fast path never reaches cross_validator)
```

---

## 7. MCP Server Map (final)

| Server | Port | Status | Wraps | Phase |
|--------|------|--------|-------|-------|
| `inference_server` | :8010 | EXISTS | ML model predict | now |
| `rag_server` | :8011 | EXISTS | FAISS+BM25+reranker | now |
| `audit_server` | :8012 | EXISTS | AuditRegistry read | now |
| `graph_inspector_server` | :8013 | NEW | graph_extractor + GNN attention | Phase 1 |
| `static_analysis_server` | :8014 | OPTIONAL | Slither (already in-process) | Phase 3 |
| `econ_sim_server` | :8015 | NEW | canned scenario engine | Phase 3 |

`static_analysis` runs Slither in-process (already implemented). A dedicated MCP server `:8014` only makes sense if Mythril is added (Phase 3) — Mythril should be isolated because it can crash or hang.

---

## 8. Memory Architecture

Four types — different storage, different staleness, different retrieval:

| Type | What | Where | Update frequency |
|------|------|-------|-----------------|
| **In-context** | this audit run's full state | `AuditState` / LangGraph `SqliteSaver` | every node |
| **Episodic** | audit history per contract | `AuditRegistry` on-chain + `data/reports/*.json` | per audit |
| **Semantic** | vulnerability knowledge, exploits | FAISS+BM25 RAG index | weekly / on new exploit |
| **Procedural** | how to audit, what each class means | agent system prompts + `ROUTING_RULES` | deliberate change only |

**Checkpointer upgrade:** `MemorySaver` → `SqliteSaver`

```python
# agents/src/orchestration/graph.py
from langgraph.checkpoint.sqlite import SqliteSaver

DB_PATH = Path(__file__).parents[2] / "data" / "checkpoints.db"

def build_graph(use_checkpointer: bool = True):
    checkpointer = SqliteSaver.from_conn_string(str(DB_PATH)) if use_checkpointer else None
    ...
```

This gives: full audit trail, resume-from-node on crash, inspectable mid-execution state. Zero schema change to `AuditState`.

---

## 9. `cross_validator` Node (Phase 2) — the key differentiator

This is the node that makes SENTINEL different from "run Slither" or "call an LLM".

```python
async def cross_validator(state: AuditState) -> dict:
    """
    Reconcile ML, static, and RAG evidence per flagged class.
    Uses STRONG LLM with a structured prompt. Returns verdicts dict.
    """
    ml_vulns    = state.get("ml_result", {}).get("vulnerabilities", [])
    static_hits = _index_static_by_class(state.get("static_findings", []))
    rag_hits    = _index_rag_by_class(state.get("rag_results", []))

    verdicts       = {}
    confirmations  = {}
    contradictions = {}

    for vuln in ml_vulns:
        cls  = vuln["vulnerability_class"]
        prob = vuln["probability"]

        slither_match = cls.lower() in static_hits  # detector name overlap
        rag_score     = max((r["score"] for r in rag_hits.get(cls, [])), default=0.0)

        # Rule-based verdict (no LLM cost for clear cases)
        if prob >= 0.50 and slither_match:
            verdict = "CONFIRMED"
            confirmations[cls]  = ["ml", "slither"]
        elif prob >= 0.50 and rag_score >= 0.80:
            verdict = "CONFIRMED"
            confirmations[cls]  = ["ml", "rag"]
        elif prob >= 0.50 and rag_score >= 0.50:
            verdict = "LIKELY"
            confirmations[cls]  = ["ml", "rag_partial"]
        elif prob >= 0.50:
            verdict = "DISPUTED"
            contradictions[cls] = ["ml_flagged", "slither_clean", "rag_nomatch"]
        else:
            verdict = "SAFE"

        verdicts[cls] = verdict

    # LLM call only for DISPUTED cases (saves cost on clear outcomes)
    disputed = [c for c, v in verdicts.items() if v == "DISPUTED"]
    if disputed:
        # single LLM call with all disputed classes for efficiency
        verdicts = await _llm_adjudicate_disputed(state, verdicts, disputed)

    return {
        "verdicts":      verdicts,
        "confirmations": confirmations,
        "contradictions": contradictions,
    }
```

The LLM call is gated to DISPUTED cases only — most contracts resolve via the rule-based path, keeping cost minimal.

---

## 10. Escalation Tiers

| Tier | Condition | Path | Cost |
|------|-----------|------|------|
| **Tier 0 — Fast pass** | all probs < DEEP_THRESHOLDS | fast path → synthesizer (no LLM) | 0 LLM calls |
| **Tier 1 — Standard** | any prob ≥ threshold | static + rag + rule-based cross_validator | 0–1 LLM calls |
| **Tier 2 — Deep** | DISPUTED verdict exists | LLM adjudication in cross_validator | 1 LLM call |
| **Tier 3 — Narrative** | CONFIRMED or LIKELY | synthesizer LLM narrative | 1 LLM call |

Maximum LLM calls per audit: 2 (adjudication + narrative). Most contracts: 0.

---

## 11. Phased Build Plan

### Phase 0 — Foundation (implement now, unblocked)

All items are unblocked today. Estimated total: 1–2 days.

- [ ] **P0-1** Extract routing logic to `agents/src/orchestration/routing.py`  
  - Move `_is_high_risk()` out of `nodes.py`  
  - Implement `DEEP_THRESHOLDS` dict + `ROUTING_RULES` dict + `compute_active_tools()`  
  - Update `_route_after_ml()` in `graph.py` to call `compute_active_tools()`  
  - Log every routing decision to `routing_decisions` in state  

- [ ] **P0-2** Scope Slither detectors per flagged class  
  - `static_analysis` node currently runs ALL detectors — expensive and noisy  
  - Build `CLASS_TO_DETECTORS: dict[str, list[str]]` map  
    - `Reentrancy → ["reentrancy-eth", "reentrancy-no-eth", "reentrancy-events-and-order"]`  
    - `IntegerUO → ["integer-overflow", "toctou", "unchecked-lowlevel"]`  
    - `Timestamp → ["timestamp"]`  
    - `DenialOfService → ["calls-loop", "costly-loop"]`  
    - `ExternalBug → ["low-level-calls", "unchecked-send", "arbitrary-send-eth"]`  
    - (etc. for remaining classes)  
  - Pass `detectors=active_detectors` to `Slither.run_detectors()` — reduces runtime 5–10×  
  - Add `function_name` extraction to each finding (from `elements[].name`)  

- [ ] **P0-3** Swap `MemorySaver` → `SqliteSaver`  
  - `pip install langgraph-checkpoint-sqlite` in agents venv  
  - One-line change in `graph.py`; checkpoint DB at `agents/data/checkpoints.db`  

- [ ] **P0-4** Add `routing_decisions` field to `AuditState`  
  - `routing_decisions: list[str]` with `Annotated[list, operator.add]` reducer  
  - Each node that makes a routing decision appends one line  

- [ ] **P0-5** Extend `AuditState` with `verdicts`, `confirmations`, `contradictions` fields  
  - Just schema additions — no logic yet (cross_validator is Phase 2)  

- [ ] **P0-6** Add `verdict` and `evidence_sources` to final report  
  - In `synthesizer` node: compute a simple rule-based verdict (no cross_validator yet)  
  - `CONFIRMED` if static_findings non-empty AND ML flagged same class  
  - `LIKELY` if ML flagged but static empty  
  - `SAFE` if fast path  
  - This gives the report structure before cross_validator is built  

- [ ] **P0-7** Write smoke test for per-class routing  
  - Test that Reentrancy above threshold routes to `["rag_research", "static_analysis"]`  
  - Test that all-safe contract routes to `[]` (fast path)  
  - Test that DoS routes to econ_assessment (when econ node exists — mark as pending)  

---

### Phase 1 — Evidence Richness (after v8 checkpoint exists)

Requires: stable v8 checkpoint at `ml/checkpoints/v8.0_best.pt`

- [ ] **P1-1** Extend `inference_server :8010` with `/hotspots` endpoint  
  - GNN: record per-edge attention weights during forward pass for top-3 flagged classes  
  - CodeBERT: token-level gradient × attention score (integrated gradients, simplified)  
  - Return: `[{class, fn_name, lines, node_ids, score}]`  
  - Store in `AuditState.ml_hotspots`  

- [ ] **P1-2** Build `graph_inspector_server :8013`  
  - New MCP server (`agents/src/mcp/servers/graph_inspector_server.py`)  
  - Tools: `/subgraph` (return subgraph JSON around node_ids), `/features` (node feature values with names from schema), `/explain` (natural-language description of each hotspot node)  
  - Input: contract_address + node_ids from `ml_hotspots`  
  - Output: `{class: {subgraph_json, feature_descriptions}}`  

- [ ] **P1-3** Add `graph_explain` node to LangGraph graph  
  - Runs after `ml_assessment`, before `evidence_router`  
  - Calls `graph_inspector_server` with hotspot node_ids  
  - Stores in `AuditState.graph_explanations`  
  - Falls back gracefully if server unavailable (non-fatal)  

- [ ] **P1-4** Enhance RAG queries with hotspot context  
  - Current: generic `"{class} vulnerability in Solidity"`  
  - New: `"{class}: external call before state update in payable function withdraw()"` (using hotspot fn_name + feature descriptions)  
  - Measurably improves RAG relevance for specific patterns  

- [ ] **P1-5** Extend `audit_check` with similar-contract lookup  
  - Beyond exact address match: query RAG index by GNN embedding similarity  
  - "this vulnerability pattern was previously flagged in a related contract" signal  

---

### Phase 2 — Routing Intelligence & Cross-Validation

Requires: Phase 0 complete, Phase 1 hotspots available

- [ ] **P2-1** Implement `cross_validator` node (see §9 above)  
  - Rule-based verdict for CONFIRMED/LIKELY/SAFE cases (no LLM cost)  
  - LLM adjudication for DISPUTED cases only  
  - Wire into graph between `audit_check` and `synthesizer`  

- [ ] **P2-2** Upgrade routing to LangGraph `Send` API  
  - Enables per-vulnerability-class fan-out (one branch per class, not one branch per tool)  
  - Allows class-specific RAG queries to run in parallel  

- [ ] **P2-3** Extend `synthesizer` prompt context  
  - Include: graph feature descriptions, static findings with line numbers, cross_validator verdict, routing trace  
  - Synthesizer narrative cites specific evidence: "Slither confirmed reentrancy-eth at line 58, matching the DAO exploit pattern from RAG"  

- [ ] **P2-4** Add `client_pool.py` for MCP connections  
  - Replace per-call SSE connection with persistent pooled clients  
  - Reduces latency on multi-tool deep-path audits  

- [ ] **P2-5** Per-class confidence signal in final report  
  - `evidence_breadth`: count of independent tools agreeing  
  - `eye_agreement`: GNN prob vs CodeBERT prob vs fused prob (from hotspots)  
  - `historical_precedent`: bool (RAG match above 0.80)  
  - `coverage_confidence`: was contract within training distribution? (windows=1 → high; windows=4 → moderate)  

---

### Phase 3 — Advanced Tools

Requires: Phase 2 complete

- [ ] **P3-1** Implement `econ_sim_server :8015`  
  - Canned scenarios for TOD and DoS classes (not full simulation)  
  - TOD: "can sequenced tx pair change conditional outcome?" (pattern matching, not execution)  
  - DoS: "is there a loop processing external calls proportional to unbounded array?"  
  - Wire `econ_assessment` node into graph for TOD/DoS classes  

- [ ] **P3-2** Add Mythril to `static_analysis` (scoped, isolated)  
  - Only invoke on functions listed in `ml_hotspots` — never full-contract  
  - Hard timeout: 60 seconds per function  
  - Run in subprocess (Mythril can OOM crash — isolate from main process)  
  - Property to check per class: Reentrancy → "ETH balance cannot decrease without state update"  

- [ ] **P3-3** Cross-contract dependency graph  
  - Orchestrator maintains a directed graph: nodes = contracts, edges = CALL relationships  
  - Build from Slither's `contract.called_contracts` across a batch  
  - Analyze dependencies before dependents (topological order)  
  - "Contract A passes user-controlled data to Contract B's privileged function — combined vulnerability"  

---

### Phase 4 — Feedback Loop Closure

- [ ] **P4-1** Model drift corrections  
  - When on-chain record has human-confirmed false positive → write to `ml/data/feedback/corrections.csv`  
  - Becomes active-learning training data for next model version  

- [ ] **P4-2** Class-aware RAG ingestion  
  - GitHub fetcher maps each new exploit writeup to its vulnerability class  
  - RAG queries for that class return richer context over time  

- [ ] **P4-3** Disagreement logging  
  - Log every case where ML flags a class but Slither finds nothing (and vice versa)  
  - Builds `ml/data/feedback/disagreements.csv` — identifies ML training gaps  

---

## 12. Success Metrics

The agents system is measurable. Build a test set of contracts with known ground truth and track:

| Metric | Target | How to measure |
|--------|--------|----------------|
| CONFIRMED precision | > 0.85 | `CONFIRMED` verdicts on known-vulnerable contracts |
| SAFE recall | > 0.95 | `SAFE` on known-clean contracts (false negative rate) |
| DISPUTED rate | < 0.15 | fraction of flagged contracts where tools disagree |
| Audit latency (Tier 1) | < 30s | end-to-end with Slither + RAG, no LLM |
| Audit latency (Tier 2) | < 60s | with LLM adjudication |
| Fast-path rate | > 0.60 | fraction of contracts taking fast path (cost control) |

---

## 13. What NOT to Build Yet

- **Triage Agent as separate process** (Proposal 1): LangGraph node boundary is sufficient
- **Debate Agent pair** (Proposal 1): `cross_validator` with LLM adjudication covers the same case at lower cost
- **Foundry fuzz tests in the agent loop**: too slow for real-time audit; defer to offline pipeline
- **M6 API routes**: blocked on stable v8 checkpoint and Phase 2 completion
- **Full Mythril symbolic execution**: never full-contract, always scoped to hotspot functions, always with hard timeout

---

## Appendix: Current State vs Target State

| Component | Current | Phase 0 | Phase 2 |
|-----------|---------|---------|---------|
| Routing | single global threshold 0.70 | per-class thresholds + tool matrix | LangGraph Send per class |
| Slither | runs ALL detectors | scoped to flagged classes | + function-level findings |
| RAG queries | generic class query | + hotspot context (Phase 1) | per-class parallel |
| Cross-validation | none | rule-based verdict | LLM for DISPUTED cases |
| Checkpointer | MemorySaver | SqliteSaver | SqliteSaver |
| Report | label + recommendation text | + verdict + evidence_sources | + confidence dimensions |
| Hotspots | none | none | Phase 1 | Phase 2 synthesizer uses them |
| Economic sim | none | none | Phase 3 |
