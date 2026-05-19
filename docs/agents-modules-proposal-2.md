Good. Now let me write the full design document.

***
# Sentinel Agents Module — Full Architecture Design
The agents module is the **reasoning and orchestration brain** of Sentinel: it receives a raw Solidity contract, uses the ML module as a first-pass evidence generator, then orchestrates deeper tools (static analyzers, graph inspector, RAG retrieval, economic simulators) through MCP, reconciles all the evidence, and produces a structured audit report that eventually lands on-chain. Everything described below is designed to directly cover ML's structural limitations while also giving you deliberate exposure to the most in-demand AI engineering patterns in 2025–2026.

Here is the overall pipeline flow:
---
## Core Design Philosophy
Before touching any code, the philosophy to lock in is this: **the ML model is a fast triage oracle, not the judge**. It produces `(class, probability, hotspot_locations)` tuples that tell the agents *where to look*, not *what the verdict is*. Every downstream tool chain exists to either confirm, refute, or add nuance to what ML flagged — and the synthesizer's job is to reconcile all that evidence into a coherent report.

This is the same pattern used in real security tooling (e.g. GitHub Advanced Security, Semgrep App, Aikido Security): cheap signal first, expensive analysis only when warranted. The agent graph's routing logic is where this cost/signal tradeoff lives.

***
## The `AuditState` — the nervous system
Every LangGraph node reads from and writes to a shared `AuditState` typed dict. Getting this right is the most important design decision because everything else is just node logic that transforms it. The current state schema is too thin; here is the full proposed design:

```python
class AuditState(TypedDict):
    # ── Input ──────────────────────────────────────
    contract_source: str          # raw Solidity UTF-8
    contract_id: str              # stable hash of source
    
    # ── Pre-processing outputs ─────────────────────
    graph_data: dict              # serialized PyG Data (v7 schema)
    token_data: dict              # {input_ids, attention_mask, windows}
    
    # ── ML evidence ────────────────────────────────
    ml_probs: dict[str, float]    # {class: probability, ...}
    ml_hotspots: list[Hotspot]    # [{class, fn_name, lines, node_ids}]
    ml_latency_ms: float
    
    # ── Graph explanation evidence ──────────────────
    graph_subgraphs: dict[str, SubgraphJSON]   # per flagged class
    graph_node_features: list[NodeFeatureRow]  # for synthesizer
    
    # ── Static analysis evidence ────────────────────
    slither_findings: list[StaticFinding]      # {rule, severity, fn, lines, msg}
    mythril_findings: list[SymbolicFinding]    # {property, result, cex_trace}
    foundry_fuzz_findings: list[FuzzFinding]   # {fn, seed, trace}
    
    # ── RAG evidence ────────────────────────────────
    rag_snippets: list[RAGSnippet]             # {text, source, score, class}
    
    # ── Economic simulation evidence ────────────────
    econ_scenarios: list[EconScenario]         # {name, params, outcome}
    
    # ── Cross-validation ────────────────────────────
    confirmations: dict[str, list[str]]        # {class: [tool1, tool2]}
    contradictions: dict[str, list[str]]       # {class: [tool1_says_yes, tool2_says_no]}
    
    # ── On-chain history ────────────────────────────
    prior_audits: list[AuditRecord]
    
    # ── Final output ────────────────────────────────
    report: AuditReport
    routing_decisions: list[str]               # trace of what was run and why
```

The `routing_decisions` field is purely for debugging and transparency — every routing choice appends a human-readable reason string (`"ml_prob[Reentrancy]=0.83 → static_analysis activated"`), so you can trace exactly why a particular tool was used for a given contract.

***
## LangGraph Node-by-Node Design
### Node 1: `pre_processing`
**New node** that runs before ML. Currently, graph extraction and tokenization happen inside the ML server; that should move here so the results can be stored in `AuditState` and shared across all tools without re-extracting.

- Calls `graph_extractor.py` and `windowed_tokenizer.py` (via MCP on `graph_inspector_server :8013`, or directly).
- Stores `graph_data` and `token_data` in state.
- Validates schema version (`FEATURE_SCHEMA_VERSION == "v7"`), rejects or re-extracts otherwise.
- **Why**: static analysis, graph explanation, and the ML model all need the same graph; computing it once and sharing it avoids desync bugs.
### Node 2: `ml_assessment`
Current node, but enhanced. Calls `inference_server :8010`.

- Existing: gets `ml_probs` and threshold-tuned labels.
- **New**: requests attention/gradient-based hotspots for top-3 flagged classes.
  - GNN: attention weights per edge type → identifies which functions/nodes drove each class prediction.
  - Transformer: token-level gradient × attention scores → identifies which code spans were most influential.
  - Returns `ml_hotspots: [{class, fn_name, lines, node_ids}]`.
- Stores everything in `AuditState`.

The hotspot extraction is the key new capability — it converts the ML model from a "black box probability emitter" into an "evidence pointer" for all downstream nodes.
### Node 3: `graph_explain` (NEW)
Calls `graph_inspector_server :8013` (new MCP server).

- Given the node IDs from `ml_hotspots`, returns:
  - A subgraph JSON (nodes + edges + features) around each hotspot.
  - Natural-language description of each hotspot node: `"function withdraw(): payable=True, has_loop=True, ext_calls=2, visibility=public → GNN weight=0.94 for Reentrancy"`.
  - Per-node feature values with their names (from schema), so the synthesizer can quote them.
- **Why**: the synthesizer cannot quote the GNN's reasoning without access to actual feature values and graph structure — currently those are lost after embedding pooling. This node preserves them explicitly.
### Node 4: `evidence_router` (enhanced)
The existing `_route_after_ml` routing logic uses a single `max(per-class probability) > 0.70` threshold. Replace this with a **per-class, per-tool routing decision matrix**:

```python
ROUTING_RULES: dict[str, list[str]] = {
    "Reentrancy":         ["static_analysis", "rag_research"],
    "IntegerUO":          ["static_analysis", "rag_research"],
    "GasException":       ["static_analysis"],
    "Timestamp":          ["static_analysis", "rag_research"],
    "TOD":                ["static_analysis", "econ_assessment"],
    "ExternalBug":        ["static_analysis", "rag_research"],
    "CallToUnknown":      ["static_analysis", "graph_explain"],
    "MishandledException":["static_analysis"],
    "UnusedReturn":       ["static_analysis"],
    "DenialOfService":    ["static_analysis", "econ_assessment", "rag_research"],
}

def route(state: AuditState) -> list[str]:
    active_tools = set()
    for cls, prob in state["ml_probs"].items():
        if prob > THRESHOLDS[cls]:           # per-class tuned thresholds
            for tool in ROUTING_RULES[cls]:
                active_tools.add(tool)
    return list(active_tools)
```

This is better than a single global threshold for several reasons:

- Some classes (e.g. `UnusedReturn`) don't benefit from economic simulation.
- `econ_assessment` is expensive; only launch it for classes that have economic exploitability.
- Per-class thresholds (already saved in your `thresholds.json` from `tune_threshold.py`) make routing more precise.

The router's routing decisions are appended to `AuditState.routing_decisions` for full traceability.
### Node 5: `rag_research` (enhanced)
The existing node calls `rag_server :8011`. Add per-class scoped queries:

- For each flagged class, issue a targeted query: `"Reentrancy vulnerability: {contract_summary} similar patterns"`.
- Use `ml_hotspots` to enrich the query: `"external call before state update in payable function"`.
- Cross-encoder reranking (already implemented with `ms-marco-MiniLM-L-6-v2`) stays.
- Return per-class `rag_snippets` with source attribution.

**Educational value**: you learn **query formulation for RAG** — how to turn structured data (graph features, class probabilities) into natural-language search queries.
### Node 6: `static_analysis` (implement for real)
This is currently a stub referenced in the codebase but not wired. Calls `static_analysis_server :8014` (new MCP server wrapping Slither + Mythril).

**Slither sub-tool**:
- Run detectors scoped to flagged classes:
  - Reentrancy: `reentrancy-eth`, `reentrancy-no-eth`, `reentrancy-events-and-order`.
  - IntegerUO: `toctou`, `integer-overflow` (in older solc), `unchecked-lowlevel`.
  - Timestamp: `timestamp`.
  - External calls: `calls-loop`, `low-level-calls`, `unchecked-send`.
- Return structured `StaticFinding`: `{rule_id, severity, function, line_start, line_end, message}`.

**Mythril sub-tool** (scoped, not full symbolic execution):
- Only invoke for functions in `ml_hotspots` (not full contract — too slow).
- Property to check per class: Reentrancy → "ETH balance cannot decrease without state update".
- Return `SymbolicFinding`: `{property, result: sat|unsat|timeout, counterexample_trace}`.

**Design decision — timeouts**: Mythril can run for hours on complex contracts. Hard-cap at 60 seconds per function, report `"timeout"` rather than blocking the agent graph.
### Node 7: `econ_assessment` (NEW)
Only activated for `TOD`, `DenialOfService`, and `ExternalBug` classes.

Calls `econ_sim_server :8015` (new MCP server, Python-based):

- Phase 1 (minimal viable): run a small set of **canned scenarios** derived from contract features:
  - "Is there a loop that processes external calls proportional to an unbounded array?" → DoS.
  - "Can a sequenced pair of transactions change the outcome of a conditional?" → TOD.
  - "Does a payable function forward ETH to an external address without a cap?" → fund drain scenario.
- Phase 2 (later): actual AMM/loan protocol simulation if the contract pattern matches.
- Returns `EconScenario`: `{name, inputs, outcome, exploitable: bool, description}`.

**Educational value**: this is where you encounter **formal specification and property-based testing** patterns, which are very transferable to Foundry fuzz tests and formal verification.
### Node 8: `cross_validator` (NEW)
This is the most intellectually interesting new node. It takes all evidence — ML probabilities, static findings, graph features, RAG snippets, economic scenarios — and produces:

```python
for each flagged class:
    confirmations[class] = [tool for tool in activated_tools if tool_found_same_class]
    contradictions[class] = [tool for tool if ml_flagged and tool_says_safe]
```

The node uses the STRONG LLM (your existing `llm/client.py` STRONG model, e.g. qwen3.5-9b-ud) to reason over the evidence:

```
You are cross-validating vulnerability evidence.
ML says: Reentrancy prob=0.87, hotspot=function withdraw() lines 45-72
Slither says: reentrancy-eth FOUND at function withdraw() line 58
Graph features: payable=True, has_loop=True, ext_calls=2, state_update_after_call=True
Similar exploit: "The DAO hack: external call to untrusted contract before balance update"

Rate: CONFIRMED | DISPUTED | NEEDS_MANUAL_REVIEW
Explain your reasoning in one sentence.
```

Then produces a `contradiction_summary` for any class where tools disagree.

**Why this matters**: you learn **structured LLM reasoning over heterogeneous evidence** — one of the core patterns in modern AI agent systems (cited by LangChain, AutoGen, and production audit tools).
### Node 9: `audit_check` (existing, enhanced)
Current node checks on-chain `AuditRegistry` via `audit_server :8012`. Add:

- Query for prior audits of **similar contracts** (by code hash similarity in RAG index, not just exact address).
- If a prior audit exists with the same vulnerability class, flag it: "this vulnerability pattern was previously flagged in a related contract, suggesting a shared library vulnerability."
- Store in `AuditState.prior_audits`.
### Node 10: `synthesizer` (enhanced)
Current node uses qwen3.5-9b-ud with a rule-based fallback. The synthesizer's full prompt context should now include:

1. ML probabilities + per-class thresholds (already in state).
2. Graph feature descriptions from `graph_explain` (new).
3. Static findings with function names and line numbers (new).
4. Cross-validator's confirmation/contradiction summary (new).
5. RAG snippets as evidence references.
6. Economic scenario results (if activated).
7. Routing decisions log (for transparency).

Output format — `AuditReport`:

```python
@dataclass
class AuditReport:
    contract_id: str
    timestamp: str
    vulnerability_classes: list[VulnerabilityAssessment]
    
    @dataclass  
    class VulnerabilityAssessment:
        class_name: str
        ml_probability: float
        verdict: Literal["CONFIRMED", "LIKELY", "DISPUTED", "SAFE"]
        evidence_sources: list[str]          # ["ml", "slither", "rag"]
        location: str                         # "function withdraw() lines 45-72"
        explanation: str                      # 1-2 sentences
        severity: Literal["CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO"]
    
    overall_risk_score: float
    routing_trace: list[str]                  # for transparency
    tools_activated: list[str]
    audit_duration_ms: int
    recommended_actions: list[str]
```

The per-vulnerability `verdict` is important: it acknowledges that ML and static analysis may disagree, and the synthesizer adjudicates rather than blindly trusting ML.

***
## MCP Layer Architecture
The MCP layer is your **"service mesh" for tools**. Here is the full port map and responsibility design:

| Server | Port | Wraps | Key Tools |
|---|---|---|---|
| `inference_server` | :8010 | ML model + hotspot extractor | `/predict`, `/hotspots` |
| `rag_server` | :8011 | FAISS+BM25+RRF+reranker | `/search`, `/search_by_class` |
| `audit_server` | :8012 | AuditRegistry (read) | `/history`, `/similar` |
| `graph_inspector_server` | :8013 NEW | graph_extractor + GNN attention | `/subgraph`, `/features`, `/explain` |
| `static_analysis_server` | :8014 NEW | Slither + Mythril | `/slither_scan`, `/mythril_check` |
| `econ_sim_server` | :8015 NEW | scenario engine | `/run_scenarios` |

**Client pool design** (per the existing M4 plan `client_pool.py`):

```python
# agents/src/mcp/client_pool.py
_pool: dict[str, MCPClient] = {}

async def get_client(url: str) -> MCPClient:
    if url not in _pool:
        _pool[url] = await MCPClient.connect(url)
    return _pool[url]
```

This eliminates per-call SSE connection overhead and is a prerequisite before M6 API wires the orchestrator to handle concurrent requests.

**Why the MCP architecture is the right choice** for this project:

Every tool capability is independently testable, independently deployable, and composable with any client. You can swap `static_analysis_server` from Slither to Semgrep without changing any agent code. You can add a new tool server without touching the orchestrator. And the `smoke_*.py` scripts you already have become the integration test suite for each server, which is exactly how production MCP tool servers are validated.

***
## LangGraph State Machine Design
The full graph structure with conditional edges:

```
pre_processing
    → ml_assessment
        → graph_explain          (always, in parallel with routing)
        → evidence_router
            → rag_research       (conditional on routing_rules)
            → static_analysis    (conditional on routing_rules)
            → econ_assessment    (conditional on routing_rules)
                → cross_validator
                    → audit_check
                        → synthesizer
                            → END
```

**Parallel fan-out pattern** — the `evidence_router` should use LangGraph's `Send` API to dispatch multiple branches in parallel (asyncio under the hood):

```python
from langgraph.types import Send

def evidence_router(state: AuditState) -> list[Send]:
    active = compute_active_tools(state)
    return [Send(tool_name, state) for tool_name in active]
```

This means Slither, Mythril, and RAG run concurrently, not sequentially — which is a major latency win.

**Checkpointer upgrade** — switch from `MemorySaver` to `SqliteSaver` (or `PostgresSaver` once M6 lands):

```python
from langgraph.checkpoint.sqlite import SqliteSaver
checkpointer = SqliteSaver.from_conn_string("agents/data/checkpoints.db")
graph = build_graph(checkpointer=checkpointer)
```

This gives you full audit trail, resume from any node on failure, and the ability to inspect mid-execution state — all critical for an audit tool where you want to explain *why* an agent made a decision.

***
## Feedback Loop — the learning circuit
The `feedback_loop.py` already polls on-chain `AuditSubmitted` events and re-indexes the RAG corpus. Extend it to close three more loops:

1. **Model drift feedback**: when the on-chain record for a contract has a human-provided correction (e.g. "this was a false positive"), write that to `ml/data/feedback/corrections.csv`. This becomes training data for the next model version.

2. **RAG corpus enrichment**: when a new exploit is published (via the GitHub fetcher in `ingestion/`), ingest the exploit writeup and map it to the vulnerability class it belongs to. Future RAG queries for that class will return richer context.

3. **Static analysis rule calibration**: if Slither consistently flags something that ML doesn't (or vice versa), log those disagreements. Over time this builds a "disagreement dataset" you can use to identify where ML needs more training data.

***
## Skills and Concepts You'll Build Through This
| Component | Skill You Learn | Industry Relevance |
|---|---|---|
| LangGraph `Send` API + parallel nodes | Async multi-tool agent orchestration | Core LangChain/LangGraph pattern used in production |
| MCP server design (6 servers) | Tool-augmented agents, composable API design | MCP is the emerging standard (Anthropic, OpenAI tooling) |
| `cross_validator` node | Structured LLM reasoning over heterogeneous evidence | Advanced prompt engineering + evidence fusion |
| Per-class routing rules | Decision logic in agentic systems | Directly transferable to any AI workflow |
| `SqliteSaver` checkpointing | Durable, inspectable agent state | Production requirement for any serious agent system |
| `graph_inspector_server` | Exposing GNN internals as APIs | XAI (Explainability) — very valued in ML Eng roles |
| `static_analysis_server` | Integrating traditional program analysis with ML/LLM | Unique combination, rare skill set |
| Feedback loop design | Online learning, data flywheel | MLOps + product thinking |
| `AuditReport` schema design | Structured output from LLMs | Critical skill — unstructured LLM output is hard to use downstream |

***
## Phased Build Plan
Rather than building everything at once, a clean phase order that respects dependencies:

**Phase 0 — Foundation (implement now, blocks everything)**
- Wire `static_analysis` node properly (Slither v1, no Mythril yet).
- Introduce `client_pool.py`.
- Upgrade checkpointer to `SqliteSaver`.
- Parameterise `build_graph(checkpointer=...)` so M6 can inject.

**Phase 1 — Evidence richness (after v7 training)**
- Implement `graph_inspector_server :8013` with subgraph + feature export.
- Add `graph_explain` node to the graph.
- Extend `inference_server` to return hotspots (attention + gradient saliency).
- Extend `AuditState` with full schema above.

**Phase 2 — Routing intelligence**
- Replace `_is_high_risk` with per-class routing matrix.
- Add `cross_validator` node.
- Add parallel `Send` fan-out routing.
- Extend `synthesizer` prompt context.

**Phase 3 — Advanced tools (learning-heavy, after Phase 2 is solid)**
- Implement `static_analysis_server` with Mythril (scoped to hotspot functions, 60s cap).
- Implement `econ_sim_server` with canned scenarios.
- Add economic assessment node.

**Phase 4 — Feedback loop closure**
- Extend feedback loop with corrections CSV.
- RAG class-aware ingestion.
- Disagreement logging between ML and static tools.

***
## The Most Important Design Decision
The single most consequential architectural decision is the **`cross_validator` node** and the `verdict` field in `AuditReport`. It is the thing that makes Sentinel different from "run Slither" or "ask an LLM." Any tool alone can produce false positives and false negatives. The value of Sentinel's agents layer is that it reconciles multiple independent evidence sources and tells the auditor: "ML, Slither, and a prior exploit match all agree → `CONFIRMED CRITICAL`" vs "ML flagged it but Slither found nothing and graph features don't support it → `DISPUTED, manual review recommended`."

That `verdict` field is also your primary evaluation metric for the agents system: you can build a test set of known-vulnerable and known-safe contracts and measure how often `CONFIRMED` maps to actually vulnerable, and how often `SAFE` maps to actually safe — exactly the kind of agent evaluation benchmark that doesn't exist widely yet but is one of the hottest topics in AI engineering right now.