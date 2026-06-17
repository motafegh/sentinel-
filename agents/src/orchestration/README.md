# Orchestration

LangGraph StateGraph that coordinates the SENTINEL audit workflow — from ML inference through multi-source evidence collection to a final vulnerability report.

## Architecture

```
START
  │
  ▼
ml_assessment          POST /predict via inference MCP :8010
  │
  ▼
quick_screen           Slither + Aderyn Tier-0 screen (every contract)
  │
  ▼
evidence_router        Per-class routing decisions, logs to state
  │
  ├─ [deep path] ───── rag_research ────────┐
  │                   static_analysis ───────┤
  │                   graph_explain ─────────┘
  │                                            │
  │                                      audit_check
  │                                            │
  │                                     cross_validator
  │                                            │
  ├─ [fast path] ─────────────────────── synthesizer
  │                                            │
  ▼                                            ▼
  END ◄───────────────────────────────────── END
```

### Fast vs Deep Path

| Signal | Fast Path | Deep Path |
|--------|-----------|-----------|
| ML all classes below `DEEP_THRESHOLDS` | Yes | No |
| `quick_screen` zero High/Critical hits | Yes | No |
| Both signals agree it is safe | **Yes** | — |
| Either signal flags risk | — | **Yes** |

Two-signal gate: fast path requires ML **and** static analysis to agree. If Slither/Aderyn fire High/Critical findings while ML is below thresholds, the contract is still escalated to deep path.

## Files

| File | Lines | Purpose |
|------|-------|---------|
| `state.py` | 160 | `AuditState` TypedDict — 16 fields flowing through the graph |
| `routing.py` | 263 | Per-class thresholds, tool routing, verdict computation |
| `nodes.py` | 1415 | 9 node implementations (the core logic) |
| `graph.py` | 241 | StateGraph builder, conditional edges, SqliteSaver checkpointing |

## `state.py` — AuditState

`TypedDict` with `total=False` — every field is optional. Nodes return only the keys they updated; LangGraph merges partial dicts automatically.

### Fields

| Field | Type | Set By | Reducer |
|-------|------|--------|---------|
| `contract_code` | `str` | Caller | — (immutable) |
| `contract_address` | `str` | Caller | — (immutable) |
| `ml_result` | `dict` | `ml_assessment` | replace |
| `ml_hotspots` | `list[dict]` | `graph_explain` | replace |
| `routing_decisions` | `list[str]` | `evidence_router` + any node | `operator.add` (append) |
| `graph_explanations` | `dict` | `graph_explain` | replace |
| `quick_screen_hits` | `dict[str, list[str]]` | `quick_screen` | replace |
| `static_findings` | `list[dict]` | `static_analysis` | replace |
| `external_call_summary` | `list[dict]` | `static_analysis` | replace |
| `rag_results` | `list[dict]` | `rag_research` | replace |
| `econ_scenarios` | `list[dict]` | (Phase 3) | replace |
| `verdicts` | `dict[str, str]` | `cross_validator` / `synthesizer` | replace |
| `confirmations` | `dict[str, list[str]]` | `cross_validator` / `synthesizer` | replace |
| `contradictions` | `dict[str, list[str]]` | `cross_validator` | replace |
| `audit_history` | `list[dict]` | `audit_check` | replace |
| `final_report` | `dict` | `synthesizer` | replace |
| `narrative` | `str \| None` | `synthesizer` | replace |
| `error` | `str \| None` | Any node | replace |

## `routing.py` — Per-Class Routing

### DEEP_THRESHOLDS

Per-class probability thresholds that trigger deep analysis. Deliberately lower than the inference threshold (0.50) — borderline cases are investigated, not skipped.

```python
DEEP_THRESHOLDS = {
    "Reentrancy":          0.35,
    "IntegerUO":           0.35,
    "GasException":        0.40,
    "Timestamp":           0.35,
    "TransactionOrderDependence": 0.35,
    "ExternalBug":         0.40,
    "CallToUnknown":       0.40,
    "MishandledException": 0.40,
    "UnusedReturn":        0.45,
    "DenialOfService":     0.30,
}
```

### ROUTING_RULES

Which tool nodes activate per flagged class:

| Class | Tools |
|-------|-------|
| Reentrancy, IntegerUO, Timestamp, TransactionOrderDependence, ExternalBug, CallToUnknown, DenialOfService | `static_analysis` + `rag_research` |
| GasException, MishandledException, UnusedReturn | `static_analysis` only |

### CLASS_TO_DETECTORS

Maps each vulnerability class to the relevant Slither detector names. Used by `static_analysis` node for detector scoping (3-8x faster than running all 90+ detectors) and by `synthesizer` for matching Slither findings to ML-flagged classes.

### Verdict Computation

**Rule-based** (`compute_verdict`):
- `CONFIRMED` — prob >= 0.50 AND (Slither match OR RAG score >= 0.80)
- `LIKELY` — prob >= 0.50 AND RAG score >= 0.50
- `DISPUTED` — prob >= 0.50 AND no corroborating evidence
- `SAFE` — below threshold

**LLM-adjudicated** (`cross_validator` node): prompts the strong LLM with all evidence per class, returns structured JSON verdicts. Falls back silently to rule-based on LLM failure.

## `nodes.py` — Node Implementations

### Node Summary

| Node | MCP Server | Parallel? | Purpose |
|------|-----------|-----------|---------|
| `quick_screen` | — (direct Slither + Aderyn) | No | Tier-0 screen on every contract |
| `evidence_router` | — (pure function) | No | Logs per-class routing decisions |
| `ml_assessment` | `:8010` | No | ML vulnerability prediction |
| `rag_research` | `:8011` | Yes (deep) | Exploit pattern retrieval |
| `static_analysis` | — (direct Slither + Aderyn) | Yes (deep) | Detector-scoped static analysis |
| `graph_explain` | `:8013` | Yes (deep) | Function-level hotspot attribution |
| `audit_check` | `:8012` | After fan-in | On-chain audit history lookup |
| `cross_validator` | — (LLM call) | After audit_check | LLM-adjudicated per-class verdicts |
| `synthesizer` | — (LLM call) | After cross_validator | Final report assembly |

### MCP Client Pattern

Each node opens a **short-lived SSE connection**, calls exactly one MCP tool, and closes the connection. The shared `_call_mcp_tool()` helper handles the full lifecycle:

```python
async with sse_client(server_url) as (read, write):
    async with ClientSession(read, write) as session:
        await session.initialize()
        result = await session.call_tool(tool_name, arguments)
```

Connection-per-call is intentional for M5 simplicity. Promotable to a module-level persistent client in M6 if RTT measurements show this is a bottleneck.

### quick_screen

Runs Slither and Aderyn on **every contract** before routing. Scoped to High/Critical-impact detectors only:

- Slither: 14 detectors (`reentrancy-eth`, `arbitrary-send-eth`, `controlled-delegatecall`, `integer-overflow`, `suicidal`, etc.)
- Aderyn: H-1 through H-5, C-1 through C-3

Non-fatal: if either tool is not installed, the screen proceeds with the remaining tool.

### static_analysis

Runs Slither directly (not via MCP — Slither is a Python library in this process). Scoping:
1. Collect classes above `DEEP_THRESHOLDS`
2. Map classes → detectors via `CLASS_TO_DETECTORS`
3. Filter `sl._detectors` to scoped set only

Also runs Aderyn on the same temp file for independent findings. Produces `external_call_summary` when ExternalBug is flagged (inter-contract call graph extraction for RAG query enrichment and LLM synthesis).

### cross_validator

Prompts the strong LLM (qwen3.5-9b-ud) with per-class evidence:
- ML tier + probability
- Slither findings mapped to this class
- RAG topic matches
- Prior audit count

Returns structured JSON: `{class: verdict}` where verdict is one of `CONFIRMED | LIKELY | DISPUTED | WATCH | SAFE`. Timeout: 30s.

### synthesizer

Assembles the final report from all available state fields:
1. Merges cross_validator verdicts (or falls back to rule-based)
2. Computes `risk_probability`, `top_vulnerability`, `overall_verdict`
3. Generates rule-based recommendation (fallback) or LLM narrative (primary)
4. Persists report to `data/reports/{contract_address}.json` for the feedback loop bridge

LLM narrative: prompts the strong LLM with structured Markdown output (Severity, Vulnerability Summary, Exploit Pattern, Recommended Fix). Timeout: 45s.

## `graph.py` — Graph Builder

### Compilation

```python
from src.orchestration.graph import build_graph

graph = build_graph()                    # with SqliteSaver (production)
graph = build_graph(use_checkpointer=False)  # no persistence (tests)
```

### Checkpointing

`SqliteSaver` persists state to `agents/data/checkpoints.db` after **every node**. If the process crashes mid-graph, it resumes from the last completed node by providing the same `thread_id`:

```python
result = await graph.ainvoke(
    None,  # state loaded from checkpoint by thread_id
    config={"configurable": {"thread_id": "audit-001"}},
)
```

Falls back to `MemorySaver` (in-process dict, lost on restart) if `langgraph-checkpoint-sqlite` is not installed.

### Module-Level Default

```python
from src.orchestration.graph import audit_graph
# Ready-to-use compiled graph instance
```

Importing the module gives a pre-compiled graph. For tests, call `build_graph(use_checkpointer=False)` directly.

## Usage

### Standalone

```python
import asyncio
from src.orchestration.graph import build_graph

async def audit():
    graph = build_graph(use_checkpointer=False)
    result = await graph.ainvoke(
        {
            "contract_code": "<solidity source>",
            "contract_address": "0x...",
        },
        config={"configurable": {"thread_id": "audit-001"}},
    )
    print(result["final_report"])

asyncio.run(audit())
```

### Smoke Test

```bash
cd agents
poetry run python scripts/smoke_langgraph.py          # mock — no services needed
poetry run python scripts/smoke_langgraph.py --live    # live — all services must be up
```
