# Orchestration — LangGraph Audit Pipeline

14-node LangGraph `StateGraph` that orchestrates the full SENTINEL audit workflow.
Conditional routing selects fast or deep path based on ML probabilities and static
analysis signals. All decision numbers are loaded from `src/config/` (Rule 5B).

## Architecture

```
START
  │
  ▼
ml_assessment          ← ML model (Module 1 MCP :8010) + model hash (P5)
  │
  ▼
quick_screen           ← Slither + Aderyn on every contract (Tier 0)
  │
  ▼
evidence_router        ← logs routing_decisions to AuditState
  │
  ├─ fast path ──────────────────────────────────────────────────────────▶ synthesizer
  │   (ML below all thresholds AND quick_screen clean)                         │
  │                                                                            │
  └─ deep path (parallel fan-out) ─────────────────────────────────┐          │
       rag_research          ← FAISS+BM25 DeFiHackLabs (MCP :8011) │          │
       static_analysis       ← Slither+Aderyn per-class detectors   │          │
       graph_explain         ← GNN hotspots (MCP :8013)             │          │
       formal_verification   ← Halmos symbolic execution (P8a)      │          │
                                            │  (all fan-in at)       │          │
                                            ▼                        │          │
                                       audit_check  ◀───────────────┘          │
                                            │                                   │
                                            ▼                                   │
                                    consensus_engine   ← A.6/A.7 weighted vote │
                                            │                                   │
                                            ▼                                   │
                                     cross_validator   ← A.4 debate (P/D/J)    │
                                            │                                   │
                                            └──────────────────────▶ synthesizer
                                                                         │
                                                                    reflection    ← A.3
                                                                         │
                                                                     explainer    ← A.8
                                                                         │
                                                                    visualizer    ← A.9
                                                                         │
                                                                        END
```

### Fast vs Deep Path

| Condition | Path |
|-----------|------|
| All ML class probs below `DEEP_THRESHOLDS` AND `quick_screen` clean | Fast → synthesizer |
| ML safe but `quick_screen` fires (High/Critical hit) | Screen-escalated deep: `static_analysis` only |
| Any ML prob ≥ class threshold | Deep: full tool fan-out per class routing rules |

The fast path still runs the full post-synthesis chain (reflection → explainer →
visualizer → END). Every audit gets a complete final report.

## Files

### `graph.py`

Builds and compiles the `StateGraph`. 14 nodes, all wrapped with `timed_node()` for
uniform START/DONE+elapsed logging. Lazy `audit_graph` singleton via PEP 562
`__getattr__` — the graph is not compiled at import time.

**Checkpointer:** `SqliteSaver` (persists to `data/checkpoints.db`) by default.
Falls back to `MemorySaver` if `langgraph-checkpoint-sqlite` is not installed.
Tests call `build_graph(use_checkpointer=False)` to avoid I/O.

### `state.py` — AuditState

`TypedDict(total=False)` — every field is optional; nodes return only the fields they
updated. LangGraph merges partial dicts back automatically.

#### Key Fields

| Field | Type | Set by | Notes |
|-------|------|--------|-------|
| `contract_code` | `str` | caller | Never mutated |
| `contract_address` | `str` | caller | Never mutated |
| `ml_result` | `dict` | `ml_assessment` | Three-tier schema: label, probabilities, confirmed, suspicious |
| `ml_hotspots` | `list[dict]` | `ml_assessment` / `graph_explain` | {class, fn_name, lines, score} |
| `quick_screen_hits` | `dict` | `quick_screen` | {slither: [...], aderyn: [...]} |
| `routing_decisions` | `Annotated[list, operator.add]` | `evidence_router` + any node | Append-reducer |
| `static_findings` | `list[dict]` | `static_analysis` | {tool, detector, impact, description, lines} |
| `rag_results` | `list[dict]` | `rag_research` | Ranked RAG chunks |
| `graph_explanations` | `dict` | `graph_explain` | {class: {subgraph_json, feature_descriptions}} |
| `symbolic_findings` | `list[dict]` | `formal_verification` | {invariant, proven, counterexample} |
| `audit_history` | `list[dict]` | `audit_check` | Historical on-chain audit records |
| `verdicts` | `dict[str, str]` | `synthesizer` / `cross_validator` | CONFIRMED/LIKELY/DISPUTED/SAFE per class |
| `consensus_verdict` | `dict` | `consensus_engine` | A.6 weighted vote per class |
| `confidence_by_class` | `dict[str, float]` | `consensus_engine` / `cross_validator` | A.7 Bayesian confidence |
| `debate_transcript` | `dict[str, str]` | `cross_validator` | A.4: {prosecutor, defender, judge} |
| `reflection_notes` | `dict` | `reflection` | A.3 self-critique |
| `metric_attribution` | `dict` | `explainer` | A.8 LIME-style {ml_pct, slither_pct, rag_pct} |
| `hotspot_visualization` | `str\|None` | `visualizer` | A.9 self-contained HTML |
| `final_report` | `dict` | `synthesizer` | Complete audit output |
| `evidence_list` | `Annotated[list, operator.add]` | any node | P2: append-reducer, consumed by fuse() |
| `verdict_provable` | `dict[str, str]` | `synthesizer` | P2: ZK-anchorable tier (deterministic evidence only) |
| `verdict_full` | `dict[str, str]` | `synthesizer` | P2: human-report tier (all evidence) |
| `tool_status` | `Annotated[dict, _merge_tool_status]` | any node | Rule 5C: explicit ran/reason per tool |
| `injection_matches` | `Annotated[list, operator.add]` | `cross_validator`, `synthesizer` | P4: injection detections |
| `model_hash` | `str` | `ml_assessment` | P5: SHA-256 of checkpoint file |

### `routing.py` — Per-Class Routing

All routing constants (`DEEP_THRESHOLDS`, `ROUTING_RULES`, `OVERALL_VERDICT_RANK`) are
loaded lazily from `src/config/` via PEP 562 `__getattr__`. No hardcoded thresholds.

`CLASS_TO_DETECTORS` maps each of the 10 vulnerability classes to the relevant Slither
detector names used by `static_analysis` for scoped detector runs and verdict matching.

Key functions:

| Function | Returns | Notes |
|----------|---------|-------|
| `compute_active_tools(ml_result)` | `list[str]` | Tool node names to fan-out to |
| `build_routing_decisions(ml_result)` | `list[str]` | Human-readable log entries for state |
| `compute_verdict(cls, prob, static_findings, rag_results, path_taken)` | `(verdict, sources)` | Rule-based, no LLM |
| `compute_overall_verdict(verdicts)` | `str` | Max-rank across all classes |
| `prob_to_severity(prob)` | `str` | CRITICAL/HIGH/MEDIUM/LOW/INFO |

### `nodes/` Package

Was the monolithic `nodes.py` (2,280 lines, deleted in P2). Now split into 14 focused
modules + a shared helpers file.

| File | Node | Path |
|------|------|------|
| `ml_assessment.py` | `ml_assessment` | Always |
| `quick_screen.py` | `quick_screen` | Always (Tier 0) |
| `evidence_router.py` | `evidence_router` | Always |
| `rag_research.py` | `rag_research` | Deep path |
| `static_analysis.py` | `static_analysis` | Deep path |
| `graph_explain.py` | `graph_explain` | Deep path |
| `formal_verification.py` | `formal_verification` | Deep path (P8a Halmos) |
| `audit_check.py` | `audit_check` | Deep path (fan-in) |
| `consensus_engine.py` | `consensus_engine` | Deep path |
| `cross_validator.py` | `cross_validator` | Deep path (A.4 debate + P4 injection guard) |
| `synthesizer.py` | `synthesizer` | Both paths (fuse() verdict producer) |
| `reflection.py` | `reflection` | Post-synthesis |
| `explainer.py` | `explainer` | Post-synthesis |
| `visualizer.py` | `visualizer` | Post-synthesis |
| `_helpers.py` | — | Shared: `_call_mcp_tool()`, `_llm_enabled()`, `AderynRunError`, tool subprocess helpers |

#### `formal_verification.py` — P8a Halmos

Runs Halmos symbolic execution on contracts in the deep path. Generates a temp Foundry
test harness, runs `forge build` + `halmos --json-output`, parses results to
`Evidence(kind=FORMAL, deterministic=True)`. Emits `symbolic_findings` to state. Fail-
soft on missing tools or compile errors — sets `tool_status["halmos"]` with
`ran=False` and reason.

#### Rule 5C in nodes

Every tool invocation (Slither, Aderyn, Halmos, MCP calls) must surface failure via
`tool_status`, not silently return `[]`. The acceptable patterns are:

1. `raise` with a precise message
2. Structured degraded return with `{"ran": False, "reason": "...", "skipped": True}`
3. Set `state["tool_status"][tool] = {"ran": False, "reason": ...}`

Silent `except FileNotFoundError: return []` is forbidden. See `CLAUDE.md §C`.

### `verdict/` Package

Verdict production split from synthesizer into a dedicated package (P2, 2026-06-24).
`fuse()` is the sole verdict producer — `_reconcile_shim.py` was deleted.

| File | Purpose |
|------|---------|
| `evidence.py` | `Evidence` dataclass: source, vuln_class, polarity, strength, reliability, kind, deterministic |
| `fuse.py` | `fuse(evidence_list, reliability_weights)` → `{verdict_provable, verdict_full}` |
| `reliability.py` | Reads L3 from `reliability_v3.yaml`, falls back to L1 |
| `emit.py` | `emit_evidence()`, `emit_halmos_evidence()` helpers for nodes |
| `verdict.py` | Verdict constants and helpers |

#### Dual-tier verdict system (P2)

| Tier | Field | Source | ZK-anchorable |
|------|-------|--------|---------------|
| Provable | `verdict_provable` | Deterministic evidence only (Slither, Aderyn, Halmos) | Yes |
| Full | `verdict_full` | All evidence (+ ML, RAG, LLM debate) | No |

The ZK circuit can only commit to deterministic evidence. The human report uses both.

### Supporting Modules

| File | Purpose |
|------|---------|
| `consensus.py` | A.6 — weighted ML/Slither/Aderyn vote per class; `ML_WEIGHT_SCALE` discounts ML |
| `confidence.py` | A.7 — Bayesian staged confidence tracking through pipeline |
| `attribution.py` | A.8 — LIME-style evidence-source breakdown (ml_pct/slither_pct/rag_pct) |
| `visualizer.py` | A.9 — generates self-contained interactive HTML hotspot report |
| `timeouts.py` | Centralized timeout constants for all tool invocations |
| `timing.py` | `timed_node()` wrapper, `step_timer` context manager |

## Usage

```python
from src.orchestration.graph import build_graph

graph = build_graph()
result = await graph.ainvoke(
    {"contract_code": "<solidity source>", "contract_address": "0x..."},
    config={"configurable": {"thread_id": "audit-001"}},
)
print(result["final_report"])
```

Resume from checkpoint after crash:
```python
result = await graph.ainvoke(
    None,   # state loaded from SqliteSaver by thread_id
    config={"configurable": {"thread_id": "audit-001"}},
)
```

Deterministic mode (no LLM calls, reproducible for ZK):
```bash
SENTINEL_DETERMINISTIC=1 poetry run python scripts/smoke_langgraph.py
```
