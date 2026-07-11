# Agents Module Ownership — 02: State, Reducers, and Routing

## Ownership Target

Understand how LangGraph carries `AuditState` between nodes, which updates replace versus merge state, and how externalized routing policy selects the fast or deep path.

## Source Reading Order

Read these executable and policy files in this order:

1. `agents/src/orchestration/state.py`
2. `agents/src/orchestration/routing.py`
3. `agents/src/config/__init__.py`
4. `agents/configs/verdicts_default.yaml`
5. `agents/src/orchestration/nodes/evidence_router.py`
6. `agents/src/orchestration/graph.py` — `_route_from_evidence_router()` only

## Items to Own

- Why `AuditState` is `TypedDict(total=False)` and why nodes return partial updates.
- The difference between normal state replacement and a reducer-backed state field.
- The reducer behavior of `routing_decisions`, `evidence_list`, `injection_matches`, and `tool_status`.
- The ownership and lifecycle of `contract_code`, `ml_result`, `quick_screen_hits`, `static_findings`, `rag_results`, and `final_report`.
- Where per-class deep thresholds and routing rules are loaded from.
- How `compute_active_tools()` converts ML probabilities into a deep-path tool set.
- Why `evidence_router` records a decision but does not select graph edges.
- How a quick-screen hit changes the conditional route even when no ML class crosses a deep threshold.
- Why routing modules must remain independent of LLM calls and raw contract-source prompt handling.

## State Trace Exercise

For each field below, fill in the source-backed contract:

| Field | First writer | Merge behavior | Primary readers | Failure meaning when absent or empty |
|---|---|---|---|---|
| `ml_result` |  |  |  |  |
| `quick_screen_hits` |  |  |  |  |
| `routing_decisions` |  |  |  |  |
| `tool_status` |  |  |  |  |
| `evidence_list` |  |  |  |  |
| `final_report` |  |  |  |  |

Then trace these two decisions using the current configuration, without changing any threshold:

1. Every ML class below its deep threshold and no quick-screen hit.
2. Every ML class below its deep threshold but a quick-screen Slither or Aderyn hit.

## Verification

Use focused routing and isolation tests:

```bash
cd agents
TMP=/tmp TEMP=/tmp TMPDIR=/tmp poetry run pytest \
  tests/test_routing_phase0.py \
  tests/test_graph_routing.py \
  tests/test_routing_isolation.py -q
```

## Completion Check

This artifact is complete when Ali can answer without notes:

- Which fields append, which fields replace, and why does that distinction matter?
- Why is `tool_status` a custom one-level-deep merge rather than a normal dict replacement?
- Where must a routing threshold be changed, and what measurement is required before changing it?
- Why can the graph take a deep path even when `compute_active_tools()` returns an empty list?
- Why must `evidence_router` and the graph conditional function agree but remain separate?

## Intentionally Out of Scope

- Evidence-strength and reliability calculations.
- Detector implementation, RAG retrieval, LLM prompts, and formal verification.
- Gateway persistence and MCP transport.
- Changing thresholds, routing rules, or state schemas.
