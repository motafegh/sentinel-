# Agents Module Ownership — 01: Request Lifecycle

## Ownership Target

Trace one Solidity audit request from graph entry to final report. Do not study implementation details of external tools yet.

## Source Reading Order

Read these executable files in this order:

1. `agents/src/orchestration/graph.py`
2. `agents/src/orchestration/state.py`
3. `agents/src/orchestration/nodes/ml_assessment.py`
4. `agents/src/orchestration/nodes/quick_screen.py`
5. `agents/src/orchestration/nodes/evidence_router.py`
6. `agents/src/orchestration/routing.py`
7. `agents/src/orchestration/nodes/synthesizer.py`

## Items to Own

- Graph entry inputs: `contract_code` and `contract_address`.
- The difference between a graph node, an edge, and a conditional-routing function.
- The exact condition for the fast path.
- The exact condition for the deep path and its parallel fan-out.
- Which state fields are added by the first three nodes.
- Why `quick_screen` can override an ML-safe fast-path decision.
- Where final-report creation begins and which later nodes enrich it.
- How an ML or tool failure remains visible through `tool_status`.

## Trace Exercise

For one contract, write a short state trace containing only:

| Step | Node or route | Reads from state | Adds or changes state | Next destination |
|---|---|---|---|---|
| 1 | `ml_assessment` |  |  |  |
| 2 | `quick_screen` |  |  |  |
| 3 | `evidence_router` |  |  |  |
| 4 | Conditional route |  |  |  |
| 5 | `synthesizer` |  |  |  |

Complete the trace for both a fast-path and a deep-path request.

## Verification

Use focused tests after reading:

```bash
cd agents
TMP=/tmp TEMP=/tmp TMPDIR=/tmp poetry run pytest \
  tests/test_graph_routing.py tests/test_smoke_e2e.py -q
```

## Completion Check

This artifact is complete when Ali can answer without notes:

- Why is `quick_screen` before routing?
- What prevents an ML-safe response from automatically ending the audit?
- Which route skips deep analysis, and why?
- Which state fields make an unavailable tool distinguishable from a clean tool result?
- Which file changes the graph topology, and which file only records routing decisions?

## Intentionally Out of Scope

- How Slither, Aderyn, RAG, Halmos, or the LLM work internally.
- Evidence weights and verdict math.
- MCP transport, gateway jobs, SQLite persistence, and ZK submission.
- Any source-code edits.
