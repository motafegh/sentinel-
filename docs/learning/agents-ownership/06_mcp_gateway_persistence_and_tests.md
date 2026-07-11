# Agents Module Ownership — 06: MCP, Gateway, Persistence, and Tests

## Ownership Target

Understand how an external client submits an audit, how orchestration reaches dependent services, and how failures and jobs remain observable.

## Source Reading Order

1. `agents/src/api/gateway.py`
2. `agents/src/api/job_store.py`
3. `agents/src/api/sqlite_job_store.py`
4. `agents/src/mcp/servers/inference_server.py`
5. `agents/src/mcp/servers/rag_server.py`
6. `agents/src/mcp/servers/graph_inspector_server.py`
7. `agents/src/mcp/servers/representation_server.py`
8. `agents/src/mcp/servers/audit_server.py`
9. `agents/tests/` — select tests below before reading test helpers

## Items to Own

- Gateway request lifecycle: create, run, complete/fail, poll, and recover after restart.
- Difference between the gateway job SQLite database and LangGraph checkpoint SQLite database.
- What an MCP server exposes and why orchestration uses it rather than importing every service directly.
- Which upstream services are health-probed and how degraded health is represented.
- What is intentionally synchronous, asynchronous, persistent, or ephemeral.
- How tests isolate the gateway from a real graph and real external services.
- Which production seams require special care: model provenance, proof artifacts, external timeouts, and job recovery.

## Request Exercise

Trace `POST /audit` to a completed report. Record the job record state transition and the boundary crossed at each stage.

## Verification

```bash
cd agents
TMP=/tmp TEMP=/tmp TMPDIR=/tmp poetry run pytest \
  tests/test_gateway.py \
  tests/test_p10_gateway.py \
  tests/test_audit_server.py \
  tests/test_inference_server.py -q
```

## Completion Check

- What survives a gateway restart and what survives a graph restart?
- What does a health endpoint mean by `degraded`?
- Why must a client poll rather than expect a completed report from the submission request?
- Which tests would be run before changing persistent job behavior?

## Intentionally Out of Scope

- Detailed FastAPI, SSE, Web3, or SQLite syntax.
- ML model architecture and ZK circuit internals.
- Evidence-fusion policy changes.
