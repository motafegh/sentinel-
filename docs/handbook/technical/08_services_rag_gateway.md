# T08 — MCP, RAG, gateway, persistence, and feedback

## Learning outcome

You can trace tool schemas and network calls across five MCP services, follow gateway job persistence/recovery, diagnose RAG index integrity, and explain feedback ingestion boundaries.

## Prerequisites

Read [AGENTS services](../10_agents_services.md). Know FastAPI, async background tasks, JSON schemas, SQLite transactions, vector/keyword retrieval, and retry/idempotency concepts.

## Source map and reading order

1. `agents/src/mcp/servers/{inference,rag,graph_inspector,representation}_server.py` and `audit/` package.
2. `agents/src/rag/build_index/`, then `retriever.py::HybridRetriever.search`.
3. `agents/src/api/models.py`, `job_store.py`, and `sqlite_job_store.py::SqliteJobStore`.
4. `api/gateway.py::{create_app,submit_audit,_run_job,_probe_services}`.
5. `agents/src/ingestion/feedback_loop.py::{OnChainListener,FeedbackIngester}`.

## Entry point and complete call chain

Each MCP server declares tool schemas and routes tool names to handlers, usually translating JSON into an HTTP/local operation and returning text content. RAG build fetches/deduplicates/chunks/embeds and writes FAISS, BM25, chunks, and metadata; retrieval combines semantic and lexical ranks with RRF. Gateway `POST /audit` validates input, persists a queued record, starts `_run_job`, marks running, invokes LangGraph with timeout, and stores completion/failure. Startup recovery moves abandoned queued/running jobs into an explicit recoverable state. Feedback listener reads chain events and sends curated records into ingestion.

## Important symbols and configuration

- Gateway 8000; MCP inference/RAG/audit/graph/representation 8010–8014; ML 8001.
- SQLite state transitions are conditional: queued → running → completed/failed.
- Health probes report dependency state but do not make absent services into success.
- RAG requires FAISS vector count to equal chunk count; mismatch is corruption.
- Gateway orchestration does not call audit MCP `submit_audit`; direct chain submission remains separately invoked.

## Annotated source excerpt

Source: `agents/src/api/gateway.py::_run_job`

```python
store.mark_running(job_id)
result = await asyncio.wait_for(
    graph.ainvoke(initial_state),
    timeout=audit_timeout_s,
)
final_report = result.get("final_report", {}) or {}
store.mark_completed(job_id, report)
```

The background task owns lifecycle transitions. Exceptions and timeouts are persisted as job failure instead of escaping and killing the service.

## Worked example

`POST /audit` returns HTTP 202 with UUID `J`. SQLite row `J` is queued and retains source for execution; public serialization omits source. `_run_job` marks running and invokes the graph. After success, a compact report is stored and `GET /audit/J` returns completed. A restart between queued/running and completion is detected by recovery logic; it is not silently presented as completed.

## Success trace

Tool request satisfies schema; dependency call succeeds; MCP content is valid JSON; RAG indexes are synchronized; gateway state transitions once; graph returns a dict; report persists across store recreation; health exposes service status; feedback is deduplicated and provenance-bearing.

## Failure trace

Partial batch tool failures remain item-local where specified. Missing RAG index raises with the build command. FAISS/chunk count mismatch raises immediately. Gateway timeout stores failure. The graph-inspector service currently has a wrong default ML URL port in source; override/configure it rather than documenting the defect as healthy behavior. Shared direct-proof files are not concurrency-safe.

## Design reasoning and rejected alternatives

Small MCP services isolate tool failure and schemas. Hybrid RAG covers semantic synonyms and exact identifiers; either-only retrieval was rejected. SQLite supplies durable single-host jobs without requiring a distributed queue. Background tasks keep submit latency low, but do not provide multi-process queue guarantees.

## Safe change walkthrough

For a new MCP tool, define strict input schema, implement handler and error payload, add list/call routing tests, declare port/dependency/health behavior, then update orchestration consumption. For a gateway field, change Pydantic model, persistence serialization/migration, public redaction, lifecycle tests, and recovery tests together.

## Guided lab

Complete [L08 — MCP, RAG, gateway, and SQLite recovery](../labs/08_services_rag_gateway_recovery.md).

## Tests and expected results

```bash
cd agents && TMPDIR=/tmp TMP=/tmp TEMP=/tmp poetry run pytest -q \
  tests/test_inference_server.py tests/test_gateway.py tests/test_p10_gateway.py
```

Expected: schema routing, job lifecycle, persistence, recovery, redaction, and health-model tests pass without requiring all live services.

## Review questions

Why can HTTP 202 precede audit execution? What does recovery guarantee? How is RAG corruption detected? Which path actually invokes `submit_audit`?

## Ownership checklist

- I can map every service to port, tools, and dependency.
- I can reproduce a job lifecycle from SQLite.
- I distinguish test/mocked MCP behavior from live dependencies.
- I know why gateway completion is not on-chain submission.
