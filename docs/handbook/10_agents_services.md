# 10 — AGENTS services, RAG, gateway, and feedback

**Read this when:** you need the five MCP servers, gateway jobs, RAG ingestion/retrieval, health monitoring, or feedback ingestion.

**Skip this if:** you only need in-process graph logic; read [orchestration](09_agents_orchestration.md).

**Estimated reading time:** 15 minutes.

## 30-second summary

AGENTS is a process topology, not one server. The gateway listens on 8000, ML on 8001, and five SSE MCP services on 8010–8014: inference, RAG, audit, graph inspector, and representation. The gateway persists asynchronous audit jobs in SQLite and monitors upstream health. RAG builds FAISS/BM25 indexes from external audit knowledge and retrieves with reciprocal-rank fusion. Feedback ingestion is a separate data loop.

## Just-enough mental model

```text
client → gateway:8000 → in-process LangGraph
                         ├→ ML API:8001
                         └→ MCP clients as nodes need them

MCP: inference:8010 | RAG:8011 | audit:8012 | graph:8013 | representation:8014

external reports → ingestion → chunk/embed/index → hybrid retrieval → RAG evidence
completed audit feedback → ingestion/label review → future DATA/ML cycle
```

MCP standardizes tool calls; it does not imply every tool is invoked by the gateway’s default flow.

## Actual runtime/source walkthrough

### Five MCP services

| Port | Service/source | Tool family |
|---:|---|---|
| 8010 | [`inference_server.py`](../../agents/src/mcp/servers/inference_server.py) | `predict`, `batch_predict`; proxies ML HTTP |
| 8011 | [`rag_server.py`](../../agents/src/mcp/servers/rag_server.py) | `search`; hybrid knowledge retrieval |
| 8012 | [`audit`](../../agents/src/mcp/servers/audit) | `get_latest_audit`, `get_audit_history`, `check_audit_exists`, `submit_audit` |
| 8013 | [`graph_inspector_server.py`](../../agents/src/mcp/servers/graph_inspector_server.py) | `get_graph_hotspots`; ML hotspots with Slither fallback |
| 8014 | [`representation_server.py`](../../agents/src/mcp/servers/representation_server.py) | `get_function_cfgs`; DATA CFG-derived structural signals |

Each uses SSE endpoints plus `/health`. Service defaults are source-derived and environment-overridable.

### Gateway and JobStore

[`gateway.py`](../../agents/src/api/gateway.py) — `agents/src/api/gateway.py::create_app` creates five public routes, schedules each audit with `asyncio.create_task`, and invokes `graph.ainvoke`. [`sqlite_job_store.py`](../../agents/src/api/sqlite_job_store.py) — `::SqliteJobStore` stores queued/running/completed/failed records. Startup marks interrupted jobs failed and begins periodic service probes. The in-memory [`JobStore`](../../agents/src/api/job_store.py) remains a test/drop-in implementation; it is not the default runtime store.

### RAG lifecycle

Fetchers under [`rag/fetchers`](../../agents/src/rag/fetchers) normalize public audit/exploit sources. [`chunker.py`](../../agents/src/rag/chunker.py) creates retrievable chunks, [`embedder.py`](../../agents/src/rag/embedder.py) creates dense vectors, and [`build_index`](../../agents/src/rag/build_index) writes index/metadata artifacts. [`retriever.py`](../../agents/src/rag/retriever.py) combines FAISS dense ranking and BM25 sparse ranking with reciprocal-rank fusion and metadata filtering.

### Feedback loop

[`ingestion/pipeline.py`](../../agents/src/ingestion/pipeline.py), [`feedback_loop.py`](../../agents/src/ingestion/feedback_loop.py), and scheduler adapters ingest completed audit/report feedback, deduplicate it, and prepare reviewable records. Feedback does not automatically become trusted ground truth or retrain a model.

## Interfaces, data shapes, and configuration

Gateway routes:

- `GET /` — service identity;
- `GET /health` — gateway/job counts plus cached service health;
- `POST /audit` — validate and enqueue source/address/metadata;
- `GET /audit/{job_id}` — status, error, or report;
- `GET /audit` — recent bounded jobs.

MCP uses JSON Schema tool inputs and returns text content containing structured JSON. Treat `status`, `failed_step`, `reason`, fallback provenance, and mock flags as part of the interface.

Configuration is environment-driven for ports, ML URL, RPC/registry addresses, operator key presence, mock modes, index locations, timeouts, and gateway DB. Document names, defaults, and prerequisites—not values from `.env`.

## Failure modes and current limitations

- Gateway health can be degraded while it still accepts jobs; callers must inspect service details.
- Gateway audit completion does not mean an on-chain transaction was attempted.
- The graph-inspector source currently defaults `SENTINEL_ML_API_URL` to port 8000, while ML runs on 8001; set the variable explicitly until that product defect is fixed.
- `submit_audit` is independently invoked and requires model, proof, RPC, key, funded/staked operator, and chain artifacts.
- Graph-inspector Slither fallback is not equivalent to GNN hotspots and must be labeled.
- RAG indexes are generated/local artifacts; empty or stale indexes reduce evidence quality.
- External source changes and embedding-model changes require index rebuilds and schema identity updates.
- Feedback is untrusted until provenance, deduplication, and label review complete.

## Common change recipe

To add an MCP tool:

1. Define a narrow JSON Schema and structured success/degraded/error result.
2. Add tool registration and dispatcher handling together.
3. Add timeouts, resource bounds, health semantics, and tests.
4. Wire a graph node only if orchestration should use it; MCP availability alone is not wiring.
5. Update the service/tool registry in metadata and cross-module contracts.

To add a RAG source, define provenance/license, normalization, stable IDs, dedup rules, metadata, contamination checks, and a rebuild/version plan.

## Verification commands

```bash
export TMPDIR=/tmp TMP=/tmp TEMP=/tmp
cd agents && poetry run pytest tests/test_gateway.py tests/test_mcp_servers.py -q  # targeted if present
cd .. && python3 docs/handbook/tools/verify_handbook.py static
curl -fsS http://127.0.0.1:8000/health                                 # live
for p in 8010 8011 8012 8013 8014; do curl -fsS "http://127.0.0.1:$p/health"; done  # live
```

If a named targeted file differs in the checkout, use `rg --files agents/tests | rg 'gateway|mcp'` to select it. Current counts are in [current status](16_current_status.md).

## Optional deep references

- [`docs/learning/05_mcp_architecture.md`](../learning/05_mcp_architecture.md)
- [`docs/learning/06_rag_hybrid_retrieval.md`](../learning/06_rag_hybrid_retrieval.md)
- [`docs/learning/07_gateway_production.md`](../learning/07_gateway_production.md)
- [Runtime flows](02_runtime_flows.md)
- [Operations](14_operations.md)

## Technical mastery layer

### Prerequisite knowledge

Know MCP tool schemas, async HTTP, SQLite state machines, vector/BM25 retrieval, and event ingestion.

### Source map and reading order

Read each MCP server’s tool list/handler, RAG build and `HybridRetriever`, gateway models/store/runner, and feedback listener/ingester. [T08](technical/08_services_rag_gateway.md) is the current source-guided companion.

### Execution trace and worked example

MCP calls validate JSON and translate to ML/RAG/audit/graph/representation operations. Gateway stores queued job, runs graph under timeout, persists report/failure, and recovers abandoned work. RAG refuses mismatched vector/chunk counts.

### Implementation practice

[L08](labs/08_services_rag_gateway_recovery.md) tests schemas, persistence, and recovery. New tool fields must update schema, handler, errors, call routing, health/dependency documentation, and consumer tests.

### Review and ownership check

Can you list five services/ports/tools and explain why gateway completion does not invoke audit MCP submission?
