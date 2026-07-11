# MCP — Model Context Protocol Servers

Five MCP servers that expose SENTINEL's capabilities as tools. Each server runs as an
independent SSE (HTTP) service, callable by LangGraph nodes, Claude Desktop, Cursor, or
any MCP-compatible client.

## Architecture

```
LangGraph nodes                     MCP Servers (SSE)
─────────────                       ─────────────────
ml_assessment     ─── :8010 ──▶    inference_server.py     (Module 1 FastAPI proxy)
rag_research      ─── :8011 ──▶    rag_server.py           (HybridRetriever proxy)
audit_check       ─── :8012 ──▶    audit/ package          (AuditRegistry Web3)
graph_explain     ─── :8013 ──▶    graph_inspector_server.py  (GNN attention / Slither)
(internal)        ─── :8014 ──▶    representation_server.py  (GNN embeddings)
```

All servers share the same SSE transport pattern:
```
GET  /sse          → persistent SSE event stream (client connects here)
POST /messages/    → JSON-RPC tool call endpoint
GET  /health       → liveness probe for Docker / monitoring
```

## Files

| File/Package | Port | Purpose |
|---|------|---------|
| `inference_server.py` | 8010 | Wraps Module 1 `/predict` as MCP tools |
| `rag_server.py` | 8011 | Wraps `HybridRetriever.search()` as MCP tool |
| `audit/` package | 8012 | Reads AuditRegistry on Sepolia via Web3 (was `audit_server.py`, split P2.5) |
| `graph_inspector_server.py` | 8013 | GNN attention hotspots or Slither fallback |
| `representation_server.py` | 8014 | Serves raw GNN node embeddings for attribution |

See `servers/README.md` for detailed per-server documentation.

## MCP Protocol Notes

### SSE Transport

Each server uses `SseServerTransport` from the MCP SDK:
- `/sse` — persistent SSE event stream (client subscribes)
- `/messages/` — JSON-RPC POST endpoint (client sends tool calls)
- `/health` — liveness probe (no MCP)

### Client Connection Pattern

LangGraph nodes open short-lived SSE connections:

```python
async with sse_client(server_url) as (read, write):
    async with ClientSession(read, write) as session:
        await session.initialize()
        result = await session.call_tool(tool_name, arguments)
```

Connection-per-call is intentional for current simplicity. Promotable to persistent
clients in a future phase.

### Mock Modes

Every server supports mock mode for development and CI:

| Server | Mock env var | Default |
|--------|-------------|---------|
| inference | `MODULE1_MOCK` | `false` |
| rag | — (always real if index exists) | — |
| audit | `AUDIT_MOCK` | `true` if no RPC |
| graph_inspector | `GRAPH_INSPECTOR_MOCK` | `false` |
| representation | `REPRESENTATION_MOCK` | `false` |

## Starting Servers

```bash
cd agents

poetry run python -m src.mcp.servers.inference_server
poetry run python -m src.mcp.servers.rag_server
poetry run python -m src.mcp.servers.audit_server
poetry run python -m src.mcp.servers.graph_inspector_server
poetry run python -m src.mcp.servers.representation_server
```

## Smoke Tests

```bash
poetry run python scripts/smoke_inference_mcp.py
poetry run python scripts/smoke_rag_mcp.py
poetry run python scripts/smoke_audit_mcp.py
```
