# MCP — Model Context Protocol Servers

Four MCP servers that expose SENTINEL's capabilities as tools. Each server runs as an independent SSE (HTTP) service, callable by LangGraph nodes, Claude Desktop, Cursor, or any MCP-compatible client.

## Architecture

```
LangGraph nodes                     MCP Servers (SSE)
─────────────                       ─────────────────
ml_assessment     ─── :8010 ──▶    inference_server.py    (Module 1 FastAPI proxy)
rag_research      ─── :8011 ──▶    rag_server.py          (HybridRetriever proxy)
audit_check       ─── :8012 ──▶    audit_server.py        (AuditRegistry Web3)
graph_explain     ─── :8013 ──▶    graph_inspector_server.py  (GNN attention / Slither)
```

All servers share the same SSE transport pattern:
```
GET  /sse          → persistent SSE event stream (client connects here)
POST /messages/    → JSON-RPC tool call endpoint
GET  /health       → liveness probe for Docker / monitoring
```

## Files

| File | Lines | Port | Purpose |
|------|-------|------|---------|
| `inference_server.py` | 501 | 8010 | Wraps Module 1 `/predict` as MCP tools |
| `rag_server.py` | 353 | 8011 | Wraps `HybridRetriever.search()` as MCP tool |
| `audit_server.py` | 717 | 8012 | Reads AuditRegistry on Sepolia via Web3 |
| `graph_inspector_server.py` | 544 | 8013 | GNN attention hotspots or Slither fallback |

## Server Details

### inference_server.py — `:8010`

Wraps Module 1's FastAPI inference server.

**Tools:**

| Tool | Required | Optional | Returns |
|------|----------|----------|---------|
| `predict` | `contract_code: str` | `contract_address: str` | Track 3 `PredictResponse` |
| `batch_predict` | `contracts: list` | — | `{"results": [...]}` (max 20) |

**Mock fallback:** When `_MOCK_MODE=true` or Module 1 is unreachable, returns realistic fake predictions matching the Track 3 schema (three-tier: `confirmed`, `suspicious`, `vulnerabilities`).

**Shared HTTP client (A-20):** A single `httpx.AsyncClient` is created at server startup and reused across all tool calls. Avoids per-call TCP+TLS handshake overhead (~20-50ms per connection).

**Configuration (`.env`):**

```bash
MODULE1_INFERENCE_URL=http://localhost:8001    # Module 1 FastAPI
MCP_INFERENCE_PORT=8010
MODULE1_TIMEOUT=30.0
MODULE1_MOCK=false
```

### rag_server.py — `:8011`

Wraps `HybridRetriever.search()` from the RAG module.

**Tool: `search`**

| Parameter | Type | Default | Max | Description |
|-----------|------|---------|-----|-------------|
| `query` | `str` | (required) | — | Natural language search query |
| `k` | `int` | 5 | 20 | Number of results |
| `filters` | `dict` | `{}` | — | Metadata filters (see RAG README) |

**Response shape per result:**
```json
{
    "chunk_id": "abc123-0",
    "content": "...",
    "doc_id": "abc123",
    "chunk_index": 0,
    "total_chunks": 3,
    "metadata": {"vuln_type": "Reentrancy", "date": "2023-03-15", ...},
    "score": 0.842
}
```

**Lazy initialization (Bug 10 fix):** `HybridRetriever()` is loaded in `_on_startup()` when the server starts, not at module import time. This prevents crashes in CI or unit tests where the index doesn't exist.

**Configuration (`.env`):**

```bash
MCP_RAG_PORT=8011
RAG_DEFAULT_K=5
```

### audit_server.py — `:8012`

Reads `AuditRegistry.sol` on Sepolia via Web3.py. Query-only in current phase — `submitAudit` is deferred until ZKML + Track 3 proof semantics are finalised.

**Tools:**

| Tool | Required | Optional | Returns |
|------|----------|----------|---------|
| `get_latest_audit` | `contract_address: str` | — | `{score, label, proof_hash, timestamp, agent, verified}` |
| `get_audit_history` | `contract_address: str` | `limit: int` (default 10, max 50) | `{count, records: [...]}` |
| `check_audit_exists` | `contract_address: str` | — | `{exists: bool, count: int}` |

**Score decoding:** `score = scoreFieldElement / 8192` (EZKL scale factor). Example: `4497 / 8192 = 0.5490`.

**Mock mode:** Auto-enabled when `SEPOLIA_RPC_URL` is absent or `AUDIT_MOCK=true`. Returns realistic fake data matching the real response shape.

**ABI loading (Bug 2 fix):** The AuditRegistry ABI is loaded lazily in `_on_startup()`, not at module import time. Mock mode starts cleanly without compiled contracts.

**Address validation:** `_validate_address()` checksums lowercase hex addresses via `Web3.to_checksum_address()` (EIP-55).

**Configuration (`.env`):**

```bash
MCP_AUDIT_PORT=8012
SEPOLIA_RPC_URL=<your-rpc>
AUDIT_REGISTRY_ADDRESS=0x14E5eFb6DE4cBb74896B45b4853fd14901E4CfAf
AUDIT_MOCK=false
```

### graph_inspector_server.py — `:8013`

Function-level hotspot attribution — tells the auditor *where* in the contract the suspicious pattern lives.

**Tool: `get_graph_hotspots`**

| Parameter | Type | Required |
|-----------|------|----------|
| `contract_code` | `str` | Yes |
| `flagged_classes` | `list[str]` | No (default: all flagged) |

**Three-tier fallback:**

| Priority | Backend | Signal | When used |
|----------|---------|--------|-----------|
| 1 | ML API `/hotspots` | Real GNN embedding-norm scores | ML API reachable |
| 2 | Slither analysis | Structural proxy scoring | ML API unreachable |
| 3 | Mock data | Deterministic stub | Both unavailable |

**GNN attention (Phase 2):** Calls `POST /hotspots` on the ML inference API. Returns L2 norm of each function-level node's GNN embedding — the real model signal, not a Slither proxy. Higher norm = GNN concentrated more message-passing signal on that node.

**Slither fallback (Phase 1):** Scores functions by:
- Detector hits × impact weight (High=3.0, Medium=2.0, Low=1.0)
- External call count × 0.5
- State write count × 0.3
- High-level calls × 0.4
- External-facing visibility × 0.2

**Analysis mode:** Returned in response as `"analysis_mode"` — `"gnn_attention"`, `"slither"`, or `"mock"`. Enables downstream consumers to distinguish ground-truth attribution from proxy scoring.

**Configuration (`.env`):**

```bash
MCP_GRAPH_INSPECTOR_PORT=8013
SENTINEL_ML_API_URL=http://localhost:8000
GRAPH_INSPECTOR_HOTSPOTS_TIMEOUT=60
GRAPH_INSPECTOR_MOCK=false
```

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

Connection-per-call is intentional for M5 simplicity. Promotable to persistent clients in M6.

### Mock Modes

Every server supports mock mode for development and CI:

| Server | Mock env var | Default |
|--------|-------------|---------|
| inference | `MODULE1_MOCK` | `false` |
| rag | — (always real if index exists) | — |
| audit | `AUDIT_MOCK` | `true` if no RPC |
| graph_inspector | `GRAPH_INSPECTOR_MOCK` | `false` |

## Starting Servers

```bash
cd agents

# Individual servers
poetry run python -m src.mcp.servers.inference_server
poetry run python -m src.mcp.servers.rag_server
poetry run python -m src.mcp.servers.audit_server
poetry run python -m src.mcp.servers.graph_inspector_server
```

## Smoke Tests

```bash
poetry run python scripts/smoke_inference_mcp.py
poetry run python scripts/smoke_rag_mcp.py
poetry run python scripts/smoke_audit_mcp.py
```
