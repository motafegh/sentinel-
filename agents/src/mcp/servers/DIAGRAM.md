# MCP Servers — Model Context Protocol SSE Servers

> **Scope:** `agents/src/mcp/servers/` — 4 SSE/HTTP servers that expose
> SENTINEL's capabilities as MCP tools. Source-of-truth: the code, not
> this file. Last verified: 2026-06-23.

---

## 1. Server Mesh at a Glance

```
   LangGraph nodes                          MCP servers (Starlette + uvicorn)
   ────────────────                         ────────────────────────────────
   ② ml_assessment       ── :8010 ──►  inference_server.py     (Module 1 /predict proxy)
   ④ rag_research        ── :8011 ──►  rag_server.py           (HybridRetriever proxy)
   ⑦ audit_check         ── :8012 ──►  audit_server.py         (Sepolia AuditRegistry Web3)
   ⑥ graph_explain       ── :8013 ──►  graph_inspector_server.py (GNN attention / Slither)

   All 4 servers share the same SSE transport pattern:
   ┌────────────────────────────────────────────────────┐
   │  Route("/sse",       handle_sse)   — persistent SSE │
   │  Mount("/messages/", handle_post)  — JSON-RPC POST  │
   │  Route("/health",    health)       — liveness probe │
   └────────────────────────────────────────────────────┘
   Transport: SSE (HTTP). stdio alternative exists but unused.
```

### Port Map

| Port | Server file | Backing system | Default env | Mock env |
|------|-------------|----------------|-------------|----------|
| 8010 | `inference_server.py` | Module 1 FastAPI `/predict` (port 8001) | `MODULE1_INFERENCE_URL` | `MODULE1_MOCK` |
| 8011 | `rag_server.py` | `HybridRetriever` (FAISS + BM25) | `MCP_RAG_PORT` | — (always real if index exists) |
| 8012 | `audit_server.py` | Sepolia `AuditRegistry.sol` via Web3.py | `SEPOLIA_RPC_URL` | `AUDIT_MOCK` (auto if no RPC) |
| 8013 | `graph_inspector_server.py` | Module 1 FastAPI `/hotspots` + Slither fallback | `SENTINEL_ML_API_URL` | `GRAPH_INSPECTOR_MOCK` |

---

## 2. Per-Server Architecture — The 4 Servers

### 2.1 sentinel-inference `:8010` (`inference_server.py`, 491 lines)

```
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  Listens:  http://0.0.0.0:8010                                          │
  │  Tools:    predict, batch_predict                                       │
  │                                                                         │
  │  ┌─────────────────────────────────────────────────────────────────┐   │
  │  │  Server instance (MCP SDK)                                       │   │
  │  │  server = Server("sentinel-inference")                           │   │
  │  │                                                                  │   │
  │  │  @server.list_tools() → [Tool(predict), Tool(batch_predict)]    │   │
  │  │  @server.call_tool()  → dispatcher                              │   │
  │  │                                                                  │   │
  │  │  predict  → _handle_predict   → _call_inference_api()           │   │
  │  │  batch    → _handle_batch     → loop _call_inference_api()      │   │
  │  └─────────────────────────────────────────────────────────────────┘   │
  │                                                                         │
  │  ┌─────────────────────────────────────────────────────────────────┐   │
  │  │  Shared httpx.AsyncClient (lifespan-managed, A-20)               │   │
  │  │  Created in _on_startup(), closed in _on_shutdown()             │   │
  │  │  Avoids per-call TCP+TLS handshake (~20-50ms)                   │   │
  │  └─────────────────────────────────────────────────────────────────┘   │
  │                                                                         │
  │  _call_inference_api(contract_code, contract_address):                   │
  │    ├─ if MODULE1_MOCK=true        → _mock_prediction()                  │
  │    ├─ try: POST {MODULE1_INFERENCE_URL}/predict                        │
  │    │     body: {"source_code": contract_code}                          │
  │    │     (contract_address is metadata, NOT sent to Module 1)          │
  │    ├─ except TimeoutException       → _mock_prediction()                │
  │    ├─ except HTTPStatusError(4xx5xx) → RAISE (don't silently fall back)│
  │    └─ except RequestError           → _mock_prediction()                │
  │                                                                         │
  │  _mock_prediction():                                                     │
  │    Full 3-tier schema (label, probabilities, confirmed, suspicious,     │
  │    vulnerabilities, tier_thresholds, thresholds, truncated,              │
  │    windows_used, num_nodes, num_edges).                                  │
  │    Heuristic: "call.value" or "transfer(" in code → Reentrancy=0.72     │
  │                                                                         │
  │  ┌─────────────────────────────────────────────────────────────────┐   │
  │  │  Transport                                                       │   │
  │  │  SseServerTransport("/messages/")                                │   │
  │  │  Route("/sse",       handle_sse)                                 │   │
  │  │  Mount("/messages/", sse_transport.handle_post_message)          │   │
  │  │  Route("/health",    health → {status, server, mock_mode, ...})  │   │
  │  └─────────────────────────────────────────────────────────────────┘   │
  └─────────────────────────────────────────────────────────────────────────┘
```

### 2.2 sentinel-rag `:8011` (`rag_server.py`, 353 lines)

```
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  Listens:  http://0.0.0.0:8011                                          │
  │  Tool:     search                                                       │
  │                                                                         │
  │  ┌─────────────────────────────────────────────────────────────────┐   │
  │  │  Server instance                                                 │   │
  │  │  server = Server("sentinel-rag")                                 │   │
  │  │  Tool: search(query, k=5, filters)                               │   │
  │  │       filters: {vuln_type, date_gte, loss_gte, source, has_summary}│  │
  │  │  → _handle_search() → _retriever.search(query, k, filters)       │   │
  │  └─────────────────────────────────────────────────────────────────┘   │
  │                                                                         │
  │  ┌─────────────────────────────────────────────────────────────────┐   │
  │  │  _retriever: HybridRetriever | None (lazy)                       │   │
  │  │  Loaded in _on_startup() (NOT at module import — Bug 10 fix)     │   │
  │  │  FAISS + BM25 + chunks from agents/data/index/                   │   │
  │  │  search() is synchronous (CPU-bound, no I/O at this scale)       │   │
  │  └─────────────────────────────────────────────────────────────────┘   │
  │                                                                         │
  │  search() is synchronous and CPU-bound (<100ms on RTX 3070).            │
  │  If latency becomes a problem in M6, wrap with asyncio.run_in_executor. │
  │                                                                         │
  │  Returns per chunk:                                                     │
  │    {chunk_id, content, doc_id, chunk_index, total_chunks,                │
  │     metadata: {vuln_type, date, loss_usd, source, has_summary, ...},    │
  │     score: <RRF score>}                                                  │
  │                                                                         │
  │  Known Track 3 vuln_type values (11 classes):                            │
  │    Reentrancy, AccessControl, ArithmeticOverflow, UncheckedReturn,       │
  │    FrontRunning, OracleManipulation, FlashLoan, LogicError,              │
  │    DenialOfService, Phishing, Other                                     │
  │  (Note: this list is for the rag_server's filter description,           │
  │   and differs from the ML model's 10 classes — they don't align 1:1.)   │
  │                                                                         │
  │  Transport: Route("/sse") + Mount("/messages/") + Route("/health")      │
  │  /health returns {chunks_indexed, default_k}                            │
  └─────────────────────────────────────────────────────────────────────────┘
```

### 2.3 sentinel-audit `:8012` (`audit_server.py`, 717 lines)

```
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  Listens:  http://0.0.0.0:8012                                          │
  │  Tools:    get_latest_audit, get_audit_history, check_audit_exists      │
  │                                                                         │
  │  ┌─────────────────────────────────────────────────────────────────┐   │
  │  │  Server instance                                                 │   │
  │  │  server = Server("sentinel-audit")                               │   │
  │  │  Tools: 3 read-only on-chain queries                             │   │
  │  └─────────────────────────────────────────────────────────────────┘   │
  │                                                                         │
  │  ┌─────────────────────────────────────────────────────────────────┐   │
  │  │  Web3 client                                                     │   │
  │  │  Connects to Sepolia via SEPOLIA_RPC_URL                         │   │
  │  │  Contract: AuditRegistry proxy 0x14E5eFb6DE4cBb74896B45b4853      │   │
  │  │            fd14901E4CfAf                                         │   │
  │  │  ABI: loaded LAZILY in _on_startup() (Bug 2 fix)                 │   │
  │  │  _ABI = None at module level (no crash in CI / mock mode)        │   │
  │  └─────────────────────────────────────────────────────────────────┘   │
  │                                                                         │
  │  AuditRegistry.sol architecture:                                        │
  │  ┌─────────────────────────────────────────────────────────────────┐   │
  │  │  Deployed as UUPS proxy on Sepolia                               │   │
  │  │  Proxy addr: 0x14E5eFb6DE4cBb74896B45b4853fd14901E4CfAf          │   │
  │  │                                                                   │   │
  │  │  AuditResult struct:                                             │   │
  │  │    scoreFieldElement  uint256    (BN254 field, /8192 = score)    │   │
  │  │    proofHash          bytes32    (keccak256 of ZK proof)         │   │
  │  │    timestamp          uint256                                    │   │
  │  │    agent              address    (submitter's wallet)             │   │
  │  │    verified           bool       (ZK proof passed on-chain)      │   │
  │  │                                                                   │   │
  │  │  Score decoding:                                                 │   │
  │  │    score = scoreFieldElement / 8192.0                           │   │
  │  │    8192 = 2^13 = EZKL scale factor                              │   │
  │  │    Example: 4497 / 8192 = 0.5490 (vulnerable)                    │   │
  │  └─────────────────────────────────────────────────────────────────┘   │
  │                                                                         │
  │  Address validation:                                                     │
  │    _validate_address() → Web3.to_checksum_address() (EIP-55)            │
  │                                                                         │
  │  Mock mode:                                                             │
  │    Auto-enabled if SEPOLIA_RPC_URL is empty                             │
  │    Or AUDIT_MOCK=true                                                    │
  │    Returns realistic fake on-chain data (same shape as real)            │
  │                                                                         │
  │  Transport: Route("/sse") + Mount("/messages/") + Route("/health")      │
  │  /health → {status, server, mock_mode, registry_address}                │
  └─────────────────────────────────────────────────────────────────────────┘
```

### 2.4 sentinel-graph-inspector `:8013` (`graph_inspector_server.py`, 527 lines)

```
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  Listens:  http://0.0.0.0:8013                                          │
  │  Tool:     get_graph_hotspots                                           │
  │                                                                         │
  │  ┌─────────────────────────────────────────────────────────────────┐   │
  │  │  Server instance                                                 │   │
  │  │  server = Server("sentinel-graph-inspector")                     │   │
  │  │  Tool: get_graph_hotspots(contract_code, flagged_classes=[])     │   │
  │  └─────────────────────────────────────────────────────────────────┘   │
  │                                                                         │
  │  Phase 2 (current): real GNN attention scores from Module 1             │
  │  Phase 1 (fallback): Slither structural scoring                         │
  │                                                                         │
  │  ┌─────────────────────────────────────────────────────────────────┐   │
  │  │  Three-tier fallback chain                                       │   │
  │  │                                                                   │   │
  │  │  ┌───────────────────────────────────────────────────────────┐   │   │
  │  │  │  Tier 1: _analyze_hotspots_gnn()                         │   │   │
  │  │  │   POST {SENTINEL_ML_API_URL}/hotspots                    │   │   │
  │  │  │   body: {"source_code": contract_code}                   │   │   │
  │  │  │   → returns hotspots + hotspot_stats + ml verdict         │   │   │
  │  │  │   Higher L2 norm = GNN concentrated more signal           │   │   │
  │  │  │                                                           │   │   │
  │  │  │   on 200 OK   → analysis_mode = "gnn_attention"          │   │   │
  │  │  │   on non-200  → return None → fall to Tier 2             │   │   │
  │  │  │   on conn err → return None → fall to Tier 2             │   │   │
  │  │  └───────────────────────────────────────────────────────────┘   │   │
  │  │           │                                                        │   │
  │  │           ▼                                                        │   │
  │  │  ┌───────────────────────────────────────────────────────────┐   │   │
  │  │  │  Tier 2: _analyze_hotspots_slither() (Phase 1 logic)      │   │   │
  │  │  │   Runs Slither directly on temp file                       │   │   │
  │  │  │   Scopes detectors to CLASS_TO_DETECTORS[flagged_classes] │   │   │
  │  │  │   score = detector_hits × impact_weight                   │   │   │
  │  │  │         + external_call_count × 0.5                        │   │   │
  │  │  │         + state_write_count   × 0.3                        │   │   │
  │  │  │         + high_level_calls    × 0.4                        │   │   │
  │  │  │         + is_external_facing  × 0.2                        │   │   │
  │  │  │                                                           │   │   │
  │  │  │   on Slither OK    → analysis_mode = "slither"            │   │   │
  │  │  │   on Slither error → fall to Tier 3                       │   │   │
  │  │  └───────────────────────────────────────────────────────────┘   │   │
  │  │           │                                                        │   │
  │  │           ▼                                                        │   │
  │  │  ┌───────────────────────────────────────────────────────────┐   │   │
  │  │  │  Tier 3: mock data (deterministic stub)                    │   │   │
  │  │  │   Used when GRAPH_INSPECTOR_MOCK=true OR Slither missing   │   │   │
  │  │  │   analysis_mode = "mock"                                   │   │   │
  │  │  └───────────────────────────────────────────────────────────┘   │   │
  │  └─────────────────────────────────────────────────────────────────┘   │
  │                                                                         │
  │  Transport: Route("/sse") + Mount("/messages/") + Route("/health")      │
  └─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. MCP Protocol — Common Patterns

### 3.1 Server Skeleton (every server follows this)

```python
# 1. Configuration (all env-overridable)
_SERVER_PORT = int(os.getenv("MCP_<NAME>_PORT", "<default>"))

# 2. Server instance (MCP SDK identity — keep stable across deploys)
server = Server("sentinel-<name>")

# 3. Tool definitions
@server.list_tools()
async def list_tools() -> list[Tool]:
    return [Tool(name=..., description=..., inputSchema={...})]

# 4. Tool dispatcher
@server.call_tool()
async def call_tool(name, arguments) -> list[TextContent]:
    if name == "...": return await _handle_...(arguments)
    else: return [TextContent(text=json.dumps({"error": f"Unknown: {name}"}))]

# 5. Lifespan (for servers with HTTP/IO state)
@asynccontextmanager
async def lifespan(app):
    await _on_startup()      # create shared client / load retriever
    try: yield
    finally: await _on_shutdown()  # close client

# 6. Starlette app + uvicorn
starlette_app = Starlette(
    lifespan=lifespan,
    routes=[
        Route("/sse",       endpoint=handle_sse),
        Mount("/messages/", app=sse_transport.handle_post_message),
        Route("/health",    endpoint=health),
    ]
)
uvicorn.run(starlette_app, host="0.0.0.0", port=_SERVER_PORT)
```

### 3.2 Client Connection Pattern (used by graph nodes)

```python
from mcp.client.sse import sse_client
from mcp import ClientSession

# In a graph node:
async with sse_client(server_url) as (read, write):
    async with ClientSession(read, write) as session:
        await session.initialize()
        result = await session.call_tool(tool_name, arguments)
        # result.content is a list[TextContent] — first item's .text is JSON string
```

Connection-per-call is intentional M5 simplicity. Promotable to module-level
persistent clients in M6 if RTT measurements show it's a bottleneck.

### 3.3 Mock Mode Decision Table

| Server | Mock env var | Default | When fallback activates |
|--------|--------------|---------|-------------------------|
| `inference_server` | `MODULE1_MOCK` | `false` | explicit OR Timeout OR RequestError (NOT on HTTP 4xx/5xx — those RAISE) |
| `rag_server` | — | — | never (assumes index exists) |
| `audit_server` | `AUDIT_MOCK` | `true` if `SEPOLIA_RPC_URL` empty | always when in mock mode |
| `graph_inspector_server` | `GRAPH_INSPECTOR_MOCK` | `false` | explicit OR Slither unavailable |

---

## 4. Common Transport — SSE + JSON-RPC

```
  ┌────────────────────────────────────────────────────────────────────────┐
  │  SseServerTransport("/messages/")                                       │
  │                                                                        │
  │  GET  /sse                                                              │
  │  ├── Client opens persistent HTTP connection (event-stream)            │
  │  ├── Server pushes server→client messages (initialized, tools/list,    │
  │  │   tools/call results) as SSE events                                  │
  │  └── Client holds the connection open for the lifetime of the session  │
  │                                                                        │
  │  POST /messages/                                                        │
  │  ├── Client sends JSON-RPC requests (initialize, tools/list,           │
  │  │   tools/call) as regular HTTP POST                                  │
  │  ├── Server processes and pushes response back on the /sse stream      │
  │  └── No response body on POST — client reads from /sse                 │
  │                                                                        │
  │  GET  /health                                                           │
  │  ├── NOT an MCP endpoint — plain HTTP JSON                              │
  │  ├── Used by Docker / Prometheus / load balancers                      │
  │  └── Returns server-specific diagnostics (e.g. chunks_indexed for rag) │
  └────────────────────────────────────────────────────────────────────────┘
```

### Why SSE + not stdio?

- stdio: subprocess transport, used when MCP server is a child process
- SSE: HTTP transport, used when MCP server is a deployable service
- All 4 agents servers use SSE because they're deployable services
- stdio alternative documented in each file's docstring (replace `run_server()` body)

---

## 5. Server Lifecycles

### 5.1 `inference_server` (HTTP client state)

```
  server startup
       │
       ▼
  _on_startup()
       │ global _http_client = httpx.AsyncClient(timeout=MODULE1_TIMEOUT)
       │
       ▼
  serve requests (predict / batch_predict share _http_client)
       │
       ▼
  _on_shutdown()
       │ await _http_client.aclose()
```

### 5.2 `rag_server` (retriever state)

```
  server startup
       │
       ▼
  _on_startup()     (called explicitly in run_server() before Starlette starts)
       │ global _retriever = HybridRetriever()    ← loads FAISS + BM25 + chunks
       │
       ▼
  serve requests (search reads from _retriever — synchronous, CPU-bound)
```

Note: rag_server loads the retriever BEFORE Starlette starts (not via lifespan)
because the dependency on `agents/data/index/` should fail-fast if missing.

### 5.3 `audit_server` (Web3 + ABI state)

```
  module import (safe, no IO)
       │
       │  _ABI = None at module level (Bug 2 fix — no crash without compiled contracts)
       │
       ▼
  server startup
       │
       ▼
  _on_startup()
       │ if _MOCK_MODE: skip ABI load
       │ else: _ABI = _load_abi()    ← reads contracts/out/AuditRegistry.sol/AuditRegistry.json
       │ Web3 provider initialized in handlers (per-call, see below)
       │
       ▼
  serve requests
       │ handler creates Web3(...) per call, calls contract.functions.getLatestAudit(addr).call()
       │
       ▼
  no explicit shutdown — Web3 provider has no persistent connection
```

### 5.4 `graph_inspector_server` (per-call HTTP, no shared state)

```
  module import (no IO)
       │
       ▼
  server startup (no startup work — fallbacks don't need init)
       │
       ▼
  serve requests
       │ _analyze_hotspots_gnn() — fresh httpx.AsyncClient per call
       │ _analyze_hotspots_slither() — fresh Slither instance per call
       │ mock — pure Python, no IO
```

Unlike `inference_server`, this server does NOT share an httpx client — each
call creates a new one. Reason: fallback chain needs to try Tier 1, and a
broken client should not poison subsequent calls. (Trade-off: more TCP
handshakes, but lower risk of one bad call breaking the next.)

---

## 6. Tool Inventory

```
  Port  Server                  Tool                    Args                Returns
  ────  ──────────────────────  ──────────────────────  ──────────────────  ───────────────
  8010  sentinel-inference      predict                 contract_code       ml_result dict
                                                          (contract_address  (3-tier schema)
                                                           optional)

  8010  sentinel-inference      batch_predict           contracts (≤20)     {results: [...]}
                                                          [{contract_code,
                                                            contract_address}]

  8011  sentinel-rag            search                  query (str)         {query, k_*
                                                          k (int, ≤20)        requested,
                                                          filters (dict):    k_returned,
                                                            vuln_type,       filters_applied,
                                                            date_gte,        results: [...]}
                                                            loss_gte,
                                                            source,
                                                            has_summary

  8012  sentinel-audit          get_latest_audit        contract_address    AuditResult or
                                                                             null

  8012  sentinel-audit          get_audit_history       contract_address,   {count, records}
                                                          limit (≤50, def 10)

  8012  sentinel-audit          check_audit_exists      contract_address    {exists, count}

  8013  sentinel-graph-         get_graph_hotspots      contract_code,      {hotspots, graph_*
        inspector                                       flagged_classes     stats, analysis_*
                                                          (optional)         mode}
```

---

## 7. /health Endpoints (for Docker, Prometheus, monitoring)

```
  ┌──────────────┬────────────────────────────────────────────────────────────┐
  │ Server       │ Response                                                    │
  ├──────────────┼────────────────────────────────────────────────────────────┤
  │ inference    │ {status, server, mock_mode, module1_url}                  │
  │ rag          │ {status, server, chunks_indexed, default_k}                │
  │ audit        │ {status, server, mock_mode, registry_address}              │
  │ graph_       │ {status, server, mock_mode, ml_api_url, fallback}         │
  │  inspector   │   fallback = "gnn_attention" | "slither" | "mock"         │
  └──────────────┴────────────────────────────────────────────────────────────┘
```

All 4 `/health` endpoints are plain JSON (no MCP), suitable for Docker
healthchecks and Prometheus blackbox monitoring.

---

## 8. Configuration Surface

```
  ┌─── Module 1 / ML backends ─────────────┬────────────────────────────────┐
  │ MODULE1_INFERENCE_URL                  │ http://localhost:8001          │
  │ MODULE1_TIMEOUT                        │ 30.0 sec                       │
  │ MODULE1_MOCK                           │ false                          │
  │ SENTINEL_ML_API_URL                    │ http://localhost:8000          │
  │ GRAPH_INSPECTOR_HOTSPOTS_TIMEOUT       │ 60 sec                         │
  │ GRAPH_INSPECTOR_MOCK                   │ false                          │
  └────────────────────────────────────────┴────────────────────────────────┘
  ┌─── Server ports ───────────────────────┬────────────────────────────────┐
  │ MCP_INFERENCE_PORT                     │ 8010                           │
  │ MCP_RAG_PORT                           │ 8011                           │
  │ MCP_AUDIT_PORT                         │ 8012                           │
  │ MCP_GRAPH_INSPECTOR_PORT               │ 8013                           │
  └────────────────────────────────────────┴────────────────────────────────┘
  ┌─── Audit (Web3) ───────────────────────┬────────────────────────────────┐
  │ SEPOLIA_RPC_URL                        │ <alchemy/infura endpoint>      │
  │ AUDIT_REGISTRY_ADDRESS                 │ 0x14E5eFb6DE4cBb74896B45b4853  │
  │                                        │ fd14901E4CfAf                 │
  │ AUDIT_MOCK                             │ auto if RPC empty              │
  │ AUDIT_HISTORY_DEFAULT_LIMIT            │ 10                             │
  └────────────────────────────────────────┴────────────────────────────────┘
  ┌─── RAG ────────────────────────────────┬────────────────────────────────┐
  │ RAG_DEFAULT_K                          │ 5                              │
  │ RAG_MAX_K (hard cap)                   │ 20                             │
  └────────────────────────────────────────┴────────────────────────────────┘
```

---

## 9. Failure Modes

```
  ┌────────────────────────────────────────┬──────────────────────────────────────┐
  │ Failure                               │ Behaviour                             │
  ├────────────────────────────────────────┼──────────────────────────────────────┤
  │ Module 1 unreachable                  │ inference_server: TimeoutException    │
  │                                        │ or RequestError → falls back to      │
  │                                        │ _mock_prediction() (no error to      │
  │                                        │ client)                              │
  │                                        │                                      │
  │ Module 1 returns 4xx/5xx              │ inference_server: HTTPStatusError     │
  │                                        │ → RAISES → call_tool returns error  │
  │                                        │ TextContent to graph node.           │
  │                                        │ ml_assessment captures as state error│
  │                                        │                                      │
  │ RAG index missing                     │ rag_server: _on_startup raises       │
  │                                        │ RuntimeError → server fails to start │
  │                                        │ (fail-fast by design)                │
  │                                        │                                      │
  │ Sepolia RPC down                      │ audit_server: handler returns        │
  │                                        │ error dict to call_tool (does not    │
  │                                        │ auto-fall-back to mock unless        │
  │                                        │ _MOCK_MODE=true)                     │
  │                                        │                                      │
  │ Module 1 /hotspots unreachable        │ graph_inspector: _analyze_hotspots_  │
  │                                        │ gnn returns None → falls back to     │
  │                                        │ Slither analysis. analysis_mode=     │
  │                                        │ "slither" in result.                 │
  │                                        │                                      │
  │ Slither not installed                 │ graph_inspector: _analyze_hotspots_  │
  │                                        │ slither errors → falls back to       │
  │                                        │ mock. analysis_mode="mock".          │
  │                                        │                                      │
  │ Tool call before _on_startup          │ rag_server: returns error dict       │
  │ (e.g. in unit tests)                 │ {error: "retriever_not_initialised"} │
  │                                        │                                      │
  │ Client sends invalid tool name        │ Every server: dispatcher returns     │
  │                                        │ TextContent with {"error":           │
  │                                        │ "Unknown tool: <name>"}              │
  │                                        │                                      │
  │ Client sends args violating inputSchema│ MCP SDK enforces before handler runs │
  │                                        │ (JSON Schema validation)              │
  │                                        │                                      │
  │ batch_predict > 20 contracts          │ inference_server: handler returns     │
  │                                        │ {"error": "batch size exceeds max 20"}│
  └────────────────────────────────────────┴──────────────────────────────────────┘
```

---

## 10. File Map (with line counts and key functions)

```
  agents/src/mcp/servers/
  ├── inference_server.py         501 lines   :8010
  │   ├─ _on_startup() / _on_shutdown()       shared httpx client
  │   ├─ _call_inference_api()                HTTP bridge to Module 1
  │   ├─ _mock_prediction()                   3-tier schema, deterministic
  │   ├─ _handle_predict() / _handle_batch_predict()
  │   └─ run_server()                         Starlette + uvicorn
  │
  ├── rag_server.py               353 lines   :8011
  │   ├─ _on_startup()                        lazy-loads HybridRetriever (Bug 10 fix)
  │   ├─ _handle_search()                     retriever.search() sync call
  │   └─ run_server()                         calls _on_startup() before Starlette
  │
  ├── audit_server.py             717 lines   :8012
  │   ├─ _load_abi()                          lazy (Bug 2 fix)
  │   ├─ _validate_address()                  Web3.to_checksum_address
  │   ├─ get_latest_audit() / get_audit_history() / check_audit_exists()
  │   └─ mock helpers (auto-enabled if RPC missing)
  │
  ├── graph_inspector_server.py   544 lines   :8013
  │   ├─ _analyze_hotspots_gnn()              Tier 1: POST /hotspots
  │   ├─ _analyze_hotspots_slither()          Tier 2: Slither structural
  │   └─ mock data                            Tier 3: deterministic stub
  │
  └── __init__.py                 (empty)
```

---

## 11. Quick Reference — Source Code Locations

| Concept | File:Line |
|---------|-----------|
| `sentinel-inference` server identity | `agents/src/mcp/servers/inference_server.py:65` |
| `predict` tool inputSchema | `inference_server.py:80-105` |
| `batch_predict` tool inputSchema | `inference_server.py:107-137` |
| `_call_inference_api()` (HTTP bridge) | `inference_server.py:174-237` |
| `_mock_prediction()` (3-tier) | `inference_server.py:240-313` |
| Shared httpx client (lifespan) | `inference_server.py:147-167` |
| Mock mode decision table (inference) | `inference_server.py:218-237` |
| `sentinel-rag` server identity | `rag_server.py:118` |
| `search` tool inputSchema | `rag_server.py:124-195` |
| Lazy retriever init (Bug 10) | `rag_server.py:97-111` |
| `_handle_search()` | `rag_server.py:219-283` |
| `sentinel-audit` server identity | `audit_server.py` (search for `server = Server`) |
| Bug 2 fix — lazy ABI | `audit_server.py:116-...` |
| AuditRegistry struct (in docstring) | `audit_server.py:17-28` |
| Score decoding (8192 scale) | `audit_server.py:24-28` |
| `sentinel-graph-inspector` server identity | `graph_inspector_server.py:106` |
| `get_graph_hotspots` tool inputSchema | `graph_inspector_server.py:115-146` |
| `_analyze_hotspots_gnn()` (Tier 1) | `graph_inspector_server.py:153-232` |
| `_analyze_hotspots_slither()` (Tier 2) | `graph_inspector_server.py:253-...` |
| Phase 2 transition note | `graph_inspector_server.py:5-32` |

---

## 12. See Also

- `~/projects/sentinel/agents/DIAGRAM.md` — top-level module diagram
- `~/projects/sentinel/agents/src/orchestration/DIAGRAM.md` — ML integration
- `~/projects/sentinel/agents/src/mcp/README.md` — text companion
- `~/projects/sentinel/agents/scripts/` — smoke tests for each server
- `~/projects/sentinel/agents/tests/test_inference_server.py` — unit tests
