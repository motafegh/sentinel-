# MCP Servers — Per-Server Reference

Five MCP SSE servers that expose SENTINEL capabilities as tools. All servers are
independently runnable and mock-capable for CI.

## `inference_server.py` — `:8010`

Wraps Module 1's FastAPI inference server.

**Tools:**

| Tool | Required | Optional | Returns |
|------|----------|----------|---------|
| `predict` | `contract_code: str` | `contract_address: str` | Track 3 `PredictResponse` |
| `batch_predict` | `contracts: list` | — | `{"results": [...]}` (max 20) |

**Mock fallback:** When `MODULE1_MOCK=true` or Module 1 is unreachable, returns realistic
fake predictions matching the Track 3 schema (three-tier: `confirmed`, `suspicious`,
`vulnerabilities`).

**Shared HTTP client:** A single `httpx.AsyncClient` is created at server startup and
reused across all tool calls. Avoids per-call TCP+TLS handshake overhead.

**Configuration (`.env`):**
```bash
MODULE1_INFERENCE_URL=http://localhost:8001
MCP_INFERENCE_PORT=8010
MODULE1_TIMEOUT=30.0
MODULE1_MOCK=false
```

---

## `rag_server.py` — `:8011`

Wraps `HybridRetriever.search()` from the RAG module.

**Tool: `search`**

| Parameter | Type | Default | Max |
|-----------|------|---------|-----|
| `query` | `str` | (required) | — |
| `k` | `int` | 5 | 20 |
| `filters` | `dict` | `{}` | — |

**Response shape per result:**
```json
{
    "chunk_id": "abc123-0",
    "content": "...",
    "score": 0.842,
    "metadata": {"vuln_type": "Reentrancy", "date": "2023-03-15"}
}
```

**Lazy initialization:** `HybridRetriever()` is loaded in `_on_startup()` — not at
import time. Prevents crashes in CI where the index doesn't exist.

**Configuration (`.env`):**
```bash
MCP_RAG_PORT=8011
RAG_DEFAULT_K=5
```

---

## `audit/` package — `:8012`

Was the monolithic `audit_server.py` (717 LOC). Split into a package in P2.5
(2026-06-25) per Rule A (single responsibility).

**Package layout:**

| File | Purpose |
|------|---------|
| `_config.py` | Environment config and constants |
| `_decode.py` | Field-element → float score decoding, label logic |
| `_handlers.py` | Tool handler implementations |
| `_lifecycle.py` | Server startup/shutdown (ABI loading, Web3 init) |
| `_server.py` | FastAPI + MCP app assembly |
| `__init__.py` | Public re-export: `app` |

Reads `AuditRegistry.sol` on Sepolia via Web3.py. Query-only in current phase —
`submitAudit` is deferred until ZKML proof semantics are finalised.

**Tools:**

| Tool | Required | Optional | Returns |
|------|----------|----------|---------|
| `get_latest_audit` | `contract_address: str` | — | `{score, label, proof_hash, timestamp, agent, verified}` |
| `get_audit_history` | `contract_address: str` | `limit: int` (default 10, max 50) | `{count, records: [...]}` |
| `check_audit_exists` | `contract_address: str` | — | `{exists: bool, count: int}` |

**Score decoding:** `score = scoreFieldElement / 8192` (EZKL scale factor). Example:
`4497 / 8192 = 0.5490`.

**Mock mode:** Auto-enabled when `SEPOLIA_RPC_URL` is absent or `AUDIT_MOCK=true`.

**Configuration (`.env`):**
```bash
MCP_AUDIT_PORT=8012
SEPOLIA_RPC_URL=<your-rpc>
AUDIT_REGISTRY_ADDRESS=0x14E5eFb6DE4cBb74896B45b4853fd14901E4CfAf
AUDIT_MOCK=false
```

---

## `graph_inspector_server.py` — `:8013`

Function-level hotspot attribution — tells the auditor *where* in the contract the
suspicious pattern lives.

**Tool: `get_graph_hotspots`**

| Parameter | Type | Required |
|-----------|------|----------|
| `contract_code` | `str` | Yes |
| `flagged_classes` | `list[str]` | No (default: all flagged) |

**Three-tier fallback:**

| Priority | Backend | Signal |
|----------|---------|--------|
| 1 | ML API `/hotspots` | Real GNN embedding-norm scores |
| 2 | Slither analysis | Structural proxy scoring |
| 3 | Mock data | Deterministic stub |

**GNN hotspots:** Calls `POST /hotspots` on ML inference API. Returns L2 norm of each
function-level GNN node embedding — higher norm = more message-passing signal
concentrated there.

**Slither fallback scoring:**
- Detector hits × impact weight (High=3.0, Medium=2.0, Low=1.0)
- External call count × 0.5
- State write count × 0.3

**Configuration (`.env`):**
```bash
MCP_GRAPH_INSPECTOR_PORT=8013
SENTINEL_ML_API_URL=http://localhost:8001
GRAPH_INSPECTOR_HOTSPOTS_TIMEOUT=60
GRAPH_INSPECTOR_MOCK=false
```

---

## `representation_server.py` — `:8014`

Serves raw GNN node embedding vectors for the `explainer` node's LIME-style attribution.
Allows the explainer to compute which contract nodes contributed most to the model's
decision without re-running the full model.

**Tool: `get_embeddings`**

| Parameter | Type | Required |
|-----------|------|----------|
| `contract_code` | `str` | Yes |
| `node_ids` | `list[int]` | No (default: all) |

**Configuration (`.env`):**
```bash
MCP_REPRESENTATION_PORT=8014
REPRESENTATION_MOCK=false
```

---

## Starting All Servers

```bash
cd agents

poetry run python -m src.mcp.servers.inference_server    # :8010
poetry run python -m src.mcp.servers.rag_server          # :8011
poetry run python -m src.mcp.servers.audit_server        # :8012
poetry run python -m src.mcp.servers.graph_inspector_server  # :8013
poetry run python -m src.mcp.servers.representation_server   # :8014
```
