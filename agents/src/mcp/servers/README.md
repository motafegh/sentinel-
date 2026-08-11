# MCP Servers — Per-Server Reference

Five MCP SSE servers expose SENTINEL capabilities as tools. Each server is independently runnable; mock behavior is explicit rather than silently substituted for failed live dependencies.

## `inference_server.py` — `:8010`

Wraps Module 1's FastAPI inference server.

**Tools:**

| Tool | Required | Optional | Returns |
|------|----------|----------|---------|
| `predict` | `contract_code: str` | `contract_address: str` | Track 3 `PredictResponse` |
| `batch_predict` | `contracts: list` | — | `{"results": [...]}` (max 20) |

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

`HybridRetriever()` is loaded during startup rather than at import time so a missing index cannot masquerade as a clean result.

**Configuration (`.env`):**
```bash
MCP_RAG_PORT=8011
RAG_DEFAULT_K=5
```

---

## `audit/` package — `:8012`

The audit service is a **read-only registry observer**. Runtime signing and submission do not live in this MCP security domain.

**Live package layout:**

| File | Purpose |
|------|---------|
| `_config.py` | Environment/config + mutable runtime dependency state |
| `_decode.py` | Historical V1 tuple decoding helpers |
| `_versioned_reads.py` | V1/V2/V3 record decoding and merged read semantics |
| `_readonly_handlers.py` | Live MCP tool declarations + fail-closed read-only dispatcher |
| `_handlers.py` | Historical compatibility handlers; not the live MCP dispatcher |
| `_submit.py` | Historical R0/V2 compatibility machinery; unreachable from live MCP dispatch |
| `_lifecycle.py` | ABI/Web3 startup and shutdown |
| `_server.py` | Starlette/SSE transport |
| `audit_server.py` | Public compatibility shim exporting the live read-only server |

### Live tools

| Tool | Required | Optional | Semantics |
|------|----------|----------|-----------|
| `get_latest_audit` | `contract_address: str` | — | Newest persisted record across V3/V2/V1 by on-chain timestamp |
| `get_audit_history` | `contract_address: str` | `limit: int` (default 10, max 50) | Merged V3/V2/V1 history, newest first |
| `check_audit_exists` | `contract_address: str` | — | Aggregate existence/count plus `counts_by_protocol` |

Every returned audit record identifies `protocol_version` explicitly.

- **V1** retains its historical scalar `score_field_element` and legacy decoded score/label fields.
- **V2** returns the ten raw `class_score_felts`, proof/model identities, and explicitly records `proof_scope=legacy_proxy_only_unbound`.
- **V3** returns the ten raw class-score field elements plus request digest, public-signal hash, target code hash, teacher/proxy/DATA/schema identities, round, policy signer, verifier, agent and verification state.

The MCP does **not** invent a scalar SAFE/VULNERABLE verdict for V2/V3 persistence records, and it does not blindly reinterpret raw field elements as probabilities. Interpretation belongs to the exact model/artifact policy layer.

### Write containment

`submit_audit` is not advertised and is not reachable through the live dispatcher. Any write-like tool name is rejected with a structured `policy_rejected` result and `attempted=false` before the historical `_submit.py` module is imported.

V3 signing belongs to the isolated policy-signer domain defined by `agents/src/security/policy_signer.py`, not to the analysis MCP process.

### Failure / mock semantics

Missing RPC, ABI, startup failure, or incompatible registry reads remain explicitly `UNAVAILABLE`; the service does **not** silently switch to mock evidence.

Mock mode is enabled only when explicitly requested, e.g.:

```bash
AUDIT_MOCK=true
```

**Configuration:**
```bash
MCP_AUDIT_PORT=8012
SEPOLIA_RPC_URL=<your-rpc>
AUDIT_REGISTRY_ADDRESS=<registry-proxy-address>
AUDIT_MOCK=false
```

---

## `graph_inspector_server.py` — `:8013`

Function-level hotspot attribution indicates where suspicious model/static-analysis signal concentrates.

**Tool: `get_graph_hotspots`**

| Parameter | Type | Required |
|-----------|------|----------|
| `contract_code` | `str` | Yes |
| `flagged_classes` | `list[str]` | No |

The server records whether real ML/static-analysis evidence ran; unavailable dependencies must not be serialized as a clean empty result.

**Configuration:**
```bash
MCP_GRAPH_INSPECTOR_PORT=8013
SENTINEL_ML_API_URL=http://localhost:8001
GRAPH_INSPECTOR_HOTSPOTS_TIMEOUT=60
GRAPH_INSPECTOR_MOCK=false
```

---

## `representation_server.py` — `:8014`

Serves GNN node embeddings used by downstream explanation tooling.

**Tool: `get_embeddings`**

| Parameter | Type | Required |
|-----------|------|----------|
| `contract_code` | `str` | Yes |
| `node_ids` | `list[int]` | No |

**Configuration:**
```bash
MCP_REPRESENTATION_PORT=8014
REPRESENTATION_MOCK=false
```

---

## Starting the servers

```bash
cd agents

poetry run python -m src.mcp.servers.inference_server          # :8010
poetry run python -m src.mcp.servers.rag_server                # :8011
poetry run python -m src.mcp.servers.audit_server              # :8012
poetry run python -m src.mcp.servers.graph_inspector_server    # :8013
poetry run python -m src.mcp.servers.representation_server     # :8014
```

A server being alive is not equivalent to its dependency being ready. Use each server's readiness endpoint/status fields when deciding whether evidence is live, mocked, or unavailable.
