# M4 — Agents / MCP / RAG

LangGraph orchestration, three MCP servers, a hybrid RAG retriever over DeFi exploit history, and a continuous ingestion pipeline that keeps the knowledge base fresh.

---

## Overview

```
LangGraph StateGraph (graph.py)
  ml_assessment ──▶ inference MCP :8010 ──▶ M1 FastAPI :8001
       │
       ▼  max(prob) ≥ 0.70?
       ├── deep ──▶ rag_research ──▶ audit_check ──▶ synthesizer
       └── fast ──────────────────────────────────▶ synthesizer

RAG pipeline (HybridRetriever)
  FAISS semantic search  +  BM25 keyword search  →  RRF fusion  →  top-k chunks

Ingestion pipeline
  DeFiHackLabs GitHub  →  chunk  →  embed  →  atomic index write
  AuditRegistry events →  feed findings back into RAG (feedback_loop.py)
```

---

## LangGraph Orchestration

### `AuditState` fields

| Field | Type | Set by |
|-------|------|--------|
| `contract_code` | `str` | Caller |
| `contract_address` | `str` | Caller |
| `ml_result` | `dict \| None` | `ml_assessment` |
| `rag_results` | `list \| None` | `rag_research` |
| `audit_history` | `list \| None` | `audit_check` |
| `static_findings` | `dict \| None` | `static_analysis` (M6) |
| `final_report` | `dict \| None` | `synthesizer` |
| `error` | `str \| None` | Any node on failure |

### Risk routing

```python
def _is_high_risk(ml_result):
    vulns = ml_result.get("vulnerabilities", [])
    return max(v["probability"] for v in vulns) > 0.70 if vulns else False
# True  → deep path: rag_research → audit_check → synthesizer
# False → fast path: synthesizer directly
```

0.70 is intentionally higher than the per-class inference threshold (0.50): deep analysis only for high-confidence detections.

### Final report schema

```json
{
  "overall_label": "vulnerable",
  "risk_probability": 0.8943,
  "top_vulnerability": "Reentrancy",
  "vulnerabilities": [{"vulnerability_class": "Reentrancy", "probability": 0.8943}],
  "rag_evidence": [...],
  "audit_history": [...]
}
```

There is **no** `confidence` field anywhere in this schema (removed in Track 3).

### Usage

```python
import asyncio
from src.orchestration.graph import build_graph

graph = build_graph()
result = asyncio.run(graph.ainvoke(
    {"contract_code": "<solidity>", "contract_address": "0x..."},
    config={"configurable": {"thread_id": "audit-001"}},
))
print(result["final_report"])
```

Resume after crash (same `thread_id`, pass `None`):

```python
result = asyncio.run(graph.ainvoke(
    None,
    config={"configurable": {"thread_id": "audit-001"}},
))
```

---

## MCP Servers

All three servers use **SSE transport** (HTTP, not stdio) — production-safe, multi-client, Docker-compatible.

Start command pattern:
```bash
cd agents
poetry run python -m src.mcp.servers.<server_module>
```

### inference_server.py — port 8010

Wraps the M1 FastAPI `/predict` endpoint.

**Tools:**

| Tool | Inputs | Returns |
|------|--------|---------|
| `predict(contract_code, contract_address?)` | Solidity source | Track 3 `vulnerabilities[]` schema |
| `batch_predict(contracts, max=20)` | List of `{contract_code, contract_address}` | List of results |

Field mapping: MCP accepts `contract_code`, forwards as `source_code` to FastAPI (these are different names by design).

Mock fallback: enabled when M1 is unreachable. Set `MODULE1_MOCK=true` to force in tests.

### rag_server.py — port 8011

Wraps `HybridRetriever.search()` over the DeFiHackLabs RAG index.

**Tool: `search(query, k=5, filters={})`**

Filter keys (all optional):

| Key | Type | Example |
|-----|------|---------|
| `vuln_type` | string | `"Reentrancy"` |
| `date_gte` | string | `"2023-01-01"` |
| `loss_gte` | number | `1000000` |
| `source` | string | `"DeFiHackLabs"` |
| `has_summary` | boolean | `true` |

Response shape per result:
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

The retriever is instantiated once at module level (not per-request) — ~400 ms load, ~5 MB RAM.

### audit_server.py — port 8012

Reads `AuditRegistry` on Sepolia. Mock mode auto-enabled when `SEPOLIA_RPC_URL` is absent.

**Tools:**

| Tool | Returns |
|------|---------|
| `get_latest_audit(contract_address)` | `{score, label, proof_hash, timestamp, agent}` |
| `get_audit_history(contract_address, limit=10)` | List of `AuditResult`, newest first |
| `check_audit_exists(contract_address)` | `{exists: bool, count: int}` |

Score decoding: `score = field_element / 8192` (EZKL scale factor).
`submit_audit` is deferred until ZKML + Track 3 proof semantics are finalised.

---

## RAG Pipeline

### Knowledge base

| Item | Value |
|------|-------|
| Source | DeFiHackLabs GitHub (`src/test/` and `past/` directories) |
| `.sol` files | 726 |
| Chunks | 752 |
| Chunk size | 1 536 chars, overlap 128 |
| Embedding model | `text-embedding-nomic-embed-text-v1.5` via LM Studio |
| Vector index | FAISS `IndexFlatL2`, 768-dim |
| Keyword index | `BM25Okapi` |
| Fusion | Reciprocal Rank Fusion, k=60 |

### Retrieval algorithm

```
FAISS: top-20 by L2 similarity
BM25:  top-20 by BM25Okapi score
RRF:   score[doc] = 1/(60 + rank_faiss) + 1/(60 + rank_bm25)
Post-filter → return top-k
```

### Building / rebuilding the index

```bash
cd agents
poetry run python -m src.rag.build_index
```

Write safety: `FileLock` + atomic `os.replace()` (temp file → real file) + rollback snapshot.
Do not change `chunk_size` or `embedding model` without a full rebuild.

---

## Ingestion Pipeline

### Daily scheduled ingestion (Dagster)

```bash
cd agents
DAGSTER_HOME=agents/.dagster \
poetry run dagster dev -f src/ingestion/scheduler_dagster.py
# → http://localhost:3000  (Dagster UI)
```

Asset: `rag_index` — full pipeline: DeFiHackLabs fetch → deduplicate → chunk → embed → atomic index write.
Schedule: `daily_ingestion_schedule` (cron `0 2 * * *`).

### Feedback loop

`src/ingestion/feedback_loop.py` polls `AuditRegistry` for new `AuditSubmitted` events, reads the corresponding final report from `agents/data/reports/{contract_address}.json` (written by `synthesizer`), and ingests the findings back into the RAG index.

This closes the loop: on-chain verified findings inform future audits.

---

## LLM Client

`src/llm/client.py` routes requests to LM Studio (OpenAI-compatible API).

```
LM_STUDIO_BASE_URL   default: http://localhost:1234/v1
LM_STUDIO_TIMEOUT    default: 120 s
```

Models are configurable via env vars:

| Env var | Default use |
|---------|------------|
| `AGENT_MODEL_FAST` | Fast reasoning (ml_assessment) |
| `AGENT_MODEL_STRONG` | Deep analysis (synthesizer) |
| `AGENT_MODEL_CODER` | Static analysis context |
| `AGENT_MODEL_EMBED` | RAG embeddings (nomic-embed-text) |

On WSL2 the Windows host gateway IP changes on reboot. Set `LM_STUDIO_BASE_URL` explicitly rather than relying on the default.

---

## Environment Variables

```bash
MCP_INFERENCE_URL=http://localhost:8010/sse
MCP_RAG_URL=http://localhost:8011/sse
MCP_AUDIT_URL=http://localhost:8012/sse
AUDIT_RAG_K=5                      # RAG chunks retrieved for deep-path analysis
MODULE1_MOCK=false                 # Force mock inference (tests/dev)
AUDIT_MOCK=false                   # Force mock audit history (tests/dev)
SEPOLIA_RPC_URL=<your-rpc>
LM_STUDIO_BASE_URL=http://localhost:1234/v1
DAGSTER_HOME=agents/.dagster
```

---

## Testing

```bash
cd agents
poetry run pytest tests/ -v
```

| Test file | Coverage |
|-----------|---------|
| `test_graph_routing.py` | Routing logic, all node paths, graph compilation (41 tests) |
| `test_inference_server.py` | MCP tool schemas, mock/live transport |
| `test_audit_server.py` | On-chain history decoding, mock mode |
| `test_github_fetcher.py` | DeFiHackLabs parsing (3 comment formats) |
| `test_retriever_filters.py` | FAISS+BM25+RRF filter behaviour |
| `test_deduplicator.py` | SHA256 hash deduplication |
| `test_chunker.py` | Chunk size and overlap |

### Smoke scripts

```bash
poetry run python scripts/smoke_langgraph.py          # mock — no services needed
poetry run python scripts/smoke_langgraph.py --live   # live — all services must be up
poetry run python scripts/smoke_inference_mcp.py
poetry run python scripts/smoke_rag_mcp.py
poetry run python scripts/smoke_audit_mcp.py
```

Loguru note: `capsys`/`caplog` do not capture loguru output. Tests that assert on log lines add a temporary list sink.

---

## File Reference

```
agents/src/
  orchestration/
    state.py               AuditState TypedDict
    nodes.py               Node functions (ml_assessment, rag_research, audit_check, synthesizer)
    graph.py               StateGraph, conditional routing, MemorySaver
  mcp/servers/
    inference_server.py    Port 8010 — predict, batch_predict
    rag_server.py          Port 8011 — search
    audit_server.py        Port 8012 — get_audit_history, get_latest_audit, check_audit_exists
  rag/
    retriever.py           HybridRetriever: FAISS + BM25 + RRF
    chunker.py             Text chunking (1536 chars, overlap 128)
    embedder.py            Embedding via LM Studio nomic-embed-text
    build_index.py         Full index rebuild with rollback
    fetchers/
      github_fetcher.py    DeFiHackLabs fetcher (3 comment format parsers)
      base_fetcher.py      Abstract base
  ingestion/
    pipeline.py            Incremental update pipeline + REPORTS_DIR bridge
    deduplicator.py        SHA256 seen_hashes.json deduplication
    feedback_loop.py       AuditRegistry event polling, exp backoff
    scheduler_cron.py      Cron scheduling
    scheduler_dagster.py   Dagster asset + daily schedule
  llm/
    client.py              LM Studio client, AGENT_MODEL_MAP

agents/data/
  index/
    faiss.index            FAISS IndexFlatL2, 752 × 768-dim
    bm25.pkl               BM25Okapi model
    chunks.pkl             752 Chunk dataclass instances
    index_metadata.json    Build ID, config hash, artifact checksums
    seen_hashes.json       726 source file hashes (deduplication)
  reports/                 Final audit report JSON per contract_address
  feedback_state.json      Last processed Sepolia block number
```

---

## Do Not Change Without Wider Plan

- Do not re-add `confidence` to any schema or routing condition.
- Do not change the MCP `contract_code` / `source_code` field split casually.
- Do not change `RAG_MAX_K` (cap inside `rag_server.py`) without considering synthesizer context window.
- Do not change `chunk_size` or `chunk_overlap` without rebuilding the index.
- Do not wire `static_analysis` node without adding `static_findings` state field, error handling, and tests.
- Do not use mock-mode audit results as real security evidence.
