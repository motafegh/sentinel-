# SENTINEL — Module 4: Agents / MCP / RAG

Load for: LangGraph orchestration, MCP servers, RAG retriever, LLM model map.
Always load alongside: **SENTINEL-CONSTRAINTS.md**

---

## Tech Stack

| Tool | Version | Role |
|------|---------|------|
| LangChain ^0.3 | | Agent framework base |
| LangGraph ^0.2 | | State machine orchestration |
| langchain-openai ^0.2 | | OpenAI-compatible → LM Studio |
| mcp 1.27.0 | | MCP protocol SDK — enforces inputSchema at protocol level |
| httpx ^0.27 | | Async HTTP for ML API calls |
| faiss-cpu ^1.8 | | Vector similarity search |
| rank-bm25 ^0.2 | | BM25Okapi keyword search |
| pydantic ^2.0 | | Structured outputs |
| filelock ^3.13 | | Single-writer lock on index writes |
| web3 ^7.15 | | Ethereum RPC (feedback loop + audit MCP) |
| dagster 1.12.22 | | RAG ingestion scheduling |
| loguru ^0.7 | | Structured logging |
| python-dotenv ^1.0 | | .env loading |
| sentence-transformers | | Cross-encoder reranking (optional; T3-B) |

---

## LLM Model Map

```
LM Studio serves models locally (port 1234, Windows host).
IP: WSL2 gateway — changes on reboot; use env var or dynamic detection.

MODEL_FAST   = "gemma-4-e2b-it"            — fast responses, ML MCP calls only
MODEL_STRONG = "qwen3.5-9b-ud"             — reasoning, RAG synthesis, report generation
MODEL_CODER  = "qwen2.5-coder-7b-instruct" — Solidity analysis, code logic review

AGENT_MODEL_MAP = {
  "static_analyzer": MODEL_CODER,    # reads Solidity structure
  "ml_intelligence": MODEL_FAST,     # calls Module 1 API only
  "rag_researcher":  MODEL_STRONG,   # reasons over text descriptions
  "code_logic":      MODEL_CODER,    # understands Solidity logic
  "synthesizer":     MODEL_STRONG,   # generates structured report (T3-A)
}

Convenience functions:
  get_fast_llm()   → get_llm(model=MODEL_FAST, temperature=0.0)
  get_strong_llm() → get_llm(model=MODEL_STRONG, temperature=0.0)
  get_coder_llm()  → get_llm(model=MODEL_CODER, temperature=0.0)
```

---

## LangGraph Orchestration

```
Files:
  agents/src/orchestration/state.py    AuditState TypedDict
  agents/src/orchestration/nodes.py    Node functions
  agents/src/orchestration/graph.py    StateGraph + conditional routing

AuditState TypedDict:
  contract_code:    str
  contract_address: str
  ml_result:        dict | None
  rag_results:      list | None
  audit_history:    list | None
  static_findings:  list[dict] | None    # set by static_analysis (Slither findings)
  final_report:     dict | None
  error:            str | None
  narrative:        str | None           # T3-A: LLM-generated markdown narrative

Graph nodes:
  ml_assessment    → calls predict tool via MultiServerMCPClient
  rag_research     → calls search (query built from top vulnerability class)
  audit_check      → calls get_audit_history
  static_analysis  → Slither direct call (not via MCP); returns list[dict] per finding
  synthesizer      → LLM-based (qwen3.5-9b-ud) with rule-based fallback (T3-A complete)
                     structured prompt → markdown with severity/exploit/fix
                     falls back to rule-based when LM Studio unavailable
                     sets narrative: str | None in AuditState

Conditional routing:
  After ml_assessment:
    _is_high_risk() == True  → parallel fan-out: [rag_research ‖ static_analysis]
                                → audit_check (fan-in, waits for both) → synthesizer
    _is_high_risk() == False → synthesizer (fast path, static_findings=[])

  _route_after_ml returns list[str] for deep path (LangGraph parallel execution)
  or str for fast path. No path_map — function returns node name(s) directly.

  _is_high_risk(ml_result):
    vulns = ml_result.get("vulnerabilities", [])
    return max(v["probability"] for v in vulns) >= 0.70 if vulns else False

Final report schema:
  overall_label, risk_probability, top_vulnerability,
  vulnerabilities[], rag_evidence[], audit_history[], static_findings[], narrative
  — NO confidence field anywhere in this schema

MultiServerMCPClient config:
  "sentinel-inference": {"url": "http://localhost:8010/sse", "timeout": 120.0}
  "sentinel-rag":       {"url": "http://localhost:8011/sse", "timeout": 30.0}
  "sentinel-audit":     {"url": "http://localhost:8012/sse", "timeout": 30.0}
```

---

## MCP Server Patterns

```
All three servers share the same wiring:
  SSE transport (not stdio — production/Docker compatible)
  load_dotenv(override=True) at top of file
  Module-level client/retriever init (fail-fast, not per-request)
  Broad except Exception in handlers (unhandled exception closes SSE session)
  Full traceback logged via logger.exception()

inference_server.py (port 8010):
  Tool: predict(contract_code, contract_address?)
    → POST localhost:8001/predict {"source_code": contract_code}
    → TimeoutException → fall back to mock (transient)
    → HTTPStatusError  → re-raise (4xx = payload bug, not transient)
    → RequestError     → fall back to mock (M1 not running)
  Tool: batch_predict(contracts, max 20)
    → sequential loop (GPU-bound)
    → partial failure: HTTPStatusError per contract recorded, batch continues

rag_server.py (port 8011):
  Tool: search(query, k=5, filters={})
  Filters (additionalProperties: False):
    vuln_type, date_gte, loss_gte, source, has_summary
  Returns: {query, k_requested, k_returned, filters_applied, results[]}
  Result shape: {chunk_id, content, doc_id, chunk_index, total_chunks, metadata{}, score}
  score field: populated from RRF ranking (or cross-encoder score when rerank=True)

audit_server.py (port 8012):
  AsyncWeb3 client (one client reused)
  Mock mode: auto-enabled when SEPOLIA_RPC_URL is missing (AUDIT_MOCK=true also forces it)
  ABI lazy-load: fixed (2026-04-29) — ABI loaded only in real mode, not at import time
  Score decoding: score = field_element / 8192  (EZKL scale factor)
  Tools:
    get_latest_audit(contract_address) → {score, label, proof_hash, timestamp, agent}
    get_audit_history(contract_address, limit=10) → list of AuditResult, newest first
    check_audit_exists(contract_address) → {exists: bool, count: int}
    submit_audit — deferred (requires valid ZK proof + MIN_STAKE tokens staked)
```

---

## RAG Pipeline Technical Facts

```
Source:              DeFiHackLabs GitHub repo
Data:                726 .sol files (src/test/ AND past/ directories)
Index:               752 chunks × 768-dim float32
Chunk size:          1536 chars, overlap 128
Embedding model:     text-embedding-nomic-embed-text-v1.5
Index schema ver:    rag_index_v2
Build ID:            20260424T185700Z-6ca204aa

Retrieval algorithm:
  FAISS: similarity_search with k=20 candidates
  BM25:  BM25Okapi.get_scores, top-20 candidates
  RRF:   score[doc] = 1/(k+rank_faiss) + 1/(k+rank_bm25), k=60
  Optional rerank: CrossEncoder (off by default) — pass rerank=True to search()
  Post-filter → top-k returned

HybridRetriever: instantiated at module level (import time), not per-request
  — startup validation: checks FAISS vector count matches chunks length; fails fast if corrupt
  — single RAM instance shared across all tool calls
  — ~400ms load, ~5MB

Ingestion pipeline safety:
  FileLock(timeout=300) on .index.lock
  Atomic writes: write to .tmp then os.replace() (POSIX rename)
  Rollback: snapshot to backups/<build_id>/ before writing

Rebuild command:
  cd ~/projects/sentinel/agents && poetry run python -m src.rag.build_index
```

---

## File Inventory

```
agents/src/
  ingestion/
    pipeline.py              DeFiHackLabsFetcher → Chunker → Embedder → index write
    deduplicator.py          SHA256 seen_hashes.json deduplication
    feedback_loop.py         Polls AuditRegistry, 1999-block batch chunks, exp backoff
    scheduler_cron.py        Cron-based scheduling
    scheduler_dagster.py     Dagster asset + daily schedule

  llm/
    client.py                LM Studio client; MODEL_FAST/STRONG/CODER; AGENT_MODEL_MAP;
                             get_fast_llm(), get_strong_llm(), get_coder_llm()

  mcp/servers/
    inference_server.py      Port 8010 — wraps ML API /predict
    rag_server.py            Port 8011 — wraps HybridRetriever
    audit_server.py          Port 8012 — reads AuditRegistry on Sepolia

  orchestration/
    state.py                 AuditState TypedDict
    nodes.py                 Node functions — ml_assessment, rag_research,
                             audit_check, static_analysis, synthesizer
    graph.py                 StateGraph definition + conditional routing

  rag/
    build_index.py           Full rebuild: lock + atomic writes + rollback snapshot
    chunker.py               chunk_size=1536, overlap=128; Chunk dataclass with score: float = 0.0
    embedder.py              nomic-embed-text-v1.5 via direct OpenAI client
    retriever.py             HybridRetriever: FAISS + BM25 + RRF (k=60)
                             search(query, k, filters, rerank=False)
                             rerank=True: CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
                             results constructed via explicit Chunk(..., score=rrf_scores[i]) ctor
                             for backward compat with old pickled chunks lacking score in __dict__
    fetchers/
      base_fetcher.py        Abstract base
      github_fetcher.py      DeFiHackLabsFetcher — scans src/test/ AND past/

agents/scripts/
  smoke_inference_mcp.py
  smoke_rag_mcp.py
  smoke_audit_mcp.py
  smoke_langgraph.py
  test_k_cap.py

agents/tests/
  test_inference_server.py
  test_rag_server.py
  test_audit_server.py
  test_chunker.py
  test_deduplicator.py
  test_github_fetcher.py
  test_retriever_filters.py   incl. TestSearchScores (score > 0, descending order)
  test_graph_routing.py

agents/data/
  index/
    faiss.index              FAISS IndexFlatL2, 752 vectors × 768-dim
    bm25.pkl                 BM25Okapi over 752 chunks
    chunks.pkl               752 Chunk dataclass instances
    index_metadata.json      Build ID, config hash, artifact SHA256s
    seen_hashes.json         726 source file hashes for deduplication
    .index.lock              FileLock — prevents concurrent writes
    backups/<build_id>/      Rollback snapshots
  feedback_state.json        Last processed Sepolia block number
```

---

## Test Patterns (Non-Obvious)

```
Loguru capture: add temporary sink (list) — capsys/caplog don't capture loguru
MagicMock pickling: use plain dict for Chunk objects — MagicMock raises PicklingError
Module-level retriever: mock HybridRetriever before import to avoid index load
pytest config: agents/pyproject.toml [tool.pytest.ini_options] addopts=""
               overrides root sentinel/pyproject.toml --cov flags
```
