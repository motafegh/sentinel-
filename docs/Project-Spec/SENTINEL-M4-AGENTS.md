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
    submit_audit — deferred; requires a valid ZK proof generated by a completed EZKL
                   pipeline (Groth16/BN254, produced by zkml/run_proof.py) AND
                   stakedBalance(msg.sender) >= MIN_STAKE (1000 SENTINEL tokens)
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

---

## Upgrade Proposals (v2)

### AGENT-1: Multi-model consensus for vulnerability classification

**Problem:** The single LLM synthesizer (qwen3.5-9b-ud) can hallucinate or produce
inconsistent classifications, with no internal check on output quality.

**Solution:** Run the synthesizer against three models (MODEL_STRONG + MODEL_CODER + a
third reasoning model); aggregate via majority vote on vulnerability_class and severity;
flag disagreements as "low confidence".

**Implementation:**
- New file: `agents/src/orchestration/consensus.py`
- New dataclass: `ConsensusResult{agreement_score, majority_verdict, dissenting_models[]}`
- Routing: apply consensus only when `max_probability >= 0.70` and `< 0.90` (high-risk
  but uncertain boundary zone); skip for clear-cut scores outside that range
- Cost: 3x LLM calls per consensus invocation; mitigate with result caching keyed on
  SHA256(contract_code)

**AuditState additions:** `consensus_result: ConsensusResult | None`

---

### AGENT-2: Persistent audit memory (cross-session)

**Problem:** Each audit starts with zero knowledge of prior audits on the same contract
or protocol, forcing repeated work and missing pattern history.

**Solution:** SQLite-backed MemoryStore keyed on `(contract_address, content_hash)`,
storing previous audit findings, risk history, and attacker patterns observed.

**Implementation:**
- New directory: `agents/src/memory/`
- New node: `memory_lookup` inserted into graph before `ml_assessment`; injects
  `prior_findings: list[dict]` into `AuditState`
- TTL: findings expire after 90 days unless the contract is re-submitted
- Privacy constraint: store only hashes and vulnerability category labels — never raw
  source code

---

### AGENT-3: Streaming audit output (SSE to client)

**Problem:** Full audit pipeline takes 30–120 s; client receives nothing until completion,
making integration fragile and UX poor.

**Solution:** Each LangGraph node emits progress events via Server-Sent Events as it
starts and completes. The synthesizer streams token-by-token.

**Event format:**
```json
{"stage": "ml_assessment",  "status": "running",   "elapsed_ms": 1200}
{"stage": "ml_assessment",  "status": "done",       "result": {...}, "elapsed_ms": 4300}
{"stage": "rag_research",   "status": "running",    "elapsed_ms": 4350}
{"stage": "synthesizer",    "status": "streaming",  "token": "The contract..."}
```

**Implementation:**
- `asyncio.Queue` per audit job, shared between LangGraph node callbacks and the SSE
  stream handler
- M6 endpoint: `GET /v1/audit/{id}/stream` — SSE; consumes the queue until a sentinel
  `{"status": "complete"}` event is pushed

---

### AGENT-4: Structured evidence linking

**Problem:** The synthesizer narrative says "function X is vulnerable" but provides no
link to specific source lines, forcing manual review to locate the issue.

**Solution:**
- `static_analysis` node (Slither) already returns per-finding data; extend to include
  line numbers and function names
- Synthesizer system prompt updated to require: "cite line numbers for every claim you
  make about a specific function or expression"
- `predictor.py` extended to return top-3 most-attended FUNCTION nodes (from GNN
  CrossAttention weights) alongside the classification result

**New AuditState field:** `evidence_links: list[{claim: str, source_line: int, confidence: float}]`

**Benefit:** Auditors can jump directly to flagged code; reduces false-positive review
time and improves trust in generated reports.

---

### AGENT-5: Feedback loop for training data collection

**Problem:** No mechanism captures human-verified labels from production audits; the DoS
class has only 377 training samples and no growth path.

**Solution:**
- M6 endpoint: `POST /v1/audit/{id}/feedback {"correct": bool, "actual_classes": [...]}`
- Handler: `agents/src/ingestion/feedback_collector.py` — appends to `quarantined_labels/`
- Human review queue: labels merge into training dataset after N=10 verified labels per
  class (configurable threshold)
- Privacy: collect only `(contract_hash, label_set)` — never raw source code

**Benefit:** Continuous dataset improvement from real production contracts; most impactful
for severely data-starved classes (DoS at 377 samples, Timestamp at 2,191).

---

### AGENT-6: Specialized sub-agent per vulnerability class

**Current state:** One `static_analysis` node handles all vulnerability classes with
generic Slither checks.

**Proposal:** Route high-probability classes to class-specific Slither check agents,
each with a targeted detector configuration.

| Specialist agent | Focus |
|-----------------|-------|
| `reentrancy_agent` | Cross-function state modification after external calls |
| `integer_agent` | Unchecked arithmetic, SafeMath usage patterns |
| `dos_agent` | Unbounded loops, mapping iterations, gas ceiling patterns |
| `access_agent` | Missing access modifiers, `tx.origin` usage |

**Implementation:**
- New file: `agents/src/orchestration/specialist_nodes.py`
- Routing: `ml_result` top-3 class probabilities determine which specialist agents are
  spawned in parallel (LangGraph fan-out)
- Each specialist produces structured findings that feed directly into the synthesizer
  prompt alongside the generic Slither output
