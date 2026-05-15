```
══════════════════════════════════════════════════════
SENTINEL SESSION HANDOVER
Generated: 2026-04-11 | Code audit + hardening session
══════════════════════════════════════════════════════

POSITION
  Phase:     3 — Intelligence Layer
  Module:    4 — Multi-Agent Audit System
  Milestone: M3 Ingestion Pipeline + Scheduling — COMPLETE ✅
             M3.x Code Hardening + Unit Tests — COMPLETE ✅
             Next: M4 MCP Servers

CODEBASE STATE
  Health: Green
  Reason: Full agents/ codebase audited, 28 bugs fixed, 66 unit
          tests passing. LM Studio confirmed working on port 1235.
          All 10 modules import cleanly. RAG retriever verified
          end-to-end with live embedding + search.

══════════════════════════════════════════════════════
COMPLETED THIS SESSION
══════════════════════════════════════════════════════

  ✓ Full code audit of agents/ — 28 bugs identified and fixed
  ✓ 4 unit test files written — 66 tests, all passing
  ✓ LM Studio port 1235 fix — .env updated, portproxy + firewall added
  ✓ load_dotenv() added to client.py — .env now loaded on import
  ✓ FreshnessPolicy dagster 1.12 incompatibility fixed
  ✓ pytest config conflict fixed (root --cov vs agents venv)
  ✓ End-to-end smoke test confirmed — all 3 LLMs + retriever working

══════════════════════════════════════════════════════
LM STUDIO — CURRENT STATE (2026-04-11)
══════════════════════════════════════════════════════

  PORT CHANGE: LM Studio moved from 1234 → 1235 on 2026-04-11

  CURRENT CONFIG:
    LM Studio port:    1235
    WSL2 gateway IP:   172.21.16.1 (verify after reboot — can change)
    Base URL:          http://172.21.16.1:1235/v1
    Set in:            agents/.env → LM_STUDIO_BASE_URL=http://172.21.16.1:1235/v1
    Loaded by:         load_dotenv() in client.py (module-level, auto)

  WINDOWS PORTPROXY (run as Admin if broken after reboot):
    netsh interface portproxy add v4tov4 listenport=1235 listenaddress=0.0.0.0 connectport=1235 connectaddress=127.0.0.1
    New-NetFirewallRule -DisplayName "LM Studio Port 1235" -Direction Inbound -Protocol TCP -LocalPort 1235 -Action Allow -Profile Any

  VERIFY CONNECTION:
    # WSL2:
    curl -s http://172.21.16.1:1235/v1/models | python3 -c "import sys,json; [print(m['id']) for m in json.load(sys.stdin)['data']]"
    # LLM smoke test:
    cd ~/projects/sentinel/agents && poetry run python src/llm/client.py

  QUICK DIAGNOSIS (run in order if connection fails):
    WSL2: cat /etc/resolv.conf | grep nameserver | awk '{print $2}'  → check IP matches .env
    PowerShell: Get-Service iphlpsvc → Start-Service if stopped
    PowerShell: netstat -ano | findstr "0.0.0.0:1235" → must show LISTENING
    WSL2: ping -c 2 172.21.16.1 → must get replies
    If port changed again: just update LM_STUDIO_BASE_URL in agents/.env — no code changes needed

  LM STUDIO MODELS (confirmed working 2026-04-11):
    gemma-4-e2b-it               FAST   ~12 tok/s  3.18GB GPU
    qwen3.5-9b-ud                STRONG ~37 tok/s  5.55GB GPU
    qwen2.5-coder-7b-instruct    CODER  code-specific, Solidity
    text-embedding-nomic-embed-text-v1.5   EMBED  768-dim, RAG

══════════════════════════════════════════════════════
28 BUGS FIXED — COMPLETE REFERENCE
══════════════════════════════════════════════════════

  FILE: pipeline.py
    FIX-7:  Atomic index writes — write to .tmp, then os.replace() (POSIX rename).
            Crash mid-write no longer corrupts the live index.
    FIX-8:  FileLock(timeout=300) around all index writes.
            Prevents concurrent pipeline + feedback_loop from corrupting each other.
    FIX-9:  BM25 import moved from inside run() body to module level.
            Import-inside-loop caused silent failures on second pipeline run.
    FIX-23: All paths anchored to __file__ (not CWD).
            Old: Path("data/index/faiss.index") — breaks if called from wrong directory.
            New: _AGENTS_DIR / "data" / "index" / "faiss.index" — always correct.

  FILE: feedback_loop.py
    FIX-1:  BM25 rebuilt after every index update.
            Old: BM25 never updated → keyword search degraded over time.
    FIX-2:  Single Deduplicator instance throughout loop.
            Old: two instances created → in-memory state diverged → re-indexing duplicates.
    FIX-3:  Removed dead code — self.pipeline = IngestionPipeline() in __init__.
            Was never used; created a full pipeline object on every FeedbackIngester init.
    FIX-4:  Block range chunked in MAX_BLOCK_RANGE=1999 batches.
            Most RPC providers cap eth_getLogs at 2000 blocks. Requests over the limit
            returned empty results silently — could miss on-chain audit findings.
    FIX-5:  get_new_events() returns None on error (vs [] for "no events").
            Main loop applies exponential backoff: min(POLL_INTERVAL * 2^n, 300s).
            Old: any RPC error returned [] → loop treated failure as "no events", no backoff.
    FIX-6:  Index paths imported from pipeline.py constants (not re-hardcoded locally).

  FILE: retriever.py
    FIX-5b: metadata key: self.metadata.get("built_at") or self.metadata.get("last_run").
            build_index.py writes "built_at"; pipeline.py writes "last_run". Both handled.
    FIX-10: FAISS ↔ chunks count validation on __init__.
            If ntotal ≠ len(chunks): raise RuntimeError("Index corruption").
            Detected immediately on startup, not silently during retrieval.
    FIX-12: _apply_filters() logs WARNING when result set is empty.
            Query and filter values included in message for debugging.
    FIX-23: Paths anchored to __file__.

  FILE: embedder.py
    FIX-13: assert len(all_vectors) == len(chunks) → explicit RuntimeError.
            Python -O flag disables all assert statements silently.
    FIX-14: _embed_batch_with_retry() — 3 attempts per batch, backoff 1s/2s/4s.
            Old: one LM Studio GPU-OOM crash killed the entire embedding run.

  FILE: client.py
    FIX-15: LM_STUDIO_BASE_URL = os.getenv("LM_STUDIO_BASE_URL", fallback).
            Old: hardcoded IP broke on every WSL2 reboot.
    FIX-16: timeout=LM_STUDIO_TIMEOUT (default 60s) added to ChatOpenAI + OpenAIEmbeddings.
            Old: no timeout → GPU OOM or model load → infinite hang, no recovery.
    ADD-2:  AGENT_MODEL_MAP marked as FORWARD DECLARATION (M4.x agents not yet built).
    NEW:    load_dotenv() called at module level so .env is loaded whether the module is
            imported by a pipeline or run directly as a script.

  FILE: build_index.py
    FIX-22: Freshness check upgraded — compares fetched .sol file count to
            metadata["total_documents"]. Warns if index may be stale.
            Old: only checked if index files existed (never caught partial updates).
    FIX-23: Paths anchored to __file__.

  FILE: chunker.py
    FIX-26: DEFAULT_CHUNK_SIZE raised 512 → 1536 chars.
            nomic-embed supports 8192 tokens. DeFiHackLabs descriptions are 800-1500 chars.
            At 512, avg 1.8 chunks/doc with unnecessary mid-sentence splits.
            At 1536, most documents fit in one chunk — cleaner semantic units.
            DEFAULT_CHUNK_OVERLAP raised 64 → 128 (scaled proportionally).
    NOTE:   Existing index built at 512. Rebuild required to apply new chunk size.

  FILE: deduplicator.py
    FIX-25: filter_new() typed as list[Document] → list[Document].
            mark_seen() typed as list[str] → None.
            TYPE_CHECKING guard avoids circular import at runtime.

  FILE: github_fetcher.py
    FIX-20: fetch() scans both src/test/ AND past/ directories.
            past/ was declared but never iterated — ~150 older exploits were silently skipped.
    FIX-21: fetch_since() no longer drops undated files.
            Old: undated docs skipped entirely (159 files, 21.9% of corpus).
            New: undated docs always included, logged at DEBUG level.
    FIX-22b: _infer_vuln_type() takes keyinfo_block: dict (not content: str).
            Old: used content[:1000] which was SPDX license + pragma — not exploit logic.
            New: uses root_cause + summary_block + keyinfo_block values only.

  FILE: scheduler_cron.py
    FIX-27: _check_cron_daemon() uses returncode == 0 (not "running" in stdout).
            Old: "not running" contains "running" → false positive, always reported as running.
    FIX-28: Removed duplicate import subprocess inside run_now() (already at module level).

  FILE: scheduler_dagster.py
    FIX-17: Collapsed fake 3-asset chain into single rag_index asset.
            Old: raw_documents → chunks → rag_index declared as 3 assets but each
            re-fetched the source independently — no data flowed between them.
            DeFiHackLabs was parsed 3× per run. Lineage graph was false.
            New: single rag_index asset owns full pipeline — honest representation.
    FIX-18: FreshnessPolicy — in dagster 1.12, FreshnessPolicy is an abstract base
            class with no constructor args. LegacyFreshnessPolicy has the right signature
            but is rejected by @asset validator (type mismatch). Removed from decorator.
            24h freshness is enforced by the cron schedule (0 2 * * *) instead.
            Revisit when upgrading dagster past 1.12.
    FIX-19: avg_chunk_size: chunk_count // max(total_docs, 1).
            Old: ZeroDivisionError on first run before any documents are indexed.

  FILE: pyproject.toml (agents/)
    FIX-8b: filelock = "^3.13" added — required for FileLock in pipeline.py.

══════════════════════════════════════════════════════
UNIT TESTS — agents/tests/
══════════════════════════════════════════════════════

  Run:    cd ~/projects/sentinel/agents && poetry run pytest
  Result: 66 passed, 0 failed, 4 warnings (numpy deprecation in faiss — not our code)
  Time:   ~0.6s (no LM Studio, no FAISS search, no network)

  test_deduplicator.py     — 16 tests
    Init (empty/existing/corrupted JSON), seen(), filter_new(), mark_seen(),
    persistence to disk, idempotency, checkpoint pattern (mark AFTER success).

  test_chunker.py          — 18 tests
    Default chunk_size=1536 (FIX-26 regression guard), custom sizes,
    single/multi-chunk docs, empty doc, chunk IDs, total_chunks consistency,
    metadata inheritance, parent metadata immutability, chunk_documents() flattening.

  test_github_fetcher.py   — 20 tests
    _extract_loss() all formats (M/K/raw/decimal/billion/old), _infer_vuln_type()
    (FIX-22b: uses keyinfo_block not content[:1000]), _extract_date(),
    fetch_since() includes undated files (FIX-21), fetch() scans past/ (FIX-20).

  test_retriever_filters.py — 12 tests
    _apply_filters() — no filter, vuln_type, date_gte, loss_gte, source, has_summary,
    combined filters, empty result WARNING logged (FIX-12).
    FAISS ↔ chunks count mismatch raises RuntimeError on init (FIX-10).

  KEY TEST PATTERNS:
    Loguru warning capture: add temporary sink (list) — capsys/caplog don't capture loguru.
    MagicMock pickling: use plain dict instead — MagicMock raises PicklingError.
    pytest config: agents/pyproject.toml [tool.pytest.ini_options] addopts="" overrides
                   root sentinel/pyproject.toml --cov flags (pytest-cov not in agents venv).

══════════════════════════════════════════════════════
MODULE 4 — COMPLETE FILE STRUCTURE
══════════════════════════════════════════════════════

  agents/
    src/
      llm/
        client.py              ✅ 3 LLMs + embedding + AGENT_MODEL_MAP + load_dotenv
      rag/
        fetchers/
          __init__.py
          base_fetcher.py      ✅ BaseFetcher ABC + Document dataclass
          github_fetcher.py    ✅ DeFiHackLabsFetcher (726 docs, FIX-20/21/22b)
        chunker.py             ✅ 1536 chars, 128 overlap (FIX-26)
        embedder.py            ✅ nomic via LM Studio direct client + retry (FIX-13/14)
        build_index.py         ✅ full rebuild + freshness check (FIX-22/23)
        retriever.py           ✅ FAISS+BM25+RRF + sync validation + filter warning
      ingestion/
        __init__.py
        deduplicator.py        ✅ SHA256 seen_hashes.json (FIX-25)
        pipeline.py            ✅ atomic writes + FileLock + anchored paths (FIX-7/8/9/23)
        scheduler_cron.py      ✅ cron manager, FIX-27/28
        scheduler_dagster.py   ✅ single rag_index asset, FIX-17/18/19
        feedback_loop.py       ✅ block chunking + backoff + BM25 update (FIX-1–6)
    data/
      defihacklabs/            ✅ 726 .sol files, 9.8MB
      index/
        faiss.index            ✅ 4.0MB, 1339 vectors × 768d
        bm25.pkl               ✅ 539KB BM25Okapi
        chunks.pkl             ✅ 604KB, 1339 Chunk objects
        index_metadata.json    ✅ built 2026-04-09, chunk_size=512
        seen_hashes.json       ✅ 726 entries
      feedback_state.json      ✅ last_block=10631088 (Sepolia, 2026-04-10)
    tests/
      __init__.py
      test_deduplicator.py     ✅ 16 tests
      test_chunker.py          ✅ 18 tests
      test_github_fetcher.py   ✅ 20 tests
      test_retriever_filters.py ✅ 12 tests
    .dagster/                  ✅ DAGSTER_HOME — persists run history
    .env                       ✅ LM_STUDIO_BASE_URL + DAGSTER_HOME
    pyproject.toml             ✅ all deps + pytest config

══════════════════════════════════════════════════════
RAG INDEX — CURRENT STATE
══════════════════════════════════════════════════════

  Built:       2026-04-09T20:53:09
  Build time:  13.7s (RTX 3070 GPU)
  Source:      DeFiHackLabs (726 docs, 2017-2026)
  Chunks:      1339 (avg 1.8/doc, 512 chars, 64 overlap — old chunk_size)
  Vectors:     1339 × 768d float32 (IndexFlatL2)
  BM25:        BM25Okapi, 1339 docs
  Dedup:       726 doc_ids in seen_hashes.json

  NOTE: chunker.py now uses 1536 chars (FIX-26) but the built index still uses
        512. The index is correct and working — rebuild when adding new sources
        to benefit from the larger chunk size.

  REBUILD COMMAND:
    cd ~/projects/sentinel/agents
    poetry run python -m src.rag.build_index

  VERIFY COMMAND (requires LM Studio running):
    poetry run python -m src.rag.retriever
    → should show Grim/Balancer/Euler, Market/Orion/HundredFinance, Templedao/Bybit

  DATA COVERAGE:
    @KeyInfo extracted:    468/726 (64.5%)
    Loss amount parsed:    464/726 (63.9%)
    Analysis URLs found:   404/726 (55.6%)
    Vuln type classified:  125/726 (17.2%)  ← "other" bucket = remaining 82.8%
    With date:             721/726 (99.3%)

══════════════════════════════════════════════════════
PIPELINE — TECHNICAL DETAILS
══════════════════════════════════════════════════════

  INGESTION PIPELINE (pipeline.py):
    Fetch → deduplicate → chunk → embed → atomic-write FAISS/BM25/chunks
    Checkpoint: mark_seen() called AFTER successful index write — crash-safe
    Lock: FileLock(timeout=300s) — pipeline + feedback_loop can't corrupt each other
    Paths: all anchored to agents/ via __file__

  DEDUPLICATION:
    First run:  726 docs, ~14s (full embedding)
    Second run: 0 new docs, ~0.1s (all skipped) ✓
    doc_id = SHA256(file_path)[:16] — stable across content edits

  CRON SCHEDULER:
    Schedule:  0 2 * * * (daily at 2 AM)
    Installed: crontab -l | grep SENTINEL_INGESTION
    Log:       ~/projects/sentinel/logs/ingestion_cron.log
    Daemon:    started via /etc/wsl.conf [boot] command
    Commands:
      poetry run python -m src.ingestion.scheduler_cron install
      poetry run python -m src.ingestion.scheduler_cron remove
      poetry run python -m src.ingestion.scheduler_cron status
      poetry run python -m src.ingestion.scheduler_cron run-now

  DAGSTER SCHEDULER:
    Asset:     rag_index (single asset — full pipeline in one step)
    Schedule:  daily_ingestion_schedule (0 2 * * *)
    UI:        poetry run dagster dev -f src/ingestion/scheduler_dagster.py
               http://localhost:3000
    DAGSTER_HOME: ~/projects/sentinel/agents/.dagster (set in .env)
    Dagster version: 1.12.22

  FEEDBACK LOOP:
    Contract:  AuditRegistry 0x14E5eFb6DE4cBb74896B45b4853fd14901E4CfAf (Sepolia)
    Threshold: score > 5734 (≈70% confidence field element)
    Polling:   every 30s via eth_getLogs, chunked in 1999-block batches (FIX-4)
    Backoff:   min(30 * 2^n, 300s) on error (FIX-5)
    State:     agents/data/feedback_state.json — last_block=10631088
    Run:       poetry run python -m src.ingestion.feedback_loop

  HYBRID RETRIEVAL (FAISS + BM25 + RRF):
    faiss_candidates = 20 per system → RRF merge → post-filter → top-k
    RRF_K = 60 (standard constant)
    Filters: vuln_type, date_gte, loss_gte, source, has_summary
    Embedding: nomic-embed-text-v1.5 via LM Studio direct client call

══════════════════════════════════════════════════════
DEPENDENCIES (agents/pyproject.toml)
══════════════════════════════════════════════════════

  Production:
    langchain ^0.3
    langchain-openai ^0.2          OpenAI-compatible → LM Studio
    langchain-community ^0.3       FAISS loader, BM25
    langgraph ^0.2                 Agent orchestration (M5)
    faiss-cpu ^1.8                 Vector similarity search
    rank-bm25 ^0.2                 BM25Okapi keyword search
    pydantic ^2.0                  Structured outputs (M7)
    mcp ^1.0                       Model Context Protocol (M4)
    httpx ^0.27                    Async HTTP for Module 1 API
    loguru ^0.7                    Consistent logging
    python-dotenv ^1.0             .env loading
    filelock ^3.13                 Single-writer lock for index files (FIX-8)
    dagster ^1.12.22               Asset orchestration + scheduling
    dagster-webserver ^1.12.22     Dagster UI
    web3 ^7.15.0                   Ethereum RPC (feedback loop)

  Dev:
    pytest ^8.0
    pytest-asyncio ^0.23

══════════════════════════════════════════════════════
MODULE 4 — MILESTONE TRACKER
══════════════════════════════════════════════════════

  COMPLETED:
    M1: LM Studio client + model routing          ✅
    M2: RAG knowledge base (full pipeline)        ✅
      M2.1 DeFiHackLabs fetcher (726 docs)        ✅
      M2.2 Chunker (now 1536 chars, FIX-26)       ✅
      M2.3 Embedder (nomic, retry, FIX-13/14)     ✅
      M2.4 FAISS + BM25 index build               ✅
      M2.5 Hybrid retriever (FAISS+BM25+RRF)      ✅
    M3: Ingestion pipeline + scheduling           ✅
      M3.1 pipeline.py (atomic + lock + paths)    ✅
      M3.2 scheduler_cron.py                      ✅
      M3.3 scheduler_dagster.py                   ✅
      M3.4 GitHub Actions (written, not pushed)   🟡
      M3.5 feedback_loop.py                       ✅
    M3.x: Code hardening + unit tests             ✅
      28 bugs fixed across all agents/ files      ✅
      66 unit tests, all passing                  ✅

  NEXT — M4: MCP Servers
    M4.1 sentinel-inference MCP server
         Exposes Module 1 /predict API as MCP tool
         Tools: predict(contract_code), batch_predict()
         File: agents/src/mcp/servers/inference_server.py
    M4.2 sentinel-rag MCP server
         Exposes HybridRetriever as MCP tool
         Tools: search(query, k, filters)
         File: agents/src/mcp/servers/rag_server.py
    M4.3 sentinel-audit MCP server
         Exposes AuditRegistry (Sepolia) as MCP tool
         Tools: submit_audit(), get_audit_history()
         File: agents/src/mcp/servers/audit_server.py

  THEN:
    M5: LangGraph graph (state + nodes + edges + routing)
    M6: Five agents (static, ml, rag, code, synthesizer)
    M7: Structured outputs (Pydantic AuditReport schema)
    M8: Evaluation framework
    M9: End-to-end audit test on a real contract

══════════════════════════════════════════════════════
PARKED TOPICS
══════════════════════════════════════════════════════

  • GitHub Actions push — needs internet (file written: .github/workflows/ingest.yml)
  • CI_MODE embedder — sentence-transformers CPU fallback for GitHub Actions
  • SWC Registry fetcher — next data source after MCP servers
  • Rekt.news scraper — HTML parsing needed
  • Immunefi fetcher — public data, no API key required
  • Dual embedding — CodeBERT + nomic for code+text hybrid
  • LLM classification for vuln_type "other" bucket (82.8% → classified)
  • Euler Finance loss cross-reference from README metadata
  • DVC versioning of index
  • v4.0 git tag (after internet stable)
  • Module 2 zkml milestone docs
  • M3.6 Dockerfile
  • TRANSFORMERS_OFFLINE=1 + HF_TOKEN → .env
  • Index rebuild with new chunk_size=1536 (FIX-26 — existing index still uses 512)
  • WSL2 VPN sharing

══════════════════════════════════════════════════════
NEXT SESSION — START HERE
══════════════════════════════════════════════════════

  1. Verify LM Studio reachable:
       HOST=$(cat /etc/resolv.conf | grep nameserver | awk '{print $2}')
       curl -s http://$HOST:1235/v1/models | python3 -c "import sys,json; [print(m['id']) for m in json.load(sys.stdin)['data']]"

       If timeout → check portproxy + firewall for port 1235 (see LM STUDIO section above)
       If port changed → update LM_STUDIO_BASE_URL in agents/.env only

  2. Run quick health check:
       cd ~/projects/sentinel/agents
       poetry run pytest                            # 66 tests, ~0.6s
       poetry run python src/llm/client.py          # FAST_OK STRONG_OK CODER_OK
       poetry run python -m src.rag.retriever       # 3 test queries with results

  3. Start M4.1 — sentinel-inference MCP server:
       mkdir -p agents/src/mcp/servers
       touch agents/src/mcp/servers/__init__.py
       touch agents/src/mcp/servers/inference_server.py

       CONCEPT:
         MCP = Model Context Protocol (Anthropic, Nov 2024)
         Standardises how agents connect to tools.
         Server exposes: tools (functions), resources (data), prompts.
         Client (agent) discovers and calls tools via standard protocol.
         Any MCP-compatible agent can use any MCP server — plug-and-play.

       SENTINEL's 3 MCP servers:
         inference_server  → Module 1 /predict endpoint → risk score [0,1]
         rag_server        → HybridRetriever.search() → similar past exploits
         audit_server      → AuditRegistry on Sepolia → on-chain audit history

       IMPLEMENTATION PATTERN:
         from mcp.server import Server
         from mcp.types import Tool, TextContent
         server = Server("sentinel-inference")
         @server.list_tools() → define available tools
         @server.call_tool()  → implement tool logic

  4. Then M4.2 (rag MCP server) and M4.3 (audit MCP server)
  5. Then M5 LangGraph graph

══════════════════════════════════════════════════════
AGENT MODEL ROUTING REFERENCE
══════════════════════════════════════════════════════

  AGENT_MODEL_MAP (forward declaration — agents not yet built):
    static_analyzer  → qwen2.5-coder-7b  reads Solidity structure + AST
    ml_intelligence  → gemma-4-e2b       calls Module 1 /predict API only
    rag_researcher   → qwen3.5-9b        reasons over past exploit descriptions
    code_logic       → qwen2.5-coder-7b  understands reentrancy/access-control
    synthesizer      → qwen3.5-9b        generates structured AuditReport

  CONDITIONAL ROUTING (planned, LangGraph M5):
    risk_score > 0.70 → full 5-node path (all agents)
    risk_score ≤ 0.70 → skip RAG + CodeLogic → fast Synthesizer path

══════════════════════════════════════════════════════
MODULE 2 ZKML — STILL PENDING
══════════════════════════════════════════════════════

  • v4.0 git tag (after internet stable)
  • Module 2 milestone docs
  • solc-select: confirmed on 0.8.20

══════════════════════════════════════════════════════
```
