```
══════════════════════════════════════════════════════
SENTINEL SESSION HANDOVER
Generated: 2026-04-10 | Session ~100+ exchanges
══════════════════════════════════════════════════════

POSITION
  Phase:     3 — Intelligence Layer
  Module:    4 — Multi-Agent Audit System
  Milestone: M3 Ingestion Pipeline + Scheduling — COMPLETE ✅
             Next: M4 MCP Servers

CODEBASE STATE
  Health: Green
  Reason: Full ingestion pipeline working end-to-end.
          Deduplication confirmed working (0.1s on second run).
          Dagster UI working with lineage tracking.
          Feedback loop connected to Sepolia, polling confirmed.
          All files committed.

══════════════════════════════════════════════════════
COMPLETED THIS SESSION
══════════════════════════════════════════════════════

  ✓ pipeline.py — incremental ingestion orchestrator
  ✓ deduplicator.py — SHA256-based seen_hashes.json tracking
  ✓ scheduler_cron.py — daily 2 AM cron job installed
  ✓ scheduler_dagster.py — Dagster asset pipeline with UI
  ✓ feedback_loop.py — on-chain AuditRegistry → RAG index
  ✓ web3 installed (poetry add web3)
  ✓ dagster + dagster-webserver installed
  ✓ .github/workflows/ingest.yml written (parked — needs internet)
  ✓ /etc/wsl.conf fixed — single [boot] section
  ✓ DAGSTER_HOME set — run history persists
  ✓ git committed — all M3 work saved

══════════════════════════════════════════════════════
LM STUDIO — PERMANENT FIX (CONFIRMED WORKING)
══════════════════════════════════════════════════════

  ROOT CAUSE: IP Helper service (iphlpsvc) stopped on reboot
  
  PERMANENT FIXES APPLIED:
    1. IP Helper set to Automatic startup
    2. Scheduled task "WSL2 LM Studio Setup" at login
       Script: $env:USERPROFILE\wsl2-lmstudio-setup.ps1
    3. Firewall rules (persist across reboots):
       "LM Studio WSL2" — TCP 1234, any profile
       "WSL2 Inbound"   — all traffic on WSL2 adapter
       "WSL2 ICMP"      — ping
       "LM Studio Port 1234" — port 1234 on WSL2 interface
    4. WSL2 adapter: "vEthernet (WSL (Hyper-V firewall))"
    5. /etc/wsl.conf — cron starts on WSL2 boot

  QUICK DIAGNOSIS (run in order if connection fails):
    PowerShell: Get-Service iphlpsvc → Start if stopped
    PowerShell: netstat -ano | findstr "0.0.0.0:1234" → must show LISTENING
    WSL2: ping -c 2 172.21.16.1 → must get replies
    WSL2: HOST=$(cat /etc/resolv.conf | grep nameserver | awk '{print $2}')
          curl -s http://$HOST:1234/v1/models | grep '"id"'

  LM STUDIO MODELS:
    gemma-4-e2b-it               FAST   3.18GB GPU
    qwen3.5-9b-ud                STRONG 5.55GB GPU
    qwen2.5-coder-7b-instruct    CODER  code-specific
    text-embedding-nomic-embed   EMBED  RAG embeddings

══════════════════════════════════════════════════════
MODULE 4 — COMPLETE FILE STRUCTURE
══════════════════════════════════════════════════════

  agents/
    src/
      llm/
        client.py              ✅ 3 models + AGENT_MODEL_MAP
      rag/
        fetchers/
          __init__.py
          base_fetcher.py      ✅ BaseFetcher ABC + Document dataclass
          github_fetcher.py    ✅ DeFiHackLabsFetcher (726 docs)
        chunker.py             ✅ 512 chars, 64 overlap
        embedder.py            ✅ nomic via LM Studio direct client call
        build_index.py         ✅ full rebuild + populates seen_hashes
        retriever.py           ✅ FAISS + BM25 + RRF hybrid search
      ingestion/
        __init__.py
        deduplicator.py        ✅ SHA256 seen_hashes.json
        pipeline.py            ✅ incremental orchestrator
        scheduler_cron.py      ✅ cron job manager
        scheduler_dagster.py   ✅ Dagster assets + schedule
        feedback_loop.py       ✅ on-chain → RAG ingestion
    data/
      defihacklabs/            ✅ 726 .sol files, 9.8MB
      exploits/                ✅ raw document cache
      index/
        faiss.index            ✅ 4.0MB, 1339 vectors × 768d
        bm25.pkl               ✅ 539KB keyword model
        chunks.pkl             ✅ 603KB, 1339 Chunk objects
        index_metadata.json    ✅ build passport
        seen_hashes.json       ✅ 726 entries, dedup registry
      feedback_state.json      ✅ last processed Sepolia block
    .dagster/                  ✅ DAGSTER_HOME — persists run history
    pyproject.toml             ✅ all deps installed
  .github/
    workflows/
      ingest.yml               🟡 written, parked (needs internet push)

══════════════════════════════════════════════════════
PIPELINE — TECHNICAL DETAILS
══════════════════════════════════════════════════════

  INGESTION PIPELINE (pipeline.py):
    Fetchers:      DeFiHackLabsFetcher (extendable — add SWC, Rekt, etc.)
    Deduplication: SHA256(file_path)[:16] as doc_id
                   seen_hashes.json tracks indexed docs
    Incremental:   FAISS.add() on existing index (no rebuild)
                   BM25 full rebuild (fast — no embedding)
                   Chunks list extended in sync with FAISS
    Error handling: per-fetcher try/except — one failing doesn't kill others
    Checkpoint:    mark_seen() called AFTER successful index update

  DEDUPLICATION RESULTS CONFIRMED:
    First run:  726 docs, 13.7s (full embedding)
    Second run: 0 new docs, 0.1s (all skipped) ✓

  BUG FIXED THIS SESSION:
    build_index.py didn't populate seen_hashes.json
    → pipeline treated everything as new on first run
    → FAISS doubled to 2678 vectors
    Fix: build_index.py now writes seen_hashes.json after build
    Fix: rebuilt index cleanly at 1339 vectors

  CRON SCHEDULER:
    Schedule:  0 2 * * * (daily at 2 AM)
    Installed: crontab -l to verify
    Logs:      ~/projects/sentinel/logs/ingestion_cron.log
    Daemon:    started via /etc/wsl.conf [boot] command
    Commands:
      poetry run python -m src.ingestion.scheduler_cron install
      poetry run python -m src.ingestion.scheduler_cron remove
      poetry run python -m src.ingestion.scheduler_cron status
      poetry run python -m src.ingestion.scheduler_cron run-now

  DAGSTER SCHEDULER:
    Assets:    raw_documents → chunks → rag_index
    Schedule:  daily_ingestion_schedule (0 2 * * *)
    UI:        poetry run dagster dev -f src/ingestion/scheduler_dagster.py
               http://localhost:3000
    DAGSTER_HOME: ~/projects/sentinel/agents/.dagster
    Export:    export DAGSTER_HOME=~/projects/sentinel/agents/.dagster
    Note:      FreshnessPolicy removed — not supported in dagster 1.12.x
    Note:      Absolute imports required inside @asset functions (Dagster subprocess issue)

  FEEDBACK LOOP:
    Contract:  AuditRegistry 0x14E5eFb6DE4cBb74896B45b4853fd14901E4CfAf (Sepolia)
    Threshold: score > 5734 (≈70% confidence, field element encoding)
    Polling:   every 30 seconds via get_logs()
    State:     agents/data/feedback_state.json (last processed block)
    Confirmed: connected to Sepolia block 10631085, polling working ✓
    Run:       poetry run python -m src.ingestion.feedback_loop

  GITHUB ACTIONS (parked):
    File:      .github/workflows/ingest.yml (written, not pushed)
    Trigger:   schedule 0 2 * * * + workflow_dispatch
    Secrets:   SEPOLIA_RPC + DEPLOYER_PRIVATE_KEY (set in GitHub UI)
    Issue:     CI has no LM Studio → need CPU embedder (CI_MODE flag)
    Push when: internet available

══════════════════════════════════════════════════════
RAG INDEX — CURRENT STATE
══════════════════════════════════════════════════════

  Source:      DeFiHackLabs (726 docs, 2017-2026)
  Chunks:      1339 (avg 1.8/doc, 512 chars, 64 overlap)
  Vectors:     1339 × 768d float32 (nomic-embed-text-v1.5)
  Build time:  12-14 seconds
  FAISS type:  IndexFlatL2

  RETRIEVER VERIFIED:
    "flash loan attack on lending protocol"
    → Grim, Balancer, Euler — all flash_loan ✓
    "reentrancy vulnerability in vault" + {vuln_type: reentrancy}
    → Market, Orion, HundredFinance ✓
    "access control" + {loss_gte: 1M}
    → Templedao ($2.3M), Bybit ($1.5B) ✓

  VERIFY COMMAND:
    cd ~/projects/sentinel/agents
    poetry run python -m src.rag.retriever

══════════════════════════════════════════════════════
DESIGN DECISIONS THIS SESSION
══════════════════════════════════════════════════════

  1. MARK SEEN AFTER SUCCESS (checkpoint pattern)
     Decision: deduplicator.mark_seen() called only after
               successful FAISS + chunks + BM25 update
     Reason: if embedding fails mid-batch, those docs are NOT
             marked seen → next run retries them automatically
             Never lose work due to mid-run crashes

  2. FAISS INCREMENTAL vs FULL REBUILD
     Decision: FAISS.add() on existing index for new docs
               BM25 full rebuild (no incremental support)
     Reason: FAISS add() is O(n_new) not O(n_total)
             BM25 rebuild is fast (no embedding step)
             Both maintain index/chunks sync correctly

  3. DAGSTER ABSOLUTE IMPORTS
     Decision: sys.path.insert(0, ...) inside @asset functions
     Reason: Dagster runs each asset in a subprocess with
             different working directory — relative imports fail
             This is a known Dagster limitation

  4. FEEDBACK LOOP POLLING vs WEBSOCKET
     Decision: polling (get_logs every 30s)
     Reason: simpler, works with HTTP RPC providers
             websocket subscriptions require persistent connection
             and some RPC providers don't support them reliably
     Production: upgrade to websocket for real-time response

  5. SCORE THRESHOLD 5734 (70%)
     Decision: only ingest findings with score > 5734 field element
     Reason: low confidence findings add noise to knowledge base
             70% threshold balances recall vs precision
             Adjustable in config — not hardcoded in pipeline logic

══════════════════════════════════════════════════════
DEPENDENCIES (agents/pyproject.toml)
══════════════════════════════════════════════════════

  langchain ^0.3
  langchain-openai ^0.2
  langchain-community ^0.3
  langgraph ^0.2
  faiss-cpu ^1.8
  rank-bm25 ^0.2
  pydantic ^2.0
  mcp ^1.0
  httpx ^0.27
  loguru ^0.7
  python-dotenv ^1.0
  dagster ^1.12.22
  dagster-webserver ^1.12.22
  web3 (latest)
  pytest ^8.0 (dev)
  pytest-asyncio ^0.23 (dev)

══════════════════════════════════════════════════════
MODULE 4 — REMAINING MILESTONES
══════════════════════════════════════════════════════

  COMPLETED:
    M1: LM Studio client + model routing          ✅
    M2: RAG knowledge base (full pipeline)        ✅
    M3: Ingestion pipeline + scheduling           ✅

  NEXT — M4: MCP Servers
    M4.1 sentinel-inference MCP server
         Exposes Module 1 API as MCP tool
         Tools: predict(contract_code), batch_predict()
    M4.2 sentinel-rag MCP server
         Exposes RAG retriever as MCP tool
         Tools: search(query, k, filters)
    M4.3 sentinel-audit MCP server
         Exposes AuditRegistry as MCP tool
         Tools: submit_audit(), get_audit_history()

  THEN:
    M5: LangGraph graph (state + nodes + edges + routing)
    M6: Five agents (static, ml, rag, code, synthesizer)
    M7: Structured outputs (Pydantic AuditReport)
    M8: Evaluation framework
    M9: End-to-end test

══════════════════════════════════════════════════════
PARKED TOPICS
══════════════════════════════════════════════════════

  • GitHub Actions push — needs internet
  • CI_MODE embedder — sentence-transformers CPU fallback
  • SWC Registry fetcher — next data source
  • Rekt.news scraper — HTML parsing
  • Immunefi fetcher — public data
  • Dual embedding — CodeBERT + nomic
  • LLM classification for vuln_type "other" bucket
  • DVC versioning of index
  • Euler Finance loss cross-reference from README
  • v4.0 git tag + Module 2 milestone docs
  • M3.6 Dockerfile
  • TRANSFORMERS_OFFLINE=1 + HF_TOKEN → .env
  • WSL2 VPN sharing

══════════════════════════════════════════════════════
NEXT SESSION — START HERE
══════════════════════════════════════════════════════

  1. Verify LM Studio reachable:
       HOST=$(cat /etc/resolv.conf | grep nameserver | awk '{print $2}')
       curl -s http://$HOST:1234/v1/models | grep '"id"'

  2. If connection fails → IP Helper check:
       PowerShell: Get-Service iphlpsvc → Start-Service if stopped
       PowerShell: netstat -ano | findstr "0.0.0.0:1234"

  3. Verify RAG still works:
       cd ~/projects/sentinel/agents
       poetry run python -m src.rag.retriever 2>&1 | tail -5

  4. Start M4.1 — sentinel-inference MCP server:
       mkdir -p agents/src/mcp/servers
       touch agents/src/mcp/servers/inference_server.py

       CONCEPT FIRST:
       MCP = Model Context Protocol (Anthropic, Nov 2024)
       Standardizes how agents connect to tools
       Server exposes: tools (functions), resources (data), prompts
       Client (agent) discovers and calls tools via standard protocol
       Any MCP-compatible agent can use any MCP server

       SENTINEL's 3 MCP servers:
         inference_server → Module 1 /predict endpoint
         rag_server       → HybridRetriever.search()
         audit_server     → AuditRegistry on Sepolia

  5. Then M4.2 sentinel-rag MCP server
  6. Then M4.3 sentinel-audit MCP server
  7. Then M5 LangGraph graph

══════════════════════════════════════════════════════
MODULE 2 ZKML — STILL PENDING
══════════════════════════════════════════════════════

  • v4.0 git tag (after internet stable)
  • Module 2 milestone docs
  • solc-select: confirmed on 0.8.20

══════════════════════════════════════════════════════
```

