══════════════════════════════════════════════════════
SENTINEL SESSION HANDOVER
Generated: 2026-04-09 | Session ~200+ exchanges
══════════════════════════════════════════════════════

POSITION
  Phase:     3 — Intelligence Layer
  Module:    4 — Multi-Agent Audit System
  Milestone: M2 RAG Knowledge Base — COMPLETE ✅
             Next: M3 Ingestion Pipeline + Scheduling

CODEBASE STATE
  Health: Green
  Reason: Full RAG pipeline working end-to-end.
          726 documents → 1339 chunks → 1339 vectors → hybrid search.
          Three LM Studio models confirmed working.
          All files committed.

══════════════════════════════════════════════════════
COMPLETED THIS SESSION
══════════════════════════════════════════════════════

  ✓ LM Studio WSL2 connection permanently fixed
  ✓ client.py updated with coder model + AGENT_MODEL_MAP
  ✓ DeFiHackLabsFetcher — 726 documents, three formats
  ✓ Chunker — 1339 chunks, 512 chars, 64 overlap
  ✓ Embedder — 768-dim vectors via nomic-embed-text
  ✓ build_index.py — FAISS + BM25 index built in 12.2s
  ✓ HybridRetriever — RRF search working, filters working
  ✓ git commit — all agents/ work committed

══════════════════════════════════════════════════════
LM STUDIO — PERMANENT FIX DETAILS
══════════════════════════════════════════════════════

  ROOT CAUSE FOUND THIS SESSION:
    IP Helper service (iphlpsvc) was stopped on every reboot
    → portproxy rules exist but nothing actually listens
    → netstat shows nothing on 0.0.0.0:1234
    → WSL2 traffic hits nothing → silent failure

  PERMANENT FIXES APPLIED:
    1. Set-Service iphlpsvc -StartupType Automatic
       → IP Helper now starts automatically on boot

    2. Scheduled task "WSL2 LM Studio Setup" created
       → runs at every Windows login
       → script: $env:USERPROFILE\wsl2-lmstudio-setup.ps1
       → does: Start-Service iphlpsvc + portproxy add

    3. Firewall rules created (persist across reboots):
       "LM Studio WSL2"     — TCP port 1234, any profile
       "WSL2 Inbound"       — all traffic on WSL2 adapter
       "WSL2 ICMP"          — ping allowed
       "LM Studio Port 1234" — port 1234 on WSL2 interface

    4. WSL2 adapter name confirmed:
       "vEthernet (WSL (Hyper-V firewall))"

  CONNECTION DETAILS:
    LM Studio port:    1234
    WSL2 bridge IP:    172.21.16.1 (verify after reboot)
    Base URL:          http://172.21.16.1:1234/v1
    Port proxy:        0.0.0.0:1234 → 127.0.0.1:1234

  IF CONNECTION FAILS AFTER REBOOT:
    Step 1: Check IP Helper running (PowerShell):
      Get-Service iphlpsvc
      Start-Service iphlpsvc (if stopped)

    Step 2: Check portproxy listening (PowerShell):
      netstat -ano | findstr "0.0.0.0:1234"
      If nothing: netsh interface portproxy add v4tov4 ...

    Step 3: Check WSL2 IP:
      cat /etc/resolv.conf | grep nameserver | awk '{print $2}'
      Update client.py if changed

    Step 4: Test ping:
      ping -c 2 172.21.16.1

    Step 5: Test curl:
      curl -s http://172.21.16.1:1234/v1/models | grep '"id"'

    Full nuclear reset guide saved in session history.

══════════════════════════════════════════════════════
LM STUDIO MODELS — CONFIRMED WORKING
══════════════════════════════════════════════════════

  gemma-4-e2b-it               FAST   3.18GB fully on GPU
                                       ~12 tok/s
  qwen3.5-9b-ud                STRONG 5.55GB 31/33 layers GPU
                                       ~37 tok/s
  qwen2.5-coder-7b-instruct    CODER  code-specific LLM
                                       Solidity analysis
  text-embedding-nomic-embed   EMBED  768-dim vectors
                                       RAG embeddings only

  AGENT MODEL ROUTING:
    static_analyzer  → qwen2.5-coder  (reads Solidity structure)
    ml_intelligence  → gemma-4-e2b    (calls Module 1 API only)
    rag_researcher   → qwen3.5-9b     (reasons over descriptions)
    code_logic       → qwen2.5-coder  (understands Solidity logic)
    synthesizer      → qwen3.5-9b     (generates structured report)

  JUST-IN-TIME LOADING:
    LM Studio auto-loads models on first request
    Penalty: ~8-18 seconds on first call to unloaded model
    Subsequent calls: instant (model stays loaded)

══════════════════════════════════════════════════════
CLIENT.PY — FINAL STATE
══════════════════════════════════════════════════════

  Location: agents/src/llm/client.py

  Constants:
    LM_STUDIO_BASE_URL = "http://172.21.16.1:1234/v1"
    LM_STUDIO_API_KEY  = "lm-studio"
    MODEL_FAST         = "gemma-4-e2b-it"
    MODEL_STRONG       = "qwen3.5-9b-ud"
    MODEL_CODER        = "qwen2.5-coder-7b-instruct"
    MODEL_EMBED        = "text-embedding-nomic-embed-text-v1.5"

  Functions:
    get_fast_llm()        → ChatOpenAI (gemma)
    get_strong_llm()      → ChatOpenAI (qwen3.5)
    get_coder_llm()       → ChatOpenAI (qwen-coder)
    get_embedding_model() → OpenAIEmbeddings (nomic)

  AGENT_MODEL_MAP dict exported for agent definitions

  Smoke test:
    cd ~/projects/sentinel/agents
    poetry run python src/llm/client.py
    → FAST_OK, STRONG_OK, CODER_OK

══════════════════════════════════════════════════════
RAG PIPELINE — FULL DETAILS
══════════════════════════════════════════════════════

  DATA SOURCE:
    DeFiHackLabs repo cloned at: agents/data/defihacklabs/
    Size: 9.8MB, 726 Solidity PoC files, 2017-2026
    Three comment formats discovered:
      Format A: @Summary (25 files) — step-by-step attack narrative
      Format B: @KeyInfo (468 files) — loss + addresses + @Analysis URLs
      Format C: free-form (159 files) — older unstructured comments

  COVERAGE AFTER FIX:
    @KeyInfo extracted:    468/726 (64.5%)
    @Summary extracted:     24/726  (3.3%)
    Loss amount parsed:    464/726 (63.9%)
    Analysis URLs found:   404/726 (55.6%)
    Vuln type classified:  125/726 (17.2%)
    With date:             721/726 (99.3%)

  KNOWN DATA GAPS:
    Euler Finance: Loss: unknown (no @KeyInfo in file — data gap not bug)
    vuln_type "other": 82.8% — keyword matching limit of source
    Future fix: LLM classification step for "other" bucket

  FILE STRUCTURE:
    agents/src/rag/
      fetchers/
        __init__.py
        base_fetcher.py     ← BaseFetcher ABC + Document dataclass
        github_fetcher.py   ← DeFiHackLabsFetcher (726 docs)
      chunker.py            ← Chunker (512 chars, 64 overlap)
      embedder.py           ← Embedder (nomic via LM Studio)
      build_index.py        ← full pipeline orchestrator
      retriever.py          ← HybridRetriever (FAISS + BM25 + RRF)

  INDEX FILES (agents/data/index/):
    faiss.index         4.0MB  — 1339 vectors × 768d (IndexFlatL2)
    bm25.pkl            539KB  — BM25Okapi keyword model
    chunks.pkl          603KB  — 1339 Chunk objects (text + metadata)
    index_metadata.json  324B  — build passport

  INDEX BUILD STATS:
    Documents:    726
    Chunks:       1339 (avg 1.8/doc, min 1, max ~4)
    Chunk sizes:  min 12 chars, max 512 chars, avg 280 chars
    Vectors:      1339 × 768d float32
    Build time:   12.2s (RTX 3070 GPU accelerated)
    Embedding:    9ms per chunk

  RETRIEVER VERIFIED WORKING:
    Query 1: "flash loan attack on lending protocol"
      → Grim (2021), Balancer (2020), Euler (2023) — all flash_loan ✓
    Query 2: "reentrancy vulnerability in vault" + {vuln_type: reentrancy}
      → 29 candidates → 11 after filter → Market, Orion, HundredFinance ✓
    Query 3: "access control privilege escalation" + {loss_gte: 1M}
      → Templedao ($2.3M), Bybit ($1.5B) — only 2 passed filter ✓

══════════════════════════════════════════════════════
KEY BUGS FIXED THIS SESSION
══════════════════════════════════════════════════════

  1. LM Studio not reachable from WSL2 (recurring)
     Root cause: IP Helper service (iphlpsvc) stopped on reboot
     Fix: Set-Service Automatic + scheduled task at login
     ADR: portproxy requires IP Helper to actually listen

  2. github_fetcher.py — low coverage (2% root cause, 7% vuln type)
     Root cause: regex only matched "// Root cause:" format (17 files)
                 @Analysis regex expected URL on same line as tag
     Fix: _extract_block_lines() shared utility — line-by-line scanning
          extracts @Summary, @KeyInfo, @Analysis blocks robustly
     Coverage after: 64.5% @KeyInfo, 55.6% URLs, 17.2% vuln type

  3. Loss regex missing @KeyInfo format
     Root cause: PATTERN_LOSS matched "// Loss ~$197M" not
                 "// @KeyInfo - Total Lost : ~59643 USD"
     Fix: PATTERN_LOSS_KEYINFO + PATTERN_LOSS_OLD two-pattern approach
     Coverage after: 63.9% of documents have loss amount

  4. Euler Finance Loss: unknown
     Root cause: Euler file has no @KeyInfo at all (data gap, not bug)
     Status: parked — future fix via README cross-reference

  5. embedder.py — BadRequestError 400
     Root cause: LangChain's embed_documents() wraps texts in dict format
                 LM Studio expects plain list of strings
     Fix: bypass LangChain wrapper, call self.embedding_model.client.create()
          directly with plain list of strings

  6. Module path errors (recurring)
     Root cause: running from wrong directory
     Fix: always cd ~/projects/sentinel/agents before poetry run commands
          use poetry run python -m src.rag.module (not direct file path)

══════════════════════════════════════════════════════
DESIGN DECISIONS MADE THIS SESSION
══════════════════════════════════════════════════════

  1. CODE-SPECIFIC LLM ADDED
     Added qwen2.5-coder-7b-instruct as MODEL_CODER
     Assigned to: StaticAnalyzerAgent, CodeLogicAgent
     Reason: code-specific models understand Solidity patterns
             (access control, reentrancy, state machine) better
             than general models

  2. EMBED DESCRIPTIONS NOT RAW SOLIDITY
     Decision: build human-readable descriptions from comments,
               embed those instead of raw Solidity code
     Reason: embedding models trained on natural language,
             descriptions match queries better than syntax
     Future stretch: dual embedding (nomic + CodeBERT)

  3. DIRECT CLIENT CALL FOR EMBEDDINGS
     Decision: bypass LangChain's embed_documents() wrapper
     Reason: LM Studio incompatibility with LangChain's format
     Implementation: self.embedding_model.client.create(input=texts)

  4. RRF_K = 60
     Decision: use standard RRF constant
     Reason: empirically proven across many retrieval tasks
     Effect: smooths rank differences, prevents rank-1 dominance

  5. faiss_candidates = 20 for retrieval
     Decision: retrieve 20 from each system before merging/filtering
     Reason: filtering can remove many results — need buffer
             20 gives good recall without excessive computation

  6. SHA256(file_path)[:16] as doc_id
     Decision: hash of file path, not content
     Reason: stable across content edits, deterministic, dedup-safe

  7. Post-retrieval filtering (not pre-filtering)
     Decision: retrieve 20, then apply metadata filters
     Reason: pre-filtering requires FAISS filtering (complex)
             post-filtering is simple and effective at our scale

══════════════════════════════════════════════════════
ARCHITECTURE — FULL MODULE 4 PLAN
══════════════════════════════════════════════════════

  COMPLETED:
    M1: LM Studio client + model routing     ✅
    M2: RAG knowledge base                   ✅
      M2.1 DeFiHackLabs fetcher              ✅
      M2.2 Chunker                           ✅
      M2.3 Embedder                          ✅
      M2.4 FAISS + BM25 index build          ✅
      M2.5 Hybrid retriever                  ✅

  NEXT — M3: Ingestion Pipeline + Scheduling
    M3.1 pipeline.py — full orchestration
    M3.2 scheduler_cron.py — Linux cron
    M3.3 scheduler_dagster.py — Dagster assets
    M3.4 scheduler_github.py — GitHub Actions workflow
    M3.5 feedback_loop.py — on-chain event → RAG update

  THEN:
    M4: MCP servers (inference, rag, audit)
    M5: LangGraph graph (state, nodes, edges)
    M6: Five agents (static, ml, rag, code, synthesizer)
    M7: Structured outputs (Pydantic AuditReport)
    M8: Evaluation framework
    M9: End-to-end test

  STORAGE ARCHITECTURE:
    Now:   Local FAISS + BM25 + DVC versioning
    Later: Swap FAISS → Pinecone/Weaviate (one-line change)
           Same HybridRetriever interface, different backend

  CONDITIONAL ROUTING (LangGraph):
    risk_score > 0.70 → full 5-node path
    risk_score ≤ 0.70 → skip RAG + CodeLogic → Synthesizer

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
  pytest ^8.0 (dev)
  pytest-asyncio ^0.23 (dev)

══════════════════════════════════════════════════════
PARKED TOPICS
══════════════════════════════════════════════════════

  • SWC Registry fetcher (next data source after pipeline built)
  • Rekt.news scraper (architecture only — HTML parsing complexity)
  • Immunefi fetcher (public data, no API key)
  • Dual embedding stretch goal (CodeBERT + nomic-embed)
  • LLM classification for vuln_type "other" bucket (17.2% → better)
  • Euler Finance loss cross-reference from README metadata
  • DVC versioning of index (after pipeline complete)
  • v4.0 git tag — internet available
  • Module 2 milestone docs
  • M3.6 Dockerfile
  • TRANSFORMERS_OFFLINE=1 + HF_TOKEN → .env
  • WSL2 VPN sharing — same netsh pattern, VPN-specific routing

══════════════════════════════════════════════════════
NEXT SESSION — START HERE (exact order)
══════════════════════════════════════════════════════

  1. Verify LM Studio reachable:
       HOST=$(cat /etc/resolv.conf | grep nameserver | awk '{print $2}')
       curl -s http://$HOST:1234/v1/models | grep '"id"'

  2. If connection fails — check IP Helper first:
       PowerShell: Get-Service iphlpsvc
       PowerShell: Start-Service iphlpsvc (if stopped)
       Then: netstat -ano | findstr "0.0.0.0:1234"

  3. Update client.py if WSL2 IP changed:
       cd ~/projects/sentinel/agents
       cat /etc/resolv.conf | grep nameserver | awk '{print $2}'
       code src/llm/client.py → update LM_STUDIO_BASE_URL if needed

  4. Verify RAG index still loads:
       poetry run python -m src.rag.retriever
       → should show 3 test queries with results

  5. Start M3.1 — pipeline.py:
       cd ~/projects/sentinel
       touch agents/src/ingestion/__init__.py
       touch agents/src/ingestion/pipeline.py
       Orchestrates: fetchers → chunker → embedder → index update
       Handles: deduplication, incremental updates, error recovery

  6. M3.2 — Cron scheduling:
       touch agents/src/ingestion/scheduler_cron.py
       Learn: crontab syntax, how cron works, when to use it
       Implement: daily 2am ingestion job

  7. M3.3 — Dagster:
       Learn: @asset, @job, freshness_policy, lineage tracking
       Implement: Dagster asset definitions for ingestion pipeline

  8. M3.4 — GitHub Actions:
       touch .github/workflows/ingest.yml
       Learn: scheduled workflows, secrets, artifact storage
       Implement: daily cloud ingestion run

  9. M3.5 — Feedback loop:
       touch agents/src/ingestion/feedback_loop.py
       Learn: on-chain event listening, web3.py event filters
       Implement: AuditRegistry → RAG ingestion trigger

══════════════════════════════════════════════════════
MODULE 2 ZKML — STILL PENDING
══════════════════════════════════════════════════════

  (carried from previous sessions)
  • v4.0 git tag after internet stable
  • Module 2 milestone docs: docs/milestones/milestone-4-zkml.md
  • solc-select confirmed back on 0.8.20

══════════════════════════════════════════════════════