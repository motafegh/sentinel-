# SENTINEL Agents — Architecture Diagram

> **Scope:** whole-module reference for `~/projects/sentinel/agents/`.
> Source-of-truth is the code; this is a single-page visual index.
> Last verified: 2026-06-27.

---

## 1. One-Page Overview

```
                          ┌──────────────────────────────────────────────────────────────┐
                          │              SENTINEL Agents module                         │
                          │   LangGraph orchestration of a 14-node audit pipeline      │
                          │   backed by 5 MCP servers, a hybrid RAG, and an LLM client │
                          └──────────────────────────────────────────────────────────────┘
                                                  │
   ┌──────────────────────────────────────────────┼────────────────────────────────────────┐
   │                                              │                                        │
   ▼                                              ▼                                        ▼
┌──────────────────────┐               ┌──────────────────────┐                ┌──────────────────────┐
│  src/orchestration/  │               │     src/mcp/servers/ │                │      src/rag/        │
│  14 LangGraph nodes  │               │  5 SSE MCP servers   │                │  Hybrid retriever    │
│  AuditState schema   │◄────SSE──────►│  :8010/11/12/13/14   │                │  FAISS + BM25        │
│  routing logic       │               │  Tool definitions    │                │  6 fetchers          │
└──────────────────────┘               └──────────┬───────────┘                └──────────┬───────────┘
            │                                     │                                        │
            │ pure-Python                         │ HTTP/SSE                               │
            │                                     ▼                                        ▼
            │                          ┌──────────────────────┐                ┌──────────────────────┐
            │                          │  src/llm/client.py   │                │  src/ingestion/      │
            │                          │  LM Studio (4 model  │                │  Build pipeline      │
            │                          │  roles: fast/strong/ │                │  Feedback loop       │
            │                          │  embed/synth)        │                │  Schedulers          │
            │                          └──────────────────────┘                └──────────────────────┘
            │                                     │                                        │
            │                                     │                                        │ polls
            ▼                                     ▼                                        ▼
   ┌──────────────────────┐         ┌──────────────────────────┐            ┌──────────────────────┐
   │  Module 1 (ml/)      │         │  LM Studio (OpenAI-compat│            │  Sepolia             │
   │  FastAPI /predict    │         │  http://.../v1)          │            │  AuditRegistry       │
   │  FastAPI /hotspots   │         │  4 GGUF models loaded    │            │  (on-chain events)   │
   └──────────────────────┘         └──────────────────────────┘            └──────────────────────┘
```

---

## 2. Orchestration — The 14-Node LangGraph

```
                                  ┌──────────────────────────────────────────────┐
                                  │   14-node StateGraph (orchestration/nodes/)   │
                                   │   AuditState TypedDict                       │
                                  └──────────────────────────────────────────────┘
                                                          │
   START                                                  ▼
     │         ┌──────────────────────────────────────────────────────────────────────┐
     │         │  TIER 0 — every contract (always runs, no shortcuts)                │
     │         ├──────────────────────────────────────────────────────────────────────┤
     │         │  ① ml_assessment         MCP :8010 → POST /predict  (Module 1)      │
     │         │  ② quick_screen          Slither + Aderyn, High/Critical only        │
     │         └──────────────────────────────────────────────────────────────────────┘
     │                                       │
     ▼                                       ▼
   ┌────────────────────────────────────────────────────────────────────────────────┐
   │  ③ evidence_router       Per-class routing decisions → state.routing_decisions │
   │                          Two-signal fast/deep gate                              │
   └────────────────────────────────────────────────────────────────────────────────┘
                  │                                                    │
                  │ [deep path]                                        │ [fast path]
                  ▼                                                    │
   ┌─────────────────────────────────────────┐                         │
   │  Parallel fan-out (asyncio.gather)      │                         │
   │  ④ rag_research   MCP :8011  search     │                         │
   │  ⑤ static_analysis  Slither + Aderyn   │                         │
   │      (scoped to ML-flagged classes)     │                         │
   │  ⑥ graph_explain  MCP :8013  hotspots   │                         │
   │      (real GNN attention, Slither fallback)                       │
   │  ⑦ formal_verification  Halmos symbolic execution                 │
   └─────────────────────────────────────────┘                         │
                  │                                                    │
                  ▼                                                    │
              ⑧ audit_check        MCP :8012  get_audit_history
                  │                  (Sepolia AuditRegistry)            │
                  ▼                                                    │
              ⑨ consensus_engine   A.6/A.7 — weighted vote (ML
                  │                  discounted, ML_WEIGHT_SCALE=0.5); │
                  │                  Bayesian confidence                │
                  ▼                                                    │
              ⑩ cross_validator    A.4 — Prosecutor/Defender/Judge
                  │                  debate (LLM, 3 sequential calls)  │
                  ├────────────────────────────────────────────┤
                  │                       [fast path rejoins here]    │
                  ▼                                                    ▼
                          ⑪ synthesizer
                          Assembles final_report (JSON + LLM narrative)
                                          │
                                          ▼
                          ⑫ reflection       A.3 — self-critique (LLM optional)
                                          │
                                          ▼
                          ⑬ explainer        A.8 — LIME attribution (ml/slither/rag %)
                                          │
                                          ▼
                          ⑭ visualizer       A.9 — interactive HTML (hotspot map)
                                          │
                                          ▼
                                         END
```

### Two-Signal Fast-Path Gate (`evidence_router`)

```
  ┌──────────────────────────────────────────────────────────────┐
  │  Signal A: ML all classes < DEEP_THRESHOLDS  (routing.py)    │
  │  Signal B: quick_screen zero High/Critical hits              │
  │                                                              │
  │      A AND B   →  FAST path   (synthesizer directly)         │
  │      A OR B    →  DEEP path   (fan-out to evidence tools)    │
  └──────────────────────────────────────────────────────────────┘
```

### Parallel vs Sequential

```
  evidence_router ──┬─► rag_research        ┐
                    ├─► static_analysis      ┤
                    ├─► graph_explain        ┤
                    └─► formal_verification  ┴─► audit_check ─► consensus ─► cross_validator ─► synthesizer
                                                                                              │
                                                                                       (rejoins fast path)
```

---

## 3. MCP Server Mesh

```
  ┌────────────────────────────────────────────────────────────────────────────────┐
  │  src/mcp/servers/        5 SSE/HTTP servers, each a Starlette app + uvicorn   │
  │  Pattern: SseServerTransport("/messages/")  +  Route("/sse")  +  Route("/health")│
  │  Shared httpx.AsyncClient (lifespan-managed) — avoids per-call TCP handshake   │
  └────────────────────────────────────────────────────────────────────────────────┘

  Port   Server name                Tool(s)                  Backing system
  ────   ─────────────────────      ─────────────────────    ─────────────────────────
  :8010  sentinel-inference         predict                  Module 1 FastAPI /predict
                                     batch_predict           (port 8001, "Module 1")
                                                              Mock mode via MODULE1_MOCK=true

  :8011  sentinel-rag               search                   HybridRetriever
                                                              (FAISS + BM25, Nomic embed)
                                                              5 fetchers (Phase A) + DeFiHackLabs

  :8012  sentinel-audit             get_audit_history        audit/ package
                                     get_latest_audit         (Sepolia AuditRegistry,
                                     check_audit_exists        0x14E5...fAf)

  :8013  sentinel-graph-inspector   get_graph_hotspots       Module 1 FastAPI /hotspots
                                                              (real GNN attention)
                                                              Fallback chain:
                                                              ML API → Slither → mock

  :8014  sentinel-representation    get_embeddings           Module 1 FastAPI /embeddings
                                                              (GNN node embedding vectors)
```

### Connection Pattern (per node call)

```
  ┌──────────────┐   async with sse_client(url)   ┌────────────────────────┐
  │ graph node   │ ──────────────────────────────► │  MCP server            │
  │ (orchestration)                              │  :80xx/sse            │
  │                                               │  → initialize()        │
  │   await session.call_tool(name, args)         │  → call_tool dispatch  │
  │   await session.close()                       │  → result dict         │
  └──────────────┘                                └────────────────────────┘
   Connection-per-call (intentional M5 simplicity; promote to pooled in M6)
```

---

## 4. State Schema — AuditState (orchestration/state.py)

```
  ┌──────────────────────────────────────────────────────────────────────────────┐
  │  AuditState TypedDict  (total=False — every field optional)                  │
  │  LangGraph merges partial dicts returned by each node                        │
  └──────────────────────────────────────────────────────────────────────────────┘

  ┌─── INPUT (immutable) ──────────────────────┐
  │  contract_code        str                  │
  │  contract_address     str                  │
  └────────────────────────────────────────────┘
  ┌─── ML EVIDENCE ────────────────────────────┐
  │  ml_result            dict                 │  ← ml_assessment (3-tier schema)
  │  ml_hotspots          list[dict]           │  ← graph_explain
  └────────────────────────────────────────────┘
  ┌─── ROUTING TRACE ──────────────────────────┐
  │  routing_decisions   list[str]  (operator  │  ← evidence_router + any node
  │                            .add reducer)   │
  └────────────────────────────────────────────┘
  ┌─── STATIC ANALYSIS (deep path) ────────────┐
  │  quick_screen_hits    dict[str, list]      │  ← quick_screen (always)
  │  static_findings      list[dict]           │  ← static_analysis (deep)
  │  external_call_summary list[dict]          │  ← static_analysis (ExternalBug)
  └────────────────────────────────────────────┘
  ┌─── EVIDENCE TOOLS (deep path) ─────────────┐
  │  rag_results          list[dict]           │  ← rag_research
  │  audit_history        list[dict]           │  ← audit_check
  │  econ_scenarios       list[dict]           │  ← Phase 3 (placeholder)
  └────────────────────────────────────────────┘
  ┌─── VERDICTS (consensus + validation) ──────┐
  │  verdicts             dict[str, str]       │  ← cross_validator
  │  confirmations        dict[str, list]      │  ← cross_validator
  │  contradictions       dict[str, list]      │  ← cross_validator
  │  consensus_verdict    dict[str, dict]      │  ← consensus_engine (A.6)
  │  confidence_by_class  dict[str, float]     │  ← consensus_engine (A.7)
  │  metric_attribution   dict[str, dict]      │  ← explainer (A.8)
  │  debate_transcript    dict[str, str]       │  ← cross_validator (A.4 debate)
  │  evidence_list        list (append-reducer) │  ← any node (P2)
  │  verdict_provable     dict[str, str]        │  ← synthesizer (ZK-anchorable)
  │  verdict_full         dict[str, str]        │  ← synthesizer (all evidence)
  └──────────────────────────────────────────┘
  ┌─── FINAL OUTPUT ───────────────────────────┐
  │  final_report         dict                 │  ← synthesizer (full enrichment)
  │  narrative            str | None           │  ← synthesizer (LLM Markdown)
  │  hotspot_visualization str | None          │  ← visualizer (A.9)
  │  reflection_notes     dict                 │  ← reflection (A.3)
  └────────────────────────────────────────────┘
  ┌─── ERROR ───────────────────────────────────┐
  │  error                str | None           │  ← any node (non-fatal)
  └──────────────────────────────────────────┘
  ┌─── RULE 5C / SECURITY / PROVENANCE (P2–P5) ────┐
  │  tool_status         dict (merge-reducer)      │  ← any node (ran/reason per tool)
  │  injection_matches   list (append-reducer)     │  ← cross_validator, synthesizer (P4)
  │  model_hash          str                       │  ← ml_assessment (SHA-256, P5)
  └──────────────────────────────────────────────┘
  ┌─── PHASE B PLACEHOLDERS (no node yet) ─────┐
  │  symbolic_findings    list[dict]           │  ← formal_verification (P8a)
  │  bytecode_analysis    dict                 │  ← Gigahorse
  │  taint_flows          list[dict]           │  ← Taint analyzer
  │  permission_graph     dict                 │  ← Access control
  └────────────────────────────────────────────┘
```

---

## 5. Cross-Module Dependencies — Where the Agents Module Sits

```
  ┌────────────────────┐
  │  contracts/        │  raw .sol files
  │  data/             │  smartbugs curated, test contracts
  └────────┬───────────┘
           │ request an audit
           ▼
  ┌────────────────────────────────────────────────────────────┐
  │  AGENTS MODULE (this repo)                                 │
  │  LangGraph pipeline + MCP servers + RAG + LLM client       │
  │                                                            │
  │  Outputs:                                                  │
  │    data/reports/{address}.json   (synthesizer)             │
  │    data/reports/{address}_hotspot.html  (visualizer)       │
  │    data/checkpoints.db           (SqliteSaver)             │
  └─────┬───────────────┬─────────────────────┬───────────────┘
        │               │                     │
        │ HTTP /predict │ HTTP /hotspots      │ Sepolia RPC
        │ /health       │                     │
        ▼               ▼                     ▼
  ┌──────────────────────┐         ┌──────────────────────────┐
  │  ml/  (Module 1)     │         │  Sepolia blockchain      │
  │  FastAPI api.py      │         │  AuditRegistry contract  │
  │  /predict, /hotspots │         │  (0x14E5...fAf)          │
  │  /health             │         │  AuditSubmitted event    │
  │  Run 12 checkpoint   │         │                          │
  │  F1=0.7004 (Run 12)  │         │                          │
  └──────────┬───────────┘         └──────────────┬───────────┘
             │                                    │
             │ training                           │ events
             ▼                                    │
  ┌──────────────────────┐                        │
  │  data_module/        │                        │
  │  v3 export 22,493    │                        │
  │  5 shards            │                        │
  │  splits 18.6K/2K/2K  │                        │
  └──────────────────────┘                        │
                                                  │
  ┌────────────────────┐                          │
  │  src/ingestion/     │ ◄────────────────────────┘
  │  feedback_loop.py   │  polls Sepolia
  │  (ingestion subscribes to AuditSubmitted events and
  │   ingests new audited contracts back into the RAG)
  └────────┬───────────┘
           │ rebuild
           ▼
  ┌────────────────────┐
  │  src/rag/          │  FAISS + BM25
  │  data/index/       │  chunks + metadata
  └────────────────────┘
```

---

## 6. RAG Pipeline (src/rag/)

```
   6 SOURCE FETCHERS (rag/fetchers/)                       BUILD PIPELINE (rag/)
   ───────────────────────────────                         ──────────────────────
   ┌───────────────────────┐                               fetch → chunk (1536ch)
   │ base_fetcher.py       │  abstract                    → embed (Nomic)
   │                       │                                    │
   │ 6 concrete fetchers:  │                                    ▼
   │  github_fetcher.py    │  DeFiHackLabs exploits  ┌──────────────────────┐
   │  code4rena_fetcher.py │  contest findings       │ Reciprocal Rank      │
   │  sherlock_fetcher.py  │  contest findings       │ Fusion (RRF)         │
   │  solodit_fetcher.py   │  aggregated findings    │ FAISS + BM25 hybrid  │
   │  immunefi_fetcher.py  │  bounty disclosures     └──────────┬───────────┘
   │  swc_registry_*      │  SWC weakness registry             │
   └───────────────────────┘                                    │
                                                                ▼
                                                  agents/data/index/
                                                  (FAISS .bin + BM25 + chunks)
                                                                ▲
                                                                │
              ┌─────────────────────────────────────────────────┘
              │  atomic write + rollback
              │  build_index.py (full rebuild)
              │  ingestion/pipeline.py (incremental)
```

### Retrieval — Hybrid FAISS + BM25 with RRF

```
  query: "{ML top class} exploit pattern in {contract snippet}"
      │
      ├──► FAISS top-K       ─┐
      │                        ├─► Reciprocal Rank Fusion (RRF)
      └──► BM25 top-K        ─┘            │
                                           ▼
                                   ranked chunks (with score)
                                   → state.rag_results
                                   → consumed by rag_research → cross_validator → synthesizer
```

---

## 7. Ingestion + Feedback Loop (src/ingestion/)

```
  ┌──────────────────────┐
  │  Raw sources         │
  │  • DeFiHackLabs      │  ← github_fetcher (3 .sol formats)
  │  • Code4rena, etc.   │  ← json_corpus_fetcher
  └──────────┬───────────┘
             │ (incremental: deduplicator.py SHA256 dedup)
             ▼
  ┌──────────────────────┐
  │  Chunk               │  chunker.py
  │  1536-char windows   │  (RecursiveCharacterTextSplitter)
  └──────────┬───────────┘
             │ (embedder.py Nomic-embed-text via LM Studio)
             ▼
  ┌──────────────────────┐
  │  Atomic write        │  pipeline.py
  │  FAISS + BM25 index  │  tmp → rename (rollback on failure)
  └──────────┬───────────┘
             │
             ▼
       agents/data/index/


  ─── FEEDBACK LOOP (production) ───────────────────────────────────────────

  ┌────────────────────┐    poll block_N       ┌─────────────────────┐
  │  scheduler_cron.py  │ ───────────────────► │  feedback_loop.py   │
  │  scheduler_dagster  │  (cron OR dagster)   │  AuditSubmitted     │
  └────────────────────┘                       │  event filter       │
                                               └──────────┬──────────┘
                                                          │ on new audit
                                                          ▼
                                               ┌─────────────────────┐
                                               │  ingest into RAG    │
                                               │  (dedup → chunk →   │
                                               │   embed → append)   │
                                               └─────────────────────┘
```

---

## 8. LLM Client (src/llm/client.py)

```
  ┌────────────────────────────────────────────────────────────────────┐
  │  LM Studio (OpenAI-compat) at  http://<wsl2-gateway>:4567/v1      │
  │  Agents talks to it via httpx                                    │
  └────────────────────────────────────────────────────────────────────┘

  4 model roles (env-configurable):
  ┌─────────────────┬──────────────────────────┬──────────────────────────────┐
  │ FAST            │  gemma-4-e2b-it           │  debate, validation, reflection
  │ STRONG          │  gemma-4-e2b-it           │  narrative, synthesis (FIX-18)
  │ CODER           │  qwen2.5-coder-7b-instruct│  Solidity analysis, P/D/J debate
  │ EMBED           │  nomic-embed-text-v1.5   │  RAG embeddings (768-dim)
  └─────────────────┴──────────────────────────┴──────────────────────────────┘

  SENTINEL_DETERMINISTIC=1 → disables all LLM calls + RAG (P5 / ZK mode)

  Timeouts (env-configurable, 2026-06-21 retune):
    CROSS_VALIDATOR_TIMEOUT_S  = 90    (single-pass fallback)
    DEBATE_TIMEOUT_S           = 240   (entire 3-role debate — outer)
    SYNTHESIZER_TIMEOUT_S      = 120   (final narrative)
    LM_STUDIO_TIMEOUT          = 60

  Test-mode kill-switch:
    AGENTS_DISABLE_LLM=1   →  LLM-calling nodes (cross_validator, synthesizer,
                              reflection, explainer) fall back to rule-based.
                              Graph runs fast + deterministic.
```

---

## 9. Verdict Pipeline (consensus + debate + synthesis)

```
  ml_result ─────┐
  static_findings┼─►  ⑨ consensus_engine
  aderyn_hits   ─┘    A.6 weighted vote  (per-class reliability, ML discounted)
                          │  ACCURACY_WEIGHTS × ML_WEIGHT_SCALE (0.5)
                          ▼
                    consensus_verdict : {class: {ml_signal, slither_match,
                                                 aderyn_match, score,
                                                 confidence, verdict}}
                          │
                          ▼
                    ⑩ cross_validator  (A.4)
                    DEBATE_MODE=on  →  3 sequential LLM calls:
                      ┌──────────────────────────────────────────┐
                      │  Prosecutor   reads source + evidence   │
                      │     │  argues VULNERABLE                  │
                      │     ▼                                    │
                      │  Defender      given prosecutor's case   │
                      │     │  argues FALSE POSITIVE              │
                      │     ▼                                    │
                      │  Judge         given both sides          │
                      │     │  renders {class: verdict} JSON      │
                      │     ▼                                    │
                      │  debate_transcript: {prosecutor,defender,│
                      │                       judge}            │
                      └──────────────────────────────────────────┘
                          │
                          ▼  (or single-pass if DEBATE_MODE=off,
                              or rule-based if AGENTS_DISABLE_LLM=1)
                    verdicts : {class: CONFIRMED|LIKELY|DISPUTED|WATCH|SAFE}
                    confirmations, contradictions
                          │
                          ▼
                    ⑪ synthesizer
                       • fuse(evidence_list) → verdict_provable + verdict_full  (P2)
                       • assembles final_report (JSON)
                       • risk_probability, top_vulnerability
                       • overall_verdict = max(verdicts by rank)
                       • LLM narrative (Markdown)
                       • persists to data/reports/{address}.json
                          │
                          ▼
                    ⑫ reflection (A.3)   unused evidence, contradictions,
                       uncertain verdicts, known failure modes
                          │
                          ▼
                    ⑬ explainer (A.8)    per-class {ml_pct, slither_pct,
                       rag_pct} attribution (sums to ~100)
                       Folds confidence_by_class + consensus_verdict +
                       reflection_notes INTO final_report
                          │
                          ▼
                    ⑭ visualizer (A.9)   interactive HTML hotspot map
                       data/reports/{address}_hotspot.html
```

---

## 10. Configuration Surface (env vars)

```
  ┌─── Module 1 / ML ──────────────────────┬────────────────────────────────┐
  │ MODULE1_INFERENCE_URL                  │ http://localhost:8001          │
  │ MODULE1_TIMEOUT                        │ 30.0 (sec)                     │
  │ MODULE1_MOCK                           │ false  (true = dev mock)       │
  │ SENTINEL_ML_API_URL                    │ http://localhost:8000  (used   │
  │                                        │   by graph_inspector_server)   │
  │ GRAPH_INSPECTOR_HOTSPOTS_TIMEOUT       │ 60                             │
  └────────────────────────────────────────┴────────────────────────────────┘
  ┌─── MCP servers ────────────────────────┬────────────────────────────────┐
  │ MCP_INFERENCE_URL                      │ http://localhost:8010/sse      │
  │ MCP_RAG_URL                            │ http://localhost:8011/sse      │
  │ MCP_AUDIT_URL                          │ http://localhost:8012/sse      │
  │ MCP_GRAPH_INSPECTOR_URL                │ http://localhost:8013/sse      │
  │ MCP_INFERENCE_PORT                     │ 8010                           │
  │ MCP_RAG_PORT                           │ 8011                           │
  │ MCP_AUDIT_PORT                         │ 8012                           │
  │ MCP_GRAPH_INSPECTOR_PORT               │ 8013                           │
  └────────────────────────────────────────┴────────────────────────────────┘
  ┌─── LLM ───────────────────────────────┬────────────────────────────────┐
  │ LM_STUDIO_BASE_URL                     │ http://<wsl-gateway>:4567/v1   │
  │ LM_STUDIO_TIMEOUT                      │ 60                             │
  │ AGENTS_DISABLE_LLM                     │ 0  (1 = rule-based fallback)   │
  │ DEBATE_MODE                            │ on  (off = single-pass debate) │
  │ CROSS_VALIDATOR_LLM_MODEL              │ fast                           │
  │ CROSS_VALIDATOR_TIMEOUT_S              │ 90                             │
  │ DEBATE_TIMEOUT_S                       │ 240                            │
  │ SYNTHESIZER_TIMEOUT_S                  │ 120                            │
  │ SYNTHESIZER_MAX_TOKENS                 │ 4096                           │
  └────────────────────────────────────────┴────────────────────────────────┘
  ┌─── Consensus / ML discounting ─────────┬────────────────────────────────┐
  │ ML_WEIGHT_SCALE                        │ 0.5   (ML alone → never        │
  │                                        │         CONFIRMED)             │
  │ ACCURACY_WEIGHTS                       │ per-class, in consensus.py     │
  └────────────────────────────────────────┴────────────────────────────────┘
  ┌─── Sepolia / on-chain ────────────────┬────────────────────────────────┐
  │ SEPOLIA_RPC_URL                        │ <alchemy/infura endpoint>      │
  │ AUDIT_REGISTRY_ADDRESS                 │ 0x14E5eFb6DE4cBb74896B45b4853  │
  │                                        │ fd14901E4CfAf                 │
  └────────────────────────────────────────┴────────────────────────────────┘
  ┌─── RAG ────────────────────────────────┬────────────────────────────────┐
  │ AUDIT_RAG_K                            │ 5                              │
  └────────────────────────────────────────┴────────────────────────────────┘
```

---

## 11. Test + Smoke Surface

```
  scripts/  (smoke tests + utilities)
  ─────────
  smoke_langgraph.py          # mock + --live modes for the full graph
  smoke_inference_mcp.py      # SSE smoke :8010
  smoke_rag_mcp.py            # SSE smoke :8011
  smoke_audit_mcp.py          # SSE smoke :8012
  test_k_cap.py               # RAG server k-cap test
  run_real_audit.py           # real-LLM E2E harness (--no-llm / --profile)
  audit_gt_labels.py          # ground-truth labels for eval
  eval_benchmark.py           # evaluation benchmark runner
  build_reliability_matrix.py # P3: builds per-tool reliability matrix from eval run
  audit_labels.py             # extended label set

  tests/  (631 passing, 3 skipped — ~42 test files)
  ──────
  test_graph_routing.py            # full graph, all 14 nodes
  test_smoke_e2e.py                # end-to-end deep/fast/screen-escalated paths
  test_eval_framework.py           # eval harness + Fbeta scoring
  test_ws4_2_selective_gating.py   # WS4.2 asymmetric debate gating
  test_ws3_hotspot_excerpts.py     # WS3 hotspot-guided debate prompts
  test_verdict_fuse.py             # fuse() dual-tier verdict (P2)
  test_verdict_evidence.py         # Evidence dataclass + emit (P2)
  test_verdict_reliability.py      # L1/L3/schema-mismatch paths (P3)
  test_verdict_integrity.py        # FN/FP invariants
  test_comment_strip.py            # Layer 1 injection defense (P4)
  test_injection_detect.py         # Layer 3 — 8 patterns (P4)
  test_adversarial_corpus.py       # 8 adversarial contracts, P4
  test_deterministic_mode.py       # SENTINEL_DETERMINISTIC=1 (P5)
  test_formal_verification.py      # Halmos node (P8a)
  test_p10_gateway.py              # SQLite JobStore + crash recovery (P10)
  test_routing_isolation.py        # routing nodes have no LLM imports
  ... (+ 27 more files covering MCP, RAG, config, eval, etc.)
```

---

## 12. File Map (one-liner per source file)

```
  agents/
  ├── src/orchestration/
  │   ├── state.py            AuditState TypedDict, 29 fields, 1 append-reducer
  │   ├── routing.py          DEEP_THRESHOLDS, ROUTING_RULES, CLASS_TO_DETECTORS,
  │   │                       compute_active_tools, compute_verdict, prob_to_severity
  │   ├── nodes/              14 async node implementations (package, P2)
  │   │   ├── ml_assessment.py, quick_screen.py, evidence_router.py
  │   │   ├── rag_research.py, static_analysis.py, graph_explain.py
  │   │   ├── formal_verification.py (P8a Halmos), audit_check.py
  │   │   ├── consensus_engine.py, cross_validator.py, synthesizer.py
  │   │   ├── reflection.py, explainer.py, visualizer.py
  │   │   └── _helpers.py     shared: _call_mcp_tool, _llm_enabled, AderynRunError
  │   ├── verdict/            verdict package (P2)
  │   │   ├── fuse.py         sole verdict producer → verdict_provable + verdict_full
  │   │   ├── evidence.py, reliability.py, emit.py, verdict.py
  │   ├── graph.py            build_graph(), conditional edges, lazy audit_graph,
  │   │                       SqliteSaver checkpointing (PEP 562)
  │   ├── consensus.py        A.6 — weighted vote, ML_WEIGHT_SCALE
  │   ├── confidence.py       A.7 — Bayesian staged confidence tracking
  │   ├── attribution.py      A.8 — LIME-style per-class evidence % breakdown
  │   ├── visualizer.py       A.9 — interactive HTML hotspot report
  │   ├── timeouts.py         Centralized timeout defaults (2026-06-21)
  │   └── timing.py           step_timer() / timed_node() (2026-06-21)
  │
  ├── src/mcp/servers/
  │   ├── inference_server.py       :8010  predict, batch_predict
  │   ├── rag_server.py             :8011  search
  │   ├── audit/                    :8012  get_audit_history, get_latest_audit, check_audit_exists (package, P2.5)
  │   ├── graph_inspector_server.py :8013  get_graph_hotspots (GNN attention / Slither fallback)
  │   └── representation_server.py  :8014  get_embeddings (GNN node vectors, NEW)
  │
  ├── src/rag/
  │   ├── retriever.py        HybridRetriever (FAISS + BM25 + RRF) (334L)
  │   ├── chunker.py          RecursiveCharacterTextSplitter (1536 chars) (199L)
  │   ├── embedder.py         Nomic-embed-text via LM Studio (228L)
  │   ├── build_index/        Full rebuild package (was build_index.py 661L, split P2.5)
  │   │   └── __main__.py, _orchestrator.py, _pipeline.py, _io.py, _metadata.py, _paths.py
  │   └── fetchers/
  │       ├── base_fetcher.py        Abstract (94L)
  │       ├── github_fetcher.py      DeFiHackLabs .sol (478L)
  │       ├── json_corpus_fetcher.py Shared base (131L)
  │       ├── code4rena_fetcher.py   C4 (17L)  ⚠ DISABLED (WS2)
  │       ├── sherlock_fetcher.py    Sherlock (15L) ⚠ DISABLED (WS2)
  │       ├── solodit_fetcher.py     Solodit (15L)  ⚠ DISABLED (WS2)
  │       ├── immunefi_fetcher.py    Immunefi (15L) ⚠ DISABLED (WS2)
  │       └── swc_registry_fetcher.py SWC (18L)     ⚠ DISABLED (WS2)
  │
  ├── src/ingestion/
  │   ├── pipeline.py         Dedup → chunk → embed → atomic write (313L)
  │   ├── deduplicator.py     SHA256 hash dedup (136L)
  │   ├── feedback_loop.py    Sepolia AuditRegistry event poll → RAG (470L)
  │   ├── scheduler_cron.py   Cron install/remove/status (60L)
  │   └── scheduler_dagster.py Dagster asset + schedule (77L)
  │
  ├── src/llm/
  │   └── client.py           LM Studio (httpx), 4 model roles (233L)
  │
  ├── src/security/           Prompt-injection defense (P4)
  │   ├── comment_strip.py    Layer 1: state-machine comment stripper
  │   ├── prompt_delimit.py   Layer 2: <<CONTRACT_SOURCE>> structural delimiter
  │   ├── injection_detect.py Layer 3: 8-pattern injection scanner
  │   └── prompt_sanitize.py  Orchestrator
  │
  ├── src/api/                Audit gateway (P10)
  │   ├── gateway.py          FastAPI /audit + /health + background health monitor
  │   ├── sqlite_job_store.py SQLite-backed jobs (crash recovery, survives restart)
  │   ├── job_store.py        Abstract interface
  │   └── models.py           Pydantic request/response models
  │
  ├── src/eval/               Evaluation framework (P0/P3)
  │   ├── pipeline_metrics.py Fbeta(β=2), macro/per-class confusion matrix
  │   ├── gates.py            9 gate assertions (must all pass for run to count)
  │   ├── reliability_matrix.py P3: per-tool TP/FP/FN/TN builder
  │   ├── reliability_fit.py  P3: Bayesian shrinkage fitter (α=5)
  │   └── run_benchmark.py, benchmarks.py, regression.py
  │
  ├── src/config/             Externalized decision numbers (P1)
  │   ├── schema.py           SentinelConfig Pydantic model
  │   └── loader.py           get_config() singleton
  │
  ├── scripts/                Smoke + E2E + eval harnesses (8 files)
  ├── tests/                  ~42 files, 631 passing, 3 skipped
  ├── data/                   index/, reports/, checkpoints.db, feedback_state.json
  └── README.md               user-facing quickstart
```

---

## See Also — Per-Subfolder DIAGRAMs

For deeper detail on each subpackage:
- `src/orchestration/DIAGRAM.md` — **ML integration in agents** (cross-module reference to Module 1)
- `src/mcp/servers/DIAGRAM.md` *(next)*
- `src/rag/DIAGRAM.md` *(next)*
- `src/ingestion/DIAGRAM.md` *(next)*
- `src/llm/DIAGRAM.md` *(next)*
