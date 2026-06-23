# Agents Module

LangGraph orchestration, four MCP servers, a hybrid RAG retriever over DeFi exploit history, an incremental ingestion pipeline, and an on-chain feedback loop.

## Overview

```
                    ┌────────────────────────────────────────────────────────────┐
                    │            LangGraph StateGraph (13 nodes)                 │
                    │                                                            │
                    │  ml_assessment → quick_screen → evidence_router            │
                    │       │                      ├─ deep ─▶ rag_research ──┐   │
                    │       │                      │         static_analysis ┤   │
                    │       │                      │         graph_explain ──┘   │
                    │       │                      │              │              │
                    │       │                      │         audit_check         │
                    │       │                      │              │              │
                    │       │                      │       consensus_engine      │  ← A.6/A.7
                    │       │                      │              │              │
                    │       │                      │        cross_validator      │  ← A.4 debate
                    │       │                      │              │              │
                    │       │                      └─ fast ─▶ synthesizer ◀──────┘
                    │       │                                     │                │
                    │       │                                reflection            │  ← A.3
                    │       │                                     │                │
                    │       │                                 explainer            │  ← A.8
                    │       │                                     │                │
                    │       │                                visualizer            │  ← A.9
                    │       │                                     │                │
                    │       │                                    END               │
                    └────────────────────────────────────────────────────────────┘
                                        │
                    ┌───────────────────┼───────────────────┐
                    ▼                   ▼                   ▼
              MCP :8010            MCP :8011           MCP :8012          MCP :8013
           inference_server      rag_server          audit_server     graph_inspector
              │                     │                    │                  │
              ▼                     ▼                    ▼                  ▼
         Module 1 FastAPI    HybridRetriever      AuditRegistry       GNN / Slither
        (ML — treated as a    (FAISS + BM25)        (Sepolia)         (hotspots)
         HINT, not authority —
         see ML_WEIGHT_SCALE)

                    ┌──────────────────────────────────────────────────────┐
                    │         RAG Pipeline                                 │
                    │  DeFiHackLabs + Code4rena/Sherlock/Solodit/          │
                    │  Immunefi/SWC → chunk → embed → FAISS               │
                    │  AuditRegistry → feedback → RAG                      │
                    └──────────────────────────────────────────────────────┘
```

**2026-06-21 — Extended Capability Phase A complete.** Added `consensus_engine`,
`reflection`, `explainer`, `visualizer` nodes; upgraded `cross_validator` to a
Prosecutor/Defender/Judge debate; expanded RAG to 5 new sources. ML predictions
are deliberately down-weighted in voting (`ML_WEIGHT_SCALE`, default 0.5) — the
agent layer does its own analysis via static tools + LLM debate rather than
trusting the ML model as ground truth. Full details:
`docs/changes/2026-06-21-agents-phase-a-extended-capability.md`.

## Module Map

```
agents/
├── src/
│   ├── orchestration/       LangGraph workflow (13 nodes, conditional routing)
│   │   ├── state.py         AuditState TypedDict (29 fields incl. Phase A/B placeholders)
│   │   ├── routing.py       Per-class thresholds, tool routing, verdict computation
│   │   ├── nodes.py         13 node implementations (consensus_engine, cross_validator
│   │   │                    debate, reflection, explainer, visualizer added 2026-06-21)
│   │   ├── consensus.py     A.6 — weighted ML/Slither/Aderyn vote (ML down-weighted)
│   │   ├── confidence.py    A.7 — Bayesian staged confidence tracking
│   │   ├── attribution.py   A.8 — LIME-style evidence-source breakdown
│   │   ├── visualizer.py    A.9 — interactive hotspot HTML generator
│   │   └── graph.py         StateGraph builder, lazy audit_graph, SqliteSaver checkpointing
│   │
│   ├── rag/                 Hybrid FAISS + BM25 retriever
│   │   ├── retriever.py     HybridRetriever with Reciprocal Rank Fusion
│   │   ├── chunker.py       RecursiveCharacterTextSplitter (1536 chars)
│   │   ├── embedder.py      Nomic-embed-text via LM Studio
│   │   ├── build_index.py   Full rebuild with atomic writes + rollback
│   │   └── fetchers/
│   │       ├── base_fetcher.py        Abstract BaseFetcher + Document dataclass
│   │       ├── github_fetcher.py      DeFiHackLabs .sol parser (3 formats)
│   │       ├── json_corpus_fetcher.py Shared base for curated JSON corpora (A.5)
│   │       ├── code4rena_fetcher.py   Code4rena contest findings
│   │       ├── sherlock_fetcher.py    Sherlock contest findings
│   │       ├── solodit_fetcher.py     Solodit aggregated findings
│   │       ├── immunefi_fetcher.py    Immunefi bounty disclosures
│   │       └── swc_registry_fetcher.py SWC weakness-classification registry
│   │
│   ├── ingestion/           Incremental pipeline + feedback loop
│   │   ├── pipeline.py      Dedup → chunk → embed → atomic write
│   │   ├── deduplicator.py  SHA256 hash-based deduplication
│   │   ├── feedback_loop.py AuditRegistry event polling, on-chain → RAG bridge
│   │   ├── scheduler_cron.py
│   │   └── scheduler_dagster.py
│   │
│   ├── mcp/servers/         MCP SSE servers (Model Context Protocol)
│   │   ├── inference_server.py       :8010 — predict, batch_predict
│   │   ├── rag_server.py             :8011 — search
│   │   ├── audit_server.py           :8012 — get_latest_audit, get_audit_history, check_audit_exists
│   │   └── graph_inspector_server.py :8013 — get_graph_hotspots
│   │
│   └── llm/
│       └── client.py        LM Studio connection, 4 model roles
│
├── scripts/                 Smoke tests (see scripts/README.md)
├── tests/                   Unit + integration tests (see tests/README.md, 9 files, 3,293 lines)
├── data/
│   ├── index/               FAISS + BM25 + chunks + metadata
│   ├── reports/             Final audit report JSON per contract_address
│   ├── feedback_state.json  Last processed Sepolia block number
│   └── checkpoints.db       LangGraph SqliteSaver checkpoint database
├── pyproject.toml
└── README.md                ← this file
```

## Quick Start

### 1. Install Dependencies

```bash
cd agents
poetry install
```

### 2. Configure Environment

```bash
cp .env.example .env   # or create manually
```

Required variables:

```bash
# LM Studio (required for RAG embeddings + LLM synthesis)
LM_STUDIO_BASE_URL=http://<wsl2-gateway-ip>:4567/v1
LM_STUDIO_TIMEOUT=60

# Sepolia RPC (required for audit_server.py + feedback_loop.py)
SEPOLIA_RPC_URL=<your-rpc-url>
AUDIT_REGISTRY_ADDRESS=0x14E5eFb6DE4cBb74896B45b4853fd14901E4CfAf

# Module 1 inference (required for ml_assessment node)
MODULE1_INFERENCE_URL=http://localhost:8001

# MCP server ports (defaults work for local development)
MCP_INFERENCE_PORT=8010
MCP_RAG_PORT=8011
MCP_AUDIT_PORT=8012
MCP_GRAPH_INSPECTOR_PORT=8013
```

### 3. Build RAG Index

```bash
poetry run python -m src.rag.build_index
# Fetches DeFiHackLabs, chunks, embeds, builds FAISS + BM25
```

### 4. Start MCP Servers

```bash
# Each in a separate terminal
poetry run python -m src.mcp.servers.inference_server
poetry run python -m src.mcp.servers.rag_server
poetry run python -m src.mcp.servers.audit_server
poetry run python -m src.mcp.servers.graph_inspector_server
```

### 5. Run an Audit

```python
import asyncio
from src.orchestration.graph import build_graph

async def audit():
    graph = build_graph(use_checkpointer=False)
    result = await graph.ainvoke(
        {
            "contract_code": "<solidity source>",
            "contract_address": "0x...",
        },
        config={"configurable": {"thread_id": "audit-001"}},
    )
    print(result["final_report"])

asyncio.run(audit())
```

### 6. Smoke Tests

```bash
poetry run python scripts/smoke_langgraph.py          # mock — no services needed
poetry run python scripts/smoke_langgraph.py --live    # live — all services must be up
poetry run python scripts/smoke_inference_mcp.py
poetry run python scripts/smoke_rag_mcp.py
poetry run python scripts/smoke_audit_mcp.py
```

## Orchestration

### Graph Topology

```
START → ml_assessment → quick_screen → evidence_router
    ├─ [deep path]  → rag_research ──┐
    │                static_analysis ─┤→ audit_check → consensus_engine → cross_validator ─┐
    │                graph_explain ───┘                                                      │
    └─ [fast path]  ─────────────────────────────────────────────────────────────────────────┤
                                                                                                ▼
                                                                                         synthesizer
                                                                                                │
                                                                                          reflection
                                                                                                │
                                                                                           explainer
                                                                                                │
                                                                                          visualizer
                                                                                                │
                                                                                               END
```

**Two-signal fast-path gate:** Fast path requires BOTH:
1. ML all class probabilities below `DEEP_THRESHOLDS`
2. `quick_screen` zero High/Critical Slither/Aderyn hits

If either signal flags risk, the contract goes to deep path. **All paths converge at
`synthesizer`**, then run the post-synthesis enrichment chain
(`reflection → explainer → visualizer`) before `END`.

**`consensus_engine` (A.6/A.7, deep path only):** weighted vote over ML/Slither/Aderyn
per class, then Bayesian-updates a confidence score. ML's vote weight is discounted by
`ML_WEIGHT_SCALE` (default 0.5) — **ML alone can never reach a CONFIRMED verdict**; it
needs at least one corroborating static-analysis hit. This is intentional: Run 12's ML
model is not yet reliable enough to be treated as ground truth, so the agent layer does
independent analysis (static tools + LLM debate) and uses ML only as a clue.

**`cross_validator` (A.4):** when `DEBATE_MODE=on` (default), runs three sequential LLM
calls — Prosecutor (argues vulnerable, reading the actual source), Defender (argues
false-positive), Judge (renders the verdict) — instead of one classification call.
Falls back to a single-pass call if `DEBATE_MODE=off`, and to rule-based verdicts in
`synthesizer` if the LLM is unavailable. Transcript stored in `state["debate_transcript"]`.

**`reflection` (A.3):** self-critique after `synthesizer` — flags unused evidence,
tool contradictions, low-confidence/DISPUTED verdicts, and known failure modes (e.g.
truncated contracts, ExternalBug's known ML over-prediction). Optional LLM-written
narrative summary; always produces a rule-based summary even without an LLM.

**`explainer` (A.8):** LIME-style attribution per verdict (`{ml_pct, slither_pct,
rag_pct}`, sums to ~100) and folds `confidence_by_class` / `consensus_verdict` /
`reflection_notes` into `final_report` so one artifact carries the full enrichment.

**`visualizer` (A.9):** renders a self-contained interactive HTML report (source with
hotspot highlighting + verdict cards with confidence/attribution bars), written to
`data/reports/{contract_address}_hotspot.html`.

### AuditState Fields

| Field | Type | Set By | Description |
|-------|------|--------|-------------|
| `contract_code` | `str` | Caller | Raw Solidity source |
| `contract_address` | `str` | Caller | On-chain address |
| `ml_result` | `dict` | `ml_assessment` | Full ML prediction response |
| `ml_hotspots` | `list[dict]` | `graph_explain` | Function-level hotspot scores |
| `routing_decisions` | `list[str]` | `evidence_router` | Per-class routing log (append-reducer) |
| `quick_screen_hits` | `dict` | `quick_screen` | Slither + Aderyn Tier-0 findings |
| `static_findings` | `list[dict]` | `static_analysis` | Slither + Aderyn deep findings |
| `external_call_summary` | `list[dict]` | `static_analysis` | Inter-contract call graph |
| `rag_results` | `list[dict]` | `rag_research` | Ranked exploit chunks |
| `audit_history` | `list[dict]` | `audit_check` | Prior on-chain audit records |
| `verdicts` | `dict[str, str]` | `cross_validator` | Per-class verdicts |
| `confirmations` | `dict[str, list[str]]` | `cross_validator` | Evidence sources per class |
| `contradictions` | `dict[str, list[str]]` | `cross_validator` | Conflicting evidence |
| `final_report` | `dict` | `synthesizer` | Complete audit report |
| `narrative` | `str \| None` | `synthesizer` | LLM-generated Markdown narrative |
| `error` | `str \| None` | Any node | Non-fatal error |
| `consensus_verdict` | `dict[str, dict]` | `consensus_engine` | Per-class weighted ML/Slither/Aderyn vote |
| `confidence_by_class` | `dict[str, float]` | `consensus_engine` | Bayesian-updated confidence, [0,1] |
| `debate_transcript` | `dict[str, str]` | `cross_validator` | `{prosecutor, defender, judge}` when DEBATE_MODE=on |
| `reflection_notes` | `dict` | `reflection` | Self-critique: unused evidence, contradictions, uncertain verdicts, failure modes |
| `metric_attribution` | `dict[str, dict]` | `explainer` | LIME-style `{ml_pct, slither_pct, rag_pct}` per class |
| `hotspot_visualization` | `str \| None` | `visualizer` | Self-contained interactive HTML report |
| `symbolic_findings`, `bytecode_analysis`, `taint_flows`, `permission_graph` | — | *(Phase B, schema only)* | Halmos/Gigahorse/taint/access-control — nodes not yet built |

### Per-Class Routing

`routing.py` defines three mappings:

**DEEP_THRESHOLDS** — probability triggers deep analysis (deliberately below inference threshold):

| Class | Threshold |
|-------|-----------|
| DenialOfService | 0.30 |
| Reentrancy, IntegerUO, Timestamp, TOD | 0.35 |
| GasException, ExternalBug, CallToUnknown, MishandledException | 0.40 |
| UnusedReturn | 0.45 |

**ROUTING_RULES** — which tools activate per class:

| Classes | Tools |
|---------|-------|
| Reentrancy, IntegerUO, Timestamp, TOD, ExternalBug, CallToUnknown, DenialOfService | `static_analysis` + `rag_research` |
| GasException, MishandledException, UnusedReturn | `static_analysis` only |

**CLASS_TO_DETECTORS** — maps classes to Slither detector names for detector scoping.

### Verdicts

| Source | Scale |
|--------|-------|
| Rule-based (`compute_verdict`) | CONFIRMED / LIKELY / DISPUTED / SAFE |
| Consensus vote (`consensus_engine`) | CONFIRMED / LIKELY / DISPUTED / SAFE (ML weight discounted) |
| LLM-adjudicated debate (`cross_validator`) | CONFIRMED / LIKELY / DISPUTED / WATCH / SAFE |

**LLM-adjudicated verdicts** run a Prosecutor/Defender/Judge debate (FAST model by
default — `CROSS_VALIDATOR_LLM_MODEL=fast`) over per-class evidence (ML tier +
probability, Slither findings, RAG topics, prior audits, and the contract source
itself). Falls back silently to rule-based on LLM failure or when
`AGENTS_DISABLE_LLM=1`.

### Checkpointing

`SqliteSaver` persists state to `data/checkpoints.db` after every node. Resume from crash with the same `thread_id`:

```python
result = await graph.ainvoke(None, config={"configurable": {"thread_id": "audit-001"}})
```

## RAG Pipeline

### Knowledge Base

| Item | Value |
|------|-------|
| Source | DeFiHackLabs GitHub (726 `.sol` exploit PoCs) |
| Chunks | ~752 |
| Chunk size | 1536 chars, 128 overlap |
| Embedding | `text-embedding-nomic-embed-text-v1.5` (768-dim) via LM Studio |
| Vector index | FAISS `IndexFlatL2` |
| Keyword index | `BM25Okapi` |
| Fusion | Reciprocal Rank Fusion (RRF_K = 60) |

### Retrieval

```
FAISS: top-20 by L2 distance     ─┐
                                    ├─ RRF fusion → metadata filter → top-k
BM25: top-20 by keyword match    ─┘
```

### Index Build / Update

| Command | Use case |
|---------|----------|
| `poetry run python -m src.rag.build_index` | Full rebuild from scratch |
| `poetry run python -m src.ingestion.pipeline` | Incremental update (new docs only) |

Write safety: `FileLock` + atomic `Path.replace()` + rollback snapshots + artifact SHA256 checksums.

### Feedback Loop

Polls `AuditSubmitted` events on Sepolia AuditRegistry. High-confidence findings (`score >= 0.70`) are ingested back into the RAG knowledge base. The synthesizer writes `data/reports/{address}.json` which the feedback loop reads to recover `vulnerability_class` (BRIDGE Issue #1).

```bash
SEPOLIA_RPC=<your-rpc> poetry run python -m src.ingestion.feedback_loop
```

## MCP Servers

| Server | Port | Tools | Backend |
|--------|------|-------|---------|
| `inference_server` | 8010 | `predict`, `batch_predict` | Module 1 FastAPI (`:8001`) |
| `rag_server` | 8011 | `search` | `HybridRetriever` (FAISS + BM25) |
| `audit_server` | 8012 | `get_latest_audit`, `get_audit_history`, `check_audit_exists` | AuditRegistry (Sepolia Web3) |
| `graph_inspector_server` | 8013 | `get_graph_hotspots` | GNN attention / Slither fallback |

All servers: SSE transport, `/health` endpoint, mock mode for dev/CI.

## LLM Client

Routes to LM Studio (OpenAI-compatible API):

| Role | Model | Use |
|------|-------|-----|
| FAST | `gemma-4-e2b-it` | Simple tasks, API calls |
| STRONG | `qwen3.5-9b-ud` | Reasoning, synthesis, reports |
| CODER | `qwen2.5-coder-7b-instruct` | Solidity analysis |
| EMBED | `nomic-embed-text-v1.5` | RAG embeddings |

## Testing

```bash
cd agents
poetry run pytest tests/ -v
```

**402 tests** (26 files, ~6,135 lines). `conftest.py` sets `AGENTS_DISABLE_LLM=1`
session-wide so the suite never depends on a live LM Studio.

| Key test files | What they cover |
|---|---|
| `test_graph_routing.py` (1,075L) | Full graph: routing, all 13 nodes, graph compilation, integration |
| `test_eval_framework.py` (471L) | Evaluation harness: benchmark scoring, gt comparison |
| `test_smoke_e2e.py` (375L) | End-to-end deep/fast/screen-escalated/ML-failure paths |
| `test_ws4_2_selective_gating.py` (383L) | WS4.2 — asymmetric debate gating (CONFIRMED+2tools skip) |
| `test_representation_server.py` (356L) | GNN embedding MCP server |
| `test_ws3_hotspot_excerpts.py` (310L) | WS3 — hotspot-guided code excerpts in debate prompts |
| `test_verdict_reconciliation.py` (268L) | 8-case reconciliation table + invariants |
| `test_verdict_integrity.py` (204L) | FN/FP invariants, DISPUTED floor enforcement |
| `test_consensus_voting.py` (161L) | A.6 — ML-discounted weighted vote |
| `test_reflection.py` (154L) | A.3 self-critique + A.4 3-role debate (mocked) |
| `test_visualizer.py` (90L) | A.9 hotspot HTML generation |
| `test_metric_attribution.py` (70L) | A.8 LIME-style attribution |
| `test_confidence_tracking.py` (49L) | A.7 Bayesian confidence |
| `test_retriever_filters.py` (236L) | FAISS+BM25+RRF filter behaviour |
| `test_audit_server.py` (345L) | On-chain history, mock mode, address validation |
| `test_inference_server.py` (335L) | MCP tool schemas, mock/live transport |
| `test_routing_phase0.py` (345L) | Per-class thresholds, tool matrix, verdict logic |
| `test_github_fetcher.py` (211L) | DeFiHackLabs parsing (3 formats) |
| `test_deduplicator.py` (159L) | SHA256 dedup, persistence |
| `test_chunker.py` (155L) | Chunk size, overlap, metadata |
| `test_static_analysis_real_slither.py` (88L) | REAL (non-mocked) Slither |
| `test_static_analysis_real_aderyn.py` (86L) | REAL (non-mocked) Aderyn |
| `test_timeouts_and_timing.py` (97L) | Centralized timeouts + `step_timer`/`timed_node` |
| `test_rag_fetchers.py` (96L) | **WS2:** 5 corpus fetchers exist but **disabled** in `build_index.py` |

`conftest.py` sets `AGENTS_DISABLE_LLM=1` for the whole session so LLM-calling nodes
fall back to rule-based. Tests that exercise the LLM path mock it locally.

**WS2 note (2026-06-22):** The 5 Phase A.5 corpus fetchers (Code4rena/Sherlock/Solodit/
Immunefi/SWC) are **disabled** — their seed corpora were synthetic hand-written placeholders
and one caused a hallucinated verdict. `build_index.py:_extra_fetchers()` returns `[]`.
Fetcher code is kept for when real data is wired per `02_RAG_BUILD_PLAN.md`.

## Environment Variables

```bash
# LM Studio
LM_STUDIO_BASE_URL=http://localhost:4567/v1
LM_STUDIO_API_KEY=lm-studio
LM_STUDIO_TIMEOUT=60

# Module 1 inference
MODULE1_INFERENCE_URL=http://localhost:8001
MODULE1_TIMEOUT=30.0
MODULE1_MOCK=false

# Sepolia / AuditRegistry
SEPOLIA_RPC_URL=<your-rpc>
AUDIT_REGISTRY_ADDRESS=0x14E5eFb6DE4cBb74896B45b4853fd14901E4CfAf
AUDIT_MOCK=false

# MCP servers
MCP_INFERENCE_PORT=8010
MCP_RAG_PORT=8011
MCP_AUDIT_PORT=8012
MCP_GRAPH_INSPECTOR_PORT=8013
MCP_INFERENCE_URL=http://localhost:8010/sse
MCP_RAG_URL=http://localhost:8011/sse
MCP_AUDIT_URL=http://localhost:8012/sse
MCP_GRAPH_INSPECTOR_URL=http://localhost:8013/sse

# RAG
AUDIT_RAG_K=5
RAG_DEFAULT_K=5

# Graph Inspector
SENTINEL_ML_API_URL=http://localhost:8000
GRAPH_INSPECTOR_HOTSPOTS_TIMEOUT=60
GRAPH_INSPECTOR_MOCK=false

# Dagster
DAGSTER_HOME=agents/.dagster

# Extended Capability Phase A (2026-06-21)
AGENTS_DISABLE_LLM=        # "1"/"true" → all LLM calls skipped, rule-based fallback used
DEBATE_MODE=on             # "off" → cross_validator single-pass instead of 3-role debate
ML_WEIGHT_SCALE=0.5        # discounts ML's consensus-vote weight — ML alone can't CONFIRM
REFLECTION_TIMEOUT_S=120
REFLECTION_MAX_TOKENS=1024

# Timeouts — every default centralized in src/orchestration/timeouts.py (2026-06-21).
# Unset here = falls through to that file's default. scripts/run_real_audit.py also
# exposes a --<name>-timeout-s CLI flag per variable + --unbounded-timeouts (sets all
# to 3600s at once, for observing true per-step timing with nothing truncated).
LM_STUDIO_TIMEOUT=60       # floor under EVERY LLM call (import-time read by client.py)
CROSS_VALIDATOR_TIMEOUT_S=90   # single-pass mode only (DEBATE_MODE=off)
DEBATE_TIMEOUT_S=240       # entire 3-role debate as ONE budget, not per-call
SYNTHESIZER_TIMEOUT_S=120
ADERYN_TIMEOUT_S=90
```

## Do Not Change Without Wider Plan

- Do not re-add `confidence` to any schema or routing condition.
- Do not change the MCP `contract_code` / `source_code` field split casually.
- Do not change `RAG_MAX_K` (cap inside `rag_server.py`) without considering synthesizer context window.
- Do not change `chunk_size` or `chunk_overlap` without rebuilding the index.
- Do not wire `static_analysis` node without adding `static_findings` state field, error handling, and tests.
- Do not use mock-mode audit results as real security evidence.
