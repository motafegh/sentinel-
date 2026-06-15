# Agents Module

LangGraph orchestration, four MCP servers, a hybrid RAG retriever over DeFi exploit history, an incremental ingestion pipeline, and an on-chain feedback loop.

## Overview

```
                    ┌──────────────────────────────────────────┐
                    │         LangGraph StateGraph (9 nodes)   │
                    │                                          │
                    │  ml_assessment → quick_screen → evidence_router
                    │       │                      ├─ deep ──────────▶ rag_research ──┐
                    │       │                      │                  static_analysis ─┤→ audit_check → cross_validator → synthesizer
                    │       │                      │                  graph_explain ──┘
                    │       │                      └─ fast ──────────────────────────────▶ synthesizer
                    └──────────────────────────────────────────┘
                                        │
                    ┌───────────────────┼───────────────────┐
                    ▼                   ▼                   ▼
              MCP :8010            MCP :8011           MCP :8012          MCP :8013
           inference_server      rag_server          audit_server     graph_inspector
              │                     │                    │                  │
              ▼                     ▼                    ▼                  ▼
         Module 1 FastAPI    HybridRetriever      AuditRegistry       GNN / Slither
           (ML model)      (FAISS + BM25)         (Sepolia)         (hotspots)

                    ┌──────────────────────────────────────────┐
                    │         RAG Pipeline                      │
                    │  DeFiHackLabs → chunk → embed → FAISS    │
                    │  AuditRegistry → feedback → RAG           │
                    └──────────────────────────────────────────┘
```

## Module Map

```
agents/
├── src/
│   ├── orchestration/       LangGraph workflow (9 nodes, conditional routing)
│   │   ├── state.py         AuditState TypedDict (16 fields)
│   │   ├── routing.py       Per-class thresholds, tool routing, verdict computation
│   │   ├── nodes.py         9 node implementations (1415 lines)
│   │   └── graph.py         StateGraph builder, SqliteSaver checkpointing
│   │
│   ├── rag/                 Hybrid FAISS + BM25 retriever
│   │   ├── retriever.py     HybridRetriever with Reciprocal Rank Fusion
│   │   ├── chunker.py       RecursiveCharacterTextSplitter (1536 chars)
│   │   ├── embedder.py      Nomic-embed-text via LM Studio
│   │   ├── build_index.py   Full rebuild with atomic writes + rollback
│   │   └── fetchers/
│   │       ├── base_fetcher.py     Abstract BaseFetcher + Document dataclass
│   │       └── github_fetcher.py   DeFiHackLabs .sol parser (3 formats)
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
    │                static_analysis ─┤→ audit_check → cross_validator → synthesizer → END
    │                graph_explain ───┘
    └─ [fast path]  → synthesizer → END
```

**Two-signal fast-path gate:** Fast path requires BOTH:
1. ML all class probabilities below `DEEP_THRESHOLDS`
2. `quick_screen` zero High/Critical Slither/Aderyn hits

If either signal flags risk, the contract goes to deep path.

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
| LLM-adjudicated (`cross_validator`) | CONFIRMED / LIKELY / DISPUTED / WATCH / SAFE |

**LLM-adjudicated verdicts** prompt the strong LLM (qwen3.5-9b-ud) with per-class evidence (ML tier + probability, Slither findings, RAG topics, prior audits). Falls back silently to rule-based on LLM failure.

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

| Test file | Coverage |
|-----------|---------|
| `test_graph_routing.py` | Routing logic, all node paths, graph compilation, full graph integration (931 lines) |
| `test_smoke_e2e.py` | End-to-end smoke tests — deep/fast/screen-escalated/ML-failure paths (375 lines) |
| `test_audit_server.py` | On-chain history decoding, mock mode, address validation, hard caps (345 lines) |
| `test_inference_server.py` | MCP tool schemas, mock/live transport, batch predict, partial failure (336 lines) |
| `test_routing_phase0.py` | Per-class thresholds, tool matrix, verdict logic, DETECTOR_TO_CLASSES (345 lines) |
| `test_retriever_filters.py` | FAISS+BM25+RRF filter behaviour, score validation, sync validation (236 lines) |
| `test_github_fetcher.py` | DeFiHackLabs parsing (3 comment formats, FIX-20/21/22b) (211 lines) |
| `test_deduplicator.py` | SHA256 hash deduplication, persistence, checkpoint pattern (159 lines) |
| `test_chunker.py` | Chunk size, overlap, metadata inheritance, edge cases (155 lines) |

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
```

## Do Not Change Without Wider Plan

- Do not re-add `confidence` to any schema or routing condition.
- Do not change the MCP `contract_code` / `source_code` field split casually.
- Do not change `RAG_MAX_K` (cap inside `rag_server.py`) without considering synthesizer context window.
- Do not change `chunk_size` or `chunk_overlap` without rebuilding the index.
- Do not wire `static_analysis` node without adding `static_findings` state field, error handling, and tests.
- Do not use mock-mode audit results as real security evidence.
