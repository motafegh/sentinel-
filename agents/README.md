# Agents Module

LangGraph orchestration pipeline, five MCP servers, a hybrid RAG retriever over DeFi
exploit history, an incremental ingestion pipeline, an on-chain feedback loop, a prompt-
injection defense layer, a production gateway, and a full evaluation framework.

**631 tests passing, 3 skipped** (as of 2026-06-26). Phases P0–P10 complete.

## Overview

```
                    ┌────────────────────────────────────────────────────────────┐
                    │            LangGraph StateGraph (14 nodes)                 │
                    │                                                            │
                    │  ml_assessment → quick_screen → evidence_router           │
                    │                           │                               │
                    │              ┌────────────┤ deep path (parallel fan-out)  │
                    │              │            │                               │
                    │        rag_research       │  formal_verification (P8a)    │
                    │        static_analysis    │  graph_explain                │
                    │              │            │                               │
                    │              └────────────▶ audit_check                  │
                    │                                    │                      │
                    │                            consensus_engine  ← A.6/A.7   │
                    │                                    │                      │
                    │                             cross_validator  ← A.4 debate │
                    │                           + P4 injection guard            │
                    │                                    │                      │
                    │            fast path ──────▶ synthesizer ◀───────────────┘
                    │                  (fuse() sole                             │
                    │                  verdict producer)                        │
                    │                                    │                      │
                    │                              reflection   ← A.3           │
                    │                                    │                      │
                    │                               explainer   ← A.8           │
                    │                                    │                      │
                    │                              visualizer   ← A.9           │
                    │                                    │                      │
                    │                                   END                     │
                    └────────────────────────────────────────────────────────────┘
                                        │
              ┌─────────────────────────┼───────────────────────┐
              ▼                         ▼                         ▼
        MCP :8010                 MCP :8011                MCP :8012
     inference_server           rag_server               audit/ package
   (Module 1 FastAPI)       (HybridRetriever)        (AuditRegistry Web3)

       MCP :8013                 MCP :8014
   graph_inspector_server   representation_server
   (GNN attention/Slither)   (GNN embeddings)
```

## Module Map

```
agents/
├── src/
│   ├── orchestration/          LangGraph workflow (14 nodes, conditional routing)
│   │   ├── nodes/              Node implementations (14 files + _helpers.py)
│   │   │   ├── ml_assessment.py
│   │   │   ├── quick_screen.py
│   │   │   ├── evidence_router.py
│   │   │   ├── rag_research.py
│   │   │   ├── static_analysis.py
│   │   │   ├── graph_explain.py
│   │   │   ├── formal_verification.py  ← P8a Halmos
│   │   │   ├── audit_check.py
│   │   │   ├── consensus_engine.py     ← A.6/A.7
│   │   │   ├── cross_validator.py      ← A.4 debate + P4 injection guard
│   │   │   ├── synthesizer.py          ← fuse() verdict + P4 injection guard
│   │   │   ├── reflection.py           ← A.3
│   │   │   ├── explainer.py            ← A.8
│   │   │   ├── visualizer.py           ← A.9
│   │   │   └── _helpers.py             shared: _call_mcp_tool, _llm_enabled, AderynRunError
│   │   ├── verdict/            Verdict production package (P2)
│   │   │   ├── evidence.py     Evidence dataclass + constructors
│   │   │   ├── fuse.py         fuse() — sole verdict producer
│   │   │   ├── reliability.py  L3→L1 fallback reliability weights
│   │   │   ├── emit.py         emit_evidence(), emit_halmos_evidence()
│   │   │   └── verdict.py      Verdict constants
│   │   ├── state.py            AuditState TypedDict (all fields)
│   │   ├── routing.py          Per-class thresholds + routing rules (config-driven)
│   │   ├── graph.py            StateGraph builder + lazy audit_graph singleton
│   │   ├── consensus.py        A.6 weighted ML/Slither/Aderyn vote
│   │   ├── confidence.py       A.7 Bayesian staged confidence tracking
│   │   ├── attribution.py      A.8 LIME-style evidence breakdown
│   │   ├── visualizer.py       A.9 interactive hotspot HTML generator
│   │   ├── timeouts.py         Centralized timeout constants
│   │   └── timing.py           timed_node() wrapper, step_timer
│   │
│   ├── verdict/ → see orchestration/verdict/
│   │
│   ├── security/               Prompt-injection defense (P4)
│   │   ├── comment_strip.py    Layer 1: state-machine Solidity comment stripper
│   │   ├── prompt_delimit.py   Layer 2: <<CONTRACT_SOURCE>> structural delimiter
│   │   ├── injection_detect.py Layer 3: 8-pattern injection scanner
│   │   └── prompt_sanitize.py  Orchestrator
│   │
│   ├── api/                    Audit gateway (P10)
│   │   ├── gateway.py          FastAPI app, /audit, /health
│   │   ├── sqlite_job_store.py SQLite-backed jobs (survives restarts)
│   │   ├── job_store.py        Abstract JobStore interface
│   │   └── models.py           Pydantic request/response models
│   │
│   ├── eval/                   Evaluation framework (P0/P3)
│   │   ├── pipeline_metrics.py Fbeta(β=2), macro/per-class confusion matrix
│   │   ├── gates.py            9 gate assertions for benchmark pass/fail
│   │   ├── run_benchmark.py    CLI benchmark runner
│   │   ├── reliability_matrix.py P3: per-tool TP/FP/FN/TN builder
│   │   ├── reliability_fit.py  P3: Bayesian shrinkage fitter (α=5)
│   │   ├── benchmarks.py       Benchmark contract/verdict definitions
│   │   └── regression.py       Regression test harness
│   │
│   ├── config/                 Externalized decision numbers (P1)
│   │   ├── schema.py           SentinelConfig Pydantic model
│   │   └── loader.py           get_config() singleton
│   │
│   ├── rag/                    Hybrid FAISS + BM25 retriever
│   │   ├── retriever.py        HybridRetriever with RRF + P7 zero-match fix
│   │   ├── chunker.py          RecursiveCharacterTextSplitter (1536 chars)
│   │   ├── embedder.py         Nomic-embed-text via LM Studio
│   │   ├── build_index/        Full rebuild package (was build_index.py, split P2.5)
│   │   └── fetchers/
│   │       ├── base_fetcher.py        Abstract BaseFetcher + Document dataclass
│   │       ├── github_fetcher.py      DeFiHackLabs .sol parser (active)
│   │       └── *.py                   Code4rena/Sherlock/Solodit/Immunefi/SWC (disabled — WS2)
│   │
│   ├── ingestion/              Incremental pipeline + feedback loop
│   │   ├── pipeline.py         Dedup → chunk → embed → atomic index write
│   │   ├── deduplicator.py     SHA256 hash-based deduplication
│   │   ├── feedback_loop.py    AuditRegistry event polling, on-chain → RAG bridge
│   │   ├── scheduler_cron.py   Cron manager (install/remove/status)
│   │   └── scheduler_dagster.py Dagster asset + daily schedule (02:00 UTC)
│   │
│   ├── mcp/servers/            MCP SSE servers
│   │   ├── inference_server.py :8010 — predict, batch_predict
│   │   ├── rag_server.py       :8011 — search
│   │   ├── audit/              :8012 — get_latest_audit, get_audit_history, check_audit_exists
│   │   ├── graph_inspector_server.py :8013 — get_graph_hotspots
│   │   └── representation_server.py  :8014 — get_embeddings
│   │
│   └── llm/
│       └── client.py           LM Studio connection, 4 model roles
│
├── configs/
│   ├── verdicts_default.yaml   L1 decision numbers (baseline policy)
│   └── reliability_v3.yaml     L3 Bayesian-fitted tool reliability weights (active)
│
├── scripts/                    Smoke tests + utilities (see scripts/README.md)
├── tests/                      Unit + integration tests (see tests/README.md)
│
├── data/
│   ├── index/                  FAISS + BM25 + chunks + metadata
│   ├── reports/                Final audit report JSON per contract_address
│   ├── jobs.db                 Gateway SQLite job store (P10)
│   ├── checkpoints.db          LangGraph SqliteSaver checkpoint database
│   └── feedback_state.json     Last processed Sepolia block number
│
├── eval/runs/                  Timestamped eval run directories (metrics + reports)
└── pyproject.toml
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

# Sepolia RPC (required for audit/ server + feedback_loop.py)
SEPOLIA_RPC_URL=<your-rpc-url>
AUDIT_REGISTRY_ADDRESS=0x14E5eFb6DE4cBb74896B45b4853fd14901E4CfAf

# Module 1 inference (required for ml_assessment node)
MODULE1_INFERENCE_URL=http://localhost:8001

# MCP server ports (defaults work for local development)
MCP_INFERENCE_PORT=8010
MCP_RAG_PORT=8011
MCP_AUDIT_PORT=8012
MCP_GRAPH_INSPECTOR_PORT=8013
MCP_REPRESENTATION_PORT=8014
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
poetry run python -m src.mcp.servers.representation_server
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

Deterministic mode (no LLM, reproducible for ZK verification):
```bash
SENTINEL_DETERMINISTIC=1 python -m scripts.smoke_langgraph
```

### 6. Smoke Tests

```bash
poetry run python scripts/smoke_langgraph.py          # mock — no services needed
poetry run python scripts/smoke_langgraph.py --live   # live — all 5 servers must be up
poetry run python scripts/smoke_inference_mcp.py
poetry run python scripts/smoke_rag_mcp.py
poetry run python scripts/smoke_audit_mcp.py
```

## Orchestration

### Graph Topology (14 nodes)

```
START → ml_assessment → quick_screen → evidence_router
    ├─ [deep path] (parallel)  → rag_research ──────────────────┐
    │                          → static_analysis ────────────────┤
    │                          → graph_explain ──────────────────┤→ audit_check
    │                          → formal_verification (P8a) ──────┘       │
    │                                                              consensus_engine
    │                                                                     │
    │                                                              cross_validator
    │                                                            (P4 injection guard)
    │                                                                     │
    └─ [fast path] ──────────────────────────────────────────────▶ synthesizer
                                                                  (fuse() verdicts)
                                                                         │
                                                                   reflection (A.3)
                                                                         │
                                                                    explainer (A.8)
                                                                         │
                                                                   visualizer (A.9)
                                                                         │
                                                                        END
```

**Two-signal fast-path gate:** Fast path requires BOTH signals to agree it's safe:
1. ML — all class probabilities below `DEEP_THRESHOLDS`
2. `quick_screen` — zero High/Critical Slither/Aderyn hits

Both paths converge at `synthesizer`, then run the full post-synthesis chain.

### Key Node Notes

**`formal_verification` (P8a):** Runs Halmos symbolic execution in the deep-path fan-
out. Generates a temp Foundry harness, runs `forge build` + `halmos --json-output`,
emits `Evidence(kind=FORMAL, deterministic=True)`. Fail-soft on missing tools —
surfaces `tool_status["halmos"]["ran"] = False`, never silently returns `[]`.

**`consensus_engine` (A.6/A.7):** Weighted vote over ML/Slither/Aderyn per class, then
Bayesian-updates confidence. `ML_WEIGHT_SCALE=0.5` — ML alone can never reach CONFIRMED.

**`cross_validator` (A.4):** Prosecutor/Defender/Judge debate. Sanitizes contract source
via `prompt_sanitize.py` (P4) before every prompt. Falls back to rule-based on LLM
failure or `AGENTS_DISABLE_LLM=1`. P6 cascade (strong-model re-judgment for ambiguous
verdicts) implemented but disabled by default — strong model over-predicts.

**`synthesizer`:** Calls `fuse()` from `verdict/fuse.py` (P2) — the sole verdict
producer. Produces two verdict tiers: `verdict_provable` (deterministic evidence only,
ZK-anchorable) and `verdict_full` (all evidence, human report). Also sanitizes contract
source (P4) before the LLM narrative call.

### AuditState Selected Fields

| Field | Type | Notes |
|-------|------|-------|
| `ml_result` | `dict` | Three-tier: label, probabilities, confirmed, suspicious |
| `quick_screen_hits` | `dict` | `{slither: [...], aderyn: [...]}` |
| `routing_decisions` | `list[str]` (append) | Per-class routing log |
| `static_findings` | `list[dict]` | Slither + Aderyn deep findings |
| `symbolic_findings` | `list[dict]` | Halmos: `{invariant, proven, counterexample}` |
| `evidence_list` | `list[Any]` (append) | P2: all Evidence items, consumed by fuse() |
| `verdict_provable` | `dict[str, str]` | P2: ZK-anchorable tier (deterministic only) |
| `verdict_full` | `dict[str, str]` | P2: human-report tier (all evidence) |
| `tool_status` | `dict` (merge) | Rule 5C: `{tool: {ran, reason}}` per tool |
| `injection_matches` | `list` (append) | P4: detected injection patterns |
| `model_hash` | `str` | P5: SHA-256 of ML checkpoint file |
| `debate_transcript` | `dict[str, str]` | A.4: `{prosecutor, defender, judge}` |
| `final_report` | `dict` | Complete audit output |

See `src/orchestration/state.py` for the full field list.

### Per-Class Routing

All thresholds and rules live in `configs/verdicts_default.yaml`, loaded by
`src/config/loader.py`. `routing.py` reads them via `get_config()`.

| Class | Deep threshold | Tools activated |
|-------|---------------|----------------|
| DenialOfService | 0.30 | static_analysis + rag_research |
| Reentrancy, IntegerUO, Timestamp, TOD | 0.35 | static_analysis + rag_research |
| ExternalBug, CallToUnknown | 0.40 | static_analysis + rag_research |
| GasException, MishandledException | 0.40 | static_analysis only |
| UnusedReturn | 0.45 | static_analysis only |

`graph_explain` and `formal_verification` always join the deep-path fan-out regardless
of class.

### Verdicts

| Source | Possible verdicts |
|--------|------------------|
| Rule-based `compute_verdict()` | CONFIRMED / LIKELY / DISPUTED / SAFE |
| `consensus_engine` (A.6) | CONFIRMED / LIKELY / DISPUTED / SAFE (ML discounted) |
| `cross_validator` debate (A.4) | CONFIRMED / LIKELY / DISPUTED / WATCH / SAFE |
| `fuse()` provable tier | deterministic-evidence-only verdict per class |

### Checkpointing

`SqliteSaver` persists state to `data/checkpoints.db` after every node. Resume from
crash:

```python
result = await graph.ainvoke(
    None,
    config={"configurable": {"thread_id": "audit-001"}},
)
```

## RAG Pipeline

| Item | Value |
|------|-------|
| Source | DeFiHackLabs GitHub (726 `.sol` exploit PoCs) |
| Chunks | ~752 |
| Chunk size | 1536 chars, 128 overlap |
| Embedding | `text-embedding-nomic-embed-text-v1.5` (768-dim) via LM Studio |
| Vector index | FAISS `IndexFlatL2` |
| Keyword index | `BM25Okapi` |
| Fusion | Reciprocal Rank Fusion (RRF_K = 60) |

Full rebuild:
```bash
poetry run python -m src.rag.build_index
```

Incremental update (new docs only):
```bash
poetry run python -m src.ingestion.pipeline
```

**P7 zero-match fix:** If RRF returns 0 results above the score floor, a keyword-only
BM25 pass runs with relaxed thresholds. Closes the "ML says vulnerable, RAG says
nothing" gap.

**WS2 note:** The 5 Phase A.5 corpus fetchers (Code4rena/Sherlock/Solodit/Immunefi/SWC)
are **disabled** — their seed corpora were synthetic placeholders and one caused a
hallucinated verdict. Re-enable with real data per `02_RAG_BUILD_PLAN.md`.

## MCP Servers

| Server | Port | Tools |
|--------|------|-------|
| `inference_server` | 8010 | `predict`, `batch_predict` |
| `rag_server` | 8011 | `search` |
| `audit/` package | 8012 | `get_latest_audit`, `get_audit_history`, `check_audit_exists` |
| `graph_inspector_server` | 8013 | `get_graph_hotspots` |
| `representation_server` | 8014 | `get_embeddings` |

All servers: SSE transport, `/health` endpoint, mock mode for dev/CI.

## Security — Prompt Injection (P4)

Three-layer defense sanitizes contract source before every LLM prompt:

1. **`comment_strip.py`** — state-machine comment stripper (preserves line count)
2. **`prompt_delimit.py`** — `<<CONTRACT_SOURCE>>` structural delimiter + frame
3. **`injection_detect.py`** — 8-pattern scanner (comment/string/role-swap/extraction/
   identifier/NatSpec/multi/import)

`injection_matches` flows through `AuditState` → `final_report["security"]["injection_detections"]`.

## Evaluation (P0/P3)

| Phase | macro_F1 | macro_Fbeta (β=2) |
|-------|----------|-------------------|
| P0 honest baseline | 0.1958 | 0.2515 |
| P2 calibrated | 0.1998 | 0.2246 |
| P3 L3 reliability | **0.3008** | **0.3821** |

Run benchmark:
```bash
poetry run python src/eval/run_benchmark.py --no-llm --output eval/runs/
```

Build L3 reliability weights:
```bash
poetry run python scripts/build_reliability_matrix.py \
    --run-dir eval/runs/<run_id> --output configs/reliability_v3.yaml
```

## LLM Client

Routes to LM Studio (OpenAI-compatible API):

| Role | Model | Use |
|------|-------|-----|
| FAST | `gemma-4-e2b-it` | Simple tasks, API calls |
| STRONG | `gemma-4-e2b-it` | Reasoning, synthesis, reports |
| CODER | `qwen2.5-coder-7b-instruct` | Solidity analysis |
| EMBED | `nomic-embed-text-v1.5` | RAG embeddings |

`SENTINEL_DETERMINISTIC=1` disables all LLM calls and RAG lookups, enabling
reproducible deterministic-mode audits for ZK proof generation.

## Testing

```bash
cd agents
poetry run pytest tests/ -v
# 631 passing, 3 skipped
```

`conftest.py` sets `AGENTS_DISABLE_LLM=1` session-wide. See `tests/README.md` for the
full file list.

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
MCP_REPRESENTATION_PORT=8014
MCP_INFERENCE_URL=http://localhost:8010/sse
MCP_RAG_URL=http://localhost:8011/sse
MCP_AUDIT_URL=http://localhost:8012/sse
MCP_GRAPH_INSPECTOR_URL=http://localhost:8013/sse
MCP_REPRESENTATION_URL=http://localhost:8014/sse

# RAG
RAG_DEFAULT_K=5
AUDIT_RAG_K=5

# Graph Inspector
SENTINEL_ML_API_URL=http://localhost:8001
GRAPH_INSPECTOR_HOTSPOTS_TIMEOUT=60
GRAPH_INSPECTOR_MOCK=false

# Orchestration flags
AGENTS_DISABLE_LLM=        # "1"/"true" → all LLM calls skipped, rule-based fallback
DEBATE_MODE=on             # "off" → cross_validator single-pass instead of 3-role debate
ML_WEIGHT_SCALE=0.5        # discounts ML's consensus vote — ML alone can't CONFIRM
SENTINEL_DETERMINISTIC=1   # disables LLM + RAG, torch deterministic mode (P5 / ZK)

# Timeouts (centralized in src/orchestration/timeouts.py)
CROSS_VALIDATOR_TIMEOUT_S=90    # single-pass only
DEBATE_TIMEOUT_S=240            # full 3-role debate budget
SYNTHESIZER_TIMEOUT_S=120
ADERYN_TIMEOUT_S=90
REFLECTION_TIMEOUT_S=120

# Gateway (P10)
DAGSTER_HOME=agents/.dagster
```

## Do Not Change Without a Wider Plan

- Decision numbers (thresholds, weights) must be changed only with before/after eval
  measurements that justify the change (Rule 5B). "It feels right" is not sufficient.
- Do not add `except Exception: return []` anywhere — surface failures via `tool_status`
  (Rule 5C, CLAUDE.md §C).
- Do not change `chunk_size` or `chunk_overlap` without rebuilding the index.
- Do not re-enable the A.5 corpus fetchers (Code4rena/Sherlock/Solodit/Immunefi/SWC)
  without replacing the synthetic placeholder data with real curated corpora.
- Do not re-enable the P6 model cascade without a prompt or fine-tuning fix — the strong
  model currently over-predicts CONFIRMED on safe contracts.
- Do not use mock-mode audit results as real security evidence.
