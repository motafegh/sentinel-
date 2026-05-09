# 2026-05-09 — M4 (Agents / RAG / LangGraph) Plan

Spec ref: `docs/Project-Spec/SENTINEL-M4-AGENTS.md`,
`docs/Project-Spec/SENTINEL-CONSTRAINTS.md`.
Status ref: `docs/STATUS.md`.

---

## 1. Current State (verified from source)

```
agents/src/
  orchestration/
    graph.py          StateGraph: ml_assessment → routing → (rag/static) → audit_check → synthesizer
                      MemorySaver checkpoint after every node
                      _route_after_ml fans out parallel branches for high-risk
    nodes.py          ml_assessment / rag_research / audit_check / synthesizer
                      MCP clients (short-lived SSE per call)
                      static_analysis node "added in M6" — NOT YET WIRED
                      _is_high_risk threshold = 0.70 on max(per-class probability)
    state.py          AuditState typed dict
  rag/
    retriever.py      FAISS + BM25 + RRF (FIX-5/-10/-12/-23 applied)
    chunker.py / embedder.py / build_index.py
    fetchers/github_fetcher.py / base_fetcher.py
  ingestion/
    feedback_loop.py  On-chain → RAG re-index loop (FIX-1..6 + FIX-BugA + Issue #1 bridge)
    pipeline.py       REPORTS_DIR shared constant
    deduplicator.py / scheduler_cron.py / scheduler_dagster.py
  llm/client.py       LM Studio + 4 model map (FAST/STRONG/CODER/EMBED)
                      LM_STUDIO_BASE_URL via .env, 60s timeout
  mcp/servers/
    inference_server.py  SSE :8010 — wraps M1 /predict
    rag_server.py        SSE :8011 — wraps RAG retriever
    audit_server.py      SSE :8012 — wraps AuditRegistry (read-only currently)

agents/scripts/
  smoke_inference_mcp.py / smoke_rag_mcp.py / smoke_audit_mcp.py
  smoke_langgraph.py    End-to-end smoke
  test_k_cap.py
```

Cross-encoder reranking (T3-B) implemented: `rerank=False` default,
`ms-marco-MiniLM-L-6-v2`. LLM synthesizer (T3-A): qwen3.5-9b-ud with
rule-based fallback.

---

## 2. Plan A — Close Visible Source Gaps

### 2.1 `static_analysis` node

`agents/src/orchestration/nodes.py` references a `static_analysis` node
(`graph.py` imports it; doc comment says "added in M6"). **Verify it
actually exists** and runs Slither / Mythril; if it is a stub, either:

- (a) implement it before M6 ships (Slither only is enough for v1)
- (b) drop `static_analysis` from `_route_after_ml` parallel branch and
  re-introduce in M6

Acceptance: `agents/scripts/smoke_langgraph.py` end-to-end produces
identical state-keys regardless of whether the deep path includes static
analysis.

### 2.2 Dagster RAG schedule

`agents/src/ingestion/scheduler_dagster.py` currently exists; verify:
- Schedule is registered against the Dagster instance loaded by an
  obvious entry point (`dagster dev` or `Definitions` object)
- A `RunFailureSensor` or equivalent emits a Prometheus / log alert
- Documented in `agents/README.md` (start command + what it ingests)

If any missing, add a follow-up file:
`agents/src/ingestion/__init__.py` — `definitions` object exporting
the schedule.

### 2.3 `submitAudit` write path on `audit_server.py`

Source comment says "submitAudit added after Track 3" — verify whether
this is now wired. If not, this becomes a soft-blocker for end-to-end
M5/M2 verification flow:

- Decide: add the write tool now, or expose it only via M6 once ZK
  proofs land (recommend the latter — write path needs ZK proof, which
  comes from M2 Option A)

### 2.4 `feedback_loop.py` reports bridge

Issue #1 fix introduced `data/reports/{contract_address}.json` as a
sidecar between synthesizer and feedback_loop. Confirm:
- `REPORTS_DIR` is created at import time (no runtime mkdir race)
- Synthesizer writes atomically (`.tmp` + `Path.replace` — same
  pattern as FIX-BugA in feedback_loop)

---

## 3. Plan B — Operational Hardening

### 3.1 MCP server lifecycle

Currently each node opens a short-lived SSE connection per call (see
`nodes.py` "MCP client pattern" docstring). For M6 latency this becomes
a bottleneck.

Action: introduce `agents/src/mcp/client_pool.py` — a module-level
async client cache keyed by URL. Switch `nodes._call_mcp_tool()` to
use it. Concurrency-safe (LangGraph runs nodes in asyncio).

This is in scope for **after** the LangGraph orchestration is
otherwise stable; do not block M6 on it.

### 3.2 LangGraph checkpointer choice

`MemorySaver` is fine for development; for M6 the orchestrator behind a
gateway needs `SqliteSaver` (or Postgres if M6 adds it).

Action: parameterise `build_graph(checkpointer=...)` so M6 can inject a
durable checkpointer without forking the module.

### 3.3 Retrieval correctness regression tests

Existing tests in `agents/tests/`:
- `test_retriever_filters.py`, `test_chunker.py`, `test_deduplicator.py`,
  `test_graph_routing.py`, `test_github_fetcher.py`,
  `test_audit_server.py`, `test_inference_server.py`

Gaps:
- No test for cross-encoder rerank ON vs OFF producing different orderings
- No test for FIX-10 (FAISS↔chunks sync detection) — write a test that
  corrupts the index pickle and asserts the retriever raises
- No test for `_is_high_risk` threshold boundary (0.70) — write
  parametric test for {0.69, 0.70, 0.71}

---

## 4. Plan C — LLM Routing & Prompt Discipline

`agents/src/llm/client.py` defines a 4-model map (FAST/STRONG/CODER/
EMBED). Unclear from source which agent uses which today.

Action: audit each call site (`grep -r "ChatOpenAI\|chat_completion" agents/src/`)
and explicitly select a model. Document the mapping in a new file:

```
agents/MODELS.md       NEW
  FAST   — used by: ml_assessment routing decisions
  STRONG — used by: synthesizer, rag_research summarisation
  CODER  — used by: static_analysis (when implemented)
  EMBED  — used by: rag.embedder, ingestion.pipeline
```

Acceptance: every `ChatOpenAI(...)` instantiation in `agents/src/`
explicitly names its model alias; no implicit defaults.

---

## 5. Plan D — Documentation Sync

Files to keep in sync once §2–§4 land:

- `agents/README.md` — start commands, ports, env vars
- `docs/Project-Spec/SENTINEL-M4-AGENTS.md` — append section listing
  what is wired vs planned (current spec is general; needs source-aware
  truth table)

---

## 6. Acceptance Criteria

- `smoke_langgraph.py` PASS end-to-end including static_analysis branch
  (or static_analysis removed from `_route_after_ml`, with ROADMAP entry)
- Dagster schedule visible from `dagster dev`; one successful tick logged
- `client_pool.py` reduces median MCP roundtrip by ≥ 30 % (measure with
  `smoke_inference_mcp.py --bench`)
- New tests for cross-encoder, FAISS sync, and `_is_high_risk` boundary
  pass in `pytest agents/tests/`
- `agents/MODELS.md` exists; `grep -nE "ChatOpenAI|OpenAIEmbeddings"
  agents/src/` shows every instance has an explicit alias
