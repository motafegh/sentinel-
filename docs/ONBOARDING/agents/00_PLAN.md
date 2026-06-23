# Onboarding Plan — Agents Module

**Session started:** 2026-06-22
**Style:** teach-onboarding (over-the-shoulder class, Frame → Execute → Checkpoint)
**Artifacts:** this folder
**Current state:** WS0-WS5 + WS6a/C.2 done; WS6a/C.1 (FastAPI gateway) is the next implementation step.

---

## Goal

By the end of this onboarding, you can:

1. **Read the 13-node graph top-to-bottom** — know what runs when, in what order, and which MCP calls fire.
2. **Trace any contract through the system** — follow `AuditState` field by field from `contract_code` in to `final_report` out.
3. **Know which module owns what** — given a behavior, name the file + the test that covers it.
4. **Recognize the design boundaries** — additive (Phase A) vs planned (Phase B/C/D) vs the "do not change" list at the bottom of `agents/README.md`.

This is a **walkthrough**, not an implementation. No code is being changed.

---

## Proposed session order

The agents module has natural seams. Walk in this order — each session builds on the previous.

| # | Session | What we'll cover | Why this order |
|---|---------|------------------|----------------|
| 1 | **Map & skeleton** | 13-node graph topology, `AuditState` schema, `graph.py` + `state.py`, where the fast/deep split happens. | Anchors the whole system in your head before opening any leaf file. |
| 2 | **Routing + the per-class matrix** | `routing.py` (DEEP_THRESHOLDS, ROUTING_RULES, CLASS_TO_DETECTORS), `compute_verdict`, the three verdict scales. | Explains *why* certain contracts go deep and how the agent decides what tools to fire. |
| 3 | **Evidence-gathering nodes** | `ml_assessment`, `quick_screen`, `rag_research`, `static_analysis`, `graph_explain`, `audit_check` — what each does, what MCP server it calls, what state it writes. | These are the "input" nodes. After this you can name the tool behind any field in `AuditState`. |
| 4 | **Decision + synthesis nodes** | `consensus_engine` (A.6), `cross_validator` (A.4 debate), `synthesizer` — how the weighted vote works, the Prosecutor/Defender/Judge sequence, how the final report is assembled. | This is where the agent's "intelligence" actually lives. Highest value to understand for an AI/ML engineer interview. |
| 5 | **Enrichment nodes** | `reflection` (A.3), `explainer` (A.8), `visualizer` (A.9) — self-critique, attribution, hotspot HTML. | These are the "post-processing" chain that turns a verdict into a presentable audit. |
| 6 | **RAG pipeline** | `fetchers/` (7 of them), `chunker.py`, `embedder.py`, `build_index.py`, `retriever.py` (FAISS+BM25+RRF), `ingestion/pipeline.py`, `feedback_loop.py`. | Explains where the 752 chunks come from and how the system stays current with on-chain audits. |
| 7 | **MCP layer** | All 5 servers (inference, rag, audit, graph_inspector, representation), the mock-mode pattern, SSE transport, why each is a separate process. | This is the architectural pattern that makes the agent pluggable. |
| 8 | **LLM client + 4 model roles** | `llm/client.py`, FAST/STRONG/CODER/EMBED, when each is called, timeouts, the `AGENTS_DISABLE_LLM=1` rule-based fallback. | Knows the cost/quality trade-offs. |
| 9 | **Eval framework (WS6a/C.2)** | `src/eval/` library (`PipelineMetrics`, `Benchmark`, `RegressionBaseline`), `scripts/eval_benchmark.py` refactor, the macro=0.2841 / micro=0.3294 baseline. | The newest piece. Worth a walk-through because it sets the bar for any future node's "did it help?" check. |
| 10 | **Tests as the second source of truth** | How `conftest.py` disables LLM, the 23 test files' coverage map, how to run a single test, the `TestCrossValidatorNode` autouse fixture pattern. | After this, you can answer "where is X tested?" in <30 seconds. |

---

## Pace and depth

- **Per session:** typically 30-60 min. One Frame + one or two meaningful steps + a Checkpoint each.
- **Mechanical steps** (e.g. "open `retriever.py`, count its lines") get a one-liner, not the full loop.
- **Architectural choices** (e.g. "why ML is down-weighted via `ML_WEIGHT_SCALE=0.5`") get the full Frame with the *what-else-could-have-worked* contrast.

---

## Where we are right now

You said **"lets go for agents module"** — that means we start at **Session 1: Map & skeleton**.

Before we open any file, I want your sign-off on the order above. If you'd rather:
- Start with the **decision nodes (Session 4)** because that's the "intelligence" core,
- Or **skip RAG (Session 6)** and come back later,
- Or **collapse 3+4 into one** because the evidence nodes make more sense next to the decision nodes that consume them,

…just say so. Otherwise, confirm the order and I'll start Session 1.

---

## Artifacts this folder will hold

- `00_PLAN.md` — this file
- `01_map_and_skeleton.md` — written after Session 1
- `02_routing.md`
- `03_evidence_nodes.md`
- `04_decision_synthesis.md`
- `05_enrichment.md`
- `06_rag_pipeline.md`
- `07_mcp_layer.md`
- `08_llm_client.md`
- `09_eval_framework.md`
- `10_tests.md`
- `99_session_wrap_up.md` — the end-of-plan master recap (§5 of the spec)

Each session file is a **walkthrough note** (what we read, what we learned, what surprised us), not a tutorial you hand to someone else. The session wrap-up at the end is the shareable artifact.
