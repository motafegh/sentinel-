# Plan: Doc 01 — The Audit Pipeline: How 14 Nodes Process a Contract

**Spec:** `docs/learning/LEARNING_DOCS_SPEC.md`
**Target:** `docs/learning/01_orchestration_pipeline.md`
**Session:** 1 of 5
**Prerequisite docs:** None (this is the foundation)

---

## Recall from previous docs

This is the first document. No previous docs to recall from. This doc IS the foundation that all others build on.

---

## Step 1: Read source files

- [ ] `agents/src/orchestration/graph.py` (280 lines) — graph builder, `_route_from_evidence_router()`, edge wiring, node registration, checkpointer setup
- [ ] `agents/src/orchestration/state.py` (254 lines) — AuditState TypedDict, `Annotated[list, operator.add]` append-reducers, all state fields
- [ ] `agents/src/orchestration/routing.py` — `compute_active_tools()`, `CLASS_TO_DETECTORS` mapping
- [ ] `agents/src/orchestration/nodes/__init__.py` — node registry (15 exports including formal_verification)
- [ ] `agents/src/orchestration/nodes/_helpers.py` — `_llm_enabled()`, `_call_mcp_tool()` helper
- [ ] `agents/src/orchestration/timing.py` — `timed_node()` wrapper, `step_timer()`
- [ ] `agents/src/orchestration/timeouts.py` — timeout env vars, `get_timeout()`

## Step 2: Read scratch files

- [ ] `~/.claude/scratch/system_finalization_statecheck_20260625.md` — P2.5/P3 findings, the dual-wire `_reconcile_shim.py` story, Rule 5C findings, the nodes.py split
- [ ] `~/.claude/scratch/p2_plan_review_20260624.md` — P2 plan review, the pairwise→uniform decision, fuse() vs legacy analysis

## Step 3: Read architecture proposal

- [ ] `docs/proposal/2026-06-23_proposal_SYSTEM_architecture-finalization.md` — §3 (guiding principles), §5 (target architecture), §10 (phased plan)

## Step 4: Write sections (following spec template)

- [ ] **TL;DR:** 14 nodes, LangGraph StateGraph, deterministic routing (not LLM), fail-soft (always produces report), dual-path (fast ~3s / deep ~60s)
- [ ] **The Problem:** Need a deterministic analysis pipeline (not agentic) that always produces a report even when external tools crash
- [ ] **How We Arrived at This Design:** invariant (always produce report) → constraint (routing in code, never LLM) → simplest graph (14 nodes, conditional edges) → stress-test (add Halmos in P8a without changing fuse) → measure (P0 baseline F1=0.1958)
- [ ] **The Solution:** Full graph topology ASCII diagram showing all 14 nodes + edges. State flow diagram showing how `AuditState` accumulates. Routing decision tree (fast path vs deep path). Parallel fan-out/fan-in explanation.
- [ ] **Key Code:**
  - `_route_from_evidence_router()` (graph.py:91-137) — the ONLY routing decision, pure function, two-signal gate
  - `AuditState` TypedDict (state.py) — key fields: `evidence_list` (append-reducer), `model_hash`, `injection_matches`, `tool_status`, `verdict_provable`, `verdict_full`
  - `build_graph()` (graph.py:144-253) — node registration, edge wiring, checkpointer setup
  - `timed_node()` wrapper (timing.py) — uniform START/DONE logging for every node
- [ ] **Design Decision:** LangGraph vs Temporal vs Airflow vs custom DAG (tradeoff table with criteria: latency, complexity, failure mode, scale ceiling, team familiarity)
- [ ] **Technology Choice:** LangGraph (5-question framework: category, alternatives, why this one, when different, migration trigger)
- [ ] **Anti-Patterns:**
  - ❌ "Smart Router" — LLM chooses which nodes to run. Tempting because sounds efficient. Breaks: non-deterministic, injectable, unverifiable, undebuggable. Right approach: routing in code (Principle 2)
  - ❌ "God Function" — one function does everything (`def analyze_contract(code): ...`). Tempting because simple. Breaks: no parallelism, no fail-soft, can't add channels. Right approach: Evidence model + fuse()
- [ ] **Mistakes & Fixes:**
  - `_reconcile_shim.py` dual-wire: two verdict systems (legacy consensus + fuse()) running in parallel, producing conflicting verdicts. Fix: Shape A flip — delete shim, fuse() is sole producer
  - `asyncio.to_thread` blocking event loop: blocking subprocess calls in async nodes freezing the loop. Fix: proper async patterns with `run_in_executor`
  - `sys.path` bug: CWD-dependent imports. Fix: `sys.path.insert(0, str(Path(__file__).resolve().parents[3]))` in every node file
- [ ] **What Would Break Without This:** Remove routing → fast path gone, every contract takes 60s, safe contracts get false positives. Remove state reducers → evidence doesn't accumulate. Remove checkpointer → no crash recovery
- [ ] **At Scale:** 61 contracts (current, ~3s fast / ~60s deep) / 610 (LLM latency dominates) / 6,100 (need parallel gateway workers) / 61,000 (need distributed workers — Temporal)
- [ ] **Try It Yourself:**
  ```
  cd agents && source .venv/bin/activate
  python -c "from src.orchestration.graph import build_graph; g = build_graph(use_checkpointer=False); print(type(g))"
  ```
- [ ] **Limitations:** Single-process (no distributed workers), SqliteSaver optional (falls back to MemorySaver silently), no job cancellation, no mid-graph resume on gateway path
- [ ] **Transferable Patterns:** (1) Fail-soft design (2) Control-flow determinism (routing in code not LLM) (3) State-machine workflows with append-reducers (4) Conditional routing as pure functions. Each with interview story + when wrong.

## Step 5: Verify

- [ ] Open `graph.py` and verify every line number cited
- [ ] Open `state.py` and verify every field mentioned
- [ ] Confirm 14 nodes listed (not 13 — `formal_verification` added in P8a)
- [ ] Confirm the node list in `__init__.py` matches the doc
- [ ] Verify the ASCII diagram shows the correct edge from `formal_verification` → `audit_check`
