# Agents Extended Capability — Phase A COMPLETE (2026-06-21)

**Plan:** `docs/plan/agents/2026-06-17-extended-capability/01_PHASE_A_EXECUTION_PLAN.md`
**Status:** ✅ All 9 Phase-A items (A.1–A.9) implemented, wired, and tested.
**Tests:** 219 baseline → **276 passing** (+57 new). Full graph runs end-to-end.

---

## What was built

| Item | Deliverable | Files |
|------|-------------|-------|
| **A.1** | Lazy `audit_graph` (PEP 562 `__getattr__`) — no import-time graph build | `orchestration/graph.py` |
| **A.2** | 10 new optional `AuditState` fields (Phase A + Phase B placeholders) | `orchestration/state.py` |
| **A.3** | `reflection` node — self-critique (rule-based + optional LLM summary) | `orchestration/nodes.py` |
| **A.4** | `cross_validator` upgraded to **Prosecutor→Defender→Judge** debate (reads source) | `orchestration/nodes.py` |
| **A.5** | 5 RAG fetchers (Code4rena/Sherlock/Solodit/Immunefi/SWC) + curated corpora + index wiring | `rag/fetchers/*`, `data/knowledge/*`, `rag/build_index.py` |
| **A.6** | `consensus.py` weighted voting + `consensus_engine` node | `orchestration/consensus.py`, `nodes.py` |
| **A.7** | `confidence.py` Bayesian staged confidence tracking | `orchestration/confidence.py` |
| **A.8** | `attribution.py` LIME-style breakdown + `explainer` node | `orchestration/attribution.py`, `nodes.py` |
| **A.9** | `visualizer.py` self-contained interactive hotspot HTML + `visualizer` node | `orchestration/visualizer.py`, `nodes.py` |

## Graph topology (new)

```
START → ml_assessment → quick_screen → evidence_router
  ├─ deep → [rag_research | static_analysis | graph_explain] → audit_check
  │         → consensus_engine → cross_validator → synthesizer
  └─ fast ─────────────────────────────────────────→ synthesizer
synthesizer → reflection → explainer → visualizer → END
```

## Ali directive baked in (2026-06-21): "ML is a HINT, not authority"

Run 12 ML is not yet reliable, so the agent layer does its own analysis and
down-weights ML:
- **`ML_WEIGHT_SCALE`** env (default **0.5**) discounts every class's ML reliability
  weight in `consensus.py`. Result: **ML alone can never reach CONFIRMED** for any
  class — corroboration from Slither/Aderyn (and the LLM debate) is required.
- The **debate** now feeds the **contract source** to Prosecutor/Defender so the LLM
  forms an independent judgement from code, not from the ML/tool summary.
- `reflection` explicitly flags ExternalBug over-prediction and ML-only verdicts as
  failure modes.

Verified: `consensus_vote(0.99, slither=False, aderyn=False, "Reentrancy") → SAFE`;
adding one strong tool → escalates.

## Testing & determinism

- New `tests/conftest.py` sets `AGENTS_DISABLE_LLM=1` for the session so the suite is
  fast/deterministic and never depends on a live LM Studio. LLM-calling nodes
  (`cross_validator`, synthesizer narrative, `reflection`) consult `_llm_enabled()`.
- `cross_validator` LLM-path unit tests re-enable LLM locally (autouse fixture).
- New test files: `test_consensus_voting.py`, `test_confidence_tracking.py`,
  `test_metric_attribution.py`, `test_reflection.py` (incl. debate),
  `test_visualizer.py`, `test_rag_fetchers.py`.

## Env / config touched

- `agents/.env`: `LM_STUDIO_BASE_URL` → `http://127.0.0.1:12345/v1` (live LM Studio).
- New env knobs: `AGENTS_DISABLE_LLM`, `DEBATE_MODE` (default on), `ML_WEIGHT_SCALE`
  (default 0.5), `REFLECTION_TIMEOUT_S`, `REFLECTION_MAX_TOKENS`.

## Not in scope this session (blocked / deferred)

- **Phase B** (Halmos/Z3/Gigahorse) and **Phase D** (ItyFuzz) — external binaries
  **not installed**; schema placeholders added (`symbolic_findings`, `bytecode_analysis`,
  `taint_flows`, `permission_graph`). **Phase C** (FastAPI gateway/eval/guards) — next.
- RAG corpora ship as **curated seed JSON** (offline, testable); replace with full
  exports + live fetch for production scale.
