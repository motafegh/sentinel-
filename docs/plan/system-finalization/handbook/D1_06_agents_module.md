> **Superseded v1 plan:** retained for history. Use [D1 v2](../D1_developer_handbook.md) and the implemented [AGENTS chapters](../../../handbook/09_agents_orchestration.md).

# D1.2e — Agents Module Doc

**Doc target:** `docs/handbook/06_agents_module.md`
**Estimated time:** 1.5h (largest module — 20 source files)
**Rule:** Every claim verified against source code.

---

## Source files to read before writing (20 files)

### Orchestration core (5 files)
1. `agents/src/orchestration/graph.py` — LangGraph definition
   - Extract: all node names, edge connections, conditional routing logic
   - Fast path vs deep path: which nodes fire on which path
   - Entry point and terminal node

2. `agents/src/orchestration/state.py` — AuditState TypedDict
   - Extract: all state fields with their types and comments
   - Key fields: contract_code, contract_address, ml_result, model_hash, tool_status, verdict_provable, verdict_full, final_report, on_chain

3. `agents/src/orchestration/routing.py` — routing logic
   - What determines fast vs deep path
   - Verify: no LLM imports, no contract_code access (routing isolation)

4. `agents/src/orchestration/timeouts.py` — centralized timeout defaults
   - CROSS_VALIDATOR_TIMEOUT_S, DEBATE_TIMEOUT_S, SYNTHESIZER_TIMEOUT_S

5. `agents/src/orchestration/attribution.py` — vulnerability attribution

### Pipeline nodes (7 files)
6. `agents/src/orchestration/nodes/quick_screen.py` — Tier-0 screen
   - Slither + Aderyn high-impact detector scan
   - tool_status for both slither and aderyn (Rule 5C)
   - Empty contract code early return with tool_status

7. `agents/src/orchestration/nodes/static_analysis.py` — full static analysis
   - Slither with all detectors + Aderyn + external call extraction
   - tool_status for slither and aderyn
   - ImportError and Exception paths return tool_status

8. `agents/src/orchestration/nodes/ml_assessment.py` — ML API call
   - Calls MCP inference tool → ML API /predict
   - Extracts model_hash from response → state["model_hash"]
   - Three-tier result: confirmed/suspicious/safe

9. `agents/src/orchestration/nodes/rag_research.py` — RAG retrieval
   - Hybrid retrieval: FAISS (semantic) + BM25 (lexical)
   - SENTINEL_DETERMINISTIC skip
   - Returns rag_results list

10. `agents/src/orchestration/nodes/cross_validator.py` — LLM debate
    - 3-role debate: Prosecutor, Defender, Judge
    - Prompt sanitization applied before debate (P4)
    - Cascade: ambiguous classes re-judged by strong model (P6, disabled by default)

11. `agents/src/orchestration/nodes/synthesizer.py:344-425` — final report builder
    - final_report dict: 29+ fields including model_provenance and on_chain
    - on_chain section: submitted, tx_hash, proof_hash, class_scores, provenance
    - Report persisted to disk for feedback_loop

12. `agents/src/orchestration/nodes/formal_verification.py` — Halmos
    - Symbolic execution via Halmos
    - Foundry test harness generation
    - tool_status with ran/reason/detail

### Verdict system (4 files)
13. `agents/src/orchestration/verdict/evidence.py` — Evidence dataclass
    - Polarity enum: SUPPORTS, REFUTES, NEUTRAL
    - Kind enum: STATISTICAL, SYNTACTIC, SEMANTIC, FORMAL, ECONOMIC
    - 7 constructors: ml, slither, aderyn, rag, debate, quick_screen, formal
    - 8 fields: kind, source, polarity, strength, reliability, deterministic, detail, vulnerability_class

14. `agents/src/orchestration/verdict/fuse.py` — fuse() verdict producer
    - FAMILIES dict: 13 source names → 6 families (ML, STATIC_SYNTAX, RAG, LLM_DEBATE, FORMAL, ECONOMIC)
    - Family discount: 1/N per family
    - Signed sum: discounted_rel × strength × polarity
    - Bands: confirmed/likely/disputed (from config)
    - Asymmetry override: strong SUPPORTS + verdict==SAFE → DISPUTED
    - Dual verdict: verdict_provable (deterministic only) + verdict_full (all evidence)

15. `agents/src/orchestration/verdict/reliability.py:73-140` — reliability weights
    - L3→L1→L0 fallback chain:
      - L3: reliability_v3.yaml (Bayesian shrinkage fitted, schema_version=1)
      - L1: verdicts_default.yaml (hand-set per-class accuracy_weights)
      - L0: hardcoded defaults (rag=0.50, debate=0.55, quick_screen=0.40)
    - ML weight always scaled by ml_weight_scale

16. `agents/src/orchestration/verdict/emit.py` — 7 emit functions
    - emit_ml, emit_static, emit_rag, emit_debate, emit_quick_screen, emit_halmos, emit_consensus
    - Each converts tool output → Evidence dataclass

### API + MCP (3 files)
17. `agents/src/api/gateway.py:212-322,328-403` — FastAPI gateway
    - POST /audit: accepts AuditRequest, creates JobRecord, spawns _run_job background task
    - GET /audit/{job_id}: polls job status
    - GET /audit: lists recent jobs
    - GET /health: service status + job counts + service probes
    - _run_job: graph.ainvoke(initial_state) → store.mark_completed

18. `agents/src/api/sqlite_job_store.py` — SQLite JobStore
    - Persists jobs to SQLite (data/jobs.db)
    - Crash recovery: recover_pending() marks RUNNING → FAILED on startup
    - Background health monitor: probes 6 services every 30s

19. `agents/src/mcp/servers/audit/_handlers.py:45-147` — 4 MCP tools
    - get_latest_audit(contract_address) — view
    - get_audit_history(contract_address, limit) — view
    - check_audit_exists(contract_address) — view
    - submit_audit(source_code, contract_address, model_hash) — write (P11)

20. `agents/src/mcp/servers/audit/_submit.py` — on-chain submission pipeline
    - Step 1: GET /fusion-embedding from ML API
    - Step 2: Run proxy model locally → 10 class scores
    - Step 3: Generate EZKL proof inline (gen_witness → prove → verify)
    - Step 3b: Build provenance manifest (EIP-191 signed)
    - Step 4: Submit on-chain via web3.py transact
    - class_score_felts overwritten from proof's publicSignals (binding guarantee)

### Config (1 file)
21. `agents/src/config/loader.py, schema.py` — config externalization
    - verdicts_default.yaml loaded via Pydantic schema
    - Wired to: consensus.py, confidence.py, routing.py, attribution.py, pipeline_metrics.py

---

## Sections to write

**1. TL;DR** (5 lines)
```
What: LangGraph 14-node audit pipeline + 4 MCP servers + FastAPI gateway
Nodes: quick_screen → static_analysis → ml_assessment → rag_research → cross_validator → synthesizer
Verdict: fuse() produces dual verdict (provable + full) from Evidence objects
Tests: cd agents && poetry run pytest (634 passed)
```

**2. Pipeline overview** (~1.5 pages)
- Node list with one-line descriptions (verify each name from `graph.py`):
  - quick_screen: Tier-0 Slither+Aderyn high-impact scan
  - static_analysis: full Slither+Aderyn+external calls
  - ml_assessment: ML API call, 3-tier result
  - rag_research: hybrid FAISS+BM25 retrieval
  - cross_validator: 3-role LLM debate
  - formal_verification: Halmos symbolic execution
  - audit_check: on-chain prior audit lookup
  - graph_explain: GNN explanation
  - explainer: human-readable explanation
  - consensus_engine: consensus evidence
  - reflection: LLM self-critique
  - visualizer: graph visualization
  - synthesizer: final report + on-chain section
  - evidence_router: routes evidence to verdict
- Fast path vs deep path (verify from `routing.py`):
  - Fast: quick_screen → ml_assessment → synthesizer (when contract is clearly safe)
  - Deep: all nodes (when contract has suspicious findings)
- ASCII diagram of graph flow with conditional edges

**3. Evidence + Verdict system** (~1 page)
- Evidence dataclass (verify from `evidence.py:15-26`):
  - 5 kinds: STATISTICAL, SYNTACTIC, SEMANTIC, FORMAL, ECONOMIC
  - 3 polarities: SUPPORTS, REFUTES, NEUTRAL
  - 8 fields: kind, source, polarity, strength, reliability, deterministic, detail, vulnerability_class
  - 7 constructors: ml, slither, aderyn, rag, debate, quick_screen, formal
- fuse() (verify from `fuse.py:24-38,93-139`):
  - FAMILIES: 13 sources → 6 families
  - Family discount: 1/N (same-family evidence doesn't double-count)
  - Signed sum: Σ(discounted_rel × strength × polarity), clamped [0,1]
  - Bands: confirmed/likely/disputed (from verdicts_default.yaml)
  - Asymmetry override: strong SUPPORTS + SAFE → DISPUTED
  - Dual verdict: verdict_provable (deterministic=True only) + verdict_full (all)
- Reliability (verify from `reliability.py:73-140`):
  - L3: reliability_v3.yaml (fitted, schema_version=1)
  - L1: verdicts_default.yaml (hand-set)
  - L0: hardcoded defaults
  - Fallback: L3 → L1 → L0

**4. MCP servers** (~1 page)
- 4 servers (verify ports from `.env`):
  - Inference (8010): proxies ML API /predict to agents
  - Audit (8012): 4 tools — 3 read-only (get_latest, get_history, check_exists) + 1 write (submit_audit)
  - RAG: retrieval interface
  - Representation (8014): graph extraction
- Audit server tools (verify from `_handlers.py:45-147`):
  - submit_audit: full pipeline (fusion → proxy → proof → transact)
  - Returns structured degraded return on any failure (Rule 5C)

**5. Gateway** (~0.5 page)
- Endpoints (verify from `gateway.py:212-322`):
  - POST /audit: 202 Accepted, job enqueued
  - GET /audit/{job_id}: job status + result
  - GET /audit: list recent jobs
  - GET /health: service status
- SQLite JobStore (verify from `sqlite_job_store.py`):
  - Persists to data/jobs.db
  - Crash recovery: recover_pending() on startup
  - Health monitor: probes 6 services every 30s

**6. Final report structure** (~0.5 page)
- Verify all fields from `synthesizer.py:344-386`:
  - contract_address, overall_label, overall_verdict, risk_probability
  - confirmed, suspicious, vulnerabilities, probabilities, tier_thresholds
  - vulnerability_verdicts, threshold, ml_truncated, num_nodes, num_edges
  - rag_evidence, audit_history, static_findings, external_call_summary
  - routing_decisions, consensus_verdict, debate_transcript
  - recommendation, narrative, error, path_taken
  - security.injection_detections
  - model_provenance: model_hash, checkpoint_path, schema_version
  - on_chain: submitted, tx_hash, proof_hash, class_scores, class_score_felts, model_hash, provenance

**7. Deep reference**
- → `docs/learning/01_orchestration_pipeline.md` (deep dive on 14-node pipeline)
- → `docs/learning/02_evidence_model_fuse.md` (deep dive on Evidence + fuse)
- → `docs/learning/05_mcp_architecture.md` (deep dive on MCP servers)
- → `docs/learning/07_gateway_production.md` (deep dive on gateway + JobStore)
- → source: all files listed above

---

## Verification checklist
- [ ] Node count and names match `graph.py` exactly
- [ ] MCP tool count = 4 (3 read + 1 write)
- [ ] final_report fields match `synthesizer.py:344-386` dict keys
- [ ] Evidence kinds = 5 (STATISTICAL, SYNTACTIC, SEMANTIC, FORMAL, ECONOMIC)
- [ ] Evidence polarities = 3 (SUPPORTS, REFUTES, NEUTRAL)
- [ ] FAMILIES dict has 13 source names → 6 families (verify from `fuse.py:24-38`)
- [ ] Reliability fallback chain: L3 → L1 → L0 (verify from `reliability.py:73-140`)
- [ ] Gateway endpoints match `gateway.py` route decorators
- [ ] `poetry run pytest` produces 634 passed
