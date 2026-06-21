# Tests — Unit & Integration

Pytest-based unit and integration tests for the agents module. All tests run without live MCP servers, LM Studio, or Sepolia RPC. Mocks isolate each layer for fast, deterministic testing.

## Files

| File | Purpose |
|------|---------|
| `test_graph_routing.py` | Routing logic, all node paths, graph compilation, full graph integration |
| `test_smoke_e2e.py` | End-to-end smoke tests — deep/fast/screen-escalated/ML-failure paths |
| `test_audit_server.py` | On-chain history decoding, mock mode, address validation, hard caps |
| `test_inference_server.py` | MCP tool schemas, mock/live transport, batch predict, partial failure |
| `test_routing_phase0.py` | Phase 0 routing — per-class thresholds, tool matrix, verdict logic |
| `test_retriever_filters.py` | FAISS+BM25+RRF filter behaviour, score validation, sync validation |
| `test_github_fetcher.py` | DeFiHackLabs parsing (3 comment formats, FIX-20/21/22b) |
| `test_deduplicator.py` | SHA256 hash deduplication, persistence, checkpoint pattern |
| `test_chunker.py` | Chunk size, overlap, metadata inheritance, edge cases |
| `test_consensus_voting.py` | **(A.6, 2026-06-21)** Weighted vote, ML-weight discount, `consensus_engine` node |
| `test_confidence_tracking.py` | **(A.7)** Bayesian confidence updating, bounds, bands |
| `test_metric_attribution.py` | **(A.8)** LIME-style attribution, `explainer` node, report folding |
| `test_reflection.py` | **(A.3/A.4)** Reflection (rule-based + LLM) and debate (3-role mock) |
| `test_visualizer.py` | **(A.9)** Hotspot HTML generation, escaping, `visualizer` node |
| `test_rag_fetchers.py` | **(A.5)** Code4rena/Sherlock/Solodit/Immunefi/SWC fetchers |
| `test_static_analysis_real_slither.py` | **(2026-06-21)** REAL (non-mocked) Slither — would have caught the detector-registration bug |
| `test_static_analysis_real_aderyn.py` | **(2026-06-21)** REAL (non-mocked) Aderyn — would have caught the dir/output-path/schema bugs |
| `test_timeouts_and_timing.py` | **(2026-06-21)** Centralized timeout config + `step_timer`/`timed_node` |
| `conftest.py` | **(2026-06-21)** Sets `AGENTS_DISABLE_LLM=1` session-wide for determinism |
| `__init__.py` | — Package marker |

**Total: 297 tests passing** (219 baseline → 276 after Extended Capability Phase A,
2026-06-21 — see `docs/changes/2026-06-21-agents-phase-a-extended-capability.md` —
→ 297 after the manual-verification bug fixes + timeout centralization, same day —
see `docs/changes/2026-06-21-agents-manual-verification-real-bugs-found.md`).

LLM-calling nodes (`cross_validator`, synthesizer narrative, `reflection`) consult
`_llm_enabled()` in `src/orchestration/nodes.py`, which reads `AGENTS_DISABLE_LLM`.
`conftest.py` sets this for the whole session so tests never depend on a live LM
Studio. Tests that specifically exercise the LLM path (e.g. `TestCrossValidatorNode`)
re-enable it locally via an autouse fixture and mock the LLM call.

## Running

```bash
cd agents
poetry run pytest tests/ -v                    # all tests
poetry run pytest tests/test_graph_routing.py -v  # specific file
poetry run pytest tests/ -k "not slow" -v      # exclude slow tests
```

## Test Details

### test_graph_routing.py (931 lines)

The largest test file — covers the full orchestration layer.

| Class | Tests | What it covers |
|-------|-------|----------------|
| `TestRouteFromEvidenceRouter` | 10 | Conditional routing: deep/fast path, quick_screen escalation |
| `TestBuildGraph` | 3 | Graph compiles, correct nodes present |
| `TestMlAssessmentNode` | 4 | Happy path, MCP error, exception, empty contract |
| `TestRagResearchNode` | 5 | List/dict responses, errors, confirmed class query, ExternalBug |
| `TestAuditCheckNode` | 4 | Happy path, missing address, exception, error dict |
| `TestSynthesizerNode` | 7 | Deep/fast/suspicious/safe paths, ML failure, truncated, required fields |
| `TestFullGraphIntegration` | 3 | Deep/fast/ML-failure end-to-end with all nodes mocked |
| `TestGraphExplainNode` | 6 | Hotspots, graph explanations, errors, flagged classes |
| `TestCrossValidatorNode` | 5 | No flagged classes, LLM failure, happy path, invalid verdict, markdown fences |
| `TestSynthesizerUsesPreComputedVerdicts` | 1 | Cross-validator verdicts flow through to synthesizer |
| `TestQuickScreenNode` | 8 | Empty contract, Slither not installed, non-fatal errors, High-impact finding |

### test_smoke_e2e.py (375 lines)

Integration tests running the full graph topology with all nodes. Slither runs in-process (or skipped gracefully). All MCP calls mocked.

| Test | What it verifies |
|------|------------------|
| `test_deep_path_vault_produces_final_report` | Reentrancy contract → deep path → all required fields |
| `test_quick_screen_hits_in_final_state` | `quick_screen_hits` always present after graph run |
| `test_deep_path_graph_explanations_present` | `graph_explanations` non-empty after deep path |
| `test_routing_decisions_logged` | `routing_decisions` populated with class info |
| `test_fast_path_safe_contract` | Safe contract → fast path → no RAG, no graph_explanations |
| `test_screen_escalated_path_when_ml_safe_but_screen_fires` | ML safe + Slither hit → deep path (not fast) |
| `test_ml_failure_still_produces_report` | ML unavailable → report still produced with error |

### test_audit_server.py (345 lines)

Tests tool handlers directly (no HTTP, no SSE). Registry object mocked.

- **Schema tests**: 3 tools declared, `contract_address` required
- **Address validation**: valid, lowercase, garbage
- **Score decoding**: score = field_element / 8192, label logic, required fields, proof_hash format
- **get_latest_audit**: mock mode, bad address, live no-audit, live RPC error
- **get_audit_history**: mock mode, limit enforcement, live empty
- **check_audit_exists**: mock mode, bad address, live mode
- **Hard cap**: limit capped at 50 even with limit=999

### test_inference_server.py (336 lines)

Tests tool handlers and mock prediction logic.

- **Tool registration**: 2 tools, required fields, optional fields
- **Mock prediction**: safe contract → safe, reentrancy → high risk, three-tier schema structure
- **_handle_predict**: TextContent return type, valid JSON, address forwarding, HTTP error handling
- **_handle_batch_predict**: processes all, index field, partial failure continues, size cap enforcement
- **call_tool dispatcher**: routes predict, routes batch_predict, unknown name → error

### test_routing_phase0.py (345 lines)

Phase 0 routing logic — per-class thresholds, tool matrix, verdict computation.

- **compute_active_tools**: all below threshold, Reentrancy activates static+rag, GasException/MishandledException only static, DoS threshold 0.30, UnusedReturn threshold 0.45, deduplication
- **build_routing_decisions**: fast/deep path strings, skipped class shows threshold
- **compute_verdict**: CONFIRMED (Slither match or RAG ≥ 0.80), LIKELY (RAG ≥ 0.50), DISPUTED (no corroboration)
- **compute_overall_verdict**: max-rank across classes
- **prob_to_severity**: CRITICAL/HIGH/MEDIUM/LOW/INFO boundaries
- **DETECTOR_TO_CLASSES**: inverted map completeness
- **evidence_router node**: logs routing decisions, fast path
- **graph compilation**: evidence_router present, SqliteSaver attached

### test_retriever_filters.py (236 lines)

HybridRetriever filter behaviour and FAISS↔chunks sync validation.

- **_apply_filters**: no filters, vuln_type, date_gte, loss_gte, source, has_summary, combined, strict filter returns empty
- **Search scores**: returned chunks have positive score, scores descending
- **FAISS↔chunks sync**: RuntimeError on count mismatch (corruption detection)

### test_github_fetcher.py (211 lines)

DeFiHackLabsFetcher parsing logic — no network, no FAISS.

- **_extract_loss**: millions, thousands, raw USD, decimal, billions, old format, no loss, malformed
- **_infer_vuln_type**: reentrancy, flash_loan, oracle, access_control, integer_overflow, fallback to other, no content slice used (FIX-22b)
- **_extract_date**: extracts YYYY-MM from path, empty string when no date
- **fetch_since**: includes undated files (FIX-21), excludes old dated, includes new dated
- **past/ directory**: files are fetched (FIX-20), no past/ dir doesn't crash

### test_deduplicator.py (159 lines)

Deduplicator seen/filter/mark cycle and persistence.

- **Init**: starts empty, loads existing, handles corrupted JSON
- **seen()**: false for unknown, true after mark_seen, doesn't affect other IDs
- **filter_new()**: returns all when none seen, filters seen, empty when all seen
- **mark_seen()**: persists to disk, empty list OK, increments count, idempotent, timestamp recorded
- **Checkpoint pattern**: docs not marked until mark_seen called, after mark they're excluded

### test_chunker.py (155 lines)

Chunker splitting, metadata inheritance, edge cases.

- **Defaults**: chunk_size=1536, accepts custom size
- **chunk_document**: short doc → 1 chunk, long doc → multiple, empty → empty list, sequential IDs, total_chunks consistent, doc_id preserved
- **Metadata inheritance**: protocol, date, chunk-specific metadata, parent not mutated
- **chunk_documents**: processes multiple docs, empty skipped, flat list, sizes within limit

### test_consensus_voting.py (A.6, added 2026-06-21)

- **TestConsensusVote**: all-agree confirms, **ML alone can never confirm (any class)**,
  ML+one tool escalates confidence, no-signal → SAFE, confidence always in [0,1],
  unknown class uses default weights, `ML_WEIGHT_SCALE` env discounts/restores ML weight
- **TestConsensusEngineNode**: emits rows for flagged+tool-hit classes only, no
  probabilities → empty, falls back to confirmed/suspicious list when probabilities absent

### test_confidence_tracking.py (A.7)

- **TestTrackConfidence**: ML-only returns clamped probability, Slither agreement
  boosts / disagreement shrinks, RAG boost only above relevance floor, always in
  bounds, full ML→Slither→RAG pipeline flow, confidence band labels

### test_metric_attribution.py (A.8)

- **TestAttributeVerdict**: percentages sum to 100, no-evidence → zeros, Slither-only,
  sub-floor RAG ignored, ML-dominant when strong
- **TestExplainerNode**: attributes + folds confidence/consensus/reflection into
  final_report, verdict rows annotated in place, empty report is safe

### test_reflection.py (A.3 + A.4 debate)

- **TestReflectionRuleBased**: rule-based critique without LLM (uncertain verdicts,
  failure modes incl. ExternalBug ML over-prediction, contradictions surfaced,
  truncated-contract failure mode), empty state is safe
- **TestReflectionLLM**: LLM summary used when enabled, falls back to rule-based on
  LLM failure
- **TestDebateMode**: 3-role debate (Prosecutor/Defender/Judge) with mocked LLM,
  3 invoke calls, `debate_transcript` populated; LLM disabled skips debate entirely

### test_visualizer.py (A.9)

- **TestGenerateHotspotHtml**: valid HTML document, contains code + verdict panel,
  hotspot lines highlighted with clickable `data-fn`, attribution bars present,
  HTML-escapes source (XSS-safe), empty state degrades gracefully
- **TestVisualizerNode**: writes `{address}_hotspot.html` to `REPORTS_DIR`, no address
  → no file written

### test_rag_fetchers.py (A.5)

- **TestSeedCorpora**: all 5 fetchers ship a working seed corpus and fetch real `Document`s
- **TestJsonCorpusBehaviour**: missing corpus → empty (no crash), custom corpus parsed,
  doc_id derived when absent, `fetch_since` filters by date, malformed JSON → empty,
  `{"findings": [...]}` wrapper format tolerated
- **TestSWCRegistry**: canonical SWC-107 (Reentrancy) / SWC-101 (overflow) IDs present
