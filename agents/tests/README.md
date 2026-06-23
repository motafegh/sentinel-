# Tests — Unit & Integration

Pytest-based unit and integration tests for the agents module. All tests run without live MCP servers, LM Studio, or Sepolia RPC. Mocks isolate each layer for fast, deterministic testing.

## Files

| File | Lines | Purpose |
|------|-------|---------|
| `test_graph_routing.py` | 1,075 | Full graph: routing, all 13 nodes, compilation, integration |
| `test_eval_framework.py` | 471 | Evaluation harness: benchmark scoring, ground-truth comparison |
| `test_ws4_2_selective_gating.py` | 383 | WS4.2 — asymmetric debate gating (CONFIRMED+2tools skip) |
| `test_smoke_e2e.py` | 375 | End-to-end deep/fast/screen-escalated/ML-failure paths |
| `test_representation_server.py` | 356 | GNN embedding MCP server tests |
| `test_audit_server.py` | 345 | On-chain history, mock mode, address validation |
| `test_routing_phase0.py` | 345 | Phase 0 routing: thresholds, tool matrix, verdict logic |
| `test_inference_server.py` | 335 | MCP tool schemas, mock/live transport, batch predict |
| `test_ws3_hotspot_excerpts.py` | 310 | WS3 — hotspot-guided code excerpts in debate prompts |
| `test_verdict_reconciliation.py` | 268 | 8-case reconciliation table + invariants |
| `test_retriever_filters.py` | 236 | FAISS+BM25+RRF filter behaviour, sync validation |
| `test_github_fetcher.py` | 211 | DeFiHackLabs parsing (3 formats, FIX-20/21/22b) |
| `test_verdict_integrity.py` | 204 | FN/FP invariants, DISPUTED floor enforcement |
| `test_consensus_voting.py` | 161 | A.6 — weighted vote, ML-weight discount |
| `test_deduplicator.py` | 159 | SHA256 dedup, persistence, checkpoint pattern |
| `test_chunker.py` | 155 | Chunk size, overlap, metadata inheritance |
| `test_reflection.py` | 154 | A.3/A.4 — self-critique + 3-role debate (mocked) |
| `test_static_analysis_real_slither.py` | 88 | REAL (non-mocked) Slither — detector registration |
| `test_static_analysis_real_aderyn.py` | 86 | REAL (non-mocked) Aderyn — dir/output-path bugs |
| `test_timeouts_and_timing.py` | 97 | Centralized timeouts + `step_timer`/`timed_node` |
| `test_rag_fetchers.py` | 96 | A.5 fetcher code exists but **disabled** in `build_index.py` |
| `test_metric_attribution.py` | 70 | A.8 — LIME-style attribution |
| `test_confidence_tracking.py` | 49 | A.7 — Bayesian confidence updating |
| `test_visualizer.py` | 90 | A.9 — hotspot HTML generation |
| `conftest.py` | 16 | Sets `AGENTS_DISABLE_LLM=1` session-wide |
| `__init__.py` | 0 | Package marker |

**Total: 402 tests** (26 files, ~6,135 lines). `conftest.py` sets `AGENTS_DISABLE_LLM=1` session-wide.

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

### test_graph_routing.py (1,075 lines)

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

### test_inference_server.py (335 lines)

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

### test_rag_fetchers.py (A.5, 96 lines)

- **TestSeedCorpora**: 5 fetcher classes exist and parse their seed files (⚠ corpora
  are synthetic placeholders; fetchers are **disabled** in `build_index.py` per WS2)
- **TestJsonCorpusBehaviour**: missing corpus → empty (no crash), malformed JSON → []
- **TestSWCRegistry**: canonical SWC-107 (Reentrancy) / SWC-101 (overflow) IDs present

### test_eval_framework.py (471 lines)

Evaluation harness: runs the benchmark suite and scores results against ground-truth
labels (`audit_gt_labels.py`). Measures precision/recall across the 10-class track.

### test_ws4_2_selective_gating.py (383 lines)

WS4.2 — tests the asymmetric debate gating: debate is skipped when all flagged
classes are CONFIRMED by ≥2 of 3 tools; debate runs for any lower-confidence case.

### test_representation_server.py (356 lines)

Tests the representation MCP server that serves GNN embeddings. Covers tool schemas,
mock fallback, and response shape validation.

### test_ws3_hotspot_excerpts.py (310 lines)

WS3 — tests the hotspot-guided code excerpt construction in the cross_validator
debate prompt. Verifies that `ml_hotspots` are correctly used to focus the LLM.

### test_verdict_reconciliation.py (268 lines)

Tests the 8-case `_reconcile_verdicts()` function table plus invariants: a class
flagged by `consensus_engine` cannot be cleared to SAFE by the debate; confidence=1.0
is never downgraded; debate can upgrade but not downgrade past DISPUTED.

### test_verdict_integrity.py (204 lines)

FN/FP asymmetry invariants: enforces that no vulnerability class is ever silently
cleared to SAFE without a recorded reason. Covers the WS1 design principle across
all graph paths.
