# Tests — Unit & Integration

Pytest-based tests for the agents module. **631 passing, 3 skipped** (as of 2026-06-26).
All tests run without live MCP servers, LM Studio, or Sepolia RPC. Mocks isolate each
layer for fast, deterministic testing.

`conftest.py` sets `AGENTS_DISABLE_LLM=1` session-wide. Nodes that call LLMs
(`cross_validator`, `synthesizer`, `reflection`) consult `_llm_enabled()` in
`src/orchestration/nodes/_helpers.py` — false when this env var is set. Tests that
explicitly exercise the LLM path re-enable it locally and mock the LLM call.

## Running

```bash
cd agents
poetry run pytest tests/ -v                              # all tests
poetry run pytest tests/test_graph_routing.py -v        # specific file
poetry run pytest tests/ -k "verdict" -v                # pattern filter
poetry run pytest tests/ -v --cov --cov-report=html     # with coverage
```

## Test Files

### Orchestration

| File | Tests | What it covers |
|------|-------|----------------|
| `test_graph_routing.py` | ~60 | Full graph: routing, all 14 nodes, compilation, integration |
| `test_smoke_e2e.py` | ~7 | End-to-end deep/fast/screen-escalated/ML-failure paths |
| `test_routing_phase0.py` | ~20 | Per-class thresholds, tool matrix, verdict logic |
| `test_routing_isolation.py` | ~4 | AST-based: routing.py + evidence_router.py have no LLM imports |
| `test_ws4_2_selective_gating.py` | ~15 | WS4.2 — debate skipped when all classes CONFIRMED by ≥2 tools |
| `test_ws3_hotspot_excerpts.py` | ~12 | WS3 — hotspot-guided code excerpts in debate prompts |

### Verdict & Evidence (P2)

| File | Tests | What it covers |
|------|-------|----------------|
| `test_verdict_fuse.py` | — | `fuse()` — evidence → dual-tier verdict (provable + full) |
| `test_verdict_evidence.py` | — | `Evidence` dataclass construction, `emit_evidence()` |
| `test_verdict_reliability.py` | — | L1/L3/schema-mismatch paths in `reliability.py` |
| `test_verdict_integrity.py` | ~15 | FN/FP asymmetry invariants; no class cleared silently |
| `test_p2_evidence_integration.py` | — | End-to-end evidence → fuse() → verdicts integration |

### Consensus & Confidence (A.6/A.7)

| File | Tests | What it covers |
|------|-------|----------------|
| `test_consensus_voting.py` | ~10 | Weighted vote, ML alone never confirms, `ML_WEIGHT_SCALE` |
| `test_confidence_tracking.py` | ~7 | Bayesian updates — Slither boosts, RAG floor, clamping |

### Security — Prompt Injection (P4)

| File | Tests | What it covers |
|------|-------|----------------|
| `test_comment_strip.py` | 16 | State-machine comment stripper — all edge cases |
| `test_prompt_delimit.py` | 5 | Delimiter framing, truncation |
| `test_injection_detect.py` | 15 | All 8 injection pattern families |
| `test_prompt_sanitize.py` | 5 | Orchestrator layer composition |
| `test_routing_isolation.py` | 4 | Routing nodes verified clean (no LLM, no contract_code) |
| `test_adversarial_corpus.py` | 8 | One adversarial contract per pattern — injection detected, verdict correct |

### Reproducibility & Determinism (P5)

| File | Tests | What it covers |
|------|-------|----------------|
| `test_deterministic_mode.py` | 8 | `SENTINEL_DETERMINISTIC=1` disables LLM + RAG, torch determinism |

### Formal Verification (P8a)

| File | Tests | What it covers |
|------|-------|----------------|
| `test_formal_verification.py` | 15 | Halmos node: Foundry harness gen, result parsing, fail-soft on missing tools |

### Gateway & API (P10)

| File | Tests | What it covers |
|------|-------|----------------|
| `test_gateway.py` | — | Gateway routes: POST /audit, GET /audit/{id}, GET /health |
| `test_p10_gateway.py` | ~13 | SQLite JobStore, crash recovery, health monitor degraded state |

### MCP Servers

| File | Tests | What it covers |
|------|-------|----------------|
| `test_inference_server.py` | ~20 | Tool schemas, mock/live transport, batch predict |
| `test_audit_server.py` | ~20 | On-chain history, mock mode, address validation, score decoding |
| `test_representation_server.py` | ~15 | GNN embedding server tools, mock fallback, response shape |

### Evaluation Framework

| File | Tests | What it covers |
|------|-------|----------------|
| `test_eval_framework.py` | ~30 | Benchmark scoring, ground-truth comparison, 10-class track |
| `test_eval_fbeta.py` | — | Fbeta(β=2) computation — macro + per-class |
| `test_run_benchmark.py` | — | CLI benchmark runner integration |

### Config (P1)

| File | Tests | What it covers |
|------|-------|----------------|
| `test_config.py` | — | Schema validation, loader singleton, L1/L3 fallback |

### RAG

| File | Tests | What it covers |
|------|-------|----------------|
| `test_retriever_filters.py` | ~15 | FAISS+BM25+RRF filter behaviour, FAISS↔chunks sync |
| `test_github_fetcher.py` | ~20 | DeFiHackLabs parsing (3 formats, FIX-20/21/22b) |
| `test_chunker.py` | ~15 | Chunk size, overlap, metadata inheritance |
| `test_deduplicator.py` | ~15 | SHA256 dedup, persistence, checkpoint pattern |
| `test_rag_fetchers.py` | ~10 | A.5 fetcher classes exist + parse seed files (⚠ disabled in pipeline) |
| `test_rag_query.py` | — | RAG query integration with HybridRetriever |

### Static Analysis (Real Tools)

| File | Tests | What it covers |
|------|-------|----------------|
| `test_static_analysis_real_slither.py` | ~5 | Non-mocked Slither — detector registration |
| `test_static_analysis_real_aderyn.py` | ~5 | Non-mocked Aderyn — dir/output-path bugs |

### Attribution & Visualization (A.8/A.9)

| File | Tests | What it covers |
|------|-------|----------------|
| `test_metric_attribution.py` | ~8 | LIME-style attribution: percentages sum to 100, ML-dominant |
| `test_visualizer.py` | ~8 | Hotspot HTML generation, XSS-safe escaping, graceful degradation |

### LangGraph Internals

| File | Tests | What it covers |
|------|-------|----------------|
| `test_reflection.py` | ~12 | A.3 rule-based critique + LLM summary + A.4 debate (mocked) |
| `test_timeouts_and_timing.py` | ~8 | Centralized timeouts, `step_timer`, `timed_node` |

## Key Test Patterns

### Mocking MCP Tools

```python
with mock.patch("src.orchestration.nodes._helpers._call_mcp_tool") as m:
    m.return_value = {"label": "safe", "probabilities": {...}}
    result = await graph.ainvoke(state, config=...)
```

### Testing Without LLM

```python
# conftest.py sets AGENTS_DISABLE_LLM=1 globally.
# To test the LLM path, re-enable locally:
with mock.patch("src.orchestration.nodes._helpers._llm_enabled", return_value=True):
    with mock.patch("src.llm.client.get_fast_llm") as m:
        m.return_value.ainvoke = AsyncMock(return_value=AIMessage(content="CONFIRMED"))
        ...
```

### Testing Rule 5C (No Silent Failures)

```python
# A mocked subprocess raising FileNotFoundError MUST NOT produce result == []
# indistinguishable from "ran clean". Assert that tool_status is set:
assert state["tool_status"]["aderyn"]["ran"] is False
assert "reason" in state["tool_status"]["aderyn"]
```
