# Scripts — Smoke Tests & Utilities

Quick connectivity and integration tests for MCP servers and the full audit graph. Each script is standalone — no pytest framework, no test fixtures.

## Files

| File | Lines | Purpose |
|------|-------|---------|
| `smoke_inference_mcp.py` | 103 | Tests inference MCP server — connect, discover tools, call predict |
| `smoke_rag_mcp.py` | 174 | Tests RAG MCP server — health, search, filters, k cap |
| `smoke_audit_mcp.py` | 144 | Tests audit MCP server — all three tools, bad address handling |
| `smoke_langgraph.py` | 189 | Full audit graph — mock or live mode (`--live` flag) |
| `test_k_cap.py` | 39 | Quick k=99 cap test for RAG server |
| `run_real_audit.py` | — | Real-LLM E2E harness with `--no-llm`, `--profile`, `--unbounded-timeouts` |
| `audit_gt_labels.py` | — | Ground-truth labels for evaluation benchmark |
| `eval_benchmark.py` | — | Evaluation benchmark runner, scores against ground truth |

## Usage

```bash
cd agents

# Inference server (must be running on :8010)
poetry run python scripts/smoke_inference_mcp.py

# RAG server (must be running on :8011)
poetry run python scripts/smoke_rag_mcp.py

# Audit server (starts its own subprocess in mock mode)
poetry run python scripts/smoke_audit_mcp.py

# Full audit graph — mock mode (no services needed)
poetry run python scripts/smoke_langgraph.py

# Full audit graph — live mode (all MCP servers must be running)
poetry run python scripts/smoke_langgraph.py --live

# Quick k cap test (RAG server must be running on :8011)
poetry run python scripts/test_k_cap.py
```

## Script Details

### smoke_langgraph.py

End-to-end test for the full LangGraph audit pipeline.

- **Mock mode** (default): Patches `_call_mcp_tool` so no MCP servers are needed. Exercises all nodes with realistic mock data.
- **Live mode** (`--live`): Uses real MCP servers. All four servers must be running.

Expected output:
```
[PASS] Graph compiled successfully
[PASS] ml_assessment ran — label=vulnerable confidence=0.82
[PASS] rag_research ran — 3 RAG chunks retrieved (deep path)
[PASS] audit_check ran  — 2 prior audits found
[PASS] synthesizer ran  — recommendation present
[PASS] path_taken=deep  (confidence 0.82 > 0.70 threshold)
```

### smoke_audit_mcp.py

Starts the audit server as a subprocess (mock mode), waits for `/health`, then exercises all three tools via MCP SSE client. Tests:
- `check_audit_exists` → `exists=True`
- `get_latest_audit` → `score=0.7314 label=vulnerable`
- `get_audit_history` → 2 records returned
- Bad address → error returned (not a crash)

### smoke_rag_mcp.py

Six sequential checks:
1. `/health` responds with chunk count
2. SSE handshake + MCP initialization succeeds
3. `list_tools()` returns the `search` tool with correct schema
4. `call_tool('search')` returns results with expected shape
5. `call_tool('search')` with filters runs without error
6. `call_tool('search')` with k cap enforced (k > 20 → 20 results max)
