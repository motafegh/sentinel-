# Scripts — Utilities & Smoke Tests

Standalone scripts for smoke testing MCP servers, running real audits, and driving the
evaluation pipeline. No pytest framework — each script is self-contained.

## Files

| File | Purpose |
|------|---------|
| `smoke_inference_mcp.py` | Tests inference MCP server — connect, discover tools, call predict |
| `smoke_rag_mcp.py` | Tests RAG MCP server — health, search, filters, k cap |
| `smoke_audit_mcp.py` | Tests audit MCP server — all three tools, bad address handling |
| `smoke_langgraph.py` | Full audit graph — mock or live mode (`--live` flag) |
| `test_k_cap.py` | Quick k=99 cap test for RAG server |
| `run_real_audit.py` | Real-LLM E2E harness — `--no-llm`, `--profile`, `--unbounded-timeouts` |
| `eval_benchmark.py` | Evaluation benchmark runner, scores against ground truth |
| `audit_gt_labels.py` | Ground-truth vulnerability labels for benchmark scoring |
| `audit_labels.py` | Extended label set for eval coverage |
| `build_reliability_matrix.py` | P3: builds per-tool reliability matrix from an eval run directory |

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

# Real audit with LLM (LM Studio + all MCP servers running)
poetry run python scripts/run_real_audit.py contracts/Vault.sol

# Real audit, no LLM, with profiling
poetry run python scripts/run_real_audit.py contracts/Vault.sol --no-llm --profile

# Build reliability matrix from a completed eval run (P3)
poetry run python scripts/build_reliability_matrix.py \
    --run-dir eval/runs/20260626T123145Z_p3_rule5c_v3 \
    --output configs/reliability_v3.yaml
```

## Script Details

### `smoke_langgraph.py`

End-to-end test for the full LangGraph audit pipeline.

- **Mock mode** (default): Patches `_call_mcp_tool` so no MCP servers are needed.
  Exercises all 14 nodes with realistic mock data.
- **Live mode** (`--live`): Uses real MCP servers. All five servers must be running.

### `smoke_audit_mcp.py`

Starts the audit server as a subprocess (mock mode), waits for `/health`, then exercises
all three tools via MCP SSE client:
- `check_audit_exists` → `exists=True`
- `get_latest_audit` → `score=0.7314 label=vulnerable`
- `get_audit_history` → 2 records returned
- Bad address → error returned (not a crash)

### `smoke_rag_mcp.py`

Six sequential checks:
1. `/health` responds with chunk count
2. SSE handshake + MCP initialization succeeds
3. `list_tools()` returns the `search` tool with correct schema
4. `call_tool('search')` returns results with expected shape
5. `call_tool('search')` with filters runs without error
6. `call_tool('search')` with k=99 is capped to 20 results

### `build_reliability_matrix.py`

P3 script. Reads `results.json` from a completed benchmark run, computes TP/FP/FN/TN
per tool per vulnerability class (excluding contracts where `tool_status[tool]["ran"]`
is False — Rule 5C), applies Bayesian shrinkage (α=5), and writes `reliability_v3.yaml`.

### `run_real_audit.py`

Full E2E audit harness for manual testing with live services.

| Flag | Effect |
|------|--------|
| `--no-llm` | Sets `AGENTS_DISABLE_LLM=1` — skips all LLM calls |
| `--profile` | Prints per-node timing breakdown |
| `--unbounded-timeouts` | Removes tool timeouts (useful for slow Halmos runs) |

Reports `success/partial/failed` in the summary line — never silently succeeds when
tools failed (Rule 5C).
