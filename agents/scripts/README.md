# Scripts — Utilities & Smoke Tests

Standalone scripts for smoke testing MCP servers, running audits, and driving the
evaluation pipeline. No pytest framework — each script is self-contained.

For the public fresh-clone portfolio entry point, start with the root
[`SHOWCASE.md`](../../SHOWCASE.md). The scripts in this directory are deeper
module/runtime exercises and may require the AGENTS environment, local tools, or
live services.

## Files

| File | Purpose |
|------|---------|
| `smoke_inference_mcp.py` | Tests inference MCP server — connect, discover tools, call predict |
| `smoke_rag_mcp.py` | Tests RAG MCP server — health, search, filters, k cap |
| `smoke_audit_mcp.py` | Tests audit MCP server — all three tools, bad address handling |
| `smoke_langgraph.py` | Full audit graph — mock-MCP or live mode (`--live` flag) |
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

# Full audit graph — MCP responses mocked; disable LLM for deterministic local smoke
AGENTS_DISABLE_LLM=1 poetry run python scripts/smoke_langgraph.py

# Full audit graph — live MCP mode (required services/toolchains must be available)
poetry run python scripts/smoke_langgraph.py --live

# Quick k cap test (RAG server must be running on :8011)
poetry run python scripts/test_k_cap.py

# Real audit with LLM (LM Studio + all required MCP/tool services running)
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

End-to-end smoke for the LangGraph orchestration path.

- **Default mode:** patches MCP calls with deterministic fixture responses, so MCP servers are not required. It exercises the real graph/orchestration code. Set `AGENTS_DISABLE_LLM=1` for a deterministic no-LLM smoke.
- **Live mode (`--live`):** uses live MCP/service dependencies and therefore requires the relevant local runtime/toolchains.

The mock payloads are interface fixtures. A passing smoke proves orchestration/report assembly against those fixtures; it is **not** evidence of vulnerability-detection quality, model quality, or live analyzer availability. Tool/analyzer failures must remain explicit in the resulting status rather than being interpreted as clean findings.

### `smoke_audit_mcp.py`

Starts the audit server as a subprocess (mock mode), waits for `/health`, then exercises
all three tools via MCP SSE client:
- `check_audit_exists` → `exists=True`
- `get_latest_audit` → mock record returned
- `get_audit_history` → versioned mock records returned
- Bad address → error returned (not a crash)

These mock results validate the read interface; they are not live on-chain evidence.

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
