# SENTINEL — Session Log: Tracks 0–2 + M4.3 + M5

**Date:** 2026-04-17  
**Commits:** `58a5925` → `04abb94`  
**Tags created:** `v1.0-binary-baseline` (pre-cleanup snapshot), `v1.3-m5-complete`

---

## Overview

This session covered four distinct work items, executed in order:

| Track | What | Files Changed |
|-------|------|---------------|
| Track 0 | File structure cleanup — archive legacy code, rename scripts | 15 files moved/renamed |
| Track 1 | 24 code quality fixes from SENTINEL_FINAL_IMPROVEMENT_LIST.md | 9 files modified |
| M4.3 | `sentinel-audit` MCP server — AuditRegistry on-chain bridge | 3 files new |
| M5 | LangGraph audit graph — full pipeline orchestration | 6 files new |

---

## Track 0 — File Structure Cleanup

### What changed

**Before:** Legacy scripts at root-level names that mixed production and one-time-use tools.  
**After:** Clean module boundaries, legacy code archived, scripts renamed to intent.

**Archived to `ml/_archive/`:**
- `ml/src/data/ast_extractor_v4_production.py` → `ml/_archive/ast_extractor_v4.py`
- `ml/src/data/tokenizer_v1_production.py` → `ml/_archive/tokenizer_v1.py`
- `ml/logs/` contents → `ml/_archive/logs/`

**New `ml/_archive/README.md`** explains why files are there and how to restore:
```bash
git restore --source=v1.0-binary-baseline ml/_archive/ast_extractor_v4.py
```

**Renamed scripts** (to clearly indicate one-time-use data generation, not inference pipeline):
- `ml/scripts/extract_graphs.py` ← was `ast_extractor_v4_production.py`
- `ml/scripts/extract_tokens.py` ← was `tokenizer_v1_production.py`

**New `ml/data_extraction/__init__.py`** — explains the module's purpose: these scripts ran once to produce `ml/data/graphs/` and `ml/data/tokens/`, and are not part of inference.

### Why

The legacy filenames (`_v4_production`) implied they were active production code. They're frozen one-time-use tools. Archiving them prevents confusion for anyone reading the codebase and removes them from the inference import graph.

---

## Track 1 — 24 Code Fixes (A-01 through A-24)

From `SENTINEL_FINAL_IMPROVEMENT_LIST.md`. All fixes in commit `b02a407`.

### Critical fixes

**A-01 — `predictor.py`: threshold validation**  
```python
# Before: no validation
# After:
if not (0 < threshold < 1):
    raise ValueError(f"threshold must be in (0, 1), got {threshold}")
```

**A-02 — `preprocess.py`: removed dead import**  
```python
# Deleted — ASTExtractor was archived, this import would crash inference
from ml.src.data.graphs.ast_extractor import ASTExtractor
```

**A-11/A-12 — `run_proof.py` + `setup_circuit.py`: checkpoint compat + assert removal**  
New trainer saves `{"model": state_dict, "epoch": ..., "best_f1": ...}` but old EZKL scripts called `load_state_dict()` on the full dict. Fixed with isinstance check:
```python
_ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)
if isinstance(_ckpt, dict) and "model" in _ckpt:
    _state_dict = _ckpt["model"]
else:
    _state_dict = _ckpt  # legacy raw state_dict
model.load_state_dict(_state_dict)
```
Also: three `assert res` → `if not res: raise RuntimeError(...)` with diagnostic messages (Python `-O` strips asserts silently in production Docker).

**A-13 — `inference_server.py`: removed `"mock": True` from mock prediction**  
The mock response leaked into tests and caused schema mismatch. Clean mock now returns only schema-valid fields.

**A-14 — `embedder.py`: embed_query retry coverage**  
`embed_query()` now calls `_embed_batch_with_retry()` instead of the bare API call — gets the same exponential backoff as batch embeds.

**A-18/A-19 — `api.py`: OOM guard + missing predictor guard**  
```python
except torch.cuda.OutOfMemoryError:
    torch.cuda.empty_cache()
    raise HTTPException(status_code=413, detail="Contract too large for GPU memory")

predictor = getattr(request.app.state, "predictor", None)
if predictor is None:
    raise HTTPException(status_code=503, detail="Model not loaded yet")
```

**A-20 — `inference_server.py`: shared httpx client**  
Module-level `_http_client: httpx.AsyncClient` initialized in `_on_startup()` and closed in `_on_shutdown()`. Removes one TCP handshake + TLS negotiation per `/predict` request.

**A-23 — `deduplicator.py`: debug logging for skipped docs**  
`filter_new()` logs every skipped document at DEBUG level — makes pipeline tracing possible without touching logic.

**A-24 — `sentinel_model.py`: typed `graphs` parameter**  
`forward(graphs: object, ...)` → `forward(graphs: Batch, ...)` for PyG `Batch` type.

### Other fixes (A-03 through A-22)
- Stale comments updated in inference_server, rag_server, retriever
- `handle_sse` return type fixed (`None` → `Response`)
- sys.path insert added to rag_server
- `get_index_info()` return type annotation corrected

---

## M4.3 — sentinel-audit MCP Server

**New file:** `agents/src/mcp/servers/audit_server.py`  
**Tests:** `agents/tests/test_audit_server.py` (21 tests, all pass)  
**Smoke test:** `agents/scripts/smoke_audit_mcp.py`  
**Updated:** `agents/.env` (added `MCP_AUDIT_PORT`, `AUDIT_REGISTRY_ADDRESS`, `AUDIT_MOCK`)

### What it does

SSE MCP server on port 8012 that exposes three tools against the on-chain `AuditRegistry.sol`:

| Tool | What it returns |
|------|----------------|
| `get_latest_audit` | Most recent audit record for a contract address |
| `get_audit_history` | All records, reverse-chronological, with optional limit |
| `check_audit_exists` | Boolean — has this contract ever been audited? |

### Score decoding

`AuditRegistry` stores scores as EZKL field elements. Decode:
```python
EZKL_SCALE_FACTOR = 8192  # 2^13 — from calibration step
score = int(field_element) / EZKL_SCALE_FACTOR  # → float in [0, 1]
```

### Mock mode

If `SEPOLIA_RPC_URL` is empty (local dev) or if the RPC call fails, the server auto-switches to deterministic mock mode. It does NOT crash — it logs a warning and returns mock data. This keeps the full LangGraph pipeline runnable without a live blockchain connection.

### Key design decisions

- **ABI loaded from `contracts/out/`** — Foundry's compiled artifact, not a hardcoded ABI string. Keeps the server in sync with contract changes automatically.
- **AsyncWeb3** — Non-blocking RPC calls (web3.py v7 `AsyncWeb3` + `AsyncHTTPProvider`).
- **Global `_MOCK_MODE` declared first in `_on_startup()`** — Python requires `global` declarations before any use. Placing it after `if _MOCK_MODE:` causes `SyntaxError: name used prior to global declaration`.

---

## M5 — LangGraph Audit Graph

**New files:**
- `agents/src/orchestration/__init__.py`
- `agents/src/orchestration/state.py`
- `agents/src/orchestration/nodes.py`
- `agents/src/orchestration/graph.py`
- `agents/scripts/smoke_langgraph.py`
- `agents/tests/test_graph_routing.py` (35 tests, all pass)

**Tag:** `v1.3-m5-complete`

### Graph topology

```
START
  │
  ▼
ml_assessment  ──── calls sentinel-inference:predict ────────────────────────┐
  │                                                                            │
  ├─ confidence > 0.70 ("deep") ──► rag_research ──► audit_check ──► synthesizer
  │                                                                      ▲
  └─ confidence ≤ 0.70 ("fast") ────────────────────────────────────────┘
                                                                         │
                                                                        END
```

### State schema (`AuditState`)

`TypedDict` with `total=False` — all fields optional. LangGraph merge semantics: each node returns only what it changed; unchanged fields are preserved.

```python
class AuditState(TypedDict, total=False):
    contract_code:    str                    # input
    contract_address: str                    # input
    ml_result:        dict[str, Any]         # set by ml_assessment
    rag_results:      list[dict[str, Any]]   # set by rag_research (deep only)
    audit_history:    list[dict[str, Any]]   # set by audit_check (deep only)
    static_findings:  dict[str, Any]         # reserved for M6
    final_report:     dict[str, Any]         # set by synthesizer
    error:            str | None             # set by any node on failure
```

### Routing helper `_is_high_risk()`

```python
def _is_high_risk(ml_result):
    # Binary threshold: confidence > 0.70 → deep path
    # Track 3 multi-label swap: max(v["probability"] for v in vulnerabilities) > 0.70
    return ml_result.get("confidence", 0.0) > 0.70
```

This is the **single place to update** when Track 3 multi-label is implemented. No other routing code changes.

### Error handling philosophy

Nodes **never raise** — exceptions are caught and written to `state["error"]`. The synthesizer always runs and produces a partial report. If ML assessment failed, the report notes it and recommends manual review. This matches the pattern: a partial audit is better than a crashed pipeline.

### Synthesizer (M5 vs M6)

M5 synthesizer is rule-based — deterministic text based on label + confidence:
- `confidence >= 0.70 and label == "vulnerable"` → "HIGH RISK" with RAG count + prior audit count
- `confidence < 0.70 and label == "vulnerable"` → "MODERATE RISK"  
- `label == "safe"` → "LOW RISK"
- `ml_result == {}` → "ML assessment failed — manual review required"

M6 will replace the rule-based block with an LLM call (the block is clearly delimited by comments for easy swap).

### MCP call pattern

```python
async def _call_mcp_tool(server_url, tool_name, arguments):
    async with sse_client(server_url) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, arguments)
            return json.loads(result.content[0].text)
```

Fresh connection per call. `_call_mcp_tool` is the single patch point for the smoke test — mocking it is sufficient to test all four nodes.

### Checkpointing

`MemorySaver` by default — in-process dict, zero setup, fast for M5. Swap to `RedisSaver` for M6 multi-replica deployment. `use_checkpointer=False` in tests for speed.

Resume after interrupt:
```python
result = await graph.ainvoke(
    None,  # None = load state from checkpoint
    config={"configurable": {"thread_id": "audit-001"}}
)
```

---

## What's Next (M6)

Five specialized agents:
1. `static_analyzer.py` — Slither + Mythril, direct call (no MCP)
2. `ml_intelligence.py` — wraps sentinel-inference with retry logic
3. `rag_researcher.py` — query builder + sentinel-rag call
4. `code_logic.py` — LLM code analysis agent (Qwen 3.5-9B-UD via LM Studio)
5. `synthesizer.py` — LLM-generated audit narrative replacing M5 rule-based text

Plus:
- `api_gateway.py` — POST /audit, POST /batch_audit, GET /audit/{id}
- `AuditReport` Pydantic model — structured output schema
- `agents/docker-compose.yml` — all services wired together
