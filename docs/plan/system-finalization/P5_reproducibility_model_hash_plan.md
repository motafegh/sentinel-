# P5 — Reproducibility Test + Model-Hash Binding (B-4)

**Date:** 2026-06-26
**Phase:** P5 (reproducibility test + model-hash binding; finalize ZK boundary contract)
**Architecture of record:** `docs/proposal/2026-06-23_proposal_SYSTEM_architecture-finalization.md` (§5.4, §10 P5 row, §10.1 P5 row)
**Pre-conditions:** P4 DONE (586 tests green, 3-layer injection defense active).
**Working memory:** `~/.claude/scratch/p5_reproducibility_20260626.md`

---

## Proposal references (verbatim)

**§10.1 P5 row:**
- *Mode:* `SENTINEL_DETERMINISTIC=1` → skip LLM debate + RAG, `torch.use_deterministic_algorithms`.
- *Hash:* **SHA-256 of the `.pt` file**, computed at audit start, in the report + on-chain anchor.

**§5.4 Reliability & determinism plumbing:**
> `deterministic` is set per source at emission. The reproducibility test (B-4) guards the provable tier and binds it to a model hash for ZK anchoring.

**§12 Definition of done:**
> The system emits a reproducible `verdict_provable` (identical across runs, bound to a model hash) and a richer `verdict_full`; only the former is anchored on-chain.

---

## Current state (source read 2026-06-26)

### What already exists

| Component | Location | Status |
|-----------|----------|--------|
| **Model hash functions** | `ml/scripts/auto_reproducibility_check.py:49-82` | `_file_hash()` (SHA-256 of `.pt` file) and `_model_state_hash()` (SHA-256 of `state_dict` tensors) exist but are NOT wired into the pipeline |
| **Checkpoint config** | `ml/src/inference/api.py:74-80` | 3-level precedence: `SENTINEL_CHECKPOINT` env > `mlops_config.json` > hardcoded default |
| **`AGENTS_DISABLE_LLM`** | `agents/src/orchestration/nodes/_helpers.py:29-31` | Disables LLM nodes (cross_validator, synthesizer narrative, reflection) |
| **`torch.use_deterministic_algorithms`** | NOWHERE | Not set in any source file |
| **Reproducibility tests** | `ml/tests/test_api.py:201-214`, `ml/tests/test_fusion_layer.py:149-162` | Within-process consistency only; no cross-run test |
| **Evidence `deterministic` flag** | `agents/src/orchestration/verdict/evidence.py` | ML/static = `True`; debate/RAG = `False` |
| **Dual verdict** | `agents/src/orchestration/verdict/fuse.py` | `verdict_provable` (deterministic only) + `verdict_full` (all evidence) |

### Sources of non-determinism

| Node | Source | Can disable? |
|------|--------|-------------|
| `ml_assessment` | ML model on CUDA (no deterministic algorithms set) | No |
| `cross_validator` | LLM debate (temp=0.0 but CUDA/llama.cpp non-determinism) | `AGENTS_DISABLE_LLM=1` |
| `rag_research` | Embedding model + BM25 retrieval | No (no disable mechanism) |
| `synthesizer` | LLM narrative generation | `AGENTS_DISABLE_LLM=1` |
| `reflection` | LLM self-critique | `AGENTS_DISABLE_LLM=1` |

### Report structure (synthesizer.py:344-372)

29 fields currently. **Missing:** `model_provenance` (model_hash, checkpoint_path, architecture, schema_version).

### On-chain anchoring (AuditRegistry.sol:28-34)

`AuditResult` struct has: `scoreFieldElement`, `proofHash`, `timestamp`, `agent`, `verified`. **Missing:** `modelHash` to bind report to model version.

---

## P5 design

### `SENTINEL_DETERMINISTIC=1` mode

When set:
1. **ML module:** `torch.use_deterministic_algorithms(True)` + `torch.manual_seed(42)` at startup
2. **Agents module:** Implies `AGENTS_DISABLE_LLM=1` (skip debate, narrative, reflection)
3. **RAG:** Skip RAG retrieval (set `rag_results = []` in `rag_research.py`)
4. **Report:** Include `model_provenance` section with hash

**Why this matters:** The product goal is a ZK-proved on-chain oracle. The anchored verdict must be reproducible. The LLM debate is non-deterministic (even at temp=0.0) — it cannot be ZK-proven. So we prove only the deterministic tier: ML model + static tools + `fuse()` math.

### Model hash propagation

```
ML API startup:
  1. Compute SHA-256 of checkpoint file
  2. Store in Predictor instance
  3. Expose via /health endpoint + /predict response

Agents pipeline:
  4. ml_assessment reads model_hash from ML API response
  5. Store in state["model_hash"]
  6. synthesizer includes in final_report["model_provenance"]

On-chain (optional, P9):
  7. Add modelHash field to AuditResult struct
  8. Anchor model_hash alongside proofHash
```

---

## P5 tasks

### T5.1 — Wire model hash into ML API

**What:** Compute model hash at startup and expose via API.

**Changes:**

1. **`ml/src/inference/predictor.py`:**
   - Add `self.model_hash: str` attribute
   - In `__init__()`, compute hash after loading checkpoint:
     ```python
     self.model_hash = self._compute_file_hash(checkpoint)
     ```
   - Add `_compute_file_hash()` method (copy from `auto_reproducibility_check.py:49-55`)

2. **`ml/src/inference/api.py`:**
   - Add `model_hash` to `/health` response (line 143-148):
     ```python
     return {"status": "healthy", "model_hash": predictor.model_hash, ...}
     ```
   - Add `model_hash` to `/predict` response (line 180-220):
     ```python
     return {"probabilities": ..., "model_hash": predictor.model_hash, ...}
     ```

3. **Tests:**
   - `ml/tests/test_api.py`: assert `/health` returns `model_hash`
   - `ml/tests/test_api.py`: assert `/predict` returns `model_hash`
   - `ml/tests/test_api.py`: assert hash is 64-char hex string

**Acceptance:**
- `curl http://localhost:8001/health` returns `{"status": "healthy", "model_hash": "abc123...", ...}`
- `/predict` response includes `model_hash`
- Hash matches `auto_reproducibility_check.py` output for same checkpoint

### T5.2 — Propagate model hash through agents pipeline

**What:** Read `model_hash` from ML API response and store in state.

**Changes:**

1. **`agents/src/orchestration/state.py`:**
   - Add `model_hash: str` field to `AuditState` TypedDict

2. **`agents/src/orchestration/nodes/ml_assessment.py`:**
   - Extract `model_hash` from MCP tool response (line 79-83):
     ```python
     result = await _h._call_mcp_tool(...)
     model_hash = result.get("model_hash", "")
     return {"ml_result": result, "model_hash": model_hash}
     ```

3. **`agents/src/mcp/servers/inference_server.py`:**
   - Forward `model_hash` from ML API response to agents (check if it's already forwarded or needs to be added)

**Acceptance:**
- `state["model_hash"]` is populated after `ml_assessment` node
- Hash matches ML API's `/predict` response

### T5.3 — Implement `SENTINEL_DETERMINISTIC=1` mode

**What:** Add deterministic mode that disables non-deterministic components.

**Changes:**

1. **`ml/src/inference/api.py` — `lifespan()` function (line 97-116):**
   - At startup, check env var:
     ```python
     if os.getenv("SENTINEL_DETERMINISTIC", "").strip().lower() in ("1", "true", "yes"):
         torch.use_deterministic_algorithms(True)
         torch.manual_seed(42)
         logger.info("SENTINEL_DETERMINISTIC mode enabled")
     ```

2. **`agents/src/orchestration/nodes/_helpers.py`:**
   - Update `_llm_enabled()` to check `SENTINEL_DETERMINISTIC`:
     ```python
     def _llm_enabled() -> bool:
         if os.getenv("SENTINEL_DETERMINISTIC", "").strip().lower() in ("1", "true", "yes"):
             return False
         return os.getenv("AGENTS_DISABLE_LLM", "").strip().lower() not in ("1", "true", "yes")
     ```

3. **`agents/src/orchestration/nodes/rag_research.py`:**
   - Check deterministic mode at start:
     ```python
     if os.getenv("SENTINEL_DETERMINISTIC", "").strip().lower() in ("1", "true", "yes"):
         logger.info("rag_research | skipped (SENTINEL_DETERMINISTIC mode)")
         return {"rag_results": []}
     ```

4. **Tests:**
   - `agents/tests/test_deterministic_mode.py`: assert `_llm_enabled()` returns `False` when `SENTINEL_DETERMINISTIC=1`
   - `ml/tests/test_api.py`: assert `/health` returns `deterministic_mode: true` when set

**Acceptance:**
- With `SENTINEL_DETERMINISTIC=1`: LLM nodes skip, RAG skips, torch deterministic algorithms enabled
- Without it: behavior unchanged

### T5.4 — Add `model_provenance` to final report

**What:** Include model hash and metadata in the audit report.

**Changes:**

1. **`agents/src/orchestration/nodes/synthesizer.py` (line 344-372):**
   - Add `model_provenance` section to report:
     ```python
     report = {
         ...
         "model_provenance": {
             "model_hash": state.get("model_hash", ""),
             "checkpoint_path": os.getenv("SENTINEL_CHECKPOINT", ""),
             "schema_version": "v9",  # from FEATURE_SCHEMA_VERSION
             "deterministic_mode": os.getenv("SENTINEL_DETERMINISTIC", "").lower() in ("1", "true"),
         },
     }
     ```

**Acceptance:**
- Report JSON includes `model_provenance` section
- `model_hash` matches ML API's hash
- `deterministic_mode` reflects env var state

### T5.5 — End-to-end reproducibility test

**What:** Assert that the same contract produces identical `verdict_provable` across two runs with `SENTINEL_DETERMINISTIC=1`.

**Test:** `agents/tests/test_reproducibility_e2e.py`

```python
@pytest.mark.asyncio
async def test_reproducibility_deterministic_mode():
    """Same contract → same verdict_provable across two runs."""
    contract = "// SPDX-License-Identifier: MIT\npragma solidity ^0.8.0;\ncontract Test { ... }"
    
    os.environ["SENTINEL_DETERMINISTIC"] = "1"
    
    # Run 1
    state1 = await run_audit(contract, "0x1234...")
    verdict1 = state1["verdict_provable"]
    
    # Run 2
    state2 = await run_audit(contract, "0x1234...")
    verdict2 = state2["verdict_provable"]
    
    # Assert identical
    assert verdict1 == verdict2
    assert state1["model_hash"] == state2["model_hash"]
```

**Acceptance:**
- Test passes with `SENTINEL_DETERMINISTIC=1`
- Test would fail without it (LLM debate introduces variance)

### T5.6 — (Optional) Add `modelHash` to on-chain anchoring

**What:** Extend `AuditRegistry.sol` to include `modelHash` in `AuditResult`.

**Changes:**

1. **`contracts/src/AuditRegistry.sol`:**
   - Add `bytes32 modelHash` field to `AuditResult` struct (line 28-34)
   - Update `submitAudit()` to accept `modelHash` parameter
   - Update `AuditSubmitted` event to include `modelHash`

2. **`agents/src/mcp/servers/audit_server.py`:**
   - Update `_decode_audit_result()` to decode `modelHash`
   - Update `submit_audit()` tool to accept and forward `modelHash`

**Acceptance:**
- Contract compiles
- Tests pass
- `modelHash` is anchored on-chain alongside `proofHash`

**Note:** This is optional for P5. The core reproducibility guarantee (T5.1-T5.5) does not require on-chain anchoring — that's P9's job. But if we do it now, it's ready for P9.

---

## P5 deliverables

| File | LOC (est.) | Description |
|------|-----------|-------------|
| `ml/src/inference/predictor.py` | modified | Add `model_hash` attribute + `_compute_file_hash()` method |
| `ml/src/inference/api.py` | modified | Expose `model_hash` via `/health` + `/predict`; add `SENTINEL_DETERMINISTIC` startup logic |
| `agents/src/orchestration/state.py` | modified | Add `model_hash: str` field |
| `agents/src/orchestration/nodes/ml_assessment.py` | modified | Extract `model_hash` from ML API response |
| `agents/src/orchestration/nodes/_helpers.py` | modified | Update `_llm_enabled()` to check `SENTINEL_DETERMINISTIC` |
| `agents/src/orchestration/nodes/rag_research.py` | modified | Skip RAG in deterministic mode |
| `agents/src/orchestration/nodes/synthesizer.py` | modified | Add `model_provenance` to report |
| `agents/src/mcp/servers/inference_server.py` | modified | Forward `model_hash` from ML API (if needed) |
| `ml/tests/test_api.py` | modified | Tests for `model_hash` in `/health` + `/predict` |
| `agents/tests/test_deterministic_mode.py` | ~50 | Tests for `SENTINEL_DETERMINISTIC` mode |
| `agents/tests/test_reproducibility_e2e.py` | ~80 | End-to-end reproducibility test |
| `contracts/src/AuditRegistry.sol` | modified | (Optional) Add `modelHash` field |

---

## Critical DoD-test gates

| Gate | Where | What it asserts |
|------|-------|-----------------|
| Model hash exposed | `test_api.py` | `/health` and `/predict` return `model_hash` |
| Hash consistency | `test_api.py` | Hash matches `auto_reproducibility_check.py` output |
| Deterministic mode | `test_deterministic_mode.py` | `SENTINEL_DETERMINISTIC=1` disables LLM + RAG |
| Report includes provenance | `test_reproducibility_e2e.py` | `final_report["model_provenance"]` present with hash |
| End-to-end reproducibility | `test_reproducibility_e2e.py` | Same contract → same `verdict_provable` across runs |
| Full suite green | `pytest -q` | 586+ passed, 3 skipped (no regressions) |

---

## Ordering & effort

1. **T5.1 — Wire model hash into ML API** (~0.5 day): Add hash computation + expose via API. Tests.
2. **T5.2 — Propagate hash through agents** (~0.25 day): Read from ML response → state. Quick.
3. **T5.3 — Implement `SENTINEL_DETERMINISTIC` mode** (~0.5 day): torch deterministic + disable LLM/RAG. Tests.
4. **T5.4 — Add `model_provenance` to report** (~0.25 day): Add section to report. Quick.
5. **T5.5 — End-to-end reproducibility test** (~0.5 day): Cross-run test with deterministic mode.
6. **T5.6 — (Optional) On-chain `modelHash`** (~1 day): Contract upgrade + decode logic.

**Total: ~3 days (without T5.6) or ~4 days (with T5.6).**

---

## Rollback plan

| Step | What fails | Rollback |
|------|-----------|----------|
| T5.1 model hash | Hash computation fails or mismatches | Don't expose via API; debug hash function |
| T5.2 propagation | `model_hash` not in MCP response | Check inference_server.py forwarding; revert ml_assessment.py |
| T5.3 deterministic mode | `torch.use_deterministic_algorithms` raises (no deterministic alternative for some op) | Use `warn_only=True` or skip torch deterministic (keep LLM/RAG disable only) |
| T5.4 report | Report structure breaks | Revert synthesizer.py; `model_provenance` is additive, not required |
| T5.5 reproducibility test | Test fails (verdicts differ across runs) | Investigate: is there still non-determinism? Check torch ops, LM Studio backend, RAG embedding model |
| T5.6 on-chain | Contract upgrade breaks existing tests | Revert contract changes; P5 core (T5.1-T5.5) is complete without it |

---

## Risks (P5-specific)

| Risk | L | I | Mitigation |
|------|---|---|-----------|
| `torch.use_deterministic_algorithms` raises on CUDA ops | High | Med | Use `warn_only=True` flag; or skip torch deterministic and rely on LLM/RAG disable only |
| LM Studio backend is non-deterministic even at temp=0.0 | Med | High | `SENTINEL_DETERMINISTIC=1` disables LLM entirely — this is the point (prove only deterministic tier) |
| RAG embedding model is non-deterministic | Med | Med | `SENTINEL_DETERMINISTIC=1` skips RAG — acceptable because RAG is `deterministic=False` anyway |
| Model hash changes on every restart (file hash vs state_dict hash) | Low | Med | Use file hash (SHA-256 of `.pt` file) — stable across restarts unless checkpoint is replaced |
| End-to-end test is flaky (timing, external services) | Med | Med | Mock ML API + MCP servers in test; use fixed contract; assert `verdict_provable` (deterministic tier only) |
| On-chain `modelHash` requires contract migration | Low | High | Defer T5.6 to P9; P5 core does not require on-chain changes |

---

## What this plan deliberately defers

- **Full ZK proof of `fuse()` math** — P9's job. P5 only ensures the deterministic tier is reproducible; P9 will prove it cryptographically.
- **Model hash in on-chain anchor (T5.6)** — Optional for P5. Can defer to P9 if contract upgrade is risky.
- **Cross-node consensus for LLM verdicts** — EXT (decentralization) phase. P5 focuses on the deterministic tier.
- **Model retrain trigger for hash update** — The hash updates automatically when the checkpoint changes; no special mechanism needed.
