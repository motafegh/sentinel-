# Plan: Doc 04 — Reproducibility & Determinism: The ZK Boundary

**Spec:** `docs/learning/LEARNING_DOCS_SPEC.md`
**Target:** `docs/learning/04_reproducibility_determinism.md`
**Session:** 2 of 5
**Prerequisite docs:** Doc 01 (Pipeline), Doc 02 (Evidence/Fuse), Doc 03 (Prompt Injection)

---

## Recall from previous docs

**From Doc 01 (Pipeline):** You learned that the pipeline has two determinism levels: control-flow determinism (which nodes run — always guaranteed) and output determinism (same contract → same verdict — only for the deterministic tier). The `_llm_enabled()` function controls whether LLM nodes run.

**From Doc 02 (Evidence/Fuse):** You learned that each `Evidence` has a `deterministic` flag. ML + static tools = `True`. LLM debate = `False`. `verdict_provable` uses only `deterministic=True` evidence. `verdict_full` uses everything. This is the ZK boundary.

**From Doc 03 (Prompt Injection):** You learned that contract source is sanitized before entering LLM prompts. But even sanitized, the LLM's output is non-deterministic — same input can produce different debate verdicts across runs.

**Connection to this doc:** This doc explains how we make the system reproducible for ZK proofs. `SENTINEL_DETERMINISTIC=1` disables all non-deterministic components (LLM, RAG). The model hash (SHA-256 of the checkpoint) binds each verdict to a specific model version. This is the foundation for P9 (EZKL ZK circuit).

**Key concepts carried forward:** `deterministic` flag on Evidence, `verdict_provable` vs `verdict_full`, `_llm_enabled()` function, fail-soft principle.

---

## Step 1: Read source files

- [ ] `agents/src/orchestration/nodes/_helpers.py` (lines 25-35) — `_llm_enabled()` with `SENTINEL_DETERMINISTIC` check
- [ ] `agents/src/orchestration/nodes/rag_research.py` (lines 75-85) — RAG skip in deterministic mode
- [ ] `agents/src/orchestration/nodes/ml_assessment.py` (lines 70-90) — `model_hash` extraction from ML API response
- [ ] `agents/src/orchestration/nodes/synthesizer.py` (lines 210-220, 344-372) — `model_provenance` in report, `injection_matches` initialization
- [ ] `agents/src/orchestration/state.py` (lines 248-254) — `model_hash: str` field in AuditState
- [ ] `ml/src/inference/predictor.py` (lines 350-380) — `_compute_file_hash()` method, SHA-256 of checkpoint
- [ ] `ml/src/inference/api.py` (lines 97-120) — `SENTINEL_DETERMINISTIC` in lifespan, `torch.use_deterministic_algorithms`
- [ ] `ml/src/inference/api.py` (lines 140-160) — `model_hash` in `/health` response
- [ ] `ml/src/inference/api.py` (lines 220-240) — `model_hash` in `/predict` response and `PredictResponse` model
- [ ] `agents/src/mcp/servers/inference_server.py` (lines 80-100) — `model_hash` in mock prediction
- [ ] `agents/tests/test_deterministic_mode.py` — 8 tests (4 LLM disable, 2 RAG skip, 2 reproducibility)

## Step 2: Read scratch files

- [ ] `~/.claude/scratch/p5_reproducibility_20260626.md` — P5 working memory, decisions (file hash vs state_dict hash, deterministic mode scope, T5.6 on-chain deferred)

## Step 3: Read ML tests

- [ ] `ml/tests/test_api.py` — `TestModelHash` class (4 tests: health returns hash, predict returns hash, consistent across requests, stable across restarts)

## Step 4: Write sections

- [ ] **TL;DR:** `SENTINEL_DETERMINISTIC=1` disables LLM + RAG, sets `torch.use_deterministic_algorithms(True)`. Model hash (SHA-256 of `.pt` file) in `/health`, `/predict`, and `final_report["model_provenance"]`. Dual verdict: `verdict_provable` (ZK-anchorable) vs `verdict_full` (human-readable)
- [ ] **The Problem:** ZK proof requires reproducible output. LLM is non-deterministic even at temp=0 (CUDA, llama.cpp backend). Can't ZK-prove a debate. Must prove only the deterministic core
- [ ] **How We Arrived at This Design:** invariant (verdict must be reproducible for ZK) → constraint (LLM non-deterministic, can't be proven) → simplest split (deterministic tier vs full) → stress-test (torch deterministic ops may not cover all ops) → measure (model hash stable across requests)
- [ ] **The Solution:** Deterministic mode flow diagram:
  ```
  SENTINEL_DETERMINISTIC=1
    → _llm_enabled() returns False (disables cross_validator, synthesizer narrative, reflection)
    → rag_research returns {"rag_results": []} (skips embedding model)
    → torch.use_deterministic_algorithms(True) + torch.manual_seed(42)
    → Only deterministic evidence reaches fuse() → verdict_provable is reproducible
  ```
  Model hash propagation chain:
  ```
  ML API startup → _compute_file_hash(checkpoint) → predictor.model_hash
    → /health response → /predict response
    → ml_assessment node extracts → state["model_hash"]
    → synthesizer → final_report["model_provenance"]["model_hash"]
  ```
  Dual verdict diagram (from Doc 02, expanded with deterministic mode)
- [ ] **Key Code:**
  - `_llm_enabled()` (_helpers.py:25-35) — checks both `SENTINEL_DETERMINISTIC` and `AGENTS_DISABLE_LLM`
  - `torch.use_deterministic_algorithms(True)` (api.py lifespan) — set at ML API startup
  - `_compute_file_hash()` (predictor.py:350-380) — SHA-256 of checkpoint file, 8MB chunks
  - `model_hash` in `PredictResponse` (api.py) — Pydantic field, 64 hex chars
  - `model_provenance` in report (synthesizer.py:344-372) — `model_hash`, `checkpoint_path`, `schema_version`
  - `model_hash` in AuditState (state.py:248-254) — propagated through state
- [ ] **Design Decision:** File hash vs state_dict hash vs ONNX hash (tradeoff table: simplicity, stability, precision, format dependence)
- [ ] **Technology Choice:** `torch.use_deterministic_algorithms` (5-question framework: category, alternatives, why this, when different, migration trigger)
- [ ] **Anti-Patterns:**
  - ❌ ZK-prove the LLM — "prove everything." Breaks: LLM is non-deterministic, ZK circuit would be astronomically large. Right: prove only deterministic tier, treat LLM as advisory
  - ❌ temp=0.0 = deterministic — "if temperature is zero, output is reproducible." Breaks: CUDA non-determinism, llama.cpp backend variations, batch effects. Right: disable LLM entirely in deterministic mode
- [ ] **Mistakes & Fixes:**
  - LM Studio at temp=0.0 is still non-deterministic. Same prompt → different outputs across runs. Root cause: CUDA non-determinism + llama.cpp batching. Fix: `SENTINEL_DETERMINISTIC=1` disables LLM entirely
  - `torch.use_deterministic_algorithms(True)` can raise on ops without deterministic alternatives. Fix: `warn_only=True` as fallback (logs warning but doesn't crash)
  - Model hash initially not propagated to report. `state["model_hash"]` field didn't exist. Fix: add to AuditState, extract in ml_assessment from ML API response, include in synthesizer's `model_provenance` section
  - Mock prediction in inference_server.py didn't include `model_hash`. Fix: add `"model_hash": "mock_model_hash_" + "0" * 46` to mock
- [ ] **What Would Break Without This:** Remove `deterministic` flag from Evidence → ZK boundary disappears, fuse() can't separate reproducible from non-reproducible. Remove `SENTINEL_DETERMINISTIC` → can't run in ZK-proof mode. Remove model hash → can't bind verdict to model version, no provenance. Remove `model_provenance` from report → no traceability
- [ ] **At Scale:** 1 model (current) / 5 models / 20 models / 100 models — need model registry, hash per model version, model versioning policy
- [ ] **Try It Yourself:**
  ```
  curl -s localhost:8001/health | python3 -m json.tool   # look for model_hash
  cd agents && source .venv/bin/activate
  SENTINEL_DETERMINISTIC=1 python3 -c "from src.orchestration.nodes._helpers import _llm_enabled; print(_llm_enabled())"  # should print False
  ```
- [ ] **Limitations:** File hash changes on re-serialization (same model, different `.pt` format). `torch.use_deterministic_algorithms` may not cover all ops (warn_only fallback). No ZK circuit yet (P9). No model versioning policy. No automatic refit trigger when model changes
- [ ] **Transferable Patterns:** (1) Dual-tier architecture — provable vs advisory (2) Determinism boundary as a flag — `deterministic: bool` on Evidence (3) Content-addressed provenance — SHA-256 hash as identity. Each with interview story + when wrong.

## Step 5: Verify

- [ ] Open `api.py` and verify `SENTINEL_DETERMINISTIC` check is in `lifespan()` function
- [ ] Open `predictor.py` and verify `_compute_file_hash()` reads file in 8MB chunks
- [ ] Open `_helpers.py` and verify `_llm_enabled()` checks `SENTINEL_DETERMINISTIC` BEFORE `AGENTS_DISABLE_LLM`
- [ ] Open `rag_research.py` and verify the `SENTINEL_DETERMINISTIC` skip returns `{"rag_results": []}`
- [ ] Confirm test count: 8 tests in `test_deterministic_mode.py` + 4 in `test_api.py::TestModelHash`
- [ ] Open `synthesizer.py` and verify `model_provenance` section in report dict
