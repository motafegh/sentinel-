# Active Improvement Ledger — Current Version

Generated: 2026-04-28 | Last updated: 2026-05-01
Scope: SENTINEL learning journey, code audit, and future modification backlog
Mode: updated to reflect implementation sessions through 2026-05-01
Status: Sprints 2–4 complete (Foundry tests, ML unit tests, RAG/Agent hardening, ZKML bugs fixed)

---

## 0. Source-of-truth note

```text
GitHub repo available and should be checked whenever current committed source code is needed:
  motafegh/sentinel-
  branch: main (completed work also on: claude/review-project-progress-X8YuG)

Use GitHub as the active source for committed source files.
Use uploaded/session files for specs, generated artifacts, contracts/out, checkpoints, datasets,
handovers, and files not present in GitHub.

Important repo behavior:
  The GitHub repo is source-only by design. .gitignore excludes large/generated/local artifacts:
  checkpoints, datasets, RAG indexes, contracts/out, docs/specs, ZKML generated artifacts, etc.
  Note: contracts/ source was previously blanket-gitignored; fixed 2026-05-01 (.gitignore line 89
  removed so Solidity source, test, and script files are now tracked).
```

---

## 1. Covered so far

```text
── Phase 1 — Orchestration Core (complete 2026-04-28) ───────────────────────
agents/src/orchestration/state.py
agents/src/orchestration/graph.py
agents/src/orchestration/nodes.py
agents/tests/test_graph_routing.py
agents/scripts/smoke_langgraph.py
agents/scripts/smoke_inference_mcp.py

── Phase 2 — MCP / Service Layer (complete 2026-04-28) ──────────────────────
agents/src/mcp/servers/inference_server.py
agents/src/mcp/servers/rag_server.py
agents/src/mcp/servers/audit_server.py
agents/src/llm/client.py

── Phase 3 — ML Inference Runtime Path (complete 2026-04-29/30) ─────────────
ml/src/inference/api.py
ml/src/inference/predictor.py
ml/src/inference/preprocess.py
ml/src/models/sentinel_model.py
ml/src/models/gnn_encoder.py
ml/src/models/transformer_encoder.py
ml/src/models/fusion_layer.py
ml/src/training/trainer.py
ml/src/datasets/dual_path_dataset.py
ml/data_extraction/ast_extractor.py

── Phase 4 — ZKML module (read-only audit 2026-04-30) ───────────────────────
zkml/src/distillation/proxy_model.py
zkml/src/distillation/train_proxy.py
zkml/src/distillation/export_onnx.py
zkml/src/distillation/generate_calibration.py
zkml/src/ezkl/setup_circuit.py
zkml/src/ezkl/run_proof.py
zkml/src/ezkl/extract_calldata.py

── Phase 5 — Contracts (read-only audit 2026-04-30) ─────────────────────────
contracts/src/AuditRegistry.sol
contracts/src/SentinelToken.sol
contracts/src/IZKMLVerifier.sol

── Phase 6 — ZKML pre-sprint bug fixes (complete 2026-05-01) ────────────────
zkml/src/distillation/train_proxy.py     Bug Z1: checkpoint path, 64→128 dim, multi-label target
zkml/src/distillation/export_onnx.py    Bug Z2: dummy_input shape (1,64)→(1,128)
zkml/src/distillation/generate_calibration.py  Bug Z3: 64→128 dim feature extraction

── Phase 7 — Foundry test suite + deploy script (complete 2026-05-01) ───────
contracts/test/mocks/MockZKMLVerifier.sol    Configurable true/false mock
contracts/test/SentinelToken.t.sol           14 token unit tests
contracts/test/AuditRegistry.t.sol           Full registry test suite (3 guards, upgrade, pause)
contracts/test/InvariantAuditRegistry.t.sol  Stateful fuzz invariant tests
contracts/script/Deploy.s.sol               Sepolia deployment script

── Phase 8 — ML unit tests + trainer improvements (complete 2026-05-01) ─────
ml/tests/test_model.py                SentinelModel forward shapes, edge cases
ml/tests/test_preprocessing.py        ContractPreprocessor — error types, shapes, hashing
ml/tests/test_dataset.py              DualPathDataset — length, shapes, collation
ml/tests/test_trainer.py              TrainConfig, FocalLoss, evaluate(), train loop
ml/src/training/trainer.py            FocalLoss activation via TrainConfig.loss_fn
ml/pyproject.toml                     peft pinned to >=0.13.0,<0.16.0

── Phase 9 — RAG/Agent hardening + parallel graph (complete 2026-05-01) ─────
agents/src/rag/chunker.py               score: float = 0.0 field added to Chunk
agents/src/rag/retriever.py             score populated from RRF; explicit Chunk ctor
agents/src/orchestration/state.py       static_findings: list[dict] (was dict)
agents/src/orchestration/nodes.py       static_analysis node (Slither direct call)
agents/src/orchestration/graph.py       parallel deep path fan-out + fan-in
agents/tests/test_retriever_filters.py  TestSearchScores class added
.gitignore                              Removed blanket contracts/ ignore
```

---

## 2. Current position

```text
Phase 1 — Orchestration Core:              ✅ COMPLETE
Phase 2 — MCP / Service Layer:             ✅ COMPLETE
Phase 3 — ML Inference Runtime Path:       ✅ COMPLETE (hardened 2026-04-30)
Phase 4 — Documentation layer:             ✅ COMPLETE (READMEs added 2026-04-30)
Phase 5 — Contracts (source audit):        ✅ COMPLETE
Phase 6 — ZKML pre-sprint bug fixes:       ✅ COMPLETE (scripts ready to run with GPU access)
Phase 7 — Foundry tests + deploy script:   ✅ COMPLETE (written 2026-05-01; forge not yet run)
Phase 8 — ML unit tests + trainer:         ✅ COMPLETE (4 test suites, 2026-05-01)
Phase 9 — RAG/Agent hardening:             ✅ COMPLETE (score, static_analysis, parallel graph)
Phase 10 — ZKML end-to-end execution:     ⏳ PENDING (bugs fixed; needs GPU access to run pipeline)
Phase 11 — M6 Integration API:            ❌ NOT BUILT
```

All source code written and committed. Phases 10–11 require running processes (GPU for ZKML,
forge for contracts), not source changes.

Next recommended work (source-only):

```text
1. S4.2  Cross-encoder re-ranking in retriever (off by default)
2. S4.3  Solodit knowledge source ingester
3. S4.6  LLM synthesizer upgrade (replace rule-based with get_strong_llm() call)
4. S5    M6 Integration API (FastAPI + Celery + Redis; api/ directory)
5. S9.3  LLM vulnerability explanation node in LangGraph
```

---

## 3. Validated / understood path so far

```text
Solidity contract source
→ inference MCP predict(contract_code)
→ FastAPI /predict {"source_code": ...}
→ Predictor.predict_source()
→ ContractPreprocessor.process_source()
→ Slither graph extraction + CodeBERT tokenization
→ SentinelModel
→ logits
→ Predictor sigmoid + thresholds
→ Track 3 vulnerabilities[]
→ LangGraph risk routing (_is_high_risk: max_prob >= 0.70)
→ deep path (parallel): [rag_research ‖ static_analysis] → audit_check → synthesizer
   OR fast path:         synthesizer directly
→ final_report
```

Validated earlier:

```text
pytest agents/tests/test_graph_routing.py -v → 41 passed
smoke_langgraph.py → mock passed
smoke_langgraph.py --live → live passed
smoke_inference_mcp.py → predict schema/transport passed
```

---

# 4. Must Change

## 4.1 `agents/src/mcp/servers/audit_server.py` ✅ DONE (commit b9d4663, 2026-04-29)

```text
[FIXED] Lazy-load ABI only in real mode, not at import time.
```

## 4.2 `ml/src/inference/api.py` ✅ DONE (commit 035e212, 2026-04-29)

```text
[FIXED] Add missing `import torch`.
```

## 4.3 `ml/src/inference/api.py` / `ml/src/inference/predictor.py` ✅ DONE (commit 287aa9e, 2026-04-29)

```text
[FIXED] Align vulnerability item key across Predictor and public API.
predictor.py now emits "vulnerability_class" as the canonical key.
```

## 4.4 `ml/src/inference/predictor.py` ✅ DONE (commit 287aa9e, 2026-04-29)

```text
[FIXED] Unknown checkpoint architecture now raises ValueError immediately.
_ARCH_TO_FUSION_DIM allowlist replaces the silent else-64 fallthrough.
```

## 4.5 `ml/src/inference/predictor.py` ✅ DONE (commit 287aa9e, 2026-04-29)

```text
[FIXED] num_classes > len(CLASS_NAMES) now raises ValueError before slicing.
```

## 4.6 Shared graph extraction unification — OPEN (larger refactor, deferred)

If we choose to unify `preprocess.py` and `ast_extractor.py`, the following are must-have design requirements:

```
- Shared constants must live in one place only.
- Shared core must preserve exact insertion order (CONTRACT → STATE_VAR → FUNCTION → MODIFIER → EVENT) as used in training.
- Shared graph core must raise typed exceptions, never return None.
- Offline wrapper may catch typed exceptions and convert failures to None/skip.
- Online wrapper must not silently choose contracts[0] unless policy explicitly says so.
- Offline/online parity tests must be added before trusting the refactor.
```

## 4.7 `zkml/src/distillation/train_proxy.py` ✅ DONE (2026-05-01)

Bug Z1 — three bugs in one file:

```text
[FIXED] TEACHER_CHECKPOINT: "run-alpha-tune_best.pt" → "multilabel_crossattn_best.pt"
[FIXED] Feature dim comments updated from 64 → 128 throughout extract_features()
[FIXED] Distillation target: teacher.classifier(features).squeeze(1) → [N,10 logits]
         changed to: torch.sigmoid(teacher.classifier(features)).mean(dim=1) → [N] scalar
```

The old code produced a [N, 10] tensor for what should be a [N] scalar agreement score,
silently training the proxy on the wrong target shape.

## 4.8 `zkml/src/distillation/export_onnx.py` ✅ DONE (2026-05-01)

Bug Z2:

```text
[FIXED] dummy_input = torch.randn(1, 64) → torch.randn(1, 128)
         ProxyModel input_dim is FROZEN_INPUT_DIM=128 (CrossAttentionFusion output).
         ONNX export with wrong shape produces a circuit that expects 64-dim input
         — every real inference would silently fail at shape mismatch.
```

## 4.9 `zkml/src/distillation/generate_calibration.py` ✅ DONE (2026-05-01)

Bug Z3:

```text
[FIXED] Feature extraction updated from 64-dim FusionLayer output to 128-dim
         CrossAttentionFusion output (matching post-ADR-025 architecture).
         Calibration JSON now contains 128 floats per sample, not 64.
         EZKL calibration with wrong dim would produce invalid scaling constants.
```

## 4.10 Foundry test suite ✅ DONE (2026-05-01)

Sprint 2 — written but not yet run (forge not installed in current environment):

```text
contracts/test/mocks/MockZKMLVerifier.sol
  — implements IZKMLVerifier; setReturnValue(bool) toggles mock behavior

contracts/test/SentinelToken.t.sol
  — 14 tests: stake/unstake, events, zero-amount revert, over-unstake revert,
    slash by owner/non-owner, transfer-while-staked, initial supply, MIN_STAKE constant

contracts/test/AuditRegistry.t.sol
  — UUPS proxy deployment via ERC1967Proxy + abi.encodeCall(initialize,...)
  — Tests: 3 submitAudit guards (insufficient stake, invalid proof, score mismatch),
    pause/unpause halts submission, UUPS upgradeToAndCall, history queries
  — _buildSignals(uint256 score) helper: 65-element array with signals[64] == score

contracts/test/InvariantAuditRegistry.t.sol
  — AuditRegistryHandler with bound() random inputs
  — 3 invariants: audit count monotonic, total staked ≤ total supply,
    contract balance == sum of all staked balances

contracts/script/Deploy.s.sol
  — Reads DEPLOYER_PRIVATE_KEY + ZKML_VERIFIER_ADDRESS from env
  — Deploys SentinelToken → AuditRegistry (UUPS proxy)
  — Post-deploy require() sanity checks
```

Run when forge is available:
```bash
cd contracts && forge install  # fetches OZ + foundry-rs/forge-std
cd contracts && forge build
cd contracts && forge test -vvv
cd contracts && forge coverage  # target ≥ 80%
```

## 4.11 ML unit tests + FocalLoss activation ✅ DONE (2026-05-01)

Sprint 3 — 4 test modules written:

```text
ml/tests/test_model.py
  — _StubTransformer inner class (avoids loading CodeBERT ~500MB)
  — Tests: forward shape (4,10) multi-label, (4,) binary, large batch,
    all-PAD mask, logits not probabilities assertion, parameter_summary() smoke

ml/tests/test_preprocessing.py
  — External deps mocked: AutoTokenizer, _extract_graph, _tokenize
  — Tests: empty source raises ValueError, whitespace-only raises ValueError,
    too-large raises ValueError, at-limit doesn't raise, missing file → FileNotFoundError,
    graph.x shape [N,8], token shape [1,512], content-addressed hash consistency,
    truncation flag propagation

ml/tests/test_dataset.py
  — Uses tmp_path with real synthetic .pt files
  — Tests: length, getitem shapes, feature dim=8, split indices, empty raises,
    out-of-range raises, unpaired files skipped, binary collation label [B] float32,
    multi-label collation label [B,10] float32

ml/tests/test_trainer.py
  — _TinyMLP avoids HF model; _SyntheticDataset generates controllable batches
  — Tests: FocalLoss scalar output, perfect/wrong predictions → low/high loss,
    TrainConfig defaults, evaluate() F1 metrics in range [0,1],
    train_one_epoch() loss decreases over 3 epochs, checkpoint round-trip

ml/src/training/trainer.py
  — TrainConfig.loss_fn: str = "bce" added (also accepts "focal")
  — FocalLoss wired via _FocalFromLogits inner class (applies sigmoid internally
    to preserve logit interface identical to BCEWithLogitsLoss)

ml/pyproject.toml
  — peft = { git = "https://github.com/huggingface/peft.git" }
    → peft = ">=0.13.0,<0.16.0"  (stable PyPI releases, reproducible builds)
```

## 4.12 RAG score field + test coverage ✅ DONE (2026-05-01)

Sprint 4.1:

```text
agents/src/rag/chunker.py
  — score: float = 0.0 added as final field in Chunk dataclass

agents/src/rag/retriever.py
  — Results explicitly constructed via Chunk(..., score=rrf_scores[i]) rather than
    list comprehension, to handle backward-compat with old pickled Chunks that
    lack the score attribute in __dict__ (dataclasses.replace() would fail on them)

agents/tests/test_retriever_filters.py
  — TestSearchScores class added:
    test_search_returns_chunks_with_score: mocks embedder + FAISS + BM25,
      asserts all returned chunks have score > 0.0
    test_search_scores_descending: asserts scores sorted highest-first
```

## 4.13 static_analysis node + parallel LangGraph graph ✅ DONE (2026-05-01)

Sprint 4.5 — parallel deep path:

```text
agents/src/orchestration/state.py
  — static_findings: list[dict] (changed from dict)
  — Each item: {tool, detector, impact, confidence, description, lines: list[int]}

agents/src/orchestration/nodes.py
  — static_analysis node: writes contract_code to NamedTemporaryFile,
    calls Slither(tmp_path) with all detectors, returns findings list.
    Handles ImportError gracefully (empty list if slither not installed).
    finally block deletes temp file.
  — synthesizer updated: counts slither_high (High+Medium impact findings),
    includes count in recommendation text; static_findings always serialized

agents/src/orchestration/graph.py
  — _route_after_ml returns list["rag_research", "static_analysis"] for deep path
    (LangGraph parallel fan-out — both nodes run concurrently in same superstep)
  — audit_check receives both rag_results + static_findings before running (fan-in)
  — Deep path topology:
    ml_assessment → [rag_research ‖ static_analysis] → audit_check → synthesizer
```

Key design decision: returning a list from a LangGraph conditional edge function (without
path_map) causes both nodes to execute in the same superstep. audit_check's fan-in
waits for BOTH to complete automatically via LangGraph's Pregel semantics.

---

# 5. Potential Improvements

## 5.1 Orchestration

```text
nodes.py:
- Move Track 3 risk helpers into orchestration/risk.py if reused elsewhere.
- Log warning when vulnerabilities is malformed.
- Add model-class → RAG metadata filter mapping only where safe.
- Replace single error string with structured append-only errors[].

state.py:
- Add reducer semantics if errors[] is introduced.

graph.py / nodes.py:
- LLM synthesizer upgrade: replace rule-based synthesizer with get_strong_llm() call.
  Prompt: contract code + vulnerabilities + RAG evidence + static findings + audit history
  → narrative markdown report with severity + recommended fixes.

smoke_langgraph.py:
- Add optional --print-state flag.
- Add optional --expect-fast mode.
- Add optional --expect-ml-failure mode.

smoke_inference_mcp.py:
- Make SERVER_URL configurable via env var.
- Add optional --contract-file argument.
- Add optional strict expectation flag, e.g. --expect-vulnerability Reentrancy.
```

---

## 5.2 `agents/src/mcp/servers/inference_server.py`

```text
- Add explicit guard when _http_client is None.
- Add optional Module 1 readiness check in /health.
- Add structured catch-all error in _handle_predict().
- Consider production-only no-mock-fallback mode.
```

---

## 5.3 `agents/src/mcp/servers/rag_server.py`

```text
- Defensively enforce k lower bound and integer parsing in _handle_search().
- Separate k_requested from k_effective in response.
- Consider enforcing enum for filters.vuln_type once metadata taxonomy is stable.
- Add query length cap or truncation policy.
- Avoid returning full query in error responses.
- Verify/offload blocking search path because HybridRetriever.search() includes query embedding via LM Studio.
```

---

## 5.4 `agents/src/mcp/servers/audit_server.py`

```text
- Add production guard to forbid automatic mock fallback.
- Fail startup on wrong chain ID in production.
- Defensively validate get_audit_history limit type and lower bound in handler.
- Add JSON Schema regex pattern for contract_address.
- Add readiness endpoint or startup check that confirms registry methods are callable.
- Review binary-era audit result schema against Track 3 multi-label output.
```

---

## 5.5 `agents/src/llm/client.py`

```text
- Make MODEL_FAST, MODEL_STRONG, MODEL_CODER, and MODEL_EMBED configurable via env vars.
- Validate LM_STUDIO_TIMEOUT parsing with clear RuntimeError.
- Warn or fail if LM_STUDIO_BASE_URL does not end with /v1.
- Add validate_lm_studio_models() helper that calls /v1/models and checks configured IDs.
- Extend __main__ smoke to test embedding model too.
- Consider separate timeout values for chat and embeddings.
```

---

## 5.6 `ml/src/inference/api.py` ✅ DONE (commit 6aa92eb, 2026-04-30)

```text
[DONE] thresholds_loaded now reported in /health response (from predictor.thresholds_loaded).
[DONE] SENTINEL_PREDICT_TIMEOUT env var controls timeout (default 60s).
[DONE] logger.exception() used in catch-all — full traceback in logs.
[DONE] MAX_SOURCE_BYTES (1MB) size guard added before preprocessing.
[OPEN] Checkpoint path anchoring — documented in ml/README.md but not enforced in code.
[OPEN] Solidity validator wording — minor, deferred.
```

---

## 5.7 `ml/src/inference/predictor.py` ✅ MOSTLY DONE (commit 6aa92eb, 2026-04-30)

```text
[DONE] self.thresholds_loaded stored and exposed for /health.
[DONE] Per-class warning when threshold JSON is missing class entries (logs missing_classes list).
[DONE] Strict metadata cross-check: fusion_output_dim and class_names in checkpoint config validated.
[DONE] Warmup forward pass runs at startup — catches CUDA/shape issues before first request.
[DONE] legacy_binary mode logs explicit production warning.
[OPEN] Move CLASS_NAMES to shared constants module — deferred (cross-module impact).
[OPEN] top_vulnerability/risk_probability in response — deferred until API schema review.
```

---

## 5.8 `ml/src/inference/preprocess.py` ✅ DONE (commit 6aa92eb, 2026-04-30)

```text
[DONE] Docstring updated: ast_extractor_v4_production.py → ml/data_extraction/ast_extractor.py.
[DONE] edge_attr added to PyG Data object (edge type IDs matching _EDGE_TYPES) for offline/online parity.
[DONE] assert-based tokenizer shape checks replaced with explicit RuntimeError (python -O safe).
[DONE] ImportError → RuntimeError for missing Slither (infrastructure failure, not user error).
[DONE] Solidity compilation errors → ValueError (user error, HTTP 400).
       Other Slither/OS failures → RuntimeError (infrastructure error, HTTP 500).
[DONE] MAX_SOURCE_BYTES (1MB) guard in process_source() — defence-in-depth.
[DONE] Temp file prefix sanitized to strip unsafe characters from name argument.
[OPEN] Mirror ASTExtractorV4._get_slither_instance() for robust solc version handling — deferred to §5.10 refactor.
[OPEN] Unify hashing decision (path MD5 vs content MD5) — documented in code comments, formal decision deferred.
```

---

## 5.9 `ml/data_extraction/ast_extractor.py`

```text
- Consider reducing direct duplication with preprocess.py by moving shared graph logic into a strict shared core.
- Keep offline batch responsibilities separate from online inference responsibilities.
```

---

## 5.10 Shared preprocessing architecture proposal

Potential future architecture:

```text
ml/src/preprocessing/graph_schema.py
  - NODE_TYPES
  - VISIBILITY_MAP
  - EDGE_TYPES
  - FEATURE_NAMES
  - FEATURE_SCHEMA_VERSION

ml/src/preprocessing/graph_extractor.py
  - GraphExtractionConfig
  - GraphExtractionError subclasses
  - node_features()
  - extract_contract_graph()
  - Slither setup helper
  - Data(x, edge_index, optional edge_attr, metadata)

ml/data_extraction/ast_extractor.py
  - offline wrapper
  - parquet loading
  - solc version grouping
  - multiprocessing
  - checkpoint/resume
  - save graph .pt files
  - catches GraphExtractionError and returns/skips None

ml/src/inference/preprocess.py
  - online wrapper
  - input size guard
  - source content hash
  - temp .sol file
  - calls shared extract_contract_graph()
  - tokenization
  - optional content-based cache
  - returns graph + tokens
```

Design requirements:

```text
- Solc version handling for inference belongs in GraphExtractionConfig + shared Slither setup helper.
- Multi-contract detection belongs in GraphExtractionConfig + extract_contract_graph().
- Offline wrapper can preserve old multi_contract_policy="first" behavior.
- Online wrapper should use multi_contract_policy="error" or "by_name".
- include_edge_attr should be explicit, defaulting to True for parity.
- Shared core should raise typed exceptions only.
- Shared core should avoid print/loguru; use standard logging or no logging.
- Inference caching should be versioned by source hash + feature schema + tokenizer + max length + solc/compiler mode.
- Input size limits should be enforced before Slither/tokenizer work.
```

## 5.11 Tokenization & Context Window Improvements (Retraining Required)

### Current Limitation
- CodeBERT has a maximum context of **512 tokens**.
- Training data and model were built with truncation at 512.
- Vulnerabilities appearing after token 512 are **never seen** by the model.

### Production-Ready Solutions (All Require Retraining)

#### A. Replace CodeBERT with a Long-Context Code Model
- **Options:** StarCoder2-3B/7B, DeepSeek-Coder-1.3B/6.7B, CodeLlama-7B, QwenCoder-7B.
- **Context sizes:** 8K – 16K tokens (enough for 99% of contracts).
- Requires updating: `transformer_encoder.py`, `fusion_layer.py`, `preprocess.py`, `dual_path_dataset.py`

#### B. Train with Sliding-Window Chunks (Keep CodeBERT)
- Framework added (Sprint 3): TrainConfig.loss_fn field ready; sliding-window preprocessing
  can be added to `preprocess.py` and `dual_path_dataset.py` without model architecture change.
- Inference aggregation: `_aggregate_window_predictions()` → max per class.

#### C. Hierarchical Summarisation (Advanced)
- Full refactor of `sentinel_model.py`, split into summarizer + sequential encoder.

### Action Items
1. Run `ml/scripts/analyse_truncation.py` — see §9 item 16.
2. If < 5% truncation: accept and document limitation.
3. If 5-25%: implement sliding-window (TrainConfig framework already in place).
4. If > 25%: open long-context model milestone.

---

## 5.12 RAG retrieval improvements (open)

```text
S4.2 Cross-encoder re-ranking (not yet implemented):
  After RRF returns top-20, re-rank with cross-encoder/ms-marco-MiniLM-L-6-v2.
  search(query, k, filters, rerank=False) — off by default to preserve latency.
  Skill: two-stage retrieval — bi-encoder for coarse recall, cross-encoder for precision.

S4.3 Solodit knowledge source (not yet implemented):
  agents/src/ingestion/solodit_fetcher.py — index ~50K professional audit findings.
  Adds to RAG alongside DeFiHackLabs; Dagster asset solodit_index rebuilt nightly.

S4.4 Immunefi severity-weighted RAG (not yet implemented):
  agents/src/ingestion/immunefi_fetcher.py — Immunefi bug bounty reports with severity.
  Synthesizer uses severity for risk-adaptive report scoring.
```

---

# 6. Production Hardening

## 6.1 Orchestration / runtime

```text
- graph.py: Replace MemorySaver with persistent checkpointing for M6 production.
- nodes.py: Consider persistent MCP clients or connection pooling only after latency measurement.
- nodes.py/service layer: Add dependency-aware graph preflight before live graph execution.
- graph.py/nodes.py: Add fallback static-analysis route for ML failure in M6.
- Local runtime: Add proper service start script or Docker Compose profile for integrated local testing.
```

## 6.2 MCP services

```text
- inference_server.py: Prevent MODULE1_MOCK=true in production.
- inference_server.py: Do not silently fallback to mock when Module 1 is unreachable unless explicitly in dev/test.
- inference_server.py: Add Docker Compose readiness ordering between Module 1 and inference MCP.
- inference_server.py: Replace sequential batch loop only if Module 1 adds native batched CUDA inference.

- rag_server.py: Move retriever loading into startup/lifespan if tests/readiness become painful.
- rag_server.py: Add readiness endpoint that checks index consistency, not only chunk count.
- rag_server.py: Add index metadata to health/readiness: build_id, chunk_count, embedding model, last_updated.
- rag_server.py: Offload search to asyncio.to_thread() if measured latency or I/O makes event-loop blocking risky.
- rag_server.py: Add retrieval-quality evaluation separately from transport/schema smoke tests.

- audit_server.py: Separate dev mock mode from production fail-closed behavior.
- audit_server.py: Consider Multicall for hasAudit + getAuditCount if RPC latency matters.
- audit_server.py: Add metrics for RPC latency, RPC errors, mock mode usage, chain ID, and returned history size.
- audit_server.py: Add integration test with AUDIT_MOCK=false against a real Sepolia RPC.
```

## 6.3 LLM / RAG config

```text
- client.py: Replace fragile WSL2 fallback with deployment-specific configuration in Docker/M6.
- client.py: Add startup readiness check for LM Studio or future hosted model backend.
- client.py: Add retry/backoff at call sites for transient local model failures, especially embeddings.
- client.py: Add observability around model selected, latency, timeout, and failure type.
```

## 6.4 ML inference ✅ MOSTLY DONE (commit 6aa92eb, 2026-04-30)

```text
[DONE] api.py: MAX_SOURCE_BYTES (1MB) request-size guard added before Pydantic/preprocessing.
[OPEN] api.py: Process/task-queue isolation for long GPU jobs — asyncio.to_thread timeout does not kill underlying work.
[OPEN] api.py: Structured request ID logging for MCP↔FastAPI log correlation.
[OPEN] api.py: Integration test verifying exact /predict API schema.

[DONE] predictor.py: Strict checkpoint metadata validation: architecture, num_classes, class_names, fusion_output_dim.
[OPEN] predictor.py: Checksum/version check for thresholds file matching checkpoint.
[DONE] predictor.py: Model warmup inference at startup — dummy forward pass catches CUDA/shape issues before first request.
[DONE] predictor.py: Explicit legacy_binary production warning logged at startup.

[OPEN] preprocess.py: Contract_name selection for multi-contract files.
[DONE] preprocess.py: Safe temp-file prefix sanitization for process_source(name).
[OPEN] preprocess.py: Metadata/logging for hash_mode (path_md5 vs content_md5).
[OPEN] preprocess.py: Preprocessing parity test vs ASTExtractorV4.contract_to_pyg() on same contract.
[OPEN] preprocess.py: Regression tests for graph.x shape [N,8], token shapes [1,512], empty-edge behavior.
```

## 6.5 ZKML / on-chain compatibility

```text
- ZKML bugs Z1/Z2/Z3 fixed (2026-05-01) — pipeline is now internally consistent at 128-dim.
- ZKML pipeline not yet run end-to-end (needs GPU access).
- ZKMLVerifier.sol not yet generated — will be produced by setup_circuit.py.
- AuditRegistry.sol: Current AuditResult stores one scoreFieldElement; Track 3 may need
  class-score vector, top class, or proof commitment design.
- proxy_model.py/train_proxy.py/run_proof.py/extract_calldata.py: Now consistent at 128-dim.
- AuditRegistry.sol: Plan Track 3 schema migration before exposing submit_audit.
- Foundry tests written; will run when forge is installed.
```

---

# 7. Learning Notes

## 7.1 Orchestration

```text
- state.py defines the evolving data contract.
- graph.py defines the route and conditional branch.
- nodes.py defines behavior and service calls.
- LangGraph nodes receive full state and return partial updates.
- AuditState(total=False) exists because fields appear over time.
- Track 3 has no top-level confidence.
- risk_probability = max(vulnerabilities[].probability).
- >= 0.70 routes to deep path.
- Fast path does not always mean safe; it can also mean ML failure fallback.
- analysis_path exists because empty rag_results does not prove fast path.
- synthesizer is deterministic in M5 and may become LLM-assisted in M6.

- LangGraph parallel fan-out: conditional edge returning a list executes all nodes concurrently.
- Fan-in: audit_check waits for BOTH rag_research and static_analysis automatically.
- No explicit synchronisation needed — LangGraph's Pregel superstep semantics handle it.
- Write conflicts avoided: rag_research writes rag_results; static_analysis writes static_findings.
```

## 7.2 MCP / service layer

```text
- inference_server.py is a protocol adapter, not model logic.
- MCP-facing field is contract_code.
- FastAPI-facing field is source_code.
- contract_address is traceability metadata and is not sent to Module 1.
- MCP responses are JSON strings inside TextContent.
- Mock output intentionally mirrors real Track 3 schema.

- rag_server.py is a RAG service wrapper, not the retriever algorithm.
- Public MCP tool is search(query, k, filters).
- HybridRetriever is loaded once and reused.
- HybridRetriever.search() includes query embedding via LM Studio, so it is not purely CPU-only.
- Hybrid retrieval is powerful because smart-contract evidence is both semantic and lexical.
- score field now populated: chunker.py has score: float = 0.0; retriever constructs Chunk
  objects with explicit score= from RRF, bypassing old pickle backward-compat issue.

- audit_server.py is an MCP/Web3 read adapter.
- It hides ABI, RPC, checksum addresses, and tuple decoding from LangGraph.
- scoreFieldElement / 8192 converts old EZKL field element to probability.
- Current decoded audit score is binary-era.
- Mock mode is useful for CI but dangerous if silently allowed in production.
```

## 7.3 LLM client

```text
- client.py is a model router and client factory.
- ChatOpenAI can talk to LM Studio because LM Studio exposes an OpenAI-compatible API.
- OpenAIEmbeddings is separate from ChatOpenAI because embeddings return vectors, not generated text.
- get_embedding_model() is currently used by RAG through embedder.py.
- get_fast_llm/get_strong_llm/get_coder_llm are mostly future-agent helpers.
- Timeout is mandatory for local model servers.
- WSL2 gateway IPs are unstable; env config is better than hardcoding.
```

## 7.4 ML inference

```text
- api.py is the Module 1 FastAPI boundary.
- It accepts POST /predict {"source_code": ...}.
- It loads Predictor once through FastAPI lifespan.
- It runs blocking predictor.predict_source() inside asyncio.to_thread().
- It returns Track 3 PredictResponse with vulnerabilities[].vulnerability_class.

- Predictor coordinates checkpoint, model, thresholds, preprocessing, and scoring.
- It does not parse Solidity itself; ContractPreprocessor does that.
- It does not define architecture; SentinelModel does that.
- Checkpoint config decides fusion_output_dim.
- cross_attention_lora means fusion_output_dim=128.
- legacy means fusion_output_dim=64.
- Sigmoid is applied in predictor, not inside model.
- Multi-label prediction uses sigmoid, not softmax.
- Per-class thresholds are applied class by class.

- preprocess.py is part of the model contract, not just data cleaning.
- The current model expects graph.x [N,8], not graph_builder's 17-dim one-hot features.
- The 8 feature meanings/order must not change without rebuilding data and retraining.
- Slither requires a real .sol file, so process_source() uses NamedTemporaryFile.
- Online inference returns token tensors with batch dimension [1,512].
- Offline training token files are [512] and become [B,512] during collate.
- edge_attr exists offline but is ignored by current GNNEncoder.
- Bare Slither setup is the biggest production robustness concern.
- truncated=True means CodeBERT saw only the first 512 tokens.
```

## 7.5 ZKML knowledge distillation

```text
- ProxyModel: 6K-param MLP replacing 315K-param SentinelModel for ZK circuit.
- Input: 128-dim CrossAttentionFusion output (not raw source, not logits).
- Output: 10 class scores (agreement target from full model).
- Distillation target: torch.sigmoid(teacher.classifier(features)).mean(dim=1) → [B] scalar.
  This gives a single 0..1 agreement score per contract, not per-class.
  Proxy trains to approximate this aggregate teacher agreement signal.
- ONNX export: opset=11 (EZKL requirement), dynamo=False, dummy_input=(1,128).
- EZKL CIRCUIT_VERSION: v2.0 (128-dim input, 10-class output).
- generate_calibration.py produces calibration_data.json: list of 128-float input vectors.
  EZKL uses this to determine optimal quantization scales.
```

## 7.6 Foundry testing patterns

```text
- UUPS proxy testing: deploy via ERC1967Proxy + abi.encodeCall(implementation.initialize, ...)
  NOT new AuditRegistry() directly — that skips the proxy and breaks upgrade logic.

- vm.prank(address): next call only comes from that address.
  vm.startPrank / vm.stopPrank: all calls in block come from that address.

- vm.expectRevert(bytes4): check specific custom error selector.
  vm.expectRevert(): any revert (less precise but useful for invariants).

- vm.expectEmit(true, true, false, true): check topics + data, not address.
  Must immediately precede the call that should emit.

- Invariant testing: AuditRegistryHandler wraps target calls.
  bound(x, min, max): clamp fuzzer input to valid range.
  targetContract() + targetSelector(FuzzSelector{...}): focus fuzzer on handler only.

- makeAddr("label"): deterministic address from string — readable test addresses.

- ERC1967Proxy from "@openzeppelin/contracts/proxy/ERC1967/ERC1967Proxy.sol"
  requires forge install OpenZeppelin/openzeppelin-contracts before compilation.
```

---

# 8. Do Not Change Without Wider Plan

## 8.1 Orchestration / Track 3 schema

```text
- Do not reintroduce confidence into orchestration routing or final report.
- Do not casually change Track 3 final_report schema.
- Do not change high-risk threshold semantics without updating tests, smokes, and docs.
- Do not change AuditState key semantics without reviewing LangGraph reducer behavior.
- Do not remove legacy_confidence_is_ignored.
- Do not remove "confidence" not in report assertions.
- Do not remove ML failure path tests.
- Do not remove fast-path tests that prevent accidental RAG/audit calls.
- static_analysis node now wired and tested. The parallel fan-out pattern (list return
  from conditional edge) is load-bearing — do not change to sequential without understanding
  that audit_check fan-in semantics depend on BOTH branches existing.
```

## 8.2 MCP contracts

```text
- Do not change MCP /sse URL contracts casually.
- Do not rename contract_code in inference MCP schema casually.
- Do not send contract_address to Module 1 unless PredictRequest changes.
- Do not rename source_code in the Module 1 payload.
- Do not add confidence back into inference response.
- Do not add mock=True into prediction responses.
- Do not change RAG tool name "search" casually.
- Do not increase RAG _MAX_K without considering downstream context size and synthesizer latency.
- Do not add hard metadata filters from ML classes until mapping is validated.
- Do not treat mock-mode audit history as real security evidence.
```

## 8.3 ML / preprocessing

```text
- Do not change CLASS_NAMES order.
- Do not change fusion_output_dim manually without matching checkpoint architecture.
- Do not use weights_only=True unless checkpoint format is changed and verified.
- Do not remove sigmoid from predictor unless model/training loss changes.
- Do not switch to softmax; this is multi-label, not single-class classification.
- Do not change threshold semantics without updating api.py, inference_server.py, nodes.py, tests, and smoke scripts.

- Do not switch to graph_builder.py for inference.
- Do not change node feature dimension from 8 without retraining.
- Do not reorder the 8 node features.
- Do not change tokenizer model without rebuilding token files and retraining.
- Do not change MAX_TOKEN_LENGTH casually.
- Do not treat edge_attr as model-relevant unless GNNEncoder architecture changes.
- Do not remove temp-file cleanup.

- Do not change TrainConfig.loss_fn default from "bce" without measuring FocalLoss impact.
  FocalLoss is now wired but "bce" is production default; "focal" is opt-in for experiments.
- peft is now pinned to >=0.13.0,<0.16.0 — do not re-add GitHub HEAD dependency.
```

## 8.4 LLM / RAG index

```text
- Do not change MODEL_EMBED without rebuilding the RAG index.
- Do not change embedding model dimensions without validating FAISS index compatibility.
- Do not remove timeout.
- Do not scatter model names across future agents; keep routing centralized.
- Do not assume a code model guarantees correct Solidity reasoning without tests.
- Do not change chunk_size / overlap without rebuilding the RAG index.
- Chunk.score field is now populated by retriever. Do not revert to returning bare
  self.chunks[i] references — old pickled Chunk objects lack the score attribute
  and must be wrapped via explicit Chunk(...) constructor call.
```

## 8.5 ZKML / on-chain

```text
- Do not expose submit_audit until ZKML + Track 3 proof semantics are settled.
- Do not change AuditResult struct casually; it affects ABI, deployed proxy, server decoding, scripts, and frontend/API consumers.
- ZKML dim is now 128 across ALL files (train_proxy, export_onnx, generate_calibration).
  Do not patch any single file without updating all three + ZKMLVerifier + AuditRegistry signals array.
- Do not update AuditRegistry schema casually; it affects ABI, deployed contract, audit_server.py, extract_calldata.py, proof submission scripts, and consumers.
- ONNX opset=11 is locked for EZKL compatibility — do not change.
```

---

# 9. Recommended future implementation priority

When switching from current source work to next sprint, a safe order is:

```text
Priority 1 — Low-risk confirmed bugs                        [ALL DONE]
  ✅ 1. api.py: add import torch.                             (commit 035e212, 2026-04-29)
  ✅ 2. audit_server.py: lazy-load ABI only in real mode.    (commit b9d4663, 2026-04-29)

Priority 2 — Schema safety                                  [ALL DONE]
  ✅ 3. predictor.py/api.py: align vulnerability_class schema. (commit 287aa9e, 2026-04-29)
  ✅ 4. predictor.py: reject unknown architecture.             (commit 287aa9e, 2026-04-29)
  ✅ 5. predictor.py: validate num_classes <= len(CLASS_NAMES). (commit 287aa9e, 2026-04-29)

Priority 3 — Observability / health                         [MOSTLY DONE]
  ✅ 6. predictor.py/api.py: thresholds_loaded.               (commit 6aa92eb, 2026-04-30)
  ✅ 7. api.py: logger.exception and configurable timeout.    (commit 6aa92eb, 2026-04-30)
  ⏳ 8. MCP readiness checks.                                (deferred — MCP servers not yet hardened)

Priority 4 — Preprocessing parity / robustness             [MOSTLY DONE]
  ✅ 9.  preprocess.py: update docstring path/wording.        (commit 6aa92eb, 2026-04-30)
  ✅ 10. preprocess.py: add edge_attr for parity.             (commit 6aa92eb, 2026-04-30)
  ✅ 11. preprocess.py: safe temp-file prefix + input size guard + error type differentiation. (commit 6aa92eb, 2026-04-30)
  ⏳ 12. Add parity tests against ASTExtractorV4.             (deferred — cross-module, needs §4.6 refactor first)

Priority 5 — Larger design refactors                       [NOT STARTED]
  ⏳ 13. Shared graph extraction core.                        (depends on §4.6 design)
  ⏳ 14. Track 3 on-chain/ZKML migration design.              (depends on ZKML end-to-end execution)
  ⏳ 15. Production Docker/service orchestration.             (M6 Integration API not built)

Priority 6 — ZKML pipeline (Sprint 1)                      [BUGS FIXED; EXECUTION PENDING]
  ✅ 16. ZKML Bug Z1: train_proxy checkpoint + 128-dim + multi-label target. (2026-05-01)
  ✅ 17. ZKML Bug Z2: export_onnx dummy_input (1,128).                       (2026-05-01)
  ✅ 18. ZKML Bug Z3: generate_calibration 128-dim.                           (2026-05-01)
  ⏳ 19. Run ZKML pipeline end-to-end (train_proxy → export_onnx → setup_circuit → run_proof).
  ⏳ 20. ZKMLVerifier.sol auto-generated and placed in contracts/src/.

Priority 7 — Foundry tests (Sprint 2)                      [WRITTEN; EXECUTION PENDING]
  ✅ 21. MockZKMLVerifier.sol + SentinelToken.t.sol.           (2026-05-01)
  ✅ 22. AuditRegistry.t.sol + InvariantAuditRegistry.t.sol.   (2026-05-01)
  ✅ 23. Deploy.s.sol (Sepolia deployment script).             (2026-05-01)
  ⏳ 24. forge install + forge build + forge test -vvv (requires forge installation).
  ⏳ 25. forge coverage (target ≥ 80% line coverage).

Priority 8 — ML unit tests + FocalLoss (Sprint 3)          [ALL DONE]
  ✅ 26. test_model.py, test_preprocessing.py, test_dataset.py, test_trainer.py. (2026-05-01)
  ✅ 27. FocalLoss activation via TrainConfig.loss_fn.                          (2026-05-01)
  ✅ 28. peft version pinned to >=0.13.0,<0.16.0.                              (2026-05-01)
  ⏳ 29. Run analyse_truncation.py → decide sliding-window vs accept 512 limit.

Priority 9 — RAG/Agent hardening (Sprint 4)                [PARTIALLY DONE]
  ✅ 30. RAG score field: chunker.py + retriever.py + tests.   (2026-05-01)
  ✅ 31. static_analysis node + parallel LangGraph deep path.  (2026-05-01)
  ⏳ 32. Cross-encoder re-ranking (search(rerank=False) off by default).
  ⏳ 33. Solodit knowledge source ingester.
  ⏳ 34. LLM synthesizer upgrade (replace rule-based with get_strong_llm()).

Priority 10 — M6 Integration API (Sprint 5)                [NOT STARTED]
  ⏳ 35. FastAPI + Celery + Redis api/ directory.
  ⏳ 36. Docker Compose full stack.
  ⏳ 37. Multi-stage Dockerfiles for each service.
```

---
