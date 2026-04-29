# Active Improvement Ledger — Current Version

Generated: 2026-04-28  
Scope: SENTINEL learning journey, code audit, and future modification backlog  
Mode: analysis/learning only — no source-code implementation performed in this session  
Status: updated through `ml/src/inference/preprocess.py` and the `ml/data_extraction/ast_extractor.py` parity discussion

---

## 0. Source-of-truth note

```text
GitHub repo available and should be checked whenever current committed source code is needed:
  motafegh/sentinel-
  branch: main

Use GitHub as the active source for committed source files.
Use uploaded/session files for specs, generated artifacts, contracts/out, checkpoints, datasets,
handovers, and files not present in GitHub.

Important repo behavior:
  The GitHub repo is source-only by design. .gitignore excludes large/generated/local artifacts:
  checkpoints, datasets, RAG indexes, contracts/out, docs/specs, ZKML generated artifacts, etc.
```

---

## 1. Covered so far

```text
agents/src/orchestration/state.py
agents/src/orchestration/graph.py
agents/src/orchestration/nodes.py
agents/tests/test_graph_routing.py
agents/scripts/smoke_langgraph.py
agents/scripts/smoke_inference_mcp.py

agents/src/mcp/servers/inference_server.py
agents/src/mcp/servers/rag_server.py
agents/src/mcp/servers/audit_server.py
agents/src/llm/client.py

ml/src/inference/api.py
ml/src/inference/predictor.py
ml/src/inference/preprocess.py

Related dependency/audit files inspected:
ml/data_extraction/ast_extractor.py
ml/src/models/gnn_encoder.py
ml/src/models/sentinel_model.py
ml/src/datasets/dual_path_dataset.py
```

---

## 2. Current position

```text
Phase 1 — Orchestration Core: complete
Phase 2 — MCP / Service Layer: complete
Phase 3 — ML Inference Runtime Path: preprocessing complete

Current completed file:
  ml/src/inference/preprocess.py

Next recommended file:
  ml/src/models/sentinel_model.py
```

Recommended next session:

```text
Phase 3 / Session 3.4 — ml/src/models/sentinel_model.py
```

Core question:

```text
How does SentinelModel consume graph.x / edge_index / batch and CodeBERT tokens,
then combine GNN + Transformer paths through CrossAttentionFusion to produce Track 3 logits?
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
→ LangGraph risk routing
→ deep/fast path
→ RAG / audit if deep
→ final_report
```

Validated earlier:

```text
pytest tests/test_graph_routing.py -v → 41 passed
smoke_langgraph.py → mock passed
smoke_langgraph.py --live → live passed
smoke_inference_mcp.py → predict schema/transport passed
```

---

# 4. Must Change

## 4.1 `agents/src/mcp/servers/audit_server.py`

```text
- Lazy-load ABI only in real mode, not at import time.
```

Reason:

```text
Current _ABI = _load_abi() runs even in mock mode.
Mock/CI can fail before the server starts if contracts/out/AuditRegistry.sol/AuditRegistry.json is missing.
Mock mode should not require ABI availability unless real Web3 calls are used.
```

---

## 4.2 `ml/src/inference/api.py`

```text
- Add missing `import torch`.
```

Reason:

```text
api.py catches torch.cuda.OutOfMemoryError and calls torch.cuda.empty_cache(),
but torch is not imported. CUDA OOM handling can therefore fail with NameError.
```

---

## 4.3 `ml/src/inference/api.py` / `ml/src/inference/predictor.py`

```text
- Align vulnerability item key across Predictor and public API.
```

Reason:

```text
Predictor currently returns:
  {"class": name, "probability": prob}

External API / MCP / LangGraph expect:
  {"vulnerability_class": name, "probability": prob}

api.py maps "class" → "vulnerability_class", but this creates schema drift.
One schema should flow through Predictor → API → MCP → LangGraph.
```

Preferred future direction:

```python
# predictor.py
{"vulnerability_class": name, "probability": round(prob, 4)}
```

---

## 4.4 `ml/src/inference/predictor.py`

```text
- Reject unknown checkpoint architecture instead of silently treating it as legacy.
```

Reason:

```text
Currently architecture == "cross_attention_lora" gives fusion_output_dim=128,
while every other architecture silently becomes fusion_output_dim=64.

A future or misspelled architecture value could load incorrectly or fail with confusing errors.
Unknown architecture should fail loudly.
```

---

## 4.5 `ml/src/inference/predictor.py`

```text
- Validate num_classes <= len(CLASS_NAMES).
```

Reason:

```text
CLASS_NAMES[:num_classes] silently returns fewer names if num_classes is larger than the class-name list.
Then zip(class_names, probabilities, thresholds) silently drops unmatched model outputs.
That could hide an output class and produce incorrect reports.
```

---

## 4.6 Shared graph extraction unification — required if/when refactor is performed

 if we choose to unify `preprocess.py` and `ast_extractor.py`, the following are must-have design requirements:

```
- Shared constants must live in one place only.
- Shared core must preserve exact insertion order (CONTRACT → STATE_VAR → FUNCTION → MODIFIER → EVENT) as used in training.”
- Shared graph core must raise typed exceptions, never return None.
- Offline wrapper may catch typed exceptions and convert failures to None/skip.
- Online wrapper must not silently choose contracts[0] unless policy explicitly says so.
- Offline/online parity tests must be added before trusting the refactor.
```

Reason:

```text
A cosmetic refactor that only moves duplicated code into a new file would not solve preprocessing drift.
The shared core needs explicit config, typed errors, solc/multi-contract policies, and parity tests.
```

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
- Consider parallel RAG + audit deep path after latency measurement.

test_graph_routing.py:
- Add/keep explicit test for path_taken="deep" when analysis_path="deep" and rag_results=[].

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

## 5.6 `ml/src/inference/api.py`

```text
- Add thresholds_loaded to /health.
- Make predict timeout configurable via env var, e.g. SENTINEL_PREDICT_TIMEOUT.
- Use logger.exception in catch-all error handler.
- Anchor default checkpoint path to project root or document required working directory.
- Improve Solidity validator wording or logic.
```

---

## 5.7 `ml/src/inference/predictor.py`

```text
- Store self.thresholds_loaded for /health.
- Warn or fail when threshold JSON is missing any active class.
- Clarify threshold field semantics when per-class thresholds are active.
- Move CLASS_NAMES to a neutral shared schema/constants module.
- Include optional top_vulnerability/risk_probability only if API schema is intentionally expanded.
```

---

## 5.8 `ml/src/inference/preprocess.py`

```text
- Update docstring reference from ml/scripts/ast_extractor_v4_production.py to ml/data_extraction/ast_extractor.py.
- Clarify “replicates exactly” claim.
- Add edge_attr to online Data for full offline/online graph object parity.
- Mirror or reuse ASTExtractorV4._get_slither_instance() logic.
- Make _add_node() naming logic match offline extractor exactly for Contract vs non-Contract objects.
- Replace assert-based tokenizer shape checks with explicit exceptions.
- Differentiate invalid Solidity errors from infrastructure/runtime extraction errors.
- Decide whether to unify hashing (path vs content) for caching; currently training uses path MD5, process_source uses content MD5.
- Enforce maximum source size (e.g., 1 MB) before Slither/tokenization.
```

Reason:

```text
Current model consumes x + edge_index only, so missing edge_attr is not currently breaking prediction.
But online/offline graph object parity, Slither setup consistency, and clearer errors matter for production.
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

### Production‑Ready Solutions (All Require Retraining)

#### A. Replace CodeBERT with a Long‑Context Code Model
- **Options:** StarCoder2‑3B/7B, DeepSeek‑Coder‑1.3B/6.7B, CodeLlama‑7B, QwenCoder‑7B.
- **Context sizes:** 8K – 16K tokens (enough for 99% of contracts).
- **Files to change:**
  - `ml/src/models/transformer_encoder.py` – replace `AutoModel.from_pretrained("microsoft/codebert-base")` with new model; adjust hidden size (768 → new hidden dim).
  - `ml/src/models/fusion_layer.py` – change `token_dim` from 768 to new model’s hidden size.
  - `ml/src/inference/preprocess.py` – update `TOKENIZER_NAME` to matching tokenizer.
  - `ml/src/datasets/dual_path_dataset.py` – may need to adjust token `.pt` loading if shape changes (likely still [512] but hidden dim different – the tokenizer still outputs IDs, not embeddings, so dataset unchanged).
  - `ml/src/training/trainer.py` – update `config` defaults (batch size, learning rate) for new model.
- **Pros:** Single model, full contract visibility, no chunking complexity.
- **Cons:** Higher VRAM usage (may need 16GB+ for 7B models), longer training time.

#### B. Train with Sliding‑Window Chunks (Keep CodeBERT)
- **Idea:** During training, chunk each contract into overlapping 512‑token windows. Each window gets the same multi‑label as the original contract.
- **At inference:** Run all windows through the model, aggregate predictions (max, mean, or voting).
- **Files to change:**
  - `ml/src/datasets/dual_path_dataset.py` – modify `__getitem__` to return a list of token windows per contract, not a single token dict. Collate function must handle variable number of windows per contract.
  - `ml/src/inference/preprocess.py` – add `_generate_sliding_windows(tokens, overlap)` method.
  - `ml/src/inference/predictor.py` – add `_aggregate_window_predictions()` (max/mean/voting) and call model on each window.
  - `ml/src/training/trainer.py` – update collation and batching for variable‑length window lists.
- **Pros:** No model architecture change; uses existing CodeBERT.
- **Cons:** Increased training data size (chunks multiply samples), slower inference (linear in number of chunks).

#### C. Hierarchical Summarisation (Advanced)
- **Idea:** Train a small model to summarise each function (or code block) into a fixed vector. Then feed a sequence of these summaries to a second model (e.g., a small transformer).
- **Files to change (major refactor):**
  - `ml/src/models/summarizer.py` (new) – small transformer or CNN to encode a function’s token sequence.
  - `ml/src/models/sequential_encoder.py` (new) – processes the sequence of function summaries.
  - `ml/src/models/sentinel_model.py` – replace `TransformerEncoder` with summarizer + sequential encoder.
  - `ml/src/inference/preprocess.py` – need to split contract into functions (using Slither) before tokenizing each function.
  - `ml/src/datasets/dual_path_dataset.py` – store per‑function token sequences.
- **Pros:** Can handle arbitrary length; interpretable.
- **Cons:** Complex, multi‑stage training; may lose fine‑grained details.

### Action Items for Ledger
1. **Analyse truncation impact** – Script that samples real contracts, tokenises them, records:
   - Count exceeding 512 tokens.
   - For known‑vulnerability contracts, whether vulnerable function appears after token 512.
2. **Select candidate long‑context model** – Evaluate StarCoder2‑3B (low VRAM) vs DeepSeek‑Coder‑6.7B. Test inference speed and memory on current GPU.
3. **Implement sliding‑window training (optional)** – As fallback if long‑context model is not feasible.
4. **Document retraining timeline** – Data regeneration, training cost, expected evaluation metrics.

### Decision Required
- After analysis, decide whether to:
  - **Accept 512‑token truncation** (document limitation, warn users).
  - **Retrain with a long‑context model** (best long‑term solution).
  - **Retrain with sliding‑window CodeBERT** (incremental improvement).
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

## 6.4 ML inference

```text
- api.py: Consider request-size limit before Pydantic/model preprocessing.
- api.py: Consider process/task-queue isolation for long GPU jobs; asyncio.to_thread timeout does not necessarily stop underlying work.
- api.py: Add structured request ID logging so inference MCP and FastAPI logs can be correlated.
- api.py: Add integration test verifying exact API schema from /predict.

- predictor.py: Add strict checkpoint metadata validation: architecture, num_classes, class_names, fusion_output_dim.
- predictor.py: Add checksum/version check for thresholds file matching checkpoint.
- predictor.py: Add model warmup inference after startup to catch CUDA/model shape issues early.
- predictor.py: Add explicit legacy mode warning/fail switch so production cannot accidentally run legacy_binary.

- preprocess.py: Add contract_name selection support for multi-contract files.
- preprocess.py: Add safe temp-file prefix sanitization for process_source(name).
- preprocess.py: Add metadata/logging for hash_mode: path_md5 vs content_md5.
- preprocess.py: Add preprocessing parity test comparing online _extract_graph() against ASTExtractorV4.contract_to_pyg() on the same contract.
- preprocess.py: Add regression test for graph.x shape [N,8], token shapes [1,512], and empty-edge behavior.
```

## 6.5 ZKML / on-chain compatibility

```text
- ZKML/on-chain layer: Audit whether current AuditRegistry + proxy files are binary-era and incompatible with Track 3 multi-label outputs.
- AuditRegistry.sol: Current AuditResult stores one scoreFieldElement; Track 3 may need class-score vector, top class, or proof commitment design.
- proxy_model.py/train_proxy.py/run_proof.py/extract_calldata.py: Appear to assume 64-dim FusionLayer and one scalar score; current model uses CrossAttentionFusion output_dim=128 and 10 classes.
- AuditRegistry.sol: Plan Track 3 schema migration before exposing submit_audit.
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
- The current model expects graph.x [N,8], not graph_builder’s 17-dim one-hot features.
- The 8 feature meanings/order must not change without rebuilding data and retraining.
- Slither requires a real .sol file, so process_source() uses NamedTemporaryFile.
- Online inference returns token tensors with batch dimension [1,512].
- Offline training token files are [512] and become [B,512] during collate.
- edge_attr exists offline but is ignored by current GNNEncoder.
- Bare Slither setup is the biggest production robustness concern.
- truncated=True means CodeBERT saw only the first 512 tokens.
```

---

# 8. Do Not Change Without Wider Plan

## 8.1 Orchestration / Track 3 schema

```text
- Do not reintroduce confidence into orchestration routing or final report.
- Do not casually change Track 3 final_report schema.
- Do not change high-risk threshold semantics without updating tests, smokes, and docs.
- Do not wire static_analysis without state fields, node behavior, tests, and failure handling.
- Do not change AuditState key semantics without reviewing LangGraph reducer behavior.
- Do not remove legacy_confidence_is_ignored.
- Do not remove "confidence" not in report assertions.
- Do not remove ML failure path tests.
- Do not remove fast-path tests that prevent accidental RAG/audit calls.
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
```

## 8.4 LLM / RAG index

```text
- Do not change MODEL_EMBED without rebuilding the RAG index.
- Do not change embedding model dimensions without validating FAISS index compatibility.
- Do not remove timeout.
- Do not scatter model names across future agents; keep routing centralized.
- Do not assume a code model guarantees correct Solidity reasoning without tests.
- Do not change chunk_size / overlap without rebuilding the RAG index.
```

## 8.5 ZKML / on-chain

```text
- Do not expose submit_audit until ZKML + Track 3 proof semantics are settled.
- Do not change AuditResult struct casually; it affects ABI, deployed proxy, server decoding, scripts, and frontend/API consumers.
- Do not patch 64 → 128 in ZKML files casually; proxy model, ONNX, EZKL setup, verifier, public signals, and registry checks must migrate together.
- Do not update AuditRegistry schema casually; it affects ABI, deployed contract, audit_server.py, extract_calldata.py, proof submission scripts, and consumers.
```

---

# 9. Recommended future implementation priority

When we later switch from learning/audit to implementation, a safe order is:

```text
Priority 1 — Low-risk confirmed bugs
  1. api.py: add import torch.
  2. audit_server.py: lazy-load ABI only in real mode.

Priority 2 — Schema safety
  3. predictor.py/api.py: align vulnerability_class schema.
  4. predictor.py: reject unknown architecture.
  5. predictor.py: validate num_classes <= len(CLASS_NAMES).

Priority 3 — Observability / health
  6. predictor.py/api.py: thresholds_loaded.
  7. api.py: logger.exception and configurable timeout.
  8. MCP readiness checks.

Priority 4 — Preprocessing parity / robustness
  9. preprocess.py: update docstring path/wording.
  10. preprocess.py: add edge_attr for parity.
  11. preprocess.py: mirror/reuse robust Slither setup.
  12. Add parity tests against ASTExtractorV4.

Priority 5 — Larger design refactors
  13. Shared graph extraction core.
  14. Track 3 on-chain/ZKML migration design.
  15. Production Docker/service orchestration.
```

---

