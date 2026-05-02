# SENTINEL — Roadmap

Last updated: 2026-05-02

This file tracks upcoming work in priority order. Completed items move to
`docs/changes/` as dated changelogs. See `docs/STATUS.md` for current module state.

---

## Immediate Next (ordered)

### Move 0 — Validate Graph Dataset for edge_attr (30 min) ⚠️ BLOCKS RETRAIN

**Why:** P0-B (`gnn_encoder.py`) degrades gracefully to zero-vectors when `edge_attr is None`
in old `.pt` files — it does not crash. However, training on zero-vectors defeats the entire
purpose of P0-B: the model learns nothing about edge relation types.
We must confirm that `ml/data/graphs/*.pt` files actually contain `edge_attr` tensors with
shape `[E]` and values in `[0, 4]` before the retrain captures any benefit from P0-B.

**New file:** `ml/scripts/validate_graph_dataset.py`
- Walk all `.pt` files in `ml/data/graphs/`
- Per file: check `hasattr(g, 'edge_attr')`, shape is `[E]` (1-D), values in `[0, NUM_EDGE_TYPES)`
- Report: total files, files with valid `edge_attr`, files with missing/wrong-shape `edge_attr`
- Exit non-zero if any file fails — CI-safe
- Note: `graph_schema.py` comments that pre-refactor files may have shape `[E, 1]` (old
  `ast_extractor.py`) — the script should detect and report this as a shape mismatch

**Verification:** Script exits 0 on current dataset; prints per-file summary.

---

### Move 1 — Confirm Audit #13 Closed (5 min)

**Why:** The ROADMAP previously listed audit item #13 (FocalLoss scalar cast) as an open
task. Source code inspection (2026-05-02) confirms it is **already fixed**:
- `focalloss.py`: `predictions.float()` / `targets.float()` cast at top of `forward()` (Fix #6, 2026-05-01)
- `trainer.py`: `_FocalFromLogits` applies `logits.float()` before sigmoid (Fix #2, 2026-05-01)

**Action:** Close this item. No code change needed. Update open items in `docs/STATUS.md`.

---

### Move 2 — T1-A: Inference Cache (2h)

**Why:** Inference cache (`cache.py`) is listed as complete in STATUS but needs
integration verification — confirm `process_source()` cache hit returns in < 50ms
on second call with same contract content.

**Files:**
- `ml/src/inference/cache.py` — already created
- `ml/src/inference/preprocess.py` — optional cache in `ContractPreprocessor.__init__()`

**Verification:** Second call to `process_source()` on same contract returns in < 50ms.

---

### Move 3 — T2-A: Prometheus Metrics (1h)

**Why:** M6 will route production traffic to the ML service. Without metrics there is
no way to detect GPU memory pressure, latency spikes, or rising error rates.
T2-B drift detection also depends on Prometheus counters being available.

**Files:**
- `ml/pyproject.toml`: add `prometheus-fastapi-instrumentator>=0.9`
- `ml/src/inference/api.py`: `Instrumentator().instrument(app).expose(app)`; custom
  `Gauge("sentinel_gpu_memory_bytes")` and `Gauge("sentinel_model_loaded")`

---

### Move 4 — T2-C: MLflow Model Registry Script (2h)

**Why:** Current model promotion = manually copy `.pt` file. No audit trail, no staged
rollout. MLflow is already running (`sqlite:///mlruns.db`).

**Files:**
- **New** `ml/scripts/promote_model.py`: CLI `--checkpoint --stage (Staging|Production) --note`;
  `mlflow.register_model()` with `val_f1_macro`, `architecture`, `git_commit` tags

---

### Move 5 — T3-A: LLM Synthesizer Upgrade (2h)

**Why:** Current synthesizer picks from 3 static strings — no Solidity-specific guidance,
no exploit analysis. `get_strong_llm()` in `agents/src/llm/client.py` is already wired.

**Files:**
- `agents/src/orchestration/nodes.py`: structured prompt → markdown with severity/exploit/fix;
  fallback to rule-based when LLM unavailable
- `agents/src/orchestration/state.py`: add `narrative: str | None = None`

---

### Move 6 — T3-B: Cross-Encoder Re-Ranking in RAG (1h)

**Why:** RRF gives good recall but imprecise ranking. A cross-encoder reads query + chunk
bidirectionally — more accurate relevance. Off by default (`rerank=False`).

**Files:**
- `agents/src/rag/retriever.py`: `rerank: bool = False` param; after RRF top-20 →
  `CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2").predict()`, re-sort
- `agents/pyproject.toml`: add `sentence-transformers>=2.2`

---

### Move 7 — T2-B: Drift Detection (3h)

**Why:** Model trained on BCCC-SCsVul-2024. Production contracts from 2026+ may degrade
silently. KS test is the standard tool; wire after T2-A so Prometheus counters are available.

⚠️ **Baseline strategy — read before implementing:**
Do NOT compute the baseline from `ml/data/graphs/` (training data).
The BCCC-SCsVul-2024 corpus is a 2024 historical snapshot; using it as a baseline
will cause the KS test to fire on virtually every modern 2026 contract, making alerts meaningless.

**Correct approach:**
- `compute_drift_baseline.py` must support `--source [warmup|training]`
- Default and recommended: `--source warmup` — collect stats from first 500 real requests,
  then write `drift_baseline.json`; alerts are suppressed during warm-up
- `--source training` is available for testing but prints a prominent warning

**Files:**
- **New** `ml/src/inference/drift_detector.py`: `DriftDetector`; KS test per stat vs baseline;
  rolling buffer 200 requests; `sentinel_drift_alerts_total` Prometheus counter on p < 0.05;
  warm-up mode suppresses alerts until N >= 500
- **New** `ml/scripts/compute_drift_baseline.py`: `--source warmup|training`;
  walks request log or `ml/data/graphs/`; writes `drift_baseline.json`
- `ml/src/inference/api.py`: `detector.update()` per request; `detector.check()` every 50

---

### Move 8 — ML Audit Items #9, #11 (45 min)

Note: Audit item #13 (FocalLoss scalar cast) is already fixed — see Move 1.

| Item | File | Fix |
|------|------|-----|
| #9 | `preprocess.py` | Temp file not cleaned on SIGKILL |
| #11 | `dual_path_dataset.py` | RAM cache loaded without integrity check |

---

## Then: Retrain

After Moves 0–8, retrain the model with the new architecture (P0-A/B/C):
- Edge relation type embeddings (P0-B) — biggest expected quality gain
- Configurable LoRA rank — experiment with r=16, r=32
- All architecture params now in MLflow for comparison

### Retrain Evaluation Protocol

Before launching the retrain, confirm all of the following:

1. `validate_graph_dataset.py` (Move 0) exits 0 — confirms `edge_attr` is present in `.pt` files
2. Held-out split is fixed — use `ml/data/splits/val_indices.npy` with the **same seed**; do NOT regenerate
3. MLflow experiment `sentinel-retrain-v2` created; baseline run ID from `sentinel-multilabel` recorded
4. **Success gate:** val F1-macro > **0.4679** on the same held-out split
5. **Per-class floor:** no class drops > 0.05 F1 from pre-retrain values (log per-class F1 in MLflow)
6. **Rollback rule:** if new F1 < 0.4679 after 40 epochs, revert to `multilabel_crossattn_best.pt`
   and investigate P0-B `edge_emb_dim` (try 8 instead of 16) before re-running

---

## Then: M6 Integration API (Sprint)

Build after the building blocks are solid:
- T1-A cache → repeated audits are fast from day one
- T2-A Prometheus → M6 can instrument from day one
- T3-A LLM synthesizer → first user-facing reports are meaningful
- T2-B drift → production degradation visible immediately

### Security Design (complete before building routes)

This is a smart contract security tool — the API itself must be secure from day one.
Design these before writing a single route:

- **Auth:** `Authorization: Bearer <key>` header; validated in a FastAPI dependency (`Depends`)
- **API key storage:** env var `SENTINEL_API_KEYS` (comma-separated); never hardcoded
- **Rate limiting:** `slowapi` or Redis token bucket; max 10 audits/min per key
- **Contract confidentiality:** audit job payloads must not be logged at INFO level — use DEBUG only
- **Input validation:** max contract size (e.g. 500KB); reject non-UTF-8 payloads before Slither

### New directory: `api/`
- `api/main.py` — FastAPI + lifespan + Prometheus
- `api/routes/audit.py` — `POST /v1/audit` → `{job_id}`; `GET /v1/audit/{id}`
- `api/routes/proof.py` — `GET /v1/proof/{id}`
- `api/tasks/audit_task.py` — Celery task wrapping `build_graph().ainvoke()`
- `api/tasks/proof_task.py` — Celery task wrapping EZKL `run_proof.py`
- `docker-compose.yml` — full stack (api, ml-server, mcp servers, redis, postgres)

---

## ZKML Pipeline Resolution (S5.5)

M2 has been "source complete, not yet run" since 2026-04-29 with no scheduled move
to resolve it. This is a decision point — one of the two options must be chosen:

**Option A — Run it:**
  - Set up EZKL environment locally (GPU + local graph data required)
  - Export ONNX model from current checkpoint
  - Run `ezkl gen-settings`, `ezkl calibrate`, `ezkl prove`
  - Verify Solidity verifier contract compiles
  - Mark M2 as fully complete

**Option B — Descope for now:**
  - Move M2 to a future sprint (S10 Advanced ZK)
  - Document the reason (environment setup cost > current value)
  - Remove M2 from active sprint tracking

Add a `S5.5 ZKML Validation` sprint row to the later sprints table once the decision is made.

---

## Later Sprints

| Sprint | Goal | Key Skills |
|--------|------|-----------|
| S6 Observability | Prometheus, OpenTelemetry/Jaeger, Loki | Distributed tracing, structured logs |
| S7 CI/CD | GitHub Actions, Semgrep, Bandit, Grype | SAST/DAST, SBOM supply-chain security |
| S8 Advanced contracts | Echidna property fuzzing, Halmos symbolic proofs | Invariant testing, mathematical guarantees |
| S9 Advanced ML | Online drift, MLflow registry, LLM explanations | KS test, canary deployment, structured LLM output |
| S10 Advanced ZK | Env config, circuit versioning on-chain, proof aggregation | Recursive SNARKs, gas optimization |

---

## Unit Test Plan for New Stateful Modules

These modules are IO-heavy and stateful — bugs are silent and expensive without tests.
Add these before or alongside each Move:

| Module | Move | Key test cases |
|--------|------|----------------|
| `cache.py` | Move 2 | Cache miss writes files; cache hit returns same object; TTL expiry evicts entry; cache key includes schema version |
| `drift_detector.py` | Move 7 | Warm-up mode suppresses alerts; KS fires on p < 0.05; buffer rolls after 200 requests |
| `promote_model.py` | Move 4 | CLI rejects unknown stage names; MLflow tags are written; dry-run mode does not register |

---

## Tool/Skill Reference

### ML / Training
- **Sliding-window NLP**: handling > 512 tokens without long-context models — T1-C (done)
- **Content-addressed caching**: Redis/disk pattern for ML feature store — T1-A (done)
- **LoRA rank tuning**: r=8 (default) → try r=16 or r=32 for more capacity
- **FocalLoss**: `TrainConfig(loss_fn="focal")` — down-weights easy negatives; FP32 cast fixed
- **KS drift detection**: `scipy.stats.ks_2samp` — production ML monitoring standard
- **MLflow Model Registry**: staged rollout (None → Staging → Production) with audit trail

### Agents / RAG
- **Cross-encoder reranking**: two-stage retrieval (bi-encoder recall → cross-encoder precision)
- **Solodit knowledge source**: ~50K professional audit findings as RAG corpus
- **Immunefi severity weighting**: bug bounty data with `bounty_usd` for risk-adaptive scoring
- **LangGraph parallel nodes**: `Send` API for concurrent rag_research ∥ static_analysis
- **Structured LLM output**: Pydantic-validated LLM responses for security reports

### Infrastructure
- **Docker multi-stage builds**: builder → runtime (no build tools in final image)
- **Celery + Redis**: async task queue for audit and proof jobs
- **OTLP / Jaeger**: W3C traceparent propagation across API → ML → RAG hops

### Smart Contract Analysis
- **Echidna**: property-based fuzzer — breaks invariants with random transaction sequences
- **Halmos**: symbolic testing — proves properties hold for ALL inputs (not just sampled)
- **Aderyn**: second-opinion static analyzer alongside Slither (different ruleset)
- **SMTChecker**: formal arithmetic verification at compile time (overflow, assertion violations)
- **Mythril**: symbolic execution — path-sensitive logical flaw detection

### ZK Infrastructure
- **EZKL**: ZK circuit from ONNX model — proving_key, verification_key, Solidity verifier
- **Recursive SNARKs / Groth16 aggregation**: batch 10 proofs → 1 on-chain verification
- **CycloneDX + Grype**: SBOM generation + CVE scanning for supply-chain security
