# SENTINEL — Roadmap

Last updated: 2026-05-02

This file tracks upcoming work in priority order. Completed items move to
`docs/changes/` as dated changelogs. See `docs/STATUS.md` for current module state.

---

## Immediate Next (ordered)

### Move 3 — T2-A: Prometheus Metrics (1h)

**Why:** M6 will route production traffic to the ML service. Without metrics there is
no way to detect GPU memory pressure, latency spikes, or rising error rates.

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

**Files:**
- **New** `ml/src/inference/drift_detector.py`: `DriftDetector`; KS test per stat vs baseline;
  rolling buffer 200 requests; `sentinel_drift_alerts_total` Prometheus counter on p < 0.05
- **New** `ml/scripts/compute_drift_baseline.py`: walks `ml/data/graphs/` → `drift_baseline.json`
- `ml/src/inference/api.py`: `detector.update()` per request; `detector.check()` every 50

---

### Move 8 — ML Audit Items #9, #11, #13 (1h)

| Item | File | Fix |
|------|------|-----|
| #9 | `preprocess.py` | Temp file not cleaned on SIGKILL |
| #11 | `dual_path_dataset.py` | RAM cache loaded without integrity check |
| #13 | `trainer.py` | FocalLoss scalar not cast to `float()` |

---

## Then: Retrain

After Moves 3–8, retrain the model with the new architecture (P0-A/B/C):
- Edge relation type embeddings (P0-B) — biggest expected quality gain
- Configurable LoRA rank — experiment with r=16, r=32
- All architecture params now in MLflow for comparison

---

## Then: M6 Integration API (Sprint)

Build after the building blocks are solid:
- T1-A cache → repeated audits are fast from day one
- T2-A Prometheus → M6 can instrument from day one
- T3-A LLM synthesizer → first user-facing reports are meaningful
- T2-B drift → production degradation visible immediately

**New directory:** `api/`
- `api/main.py` — FastAPI + lifespan + Prometheus
- `api/routes/audit.py` — `POST /v1/audit` → `{job_id}`; `GET /v1/audit/{id}`
- `api/routes/proof.py` — `GET /v1/proof/{id}`
- `api/tasks/audit_task.py` — Celery task wrapping `build_graph().ainvoke()`
- `api/tasks/proof_task.py` — Celery task wrapping EZKL `run_proof.py`
- `docker-compose.yml` — full stack (api, ml-server, mcp servers, redis, postgres)

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

## Tool/Skill Reference

### ML / Training
- **Sliding-window NLP**: handling > 512 tokens without long-context models — T1-C (done)
- **Content-addressed caching**: Redis/disk pattern for ML feature store — T1-A (done)
- **LoRA rank tuning**: r=8 (default) → try r=16 or r=32 for more capacity
- **FocalLoss**: `TrainConfig(loss_fn="focal")` — down-weights easy negatives
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
