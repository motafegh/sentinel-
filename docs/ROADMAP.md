# SENTINEL — Roadmap

Last updated: 2026-05-05 (v3 training complete)

This file tracks upcoming work in priority order. Completed items move to
`docs/changes/` as dated changelogs. See `docs/STATUS.md` for current module state.

---

## Completed Pre‑Retrain Moves

These moves were listed as “Immediate Next” on 2026-05-02. They have all been
implemented, except for a single pending sub‑item noted below.

| Move | Description | Status |
|------|-------------|--------|
| Move 0 — Validate Graph Dataset | `validate_graph_dataset.py` created, exits 0 on current data | ✅ Done |
| Move 1 — Confirm Audit #13 Closed | FocalLoss FP32 cast already present; closed | ✅ Done |
| Move 2 — T1-A Inference Cache | `cache.py` integrated; cache hit < 50ms | ✅ Done |
| Move 3 — T2-A Prometheus Metrics | `prometheus-fastapi-instrumentator` added to `api.py` | ✅ Done |
| Move 4 — T2-C MLflow Model Registry | `promote_model.py` CLI added | ✅ Done |
| Move 5 — T3-A LLM Synthesizer Upgrade | qwen3.5-9b-ud + rule‑based fallback implemented | ✅ Done |
| Move 6 — T3-B Cross‑Encoder Re‑ranking | `rerank` param added, ms‑marco‑MiniLM‑L‑6‑v2 wired | ✅ Done |
| Move 7 — T2-B Drift Detection | `drift_detector.py` + `compute_drift_baseline.py` added | ✅ Done |
| Move 8 — ML Audit Items #9, #11 | Item #11 (RAM cache integrity) fixed; Item #9 (preprocess temp file on SIGKILL) still pending | ⚠️ Partial |

**Remaining from Move 8:**
- **Audit item #9** in `ml/src/inference/preprocess.py`: temporary file not cleaned up on SIGKILL.  
  This is a low‑priority hardening task that does not block retrain.

---

## Recently Completed

- **Fresh retrain (v3) — DONE** — `multilabel-v3-fresh-60ep` trained 60 epochs on `sentinel-retrain-v3`.
  Best raw F1-macro: **0.4715**. Tuned (per-class thresholds): **0.5069** ✅ — beats 0.4884 gate.
  Threshold JSON: `ml/checkpoints/multilabel-v3-fresh-60ep_best_thresholds.json`.
  See `docs/changes/2026-05-05-v3-training-complete.md`.

---

## In Progress

1. **v4 retrain** — v3 plateaued at raw F1 0.4715 with no improvement from epoch ~54.
   Recommended changes for v4:
   - `--loss-fn focal --focal-gamma 2.0` — penalise easy negatives, lift weak classes
   - `--lora-r 16 --lora-alpha 32` — double LoRA rank to break capacity ceiling
   - Weighted sampler for DenialOfService (137 support — 39× underrepresented)
   - Success gate: tuned F1-macro > **0.5069** on same `val_indices.npy`
   - Full command in `docs/changes/2026-05-05-v3-training-complete.md`

2. **Autoresearch setup** — now unblocked:
   - Implement `ml/scripts/auto_experiment.py` (thin CLI wrapper printing `SENTINEL_SCORE`).
   - Write `ml/autoresearch/program.md` (metric=tuned F1-macro, knobs: focal gamma, LoRA r, class weights).

3. **M6 Integration API** — build after the building blocks are solid:
   - Design auth/rate‑limit (see Security Design below).
   - Create `api/` directory and wire routes (`POST /v1/audit`, etc.).
   - Docker‑compose the full stack.

---

## Retrain Evaluation Protocol (v4)

Before launching v4, confirm all of the following:

1. `validate_graph_dataset.py` exits 0 — data unchanged from v3 run
2. Held-out split is fixed — use `ml/data/splits/val_indices.npy` with the **same seed**; do NOT regenerate
3. MLflow experiment `sentinel-retrain-v4` created
4. **Success gate:** tuned val F1-macro > **0.5069** on the same held-out split
5. **Per-class floor:** no class drops > 0.05 F1 from v3 tuned values (see `docs/changes/2026-05-05-v3-training-complete.md`)
6. **Rollback rule:** if tuned F1 < 0.5069 after completion, revert to v3 checkpoint and adjust hyperparameters

---

## M6 Integration API: Security Design (Complete Before Building Routes)

- **Auth:** `Authorization: Bearer <key>` header; validated via FastAPI dependency
- **API key storage:** env var `SENTINEL_API_KEYS` (comma‑separated); never hardcoded
- **Rate limiting:** `slowapi` or Redis token bucket; max 10 audits/min per key
- **Contract confidentiality:** audit job payloads must not be logged at INFO level — use DEBUG only
- **Input validation:** max contract size (e.g. 500KB); reject non‑UTF‑8 payloads before Slither

---

## ZKML Pipeline Resolution (S5.5)

M2 has been "source complete, not yet run" since 2026-04-29. Choose one of:

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

## Post-M6: Multi-Contract Parsing (Move 9)

**Why:** Users submitting a file with multiple contracts currently only get the first
non‑dependency contract analysed. The second is silently ignored.

### What already exists
`GraphExtractionConfig` in `ml/src/preprocessing/graph_extractor.py` has:
```python
multi_contract_policy: str = "first"   # also supports "by_name"
target_contract_name:  str | None = None
```

The correct change is adding an `"all"` value to the existing `multi_contract_policy` field.

### Implementation scope
**Primary:** `graph_extractor.py` — `_select_contract()` and `extract_contract_graph()`
   support `policy="all"` returning a list of `Data` objects.
**Propagation:**
- `preprocess.py` — new `process_source_all_contracts()` entry point.
- `predictor.py` — new `predict_source_multi()` with max‑aggregation across contracts.
- `api.py` — extend `PredictResponse` with optional `contracts_analysed` field.
- `cache.py` — decide caching strategy (single key or per‑contract sub‑keys).
**Documentation:** update `ml/README.md` Known Limitation #2.

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

| Module | Key test cases |
|--------|----------------|
| `cache.py` | Cache miss writes files; cache hit returns same object; TTL expiry evicts entry; cache key includes schema version |
| `drift_detector.py` | Warm-up mode suppresses alerts; KS fires on p < 0.05; buffer rolls after 200 requests |
| `promote_model.py` | CLI rejects unknown stage names; MLflow tags are written; dry-run mode does not register |
| Multi-contract pipeline (Move 9) | Two-contract file returns two graphs; max aggregation takes highest prob per class; cache stores and retrieves multi-result correctly; `contracts_analysed` list in response matches input file |

---

## Tool/Skill Reference

### ML / Training
- **Sliding-window NLP**: handling > 512 tokens without long-context models — T1-C (done)
- **Content-addressed caching**: Redis/disk pattern for ML feature store — T1-A (done)
- **LoRA rank tuning**: r=8 used in v3 → try r=16 for v4 (v3 plateaued at raw F1 0.4715, suggesting capacity ceiling)
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
