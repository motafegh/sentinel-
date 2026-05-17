# SENTINEL — Roadmap

Last updated: 2026-05-17 (v6.0 training running — epoch 1/100)

This file tracks upcoming work in priority order. Completed items move to
`docs/changes/` as dated changelogs. See `docs/STATUS.md` for current module state.

---

## Current Priority: v6.0 Training and Evaluation

v6.0 training is running (PID 450936). All pipeline steps complete.

| Item | ETA | Notes |
|------|-----|-------|
| v6.0 training (100ep, patience=30) | ~57h | ~34 min/epoch; early stopping expected ~ep 40–60 |
| Threshold tuning (val set) | After training | `ml/scripts/tune_thresholds.py` |
| Behavioral evaluation | After tuning | `ml/scripts/manual_test.py` — gate: ≥80% detection, ≥80% specificity |
| If gate passes → promote | Same day | `promote_model.py` → MLflow Production |
| If gate fails → Phase 4 augmentation | See below | DoS/Timestamp synthetic data |

---

## Recently Completed (v5→v6 journey, 2026-05-09 to 2026-05-17)

| Move | Status | Key Output |
|------|--------|-----------|
| v5.0 full retrain (60ep) | ✅ Done | F1=0.5828 — val gate cleared, behavioral FAILED (36%/33%) |
| v5.1 root cause analysis | ✅ Done | 3 bugs found: interface filter, function pooling, leaky splits |
| Dataset deduplication | ✅ Done | 68,523 → 44,470 rows; 34.9% leakage eliminated |
| v5.2 retrain (60ep) | ✅ Done | F1=0.3422 — behavioral FAILED (36%/33%) |
| v5.3 killed epoch 47 | ✅ Done | F1=0.2559 — schema bugs were root cause |
| Feature schema v4 (4 bugs fixed) | ✅ Done | return_ignored, uses_block_globals, ext_call_count, loc normalization |
| Deep audit (26 tasks, 9 bugs) | ✅ Done | BUG-6 wrong contract selection (47.4%!), BUG-1/2/3/7/8/9 |
| Feature schema v5 (5 more bugs fixed) | ✅ Done | loc/complexity/contract-selection/return_ignored/visibility |
| Graph re-extraction v7 | ✅ Done | 41,521 fresh + 2,702 in-place patched |
| Windowed tokenization | ✅ Done | [4,512] per contract; max_windows=4 |
| Timestamp label relabeling | ✅ Done | 972 unverified labels removed (1,933→961) |
| v6 GNN arch (256-dim, 6-layer, conv3b/4b) | ✅ Done | WindowAttentionPooler, classifier 384→192→10 |
| v6 training config (ASL, 100ep, patience=30) | ✅ Done | AsymmetricLoss(γ⁻=4, γ⁺=1), lora_lr=0.3× |
| v6.0 training launch | ✅ Done (running) | PID 450936, ~34 min/epoch |

See `docs/changes/2026-05-16-all-training-runs-summary.md` for complete v4→v5.3 history.

---

## If v6.0 Behavioral Gate Passes

1. Promote checkpoint → MLflow Production
2. Run ZKML pipeline (M2 — source complete, never run):
   - Export ONNX from v6 checkpoint (proxy MLP Linear(128→64→32→10), opset=11)
   - `ezkl gen-settings`, `ezkl calibrate`, `ezkl prove`
   - Deploy Solidity verifier + AuditRegistry on-chain
3. M6 Integration API (design auth/rate-limit before writing routes):
   - Bearer token auth; env var `SENTINEL_API_KEYS`; rate limit 10 audits/min
   - `POST /v1/audit` endpoint wiring ML → RAG → synthesis
4. M5 Contracts (never built):
   - `forge install + build + test` — contracts/lib/ is empty

---

## If v6.0 Behavioral Gate Fails — Phase 4 Data Augmentation

The remaining root causes not fixed in v6 schema:

| Problem | Fix |
|---------|-----|
| DoS: 7 pure-label contracts (98.1% Reentrancy co-occurrence) | ~500 clean ETH-transfer-loop-only contracts → break co-occurrence |
| Timestamp: positive class halved (961 after cleanup) | ~500 clean block.timestamp-only contracts to compensate |
| BUG-7: EMITS edges never created | New extraction needed with IR-level event scan |
| BUG-8: INHERITS edges never created | New extraction needed with parent contract node_map |
| BUG-3: visibility=2 for private | Normalize to binary in next schema bump |

Then: rebuild CSV, re-extract, retokenize, rebuild cache, retrain.

---

## Deferred Backlog (not blocking v6)

| Item | Priority | Notes |
|------|----------|-------|
| ZKML pipeline execution | P1 | Blocked on checkpoint; source code complete since 2026-04-29 |
| M6 Integration API | P2 | Design auth before writing routes; `api/` dir doesn't exist |
| M5 Contracts | P2 | forge never run; contracts/lib/ empty |
| Fix #6 downstream update | P2 | `threshold` → `thresholds` breaking rename; consumers not updated |
| Multi-contract parsing (Move 9) | P3 | `multi_contract_policy="all"` not implemented; single contract only |
| Preprocess temp file on SIGKILL (audit #9) | P3 | Low priority hardening |
| S6 Observability | P3 | Prometheus + OpenTelemetry + Loki |
| S7 CI/CD | P3 | GitHub Actions, Semgrep, Bandit, Grype |
| S8 Advanced contracts | P4 | Echidna fuzzing, Halmos symbolic proofs |
| S9 Advanced ML | P4 | Online drift, LLM explanations |
| S10 Advanced ZK | P4 | Recursive SNARKs, proof aggregation |

---

## M6 Integration API: Security Design (Complete Before Building Routes)

- **Auth:** `Authorization: Bearer <key>` header; validated via FastAPI dependency
- **API key storage:** env var `SENTINEL_API_KEYS` (comma-separated); never hardcoded
- **Rate limiting:** `slowapi` or Redis token bucket; max 10 audits/min per key
- **Contract confidentiality:** audit job payloads must not be logged at INFO level — use DEBUG only
- **Input validation:** max contract size (e.g. 500KB); reject non-UTF-8 payloads before Slither

---

## Post-M6: Multi-Contract Parsing (Move 9)

`GraphExtractionConfig.multi_contract_policy` exists with `"first"` / `"by_name"` / `"most_derived"`.
Adding `"all"` would analyse every contract in a file. Requires:

- `graph_extractor.py` — `_select_contract()` returns list when `policy="all"`
- `preprocess.py` — `process_source_all_contracts()` entry point
- `predictor.py` — `predict_source_multi()` with max-aggregation across contracts
- `api.py` — optional `contracts_analysed` field in `PredictResponse`
- `cache.py` — per-contract sub-key caching strategy

---

## Tool/Skill Reference

### ML / Training
- **Windowed tokenization**: [4, 512] tensors per contract; CodeBERT processes [B*4, 512] per batch; WindowAttentionPooler aggregates window CLS tokens
- **AsymmetricLoss**: γ⁻=4 down-weights easy negatives (p≈0, y=0); critical for DoS (215 train samples)
- **LoRA rank tuning**: r=16 on Q+V of all 12 CodeBERT layers; lora_lr_mult=0.3 relative to base
- **Content-addressed caching**: md5_stem = hash(relative_path) — uniquely identifies contract across BCCC multi-folder layout
- **KS drift detection**: `scipy.stats.ks_2samp` — production ML monitoring
- **MLflow Model Registry**: staged rollout (None → Staging → Production)

### Agents / RAG
- **Cross-encoder reranking**: bi-encoder recall → cross-encoder precision
- **Solodit knowledge source**: ~50K professional audit findings
- **LangGraph parallel nodes**: `Send` API for concurrent rag_research ∥ static_analysis

### Infrastructure
- **Docker multi-stage builds**: builder → runtime (no build tools in final image)
- **Celery + Redis**: async task queue for audit and proof jobs
- **OTLP / Jaeger**: W3C traceparent propagation

### Smart Contract Analysis
- **Slither IR**: all feature extraction goes through SlithIR — HighLevelCall, LowLevelCall, Transfer, Send, SolidityVariableComposed, etc.
- **solc-select**: versions 0.4.0–0.8.31 all installed; use inside venv
- **Echidna / Halmos**: property fuzzing + symbolic proofs (not yet integrated)

### ZK Infrastructure
- **EZKL**: ZK circuit from ONNX model — proving_key, verification_key, Solidity verifier; scale=8192 little-endian
- **Groth16**: Solidity verifier compiled from circuit; deployed on AuditRegistry
