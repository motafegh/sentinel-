# SENTINEL — Full Audit + Next ML Improvements (2026-05-01)

## Current State Audit

### What Is Complete (source-code level)

| Module | Status | Notes |
|--------|--------|-------|
| M1 ML Core — models | ✅ Complete | sentinel_model, gnn_encoder, transformer_encoder, fusion_layer |
| M1 ML Core — inference | ✅ Complete & hardened | api.py, predictor.py, preprocess.py |
| M1 ML Core — training | ✅ Complete | AMP/TF32, FocalLoss opt-in, peft pinned ≥0.13.0,<0.16.0 |
| M1 ML Core — tests | ✅ 6 test files | test_api (pre-existing 210 lines), test_model, test_preprocessing, test_dataset, test_trainer |
| M1 ML Scripts | ✅ Complete | train.py, tune_threshold.py, analyse_truncation.py, build_multilabel_index.py, create_splits.py |
| M2 ZKML | ✅ Source complete | Z1/Z2/Z3 bugs fixed; pipeline not yet run (GPU needed) |
| M3 MLOps | ✅ MLflow + DVC + Dagster | Model registry promotion script missing |
| M4 Agents/RAG | ✅ Core complete | RAG score fixed, static_analysis wired, LLM synthesizer not yet built |
| M5 Contracts | ✅ Source complete | Foundry tests written, forge not yet run |
| M6 Integration API | ❌ Not built | api/ directory does not exist |

### Discovered in This Audit (not previously tracked)
- `ml/utils/hash_utils.py` — content-addressed MD5 hashing **already exists**, not yet wired to inference cache
- `agents/tests/test_api.py` (210 lines) — pre-existing test for /predict + /health (not one we created)
- `ml/src/models/fusion/`, `gnn/`, `transformer/` — sub-package `__init__` stubs exist
- `edge_attr` is computed in `preprocess.py` but **never consumed** by GNNEncoder — dead code

### What Requires Execution (not source work)
```
forge install + forge build + forge test      (Foundry not installed in env)
python zkml/.../train_proxy.py                (GPU access needed)
python ml/scripts/analyse_truncation.py      (training data access needed)
```

---

## ML Module Deep Audit

### Architecture: ✅ Sound

GNN + CodeBERT + CrossAttentionFusion pipeline is correct end-to-end. All 8 original review
fixes applied. Defensive RuntimeErrors at all critical shape boundaries. Checkpoint metadata
validation at startup. Warmup forward pass catches CUDA/shape issues before first request.

### Gaps Found

| Gap | Location | Risk |
|-----|----------|------|
| No inference cache | preprocess.py | ~5s Slither per repeated contract (hash_utils.py exists but unused) |
| No node-count guard | preprocess.py | 0-node graph → garbage logits, no warning |
| graph.x dim not validated at runtime | preprocess.py | Wrong-dim graph silently crashes GNNEncoder |
| 512-token silent truncation | preprocess.py + predictor.py | Missed vulns in long contracts; `truncated=True` but no mitigation |
| edge_attr computed but never read | preprocess.py | Dead code; misleading; wastes memory |
| No Prometheus metrics | api.py | Invisible in production: no latency/error/GPU tracking |
| No drift detection | missing file | Silent model degradation when production contracts differ from training |
| No MLflow model registry | missing script | Unsafe model promotion (manual file copy, no audit trail) |
| RAM cache no version key | dual_path_dataset.py | Stale cache on schema change, silent wrong labels |
| LoRA config hardcoded | transformer_encoder.py | Can't tune r/lora_alpha without code change |

---

## Recommended ML Improvements — Prioritized With Rationale

### Tier 1 — High Value, Source-Code Only, Low Risk

#### T1-A: Content-Addressed Inference Cache
**Why:** Slither + CodeBERT tokenization takes ~3-5s per contract. The feedback loop, CI pipelines, and repeated audits of the same contract all pay this cost repeatedly. `hash_utils.py` already has `get_contract_hash_from_content()` — the key infrastructure exists unused.

**What to build:**
- New `ml/src/inference/cache.py`: `InferenceCache(cache_dir, ttl_seconds=86400)` with `get(hash) -> (graph, tokens) | None` and `put(hash, graph, tokens)`. Files stored as `{key}_graph.pt` / `{key}_tokens.pt`. Key = `md5(source_content)_v1` (version suffix for schema invalidation).
- `preprocess.py` `process_source()`: cache lookup at top, write on miss. Transparent to caller.

**Files:** `ml/src/inference/cache.py` (new), `ml/src/inference/preprocess.py`

---

#### T1-B: Runtime Feature Dimension + Empty-Graph Validation
**Why:** If `graph.x.shape[1] != 8`, the GATConv weight matrix is mismatched — this crashes deep in the GNN, not at the validation boundary where the error is actionable. If Slither returns 0 nodes (common on complex multi-contract files), cross-attention has nothing to attend over and produces garbage logits with no warning.

**What to add to `preprocess.py._extract_graph()`:**
```python
if graph.x.shape[1] != 8:
    raise RuntimeError(f"Node feature dim mismatch: expected 8, got {graph.x.shape[1]}")
if graph.num_nodes == 0:
    logger.warning("0 AST nodes extracted — Slither may have failed to parse contract")
```

**Files:** `ml/src/inference/preprocess.py`

---

#### T1-C: Sliding-Window Tokenization (No Retraining Required)
**Why:** Contracts longer than ~400 tokens have their tail silently truncated. Functions defined late in the file — complex logic, withdrawal patterns — are invisible to CodeBERT. The GNN still sees the full AST regardless. At inference, we can split the source into overlapping 512-token windows, run each through TransformerEncoder, and aggregate via max-probability per class. No retraining needed; the model is already correct on each window.

**What to build:**
- `preprocess.py`: `_tokenize_sliding_window(source, stride=256, max_windows=8)` → `list[dict]`. Returns `[single_dict]` for short contracts (no overhead).
- `predictor.py`: `_aggregate_window_predictions(probs_list)` → `torch.stack(probs_list).max(dim=0).values`. Add `windows_used: int` to response dict.
- Keep `process_source()` as single-window for backward compat; add `process_source_windowed()` that returns the list.

**Files:** `ml/src/inference/preprocess.py`, `ml/src/inference/predictor.py`

---

#### T1-D: Remove / Annotate Dead `edge_attr` Code
**Why:** `preprocess.py` builds `edge_attr` (edge type IDs, shape [E, 1]) but GNNEncoder only uses `edge_index`. Dead code at inference is misleading and wastes memory for large contracts.

**What:** Add comment explaining it's reserved for future GATv2/RGAT compatibility, or delete outright.

**Files:** `ml/src/inference/preprocess.py`

---

### Tier 2 — Production Observability + ML Safety

#### T2-A: Prometheus Metrics in ML API
**Why:** M6 will route real traffic to this service. Without metrics there is no way to detect: GPU memory pressure, latency spikes, rising error rates. `prometheus-fastapi-instrumentator` adds full auto-instrumentation in 5 lines.

**What:**
- `ml/pyproject.toml`: add `prometheus-fastapi-instrumentator>=0.9`
- `api.py`: `Instrumentator().instrument(app).expose(app)` after app creation. Auto-adds `/metrics` endpoint with request count, latency histograms, status codes.
- Custom gauges: `Gauge("sentinel_gpu_memory_bytes", ...)` polled per request, `Gauge("sentinel_model_loaded", ...)` set in lifespan.

**Files:** `ml/src/inference/api.py`, `ml/pyproject.toml`

---

#### T2-B: Online Drift Detection
**Why:** The model trained on BCCC-SCsVul-2024. Production contracts from 2026+ may be systematically longer, have more nodes, or use patterns underrepresented in training. The model degrades silently — predictions look valid but are wrong. KS test (Kolmogorov-Smirnov) tests whether two empirical distributions come from the same underlying distribution, with no assumptions about shape.

**What to build:**
- New `ml/src/inference/drift_detector.py`: `DriftDetector` class.
  - Loads baseline stats (node_count mean/std, token_length mean/std, 8 feature-dimension means) from `ml/checkpoints/drift_baseline.json`.
  - `update(graph, tokens)`: appends sample to rolling buffer (last 200 requests).
  - `check() -> dict[str, float]`: runs `scipy.stats.ks_2samp` for each tracked stat vs baseline. Returns p-values.
  - When p-value < 0.05: log warning + increment Prometheus counter `sentinel_drift_alerts_total{feature="node_count"}`.
- New `ml/scripts/compute_drift_baseline.py`: walks training graphs directory, computes stats → `drift_baseline.json`.
- `api.py`: instantiate in lifespan; call `detector.update()` per request; `detector.check()` every 50 requests.

**Files:** `ml/src/inference/drift_detector.py` (new), `ml/scripts/compute_drift_baseline.py` (new), `ml/src/inference/api.py`

---

#### T2-C: MLflow Model Registry Promotion Script
**Why:** Current promotion = manually copy a .pt file and restart the server. No audit trail, no metadata, no staged rollout, no rollback record. MLflow's Model Registry is already running (`sqlite:///mlruns.db`) — the infrastructure exists.

**What to build:**
- New `ml/scripts/promote_model.py`: CLI with `--checkpoint`, `--stage` (Staging|Production), `--note`.
  - Reads checkpoint config, loads val_f1_macro, per-class F1s, threshold JSON.
  - Tags with: val_f1_macro, architecture, git_commit, threshold_path.
  - Transitions stage: `None → Staging → Production` gated on `--promote` flag.
- Optional hook in `predictor.py`: if `SENTINEL_USE_REGISTRY=1`, poll MLflow for "Production" checkpoint at startup instead of using file path.

**Files:** `ml/scripts/promote_model.py` (new), `ml/src/inference/predictor.py` (optional env hook)

---

### Tier 3 — Agents/RAG (Adjacent to ML)

#### T3-A: LLM Synthesizer Upgrade
**Why:** Current synthesizer picks from 3 static strings. It produces no Solidity-specific guidance, no fix recommendations, no exploit scenario analysis. `get_strong_llm()` in `client.py` already wraps LM Studio — the call site is wired and ready.

**What:** Modify `nodes.py` `synthesizer()`:
- If LLM available: structured prompt → markdown report with severity, exploit path, fix. Parse into `final_report["narrative"]`.
- If LLM unavailable: fall back to rule-based output unchanged.
- Add `narrative: str` optional field to `AuditState`.

**Files:** `agents/src/orchestration/nodes.py`, `agents/src/orchestration/state.py`

---

#### T3-B: Cross-Encoder Re-Ranking in RAG
**Why:** RRF gives good recall but imprecise ranking. A cross-encoder reads query + chunk bidirectionally — much more accurate relevance scoring. Off by default (rerank=False) to preserve latency.

**What:** Add `rerank: bool = False` to `retriever.search()`. When True: after RRF top-20, call `CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2").predict([(query, c.content) for c in top20])`, re-sort.

**Files:** `agents/src/rag/retriever.py`, `agents/pyproject.toml`

---

## Implementation Order

```
T1-B  Feature dim + empty graph validation    30 min   Zero risk, pure defensive
T1-D  edge_attr comment/cleanup               15 min   Zero risk
T1-A  Inference cache                          2h      New cache.py + preprocess wiring
T1-C  Sliding-window tokenization              3h      New windowing + aggregation in predictor
T2-A  Prometheus metrics                       1h      Library wiring + 2 custom gauges
T2-C  MLflow registry script                   2h      New standalone script
T2-B  Drift detection                          3h      New module + baseline script
T3-A  LLM synthesizer                          2h      nodes.py modification with fallback
T3-B  Cross-encoder reranking                  1h      retriever extension
```

## Files to Create / Modify

| File | Action | Tier |
|------|--------|------|
| `ml/src/inference/cache.py` | Create | T1-A |
| `ml/src/inference/preprocess.py` | Modify (cache, dim check, sliding-window, edge_attr) | T1-A/B/C/D |
| `ml/src/inference/predictor.py` | Modify (window aggregation, windows_used field) | T1-C |
| `ml/src/inference/api.py` | Modify (Prometheus, drift check call) | T2-A/B |
| `ml/src/inference/drift_detector.py` | Create | T2-B |
| `ml/scripts/compute_drift_baseline.py` | Create | T2-B |
| `ml/scripts/promote_model.py` | Create | T2-C |
| `ml/pyproject.toml` | Modify (prometheus-fastapi-instrumentator) | T2-A |
| `agents/src/orchestration/nodes.py` | Modify (LLM synthesizer with fallback) | T3-A |
| `agents/src/orchestration/state.py` | Modify (narrative field) | T3-A |
| `agents/src/rag/retriever.py` | Modify (rerank param) | T3-B |
| `agents/pyproject.toml` | Modify (sentence-transformers) | T3-B |

## Verification

- **T1-A**: Run `process_source()` twice on same contract — second call < 0.05s; cache dir contains `*_graph.pt` + `*_tokens.pt` files.
- **T1-B**: Pass graph with 9 features → RuntimeError with clear dim message. Pass Slither-empty contract → WARNING log, no crash.
- **T1-C**: 2000-token contract → `process_source_windowed()` returns ≥2 dicts; predictor returns `windows_used >= 2` in response.
- **T2-A**: `GET /metrics` returns Prometheus text format with `http_request_duration_seconds_bucket` and `sentinel_gpu_memory_bytes`.
- **T2-B**: `DriftDetector.check()` returns dict of floats in [0,1]; `compute_drift_baseline.py` writes valid JSON with expected keys.
- **T2-C**: `python ml/scripts/promote_model.py --checkpoint ... --stage Staging` → model visible in `mlflow ui` under "sentinel-audit" registry.
- **T3-A**: Full graph run with LM Studio → `final_report["narrative"]` non-empty; without LM Studio → rule-based report, no crash.
- **T3-B**: `retriever.search("reentrancy", k=5, rerank=True)` → 5 chunks with scores from cross-encoder.
