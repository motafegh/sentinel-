# ml/src/inference — SENTINEL Inference Pipeline

FastAPI inference server with prediction, hotspot extraction, drift detection, and caching.

---

## Files

| File | Lines | Purpose |
|------|-------|---------|
| `api.py` | 432 | FastAPI app — `/predict`, `/hotspots`, `/health`, `/metrics` |
| `predictor.py` | 804 | `Predictor` — checkpoint loading, warmup, inference, hotspot extraction |
| `drift_detector.py` | 253 | `DriftDetector` — KS-based feature distribution drift detection |
| `preprocess.py` | 625 | `ContractPreprocessor` — Slither graph extraction + CodeBERT tokenization |
| `cache.py` | 161 | `InferenceCache` — content-addressed disk cache with TTL |
| `__init__.py` | 0 | Empty |

---

## api.py

### FastAPI Application

**Title:** "SENTINEL Vulnerability API" | **Version:** 3.0.0

**Lifespan:** Loads `Predictor` and `DriftDetector` once at startup.

**Configuration:** Reads `ml/mlops_config.json` (env vars override):
- `SENTINEL_CHECKPOINT` — checkpoint path
- `SENTINEL_DRIFT_BASELINE` — drift baseline JSON
- `SENTINEL_DRIFT_CHECK_INTERVAL` — KS check frequency (default 50)
- `SENTINEL_PREDICT_TIMEOUT` — inference timeout (default 60s)
- `SENTINEL_CONFIG` — config file path

**Endpoints:**

#### POST /predict

Request: `{"source_code": "pragma solidity ..."}`
Response: `PredictResponse` with:
- `label`: "safe" | "suspicious" | "confirmed_vulnerable"
- `probabilities`: full 10-class probability dict
- `confirmed`: list of `{vulnerability_class, probability, tier}`
- `suspicious`: list of `{vulnerability_class, probability, tier}`
- `vulnerabilities`: legacy alias for confirmed
- `tier_thresholds`: `{confirmed: [per-class], suspicious: 0.25, noteworthy: 0.10}`
- `thresholds`: per-class tuned thresholds
- `truncated`, `windows_used`, `num_nodes`, `num_edges`

Validation: rejects source without `pragma` or `contract`. Rejects >1MB.

#### POST /hotspots

Same request/response as /predict plus:
- `hotspots`: top-20 functions by GNN embedding norm (fn_name, node_id, score, lines, node_type)
- `hotspot_stats`: total_function_nodes, num_nodes, attention_source

#### GET /health

Returns model metadata: checkpoint path, architecture, thresholds_loaded, model_epoch, model_f1_val.

#### GET /metrics

Prometheus metrics (auto-instrumented via `prometheus-fastapi-instrumentator`).

Custom gauges: `sentinel_model_loaded`, `sentinel_gpu_memory_bytes`.

---

## predictor.py

### Predictor

Loads a trained SentinelModel checkpoint and scores Solidity contracts.

**Constructor:**
```python
Predictor(checkpoint, threshold=0.50, device=None,
          tier_confirmed_threshold=None, tier_suspicious_threshold=None)
```

**Key behaviors:**
- Reads architecture from checkpoint config — maps to `fusion_output_dim` via `_ARCH_TO_FUSION_DIM`
- Loads per-class thresholds from `{checkpoint_stem}_thresholds.json`
- Strips `._orig_mod.` infix from torch.compile state dicts
- Resizes edge_embedding if checkpoint used fewer edge types
- Warmup forward pass: 3-node graph with CONTRACT+FUNCTION+STATE_VAR, [1,4,512] token batch
- Sets `_current_epoch = 9999` so prefix is always active

**Architecture allowlist:**
```python
_ARCH_TO_FUSION_DIM = {
    "four_eye_v8": 128, "three_eye_v8": 128, "three_eye_v7": 128,
    "three_eye_v5": 128, "cross_attention_lora": 128,
    "legacy": 64, "legacy_binary": 64,
}
```

**Public methods:**
- `predict(sol_path)` — score a .sol file
- `predict_source(source_code)` — score raw Solidity string (training-aligned forward path)
- `predict_with_hotspots(source_code)` — full inference + GNN attention hotspots

**Forward path:** Always uses batched multi-window format `[1, 4, 512]`:
- Short contracts: 1 real window + 3 zero-padded
- Long contracts: up to 4 sliding windows
- Identical shape to training — WindowAttentionPooler and CrossAttentionFusion see the same context

**Three-tier classification:**
- CONFIRMED: `prob >= per_class_threshold` (tuned per class, default 0.55)
- SUSPICIOUS: `prob >= 0.25`
- NOTEWORTHY: `prob < 0.25` (probabilities dict only)

---

## drift_detector.py

### DriftDetector

Rolling KS-based drift detector for inference request monitoring.

**Stats monitored:** `num_nodes`, `num_edges`, `confirmed_count`, `suspicious_count`

**Configuration:**
- `KS_ALPHA = 0.05` — p-value threshold for alerts
- `MIN_SAMPLES_FOR_KS = 30` — minimum buffer entries before KS runs
- `buffer_size = 200` — rolling window size

**Modes:**
- **Warm-up mode:** Collects stats, suppresses alerts until `n_warmup` requests
- **Active mode:** Runs KS tests against baseline, fires Prometheus counter on drift

**Baseline:** `ml/data/drift_baseline_run12.json` (real synthetic warmup data, 4 stats x 500 samples)

**Public methods:**
- `update_stats(stats)` — record per-request feature statistics
- `check()` — run KS tests, return {stat: p_value}
- `dump_warmup_stats()` / `dump_warmup_to_jsonl()` — export buffer for baseline building

---

## preprocess.py

### ContractPreprocessor

Converts Solidity source into (graph, tokens) for SentinelModel.

**Constructor:** Loads CodeBERT tokenizer, purges orphaned temp files.

**Public methods:**
- `process(sol_path)` — contract file on disk
- `process_source(source_code)` — raw string (HTTP API)
- `process_source_windowed(source_code)` — sliding-window for long contracts

**Key behaviors:**
- Writes temp file for Slither (solc requires real path)
- solc version detection from pragma
- Content-addressed caching via InferenceCache
- Exception translation: SolcCompilationError -> ValueError (HTTP 400), SlitherParseError -> RuntimeError (HTTP 500)

**Tokenization:**
- Single window: `[1, 512]` with truncation detection
- Sliding window: overlapping 512-token windows, stride=256, max 4 windows
- Matches offline training pipeline (retokenize_windowed.py) exactly

**Constants:**
- `TOKENIZER_NAME = "microsoft/codebert-base"` (shared vocab with graphcodebert-base)
- `MAX_TOKEN_LENGTH = 512`
- `MAX_SOURCE_BYTES = 1 * 1024 * 1024` (1 MB)

---

## cache.py

### InferenceCache

Content-addressed disk cache for (graph, tokens) pairs.

**Key format:** `{content_md5}_{FEATURE_SCHEMA_VERSION}`

**Storage:** `{cache_dir}/{key}_graph.pt` + `{key}_tokens.pt`

**Features:**
- TTL-based expiry (default 86400s = 24h)
- Atomic writes via tmp file + rename
- Schema validation on load (NODE_FEATURE_DIM check)
- Thread-safe reads, last-writer-wins for concurrent writes

**Default dir:** `~/.cache/sentinel/preprocess/`
