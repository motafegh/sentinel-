# MLOps — State Audit & Redesign Plan (2026-06-14)

> **Purpose:** Working reference for the MLOps layer redesign. Written from
> source-code read, not docs. Do not re-read source files when starting this work —
> use this document as the entry point.
>
> **Context:** Run 12 is training (ep46, f1_tuned=0.6941 @ ep30). Run 13 will
> drop GasException (NUM_CLASSES 10→9), inject 658 BCCC ME contracts, strip
> Solidifi `bug_*` prefix. The MLOps layer must track and serve ALL of this.

---

## 1. Should MLOps Be a Separate Top-Level Folder?

**Answer: Yes. Partially.**

The current split is:

| Location | What it contains |
|---|---|
| `ml/src/inference/` | FastAPI server, predictor, preprocessor, cache, drift detector |
| `ml/scripts/promote_model.py` | MLflow registry CLI |
| `ml/scripts/compute_drift_baseline.py` | Drift baseline builder |
| `ml/scripts/run_overnight_experiments.py` | Sequential MLflow sweep launcher |
| `agents/src/ingestion/scheduler_dagster.py` | Dagster schedule for RAG ingestion |
| `ml/data/drift_baseline.json` | Placeholder — NOT real |

**Recommended structure going forward:**

```
sentinel/
  ml/                          ← model code STAYS here (tight coupling with GNN, trainer)
    src/
      inference/               ← inference server STAYS in ml/ (shared venv, shared model code)
    scripts/
      promote_model.py
      compute_drift_baseline.py
  mlops/                       ← NEW top-level module (operational concerns)
    docker/
      docker-compose.yml       ← inference server + monitoring stack
      Dockerfile.inference
    monitoring/
      prometheus.yml
      grafana/
    model_registry/
      promote.py               ← move from ml/scripts/promote_model.py
      validate_checkpoint.py   ← NEW: runs smoke suite before promoting
    experiments/
      run_overnight.py         ← move from ml/scripts/
      compare_runs.py          ← NEW: compare two MLflow run IDs
    deployment/
      deploy.sh
      rollback.sh
```

**Why the inference server stays in `ml/`:**
It imports `ml.src.models.*`, `ml.src.preprocessing.*` directly. Moving it to `mlops/` would require packaging `ml` as an installable — that's a larger refactor than the value justifies. The inference server is a serving wrapper around model code; it belongs alongside model code.

**Why everything else should move to `mlops/`:**
Experiment management, deployment config, Docker, monitoring, and model promotion are operational concerns that don't depend on model internals. They also have different lifecycle (change when infrastructure changes, not when model changes).

---

## 2. Is MLOps Fully Implemented?

**No. ~55% done.**

### What IS implemented and works

| Component | File | Status | Notes |
|---|---|---|---|
| FastAPI inference server | `ml/src/inference/api.py` | ✅ Implemented | Three-tier schema, Prometheus metrics, timeout handling |
| Predictor (checkpoint loader) | `ml/src/inference/predictor.py` | ✅ Implemented | Handles LoRA, `._orig_mod.` stripping, BF16, per-class thresholds |
| Preprocessor | `ml/src/inference/preprocess.py` | ✅ Implemented | Delegates to `graph_extractor.py` (healthy seam) |
| Inference cache | `ml/src/inference/cache.py` | ✅ Implemented | Content-hash keyed cache |
| Drift detector | `ml/src/inference/drift_detector.py` | ✅ Implemented | KS test, rolling buffer, warm-up phase |
| MLflow tracking | `mlruns.db` (project root) | ✅ Implemented | All runs logged including sentinel-v12 (Run 12) |
| Model promotion CLI | `ml/scripts/promote_model.py` | ✅ Implemented | Staging / Production stages, dry-run flag |
| Drift baseline builder | `ml/scripts/compute_drift_baseline.py` | ✅ Implemented | `--source warmup | training` |
| DVC checkpoint tracking | `ml/checkpoints.dvc` | ✅ Initialised | Remote TBD |
| Prometheus metrics | In `api.py` | ✅ Wired | model_loaded gauge, gpu_mem_bytes gauge |

### What is NOT implemented

| Missing piece | Priority | Notes |
|---|---|---|
| **Drift baseline is a placeholder** | HIGH | `ml/data/drift_baseline.json` is not real — computed from zero traffic. The `/health` endpoint says `drift_baseline_loaded: false`. No alerts will fire until baseline is set. |
| **Docker Compose / deployment config** | HIGH | No containerisation. Inference server is launched manually. |
| **Checkpoint auto-selection** | HIGH | `api.py` hardcodes `GCB-P1-Run4-no-asl-pw_best.pt` as default (see §3 below) |
| **Model promotion workflow discipline** | MEDIUM | `promote_model.py` exists but there's no documented process for when/how to promote Run 12→Staging→Production |
| **CI/CD for inference server** | MEDIUM | No tests that spin up the FastAPI server and hit `/predict` |
| **Drift baseline from real data** | LOW (blocked) | Cannot be computed until M6 ships traffic; use synthetic bridge |
| **`/debug/warmup_dump` endpoint** | LOW | Planned in M3 plan but never built |
| **`is_warming_up` property on drift_detector** | LOW | Planned, never built |

---

## 3. Critical Staleness Issues (must fix before Run 12 can be served)

### 3.1 Checkpoint hardcoded to Run 4

```python
# ml/src/inference/api.py:39
CHECKPOINT: str = os.getenv(
    "SENTINEL_CHECKPOINT",
    "ml/checkpoints/GCB-P1-Run4-no-asl-pw_best.pt",  # ← RUN 4, F1=0.3362
)
```

Run 12's checkpoint: `ml/checkpoints/GCB-P1-Run12-v3dospatched-20260613_best.pt`.
Fix: change the default, OR wire `SENTINEL_CHECKPOINT` env var into all startup scripts.

### 3.2 Schema comments reference old experiment names

`api.py` docstring says "CHECKPOINT: GCB-P1-Run4-no-asl-pw_best.pt (epoch 32, F1=0.3362)".
Must update with Run 12 info after Run 12 completes.

### 3.3 preprocess.py schema version comments are stale

```python
# preprocess.py comment:
#   graph.x [N, NODE_FEATURE_DIM] float32 (13 in v5; was 8 in v4)
```

Actual: `NODE_FEATURE_DIM=12`, `FEATURE_SCHEMA_VERSION="v9"`. The code is correct (it imports from `graph_schema.py`); the comment is wrong.

### 3.4 NUM_CLASSES hardcoded in api.py / predictor.py comments

Both files reference 10 classes throughout comments and docstrings. After Run 13, NUM_CLASSES=9 (GasException dropped). These are comments, not code — the code reads from the checkpoint config. But they'll be confusing.

### 3.5 predictor.py loads old checkpoint config field names

`predictor.py` reads `architecture` from checkpoint config and checks against `_ARCH_TO_FUSION_DIM`. This was built for pre-Run-12 checkpoints. Run 12's checkpoint structure should be verified — the trainer.py saves checkpoint config, and predictor.py must read the right keys.

Key checkpoint fields to verify (from trainer.py):
- `epoch`, `best_f1`, `model_state_dict`, `optimizer_state_dict`
- `config` dict: `num_classes`, `architecture`, `fusion_output_dim`, `class_names`, `drop_complexity_feature`

### 3.6 drift_detector.py warm-up count

Currently `warm_up_size=500` (hardcoded). After Run 12/13 the model may produce different feature distributions. The warm-up count should be configurable via env var.

---

## 4. The Preprocessing Seam — CURRENTLY HEALTHY

**This is the most important correctness point:**

`preprocess.py` calls `ml.src.preprocessing.graph_extractor.extract_contract_graph()` — the SAME function that data_module uses (via `sentinel_data.representation.graph_extractor`).

Verify: `data_module/sentinel_data/representation/graph_extractor.py` is a symlink or copy of `ml/src/preprocessing/graph_extractor.py`.

```bash
# Verify they are the same file
diff ml/src/preprocessing/graph_extractor.py \
     data_module/sentinel_data/representation/graph_extractor.py
```

If they diverge at any point, inference will receive different feature vectors than training received. This would be a silent catastrophe.

**One known gap:** `preprocess.py` uses `AutoTokenizer.from_pretrained("microsoft/graphcodebert-base")` while the data_module uses `windowed_tokenizer.py`'s `tokenize_windowed_contract()`. These must produce the same tokenized output. Verify that `preprocess.py`'s `_tokenize()` and `_tokenize_sliding_window()` match `windowed_tokenizer.py`'s logic exactly (padding, truncation, stride).

---

## 5. What MLOps Needs for Run 13 Readiness

These are the concrete steps, in order:

### Step 1: Wire checkpoint into a managed config (30 min)
- Create `ml/mlops_config.json` or env-file: `SENTINEL_CHECKPOINT`, `SENTINEL_NUM_CLASSES`, `SENTINEL_EXPERIMENT`
- Update `api.py` to read checkpoint path from config, not hardcoded default
- Write a script `ml/scripts/set_active_checkpoint.py <run_name>` that updates the config

### Step 2: Validate Run 12 checkpoint loads cleanly (1 hr)
- Run `uvicorn ml.src.inference.api:app` with `SENTINEL_CHECKPOINT=ml/checkpoints/GCB-P1-Run12-v3dospatched-20260613_best.pt`
- Hit `/health` and verify: `architecture`, `thresholds_loaded`, `model_epoch`, `model_f1_val` are correct
- Hit `/predict` with a known-vulnerable contract (Solidifi test file)
- Check all 10 class probabilities are in response (not 9 yet — Run 13 drops GasException)

### Step 3: Handle NUM_CLASSES=9 (Run 13 prerequisite, 2 hr)
- After Run 13 checkpoint is ready, verify predictor.py reads `num_classes` from checkpoint config (it does — `cfg.get("num_classes")`)
- Update `api.py`'s `PredictResponse` model: `probabilities: dict[str, float]` is already dynamic (not hardcoded class list) — this should work transparently
- Test that `confirmed`/`suspicious` lists don't break when GasException is absent

### Step 4: Drift baseline (1 hr, after Step 2)
- Run `ml/scripts/compute_drift_baseline.py --source training` against the v3 train split
- Save result to `ml/data/drift_baseline_run12.json`
- Update `SENTINEL_DRIFT_BASELINE` env var in startup config
- Accept that alerts will be noisy at first (training ≠ production distribution)

### Step 5: Create Docker Compose (2 hr, in new mlops/ folder)
- Service: `inference` — wraps `uvicorn ml.src.inference.api:app`
- Service: `prometheus` — scrapes `/metrics` at `/metrics`
- Service: `grafana` — connects to Prometheus (optional for now)
- env file: checkpoint path, drift baseline path, ports

---

## 6. MLflow Experiments Inventory

From `mlruns.db` (confirmed via training logs):

| Experiment | Status | Best run |
|---|---|---|
| `sentinel-training` | Historical | binary classifier |
| `sentinel-multilabel` | Historical | Run 11 ep1 F1=0.3293 |
| `sentinel-v12` | **ACTIVE** | Run 12, ep30 f1_tuned=0.6941 ★ |

**Run 12 MLflow run ID:** Not captured in memory — query `mlflow ui` to find it.

After Run 12 completes:
```bash
python ml/scripts/promote_model.py \
  --checkpoint ml/checkpoints/GCB-P1-Run12-v3dospatched-20260613_best.pt \
  --stage Staging \
  --note "Run 12 — f1_tuned=0.6941, ep30, v3 clean data, DoS patch"
```

---

## 7. Key File Index (for this module)

| File | Lines | Role | Freshness |
|---|---|---|---|
| `ml/src/inference/api.py` | 402 | FastAPI server | Stale: Run 4 default, schema comments outdated |
| `ml/src/inference/predictor.py` | 760 | Checkpoint loader + inference | OK: handles LoRA/BF16 |
| `ml/src/inference/preprocess.py` | 625 | Solidity → graph+tokens | OK: delegates to graph_extractor |
| `ml/src/inference/cache.py` | 161 | Content-hash cache | OK |
| `ml/src/inference/drift_detector.py` | 193 | KS drift detection | OK: baseline is placeholder |
| `ml/scripts/promote_model.py` | ~150 | MLflow promotion CLI | OK |
| `ml/scripts/compute_drift_baseline.py` | ~100 | Drift baseline builder | OK |
| `ml/data/drift_baseline.json` | — | Placeholder | NOT REAL — replace after Step 4 |

---

## 8. Priority Order for Implementation

1. **Wire checkpoint config** (unblocks serving Run 12) — 30 min
2. **Validate Run 12 loads cleanly** — 1 hr
3. **Docker Compose** (unblocks any deployment) — 2 hr
4. **Drift baseline from training data** (synthetic bridge) — 1 hr
5. **Run 13: NUM_CLASSES=9 validation** — after Run 13 trains
6. **Create top-level `mlops/` folder** and move operational scripts — 2 hr (housekeeping)
7. **CI for inference server** (pytest hitting `/predict`) — 1 hr
