# SENTINEL — Module 3: MLOps

Load for: MLflow experiments, DVC, Dagster RAG ingestion, drift detection.

---

## Tech Stack

| Tool            | Role                                                |
| --------------- | --------------------------------------------------- |
| MLflow ^2.17    | Experiment tracking, metric logging, model registry |
| DVC ^3.49       | Dataset and model versioning                        |
| Dagster 1.12.22 | Asset orchestration (RAG ingestion pipeline)        |

---

## MLflow

```
Tracking URI: sqlite:///mlruns.db (project root)

Experiments:
  "sentinel-training"     — binary model (historical)
  "sentinel-multilabel"   — Track 3 multi-label runs (baseline)
  "sentinel-retrain-v2"   — v2 paused run (batch-size mismatch)
  "sentinel-retrain-v3"   — v3 complete; F1-macro 0.5069 tuned ✅
  "sentinel-retrain-v4"   — v4 complete; best fallback multilabel-v4-finetune-lr1e4_best.pt ✅
  "sentinel-v5.2"         — ACTIVE; run v5.2-jk-20260515c-r3 in progress (epoch 21 F1=0.3130)

Per training run params logged:
  num_classes, epochs, batch_size, lr, weight_decay, threshold,
  grad_clip, warmup_pct, num_workers, device,
  architecture="v5_three_eye",
  lora_r, lora_alpha, lora_dropout, lora_target_modules,
  use_edge_attr, gnn_edge_emb_dim, gnn_hidden_dim, gnn_heads, gnn_dropout,
  fusion_output_dim,
  focal_gamma, focal_alpha,  ← always logged (even when loss_fn="bce")
  label_csv, resume_from, resume_model_only, remaining_epochs, start_epoch,
  pos_weight_{classname} × 10,
  eval_threshold,            ← threshold used during val-time early stopping (v5.2 fix: 0.35)
  gradient_accumulation_steps,
  gnn_use_jk,               ← bool; True for v5.2 JK-attention aggregation
  gnn_lr_multiplier,        ← GNN param group LR = base_lr × gnn_lr_multiplier (v5.2: 2.5)
  lora_lr_multiplier        ← LoRA param group LR = base_lr × lora_lr_multiplier (v5.2: 0.5)

Metrics per epoch:
  train_loss, val_f1_macro, val_f1_micro, val_hamming,
  val_f1_{classname} × 10,
  val_f1_gnn_eye,            ← auxiliary GNN-eye head F1-macro (λ=0.3 aux loss)
  val_f1_tf_eye,             ← auxiliary transformer-eye head F1-macro
  val_f1_fused_eye,          ← auxiliary fused-eye head F1-macro
  gnn_grad_share             ← fraction of total gradient norm attributable to GNN (per step)

Start UI: mlflow ui --port 5000
```

### Model Registry (T2-C)

```
promote_model.py --checkpoint ... --stage Staging|Production --note "..."
Tags written: val_f1_macro, architecture, git_commit, threshold_path
--dry-run flag: preview without writing to MLflow
```

---

## DVC

```
Tracked: ml/checkpoints/ (model .pt files), ml/data/ (graphs, tokens, splits)
Remote: configured per environment — ask for current remote path
Commands:
  dvc pull          — fetch latest data/checkpoints
  dvc push          — push new checkpoint after retrain
  dvc status        — check local vs remote diff
```

---

## Drift Detection (T2-B)

```
File: ml/src/inference/drift_detector.py

DriftDetector — KS-based feature drift monitoring
  rolling buffer: 200 requests
  warm-up suppression: alerts suppressed until N≥500 real requests
  baseline: drift_baseline.json (from warmup data — NOT training data)

⚠️  DO NOT compute drift baseline from ml/data/graphs/ (training data)
    BCCC-SCsVul-2024 is a 2024 historical snapshot.
    KS test will fire on virtually every modern 2026 contract → alerts useless.

Correct strategy:
  Phase 1 (warm-up): collect stats from first 500 real audit requests; suppress alerts.
  Phase 2 (active):  write drift_baseline.json from warm-up data; enable KS alerts.

compute_drift_baseline.py --source warmup|training
  --source training: available for offline testing with a prominent warning printed.

Env vars:
  SENTINEL_DRIFT_BASELINE        (default ml/data/drift_baseline.json)
  SENTINEL_DRIFT_CHECK_INTERVAL  (default 50 requests)

Prometheus counter: sentinel_drift_alerts_total{stat}  (exposed on /metrics)
```

---

## Dagster (RAG Ingestion)

```
Asset:    rag_index — full ingestion pipeline
          (DeFiHackLabs → chunk → embed → FAISS+BM25 index)
Schedule: daily_ingestion_schedule (cron: 0 2 * * *)
Home:     agents/.dagster (DAGSTER_HOME env var)

Files:
  agents/src/ingestion/scheduler_dagster.py   Dagster asset + daily schedule

Start UI:
  cd ~/projects/sentinel/agents
  poetry run dagster dev -f src/ingestion/scheduler_dagster.py
  → http://localhost:3000

Design note (ADR-027):
  Single asset encapsulates full pipeline with honest lineage.
  Old 3-asset chain had no real data flow and re-fetched the source 3×.
```

---

## Upgrade Proposals (v2)

### MLOPS-1: Automated Hyperparameter Search (Optuna)

**Problem:** All hyperparameters (lr, aux_loss_weight, gnn_lr_multiplier, lora_lr_multiplier,
eval_threshold, gnn_dropout) were chosen through analysis and intuition. There is no systematic
evidence that the current values are near-optimal, and interactions between GNN LR multiplier
and LoRA LR multiplier are non-trivial.

**Solution:** `ml/scripts/optuna_search.py` — Optuna TPE sampler, 50 trials, objective is
val F1-macro after 20 epochs evaluated on a 10% stratified subsample of the deduped dataset.
MedianPruner kills trials whose F1 trajectory at epoch 5 is below the median of completed
trials, saving ~60% of compute.

**Search space:**
```python
lr                  = trial.suggest_float("lr", 1e-5, 5e-4, log=True)
aux_loss_weight     = trial.suggest_float("aux_loss_weight", 0.1, 0.5)
gnn_lr_multiplier   = trial.suggest_float("gnn_lr_multiplier", 1.0, 5.0)
lora_lr_multiplier  = trial.suggest_float("lora_lr_multiplier", 0.1, 1.0)
eval_threshold      = trial.suggest_float("eval_threshold", 0.25, 0.45)
gnn_dropout         = trial.suggest_float("gnn_dropout", 0.1, 0.4)
```

**Integration:**
```
ml/scripts/optuna_search.py
  --n-trials 50
  --subsample-frac 0.10
  --base-config ml/configs/v5.2_base.yaml   ← non-searched params come from here
  --mlflow-experiment sentinel-optuna
  --output-config ml/configs/v5.2_optuna_best.yaml  ← auto-generated for full retrain

Pruner:  MedianPruner(n_startup_trials=5, n_warmup_steps=5)
Sampler: TPESampler(seed=42)
```

**Output:** Best trial config auto-generates a `train.py` launch command. All 50 trial runs
logged to MLflow experiment `"sentinel-optuna"` with full param/metric history.

---

### MLOPS-2: Automated Retraining Pipeline

**Problem:** Retraining is currently a manual decision. There is no path from drift detector
alerts to a new production checkpoint without human intervention at every step.

**Solution:** Dagster job `retrain_pipeline` triggered by drift gate asset; runs the full
validate → train → tune → gate → promote sequence with a single required human approval
before Production promotion.

**Pipeline stages:**
```
Dagster assets (agents/src/pipelines/retrain_pipeline.py):

drift_gate           Reads sentinel_drift_alerts_total from Prometheus.
                     Fires when > 3 alerts in a rolling 500-request window.
                     Writes trigger file: ml/data/retrain_trigger.json
                     Guard: skips if last retrain < 48h ago (prevents thrashing).

retrain_job          Runs: python ml/scripts/train.py --resume ml/checkpoints/current_best.pt
                     Uses current best checkpoint as starting point.
                     Logs to MLflow experiment "sentinel-v5.x-retrain".

evaluation_gate      Runs: python ml/scripts/tune_threshold.py + ml/scripts/manual_test.py
                     Fails asset if any per-class F1 < (v4 floor − 0.05).
                     Blocks pipeline until gate passes; does NOT auto-retry.

staging_promotion    Runs: python ml/scripts/promote_model.py --stage Staging
                     Sends Slack/webhook notification for human review.

production_promotion MANUAL TRIGGER ONLY via Dagster UI sensor confirmation.
                     Runs: python ml/scripts/promote_model.py --stage Production
                     Updates ml/checkpoints/current_best.pt symlink.
```

**Guard detail:** `drift_gate` checks `ml/data/last_retrain_timestamp.json` and raises
`SkipReason("Last retrain < 48h ago")` to prevent alert storms triggering repeated retrains.

---

### MLOPS-3: Feature Store for Graph Embeddings

**Problem:** Graph extraction via Slither costs 3–5s per contract. The full SentinelModel
forward pass (GNN + CodeBERT) costs ~200ms. For previously-seen contracts, both are wasted
on every inference request.

**Solution:** Content-addressed feature store keyed by contract MD5 hash, storing extracted
graph embeddings and token embeddings alongside their schema version for invalidation.

**Implementation:**
```
ml/src/feature_store/
  store.py             FeatureStore class
                         get(contract_hash, schema_version) → FeatureRecord | None
                         put(contract_hash, schema_version, graph_emb, token_emb)
                         invalidate_stale(current_schema_version)  ← call on startup
  backends/
    sqlite_backend.py  Structured metadata (hash, schema_version, inserted_at, hit_count)
                       in SQLite; tensors stored as memory-mapped .npy files in
                       ml/data/feature_store/{hash[:2]}/{hash}.npy
    duck_backend.py    DuckDB alternative for bulk analytics queries on the store

FeatureRecord:
  graph_embedding:  np.ndarray [128]   (CrossAttentionFusion input — GNN side)
  token_embedding:  np.ndarray [768]   (CodeBERT CLS token)
  schema_version:   str                must match FEATURE_SCHEMA_VERSION constant ("v3")
  inserted_at:      datetime
```

**Invalidation rules:**
- `schema_version` mismatch → re-extract (FEATURE_SCHEMA_VERSION bump invalidates entire store)
- TTL: entries older than 30 days evicted on startup via `invalidate_stale()`
- Manual purge: `python ml/scripts/purge_feature_store.py --older-than 30d`

**Expected latency improvement:** audit latency drops from ~5s (Slither + model) to ~50ms
(store lookup + classifier forward only) for previously-seen contracts.

---

### MLOPS-4: Canary Deployment with A/B Testing

**Problem:** Model promotions to Production are currently binary (all-or-nothing). A regression
introduced during a v5.x→v6.x promotion would affect all users immediately with no automated
rollback.

**Solution:** Weighted-random routing in `ml/src/inference/api.py` between a production
Predictor and a canary Predictor. 5% of traffic routes to the canary. Automated rollback
triggers on error rate or latency regression.

**API changes (`ml/src/inference/api.py`):**
```python
# Two Predictor instances loaded at startup
primary_predictor = Predictor.from_checkpoint(PRODUCTION_CHECKPOINT)
canary_predictor  = Predictor.from_checkpoint(CANARY_CHECKPOINT)   # None if no canary active
CANARY_WEIGHT = float(os.getenv("SENTINEL_CANARY_WEIGHT", "0.05"))  # 5% default

# Per-request routing
def route_request(contract_source: str) -> tuple[Predictor, str]:
    if canary_predictor and random.random() < CANARY_WEIGHT:
        return canary_predictor, "canary"
    return primary_predictor, "production"

# Response header
response.headers["X-Model-Version"] = model_version  # "production" or "canary"
```

**Canary gate (automated rollback conditions):**
```
Prometheus metrics tagged with model_version:
  sentinel_request_errors_total{model_version}
  sentinel_response_time_seconds{model_version, quantile="0.99"}
  sentinel_f1_live{model_version, class}   ← scored from human-feedback labels

Rollback triggers (checked every 1000 canary requests):
  canary error_rate    > 2%         → set CANARY_WEIGHT=0, alert ops
  canary p99 latency   > 10s        → set CANARY_WEIGHT=0, alert ops
  canary F1-macro drop > 3% vs prod → set CANARY_WEIGHT=0, alert ops

Full rollout: canary F1 ≥ prod F1 − 1% over 1000 requests → promote canary to production
```

**Env vars:**
```
SENTINEL_CANARY_CHECKPOINT   path to canary .pt file (empty string = no canary)
SENTINEL_CANARY_WEIGHT       float 0.0–1.0 (default 0.05)
```

---

### MLOPS-5: Dataset Versioning and Lineage

**Problem:** There is currently no way to reproduce a specific checkpoint from a given
training run. The dataset (multilabel_index_deduped.csv), graph extraction parameters,
and code commit are not formally linked to checkpoint files.

**Solution:** DVC pipeline from raw CSV to cached dataset, with MLflow tags capturing the
full provenance triple: dataset DVC hash + code git hash + config hash.

**DVC pipeline (`ml/dvc.yaml`):**
```yaml
stages:
  extract_graphs:
    cmd: python ml/scripts/extract_graphs.py --config ml/configs/extraction_v3.yaml
    deps: [ml/data/processed/multilabel_index_deduped.csv, ml/src/preprocessing/]
    outs: [ml/data/graphs/]
    params: [ml/configs/extraction_v3.yaml]

  extract_tokens:
    cmd: python ml/scripts/extract_tokens.py
    deps: [ml/data/processed/multilabel_index_deduped.csv]
    outs: [ml/data/tokens/]

  build_splits:
    cmd: python ml/scripts/build_splits.py --seed 42 --deduped
    deps: [ml/data/processed/multilabel_index_deduped.csv]
    outs: [ml/data/splits/deduped/]

  build_cache:
    cmd: python ml/scripts/create_cache.py
    deps: [ml/data/graphs/, ml/data/tokens/, ml/data/splits/deduped/]
    outs: [ml/data/cached_dataset_deduped.pkl]
```

**MLflow tags written by `train.py` at run start:**
```python
mlflow.set_tag("dvc_dataset_version",  subprocess.check_output(["dvc", "data", "status", "--md5"]).decode().strip())
mlflow.set_tag("git_commit",           subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip())
mlflow.set_tag("config_hash",          hashlib.sha256(open(config_path, "rb").read()).hexdigest()[:12])
mlflow.set_tag("python_env_hash",      subprocess.check_output(["pip", "freeze"]).decode().__hash__().__str__())
```

**Reproduction guarantee:** Any checkpoint in the MLflow registry can be re-trained from
scratch by checking out `git_commit`, running `dvc checkout` to the `dvc_dataset_version`,
and re-running `train.py` with the logged config. The `config_hash` detects config drift.
