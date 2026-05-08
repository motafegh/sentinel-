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
  "sentinel-retrain-v4"   — NEXT retrain (to be created)

Per training run params logged:
  num_classes, epochs, batch_size, lr, weight_decay, threshold,
  grad_clip, warmup_pct, num_workers, device,
  architecture="cross_attention_lora",
  lora_r, lora_alpha, lora_dropout, lora_target_modules,
  use_edge_attr, gnn_edge_emb_dim, gnn_hidden_dim, gnn_heads, gnn_dropout,
  fusion_output_dim,
  focal_gamma, focal_alpha  ← always logged (even when loss_fn="bce")
  label_csv, resume_from, resume_model_only, remaining_epochs, start_epoch,
  pos_weight_{classname} × 10

Metrics per epoch:
  train_loss, val_f1_macro, val_f1_micro, val_hamming,
  val_f1_{classname} × 10

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
