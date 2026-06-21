# ml/deploy — SENTINEL Inference Deployment

Docker-based deployment stack for the SENTINEL inference API.

---

## Files

| File | Purpose |
|------|---------|
| `Dockerfile.inference` | Multi-layer Docker image for inference server |
| `docker-compose.yml` | Inference + Prometheus stack |
| `prometheus.yml` | Prometheus scrape config (15s interval) |
| `.env.example` | Environment variable template (8 vars) |
| `README.md` | Full deployment guide + troubleshooting |

---

## Stack

- **Inference server:** FastAPI on Python 3.12.1, GPU-ready
- **Monitoring:** Prometheus on internal bridge network
- **Healthcheck:** Built-in Docker healthcheck via `/health`

## Quick Start

```bash
cd ml/deploy
cp .env.example .env    # edit with your values
docker compose up -d
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SENTINEL_CHECKPOINT` | Run 12 FINAL | Path to model checkpoint |
| `SENTINEL_CONFIG` | `ml/mlops_config.json` | MLOps config file |
| `SENTINEL_DRIFT_BASELINE` | `ml/data/drift_baseline_run12.json` | Drift baseline |
| `SENTINEL_DRIFT_CHECK_INTERVAL` | `50` | KS check frequency |
| `SENTINEL_PREDICT_TIMEOUT` | `60` | Inference timeout (seconds) |
| `TRANSFORMERS_OFFLINE` | `1` | Prevent HuggingFace network calls |
| `TRITON_CACHE_DIR` | `/tmp/triton_cache` | Triton JIT cache (WSL2) |
| `PYTHONPATH` | `.` | Python module path |

## Prometheus Metrics

Scraped every 15 seconds. Custom gauges:
- `sentinel_model_loaded` — 1 if predictor loaded
- `sentinel_gpu_memory_bytes` — current GPU memory
- `sentinel_drift_alerts_total{stat=<name>}` — drift alert counter
