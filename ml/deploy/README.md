---
title: SENTINEL MLOps Deployment Guide
date: 2026-06-17
module: ml
phase: q4
type: deployment
descriptor: docker_compose_deployment
status: ACTIVE
---

# SENTINEL MLOps — Docker Deployment Guide (Q4 MLOps Phase C.6)

> One-stop reference for deploying the SENTINEL inference API + Prometheus
> monitoring stack. Covers prerequisites, quick start, configuration,
> model updates, and troubleshooting.

---

## 1. What This Stack Runs

| Service    | Image                       | Port  | Purpose |
|------------|-----------------------------|-------|---------|
| inference  | `sentinel-inference:1.0.0` (built locally) | 8001  | FastAPI serving Run 12 (F1=0.7004) |
| prometheus | `prom/prometheus:v2.55.0`   | 9090  | Scrapes `/metrics` from inference; UI at http://localhost:9090 |

Both services run on the internal `sentinel-net` Docker bridge network.

---

## 2. Prerequisites

| Tool                    | Version       | Notes |
|-------------------------|---------------|-------|
| Docker                  | 24+           | `docker --version` |
| Docker Compose          | v2 (built-in) | `docker compose version` |
| NVIDIA Container Toolkit| latest        | Only if you want GPU passthrough (RTX 3070 needs this) |
| Disk space              | ~10 GB        | Image (~5 GB) + Run 12 checkpoint (~280 MB) + Prometheus TSDB |

The stack does NOT need internet at runtime (TRANSFORMERS_OFFLINE=1 baked in).

---

## 3. Quick Start

```bash
# 1. Copy the env template
cd ml/deploy
cp .env.example .env

# 2. Build the inference image (first time only — ~5 min)
cd ..
docker compose -f ml/deploy/docker-compose.yml build

# 3. Start the stack
cd ml/deploy
docker compose up -d

# 4. Wait for the inference container to pass healthcheck (~60s for model load)
docker compose ps
# Expected: both services "healthy"

# 5. Smoke test the API
curl -s http://localhost:8001/health | jq
# Expected: {"status":"ok","predictor_loaded":true,...}

# 6. Make a real prediction
curl -s -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d '{"source_code":"pragma solidity ^0.8.0; contract C { uint x; function f() public { x = 1; } }"}' | jq

# 7. Check Prometheus is scraping
open http://localhost:9090/targets
# Expected: sentinel-inference job "UP"

# 8. Stop the stack
docker compose down
```

---

## 4. Configuration

### Two-layer config (file + env vars)

The MLOps config is split across **two layers** with env vars taking
precedence over the file:

| Layer | File | Env var override |
|---|---|---|
| MLOps config | `ml/mlops_config.json` | `SENTINEL_CONFIG` (changes the file path) |
| Per-field   | `mlops_config.json.checkpoint` | `SENTINEL_CHECKPOINT` |
| Per-field   | `mlops_config.json.drift_baseline` | `SENTINEL_DRIFT_BASELINE` |

The shipped `mlops_config.json` is the **canonical** source of truth for
Run 12 — no env vars needed for a default deployment.

### Updating the active model (without rebuilding the image)

Use `set_active_checkpoint.py` to point the config at a new checkpoint:

```bash
# Atomic update of mlops_config.json
python ml/scripts/set_active_checkpoint.py GCB-P1-Run12-v3dospatched-20260613_FINAL

# Restart the inference container to pick up the new config
docker compose -f ml/deploy/docker-compose.yml restart inference
```

This avoids rebuilding the Docker image — the bind mount shares the
updated `mlops_config.json` with the running container.

### Building the image with a different checkpoint

```bash
# Edit mlops_config.json first
python ml/scripts/set_active_checkpoint.py GCB-P1-Run13-v4bcccme-20260630_FINAL

# Then rebuild
docker compose -f ml/deploy/docker-compose.yml build --no-cache inference
docker compose -f ml/deploy/docker-compose.yml up -d inference
```

---

## 5. Monitoring — Prometheus Metrics

### Custom SENTINEL metrics (defined in `ml/src/inference/api.py`)

| Metric                          | Type    | Labels      | Meaning |
|---------------------------------|---------|-------------|---------|
| `sentinel_model_loaded`         | gauge   | —           | 1 = Run 12 loaded, 0 = not loaded |
| `sentinel_gpu_memory_bytes`     | gauge   | —           | Current GPU memory allocated (bytes) |
| `sentinel_drift_alerts_total`   | counter | `stat`      | KS drift alerts (p<0.05), per stat name |
| `http_requests_total`           | counter | method,status,handler | Default FastAPI instrumentation |
| `http_request_duration_seconds` | histogram | method,status,handler | Default FastAPI instrumentation |

### Drift detection

The `sentinel_drift_alerts_total{stat="num_nodes"}` counter fires whenever
the Kolmogorov–Smirnov test detects a significant distributional shift
between the current request window and the baseline file at
`ml/data/drift_baseline_run12.json`.

**Useful PromQL queries:**

```promql
# Drift alerts per stat, last 1h
rate(sentinel_drift_alerts_total[1h])

# Model is healthy
sentinel_model_loaded == 1

# Request rate
rate(http_requests_total{handler="/predict"}[5m])

# p95 latency for /predict
histogram_quantile(0.95, rate(http_request_duration_seconds_bucket{handler="/predict"}[5m]))
```

### Alertmanager (future)

The prometheus.yml does NOT include alerting rules yet. Adding
`rule_files:` and an Alertmanager service is a Phase C+ follow-up.

---

## 6. GPU Support

The inference API is GPU-aware but the compose file does NOT enable GPU
by default. To enable on an NVIDIA host (RTX 3070 etc.):

### 1. Install nvidia-container-toolkit

```bash
# Ubuntu / WSL2
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### 2. Uncomment the GPU block in `docker-compose.yml`

In the `inference` service, uncomment:

```yaml
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

### 3. Restart

```bash
docker compose -f ml/deploy/docker-compose.yml up -d inference
docker compose -f ml/deploy/docker-compose.yml exec inference nvidia-smi
# Should show the GPU
```

Without GPU, the inference still works (CPU fallback) but is ~10x slower.

---

## 7. Volumes

| Host path                  | Container path             | Mode | Purpose |
|----------------------------|----------------------------|------|---------|
| `../../ml/checkpoints`     | `/app/ml/checkpoints`      | ro   | Model weights (~280 MB for Run 12) |
| `../../ml/data`            | `/app/ml/data`             | ro   | Drift baseline, warmup stats |
| `../../ml/calibration`     | `/app/ml/calibration`      | ro   | Per-run temperature files |
| `../../ml/mlops_config.json` | `/app/ml/mlops_config.json` | ro   | Active config |
| (named volume)             | `/prometheus`              | rw   | Prometheus TSDB retention (30d) |

In production, replace the bind mounts with a named volume and a DVC pull
step at container start. Bind mounts are dev-friendly but expose host
paths into the container.

---

## 8. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Container exits with "Checkpoint not found" | Wrong path in `mlops_config.json` | Run `set_active_checkpoint.py` to point at the right file |
| `curl /health` returns 503 | Model still loading (~30s) | Wait, then retry. Or check `docker logs sentinel-inference` |
| Prometheus target shows "DOWN" | Wrong hostname in prometheus.yml | Should be `inference:8001` (Docker DNS), not `localhost:8001` |
| `sentinel_drift_alerts_total` not present | Detector still in warm-up mode | Baseline not loaded — check `mlops_config.json` points at `drift_baseline_run12.json` |
| `sentinel_model_loaded == 0` | Model failed to load (CUDA OOM, etc.) | Check `docker logs sentinel-inference` for traceback |
| GPU not detected inside container | nvidia-container-toolkit not installed | See §6 above |

---

## 9. C.5 — End-to-End Smoke Test (Run in Docker-enabled Environment)

The C.5 step from the Q4 plan requires Docker. To run it:

```bash
# 1. Start the stack
cd ml/deploy
cp .env.example .env
docker compose up -d

# 2. Wait for both services healthy
docker compose ps

# 3. Verify Run 12 is loaded
curl -s http://localhost:8001/health | jq
# Expect: "predictor_loaded": true, "model_f1_val": 0.6800 (Run 12 best)

# 4. Make a real prediction (use a contract from the v0.1 quickstart benchmark)
curl -s -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d "$(cat ../../data_module/benchmarks/benchmark_v0.1_quickstart/contracts/*.sol | python -c 'import sys,json; print(json.dumps({"source_code": sys.stdin.read()}))')" | jq

# 5. Verify Prometheus is scraping
curl -s http://localhost:9090/api/v1/query?query=sentinel_model_loaded | jq
# Expect: .data.result[0].value[1] == "1"

# 6. Tear down
docker compose down
```

**Status 2026-06-17:** Docker is not available in the dev environment, so
C.5 was NOT executed locally. The YAML, Dockerfile, and prometheus.yml
were authored against the Q4 plan §C.1–C.4 and reviewed for correctness.
C.5 must be run in a Docker-enabled environment (CI or production host)
to verify the full stack.

---

## 10. Cross-References

- **Q4 plan (the design doc for this work):** `docs/proposal/MLOps/2026-06-15_ml_q4_proposal_mlops_q4_implementation_plan.md`
- **mlops_config.json (canonical config):** `ml/mlops_config.json`
- **Drift detector (the monitor):** `ml/src/inference/drift_detector.py`
- **B.4 baseline file (real, not placeholder):** `ml/data/drift_baseline_run12.json`
- **set_active_checkpoint.py (atomic config update):** `ml/scripts/set_active_checkpoint.py`
- **Prometheus metrics definition:** `ml/src/inference/api.py:69-71` (custom gauges) and `prometheus-fastapi-instrumentator` (default FastAPI metrics)

---

## 11. Phase C Status

| Step | Status | Notes |
|---|---|---|
| C.1 Dockerfile.inference | ✅ Done 2026-06-17 | Python 3.12.1 slim, multi-layer build, healthcheck, GPU-ready |
| C.2 docker-compose.yml | ✅ Done 2026-06-17 | inference + prometheus on internal bridge |
| C.3 prometheus.yml | ✅ Done 2026-06-17 | 15s scrape interval, custom SENTINEL labels |
| C.4 .env.example | ✅ Done 2026-06-17 | All env vars documented with defaults |
| C.5 E2E smoke test | ⏸️ DEFERRED 2026-06-17 | Requires Docker (not in dev env) |
| C.6 README | ✅ Done 2026-06-17 | This file |
