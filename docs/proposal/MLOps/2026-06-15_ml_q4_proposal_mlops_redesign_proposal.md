---
title: SENTINEL MLOps Q4 Proposal — Redesign Proposal
date: 2026-06-15
module: ml
phase: q4
type: proposal
descriptor: redesign_proposal
status: ACTIVE
---

# MLOps Redesign Proposal (2026-06-15)

> **Purpose:** Forward-looking design decisions for the MLOps layer. Answers the
> 6 open questions from the current state findings, establishes the architecture
> for Q4 2026, and defers the Run 13 transition to Phase D.
>
> **Audience:** Tech lead, Ali (for sign-off on G1–G5 decisions).
>
> **Related:** File 2 (current state findings) for the "why"; File 4 (implementation plan) for the "how".

---

## 1. Design Principles

The redesign follows 5 principles, in priority order:

1. **Inference server stays in `ml/`** — tight coupling with model code (`ml.src.models.*`,
   `ml.src.preprocessing.*`) is a feature, not a bug. Moving to a top-level `mlops/`
   would require packaging `ml` as installable, which is a larger refactor than
   the value justifies.

2. **Promote from config, not from hardcoded paths** — the FastAPI default still
   points to Run 4. This is the central gap. We need a single source of truth for
   "which model is the API serving right now?"

3. **Drift monitoring must actually work** — if a baseline is invalid, the
   detector must say so loudly, not silently disable itself.

4. **Defer Run 13 work** — Phase A–C completes Q4 (next 3 weeks) with the
   current Run 12 model. Run 13 re-validation is Phase D (after Run 13 trains).

5. **Deploy-ready artifacts** — checkpoint, thresholds, calibration, drift
   baseline all live next to the model. Docker Compose, env file, startup script
   are all reproducible from a fresh clone.

---

## 2. The 6 Problems We're Solving

| # | Problem | Severity | Status |
|---|---|---|---|
| P1 | Drift detector silently fails with placeholder baseline | HIGH | `[DECIDED]` §3.5 |
| P2 | Inference server defaults to Run 4 (F1=0.3362), not Run 12 (F1=0.7004) | HIGH | `[DECIDED]` §3.2 |
| P3 | No real drift baseline exists | MEDIUM | `[DECIDED]` §3.3 |
| P4 | No Docker Compose — can't deploy | MEDIUM | `[DECIDED]` §3.4 |
| P5 | Stale "10-class" comments + duplicate calibration files | LOW | `[DECIDED]` §3.6 |
| P6 | DVC tracking inconsistent with filesystem | LOW | `[DEFER]` — separate housekeeping pass |

---

## 3. Design Decisions

### 3.1 D1: MLOps stays in `ml/` (not top-level `mlops/`) `[DECIDED]`

**Decision:** Keep all MLOps code in `ml/`. Do not create a top-level `mlops/` folder.

**Rationale:**
- Inference server imports `ml.src.models.*` and `ml.src.preprocessing.*` directly
- Moving to `mlops/` would require packaging `ml` as installable (pyproject.toml
  config, version pins, CI matrix expansion) — ~2 weeks of work
- The operational scripts (`promote_model.py`, `compute_drift_baseline.py`) already
  live in `ml/scripts/`, which is the right place
- Docker Compose (when added in Phase C) will live in `ml/deploy/` — keeps
  deploy config next to deploy code

**Trade-offs:**
- ✓ Single venv, single CI, no packaging overhead
- ✗ `ml/` becomes larger and more heterogeneous
- ✗ Coupling between model and serving is implicit (not enforced by directory boundary)

**Counter-argument considered:** "But the audit said to create `mlops/`." — The
audit's recommendation is one valid option, not the only one. We weigh complexity
vs. value and choose to keep the coupling.

---

### 3.2 D2: Promote from config file (mlops_config.json) `[DECIDED]`

**Decision:** Create `ml/mlops_config.json` as the single source of truth for
"which model is the API serving." Env vars override config; config overrides hardcoded defaults.

**File structure:**
```json
{
  "checkpoint": "ml/checkpoints/GCB-P1-Run12-v3dospatched-20260613_FINAL.pt",
  "thresholds": "ml/checkpoints/GCB-P1-Run12-v3dospatched-20260613_FINAL_thresholds.json",
  "num_classes": 10,
  "experiment": "sentinel-retrain-v2",
  "drift_baseline": "ml/data/drift_baseline_run12.json",
  "drift_check_interval": 50,
  "predict_timeout": 60
}
```

**Code change:** Update `api.py` to read from this file:
```python
def _load_config() -> dict:
    config_path = os.getenv("SENTINEL_CONFIG", "ml/mlops_config.json")
    if Path(config_path).exists():
        with open(config_path) as f:
            return json.load(f)
    return {}

_CONFIG = _load_config()
CHECKPOINT: str = os.getenv("SENTINEL_CHECKPOINT", _CONFIG.get("checkpoint", "<default>"))
DRIFT_BASELINE_PATH: str = os.getenv("SENTINEL_DRIFT_BASELINE", _CONFIG.get("drift_baseline", "ml/data/drift_baseline.json"))
```

**Promotion workflow:**
1. `python ml/scripts/promote_model.py --checkpoint <new.pt> --stage Staging --val-f1-macro 0.7004 --note "Run 12"`
2. After promote succeeds, run `python ml/scripts/set_active_checkpoint.py <new_checkpoint_name>`
3. This script updates `mlops_config.json` (atomic write) to point at the new checkpoint
4. Restart the API server (or send SIGHUP — not yet supported; restart is fine for now)
5. Hit `/health` to verify the new model is loaded

**Why not just env vars?** Env vars are great for one-off overrides, but they
don't survive a config loss or a new operator reading the repo. A file makes
the active model visible to anyone with shell access.

**Why not just MLflow?** MLflow is great for the registry (Staging/Production/None),
but the API server doesn't currently query MLflow at startup. Adding that would
be more code than a 10-line config file. (We can revisit if/when we need blue/green
or A/B deploys.)

---

### 3.3 D3: Build drift baseline from warmup traffic (not training data) `[DECIDED]`

**Decision:** Use `compute_drift_baseline.py --source warmup` after collecting 30+
real inference requests. Do not use `--source training`.

**Why not training data:** The script's own warning (`compute_drift_baseline.py:110-116`):
> "WARNING: --source training uses BCCC-SCsVul-2024 data. This baseline will fire
> alerts on most modern 2026 contracts."

**How we'll collect warmup traffic:**
1. Start the API server with the new Run 12 config
2. Send 30+ real contracts through `/predict` (can be the SmartBugs Curated set
   as a smoke test)
3. Call `DriftDetector.dump_warmup_to_jsonl("ml/data/warmup_run12.jsonl")` — a
   one-line helper we'll add
4. Run `python ml/scripts/compute_drift_baseline.py --source warmup --warmup-log
   ml/data/warmup_run12.jsonl --output ml/data/drift_baseline_run12.json`
5. Update `mlops_config.json` to point at the new baseline

**Output JSON shape (warmup source):**
```json
{
  "source": "warmup",
  "num_nodes": [list of floats from N requests],
  "num_edges": [list of floats],
  "confirmed_count": [list of floats],
  "suspicious_count": [list of floats]
}
```

**Why we add `dump_warmup_to_jsonl()` (not just dump_warmup_stats()):** The existing
method returns a list of dicts. The script reads JSONL. Either format works, but
JSONL is the script's expectation. One-line method.

---

### 3.4 D4: Docker Compose for inference + Prometheus (Grafana deferred) `[DECIDED]`

**Decision:** Phase C delivers:
- `ml/deploy/docker-compose.yml` — inference service + Prometheus service
- `ml/deploy/Dockerfile.inference` — Python 3.12 + venv + dependencies
- `ml/deploy/prometheus.yml` — scrape config for `/metrics` on port 8001
- `ml/deploy/.env.example` — all env vars with sensible defaults
- Grafana is **deferred** (out of scope for Q4; can be added in 1 hour when needed)

**Service definitions:**

```yaml
# ml/deploy/docker-compose.yml
version: "3.8"
services:
  inference:
    build:
      context: ..
      dockerfile: deploy/Dockerfile.inference
    container_name: sentinel-inference
    ports:
      - "8001:8001"
    env_file:
      - .env
    volumes:
      - ../../ml/checkpoints:/app/ml/checkpoints:ro
      - ../../ml/data:/app/ml/data:rw
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8001/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  prometheus:
    image: prom/prometheus:latest
    container_name: sentinel-prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus-data:/prometheus
    depends_on:
      inference:
        condition: service_healthy

volumes:
  prometheus-data:
```

**Deployment command:** `cd ml/deploy && docker compose up -d`

**Why Grafana is deferred:** Adds another service to maintain + dashboard JSON
to version-control. Not on the critical path for Run 12 serving. Easy to add later
(1 hour of work).

---

### 3.5 D5: Drift detector validates baseline on load `[DECIDED]`

**Decision:** DriftDetector.__init__ validates that the loaded baseline has at
least one known stat name. If not, it forces warm-up mode and logs a loud warning.

**Code change (sketch, full impl in Phase A.1):**
```python
# In DriftDetector.__init__, after loading baseline:
if self._baseline is not None:
    KNOWN_STAT_NAMES = {"num_nodes", "num_edges", "confirmed_count", "suspicious_count"}
    actual_stat_names = set(self._baseline.keys())
    valid_stat_names = actual_stat_names & KNOWN_STAT_NAMES
    if not valid_stat_names:
        logger.warning(
            f"DriftDetector: baseline at {bp} contains no known stat names "
            f"(found: {sorted(actual_stat_names)}). Treating as warm-up mode — "
            f"alerts will be suppressed until a real baseline is provided."
        )
        self._baseline = None
        self._warmup_done = False
    else:
        logger.info(
            f"DriftDetector: baseline loaded with {len(valid_stat_names)} known stats: "
            f"{sorted(valid_stat_names)}"
        )
```

**Behavior:**
- Placeholder baseline → detector enters warm-up mode, suppresses alerts, logs loudly
- Real baseline (warmup or training) → detector validates stat names, alerts fire
- Missing file → existing behavior (warm-up mode with default n_warmup=500)

**Backward compat:** None needed — this is a defensive check that makes the
"loaded" state actually meaningful.

---

### 3.6 D6: Update stale comments + resolve duplicate calibration files `[DECIDED]`

**Decision:**
- Replace "10-class" with "9- or 10-class" (or "NUM_CLASSES-class") in `api.py:7, 166`
- Move `ml/calibration/run12/temperatures_run12.json` to `docs/.bin/2026-06-15_ml_q4_proposal_mlops_duplicate_calibration_cleanup/`
- Keep `ml/calibration/run12/temperatures_run12_stats.json` and `_ece_comparison.png` (these are unique to the subdir)
- Update `ml/src/inference/preprocess.py:39, 55` to say "v9" (or "12 in v9; was 8 in v4")

**Effort:** 10 minutes total.

---

## 4. What Stays the Same (deliberately NOT redesigned)

| Thing | Why it stays |
|---|---|
| `_ARCH_TO_FUSION_DIM` allowlist in predictor.py | It works. Adding "v9" / "v10" entries is trivial. |
| Per-class thresholds from companion JSON | Working as designed. Run 12 has 10 entries. Run 13 will have 9. |
| Content-hash cache | Working as designed. Schema version invalidation is automatic. |
| `promote_model.py` gate logic (F1-must-beat-current + baseline-must-exist) | Working as designed. We just need to use the script. |
| Prometheus `/metrics` exposure via prometheus-fastapi-instrumentator | Auto-instrumented. Custom gauges for model_loaded + gpu_memory are already added. |
| BF16 → float() for inference | Working. Stripping `_orig_mod.` prefix works. |

---

## 5. Architecture Diagram (text-based)

```
                    External consumers
                          │
        ┌─────────────────┼─────────────────┐
        │                 │                 │
   agents (MCP)      direct API       CI / smoke tests
        │                 │                 │
        └────────┬────────┴────────┬────────┘
                 │                 │
                 ▼                 ▼
           http://localhost:8001 (uvicorn)
                 │
                 ▼
   ┌─────────────────────────────────────────┐
   │ api.py (FastAPI)                        │
   │   /health /predict /hotspots /metrics   │
   └─────────────────────────────────────────┘
        │                │              │
        ▼                ▼              ▼
   predictor.py    preprocess.py   drift_detector.py
        │                │              │
        │                │              │
        ▼                ▼              ▼
   checkpoint       graph_extractor    ml/data/
   + thresholds     + tokenizer        drift_baseline.json
        │                │              │
        ▼                ▼              ▼
   ml/checkpoints/   ml/data/graphs/   Prometheus counter
   *FINAL.pt        *tokens.pt        sentinel_drift_alerts_total
   *_thresholds.json
```

**Operational concerns (separate from serving):**
- `ml/scripts/promote_model.py` — promotes checkpoint to MLflow Staging/Production
- `ml/scripts/compute_drift_baseline.py` — builds drift baseline JSON
- `ml/scripts/set_active_checkpoint.py` — **NEW** — updates `mlops_config.json` (Phase B.2)
- `ml/scripts/dump_warmup_to_jsonl.py` — **NEW** — one-off script for warmup dump (Phase B.4)

---

## 6. Decisions That Are Explicitly DEFERRED

### 6.1 Top-level `mlops/` folder — DEFERRED (not needed)

Per §3.1.

### 6.2 MLflow-as-config — DEFERRED (not needed yet)

The `promote_model.py` script registers to MLflow, but the API server doesn't
read from MLflow at startup. Reading from MLflow would enable blue/green
deploys, A/B testing, and rollbacks via MLflow UI. But that's a Phase D+
feature. For now, `mlops_config.json` is simpler and sufficient.

### 6.3 `/debug/warmup_dump` HTTP endpoint — DEFERRED (replaced by helper method)

The original audit suggested adding a `/debug/warmup_dump` endpoint to api.py
(audit §2 row 7). We can do this, but it's actually more code than a one-line
`dump_warmup_to_jsonl(path)` method on DriftDetector that we call from a
one-off script. Easier to test, easier to remove if we later build the HTTP endpoint.

### 6.4 Grafana dashboards — DEFERRED (out of Q4 scope)

Per §3.4. 1 hour of work to add later when dashboards are actually needed.

### 6.5 Run 13 transition (NUM_CLASSES=9) — DEFERRED to Phase D

Per `2026-06-15_ml_q4_proposal_mlops_risks_dependencies.md` §R7. Phase A–C
completes Q4 with Run 12 (10-class). Run 13 re-validation happens after Run 13
trains (target: 3 weeks from now).

---

## 7. Success Criteria

This redesign is complete when:

| ID | Criterion | Verified by |
|---|---|---|
| S1 | `/health` returns `checkpoint: ml/checkpoints/GCB-P1-Run12-v3dospatched-20260613_FINAL.pt` | `curl localhost:8001/health` |
| S2 | `/predict` returns probabilities dict with 10 classes (CallToUnknown, ..., UnusedReturn) | `curl -X POST .../predict -d @test.sol` |
| S3 | Drift detector logs warning if baseline is invalid, suppresses alerts correctly | Unit test or smoke |
| S4 | `python ml/scripts/set_active_checkpoint.py GCB-P1-Run12-v3dospatched-20260613_FINAL` updates `mlops_config.json` | `cat mlops_config.json` |
| S5 | `python ml/scripts/compute_drift_baseline.py --source warmup --warmup-log ml/data/warmup_run12.jsonl --output ml/data/drift_baseline_run12.json` produces a valid baseline | `cat ml/data/drift_baseline_run12.json \| jq '.num_nodes \| length'` (should be 30+) |
| S6 | `cd ml/deploy && docker compose up -d` starts inference + Prometheus, both healthy | `docker compose ps` |
| S7 | No "10-class" hardcoded references remain (use NUM_CLASSES constant) | `grep -rn "10-class" ml/src/` returns 0 results |

---

## 8. Open Questions (for Ali)

| Q | Question | Default answer if not answered |
|---|---|---|
| OQ1 | Should MLOps use a config file or env-only? | Use config file (per §3.2) |
| OQ2 | Should we add `/debug/warmup_dump` HTTP endpoint now or defer? | Defer to Phase D (use helper method instead) |
| OQ3 | Should the API server read from MLflow for blue/green deploys? | Defer to post-Run-13 |
| OQ4 | Should the F1-must-beat-current gate also apply to Staging? | No (Staging is for evaluation, not production-grade gates) |

---

## 9. References

- **File 2 (current state):** `2026-06-15_ml_q4_proposal_mlops_current_state_findings.md`
- **File 4 (implementation plan):** `2026-06-15_ml_q4_proposal_mlops_q4_implementation_plan.md`
- **File 5 (risks):** `2026-06-15_ml_q4_proposal_mlops_risks_dependencies.md`
- **Audit (prior):** `ml/MLOPS_STATE_AND_REDESIGN_2026-06-14.md`
- **Run 12 launch:** `~/.claude/projects/.../memory/2026-06-13_project_run12_launch.md`
