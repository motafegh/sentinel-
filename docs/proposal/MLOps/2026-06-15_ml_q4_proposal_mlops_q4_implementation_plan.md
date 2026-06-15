---
title: SENTINEL MLOps Q4 Proposal — Q4 Implementation Plan
date: 2026-06-15
module: ml
phase: q4
type: proposal
descriptor: q4_implementation_plan
status: ACTIVE
---

# MLOps Q4 Implementation Plan (2026-06-15)

> **Purpose:** Concrete work plan for the next 3 weeks (Q4 2026), structured as
> 3 phases (A, B, C) plus a deferred Phase D. Each step has effort estimate,
> files to modify, success criterion, and verification command.
>
> **Related:** File 3 (redesign proposal) for design decisions; File 5 (risks) for blockers.

---

## Summary

| Phase | Title | Effort | When (Q4 weeks) | Status |
|---|---|---|---|---|
| **A** | Bugs + housekeeping | 1.5 hr | Week 1 | Ready to start |
| **B** | Wire Run 12 into the API | 3 hr | Week 1-2 | Blocked on A |
| **C** | Docker + deployment | 3 hr | Week 2-3 | Blocked on B |
| **D** | Post-Run-13 transition | ~1 day | After Run 13 trains | DEFER |

**Total Q4 effort:** ~7.5 hours active work + monitoring.

---

## Phase A: Bugs + Housekeeping (~1.5 hr)

### A.1 Fix drift detector silent failure — 30 min

**Problem:** Detector thinks placeholder baseline is real, no alerts ever fire.

**Files:**
- `ml/src/inference/drift_detector.py:91-110` — add baseline validation in `__init__`
- `ml/src/inference/drift_detector.py:149-153` — defensive check in `check()`

**Implementation:**
```python
# Add at top of file (after KS_ALPHA = 0.05):
_KNOWN_STAT_NAMES = frozenset({"num_nodes", "num_edges", "confirmed_count", "suspicious_count"})

# In __init__, replace lines 92-100:
if baseline_path is not None:
    bp = Path(baseline_path)
    if bp.exists():
        with open(bp) as f:
            loaded_baseline = json.load(f)
        # D5 fix: validate baseline has known stat names
        loaded_stat_names = set(loaded_baseline.keys()) if isinstance(loaded_baseline, dict) else set()
        valid_stat_names = loaded_stat_names & _KNOWN_STAT_NAMES
        if not valid_stat_names:
            logger.warning(
                f"DriftDetector: baseline at {bp} contains no known stat names "
                f"(found: {sorted(loaded_stat_names)}). Treating as warm-up mode — "
                f"alerts will be suppressed until a real baseline is provided."
            )
            self._baseline = None
            self._warmup_done = False
        else:
            self._baseline = loaded_baseline
            self._warmup_done = True
            logger.info(
                f"DriftDetector: baseline loaded from {bp} "
                f"({len(valid_stat_names)} stats: {sorted(valid_stat_names)})"
            )
    else:
        # existing "not found" branch
        ...
```

**Success criterion:** With the placeholder `drift_baseline.json` in place,
starting the API logs a warning: "DriftDetector: baseline at ... contains no known stat names ... Treating as warm-up mode".

**Verification command:**
```bash
# Start API, check log
uvicorn ml.src.inference.api:app --port 8001 2>&1 | grep "DriftDetector"
# Should see: "DriftDetector: baseline at ... contains no known stat names"
```

**Rollback:** Revert the changes; the original behavior is preserved (just
silently broken).

---

### A.2 Update stale "10-class" comments — 5 min

**Files:**
- `ml/src/inference/api.py:7` — docstring
- `ml/src/inference/api.py:166` — PredictResponse comment
- `ml/src/inference/preprocess.py:39, 55` — schema comments

**Changes:**
- api.py:7 — "full 10-class vector" → "full NUM_CLASSES-class vector"
- api.py:166 — "Full 10-class probability vector" → "Full NUM_CLASSES-class probability vector"
- preprocess.py:39 — "currently 13 in v5, was 8 in v1/v4" → "currently 12 in v9"
- preprocess.py:55 — "(13 in v5; was 8 in v4)" → "(12 in v9; was 8 in v4)"

**Success criterion:** `grep -rn "10-class" ml/src/` returns 0 results.

**Verification command:** `grep -rn "10-class\|13 in v5" ml/src/`

---

### A.3 Resolve duplicate Run 12 calibration files — 5 min

**Files:**
- Move: `ml/calibration/run12/temperatures_run12.json` → `docs/.bin/2026-06-15_ml_q4_proposal_mlops_duplicate_calibration_cleanup/`
- Keep: `ml/calibration/run12/temperatures_run12_stats.json`, `temperatures_run12_ece_comparison.png`
- Keep: `ml/calibration/temperatures_run12.json` (root level, canonical)

**Why:** Root-level `temperatures_run12.json` is the canonical artifact (per
`promote_model.py:200` which logs the threshold file name). Subdir copy is duplicate.

**Commands:**
```bash
mkdir -p docs/.bin/2026-06-15_ml_q4_proposal_mlops_duplicate_calibration_cleanup
mv ml/calibration/run12/temperatures_run12.json docs/.bin/2026-06-15_ml_q4_proposal_mlops_duplicate_calibration_cleanup/
# Verify only the canonical remains at root
ls ml/calibration/temperatures_run12.json  # should exist
ls ml/calibration/run12/  # should show stats.json + ece_comparison.png only
```

**Success criterion:** Only one `temperatures_run12.json` exists (at root level).

---

### A.4 Decide DVC tracking policy — 15 min

**Files:** `ml/checkpoints.dvc`

**Decision (per §3.6 D6):** Option B — track only canonical checkpoints
(1 per run + 1 `*_FINAL.pt` immutable copy). Archive the rest.

**Implementation:**
```bash
# Inspect current state
dvc list ml/ --recursive
dvc status

# Decide which checkpoints to keep
# Keep: GCB-P1-Run12-v3dospatched-20260613_FINAL.pt (+ state + thresholds)
# Keep: One checkpoint per other run for historical reference
# Move: All smoke-test checkpoints, intermediate state.json files

# Create archive directory
mkdir -p ml/checkpoints/_archive/

# Move non-canonical checkpoints
mv ml/checkpoints/GCB-P1-Run10-smoke-test_best.pt ml/checkpoints/_archive/
mv ml/checkpoints/GCB-P1-Run7-smoke-test_best.pt ml/checkpoints/_archive/
mv ml/checkpoints/GCB-P1-Run8-smoke-20260605_best.pt ml/checkpoints/_archive/
# ... etc.

# Re-track with dvc
dvc add ml/checkpoints
dvc commit -m "Q4 MLOps: canonicalize checkpoint tracking"
```

**Success criterion:** `dvc status` shows clean. `dvc list ml/checkpoints/`
shows only the canonical files.

**Caveat:** This is judgment-call work. May take longer than 15 min if there
are many smoke-test variants. Defer to a quiet afternoon.

---

### A.5 Smoke test: does Run 12 still load cleanly? — 30 min

**Purpose:** Regression guard. Before changing anything, verify the baseline works.

**Steps:**
1. Activate venv: `source ml/.venv/bin/activate`
2. Set env var: `export SENTINEL_CHECKPOINT=ml/checkpoints/GCB-P1-Run12-v3dospatched-20260613_FINAL.pt`
3. Start API: `uvicorn ml.src.inference.api:app --port 8001 &`
4. Wait for "Predictor ready" log
5. Hit health: `curl -s localhost:8001/health | jq`
6. Expected output:
   ```json
   {
     "status": "ok",
     "predictor_loaded": true,
     "checkpoint": "ml/checkpoints/GCB-P1-Run12-v3dospatched-20260613_FINAL.pt",
     "architecture": "four_eye_v8",
     "thresholds_loaded": true,
     "model_epoch": 51,
     "model_f1_val": 0.6800766276074683
   }
   ```
7. Test predict: find a small Solidity contract, e.g., `data_module/benchmarks/benchmark_v0.1_quickstart/` and run a request
8. Verify response has 10 classes in probabilities dict
9. Kill API: `pkill -f "uvicorn ml.src.inference.api"`

**Success criterion:** All steps complete without exceptions. Response
matches expected shape.

**If it fails:** Stop. The current code has a bug, or the Run 12 checkpoint
is corrupted. Investigate before proceeding to Phase B.

---

### Phase A total: ~1.5 hr

---

## Phase B: Wire Run 12 into the API (~3 hr)

### B.1 Create `mlops_config.json` — 15 min

**File:** `ml/mlops_config.json` (new)

**Contents:**
```json
{
  "checkpoint": "ml/checkpoints/GCB-P1-Run12-v3dospatched-20260613_FINAL.pt",
  "thresholds": "ml/checkpoints/GCB-P1-Run12-v3dospatched-20260613_FINAL_thresholds.json",
  "num_classes": 10,
  "experiment": "sentinel-retrain-v2",
  "drift_baseline": "ml/data/drift_baseline.json",
  "drift_check_interval": 50,
  "predict_timeout": 60
}
```

**Why these values:**
- `checkpoint`: the canonical Run 12 immutable copy
- `thresholds`: auto-discovered by predictor.py (companion JSON to checkpoint)
- `num_classes`: 10 (will become 9 in Phase D)
- `experiment`: matches the existing MLflow experiment name
- `drift_baseline`: will be replaced in B.4 with a real baseline
- Other values: defaults that match the current env vars

**Success criterion:** File exists, valid JSON.

**Verification:** `python -c "import json; print(json.load(open('ml/mlops_config.json')))"`

---

### B.2 Update `api.py` to read from config — 30 min

**File:** `ml/src/inference/api.py:40-59` (env-var-reading block)

**Add a config loader at the top of the file (after imports, before existing env-var lines):**
```python
def _load_mlops_config() -> dict:
    """Load mlops_config.json if present. Env vars take precedence over file values."""
    config_path = os.getenv("SENTINEL_CONFIG", "ml/mlops_config.json")
    if Path(config_path).exists():
        with open(config_path) as f:
            return json.load(f)
    return {}

_CONFIG = _load_mlops_config()
```

**Modify the env-var-reading block (lines 40-59) to use config as default:**
```python
import json
from pathlib import Path  # already imported

DRIFT_BASELINE_PATH: str = os.getenv(
    "SENTINEL_DRIFT_BASELINE",
    _CONFIG.get("drift_baseline", "ml/data/drift_baseline.json"),
)
DRIFT_CHECK_INTERVAL: int = int(os.getenv(
    "SENTINEL_DRIFT_CHECK_INTERVAL",
    str(_CONFIG.get("drift_check_interval", 50)),
))

CHECKPOINT: str = os.getenv(
    "SENTINEL_CHECKPOINT",
    _CONFIG.get("checkpoint", "ml/checkpoints/GCB-P1-Run4-no-asl-pw_best.pt"),  # fallback only
)
PREDICT_TIMEOUT: float = float(os.getenv(
    "SENTINEL_PREDICT_TIMEOUT",
    str(_CONFIG.get("predict_timeout", 60)),
))
```

**Also update the docstring (line 13) to point at mlops_config.json:**
```python
"""
...
CHECKPOINT: read from mlops_config.json (`checkpoint` field) or SENTINEL_CHECKPOINT
            env var. Defaults to Run 4 path for backward compat.
...
"""
```

**Success criterion:** API loads Run 12 from `mlops_config.json` even without `SENTINEL_CHECKPOINT` env var set.

**Verification:** Same as A.5, but without setting the env var.

---

### B.3 Create `set_active_checkpoint.py` script — 30 min

**File:** `ml/scripts/set_active_checkpoint.py` (new)

**Purpose:** Update `mlops_config.json` to point at a new checkpoint. Atomic write.

**Skeleton:**
```python
#!/usr/bin/env python3
"""
set_active_checkpoint.py — Update mlops_config.json to point at a new checkpoint.

Usage:
    python ml/scripts/set_active_checkpoint.py GCB-P1-Run12-v3dospatched-20260613_FINAL
    python ml/scripts/set_active_checkpoint.py GCB-P1-Run13-v4bcccme-20260630_FINAL --dry-run
"""
import argparse
import json
import sys
from pathlib import Path

CONFIG_PATH = Path("ml/mlops_config.json")


def main():
    parser = argparse.ArgumentParser(...)
    parser.add_argument("checkpoint_name", help="Filename of the .pt checkpoint (no path)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    
    checkpoint_path = Path("ml/checkpoints") / args.checkpoint_name
    if not checkpoint_path.exists():
        print(f"ERROR: checkpoint not found: {checkpoint_path}", file=sys.stderr)
        sys.exit(1)
    
    # Read existing config
    if not CONFIG_PATH.exists():
        print(f"ERROR: {CONFIG_PATH} not found", file=sys.stderr)
        sys.exit(1)
    with CONFIG_PATH.open() as f:
        config = json.load(f)
    
    # Update
    config["checkpoint"] = str(checkpoint_path)
    # Auto-discover thresholds
    thresholds = checkpoint_path.with_name(f"{checkpoint_path.stem}_thresholds.json")
    if thresholds.exists():
        config["thresholds"] = str(thresholds)
    
    if args.dry_run:
        print(json.dumps(config, indent=2))
        return 0
    
    # Atomic write
    tmp = CONFIG_PATH.with_suffix(".tmp")
    with tmp.open("w") as f:
        json.dump(config, f, indent=2)
    tmp.rename(CONFIG_PATH)
    print(f"Updated {CONFIG_PATH} → checkpoint={checkpoint_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

**Success criterion:** Running the script updates `mlops_config.json` and the
new checkpoint path is reflected.

**Verification:**
```bash
python ml/scripts/set_active_checkpoint.py GCB-P1-Run12-v3dospatched-20260613_FINAL
cat ml/mlops_config.json | jq .checkpoint  # should show new path
```

---

### B.4 Build real drift baseline (warmup source) — 1 hr

**Prerequisites:** A.1 (drift detector fix) and B.2 (Run 12 wired into API) complete.

**Steps:**

1. **Add `dump_warmup_to_jsonl(path)` method to DriftDetector** (5 min)
   - File: `ml/src/inference/drift_detector.py`
   - Add after `dump_warmup_stats()`:
   ```python
   def dump_warmup_to_jsonl(self, path: str | Path) -> None:
       """Dump the current rolling buffer to a JSONL file for baseline building."""
       import json
       path = Path(path)
       with path.open("w") as f:
           for stats in self._buffer:
               f.write(json.dumps(stats) + "\n")
       logger.info(f"Dumped {len(self._buffer)} warmup samples to {path}")
   ```

2. **Start the API** with Run 12 wired in:
   ```bash
   uvicorn ml.src.inference.api:app --port 8001 &
   ```

3. **Send 50 real inference requests** through the API. Use the SmartBugs Curated
   contracts or the v0.1 quickstart benchmark contracts as a smoke source:
   ```bash
   for f in data_module/benchmarks/benchmark_v0.1_quickstart/contracts/*.sol; do
     curl -s -X POST localhost:8001/predict \
       -H "Content-Type: application/json" \
       -d "{\"source_code\": \"$(cat $f | python -c 'import sys,json; print(json.dumps(sys.stdin.read()))')\"}" \
       > /dev/null
   done
   ```

4. **Dump the warmup buffer** to JSONL:
   ```python
   # In a Python shell with the API still running, OR add a small one-off script
   from ml.src.inference.drift_detector import DriftDetector
   # Reload the detector from a script — but it lives in the API process
   ```
   
   **Better approach:** Add a `/debug/warmup_dump` endpoint to api.py. But we
   deferred this in the redesign. Alternative: write a one-off script that
   imports DriftDetector, calls update_stats() 30+ times with synthetic but
   realistic data, then dumps. Not as good as real traffic, but unblocks the
   baseline-building.

   **Decision:** Use the synthetic approach for now. Real traffic comes when
   agents are wired in. The script `ml/scripts/build_warmup_baseline.py`
   (new, ~30 lines) does this.

5. **Run the baseline builder**:
   ```bash
   python ml/scripts/compute_drift_baseline.py \
     --source warmup \
     --warmup-log ml/data/warmup_run12.jsonl \
     --output ml/data/drift_baseline_run12.json
   ```

6. **Update `mlops_config.json`** to point at the new baseline:
   ```json
   { "drift_baseline": "ml/data/drift_baseline_run12.json", ... }
   ```

7. **Verify:** Restart the API; the drift detector should log:
   "DriftDetector: baseline loaded from ... (4 stats: ['confirmed_count', 'num_edges', 'num_nodes', 'suspicious_count'])"

**Success criterion:** Drift detector loads a real baseline (not placeholder).
4 stats present. Detector enters active mode.

**Verification:** `curl -s localhost:8001/health | jq` (and check logs).

---

### B.5 Add tests for inference layer — 1 hr (if time permits)

**Purpose:** Regression guard for A.1, B.2, B.4.

**Files:**
- `ml/tests/inference/test_drift_detector.py` (new)
- `ml/tests/inference/test_api_config.py` (new)

**Test cases (sketch):**
- `test_drift_detector_placeholder_baseline_triggers_warmup_mode` — verify
  loading placeholder → `_warmup_done = False`
- `test_drift_detector_real_baseline_enters_active_mode` — verify loading
  valid baseline → `_warmup_done = True`
- `test_drift_detector_no_alerts_during_warmup` — verify update_stats during
  warmup doesn't fire alerts
- `test_api_config_loads_mlops_config_json` — verify `_load_mlops_config()`
  returns expected dict
- `test_api_config_env_var_overrides_config_file` — verify SENTINEL_CHECKPOINT
  wins over mlops_config.json

**Effort:** 1 hour. Optional but high value (catches regressions).

**Note:** Audit didn't find any unit tests for the inference layer. Adding
even 5 tests is a significant improvement.

---

### Phase B total: ~3 hr

---

## Phase C: Docker + Deployment (~3 hr)

### C.1 Create `ml/deploy/Dockerfile.inference` — 30 min

**File:** `ml/deploy/Dockerfile.inference` (new)

**Skeleton:**
```dockerfile
FROM python:3.12.1-slim

WORKDIR /app

# Install poetry + dependencies
RUN pip install --no-cache-dir poetry==1.8.0
COPY pyproject.toml poetry.lock ./
RUN poetry config virtualenvs.create false \
 && poetry install --no-dev --no-interaction

# Copy code
COPY ml/ ./ml/
COPY data_module/ ./data_module/

# Expose port
EXPOSE 8001

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD curl -f http://localhost:8001/health || exit 1

# Entrypoint
CMD ["uvicorn", "ml.src.inference.api:app", "--host", "0.0.0.0", "--port", "8001"]
```

**Caveat:** This is a first draft. May need adjustments for solc-select
(multi-version Solidity compiler) and other native deps. Iterate as needed.

**Success criterion:** `docker build -f ml/deploy/Dockerfile.inference -t sentinel-inference .` succeeds.

---

### C.2 Create `ml/deploy/docker-compose.yml` — 30 min

See File 3 §3.4 for the full YAML. The Compose file is the deliverable.

**Success criterion:** `cd ml/deploy && docker compose config` validates
(doesn't error).

---

### C.3 Create `ml/deploy/prometheus.yml` — 15 min

**File:** `ml/deploy/prometheus.yml` (new)

```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: sentinel-inference
    metrics_path: /metrics
    static_configs:
      - targets: ["inference:8001"]
        labels:
          service: sentinel
          environment: local
```

**Success criterion:** Prometheus can scrape the inference service. Verify
in Prometheus UI: http://localhost:9090/targets (after `docker compose up`).

---

### C.4 Create `ml/deploy/.env.example` — 15 min

**File:** `ml/deploy/.env.example` (new)

```bash
# SENTINEL API configuration
SENTINEL_CONFIG=/app/ml/mlops_config.json
# Uncomment to override the config file:
# SENTINEL_CHECKPOINT=/app/ml/checkpoints/GCB-P1-Run12-v3dospatched-20260613_FINAL.pt
# SENTINEL_DRIFT_BASELINE=/app/ml/data/drift_baseline_run12.json

# uvicorn
UVICORN_HOST=0.0.0.0
UVICORN_PORT=8001

# Python
TRANSFORMERS_OFFLINE=1
PYTHONUNBUFFERED=1
```

**Success criterion:** `cp .env.example .env && docker compose up -d` starts
the stack with these defaults.

---

### C.5 End-to-end smoke test in Docker — 30 min

**Steps:**
1. `cd ml/deploy`
2. `cp .env.example .env`
3. `docker compose up -d`
4. Wait for healthcheck: `docker compose ps` (both services should be "healthy")
5. `curl -s localhost:8001/health | jq` — verify Run 12 is loaded
6. `curl -s localhost:8001/predict -d @test_contract.json` — verify prediction
7. `curl -s localhost:9090/api/v1/query?query=sentinel_model_loaded` — verify
   Prometheus has the metric (should be 1)
8. `docker compose down`

**Success criterion:** All 4 steps pass.

**If it fails:** Check `docker compose logs`. Common issues:
- GPU passthrough not configured (RTX 3070 needs `nvidia-docker2`)
- Volume mounts not finding checkpoint files
- mlops_config.json path not right inside the container

---

### C.6 Document the deployment process — 30 min

**File:** `ml/deploy/README.md` (new)

Contents:
- Prerequisites (Docker, nvidia-docker2, GPU)
- Quick start (`docker compose up -d`)
- Configuration (mlops_config.json vs .env)
- Updating the active model (`set_active_checkpoint.py` + restart)
- Monitoring (Prometheus UI at :9090)
- Troubleshooting

**Success criterion:** A new operator can follow README.md to deploy.

---

### Phase C total: ~3 hr

---

## Phase D: Post-Run-13 Transition — DEFERRED

**Trigger:** Run 13 training completes (target: ~3 weeks from now).

**Steps (sketch — refine when Run 13 is ready):**

1. **Test that Run 12 still loads** with `trainer.py:CLASS_NAMES = 9 entries`
   - Per `2026-06-15_ml_q4_proposal_mlops_current_state_findings.md` §3.2
   - Likely fails; need to pick Option A (keep CLASS_NAMES with 10 entries,
     mark GasException as @deprecated)
2. **Update `mlops_config.json`** with Run 13 paths:
   - `checkpoint`: `ml/checkpoints/GCB-P1-Run13-v4bcccme-20260630_FINAL.pt`
   - `num_classes`: 9
3. **Re-run `set_active_checkpoint.py`** for Run 13
4. **Re-build drift baseline** with Run 13 (B.4 redo)
5. **Re-run smoke tests** (A.5 redo)
6. **Promote Run 13 to Staging** in MLflow (`promote_model.py`)
7. **Re-validate `preprocess.py` comment** "10-class" → "9-class" (no more
   conditional)

**Effort estimate:** ~1 day (mostly testing + config updates).

**Why deferred:** Run 13 hasn't trained yet. Specifying the exact transition
now is premature.

---

## Files Created / Modified / Moved

### Created (new)
- `ml/mlops_config.json`
- `ml/scripts/set_active_checkpoint.py`
- `ml/scripts/build_warmup_baseline.py` (synthetic warmup generator for B.4)
- `ml/tests/inference/test_drift_detector.py` (optional)
- `ml/tests/inference/test_api_config.py` (optional)
- `ml/deploy/Dockerfile.inference`
- `ml/deploy/docker-compose.yml`
- `ml/deploy/prometheus.yml`
- `ml/deploy/.env.example`
- `ml/deploy/README.md`

### Modified
- `ml/src/inference/api.py` — config loader + env-var fallbacks (B.2)
- `ml/src/inference/drift_detector.py` — baseline validation (A.1) + dump_warmup_to_jsonl (B.4)
- `ml/src/inference/api.py:7, 166` — comment updates (A.2)
- `ml/src/inference/preprocess.py:39, 55` — comment updates (A.2)
- `ml/mlops_config.json` — updated in B.1, B.4, D.2

### Moved
- `ml/calibration/run12/temperatures_run12.json` → `docs/.bin/2026-06-15_ml_q4_proposal_mlops_duplicate_calibration_cleanup/` (A.3)
- `ml/checkpoints/GCB-P1-Run{6,7,8,9,10,11}-*_smoke*.pt` → `ml/checkpoints/_archive/` (A.4)

### Not modified (deliberately)
- `ml/src/inference/predictor.py` — works as designed
- `ml/src/inference/cache.py` — works as designed
- `ml/scripts/promote_model.py` — works as designed
- `ml/scripts/compute_drift_baseline.py` — works as designed

---

## Effort Summary

| Phase | Effort | Status |
|---|---|---|
| A.1 Fix drift detector | 30 min | Ready |
| A.2 Update stale comments | 5 min | Ready |
| A.3 Resolve duplicate calibration | 5 min | Ready |
| A.4 Decide DVC policy | 15 min | Ready (judgment call) |
| A.5 Smoke test Run 12 loads | 30 min | Ready |
| **Phase A total** | **1.5 hr** | **Ready to start** |
| B.1 Create mlops_config.json | 15 min | Blocked on A |
| B.2 Update api.py to read config | 30 min | Blocked on A |
| B.3 Create set_active_checkpoint.py | 30 min | Blocked on A |
| B.4 Build real drift baseline | 1 hr | Blocked on B.1, B.2 |
| B.5 Add inference tests (optional) | 1 hr | Optional |
| **Phase B total** | **2-3 hr** | **Blocked on A** |
| C.1 Create Dockerfile | 30 min | Blocked on B |
| C.2 Create docker-compose.yml | 30 min | Blocked on B |
| C.3 Create prometheus.yml | 15 min | Blocked on B |
| C.4 Create .env.example | 15 min | Blocked on B |
| C.5 E2E smoke test | 30 min | Blocked on C.1-C.4 |
| C.6 Document deployment | 30 min | Blocked on C.5 |
| **Phase C total** | **2.5-3 hr** | **Blocked on B** |
| Phase D (Run 13) | ~1 day | DEFERRED |
| **GRAND TOTAL Q4** | **~7.5 hr active** | |

---

## Recommended Execution Order

**Week 1:** Phase A (1.5 hr) + start Phase B (2 hr) = 3.5 hr
**Week 2:** Finish Phase B (1 hr) + start Phase C (2 hr) = 3 hr
**Week 3:** Finish Phase C (1 hr) + buffer = 1 hr
**After Run 13:** Phase D (1 day)

---

## References

- **File 2 (state):** `2026-06-15_ml_q4_proposal_mlops_current_state_findings.md`
- **File 3 (design):** `2026-06-15_ml_q4_proposal_mlops_redesign_proposal.md`
- **File 5 (risks):** `2026-06-15_ml_q4_proposal_mlops_risks_dependencies.md`
- **Audit (prior):** `ml/MLOPS_STATE_AND_REDESIGN_2026-06-14.md`
