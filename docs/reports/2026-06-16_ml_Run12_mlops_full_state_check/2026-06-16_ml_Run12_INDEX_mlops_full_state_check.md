---
title: SENTINEL MLOps Full State Check — Run 12
date: 2026-06-16
module: ml
run: Run12
what: INDEX
descriptor: mlops_full_state_check
status: ACTIVE
---

# MLOps Full State Check — Run 12 (2026-06-16)

> **Method:** Source-code-first verification of `ml/src/inference/`, `ml/scripts/`, artifacts, DVC, MLflow, and tests. Docs used only for cross-reference.
>
> **Scratch working notes:** `~/.claude/scratch/mlops_full_state_check_20260616.md`

---

## TL;DR

- **Phase A is done but uncommitted.** Drift-detector silent-failure fix verified; stale comments cleaned; duplicate calibration archived; checkpoints canonicalized.
- **Phase B has not started.** No `mlops_config.json`; API still defaults to a Run 4 checkpoint that no longer exists.
- **🔴 New critical bug:** FastAPI `PredictResponse.tier_thresholds` declares `dict[str, float]`, but `predictor.py` returns `confirmed` thresholds as a list. `/predict` and `/hotspots` will HTTP 500 when serving Run 12 with per-class thresholds.
- **Production promotion is still blocked** (placeholder drift baseline + the new schema mismatch).

---

## What Was Checked

| Area | Files / Commands | Result |
|---|---|---|
| Inference source | `ml/src/inference/{api,predictor,preprocess,cache,drift_detector}.py` | Read line-by-line |
| MLOps scripts | `ml/scripts/{promote_model,compute_drift_baseline}.py` + smoke tests | Read + executed |
| Tests | `ml/tests/{test_api,test_predictor,test_drift_detector}.py` | Executed |
| Artifacts | `ml/checkpoints/`, `ml/calibration/`, `ml/data/drift_baseline.json` | Listed + inspected |
| DVC | `.dvc/config`, `ml/.venv/bin/dvc status` | Clean; only Run 12 FINAL tracked |
| MLflow | `mlruns.db` via `MlflowClient` | Run 12 v1 in Staging |
| Docker | `ml/docker/Dockerfile.slither`, `data_module/docker/Dockerfile.data` | No inference deploy artifacts |
| Downstream consumers | `agents/src/mcp/servers/inference_server.py` | Calls `localhost:8001/predict` |

---

## Verification Results

| Test | Command | Result |
|---|---|---|
| Drift detector fix | `ml/scripts/smoke/test_drift_detector_a1.py` | ✅ 5/5 pass |
| Run 12 loads | `ml/scripts/smoke/test_run12_loads_a5.py` | ✅ PASS |
| Drift unit tests | `pytest ml/tests/test_drift_detector.py` | ✅ 5 pass |
| Predictor unit tests | `pytest ml/tests/test_predictor.py` | ✅ 6 pass |
| API tests (Run 12 override) | `SENTINEL_CHECKPOINT=...FINAL.pt pytest ml/tests/test_api.py` | ❌ 4 fail — `tier_thresholds.confirmed` type mismatch |
| API health (default checkpoint) | `pytest ml/tests/test_api.py::test_health_returns_ok` | ❌ ERROR — default Run 4 checkpoint missing |

---

## Critical Finding: API Schema Mismatch

**File:** `ml/src/inference/api.py:185`
**Declared type:**
```python
tier_thresholds: dict[str, float]
```

**File:** `ml/src/inference/predictor.py:750-754`
**Actual value:**
```python
"tier_thresholds": {
    "confirmed":  self.thresholds.cpu().tolist(),  # list[float]
    "suspicious": susp_thr,
    "noteworthy": 0.10,
}
```

**Failure (from `pytest ml/tests/test_api.py`):**
```
tier_thresholds.confirmed
  Input should be a valid number [type=float_type,
  input_value=[0.4000000059604645, 0.5, ...],
  input_type=list]
```

**Impact:** Real HTTP requests to `/predict` and `/hotspots` will 500 once Run 12 is wired.

**Fix sketch:** Update `PredictResponse.tier_thresholds` to accept `dict[str, float | list[float]]`, or split the field. Update `test_api.py` accordingly.

---

## Other Findings

1. **API default checkpoint missing.** `api.py` defaults to `ml/checkpoints/GCB-P1-Run4-no-asl-pw_best.pt`, which was archived. Expected to be fixed in Phase B via `mlops_config.json`.
2. **No deployment artifacts.** Phase C not started: no `ml/deploy/`, no `Dockerfile.inference`, no `docker-compose.yml`.
3. **Placeholder drift baseline.** `ml/data/drift_baseline.json` is still a placeholder. Detector now correctly enters warm-up mode.
4. **Uncommitted working tree.** Phase A changes are not committed; `git status` shows many modified/deleted/untracked files.

---

## Recommendations

1. Fix the `tier_thresholds` Pydantic schema mismatch before Phase B.
2. Commit or stash Phase A changes.
3. Implement Phase B (`ml/mlops_config.json`, config loader, `set_active_checkpoint.py`).
4. Get `pytest ml/tests/test_api.py` green with the Run 12 checkpoint.
5. Build a real/synthetic warmup baseline (Phase B.4).
6. Proceed to Phase C Docker + Prometheus.

---

## References

- Q4 Proposal: `docs/proposal/MLOps/`
- Implementation plan: `docs/proposal/MLOps/2026-06-15_ml_q4_proposal_mlops_q4_implementation_plan.md`
- MEMORY.md: `~/.claude/projects/-home-motafeq-projects-sentinel/memory/MEMORY.md`
