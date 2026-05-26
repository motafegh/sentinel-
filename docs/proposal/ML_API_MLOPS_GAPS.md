# ML API & MLOps — Gap Analysis and Plan

**Date:** 2026-05-27  
**Context:** Run 4 complete (F1=0.3362, ep32); three-tier output designed; pipeline verified.  
**Scope:** What needs doing in `ml/src/inference/api.py`, MLOps tooling, and operations before the system is production-ready.

---

## 1. Uncommitted Work (commit before anything else)

The following changes are clean, verified (compare_pipelines FAIL=0), and should be committed now:

### Commit 1 — Pipeline alignment fixes + Run 4 verification

**Modified files (inference pipeline fixes):**
- `ml/src/inference/predictor.py` — `_score_windowed()` batched `[1,4,512]`; `predict()` delegates to windowed; `_warmup()` 3-node graph
- `ml/src/inference/preprocess.py` — window advance fix (`_CONTENT_CAP - stride`); solc binary detection; tokenizer identity comment
- `ml/src/preprocessing/graph_schema.py` — v7→v8 comment fixes (4 stale comments)
- `ml/scripts/tune_threshold.py` — minor fix
- `docs/proposal/EXECUTION_PLAN.md` — Run 4 status + GATE-GCB-4 section
- `docs/changes/INDEX.md` — three new entries (2026-05-26 session, agents plan v2, proposal)

**New files (compare_pipelines tooling + docs):**
- `ml/scripts/compare_pipelines.py` — full 60+ check pipeline parity verifier
- `ml/scripts/full_graph_diagnostic.py` — diagnostic script
- `ml/scripts/show_full_input.py` — input inspection script
- `docs/changes/2026-05-26-pipeline-alignment-and-inference-evaluation.md`
- `docs/ml/gcb-p1-run4-final-analysis.md`
- `docs/proposal/2026-05-27-three-tier-inference-output.md`
- `docs/AGENTS_PLAN_V2.md`
- `docs/proposal/AGENTS_MODULE_PROPOSAL.md`

**Do NOT commit:**
- `sentinel_pipeline_comparison_audit.md` — working file in root dir; move to `docs/` or discard

---

## 2. ML Inference API (`ml/src/inference/api.py`) — What Needs Updating

### 2.1 Schema is Stale (highest priority)

Current `PredictResponse` schema:

```python
class PredictResponse(BaseModel):
    label:           str                     # "vulnerable" | "safe"  ← stale
    vulnerabilities: list[VulnerabilityResult]  # above-threshold only
    thresholds:      list[float]
    truncated:       bool
    windows_used:    int
    num_nodes:       int
    num_edges:       int
```

**After three-tier predictor.py update, `PredictResponse` must add:**

```python
class PredictResponse(BaseModel):
    # Updated label values
    label: str  # "safe" | "suspicious" | "confirmed_vulnerable"

    # NEW — full probability vector, always present
    probabilities: dict[str, float]

    # NEW — tiered findings
    confirmed:  list[VulnerabilityResult]   # prob >= 0.55
    suspicious: list[VulnerabilityResult]   # 0.25 <= prob < 0.55

    # Legacy — preserved for backward compat, contains confirmed only
    vulnerabilities: list[VulnerabilityResult]

    # NEW — tier boundaries used
    tier_thresholds: dict[str, float]   # {"confirmed": 0.55, "suspicious": 0.25, ...}

    thresholds:   list[float]           # per-class tuned thresholds (unchanged)
    truncated:    bool
    windows_used: int
    num_nodes:    int
    num_edges:    int
```

**`VulnerabilityResult` needs a `tier` field:**
```python
class VulnerabilityResult(BaseModel):
    vulnerability_class: str
    probability:         float
    tier:                str | None = None  # "CONFIRMED" | "SUSPICIOUS" (None in legacy field)
```

### 2.2 Checkpoint Pointer is Wrong

Current default in `api.py`:
```python
CHECKPOINT: str = os.getenv(
    "SENTINEL_CHECKPOINT",
    "ml/checkpoints/multilabel_crossattn_v2_best.pt",   # ← stale
)
```

This is an old v2 checkpoint. The current best is Run 4:
```python
CHECKPOINT: str = os.getenv(
    "SENTINEL_CHECKPOINT",
    "ml/checkpoints/GCB-P1-Run4-no-asl-pw_best.pt",
)
```

This is a one-line change but failing to update it means anyone starting the API with defaults
loads a significantly weaker model (v2 tuned F1≈0.48 but behavioral test ❌ vs Run 4 F1=0.3362
but correct architecture + verified pipeline).

### 2.3 `/health` Should Report Tier Thresholds

Current `/health` output:
```json
{"status": "ok", "predictor_loaded": true, "checkpoint": "...", "architecture": "...", "thresholds_loaded": true}
```

Should add:
```json
{
  "tier_thresholds": {"confirmed": 0.55, "suspicious": 0.25, "noteworthy": 0.10},
  "model_f1_val": 0.3362,
  "model_epoch": 32
}
```

The `f1` and `epoch` are read from the checkpoint's saved `config` dict (already loaded at startup).
This gives monitoring and debugging immediate confirmation the right checkpoint is loaded.

### 2.4 Drift Detector Is Tracking Wrong Features

Current drift detection tracks:
```python
drift_detector.update_stats({
    "num_nodes": float(result["num_nodes"]),
    "num_edges": float(result["num_edges"]),
})
```

These are useful but coarse. After three-tier output, the more informative drift features are:
- `max_confirmed_prob` — average max probability in CONFIRMED tier (if this drops, model confidence is drifting)
- `suspicious_count` — average number of SUSPICIOUS classes per contract (distribution shift signal)
- `num_nodes`, `num_edges` — keep existing

This is a low-priority improvement; the existing drift features are better than nothing.

### 2.5 `/predict` Response Mapping Gap

The endpoint maps `result["vulnerabilities"]` directly. After three-tier output, it needs to also map `confirmed`, `suspicious`, and `probabilities`. The `result` dict is the predictor's `_format_result()` output, so the endpoint just needs to pass the new fields through.

```python
# Current (stale after predictor update)
return PredictResponse(
    label=result["label"],
    vulnerabilities=[VulnerabilityResult(...) for v in result["vulnerabilities"]],
    thresholds=result["thresholds"],
    ...
)

# Updated
return PredictResponse(
    label=result["label"],                           # now "safe"|"suspicious"|"confirmed_vulnerable"
    probabilities=result["probabilities"],
    confirmed=[VulnerabilityResult(**v) for v in result.get("confirmed", [])],
    suspicious=[VulnerabilityResult(**v) for v in result.get("suspicious", [])],
    vulnerabilities=[VulnerabilityResult(**v) for v in result.get("vulnerabilities", [])],
    tier_thresholds=result.get("tier_thresholds", {}),
    thresholds=result["thresholds"],
    truncated=result["truncated"],
    windows_used=result.get("windows_used", 1),
    num_nodes=result["num_nodes"],
    num_edges=result["num_edges"],
)
```

---

## 3. MLOps Gaps — What's Missing vs the M3 Plan

### 3.1 `exercise_drift_detector.py` — Does Not Exist

**Status:** Planned in M3 plan §2.1, never built.

**What it should do:**
- Generate 600 synthetic feature vectors matching the Run 4 training distribution
- Assert zero drift alerts during warm-up phase (first 500 vectors)
- Inject a clear distribution shift (multiply `num_nodes` by 3× for last 100 vectors)
- Assert at least one KS alert fires within 50 post-shift requests
- Exit 0 on PASS, 1 on FAIL — wire into CI

**Why it matters:** `drift_detector.py` is exercised in production but never tested in CI.
A bug in the KS test or rolling buffer could go undetected until a real distribution shift
occurs in production.

**File:** `ml/scripts/exercise_drift_detector.py`  
**Effort:** ~2 hours

### 3.2 `promote_model.py` — Missing Gates

**Status:** Script exists, works. Missing two gates from the M3 plan.

**Missing: `--require-baseline` flag**
```python
# Should fail if:
# - drift_baseline.json doesn't exist at the given path
# - drift_baseline.json was generated from --source training (check metadata key "source")
# This prevents promoting a model to Production without a proper warm-up baseline
```

**Missing: Previous-Production F1 comparison**
Currently `promote_model.py --stage Production` only checks that the new model's F1 exceeds
a user-supplied `--val-f1-macro` value. It should also:
1. Query the MLflow registry for the current Production model's F1
2. Fail if new model's F1 does not exceed previous Production model's F1

This prevents accidental downgrade when promoting a model that passed the absolute gate
but is worse than what's currently in production.

**Effort:** ~3 hours

### 3.3 Run 4 Not Registered in MLflow

The current best checkpoint (`GCB-P1-Run4-no-asl-pw_best.pt`, F1=0.3362) has never been
registered in the MLflow Model Registry via `promote_model.py`. It should be at minimum
in Staging.

**Command to run once promote_model.py gates are added:**
```bash
cd ~/projects/sentinel
source ml/.venv/bin/activate
MLFLOW_TRACKING_URI=sqlite:///mlruns.db python ml/scripts/promote_model.py \
    --checkpoint ml/checkpoints/GCB-P1-Run4-no-asl-pw_best.pt \
    --stage Staging \
    --val-f1-macro 0.3362 \
    --note "Run4 ep32: GCB+prefix+8L GNN, no-ASL-pw, F1=0.3362 (all-time best). Pipeline verified FAIL=0."
```

Note: Do NOT promote to Production yet — three-tier output needs to be implemented first,
then api.py updated, then an integration smoke test before Production.

### 3.4 Threshold JSON Not Registered with Checkpoint

The per-class threshold JSON file (`GCB-P1-Run4-no-asl-pw_thresholds.json`) is produced by
`tune_threshold.py` alongside the checkpoint. The `promote_model.py` script logs the checkpoint
file as an MLflow artifact but does NOT log the companion threshold JSON.

When someone deploys from the MLflow registry, they get the weights but not the thresholds —
which means inference falls back to uniform 0.5 thresholds (silent degradation).

**Fix:** In `promote_model.py`, also log `{checkpoint_stem}_thresholds.json` as an artifact
if it exists alongside the checkpoint.

### 3.5 Drift Baseline — Blocked on Production Traffic

The M3 plan correctly identifies that `compute_drift_baseline.py --source warmup` requires
real production traffic that doesn't exist yet. This is blocked until M6 (API gateway)
ships and starts receiving real requests.

**Current workaround:** The drift detector starts in warm-up mode and suppresses all alerts
for the first `N_WARMUP` requests. This is correct behavior.

**Action needed (none until M6):** Once M6 ships, run:
```bash
python ml/scripts/compute_drift_baseline.py \
    --source warmup \
    --output ml/data/drift_baseline.json
```
Then restart the inference API (which will load the baseline on startup).

### 3.6 DVC — Data Not Tracked

`ml/checkpoints.dvc` exists (checkpoints tracked). `ml/data/` (41K graphs, 44K tokens, 2.2GB)
is NOT under DVC. This means the training data is not reproducible from the repository.

**Decision required:** Is a DVC remote available? Options:
- Local (NAS/external drive) — simplest, no cloud cost
- S3/GCS — standard, costs money
- Defer — accept non-reproducibility for now

**Recommendation:** Defer until a second machine needs the data. Document the decision in
`docs/Project-Spec/SENTINEL-M3-MLOPS.md`.

---

## 4. Implementation Order

### Immediate (do these now, unblocked):

| # | Task | File | Effort | Depends on |
|---|------|------|--------|-----------|
| 1 | Commit pipeline alignment work | git | 10 min | nothing |
| 2 | Update `CHECKPOINT` default in api.py | `ml/src/inference/api.py:64` | 1 min | nothing |
| 3 | Implement three-tier output in predictor.py | `ml/src/inference/predictor.py` | 2–3 hrs | nothing |
| 4 | Update `PredictResponse` schema in api.py | `ml/src/inference/api.py` | 1–2 hrs | task 3 |
| 5 | Update `/predict` endpoint mapping in api.py | same file | 30 min | task 4 |
| 6 | Update `/health` to report tier thresholds + F1 | same file | 30 min | task 3 |
| 7 | Add `--require-baseline` to promote_model.py | `ml/scripts/promote_model.py` | 1–2 hrs | nothing |
| 8 | Add previous-Production F1 gate to promote_model.py | same file | 1 hr | task 7 |
| 9 | Log threshold JSON artifact in promote_model.py | same file | 30 min | nothing |
| 10 | Write `exercise_drift_detector.py` | `ml/scripts/` | 2 hrs | nothing |
| 11 | Register Run 4 in MLflow Staging | CLI command | 5 min | task 7 |

### Blocked (need M6 or other prerequisites):

| # | Task | Blocked on |
|---|------|-----------|
| B1 | Drift baseline from warm-up traffic | M6 API gateway running + receiving traffic |
| B2 | Promote Run 4 to Production | Three-tier output implemented + api.py updated + integration smoke test |
| B3 | DVC remote for ml/data/ | Decision on remote storage provider |

---

## 5. Files Changed Summary

| File | Change type | Priority |
|------|-------------|----------|
| `ml/src/inference/api.py` | Schema update (PredictResponse + label values + probabilities + tier fields) + checkpoint default + health endpoint | HIGH — blocks agents integration |
| `ml/src/inference/predictor.py` | Three-tier `_format_result()` | HIGH — prerequisite for api.py |
| `ml/scripts/promote_model.py` | `--require-baseline` flag + prev-Production F1 gate + threshold JSON artifact | MEDIUM — process improvement |
| `ml/scripts/exercise_drift_detector.py` | NEW — CI smoke test for drift detector | MEDIUM — CI quality |
| `ml/data/drift_baseline.json` | Generated by compute_drift_baseline.py when traffic arrives | LOW — blocked on M6 |
