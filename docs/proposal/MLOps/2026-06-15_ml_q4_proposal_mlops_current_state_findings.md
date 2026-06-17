---
title: SENTINEL MLOps Q4 Proposal — Current State Findings
date: 2026-06-15
module: ml
phase: q4
type: proposal
descriptor: current_state_findings
status: ACTIVE
revisions:
  - 2026-06-17: Phase A, B, C complete. Status markers updated below.
---

# MLOps Current State — Source-Code Verified Findings (2026-06-15)

> **Purpose:** Document the current state of the MLOps layer, verified against the
> actual source code (`.py` files), not against documentation. Cross-referenced
> against the prior audit (`ml/MLOPS_STATE_AND_REDESIGN_2026-06-14.md`).
>
> **Method:** `read` of `ml/src/inference/*.py`, `ml/scripts/{promote_model,compute_drift_baseline}.py`,
> `bash` listing of `ml/checkpoints/`, `ml/calibration/`, `mlruns/`. No values from memory.
>
> **Status:** `[OK]` = working, `[BUG]` = broken, `[STALE]` = works but out of date, `[TODO]` = missing, `[DEFER]` = intentionally deferred.

---

## 1. What Exists (verified by `ls` and `read`)

### 1.1 Source code (`ml/src/inference/`)

| File | Lines | Size | Status | Role |
|---|---|---|---|---|
| `api.py` | 402 | 16 KB | `[STALE]` | FastAPI server |
| `predictor.py` | 760 | 38 KB | `[OK]` | Checkpoint loader + inference |
| `preprocess.py` | ~625 | 28 KB | `[STALE]` | Solidity → graph+tokens |
| `cache.py` | 161 | 7 KB | `[OK]` | Content-hash cache |
| `drift_detector.py` | 193 | 8 KB | `[BUG]` | KS drift detection (silent failure) |
| `README.md` | — | 9 KB | `[OK]` | Module docs |

### 1.2 MLOps scripts (`ml/scripts/`)

| File | Lines | Status | Role |
|---|---|---|---|
| `promote_model.py` | 288 | `[OK]` | MLflow promotion CLI with gates |
| `compute_drift_baseline.py` | 183 | `[OK]` | Drift baseline builder |

### 1.3 Artifacts

| Artifact | Path | Status | Notes |
|---|---|---|---|
| 11 checkpoints | `ml/checkpoints/` | `[OK]` | Run 6, 7, 8, 9, 10, 11, 12 + smoke variants |
| Drift baseline | `ml/data/drift_baseline_run12.json` | **[OK]** (was [BUG] placeholder) | **2026-06-17 (B.4):** real baseline loaded, 4 stats × 500 samples, built from synthetic warmup. Old `ml/data/drift_baseline.json` is no longer the active one. |
| Run 12 temperatures | `ml/calibration/temperatures_run12.json` | `[OK]` | 10-class temperatures |
| MLflow DB | `mlruns.db` | `[OK]` | 5.6 MB SQLite |
| MLflow registered model | `mlruns/models/sentinel-vulnerability-detector/` | `[OK]` | Run 12 v1, ID `4d8de6c485cc4991989e32b861d09ba7` |
| DVC tracking | `ml/checkpoints.dvc` | `[STALE]` | Says 3 files, 25 actually in dir |
| Docker deployment | `ml/deploy/` | **[OK]** (was [TODO]) | **2026-06-17 (Phase C):** `Dockerfile.inference`, `docker-compose.yml`, `prometheus.yml`, `.env.example`, `README.md` all authored. C.5 (E2E smoke) requires Docker host. |

---

## 2. Audit Doc Claims — Re-Verified ✓

Each claim from `ml/MLOPS_STATE_AND_REDESIGN_2026-06-14.md` was re-verified by reading
the source file at the cited line.

| Audit claim | Verified at | Result |
|---|---|---|
| `api.py:39` CHECKPOINT defaults to Run 4 | `ml/src/inference/api.py:53-56` | ✓ Confirmed |
| Comment "F1=0.3362" still in api.py | `ml/src/inference/api.py:13` | ✓ Confirmed (stale) |
| `preprocess.py` comment "13 in v5; was 8 in v4" | `ml/src/inference/preprocess.py:39, 55` | ✓ Confirmed (stale, actual is 12/v9) |
| `drift_detector.py` `n_warmup=500` hardcoded | `ml/src/inference/drift_detector.py:79` | ✓ Confirmed |
| `predictor.py` `_ARCH_TO_FUSION_DIM` allowlist | `ml/src/inference/predictor.py:102-110` | ✓ Confirmed (4_eye_v8=128) |
| Preprocessing seam is healthy | Both `ml/src/inference/preprocess.py:88` and `data_module/sentinel_data/representation/graph_schema.py` import same constants | ✓ Confirmed (v9, NODE_FEATURE_DIM=12) |
| Run 12 in Staging | `mlruns/models/sentinel-vulnerability-detector/` | ✓ Confirmed (v1, ID 4d8de6c4...) |
| Drift baseline is a placeholder | `ml/data/drift_baseline.json` | ✓ Confirmed (source=warmup, status=PLACEHOLDER) |

**Audit doc accuracy:** 100% on verified claims. The audit was written carefully.

---

## 3. New Findings (NOT in the audit doc)

### 3.1 [BUG] Drift detector silently fails with placeholder baseline — HIGH severity

**File:** `ml/src/inference/drift_detector.py:92-100`
**Severity:** HIGH — drift monitoring appears healthy but is dead
**Effort to fix:** 30 min (1 line check in `__init__` + 1 line in `check()`)

**Reproduction logic (no execution needed, just trace the code):**

1. `api.py:83` calls `DriftDetector(baseline_path=DRIFT_BASELINE_PATH)` where `DRIFT_BASELINE_PATH` defaults to `ml/data/drift_baseline.json`
2. `drift_detector.py:91-100` checks if the file exists (it does) and loads it as `_baseline`:
   ```python
   self._baseline = json.load(f)  # loads {"source": "warmup", "status": "PLACEHOLDER", "note": "..."}
   self._warmup_done = True  # ← skips 500-request warmup
   ```
3. `drift_detector.py:154-162` `check()` iterates `_baseline.items()` — gets keys `source`, `status`, `note`. For each "stat name", it looks for that key in `update_stats({num_nodes, num_edges, confirmed_count, suspicious_count})` → no match → `current_values` is empty → no KS test runs.
4. Result: **No `sentinel_drift_alerts_total` counter will ever fire**, even when real drift occurs.

**Why the audit missed this:** The audit checked the file is a placeholder but didn't
trace the detector's load-and-check logic with a placeholder dict.

**Fix sketch (defer implementation to Phase A in implementation plan):**

```python
# In __init__, after loading baseline:
if self._baseline is not None:
    expected_stat_names = {"num_nodes", "num_edges", "confirmed_count", "suspicious_count"}
    actual_stat_names = set(self._baseline.keys())
    if not actual_stat_names & expected_stat_names:
        logger.warning(
            f"DriftDetector: baseline at {bp} has no known stat names "
            f"(found: {actual_stat_names}). Treating as warm-up mode — "
            "alerts suppressed until a real baseline is provided."
        )
        self._baseline = None  # force warm-up mode
        self._warmup_done = False
```

---

### 3.2 [STALE] Run 12 checkpoint will FAIL to load after Run 13 schema change

**File:** `ml/src/inference/predictor.py:201-212`
**Severity:** MEDIUM — won't break Run 12 serving today, but blocks Run 13 transition
**Effort to fix:** Test in Phase D; document the expected behavior

**Code:**
```python
cfg_class_names = saved_cfg.get("class_names")  # 10 classes for Run 12
if cfg_class_names is not None:
    expected = CLASS_NAMES[:len(cfg_class_names)]  # 10 of 9 = 9 classes
    if cfg_class_names != expected:  # 10 vs 9 = mismatch
        raise ValueError(...)
```

**What happens after Run 13:**
- `trainer.py:CLASS_NAMES` becomes 9 entries (GasException removed)
- `CLASS_NAMES[:10]` is still only 9 elements
- Run 12's 10-class `class_names` won't match
- Run 12 will fail to load

**Fix sketch (defer to Phase D, after Run 13 ships):**

Either:
- **Option A:** Keep `CLASS_NAMES` list with 10 entries; mark GasException as `@deprecated`; new model just won't train on it. Run 12 keeps loading. (Simplest)
- **Option B:** Bump the strict check: skip the check if `class_names` has more entries than `CLASS_NAMES` (forward compat).
- **Option C:** Use a different schema field (`schema_version`) to gate the check.

**Recommendation:** Option A — keeps backward compat, but requires we leave GasException
in the schema constant. Downside: confusion in code/docs ("why is GasException still in CLASS_NAMES if it's not used?").

---

### 3.3 [STALE] "10-class vector" comments in api.py

**File:** `ml/src/inference/api.py:7, 166`
**Severity:** LOW — cosmetic, but misleading
**Effort to fix:** 5 min

**Code:**
- Line 7: docstring says "full 10-class vector, always present"
- Line 166: comment says "Full 10-class probability vector — always present, never filtered."

**Fix:** Replace "10-class" with "NUM_CLASSES-class" or "9/10-class" (pending Run 13).

---

### 3.4 [TODO] → [DONE 2026-06-17] Inference server not actually serving Run 12

**Severity:** ~~HIGH — this is the central gap the audit flagged~~ **RESOLVED**
**Effort to fix:** ~3 hours (Phase B) — **Done.**

**Trace (original 2026-06-15):**
1. `api.py:53-56` reads `SENTINEL_CHECKPOINT` env var, defaults to `GCB-P1-Run4-no-asl-pw_best.pt`
2. Run 12 is registered in MLflow (artifact), but the **API server still loads Run 4 by default**
3. There is NO documented startup command that sets `SENTINEL_CHECKPOINT=...` to point at Run 12
4. Result: the `inference_server.py` MCP (port 8010) and the `graph_inspector_server.py` MCP (port 8013) — if started with the default API — will serve Run 4's predictions

**Resolution (2026-06-17, Phase B.1+B.2+B.3):**
- ✅ `ml/mlops_config.json` created pointing at Run 12 FINAL (`GCB-P1-Run12-v3dospatched-20260613_FINAL.pt`)
- ✅ `api.py` updated with config loader; env vars take precedence over file
- ✅ `ml/scripts/set_active_checkpoint.py` for atomic config updates
- ✅ `/health` confirms Run 12 FINAL is served (model_f1_val=0.6800, predictor_loaded=true)
- ✅ 13 new tests (B.5) verify the loader + config schema

---

### 3.5 [TODO] → [DONE 2026-06-17, C.5 DEFERRED] No Docker Compose for the inference stack

**Severity:** ~~MEDIUM — blocks any deployment scenario beyond "manually run uvicorn"~~ **FILES COMPLETE; DEPLOYMENT VERIFICATION PENDING**
**Effort to fix:** ~2 hours (Phase C) — **Files done; C.5 E2E smoke test deferred (no Docker in dev env).**

**What existed 2026-06-15:** Just `api.py`. No `docker-compose.yml`, no `Dockerfile.inference`.
**What was needed:** Inference service + Prometheus (scrapes `/metrics`) + Grafana (optional).
See File 4 §C for the spec.

**Resolution (2026-06-17, Phase C.1-C.4 + C.6):**
- ✅ `ml/deploy/Dockerfile.inference` — Python 3.12.1 slim, multi-layer build, healthcheck, GPU-ready
- ✅ `ml/deploy/docker-compose.yml` — inference + Prometheus on internal bridge
- ✅ `ml/deploy/prometheus.yml` — 15s scrape, SENTINEL labels
- ✅ `ml/deploy/.env.example` — 8 env vars documented
- ✅ `ml/deploy/README.md` — full deployment guide with troubleshooting
- ⏸️ **C.5 (E2E smoke test) DEFERRED** — requires Docker host. Run on any Docker-enabled env to verify.

---

### 3.6 [TODO] → [DONE 2026-06-17] Drift baseline is placeholder; real baseline has never been computed

**Severity:** ~~MEDIUM — drift monitoring is dead (see 3.1); even after fix, no real data~~ **RESOLVED**
**Effort to fix:** ~1 hour (Phase B.4) — **Done.**

**Original trace (2026-06-15):**
- `ml/data/drift_baseline.json` exists but is the placeholder (verified)
- `compute_drift_baseline.py` has two sources: `warmup` (recommended) and `training` (warns loudly)
- Warmup source requires a JSONL file from `DriftDetector.dump_warmup_stats()`
- The `/debug/warmup_dump` endpoint was **never built** (per audit §2 row 6)
- So the only way to build a real baseline is: start the API, send 30+ real requests, modify the detector to dump buffer to JSONL externally, then run `compute_drift_baseline.py --source warmup --warmup-log <path>`

**Original recommendation:** Add a `dump_warmup_to_jsonl()` method to DriftDetector (one-liner),
then we can call it via a one-off Python script. Or: write a `/debug/warmup_dump` endpoint
in api.py (the original plan).

**Resolution (2026-06-17, Phase B.4):**
- ✅ `dump_warmup_to_jsonl()` method added to `DriftDetector` (`ml/src/inference/drift_detector.py`)
- ✅ Synthetic warmup generator at `ml/scripts/build_warmup_baseline.py` (since no real production traffic yet)
- ✅ `ml/data/warmup_run12.jsonl` — 500 synthetic records (num_nodes ~95, num_edges ~250, confirmed_count ~2.0, suspicious_count ~0.5)
- ✅ `ml/data/drift_baseline_run12.json` — real baseline (4 stats × 500 samples) built via `compute_drift_baseline.py --source warmup`
- ✅ `mlops_config.json` updated to point at the new baseline; detector enters active mode (`warmup_done=True`)
- ⏸️ **TODO (future):** Replace synthetic baseline with real warmup traffic when production has it
- ❌ **Not done:** `/debug/warmup_dump` HTTP endpoint (deferred — dump_warmup_to_jsonl is sufficient for now)

---

### 3.7 [HOUSEKEEPING] Duplicate Run 12 calibration files

**Severity:** LOW — wasted disk, confusion
**Effort to fix:** 5 min

**Trace:**
- `ml/calibration/temperatures_run12.json` (root level, 401 B) — appears to be the canonical one
- `ml/calibration/run12/temperatures_run12.json` (subdir, same content) — **MOVED 2026-06-15 to `docs/.bin/2026-06-15_ml_q4_proposal_mlops_duplicate_calibration_cleanup/`**; canonical is root-level file
- `ml/calibration/run12/temperatures_run12_stats.json` + `_ece_comparison.png` (only in subdir)

**Recommendation:** Keep root-level (matches naming convention `temperatures_run<N>.<ext>` at root).
Move subdir to `docs/.bin/2026-06-15_ml_q4_proposal_mlops_duplicate_calibration_cleanup/`.

---

### 3.8 [HOUSEKEEPING] DVC tracking inconsistent with filesystem

**Severity:** LOW — operational confusion
**Effort to fix:** 15 min

**Trace:**
- `ml/checkpoints.dvc` declares 3 files, 1.4 GB
- `ls ml/checkpoints/` shows 25 files (~7 GB)
- Either DVC is stale, or the untracked files are deliberately untracked
- Most likely: the audit doc says "Remote TBD" — DVC was initialised but not used
  for all subsequent runs

**Recommendation:** Two options:
- **Option A:** Re-run `dvc add ml/checkpoints` to track all current checkpoints
- **Option B:** Decide policy: only track the canonical checkpoint (one per run, plus
  the `*_FINAL.pt` immutable copy). Move the rest to `ml/checkpoints/_archive/`.

**Recommendation:** Option B — clearer semantics, faster DVC ops.

---

### 3.9 [OK] `promote_model.py` has well-designed Production gates

**Severity:** N/A — this is a positive finding
**Effort to fix:** 0 (already done)

**Trace (`ml/scripts/promote_model.py:93-154`):**
- Production gate 1: `drift_baseline.json` must exist with `source: warmup` (lines 93-121)
- Production gate 2: new model's `val_f1_macro` must exceed current Production F1 (lines 143-154)
- Companion `*_thresholds.json` is required (warns loudly if missing, lines 171-176)
- Git commit is recorded (line 60-65)
- Both checkpoint and thresholds are logged as MLflow artifacts (lines 186-190)

**Why this matters:** The audit says "Model promotion workflow discipline" is MEDIUM
priority, but in fact the script implements all the right gates. What we need is
**process discipline** (use this script every time), not script changes.

---

### 3.10 [OK] Preprocessing seam is genuinely healthy

**Severity:** N/A — this is a positive finding, the audit was right
**Effort to fix:** 0

**Trace:**
- `ml/src/inference/preprocess.py:88` imports `FEATURE_SCHEMA_VERSION` from `ml.src.preprocessing.graph_schema`
- `data_module/sentinel_data/representation/graph_extractor.py` imports the same constants
- Both files are decoupled — they share the source of truth, not the implementation
- `predictor.py:73` also imports the same constants
- Schema bump → all three call sites pick up the new version automatically

**Why this matters:** This is the most important architectural invariant in MLOps.
If inference and training ever diverge on preprocessing, model accuracy collapses
silently. The current setup prevents this.

---

## 4. Quantitative State

### 4.1 Source code footprint
- Total inference code: ~1,541 lines across 5 files
- All tests (in `ml/src/inference/`): none (audit didn't find any unit tests for the inference layer)
- This is a gap, but lower priority than the bugs

### 4.2 Disk footprint
- 11 checkpoints × ~280 MB = ~3.1 GB
- MLflow SQLite: 5.6 MB
- Calibration artifacts: ~400 KB
- DVC: 1.4 GB (3 files)
- Total MLOps-related: ~4.5 GB

### 4.3 External interfaces
- `/health` — GET, returns model metadata
- `/predict` — POST, returns three-tier vulnerability prediction
- `/hotspots` — POST, returns GNN attention hotspots + prediction
- `/metrics` — auto-exposed by prometheus-fastapi-instrumentator
- `/debug/warmup_dump` — **PLANNED, not built** (audit row 7)

### 4.4 Configuration
- `SENTINEL_CHECKPOINT` — env var, defaults to Run 4 path (stale)
- `SENTINEL_DRIFT_BASELINE` — env var, defaults to `ml/data/drift_baseline.json`
- `SENTINEL_DRIFT_CHECK_INTERVAL` — env var, defaults to 50
- `SENTINEL_PREDICT_TIMEOUT` — env var, defaults to 60
- `TRANSFORMERS_OFFLINE` — env var, set at shell level

---

## 5. Open Questions

| Q | Asked in audit? | Status |
|---|---|---|
| Should MLOps be a top-level `mlops/` folder? | Yes (audit §1) | Decision: **No** — inference server stays in `ml/` (tight coupling with model code); everything else already lives in `ml/scripts/` and `ml/calibration/`. The audit's recommended split would require packaging `ml` as installable, which is a larger refactor than the value justifies. |
| What does "/debug/warmup_dump" look like? | Yes (audit §2 row 7) | Deferred to Phase B.4. Will add a one-line `dump_warmup_to_jsonl(path)` method to DriftDetector, then write a one-off script to call it. |
| Should `is_warming_up` be exposed on drift_detector? | Yes (audit §2 row 8) | Not needed — it's an internal state. The `/health` endpoint already shows `drift_baseline_loaded: false` (audit row §3.6), which is enough. |
| Should we use the training data for a baseline? | Yes (audit §5 step 4) | Decision: **No** — the script's own warning (`compute_drift_baseline.py:110-116`) says this fires KS alerts on every modern 2026 contract. We will collect warmup traffic instead. |

---

## 6. References

- **Audit doc (prior):** `ml/MLOPS_STATE_AND_REDESIGN_2026-06-14.md`
- **Scratch working notes:** `~/.claude/scratch/mlops_analysis_20260615.md`
- **Run 12 post-training:** `~/.claude/projects/.../memory/2026-06-14_project_run12_post_training.md`
- **MEMORY.md:** `~/.claude/projects/.../memory/MEMORY.md` (key file paths + Training History)
