---
title: SENTINEL MLOps Q4 Proposal — Risks and Dependencies
date: 2026-06-15
module: ml
phase: q4
type: proposal
descriptor: risks_dependencies
status: ACTIVE
---

# MLOps Q4 Risks and Dependencies (2026-06-15)

> **Purpose:** Document cross-module dependencies, internal/external risks, and
> decision gates for the MLOps Q4 proposal. This is the file the project manager
> and tech lead read first when something goes wrong.
>
> **Related:** File 3 (design) for the "what"; File 4 (plan) for the "when".

---

## 1. Dependency Map

### 1.1 Upstream (must be done before MLOps can serve a model)

```
[Run 13 training in ml/src/training/trainer.py]
            │
            ↓ produces
[Run 13 checkpoint at ml/checkpoints/*_FINAL.pt]
            │
            ↓
[Run 13 thresholds at ml/checkpoints/*_thresholds.json]
            │
            ↓
[MLOps Phase B.1-B.2 wires the new checkpoint into mlops_config.json]
            │
            ↓
[MLOps Phase B.4 builds a new drift baseline]
            │
            ↓ ready to serve
```

**For Q4 (Run 12):** No upstream dependency. Run 12 checkpoint + thresholds already exist.
**For Phase D (Run 13):** Must wait for `docs/plans/2026-06-14_Run13_4_fixes_preparation.md` to complete (5 fixes + v4 export + retrain).

### 1.2 Downstream (MLOps changes block these)

```
[MLOps Phase A complete]
            │
            ├──→ [agents/src/mcp/servers/inference_server.py] can use the new api.py with confidence (D5 fix means no silent drift failure)
            │
            ├──→ [contracts/test/AuditRegistry.t.sol] can run integration tests against a live API
            │
            └──→ [CI pipeline for inference server (Phase B.5)] is unblocked
```

**For Q4 (Run 12):**
- `agents/src/mcp/servers/inference_server.py` calls `http://localhost:8001/predict` and
  `http://localhost:8001/hotspots` — once Run 12 is wired, the MCP will return Run 12 predictions
  (not Run 4)
- `contracts/test/SentinelTest.t.sol` does NOT depend on the API server (it mocks
  the ZKMLVerifier), so no immediate dependency

### 1.3 Sideways (parallel work in other modules)

| Module | Work happening in parallel | MLOps coupling |
|---|---|---|
| `data_module/` | v4 export prep (BCCC ME injection, bug_* strip, drop GasException) | None directly. MLOps doesn't read from data_module exports. |
| `zkml/` | Artifact regeneration (proxy retrain → ONNX → circuit → ZKMLVerifier.sol) | MLOps serves the *inference* path. ZK is a separate path. No direct coupling. |
| `contracts/` | V2 redesign (AuditResultV2 with classScores[9]) | MLOps doesn't write on-chain. Indirect coupling: when contracts deploy V2, agents will fetch multi-class scores from MLOps responses (`probabilities` dict). |
| `agents/` | Routing fix (TOD → TransactionOrderDependence), threshold re-calibration | MLOps serves the `/predict` endpoint that agents call. Once MLOps is wired, agents can be tested against real Run 12 predictions. |

**Implication:** MLOps is a relatively isolated module. We can complete Phases A–C
without coordinating with the other modules' parallel work, as long as we don't
break the existing API contract (predict response shape stays the same).

---

## 2. Risk Register

### R1: Drift detector fix introduces regression in normal mode

**Description:** The fix in A.1 adds validation to baseline loading. If a previously-working
baseline (from `--source training`) is no longer recognized, we'd silently break drift detection.

**Likelihood:** LOW — `compute_drift_baseline.py` produces `{"source": ..., "stat1": [...], "stat2": [...]}` format. The validation checks for known stat names, which are present in both training and warmup sources.

**Impact:** MEDIUM — drift monitoring goes back to being silently broken.

**Mitigation:**
- Add a test (B.5): `test_drift_detector_training_baseline_recognized` — load a known training-source baseline, verify `_warmup_done = True`
- Add a test: `test_drift_detector_placeholder_baseline_rejected` — load the placeholder, verify `_warmup_done = False` and a warning is logged
- Manual smoke: after A.1, run the API with the placeholder, then with a real baseline (B.4), verify both behave correctly

**Backout:** Revert the changes in A.1. Original (silently broken) behavior is preserved.

---

### R2: `mlops_config.json` becomes stale or out of sync with MLflow

**Description:** Operators update `mlops_config.json` to point at a new checkpoint, but forget to promote to MLflow Staging/Production. The API serves the new model, but MLflow still has the old one as "current."

**Likelihood:** MEDIUM — humans forget things. There's no automated sync.

**Impact:** MEDIUM — confusion in dashboards, audit trails wrong.

**Mitigation:**
- `set_active_checkpoint.py` (B.3) can include a `--also-promote-mlflow` flag that calls `promote_model.py` automatically
- Document in the deployment README (C.6): "Always run `set_active_checkpoint.py --also-promote-mlflow` after a new model is validated"
- Optionally: add a startup check in `api.py` that warns if `mlops_config.json` checkpoint != MLflow current Staging version

**Backout:** Manual sync between `mlops_config.json` and MLflow. Annoying but easy.

---

### R3: Building drift baseline from warmup traffic is too noisy

**Description:** The drift detector might fire alerts on the first day of production traffic, even though the model is fine. "Alert fatigue" → operators ignore the alerts.

**Likelihood:** MEDIUM — first 100-200 requests are likely to have unusual distributions (e.g., mostly SolidiFI-style synthetic contracts, or mostly small ERC20 tokens).

**Impact:** MEDIUM — drift alerts lose credibility.

**Mitigation:**
- Use the `--source warmup` approach with N ≥ 30 (script enforces this, see `compute_drift_baseline.py:87-93`)
- The first 500 requests are warm-up (n_warmup=500 default in DriftDetector) anyway, so alerts are suppressed
- After 500 requests, the warmup-period KS samples are representative enough
- If too noisy: bump `KS_ALPHA` from 0.05 to 0.01 (drift_detector.py:56) — fewer false positives, but slower detection
- Alternative: tune `n_warmup` to 1000 or 2000 for first production week, then reduce

**Backout:** If too noisy after tuning, switch back to placeholder and accept no drift monitoring temporarily.

---

### R4: Docker Compose doesn't handle GPU correctly

**Description:** The inference server uses RTX 3070 (8GB VRAM). Docker's GPU passthrough (`nvidia-docker2`) is finicky on WSL2 + Windows host.

**Likelihood:** HIGH — WSL2 GPU passthrough has historically been buggy.

**Impact:** MEDIUM — can fall back to running `uvicorn` directly on the host (no Docker).

**Mitigation:**
- Test GPU passthrough early in Phase C (don't leave it for C.5 smoke test)
- If GPU passthrough fails, document the fallback: "On WSL2, run `uvicorn` directly. Docker is for production deployment only."
- Use `runtime: nvidia` in docker-compose.yml (Compose v2 syntax) instead of `deploy.resources` (Compose v3 syntax, which doesn't work in Compose v2)

**Backout:** Don't use Docker in dev. Use Docker only in production (where GPU passthrough is properly configured).

---

### R5: Run 12 checkpoint fails to load after Run 13 schema change

**Description:** `predictor.py:201-212` strict cross-check. After Run 13 drops GasException, `trainer.py:CLASS_NAMES` becomes 9 entries. Run 12's 10-class `class_names` won't match the new 9-class `expected` slice. Run 12 will fail to load.

**Likelihood:** HIGH (if we don't preemptively fix)

**Impact:** MEDIUM — the API can't serve Run 12 anymore after Run 13 schema change.

**Mitigation (preemptive, in Phase A.5 smoke test):**
- Verify Run 12 loads cleanly TODAY (before any Run 13 changes)
- Document the expected behavior in a comment in `predictor.py`: "If CLASS_NAMES shrinks below the checkpoint's class_names, the load will fail. Workaround: keep CLASS_NAMES backward-compatible."

**Mitigation (in Phase D):**
- Option A: Keep `CLASS_NAMES` list with 10 entries; mark GasException as `@deprecated`; new model just won't train on it. Run 12 keeps loading. (Recommended)
- Option B: Bump the strict check to skip if checkpoint has MORE classes than current `CLASS_NAMES` (forward compat).

**Backout:** Don't drop GasException from `trainer.py:CLASS_NAMES` in Run 13. (Would block the cleaner fix but unblocks MLOps.)

---

### R6: Phase B.4 baseline is built from synthetic data, not real traffic

**Description:** Per the implementation plan, B.4 uses a one-off Python script to generate synthetic warmup traffic (since agents aren't wired in yet). This isn't real production data, so the baseline may not be representative.

**Likelihood:** HIGH (we explicitly chose this path)

**Impact:** LOW — the synthetic data can be reasonable samples (e.g., uniform distribution of contract sizes from the v3 training set). It's better than the placeholder (which has zero data). It's worse than real production traffic (which we don't have yet).

**Mitigation:**
- Document the synthetic baseline clearly in `mlops_config.json` comment: "SYNTHETIC: replace after 30+ real production requests"
- Add a "baseline age" field to the baseline JSON: `{"source": "warmup-synthetic", "created_at": "...", "n_samples": 30}`
- After agents are wired in (Phase D+ or separate work), re-build the baseline from real traffic
- Add a CI check (later): "If baseline source == 'warmup-synthetic' AND > 7 days old, fail CI"

**Backout:** Use the placeholder. No drift monitoring, but no false alerts.

---

### R7: Run 13 changes break MLOps (cross-module coupling)

**Description:** Run 13 plan (`docs/plans/2026-06-14_Run13_4_fixes_preparation.md`) includes:
1. Drop GasException (NUM_CLASSES 10→9) — affects api.py, predictor.py, drift_detector.py, calibration
2. Extend L4 to drop `loc` (graph_schema.py) — affects preprocess.py
3. Strip Solidifi `bug_*` prefix — affects data pipeline, not MLOps
4. Inject 658 BCCC ME — affects data pipeline, not MLOps
5. ExternalBug label review — affects data pipeline, not MLOps

**Likelihood:** HIGH (these are intentional changes; they will affect MLOps)

**Impact:** MEDIUM — Phase A–C should NOT depend on Run 13 changes; Phase D is the buffer.

**Mitigation:**
- Phase A–C explicitly targets Run 12 (NUM_CLASSES=10)
- Phase D is reserved for the Run 13 transition (re-validation, config update, baseline rebuild)
- Coordination point: after Run 13 trains, we sequence (in order): data_module v4 export → ml train Run 13 → MLOps Phase D
- MLOps is **downstream** of Run 13; we don't block Run 13 work

**Backout:** Revert Run 13 schema changes in trainer.py (re-add GasException to CLASS_NAMES). Keeps MLOps working with Run 12 indefinitely.

---

### R8: Inference layer has no unit tests (audit gap)

**Description:** Audit found no unit tests for `ml/src/inference/`. If we change code in A.1, B.2, etc., we have no automated safety net.

**Likelihood:** HIGH (we know it's a gap)

**Impact:** MEDIUM — changes might introduce regressions that are caught only in production.

**Mitigation:**
- Add at least 5 tests in Phase B.5 (estimated 1 hour)
- Focus on: drift detector baseline validation (A.1), config loader (B.2), set_active_checkpoint.py (B.3)
- Long-term: add tests for predictor.py (checkpoint loading), preprocess.py (tokenization), cache.py (TTL eviction)

**Backout:** Manual smoke testing after each change. Slower but workable.

---

## 3. Decision Gates (where Ali sign-off is needed)

These are the points where the plan asks for explicit Ali approval before proceeding.

| # | Gate | File | What's at stake |
|---|---|---|---|
| G1 | Approve Phase A: bug fix + housekeeping | File 4 §A | Drift monitoring stays dead if not approved |
| G2 | Approve config file approach (mlops_config.json vs env-only) | File 3 §3.2 | API re-start breaks silently if wrong choice |
| G3 | Approve Docker Compose scope (inference + Prometheus only, Grafana deferred) | File 3 §3.4 | Scope creep, delays Phase C |
| G4 | Approve NOT using training data for baseline (synthetic bridge) | File 5 §R3 | False alerts, "alert fatigue" |
| G5 | Confirm MLOps stays in `ml/` (not top-level `mlops/`) | File 3 §3.1 | Larger refactor than value justifies |
| G6 | Choose Run 13 strategy (Option A keep GasException in CLASS_NAMES, or Option B bump strict check) | File 5 §R5 | Determines Phase D implementation |

---

## 4. External Risks (beyond our control)

### X1: WSL2 GPU driver updates break inference

**Description:** Windows Update occasionally updates the NVIDIA driver, which can break the WSL2 GPU passthrough. Symptoms: `torch.cuda.is_available()` returns False, inference falls back to CPU (very slow).

**Likelihood:** LOW (rare, but happens)

**Impact:** HIGH (API unusable)

**Mitigation:** Document the recovery steps in `ml/deploy/README.md`. Test GPU periodically.

### X2: WSL2 filesystem corruption

**Description:** WSL2 has had filesystem corruption issues in the past, especially with cross-OS file operations (e.g., editing files in WSL from Windows editor).

**Likelihood:** LOW (improved in 2024+)

**Impact:** HIGH (entire project could be lost)

**Mitigation:** Regular `git push` to remote. DVC for checkpoint backups. Don't edit WSL files from Windows editors (use VSCode's WSL extension instead).

### X3: MLflow server downtime

**Description:** `mlruns.db` is local SQLite, not a server. MLflow API calls go to this local file. No external dependency.

**Likelihood:** NONE (we use local SQLite, not a remote server)

**Impact:** N/A

**Mitigation:** N/A — note this in case someone migrates to a remote MLflow server in the future.

---

## 5. Internal Coupling

### 5.1 `api.py` ↔ `predictor.py`

**Current state:** `api.py:80` does `Predictor(checkpoint=CHECKPOINT)`. Predictor is created once at startup.

**Risk:** If we change the checkpoint path resolution (B.2), we need to make sure Predictor still works. The change is at the boundary (`CHECKPOINT` constant), so Predictor is untouched.

**Mitigation:** Smoke test (A.5) catches regressions.

### 5.2 `predictor.py` ↔ `preprocess.py`

**Current state:** `predictor.py:67` imports `ContractPreprocessor` from preprocess.py. `predictor.py:345` creates one in `__init__`.

**Risk:** If we change the preprocessing logic, inference diverges from training. The audit says the seam is healthy — same `graph_extractor.py` is used by both.

**Mitigation:** Don't change preprocessing in MLOps scope. If we need to (e.g., for Run 13), coordinate with data_module.

### 5.3 `drift_detector.py` ↔ `api.py`

**Current state:** `api.py:83` creates `DriftDetector(baseline_path=...)`. `api.py:282-283` calls `check()` every 50 requests.

**Risk:** A.1 changes DriftDetector's contract (validates baseline). Need to verify api.py still works.

**Mitigation:** Smoke test (A.5) + unit test (B.5).

### 5.4 `promote_model.py` ↔ MLflow

**Current state:** `promote_model.py:51` uses `MlflowClient()`. Default tracking URI is `sqlite:///mlruns.db`.

**Risk:** If MLflow schema changes, the script might break. Low risk (MLflow is stable).

**Mitigation:** None needed for Q4. Future-proofing: pin mlflow version in `pyproject.toml`.

---

## 6. Migration / Backout Plans

### 6.1 If Phase A.1 (drift detector fix) breaks something

**Symptoms:** API fails to start, or drift alerts fire constantly.

**Backout:**
```bash
git diff ml/src/inference/drift_detector.py  # review the changes
git checkout ml/src/inference/drift_detector.py  # revert
# Restart API
```

**Time to backout:** 1 minute.

### 6.2 If Phase B.2 (config loader) breaks startup

**Symptoms:** API fails with "no attribute '_CONFIG'" or similar.

**Backout:**
```bash
git checkout ml/src/inference/api.py
# Restart API
```

**Time to backout:** 1 minute.

### 6.3 If Phase B.4 (real baseline) gives noisy alerts

**Symptoms:** `sentinel_drift_alerts_total` counter increments rapidly.

**Backout:**
```bash
# Replace real baseline with placeholder
cp ml/data/drift_baseline.json ml/data/drift_baseline_run12.json.bak
# (Restore the placeholder, or set mlops_config.json to use placeholder)
```

**Time to backout:** 5 minutes.

### 6.4 If Phase C (Docker) doesn't work on WSL2

**Symptoms:** `docker compose up` fails with GPU passthrough error.

**Backout:** Document the fallback (run `uvicorn` directly on the host). Docker is for production only.

**Time to backout:** 0 minutes (it's a fallback, not a backout).

---

## 7. Open Questions for Ali

In addition to the decision gates (G1–G6):

| Q | Question | Default if not answered |
|---|---|---|
| OQ5 | When agents are ready to consume Run 12, do we re-run Phase B.4 with real agent traffic? | Yes (rebuild baseline from real traffic when available) |
| OQ6 | Should we add unit tests in B.5, or defer all testing to a separate QA pass? | Defer to QA pass (lower priority than the 7.5 hr of MLOps work) |
| OQ7 | Should the Docker image be published to a registry (Docker Hub, GHCR) for production use? | Defer — local image is fine for dev; production deploy is a separate effort |

---

## 8. References

- **File 1 (index):** `README.md`
- **File 2 (state):** `2026-06-15_ml_q4_proposal_mlops_current_state_findings.md`
- **File 3 (design):** `2026-06-15_ml_q4_proposal_mlops_redesign_proposal.md`
- **File 4 (plan):** `2026-06-15_ml_q4_proposal_mlops_q4_implementation_plan.md`
- **Audit (prior):** `ml/MLOPS_STATE_AND_REDESIGN_2026-06-14.md`
- **Run 13 plan:** `docs/plans/2026-06-14_Run13_4_fixes_preparation.md`
- **MEMORY.md:** `~/.claude/projects/.../memory/MEMORY.md`
