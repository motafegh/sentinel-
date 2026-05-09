# 2026-05-09 — M3 (MLOps) Plan

Spec ref: `docs/Project-Spec/SENTINEL-M3-MLOPS.md`.
Status ref: `docs/STATUS.md` ("Drift Detection Baseline Note").

---

## 1. Current State (verified from source)

```
ml/scripts/promote_model.py            MLflow registry promotion CLI
                                        Stages: None → Staging → Production
                                        --dry-run, exit codes 0/1
                                        Default tracking URI: sqlite:///mlruns.db
ml/scripts/compute_drift_baseline.py   Build drift_baseline.json
                                        --source warmup | training
                                        Strong warning if --source training
ml/src/inference/drift_detector.py     KS test, warm-up phase, rolling buffer
ml/scripts/run_overnight_experiments.py Sequential MLflow sweep launcher
                                        --start-from N to resume

DVC:
  ml/checkpoints.dvc                  Tracked
  .dvc/                               Initialised (Track 1)

Dagster:
  agents/src/ingestion/scheduler_dagster.py   RAG ingestion schedule (M4)
  agents/src/ingestion/scheduler_cron.py      Cron alternative
```

MLflow experiments observed in the project history:
- `sentinel-training`, `sentinel-retrain-v2`, `sentinel-retrain-v3`
- `sentinel-retrain-v4` planned (not yet created)

---

## 2. Plan A — Drift Baseline Workflow (highest priority post-v4)

The current state is ambiguous: `compute_drift_baseline.py` accepts
`--source training` but the right path is `--source warmup`, which
requires real production traffic that does not yet exist.

### 2.1 Decide on a synthetic warm-up bridge

Until M6 ships traffic, we have three options:

| Option | Cost | Risk |
|---|---|---|
| Wait for M6 | Zero now; blocks alerting | Low |
| Replay Solodit/recent contracts as a "synthetic warm-up" feed | One script | Medium — drift baseline tied to a non-prod distribution |
| Use 2024 training data, accept noisy alerts at start | Already supported | High — alerts will be useless for ~weeks |

Recommendation: **wait for M6**, but in the meantime add a script that
stress-tests `drift_detector.py` with synthetic inputs so the
detector is known-good before the M6 cutover. New file:

```
ml/scripts/exercise_drift_detector.py    NEW
  • Generates 600 synthetic feature vectors
  • Asserts no alerts during warm-up (first 500)
  • Injects a clear distribution shift after warm-up; asserts at
    least one alert fires within 50 requests
  • Exit 0 on PASS, 1 on FAIL — wire into CI
```

### 2.2 Production-readiness checklist

- [ ] `compute_drift_baseline.py --source training` emits `WARNING:
      using training data; expect noisy alerts` (verify in source —
      currently a docstring warning, promote to logger.warning at runtime)
- [ ] `/debug/warmup_dump` endpoint added to `ml/src/inference/api.py` —
      exposes the rolling warm-up buffer as JSONL
- [ ] `drift_detector.py` ships an `is_warming_up` property used by `/health`

---

## 3. Plan B — MLflow Registry Discipline

`promote_model.py` works; what's missing is process.

### 3.1 Promotion gates

Adopt the following rule (codify in `ml/scripts/promote_model.py`
docstring):

```
None → Staging      requires: tuned val F1-macro > previous Staging
Staging → Production requires:
  • tuned val F1-macro > previous Production
  • no per-class F1 below SENTINEL-EVAL-BACKLOG floor
  • drift_baseline.json built from warmup data exists
  • predictor.py loaded checkpoint successfully in CI
```

Implementation: `promote_model.py` already validates F1; extend with a
`--require-baseline ml/data/drift_baseline.json` flag that fails if the
file does not exist or was generated from `--source training`.

### 3.2 Run-name conventions

Lock in the convention already in use:

```
multilabel-v<n>-<knob>          e.g. multilabel-v4-focal-r16
sentinel-retrain-v<n>           experiment name
```

Document this in `docs/Project-Spec/SENTINEL-M3-MLOPS.md` (append, do
not rewrite).

---

## 4. Plan C — DVC Hygiene

```
ml/checkpoints.dvc                Tracked checkpoints
ml/data/                          Currently NOT under DVC (graphs are
                                  68k files; recommend adding)
```

Decision needed: bring `ml/data/graphs/` and `ml/data/tokens/` under DVC?

- Pros: reproducible v3/v4 comparisons, shareable across machines
- Cons: ~10–30 GB DVC remote needed, push/pull friction
- Recommendation: yes, but only after a remote is provisioned (S3/GCS/
  local FS); track in `docs/ROADMAP.md` not as a sprint blocker

---

## 5. Plan D — Dagster Schedule Validation

Currently in `agents/src/ingestion/scheduler_dagster.py`. Verify:
- [ ] Schedule has a clear cron string and is validated at import
- [ ] Failures emit Prometheus counter increments (M4 cross-cut)
- [ ] No silent retries on RAG ingestion failures

If any of the above is missing, a follow-up task should be filed in
`docs/ROADMAP.md` under M4.

---

## 6. Acceptance Criteria

- `exercise_drift_detector.py` PASS in CI
- `promote_model.py --require-baseline` flag implemented
- ADR or appendix in `SENTINEL-M3-MLOPS.md` records the promotion gates
- `docs/STATUS.md` "Drift Detection Baseline Note" replaced with a link
  to the implemented workflow once M6 supplies real traffic
