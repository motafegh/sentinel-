# 2026-05-27 — Three-Tier Inference Output + Agents Integration + MLOps Gates

**Context:** Run 4 complete (F1=0.3362, ep32); pipeline verified FAIL=0; continuing from previous session.

---

## Summary

Four sequential work items completed in this session:

1. **Three-tier inference output** (`predictor.py`) — CONFIRMED / SUSPICIOUS tiers with full probability vector
2. **API schema update** (`api.py`) — PredictResponse aligned to three-tier output
3. **MLOps promotion gates** (`promote_model.py`) — drift baseline check + Production F1 gate + threshold artifact
4. **Agent schema integration** — all orchestration files updated to consume three-tier `ml_result`
5. **Drift detector CI test** (`exercise_drift_detector.py`) — four-phase smoke test for CI
6. **Agent test suite fix** — `test_graph_routing.py` fully rewritten for current architecture

Commits: `ac1542f`, `76b02f7`, `d85e291`

---

## 1. Three-Tier Predictor Output (`ml/src/inference/predictor.py`)

### What Changed

`_format_result()` completely rewritten. Previous schema returned only `label` ("vulnerable"/"safe") and `vulnerabilities` (above-threshold only).

New output dict:

```python
{
    "label":            "safe" | "suspicious" | "confirmed_vulnerable",
    "probabilities":    {"Reentrancy": 0.72, "IntegerUO": 0.54, ...},  # all 10 classes
    "confirmed":        [{"vulnerability_class": str, "probability": float, "tier": "CONFIRMED"}],
    "suspicious":       [{"vulnerability_class": str, "probability": float, "tier": "SUSPICIOUS"}],
    "vulnerabilities":  [...],  # legacy alias for confirmed only (backward compat)
    "tier_thresholds":  {"confirmed": 0.55, "suspicious": 0.25, "noteworthy": 0.10},
    "thresholds":       [...],  # per-class tuned thresholds (unchanged)
    "truncated":        bool,
    "windows_used":     int,
    "num_nodes":        int,
    "num_edges":        int,
}
```

### Tier Boundaries

| Tier | Condition | Meaning |
|------|-----------|---------|
| CONFIRMED | prob ≥ 0.55 | Model confident — high priority investigation |
| SUSPICIOUS | 0.25 ≤ prob < 0.55 | Borderline — warrants agent-layer corroboration |
| (not returned) | prob < 0.25 | Present in `probabilities` dict but not in tier lists |

**Evidence for 0.25 threshold:** 8 of 12 missed classes in 20-contract eval had prob 0.25–0.54. Binary 0.50 threshold was discarding real signal.

### Configurable Per-Deployment

```python
Predictor(
    tier_confirmed_threshold=0.55,   # override default
    tier_suspicious_threshold=0.25,  # override default
)
```

---

## 2. API Schema Update (`ml/src/inference/api.py`)

### Changes

- **Checkpoint default** updated: `multilabel_crossattn_v2_best.pt` → `GCB-P1-Run4-no-asl-pw_best.pt`
- **`VulnerabilityResult`** schema: added `tier: str | None = Field(None, ...)`
- **`PredictResponse`** schema: added `probabilities`, `confirmed`, `suspicious`, `tier_thresholds` fields; `vulnerabilities` retained as legacy backward-compat alias
- **`/health`** endpoint: now returns `tier_thresholds`, `model_epoch`, `model_f1_val` (read from checkpoint `config` dict at startup)
- **`/predict`** drift tracking: added `confirmed_count` and `suspicious_count` alongside `num_nodes`/`num_edges`

---

## 3. MLOps Promotion Gates (`ml/scripts/promote_model.py`)

Three missing gates added (all from ML_API_MLOPS_GAPS.md §3.2):

### Gate 1: `--require-baseline` (Production only)

```bash
python ml/scripts/promote_model.py \
    --checkpoint ... --stage Production \
    --require-baseline ml/data/drift_baseline.json
```

Fails if:
- `drift_baseline.json` does not exist → error with rebuild instructions
- `drift_baseline.json` has `source='training'` → error (training-source baseline fires on all prod contracts)

### Gate 2: Previous-Production F1 Comparison

When `--stage Production`, queries MLflow for current Production model's `val_f1_macro` and fails if new model's F1 does not exceed it. Prevents accidental Production downgrade.

### Gate 3: Companion Threshold JSON Artifact

Automatically logs `{checkpoint_stem}_thresholds.json` as MLflow artifact alongside the checkpoint if it exists. Previously the checkpoint was logged without thresholds, causing silent fallback to uniform 0.5 thresholds on deploy.

---

## 4. Drift Detector CI Smoke Test (`ml/scripts/exercise_drift_detector.py`)

Four-phase CI test — exits 0 on pass, 1 on fail:

| Phase | What it does | Assertion |
|-------|-------------|-----------|
| 1 — Warm-up | Feed 500 in-distribution vectors, no baseline | `check()` returns `{}` throughout; `warmup_done` flips at exactly 500 |
| 2 — Build baseline | `dump_warmup_stats()` → write `drift_baseline.json` | Buffer has ≥ 30 entries |
| 3 — In-distribution | Feed 100 more in-dist vectors with baseline loaded | Zero KS alerts (tolerate ≤1 statistical false positive) |
| 4 — Shift detection | Feed 100 vectors with 3× num_nodes/num_edges | At least 1 KS alert fires within 50 shifted requests |

**Observed:** Alert fires at shifted request 10 (p=0.042 for `num_nodes`).

---

## 5. Agent Schema Integration

All six agent files updated to consume three-tier `ml_result`.

### routing.py

`_iter_class_probs()` helper: reads from `probabilities` dict (all 10 classes, always present in three-tier schema) with fallback to `vulnerabilities` list (legacy schema). `compute_active_tools()` and `build_routing_decisions()` both use it.

**Before:** Only classes in `vulnerabilities` (above-threshold) were evaluated for routing.
**After:** All 10 classes evaluated — a class at prob=0.31 (above DenialOfService DEEP_THRESHOLD=0.30) now correctly triggers routing even if it's in the SUSPICIOUS tier.

### nodes.py

| Node | Change |
|------|--------|
| `evidence_router` | logs `len(probabilities)` instead of `len(vulnerabilities)` |
| `ml_assessment` | logs from `confirmed` tier first, then `suspicious`, with tier tag |
| `rag_research` | uses `confirmed or suspicious` for query topic (picks highest-prob confirmed class) |
| `static_analysis` | scopes Slither detectors from `probabilities` dict (all 10 classes above 0.35), not just `vulnerabilities` |
| `synthesizer` | reads `confirmed + suspicious` for verdicts; updated recommendation for new label values; adds tier counts to LLM prompt; adds `confirmed`, `suspicious`, `probabilities`, `tier_thresholds` to `final_report` |

### inference_server.py

`_mock_prediction()` rewritten to return full three-tier schema (including `probabilities` dict, `confirmed`/`suspicious` lists, `tier_thresholds`, `thresholds`).

### test_routing_phase0.py

`_ml()` helper updated to produce three-tier schema including `probabilities` dict. 46 tests still pass.

### test_graph_routing.py

Fully rewritten. Previous version referenced:
- `_route_after_ml` (old name, replaced by `_route_from_evidence_router`)
- `_is_high_risk` (old binary-confidence routing helper, never existed in current code)
- `"confidence"` field in ml_result (removed in Track 3)
- `label == "vulnerable"` (now "confirmed_vulnerable")

New version: `_ml_three_tier()` helper, 32 tests covering all nodes + routing + full graph integration.

---

## 6. Test Results

| Suite | Tests | Result |
|-------|-------|--------|
| `tests/test_routing_phase0.py` | 46 | ✅ PASS |
| `tests/test_graph_routing.py` | 32 | ✅ PASS |
| `ml/scripts/exercise_drift_detector.py` | 4 phases | ✅ PASS |

---

## Still Pending

| Task | Status |
|------|--------|
| Register Run 4 in MLflow Staging | Unblocked — run `promote_model.py --stage Staging` |
| ExternalBug agent-layer enhancement | Designed in AGENTS_PLAN_V2.md; not implemented |
| Graph inspector Phase 1 (:8013) | Designed; not implemented |
| Cross-validator Phase 2 node | Designed; not implemented |
| Drift baseline from warm-up traffic | Blocked until M6 API gateway live |
