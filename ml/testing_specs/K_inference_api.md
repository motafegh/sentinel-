# K — Inference API Validation

> Always load `00_rules.md` before following this procedure.
> Apply Rule 2 (gate assertions + completion attestation) at every step.
>
> **Last revised: 2026-06-14** (post-Run-12 launch). **Critical update**: the
> `SENTINEL_CHECKPOINT` default in `api.py` is still `ml/checkpoints/GCB-P1-Run4-no-asl-pw_best.pt`
> (historical). The active best as of Run 12 is
> `ml/checkpoints/GCB-P1-Run12-v3dospatched-20260613_best.pt`. **Always set
> `SENTINEL_CHECKPOINT` env var explicitly when validating**. Forward-looking:
> Run 13 plan drops `GasException` → `NUM_CLASSES=9` (10 → 9); the response
> schema (10-element vectors) will change. This file will need a major update
> after Run 13 launches.

---

## When This File Applies

- Validating a freshly deployed or restarted API instance
- After a checkpoint change (new promotion or rollback)
- After a `FEATURE_SCHEMA_VERSION` bump (cache invalidation required)
- When API responses look wrong (wrong class names, zero-length confirmed list, etc.)
- Before reporting any inference result from the API as ground truth

Always load alongside: `C_diagnostic_checks.md` C.2 (behaviour checks that
must pass before promotion) and `I_regression_guard.md` I.3.1 (behaviour
checks required before promotion).

---

## K.1 — Read Before Asserting

Read `ml/src/inference/api.py` docstring **before** writing any assertions about
response field names or tier thresholds. The schema evolves. Do not assume field
names from memory.

Current response schema version (as of `api.py` header): **Three-tier suspicion
output (2026-05-27)**. `PredictResponse` fields:

| Field | Type | Notes |
|---|---|---|
| `label` | `str` | `"safe"` \| `"suspicious"` \| `"confirmed_vulnerable"` |
| `probabilities` | `dict[str, float]` | Full 10-class vector, always present |
| `confirmed` | `list[VulnerabilityResult]` | prob ≥ `tier_confirmed_threshold` (default 0.55) |
| `suspicious` | `list[VulnerabilityResult]` | `tier_suspicious_threshold` ≤ prob < `tier_confirmed_threshold` |
| `vulnerabilities` | `list[VulnerabilityResult]` | Legacy alias for `confirmed` (backward compat) |
| `tier_thresholds` | `dict[str, float]` | `{"confirmed": 0.55, "suspicious": 0.25, "noteworthy": 0.10}` |
| `thresholds` | `list[float]` | Per-class tuned decision thresholds (10-element list) |
| `truncated` | `bool` | True if source was truncated to `MAX_SOURCE_BYTES` |
| `windows_used` | `int` | Token windows scored (≥1; >1 for long contracts) |
| `num_nodes` | `int` | Graph node count |
| `num_edges` | `int` | Graph edge count |

`VulnerabilityResult` fields: `vulnerability_class` (str), `probability` (float),
`tier` (str\|None). Note: **`vulnerability_class`**, not `class` — Bug 3 fix in
`api.py` changed `v['class']` → `v['vulnerability_class']`; old consumers using
`class` will silently receive `None`.

---

## K.2 — Endpoints

Read `api.py` for the full endpoint list. Three endpoints exist:

| Endpoint | Method | Purpose |
|---|---|---|
| `GET /health` | GET | Liveness check; reports `predictor_loaded`, `architecture`, `thresholds_loaded`, `tier_thresholds`, `model_epoch`, `model_f1_val` |
| `POST /predict` | POST | Full multi-label prediction; body: `{"source_code": "..."}` |
| `POST /hotspots` | POST | GNN attention hotspots + full prediction; returns `hotspots` (top-20 functions by embedding norm), `hotspot_stats`, plus all `PredictResponse` fields |
| `GET /metrics` | GET | Prometheus metrics (exposed by `prometheus-fastapi-instrumentator`) |

Input validation applied by both `/predict` and `/hotspots`:
- `source_code` minimum 10 characters
- Must contain `"pragma"` or `"contract"` (case-insensitive); otherwise HTTP 400
- `len(source_code.encode()) > MAX_SOURCE_BYTES` → HTTP 413 (read
  `ContractPreprocessor.MAX_SOURCE_BYTES` from `preprocess.py` for the current limit;
  do not hardcode it)

---

## K.3 — Threshold Source Verification

The API serves security classifications. The thresholds used must be verifiable.
Do not assume the deployed model is using tuned thresholds.

### K.3.1 — Verify via `/health`

```bash
curl -s http://localhost:8000/health | python -m json.tool
```

Confirm:
- `"predictor_loaded": true` — model is loaded
- `"thresholds_loaded": true` — per-class tuned thresholds were found and applied
- `"architecture"` matches the expected run version (e.g. `"four_eye_v8"`)
- `"model_epoch"` and `"model_f1_val"` match the promoted checkpoint's recorded values

If `thresholds_loaded` is `false`: the deployed model is using fallback uniform
thresholds. This means `<checkpoint_stem>_thresholds.json` was not present at
startup. Rebuild with `tune_threshold.py` and restart.

### K.3.2 — Trace the Threshold Load Path

Read `ml/src/inference/predictor.py` to understand the threshold load order:

1. `Predictor.__init__` loads the checkpoint `.pt` file
2. It looks for a companion `<checkpoint_stem>_thresholds.json` in the same directory
3. If found: loads per-class thresholds (10 floats in `CLASS_NAMES` index order)
4. If missing: falls back to uniform 0.5 for all classes
5. The `thresholds_loaded` flag on the predictor reflects whether step 3 succeeded

The `thresholds` field in `PredictResponse` is the 10-element list actually used
for the response. Verify it matches the `_thresholds.json` companion file values.

### K.3.3 — Tier Threshold Verification

The tier boundaries used by the API (`confirmed`, `suspicious`) are set in
`predictor.py` as `tier_confirmed_threshold` and `tier_suspicious_threshold`.
Read from `predictor.py` — do not assume the values from the `api.py` docstring
comment (0.55 and 0.25) are current. Verify from `/health`:

```python
health = requests.get("http://localhost:8000/health").json()
assert health["tier_thresholds"]["confirmed"]  == expected_confirmed_threshold
assert health["tier_thresholds"]["suspicious"] == expected_suspicious_threshold
```

---

## K.4 — Cache Invalidation Verification

Read `ml/src/inference/cache.py` before diagnosing a stale-cache issue.

### K.4.1 — Cache Key Format

Cache key: `"{content_md5}_{FEATURE_SCHEMA_VERSION}"`

When `FEATURE_SCHEMA_VERSION` is bumped in `graph_schema.py`:
- All existing cache entries become unreachable (new key suffix)
- Old files remain on disk until TTL expiry (default 86 400 s = 24 h) or manual
  deletion
- No manual cache flush is required after a schema version bump — the key change
  handles invalidation automatically

When the checkpoint changes (promotion or rollback):
- The cache key does NOT encode the checkpoint — it encodes only source content
  and schema version
- If the feature schema version did NOT change between checkpoints, old cached
  `(graph.pt, tokens.pt)` pairs **will be reused** for previously-seen contracts
- This is correct if the schema is unchanged; the graph featurisation is identical
- If model behaviour changes because of a weight change only (same schema), the
  cache is still valid — model weights are not part of the preprocessing cache

### K.4.2 — Stale Cache Detection

The cache performs a schema guard on every `get()` call:

```python
if graph.x.shape[1] != NODE_FEATURE_DIM:
    # evict and treat as miss
```

This catches any `.pt` file whose feature dimension no longer matches the current
schema. If you see frequent cache eviction warnings in logs, a schema version bump
may have been applied without the old files being cleared. This is not a bug —
files expire after TTL. To force immediate invalidation:

```bash
rm -rf ~/.cache/sentinel/preprocess/
```

(Default cache dir is `~/.cache/sentinel/preprocess/`. Read `InferenceCache.__init__`
`cache_dir` default in `cache.py` to confirm the actual path before deleting.)

---

## K.5 — Known-Positive / Known-Negative Round-Trip Tests

These tests must be run manually; they are not automated in the smoke suite.

### K.5.1 — Known-Positive Test

For each of the 10 vulnerability classes, run one contract from
`ml/scripts/test_contracts/` (or the SmartBugs Curated set) that is confirmed
positive for that class. For each:

```bash
curl -s -X POST http://localhost:8000/predict \
    -H 'Content-Type: application/json' \
    -d '{"source_code": "..."}' | python -m json.tool
```

Verify:
- `probabilities[<class_name>]` > the per-class tuned threshold (from `thresholds` field)
- The class appears in `confirmed` or `suspicious` (depending on probability magnitude)
- `label` is not `"safe"` for a confirmed-positive contract

### K.5.2 — Known-Negative (NonVulnerable) Test

Run at least 5 clean contracts with no known vulnerabilities. For each, verify:
- `confirmed` list is empty
- `probabilities` values are all below `tier_thresholds["confirmed"]`
- `label` is `"safe"` or `"suspicious"` (not `"confirmed_vulnerable"`)

### K.5.3 — FP Probe Regression Check

For any contract previously used as a False Positive probe in prior runs: confirm
its prediction has not regressed. Read the probe contract's expected prediction
from the relevant run doc before testing — do not assume from memory.

### K.5.4 — Hotspot Endpoint Sanity

For one known-positive Reentrancy contract, call `/hotspots` and verify:
- `hotspots` list is non-empty
- `hotspot_stats["attention_source"]` is present
- `hotspot_stats["total_function_nodes"]` > 0
- The top-scored hotspot `fn_name` is a function name present in the contract source
- `label`, `probabilities`, `confirmed`/`suspicious` match the `/predict` response for
  the same contract (they should be identical)

---

## K.6 — Drift Monitoring Verification

Read `ml/src/inference/drift_detector.py` before interpreting drift signals.

The drift detector updates per request and runs a KS check every
`DRIFT_CHECK_INTERVAL` requests (default 50; override via
`SENTINEL_DRIFT_CHECK_INTERVAL` env var). It tracks four features per request:
`num_nodes`, `num_edges`, `confirmed_count`, `suspicious_count`.

Verify drift is surfacing:
1. Send > 50 requests with the API running
2. Check API logs for any `DriftDetector` warning messages
3. Confirm `drift_detector` is not `None` in `app.state` (will be `None` if
   `DRIFT_BASELINE_PATH` does not exist at startup — check `/health` for
   `predictor_loaded` but no dedicated drift-loaded flag)

If the drift baseline is missing at startup, the `DriftDetector` is initialised
without a reference distribution. Drift checks will not fire. Rebuild the baseline
per `I.3.5` before Production deployment.

---

## K.7 — Environment Variables

Read `api.py` for the full list. Key overrides:

| Env var | Default | Effect |
|---|---|---|
| `SENTINEL_CHECKPOINT` | `ml/checkpoints/GCB-P1-Run4-no-asl-pw_best.pt` (STALE; Run 4 baseline) | Override checkpoint path. **Always set explicitly when validating** — verify from `/health` `checkpoint` field. Active best: `ml/checkpoints/GCB-P1-Run12-v3dospatched-20260613_best.pt`. |
| `SENTINEL_DRIFT_BASELINE` | `ml/data/drift_baseline.json` | Override drift baseline path |
| `SENTINEL_DRIFT_CHECK_INTERVAL` | `50` | Requests between KS checks |
| `SENTINEL_PREDICT_TIMEOUT` | `60` | Seconds before HTTP 504 |

The checkpoint default in `api.py` is the **Run 4 checkpoint**, not the current
best. Always verify `SENTINEL_CHECKPOINT` is set to the correct promoted checkpoint
before running any validation. Confirm from `/health` `checkpoint` field.

**Active best (post-Run-12 launch)**: `ml/checkpoints/GCB-P1-Run12-v3dospatched-20260613_best.pt`
(path subject to change; verify from `MEMORY.md` Current State).

---

## K.8 — Completion Attestation

After completing API validation, append to the relevant run or deployment doc:

```
## Procedure Attestation — K_inference_api — <ISO date>
API instance: <url or local>
Checkpoint: <path from /health checkpoint field>
Steps completed:
  K.3.1 /health verified:
    predictor_loaded:    true/false
    thresholds_loaded:   true/false
    architecture:        <value>
    model_epoch:         <value>
    model_f1_val:        <value>
  K.3.2 threshold load path traced:   YES/NO
  K.3.3 tier thresholds verified:
    confirmed:           <value>
    suspicious:          <value>
  K.4.2 cache invalidation checked:   YES/NO/N/A
  K.5.1 known-positive round-trip:    PASS/FAIL (N classes tested: N)
  K.5.2 known-negative round-trip:    PASS/FAIL (N contracts tested)
  K.5.3 FP probe regression:          PASS/FAIL/N/A
  K.5.4 hotspot endpoint sanity:      PASS/FAIL/N/A
  K.6 drift monitoring verified:      YES/NO/N/A
Steps skipped:    [any skipped + explicit reason]
Issues found:     [describe or "none"]
Written to:       [path of this attestation]
```
