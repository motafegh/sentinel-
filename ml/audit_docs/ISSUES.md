# SENTINEL ML Audit Issues

Next BUG-ID to assign: BUG-2

---

## How to File a New Issue

Follow `H_issue_triage.md` H.5:
1. Assign the next sequential ID from "Next BUG-ID" above, then increment it
2. Record: symptom, trigger condition, epoch first observed, config values at the time,
   alert code (if any), and resolution or investigation path
3. Write the entry before the session closes (Rule 3 — no floating findings)

---

## Open Issues

*(none)*

---

## Resolved Issues

### BUG-1 — False-positive claim: API `tier_thresholds` schema mismatch (CLOSED 2026-06-17)

**Status:** CLOSED — false positive. No code change required.
**Filed:** 2026-06-17
**Closed:** 2026-06-17 (same session)
**Reporter:** Self-flag (Ali asked to investigate the P0 from MEMORY.md)

**Symptom (claimed):**
`/predict` and `/hotspots` return HTTP 500 when serving Run 12 with per-class
thresholds loaded. `ml/tests/test_api.py` reported 4 failures.

**Investigation:**
- `ml/src/inference/api.py:209` declares `tier_thresholds: dict[str, float | list[float]]`
  (Pydantic v2 union — accepts both scalar and list).
- `ml/src/inference/predictor.py:750-754` returns `{"confirmed": list, "suspicious": float, "noteworthy": float}`.
- `pytest ml/tests/test_api.py` → **18/18 PASS** (re-run 2026-06-17).

**Root cause of false positive:**
- The original claim (in MEMORY.md and the 2026-06-16 MLOps state check report) was based
  on reading the type as `dict[str, float]` — likely from a stale mental model of Pydantic v1
  syntax, or from skimming the type without recognising the `|` as a union.
- The "4 fail" claim in the original report was not reproducible on re-run.

**Resolution:**
- No code change required — the API was always correct.
- Updated docs: MEMORY.md, MLOps state check report, K_inference_api.md spec.
- This BUG-1 entry exists so the next session sees the resolution and doesn't redo the work.

**Residual minor inconsistency (not a bug, not part of BUG-1):**
- `/health` returns scalar `tier_thresholds.confirmed`; `/predict` returns per-class list.
- Not a 500. Not user-blocking. Flagged for future cleanup.

**Cross-references:**
- `docs/learning_sentinel/2026-06-17_step1_api_bug_investigation.md`
- `docs/reports/2026-06-16_ml_Run12_mlops_full_state_check/2026-06-16_ml_Run12_INDEX_mlops_full_state_check.md` (§REVISION)
- `docs/changes/2026-06-17-ml-api-schema-claim-correction.md`
- `~/.claude/projects/-home-motafeq-projects-sentinel/memory/MEMORY.md`

