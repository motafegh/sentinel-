---
date: 2026-06-17
module: ml
run: Run12
what: changes
descriptor: api_schema_claim_correction
status: ACTIVE
---

# 2026-06-17 — API Schema Claim Correction (false positive)

## What changed

A "critical API schema bug" claim in `MEMORY.md` (and the 2026-06-16 MLOps state check
report) was investigated and found to be a **false positive**. The Pydantic schema at
`ml/src/inference/api.py:209` is `dict[str, float | list[float]]` (union type), not
`dict[str, float]` as claimed. The full test suite `pytest ml/tests/test_api.py` passes
**18/18** with Run 12's per-class thresholds loaded.

## No code changes

The API was always correct. The fix was to the **docs**, not the code.

## Files updated

1. `~/.claude/projects/.../memory/MEMORY.md` — false P0 claim removed; production gate
   reason corrected to "drift baseline placeholder" only.
2. `docs/reports/2026-06-16_ml_Run12_mlops_full_state_check/2026-06-16_ml_Run12_INDEX_mlops_full_state_check.md` —
   REVISION section added at top; false claims in body corrected; verification
   results table updated.
3. `ml/testing_specs/K_inference_api.md` — `tier_thresholds` type corrected to
   `dict[str, float | list[float]]`; `/health` vs `/predict` shape inconsistency flagged.
4. `ml/audit_docs/ISSUES.md` — BUG-1 filed as closed (false positive) for traceability.

## Files created

5. `docs/learning_sentinel/2026-06-17_step1_api_bug_investigation.md` — full
   investigation log with the 3-layer WHY/WHAT/HOW framework, the 30-second
   verification ritual, and the 5 failure modes for stale docs.

## Residual minor inconsistency (NOT a bug, deferred)

- `/health` (api.py:239) returns `predictor.tier_confirmed_threshold` — scalar default
- `/predict` (api.py:340) returns per-class list from `result["tier_thresholds"]["confirmed"]`

Two endpoints disagree on the shape of `tier_thresholds.confirmed`. Not a 500; flagged
for agent-consumer awareness. Recommend a future cleanup pass to align both endpoints.

## Lesson

Rule #4 in action: **Trust source code only. Distrust all docs.** The 30-second verification
ritual (read type → read producer → run test) caught a false-positive critical bug claim
that would otherwise have triggered ~30 min of unnecessary code changes.

## Cross-references

- `docs/learning_sentinel/2026-06-17_step1_api_bug_investigation.md` — full investigation
- `ml/audit_docs/ISSUES.md` — BUG-1 closed entry
- `~/.claude/scratch/sentinel_session_plan_20260617.md` — session plan scratch
