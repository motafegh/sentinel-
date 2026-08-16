# Logical V3 evidence snapshot index / addendum

**Snapshot root:** `docs/plan/ml-R4/evidence/2026-08-15_phase8_logical_v3/`
**Original snapshot commit:** `44fbb9c1d2033be8002fe404d650cf09f08b0f29`
**Original JSON checksum ledger:** `SHA256SUMS.txt`
**Current interpretation:** hardened post-regeneration V3 evidence; training and selector promotion remain unauthorized.

## Why this addendum exists

The original snapshot intentionally copied several stage-specific reports without rewriting their historical contents. One of them, `logical_v3_summary.json`, was produced immediately after the logical-V3 rebuild and **before** the later sensitivity, selector, negative-queue, CUDA, and hardened snapshot work completed.

That file is therefore classified here as:

`HISTORICAL_BUILD_STAGE_SUMMARY`

Its recorded status is:

`LOGICAL_V3_REBUILD_COMPLETE_RESEARCH_REGENERATION_PENDING`

and its `next` instructions say to regenerate downstream research reports. Those statements were correct at the time that build-stage report was produced, but they are **not current execution instructions** and must not be interpreted as pending work today.

The file is preserved unchanged because rewriting it would destroy evidence chronology.

## Current snapshot interpretation

The downstream work that `logical_v3_summary.json` described as pending was subsequently completed under hardened source commit:

`83bd566b9c4f4f653e530c2c0f5c990858dd759d`

The final snapshot was then created with `coherence=PASS`, all 11 JSON files in `SHA256SUMS.txt` verified, and committed at:

`44fbb9c1d2033be8002fe404d650cf09f08b0f29`

For post-rebuild/current evidence, use these files instead of the old `next` field in `logical_v3_summary.json`:

- `logical_v3_acceptance.json` — corrected accepted V3 outcome/role accounting;
- `representation_sensitivity_v1.json` — lineage-bound sensitivity evidence;
- `bounded_window_selector_v1.summary.json` — bounded CPU selector evidence summary;
- `selector_gpu_compare_v1.json` — lineage-bound CUDA comparison;
- `confirmed_negative_review_queue_v1.json` — hardened 200-cell / 200-group review queue, still not negative truth;
- `snapshot_coherence_v1.json` — original cross-report coherence result;
- `SHA256SUMS.txt` — original committed JSON checksum ledger.

## Integrity rule

This addendum is **non-destructive contextual metadata**. It was added after the original snapshot commit and is intentionally not retroactively inserted into `SHA256SUMS.txt`; that checksum file continues to bind the exact 11 JSON artifacts that formed the original durable snapshot.

Fresh-clone CI separately requires this addendum and re-verifies:

1. the exact JSON inventory in `SHA256SUMS.txt`;
2. every listed JSON SHA-256;
3. cross-report lineage/coherence using the current strengthened verifier;
4. the full 8×25 queue shape, deterministic candidate IDs, global group uniqueness, allowed `UNKNOWN`/`NOT_REVIEWED` states, and no negative-truth claim;
5. the explicit historical classification of `logical_v3_summary.json` recorded here.

## Current restart authority

Do not restart Phase-8 work from the `next` list inside `logical_v3_summary.json`.

Use:

`docs/plan/ml-R4/runs/2026-08-16_PHASE8_v3_hardened_evidence_snapshot_closeout.md`

The current primary next track remains R4-GAP-007 confirmed-negative pilot review using the committed hardened queue. Queue membership is review reservation only; no target `0` exists yet. Full training and guarded-selector promotion remain unauthorized.
