# Learning Docs — Sentinel v2 Data Module Build

**Date:** 2026-06-09 (revised 2026-06-12)
**Purpose:** A learning-oriented companion to the 9 actionable plans. Every stage is taught incrementally (motivation → business logic → edge cases → broader impact) BEFORE code is written. You confirm understanding; we move on.

---

## How this works

For each of the 9 stages (Stage 0–Stage 8), you'll get a learning doc that covers **three layers**:

1. **The problem** — why this stage exists, what failed before, the different branches considered
2. **The solution** — what the design decisions are, why this branch won, the edge cases
3. **The broader context** — what this stage impacts downstream, what changes if you change this stage

You read it. You ask questions. You say "got it" or "explain X more." We then run the exit-criteria tests, you confirm the code matches what you understood, and we move on.

## Files

| File | What it is | Status |
|---|---|---|
| [`LEARNING_CHECKLIST.md`](LEARNING_CHECKLIST.md) | Running checklist of all concepts you should understand across all 9 stages. Tick off as you go. | 🔵 Active |
| [`stage_0_skeleton.md`](stage_0_skeleton.md) | Stage 0: skeleton + package split (BCCC failure, 3 branches, v9 schema, 8 bugs) | ✅ Complete |
| [`stage_1_ingest_preprocess.md`](stage_1_ingest_preprocess.md) | Stage 1: ingestion + preprocessing (5-step pipeline, two-pass compile, dedup@0.85) | ✅ Complete |
| [`stage_2_representation.md`](stage_2_representation.md) | Stage 2: representation port (thin adapter, byte-identical regression, 13-issue preservation) | ✅ Complete |
| [`stage_3_labeling.md`](stage_3_labeling.md) | Stage 3: crosswalks + parsers + merger (5 critical-path sources, 99% co-occurrence prevention, 22,356 merged labels) | ✅ Complete (2026-06-12) |
| [`stage_4_verification.md`](stage_4_verification.md) | Stage 4: BCCC-failure catcher (6 components, 9 design decisions, Phase 5 regression, SmartBugs 94.4% recall) | ✅ Complete (2026-06-12) |
| [`stage_5_splitting_registry.md`](stage_5_splitting_registry.md) | Stage 5: splitting + registry (4 strategies, NonVulnerable 3:1 cap, SQLite catalog) | 🔵 NOT STARTED |
| [`stage_6_analysis.md`](stage_6_analysis.md) | Stage 6: analysis (complexity_proxy_risk, co-occurrence, drift monitor) | 🔵 NOT STARTED |
| [`stage_7_export_seam.md`](stage_7_export_seam.md) | Stage 7: export + seam swap (7 gates, predictor fix, EMITS fix, Docker) | 🔵 NOT STARTED |
| [`stage_8_run11_launch.md`](stage_8_run11_launch.md) | Stage 8: Run 11 launch (12-condition checklist, timestamped run, watcher) | 🔵 NOT STARTED |

## Current build state (2026-06-12)

**Stages 0–4: CODE-COMPLETE.** 463 tests pass, 79 skipped. 22,356 contracts labeled (SolidiFI 283 + DIVE 22,073). All 5 critical-path crosswalks built (3 of 5 parsers operational; DeFiHackLabs + Web3Bugs deferred to v2.1). Stage 4's 12 exit criteria all met — 94.4% SmartBugs aggregate recall (above 90% threshold), gate PASSES on the v2 baseline corpus.

**Stages 5–8: NOT STARTED.** Stage 5 (Splitting + Registry) is next. Learning docs for all 9 stages written.

## Workflow

```
For each stage:

  1. I post the stage's learning doc
  2. You read it (problem → solution → impact)
  3. You ask questions or say "got it"
  4. I run the exit-criteria tests
  5. You confirm the code matches what you understood
  6. I tick off items in LEARNING_CHECKLIST.md
  7. We move to the next stage
```

## What "understanding" looks like

For each stage, you should be able to answer, without looking at the doc:

1. **What problem does this stage solve? Why does that problem exist? What were the alternative approaches?**
2. **What does the solution look like at a high level? Why was this design chosen over the alternatives? What are the edge cases?**
3. **What does this stage impact downstream? What could break if we got this stage wrong? What metrics would tell us we got it right?**

If you can answer all three from memory, the stage is mastered.

## References

- The 9 actionable plans: [`../`](../)
- The binding proposal: [`../../Sentinel_v2_Data_Module_Integration_Proposal.md`](../../Sentinel_v2_Data_Module_Integration_Proposal.md)
- Project memory: `~/.claude/projects/-home-motafeq-projects-sentinel/memory/MEMORY.md` §"Sentinel v2 Data Module Build"
- The audit (applied patches + verified facts): [`../../Sentinel_v2_Dataset_Proposal.md`](../../Sentinel_v2_Dataset_Proposal.md) and the archived `../archive/AUDIT_PATCHES_applied_2026-06-08.md`
- The post-implementation audit for Stage 4: `Data/audit/07_verification_stage4_audit.md`
- The verification design record: `docs/decisions/ADR-0005-verification-design.md`
