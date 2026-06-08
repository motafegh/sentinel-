# Actionable Plans — Sentinel v2 Data Module Build

**Date:** 2026-06-08
**Owner:** SENTINEL data engineering + ML engineering
**Source proposal:** [`../Sentinel_v2_Data_Module_Integration_Proposal.md`](../Sentinel_v2_Data_Module_Integration_Proposal.md)
**Audit reference:** [`../archive/AUDIT_PATCHES_applied_2026-06-08.md`](../archive/AUDIT_PATCHES_applied_2026-06-08.md) — 30 verified facts (F1–F30) cross-checked against `~/.claude/projects/.../memory/` + live `ml/src/`. **All patches have been applied to the 9 plans below; this document is archived for reference only.**
**Build window:** Jun 9 – Aug 5, 2026 (8 weeks + 1 day launch)

---

## ⚠ CRITICAL: Schema version is v9, not v8

**The active schema (per `ml/src/preprocessing/graph_schema.py:161,175,218`, verified 2026-06-08) is v9:**

| Constant | Value | Was wrong in proposal |
|---|---|---|
| `FEATURE_SCHEMA_VERSION` | `"v9"` | Proposal §2 said `"v8"` |
| `NODE_FEATURE_DIM` | `12` | Proposal said `11` |
| `NUM_NODE_TYPES` | `14` | Proposal said `13` |
| `NUM_EDGE_TYPES` | `12` | Proposal said `11` |
| `_MAX_TYPE_ID` | `13.0` | Proposal said `12.0` |
| `gnn_num_layers` | `8` | Unchanged |
| `NUM_CLASSES` | `10` | Unchanged (class order LOCKED — see C-2 below) |

**All schema references in all 8 plans and the proposal must use v9.** The Stage 0 stub, the Stage 2 port, the Stage 7 export, and the Stage 8 Run 11 launch all use v9.

---

## ⚠ CRITICAL: 8 code bugs already fixed — do NOT re-fix

Verified against live `ml/src/` (see `AUDIT_PATCHES.md` F2–F6, F11):

| Bug | Status | Location | Plan action |
|---|---|---|---|
| **A20** label=0 hardcode | ✅ FIXED | `ast_extractor.py:290,342,395` | Stage 2 regression test must preserve |
| **A9** `now` keyword miss | ✅ FIXED | `graph_extractor.py:587-605` | Stage 2 regression test must preserve |
| **A15** def_map by name | ✅ FIXED | `graph_extractor.py:1147-1179` (scope_key two-tier) | Stage 2 regression test must preserve |
| **A34** prefix sort dim | ✅ FIXED | `sentinel_model.py:356,367` | Stage 2 regression test must preserve |
| **A38** NaN before backward | ✅ FIXED | `trainer.py` isfinite guard | (no plan change) |
| Resume overwrite | ✅ FIXED | `trainer.py:383,1184,1206,1212` (default `resume_model_only=False`) | Stage 8 launch uses full-resume default |
| `_compute_return_ignored` | ✅ FIXED | `graph_extractor.py` (checks `id()` in subsequent IR ops) | (no plan change) |
| EMITS edge bug | ⚠ OPEN (Interp-6) | `graph_extractor.py` | Stage 7 must fix during seam swap |

**3 still-open bugs:**

| Bug | Location | Plan action |
|---|---|---|
| CALL_ENTRY cross-function for external calls | `graph_extractor.py:1001` (self-loop only) | Stage 2 port preserves; full fix is post-Run-11 |
| Predictor tier threshold | `predictor.py:150,168,752` (hardcoded 0.55) | **Stage 7 seam swap must fix this** |
| 99% DoS↔Reentrancy co-occurrence | Source: BCCC folder-based labeling | Stage 3 merger must prevent |

---

## ⚠ CRITICAL: Operational facts (apply to ALL plans)

Per `~/.claude/projects/.../memory/MEMORY.md` and CLAUDE.md global rules:

| Fact | Source | Plan impact |
|---|---|---|
| **MLflow backend is `sqlite:///mlruns.db`** | MEMORY + `project_v4_analysis.md` | Stage 8 launch command — `file:///` has corrupt experiments 1,2,3 |
| **Use `wsl -- bash -c '...'` for WSL commands** | CLAUDE.md + MEMORY operational facts | All plans (PowerShell host errors on inline commands) |
| **98 solc versions pre-installed in `~/.solc-select/artifacts/`** | `reference_solc.md` | Stage 0/7 Dockerfile uses 6 baseline + on-demand for rest |
| **`TRANSFORMERS_OFFLINE=1` + `TRITON_CACHE_DIR=/tmp/triton_cache` required at shell level** | MEMORY + `ml/README.md:39-40` | Stage 0 docs + Stage 8 launch |
| **Run 7 had 30+ audit fixes before launching** (A1–A38) | `docs/pre-run-fixes/validated_audition.md` | Stage 2 regression test scope = all 9 source files, not just 4 |
| **Run 9 silent-overwrite incident** (2026-06-06) | `project_run9_resume.md` | Stage 8 — timestamped `--run-name`, watcher `F1 > 0.1` floor |
| **Run 8 watcher at `ml/scripts/run8_watcher.sh` is proven** | `project_run8_audit_findings.md` | Stage 8 — copy + repurpose, don't reinvent |

---

## Purpose

This folder contains the per-stage actionable plans that execute the integration proposal. Each plan is a design + intent document — what to build, why, and how to verify it. They are **not** code; code lives in `Data/sentinel_data/` and is produced by executing these plans.

The plans are ordered. Each stage's exit criteria are the precondition for the next stage's start.

---

## Plan index

| # | Stage | Plan file | Dates | Days | Key gate |
|---|---|---|---|---|---|
| 0 | Skeleton + Data/ restructure | [`01_stage_0_skeleton.md`](01_stage_0_skeleton.md) | Jun 9–15 | 4–5 | `poetry install` works; `sentinel-data --help` runs; config.yaml lists 17 sources |
| 1 | Ingestion + Preprocessing | [`02_stage_1_ingest_preprocess.md`](02_stage_1_ingest_preprocess.md) | Jun 16–22 | 5 | 30 ScaBench files preprocess end-to-end |
| 2 | Representation (port from ml/) | [`03_stage_2_representation.md`](03_stage_2_representation.md) | Jun 23–29 | 5 | Byte-identical regression test passes for all 9 source files (A1–A38 preserved) |
| 3 | Labeling (parsers + crosswalks) | [`04_stage_3_labeling.md`](04_stage_3_labeling.md) | Jun 30–Jul 20 | 15–20 | 17 crosswalk YAMLs + 17 parsers; 99% DoS↔Reentrancy co-occurrence de-duplicated by merger |
| 4 | Verification (BCCC-failure catcher) | [`05_stage_4_verification.md`](05_stage_4_verification.md) | Jul 21–27 | 5 | Phase 5 BCCC regression test passes (±0.5% per-class) |
| 5 | Splitting + Registry | [`06_stage_5_splitting_registry.md`](06_stage_5_splitting_registry.md) | Jul 28–Aug 3 | 5 | `load_artifact("sentinel-v2-dryrun-2026-08")` works; leakage auditor = 0 |
| 6 | Analysis | [`07_stage_6_analysis.md`](07_stage_6_analysis.md) | Aug 4–5 | 2 | `feature_dist` flags synthetic complexity skew; `complexity_proxy_risk.md` GREEN |
| 7 | Export + Seam Swap | [`08_stage_7_export_seam.md`](08_stage_7_export_seam.md) | Aug 6–17 | 8–10 | All **7** v2-readiness gates GREEN; predictor.py tier bug fixed; Docker build succeeds |
| 8 | Run 11 launch | [`09_stage_8_run11_launch.md`](09_stage_8_run11_launch.md) | Aug 18 | 1 | Run 11 starts cleanly; first-epoch val F1 logged; per-class P/R reported separately |

**Total: 50–58 working days over ~10 weeks.** (Stage 3 budget grew to 3 weeks because of the 5 new sources from friend: 17 crosswalks × 1 day avg, harder ones 2-3 days. Stages 4-8 shifted to Aug; Run 11 launches Aug 18.)

---

## How to read these plans

Each plan is structured identically:

1. **Header** — date, stage number, owner, source proposal sections, exit criteria
2. **Goal** — one-paragraph summary of what the stage delivers
3. **Why this stage** — why the stage is positioned where it is in the build
4. **Design decisions** — the D-N.M decisions that frame the implementation
5. **Tasks** — ordered list, each with: why, exit condition, commit message
6. **"What NOT to fix"** — list of bugs that are already fixed and must be preserved, not re-fixed
7. **Final exit criteria check** — the 7–16 testable conditions
8. **Risk register** — known risks and mitigations

The plans do **not** contain code. The code is produced by executing the tasks; the design decisions in each plan are the contract for what the code must do.

---

## Cross-stage dependencies

```
Stage 0 (skeleton)
   ↓
Stage 1 (ingest + preprocess)  ←─────────────────┐
   ↓                                             │
Stage 2 (representation port)                    │
   ↓ (36-issue regression test gates Stage 7)    │
Stage 3 (labeling)                               │
   ↓                                             │
Stage 4 (verification)  ←─ (BCCC regression test)│
   ↓ (gates Stage 7 export)                      │
Stage 5 (splitting + registry)                   │
   ↓                                             │
Stage 6 (analysis)  ←────────────────────────────┘
   ↓
Stage 7 (export + seam swap)  ←─ (all 7 gates, predictor fix)
   ↓
Stage 8 (Run 11 launch)
```

The arrows are hard dependencies. Stage N+1 cannot start until Stage N's exit criteria are met.

---

## Critical tests (the gates that prevent silent failure)

These tests are the structural defense against the BCCC class of failure. They are written in the stage indicated and are run in every subsequent CI:

| Test | Written in | What it prevents |
|---|---|---|
| **36-issue pre-Run-8 audit regression test** (all 9 source files) | Stage 2 | Any A1–A38 fix being lost during the port; logic change in any moved file |
| **Byte-identical regression test** (per-file) | Stage 2 | Silent logic change during the port from `ml/` to `sentinel_data/` |
| **BCCC Phase 5 regression test** (p5_s1 → p5_s6 each) | Stage 4 | Verification module being less rigorous than the ad-hoc Phase 5 scripts |
| **Dual-path seam swap test** (8 fixed bugs preserved) | Stage 7 | Loader-level difference + any fixed bug regressing during seam swap |
| **End-to-end round-trip** (CI job) | Stage 7 | Export → import breaking under realistic load |
| **7 v2-readiness gates** (final check) | Stage 7 | Any of the 7 conditions being violated |
| **Run 11 pre-launch checklist** (12 conditions) | Stage 8 | Missed launch condition (sqlite backend, timestamped run-name, etc.) |

The 36-issue regression test + the byte-identical regression test together are the proof that the move was a move, not a rewrite. The Phase 5 regression test is the proof that the new module is *at least as good* as the 14 days of BCCC debugging work.

---

## Open questions blocking the build (need Ali's nod)

| # | Question | Stage | Impact if unresolved |
|---|---|---|---|
| 1 | How is `sentinel-data` distributed to `sentinel-ml`? (path dep / PyPI / git tag) | 0 | Stage 7's `pyproject.toml` change |
| 2 | Confirm the Dockerfile base image (`python:3.12.1-bookworm` per F14, NOT slim) and solc versions | 0 | Stage 7's Docker build |
| 3 | DVC remote backend (S3 / GCS / local-only) | 0 | Stage 5's catalog (no impact on local builds) |
| 4 | Confirm the 10-class taxonomy class order matches the v1 checkpoint's class order | 3 | The v1 checkpoint may need a remap layer if order differs |
| 5 | **Schema version** — confirmed v9 (per F1) | 2 | The stub from Stage 0 must use v9; affects the regression test |
| 6 | Export shard size default (5,000 per `config.yaml` proposed) | 7 | Minor; tunable |
| 7 | Run 11 launch date (2026-08-05 proposed) | 8 | The whole build schedule |
| 8 | Should `_add_icfg_edges` cross-function external calls be added in Stage 2 (port) or post-Run-11 (v2.1)? | 2 | Affects the regression test scope; current plan = preserve partial fix |
| 9 | Should `pdg_builder.py` be shipped in v2 or deferred to v3.1? | 2 | Per AUDIT_PATCHES 2-P9, the plan defers to v3.1; confirm |

**Q5 (schema) is now confirmed v9** — the stub from Stage 0 must be updated to v9 constants before any code is written. Q8 and Q9 are recommendations from the audit that need your nod.

---

## Approval workflow

Each stage plan is reviewed by Ali before its stage starts. The review checks:
- Design decisions are consistent with the proposal
- Exit criteria are testable
- Risk register is comprehensive
- No new design decisions have been silently introduced
- "What NOT to fix" section is complete (the 8 already-fixed bugs are listed)

Reviews are recorded as `docs/decisions/ADR-NNNN-<stage>.md` (one per stage). The ADRs are append-only.

---

## Status tracking

After each stage completes, update the `## Build Summary` table in `09_stage_8_run11_launch.md` with the stage's status (e.g. `✅ Complete` or `🔴 Blocked — see risk register`).

The current status (as of 2026-06-08) is **all stages pending**. The first stage starts on 2026-06-09 after Ali's review of Stage 0.

---

**End of INDEX. Read [`AUDIT_PATCHES.md`](AUDIT_PATCHES.md) for the full audit, then start with [`01_stage_0_skeleton.md`](01_stage_0_skeleton.md).**
