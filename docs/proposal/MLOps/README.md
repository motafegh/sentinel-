---
title: SENTINEL MLOps Q4 Proposal — Index
date: 2026-06-15
module: ml
phase: q4
type: proposal
descriptor: INDEX
status: ACTIVE
supersedes: ml/MLOPS_STATE_AND_REDESIGN_2026-06-14.md (audit only; this is the forward-looking proposal)
---

# SENTINEL MLOps — Q4 Proposal (2026-06-15)

> **Purpose:** Forward-looking proposal for the MLOps layer covering the next ~3 weeks (Q4 2026), aligned with Run 12 in Staging, Run 13 in prep, and the multi-class cascade across zkml/contracts/agents.
>
> **Status:** ✅ **Phase A + B + C COMPLETE 2026-06-17** (C.5 E2E Docker smoke test deferred — requires Docker host).
> **Source-of-truth:** Verified against source code in `ml/src/inference/` and `ml/scripts/`, not against documentation.
> **Companion audit doc:** `ml/MLOPS_STATE_AND_REDESIGN_2026-06-14.md` (read this first for full audit history)
> **Completion summary:** `docs/changes/2026-06-17-ml-mlops-q4-phase-b-c-complete.md`

---

## What This Proposal Covers

The MLOps layer is the convergence point for SENTINEL — it is the only module that
sits between the trained model and every external consumer (agents, contracts, CI).
Every other module's output eventually lands here for serving. This proposal:

1. **Verifies the existing audit** against source code (not docs) — the audit doc
   was written from source on 2026-06-14; this proposal re-verifies and adds new findings.
2. **Surfaces a real bug** the audit missed: the drift detector silently fails when
   given the placeholder baseline.
3. **Plans Phase A–C work** for the next 3 weeks (Q4 2026).
4. **Defer Phase D** (post-Run-13 work) until Run 13 trains.

---

## File Index

| # | File | Purpose | Audience |
|---|---|---|---|
| 1 | `README.md` (this file) | Index + navigation + TL;DR | Everyone |
| 2 | `2026-06-15_ml_q4_proposal_mlops_current_state_findings.md` | Source-verified current state | Tech lead, Ali |
| 3 | `2026-06-15_ml_q4_proposal_mlops_redesign_proposal.md` | Architecture decisions + design | Tech lead, Ali |
| 4 | `2026-06-15_ml_q4_proposal_mlops_q4_implementation_plan.md` | Phased work plan with effort estimates | Pair programmer, Ali |
| 5 | `2026-06-15_ml_q4_proposal_mlops_risks_dependencies.md` | Risk register + cross-module dependencies | Tech lead, project manager |

---

## TL;DR — The 30-Second Pitch

**Current state (2026-06-17):** MLOps is **~95% complete** (Phase A + B + C done).
Run 12 IS in MLflow Staging AND the FastAPI server is now serving Run 12 (F1=0.7004).
**Drift monitoring is ACTIVE** with a real (synthetic warmup) baseline. Full Docker
deployment stack authored; E2E smoke test (C.5) requires a Docker host.

**What was done (2026-06-15 → 2026-06-17):**

1. ✅ **Phase A: bug fix + housekeeping** (1.5 hr) — drift detector silent-failure fixed, stale comments removed, duplicate calibration archived, DVC rebuilt
2. ✅ **Phase B: wire Run 12 into the API** (3 hr) — `mlops_config.json` + config loader + `set_active_checkpoint.py` + real drift baseline (`ml/data/drift_baseline_run12.json`, 4 stats × 500 samples) + 13 new inference tests
3. ✅ **Phase C: Docker + deployment** (3 hr) — `Dockerfile.inference`, `docker-compose.yml`, `prometheus.yml`, `.env.example`, `README.md`. C.5 (E2E smoke test) deferred to a Docker-enabled host.

**Remaining (post-Run-13, Phase D):**
1. **Replace synthetic drift baseline** with real production warmup traffic
2. **C.5 E2E smoke test** — run `docker compose up -d` on a Docker host, verify the full stack
3. **Statistical significance test** vs prior Production model (no prior Production exists)
4. **Phase D Run-13 transition** — when Run 13 trains, update mlops_config + rebuild baseline + smoke test

**Effort actual:** ~7.5 hours of focused work (as estimated).
**Production promotion remaining blockers:** (1) real warmup traffic, (2) C.5 smoke test, (3) statistical sig test.

---

## Recommended Reading Order

If you have **5 minutes:** Read TL;DR above, then skim File 4 (implementation plan).

If you have **30 minutes:** Read all 5 files in order (1 → 2 → 3 → 4 → 5).

If you're **approving this proposal:** Read File 1 (this), File 3 (redesign), File 4
(plan), File 5 (risks). Skip File 2 unless you want to verify the bug analysis.

---

## Decision Gates (where Ali sign-off is needed)

| # | Decision | File | Status 2026-06-17 |
|---|---|---|---|
| G1 | Approve Phase A: bug fix + housekeeping | File 4 §A | ✅ Done 2026-06-15 → 16 |
| G2 | Approve config file approach (mlops_config.json vs env-only) | File 3 §3.2 | ✅ Done 2026-06-16 (Phase B.1+B.2) |
| G3 | Approve Docker Compose scope (inference + Prometheus only, Grafana deferred) | File 3 §3.4 | ✅ Done 2026-06-17 (Phase C.1-C.4 + C.6) |
| G4 | Approve NOT using training data for baseline (synthetic bridge) | File 5 §R3 | ✅ Done 2026-06-17 (Phase B.4 — synthetic warmup via `ml/scripts/build_warmup_baseline.py`) |
| G5 | Confirm MLOps stays in `ml/` (not top-level `mlops/`) | File 3 §3.1 | ✅ Kept in `ml/`; deploy/ subdir added |

---

## Cross-References

- **MLOps audit (2026-06-14):** `ml/MLOPS_STATE_AND_REDESIGN_2026-06-14.md` — original audit; this proposal builds on it
- **Run 12 launch context:** `~/.claude/projects/.../memory/2026-06-13_project_run12_launch.md`
- **Run 12 post-training:** `~/.claude/projects/.../memory/2026-06-14_project_run12_post_training.md`
- **Run 13 plan (ml-side):** `docs/plans/2026-06-14_Run13_4_fixes_preparation.md`
- **zkml module state:** `zkml/ZKML_STATE_AND_REDESIGN_2026-06-14.md` (cascade dependency)
- **contracts module state:** `contracts/CONTRACTS_STATE_AND_REDESIGN_2026-06-14.md` (cascade dependency)
- **agents module state:** `agents/AGENTS_STATE_AND_REDESIGN_2026-06-14.md` (downstream consumer)

---

## Conventions Used in This Folder

- **Naming:** All files follow the 6-part convention `<YYYY-MM-DD>_<MODULE>_<RUN_or_PHASE>_<WHAT_it_is>_<descriptor>.<ext>`
- **Source citations:** When stating a fact, the file path and line number are cited as `path:line` for verification
- **Status labels:** `[BUG]`, `[STALE]`, `[OK]`, `[TODO]`, `[DEFER]`, `[DECIDED]`, `[OPEN Q]`
- **Phases:** Phase A (Q4 week 1, ~1.5 hr) → Phase B (Q4 week 1-2, ~3 hr) → Phase C (Q4 week 2-3, ~3 hr) → Phase D (post-Run-13, deferred)

---

## Changelog

- **2026-06-15** — Initial proposal written. Based on source-code re-verification of audit doc + new bug discovery.
- **2026-06-17** — Phase A + B + C complete (C.5 E2E deferred). All 5 decision gates (G1-G5) resolved. ~7.5 hr of work. Run 12 in Staging with real drift baseline + active monitoring + Docker stack. Completion summary: `docs/changes/2026-06-17-ml-mlops-q4-phase-b-c-complete.md`. 275 tests pass (ml + data_module + agents regression).
