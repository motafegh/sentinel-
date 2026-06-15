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
> **Status:** DRAFT — pending Ali review and sign-off on Phase A items.
> **Source-of-truth:** Verified against source code in `ml/src/inference/` and `ml/scripts/`, not against documentation.
> **Companion audit doc:** `ml/MLOPS_STATE_AND_REDESIGN_2026-06-14.md` (read this first for full audit history)

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

**Current state:** MLOps is ~55% complete and Run 12 IS in MLflow Staging but the
inference server is **not yet serving Run 12** — the FastAPI default still loads
Run 4 (F1=0.3362). Drift monitoring is **silently dead** (the detector thinks
the placeholder baseline is real, so no KS alerts ever fire).

**What we need to do:**

1. **Fix the drift detector bug** (30 min) — drift monitoring appears healthy but is dead
2. **Wire Run 12 into the API** (3 hr) — config file + env var + smoke test
3. **Build a real drift baseline** (1 hr) — collect warmup traffic, run `compute_drift_baseline.py --source warmup`
4. **Docker Compose** for the inference stack (2 hr) — unblocks any deployment
5. **Housekeeping**: resolve duplicate calibration files, update stale comments, decide DVC tracking policy (30 min)

**Effort:** ~7-8 hours of work over Q4 2026 (next 3 weeks).
**Blocks:** Production promotion of Run 12; Run 13 serving; any deployment.
**Unblocks once done:** Agent `/predict` calls can hit the real model; zkml artifacts
can be served end-to-end; agents routing calibration becomes possible.

---

## Recommended Reading Order

If you have **5 minutes:** Read TL;DR above, then skim File 4 (implementation plan).

If you have **30 minutes:** Read all 5 files in order (1 → 2 → 3 → 4 → 5).

If you're **approving this proposal:** Read File 1 (this), File 3 (redesign), File 4
(plan), File 5 (risks). Skip File 2 unless you want to verify the bug analysis.

---

## Decision Gates (where Ali sign-off is needed)

| # | Decision | File | Risk if not aligned |
|---|---|---|---|
| G1 | Approve Phase A: bug fix + housekeeping | File 4 §A | Drift monitoring stays dead |
| G2 | Approve config file approach (mlops_config.json vs env-only) | File 3 §3.2 | API re-start breaks silently |
| G3 | Approve Docker Compose scope (inference + Prometheus only, Grafana deferred) | File 3 §3.4 | Scope creep, delays Phase C |
| G4 | Approve NOT using training data for baseline (synthetic bridge) | File 5 §R3 | False alerts, "alert fatigue" |
| G5 | Confirm MLOps stays in `ml/` (not top-level `mlops/`) | File 3 §3.1 | Larger refactor than value justifies |

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
