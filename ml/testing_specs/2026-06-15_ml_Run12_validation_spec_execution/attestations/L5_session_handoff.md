# Procedure Attestation — L_release_readiness L.5 — 2026-06-15

**Scope:** Session handoff (L.5) — not a Production release gate (L.4)  
**Current model stage:** Staging (not Production)

## L.5.1 Mandatory Writes — Status

| Item | Done? | Location |
|------|-------|----------|
| MEMORY.md Current State updated | YES | Reflects Run 12 complete, Wild eval done, manual inspection done, Run 13 plan with ExternalBug fix added |
| MEMORY.md Training History updated | YES | Run 12 row: ep50 F1_tuned=0.7004 in Staging |
| Open bugs / findings written | YES | All FIND-R12-* in C attestation + A attestation + manual inspection report |
| Architecture decisions as ADR | N/A | No new architecture decisions this session (planned Run 13 changes are label/data changes, not architecture) |
| Open questions externalised | YES — see below | |

## L.5.2 Floating Findings Check

- Any finding only in conversation? **NO** — all written to attestation files and gate_reports
- Any plan only in conversation? **NO** — Run 13 plan in `docs/plans/2026-06-14_Run13_4_fixes_preparation.md`, ExternalBug finding added to MEMORY.md
- Can a fresh session resume from written docs? **YES** — MEMORY.md Current State is complete

## L.5.3 Handoff Summary

**Completed this session:**
- Manual inspection of 9 OOD contracts (Timestamp/Reentrancy/ExternalBug) — verdict documented
- Full testing spec audit: 4 attestations written (A, C, F.3, I-Staging)
- 4 gate reports written (C1 diagnostics, I3.2 calibration, A1 contamination, I3.5 drift)
- Folder created: `ml/testing_specs/2026-06-15_ml_Run12_validation_spec_execution/`

**Open (not finished):**
- C.2.1 smoke inference on honest benchmark (66 OOD contracts) — needs a run
- K (Inference API) — `/health` endpoint + round-trip tests — not done at all
- Production promotion — blocked on real drift baseline

**Next session must read:**
- `ml/testing_specs/2026-06-15_ml_Run12_validation_spec_execution/attestations/` — status of all gates
- `MEMORY.md` Current State — Run 13 plan has 5 fixes now (ExternalBug added)
- `ml/testing_specs/K_inference_api.md` — if validating API next

**Blockers:**
- Production promotion: drift baseline placeholder + smoke not re-run
- Run 13 launch: D + E + F.1 spec pass required first

## L.4 Release Gate — Staging vs Production

| Gate | Staging | Production |
|------|---------|------------|
| Contamination check | ✅ PASS | ✅ PASS |
| Calibration files | ✅ PASS | ✅ PASS |
| F1 two-source | ✅ PASS (epoch_summary + MEMORY.md) | ✅ PASS |
| Smoke suite | ⚠️ UNVERIFIED (not re-run) | ❌ BLOCKED |
| Drift baseline (warmup) | N/A | ❌ BLOCKED — placeholder |
| API validation (K) | N/A for Staging | ❌ NOT DONE |
| Behaviour checks | PARTIAL | ❌ INCOMPLETE |

**Production gate: BLOCKED. Do not promote to Production until drift baseline and API validation complete.**

## Written to

`ml/testing_specs/2026-06-15_ml_Run12_validation_spec_execution/attestations/L5_session_handoff.md`
