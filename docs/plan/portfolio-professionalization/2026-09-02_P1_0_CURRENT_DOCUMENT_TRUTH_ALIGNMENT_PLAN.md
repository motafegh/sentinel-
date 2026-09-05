# SENTINEL P1.0 Current-Document Truth-Alignment Record

**Date:** 2026-09-02  
**Last reconciled:** 2026-09-05  
**Status:** **COMPLETE — B-001 CLOSED**  
**Parent:** `2026-09-02_P0_PORTFOLIO_READINESS_AUDIT.md` / B-001  
**Live program status:** [`CURRENT_STATUS.md`](CURRENT_STATUS.md)  
**Scope:** current-facing documentation only; no product, DATA/ML semantic, artifact, or architecture implementation change

## Goal

Close portfolio blocker B-001 by making prominent documents that present themselves as current agree with the September 2 R4 authority.

This file is now a completed phase record. It must not be used as the live “what should we do next?” surface; use `CURRENT_STATUS.md` for that.

## Authority used

1. `CLAUDE.md` current stable baseline and restart order;
2. `docs/plan/ml-R4/PLAN_STATUS_MATRIX.md`;
3. `docs/handbook/16_current_status.md`;
4. R4-D-011 / `ADR-R4-011-v10-v26-physical-representation-acceptance.md`;
5. R4-D-012 / `ADR-R4-012-target-aware-guarded-selector-promotion.md`.

## Current facts preserved by the pass

- historical R4 G0–G7 remain PASSED and immutable;
- Phase 8 is `IN_PROGRESS`; G8 is open;
- R4-D-008 repaired-v2 physical DATA remains accepted historical/reproducibility evidence;
- R4-D-009 logical V3 remains accepted grouping/role authority;
- R4-D-010 preserves graph schema v9 for history but makes it ineligible for a new full training run;
- R4-D-011 accepts the exact 22,540-identity V10 V2.6 physical representation lineage, digest `d9f925588913e66476cfbc097bace7daa7e673295fe2a243760313d0bef5ebdd`;
- R4-D-012 promotes `target_aware_guarded_v1` only for construction/evaluation of a new versioned candidate; R4-D-011 remains immutable/current physical authority until that candidate is separately accepted;
- confirmed negatives remain zero; candidate #2 has primary support only and still requires independent agreement;
- threshold fitting, calibration fitting, untouched acceptance, model-quality claims, G8, and the 100-epoch/full training run remain unauthorized;
- Run12 remains the historical operational teacher baseline;
- AGENTS gateway completion remains off-chain; audit MCP remains read-only;
- production signing/broadcast is not implemented/claimed;
- retained EZKL proof proves proxy computation only and `check_mode="UNSAFE"` remains a production-assurance limitation.

## Files aligned

1. `README.md`;
2. `docs/handbook/00_README.md`;
3. `docs/handbook/01_architecture.md`;
4. `data_module/README.md`;
5. `ml/README.md`;
6. `contracts/README.md`;
7. `zkml/README.md`.

`agents/README.md` was reviewed and left unchanged because its off-chain/read-only/failure-boundary language was already aligned.

## Change rule used

This was a truth-alignment pass, not the final public README redesign:

- corrected stale milestone/state assertions;
- corrected stale physical-representation descriptions;
- preserved useful architecture/trust explanations;
- preserved historical facts only when clearly labeled historical;
- did not edit protected historical R4 plans/evidence/ADRs;
- did not invent model-quality, negative-truth, threshold/calibration, production-signer, or broader ZK claims.

## Validation result

The aligned current-facing surfaces agreed on the R4-D-011/R4-D-012 boundary, removed stale G6/G7-era “current” assertions, and preserved historical v9 only as historical/reproducibility context.

The final P1.0 documentation head passed the existing `Handbook` workflow. The broader portfolio program later rebuilt the root README as a recruiter/senior-engineer landing page while preserving these same technical boundaries; that later README head also passed `Handbook` and `SENTINEL system alignment`.

A green handbook check is still not treated as proof of every semantic statement because parts of currentness validation remain phrase-based; that is a P5 responsibility.

## Exit gate

**PASSED. B-001 is CLOSED.**

A reviewer can move from root README → handbook → DATA/ML/contracts/ZKML module READMEs without encountering the original contradictory current R4 milestone/representation state.

## Current routing

Do **not** restart P1.0 or route automatically to the old P1.1 checklist. The live program restart is recorded in [`CURRENT_STATUS.md`](CURRENT_STATUS.md); as of the 2026-09-05 reconciliation, P3 canonical architecture/trust presentation is the next major phase after this state-reconciliation pass.
