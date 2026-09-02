# SENTINEL P1.0 Current-Document Truth-Alignment Plan

**Date:** 2026-09-02  
**Status:** READY  
**Parent:** `2026-09-02_P0_PORTFOLIO_READINESS_AUDIT.md` / B-001  
**Scope:** current-facing documentation only; no product, DATA/ML semantic, artifact, or architecture implementation change

## Goal

Close portfolio blocker B-001 by making every prominent document that presents itself as current agree with the September 2 R4 authority.

## Authority used for this pass

1. `CLAUDE.md` current stable baseline and restart order;
2. `docs/plan/ml-R4/PLAN_STATUS_MATRIX.md`;
3. `docs/handbook/16_current_status.md`;
4. R4-D-011 / `ADR-R4-011-v10-v26-physical-representation-acceptance.md`;
5. R4-D-012 / `ADR-R4-012-target-aware-guarded-selector-promotion.md`.

## Current facts that all patched documents must preserve

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

## Files in scope

1. `README.md`
2. `docs/handbook/00_README.md`
3. `docs/handbook/01_architecture.md`
4. `data_module/README.md`
5. `ml/README.md`
6. `contracts/README.md`
7. `zkml/README.md`

`agents/README.md` is already substantially current. Change it only if a cross-link or wording dependency requires it.

## Change rule

This is a **truth-alignment pass, not the final portfolio README redesign**.

- Correct stale milestone/state assertions.
- Correct stale physical representation descriptions (`v9` vs accepted V10 V2.6/current-next guarded lineage boundary).
- Keep useful architecture/trust explanations.
- Keep historical facts when clearly labeled historical.
- Do not yet add marketing copy, screenshots, badges, licensing text, demo scaffolding, or broad folder reorganization.
- Do not edit historical plans/evidence/ADRs to make them look current.

## Validation

After edits:

1. search current-facing files for stale phrases such as `through G6`, `Phase 7 must pass`, `Phase 8 is now the next`, and statements that present v9 as the future/new-training physical schema;
2. compare all current-state sections against `PLAN_STATUS_MATRIX.md`, R4-D-011, R4-D-012, and `16_current_status.md`;
3. ensure no new quality/negative/calibration/production claims were introduced;
4. run/review the handbook static check; if it remains phrase-only for these facts, record that as P5 work rather than treating green CI as proof of semantic freshness.

## Exit gate

B-001 closes only when a recruiter or engineer can move from root README → handbook → DATA/ML/contracts/ZKML module READMEs without encountering contradictory current R4 state.
