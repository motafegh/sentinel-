# SENTINEL P1.0 Current-Document Truth-Alignment Plan

**Date:** 2026-09-02  
**Status:** COMPLETE — B-001 CLOSED  
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

1. `README.md`
2. `docs/handbook/00_README.md`
3. `docs/handbook/01_architecture.md`
4. `data_module/README.md`
5. `ml/README.md`
6. `contracts/README.md`
7. `zkml/README.md`

`agents/README.md` was reviewed and left unchanged because its current off-chain/read-only/failure-boundary language was already aligned.

## Change rule used

This remained a **truth-alignment pass, not the final portfolio README redesign**.

- Corrected stale milestone/state assertions.
- Corrected stale physical representation descriptions (`v9` vs accepted V10 V2.6/current-next guarded lineage boundary).
- Preserved useful architecture/trust explanations.
- Preserved historical facts only when clearly labeled historical.
- Did not add marketing copy, screenshots, badges, licensing text, demo scaffolding, or broad folder reorganization.
- Did not edit historical R4 plans/evidence/ADRs.

## Validation result

- Root README, handbook entry/architecture, and DATA/ML/contracts/ZKML README current-state sections now agree on the R4-D-011/R4-D-012 boundary.
- Stale assertions such as `through G6`, `Phase 7 must pass`, and `Phase 8 is now the next authorized step` were removed from the patched current-facing surfaces.
- Historical v9 is explicitly distinguished from the accepted V10 V2.6 physical lineage and the still-pending guarded-selector successor.
- No new model-quality, confirmed-negative, threshold/calibration, production-signer, or expanded-ZK-proof claim was introduced.
- PR scope remained contained to seven current-facing docs plus portfolio planning/audit records; no product source, R4 evidence, artifact, workflow, or runtime configuration changed.
- GitHub Actions `Handbook` run `33664914261` on the final documentation head passed all steps: canonical handbook static validation, active entry-point boundary assertions, validator unit tests, and inventory.
- The P0 finding that parts of the handbook CI remain phrase-based is intentionally still open for later P5 hardening; a green check is not being treated as proof of every semantic statement.

## Exit gate

**PASSED.** A reviewer can now move from root README → handbook → DATA/ML/contracts/ZKML module READMEs without encountering the previously identified contradictory current R4 milestone/representation state.

**B-001 status: CLOSED.**

## Next portfolio responsibility

Proceed to P1.1 repository hygiene and identity foundation, prioritizing non-destructive technical hygiene first:

1. DVC/tmp and `.gitignore` cleanup;
2. DVC/artifact-retrieval contract cleanup;
3. stale PR/branch containment and repository-size policy audit;
4. public security policy;
5. GitHub description/topics;
6. explicit user decision for repository rename and license before those irreversible/public-identity choices are applied.
