# SENTINEL P0 Portfolio Readiness Audit

**Original audit date:** 2026-09-02  
**Last reconciled:** 2026-09-05  
**Status:** **P0 COMPLETE — historical baseline; dispositions reconciled below**  
**Branch:** `portfolio/professionalization-2026-09-02`  
**Parent plan:** `2026-09-02_SENTINEL_PORTFOLIO_PROFESSIONALIZATION_MASTER_PLAN.md`  
**Live program status:** [`CURRENT_STATUS.md`](CURRENT_STATUS.md)

## 1. How to read this file

This document records the **P0 baseline audit** that started the portfolio-professionalization program. Statements in the original findings describe what was true when the audit was performed on 2026-09-02.

Do **not** treat an old finding below as current repository state without checking its disposition table. The canonical live portfolio-program status is `CURRENT_STATUS.md`.

The audit remains subordinate to `CLAUDE.md`, executable source/config/tests, current R4 machine-readable authority, accepted ADRs/evidence, and the canonical handbook. It does not grant any new DATA/ML training, model-quality, production, signer/broadcaster, or ZK authority.

## 2. Original P0 verdict

At audit time SENTINEL was technically substantial enough to be a strong portfolio project, but the repository was **not yet CV-ready**. The dominant problem was not engineering depth; it was mismatch between current technical authority and the public/repository surface.

Major baseline problems included:

- current-facing documentation lagging behind Phase-8 / R4-D-011 / R4-D-012 authority;
- machine-local DVC configuration and tracked runtime state;
- weak public identity/security/developer-entry surfaces;
- a multi-environment monorepo that was not clearly explained;
- stale PR/branch surface;
- repository-size/history ambiguity;
- no bounded public showcase;
- CI/currentness presentation that mixed historical evidence workflows with normal project health.

**Original P0 verdict:** `NOT_PORTFOLIO_READY_YET`, with no architectural redesign required.

That verdict was a starting condition, not a permanent project label.

## 3. Authority used by the audit

When facts conflicted, P0 used the project authority order:

1. executable source/config/tests;
2. current machine-readable R4 governance/evidence;
3. canonical handbook;
4. ADRs/registers;
5. historical/supplementary documentation.

Current DATA/ML truth was anchored to `docs/handbook/16_current_status.md`, `docs/plan/ml-R4/PLAN_STATUS_MATRIX.md`, and the R4-D-010/011/012 decision chain.

## 4. Original blocker

### B-001 — public entry documentation materially contradicted current R4 authority

At P0, the root README and several current-facing handbook/module README sections still described older G6/G7-era state as current while actual authority had advanced to:

- historical G0–G7 PASSED/immutable;
- Phase 8 `IN_PROGRESS`;
- R4-D-011 exact V10 V2.6 physical acceptance;
- R4-D-012 guarded-selector authorization only for a fresh successor candidate;
- full repaired training still unauthorized.

**Current disposition:** **CLOSED.** The bounded P1.0 truth-alignment pass updated the root README, handbook entry/architecture, DATA/ML/contracts/ZKML READMEs, and preserved AGENTS where it was already aligned. The later public README redesign kept those claim boundaries intact.

## 5. MUST findings — reconciled disposition

### M-001 — rebuild the root README as a public landing page

**Original finding:** root README was stale and internally oriented.

**Current disposition:** **SUBSTANTIALLY COMPLETE.** It now leads with problem/value, implemented capabilities, a concise architecture, engineering highlights, technology stack, current limitations, developer/evidence links, and explicit AI-assisted ownership. The runnable showcase/example output remains P4 rather than being faked inside the README.

### M-002 — establish an intentional GitHub repository identity

**Original finding:** description/topics/homepage/license/release were unset and repository name needed intentional review.

**Current disposition:** **OPEN.** Repository name and license remain explicit owner decisions. Description/topics still need to be set when repository-settings write access is available. Homepage/social preview should wait for a real destination/stable public architecture.

### M-003 — add a public security policy

**Original finding:** `SECURITY.md` absent.

**Current disposition:** **CLOSED.** Root `SECURITY.md` now defines supported reporting scope, sensitive-report handling, project/research boundaries, and avoids invented SLA/bounty/private-contact claims.

### M-004 — repair DVC and artifact-retrieval semantics

**Original finding:** root DVC used a machine-local `/mnt/d/...` remote and repository artifact boundaries were unclear.

**Current disposition:** **CLOSED at current public-contract scope.** Machine-local default remote was removed from tracked config. Root and `data_module/` DVC contexts, local/private artifact expectations, fresh-clone limitations, and safe local-remote configuration are documented. No claim is made that current heavy R4/Run12/proving artifacts are universally downloadable or reproducible from a fresh clone.

### M-005 — stop tracking runtime/DVC cruft and repair ignore rules

**Original finding:** `.dvc/tmp` lock/runtime files tracked and ignore rules accumulated residue.

**Current disposition:** **CLOSED.** Tracked DVC runtime files were removed; `.dvc/tmp`, local DVC config, environment/credential material, caches, and generated ML model/checkpoint classes are appropriately ignored; the required R4 review ZIP exception remains preserved.

### M-006 — define the monorepo/development environment contract

**Original finding:** root/ML/DATA/AGENTS/Contracts/ZKML setup boundaries were confusing and root pytest metadata was misleading.

**Current disposition:** **SUBSTANTIALLY COMPLETE.** Root `DEVELOPMENT.md` defines the actual multi-environment monorepo, module-owned setup/test paths, DVC/artifact boundaries, fresh-clone expectations, and validation commands. Root pytest scope was corrected and DATA no longer forces a regional primary package index. DATA still lacks a committed Poetry lockfile; that remains a P5 reproducibility item.

### M-007 — create one lightweight reproducible showcase path

**Original finding:** no clear five-minute demo/showcase.

**Current disposition:** **OPEN — P4.** Must demonstrate or honestly replay a bounded Solidity audit/example without requiring full historical DATA, multi-day training, production credentials, or fake success.

### M-008 — produce one canonical current architecture/trust-boundary presentation

**Original finding:** architecture was strong but distributed and current-state presentation needed consolidation.

**Current disposition:** **OPEN / NEXT — P3.** README now has a high-level diagram; the deeper authoritative architecture/trust views still need consolidation so one canonical view exists per question.

### M-009 — reorganize CI presentation and strengthen documentation freshness validation

**Original finding:** many phase-specific workflows appear current and parts of handbook validation are phrase-based.

**Current disposition:** **OPEN — P5.** Latest professionalization head has green Handbook and system-alignment checks, but current-vs-historical workflow presentation and semantic currentness checks still need improvement.

### M-010 — resolve stale open PR hygiene

**Original finding:** six obsolete May/June PRs remained open.

**Current disposition:** **CLOSED.** Obsolete PRs were reviewed/closed; obsolete remote branches were subsequently removed. Only `main` and the current professionalization branch are intended to remain during this work.

### M-011 — audit repository size and artifact/history policy

**Original finding:** repository size around 400 MB required explanation before portfolio release.

**Current disposition:** **CLOSED.** Current-tree audit shows compact active evidence/artifacts; repository weight is primarily historical Git storage. Protected evidence is retained, future heavy model/data artifacts stay out of Git, partial clone is documented, and history rewriting is explicitly rejected unless a separate evidence-preserving migration is justified.

Canonical audit: `2026-09-04_REPOSITORY_WEIGHT_AND_ARTIFACT_AUDIT.md`.

### M-012 — targeted module README truth alignment

**Original finding:** DATA/ML/contracts and parts of ZKML were stale; AGENTS was substantially current.

**Current disposition:** **CLOSED for the audited current-facing surfaces.** Future changes should re-open only concrete contradictions, not trigger broad ceremonial rewrites.

### M-013 — add a stable portfolio release after cleanup

**Original finding:** no releases.

**Current disposition:** **OPEN — P7.** Do not create the release until architecture/showcase/CI/evidence presentation reaches a coherent stable boundary.

### M-014 — bounded credential/security hygiene check

**Original finding:** no obvious key literals found, but P0 search was not a full secret scan.

**Current disposition:** **PARTIAL / adequate for current phase.** A later bounded scan found no obvious tracked `.env`/PEM/key material or common credential literals; deployment reads private key from environment; ignore controls were hardened. A dedicated history/CI secret scan remains P5.

## 6. SHOULD findings — current position

| ID | Baseline responsibility | Current status |
|---|---|---|
| S-001 | contain generated/historical report clutter | open; revisit only where current navigation is materially harmed |
| S-002 | technical case study | P6 |
| S-003 | current validation matrix | P5 |
| S-004 | issue/PR ergonomics where useful | optional/conditional |
| S-005 | selective metadata normalization | partially addressed; continue only where misleading |
| S-006 | social-preview/architecture visual | P7 after P3 stabilizes |
| S-007 | surface AI-assisted engineering ownership | substantially complete in root README |
| S-008 | stale branch surface / main protection policy | obsolete branches removed; main protection policy remains separate later repository-policy work |

## 7. Protected exclusions that remain permanent

Portfolio cleanup must not destroy or rewrite for aesthetics:

- `docs/plan/ml-R4/` decisions/evidence/review bundles/hashes/manifests;
- G0–G7 historical identities and R4-D-008/009/010/011/012 evidence;
- Run12 historical lineage references;
- retained ZKML/circuit/verifier reproducibility lineage;
- V1/V2/V3 compatibility source/tests;
- historical artifacts still needed to substantiate current claims.

## 8. P0 closure result

P0 itself remains **COMPLETE**. Its purpose was to establish the baseline and classify responsibilities, not to stay as the live status document forever.

Current portfolio readiness is tracked in [`CURRENT_STATUS.md`](CURRENT_STATUS.md). The project is materially more professional than at P0, but it is **not yet final portfolio-ready** because major remaining gates include canonical architecture (P3), bounded showcase (P4), CI/reproducibility presentation (P5), case study (P6), identity/release decisions (P7), and final adversarial/CV audit (P8).
