# SENTINEL Portfolio Professionalization — Current Status

**Last reconciled:** 2026-09-06  
**Branch:** `portfolio/professionalization-2026-09-02`  
**PR:** #72  
**Role:** canonical live status for the portfolio-professionalization program

This file is the live execution/status surface for the portfolio program. Dated audit/phase files remain evidence of what was observed or planned at that time; they must not be read as current repository state when this file records a later disposition.

The portfolio program remains subordinate to `CLAUDE.md`, executable source/config/tests, current R4 machine-readable authority, accepted ADRs/evidence, and the canonical handbook. Nothing here grants DATA/ML training, model-quality, production, signer/broadcaster, or expanded ZK authority.

## Current technical truth that portfolio work must preserve

- Historical R4 G0–G7 remain PASSED and immutable.
- Phase 8 is `IN_PROGRESS`; G8 is open.
- R4-D-011 accepts the exact V10 V2.6 physical representation lineage and digest `d9f925588913e66476cfbc097bace7daa7e673295fe2a243760313d0bef5ebdd`.
- R4-D-012 permits `target_aware_guarded_v1` only for a fresh versioned candidate requiring separate physical acceptance.
- Confirmed negatives remain zero; candidate #2 has primary-review support only and still requires genuinely independent agreement.
- Threshold fitting, calibration fitting, untouched acceptance, repaired model-quality promotion, and full Phase-8 training remain unsupported/unauthorized.
- Run12 remains the historical operational ML baseline.
- Gateway/LangGraph completion is off-chain; the live audit MCP is read-only.
- A production signer/broadcaster is not claimed.
- The retained EZKL proof proves the compact proxy computation only; V3 policy/context attestation is separate, and retained `check_mode="UNSAFE"` remains a production-assurance limitation.

## Program progress

| Phase / item | Status | Current disposition |
|---|---|---|
| **P0 readiness audit** | **COMPLETE** | Baseline audit captured and findings dispositioned. |
| **B-001 current-doc truth alignment** | **CLOSED** | Root/handbook/module current-facing docs aligned to current R4 authority. |
| **P1 repository hygiene foundation** | **SUBSTANTIALLY COMPLETE** | Runtime/DVC cruft, ignore rules, machine-local DVC remote, stale PRs/branches, security policy, developer contract, repository-weight policy, dependency locking, and repository scanning controls addressed. GitHub metadata + rename/license decisions remain. |
| **M-003 SECURITY.md** | **CLOSED** | Public security/reporting policy added without invented SLA, bounty, or private contact. |
| **M-004 DVC/artifact semantics** | **CLOSED at public-contract level** | Machine-local default remote removed; two DVC contexts and fresh-clone limitations documented. |
| **M-005 runtime/ignore hygiene** | **CLOSED** | Root `.dvc/tmp` tracked runtime files removed and ignore rules hardened. |
| **M-006 environment contract** | **CLOSED at current portfolio scope** | `DEVELOPMENT.md` defines the multi-environment monorepo; root pytest scope corrected; DATA has a committed Poetry 2.1.3-generated lock enforced by read-only zero-drift CI. Heavy/local artifact availability remains a separate documented boundary. |
| **M-010 stale PR/branch hygiene** | **CLOSED** | Obsolete PRs closed; obsolete remote branches removed. |
| **M-011 repository size/history policy** | **CLOSED** | Current-tree audit complete; historical Git storage distinguished from current artifacts; no history rewrite authorized. |
| **M-012 module README truth alignment** | **CLOSED for audited surfaces** | DATA/ML/contracts/ZKML current-state sections aligned; AGENTS already aligned. |
| **M-014 credential/security hygiene** | **CLOSED at repository-control scope / EXTERNAL RELEASE ACTION OPEN** | Current tree and baseline-aware full-history scan both pass. Four reviewed historical blobs contain one provider-RPC credential-shaped endpoint; new occurrences remain blocking. External revocation/rotation status cannot be proven from Git and must be confirmed before release if not already handled. |
| **P2 root README / public landing page** | **COMPLETE at current scope** | Recruiter/senior-engineer landing page, architecture summary, limitations, setup/validation navigation, AI-assisted ownership disclosure, showcase entry point, and case-study navigation are present. |
| **State reconciliation** | **COMPLETE** | Master/P0/P1 records reconciled, one canonical status file established, duplicate M-011 audit removed, and `CLAUDE.md` restart/memory routing corrected. |
| **P3 canonical architecture/trust presentation** | **COMPLETE / VALIDATED** | Four canonical architecture views established and current-facing DATA/ML seams reconciled. |
| **P4 bounded showcase / demo** | **COMPLETE / VALIDATED** | `SHOWCASE.md` + `tools/showcase_sentinel.py` provide a fresh-clone standard-library boundary demo with dedicated CI. |
| **P5 CI/testing/security/reproducibility presentation** | **COMPLETE / VALIDATED** | Two-layer handbook/R4 semantic validation, `VALIDATION.md`, committed DATA lock + strict CI, and current/history secret scanning are implemented and validated. |
| **P6 technical case study/evidence package** | **IN PROGRESS** | Case-study format/index established and Case Study 01 — `Unknown Is Not Negative` — published from accepted R4 policy/evidence. Further cases remain separate coherent chunks. |
| **P7 GitHub identity/release surface** | **PENDING** | Description/topics, explicit rename/license decisions, external provider-credential revocation/rotation confirmation, first stable portfolio release/tag, optional social preview. |
| **P8 final CV/interviewer audit** | **PENDING** | Recruiter skim, engineer audit, adversarial credibility pass, then derive CV wording. |

## P5 validation / reproducibility / security closure

P5 closed three earlier presentation/control gaps without redefining historical R4 evidence.

### Current vs historical authority validation

The handbook has two explicit machine-check layers:

1. `docs/handbook/tools/verify_handbook.py` preserves structural/source and historical G6/G7/runtime-compatibility checks;
2. `docs/handbook/tools/verify_current_r4.py` validates `_meta/current_r4.json` against current logical-V3 evidence, D-011 physical acceptance, D-012 selector/ADR evidence, Phase-8 hold state, and current-facing docs.

The current-R4 prose matcher is semantic/case-insensitive rather than forcing presentation capitalization to become authority.

### Public validation story

Root [`VALIDATION.md`](../../../VALIDATION.md) states what each current CI surface proves/does not prove, distinguishes normal PR checks from retained `r4-phase*` research/evidence workflows, provides the module/heavy validation matrix, and preserves `PASS` / `NOT_RUN` / `unsupported` / `unauthorized` semantics.

### DATA lock reproducibility

`data_module/poetry.lock` is committed from exact Poetry 2.1.3 generation. `.github/workflows/data-reproducibility.yml` is read-only: it regenerates the resolution and fails if the lockfile is untracked or changes.

The lock closes dependency-resolution reproducibility only; it does not make protected/local R4 physical artifacts available.

### Secret scanning and historical finding

`tools/security/scan_repository_secrets.py` provides dependency-free high-signal scans for current tracked files and reachable Git history.

The full P5 baseline established:

- current tracked tree: PASS;
- reachable blobs scanned: **92,243**;
- blob content scanned: about **3.36 GB**;
- reviewed historical findings: **4 exact blob identities**;
- finding class: one provider-RPC credential-shaped endpoint repeated in obsolete ZKML helper/generated shell material.

The exact identities are recorded in `tools/security/known_history_findings.json`; no credential value is reproduced in current documentation. The baseline-aware full-history rerun passed and new occurrences remain blocking. Current-tree scanning stays normal PR CI; full-history scanning runs on `main`, schedule, or manual execution.

External provider credential revocation/rotation cannot be established from repository evidence and remains a P7 release prerequisite. No history rewrite is authorized merely to remove those old identities.

### Exact P5 validation boundary

P5 was formally closed on commit `550851eaaedb1e3f7b3cf17cb0f09f4efcf39661` after all five current pull-request workflows completed successfully:

- Handbook;
- Portfolio showcase;
- DATA reproducibility;
- SENTINEL system alignment;
- Security hygiene.

Later P6 documentation commits must continue to pass applicable current checks; the P5 closure claim refers to the exact validated P5 boundary above.

## P6 technical case-study boundary

P6 does not rewrite R4 history into marketing copy. It curates a small number of distinct engineering decisions for external technical review while keeping primary authority in source/tests, machine-readable evidence, the handbook, and ADRs.

The common case-study structure is:

1. decision in one sentence;
2. problem;
3. evidence;
4. shortcut rejected;
5. decision;
6. implementation and validation;
7. result;
8. remaining limitation;
9. evidence trail.

Published so far:

1. [`Unknown Is Not Negative`](../../case-studies/01_unknown_is_not_negative.md) — explains why historical zero/source absence/unsupported state cannot become negative truth, why `target=0` requires class-specific confirmed-negative evidence, and why the project refuses conventional binary evaluation claims while confirmed negatives remain zero.

The package index is [`docs/case-studies/README.md`](../../case-studies/README.md). Additional cases remain pending and should be added one coherent responsibility at a time.

## Remaining P0 MUST-item disposition

| ID | Status | Remaining responsibility |
|---|---|---|
| M-001 public README | **CLOSED at current scope** | Reassess only if later P6/P7 material warrants a small landing-page refinement. |
| M-002 GitHub identity | **OPEN** | Description/topics; explicit owner decision on repo name and license. Homepage only if a real destination exists. |
| M-003 security policy | **CLOSED** | — |
| M-004 DVC/artifact contract | **CLOSED at current scope** | Future public heavy-artifact distribution is optional/separate and must be hash/version bound. |
| M-005 runtime/ignore hygiene | **CLOSED** | — |
| M-006 environment contract | **CLOSED at current portfolio scope** | Heavy/local artifact prerequisites remain explicit operational limitations. |
| M-007 lightweight showcase | **CLOSED** | Fresh-clone boundary showcase + dedicated CI validation added. |
| M-008 canonical architecture | **CLOSED** | P3 validated. |
| M-009 CI presentation/currentness | **CLOSED / VALIDATED** | P5 separates G6/G7 compatibility validation from current D-009/D-011/D-012 semantic authority and publishes the validation matrix. |
| M-010 stale PR hygiene | **CLOSED** | — |
| M-011 size/history policy | **CLOSED** | Optional future object-level history inventory only if size becomes operationally blocking. |
| M-012 module README truth | **CLOSED for audited surfaces** | Re-check only when later changes create contradictions. |
| M-013 stable release | **OPEN** | P7 after earlier gates. |
| M-014 secret hygiene | **CLOSED AT REPOSITORY-CONTROL SCOPE** | External provider-credential revocation/rotation confirmation remains a P7 release prerequisite. |

## Public/GitHub identity decisions still intentionally unresolved

These are not technical-cleanup defaults:

1. keep or rename repository `sentinel-`;
2. choose a license or intentionally remain unlicensed for now;
3. set repository description/topics when repository-setting write access is available;
4. decide whether a homepage/social-preview is useful after the public case-study/release surface stabilizes.

## Next execution order

Current default sequence:

`continue P6 case studies → P7 GitHub identity/release → P8 final portfolio audit → merge PR #72 to main`

Do not merge PR #72 merely because an intermediate phase passes. Merge only after the professionalization program reaches a coherent final validation boundary.
