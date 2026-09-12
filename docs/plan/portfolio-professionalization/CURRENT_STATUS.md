# SENTINEL Portfolio Professionalization — Current Status

**Last reconciled:** 2026-09-12  
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
- The public contribution boundary must remain explicit: the original Sentinel era was Ali's long-running AI-assisted learning/building work; the later R4 continuation is substantially AI-led research under Ali's direction and must not be represented as independent authorship/mastery of every current subsystem.

## Program progress

| Phase / item | Status | Current disposition |
|---|---|---|
| **P0 readiness audit** | **COMPLETE** | Baseline audit captured and findings dispositioned. |
| **B-001 current-doc truth alignment** | **CLOSED** | Root/handbook/module current-facing docs aligned to current R4 authority. |
| **P1 repository hygiene foundation** | **SUBSTANTIALLY COMPLETE / SMALL CLEANUP REOPENED** | Core hygiene work remains complete. Three accidental no-work remote refs created during connector operation now require deletion before final branch-hygiene closure. GitHub metadata + rename/license decisions remain P7 work. |
| **M-003 SECURITY.md** | **CLOSED** | Public security/reporting policy added without invented SLA, bounty, or private contact. |
| **M-004 DVC/artifact semantics** | **CLOSED at public-contract level** | Machine-local default remote removed; two DVC contexts and fresh-clone limitations documented. |
| **M-005 runtime/ignore hygiene** | **CLOSED** | Root `.dvc/tmp` tracked runtime files removed and ignore rules hardened. |
| **M-006 environment contract** | **CLOSED at current portfolio scope** | `DEVELOPMENT.md` defines the multi-environment monorepo; root pytest scope corrected; DATA has a committed Poetry 2.1.3-generated lock enforced by read-only zero-drift CI. Heavy/local artifact availability remains a separate documented boundary. |
| **M-010 stale PR/branch hygiene** | **REOPENED — SMALL REF CLEANUP** | Earlier obsolete PR/branch cleanup remains valid. Delete accidental refs `tmp-should-not-create`, `noop`, and `ignore-me`; they contain no unique work and are not part of PR #72. |
| **M-011 repository size/history policy** | **CLOSED** | Current-tree audit complete; historical Git storage distinguished from current artifacts; no history rewrite authorized. |
| **M-012 module README truth alignment** | **CLOSED for audited surfaces** | DATA/ML/contracts/ZKML current-state sections aligned; AGENTS already aligned. |
| **M-014 credential/security hygiene** | **CLOSED at repository-control scope / EXTERNAL RELEASE ACTION OPEN** | Current tree and baseline-aware full-history scan both pass at the P5 boundary. Four reviewed historical blobs contain one provider-RPC credential-shaped endpoint; new occurrences remain blocking. External revocation/rotation status cannot be proven from Git and must be confirmed before release if not already handled. |
| **P2 root README / public landing page** | **COMPLETE at current scope** | Recruiter/senior-engineer landing page, architecture summary, limitations, setup/validation navigation, explicit project-era/AI-assistance contribution boundary, showcase entry point, and case-study navigation are present. |
| **State reconciliation** | **COMPLETE** | Master/P0/P1 records reconciled, one canonical status file established, duplicate M-011 audit removed, `CLAUDE.md` restart/memory routing corrected, and the later main README contribution-boundary change merged into the portfolio branch. |
| **P3 canonical architecture/trust presentation** | **COMPLETE / VALIDATED AT P5 BOUNDARY** | Four canonical architecture views established and current-facing DATA/ML seams reconciled. |
| **P4 bounded showcase / demo** | **COMPLETE / VALIDATED AT P5 BOUNDARY** | `SHOWCASE.md` + `tools/showcase_sentinel.py` provide a fresh-clone standard-library boundary demo with dedicated CI. |
| **P5 CI/testing/security/reproducibility presentation** | **COMPLETE / VALIDATED** | Two-layer handbook/R4 semantic validation, `VALIDATION.md`, committed DATA lock + strict CI, and current/history secret scanning are implemented and validated at the exact P5 closure commit. |
| **P6 technical case study/evidence package** | **CONTENT COMPLETE / CURRENT-HEAD AUTOMATED VALIDATION PENDING** | Seven-case evidence-first technical-review package complete and indexed. Source/evidence review and branch/main reconciliation are complete; fresh Actions results have not yet been observed for the latest connector-authored commits. |
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

P6 does not rewrite R4 history into marketing copy. It curates distinct engineering decisions for external technical review while keeping primary authority in source/tests, machine-readable evidence, the handbook, and ADRs.

The common case-study structure remains:

1. decision in one sentence;
2. problem;
3. evidence;
4. shortcut rejected;
5. decision;
6. implementation and validation;
7. result;
8. remaining limitation;
9. evidence trail.

The intended seven-case core is now published:

1. [`Unknown Is Not Negative`](../../case-studies/01_unknown_is_not_negative.md) — preserve unknown/unsupported state rather than manufacture negative truth.
2. [`Leakage Grouping Was an Evidence Problem`](../../case-studies/02_leakage_grouping_was_an_evidence_problem.md) — make dataset identity/grouping an evidence contract.
3. [`Version the Representation Instead of Patching History`](../../case-studies/03_version_the_representation_instead_of_patching_history.md) — preserve v9 history while moving corrected call semantics into V10.
4. [`Fail Closed on Unexplained Structural Drift`](../../case-studies/04_fail_closed_on_unexplained_structural_drift.md) — keep physical acceptance false until the exact candidate's transitions are reconciled.
5. [`Promote a Selector Without Rewriting the Accepted Lineage`](../../case-studies/05_promote_a_selector_without_rewriting_the_accepted_lineage.md) — separate selector evidence, physical lineage acceptance, and training authority.
6. [`Tool Silence Is Not a Clean Result`](../../case-studies/06_tool_silence_is_not_a_clean_result.md) — keep unavailable/degraded execution distinct from a successful zero-finding result.
7. [`A Valid Proof Does Not Prove the Whole Audit`](../../case-studies/07_a_valid_proof_does_not_prove_the_whole_audit.md) — separate proxy proof, V3 context attestation, registry persistence, and transaction authority.

The package index is [`docs/case-studies/README.md`](../../case-studies/README.md). The detailed closeout is [`2026-09-12_P6_TECHNICAL_CASE_STUDY_CLOSEOUT.md`](2026-09-12_P6_TECHNICAL_CASE_STUDY_CLOSEOUT.md).

### P6 branch/main reconciliation

`main` advanced during P6 with README commit `b0c0e031b35785977a8be2c59cee3dff9f195d22`, adding the project-era contribution boundary. Because the portfolio branch also changed README, PR #72 temporarily became conflicted.

Merge commit `bd58f79289e47fd0afbaf81a3b77cd47e4bf1d5c` reconciled the two lines without force-pushing. The resulting README retains the portfolio presentation and the newer contribution disclosure. The portfolio branch is now ahead of current `main` and zero commits behind.

### P6 current-head validation state

The new P6 files were source/evidence reviewed and the case-study directory was re-enumerated from GitHub. However, GitHub Actions did not start for the connector-authored P6 commits, and a separate local fresh-clone validation attempt was blocked by DNS/network availability in the execution environment.

Therefore do not call the latest P6 head CI-validated yet. Fresh applicable checks remain a pre-release/final-merge requirement.

## Remaining P0 MUST-item disposition

| ID | Status | Remaining responsibility |
|---|---|---|
| M-001 public README | **CLOSED at current scope** | Reassess only if P7/P8 reveals a material public-landing issue. |
| M-002 GitHub identity | **OPEN** | Description/topics; explicit owner decision on repo name and license. Homepage only if a real destination exists. |
| M-003 security policy | **CLOSED** | — |
| M-004 DVC/artifact contract | **CLOSED at current scope** | Future public heavy-artifact distribution is optional/separate and must be hash/version bound. |
| M-005 runtime/ignore hygiene | **CLOSED** | — |
| M-006 environment contract | **CLOSED at current portfolio scope** | Heavy/local artifact prerequisites remain explicit operational limitations. |
| M-007 lightweight showcase | **CLOSED** | Fresh-clone boundary showcase + dedicated CI validation added. |
| M-008 canonical architecture | **CLOSED** | P3 validated at the P5 validation boundary. |
| M-009 CI presentation/currentness | **CLOSED / VALIDATED AT P5 BOUNDARY** | P5 separates G6/G7 compatibility validation from current D-009/D-011/D-012 semantic authority and publishes the validation matrix. Latest-head checks still need to run before release/merge. |
| M-010 stale PR/branch hygiene | **REOPENED — CLEANUP REQUIRED** | Delete `tmp-should-not-create`, `noop`, and `ignore-me`; no unique work is stored there. |
| M-011 size/history policy | **CLOSED** | Optional future object-level history inventory only if size becomes operationally blocking. |
| M-012 module README truth | **CLOSED for audited surfaces** | Re-check only when later changes create contradictions. |
| M-013 stable release | **OPEN** | P7 after current-head validation and release decisions. |
| M-014 secret hygiene | **CLOSED AT REPOSITORY-CONTROL SCOPE** | External provider-credential revocation/rotation confirmation remains a P7 release prerequisite. |

## Public/GitHub identity decisions still intentionally unresolved

These are not technical-cleanup defaults:

1. keep or rename repository `sentinel-`;
2. choose a license or intentionally remain unlicensed for now;
3. set repository description/topics when repository-setting write access is available;
4. decide whether a homepage/social-preview is useful after the public case-study/release surface stabilizes.

## Next execution order

Current default sequence:

`current-head validation + accidental-ref cleanup → P7 GitHub identity/release preparation and owner decisions → P8 final portfolio audit → final green validation → merge PR #72 to main`

Do not merge PR #72 merely because an intermediate phase passes. Merge only after the professionalization program reaches a coherent final validation boundary.
