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
| **P2 root README / public landing page** | **COMPLETE at current scope** | Recruiter/senior-engineer landing page, architecture summary, limitations, setup/validation navigation, AI-assisted ownership disclosure, and showcase entry point are present. |
| **State reconciliation** | **COMPLETE** | Master/P0/P1 records reconciled, one canonical status file established, duplicate M-011 audit removed, and `CLAUDE.md` restart/memory routing corrected. |
| **P3 canonical architecture/trust presentation** | **COMPLETE / VALIDATED** | Four canonical architecture views established and current-facing DATA/ML seams reconciled. Handbook + system-alignment passed on P3 head `a78ae31f44d63a28cd527cc7735edd305fd371b3`. |
| **P4 bounded showcase / demo** | **COMPLETE / VALIDATED** | `SHOWCASE.md` + `tools/showcase_sentinel.py` provide a fresh-clone standard-library boundary demo; human and JSON modes passed dedicated Portfolio showcase CI. |
| **P5 CI/testing/security/reproducibility presentation** | **COMPLETE AT REPOSITORY-CONTROL SCOPE** | Two-layer handbook/R4 semantic validation, `VALIDATION.md`, committed DATA lock + strict CI, and current/history secret scanning are implemented. Baseline-aware full-history and current-tree security scans both passed; exact-head branch CI remains the normal merge/release gate. |
| **P6 technical case study/evidence package** | **NEXT** | Curate the strongest engineering decisions into a small external-review evidence package without changing their technical truth. |
| **P7 GitHub identity/release surface** | **PENDING** | Description/topics, explicit rename/license decisions, external provider-credential revocation/rotation confirmation, first stable portfolio release/tag, optional social preview. |
| **P8 final CV/interviewer audit** | **PENDING** | Recruiter skim, engineer audit, adversarial credibility pass, then derive CV wording. |

## P3 architecture reconciliation boundary

P3 changed documentation/architecture presentation only; no product source, R4 evidence, DATA/ML artifacts, contract semantics, or runtime behavior changed.

Canonical architecture ownership is now:

1. `docs/handbook/01_architecture.md` — whole-system ownership, normal request flow, DATA/ML lifecycle, and proof/attestation/on-chain trust;
2. `docs/handbook/02_runtime_flows.md` — runtime/security flow mechanics and write authority;
3. `docs/handbook/12_security_and_trust.md` — threat model and trust controls;
4. `docs/handbook/11_cross_module_contracts.md` — producer/consumer compatibility registry;
5. `docs/handbook/16_current_status.md` — volatile technical gate/artifact state.

The important architectural correction is explicit:

```text
current live analysis runtime
  uses historical Run12

current R4 repair lifecycle
  D-009 logical V3
  → D-011 accepted V10 V2.6 physical representation
  → D-012 fresh guarded-token successor pending acceptance
  → later repaired teacher only if explicitly authorized
```

The accepted R4 physical lineage is therefore not presented as though it already powers Run12.

## P4 showcase boundary

Fresh-clone entry point:

```bash
python3 tools/showcase_sentinel.py
```

It checks committed source/config for the 14-node LangGraph topology, exact three-tool read-only audit-MCP surface, retained ZKML proxy/settings, and current R4 Phase-8 authority. It explicitly emits `NOT_RUN` for live ML inference, external analyzers/formal tools, live LangGraph execution, proof generation, and V3 signing/broadcast.

`SHOWCASE.md` documents expected output and claim boundaries; `.github/workflows/showcase.yml` keeps this public boundary executable.

## P5 validation / reproducibility / security boundary

P5 closes three earlier presentation/control gaps without redefining historical R4 evidence.

### Current vs historical authority validation

The handbook now has two explicit machine-check layers:

1. `docs/handbook/tools/verify_handbook.py` preserves structural/source and historical G6/G7/runtime-compatibility checks;
2. `docs/handbook/tools/verify_current_r4.py` validates `_meta/current_r4.json` against current logical-V3 evidence, D-011 physical acceptance, D-012 selector/ADR evidence, Phase-8 hold state, and current-facing docs.

This avoids both failure modes: pretending G7 is the latest authority, or rewriting historical compatibility metadata to look current.

### Public validation story

Root [`VALIDATION.md`](../../../VALIDATION.md) states what each current CI surface proves/does not prove, distinguishes normal PR checks from retained `r4-phase*` research/evidence workflows, provides the module/heavy validation matrix, and preserves `PASS` / `NOT_RUN` / `unsupported` / `unauthorized` semantics.

### DATA lock reproducibility

`data_module/poetry.lock` is now committed from an exact Poetry 2.1.3 CI generation. The final `.github/workflows/data-reproducibility.yml` is read-only: it regenerates the resolution and fails if the lockfile is untracked or changes.

The lock closes dependency-resolution reproducibility only; it does not make protected/local R4 physical artifacts available.

### Secret scanning and historical finding

`tools/security/scan_repository_secrets.py` provides dependency-free high-signal scans for current tracked files and reachable Git history.

The first full baseline established:

- current tracked tree: PASS;
- reachable blobs scanned: **92,243**;
- blob content scanned: about **3.36 GB**;
- reviewed historical findings: **4 exact blob identities**;
- finding class: one provider-RPC credential-shaped endpoint repeated in obsolete ZKML helper/generated shell material;
- other configured high-signal categories: no findings in that baseline.

The four exact identities are recorded in `tools/security/known_history_findings.json`; no credential value is reproduced in current documentation. The baseline-aware rerun then passed with all four classified as `KNOWN_HISTORICAL` and no new configured high-signal finding. Current-tree scanning also passed.

New occurrences remain blocking. Current-tree scanning stays normal PR CI; the expensive full-history scan runs on `main`, scheduled, or manual execution.

**Important:** this is not evidence that the historical external provider credential is revoked. Before P7 release, confirm external revocation/rotation or perform it if still needed. No Git history rewrite is authorized merely to remove the old blob identities.

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
| M-009 CI presentation/currentness | **CLOSED** | P5 separates G6/G7 compatibility validation from current D-009/D-011/D-012 semantic authority and publishes the validation matrix. |
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

## Reconciliation and validation boundary

The core governance-routing correction is commit `19883bcd62b41f9e691e3cc54f85de0c0fb14740`.

P3 was validated on `a78ae31f44d63a28cd527cc7735edd305fd371b3` with both `Handbook` and `SENTINEL system alignment` successful. P4 added a separately validated public showcase.

P5 additionally established a successful full-history/current-tree security baseline with the final exact-history classification logic, and a successful CI bootstrap of the exact generated DATA lock before the workflow was returned to strict read-only mode. The professionalization branch must still pass its applicable current checks on the final branch head before merge/release.

## Next execution order

Current default sequence:

`P6 technical case study → P7 GitHub identity/release → P8 final portfolio audit → merge PR #72 to main`

Do not merge PR #72 merely because an intermediate phase passes. Merge only after the professionalization program reaches a coherent final validation boundary.
