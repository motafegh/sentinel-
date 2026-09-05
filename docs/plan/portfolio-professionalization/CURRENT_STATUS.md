# SENTINEL Portfolio Professionalization — Current Status

**Last reconciled:** 2026-09-05  
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
| **P1 repository hygiene foundation** | **SUBSTANTIALLY COMPLETE** | Runtime/DVC cruft, ignore rules, machine-local DVC remote, stale PRs/branches, security policy, secret-exposure pass, developer contract, and repository-weight policy addressed. GitHub metadata + rename/license decisions remain. |
| **M-003 SECURITY.md** | **CLOSED** | Public security/reporting policy added without invented SLA, bounty, or private contact. |
| **M-004 DVC/artifact semantics** | **CLOSED at public-contract level** | Machine-local default remote removed; two DVC contexts and fresh-clone limitations documented. |
| **M-005 runtime/ignore hygiene** | **CLOSED** | Root `.dvc/tmp` tracked runtime files removed and ignore rules hardened. |
| **M-006 environment contract** | **SUBSTANTIALLY COMPLETE** | `DEVELOPMENT.md` defines the multi-environment monorepo; root pytest scope corrected; forced DATA regional package index removed. DATA locking remains a P5 responsibility. |
| **M-010 stale PR/branch hygiene** | **CLOSED** | Obsolete PRs closed; obsolete remote branches removed. |
| **M-011 repository size/history policy** | **CLOSED** | Current-tree audit complete; historical Git storage distinguished from current artifacts; no history rewrite authorized. |
| **M-012 module README truth alignment** | **CLOSED for audited surfaces** | DATA/ML/contracts/ZKML current-state sections aligned; AGENTS already aligned. |
| **M-014 credential/security hygiene** | **PARTIAL / adequate for current phase** | Bounded tracked-repository scan found no obvious committed secret material; dedicated historical/CI scan remains P5. |
| **P2 root README / public landing page** | **COMPLETE at current scope** | Recruiter/senior-engineer landing page, architecture summary, limitations, setup navigation, AI-assisted ownership disclosure, and showcase entry point are present. |
| **State reconciliation** | **COMPLETE** | Master/P0/P1 records reconciled, one canonical status file established, duplicate M-011 audit removed, and `CLAUDE.md` restart/memory routing corrected. |
| **P3 canonical architecture/trust presentation** | **COMPLETE / VALIDATED** | Four canonical architecture views established and current-facing DATA/ML seams reconciled. Handbook + system-alignment passed on P3 head `a78ae31f44d63a28cd527cc7735edd305fd371b3`. |
| **P4 bounded showcase / demo** | **COMPLETE / VALIDATED** | `SHOWCASE.md` + `tools/showcase_sentinel.py` provide a fresh-clone standard-library boundary demo; human and JSON modes passed the dedicated Portfolio showcase CI check. |
| **P5 CI/testing/security/reproducibility presentation** | **NEXT** | Current-vs-historical workflow presentation, semantic doc-currentness hardening, DATA locking decision, historical/CI secret scan, and validation matrix. |
| **P6 technical case study/evidence package** | **PENDING** | Curate strongest engineering decisions with traceable evidence. |
| **P7 GitHub identity/release surface** | **PENDING** | Description/topics, explicit rename/license decisions, first stable portfolio release/tag, optional social preview. |
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

### Known machine-readable documentation limitation deferred to P5

`docs/handbook/_meta/handbook.toml` and `verify_handbook.py` still contain historical G6/G7/v9 compatibility checks. They remain useful for source/runtime compatibility but do not machine-validate the later D-009/D-011/D-012 semantic/physical authority chain. P5 must improve that validator rather than merely changing metadata labels.

## P4 showcase boundary

The public showcase deliberately avoids a fake one-command production audit.

Fresh-clone entry point:

```bash
python3 tools/showcase_sentinel.py
```

It checks committed source/config for:

1. 14-node LangGraph topology (`ml_assessment` → `visualizer`);
2. exact three-tool read-only audit-MCP surface;
3. retained `128→64→32→10` ZKML proxy, 138 public signals, and `check_mode="UNSAFE"` limitation;
4. current R4 Phase-8 / R4-D-011 / no-confirmed-negative / no-full-training boundary.

It explicitly emits `NOT_RUN` for live ML inference, external analyzers/formal tools, live LangGraph audit execution, ZK proof generation, and V3 signing/broadcast. `PASS` therefore means the checked repository architecture/governance boundary matches source/config; it does not mean a contract audit succeeded or product/model quality was established.

`SHOWCASE.md` documents the expected output and the optional deeper AGENTS orchestration smoke. `.github/workflows/showcase.yml` keeps the fresh-clone demo executable on PRs/main.

## Remaining P0 MUST-item disposition

| ID | Status | Remaining responsibility |
|---|---|---|
| M-001 public README | **CLOSED at current scope** | Reassess only if later P6/P7 material warrants a small landing-page refinement. |
| M-002 GitHub identity | **OPEN** | Description/topics; explicit owner decision on repo name and license. Homepage only if a real destination exists. |
| M-003 security policy | **CLOSED** | — |
| M-004 DVC/artifact contract | **CLOSED at current scope** | Future public heavy-artifact distribution is optional/separate and must be hash/version bound. |
| M-005 runtime/ignore hygiene | **CLOSED** | — |
| M-006 environment contract | **SUBSTANTIALLY COMPLETE** | DATA lock/reproducibility remains P5. |
| M-007 lightweight showcase | **CLOSED** | Fresh-clone boundary showcase + dedicated CI validation added. |
| M-008 canonical architecture | **CLOSED** | P3 validated. |
| M-009 CI presentation/currentness | **OPEN / NEXT** | P5; includes D-009/D-011/D-012 semantic validator hardening. |
| M-010 stale PR hygiene | **CLOSED** | — |
| M-011 size/history policy | **CLOSED** | Optional future object-level history inventory only if size becomes operationally blocking. |
| M-012 module README truth | **CLOSED for audited surfaces** | Re-check only when later changes create contradictions. |
| M-013 stable release | **OPEN** | P7 after earlier gates. |
| M-014 secret hygiene | **PARTIAL** | Dedicated history/CI scan under P5. |

## Public/GitHub identity decisions still intentionally unresolved

These are not technical-cleanup defaults:

1. keep or rename repository `sentinel-`;
2. choose a license or intentionally remain unlicensed for now;
3. set repository description/topics when repository-setting write access is available;
4. decide whether a homepage/social-preview is useful after the public case-study/release surface stabilizes.

## Reconciliation and validation boundary

The core governance-routing correction is commit `19883bcd62b41f9e691e3cc54f85de0c0fb14740`.

P3 was validated on `a78ae31f44d63a28cd527cc7735edd305fd371b3` with both `Handbook` and `SENTINEL system alignment` successful.

P4 adds the dependency-light showcase, documentation, README entry point, and dedicated showcase workflow. The professionalization branch must continue to pass its relevant current CI checks before final merge. Use actual completed CI results rather than assuming success from earlier heads.

## Next execution order

Current default sequence:

`P5 CI/reproducibility/security presentation → P6 technical case study → P7 GitHub identity/release → P8 final portfolio audit → merge PR #72 to main`

Do not merge PR #72 merely because an intermediate phase passes. Merge only after the professionalization program reaches a coherent final validation boundary.