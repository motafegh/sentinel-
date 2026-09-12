# SENTINEL Portfolio Professionalization — Current Status

**Last reconciled:** 2026-09-12  
**Branch:** `portfolio/professionalization-2026-09-02`  
**PR:** #72  
**Role:** canonical live status for the portfolio-professionalization program

This file is the live execution/status surface for the portfolio program. Dated plans/audits remain evidence of what was observed at the time, but they do not override this later disposition.

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
- The public contribution boundary remains explicit: the original Sentinel era was Ali's long-running AI-assisted learning/building work; the later R4 continuation is substantially AI-led research under Ali's direction and must not be represented as independent authorship/mastery of every current subsystem.

## Program progress

| Phase / item | Status | Current disposition |
|---|---|---|
| **P0 readiness audit** | **COMPLETE** | Baseline audit captured and findings dispositioned. |
| **P1 repository hygiene foundation** | **SUBSTANTIALLY COMPLETE / EXTERNAL REF CLEANUP OPEN** | Core hygiene is complete; three accidental no-work refs still require deletion through a surface that supports ref deletion. |
| **P2 root README / public landing page** | **COMPLETE** | External-facing landing page, architecture summary, limitations, contribution-era disclosure, showcase, case-study navigation, and license boundary are present. |
| **P3 canonical architecture/trust presentation** | **COMPLETE / VALIDATED AT P5 BOUNDARY** | Canonical runtime, DATA/ML, proof/provenance, and ownership views established. |
| **P4 bounded showcase / demo** | **COMPLETE / VALIDATED AT P5 BOUNDARY** | `SHOWCASE.md` + `tools/showcase_sentinel.py` provide a dependency-light fresh-clone boundary demo. |
| **P5 CI/testing/security/reproducibility presentation** | **COMPLETE / VALIDATED** | Current/historical validation split, DATA lock/reproducibility CI, security scanning, and public validation semantics established. |
| **P6 technical case-study package** | **COMPLETE** | Seven distinct evidence-first engineering case studies are complete and indexed. |
| **P7 GitHub identity/release surface** | **COMPLETE AT REPOSITORY-CONTENT SCOPE / EXTERNAL OPERATIONS OPEN** | Repository name retained; MIT + third-party/artifact notices added; release semantics prepared. GitHub description/topics and external credential/ref operations remain outside this connector's mutation scope. |
| **P8 final portfolio audit** | **IN PROGRESS** | Public-content credibility passes are being executed. Final merge/release remains gated by exact-head CI and external blockers. |

## Exact validated baseline

P5 was formally closed on commit `550851eaaedb1e3f7b3cf17cb0f09f4efcf39661` after all five current pull-request workflows completed successfully:

- Handbook;
- Portfolio showcase;
- DATA reproducibility;
- SENTINEL system alignment;
- Security hygiene.

The P5 history scan inspected **92,243 reachable blobs / about 3.36 GB** and baselined four reviewed historical blob identities containing the same provider-RPC credential-shaped endpoint. The current tracked tree passed. The credential value is not reproduced in current documentation.

Repository evidence cannot prove that the external provider credential was revoked or rotated. That remains a hard release prerequisite if the historical value represented a live credential.

## P6 closeout

The seven-case core is complete:

1. [`Unknown Is Not Negative`](../../case-studies/01_unknown_is_not_negative.md)
2. [`Leakage Grouping Was an Evidence Problem`](../../case-studies/02_leakage_grouping_was_an_evidence_problem.md)
3. [`Version the Representation Instead of Patching History`](../../case-studies/03_version_the_representation_instead_of_patching_history.md)
4. [`Fail Closed on Unexplained Structural Drift`](../../case-studies/04_fail_closed_on_unexplained_structural_drift.md)
5. [`Promote a Selector Without Rewriting the Accepted Lineage`](../../case-studies/05_promote_a_selector_without_rewriting_the_accepted_lineage.md)
6. [`Tool Silence Is Not a Clean Result`](../../case-studies/06_tool_silence_is_not_a_clean_result.md)
7. [`A Valid Proof Does Not Prove the Whole Audit`](../../case-studies/07_a_valid_proof_does_not_prove_the_whole_audit.md)

Detailed closeout: [`2026-09-12_P6_TECHNICAL_CASE_STUDY_CLOSEOUT.md`](2026-09-12_P6_TECHNICAL_CASE_STUDY_CLOSEOUT.md).

During P6, `main` advanced with README contribution-boundary commit `b0c0e031b35785977a8be2c59cee3dff9f195d22`. Two-parent merge `bd58f79289e47fd0afbaf81a3b77cd47e4bf1d5c` reconciled that change without rewriting portfolio history.

## P7 final decisions

Decision record: [`2026-09-12_P7_DECISION_RESOLUTION.md`](2026-09-12_P7_DECISION_RESOLUTION.md).

Supporting records:

- [`2026-09-12_P7_GITHUB_IDENTITY_AND_RELEASE_PREPARATION.md`](2026-09-12_P7_GITHUB_IDENTITY_AND_RELEASE_PREPARATION.md)
- [`2026-09-12_P7_LICENSE_CONTENT_INVENTORY.md`](2026-09-12_P7_LICENSE_CONTENT_INVENTORY.md)
- [`2026-09-12_P7_RELEASE_NOTE_TEMPLATE.md`](2026-09-12_P7_RELEASE_NOTE_TEMPLATE.md)

Final decisions:

- **Repository name:** retain `sentinel-` for this cycle; the imperfect trailing hyphen is accepted as minor naming debt rather than creating migration churn for cosmetic value.
- **License:** MIT for original project code/documentation, with [`THIRD_PARTY_NOTICES.md`](../../../THIRD_PARTY_NOTICES.md) making third-party/generated/data/model/proof rights explicit.
- **Homepage:** leave unset unless a real durable destination exists.
- **Release semantics:** use a portfolio/research snapshot, recommended tag `portfolio-2026-09`, rather than a product-maturity `v1.0.0` claim.
- **Description/topics:** recommended values are prepared, but repository-settings mutation is not exposed by the current GitHub connection; this is operationally deferred rather than analytically unresolved.

### License boundary

Observed license context:

- all four local current Solidity files under `contracts/src/` declare MIT;
- generated `ZKMLVerifier.sol` declares MIT;
- OpenZeppelin dependencies are MIT upstream;
- `forge-std` is Apache-2.0 upstream and remains a Git submodule;
- retained model/proof/data/research artifacts are not automatically relicensed by the root MIT license.

## P8 audit boundary

Protocol: [`2026-09-12_P8_FINAL_PORTFOLIO_AUDIT_PROTOCOL.md`](2026-09-12_P8_FINAL_PORTFOLIO_AUDIT_PROTOCOL.md).

The audit checks:

1. recruiter/hiring-manager clarity;
2. senior-engineer source-to-claim traceability;
3. adversarial overclaim/credibility resistance;
4. repository hygiene/release integrity;
5. exact candidate CI.

The public surfaces currently reviewed—README, architecture, current-status handbook, showcase, validation guide, and case-study index—consistently preserve the major claim boundaries. The P8 closeout will record the exact disposition.

## Remaining MUST-item disposition

| ID | Status | Remaining responsibility |
|---|---|---|
| M-001 public README | **CLOSED** | — |
| M-002 GitHub identity | **CONTENT DECISIONS CLOSED / SETTINGS DEFERRED** | Optional description/topics update requires a GitHub-settings-capable surface. |
| M-003 security policy | **CLOSED** | — |
| M-004 DVC/artifact contract | **CLOSED** | Future heavy-artifact distribution remains optional/separate. |
| M-005 runtime/ignore hygiene | **CLOSED** | — |
| M-006 environment contract | **CLOSED** | Heavy/local prerequisites remain explicit. |
| M-007 lightweight showcase | **CLOSED** | — |
| M-008 canonical architecture | **CLOSED** | — |
| M-009 CI presentation/currentness | **CLOSED / FINAL-HEAD CHECK STILL REQUIRED** | Exact P8 candidate must pass current checks before merge. |
| M-010 stale PR/branch hygiene | **EXTERNAL CLEANUP OPEN** | Delete `tmp-should-not-create`, `noop`, and `ignore-me`; they contain no unique intended work. |
| M-011 size/history policy | **CLOSED** | No history rewrite authorized. |
| M-012 module README truth | **CLOSED for audited surfaces** | — |
| M-013 stable release | **PREPARED / BLOCKED** | Publish only after merge, final validation, ref cleanup, and credential disposition. |
| M-014 secret hygiene | **REPOSITORY CONTROL CLOSED / EXTERNAL CONFIRMATION OPEN** | Confirm provider credential revocation/rotation before release if applicable. |

## Current blockers

1. **External provider credential disposition:** Git cannot establish revocation/rotation. No release may claim this prerequisite satisfied without external confirmation.
2. **Accidental refs:** `tmp-should-not-create`, `noop`, and `ignore-me` should be deleted; the exposed connector provides ref update/create but not delete.
3. **Final candidate CI:** exact P8 candidate must complete the five current validation surfaces successfully.

## Next execution order

`finish P8 credibility audit → obtain exact-candidate CI → HOLD merge/release until external credential + ref-cleanup prerequisites are satisfied → merge PR #72 → create portfolio snapshot release`

Do not weaken the release/merge gate merely to finish the portfolio program cosmetically.
