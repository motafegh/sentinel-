# P6 Technical Case-Study Package — Closeout

**Date:** 2026-09-12  
**Branch:** `portfolio/professionalization-2026-09-02`  
**PR:** #72  
**Scope:** external technical-review case-study package only; no change to R4 technical authority

## Outcome

P6's intended content scope is complete. The package now contains seven distinct engineering decisions selected because they expose evidence quality, failure handling, lineage discipline, or trust-boundary reasoning rather than merely listing features.

Published cases:

1. `01_unknown_is_not_negative.md` — preserve unknown/unsupported state instead of manufacturing negative truth.
2. `02_leakage_grouping_was_an_evidence_problem.md` — treat dataset grouping as evidence authority rather than convenient preprocessing.
3. `03_version_the_representation_instead_of_patching_history.md` — preserve accepted v9 history and move corrected call semantics to a versioned V10 lineage.
4. `04_fail_closed_on_unexplained_structural_drift.md` — keep physical acceptance false until every observed candidate transition is reconciled.
5. `05_promote_a_selector_without_rewriting_the_accepted_lineage.md` — separate selector-policy support, physical candidate acceptance, and training authority.
6. `06_tool_silence_is_not_a_clean_result.md` — distinguish unavailable/degraded execution from successful zero-finding evidence.
7. `07_a_valid_proof_does_not_prove_the_whole_audit.md` — separate proxy-proof integrity, V3 context attestation, persistence, and transaction authority.

The package index at `docs/case-studies/README.md` defines these seven as the intended P6 core. New cases should be added only when they introduce a materially different engineering decision useful to an external reviewer.

## Authority boundary

The case studies are explanatory portfolio artifacts only. They do not supersede:

1. executable source/config/tests;
2. committed R4 machine-readable policy/evidence/manifests;
3. the canonical handbook;
4. accepted ADRs and decision records.

No case study grants DATA/ML training, model-quality, threshold/calibration, production, signing/broadcast, or expanded ZK authority.

## Evidence review performed

The new cases were checked against the current primary owners relevant to their claims, including:

- ADR-R4-010 — versioned external-call representation correction;
- ADR-R4-011 — exact V10 V2.6 physical acceptance and training hold;
- ADR-R4-012 — guarded selector promotion for a new lineage;
- the V2.5/V2.6 full-population structural-analysis record;
- current AGENTS orchestration/README evidence-status semantics;
- current ZKML and contracts trust-boundary documentation.

The case-study directory was then re-enumerated from GitHub and contains exactly the seven case files plus its index.

## Main-branch reconciliation during P6

While P6 was being completed, `main` advanced by one README-only commit:

- `b0c0e031b35785977a8be2c59cee3dff9f195d22` — `docs: clarify Sentinel research story and project eras`.

That change introduced an important contribution boundary: the original Sentinel work is Ali's long-running AI-assisted learning/building work, while the later R4 continuation is substantially AI-led research under Ali's direction and must not be represented as independent authorship/mastery of every current subsystem.

Because both `main` and the portfolio branch modified `README.md`, PR #72 became conflicted. The conflict was resolved with a real two-parent merge:

- `bd58f79289e47fd0afbaf81a3b77cd47e4bf1d5c` — `merge main: preserve project-era contribution boundary`.

The resulting README retains the richer portfolio landing page and the newer contribution-era disclosure, and links the P6 case studies from the engineering highlights. The portfolio branch is now ahead of current `main` and no longer behind it.

## Validation status

The prior P5 validation boundary remains valid for P5 itself: commit `550851eaaedb1e3f7b3cf17cb0f09f4efcf39661` passed the five current pull-request workflows recorded in `CURRENT_STATUS.md`.

For the P6 commits and the later README merge, GitHub Actions did not start for the connector-authored commits. A direct local fresh-clone validation attempt was also unavailable because the execution container could not resolve `github.com`.

Therefore the correct P6 status is:

- source/evidence review: **COMPLETE**;
- case-study content/package construction: **COMPLETE**;
- branch/main conflict reconciliation: **COMPLETE**;
- current-head automated CI validation: **PENDING / NOT YET OBSERVED**.

Do not describe the latest P6 head as CI-validated until applicable checks have actually run successfully.

## Repository-hygiene note

During connector operation, three accidental no-work remote refs were created:

- `tmp-should-not-create`;
- `noop`;
- `ignore-me`.

They contain no unique work and are not part of PR #72. The available GitHub connector currently exposes no delete-ref action, so they remain explicit cleanup debt. M-010 branch hygiene should not be considered fully closed again until these refs are deleted.

## P6 stop line

Do not expand P6 into a chronological rewrite of Sentinel or duplicate the same R4 episode across multiple portfolio pages. The seven-case package is sufficient for the intended technical-review surface.

Next responsibility after current-head validation/cleanup is P7: GitHub identity and release preparation. Repository rename/license choices and any external credential-revocation confirmation remain owner/external decisions rather than automatic technical defaults.
