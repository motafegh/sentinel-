# GAP-007 v2 publication and CI recovery

Date: 2026-11-26

Baseline: 012a603fe7f14a129525c1512e326430db561704 (local and GitHub main matched).

## Execution plan

1. Preserve all existing local untracked files and immutable R4 data/evidence roots.
2. Rebuild the v2 reviewer archive deterministically and verify the manifest, source and archive digest.
3. Restore the existing CI contract without relaxing its gates: convert only reported Markdown two-space hard breaks to equivalent backslash hard breaks, and restore the explicit current DATA policy reference to CLAUDE.md.
4. Publish the exact v2 archive and checksum; use a narrow gitignore exception. Add CI regeneration and tracked-byte comparison.
5. Validate local handbook and R4 repository-safe checks, then publish only the intended changes.
6. Obtain a genuinely isolated reviewer separately. This session is not eligible as a blind reviewer.

## Initial observations

- Existing local v2 archive rebuilt successfully; SHA-256 before and after was 9036d5744ef903f7ba1204edb3a248f234ca1675349b8b8d19a0e6f9bab4fc36.
- The v2 ZIP is ignored and its matching checksum is untracked at baseline.
- Phase 8 CI run 39451527520 passed compilation, regression, snapshot and historical G6 steps but failed the existing cumulative whitespace gate: 115 intentional Markdown hard breaks across 35 files.
- Handbook run 39451527501 passed static and current R4 checks but failed the entry-point phrase assertion because CLAUDE.md lacked data-vnext-policy-v1.

No candidate verdict, target-zero, model training or promotion authority changes. Candidate #2 remains UNKNOWN / PENDING_REVIEW / target None.

## Local verification results

- Deterministic rebuild reproduced the existing archive hash; source and per-member manifest validation passed.
- Handbook static: 145 passed, 0 failed.
- Current R4 validator: 56 passed, 0 failed.
- Handbook unit tests: 13 passed; inventory emitted successfully.
- All 12 active entry-point documentation assertions passed.
- Committed logical V3 snapshot: PASS; 11 JSON checksums, 60 coherence checks; 200 reserved cells / 200 groups; no training authorization.
- Historical G6: PASS; 22,493 contracts and zero confirmed-negative rows.
- Cumulative original-base and staged whitespace gates passed with no gate weakening.
- New Phase 8 CI step rebuilds the tracked archive and checks the tracked archive/checksum are unchanged. The fresh GitHub CI run remains to be observed.
- Independent reviewer has not been dispatched and no adjudication has been performed.

Next: give only the v2 ZIP to a reviewer isolated from all candidate history, then review admissibility before any formal adjudication.
