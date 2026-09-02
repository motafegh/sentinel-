# SENTINEL P1.1 Repository Hygiene and Identity Plan

**Date:** 2026-09-02  
**Status:** IN_PROGRESS  
**Parent:** P0 Portfolio Readiness Audit / P1.1  
**Scope:** repository hygiene, artifact/public identity foundation; no R4 semantic/product implementation changes

## Goal

Make the repository structurally intentional and externally defensible before the final README/demo/release work, while preserving all accepted R4/history/reproducibility evidence.

## Workstreams

### H1 — Safe runtime/ignore hygiene — execute now

- add `.dvc/tmp/` to Git ignore;
- remove tracked `.dvc/tmp` runtime lock/timestamp files only;
- repair malformed `.gitignore` compressed/docs pattern residue;
- preserve the explicit R4 independent-review ZIP exception and all protected evidence.

### H2 — DVC/artifact contract — investigate before changing

Current evidence shows:

- root `.dvc/config` has `no_scm = True`, remote `localbackup`, URL `/mnt/d/sentinel-dvc-remote`;
- root `.dvc/tmp/*` is tracked runtime state;
- `data_module/.dvc/config` exists as an empty nested DVC root;
- repository code search did not surface consumers of `localbackup`/the machine-local remote.

Do **not** remove/consolidate either DVC root or change the remote until current artifact pointers/workflows/local reproducibility assumptions are mapped. The public contract must ultimately distinguish committed evidence, locally reproducible artifacts, and externally retrievable artifacts.

### H3 — Repository size/history policy — local-clone audit required

GitHub reports roughly 406 MB. The connector cannot reliably attribute packed Git history size. Before any history rewrite or large-blob migration, run a local audit (`git count-objects`, large-blob inventory/equivalent) and classify source/evidence/generated/binary history. No history rewrite is authorized by this plan.

### H4 — PR/branch hygiene

- inspect stale open PRs individually and close only clearly superseded ones;
- inventory old branches and preserve branches/commits still referenced by evidence or useful history;
- do not merge stale branches for cosmetic cleanup;
- decide lightweight `main` protection later after checking actual ruleset state.

### H5 — Public project health/identity

Safe later actions:

- add `SECURITY.md` with honest reporting/scope wording;
- set concise GitHub description and useful topics;
- normalize only misleading public metadata.

Explicit decision gates:

- repository rename (`sentinel-` vs intentional new name);
- license choice.

Do not apply rename/license automatically because they have public-link/legal/career implications.

## Protected exclusions

Do not delete, rewrite, rename, or compact for aesthetics:

- `docs/plan/ml-R4/` decisions/evidence/review bundles/hashes/manifests;
- accepted G7, R4-D-008, R4-D-009, R4-D-010/011/012 identities/evidence;
- Run12 historical lineage references;
- retained ZKML/circuit/verifier reproducibility lineage;
- V1/V2/V3 compatibility source/tests;
- historical artifacts still required to substantiate current claims.

## Validation per slice

For every cleanup slice:

1. inspect exact branch diff;
2. prove deleted files are generated/runtime or otherwise safely superseded;
3. verify protected paths are unchanged;
4. run the relevant existing CI/checks when triggered;
5. record deferred items instead of guessing where local-only evidence is required.

## Exit gate

P1.1 completes when:

- no known runtime temp/lock cruft is tracked;
- ignore rules are coherent;
- public DVC/artifact semantics are explicit and no longer machine-specific by default, or the remaining machine-local contract is clearly isolated/documented pending a safe migration;
- stale PR/branch surface is intentionally contained;
- repository-size policy is evidence-backed;
- security/public metadata foundation is present;
- rename/license decisions are either applied after explicit user choice or recorded as deliberate pending choices;
- no protected evidence/history was damaged.
