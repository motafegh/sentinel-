# SENTINEL P1.1 Repository Hygiene and Identity Plan

**Created:** 2026-09-02  
**Last reconciled:** 2026-09-05  
**Status:** **SUBSTANTIALLY COMPLETE — identity decisions remain open**  
**Parent:** P0 Portfolio Readiness Audit / P1  
**Live status:** [`CURRENT_STATUS.md`](CURRENT_STATUS.md)  
**Scope:** repository hygiene, artifact/public-identity foundation; no R4 semantic/product implementation changes

## Goal

Make the repository structurally intentional and externally defensible while preserving accepted R4/history/reproducibility evidence.

This file is the phase record. `CURRENT_STATUS.md` is the live program dashboard.

## Workstream disposition

### H1 — safe runtime/ignore hygiene — **COMPLETE**

Completed:

- root `.dvc/tmp` runtime files removed from Git;
- `.dvc/tmp/` and machine-local DVC config ignored;
- malformed/redundant root ignore rules cleaned;
- environment/key/cache/runtime ignore controls hardened;
- ML checkpoint/model binary classes protected from future accidental Git inclusion;
- required R4 independent-review ZIP exception preserved.

No protected R4 evidence was deleted.

### H2 — DVC/artifact contract — **COMPLETE at current public-contract scope**

Findings established that two real DVC contexts exist:

1. root `.dvc/` for repository/local artifact operations;
2. `data_module/.dvc/` plus `data_module/dvc.yaml` for the historical module-local DATA lifecycle.

Completed:

- removed the public machine-local default remote `/mnt/d/sentinel-dvc-remote`;
- retained root `no_scm=True` rather than making an unrelated semantic DVC change;
- documented safe local remote configuration through ignored/local config;
- documented that `dvc repro` in `data_module/` does not automatically reconstruct the accepted R4-D-011 protected-local physical lineage;
- documented fresh-clone availability boundaries for Run12, R4 physical representations, RAG/runtime state, and proving artifacts.

No public heavy-artifact host was invented. If one is added later, it must be versioned/hash-bound explicitly.

### H3 — repository size/history policy — **COMPLETE**

Canonical audit:

`2026-09-04_REPOSITORY_WEIGHT_AND_ARTIFACT_AUDIT.md`

Disposition:

- GitHub repository size remains roughly 396 MB;
- current active evidence/model/data files inspected are comparatively small and justified;
- the dominant remaining weight is historical Git storage rather than obvious current-tree bloat;
- future raw datasets, generated representations, checkpoints, and heavy proving material remain outside normal Git;
- history rewrite is rejected for ordinary portfolio cleanup because R4 provenance binds exact commits;
- partial clone (`git clone --filter=blob:none ...`) is documented as the non-destructive clone mitigation.

A future object-level history migration is optional and requires a separately justified evidence-preserving plan.

### H4 — PR/branch hygiene — **COMPLETE**

Completed:

- obsolete May/June PRs reviewed and closed rather than merged cosmetically;
- obsolete remote branches removed after confirming their useful work/history was already represented by current `main`/Git history;
- during professionalization the intended remote branch surface is `main` plus `portfolio/professionalization-2026-09-02`;
- PR #72 remains the current draft professionalization PR.

After the professionalization program is fully validated and merged, the temporary professionalization branch can be removed.

Main-branch protection/rules remain a separate later repository-policy decision, not part of branch cleanup.

### H5 — public project health/identity — **PARTIAL**

Completed:

- root `SECURITY.md` added;
- bounded secret/credential exposure review performed with no obvious committed credential material found;
- key/credential ignore protections hardened;
- root `DEVELOPMENT.md` added and public setup/artifact boundaries clarified;
- root README rebuilt as an external landing page.

Still open:

- set concise GitHub repository description;
- set accurate topics;
- explicitly decide whether to keep or rename `sentinel-`;
- explicitly decide license;
- homepage/social preview only after a meaningful destination/visual is stable.

Description/topics have not been silently claimed as complete because the current connector has not exposed repository-settings write capability.

## Additional developer-experience repairs completed during P1

Although originally tracked separately under M-006, the hygiene foundation also corrected setup metadata that affected public repository credibility:

- root pytest scope no longer falsely includes AGENTS and nonexistent `api/tests`;
- AGENTS/DATA keep their module-owned environments/test configuration;
- DATA no longer forces a regional primary package index;
- `DEVELOPMENT.md` explains the real multi-environment monorepo rather than inventing a universal environment.

DATA still has no committed Poetry lockfile; that remains a P5 reproducibility responsibility.

## Protected exclusions

Do not delete, rewrite, rename, or compact for aesthetics:

- `docs/plan/ml-R4/` decisions/evidence/review bundles/hashes/manifests;
- accepted G0–G7 and R4-D-008/009/010/011/012 identities/evidence;
- Run12 historical lineage references;
- retained ZKML/circuit/verifier reproducibility lineage;
- V1/V2/V3 compatibility source/tests;
- historical artifacts still needed to substantiate current claims.

## Validation discipline used

Each cleanup slice followed the same rule:

1. inspect the exact target and authority;
2. delete/change only runtime, generated, misleading, or safely superseded material;
3. preserve protected paths;
4. run/observe relevant repository checks;
5. record unavailable/local-only boundaries rather than guessing.

Current-facing README work on the branch has passed both `Handbook` and `SENTINEL system alignment` CI at head `b6a4ad7480c41d86935443921193a7d304be3c40`. Later reconciliation commits must pass the same checks before final merge.

## Exit assessment

P1.1 is **substantially complete**, not fully closed, because public GitHub identity still has intentional unresolved decisions/settings:

- description/topics;
- repository-name choice;
- license choice.

Those remaining identity responsibilities do not block progression to P3 architecture work. They are retained in `CURRENT_STATUS.md` and will be resolved before the P7 stable portfolio release/final P8 audit.

No R4 semantic/product truth was changed by this phase.
