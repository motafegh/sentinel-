# SENTINEL P0 Portfolio Readiness Audit

**Date:** 2026-09-02  
**Status:** P0 COMPLETE — implementation not started  
**Branch:** `portfolio/professionalization-2026-09-02`  
**Parent plan:** `2026-09-02_SENTINEL_PORTFOLIO_PROFESSIONALIZATION_MASTER_PLAN.md`  
**Audit target:** public GitHub/CV readiness without changing technical truth

---

## 1. Executive result

SENTINEL is technically substantial enough to be a strong portfolio project, but the current public repository is **not yet CV-ready**.

The dominant problem is not lack of engineering depth. It is a mismatch between the current engineering authority and the public-facing repository surface:

- current R4 authority is Phase 8 with R4-D-011 physical V10 acceptance and R4-D-012 guarded-selector promotion for a new candidate;
- several public entry documents still describe G6/G7-era state as current;
- artifact retrieval depends on a machine-local DVC remote;
- the repository lacks an intentional public identity layer (description/topics/license decision/security policy/release);
- the development/runtime entry path is module-specific and difficult for a new reviewer to discover;
- CI is extensive and currently functioning, but the active Actions surface mixes historical R4 gates with current checks and the handbook freshness check can pass stale state because part of it is phrase-based;
- repository/runtime/history hygiene needs cleanup before public portfolio publication.

**P0 verdict:** `NOT_PORTFOLIO_READY_YET`, with no architectural redesign required.

The project should proceed through bounded professionalization. The first repair must be **truth alignment**, before aesthetic README or GitHub decoration.

---

## 2. Audit method and authority

The audit used current `main` source/config/docs plus repository metadata and GitHub Actions/PR state.

When facts conflicted, this audit followed `CLAUDE.md` authority:

1. executable source/config/tests;
2. current machine-readable R4 governance/evidence;
3. canonical handbook;
4. ADRs/registers;
5. historical/supplementary documentation.

Current DATA/ML truth was anchored to `docs/handbook/16_current_status.md` and the current R4 decision chain, including R4-D-011 and R4-D-012.

This audit does **not** reinterpret DATA/ML evidence and does not grant any new training/model/production authority.

---

# 3. Priority findings

## BLOCKER

### B-001 — Public entry documentation materially contradicts current R4 authority

**Evidence**

- Root `README.md` says the current stable baseline is G7 / 22,493 historical contracts and says Phase 8 is the next authorized step.
- `docs/handbook/00_README.md` still presents older G6/Phase-7-era navigation/state.
- `docs/handbook/01_architecture.md` says stable main has passed G6 and says Phase 7 remains candidate work.
- `data_module/README.md` describes the older G7/v9 line.
- `ml/README.md` says Phase 7 must pass before Phase 8 retraining.
- `contracts/README.md` says R4 repair is through G6 and Phase 7 still needs local G7 representation binding.
- Current authority instead says historical G0-G7 remain passed; Phase 8 is in progress; R4-D-011 accepts the exact V10 V2.6 physical lineage; R4-D-012 permits guarded selection only for a fresh versioned candidate; full training remains unauthorized.

**Why blocker**

A recruiter or engineer following the repository's advertised documentation can receive factually stale project state. This is a credibility/truth problem, not merely a presentation preference.

**Required repair**

Before broader portfolio polishing, perform a bounded current-state alignment of all externally prominent entry documents. Do not rewrite deep historical records; update only documents that present themselves as current.

**Acceptance**

- Root README and canonical current docs do not contradict `docs/handbook/16_current_status.md` / current R4 authority.
- Historical numbers are explicitly labeled historical when retained.
- No document claims repaired teacher/model quality, threshold/calibration support, accepted negatives, full-training authorization, production signer/broadcaster, or broader ZK proof scope than current evidence supports.

---

# 4. MUST items before CV publication

## M-001 — Rebuild the root README as a public landing page

Current README has useful trust-boundary content but is both stale and written primarily as an internal status document.

Required public sequence:

1. one-sentence value proposition;
2. problem;
3. what SENTINEL actually does;
4. canonical architecture visual;
5. strongest implemented capabilities;
6. technology stack by responsibility;
7. example/demo result;
8. repository map;
9. quick-start/demo path;
10. validation/reproducibility story;
11. current limitations/status;
12. links to deeper handbook/R4 evidence.

Avoid badge spam and long R4 terminology in the first screen.

---

## M-002 — Establish an intentional GitHub repository identity

Current repository metadata observed on 2026-09-02:

- name: `sentinel-`;
- description: unset;
- topics: none;
- homepage: unset;
- detected license: none;
- releases: none.

Required:

- add a concise description;
- add accurate topics;
- explicitly decide whether `sentinel-` is retained or renamed;
- explicitly decide licensing before adding a license;
- define whether a homepage is useful only after a real destination exists.

**Decision-required items:** repository rename and license choice are not automatic cleanup actions.

---

## M-003 — Add a public security policy

`SECURITY.md` is absent.

Because SENTINEL is itself a security project, the repository should state:

- how to report a vulnerability in SENTINEL;
- what is in/out of scope;
- that public issues should not be used for undisclosed sensitive vulnerabilities where inappropriate;
- supported/current development status;
- no bug-bounty promise unless one actually exists.

Do not fabricate contact channels or response SLAs.

---

## M-004 — Repair DVC and artifact-retrieval semantics

Current root `.dvc/config` sets:

```text
remote = localbackup
url = /mnt/d/sentinel-dvc-remote
```

This is machine-local WSL/Windows state and cannot serve a fresh external clone. `data_module/` also contains a second `.dvc` root with an empty config.

Required:

- decide the canonical DVC root;
- remove machine-specific default remote state from the public contract;
- clearly state which artifacts are committed, downloadable, reproducible, or intentionally local/unpublished;
- provide a lightweight demo path that does not pretend all historical training/proving artifacts are publicly downloadable;
- if a public artifact remote is later added, version and document it explicitly.

Full multi-GB artifact publication is **not** required merely for portfolio appearance.

---

## M-005 — Stop tracking runtime/DVC cruft and repair ignore rules

Current tree tracks `.dvc/tmp` files including lock/rwlock state. `.gitignore` ignores `.dvc/cache/` but not `.dvc/tmp/` and contains accumulated/malformed-looking residue, including the merged-looking `*.tar.zstdocs/IRVLESS*.txt` line.

Required:

- ignore `.dvc/tmp/`;
- untrack runtime lock/temp files without touching DVC evidence/artifact pointers;
- clean malformed/redundant ignore entries;
- preserve explicit exceptions that are required R4 review evidence;
- run a targeted tracked-temp/generated-file audit.

---

## M-006 — Define the monorepo/development environment contract

Current repository is not one simple Python environment:

- root `pyproject.toml` calls itself `sentinel-workspace`, packages only `ml`, and has root pytest paths including `api/tests` even though there is no root `api/` module;
- `agents/` has an independent Poetry environment and explicitly overrides root pytest behavior;
- `data_module/` has its own Poetry package and requires Python 3.12;
- contracts use Foundry;
- ZKML shares parts of the ML Python environment/runtime;
- external binaries/services are required for some paths.

This can be a valid multi-environment monorepo. It simply needs to be explicit.

Required:

- document environment ownership per module;
- remove/repair misleading root workspace/test configuration;
- provide one top-level developer command index/task surface for common validation;
- do not force incompatible/heavy components into one environment merely for aesthetics.

Also review the use of the Tsinghua PyPI mirror as root/default/primary package source; a public global setup should not unexpectedly depend on a regional mirror unless documented/optional.

---

## M-007 — Create one lightweight, reproducible showcase path

No clear root-level five-minute demo path is currently surfaced.

Required outcome:

A reviewer can run or inspect one bounded example without requiring the entire historical dataset, multi-day GPU training, production RPC credentials, or a new EZKL ceremony.

Preferred shape:

```text
small Solidity example
→ bounded available analysis path
→ structured report/example output
→ explanation of which evidence channels ran vs were unavailable
```

Where live execution requires unavailable artifacts/services, ship a committed example output plus an honest replay/inspection path rather than faking execution.

---

## M-008 — Produce one canonical current architecture/trust-boundary diagram

The existing architecture is strong, but current public architecture docs contain stale DATA/ML state.

The canonical external diagram must distinguish:

1. off-chain client/gateway/LangGraph analysis path;
2. DATA/representation/teacher lifecycle;
3. `fusion[128]` → retained proxy/EZKL proof path;
4. V3 EIP-712 context-attestation + registry path;
5. boundaries that are implemented, historical, candidate, or intentionally external.

It must visibly state that:

- gateway completion is off-chain;
- audit MCP is read-only;
- production signing/broadcast is not implemented/claimed;
- retained proof proves proxy computation only;
- Run12 is historical operational baseline;
- repaired full teacher training remains unauthorized today.

---

## M-009 — Reorganize CI presentation and strengthen documentation freshness validation

GitHub Actions is active and extensive. Recent `main` runs include successful Handbook and R4 compatibility/gate workflows.

The problem is presentation and semantics:

- many historical phase-specific R4 workflows still run/appear as current active CI;
- a fresh push can trigger old-named gates such as `R4 Phase 6 G6 gate`, which is confusing after G6/G7 are historical;
- the Handbook workflow's explicit entry-point check partly asserts phrase presence (`R4`, `Run12`, `V3`, etc.), so stale G6/G7 claims can coexist with a green check.

Required:

- identify which workflows are current required CI vs historical/reproducibility workflows;
- preserve historical workflow files where valuable, but prevent obsolete gates from presenting as the current acceptance story;
- expose a small set of meaningful active checks;
- strengthen current-document validation against machine-readable/current-state facts rather than keyword presence alone;
- only add README CI badges for checks whose meaning is stable and clear.

The branches API currently reports `main` as `protected=false`; separate repository-ruleset state was not verifiable through the current integration and must not be guessed.

---

## M-010 — Resolve stale open PR hygiene

Six old PRs remain open from May/June-era work, including draft PRs whose base SHA is far behind current `main`.

Required:

- inspect each;
- close/archive obsolete ones with a concise reason;
- preserve anything still useful through branch/history references;
- do not merge stale branches merely to clear the UI.

This audit's portfolio PR is separate current work.

---

## M-011 — Audit repository size and artifact/history policy

GitHub reports repository size around **406,754 KB**.

The current tree includes evidence/data artifacts (for example committed R4 parquet exports), and project history is long. Size alone is not a defect, but it is high enough to demand an intentional policy.

Required before portfolio release:

- run a local history-size audit (`git count-objects`, large-blob inventory / equivalent);
- identify whether size is current-source evidence, historical blobs, generated reports, models, or accidental binaries;
- keep small/valuable evidence in Git where justified;
- use DVC/release assets/external artifact storage for large reproducible binaries when appropriate;
- do not rewrite Git history merely to make a number smaller unless a concrete benefit and safe migration justify it.

The GitHub connector is insufficient to attribute all packed-history size precisely; this sub-audit requires a local clone during implementation.

---

## M-012 — Targeted module README truth alignment

Do not blindly rewrite every module README.

Current audit:

- `agents/README.md`: substantially current and strong; preserve its explicit off-chain/read-only/failure-boundary language.
- `zkml/README.md`: mostly strong and correctly limits proof scope / exposes `check_mode="UNSAFE"`; refresh only stale R4 sequence language.
- `contracts/README.md`: protocol/trust sections are useful, but R4 relationship section is stale.
- `data_module/README.md`: materially stale relative to repaired-v2/V3/V10 authority.
- `ml/README.md`: materially stale relative to current Phase-8 state.

Acceptance: current-facing sections agree with authority while historical anchors remain clearly labeled.

---

## M-013 — Add a stable portfolio release after the cleanup passes

There are currently no GitHub releases.

Do not create a release during P0. After professionalization and validation, create one portfolio/stable snapshot whose notes state:

- what is implemented;
- what remains research/in progress;
- reproducible demo/validation route;
- exact known limitations;
- current R4 no-training/model-claim boundary where applicable.

A release is a communication/version anchor, not a declaration of production readiness.

---

## M-014 — Add a bounded credential/security hygiene check

Targeted code searches during P0 did not surface obvious `PRIVATE_KEY` or `API_KEY` literals, and `.env` patterns are ignored. This is **not equivalent to a full secret scan**.

Before portfolio publication:

- run a proper secret/history scan locally/CI;
- verify example configs contain placeholders only;
- review RPC/key/signer-related scripts for safe example behavior;
- do not publish operational credentials or private artifact endpoints.

---

# 5. SHOULD items

## S-001 — Contain generated/historical report clutter

`agents/` currently exposes multiple top-level historical test/report directories such as broken/fixed/quarantine-era audit-report outputs.

Preserve useful debugging evidence, but move or clearly classify it under an archive/evidence/test-fixture area if references permit. Do not leave experimental output folders competing visually with `src/`, `tests/`, and current docs.

## S-002 — Create a technical case study

Use a small number of high-value engineering decisions rather than a chronological diary. Strong candidates:

- historical zero ≠ confirmed negative;
- leakage grouping repair after the 10,327-contract component;
- v9 external-call representation defect and V10 repair;
- deterministic V2.6 full-population evidence;
- explicit refusal to train without adequate evaluation/negative evidence.

The case study should trace claims to code/tests/ADRs/evidence.

## S-003 — Add a current validation matrix

Expose, in one public document, what can be validated on:

- ordinary CPU fresh clone;
- module-specific Python environment;
- CUDA environment;
- Foundry/local Anvil;
- heavy/private-local artifact environment.

Do not present unavailable heavy validation as a one-command fresh-clone test.

## S-004 — Improve issues/PR contribution ergonomics only where useful

If public issues are intended:

- add focused issue templates (bug / documentation / security redirect as appropriate);
- add a concise PR template that asks for validation and claim impact;
- avoid enterprise-style ceremony for a personal project.

## S-005 — Normalize project metadata selectively

Review inconsistent author metadata, package descriptions, Python-version statements, and environment names. Standardize only where it improves clarity; do not churn lockfiles or package boundaries without technical reason.

## S-006 — Add a social-preview image / architecture visual

After the canonical architecture is stable, create a clean GitHub social-preview image and optionally one screenshot/example report visual. This should support comprehension, not imitate a commercial SaaS product.

## S-007 — Surface AI-assisted engineering ownership intentionally

`CLAUDE.md` openly records the learning-by-doing/AI-assisted ownership model. Do not hide that history. Public-facing wording should emphasize that AI assistance may contribute code while architecture decisions, evidence discipline, validation, project ownership, and technical understanding are explicitly governed.

This is more credible than either pretending no AI was used or centering the entire README on AI tooling.

## S-008 — Reduce stale branch surface and decide main-protection policy

The repository currently exposes dozens of old remote branches, including many `claude/*`, historical `r4/*`, documentation, and system-alignment branches. Their existence preserves useful history, but the accumulated branch list makes it harder to distinguish active work from abandoned experiments. The branches API also reports `main` as `protected=false`.

For a solo personal project this is not automatically a release blocker, and GitHub ruleset state still requires separate verification. Before final portfolio release:

- identify branches still referenced by open PRs, accepted evidence, or reproducibility records;
- delete only branches that are clearly obsolete and whose commits remain safely reachable where needed;
- preserve historical R4/evidence anchors that still have documentary value;
- decide whether lightweight `main` protection (for example PR/check requirements) improves the public engineering story without adding unnecessary ceremony.

---

# 6. OPTIONAL items

These are not required for the first CV-ready release:

- `CONTRIBUTING.md` if outside contributions are actually desired;
- `CODE_OF_CONDUCT.md` if a contributor community develops;
- GitHub Pages/project website;
- Discussions;
- polished demo video/GIF;
- interactive hosted demo;
- Dockerizing the entire five-module/heavy-artifact system;
- one universal Python environment;
- extensive badge collections;
- public bug bounty;
- full historical dataset/model/proving artifact hosting.

Do not let these delay portfolio readiness.

---

# 7. Protected paths / cleanup exclusions

Professionalization must not destroy or silently rewrite reproducibility authority.

Treat the following categories as protected unless a separate evidence-preserving migration is explicitly designed:

- `docs/plan/ml-R4/` current plans, ADRs, decision/register artifacts, evidence snapshots, review bundles, hashes, manifests;
- accepted historical G7 publication/evidence artifacts;
- R4-D-008 repaired-v2 evidence roots and recorded identities/digests;
- R4-D-009 V3 logical evidence and durable snapshot;
- R4-D-010/011/012 decision/evidence records;
- Run12 historical model lineage/artifact references;
- retained ZKML proxy/circuit/verifier lineage required for reproducibility;
- V1/V2/V3 contract compatibility source/tests;
- source/tests whose historical semantics are explicitly preserved for compatibility.

Rules:

- prefer archive/navigation improvements over destructive deletion;
- never edit historical evidence files to make current claims look cleaner;
- never rename accepted artifact identities/digests for aesthetics;
- if paths must move, first prove all source/docs/workflow references and reproducibility contracts remain valid.

---

# 8. Explicit non-goals

Portfolio professionalization does **not** authorize:

- Phase-8 full training;
- new model-quality claims;
- threshold/calibration fitting without evidence;
- negative-label invention;
- selector mutation of R4-D-011 history;
- proxy/circuit regeneration before a selected repaired teacher;
- production chain deployment;
- adding a production signer/broadcaster;
- making `UNSAFE` EZKL settings look production-safe;
- rewriting the system architecture solely to look sophisticated;
- deleting historical evidence to make the repo look smaller;
- forcing every module into one dependency environment.

---

# 9. Recommended execution order after P0

The master plan phases remain useful, with one precedence rule:

### P1.0 — Truth alignment first

Close **B-001** with minimal current-state corrections before aesthetic/public-identity work.

### P1.1 — Hygiene and identity foundation

- DVC/tmp/gitignore cleanup;
- artifact/repository-size policy audit;
- stale PR and stale-branch containment;
- GitHub description/topics;
- explicit repo-name/license decisions;
- `SECURITY.md`.

### P2 — Landing/documentation architecture

- professional root README;
- canonical architecture;
- targeted module README alignment;
- public current-status/limitations navigation.

### P3/P4 — Showcase and developer experience

- monorepo environment contract;
- top-level command index/task runner;
- lightweight demo + example report;
- reproducibility matrix.

### P5 — CI/security/reproducibility polish

- active-vs-historical workflow separation;
- stronger documentation current-state validation;
- secret/security hygiene checks;
- stable public validation surface.

### P6/P7 — Case study and release

- technical case study;
- final evidence/claim review;
- stable portfolio release and GitHub presentation metadata.

### P8 — External-reader audit

Re-run three paths:

- recruiter: 2 minutes;
- senior engineer/security reviewer: 10 minutes;
- deep reviewer: trace claims to evidence.

---

# 10. P0 gate result

P0 requirements from the master plan are satisfied:

- BLOCKER/MUST/SHOULD/OPTIONAL inventory exists;
- protected evidence/history boundaries are explicit;
- stale/unsafe public claims are identified;
- environment/artifact boundaries are understood sufficiently to avoid a fake one-command setup;
- no product implementation or R4 semantic change was performed;
- execution precedence is defined.

**P0 status: `COMPLETE`.**

**Next authorized portfolio responsibility:** close B-001 with a bounded current-document truth-alignment pass, then proceed into P1 hygiene/identity work.