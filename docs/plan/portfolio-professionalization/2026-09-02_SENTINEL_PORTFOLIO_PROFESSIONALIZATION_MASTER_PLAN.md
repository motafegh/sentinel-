# SENTINEL Portfolio Professionalization Master Plan

**Created:** 2026-09-02  
**Last reconciled:** 2026-09-05  
**Status:** **IN PROGRESS**  
**Branch:** `portfolio/professionalization-2026-09-02`  
**Live status:** [`CURRENT_STATUS.md`](CURRENT_STATUS.md)  
**Scope:** whole-repository professionalization for public GitHub / CV use

## 1. Purpose

SENTINEL already contains substantial engineering, research, governance, ML, agentic, ZK, and smart-contract work. This program does **not** exist to invent features or make unfinished work look complete. It exists to make the real project easy to understand, inspect, validate, run where practical, and discuss professionally when linked from a CV or portfolio.

The target repository must communicate quickly and truthfully:

1. what SENTINEL is and why it exists;
2. what was actually engineered and validated;
3. how a technical reviewer can inspect architecture, code, evidence, tests, and limitations without navigating internal research history first.

For current progress and finding disposition, use `CURRENT_STATUS.md`. Dated audit/phase files are historical execution records, not live dashboards.

## 2. Governing constraints

This program remains subordinate to `CLAUDE.md`, executable source/config/tests, current R4 machine-readable authority, accepted ADRs/evidence, and the canonical handbook.

Permanent constraints:

- Do not change DATA/ML truth for portfolio appearance.
- Do not claim Phase 8 / G8 complete while it remains open.
- Do not present Run12 as the repaired R4 teacher.
- Do not invent model-quality, FPR, threshold, calibration, production, ZK, blockchain, signer, or broadcaster claims.
- Do not collapse `unknown`, `unsupported`, `unavailable`, or `not-run` into negative/clean outcomes.
- Do not erase protected historical/R4 evidence merely because it is complex.
- Do not rewrite Git history merely to reduce repository size; exact commit identities are part of the R4 evidence/provenance chain.
- Do not redesign working architecture only for presentation.
- Avoid ceremony-heavy OSS scaffolding without a practical need.
- Prefer current-navigation/versioning over destructive historical cleanup when evidence still matters.
- Repository rename and licensing remain explicit owner decisions.

## 3. External-review personas

### Recruiter / hiring manager — 2-minute path

Must be able to determine the problem domain, project purpose, strongest capabilities, technology stack, maturity, engineering ownership, and where to see architecture/demo/results without first learning R4 terminology.

### Senior engineer / security engineer — 10-minute path

Must be able to determine module boundaries, request/data/evidence flows, trust boundaries, major design decisions, failure semantics, validation/reproducibility story, and implemented-vs-historical-vs-experimental scope.

### Deep technical reviewer

Must be able to trace public claims to source/tests, current R4 authority, ADRs, reproducibility/evidence artifacts, example outputs, and explicit limitations.

## 4. Target public information architecture

The repository should present information in this order:

1. root README — external landing page;
2. canonical architecture/system overview;
3. bounded demo/example audit;
4. developer setup and validation commands;
5. module navigation (`data_module`, `ml`, `agents`, `zkml`, `contracts`);
6. technical case study/engineering decisions;
7. current status and limitations;
8. deep handbook/R4 evidence/historical records.

Internal evidence remains available, but external visitors should not traverse dated plans before understanding the system.

## 5. Current program position

As of 2026-09-05:

- **P0 audit:** complete;
- **B-001 truth alignment:** closed;
- **P1 hygiene foundation:** substantially complete;
- **P2 root landing-page work:** substantially complete;
- **P3 canonical architecture/trust presentation:** next;
- **P4–P8:** pending in sequence.

Completed work includes current-facing truth alignment, DVC/runtime cleanup, `.gitignore` hardening, stale PR/branch cleanup, `SECURITY.md`, bounded secret/exposure review, `DEVELOPMENT.md`, multi-environment setup clarification, regional package-index cleanup, repository-weight/history policy, and the public root README redesign.

The remaining detailed disposition is intentionally centralized in [`CURRENT_STATUS.md`](CURRENT_STATUS.md) so this master plan does not become stale again.

## 6. Execution program

### P0 — Portfolio readiness audit and scope freeze — **COMPLETE**

Purpose: identify BLOCKER/MUST/SHOULD/OPTIONAL findings, protect R4/evidence paths, and understand environment/artifact boundaries before broad cleanup.

Historical audit: [`2026-09-02_P0_PORTFOLIO_READINESS_AUDIT.md`](2026-09-02_P0_PORTFOLIO_READINESS_AUDIT.md).

### P1 — Repository identity and hygiene foundation — **SUBSTANTIALLY COMPLETE**

Responsibilities:

- runtime/temp/DVC hygiene;
- coherent ignore rules;
- secret/exposure posture;
- DVC/artifact public contract;
- stale PR/branch cleanup;
- repository size/history policy;
- security policy;
- explicit GitHub identity decisions.

Remaining P1/P7 identity work: repository description/topics when settings write access is available, and explicit owner decisions on repository name and license.

### P2 — Root README and public documentation architecture — **SUBSTANTIALLY COMPLETE**

Required qualities:

- one-sentence value proposition and problem framing;
- major implemented capabilities;
- high-level architecture;
- technology stack by responsibility;
- repository/module map;
- development/validation path;
- current status/limitations;
- engineering highlights;
- deeper evidence links;
- explicit AI-assisted engineering ownership without centering the project on tooling.

The root README now follows progressive disclosure and preserves current R4 claim boundaries. The runnable showcase/example remains P4 rather than being fabricated inside P2.

### P3 — Canonical architecture and trust-boundary presentation — **NEXT**

Produce/update one authoritative view per question:

1. **Whole-system architecture** — DATA, ML, AGENTS, ZKML, Contracts.
2. **Normal analysis request flow** — client/gateway → LangGraph → ML/MCP/static/RAG/formal evidence → synthesis/report.
3. **Verifiability/on-chain trust path** — fusion → proxy → EZKL proof scope → V3 attestation → registry, including what is *not* proved.
4. **DATA/ML lifecycle** — source/evidence semantics → grouping/roles → representations → evaluation/training gates.

Rules:

- avoid duplicate diagrams that can drift;
- label off-chain/on-chain and proof/attestation boundaries accurately;
- make historical/current/candidate/external status visually clear;
- do not alter architecture merely to make the diagram prettier.

P3 gate: a senior engineer can understand module ownership and major trust boundaries without reading source first.

### P4 — Runnable showcase and developer experience — **PENDING**

Provide at least one bounded example that does not require full DATA regeneration, multi-day GPU training, production RPC credentials, or a new proving ceremony.

Preferred shape:

`small Solidity fixture → bounded available Sentinel analysis/replay → structured expected output → explicit evidence-channel status`

If live execution depends on unavailable artifacts/services, ship a committed inspectable/replay output instead of faking execution.

Also evaluate a small common command surface (`make`, `just`, or scripts) only if it adds clarity without hiding incompatible environments.

### P5 — CI, testing, security and reproducibility presentation — **PENDING**

Responsibilities:

- distinguish current normal CI from historical R4 evidence workflows;
- improve semantic documentation-currentness validation beyond phrase presence;
- present module-specific tests and heavy/manual gates clearly;
- close the DATA lock/reproducibility decision;
- add a current validation matrix;
- perform a dedicated history/CI secret scan;
- evaluate dependency/security automation only when it adds real signal.

### P6 — Technical case study and evidence package — **PENDING**

Curate a small number of evidence-backed engineering decisions, such as:

- historical zero ≠ confirmed negative;
- leakage grouping repair after the 10,327-contract component;
- v9 external-call representation defect and V10 remediation;
- deterministic full-population drift reconciliation;
- guarded-selector promotion discipline;
- explicit refusal to manufacture unsupported evaluation evidence;
- AGENTS `ran=false` vs `ran=true/findings=[]` semantics;
- ZK proof vs V3 provenance boundary.

Each case study should state problem, evidence, shortcut rejected, decision, implementation/validation, result, and remaining limitation.

### P7 — GitHub identity and stable portfolio release — **PENDING**

After earlier gates:

- set accurate repository description/topics;
- decide repository name intentionally;
- decide license intentionally;
- create a stable tag/release describing included scope and open limitations;
- optionally add a social preview after architecture/showcase stabilizes.

Do not call the release production-ready unless that is actually true.

### P8 — Final CV/interviewer readiness audit — **PENDING**

Run four passes:

1. recruiter skim;
2. senior-engineer audit;
3. adversarial credibility audit for stale/inflated/unsupported claims, dead links, setup assumptions, secrets, and mismatched examples;
4. derive CV wording only from the verified repository state.

Portfolio-ready requires all BLOCKER/MUST items closed or explicitly accepted as visible limitations, plus accurate README, canonical architecture, bounded showcase, understandable CI/validation story, traceable claims, and a stable portfolio boundary.

## 7. Priority model

### MUST before CV publication

- truthful/current README;
- intentional repository identity;
- no obvious tracked runtime cruft/secrets;
- canonical architecture/trust overview;
- explicit current limitations;
- credible bounded showcase/setup path;
- understandable validation/CI story;
- licensing status intentionally decided;
- portfolio claims traceable to evidence.

### SHOULD before first portfolio release

- technical case study;
- current CI consolidation/presentation;
- current validation matrix;
- representative module navigation polish;
- useful developer command surface;
- social preview/representative visual if it materially improves comprehension.

### OPTIONAL / later

- GitHub Pages or separate project site;
- polished video walkthrough;
- hosted demo;
- broader contributor/community scaffolding;
- additional screenshots/animations.

Optional work must not block the first professional repository release.

## 8. Scope stop-lines

Separate the task if work becomes:

- new DATA/ML model research done only to make the portfolio look complete;
- architecture redesign without an actual engineering defect;
- broad source refactoring unrelated to correctness/readability/reproducibility;
- rewriting historical evidence to simplify the story;
- destructive history surgery for aesthetics;
- production deployment, real keys/funds, or external infrastructure;
- legal licensing choice without an intentional owner decision;
- an endless attempt to document every internal file before the repository can be shown.

Portfolio professionalization must expose strong engineering, not become another multi-month product phase.

## 9. Execution discipline

For each coherent chunk:

1. read only the authority needed for that responsibility;
2. confirm current repository/source state;
3. make the smallest coherent change set;
4. validate claims, links, commands, and affected checks;
5. update `CURRENT_STATUS.md` when the program state changes materially;
6. keep dated audit/phase files as historical records rather than pretending they are live status;
7. commit by coherent responsibility;
8. do not mix unrelated active R4 implementation into portfolio commits.

If `main` advances materially during this program, synchronize/re-evaluate before making new current-state claims.

## 10. Default execution order

`P0 audit → P1 hygiene/identity → P2 README → P3 architecture → P4 showcase/DX → P5 CI/security/reproducibility → P6 case study → P7 release/identity → P8 final audit`

Current restart point is **P3 after the 2026-09-05 state-reconciliation pass**.

## 11. Definition of success

This program succeeds when SENTINEL no longer requires its author to verbally explain why the repository is technically substantial.

The repository itself should show a serious learning-by-doing engineering system spanning smart-contract security, ML representations/inference, agentic evidence orchestration, reproducibility/provenance, and verifiable/on-chain components while remaining precise about what is historical, current, experimental, unsupported, unavailable, or still under active R4 development.

The professional signal must come from clarity, engineering decisions, inspectable evidence, validation, failure handling, and intellectual honesty—not inflated feature counts or marketing claims.
