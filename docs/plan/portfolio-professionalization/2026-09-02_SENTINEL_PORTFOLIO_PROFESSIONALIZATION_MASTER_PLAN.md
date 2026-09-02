# SENTINEL Portfolio Professionalization Master Plan

**Date:** 2026-09-02  
**Status:** PLANNED — no portfolio implementation started by this plan  
**Branch:** `portfolio/professionalization-2026-09-02`  
**Scope:** whole-repository professionalization for public GitHub / CV use  

---

## 1. Purpose

SENTINEL already contains substantial engineering, research, governance, ML, agentic, ZK, and smart-contract work. The purpose of this program is **not to invent more features or make unfinished work look complete**. It is to make the existing project easy to understand, inspect, run where practical, verify, and discuss professionally when linked from a CV or portfolio.

The target outcome is a repository that communicates three things quickly and truthfully:

1. **What SENTINEL is and why it exists.**
2. **What was actually engineered and validated.**
3. **How a technical reviewer can inspect the architecture, code, evidence, tests, and current limitations without navigating internal research history first.**

This is a presentation, developer-experience, repository-hygiene, evidence-communication, and public-documentation program. It must preserve current technical authority and R4 evidence semantics.

---

## 2. Governing constraints

This plan remains subordinate to `CLAUDE.md`, executable source/config/tests, current R4 machine-readable authority, and accepted ADRs/evidence.

Permanent constraints for this program:

- Do not change DATA/ML truth merely to improve portfolio appearance.
- Do not claim Phase 8 / G8 is complete while it remains open.
- Do not present Run12 as the repaired R4 teacher.
- Do not invent model-quality, calibration, FPR, threshold, production, ZK, or blockchain claims unsupported by current evidence.
- Do not erase or rewrite protected historical artifacts merely because they look complicated.
- Do not collapse `unknown`, `unsupported`, `unavailable`, or `not-run` into negative/clean outcomes.
- Do not redesign working architecture only for aesthetic reasons.
- Do not turn the repository into ceremony-heavy open-source scaffolding without a practical need.
- Prefer **version/archive/navigation cleanup** over destructive deletion when historical material still has reproducibility value.
- Any repo rename, licensing choice, or other public-identity choice with legal/career implications remains an explicit decision rather than an automatic cleanup action.

---

## 3. External-review personas

Professionalization is accepted only if the repository works for all three audiences below.

### A. Recruiter / hiring manager — 2 minute path

Must be able to determine:

- problem domain;
- project purpose;
- strongest capabilities;
- technologies used;
- current project maturity;
- what Ali personally engineered/learned;
- where to see architecture/demo/results.

They should not need to understand R4 gate terminology first.

### B. Senior engineer / security engineer — 10 minute path

Must be able to determine:

- system boundaries and module responsibilities;
- request/evidence/data flow;
- important design decisions;
- reliability/failure semantics;
- test/CI/reproducibility story;
- what is implemented vs historical vs experimental;
- where deeper source/evidence lives.

### C. Deep technical reviewer

Must be able to trace public claims to:

- source/tests;
- accepted plans/ADRs/registers;
- reproducibility artifacts;
- example outputs;
- exact limitations and open work.

---

## 4. Known initial findings

These are starting findings, not the final audit inventory.

### Public GitHub surface

- Repository currently has no GitHub description.
- Repository currently has no topics.
- Repository currently has no detected license.
- Repository currently has no release.
- Repository name `sentinel-` should be reviewed for intentionality before CV publication.

### Landing-page accuracy

- Root `README.md` is stale relative to the current 2026-09-02 R4 authority and therefore is not yet safe as the primary external project description.
- Internal R4 terminology currently has a much stronger presence than a recruiter-friendly product/system narrative.

### Repository hygiene

- `.dvc/tmp` runtime/lock files are tracked and should not be part of the professional repository surface.
- `.gitignore` has accumulated project-specific and malformed-looking residue that deserves a controlled cleanup audit.
- Repository size is large enough to justify a tracked-history / generated-artifact / DVC audit before portfolio publication.
- Historical test/report/output folders and dated internal artifacts need navigation/containment review rather than blind deletion.

### Workspace / developer experience

- Root `pyproject.toml` presents a workspace-level package but packages only `ml` and declares root test paths that do not cleanly describe all current module environments.
- `agents/` has its own Poetry configuration and pytest override, indicating that the repository should explicitly document whether it is a multi-environment monorepo or a unified workspace.

### CI / project history

- `.github/workflows/` contains many phase-specific R4 workflows. This demonstrates engineering activity but creates a noisy active CI surface for an external visitor.
- Historical workflow preservation and current active CI presentation need to be separated.

These findings must be revalidated during Phase P0 before implementation.

---

## 5. Target public information architecture

The eventual repository should present information in this order:

1. **Root README — external landing page**
2. **Canonical architecture / system overview**
3. **Quick demo / example audit**
4. **Developer setup and validation commands**
5. **Module navigation** (`data_module`, `ml`, `agents`, `zkml`, `contracts`)
6. **Technical case study / engineering decisions**
7. **Current status and limitations**
8. **Deep handbook / R4 evidence / historical records**

Internal evidence remains available, but external visitors should not have to traverse dated plans before understanding the system.

---

# 6. Execution program

## P0 — Portfolio readiness audit and scope freeze

### Goal

Create an evidence-backed inventory of what needs repair before changing the public surface.

### Work

Audit:

- root tree and top-level naming;
- root and module READMEs;
- current handbook navigation;
- architecture diagrams;
- setup/install paths;
- dependency/environment boundaries;
- test entry points;
- CI workflows and current status visibility;
- GitHub metadata;
- license/security/community files;
- tracked generated/runtime files;
- DVC configuration and tracked large artifacts;
- repository history/size hot spots;
- stale branches and PRs;
- releases/tags;
- example/demo assets;
- screenshots/reports currently suitable or unsuitable for public use;
- claims in README/docs against current source/R4 authority.

Classify each finding:

- **BLOCKER** — damages truth, reproducibility, security, or professional credibility;
- **MUST** — needed before CV publication;
- **SHOULD** — materially improves engineering presentation;
- **OPTIONAL** — useful but not needed for the first professional release.

### Deliverable

A compact portfolio-readiness audit/checklist appended to this program or created as a working record if the evidence volume requires it.

### P0 gate

Do not begin broad cleanup until:

- all BLOCKER/MUST items are identified;
- protected historical/evidence paths are explicitly excluded from destructive cleanup;
- public claims that are stale or unsafe are identified;
- module/environment boundaries are understood well enough not to create a fake one-command setup.

---

## P1 — Repository identity and hygiene foundation

### Goal

Make the repository itself look intentional before polishing prose.

### Work

1. Review repository name and decide whether to keep `sentinel-` or rename it.
2. Define concise GitHub description and useful topics.
3. Decide license intentionally; do not add one without understanding its effect.
4. Correct `.gitignore` and DVC runtime hygiene.
5. Stop tracking machine-local/runtime lock/temp residue.
6. Audit large tracked files and repository-size causes.
7. Separate generated/reproducible outputs from source where appropriate.
8. Review stale PRs/branches for archive/close/delete decisions without losing useful history.
9. Review top-level directory clutter and move only clearly historical material when doing so preserves links/evidence.

### P1 gate

- no known machine-local runtime cruft tracked;
- no accidental secret/environment artifacts tracked;
- repo identity is intentional;
- license status is explicit even if the decision is to remain unlicensed temporarily;
- no protected R4 artifact/history has been destroyed;
- top-level tree has a defensible reason for every major directory.

---

## P2 — Root README and public documentation architecture

### Goal

Replace the current stale landing experience with an accurate professional project entry point.

### Root README required structure

1. Project name + one-sentence value proposition.
2. Short problem statement.
3. What SENTINEL does.
4. High-level architecture visual.
5. Major capabilities.
6. Technology stack grouped by responsibility, not badge spam.
7. Demonstration / example result.
8. Repository/module map.
9. Quick-start / demo path.
10. Validation / testing / reproducibility summary.
11. Current status and explicit limitations.
12. Engineering highlights / selected hard problems solved.
13. Deeper documentation links.
14. License/security links when applicable.

### Documentation rules

- README must not become the internal handbook.
- Prefer progressive disclosure: summary first, deep evidence by link.
- Avoid unsupported marketing language such as “production-ready”, “state of the art”, or “provably secure”.
- Explicitly distinguish implemented, historical, experimental, and planned capabilities.
- Keep R4 details accurate but move dense gate history behind a concise current-status summary.
- Module READMEs should explain responsibility, interfaces, run/test path, and links to deeper docs.
- Stale module READMEs should be updated or clearly marked historical; no contradictory “current” documents.

### P2 gate

A first-time reviewer can answer in under one README traversal:

- What is Sentinel?
- What problem does it solve?
- How is it architected?
- What is currently working?
- What is not claimed?
- How do I inspect or try it?

---

## P3 — Canonical architecture and trust-boundary presentation

### Goal

Expose the technical design clearly without changing architecture for presentation alone.

### Required canonical views

At minimum produce/update:

1. **Whole-system architecture**
   - `data_module`
   - `ml`
   - `agents`
   - `zkml`
   - `contracts`

2. **Normal analysis request flow**
   - client/gateway
   - LangGraph orchestration
   - ML/MCP/static/RAG/formal evidence
   - synthesis/report output

3. **Verifiability / on-chain trust path**
   - teacher/fusion boundary
   - retained proxy
   - EZKL proof scope
   - V3 attestation / registry scope
   - explicit statement of what proof does **not** prove

4. **DATA/ML lifecycle view**
   - source data
   - evidence/label semantics
   - leakage grouping/roles
   - representations
   - training/evaluation gates

### Diagram rules

- One canonical diagram per question; avoid duplicate diagrams that drift.
- Use GitHub-renderable Mermaid/ASCII where appropriate.
- Every trust boundary must label off-chain/on-chain and proof/attestation responsibility accurately.
- Historical protocols should be visually secondary to current V3/current R4 state.

### P3 gate

A senior engineer can understand module ownership and major trust boundaries without reading source first.

---

## P4 — Runnable showcase and developer experience

### Goal

Provide a credible “show me” path instead of requiring the reviewer to reconstruct the full research environment.

### Work

Define at least two setup levels:

#### Level A — lightweight showcase

A bounded, reproducible example using a small Solidity fixture and committed/small dependencies where possible. It should demonstrate useful Sentinel behavior without requiring full DATA regeneration, long training, or every external service.

Expected outputs should be committed or easily comparable so the reviewer knows what success looks like.

#### Level B — full/advanced development setup

Document the real module environments, services, GPU/data requirements, DVC artifacts, local LLM/tool requirements, contracts/ZK requirements, and limitations.

### Developer-entry-point review

Evaluate whether a simple task interface such as `make`, `just`, or repository scripts would materially improve:

- setup checks;
- core tests;
- docs/invariant checks;
- lightweight demo;
- module-specific validation.

Do not create a wrapper that hides incompatible environments or silently skips missing dependencies.

### P4 gate

- at least one bounded showcase path is reproducible;
- expected behavior/output is visible;
- advanced prerequisites are explicit;
- missing tools produce explicit degraded/error states, never fake success;
- README commands have been executed against a clean-enough environment or CI equivalent.

---

## P5 — CI, testing, security and reproducibility presentation

### Goal

Make engineering quality visible from the repository surface.

### CI work

- Inventory historical phase-specific workflows.
- Define a small set of clearly current checks for normal pushes/PRs.
- Preserve historical R4 workflows when they remain evidentially useful, but reduce confusion about which workflows represent current project health.
- Make failures actionable and names understandable to outsiders.
- Add README badges only for meaningful current checks.

### Testing/reproducibility work

Document:

- core test commands by module;
- what requires GPU/external binaries/services;
- what CI covers vs local evidence;
- deterministic/bound artifact boundaries;
- DVC/data acquisition or regeneration boundaries;
- why some expensive R4 checks are evidence gates rather than ordinary PR tests.

### Security/public-project files

At minimum evaluate and normally add:

- `SECURITY.md` with vulnerability-reporting guidance and project security scope;
- dependency/security scanning if it adds real signal;
- secret scanning / dependency update posture supported by the repository.

Evaluate `CONTRIBUTING.md`, issue templates, PR templates, `CODE_OF_CONDUCT.md`, and `CODEOWNERS` only if they serve actual collaboration needs. Do not add decorative community files solely for a score.

### P5 gate

A reviewer can distinguish:

- passing normal CI;
- expensive/manual evidence gates;
- module-specific tests;
- known unsupported environments;
- security reporting path;
- reproducibility claims.

---

## P6 — Technical case study and evidence package

### Goal

Convert Sentinel’s strongest engineering history into concise proof of engineering judgment.

### Recommended case-study themes

The final selection should use only evidence that remains current and traceable. Strong candidates include:

1. **Label-reality correction** — discovering that historical zero labels were not valid negatives and redesigning supervision semantics around explicit uncertainty.
2. **Leakage-grouping correction** — identifying address-literal over-grouping and replacing it with defensible artifact/code/family authority.
3. **Representation semantic defect** — identifying the v9 external-call limitation and evolving to V10 rather than training on known-bad semantics.
4. **Deterministic structural validation** — investigating full-population graph drift instead of weakening equivalence gates.
5. **Selector promotion discipline** — promoting a guarded selector only for a new versioned candidate while preserving the accepted rollback root.
6. **Explicit refusal to overclaim** — keeping threshold/calibration/untouched roles unsupported rather than manufacturing evaluation evidence.
7. **Agent evidence semantics** — preserving `ran=false` versus `ran=true/findings=[]` and explicit degraded states.
8. **ZK/provenance boundary honesty** — separating proxy proof from teacher/source/agent execution claims.

### Case-study format

For each selected theme:

- original problem;
- evidence that exposed it;
- incorrect/easy shortcut rejected;
- engineering decision;
- implementation/validation approach;
- measurable/inspectable result;
- limitation that remains.

Keep this concise enough for a technical interviewer to scan. Link to deeper ADR/evidence rather than duplicating entire internal records.

### Public example artifacts

Create a curated `docs/showcase/` or equivalent only if it improves navigation. Candidate contents:

- current architecture diagram;
- one example audit/report;
- selected engineering case study;
- reproducibility/validation summary.

Do not duplicate authoritative R4 artifacts into a second pseudo-authority.

### P6 gate

Every impressive statement in the portfolio layer has an inspectable source/evidence trail and a clearly stated scope.

---

## P7 — GitHub release and portfolio surface

### Goal

Create a stable version a CV can link to rather than pointing only at an ever-changing development head.

### Work

- Decide first portfolio release boundary only after earlier gates pass.
- Create a version/tag/release that describes what is included and what remains open.
- Ensure current `main` and release documentation agree.
- Add GitHub metadata/topics/description and optional social preview.
- Ensure release notes link to demo, architecture, case study, limitations, and reproducibility.
- Do not call the release “production” unless the system actually satisfies such criteria.

### P7 gate

The public release is self-consistent, reproducible at its stated scope, and contains no known stale portfolio claims.

---

## P8 — Final CV / interviewer readiness audit

### Goal

Evaluate the repository as an employer would, not as its authors would.

### Final review passes

#### Pass 1 — recruiter skim

From the GitHub landing page alone, verify clarity of:

- purpose;
- value;
- technologies;
- maturity;
- highlights;
- demo/architecture links.

#### Pass 2 — engineer audit

Verify:

- source organization;
- current architecture;
- failure semantics;
- tests/CI;
- setup commands;
- evidence/claim traceability;
- code/comment/documentation quality in representative files.

#### Pass 3 — adversarial credibility audit

Actively search for:

- stale claims;
- contradictory docs;
- inflated metrics;
- “production-ready” implications;
- hidden setup assumptions;
- dead links;
- tracked runtime/generated junk;
- accidental secrets/credentials;
- unsupported proof/security claims;
- screenshots/examples that no longer match source behavior.

#### Pass 4 — CV extraction

Only after the repo passes the above, derive the CV project entry from verified facts:

- 1-line project description;
- 2–4 impact/engineering bullets;
- technology line;
- GitHub/release link.

The CV wording must be downstream of repository evidence, not the other way around.

### P8 gate — PORTFOLIO READY

Portfolio-ready status requires all BLOCKER and MUST findings closed or explicitly accepted as visible limitations, plus:

- accurate root README;
- canonical architecture;
- reproducible showcase path;
- understandable validation/CI story;
- explicit current limitations;
- clean enough repository surface;
- public claims traceable to evidence;
- stable release/tag or equally stable portfolio boundary;
- final recruiter + engineer + adversarial review passed.

---

# 7. Priority model

## MUST before CV publication

- truthful/current README;
- intentional repo identity;
- no obvious tracked runtime cruft/secrets;
- canonical architecture overview;
- explicit current limitations;
- clear module map;
- credible setup/showcase path;
- understandable tests/validation story;
- GitHub description/topics;
- licensing status consciously decided;
- portfolio claims evidence-backed.

## SHOULD before first portfolio release

- current CI consolidation/presentation;
- `SECURITY.md`;
- technical case study;
- repository-size cleanup where safe;
- stale PR/branch cleanup;
- release/tag;
- representative module README refreshes;
- improved developer command surface.

## OPTIONAL / later

- GitHub Pages/project website;
- polished video walkthrough;
- extra screenshots/animations;
- broader contributor/community scaffolding;
- demo deployment;
- separate portfolio microsite.

Optional work must not block the first professional repository release.

---

# 8. Scope stop-lines

During this program, stop and separate the task if work becomes:

- new DATA/ML model research rather than documentation of current state;
- new Phase-8 implementation needed only to make the portfolio look finished;
- architecture redesign without an actual engineering defect;
- broad source refactoring unrelated to readability/reproducibility/public correctness;
- rewriting historical evidence to simplify the story;
- production deployment, real keys/funds, or external infrastructure;
- legal licensing choice without an intentional decision;
- an endless attempt to document every internal file before the repository can be shown.

Portfolio polish should expose strong engineering; it must not become a second multi-month feature project.

---

# 9. Execution discipline

For each phase:

1. Re-read only the authority needed for that phase.
2. Confirm the current source/repository state before acting.
3. Make the smallest coherent set of changes.
4. Validate links, commands, claims, and affected tests/checks.
5. Update this plan/checklist with completed findings and newly discovered blockers.
6. Commit by coherent responsibility rather than one giant cosmetic commit.
7. Do not mix unrelated active R4 implementation into portfolio commits.

If `main` advances materially during this program, rebase/synchronize before making state claims.

---

# 10. Recommended execution order

The default order is:

`P0 audit → P1 hygiene/identity → P2 README/info architecture → P3 canonical architecture → P4 demo/DX → P5 CI/security/reproducibility → P6 case study/evidence → P7 release → P8 final CV audit`

Exception: a P0 BLOCKER may be fixed immediately if it is clearly unsafe (for example exposed credentials), but the fix must still preserve evidence/history correctly.

---

# 11. Definition of success

This program succeeds when SENTINEL no longer needs Ali to verbally explain why the repository is impressive.

The repository itself should make clear that SENTINEL is a serious learning-by-doing engineering system spanning smart-contract security, ML representations/inference, agentic evidence orchestration, reproducibility/provenance, and verifiable/on-chain components — while remaining precise about which parts are historical, current, experimental, unsupported, or still under active R4 development.

The professional signal should come from **clarity, engineering decisions, inspectable evidence, reproducibility, failure handling, and intellectual honesty**, not from inflated feature counts or marketing claims.
