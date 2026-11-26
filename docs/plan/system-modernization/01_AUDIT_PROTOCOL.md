# SENTINEL S4 - Audit, Verification and Modernization Decision Protocol

**Status:** ACTIVE upon program establishment

**Established:** 2026-11-26

**Parent:** [00_MASTER_PROGRAM.md](00_MASTER_PROGRAM.md) | [Current status](CURRENT_STATUS.md)

## 1. Purpose and application

This protocol makes S4 findings, comparative research, modernization decisions and remediation claims reproducible and reviewable. It applies to human and AI-assisted work, all modules and cross-module seams, and both local and remote evidence. It does not change domain semantics or authorize activities prohibited by CLAUDE.md, R4 or the technical-completion roadmap.

The governing distinction is between what exists, what was executed, what the evidence establishes, and what we decide to do. An audit is not complete merely because files were read; a fix is not accepted merely because code changed; a technology is not better merely because it is newer.

## 2. Audit units, scope and baseline

A work slice starts with a short record: slice ID (S4-XX-NNN), question, module/owner, actual source roots and entry points, baseline commit, related historical findings/decisions, expected outputs, safe commands, prerequisites, budget, stop condition, acceptance checks and excluded scope. A slice may be read-only, investigation, experiment, remediation or acceptance. Do not combine these implicitly.

Record source commit (full SHA), branch and tracked/untracked status; remote ref and exact CI head where relevant. Record hashes of relevant manifests, configs, locks, artifacts, fixtures, model/checkpoint or dataset lineage; interpreter/package/tool/compiler versions, OS/CUDA/device where pertinent. Never capture secret values or private environment dumps. Record protected artifact locations by safe identifier and availability; do not rebuild or change accepted roots merely to verify presence.

Classify each in-scope path as active runtime, active build/quality, supported historical compatibility, experimental, dormant, generated/vendor or archive. Confirm reachability by entry points/callers/configs and tests; unknown reachability stays unknown. Generated verifier and vendored code require interface/provenance/security review, not mechanical refactoring. Preserve historical artifacts and commits.

Coverage matrix: subsystem/path, lifecycle, relevant interface/invariant, review method, source inspected, tests executed, result, exclusions, reason and next owner. Distinguish inventory-only, static-reviewed, dynamically tested and acceptance-qualified. Sampling records population, selection method, coverage and limits. No full-audit claim from sampled inspection or a scan. Exact commands and source locations must be recoverable.

## 3. Evidence hierarchy and admissibility

Use source/config/tests for behavior; committed domain policy and versioned artifacts for semantics; measured results for claims about the tested conditions; current authoritative upstream documentation/release notes/advisories for version-specific external facts. Historical reports establish historical observations, not current correctness. A comment, README, agent assertion, tool silence, mock, missing result, or an old green run does not prove behavior.

Evidence states:

| State | Required interpretation |
|---|---|
| OBSERVATION | Inspectable source/tool fact, no defect verdict yet. |
| HYPOTHESIS | Plausible problem or improvement needing tests or further trace. |
| SOURCE_CONFIRMED | Exact current code/config demonstrates bounded behavior; exposure/runtime effects may remain unverified. |
| REPRODUCED | Controlled execution demonstrates stated behavior on identified baseline. |
| CORROBORATED | Independent evidence supports same claim, with method/dependencies identified. |
| INCONCLUSIVE | Conflicting or insufficient evidence; record competing explanations. |
| NOT_RUN / UNAVAILABLE | No normal evidence due to missing tool/artifact, budget or prerequisite; do not infer success. |

These are evidence states, not finding lifecycle or severity. Source proof may suffice for a deterministic narrow defect; security exploitability, model quality, utility or performance need appropriate additional evidence. Multiple tools/AI reviews sharing source, implementation or prior conclusion are not automatically independent. For independent blind evaluation follow its owning policy, isolate from candidate history and record contamination if detected. Do not expose blind reviewer to S4 records or expected conclusions.

Evidence record (minimal): evidence ID S4-E-NNNN; slice/finding IDs; UTC timestamp or recorded date/time zone; source SHA and path:line/symbol; command/working directory or method; environment/artifact/fixture IDs; expected/actual output; exit code and elapsed/resource measures when relevant; raw log or digest/path (redacted where needed); reproduction steps; limits/confounders; verifier/independence status. For upstream facts, include exact URL, version/date and access date; verify applicability to installed/locked version. Preserve raw failures and link summaries rather than rewrite them.

Store small sanitized evidence in Git; protected/high-volume logs, corpus, checkpoints, secrets or potentially harmful fixtures stay local under appropriate controls, with safe manifest/hash and access limitation. Never publish secrets, credentials, private data, live exploit artifacts or unauthorized third-party material. Redaction must be described; do not invent values. A Git commit alone is not proof external credentials were revoked.

## 4. Finding register and historical reconciliation

One canonical S4 finding ID S4-F-NNNN per root cause or distinct requirement; link all manifestations, affected modules and prior IDs. Seed IDs are leads only. No duplicated findings to inflate counts. Keep evidence references separate from prose. Minimum fields:

- ID, concise title, module/owner, scope/lifecycle, baseline SHA, affected path:line/symbol and cross-module seam;
- invariant/expected behavior with authority, actual behavior, reproduction or source proof, evidence state and evidence IDs;
- root cause vs symptom, trigger/preconditions, affected assets/users/claims, impact and reachability/exposure, confidence and unresolved ambiguity;
- historical references/disposition, severity and rationale, delivery priority and dependencies;
- proposed options (including retain/no-change), decision link if needed, acceptance tests, rollback and residual limitations;
- lifecycle state, last verification SHA/date, remediation/acceptance commit and reviewer where applicable.

Historical D2 reconciliation is mandatory for all 84 unique entries (86 raw rows minus two recorded duplicates); preserve original IDs, duplicate tombstones, severity, source baseline and acceptance references. Do not reclassify old evidence as new. Reconcile R0, V3, R4, later module and portfolio findings relevant to current scope. Reproduce closure only to the extent needed to establish current behavior; link existing accepted evidence, re-test when changed conditions or new contradictions warrant it.

| Historical disposition | Meaning |
|---|---|
| RESOLVED | Current applicable behavior meets intended invariant; cite fix and current/relevant regression evidence. |
| PARTIALLY_RESOLVED | Some failure modes addressed; residual scope identified. |
| STILL_PRESENT | Current applicable behavior reproduces or source-confirms issue. |
| SUPERSEDED | Architecture/requirement changed by accepted decision; old claim no longer applies, successor mapped. |
| REGRESSED | Previously resolved behavior fails now, with before/after evidence. |
| UNVERIFIED | Evidence insufficient or prerequisite unavailable. |
| NOT_APPLICABLE | Explicitly outside current target scope with rationale, not silent deletion. |

New findings use historical reference NONE or links to related but distinct issues. Preserve historical disposition separately from current finding lifecycle. A single historical item may map to several present findings; record mapping and avoid counting duplicated root causes. Do not declare all old P0s open based on original reports.

Lifecycle: CANDIDATE -> CONFIRMED or REJECTED or BLOCKED; CONFIRMED -> ACCEPTED_FOR_WORK or DEFERRED or ACCEPTED_RISK (explicit authority); ACCEPTED_FOR_WORK -> IMPLEMENTING -> VALIDATING -> CLOSED, or back to confirmed/blocked when tests fail. A rejected hypothesis records rationale; a closed item can reopen with new evidence. Risk acceptance does not prove repair. Evidence gaps can be confirmed as gaps without proving underlying defect.

## 5. Severity, confidence and delivery priority

Severity describes consequence within stated scope and credible preconditions, not coding effort or ordering. Record both demonstrated and plausible impact, distinguish local trusted artifact from reachable untrusted input, and do not claim exploitability from a sensitive call alone.

| Severity | Criterion |
|---|---|
| P0 CRITICAL | Confirmed or strongly evidenced critical security/trust/integrity violation, e.g. fabricated clean evidence reaching security decisions, unauthorized execution/write/signing, destructive loss, invalid acceptance authority. Contain affected scope immediately if safe. |
| P1 HIGH | Material correctness, reliability, scientific validity, security or reproducibility failure with substantial impact or plausible reachable exposure, requiring prioritized correction. |
| P2 MEDIUM | Bounded defect, meaningful maintenance/operational friction or incomplete safeguards with limited immediate impact. |
| P3 LOW | Minor issue, polish or low-risk improvement; not a reason to disrupt accepted behavior. |

An EVIDENCE_GAP is a finding category, not a severity shortcut; assign impact if missing evidence blocks a critical claim or mark severity pending. Categories include correctness, security, scientific validity, interoperability, reproducibility, reliability, performance, maintainability, governance, usefulness, opportunity. Modernization opportunity is not automatically a defect.

Confidence HIGH/MEDIUM/LOW must explain why (deterministic source, reproduced test, plausible but untested exposure, sampling limits). Do not assign precise numeric confidence unsupported by measurement. Urgency/priority: NOW (containment/baseline blocker), NEXT (unblocks important work), PLANNED (valuable with prerequisites), DEFERRED (cost/risk/low value or blocked). Consider impact, exposure, evidence, dependency graph, cost, reversibility, learning and user utility. Do not label every foundational task P0.

## 6. Source and security audit procedure

1. Read current governing requirements and inspect entry points, callers, config, tests and data/control flows. Identify invariants and cross-module meaning, not merely names/shapes.
2. Use AST/lint/type/static analysis, dependency/advisory scanners, coverage and complexity tools for candidates; manually verify reachability and semantics. No blanket changes, auto-fix of generated files or scanner counts as defects.
3. Trace failure and success paths including missing dependency, malformed input, empty results, timeouts, cancellation, retry, partial writes, cache staleness, concurrent access and version/identity mismatch. Distinguish skipped, unavailable, unsupported, clean and failed.
4. For security, map assets, actors, trust boundaries, entry points, privilege, attack preconditions and consequence. Test synthetic benign/malicious fixtures inside disposable isolated workspaces with minimal permissions and no real credentials/funds; do not execute untrusted contract code or unsafe artifacts outside containment. Never probe live third-party systems without explicit authorization.
5. For artifact loading, trace path selection, ownership/permissions, integrity verification before deserialization, source trust and network ingress. Consider safe formats, allowlisted tensor/schema loads or isolated legacy conversion; verify numeric/shape/schema/lineage parity. Unrestricted pickle is a risk surface, not proof of reachable exploit.
6. For AI/agent/RAG, treat retrieved documents, contract comments, tool output and model responses as untrusted data. Test prompt injection, fabricated citations, duplicate evidence, incorrect tool status, malformed structured output and adversarial retrieval. Model assertions cannot authorize sensitive actions or fabricate verified evidence.
7. Establish smallest deterministic reproduction; use unit/integration/end-to-end as appropriate. If a safe reproduction is impossible, retain bounded source finding/hypothesis and blocker. Do not weaken checks to get green results.

Containment before broad repair: stop affected unsafe paths and record scope if an actual serious exposure is found. Avoid unnecessary exposure of exploit detail; follow SECURITY.md. Do not deploy, rotate external keys, move funds or change production state without authorization.

## 7. Baseline tests, metrics and acceptance

Maintain test matrix per module/seam: test command, purpose/claim, exact environment, prerequisites, run ID/commit, result (PASS, FAIL, SKIP, ERROR, NOT_RUN, BLOCKED), counts and reason. Do not hide setup/collection errors or count skipped tests as passed; compare denominators and test identities before comparing totals. Use fresh clone/clean environment for reproducibility claims and protected-local runs for artifact-dependent claims. Do not assume a module lock passes because an installed environment works, or that CI proves clean-clone runtime.

Run narrow regressions first, then relevant module suites and cross-interface checks, integration/smoke and exact-commit CI according to blast radius. Historical gate replay uses frozen artifact identities and original semantics; a stale assertion may be repaired only when evidence proves the invariant is retained. Do not rewrite history or downgrade gate requirements.

For performance measure environment, workload, cold/warm condition, repeats, latency distribution, throughput, memory/VRAM/disk, failures and cost as appropriate. Report uncertainty; do not infer general improvement from one favorable run. Compare equivalent workloads/configs.

For ML/DATA preserve contract x class truth, role masks, leakage groups, class support, immutable lineage and independent evaluation. Positive fit is not discrimination; proxy fidelity is not model accuracy; historical zeros are not negatives. No calibration/threshold claims without eligible roles. For AGENTS measure correctness, evidence/provenance completeness, tool status, routing/abstention, false-clean under failures, report utility and resource cost. For ZK/chain verify exact proof scope, public-signal/identity parity, mutation rejection, attestation vs finality and upgrade/storage invariants.

Define success criteria before comparison or remediation; numerical gates need evidence and explicit scope. If adequate sample/support is absent, report unsupported or inconclusive, not fabricated thresholds.

## 8. Modernization decision and experimental protocol

Each consequential option has a decision/experiment ID S4-D-NNNN / S4-X-NNNN, question and reason, target task, baseline/source identity, alternatives (retain/no-change, simpler approach, repair, refactor, modernize, replace, retire), relevant finding IDs and upstream sources. Include exact versions, compatibility constraints, data/role/fixture selection, environment, seeds, repetitions, budgets, measures, criteria, security and stop rules before running. Separate discovery/validation data where applicable; never tune on held-out acceptance populations. Record preregistration changes with reason rather than silently moving goalposts.

Use isolated scratch worktree/environment, pinned lock/tool versions, safe local models by default where feasible, reproducible scripts/config, no protected artifact overwrites. No unapproved cloud data transfer, expensive training, signing, broadcast or irreversible deployment. Compare current implementation and appropriate simpler baseline under comparable conditions; document differences and confounders.

Decision criteria are correctness and evidence quality, security/trust, usefulness, performance/resources, maintainability/complexity, supported dependency lifecycle/licensing, reproducibility, migration and rollback, total cost and learning value. Weight only when a justified decision rule exists; do not invent numerical scoring. Report negative/inconclusive results, limits and alternatives rejected. Prefer smallest design that meets actual requirements, not newest framework or maximal abstraction.

Decision statuses: PROPOSED, EXPERIMENT_READY, INVESTIGATING, ACCEPTED, REJECTED, DEFERRED, SUPERSEDED. Accepted choice records evidence, rationale/tradeoffs, owner/authority, scope/non-goals, compatibility/rollback, acceptance tests, migration ordering, remaining risks. Domain-policy changes require owning ADR/decision; S4 record links rather than replaces it.

## 9. Implementation and closure

Before code: link accepted finding/decision, exact baseline, scope/changed interfaces, migration/rollback, tests and expected effect; write working record for nontrivial work. Preserve unrelated dirty files, inspect diff, avoid broad formatting/dependency refresh unrelated to slice. Implement narrow coherent changes, add regression where possible, preserve source/config/artifact identity. For semantic changes create versioned successor and explicit migration/rebind; do not modify accepted historical bytes.

Validate before and after with equivalent tests; include negative and failure cases, cross-module consumers and docs/claims. Record failed attempts, residual limitations, environment differences and exact commit. Confirm staged files are intended; git diff --check and applicable governance checks. Use main for validated slices, no force-push; verify remote SHA and exact-commit CI, distinguish local pass from remote pending/fail. If CI cannot complete, record pending/blocker and do not call it green.

Closure requires the original invariant/acceptance test is satisfied under stated scope, evidence linked, no new material regression, domain authority preserved and rollback documented. Otherwise mark partial/blocked or reopen. For retained/rejected options, record evidence-based rationale rather than invent a fix. Independent review is required when owning policy demands it; no assistant self-certifies blind independence.

## 10. Stage gates and progress reporting

PASS: required outputs, evidence and invariants met. PASS_WITH_LIMITATIONS: explicitly bounded claims established, non-critical limitations recorded with owners, restrictions and follow-up. BLOCKED: required evidence/prerequisite/authority absent; not a pass. FAIL: executed gate did not establish required invariant. NOT_STARTED/READY/IN_PROGRESS are execution states, not gate results. A stage may finish its audit with open remediation work, but it cannot claim unconditional acceptance of affected capability.

At each milestone update CURRENT_STATUS with active slice, source/evidence commit, work completed, finding/decision IDs, test/CI results including pending, residual risk/blocked scope, protected state and one concrete next action. Provide the owner a concise explanation of what changed and why, key evidence and any decision made without explicit direction. For learning, connect concept -> real code -> failure prevented -> practical checkpoint; bound time/effort and avoid invented mastery claims.

Final S4 acceptance requires reconciled scope/coverage, historical/current findings, decision and implementation dispositions, integrated test evidence, current claims/limitations, authority and reproducibility. Required unresolved gates or critical uncontrolled exposure prevent unconditional closeout. Production readiness, full training, model promotion and live chain actions remain separate domain decisions.

## 11. Minimal working templates

Use these fields in the owning record or machine-readable equivalent; do not create empty files for every slice.

**Slice:** ID | question | source SHA | scope/exclusions | authority/owner | prerequisites | methods/budget | evidence outputs | exit/stop | next.

**Finding:** ID | invariant | actual | affected path/flow | evidence IDs/state | root cause | impact/exposure | confidence/severity/priority | historical links | lifecycle | acceptance/rollback | residual | verification SHA.

**Experiment:** ID | hypothesis | control/candidates | versions | task/population/roles | budget/seeds | metrics/criteria | results/limits | decision link.

**Gate:** stage | required outputs | checks and evidence | result | limitations/owners | source/evidence SHA | next.

The protocol is deliberately evidence-oriented rather than ceremony-oriented: use one concise record per coherent slice, reuse existing source and evidence, and stop recording when the next action and claims are unambiguous.
