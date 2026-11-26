# SENTINEL S4 - Comprehensive System Audit, Modernization and Evolution Program

**Status:** APPROVED FOR PROGRAM ESTABLISHMENT; S4-00 ready

**Established:** 2026-11-26

**Reconnaissance baseline:** `61854a32a2badbcd84b98cef3f3976f070b1341e`

[Current status](CURRENT_STATUS.md) | [Audit protocol](01_AUDIT_PROTOCOL.md)

## 1. Mandate and intended outcome

SENTINEL is an evidence-aware smart-contract security research and engineering system. Its original learning/building era, later R4 research and remediation, and portfolio professionalization have created substantial capabilities, historical evidence and potential technical debt. S4 reassesses the whole system rather than continuing old designs by inertia.

Answer three independent questions:

1. Is the implemented system correct, secure, reliable, reproducible and maintainable within its stated boundaries?
2. Does each active capability contribute to a coherent and useful security workflow, and do guarantees compose across interfaces?
3. Can better designs, methods, technologies or operational models deliver measurable benefits at acceptable cost and risk?

Success means evidence-backed improvements in actual usefulness, scientific validity, correctness, security, reproducibility or maintainability, with residual limitations explicit. It is not measured by numbers of findings, agents, features, newer dependencies, lines changed, or documents.

S4 is not a rewrite, dependency-upgrade campaign, new taxonomy or production launch. Retain what works; repair demonstrated defects; simplify unnecessary complexity; experiment with alternatives when a meaningful question justifies it. Learning and ownership accompany concrete engineering.

## 2. Authority and scope

Read [CLAUDE.md](../../../CLAUDE.md) and applicable module instructions first.

| Existing authority | S4 relationship |
|---|---|
| Executable source/config/tests | Establish implemented behavior; measured runs establish bounded observed behavior. |
| [R4 control plane](../ml-R4/PLAN_STATUS_MATRIX.md) | Owns DATA/ML semantics, lineage, evaluation and launch gates. S4 cannot supersede it implicitly. |
| [Technical completion roadmap](../system-finalization/technical-completion/00_MASTER_TECHNICAL_COMPLETION_ROADMAP.md) | Owns remaining module-specific completion dependencies. Coordinate overlapping slices rather than duplicate work. |
| [D2 audit package](../../audit/system-engineering/2026-07-14_SYSTEM_D2_source_engineering_audit/2026-07-14_SYSTEM_D2_AUDIT_package_index.md) | Historical finding and acceptance baseline. Reconcile, do not automatically reopen or inherit severity. |
| R0 containment, V3 alignment, accepted R4 ADRs/artifacts | Preserve validated fixes, safety invariants and immutable evidence/rollback roots. |
| [Portfolio status](../portfolio-professionalization/CURRENT_STATUS.md), [handbook](../../handbook/01_architecture.md) | Navigation and claim boundaries; reconcile currentness, do not repeat completed work. |
| S4 master/protocol | Own new audit methodology, modernization research, sequencing and S4 gates. Findings are not domain policy. |

Scope: all DATA, ML/inference, AGENTS/LLM/RAG/MCP, ZKML, contracts, cross-module interfaces, dependencies/supply chain, CI, operations, security, tests, docs and repository governance. Classify assets as active, supported historical, experimental, dormant, generated/vendor or archival. Audit historical paths when compatibility, retention, security or reproducibility requires it.

If source behavior contradicts accepted semantics, resolve with the appropriate owner; neither source nor prose silently changes the other. Use versioned decisions for material semantic, protocol, interface, architectural or authority changes, not every reversible refactor.

Ali retains project ownership and delegated project direction, architecture/design and engineering decisions on 2026-11-26. Assistants make evidence-based choices and record trade-offs without repeated approval. Separate authorization remains required for destructive history/data changes, production exposure, real keys/funds, irreversible deployment/upgrade or material external cost/private data transfer. S4 grants none of these permissions.

## 3. Protected boundaries

- Historical R4 G0-G7 remain accepted for their exact frozen evidence; G8 is open. Replay failures require investigation, not automatic historical invalidation or dismissal.
- R4-D-009 logical V3, R4-D-011 V10/V2.6 graph/control and R4-D-013 guarded-token roots remain immutable with their recorded full digests. Preserve [4,512] tokens and four_eye_v8/v8.1 architecture unless a separate accepted successor changes them.
- R4-D-014 authorizes bounded positive-only control/evaluation; R4-D-015 closed M4 (8 epochs/120 steps) without discrimination evidence or full-training permission. Run12 remains historical operational inference; M4/experimental checkpoints are not its automatic replacements.
- R4-D-016 governs CallToUnknown and unchanged ten-class order. Candidate #2 remains UNKNOWN/PENDING_REVIEW/target None pending admissible isolated independent review; v2 ZIP is published and verified. Candidate #1 needs fresh primary review if reused. Keep blind reviewers outside candidate-history context.
- Confirmed negatives remain zero at baseline. Unknown, unavailable, failure, silence and historical zero are not negatives. Accepted negatives are evaluation-only absent separate optimizer authority. No pseudo-negatives, PU/nnPU launch, threshold/calibration fitting, untouched acceptance reuse, full M5 training, G8 passage or Run12 learned-state reuse by S4 alone.
- Gateway completion is off-chain; audit MCP is read-only; signing, broadcasting and finality are separate trust domains. Preserve V1/V2 reads, V3 context binding and UUPS storage compatibility.
- Retained EZKL proof establishes proxy computation only; context attestation is separate. Investigate check_mode="UNSAFE" without stronger assurance claims. Preserve historical proof/key/verifier artifacts.
- No weakening gates, invented evidence, hidden failures, in-place mutation of accepted roots or hostile execution outside authorized containment. Use versioned successors and rollback by selection.

## 4. Initial reconnaissance seeds (not adjudicated S4 findings)

At baseline: 292 core Python/Solidity files, 64,738 physical lines (including generated verifier, comments and blanks); 90,066 tracked paths, 86,168 under docs/.bin; 25 workflows. Counts are scope signals, not quality measures.

| Seed | Observation | First owner |
|---|---|---|
| S4-SEED-01 | Root Poetry configuration rejected local Poetry 2.2.1 source priority; AGENTS lock check failed; ML/DATA lock checks passed. | S4-02 |
| S4-SEED-02 | AGENTS locked LangChain 0.3.28/LangGraph 0.2.76 vs local 1.3.1/1.2.0; DATA Slither lock 0.10.0 vs installed 0.11.5. | S4-02 |
| S4-SEED-03 | Unrestricted torch.load/pickle.load in ML, DATA, RAG and ZKML; source sites confirmed, exposure not established. | S4-07/module owners |
| S4-SEED-04 | graph_extractor.py 2,327 lines; trainer.py 2,091; DATA cli.py 1,053; predictor.py 931; AGENTS _submit.py 888. Size alone is not a defect. | S4-03/04/05/08 |
| S4-SEED-05 | AST scan: 15 broad-exception handlers containing pass in DATA, 2 in ML; unchecked-scope feature path catches exception then returns 0.0. Reachability/semantics need investigation. | S4-03/04 |
| S4-SEED-06 | Current handbook/current_r4 metadata omitted D-015/D-016; candidate #1 wording stale. | S4-08/R4 |
| S4-SEED-07 | Six of seven exact-baseline CI workflows passed; historical G4 failed on missing review digest reference after frozen artifact hash checks passed. | S4-00/02 |
| S4-SEED-08 | Historical repository weight and nine pre-existing untracked WSL files; no deletion authority. | S4-00/08 |

[Exact G4 run](https://github.com/motafegh/sentinel-/actions/runs/45817256741). D2 recorded 84 unique findings/requirements (6 P0, 62 P1, 15 P2, 1 P3 evidence gap) at an older baseline. Reconcile every unique entry to current disposition; these are not current open-defect counts.

## 5. Execution architecture

S4-00 precedes other work. S4-01 frames usefulness/scope; S4-02 can start after S4-00. S4-03 through 06 can proceed in bounded parallel slices; S4-07/08 are cross-cutting. S4-09/10 run incrementally when prerequisites exist; do not wait for a global report before repairing a demonstrated serious defect. S4-11 integrates results.

Stage gates are PASS, PASS_WITH_LIMITATIONS, BLOCKED or FAIL per protocol. Audit completion does not imply remediation or release acceptance. Each stage covers source, tests, configuration, relevant historical findings, interfaces, security and measurable alternatives; record explicit exclusions.

### S4-00 - Baseline and audit initialization

- Freeze commit, local/remote state, lifecycle/ownership inventory, config/lock/runtime fingerprints, protected artifact identities, workflow triggers and exact-commit results.
- Identify test commands/prerequisites; run safe smoke and baseline suites where feasible. Record failures, skips, collection errors and unavailable artifacts.
- Reproduce G4 assertion failure and trace exact evidence. Repair verified non-semantic register/validator drift if warranted; preserve frozen evidence and gate semantics.
- Inventory D2 registry/acceptance and R0/V3/R4/portfolio closeouts; initialize reconciliation queue. Inventory untracked files safely without secret contents or unrelated staging.

Outputs: baseline inventory, coverage/run evidence, initial findings register and reconciliation queue. Gate S4-G00: exact baseline, every module inventory and test/limitation status, critical discrepancies assigned, protected state unchanged.

### S4-01 - Architecture, usefulness and scope

- Trace actual entry points, runtime/offline flows, interfaces, state ownership, trust/deployment topology. Separate Run12 operations, R4 research and proof/chain.
- Define intended users, tasks, inputs/outputs and failure expectations; distinguish intended from validated usefulness.
- Compare deterministic analyzer/report baseline and incremental ML, RAG, agent, formal, proof/chain contributions and costs. Identify complexity not serving a meaningful task.
- Map semantic contracts, coupling, duplication and alternatives. Propose scope/non-goals, including separate research or retained legacy tracks; record material product decisions and trade-offs under delegated authority.

Outputs: architecture/capability matrix, use-case hypotheses, interface map, options/scope proposal. Gate S4-G01: actual vs intended behavior distinguished; each significant capability has purpose, testable claim, trust boundary or explicit open question.

### S4-02 - Dependencies, infrastructure and CI

- Compare manifests, locks, installed and CI versions; Python/OS/CUDA/ABI, CPU/GPU, runtime/dev extras, external tools.
- Audit clean install/import/smoke, resolver constraints, package-source trust, build isolation, lock drift and artifact prerequisites. Evaluate root workspace retention, PEP 621, Poetry vs uv; no blanket migration.
- Check exact-version authoritative advisories, reachability/exposure, licensing/provenance, CI action pins/permissions, SBOM and automation. Advisory presence alone is not exploitability.
- Separate current CI, module suites, integration, security, historical replay, expensive research and release gates; review trigger coverage, false green/red risks and cost.
- Propose reproducible environments, bootstrap/test commands and rollback; do not update historical lineage in place.

Outputs: compatibility matrix, dependency risks, tooling options, CI design. Gate S4-G02: locks validated or blocked, clean-clone limits measured, no implicit promotion.

### S4-03 - DATA and scientific evidence

- Trace acquisition/licenses/provenance, compiler versions, normalization/dedup/grouping, label evidence/roles, verification, graph/tokens, export/registry/DVC.
- Investigate silent failures, alignment, instability, leakage, semantic drift, compiler coverage, resource usage with representative malformed/legacy/multi-contract/long cases.
- Review negative-evidence acquisition feasibility, class support, independent review, sampling bias and role eligibility under R4; coordinate GAP-007 without adjudicating in S4.
- Evaluate parsers, representations, stores, versioning and annotation in isolated successors where justified; verify downstream parity and immutable roots.

Outputs: lineage/coverage map, findings, scientific gaps, tests/experiments. Gate S4-G03: active paths and relevant historical findings accounted for; assumptions validated or bounded; accepted inputs protected.

### S4-04 - ML, inference and MLOps

- Trace dataset/batch/mask/model/objective/metric/checkpoint/serving/agents/proxy seams; distinguish Run12 v9 from R4 V10.
- Audit train/serve parity, shape/class/config, numeric behavior, deserialization, thresholds/calibration, admission/timeouts, concurrency, reproducibility, telemetry.
- Reassess four-eye architecture and positive-only limitations, label mechanism and alternative learning assumptions. Compare simpler baselines, graph/code embeddings, GNN/code models, self-supervision/hybrid designs where justified.
- Plan bounded ablations, per-class support and uncertainty, latency/VRAM/disk and compatibility on available hardware. Evaluate only eligible populations; separate fit, discrimination, calibration, runtime quality. No unapproved optimizer authority.

Outputs: training/serving map, diagnostics, evaluation feasibility, alternatives/artifact strategy. Gate S4-G04: claims bounded; launch/promotion holds preserved.

### S4-05 - AGENTS, LLM, RAG and security tools

- Trace graph, routing, state/reducers/checkpoints, tool/MCP contracts, fusion/consensus, prompts/structured output, evidence IDs and report. Audit double counting, deterministic vs LLM authority and abstention.
- Validate real Slither/Aderyn, symbolic/formal (including Halmos assumptions), graph and compiler capabilities. Compare tools only for genuine gaps.
- Audit retrieval, embeddings/reranking, source citations, freshness, feedback promotion, prompt injection/poisoning, serialization and concurrent updates.
- Compare local/open and authorized external models, adaptive vs fixed, single vs multi-agent, retrieval/tool-selection/state strategies. Use deterministic/no-LLM/no-RAG baselines, held-out cases, missing-tool/adversarial tests and resource budgets. No unauthorized data transfer.
- Investigate durable jobs, recovery/cancellation, retries, report consistency, latency/cost and analyst usefulness.

Outputs: graph/tool capability matrix, eval protocol/results, failure traces and options. Gate S4-G05: significant tool/report claims supported or explicitly unsupported; no false-clean evidence; autonomy bounded.

### S4-06 - ZKML and blockchain

- Trace teacher/fusion/distillation/ONNX/calibration/EZKL settings, circuit/SRS/keys/proof/public signals/verifier/V3 attestation; test exact identities, parity and mutation.
- Research pinned EZKL check_mode semantics using authoritative source. Evaluate safer settings, migration cost, proxy utility and alternative trust designs; proxy fidelity is not vulnerability correctness.
- Audit compiler/deps, storage/initializers, auth, replay/domain binding, signal mutations, staking, pause/rotation, V1/V2 reads and digest parity. Run Foundry unit/fuzz/invariant and isolated local-chain tests if available.
- Reassess signer/quorum/finality scope; preserve separate trust domain. New teacher/verifier awaits authorized handoff; no real-chain actions.

Outputs: trust/claim matrix, lineage, compatibility/security tests, utility/migration options. Gate S4-G06: exact claim and artifact scope, risk and prerequisites mapped; no unauthorized migration.

### S4-07 - Cross-system security, reliability and operations

- Threat model actual actors/assets, ingress, service identities, files/archives, deserialization, subprocess/network/RPC, RAG/LLM/tool input, report writes, secrets and chain policy.
- Reconcile D2/R0 containment via regression; investigate path traversal, unsafe loads, command execution, resource exhaustion, auth, untrusted evidence and sensitive logging. Synthetic isolated fixtures only.
- Test recovery/retry/idempotency, cancellation, concurrent admission, atomic state, schema/version mismatch and outage. Preserve unavailable vs clean and read-only MCP.
- Review logs/traces/metrics, readiness, configuration, deployment isolation, restore and performance; measure cold/warm/degraded end-to-end cases.

Outputs: threat model, verified findings, failure-injection/integration evidence, operational scope. Gate S4-G07: critical risks tested or explicitly blocked; containment preserved; no unsupported public-readiness claim.

### S4-08 - Maintainability, repository and governance

- Assess reachability, responsibility, coupling, duplication, dead/legacy paths, testability, types, lint/formatting, errors. Metrics are signals, not automatic defects.
- Reconcile handbook, current-state validators, plans/ADRs and instructions; single authority with checked/generated projections where useful.
- Investigate docs/.bin, archive access, licensing, clone/index/agent-context cost, untracked files and retention. No history rewrite or protected deletion.
- Reconcile every unique D2 entry and relevant later findings with current evidence; preserve unresolved items/non-goals.

Outputs: lifecycle/ownership map, debt register, documentation/CI and retention options. Gate S4-G08: concrete friction/risk supports change; navigation and authority unambiguous; history preserved.

### S4-09 - Modernization experiments and decisions

- Register question, baseline, pinned candidates, data/roles, budgets and criteria before execution; include no-change/simpler controls.
- Compare retain, repair, refactor, modernize, replace or retire across correctness, safety, latency/resources, maintainability, provenance, compatibility, cost, rollback and learning.
- Research current official releases/docs/advisories for selected alternatives; do not select by recency, popularity or unrelated benchmark.
- Isolate experiments; preserve adverse/inconclusive results and rejected alternatives; record material scope decisions; route domain changes through their owning authority.

Outputs: decision index and experiment packets. Gate S4-G09: consequential choices supported by inspectable comparisons and compatibility/rollback or explicitly deferred.

### S4-10 - Prioritized implementation

- Convert confirmed findings/accepted choices into bounded slices with owner, prerequisites, risk, tests, before/after evidence and rollback.
- Prioritize containment/baseline correctness, reproducibility and high-value capabilities. Preserve refactor behavior; version semantic changes.
- Run narrow then relevant wider suites, seam contracts, docs and CI. Use canonical main for validated slices; isolated scratch worktrees/environments as needed. No force-push or unrelated staging.
- Stop/re-plan when assumptions fail; integrate progressively rather than accumulate indefinite audit backlog.

Outputs: living execution roadmap, validated commits, residual risk. Gate S4-G10: accepted slices closed by evidence; others explicitly deferred/blocked/rejected/scheduled; no hidden regression.

### S4-11 - Integrated acceptance and learning closeout

- Re-run clean-clone/module/integration scenarios at exact accepted identities, including degraded/adversarial paths; compare before/after outcomes and costs.
- Reconcile findings, historical dispositions, decisions, claim matrix, docs and CI; distinguish current operational, restricted, research-only, rejected and deferred capabilities.
- Deliver owner-oriented architecture/operation walkthrough, key decisions/failure modes and practical exercises; state next maintenance roadmap. Production-readiness requires separate explicit acceptance.

Outputs: integrated acceptance report, claim/risk matrix, reproducible runbook, learning closeout. Gate S4-G11: every stage has bounded disposition; critical residuals have owners/restrictions; required unresolved gates preclude unconditional completion.

## 6. Cadence and decision discipline

Severity (impact/exposure) differs from delivery priority. Prioritize risk, dependencies, evidence confidence, usefulness, cost and reversibility. Serious demonstrated exposure or invalid baseline may interrupt sequence; speculative concerns do not authorize broad rewrites.

Each slice has a question, scope, owner, prerequisites, command/resource budget, acceptance criteria and stop condition. Checkpoint durable findings before expanding scope. If blocked, record why and choose independent bounded work. Keep R4 independent review eligible in parallel; one accepted negative does not unlock M5.

Routine approvals are delegated by CLAUDE.md. Ask only for non-routine scope/safety decisions. Integrate brief concept-to-code walkthroughs, trade-offs and practical checkpoints with real milestones; avoid endless theory and inflated individual authorship claims.

## 7. Records and restart

Start with this master, protocol and CURRENT_STATUS.md. S4-00 creates 02_BASELINE_INVENTORY.md and 03_FINDINGS_REGISTER.md (or concise indexes into machine-readable evidence). Create 04_MODERNIZATION_DECISIONS.md and 05_EXECUTION_ROADMAP.md when actual decisions/slices exist. Use module-audits/, experiments/ and evidence/ as needed. Link existing evidence rather than duplicate authority or create placeholders.

CURRENT_STATUS.md is the S4 execution pointer, not domain authority. Record source/evidence commits, gate/slice state, blockers, next action and protected boundaries at meaningful milestones. New sessions read CLAUDE.md, status, master, protocol, then active slice authority. Blind reviews follow isolated bundle rules, never S4 context.

Next authorized action: S4-00 baseline initialization and safe verification, not implementation of untested modernization proposals.
