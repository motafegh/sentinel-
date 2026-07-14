# SENTINEL D2 source-engineering master audit report

**Baseline:** `4b5bd333c63ab7a7ec83810fbbae54f3ebf1b493`
**Audit branch:** `codex/source-engineering-audit`
**Audit mode:** documentation and isolated verification only
**Decision:** `APPROVED_FOR_R0_PLANNING`
**Runtime remediation:** not performed

## Executive answer

SENTINEL is not production-ready and is not yet a decentralized oracle. It is a substantial single-host research system with useful DATA, ML, orchestration, ZKML, and contract components, but the trust guarantees do not compose across their interfaces.

The baseline has **84 unique audit findings/requirements** after two exact duplicate merges: **6 P0, 62 P1, 15 P2, and 1 P3 evidence gap**. All six unique P0s are independently primary-confirmed. No accepted P1 remains an unclassified candidate; each is reproduced/source-confirmed or carries an explicit prerequisite measurement blocker.

The immediate risks are evidence fabrication during ML outage, filesystem escape through public inputs and archives, dataset release verification that does not bind the release, proof replay across arbitrary target/model identities, and a broadly bound unauthenticated control plane that includes key-capable submission functionality.

The correct next move is compatibility-preserving stabilization, not adding quorum around current labels. V3 should finalize a canonical deterministic commitment produced under an immutable execution manifest. LLM/RAG narrative stays advisory. A governed, staked pilot of 5–9 operators uses an immutable active-set snapshot and `ceil(2N/3)` EIP-712 attestations.

## What was audited

The review covered 429 Python/Solidity/shell files across DATA, ML, AGENTS/services, ZKML, and contracts, while distinguishing deployed runtime, build/quality tooling, research, fixtures, and archive material. Executable source and tracked artifacts were authoritative. Existing Markdown was treated only as historical evidence.

Five detailed appendices were normalized into one registry:

| Track | Raw findings | Primary conclusion |
|---|---:|---|
| DATA | 14 | Release integrity, alignment, preprocessing, split, verification, and fresh-clone gates are not trustworthy enough for model promotion. |
| ML | 22 | Core model/data serving semantics and artifact identity are inconsistent; clean deployment and acceptance-quality measurement are unavailable. |
| ZKML/contracts | 18 | Proofs validate a narrow proxy computation but are replayable across audit identities; no quorum/finality protocol exists. |
| AGENTS/services | 21 | The graph is feature-rich, but outage fallback, schema loss, persistence, capacity, and service boundaries prevent trustworthy operator execution. |
| Cross-system | 11 | Gateway reports and chain submissions are separate products with no canonical end-to-end identity or manifest. |

Two exact duplicates are tombstoned in the registry: `D2-X-009 → D2-AGT-001` and `D2-AGT-013 → D2-ZKC-004`.

## Six P0 blockers

| ID | Confirmed defect | Production consequence |
|---|---|---|
| D2-AGT-001 | ML transport outage returns plausible mock prediction and downstream `ran=true` | Fabricated evidence can enter reports/evaluation/finality paths. |
| D2-AGT-002 | User-controlled report identifier escapes the report directory | Public request can overwrite JSON reachable to the service account. |
| D2-DATA-001 | ZIP containment uses an unsafe prefix test | Archive ingestion can write outside its workspace. |
| D2-DATA-002 | Export hash excludes semantic manifest and warm cache ignores missing expected files | Altered/incomplete dataset can still verify successfully. |
| D2-ZKC-001 | Proof/public signals do not bind target, runtime code, model, manifest, or round | One valid proof can create verified histories for unrelated identities. |
| D2-X-001 | Broadly bound services have no application auth boundary and expose key-capable submission code | Reachable deployment permits compute/gas/stake abuse and compounds downstream defects. |

These are containment gates. Public or production operation should not proceed while any remains open.

## Systemic P1 conclusions

### DATA and scientific provenance

Graph and token writers can publish misaligned rows; label orchestration is unimplemented; regex normalization mutates valid Solidity; worker count changes deduplication; leakage controls warn and continue; empty verification can pass; registry versions are replaceable; final exports are not registered as immutable releases. The active dataset may be useful experimentally, but the executable pipeline cannot reproduce or prove its identity from a clean clone.

### ML validity and artifact security

The GNN decodes normalized categorical node types with the wrong divisor. Offline training strips comments while online serving does not; padded-window IDs/masks differ and synthetic windows can affect pooling. Checkpoints use unrestricted pickle loading and are identified only after load. The active checkpoint is absent from clean artifact control. Timeout does not cancel worker inference; admission is unbounded. Hotspot “attention” is embedding magnitude, and recorded Run12 temperatures are not applied by serving.

### AGENTS execution integrity

Production RAG metadata is inert at the fusion seam. Derived consensus is counted beside its own parents. `provable` means an emitter asserted determinism, not that an execution manifest/proof binds it. The gateway strips evidence, tool statuses, and dual verdicts; durable jobs are disconnected from graph recovery; request-level `no_llm` is ineffective; service URLs and reliability weights depend on inconsistent launch configuration; shared state reducers and capacity controls are insufficient.

### Proof, contracts, and protocol

EZKL proves the fixed proxy over supplied fusion values—not the Solidity source, deployed code, teacher execution, preprocessing, AGENTS evidence, or vulnerability truth. Proxy score/logit semantics and model/proof hashes diverge. Proof files and nonces are shared. Fixed gas/receipt handling is unsafe. V1/V2 compatibility and feedback are incomplete. Stake secures individual submissions only; there is no round, quorum, immutable active set, finality, objective equivocation path, timelock, or safe verifier lifecycle.

### Cross-system composition

The gateway never invokes the submission flow. A completed report is advisory, while chain submission is a separate MCP call with independently supplied values. Configuration namespaces conflict. No object binds target/source, DATA, teacher, deterministic tools/fusion, proxy/circuit/verifier, proof, report, operator set, and chain state. V2 events are not consumed by the feedback loop. A clean clone cannot reproduce the evidence chain.

## Evidence quality

| Suite | Result | Acceptance |
|---|---|---|
| DATA | 465 passed, 13 failed, 111 skipped | Failed |
| ML | 159 passed, 20 failed, 16 skipped, 22 setup/collection errors | Failed |
| AGENTS | 622 passed, 11 failed, 1 skipped | Failed |
| ZKML | 34 passed, 3 skipped | Partial only |
| Foundry tracked tests | 52 passed with separately installed libraries | Not fresh-clone evidence |

One warm proof and two real-verifier local transaction observations exist, but DATA throughput, teacher GPU/VRAM/concurrency, full AGENTS latency/cost, multi-worker recovery, and V3 quorum gas/storage are blocked or not yet applicable. D2 does not change decision thresholds or claim performance from missing measurements.

## Current versus target architecture

| Concern | Current baseline | V3 decision |
|---|---|---|
| Audit identity | Free-form address/job plus unrelated hashes | Typed chain/target/runtime-code/block/source/round identity |
| Release identity | Independent mutable files and configs | Immutable signed system manifest and complete artifact DAG |
| Deterministic result | In-memory/report verdicts with incomplete status | Canonical score/verdict/evidence/status roots |
| Proof | Proxy proof detached from identity | Versioned proof envelope bound into commitment |
| Narrative | Can participate in AGENTS decisions | Separate advisory root; cannot change finality |
| Jobs | Process-local tasks plus SQLite rows | Durable lease/heartbeat/retry/idempotent stage model |
| Services | Broad bind, no app auth | Private service identity, scoped API authorization, quotas |
| Signing | Key in audit MCP process | Policy-isolated HSM/KMS signer for typed commitments only |
| Operators | Individual staked submissions | 5–9 governed/staked active-set snapshot |
| Finality | None | `ceil(2N/3)` unique EIP-712 attestations |
| Governance | Immediate owner controls | Multisig + timelock; pause-only guardian; versioned verifiers |
| Migration | V1/V2 partial | Preserve reads, shadow V3, governed cutover, no reinterpretation |

## Remediation decision

The remediation roadmap has five ordered waves:

1. **R0 — containment:** remove fabricated/mock production evidence, validate/contain filesystem writes, authenticate service boundaries, isolate signer, fail receipts/gas correctly, and stop unbound proof claims/writes.
2. **R1 — deterministic release:** immutable DATA and model/proof bundles; canonical schemas; train/serve alignment; explicit evidence/status; system manifest and cross-language vectors.
3. **R2 — durable operators:** leases, retries, idempotency, isolated workspaces/nonces, bounded admission, atomic reports/indexes, typed errors, traces and metrics.
4. **R3 — V3 protocol:** operator vault/set, verifier registry, typed commitment, quorum coordinator, governance, V1/V2 migration, shadow operation.
5. **R4 — measured acceptance:** accuracy/calibration/leakage, independent-operator equality, latency/VRAM/throughput, crash recovery, gas/storage, adversarial security, and permissionless-admission evidence.

No behavioral redesign should skip R0/R1 and jump directly to contracts. Quorum over inconsistent executions is not integrity.

## Production-readiness decision

| Claim | Decision |
|---|---|
| Safe public gateway | **No** |
| Reproducible DATA/ML release | **No** |
| Scientifically acceptance-grade classifier | **No** |
| End-to-end deterministic audit commitment | **No** |
| Proof bound to target/model/audit | **No** |
| Decentralized quorum/finality | **No** |
| Architecture ready for phased implementation planning | **Yes, approved for R0 planning** |

## Review decision

Ali approved the following D2 decisions on 2026-07-14:

1. the 84-item normalized registry and severity assignments;
2. the six P0 containment blockers;
3. the V3 identity/manifest/commitment and proof truth boundary;
4. the governed 5–9 operator pilot with `ceil(2N/3)` quorum;
5. the five-wave remediation order and no-runtime-change D2 closure.

The approval condition is mandatory: every R0–R4 wave closes only through the acceptance matrix and measured before/after evidence. Approval authorizes a new implementation plan/branch beginning with R0. It does not authorize deployment, unplanned runtime changes, policy-number changes, or a production-readiness claim.
