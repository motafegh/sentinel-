# SENTINEL D2 acceptance matrix

**Baseline:** `4b5bd333c`
**Matrix state:** prepared for review
**Meaning of `complete`:** evidence exists for the audit deliverable; it does not mean the runtime defect is remediated

## D2 audit-package acceptance

| Requirement | State | Authoritative evidence | Remaining action |
|---|---|---|---|
| Runtime baseline locked | Complete | Git baseline recorded in every canonical artifact | None |
| Audit-only mutation boundary | Complete | `git diff --check`; executable/configuration diff from `4b5bd333c` is empty | Preserve through commit |
| DATA source audit | Complete | DATA appendix, 14 IDs | None |
| ML source audit | Complete | ML appendix, 22 IDs and exact suite classification | None |
| ZKML/contracts source audit | Complete | ZKML/contracts appendix, 18 IDs | None |
| AGENTS/services source audit | Complete | AGENTS appendix, 21 IDs | None |
| Cross-system architecture/threat audit | Complete | Cross-system appendix, 11 IDs and 429-file disposition | None |
| Stable unified registry | Complete | 86 raw rows normalized to 84 unique; two tombstoned duplicates | Ali confirms severities/merges |
| P0 independent verification | Complete | Six unique P0s in verification ledger | Runtime remediation belongs to R0 |
| P1 adjudication | Complete for audit | Every accepted P1 reproduced, source-confirmed, or explicitly measurement-blocked | Ali confirms acceptance dispositions |
| Baseline test evidence | Complete | Exact DATA/ML/AGENTS/ZKML/Foundry counts and failure classification | Rerun against remediated release later |
| Fresh-clone evidence | Complete as blocker | Missing DATA/checkpoint/compiler/proving/Foundry/knowledge chain explicitly recorded | Implement R1 bootstrap/release |
| Scientific evidence | Complete as blocker | Leakage/calibration/proxy/fusion limitations recorded | Measure after R1 |
| Performance evidence | Complete as blocker | Available point observations separated from missing acceptance measurements | Measure in R4 |
| Current executable architecture | Complete | Current architecture artifact | None |
| Decision-complete V3 architecture | Complete | Target architecture fixes types, states, quorum, trust, governance, migration | Ali architecture decision |
| Master audit report | Complete | Answer-first master report | Ali review |
| Remediation roadmap | Complete | R0–R4 packages, gates, dependencies, rollback | Ali ordering decision |
| Runtime acceptance matrix | Complete | This matrix, sections below | Apply during implementation |
| Review record | Prepared | Review artifact with explicit decisions | Ali must approve/request changes/reject |
| Internal package validation | Complete | 86/86 IDs matched; 84 unique after two merges; 13 links resolve; no placeholders/unbalanced fences; `git diff --check` passed | Preserve through commit |
| Documentation commit | Pending | Git commit on audit branch | Commit after validation |
| D2 final status | `REVIEW_REQUIRED` | Review is a human governance gate | Ali decision |

## Runtime remediation acceptance

### R0 containment

| Invariant | Current | Closure evidence required | IDs |
|---|---|---|---|
| ML/chain outage never becomes successful evidence | Fail | Fault injection produces explicit degraded/failed state through report/API/finality | D2-AGT-001, D2-AGT-012, D2-AGT-016 |
| Report path cannot escape workspace | Fail | Traversal/symlink/absolute/Unicode tests with no outside write | D2-AGT-002 |
| Archive extraction cannot escape workspace | Fail | ZIP traversal/symlink/limit tests | D2-DATA-001 |
| Dataset release commitment binds semantics and exact files | Fail | Mutation/add/delete/cache tests fail closed | D2-DATA-002 |
| Public mutation and expensive routes require auth/scope/quota | Fail | 401/403, scope, tenant, body/rate/concurrency tests | D2-X-001, D2-AGT-011 |
| Analysis process has no raw signing key | Fail | Secret scan/process boundary plus policy-signer rejection tests | D2-X-001, D2-ZKC-014 |
| Proof cannot support cross-identity verified claim | Fail | Cross-target/model/chain/round replay tests | D2-ZKC-001 |
| Failed/reverted transaction cannot be reported submitted | Fail | Gas estimation and receipt-state tests | D2-ZKC-003 |

### R1 deterministic release and commitment

| Invariant | Current | Closure evidence required | Primary IDs |
|---|---|---|---|
| One immutable DATA release is reproducible | Fail | Clean bootstrap, complete signed descriptor, byte-identical rebuild | D2-DATA-003–013, D2-X-007 |
| Graph/token/label/ID rows are atomic | Fail | Missing-middle/asymmetry/shard-boundary tests | D2-DATA-003 |
| Stored Solidity preserves defined semantics | Fail | Lexer/import/source-bundle goldens and compile-exact-bytes | D2-DATA-005 |
| Split/leakage/verification gates fail closed | Fail | Group-disjoint split, independent leakage audit, empty/skipped/error cases | D2-DATA-008, D2-DATA-009 |
| Train and serve tensors are equivalent | Fail | Golden tensor tests across comments/long source/padding; exhaustive node IDs | D2-ML-001–003 |
| Model is authenticated before safe load | Fail | Wrong/missing/tampered checkpoint rejected pre-deserialization | D2-ML-004, D2-ML-005 |
| Teacher bundle binds DATA/preprocessing/schema/tools/calibration | Fail | Complete immutable bundle and promotion test | D2-ML-006, D2-ML-013–016 |
| API represents no-contract/error/degraded states truthfully | Fail | Typed response-contract tests | D2-ML-008, D2-ML-009 |
| Evidence correlation and derivation are explicit | Fail | Derived consensus not independent; production RAG seam tests | D2-AGT-003, D2-AGT-009 |
| Canonical result retains evidence/status/dual truth boundary | Fail | State→report→API→CAS byte-identical round trip | D2-AGT-004, D2-AGT-010, D2-AGT-016 |
| One system manifest binds all release artifacts | Fail | Artifact DAG validation and atomic promotion | D2-X-005, D2-X-008 |
| Python/Solidity hashes and layouts agree | Fail | Golden typed-digest/signal/Merkle vectors | D2-X-008, D2-ZKC-008, D2-ZKC-009 |

### R2 durable operator plane

| Invariant | Current | Closure evidence required | IDs |
|---|---|---|---|
| Accepted work has durable lease/heartbeat/retry ownership | Fail | Kill/restart/lease/two-worker/retry/dead-letter tests | D2-AGT-005, D2-X-004 |
| Admission and inference are bounded/cancellable | Fail | Saturation, timeout, cancellation acknowledgement, fairness tests | D2-AGT-017, D2-ML-009 |
| Parallel branch errors merge structurally | Fail | Multi-failure reducer tests preserve all errors | D2-AGT-015 |
| Proof workspaces and nonces are isolated | Fail | Concurrent proof/transaction tests | D2-ZKC-004 |
| Reports and RAG generations publish atomically | Fail | Concurrent round and partial-generation tests | D2-AGT-019 |
| Chain feedback is versioned, reorg-safe, exactly-once | Fail | V1/V2/V3 decode, reorg, duplicate, retry tests | D2-X-006, D2-AGT-020 |
| One run is observable end to end | Fail | Trace/metric/error/retry/proof/tx/event reconciliation | D2-X-010, D2-ML-020, D2-AGT-007 |

### R3 V3 protocol

| Invariant | Current | Closure evidence required | IDs |
|---|---|---|---|
| Round identity is immutable and replay-safe | Missing | EIP-712 cross-domain/identity/nonce/deadline vectors | D2-ZKC-001, D2-X-005 |
| Proof envelope binds circuit/verifier/proof/signals/scores | Missing | Real-verifier mutation/layout/range tests | D2-ZKC-002, D2-ZKC-007–009 |
| Active-set snapshot is immutable per round | Missing | Join/exit/post-snapshot tests | D2-ZKC-010, D2-ZKC-011 |
| Quorum is correct for N=5…9 | Missing | Threshold/bitmap/duplicate/inactive tests | D2-ZKC-011 |
| Equivocation/slashing is objective | Missing | Conflicting EIP-712 attestation and stake lifecycle tests | D2-ZKC-010, D2-ZKC-011 |
| Governance cannot immediately rewrite trust | Fail | Timelock/multisig/guardian/storage-layout tests | D2-ZKC-012, D2-ZKC-015 |
| V1/V2 history remains readable and unambiguous | Partial | Compatibility/event migration and shadow-cutover tests | D2-ZKC-009, D2-X-006 |
| LLM/RAG cannot affect finality | Not enforced end to end | Mutation/isolation tests over advisory root | D2-AGT-010, D2-X-002 |

### R4 measured production decision

| Evidence gate | Baseline state | Required measurement |
|---|---|---|
| DATA release quality | Blocked | Leakage, label coverage/quality, benchmark execution, rebuild equality |
| Teacher quality | Not acceptance-grade | Held-out per-class precision/recall/Fβ, calibration, unsupported-class policy, drift |
| Teacher/proxy agreement | Incomplete | Per-class/contract agreement and error bounds under final manifest |
| Operator determinism | Missing | Byte-identical commitments across independent clean operators |
| DATA performance | Blocked | Throughput, memory, scaling on pinned release |
| Teacher serving | Blocked | CPU/GPU latency, throughput, VRAM, concurrency, timeout/cancel |
| AGENTS deterministic/advisory cost | Blocked | Separate stage/tool/LLM latency and cost distributions |
| Recovery/liveness | Missing | Crash, lease, queue saturation, RPC/reorg, operator loss |
| Proof/quorum gas and storage | Incomplete | Worst-case real verifier plus N=9 attestations and growth model |
| Security | Failed baseline | Auth/tenant/quota/signer/supply-chain/path/replay/governance adversarial suite |
| Permissionless admission | Out of pilot scope | Separate economic/Sybil/concentration/churn/governance evidence |

## Decision rule

Production readiness requires every R0–R3 invariant to be `PASS` and every applicable R4 evidence gate to meet a measured, versioned acceptance policy. `BLOCKED`, `SKIPPED`, `UNAVAILABLE`, historical-only, or local-untracked evidence cannot satisfy a gate.
