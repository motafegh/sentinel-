# SENTINEL D2 verification and performance evidence

**Audit date:** 2026-07-14
**Baseline:** `4b5bd333c63ab7a7ec83810fbbae54f3ebf1b493`
**Branch:** `codex/source-engineering-audit`
**Mode:** documentation-only independent audit
**Status:** `COMPLETE_FOR_REVIEW`

This ledger is updated after each primary verification cluster. Raw command output is retained in audit-host temporary files when it may contain environment-specific paths or sensitive configuration. Secret values are never copied into this package.

## Verification rules

- Executable source is authoritative; existing prose is corroborating evidence only.
- Runtime claims use isolated temporary directories, processes, or local Anvil where feasible.
- Architecture/configuration claims may be reproduced by an independent source trace when executing the unsafe behavior would add no evidence.
- Missing services, artifacts, hardware, or dependencies are recorded as unavailable or blocked, never converted into passes.
- No probe may alter canonical datasets, models, proofs, reports, contracts, or runtime configuration.

## P0 primary verification

| Finding | Result | Method | Evidence |
|---|---|---|---|
| D2-AGT-001 | primary-reproduced | Isolated HTTP failure injection | ML transport failure returned successful-looking mock evidence and downstream `ran=true`. |
| D2-AGT-002 | primary-reproduced | Temporary report root | An unvalidated request identifier wrote outside the allocated report directory. |
| D2-DATA-001 | primary-reproduced | Temporary ZIP workspace | Sibling-prefix traversal escaped the extraction destination. |
| D2-DATA-002 | primary-reproduced | Temporary minimal export | Verification remained true after cached-shard removal and uncommitted manifest changes. |
| D2-ZKC-001 | primary-reproduced | Isolated local Anvil with tracked verifier/proof | The same proof/signals were accepted for two targets and two model hashes. |
| D2-X-001 | primary-reproduced | Independent route/auth/signing source trace | Seven wildcard bind sites were found; no route-layer authentication control was present; the audit MCP exposes submission code backed by an operator signing key. |
| D2-X-009 | merged-duplicate | Registry normalization | Exact cross-system restatement of D2-AGT-001. |

### D2-X-001 command record

The primary audit searched the executable gateway/MCP route wiring for authentication dependencies and separately traced audit submission/signing authority. Raw redacted outputs are `/tmp/d2_x001_routes_auth.txt` and `/tmp/d2_x001_signer_trace.txt` on the audit host. No network request or transaction was sent.

Result: **confirmed P0 architecture defect**. The finding does not claim that every deployment is internet-accessible; it establishes that application code supplies no authentication boundary while default server wiring binds broadly and a key-capable write operation exists.

## Baseline and performance ledger

### Runtime-delta guard

The locked runtime baseline is `4b5bd333c`. The following executable/configuration scope has no delta from that baseline:

```text
data_module/sentinel_data
ml/src
ml/scripts
zkml/src
agents/src
contracts/src
contracts/script
contracts/test
contracts/foundry.toml
```

Audit-branch changes are confined to the dated package under `docs/audit/system-engineering/`.

### Suite evidence

| Surface | Environment | Result | D2 interpretation |
|---|---|---|---|
| D1 handbook validation | Clean D2 worktree | 285/285 static checks; 10/10 validator tests; 7/7 safe lab preflights | Confirms documentation inventory tooling, not runtime readiness. |
| DATA | Root venv with D2 source | 465 passed, 13 failed, 111 skipped; 589 collected; 10.24 s | Fails acceptance: four ICFG regressions plus ignored/local corpus prerequisites. |
| ML | ML venv with D2 DATA package | 159 passed, 20 failed, 16 skipped, 22 collection/setup errors; 217 collected; 26.67 s | Fails acceptance: active checkpoint and compiler/export prerequisites are absent; remaining failures include seam/API defects and stale tests. |
| AGENTS/services | AGENTS venv with D2 source and `/tmp` temp roots | 622 passed, 11 failed, 1 skipped; 634 collected; 26.16 s | Fails acceptance: six fresh-clone data failures, three toolchain-resolution failures, and two config/isolation failures. |
| AGENTS gateway/store focus | Same AGENTS environment | 57 passed; 1.51 s | Local CRUD behavior is covered; distributed durability/backpressure remains untested and absent in source. |
| ZKML | ZKML environment | 34 passed, 3 skipped | Skips require ignored teacher/proving/GPU prerequisites; not a complete release gate. |
| Foundry tracked tests | Local installed libraries supplied | 52 passed | Not fresh-clone evidence because `contracts/lib` is absent from the repository/bootstrap path. |
| Ignored local V2 Foundry test | Local-only test | 14 passed | Useful corroboration only; the test is not present in a fresh clone. |

No failed, errored, skipped, ignored, or unavailable item is counted as a pass.

### Primary P1 adjudication

The unified registry contains the per-ID disposition. Primary adjudication used three methods:

1. **Isolated deterministic probes** for graph/job reducer behavior, launch-directory configuration, schema validation, temporary-file cleanup, artifact gating, and chain submission.
2. **Complete executable-interface traces** for static absence/architecture claims such as no authentication dependency, no quorum state machine, no execution-manifest binding, duplicated configuration namespaces, and disconnected gateway/submission flows.
3. **Prerequisite classification** where a clean container, GPU/checkpoint, multi-worker load, live RPC, or worst-case real-verifier gas run was not available.

| Cluster | Primary result | Accepted IDs/evidence boundary |
|---|---|---|
| DATA alignment/orchestration | Confirmed | Independent graph/token writers skip different rows; chunker publishes the graph index to both modalities; label is an explicit `NOT IMPLEMENTED`; composite `run` omits required stage arguments. `D2-DATA-003`, `004`. |
| DATA source/provenance | Confirmed | Regex comment removal mutates Solidity strings; parallel workers instantiate independent deduplicators; mutable manifests/catalog replacement and missing immutable snapshot checks are present. `D2-DATA-005`–`007`, `011`. |
| DATA science/release gates | Confirmed | Missing dedup groups warn and continue; leakage audit is not a publication gate; empty evidence yields provisional pass; export does not consume verification; benchmark evaluator is explicitly unimplemented. `D2-DATA-008`–`010`, `012`, `013`. |
| ML tensor semantics | Confirmed | Exhaustive node-ID round trip shifts IDs 7–12; offline/online comment and padding policies differ; synthetic windows can reach pooling. `D2-ML-001`–`003`. |
| ML artifact/deployment | Confirmed; clean build blocked | Unsafe unrestricted pickle loads occur before hash identity; active checkpoint is ignored/untracked; image/artifact/compiler acquisition is incomplete. `D2-ML-004`–`007`, `012`, `015`. |
| ML API/concurrency | Confirmed; production load blocked | `windows_used=0` cannot validate; request timeout wraps a worker thread without cancellation; no admission semaphore exists; hotspot score is embedding magnitude; startup purge is process-unscoped. `D2-ML-008`–`011`. |
| ML lifecycle | Confirmed | Promotion/reproducibility evidence can pass without the claimed immutable bundle or repeated inference; production serving does not apply recorded Run12 temperatures. `D2-ML-013`, `014`, `016`–`022`. |
| ZK/proof semantics | Confirmed | Public signals constrain proxy input/output only; fixed parameters, copied class/score semantics, and divergent hashes do not bind teacher/source/identity. `D2-ZKC-002`, `005`–`009`. |
| Chain/protocol | Confirmed | Fixed gas and receipt handling are unsafe; proof files/nonces are shared; V1/V2 verifier paths differ; stake is per submission; no round/quorum/finality; owner controls are immediate. `D2-ZKC-003`, `004`, `009`–`015`. |
| AGENTS/services | Confirmed | The primary 12-item source/probe ledger passed after resolving the configured reliability path from both launch directories. It covers jobs, `no_llm`, service URL, fusion correlation, provable semantics, chain mock fallback, shared proof/nonce, reliability, reducers, tool status, capacity, and feedback truth language. `D2-AGT-005`, `006`, `008`–`017`, `020`. |
| Cross-system | Confirmed | Gateway completion and chain submission are separate entry points; config namespaces conflict; no distributed lease/manifest; V2 feedback is unsupported; artifacts/schema promotion are not atomic. `D2-X-002`–`008`. |

Raw primary AGENTS and authentication traces are retained on the audit host at `/tmp/d2_agt_p1_primary.txt`, `/tmp/d2_x001_routes_auth.txt`, and `/tmp/d2_x001_signer_trace.txt`. They contain environment paths and are deliberately not promoted as portable evidence artifacts.

### Fresh-clone evidence chain

| Required object | Tracked/bootstrap state | Decision |
|---|---|---|
| DATA release, splits, corpora | Not tracked; DVC configuration points to a private absolute remote and no release lock exists | Blocked |
| Run12 teacher checkpoint/threshold bundle | Configured path exists only in the developer workspace; checkpoint is ignored and absent from clean worktree | Blocked |
| Hugging Face snapshot and compiler inventory | Not bound by an immutable system manifest; supported solc versions are not provisioned by clean serving | Blocked |
| Proxy/ONNX/circuit/proving bundle | Some EZKL artifacts are tracked; proving key/SRS and teacher prerequisites are absent | Blocked |
| Foundry libraries and V2 test | Local libraries/test can pass but are not registered for clean bootstrap | Blocked |
| AGENTS knowledge seeds/toolchain | Required ignored corpora and resolvable `solc` are absent from the clean test process | Blocked |

Therefore a fresh clone cannot reproduce DATA → teacher → deterministic AGENTS result → proxy proof → contract record. `D2-X-007` remains open and is a release gate.

### Scientific evidence

The baseline cannot support a production accuracy or oracle-truth claim:

- DATA split/leakage/verification are not cryptographically bound to the export.
- Run12 preprocessing and GNN type semantics are inconsistent across training/serving.
- Recorded temperature calibration is not applied by production serving.
- Proxy outputs and public-signal semantics are internally inconsistent and prove only the proxy computation.
- AGENTS fusion double-counts derived consensus and its L3 reliability path depends on launch context.
- LLM/RAG narrative is non-deterministic advisory material and must remain outside consensus.

Historical metrics remain useful experiment records, but none is promoted to D2 acceptance evidence for the current executable bundle.

### Performance evidence and blockers

| Measurement | Available evidence | Acceptance state |
|---|---|---|
| DATA throughput/memory | No portable corpus/release in clean worktree | Blocked |
| Teacher latency, throughput, VRAM, concurrency | No clean authenticated checkpoint/compiler/HF bundle; no controlled GPU run | Blocked |
| AGENTS end-to-end latency/cost | Live ML, MCP, LM Studio, and tool services unavailable during cross-system probe | Blocked |
| Job crash/recovery under multiple workers | Architecture lacks durable leases; no representative worker deployment | Blocked |
| Warm local proof time | One historical local run: witness 0.503 s, prove 2.143 s, verify 0.023 s | Informational only |
| Real-verifier submission gas | Primary replay probe: 1,002,486 and 1,022,770 gas for two local submissions | Valid point observations, not worst-case capacity evidence |
| V3 quorum gas/storage for 5–9 operators | V3 contracts do not yet exist | Required before deployment |

`D2-ML-021` and `D2-ZKC-017` remain evidence requirements. No performance target or policy number is changed by D2.

## Evidence conclusion

The audit evidence is complete enough to decide architecture and remediation ordering. It is intentionally **not** production acceptance evidence. The unremediated baseline fails trust, reproducibility, scientific, fresh-clone, and operational gates. Live/hardware measurements must be captured against an immutable remediated release candidate, not retroactively inferred for this baseline.
