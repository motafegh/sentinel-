# R0.6 — R0 wave closure

Candidate `c39de51ac` is the integrated R0 containment release. All 8 global
rows are closed with measured before/after evidence. The R0 wave is complete.

## Closure summary

| Row ID | Package | Invariant | Status |
|---|---|---|---|
| R0-EVIDENCE-OUTAGE | R0.1 | ML/chain outage never becomes successful evidence | PASS |
| R0-REPORT-CONTAINMENT | R0.2 | Report path cannot escape workspace | PASS |
| R0-ARCHIVE-CONTAINMENT | R0.2 | Archive extraction cannot escape workspace | PASS |
| R0-DATA-RELEASE-TRUST | R0.5 | Dataset release commitment binds semantics and exact files | PASS |
| R0-AUTHORIZATION-LIMITS | R0.3 | Public mutation routes require auth/scope/quota | PASS |
| R0-SIGNER-ISOLATION | R0.3 | Analysis process has no raw signing key | PASS |
| R0-PROOF-IDENTITY | R0.4 | Proof cannot support cross-identity verified claim | PASS |
| R0-TRANSACTION-TRUTH | R0.4 | Failed/reverted transaction cannot be reported submitted | PASS |

## Integrated measurement

- **8/8 frozen probes** pass on integrated commit `c39de51ac`
- **716 AGENTS tests** pass, 9 pre-existing environmental failures (6 RAG seed corpora + 3 solc)
- **91 DATA tests** pass (export + archive safety + ingestion)
- **Zero new regressions** introduced across all packages
- **Worktree clean**, `git diff --check` clean

## What R0 changed (code summary)

| Module | Change |
|---|---|
| `agents/src/contracts/execution.py` | New — canonical execution status with digest binding |
| `agents/src/persistence/` | New package — job-scoped paths, atomic writes, structured status |
| `agents/src/security/auth.py` | New — bearer token auth dependency |
| `data_module/.../archive_safety.py` | New — safe ZIP extraction with containment |
| `agents/src/config/runtime.py` | New — typed runtime profiles (test/dev/prod) |
| `scripts/r0_evidence/` | New — frozen evidence probes + matrix validator |
| `gateway.py` | Auth on POST /audit, loopback default, job_id in state |
| `synthesizer.py` | Fail-closed ML evidence, structured persistence status |
| `audit/_config.py` | Operator key removed |
| `audit/_submit.py` | Submission disabled, proof identity requirements documented |
| `audit/_handlers.py` | submit_audit unadvertised |
| `chunker.py` + `export.py` | Bidirectional file-set check, structured verification |
| `promote_model.py` | F1 gate terminal on MLflow failure |
| 5 MCP servers | Loopback default |

## What R0 did NOT change

- ML model architecture, training, or inference logic
- Agent pipeline, routing, or verdict fusion
- ZK proof generation or circuit
- Smart contracts
- DATA export format or schema
- Any runtime behavior for valid inputs

## What remains for R1–R4

- R1: Complete deterministic release types, canonical evidence schema
- R2: Durable job control, nonce allocator
- R3: V3 typed identity, proof replay prevention, quorum, governance, on-chain finality
- R4: Scientific and operational acceptance

## Authorization boundary

R0 wave closure authorizes merging `codex/r0-containment` into `main` upon Ali's
explicit approval. Deployment, live-chain writes, key movement, contract
administration, artifact deletion, and model promotion remain separately
unauthorized.
