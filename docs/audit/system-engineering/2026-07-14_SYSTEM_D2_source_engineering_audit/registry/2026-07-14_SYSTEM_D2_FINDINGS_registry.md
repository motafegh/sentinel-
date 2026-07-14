# SENTINEL D2 unified findings registry

**Audit baseline:** `4b5bd333c63ab7a7ec83810fbbae54f3ebf1b493`
**Registry version:** 1
**Audit mode:** documentation-only
**Decision state:** `APPROVED_FOR_R0_PLANNING`

## Registry contract

This is the canonical index for D2 findings. The appendices own detailed evidence, impact, recommendations, migration notes, and tests; this registry owns stable identity, severity, adjudication, duplicate handling, remediation ownership, and closure wave.

Evidence states:

- `primary-reproduced`: the primary audit exercised the behavior in an isolated fixture or local chain.
- `primary-source-confirmed`: the primary audit independently traced the complete executable interface or absence that defines the finding. This is appropriate for static architecture, configuration, schema, and governance claims that do not gain evidence from executing an unsafe action.
- `track-reproduced`: a bounded audit track executed the behavior; the primary audit accepted the evidence after source reconciliation.
- `blocked-prerequisite`: the source defect/gap is accepted, but a required hardware, artifact, live-service, or external-chain measurement could not be obtained. The missing measurement remains an acceptance blocker.
- `merged-duplicate`: no separate remediation item; the row remains for traceability and points to its canonical ID.
- `evidence-gap`: a measurement requirement, not a proven runtime defect.

Closure waves:

- `R0`: immediate containment and evidence-integrity blockers.
- `R1`: deterministic data/model/evidence contracts and reproducible release bundles.
- `R2`: durable operator execution, authenticated services, observability, and recovery.
- `R3`: V3 quorum/finality protocol, governance, and migration.
- `R4`: calibrated quality, performance, scaling, and maintainability.

## Normalization result

The five appendices contain 86 rows: 7 P0, 63 P1, 15 P2, and 1 P3. Two are exact duplicates:

- `D2-X-009` merges into `D2-AGT-001`.
- `D2-AGT-013` merges into `D2-ZKC-004`.

The registry therefore contains **84 unique findings/requirements**: 6 P0, 62 P1, 15 P2, and 1 P3. Broader cross-system findings that aggregate several module defects remain separate because they define an additional system invariant and acceptance gate; their related IDs are recorded instead of being silently discarded.

## DATA

| ID | Sev | Disposition | Evidence | Owner | Related/canonical | Wave |
|---|---|---|---|---|---|---|
| D2-DATA-001 | P0 | accepted | primary-reproduced | DATA ingestion | D2-X-007 | R0 |
| D2-DATA-002 | P0 | accepted | primary-reproduced | DATA export | D2-X-007, D2-X-008 | R0 |
| D2-DATA-003 | P1 | accepted | track-reproduced + primary-source-confirmed | DATA export / ML dataset | D2-ML-015, D2-X-008 | R1 |
| D2-DATA-004 | P1 | accepted | track-reproduced + primary-source-confirmed | DATA orchestration | D2-DATA-009 | R1 |
| D2-DATA-005 | P1 | accepted | track-reproduced + primary-source-confirmed | DATA preprocessing | D2-ML-002, D2-ML-015 | R1 |
| D2-DATA-006 | P1 | accepted | track-reproduced + primary-source-confirmed | DATA ingestion/provenance | D2-DATA-011 | R1 |
| D2-DATA-007 | P1 | accepted | track-reproduced + primary-source-confirmed | DATA preprocessing | D2-DATA-008 | R1 |
| D2-DATA-008 | P1 | accepted | track-reproduced + primary-source-confirmed | DATA splitting/science | D2-X-007 | R1 |
| D2-DATA-009 | P1 | accepted | track-reproduced + primary-source-confirmed | DATA verification/export | D2-DATA-004, D2-X-008 | R1 |
| D2-DATA-010 | P1 | accepted | track-reproduced + primary-source-confirmed | DATA representation | D2-ML-001, D2-ML-015, D2-X-008 | R1 |
| D2-DATA-011 | P1 | accepted | track-reproduced + primary-source-confirmed | DATA registry | D2-DATA-002, D2-X-007 | R1 |
| D2-DATA-012 | P1 | accepted | primary-source-confirmed | DATA packaging | D2-X-007 | R1 |
| D2-DATA-013 | P1 | accepted | primary-source-confirmed | DATA evaluation | D2-ML-021 | R4 |
| D2-DATA-014 | P2 | accepted | primary-source-confirmed | DATA maintainers | — | R4 |

## ML

| ID | Sev | Disposition | Evidence | Owner | Related/canonical | Wave |
|---|---|---|---|---|---|---|
| D2-ML-001 | P1 | accepted | track-reproduced + primary-source-confirmed | ML modeling / DATA representation | D2-DATA-010 | R1 |
| D2-ML-002 | P1 | accepted | primary-source-confirmed | ML data/serving | D2-DATA-005 | R1 |
| D2-ML-003 | P1 | accepted | primary-source-confirmed | ML data/model/serving | D2-X-008 | R1 |
| D2-ML-004 | P1 | accepted | primary-source-confirmed | ML artifact security | D2-ML-005, D2-X-007 | R0 |
| D2-ML-005 | P1 | accepted | primary-source-confirmed | ML artifact security | D2-ML-004, D2-X-005 | R0 |
| D2-ML-006 | P1 | accepted | primary-source-confirmed | ML release engineering | D2-X-007 | R1 |
| D2-ML-007 | P1 | accepted | primary-source-confirmed; clean build blocked | ML deployment | D2-ML-012, D2-X-007 | R1 |
| D2-ML-008 | P1 | accepted | track-reproduced + primary-source-confirmed | ML API | — | R1 |
| D2-ML-009 | P1 | accepted | primary-source-confirmed; load measurement blocked | ML serving | D2-AGT-017 | R2 |
| D2-ML-010 | P1 | accepted | primary-source-confirmed | ML interpretability/API | D2-AGT-020 | R1 |
| D2-ML-011 | P1 | accepted | track-reproduced + primary-source-confirmed | ML serving | — | R2 |
| D2-ML-012 | P1 | accepted | primary-source-confirmed; container run blocked | ML deployment | D2-ML-007, D2-X-007 | R1 |
| D2-ML-013 | P1 | accepted | track-reproduced + primary-source-confirmed | ML release governance | D2-X-008 | R1 |
| D2-ML-014 | P1 | accepted | track-reproduced + primary-source-confirmed | ML reproducibility | D2-ML-015 | R1 |
| D2-ML-015 | P1 | accepted | primary-source-confirmed | ML/DATA/ZKML provenance | D2-DATA-003, D2-DATA-010, D2-X-008 | R1 |
| D2-ML-016 | P2 | accepted | primary-source-confirmed | ML calibration | D2-ML-021 | R4 |
| D2-ML-017 | P2 | accepted | primary-source-confirmed | ML monitoring | D2-X-010 | R2 |
| D2-ML-018 | P2 | accepted | primary-source-confirmed | ML serving/cache | — | R4 |
| D2-ML-019 | P2 | accepted | primary-source-confirmed | ML serving | — | R2 |
| D2-ML-020 | P2 | accepted | primary-source-confirmed | ML operations | D2-X-010 | R2 |
| D2-ML-021 | P3 | evidence-gap | blocked-prerequisite | ML performance | D2-DATA-013 | R4 |
| D2-ML-022 | P2 | accepted | primary-source-confirmed | ML API/testing | D2-X-008 | R1 |

## ZKML and contracts

| ID | Sev | Disposition | Evidence | Owner | Related/canonical | Wave |
|---|---|---|---|---|---|---|
| D2-ZKC-001 | P0 | accepted | primary-reproduced | Protocol / contracts / ZKML | D2-X-005 | R0 |
| D2-ZKC-002 | P1 | accepted | primary-source-confirmed | ZKML/protocol | D2-AGT-010, D2-AGT-020 | R1 |
| D2-ZKC-003 | P1 | accepted | track-reproduced + primary-source-confirmed | Chain submission | — | R0 |
| D2-ZKC-004 | P1 | accepted | track-reproduced + primary-source-confirmed | Proof workers / chain submission | D2-AGT-013, D2-X-004 | R2 |
| D2-ZKC-005 | P1 | accepted | track-reproduced + primary-source-confirmed | ZKML science | D2-ML-016 | R1 |
| D2-ZKC-006 | P1 | accepted | primary-source-confirmed | ZKML promotion/science | D2-X-008 | R1 |
| D2-ZKC-007 | P1 | accepted | primary-source-confirmed | ZKML artifact lifecycle | D2-ZKC-015, D2-X-008 | R1 |
| D2-ZKC-008 | P1 | accepted | primary-source-confirmed | ML/ZKML/AGENTS/contracts | D2-X-005, D2-X-008 | R1 |
| D2-ZKC-009 | P1 | accepted | primary-source-confirmed | Contracts/protocol | D2-X-008 | R3 |
| D2-ZKC-010 | P1 | accepted | primary-source-confirmed | Protocol economics | D2-ZKC-011 | R3 |
| D2-ZKC-011 | P1 | accepted | primary-source-confirmed | Protocol | D2-X-002 | R3 |
| D2-ZKC-012 | P1 | accepted | primary-source-confirmed | Protocol governance | — | R3 |
| D2-ZKC-013 | P1 | accepted | primary-source-confirmed | ZKML/contracts release engineering | D2-X-007 | R1 |
| D2-ZKC-014 | P1 | accepted | primary-source-confirmed | Secrets/signer operations | D2-X-001 | R0 |
| D2-ZKC-015 | P1 | accepted | primary-source-confirmed | Verifier/deployment governance | D2-ZKC-007, D2-X-008 | R3 |
| D2-ZKC-016 | P2 | accepted | primary-source-confirmed | ZKML/contracts testing | — | R1 |
| D2-ZKC-017 | P2 | accepted | blocked-prerequisite | Protocol performance | D2-ML-021 | R4 |
| D2-ZKC-018 | P2 | accepted | primary-source-confirmed | ZKML/contracts packaging | D2-X-003, D2-X-007 | R1 |

## AGENTS and services

| ID | Sev | Disposition | Evidence | Owner | Related/canonical | Wave |
|---|---|---|---|---|---|---|
| D2-AGT-001 | P0 | accepted | primary-reproduced | Inference MCP / orchestration | D2-X-009 merged here | R0 |
| D2-AGT-002 | P0 | accepted | primary-reproduced | Gateway/reporting | D2-X-001 | R0 |
| D2-AGT-003 | P1 | accepted | primary-reproduced | RAG/verdict fusion | — | R1 |
| D2-AGT-004 | P1 | accepted | primary-reproduced | Gateway/report schema | D2-AGT-016, D2-X-002 | R1 |
| D2-AGT-005 | P1 | accepted | track-reproduced + primary-source-confirmed | Gateway/job store | D2-X-004 | R2 |
| D2-AGT-006 | P1 | accepted | primary-source-confirmed | Gateway/LLM policy | — | R1 |
| D2-AGT-007 | P2 | accepted | primary-source-confirmed | Orchestration/reporting | D2-X-010 | R2 |
| D2-AGT-008 | P1 | accepted | primary-source-confirmed | Graph MCP/config | D2-X-003 | R1 |
| D2-AGT-009 | P1 | accepted | primary-source-confirmed | Evidence fusion/science | — | R1 |
| D2-AGT-010 | P1 | accepted | primary-source-confirmed | Evidence/ZK boundary | D2-ZKC-002, D2-X-005 | R1 |
| D2-AGT-011 | P1 | accepted | primary-source-confirmed | Service security | D2-X-001 | R0 |
| D2-AGT-012 | P1 | accepted | primary-source-confirmed | Audit MCP | D2-AGT-001 | R0 |
| D2-AGT-013 | P1 | merged-duplicate | merged-duplicate | — | canonical D2-ZKC-004 | — |
| D2-AGT-014 | P1 | accepted | track-reproduced + primary-source-confirmed | Verdict configuration | D2-X-003 | R1 |
| D2-AGT-015 | P1 | accepted | track-reproduced + primary-source-confirmed | Orchestration state | D2-X-004 | R2 |
| D2-AGT-016 | P1 | accepted | primary-source-confirmed | Orchestration/report schema | D2-AGT-004 | R1 |
| D2-AGT-017 | P1 | accepted | primary-source-confirmed; load measurement blocked | Gateway capacity | D2-X-004 | R2 |
| D2-AGT-018 | P1 | accepted | track-reproduced + primary-source-confirmed | AGENTS packaging/testing | D2-X-007 | R1 |
| D2-AGT-019 | P2 | accepted | primary-source-confirmed | Reporting/feedback | D2-X-006 | R2 |
| D2-AGT-020 | P1 | accepted | primary-source-confirmed | Feedback truth boundary | D2-ZKC-002, D2-X-006 | R1 |
| D2-AGT-021 | P2 | accepted | track-reproduced + primary-source-confirmed | Test/config isolation | — | R1 |

## Cross-system

| ID | Sev | Disposition | Evidence | Owner | Related/canonical | Wave |
|---|---|---|---|---|---|---|
| D2-X-001 | P0 | accepted | primary-source-confirmed | Platform security / signer operations | D2-AGT-002, D2-AGT-011, D2-ZKC-014 | R0 |
| D2-X-002 | P1 | accepted | primary-source-confirmed | Product/protocol architecture | D2-AGT-004, D2-ZKC-011 | R3 |
| D2-X-003 | P1 | accepted | primary-source-confirmed | Platform configuration | D2-AGT-008, D2-AGT-014, D2-ZKC-018 | R1 |
| D2-X-004 | P1 | accepted | primary-source-confirmed | Gateway/workers | D2-AGT-005, D2-AGT-015, D2-AGT-017, D2-ZKC-004 | R2 |
| D2-X-005 | P1 | accepted | primary-source-confirmed | System identity/provenance | D2-ZKC-001, D2-ZKC-008, D2-AGT-010 | R1 |
| D2-X-006 | P1 | accepted | primary-source-confirmed | Chain feedback/RAG | D2-AGT-019, D2-AGT-020 | R2 |
| D2-X-007 | P1 | accepted | primary-source-confirmed | Release engineering | D2-DATA-012, D2-ML-006, D2-ML-007, D2-ML-012, D2-AGT-018, D2-ZKC-013 | R1 |
| D2-X-008 | P1 | accepted | primary-source-confirmed | System manifest/promotion | D2-DATA-002, D2-DATA-010, D2-ML-015, D2-ZKC-008, D2-ZKC-009 | R1 |
| D2-X-009 | P0 | merged-duplicate | merged-duplicate | — | canonical D2-AGT-001 | — |
| D2-X-010 | P2 | accepted | primary-source-confirmed | Platform observability | D2-AGT-007, D2-ML-020 | R2 |
| D2-X-011 | P2 | accepted | primary-source-confirmed | Architecture governance | — | R4 |

## Primary verification disposition

All six unique P0s are primary-reproduced or primary-source-confirmed in the verification ledger. Every accepted P1 has been adjudicated into one of three states:

1. behavior reproduced by the primary or bounded track and independently reconciled to executable source;
2. static interface/absence claim independently confirmed from executable source and tracked artifact state; or
3. accepted source defect with an explicitly blocked live/hardware/artifact measurement.

No P0/P1 is silently left as `candidate`. The blocked measurements do not weaken the remediation requirement; they prevent performance or production-readiness acceptance until captured against the remediated release candidate.

## Registry closure rules

A finding closes only when its appendix-specific regression tests pass and the acceptance matrix maps the result to immutable evidence. Closing an umbrella cross-system finding additionally requires every linked module root cause and the composed end-to-end invariant to pass. Severity changes require a written rationale in the review record. IDs are never reused or deleted; rejected or merged rows remain tombstoned for traceability.
