# Sentinel Portfolio Research Snapshot — 2026-09

> **Template status:** PRE-RELEASE DRAFT. Do not publish until P7/P8 blockers and final validation are closed.

## What this release represents

This release is a public portfolio/research snapshot of Sentinel's current engineering state. It is **not** a production-readiness claim, a repaired-model-quality release, or an assertion that every repository subsystem was independently authored or mastered by one person.

Sentinel began as Ali's long-running AI-assisted learning/building project across Python, data preparation, repeated ML training, Linux, graph/agent experiments, and dataset/model-quality diagnosis. The later R4 continuation is substantially AI-led research under Ali's direction. This release preserves that contribution boundary explicitly.

## What is included

### Public project surface

- recruiter/senior-engineer README with explicit claim and contribution boundaries;
- canonical architecture and trust-path documentation;
- multi-environment development/setup contract;
- validation and reproducibility guide;
- public security/reporting policy;
- fresh-clone bounded showcase;
- evidence-first engineering case-study package.

### Engineering case studies

The curated package covers seven decisions:

1. unknown is not negative;
2. leakage grouping is an evidence problem;
3. version the representation instead of patching history;
4. fail closed on unexplained structural drift;
5. promote a selector without rewriting the accepted lineage;
6. tool silence is not a clean result;
7. a valid proof does not prove the whole audit.

### Reproducibility and repository controls

- DATA dependency lock and zero-drift CI contract;
- current-R4 semantic validation separate from historical compatibility validation;
- current-tree and baseline-aware reachable-history secret scanning;
- DVC/local-artifact boundaries documented explicitly;
- fresh-clone showcase reports unexercised capabilities as `NOT_RUN` rather than treating them as clean.

## Current technical boundary

At this snapshot:

- historical R4 G0–G7 are passed and immutable;
- Phase 8 / G8 remains open;
- R4-D-011 accepts the exact V10 V2.6 physical representation lineage;
- R4-D-012 permits the guarded token selector only for a new versioned successor candidate that still requires physical acceptance;
- confirmed negatives remain zero;
- threshold fitting, calibration fitting, and untouched acceptance remain unsupported for the repaired path;
- no repaired R4 teacher has been trained or promoted;
- Run12 remains the historical operational ML baseline;
- the live audit MCP is read-only and the normal gateway path produces an off-chain report;
- the retained ZKML proof covers the compact proxy computation only;
- V3 context/provenance attestation is separate from the proof statement;
- no production signer/broadcaster is claimed.

## Start here

- `README.md` — project identity, architecture, boundaries, and navigation;
- `SHOWCASE.md` — dependency-light fresh-clone demonstration;
- `VALIDATION.md` — what current checks prove and do not prove;
- `docs/handbook/01_architecture.md` — canonical architecture/trust views;
- `docs/handbook/16_current_status.md` — exact current technical state;
- `docs/case-studies/README.md` — curated engineering decisions;
- `docs/plan/ml-R4/` — primary DATA/ML evidence and decision trail.

## Known limitations

- the repository is a multi-environment research monorepo, not a one-command production application;
- large historical/local DATA, model, RAG, and proving artifacts are not guaranteed in every fresh clone;
- the current operational ML runtime remains historical Run12;
- repaired-model discrimination is not established while confirmed-negative evidence remains absent;
- retained ZK assurance is proxy-only and the current bundle records `check_mode="UNSAFE"` as a production-assurance limitation;
- no production transaction signer/broadcaster is part of the current analysis service.

## Release validation

**Replace this section only after final checks run.**

Required final record:

- final commit SHA: `<FINAL_SHA>`;
- PR #72 merged: `<YES/NO>`;
- Handbook: `<PASS>`;
- Portfolio showcase: `<PASS>`;
- DATA reproducibility: `<PASS>`;
- SENTINEL system alignment: `<PASS>`;
- Security hygiene: `<PASS>`;
- accidental temporary refs removed: `<YES>`;
- external provider-credential revocation/rotation disposition confirmed: `<STATUS>`;
- repository name decision: `<NAME>`;
- license decision: `<LICENSE>`.

Do not publish this template with placeholders or unverified `PASS` values.
