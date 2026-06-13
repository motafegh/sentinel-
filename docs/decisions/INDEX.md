# ADR Index — sentinel-data

Architectural Decision Records are **append-only**. Never edit a merged ADR.

| # | Title | Date | Status |
|---|---|---|---|
| [ADR-0001](ADR-0001-sentinel-data-skeleton.md) | Why sentinel-data is a separate package | 2026-06-09 | Accepted |
| [ADR-0002](ADR-0002-code-bug-state-at-build-start.md) | Code bug state at build start (8 fixed, 3 open) | 2026-06-09 | Accepted |
| [ADR-0005](ADR-0005-verification-design.md) | Verification design (Stage 4) | 2026-06-12 | Accepted |
| [ADR-0006](ADR-0006-splitting-and-registry-design.md) | Splitting + registry design (Stage 5) | 2026-06-12 | Accepted |
| [ADR-0007](ADR-0007-analysis-design.md) | Analysis design (Stage 6) | 2026-06-12 | Accepted |
| [ADR-0008](ADR-0008-export-and-seam-swap-design.md) | Export + seam swap design (Stage 7) | 2026-06-13 | Accepted |
| [ADR-0009](ADR-0009-canonical-class-vocabulary.md) | Canonical 10-class vocabulary — labeling order as the single source of truth | 2026-06-12 | Accepted |

**Scope note:** the ML module's training/inference ADRs live in
[`docs/ml/adr/INDEX.md`](../ml/adr/INDEX.md) (a separate, ML-scoped
index). This index covers the SENTINEL v2 data module + cross-cutting
integration ADRs.

**Format:** Append-only. Never edit a merged ADR; write a new one to
supersede.
