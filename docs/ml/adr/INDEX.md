# SENTINEL ML — Architecture Decision Records

This directory contains the Architecture Decision Records (ADRs) for SENTINEL's ML
module. ADRs capture *why* a decision was made, the alternatives considered, and
the consequences. They complement the chronological changelog at
`docs/changes/INDEX.md` (which records *what* changed when).

Each ADR is locked once accepted. Superseding decisions are recorded in the
`Superseded by:` field and the new ADR explicitly references its predecessor.

## Tier structure

| Tier | Status | Scope |
|------|--------|-------|
| **Tier 1** | Done (this directory) | Foundational design choices. Lock the architecture. |
| **Tier 2** | Deferred to future session | Implementation details, sub-system design. |
| **Tier 3** | Backlog | Hyperparameters, training recipes, calibration thresholds. |

## Index

| # | Title | Status | Date | One-line summary |
|---|-------|--------|------|------------------|
| 0001 | [Schema versioning](0001-schema-versioning.md) | Accepted | 2026-06-06 | Single `FEATURE_SCHEMA_VERSION` string, asserted at module load, suffixed in all cache keys. |
| 0002 | [Multi-label formulation](0002-multi-label-formulation.md) | Accepted | 2026-06-06 | 10-class sigmoid multi-hot. `class_9` reserved (SENTINEL phase 2, off by default). |
| 0003 | [Dual-path GNN+CodeBERT Four-Eye architecture](0003-dual-path-four-eye-architecture.md) | Accepted | 2026-06-06 | GNN + LoRA-CodeBERT cross-attend; 4 eyes (GNN/TF/Fused/CFG) summed for final logits. |
| 0004 | [Three-phase GAT routing](0004-three-phase-gat-routing.md) | Accepted | 2026-06-06 | 8-layer GAT split Ph1 structural+CONTAINS, Ph2 CF+ICFG+EXTERNAL_CALL sub-routed, Ph3 REVERSE_CONTAINS round-trip. |
| 0005 | [BCCC-SCsVul-2024 as primary dataset](0005-bccc-dataset-choice.md) | Accepted | 2026-06-06 | 41,576 deduped contracts. SmartBugs-curated held out as OOD benchmark. 87.9% pre-0.8 Solidity. |
| 0006 | [Loss formulation](0006-loss-formulation.md) | Accepted | 2026-06-06 | ASL γ⁻=2 γ⁺=1 per-eye + aux BCE pathway (0→0.30 over 8ep) + 0.005 JK entropy. |

## Deferred (Tier 2 backlog)

These were identified during Tier 1 writing but are deferred to a future session:

- **0007:** Slither IR as the canonical extraction source (vs hand-rolled AST walk)
- **0008:** Windowed tokenization strategy (linspace subsample, stride 256, max 4 windows)
- **0009:** Cache architecture (`.pkl` vs `.parquet` vs LMDB, invalidation by schema version)
- **0010:** Pre-flight gate methodology (smoke tests 1-8, gate criteria for new runs)
- **0011:** Sampling strategy (WeightedRandomSampler, BCCC class imbalance mitigation)
- **0012:** Training kill criteria (F1-macro regression, aux BCE explosion, JK collapse)

## How to add a new ADR

1. Copy `_template.md` to `NNNN-short-title.md` (use the next available 4-digit number).
2. Fill in the front matter, Context, Decision, Consequences, Alternatives
   Considered, and References sections.
3. Add a row to this INDEX.
4. If the ADR supersedes a prior ADR, update the prior ADR's `Superseded by:` field.
5. Add code-level cross-link comments (single line) in any source files that
   implement the decision. Keep these to one line per file — full docstrings
   are not required.

## How to read ADRs

Start with **0001** (schema versioning) and **0002** (multi-label formulation) —
these define the data shapes. Then **0003** (architecture), **0004** (GNN
routing), **0005** (dataset), **0006** (loss). The references section of each
ADR links forward and backward.

For the chronological view of *what* changed when, see
[`docs/changes/INDEX.md`](../../changes/INDEX.md). For the SENTINEL project's
live state, see `~/.claude/projects/-home-motafeq-projects-sentinel/memory/MEMORY.md`.
