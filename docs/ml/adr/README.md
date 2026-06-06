# ML Module Architecture Decision Records (ADRs)

This directory captures the **why** behind significant architectural decisions in the
SENTINEL ML module. Each ADR is a short, structured document that records one decision
and its context, so future contributors (and future Ali) can understand the reasoning
without re-discovering it from scratch.

## When to write an ADR

Write an ADR when:

- The decision is **non-obvious** — would not be immediately clear from reading the code
- The decision was **expensive to make** — took real work, runs, or analysis
- The decision is **likely to be revisited** — schema changes, refactors, technology swaps
- Future contributors will be **confused without context** — "why is this here?"

Do NOT write an ADR for:

- Routine code style choices (use the project's standard)
- Bug fixes with clear commits (the commit message is enough)
- Decisions that are obvious from the code itself

## Format

Each ADR uses MADR-lite (Markdown ADR). Template at [`_template.md`](_template.md).
~80-150 lines per ADR. Keep it short and honest.

### Status lifecycle

```
Proposed  → Accepted  → Superseded  → Deprecated
            ↓
          Rejected (terminal — never adopted)
```

- **Proposed** — drafted but not yet decided on
- **Accepted** — decision is in effect
- **Superseded** — replaced by a later ADR (link to it)
- **Deprecated** — no longer applies (project moved on, domain changed)
- **Rejected** — considered and explicitly not adopted (record why)

## Index

See [`INDEX.md`](INDEX.md) for the full table of ADRs with status, date, and one-line
summary.

## Tier structure

ADRs are written in priority tiers. Tier 1 covers foundational decisions; Tier 2 covers
process and methodology; Tier 3 covers deferred or future work.

**Tier 1 (this session):** 0001-0006 — schema, multi-label, architecture, GNN phases,
dataset, loss.

**Tier 2 (future):** 0007-0012 — Slither IR, windowed tokenization, cache architecture,
pre-flight gates, sampling, training kill criteria.

## References

- Project memory: `~/.claude/projects/-home-motafeq-projects-sentinel/memory/MEMORY.md`
- Chronological changelog: `docs/changes/INDEX.md`
- Pre-run fix proposals: `docs/pre-run9-fixes/`
- MADR standard: <https://adr.github.io/madr/>
