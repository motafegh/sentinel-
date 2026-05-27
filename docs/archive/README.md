# SENTINEL Documentation Archive

This directory contains historical documents that have been superseded by newer versions but are retained for reference and historical context.

## Structure

### audit-reports/
External adversarial audit reports of the ML module:
- `sentinel_ml_adversarial_audit.md` (2026-05-23) — First adversarial audit
- `25-05-2026-sentinel-ml-adversarial-audit.md` (2026-05-24) — Second adversarial audit  
- `audit-on-the-audit.md` (2026-05-25) — Meta-audit reviewing prior audits

**Note:** These audits identified critical data quality issues (C-2 label contamination) that explain the F1 ceiling (~0.287) observed across v7, v8-AB, and PLAN-3A runs.

### historical-proposals/
Major architectural proposals that were implemented or superseded:
- `SENTINEL-v7-Comprehensive-Improvement-Proposal.md` — v7 architecture proposal (fully implemented 2026-05-18)
- `phases-v8-and-earlier.md` — v8 roadmap document (historical record)
- `2026-05-23-graphcodebert-gnn-prefix-injection-proposal.md` — GCB prefix injection proposal (Phase 3.6 direction)

### superseded-plans/
Planning documents replaced by updated versions:
- `AGENTS_PLAN.md` — Superseded by `AGENTS_PLAN_V2.md` (2026-05-27)
- `agents-module-suggestions.md` — Early agents suggestions
- `agents-modules-proposal-2.md` — Second iteration of agents proposal

## Other Archived Documents (root level)

| File | Description |
|------|-------------|
| `data-quality-findings.md` | Data quality analysis identifying label noise issues |
| `jk-attention-collapse-findings.md` | Analysis of JK attention weight collapse in three-phase GNN |
| `v8.0-B-training.md` | Training run record for v8.0-B (killed early) |
| `gcb-p1-run1-analysis-and-imp-all.md` | GraphCodeBERT P1 Run 1 post-mortem |
| `gcb-p1-run4-final-analysis.md` | GraphCodeBERT P1 Run 4 final analysis |
| `sentinel-c2-concrete-data-fixing-solutions.md` | Proposed solutions for C-2 label contamination |
| `IMPROVEMENT_BACKLOG.md` | Historical improvement backlog (superseded by current planning) |
| `ML_API_MLOPS_GAPS.md` | ML API and MLOps gaps analysis |
| `model-evolution-analysis-plan.md` | Model evolution analysis plan (historical) |
| `2026-05-27-three-tier-inference-output.md` | Three-tier inference output proposal |

## Referencing Archived Documents

When citing archived documents in new work:
1. Prefer current/active documents when available
2. Reference archived docs with clear "historical" or "superseded" notation
3. Use [docs/CHANGELOG.md](../CHANGELOG.md) as the authoritative project history

## Cleanup Policy

Documents are moved to archive when:
- They are superseded by a newer version (e.g., AGENTS_PLAN → AGENTS_PLAN_V2)
- They describe completed/historical phases no longer relevant to current work
- They contain analysis of killed experiments or abandoned directions
- They are external audits whose findings have been addressed or incorporated

Archived documents are **never deleted** — they provide valuable context for understanding design decisions and avoiding past mistakes.
