# SENTINEL — Documentation Index

**Last updated:** 2026-05-27 (post-cleanup reorganization)

## Quick Start

| Document | Purpose |
|----------|---------|
| [STATUS.md](STATUS.md) | **Current state** — pipeline status, training results, data locations, active checkpoints |
| [CHANGELOG.md](CHANGELOG.md) | **Project history** — complete timeline of all changes from foundation through current phase |
| [AGENTS_PLAN_V2.md](AGENTS_PLAN_V2.md) | **Active agents plan** — Phase 0 complete, next steps for agents module development |
| [ROADMAP.md](ROADMAP.md) | **Forward-looking plan** — upcoming work, deferred backlog, module completion targets |

---

## Core Reference

### Architecture & Design
- [SENTINEL-architecture.md](../Project-docs/SENTINEL-architecture.md) — System architecture overview
- [SENTINEL-modules.md](../Project-docs/SENTINEL-modules.md) — Module breakdown and responsibilities
- [ZKML_PIPELINE.md](ZKML_PIPELINE.md) — ZK proof pipeline design and EZKL integration
- [CONTRACTS.md](CONTRACTS.md) — Smart contract architecture (AuditRegistry, SentinelToken, ZKMLVerifier)
- [ENCODING_REFERENCE.md](ENCODING_REFERENCE.md) — On-chain encoding format specification
- [DESIGN_ISSUES_AND_ABLATIONS.md](DESIGN_ISSUES_AND_ABLATIONS.md) — Open design questions, risks, and experiment candidates

### Project Specification
See [Project-Spec/](Project-Spec/) for formal module specifications:
- [SENTINEL-OVERVIEW.md](Project-Spec/SENTINEL-OVERVIEW.md) — Project overview
- [SENTINEL-INDEX.md](Project-Spec/SENTINEL-INDEX.md) — Specification index
- [SENTINEL-M1-ML.md](Project-Spec/SENTINEL-M1-ML.md) — ML module specification
- [SENTINEL-M2-ZKML.md](Project-Spec/SENTINEL-M2-ZKML.md) — ZKML module specification
- [SENTINEL-M3-MLOPS.md](Project-Spec/SENTINEL-M3-MLOPS.md) — MLOps module specification
- [SENTINEL-M4-AGENTS.md](Project-Spec/SENTINEL-M4-AGENTS.md) — Agents module specification
- [SENTINEL-M5-M6-PLATFORM.md](Project-Spec/SENTINEL-M5-M6-PLATFORM.md) — Platform modules (Contracts + API)
- [SENTINEL-CONSTRAINTS.md](Project-Spec/SENTINEL-CONSTRAINTS.md) — Technical and business constraints
- [SENTINEL-ADR.md](Project-Spec/SENTINEL-ADR.md) — Architecture Decision Records
- [SENTINEL-COMMANDS.md](Project-Spec/SENTINEL-COMMANDS.md) — CLI commands reference
- [SENTINEL-EVAL-BACKLOG.md](Project-Spec/SENTINEL-EVAL-BACKLOG.md) — Evaluation backlog

---

## ML Module Documentation

### Current State & Results
- [STATUS.md](STATUS.md) — Training status, v7/v8 results, comparison tables
- [docs/ml/](ml/) — ML analysis reports:
  - [v8-vs-v7-comparison-results.md](ml/v8-vs-v7-comparison-results.md) — Full v7 vs v8-AB comparison
  - [v8-AB-training-analysis.md](ml/v8-AB-training-analysis.md) — v8.0-AB training deep dive
  - [plan-3a-results.md](ml/plan-3a-results.md) — PLAN-3A ICFG-only ablation results
  - [v8-vs-v7-comparison-plan.md](ml/v8-vs-v7-comparison-plan.md) — Comparison methodology

### Educational Guides
See [docs/ml-educational-docs/](ml-educational-docs/) for structured learning:
- [README.md](ml-educational-docs/README.md) — Guide index
- [ML-T1-ARCHITECTURE.md](ml-educational-docs/ML-T1-ARCHITECTURE.md) — Model architecture tutorial
- [ML-T2-TRAINING-PIPELINE.md](ml-educational-docs/ML-T2-TRAINING-PIPELINE.md) — Training pipeline tutorial
- [ML-T3-DATA-PIPELINE.md](ml-educational-docs/ML-T3-DATA-PIPELINE.md) — Data pipeline tutorial
- [ML-E1-DESIGN-RATIONALE.md](ml-educational-docs/ML-E1-DESIGN-RATIONALE.md) — Design decisions explained
- [ML-E2-TRAINING-GUIDE.md](ml-educational-docs/ML-E2-TRAINING-GUIDE.md) — Training guide

---

## Agents Module Documentation

### Active Plans
- [AGENTS_PLAN_V2.md](AGENTS_PLAN_V2.md) — Current agents architecture and implementation plan (supersedes V1)
- [AGENTS_MODULE_PROPOSAL.md](proposal/AGENTS_MODULE_PROPOSAL.md) — Full agents module proposal

### Implementation Status
Phase 0 is **COMPLETE** (46/46 tests pass):
- `agents/src/orchestration/routing.py` — Per-class thresholds, routing rules, tool matrix
- `agents/src/orchestration/state.py` — AuditState with verdicts/confirmations/contradictions
- `agents/src/orchestration/nodes.py` — Evidence router, scoped Slither detectors, synthesizer
- `agents/src/orchestration/graph.py` — LangGraph topology with SqliteSaver checkpointer

Topology: `START → ml_assessment → evidence_router → [rag_research ‖ static_analysis] → audit_check → synthesizer → END`

---

## Changelog & Session Logs

- [CHANGELOG.md](CHANGELOG.md) — Single authoritative changelog (project-wide)
- [changes/INDEX.md](changes/INDEX.md) — One-line summary of every dated changelog
- [changes/*.md](changes/) — Full session-level detail per date

---

## Archive

Historical documents kept for reference but superseded by current versions:

### Superseded Plans
- [archive/superseded-plans/AGENTS_PLAN.md](archive/superseded-plans/AGENTS_PLAN.md) — Replaced by AGENTS_PLAN_V2.md
- [archive/superseded-plans/agents-module-suggestions.md](archive/superseded-plans/agents-module-suggestions.md) — Early agents suggestions
- [archive/superseded-plans/agents-modules-proposal-2.md](archive/superseded-plans/agents-modules-proposal-2.md) — Second agents proposal iteration

### Historical Proposals
- [archive/historical-proposals/SENTINEL-v7-Comprehensive-Improvement-Proposal.md](archive/historical-proposals/SENTINEL-v7-Comprehensive-Improvement-Proposal.md) — v7 improvement proposal (implemented)
- [archive/historical-proposals/phases-v8-and-earlier.md](archive/historical-proposals/phases-v8-and-earlier.md) — v8 roadmap (historical)
- [archive/historical-proposals/2026-05-23-graphcodebert-gnn-prefix-injection-proposal.md](archive/historical-proposals/2026-05-23-graphcodebert-gnn-prefix-injection-proposal.md) — GCB prefix injection proposal

### Audit Reports
- [archive/audit-reports/](archive/audit-reports/) — External adversarial audits:
  - `sentinel_ml_adversarial_audit.md` — First adversarial audit (2026-05-23)
  - `25-05-2026-sentinel-ml-adversarial-audit.md` — Second adversarial audit (2026-05-24)
  - `audit-on-the-audit.md` — Meta-audit reviewing prior audits (2026-05-25)

### Other Archived Documents
- [archive/data-quality-findings.md](archive/data-quality-findings.md) — Data quality analysis
- [archive/jk-attention-collapse-findings.md](archive/jk-attention-collapse-findings.md) — JK attention analysis
- [archive/v8.0-B-training.md](archive/v8.0-B-training.md) — v8.0-B training run record
- [archive/gcb-p1-run1-analysis-and-imp-all.md](archive/gcb-p1-run1-analysis-and-imp-all.md) — GCB-P1 Run 1 analysis
- [archive/gcb-p1-run4-final-analysis.md](archive/gcb-p1-run4-final-analysis.md) — GCB-P1 Run 4 analysis
- [archive/sentinel-c2-concrete-data-fixing-solutions.md](archive/sentinel-c2-concrete-data-fixing-solutions.md) — C-2 data fix solutions
- [archive/IMPROVEMENT_BACKLOG.md](archive/IMPROVEMENT_BACKLOG.md) — Old improvement backlog
- [archive/ML_API_MLOPS_GAPS.md](archive/ML_API_MLOPS_GAPS.md) — ML API/MLOps gaps analysis

---

## Module README Files

For module-specific documentation, see inline README files:
- [ml/README.md](../ml/README.md) — ML module overview
- [agents/README.md](../agents/README.md) — Agents module overview
- [contracts/README.md](../contracts/README.md) — Contracts module overview
- [zkml/README.md](../zkml/README.md) — ZKML module overview
