# D1 v3 — SENTINEL technical mastery handbook

**Date:** 2026-07-13
**Verified implementation baseline:** `ac78c057b`
**Audience:** Ali and future maintainers
**Delivery:** Markdown plus standard-library validation; no rendered docs site

## Objective

Create a source-grounded learning and implementation system that explains the complete system progressively, adds intermediate-to-advanced source companions and controlled ownership labs, and separates implemented behavior, disconnected behavior, degraded operation, planned work, and artifact availability.

The implemented canonical module is [`docs/handbook`](../../handbook). D1 v3 extends the v2 compact chapters with ten guides under `technical/`, ten matching labs under `labs/`, symbol-aware validation, lab preflights, and documentation unit tests. The page-level plans in [`handbook/`](handbook) are retained as superseded v1 planning history.

## Scope and truth policy

- Cover DATA, ML, ZKML, Contracts, AGENTS orchestration/services, gateway, RAG, MLOps, interpretability, evaluation, security, operations, and artifact management.
- Source is authoritative; use relative links plus `path::symbol` anchors.
- Keep volatile test counts only in `16_current_status.md` and bind them to date/commit.
- Classify artifacts as tracked, DVC-managed local, regenerated, ignored/private, or ignored local.
- Document product defects and failing tests without changing runtime code during D1.
- Never copy environment values, RPC credentials, private keys, or private endpoint material.

## Canonical deliverables

The 18-page registry is defined in [`docs/handbook/_meta/handbook.toml`](../../handbook/_meta/handbook.toml), from `00_README.md` through `17_reference.md`. Every page follows the compact progressive template and links to deeper learning/ADR/report material rather than duplicating it.

Documentation interfaces:

- `_meta/handbook.toml`: pages, source ownership, ports/routes, critical compatibility facts, test tiers, and artifact classes.
- `tools/verify_handbook.py static`: links, tracked paths, template/navigation, secret patterns, volatile counts, and source-derived critical facts.
- `tools/verify_handbook.py inventory`: discovered services/routes/nodes/stages/schema/proxy/contracts/test files/artifacts.
- `tools/verify_handbook.py live`: explicit module, service, GPU, EZKL, and Anvil checks without hidden failures.
- `tools/verify_handbook.py lab`: list and preflight lab source, artifact, and executable requirements.
- `technical/` and `labs/`: ten source-guided companions and ten controlled ownership exercises.
- `tools/tests/`: standard-library validator unit tests.
- `.github/workflows/handbook.yml`: fast static validation when handbook or declared source anchors change.

## Required current truths

- Gateway 8000, ML 8001, five MCP services 8010–8014, Anvil 8545.
- Gateway audits do not invoke MCP `submit_audit`; reports begin unsubmitted.
- DATA has ten lifecycle stages, while the current label CLI adapter remains incomplete.
- The 14-node graph begins at `ml_assessment` and ends at `visualizer` after reflection/explanation/visualization.
- EZKL proves proxy computation over 128 inputs and ten outputs—not `verdict_provable` or teacher execution.
- Circuit v2.0 uses public inputs/outputs, fixed parameters, 138 signals, and `UNSAFE` check mode.
- Provenance is an off-chain operator assertion and may be unsigned.
- Submission gaps include separate invocation, placeholder fields, proof-hash mismatch, model-hash propagation, and shared proof-file concurrency.
- Prompt sanitization detects on original source, then strips comments, then delimits; pattern names are source-derived.
- Reliability uses measured precision shrunk toward the L1 prior with alpha 5.
- Fresh-clone claims exclude ignored DATA, ignored local V2 test, local/untracked DVC pointers, proving key, and SRS.

## Implementation order and acceptance

1. Establish metadata, status, source/artifact inventory.
2. Write DATA → ML → ZKML → Contracts → AGENTS chapters.
3. Stabilize cross-module, security, and evaluation claims.
4. Write distinct runtime flows, operations, playbooks, reference, learning paths, and root README.
5. Run static/inventory validation, inspect the documentation-only diff, and preserve failures honestly.

Acceptance requires every active subsystem and public interface to have a canonical owner and deep guide; every subsystem to have a worked success/failure trace and lab; static/unit/safe-preflight checks to pass; commands to be tiered with prerequisites; fresh-clone limitations to be explicit; and no planned/degraded/local-only behavior to be presented as production-complete.

## Implemented v3 expansion

- All 18 canonical pages retain their progressive summary and now include a technical-mastery layer.
- Ten technical companions provide source maps, call chains, annotated excerpts, worked shapes/state, success/failure traces, design reasoning, safe changes, tests, and ownership checks.
- Ten labs provide controlled test/fixture edits, expected success/failure, reset steps, readiness tiers, and review rubrics.
- Metadata registers guide ownership, objectives, source symbols, labs, prerequisites, artifacts, and safe-preflight eligibility.
- Validation resolves Python symbols with AST, Solidity symbols with constrained parsing, validates learning templates and coverage, exposes `lab` preflights, and has standard-library unit tests.
- CI remains documentation-fast: static contract plus validator unit tests only.
