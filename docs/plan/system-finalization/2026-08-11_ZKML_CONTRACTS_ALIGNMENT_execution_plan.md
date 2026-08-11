# SENTINEL ZKML + Contracts Alignment — Execution Plan

**Date:** 2026-08-11  
**Branch:** `system/zkml-contracts-alignment`  
**Baseline:** canonical `main` after R4 Phase 2 merge  
**Scope:** `zkml/`, `contracts/`, and their active seams with `ml/` and `agents/`.

## Objective

Reconstruct the current executable ZKML/on-chain path, identify stale or contradictory interfaces relative to the current ML/AGENTS system and R0 containment rules, then finish all remotely safe alignment work without changing R4 DATA/ML label-policy decisions.

## Constraints

- Source code is canonical; docs are secondary evidence only.
- Do not alter protected DATA/ML artifacts, checkpoints, thresholds, or R4 Phase-3 state.
- Do not change decision thresholds/numeric policy without measurement.
- Preserve R0 fail-closed, proof-scope, signer, artifact-identity, and transaction-state-machine guarantees.
- No claim of successful EZKL proving, Foundry integration, chain submission, or deployment without an execution environment that actually runs it.
- Keep this work isolated from `r4/phase3-evidence-ledger`.

## Work packages

### A. Executable contract reconstruction
1. Trace ML `/fusion-embedding` output contract.
2. Trace AGENTS/direct-audit submission inputs and proof/public-signal construction.
3. Trace proxy model/ONNX/EZKL input-output shapes and artifact identities.
4. Trace Solidity registry/verifier/staking submission and storage semantics.
5. Build one end-to-end interface matrix: producer → payload → consumer → identity → failure semantics.

### B. Staleness and alignment audit
Audit for:
- dimensional/class-order mismatches;
- stale model/proxy/verifier identities;
- proof/public-signal mismatch;
- model-hash/proof-hash inconsistencies;
- sigmoid/logit ambiguity;
- unsafe EZKL settings or generated-artifact assumptions;
- mutable shared proof files/races;
- signer/transaction-state bypasses;
- legacy write paths that violate R0 containment;
- docs/tests that assert behavior no longer implemented.

### C. Remote-safe implementation
Only after findings are explicit:
- centralize duplicated interface constants where this is behavior-preserving;
- add fail-closed validation at module boundaries;
- bind artifact/model/proxy/proof identities consistently;
- remove or quarantine obsolete compatibility paths when their replacement already exists;
- add static/unit tests that do not require proving/deployment;
- add deterministic local integration runner(s) for the eventual machine-dependent checks.

### D. Local-only validation handoff
Prepare one command/checklist for later local execution covering:
- ZKML Python tests;
- ONNX/proxy artifact compatibility;
- EZKL setup/witness/prove/verify on the intended version;
- Foundry tests;
- Anvil direct submission;
- adversarial proof/public-signal/model-identity cases;
- full ML → ZKML → registry seam.

## Deliverables

- source-backed alignment audit;
- interface/identity matrix;
- prioritized defect register (`P0` correctness/security, `P1` integration/reproducibility, `P2` cleanup/docs);
- remote-safe fixes + tests where justified;
- explicit local-only blockers and one-command validation handoff;
- draft PR kept unmerged until required local execution passes.

## Stop line

This track must not redesign the ML architecture or decide DATA vNext label policy. If a ZKML/contracts interface depends on a future R4 output, define a versioned boundary and preserve backward compatibility rather than guessing the future artifact.
