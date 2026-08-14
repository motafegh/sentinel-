# Phase-8 Remote Data-Repair Handoff

**Date:** 2026-08-14
**Status:** READY FOR REPOSITORY-ONLY IMPLEMENTATION; LOCAL DATA REBUILD REQUIRED AFTERWARD
**Decision:** Do not launch the 100-epoch Phase-8 run from the current DATA vNext v1 artifacts.
**Purpose:** Let a remote repository assistant complete every defensible code, test, specification, and documentation repair before handing control back to the machine that holds the Git-ignored Solidity corpus and representations.

## Start state and authority

Work from the latest remote `main` containing this handoff. Read, in order:

1. `CLAUDE.md`
2. `data_module/CLAUDE.md`
3. `ml/CLAUDE.md`
4. `docs/plan/ml-R4/PLAN_STATUS_MATRIX.md`
5. `docs/plan/ml-R4/runs/2026-08-14_PHASE8_real_data_readiness_audit.md`
6. this handoff
7. `docs/plan/ml-R4/runs/2026-08-14_PHASE8_pretraining_launch_handoff.md`

The real-data audit is the evidence authority for this repair tranche. The pretraining launch handoff remains useful for runner commands and already-passed execution gates, but its launch authorization is superseded by the real-data audit.

## Objective

Implement and validate all repairs that can be proved from repository code and small checked-in synthetic/regression fixtures. Continue autonomously until the next honest step requires the local, Git-ignored raw `.sol` population, compiler installations, generated parquet ledgers, or representation tree.

The result must make the later local rebuild deterministic and reviewable. It must not claim that repository-only tests prove the physical 22,823-record corpus or the regenerated 21,657-plus representation population is correct.

## Required workstreams

### 1. Solidity-safe normalization and promotion order

- Replace regex-only comment stripping with a lexical-state-aware implementation that preserves comment markers inside single- and double-quoted strings, escapes, multiline comments, line comments, and line structure where provenance depends on it.
- Compile or otherwise validate the exact normalized text that will be promoted, rather than only the pre-normalized input.
- Add focused fixtures for URL strings, escaped quotes, comment-like string content, multiline comments containing delimiter-like text, orphan-fragment regressions, and idempotent normalization.
- Preserve deterministic content IDs and metadata rules, but introduce a new artifact/schema/version boundary wherever repaired normalized bytes can change IDs. Never overwrite or relabel historical v1/v2 outputs in place.

### 2. Deduplication and provenance

- Remove shared Ethereum-address literals as a deletion criterion. A shared address may define a leakage/base-family relationship, but it is not proof that two distinct sources are duplicates.
- Separate source-record identity, exact byte/text identity, normalized-code identity, and leakage-family identity in the data contract.
- Make duplicate provenance aggregation deterministic under multiprocessing; filename collision must not silently erase source paths or label claims.
- Preserve conflicting claims for explicit adjudication. Do not manufacture a contract-level truth by silently choosing one source row.
- Add tests proving that content-distinct files sharing an address survive, exact duplicates are represented once with all provenance retained, and normalized duplicates receive a common grouping identity before role assignment.

### 3. Version-aware compiler invocation

- Make optional compiler flags conditional on the selected `solc` version/capability. In particular, do not pass `--allow-paths` to versions that reject it.
- Add a regression test for the Solidity 0.4.9 behavior without requiring every developer machine to have that binary. Prefer a command-construction unit test plus a clearly gated integration test if a real compiler is available.
- Keep compiler choice, flags, exit result, and normalized-output validation auditable in metadata.

### 4. Graph target alignment

- Stop selecting libraries as vulnerability-bearing contract targets by fallback.
- Support explicit target-contract binding when source provenance identifies the intended declaration; otherwise use a deterministic, documented contract-only rule and fail closed when selection is ambiguous.
- Retain full-file provenance separately from the selected declaration.
- Add fixtures covering a file with `SafeMath` plus an application contract, multiple application contracts, interfaces/libraries only, explicit by-name selection, and an unknown requested name.
- Version graph representation metadata when target-selection semantics change.

### 5. Long-contract token evidence

- Add deterministic, testable coverage accounting at representation-build time: pre-subsampling token/window count, selected ranges, retained-token ratio, and—when line/site evidence exists—whether referenced locations are covered.
- Do not silently change the frozen model tensor contract `[4, 512]` or the existing model architecture in this tranche.
- Implement only a repository-supported, backward-compatible selection improvement if its evidence contract and tests are clear. Otherwise write a versioned design decision comparing contract-aware, vulnerability/site-aware, and bounded hierarchical approaches, with the exact local experiment needed to choose among them.
- The remote result must not claim that four windows are adequate merely because shapes and masks pass.

### 6. Rebuild/versioning interfaces

- Define a new DATA/representation version path for repaired artifacts. Historical evidence ledgers, roles, vNext v1 exports, and v9/v2.1 representations are immutable inputs, not mutation targets.
- Provide deterministic commands for: raw acquisition/provenance verification; preprocessing; evidence-ledger rebuilding; normalized/base-family grouping; role freezing; vNext export; graph/token regeneration; and all acceptance profilers.
- Ensure direct SmartBugs `time_manipulation` provenance can be distinguished from `bad_randomness` without deriving truth from filenames alone at training time. The new ledger should bind the physical ingestion and preprocessing provenance explicitly.
- Keep role assignment group-atomic after final normalized/base-family grouping.

## Repository validation expectations

Run the narrow tests for every touched module, then the applicable `data_module` and `ml` suites documented in their local instructions. At minimum:

- syntax/static checks for every changed Python file;
- normalization, deduplication, compiler-command, provenance, graph-selection, token-window, and versioning regression tests;
- the existing DATA-vNext policy/role/schema validators that do not require protected local artifacts;
- the existing ML unit suite that does not require the full representation tree;
- `git diff --check` and a final status/scope review.

If a test needs unavailable local data or compiler binaries, mark it explicitly as a local acceptance command; do not weaken or fake the test to obtain a pass.

## Prohibited actions and claims

- Do not launch the 100-epoch Phase-8 run.
- Do not generate or commit model checkpoints, large corpora, generated representation trees, local databases, secrets, or machine-specific absolute paths.
- Do not overwrite historical DATA artifacts or revise old hashes as though history had always contained the repair.
- Do not invent negative labels, convert weak labels to strong truth, or move duplicate families across roles independently.
- Do not report the 65 address-dropped contracts, the Solidity 0.4.9 contract, the five direct Timestamp contracts, the normalization-damaged population, or the wrong-target graphs as recovered until the local physical rebuild proves it.
- Do not authorize full training from repository-only evidence.

## Required stopping handoff

Continue through repository implementation, tests, documentation, and code review. Stop when the next action needs the local physical corpus or GPU environment. Commit and push the repository work, then publish one durable local-execution handoff containing:

1. the base and final commit hashes;
2. every changed file grouped by workstream;
3. exact tests run, pass/fail/skip counts, and any unavailable-tool reason;
4. all new artifact/schema/extractor version identifiers;
5. exact local rebuild commands in execution order;
6. expected invariant/count changes expressed as hypotheses, not fabricated results;
7. a rollback/recovery note that leaves historical artifacts intact;
8. a checklist that reruns:
   - `p8_audit_real_data.py`,
   - `p8_audit_representations.py`,
   - `p8_audit_token_coverage.py`,
   - the physical-binding validator,
   - the bounded real-data GPU micro-smoke,
   - and only after those pass, a fresh full-run launch decision.

The local operator will pull that commit, rebuild from the protected `.sol` sources, compare actual counts and recovery against the 2026-08-14 audit, and decide whether Phase 8 is newly authorized.

## Remote completion criterion

Remote work is complete only when all repository-safe high-confidence repairs above are either implemented and tested or explicitly deferred with a concrete technical reason and local experiment. “Code reviewed” or “tests pass” alone is not completion. The final remote status must remain:

> Repository repair complete; physical DATA rebuild and acceptance pending locally; 100-epoch training not yet authorized.
