# DATA D2 — guarded selector implementation working record

Date: 2026-09-26
Work package: D2 — implementation
State: IN_PROGRESS
Branch: `work/data-d0-selector-contract`
D0 authority: `2026-09-26_data_selector_contract_reconstruction_working.md`
D1 authority: `2026-09-26_data_selector_interface_lineage_design_working.md`

## D2 increment 1 — selector + lineage primitives

Status: **CLOSED / PASS**
Implementation commit: `7fd12a62ad901126538c024986466786435cabfc`

### Added production owners

- `data_module/sentinel_data/representation/r4_window_selector.py`
  - pure DATA-owned selector implementation;
  - exact historical `historical_linspace_v1` control indices;
  - exact retained greedy marginal-union target-token coverage candidate;
  - deterministic lowest-window-index tie break;
  - historical-first then ascending fill;
  - strict target-coverage improvement guard;
  - immutable `SelectorDecision` with invariant validation;
  - missing/empty target evidence fails closed rather than becoming control fallback.

- `data_module/sentinel_data/representation/r4_guarded_lineage.py`
  - fresh guarded token/candidate schema identities;
  - immutable R4-D-011 graph-parent authority;
  - canonical selector configuration and digest;
  - canonical JSON/SHA-256 helpers;
  - canonical target-evidence construction/validation;
  - canonical selector metadata construction/validation.

### Added focused tests

- `data_module/tests/test_representation/test_r4_window_selector.py`
- `data_module/tests/test_representation/test_r4_guarded_lineage.py`

The tests compare production selector behavior directly with the retained research implementation rather than making production import that research module.

### Historical owners unchanged

No changes were made to:

- `ml/src/data_extraction/bounded_window_selector.py`;
- `ml/src/data_extraction/windowed_tokenizer.py`;
- `data_module/sentinel_data/representation/tokenizer.py`;
- `data_module/sentinel_data/representation/r4_orchestrator.py`;
- `data_module/sentinel_data/vnext/r4_v10_binding.py`;
- R4-D-011 artifacts/evidence.

### Validation performed

Local isolated validation of the exact authored source/tests established:

- Python syntax/import construction passes for both new modules and both test files;
- historical rounded-linspace equivalence against retained NumPy behavior for `total_windows=0..4999` and `max_windows=1..8`: zero mismatches;
- 2,360 deterministic randomized guarded-selector comparisons against the retained algorithm: selected/control/candidate indices, fallback state, target coverage and retained coverage all matched;
- canonical selector-config digest: `7ce20027e124aef763b6e448bd70d0c562b6d33cb2156814a9ed19e04bb25151`;
- target-evidence and selector-metadata round-trip validation passes;
- tampered target evidence, selector metadata and graph-parent identity fail closed;
- all authored focused test logic passes in the isolated harness.

Repository push workflow `Handbook` run `36262215432` completed `success` on commit `7fd12a6`. That workflow is not a substitute for the project Python pytest suite; a full repository pytest invocation still belongs to the normal local/CI verification boundary when available.

### D2 state after increment 1

The production selector and lineage primitives now exist without changing any accepted historical path. This is sufficient to proceed to the guarded-tokenization seam, but not to generate a physical candidate.

## Exact next increment

D2 increment 2 only:

1. add `r4_guarded_tokenizer.py`;
2. reuse `r4_target_spans.target_contract_char_spans()` unchanged;
3. tokenize the persisted repaired source under the frozen GraphCodeBERT `[4,512]` contract;
4. map exact target spans to token ranges;
5. call the new production selector;
6. emit established coverage fields plus canonical `target_evidence` and `token_selector` metadata;
7. add focused tokenizer tests for invalid evidence, padding/shape, control fallback, strict improvement, metadata consistency and determinism.

Do not yet implement `r4_guarded_candidate.py`, full-population generation or `r4_guarded_binding.py` in that increment.


## D2 increment 2 — guarded tokenization seam

Status: **CLOSED / PASS (focused local verification; repository Phase-8 CI wired for PR/main)**

Implementation commits:

- `25f1cb211785834de2d345616e17c43bc0da2021` — guarded tokenization seam;
- `f595305e2b9a6e4c16712d0531b6c89d93706a09` — focused tokenizer tests;
- `fa7dfe72eafcc71f92347d7556808bedc730e881` — fail-closed tokenizer hardening;
- `6418844118ca4b3c6860355f496473c3789b6a25` — Phase-8 CI coverage for the new modules/tests.

### Added production owner

`data_module/sentinel_data/representation/r4_guarded_tokenizer.py`

The seam now:

1. consumes the exact persisted repaired Solidity source bytes;
2. requires a non-empty unique requested-target tuple;
3. reuses `r4_target_spans.target_contract_char_spans()` unchanged;
4. requires the frozen fast
   `microsoft/graphcodebert-base` tokenizer identity;
5. obtains raw token IDs + offset mappings without a second comment-removal
   transform;
6. maps target character spans to exact raw-token ranges;
7. computes the frozen overflow-window geometry from
   `window_size=512`, tokenizer special-token count and `stride=256`;
8. invokes the DATA-owned `target_aware_guarded_v1` selector;
9. requires the computed window count to equal the tokenizer's actual overflow
   window count;
10. emits exactly `[4,512]` long tensors with stable padding;
11. preserves the existing `r4-token-coverage-v1` top-level telemetry for the
    emitted windows;
12. binds canonical `target_evidence`, `token_selector`,
    selector-config digest and guarded token-lineage parent identity.

Invalid requested targets, zero-token target mappings, tokenizer identity drift,
shape/window-count drift, invalid matrices and selector/metadata failures all
fail closed before a successful token artifact can be returned.

### Focused tests

Added:

`data_module/tests/test_representation/test_r4_guarded_tokenizer.py`

Coverage includes:

- under-cap control fallback + padding to `[4,512]`;
- exact repaired-source reuse without a second source mutation;
- over-cap strict target-coverage improvement;
- target-evidence/selector metadata cross-binding;
- missing/empty target evidence failure;
- tokenizer identity failure;
- repeated deterministic tensors/metadata;
- exact character-span -> token-range mapping;
- frozen overflow-window geometry.

### Validation

Focused isolated harness against the authored seam/tests:

- `9 passed`;
- source compiles;
- strict-improvement and control-fallback branches both exercised;
- repeated tensors and metadata are identical;
- hardened fail-closed tokenizer validation retained the same `9/9` pass.

Repository `Handbook` push validation passed for both implementation and
hardening commits:

- run `36263826004`: success;
- run `36263940100`: success.

The stronger `R4 Phase 8 real-data repository repair` workflow is configured
to run only on `main`, pull requests to `main`, or manual dispatch. This
branch does not receive that workflow on ordinary push. Commit `6418844`
therefore adds the three new production modules and three focused tests to that
workflow's compile/regression lists so they cannot be omitted at the PR/main
verification boundary.

No claim is made that the full Phase-8 repository regression suite ran on this
branch.

### D2 state after increment 2

The selector, lineage metadata and guarded tokenization seam now exist. The
system still cannot generate a physical guarded representation candidate; the
accepted R4-D-011 builder/root remains untouched.

## Exact next increment

D2 increment 3 only: implement the fresh physical candidate assembler
`r4_guarded_candidate.py` with focused tests.

Required boundary:

1. consume the exact R4-D-011 accepted root/acceptance record as immutable
   parent;
2. enumerate parent identities rather than rediscovering a new population;
3. require a fresh candidate root and reject parent/candidate aliasing;
4. preserve graph bytes exactly from the parent;
5. validate inherited requested/actual targets and runtime identity before
   token generation;
6. generate only new guarded token payloads through
   `r4_guarded_tokenizer.py`;
7. write new sidecars carrying inherited graph semantics plus guarded lineage
   metadata;
8. emit structured per-identity failures;
9. prove parent files remain unchanged in focused tests.

Do not run full-population generation and do not implement the successor
population binder in the same increment.
