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
