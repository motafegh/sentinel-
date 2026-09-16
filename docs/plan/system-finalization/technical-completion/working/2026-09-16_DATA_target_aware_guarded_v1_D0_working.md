# DATA D0 — `target_aware_guarded_v1` source/evidence reconstruction

**Date:** 2026-09-16  
**Branch:** `technical-completion/data-target-aware-guarded-v1`  
**Base `main`:** `57f39d652cbec1092084b4efbf41bec6a117ba07`  
**Work package:** D0 — source/evidence reconstruction  
**State:** `DESIGN_READY` after the reconstruction below  

## 1. Exact question

Reconstruct the executable contract promoted by R4-D-012 for `target_aware_guarded_v1`, identify the immutable historical-control behavior and all relevant representation seams/consumers, and establish a versioning/integration boundary that can enter D1/D2 without modifying R4-D-011.

## 2. Authorities inspected

Controlling/current:

- `CLAUDE.md`
- `docs/plan/system-finalization/technical-completion/00_MASTER_TECHNICAL_COMPLETION_ROADMAP.md`
- `docs/plan/system-finalization/technical-completion/01_DATA_REPRESENTATION_COMPLETION_PLAN.md`
- `docs/plan/ml-R4/PLAN_STATUS_MATRIX.md`
- `docs/plan/ml-R4/runs/2026-09-02_PHASE8_v10_v26_physical_acceptance_and_no_launch.md`
- `docs/plan/ml-R4/adrs/ADR-R4-011-v10-v26-physical-representation-acceptance.md`
- `docs/plan/ml-R4/runs/2026-09-02_PHASE8_selector_promotion_review.md`
- `docs/plan/ml-R4/adrs/ADR-R4-012-target-aware-guarded-selector-promotion.md`
- `docs/plan/ml-R4/evidence/2026-09-02_selector_promotion/decision.json`

Executable/research source and tests:

- `ml/src/data_extraction/windowed_tokenizer.py`
- `ml/src/data_extraction/bounded_window_selector.py`
- `data_module/sentinel_data/representation/tokenizer.py`
- `data_module/sentinel_data/representation/r4_orchestrator.py`
- `data_module/sentinel_data/representation/r4_target_spans.py`
- `data_module/sentinel_data/representation/r4_sensitivity.py`
- `data_module/sentinel_data/preprocessing/r4_versions.py`
- `docs/plan/ml-R4/scripts/p8_compare_bounded_window_selector_v1.py`
- `docs/plan/ml-R4/scripts/p8_verify_v10_bound_token_control_equivalence.py`
- `data_module/tests/test_representation/test_bounded_window_selector.py`
- `data_module/tests/test_representation/test_windowed_tokenizer_coverage.py`
- `data_module/tests/test_representation/test_r4_v10_orchestrator.py`
- `ml/src/datasets/vnext_dataset.py`

## 3. R4-D-012 machine boundary

`decision.json` records:

- decision `PROMOTED_FOR_NEW_VERSIONED_CANDIDATE_ONLY`;
- selector `target_aware_guarded_v1`;
- physical candidate build authorized;
- R4-D-011 mutation unauthorized;
- training unauthorized;
- selector implementation SHA-256 `9eea0f837f77a512628efaa3dde444f039be81e98bf1828aa61b9099d2c87866`;
- target-span implementation SHA-256 `7e6f2017e43425285fd51186cd0788ec2ba5f2fe56c90f6dc7b9ef190ee4a910`;
- CPU evidence: 1,018 analyzed, 737 over-cap, 476 improved, 261 control fallbacks, 0 regressed, 0 failed;
- median target coverage 0.6300634455832114 control versus 0.8794466403162056 guarded.

The retained executable source corresponding to the accepted research contract is `ml/src/data_extraction/bounded_window_selector.py`, with target spans supplied by `r4_target_spans.py`. The research source is still present on current `main`; it should be reused rather than semantically reimplemented.

## 4. Current raw-source → token artifact path

### 4.1 Repaired source and graph target

`r4_orchestrator._extract_one()` receives the accepted repaired Solidity file `<preprocessed>/<source>/<sha256>.sol` and repaired metadata.

`_select_targets()` resolves file-graph target names from:

1. a unique explicit target carried by ingestion provenance when present; and
2. repaired preprocessing `contract_names` through `resolve_file_graph_targets()`.

The resulting target tuple becomes sidecar `requested_contract_names` and is the exact target-name input reused by selector research/control-equivalence.

### 4.2 Existing historical token path

`windowed_tokenizer.tokenize_windowed_contract_strict()`:

1. obtains raw GraphCodeBERT code tokens with `add_special_tokens=False`, no truncation;
2. creates overflow windows using `microsoft/graphcodebert-base`, `max_length=512`, padding to 512, `stride=256`, truncation and overflow enabled;
3. selects at most four windows with `_selected_window_indices()`;
4. pads under-cap results to exactly four tensor rows;
5. emits `[4,512]` `input_ids` and `attention_mask` plus coverage metadata.

For repaired preprocessing the orchestrator calls this path with `strip_comments=False`, because repaired source has already passed lexical comment handling.

### 4.3 R4-D-011 V10 path

The accepted V10 path is intentionally different: when `accepted_tokens_dir` is provided, `_extract_one()` loads the accepted token payload, verifies `sha256` and `source`, and byte-copies the `.tokens.pt` file. The V10 sidecar records:

`token_lineage = "accepted_v9_byte_copy"`.

This is immutable R4-D-011 behavior and must not be edited into guarded behavior.

Artifacts are saved as:

- `<sha256>.pt` — graph;
- `<sha256>.tokens.pt` — token payload;
- `<sha256>.rep.json` — sidecar.

## 5. Exact historical control contract

Named research/control identity: `historical_linspace_v1`.

For `total_windows <= count`, select every real window in ascending order.

For over-cap input:

```python
[round(value) for value in np.linspace(0, total_windows - 1, count)]
```

with `count=4` in the frozen architecture. Python `round`/NumPy linspace behavior is therefore part of the historical executable contract.

Current repository-safe control test example:

- `total_windows=20`, `count=4` → `[0, 6, 13, 19]`;
- `total_windows=3`, `count=4` → `[0, 1, 2]`.

The full R4-D-011 control-equivalence verifier dynamically rebuilt `historical_linspace_v1` for all 22,540 accepted identities and compared both tensor payload and selected indices against the accepted token file/sidecar. R4-D-012 relies on that 22,540/22,540 equality result.

## 6. Exact target evidence used by accepted selector research

The accepted CPU comparison uses:

1. the repaired source text;
2. sidecar `requested_contract_names` (missing/empty is an error);
3. `target_contract_char_spans(source_text, requested_contract_names)`;
4. GraphCodeBERT raw token `offset_mapping` over the repaired-v2 comment-stripped, offset-preserving source view;
5. `char_spans_to_token_ranges()` to map each requested declaration-body span to a half-open token range.

`target_contract_char_spans()` requires every requested target name to resolve exactly once and to have a balanced declaration body. Comments and strings are masked while offsets are preserved. A missing/ambiguous/unbalanced target is an error, not evidence for a heuristic fallback.

The selector comparison deliberately uses `prepare_source_for_tokenization()`, which lexically replaces comments with spaces while preserving source length/newline positions. That makes the original-source character spans valid on the tokenized view.

## 7. Exact window geometry

The research selector derives real code-token ranges before special tokens:

- `content_capacity = window_size - tokenizer.num_special_tokens_to_add(pair=False)`;
- default GraphCodeBERT geometry is therefore normally 510 content tokens;
- configured tokenizer stride is 256;
- range step is `content_capacity - stride` (normally 254);
- windows are half-open `[start, end)` ranges and continue until the last range reaches `total_tokens`.

The research path validates that this independently reconstructed range count exactly equals the tokenizer overflow-window count before indexing tensors.

## 8. Exact `target_aware_guarded_v1` selection semantics

### 8.1 Greedy candidate

`target_aware_greedy_indices()` repeatedly chooses the unselected window giving the largest **marginal union coverage gain** over the target token ranges.

Candidate tie-breaking is deterministic:

```python
max(candidates, key=lambda item: (item[0], -item[1]))
```

Therefore equal marginal gain chooses the lowest window index.

Greedy accumulation stops when:

- four windows have been selected;
- no candidates remain; or
- the best marginal gain is `<= 0`.

If fewer than four windows have positive target gain, remaining slots are filled first from the historical linspace indices in historical order, then from remaining indices in ascending order. Duplicate indices are never added. The final selected list is sorted ascending.

If no target ranges are supplied to the low-level greedy helper it returns historical linspace indices. The accepted production-level research path, however, requires non-empty valid target names/spans before invoking the selector.

### 8.2 Guard

Compute target-token union coverage for both:

- historical control indices; and
- greedy candidate indices.

For `target_aware_guarded_v1`:

```text
if greedy_target_coverage_tokens <= control_target_coverage_tokens:
    choose historical control
    used_control_fallback = true
else:
    choose greedy candidate
    used_control_fallback = false
```

So candidate use requires **strictly greater** target-token coverage. Equality deliberately falls back to control. A regression also falls back, although the accepted full CPU evidence observed zero regressions.

No overall-retention threshold participates in the guard. Overall retained-token coverage is telemetry only; R4-D-012 explicitly accepts that median overall retention decreased while target coverage improved.

### 8.3 Under-cap behavior

When real window count is `<= 4`, historical control selects all real windows. The greedy result cannot cover more target tokens than selecting all real windows, so guarded selection resolves to the historical control on equality. Padding then restores the frozen four-row tensor shape.

### 8.4 Over-cap behavior

For `>4` real windows:

- strictly better greedy target coverage → guarded candidate indices;
- equal or lower greedy target coverage → historical control indices.

The accepted CPU population produced 476 strict improvements and 261 control fallbacks among 737 over-cap records.

## 9. Failure versus fallback boundary

This distinction is important for D1/D2:

**Guard fallback** is a valid-evidence decision: valid target spans exist, both selectors are evaluated, and the candidate is not strictly better on target-token coverage.

**Invalid/missing target evidence** is not a guard fallback in the accepted research harness. Missing `requested_contract_names`, declaration ambiguity, unbalanced bodies, zero-token span mapping, tokenizer/range divergence, or tokenizer failure is recorded as a failure.

Production integration must therefore fail closed/produce structured build failure for invalid target evidence rather than silently reporting successful target-aware selection.

## 10. Current selector/coverage metadata

The retained research `SelectionResult` supplies:

- strategy;
- selected indices;
- control indices;
- selected target-coverage token count;
- control target-coverage token count;
- selected retained-token count;
- control retained-token count;
- `used_control_fallback`.

The research tokenizer additionally reports:

- total code tokens;
- total windows;
- target token ranges/count;
- target/control coverage ratios;
- selected/control retained ratios.

The historical token payload/sidecar already records pre-subsampling counts, selected indices, selected code-token ranges, retained unique code tokens/ratio, content tokens per window and tokenizer identity.

D1 may add explicit candidate-lineage metadata but must not alter selector decisions. At minimum the new physical candidate must make requested selector, actually applied selector, fallback state/reason, control/candidate/final indices, target-span identity/evidence, coverage values, tokenizer geometry and parent lineage auditable.

## 11. Consumers and immutable assumptions

### 11.1 R4-D-011 orchestrator/build path

`r4_orchestrator.py` assumes V10 construction receives an accepted V9 token source, byte-copies it, and labels it `accepted_v9_byte_copy`. This is historical-control/accepted-parent behavior and must remain unchanged.

### 11.2 Full-population control-equivalence verifier

`p8_verify_v10_bound_token_control_equivalence.py` consumes:

- D-011 sidecar `requested_contract_names`;
- token-file `selected_window_indices`;
- sidecar `selected_window_indices` when present;
- bound tensors.

It requires dynamic `historical_linspace_v1` indices and tensors to equal the accepted payloads. This remains immutable historical evidence.

### 11.3 Sensitivity/research reporting

`r4_sensitivity.py` and selector research consume pre-subsampling window counts and graph target/component metadata. These are diagnostics/research inputs, not authority to mutate accepted artifacts.

### 11.4 ML dataset adapter

`VNextTrainingDataset` consumes only graph tensors plus token `input_ids`/`attention_mask`; it does not interpret selected-window indices or selector metadata. Therefore a new accepted physical lineage can remain architecture-compatible while still requiring a separate physical binding and later ML authorization.

## 12. Historical-control tests that must remain immutable

- `test_windowed_tokenizer_coverage.py::test_selected_window_indices_preserve_historical_linspace_rule`
- historical strict-tokenizer `[4,512]`/coverage telemetry tests in the same file;
- `test_r4_v10_orchestrator.py::test_v10_extract_copies_accepted_token_bytes` including `token_lineage == "accepted_v9_byte_copy"`;
- the full-population R4-D-011 control-equivalence report/evidence;
- current bounded-selector tests for strict non-regression and tie-to-control semantics.

New guarded tests must be additive. They must not weaken or rewrite these controls.

## 13. D1 versioning/integration decision

D0 finds no semantic contradiction between current source and R4-D-012. D1 can proceed with the following bounded design:

1. Keep `ml/src/data_extraction/bounded_window_selector.py` as the retained algorithm authority instead of copying/rephrasing its selector logic.
2. Keep `windowed_tokenizer.py` historical behavior unchanged.
3. Keep the existing R4-D-011 V10 generation path unchanged.
4. Add a focused fresh-candidate assembly seam that:
   - reads the immutable R4-D-011 graph + sidecar as parent;
   - validates source/contract identity and graph-parent identity;
   - derives target spans from the parent sidecar requested names and accepted repaired source;
   - dynamically produces guarded token payloads using the retained selector implementation;
   - copies parent graph bytes unchanged into a fresh output root;
   - writes new token/sidecar metadata with explicit selector lineage and parent binding.
5. Invalid target evidence fails that identity explicitly; it does not silently select control.
6. Guard fallback remains only the accepted `greedy_target <= control_target` rule.

This seam is preferable to rerunning Slither for a token-only successor because R4-D-012 requires the accepted V2.6 graph parent to remain unchanged and permits differences only in declared selector/token fields.

Candidate physical naming/metadata will be fixed in D1 before full implementation. No D4/D5 generation is authorized by this record alone.

## 14. Rejected interpretations

- **Rejected:** use total retained-code ratio as a second guard. R4-D-012 did not do this.
- **Rejected:** choose greedy on target-coverage equality. Accepted semantics explicitly fall back to control.
- **Rejected:** use a heuristic target when requested names/spans fail. Accepted research treated these as failures.
- **Rejected:** modify `windowed_tokenizer._selected_window_indices()` to become target-aware. That would mutate the historical control.
- **Rejected:** change R4-D-011 `token_lineage` or regenerate its token files.
- **Rejected:** rerun graph extraction merely to construct the token-selector successor when accepted graph bytes can be inherited and verified.

## 15. Current work-package state and next executable step

- **D0:** `DESIGN_READY` — exact executable contract reconstructed; no contradiction found.
- **D1:** ready to implement the explicit candidate-lineage/metadata interface and constants.
- **D2:** may follow D1 using a separate guarded candidate assembly module; existing selector source and R4-D-011 path remain intact.
- **D3:** add focused selector/candidate assembly tests and preserve all historical-control tests.
- **D4/D5:** not yet permitted; bounded implementation/test validation must pass first.

**Next step:** implement D1 metadata/versioning types/constants plus the isolated guarded-token candidate assembler, then add D3 tests for strict-improvement selection, tie fallback, under-cap behavior, deterministic tie handling, malformed target evidence, exact `[4,512]` shape, parent graph byte identity, and sidecar/token lineage isolation.
