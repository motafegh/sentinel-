# DATA D0 — selector contract reconstruction working record

Date: 2026-09-26
Work package: D0 — source/evidence reconstruction
State: AUDITING
Branch: `work/data-d0-selector-contract`
Base: `main@57f39d652cbec1092084b4efbf41bec6a117ba07`

## Exact question

Reconstruct the executable contract required for the R4-D-012
`target_aware_guarded_v1` selector, identify the production integration seam,
and identify historical behavior that must remain immutable before any D1/D2
design or implementation.

This record is an investigation aid only. R4 decisions/ADRs and executable
source/config/tests remain authoritative.

## Governing boundary

- R4-D-011 accepts the exact V10 V2.6 physical root and digest as immutable.
- R4-D-012 promotes `target_aware_guarded_v1` only for a fresh versioned
  candidate.
- Graph schema remains `v10`; extractor remains
  `v2.6-r4-call-semantics-deterministic-cfg-mutators`.
- Token tensor shape remains `[4,512]`.
- Training remains unauthorized.
- Historical selector behavior must remain available as
  `historical_linspace_v1`.

## Evidence established so far

### 1. Current production selector

`ml/src/data_extraction/windowed_tokenizer.py` still uses the historical
selection rule:

- if `total_windows <= max_windows`, select `range(total_windows)`;
- otherwise select
  `[round(i) for i in np.linspace(0, total_windows - 1, max_windows)]`.

The live DATA adapter
`data_module/sentinel_data/representation/tokenizer.py` re-exports that ML
implementation. No production guarded-selector interface exists there today.

### 2. Current representation-build seam

`data_module/sentinel_data/representation/r4_orchestrator.py` is the current
strict graph/token/sidecar construction seam.

For the accepted V10 path it requires an accepted historical token source and
copies the accepted token payload. Its V10 sidecar records
`token_lineage = "accepted_v9_byte_copy"`. Therefore a guarded candidate
cannot be introduced by silently changing this accepted path; it needs a fresh
versioned build mode/lineage.

When tokenization is performed directly, the orchestrator writes:

- `<sha256>.tokens.pt`;
- `<sha256>.rep.json`;
- coverage fields including `selected_window_indices`;
- the frozen `[4,512]` token tensors.

### 3. Retained selector implementation

The retained research implementation is
`ml/src/data_extraction/bounded_window_selector.py`.

Named strategies:

- `historical_linspace_v1`;
- `target_aware_greedy_v1`;
- `target_aware_guarded_v1`.

The retained guarded semantics currently reconstruct as follows:

1. Build historical control indices with the exact linspace rule.
2. Convert requested target-contract character spans to token ranges.
3. Greedily choose windows by maximum **marginal union target-token coverage**.
4. Greedy ties are deterministic: the lowest window index wins because
   candidates are compared as `(gain, -index)`.
5. Stop greedy target selection when no candidate has positive marginal gain.
6. Fill any remaining slots first from historical linspace indices, then from
   remaining indices in ascending order.
7. Sort the final selected indices.
8. For `target_aware_guarded_v1`, use the greedy candidate only when its
   target-token coverage is **strictly greater** than the historical control.
   If coverage is equal or lower, use the historical control.
9. The research tokenizer pads fewer than four real windows to preserve exact
   `[4,512]` tensors.

This establishes the central improvement criterion and fallback rule without
inventing new semantics.

### 4. Target evidence used by selector research

The durable research path reads
`requested_contract_names` from representation sidecars, resolves exact
contract declaration/body character spans with
`r4_target_spans.target_contract_char_spans()`, and maps those spans onto the
GraphCodeBERT token offsets used by selector comparison.

Multi-target file-graph unions are supported; target coverage is union coverage
across all requested target ranges.

### 5. Production-token-view alignment — D0 item 1 CLOSED

The authoritative repaired source path is:

`raw Solidity -> flatten_contract() -> normalize(..., preserve_line_structure=True) -> compile exact normalized bytes -> stage -> content-addressed repaired .sol -> representation/tokenization`.

Source evidence:

- `r4_pipeline._prepare_one()` normalizes `flat.content` with
  `preserve_line_structure=True`.
- `normalizer.normalize()` calls `strip_comments_lexically()`, replaces
  non-newline comment characters with spaces, preserves newlines, strips
  trailing horizontal whitespace, preserves blank-line structure in R4 mode,
  and ensures a terminal newline.
- The SHA used as `normalized_text_sha256` and therefore the repaired artifact
  filename is computed from that exact normalized content.
- The exact normalized content is compiled before promotion.
- The same exact normalized content is written to staging and then copied into
  the repaired physical `<normalized_text_sha256>.sol` artifact.
- `r4_orchestrator._extract_one()` therefore correctly invokes
  `tokenize_windowed_contract_strict(..., strip_comments=False)` for direct
  repaired tokenization. Comment removal has already happened upstream.

The repaired `.sol` bytes, not the original raw or flattened bytes, are thus
the authoritative selector/tokenizer source view.

#### Research selector compatibility

`bounded_window_selector.prepare_source_for_tokenization()` calls
`strip_comments_lexically()` again. On a valid repaired R4 artifact this is
not a newly authorized preprocessing transform: the artifact is already
comment-free except for comment-like marker text protected inside Solidity
strings, which the lexical scanner preserves.

This compatibility call is byte-idempotent for the repaired source contract
established above. The preprocessing regression suite additionally proves
`normalize(..., preserve_line_structure=True)` is idempotent and preserves
comment markers inside strings.

D1/D2 should therefore treat **the persisted repaired `.sol` bytes as the
single authoritative input view**. A production selector interface should not
create an independent second normalization policy. If it retains a defensive
idempotence check, that check must prove byte equality or fail explicitly rather
than silently changing the source view.

No contradiction was found between preprocessing, historical tokenization,
selector research, or R4-D-012.

### 6. Historical-control equivalence is already proven

The full-population verifier
`docs/plan/ml-R4/scripts/p8_verify_v10_bound_token_control_equivalence.py`
dynamically reconstructs `historical_linspace_v1` and compares it with
R4-D-011 bound tensors and indices.

Tracked R4 evidence records:

- 22,540 / 22,540 checked;
- exact `input_ids` match;
- exact `attention_mask` match;
- exact selected-window-index match;
- zero mismatches/failures.

This historical behavior is an immutable control requirement for later D1-D3.

## Historical tests that must remain controls

At minimum preserve the intent of:

- `data_module/tests/test_preprocessing/test_r4_repair.py` for lexical
  normalization, line preservation and repaired-normalization idempotence;
- `data_module/tests/test_representation/test_windowed_tokenizer_coverage.py`
  for exact historical linspace indices and frozen-shape behavior;
- `data_module/tests/test_representation/test_bounded_window_selector.py`
  for target-aware coverage, tie fallback, multi-target spans, and
  offset-preserving token-source semantics;
- the full-population R4-D-011 control-equivalence verifier/evidence.

The older research-script test
`test_bounded_window_strategy.py` is historical evidence and must not become
the production semantic owner if the focused selector module already owns the
same behavior.

## D0 remaining audit, in order

1. **CLOSED** — trace the exact preprocessing source view that reaches
   tokenization and establish comment-removal/idempotence semantics.
2. Trace target-name provenance:
   raw/preprocessing metadata -> `_select_targets()` ->
   `requested_contract_names` -> target spans.
3. Trace representation/version constants and binding utilities to determine
   which lineage fields must change for a fresh guarded candidate.
4. Identify every executable consumer of:
   - `selected_window_indices`;
   - token coverage metadata;
   - token-lineage identity;
   - `requested_contract_names`.
5. Reconcile under-cap behavior and every malformed/missing-target case with the
   accepted R4-D-012 evidence, including the exact structured fallback reason
   that D1 will need to expose.
6. Produce the D0 selector-contract table and integration/versioning boundary.
7. Only if that contract is unambiguous, mark D0 exit as satisfied and move to
   D1. No production code changes before that point.

## Current D0 status

`AUDITING`.

D0 item 1 is closed. No contradiction with R4-D-011/R4-D-012 has been
established. The production gap remains real: historical selection is the live
build behavior, while the guarded selector exists only in retained research
code/evidence and has not been integrated into a fresh physical lineage.

## Next executable step

Audit D0 item 2 only: trace target-name provenance from ingestion/preprocessing
metadata through `_select_targets()`, persisted
`requested_contract_names`, target-span resolution and selector input.
