# DATA D0 — guarded-selector source/evidence reconstruction

**Date:** 2026-09-19  
**Branch:** `agent/data-target-aware-guarded-v1`  
**Base main:** `57f39d652cbec1092084b4efbf41bec6a117ba07`  
**Work package:** D0 — source/evidence reconstruction  
**Status:** D0 COMPLETE / D1 DESIGN READY

## 1. Question

Reconstruct the exact executable contract authorized by R4-D-012 for
`target_aware_guarded_v1`, identify the safe production seam for a fresh
guarded-token physical candidate, and preserve all historical-control behavior
and R4-D-011 artifacts unchanged.

This record is an investigation/design aid. R4 decisions and executable
source/tests remain authoritative.

## 2. Authorities and source inspected

Controlling owners:

- `CLAUDE.md`
- `docs/plan/system-finalization/technical-completion/00_MASTER_TECHNICAL_COMPLETION_ROADMAP.md`
- `docs/plan/system-finalization/technical-completion/01_DATA_REPRESENTATION_COMPLETION_PLAN.md`
- `docs/plan/ml-R4/PLAN_STATUS_MATRIX.md`
- R4-D-011 run record and ADR
- R4-D-012 selector-promotion run record and ADR
- machine records under:
  - `docs/plan/ml-R4/evidence/2026-09-02_selector_promotion/`
  - `docs/plan/ml-R4/evidence/2026-09-02_selector_control_equivalence/`
  - `docs/plan/ml-R4/evidence/2026-08-15_phase8_logical_v3/`

Executable source/tests inspected:

- `ml/src/data_extraction/windowed_tokenizer.py`
- `ml/src/data_extraction/bounded_window_selector.py`
- `data_module/sentinel_data/representation/r4_target_spans.py`
- `data_module/sentinel_data/representation/r4_orchestrator.py`
- `data_module/sentinel_data/preprocessing/r4_versions.py`
- `data_module/sentinel_data/vnext/r4_binding.py`
- `data_module/sentinel_data/vnext/r4_v10_binding.py`
- `data_module/sentinel_data/vnext/representations.py`
- `data_module/sentinel_data/export/token_writer.py`
- Phase-8 selector comparison/control-equivalence scripts
- bounded-selector, tokenizer-coverage, and V10-orchestrator tests
- current ML dataset adapters that consume token payloads

## 3. Evidence identity

The current `bounded_window_selector.py` blob is identical on:

- current `main`;
- hardened evidence source commit
  `83bd566b9c4f4f653e530c2c0f5c990858dd759d`;
- control-equivalence source commit
  `735eda59dd02ab38ee5f14135f64b75a9a3a1111`.

The current `r4_target_spans.py` blob is also identical on all three refs.

Therefore the exact implementation retained in the repository is the accepted
research implementation; selector semantics do not need to be reconstructed
from prose or summary statistics.

R4-D-012 records the selector implementation SHA-256
`9eea0f837f77a512628efaa3dde444f039be81e98bf1828aa61b9099d2c87866`
and target-span implementation SHA-256
`7e6f2017e43425285fd51186cd0788ec2ba5f2fe56c90f6dc7b9ef190ee4a910`.

## 4. Current raw-source -> artifact path

### 4.1 Source and graph target

The repaired source is read from the immutable repaired preprocessing lineage.
`r4_orchestrator._select_targets()` resolves the requested file-graph targets
from explicit ingestion provenance plus the source declaration structure.

The accepted V10 sidecar persists:

- `requested_contract_names`;
- `actual_contract_names`;
- graph component count and graph/runtime provenance.

These requested names are the target-contract input used by the selector
research and by the full-population historical-control equivalence verifier.

### 4.2 Current production tokenization

`windowed_tokenizer.tokenize_windowed_contract_strict()` currently:

1. reads the Solidity source;
2. optionally strips comments lexically;
3. tokenizes the full source without truncation to count raw code tokens;
4. asks GraphCodeBERT for overlapping 512-token overflow windows with
   `stride=256`;
5. selects at most four windows using the historical linspace rule;
6. pads with all-pad windows until the tensor is exactly `[4,512]`;
7. returns `input_ids`, `attention_mask`, selected indices, token ranges,
   and retained-token coverage telemetry.

For repaired preprocessing through `r4_orchestrator`, comment stripping is
disabled at this seam because preprocessing has already performed the lexical
normalization.

### 4.3 R4-D-011 V10 path

The accepted V10 V2.6 builder does **not** retokenize. When
`accepted_tokens_dir` is supplied, it byte-copies the accepted historical token
artifact and writes a V10 graph plus sidecar. The sidecar records
`token_lineage="accepted_v9_byte_copy"`.

R4-D-011 therefore remains an immutable V10 graph parent with historical-control
token bytes.

## 5. Exact selector semantics

The authoritative selector names are:

- control: `historical_linspace_v1`;
- research greedy candidate: `target_aware_greedy_v1`;
- promoted candidate: `target_aware_guarded_v1`.

### 5.1 Source view and target spans

The accepted research path uses an offset-preserving, comment-stripped source
view. Comment bytes are replaced rather than removed, so source length and
newline positions are preserved.

Target character spans are computed from the original/preprocessed source using
the requested contract names. Every requested name must resolve exactly once.
Brace matching ignores comments and strings and requires balanced declaration
bodies.

Character spans are mapped to raw GraphCodeBERT token-index ranges using the
fast tokenizer's offset mapping. A non-empty character span that maps to zero
tokens is an error.

### 5.2 Window ranges

Let:

- `window_size = 512`;
- `special_tokens = tokenizer.num_special_tokens_to_add(pair=False)`
  (fallback 2);
- `content_capacity = window_size - special_tokens`;
- `stride = 256`;
- `step = content_capacity - stride`.

Window ranges are deterministic half-open raw-code-token intervals generated
from token 0 until the final interval reaches the token count.

The research path verifies that this computed range count is exactly equal to
the tokenizer overflow-window count before selecting tensors.

### 5.3 Historical control

If total windows are at or below the requested count, select every real window
in increasing index order.

If over cap, the exact historical rule is:

`[round(x) for x in np.linspace(0, total_windows - 1, count)]`.

For the frozen contract, `count=4`.

This is the behavior already proven equivalent to the accepted R4-D-011 token
population for all 22,540 identities.

### 5.4 Greedy candidate

The greedy candidate repeatedly chooses the not-yet-selected window with the
largest **marginal increase in union target-token coverage**.

Tie breaking is deterministic: maximize `(gain, -window_index)`, therefore the
smallest window index wins equal-gain ties.

Greedy selection stops if:

- the requested count is reached;
- no candidate remains; or
- the best marginal target-coverage gain is non-positive.

It then fills remaining slots first with historical-linspace indices that are
not already selected, then with the earliest still-unselected indices. The
final selected list is sorted and truncated to the requested count.

### 5.5 Guard

For `target_aware_guarded_v1`:

1. compute the historical-control target-token coverage;
2. compute the greedy candidate target-token coverage;
3. use the greedy candidate **only if its target-token coverage is strictly
   greater** than control;
4. if candidate coverage is equal to or below control, use the historical
   control and mark control fallback.

Therefore equality intentionally falls back. No retained-total-token criterion
participates in the guard.

This matches the accepted CPU evidence: 737 over-cap records, 476 improved,
261 equal/control-fallback, zero regressions, zero failures.

### 5.6 Under-cap behavior

For `total_windows <= 4`, historical control selects all real windows. The
greedy path ultimately contains the same complete set, so the guarded comparison
is a target-coverage tie and the effective output is the historical control.
Padding then restores the frozen `[4,512]` tensor shape when fewer than four
real windows exist.

## 6. Invalid/missing target evidence — resolved interpretation

There are two distinct cases and they must not be conflated:

1. **Valid target evidence, candidate does not strictly improve target
   coverage:** use the accepted guarded fallback to historical linspace.
2. **Target evidence cannot be established or validated:** fail closed as a
   structured target-evidence error.

The retained accepted research pipeline does not silently convert malformed,
ambiguous, or zero-token target spans into a clean guarded result:
`target_contract_char_spans()` and `char_spans_to_token_ranges()` raise.

The lower-level selector can return control for an explicitly empty
`target_ranges` list, but the accepted end-to-end research path requires
non-empty valid target spans before selection. Production integration will keep
that fail-closed boundary. This also reconciles the DATA plan's D2 requirement
that malformed/missing target evidence be an explicit fallback/error path rather
than an implicit clean result.

No new selector semantic is introduced by this interpretation.

## 7. Consumers and historical-control boundaries

### Must remain immutable / historical-control specific

- `windowed_tokenizer._selected_window_indices()` and its existing tests prove
  historical linspace behavior.
- R4-D-011 representation artifacts and sidecars remain byte/hash immutable.
- `r4_v10_binding.bind_v10_candidate()` is historical V10 acceptance machinery:
  it explicitly requires `token_lineage="accepted_v9_byte_copy"` and exact
  accepted-V9 token byte equality. It must **not** be loosened for the guarded
  lineage.
- `p8_verify_v10_bound_token_control_equivalence.py` is historical-control
  evidence and compares dynamic control indices/tensors with R4-D-011.
- existing V10 orchestrator tests that prove accepted-token byte-copy behavior
  stay historical-control tests.

### Generic consumers

- `r4_binding._validate_tokens()` binds token coverage fields in payload and
  sidecar and enforces shape/padding; a guarded-lineage binder can reuse the
  generic validation mechanics but needs new selector-aware identity checks.
- ML dataset adapters consume only `input_ids` and `attention_mask` from
  token payloads; they do not depend directly on selected-window indices.
- `token_writer.py` consumes only `input_ids` when creating legacy shards.
- downstream training identity is controlled by manifest/binding digest, not by
  selected-window metadata alone.

## 8. D1 design decision

Use a **fresh focused guarded-token candidate module** rather than modifying the
R4-D-011 V10 generation path.

Rationale:

- R4-D-012 requires a fresh token lineage while preserving accepted V10 graph
  semantics and bytes;
- copying the accepted R4-D-011 graph artifact is stronger and cheaper than
  re-running Slither merely to change token selection;
- the full-population control-equivalence verifier already proves the retained
  research tokenization mechanics reproduce the accepted token population;
- keeping the accepted builder/binder unchanged makes historical-control
  regression easier to prove.

Planned identities:

- selector requested: `target_aware_guarded_v1`;
- rollback/control selector: `historical_linspace_v1`;
- graph schema/extractor: unchanged V10 /
  `v2.6-r4-call-semantics-deterministic-cfg-mutators`;
- token tensor: unchanged `[4,512]`;
- guarded representation lineage: fresh versioned identity, never
  `accepted_v9_byte_copy`;
- fresh candidate root name, distinct from the R4-D-011 root.

The exact lineage/root constants will live in the existing R4 version-owner
module so builders, tests, binders, and later acceptance evidence cannot diverge
by string literal.

## 9. Required selector metadata

The fresh token payload and sidecar must bind enough evidence to answer:

- selector requested;
- selector actually effective;
- whether historical-control fallback occurred;
- explicit fallback reason;
- greedy candidate indices;
- historical-control indices;
- final selected indices;
- requested target contract names;
- target character spans and token ranges;
- target-token count and candidate/control/final coverage;
- pre-subsampling token/window counts;
- final retained-token coverage;
- tokenizer/window/stride configuration;
- graph-parent identity and graph hash;
- fresh representation/token lineage identity.

Malformed target evidence must never be labeled
`target_aware_guarded_v1` success.

## 10. D0 exit decision

D0 is complete:

- accepted selector semantics are reconstructable from unchanged executable
  source;
- historical-control behavior and immutable consumers are identified;
- the target-evidence error/fallback distinction is explicit;
- a safe fresh-lineage integration seam is identified;
- no contradiction was found between current source and R4-D-012 evidence.

## 11. Next executable step

Proceed with D1/D2 on this branch:

1. add fresh selector/lineage constants;
2. implement a focused guarded-token candidate builder that copies the exact
   accepted V10 graph, derives target evidence from the accepted requested
   contract names, dynamically creates guarded token tensors, and writes
   selector-aware token/sidecar metadata;
3. leave R4-D-011 builder/binder/control verifier unchanged;
4. add focused unit/property tests before any full-population generation.
