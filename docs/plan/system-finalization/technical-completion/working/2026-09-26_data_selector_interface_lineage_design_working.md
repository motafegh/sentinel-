# DATA D1 — selector interface and lineage design working record

Date: 2026-09-26
Work package: D1 — selector interface and lineage design
State: DESIGNING
Branch: `work/data-d0-selector-contract`
D0 authority record:
`2026-09-26_data_selector_contract_reconstruction_working.md`

## Scope for this increment

Define only the typed selector decision/result boundary and structured fallback
semantics. Do not implement token generation or modify the accepted R4-D-011
builder/binder in this increment.

## D1-1 decision — explicit selector decision object

The production selector boundary should return one immutable structured result
rather than a bare list of indices.

Recommended semantic shape:

```text
SelectorDecision
  requested_strategy
  effective_strategy
  selected_indices
  control_indices
  candidate_indices
  used_control_fallback
  fallback_reason
  total_windows
  max_windows
  target_coverage_tokens
  control_target_coverage_tokens
  candidate_target_coverage_tokens
  retained_tokens
  control_retained_tokens
  candidate_retained_tokens
```

This is a semantic schema, not yet a Python implementation/API commitment.

### Required invariants

- `requested_strategy` is the policy requested by the build, e.g.
  `target_aware_guarded_v1`.
- `effective_strategy` identifies which policy actually produced the emitted
  token windows:
  - guarded strategy when strict target-coverage improvement wins;
  - `historical_linspace_v1` whenever control is emitted.
- `selected_indices` are the emitted real-window indices.
- `control_indices` are always present.
- valid target evidence is a precondition for producing a `SelectorDecision`;
  therefore `candidate_indices` are always present on a successful decision.
- `used_control_fallback` is true exactly when requested guarded policy emits
  historical-control indices.
- indices are deterministic, sorted, unique and in range.
- coverage/retention counts refer to unique pre-special-token code-token
  positions, matching retained selector research semantics.
- a decision object never claims valid target-aware selection when target
  evidence is unavailable.

## D1-1 fallback reason design

Use a small closed vocabulary tied directly to D0 semantics.

The successful decision vocabulary has exactly one fallback reason:

- `candidate_target_coverage_not_strictly_greater`
  - valid target evidence exists;
  - candidate target coverage is equal to or lower than control;
  - historical control is emitted;
  - exact candidate/control coverage counts remain bound so equality versus
    lower coverage is inspectable.

- no fallback reason / null
  - guarded candidate won by strict target-coverage improvement.

Under-cap is **not** a separate fallback policy. When `total_windows <=
max_windows`, candidate and control contain all real windows, their target
coverage ties, and the ordinary strict-improvement guard therefore emits the
historical control. `total_windows` and `max_windows` provide descriptive
under-cap telemetry without multiplying policy reason codes.

Missing, malformed or unresolvable target evidence does **not** produce a
`SelectorDecision`. The durable research/control-equivalence paths reject such
inputs before selection, and R4-D-012 does not authorize silent substitution of
historical control. Those conditions belong to a separate structured build
failure/error record and no successful guarded token artifact is emitted.

Hard source/tokenizer/artifact failures likewise fail the build path before a
decision object exists.

## Why requested and effective strategy are separate

R4-D-012 authorizes `target_aware_guarded_v1` as the requested policy, but the
guard explicitly emits the historical control for some contracts. Recording
only `strategy=target_aware_guarded_v1` would hide which policy actually
produced the tensor.

The pair:

```text
requested_strategy = target_aware_guarded_v1
effective_strategy = historical_linspace_v1
```

plus the fallback reason makes artifact provenance explicit without changing
the accepted semantics.

## Serialization boundary

D1 should use one canonical mapping for both token payload and sidecar selector
metadata. The same decision fields must round-trip identically through both
artifacts and be validated by the successor binder.

Do not allow separate ad-hoc dictionaries in token and sidecar writers.

Large/raw diagnostics such as Python exceptions should not become unstable
binding inputs. The bound form should use:

- the stable guarded fallback reason;
- deterministic target evidence;
- deterministic indices/counts/config.

Selection/build failures use a separate failure record and are not serialized as
successful selector metadata.

The exact canonical serialization helper belongs to the next D1 increment.

## Compatibility boundary

Historical `windowed_tokenizer.py` may continue to expose its existing bare
coverage fields for historical callers.

The new typed decision boundary should live in a focused versioned selector
module/interface. Existing historical tests must continue to exercise the old
path unchanged.



## D1-2 decision — canonical selector, target-evidence and lineage metadata

The guarded candidate should preserve the existing top-level token coverage
fields for compatibility and add one canonical nested selector record that is
serialized identically into both the token payload and representation sidecar.

### Versioned identities

Freeze the following design identities for the guarded candidate:

- selector decision schema:
  `sentinel-r4-token-selector-decision-v1`;
- target-evidence schema:
  `sentinel-r4-selector-target-evidence-v1`;
- token lineage ID:
  `r4-v10-v26-target-aware-guarded-v1`;
- protected candidate root name:
  `representations-r4-v10-v26-target-aware-guarded-v1-candidate`;
- successor binding-report schema:
  `sentinel-r4-v10-guarded-candidate-binding-v1`.

These are new identities. Existing R4-D-011 constants and root names remain
unchanged.

### Parent-lineage binding

Every guarded candidate sidecar must identify the immutable physical parent by
authority, not by a machine-specific absolute path:

```text
graph_parent
  decision_id = R4-D-011
  acceptance_schema = sentinel-r4-v10-v26-physical-acceptance-v1
  binding_digest_sha256 = d9f925588913e66476cfbc097bace7daa7e673295fe2a243760313d0bef5ebdd
  graph_schema_version = v10
  extractor_version = v2.6-r4-call-semantics-deterministic-cfg-mutators
```

The successor binder must additionally hash-compare every candidate graph file
against the corresponding R4-D-011 graph file. The parent binding digest is the
governance anchor; per-identity graph-byte equality is the mechanical proof.

### Selector configuration identity

Selector semantics that affect physical token bytes must be represented by one
canonical configuration mapping and SHA-256 digest.

Canonical semantic fields:

```text
selector_config
  requested_strategy = target_aware_guarded_v1
  candidate_strategy = target_aware_greedy_v1
  control_strategy = historical_linspace_v1
  tokenizer_name = microsoft/graphcodebert-base
  window_size = 512
  stride = 256
  max_windows = 4
  target_metric = union_requested_target_token_coverage_v1
  guard = candidate_target_coverage_strictly_greater_v1
  greedy_tie_break = lowest_window_index_v1
  fill_policy = historical_control_then_ascending_v1
  source_view = repaired_preprocessed_bytes_v1
```

The mapping is serialized as canonical JSON
(`sort_keys=True`, separators `,` and `:`) and hashed to
`selector_config_sha256`.

Runtime-derived values such as `content_tokens_per_window` remain per-artifact
telemetry and must agree with the frozen tokenizer/window contract, but they do
not create an independent selector policy name.

### Canonical target-evidence record

For each successful selector decision, target evidence must be deterministic and
bound rather than inferred later:

```text
target_evidence
  schema
  contract_id
  requested_contract_names
  target_char_spans
  target_token_ranges
  target_tokens
  selector_config_sha256
  sha256
```

`sha256` is the canonical digest of the record excluding its own digest field.

The requested names and spans retain their deterministic source/declaration
order. The successor binder must verify that the sidecar
`requested_contract_names` equals the target-evidence names exactly.

Invalid target evidence produces a structured build failure and therefore no
successful `target_evidence` or selector decision artifact.

### Canonical selector decision record

Both `*.tokens.pt` and `*.rep.json` must contain the same mapping under
`token_selector`:

```text
token_selector
  schema = sentinel-r4-token-selector-decision-v1
  requested_strategy
  effective_strategy
  selector_config_sha256
  selected_indices
  control_indices
  candidate_indices
  used_control_fallback
  fallback_reason
  total_windows
  max_windows
  target_evidence_sha256
  target_coverage_tokens
  control_target_coverage_tokens
  candidate_target_coverage_tokens
  retained_tokens
  control_retained_tokens
  candidate_retained_tokens
```

Invariants:

- `requested_strategy == target_aware_guarded_v1`;
- `effective_strategy` is either `target_aware_guarded_v1` or
  `historical_linspace_v1`;
- `selected_indices` equals `candidate_indices` exactly when the strict
  target-coverage guard succeeds;
- otherwise `selected_indices == control_indices`,
  `used_control_fallback == true`, and
  `fallback_reason == candidate_target_coverage_not_strictly_greater`;
- `candidate_indices` and candidate coverage are always present on a
  successful decision;
- all index arrays are sorted, unique and in range;
- `target_evidence_sha256` binds the exact names/spans/token ranges used by the
  decision.

### Existing top-level coverage compatibility

Do **not** replace the established `r4-token-coverage-v1` fields. Their meaning
does not change: they describe the **emitted** token windows.

Continue to persist at top level:

- `coverage_schema_version`;
- `pre_subsampling_window_count`;
- `pre_subsampling_code_tokens`;
- `selected_window_indices`;
- `selected_code_token_ranges`;
- `retained_unique_code_tokens`;
- `retained_token_ratio`;
- `content_tokens_per_window`;
- `coverage_interpretation`.

For the guarded lineage:

- top-level `selected_window_indices` must equal
  `token_selector.selected_indices`;
- top-level retained coverage must describe those same emitted indices;
- the token payload and sidecar must still agree on all existing coverage
  fields.

This keeps existing coverage/sensitivity tooling useful without pretending it
knows the new selector semantics.

### Token-lineage fields

The new token payload and sidecar should both carry:

```text
token_lineage = r4-v10-v26-target-aware-guarded-v1
token_lineage_parent_decision = R4-D-011
token_lineage_parent_binding_digest_sha256 = d9f925...
selector_config_sha256 = <canonical selector config digest>
token_selector = <canonical selector decision mapping>
target_evidence = <canonical target-evidence mapping>
```

The sidecar additionally carries the `graph_parent` authority mapping because
it binds the complete graph/token/sidecar representation triple.

### Successor binding digest record

The guarded binder must build its population digest from sorted logical records,
not local paths. Each record must bind at least:

```text
source
contract_id
graph_sha256
parent_graph_sha256
tokens_sha256
sidecar_sha256
graph_schema_version
extractor_version
token_lineage
parent_decision_id
parent_binding_digest_sha256
selector_config_sha256
target_evidence_sha256
requested_strategy
effective_strategy
used_control_fallback
fallback_reason
selected_indices
control_indices
candidate_indices
```

Before admitting a record, the binder must prove
`graph_sha256 == parent_graph_sha256`. The duplicate values remain in the
record intentionally: they make the parent comparison explicit and auditable in
the digest input.

The binding report may aggregate counts such as guarded-selected versus
control-fallback and under-cap versus over-cap, but aggregate counts never
replace per-identity records.

### Acceptance boundary

The successor binding report remains diagnostic:

- `physical_acceptance = false`;
- `training_authorized = false`.

A passing digest is evidence for D6 review, not authority to accept the lineage
or launch training.

## D1-2 status

Canonical artifact/lineage metadata: **DESIGNED**.

No production source or artifact-generation path changed in this increment.


## Current D1 state

D1-1 selector decision semantics: **RECONCILED**.

D1-2 canonical artifact/lineage metadata: **DESIGNED**.

No source implementation changed.

## Next D1 increment

Review the proposed interface/metadata against the exact D1 exit criterion and
define the focused module/API ownership:

- which existing research logic is promoted/refactored versus wrapped;
- where the validated target-evidence boundary lives;
- where canonical serialization/digest helpers live;
- where the successor binder lives;
- which historical modules remain untouched.

Only after that ownership/API review passes should D1 close and D2
implementation begin.
