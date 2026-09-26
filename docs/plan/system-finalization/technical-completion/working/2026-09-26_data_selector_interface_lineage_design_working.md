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

## Current D1 state

D1-1 selector decision semantics: **RECONCILED / READY FOR LINEAGE METADATA
DESIGN**.

No source implementation changed.

## Next D1 increment

Design the canonical artifact metadata/lineage schema:

- fresh representation/token lineage identifier;
- R4-D-011 parent identity;
- selector config identity;
- deterministic target-evidence identity;
- token/sidecar serialization fields;
- successor binding-digest record shape.

Do not implement generation before that schema is fixed.
