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
  target_evidence_status
  target_evidence_detail
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
- `candidate_indices` are present whenever valid target evidence allowed the
  guarded candidate to be computed; they may be absent when target evidence
  could not be established.
- `used_control_fallback` is true exactly when requested guarded policy emits
  historical-control indices.
- indices are deterministic, sorted, unique and in range.
- coverage/retention counts refer to unique pre-special-token code-token
  positions, matching retained selector research semantics.
- a decision object never claims valid target-aware selection when target
  evidence is unavailable.

## D1-1 fallback reason design

Use a small closed vocabulary tied directly to D0 semantics.

Proposed reason codes:

- `under_cap_all_windows`
  - expected;
  - all real windows fit within the four-window cap;
  - historical control emits every real window before padding.

- `no_strict_target_coverage_improvement`
  - expected guarded fallback;
  - valid target evidence exists;
  - candidate target coverage is equal to or lower than control;
  - preserve exact candidate/control coverage counts so equality versus lower
    can be inspected without multiplying reason-code variants.

- `target_evidence_unavailable`
  - exceptional;
  - historical tokenization remains valid but requested target evidence cannot
    be validated/resolved;
  - preserve the original validation failure in
    `target_evidence_detail`;
  - full D4/D5 generation must treat any occurrence as a discrepancy requiring
    review because the R4-D-011 parent proved target-span resolution for
    22,540 / 22,540 identities.

- no fallback reason / null
  - guarded candidate won by strict target-coverage improvement.

Hard source/tokenizer/artifact failures do not produce a
`SelectorDecision`; they fail the build path.

## Target evidence status

Keep fallback reason and evidence validity separate.

Proposed closed status:

- `valid`;
- `unavailable_or_invalid`.

This avoids misleading combinations such as treating equal target coverage as
invalid evidence.

For `valid`, target-span/token-range evidence must be available to metadata
binding.

For `unavailable_or_invalid`, target-aware coverage fields that cannot be
meaningfully computed must remain absent/null rather than fabricated as zero.

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

- stable reason/status codes;
- deterministic target evidence;
- deterministic indices/counts/config;
- a normalized diagnostic detail string only where required for exceptional
  evidence.

The exact canonical serialization helper belongs to the next D1 increment.

## Compatibility boundary

Historical `windowed_tokenizer.py` may continue to expose its existing bare
coverage fields for historical callers.

The new typed decision boundary should live in a focused versioned selector
module/interface. Existing historical tests must continue to exercise the old
path unchanged.

## Current D1 state

D1-1 selector decision semantics: **DESIGNED / READY FOR REVIEW AGAINST
LINEAGE METADATA**.

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
