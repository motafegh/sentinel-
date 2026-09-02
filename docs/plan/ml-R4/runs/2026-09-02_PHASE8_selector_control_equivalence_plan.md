# Phase-8 full-population selector control-equivalence plan

Date: 2026-09-02
Status: PASS; selector-promotion decision still required
Scope: R4-B006 prerequisite evidence only; no selector promotion or training

## Objective

Prove over the exact R4-D-011 population that the research selector's
`historical_linspace_v1` control path reproduces the token tensors already bound
to the accepted V2.6 physical representation. This isolates any later guarded
selector comparison from tokenization, source, population, or lineage drift.

## Inputs

- R4-D-011 acceptance manifest and exact protected-local V2.6 root;
- accepted repaired-v2 preprocessed sources;
- each V2.6 sidecar's exact requested contract names;
- locally cached `microsoft/graphcodebert-base` fast tokenizer;
- current `bounded_window_selector.py` control implementation.

## Execution

Use a read-only multiprocessing verifier. Each worker loads the tokenizer once,
then for each assigned identity:

1. reads the accepted preprocessed source and V2.6 sidecar;
2. resolves the requested-contract character spans;
3. dynamically tokenizes with `historical_linspace_v1`;
4. compares `input_ids`, `attention_mask`, and selected window indices with the
   accepted bound token payload;
5. emits only compact identity/digest evidence, never replacement artifacts.

The final report must bind the source commit, acceptance-manifest SHA-256,
physical root, R4-D-011 binding digest, tokenizer identity, worker count, and
complete deterministic population digest.

## Exit gate

Pass only if:

- exactly 22,540 identities are enumerated and checked;
- every source, sidecar, and token payload exists and is readable;
- every dynamically produced control tensor is byte-equal to its bound tensor;
- every selected-window index list matches;
- failures and mismatches are zero.

A pass is prerequisite evidence only. It does not promote
`target_aware_guarded_v1`, prove that candidate's quality, authorize objective
changes, or authorize training. Any promotion still requires a separate
evidence review and recorded decision.

## Stop lines

- Do not edit, regenerate, or overwrite the R4-D-011 root.
- Do not download or substitute a tokenizer.
- Do not use Run12 weights or run model training.
- Fail closed on population, lineage, tokenizer, tensor, or metadata drift.

## Result

The full eight-worker run completed against published source commit
`735eda59dd02ab38ee5f14135f64b75a9a3a1111`:

- 22,540 expected, enumerated, and checked identities;
- 22,540 exact `input_ids`, `attention_mask`, and selected-index matches;
- zero missing inputs, failures, or mismatches;
- deterministic population-result digest
  `4cddfe353afb5ea6cbfc2f5f18e9f4b1e774242132bda5e87823c6a8cdaf57ec`;
- local report SHA-256
  `636838f376d8991e9ac07d26105aa2f907e535bbf90e4504e11d663f0c656021`.

The compact tracked snapshot is
`evidence/2026-09-02_selector_control_equivalence/report.json`. This closes the
control-equivalence prerequisite, not R4-B006 itself. Guarded-selector promotion
still requires source-first review and a separate decision; training remains
unauthorized.
