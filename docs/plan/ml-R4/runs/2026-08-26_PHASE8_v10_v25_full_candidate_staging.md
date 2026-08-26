# Phase-8 V10 V2.5 full-candidate staging protocol

Date: 2026-08-26
Status: READY FOR PROTECTED-LOCAL VALIDATION
Scope: R4-B008 V10 physical candidate construction only; no physical acceptance or training authority

## Starting authority

The bounded structural tranche is closed by
`reviews/R4-GAP-008/2026-08-26_v10_v25_bounded_structural_closure.md`:
20 / 20 previously unexpected identities are resolved as 8 exact node-index
labelled-multigraph equivalence cases and 12 deterministic persistent-storage
WRITE corrections, with zero unexplained drift.

The full-gate code and evidence chain have also passed protected-local validation:

- `p8_audit_v10_transition_v3.py` and its dependencies compile;
- focused full-gate tests pass 12 / 12;
- `p8_validate_v10_v25_evidence_chain.py` returns pass;
- original transition-audit SHA-256:
  `5793b059e7e5149424e10a5361a5b0e420b1f86f3630920e36344c5737fd4f9b`;
- bounded V2.5 report SHA-256:
  `cffcb74c531df47a211d2960772de8430fc2eff662ee991a617c29fa1dfe3a38`;
- merged semantic evidence SHA-256:
  `483012e384661ae015c39f42c686ead982d9fc016c8f80f386de8ca70dbc654b`.

Physical acceptance and training authorization remain false.

## Why full generation must be staged

The required runtime distribution is intentionally heterogeneous:

- every ordinary identity must be generated under exact Slither 0.10.0;
- the identity declared in `V10_SLITHER_RUNTIME_EXCEPTIONS` must be generated
  under its exact identity-bound Slither 0.11.5 runtime.

`p8_generate_v10_candidate.py --mode full` executes in one Python environment.
It therefore cannot by itself produce the final required 22,539 + 1 runtime
split. The binder correctly rejects a runtime-exception identity produced under
the primary runtime.

The full build must not work around that invariant by relabelling sidecars,
mutating graph payload version fields, or upgrading the population to 0.11.5.

## Approved staged construction

### Stage A — primary attempt

Run the ordinary full generator in a fresh disposable/protected attempt root
under `ml/.venv/bin/python`, exact Slither 0.10.0, with
`PYTHONPATH=.:data_module`.

This attempt is expected to be incomplete only at the declared runtime-exception
identity. It is not itself a candidate and its nonzero full-generation result is
not treated as acceptance evidence.

### Stage B — fail-closed primary staging

Run `p8_stage_v10_v25_primary_attempt.py` with the failed/incomplete primary
attempt and a second fresh candidate root.

The staging tool must refuse transfer unless all of the following hold:

1. accepted-V9 population resolves every declared runtime exception uniquely;
2. primary-attempt sidecar inventory equals accepted V9 minus exactly those
   exception identities;
3. structured failure records contain exactly those exception identities and no
   others;
4. every transferable sidecar is graph schema V10 and extractor
   `v2.5-r4-call-semantics-deterministic-cfg`;
5. every transferable artifact reports full non-degraded analysis, zero
   unclassified call IR, zero call-mapping errors, and exact classified/emitted
   call-count equality;
6. every transferable artifact is bound to Slither 0.10.0 with runtime role
   `primary` and required physical runtime 0.10.0;
7. every candidate token file is byte-identical to its accepted-V9 token file;
8. the output root is fresh.

Only validated graph/token/sidecar triples are hardlinked or copied. Primary
attempt failure/manifests are not carried into the final candidate root. A
root-level primary-stage report records the attempt provenance and the exact
missing exception identities.

### Stage C — identity-bound exception fill

Use regression generation against the staged candidate root under the exact
identity-bound runtime (currently Slither 0.11.5) for exactly the declared
exception identity. The destination triple must not already exist. Save the
regression report in the final lineage.

No ordinary identity may be generated under 0.11.5.

### Stage D — complete candidate binding

Run `p8_generate_v10_candidate.py --mode bind` on the filled root.

Binding must prove:

- exactly 22,540 candidate identities;
- exact accepted-V9 population equality;
- graph schema V10 / extractor V2.5 on every sidecar and graph payload;
- exact accepted-V9 token bytes for every identity;
- zero degraded parse-only analysis;
- zero unclassified call IR and call mapping errors;
- classified/emitted equality;
- exact runtime distribution required by `V10_SLITHER_RUNTIME_EXCEPTIONS`.

### Stage E — complete V3 transition audit

Run the evidence-chain preflight and then
`p8_audit_v10_transition_v3.py` against:

- accepted V9;
- the same immutable frozen V2.3 structural reference;
- the freshly bound V2.5 candidate;
- the passed bounded V2.5 report;
- the exact merged semantic WRITE evidence.

V3 must re-prove the 8 index-equivalence identities and the 12 WRITE identities
against the actual full candidate and must reject every additional non-parse-only
structural difference.

## Stop lines

- Do not mutate or overwrite the existing protected V2.4 canonical candidate
  while building this lineage.
- Do not use a population-wide Slither 0.11.5 generation.
- Do not manually edit runtime/version fields in artifacts.
- Do not treat the expected primary-stage exception as a waived failure; it must
  be isolated, recorded, filled under its required runtime, and then proven by
  the ordinary binder.
- Do not declare physical acceptance from a passing binder or V3 report alone;
  explicit report review and a physical-acceptance decision record remain
  required.
- Do not authorize training. Training authority is a separate later gate.
