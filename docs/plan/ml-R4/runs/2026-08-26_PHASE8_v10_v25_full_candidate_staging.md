# Phase-8 V10 V2.5 full-candidate staging protocol

Date: 2026-08-26
Status: READY FOR PROTECTED-LOCAL PRIMARY ATTEMPT AFTER DETERMINISTIC HARDENING
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
- current deterministic bounded V2.5 report SHA-256:
  `67192b2a81383af74f70ed3ed6e1c0dfbd50d6b9525a9a939a250653e2a53adc`;
- current deterministic merged semantic evidence SHA-256:
  `16e264fbed941ab16ead47dacd4e19c7a02511539e0950664e2cdc28373bfa8e`.

The protected-local staging/runtime preflight also passed on 2026-08-27:

- primary environment: Slither 0.10.0 / crytic-compile 0.3.11;
- identity-bound exception environment: Slither 0.11.5 / crytic-compile 0.3.11;
- the declared exception resolves to exactly one repaired-preprocessing identity;
- the protected V2.4 candidate and frozen V2.3 structural-reference roots are present;
- staging-tool focused tests pass.

The final Stage-A driver/staging readiness pass also succeeded on 2026-08-27:

- `p8_generate_v10_v25_primary_attempt.py` compiles;
- primary-attempt + staging focused tests pass **9 / 9**;
- accepted V9 population = 22,540;
- repaired-preprocessed population = 22,540;
- ordinary primary partition = 22,539;
- declared runtime-exception partition = 1;
- deferred identity = `dive/caa35c1a5906269bbe5e70de780d105c2968ece4fc038d7f7208efee681aeec9`;
- population partition preflight = PASS.

**Stage A itself has not yet executed.** No fresh 22,539-artifact V2.5 primary attempt may be claimed until the Stage-A report exists and passes.

Physical acceptance and training authorization remain false.

## 2026-08-27 pre-execution hardening addendum

A fresh restart audit found that the bounded V2.5 report was retained only as
an untracked protected-local file while its exact merged semantic WRITE input
was no longer present. Re-running the semantic probe also exposed byte-level
nondeterminism in informational node fields and list ordering. Before Stage A:

1. make semantic WRITE evidence aggregation deterministic and cover it with a
   focused regression;
2. regenerate three bounded V2.5 repeats and a newly SHA-bound bounded report;
3. persist the complete semantic evidence and bounded report outside `/tmp`;
4. rerun the evidence-chain preflight successfully;
5. require Stage B to accept only exact `IdentityBoundRuntimeDeferred` records
   for the declared source/identity/runtime set.

This addendum is complete. The semantic-evidence probe now canonicalizes record
ordering and deterministically aggregates duplicate Slither views while using
expression-level roots rather than unstable propagated state-write aliases.
Two independent randomized-process runs produced byte-identical base and
expansion evidence. Stage B now rejects any failure row that is not an exact
source/identity/runtime-bound `IdentityBoundRuntimeDeferred` record.

The replacement protected-local evidence chain is persistent under
`data_module/data/r4-v10-v25-evidence-deterministic-v2/`:

- semantic base SHA-256:
  `00503d3e2823513e88303cce17db5b23c777f8b7ee3b59a7bb8901ce1d3a6d4e`;
- semantic expansion SHA-256:
  `49ed42dea79250a95261949ca717b031ee6e8970329c288f7b0d481d6e22fc49`;
- merged semantic evidence SHA-256:
  `16e264fbed941ab16ead47dacd4e19c7a02511539e0950664e2cdc28373bfa8e`;
- bounded V2.5 report SHA-256:
  `67192b2a81383af74f70ed3ed6e1c0dfbd50d6b9525a9a939a250653e2a53adc`;
- evidence-chain preflight SHA-256:
  `1d28f9b2f4a597ff04f62052cad95713dafd6169f5d0f97de100fde452e542cb`.

Three new persistent 20-identity repeats all passed under Slither 0.10.0. The
replacement bounded report again proves exactly 8 node-index equivalence cases
plus 12 deterministic storage-WRITE corrections, zero unexplained drift, and no
blockers. Focused hardening/evidence tests pass 30/30. This remains evidence
reproducibility hardening, not a change to V10 semantics, physical acceptance,
or training authority.

## 2026-08-30 Stage-A interruption and fail-closed resume addendum

The first population-wide Stage-A execution was cleanly stopped before host
shutdown.  Its protected attempt root contains 13,974 apparent
graph/token/sidecar file triples, no incomplete file set, and no final report.
That inventory is interrupted build material only; it is not a passed Stage-A
attempt and it grants no acceptance or training authority.

Because the Stage-A artifacts are identity-isolated and the sidecar is written
only after the graph and accepted-V9 token copy, an opt-in fail-closed resume is
permitted for this non-canonical attempt root under these constraints:

1. the default remains fresh-root-only; a nonempty root requires explicit
   `--resume`;
2. the accepted-V9 and repaired-preprocessing populations, exact primary
   runtime, extractor, schema, and exception partition are revalidated before
   reuse;
3. every existing artifact identity must belong to the ordinary primary
   partition; extra identities and the declared exception are rejected;
4. every complete existing triple must pass the same sidecar/runtime/call/token
   checks as primary staging plus graph-payload and token-payload loading and
   schema/identity checks; any complete-but-invalid triple aborts;
5. an incomplete file set is moved to a sibling quarantine root before that
   identity is regenerated, so interrupted bytes are retained rather than
   silently overwritten;
6. only absent or quarantined-incomplete identities are sent to workers;
7. source manifests and the final Stage-A report are recomputed over the full
   ordinary population and record reused/generated/quarantined counts.

The resumed attempt must still produce the ordinary 22,539-identity inventory,
the exact structured deferred exception row, zero unexpected failures, and a
passing Stage-A report before Stage B is allowed.

The first resume exposed a process-start safety issue after successfully
validating and extending the root to 15,211 complete triples.  Payload
validation initializes PyTorch worker threads before the representation pool is
created; Linux's default `fork` context then inherited locked thread state and
all four children stopped on futex waits with no compiler descendants.  The
service was stopped with zero incomplete file sets.  Resumed multiprocessing
must therefore use Python's clean `spawn` context, never post-validation
`fork`, and the pool-start contract is covered by a focused regression.

## Why full generation must be staged

The required runtime distribution is intentionally heterogeneous:

- every ordinary identity must be generated under exact Slither 0.10.0;
- the identity declared in `V10_SLITHER_RUNTIME_EXCEPTIONS` must be generated
  under its exact identity-bound Slither 0.11.5 runtime.

`p8_generate_v10_candidate.py --mode full` executes in one Python environment.
It therefore cannot by itself produce the final required 22,539 + 1 runtime
split. In addition, the V10 compatibility layer may emit parse-only diagnostic
output after a full-analysis failure, so a primary-runtime attempt must not rely
on the exception naturally failing before artifact emission.

The full build must not work around that invariant by relabelling sidecars,
mutating graph payload version fields, or upgrading the population to 0.11.5.

## Approved staged construction

### Stage A — explicit primary attempt

Run `p8_generate_v10_v25_primary_attempt.py` in a fresh protected attempt root
under `ml/.venv/bin/python`, exact Slither 0.10.0, with
`PYTHONPATH=.:data_module`.

The Stage-A driver first requires exact accepted-V9 / repaired-preprocessing
population equality. It then partitions that complete population using
`V10_SLITHER_RUNTIME_EXCEPTIONS`:

- every ordinary identity is sent through the normal V10 V2.5 extraction path;
- every declared runtime-exception identity is **not invoked at all** in the
  primary process and is recorded as a structured
  `IdentityBoundRuntimeDeferred` failure row.

The attempt passes only when all ordinary identities are written successfully,
no unexpected ordinary failure exists, the physical sidecar inventory equals
accepted V9 minus exactly the declared exception set, and every declared
exception is represented by its deferred-runtime record.

This root is an attempt, not a bound candidate. No exception triple may be
present in it.

### Stage B — fail-closed primary staging

Run `p8_stage_v10_v25_primary_attempt.py` with the Stage-A attempt and a second
fresh candidate root.

The staging tool must refuse transfer unless all of the following hold:

1. accepted-V9 population resolves every declared runtime exception uniquely;
2. primary-attempt sidecar inventory equals accepted V9 minus exactly those
   exception identities;
3. structured failure records contain exactly those source/exception identities
   and no others, use `IdentityBoundRuntimeDeferred`, and name the exact required
   identity-bound Slither runtime;
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
- Do not invoke a declared identity-bound exception in the Stage-A primary
  extraction process; it must remain physically absent until Stage C.
- Do not treat the deferred exception as a waived failure; it must be isolated,
  recorded, filled under its required runtime, and then proven by the ordinary
  binder.
- Do not declare physical acceptance from a passing binder or V3 report alone;
  explicit report review and a physical-acceptance decision record remain
  required.
- Do not authorize training. Training authority is a separate later gate.
