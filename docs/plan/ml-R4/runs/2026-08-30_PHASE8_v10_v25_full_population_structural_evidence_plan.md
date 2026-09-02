# Phase-8 V10 V2.5 full-population structural-evidence plan

Date: 2026-08-30
Status: COMPLETE; exact V2.6 root accepted by R4-D-011 on 2026-09-02
Scope: R4-B008 full-population structural evidence; no selector promotion or training authority

## 2026-09-01 V2.6 continuation

The reviewed next version recognizes only Solidity collection `push` and `pop`
calls whose receiver resolves to persistent storage. It preserves call-node
priority and does not promote memory collections or arbitrary member calls.
The extractor version is
`v2.6-r4-call-semantics-deterministic-cfg-mutators`.

Fresh Stages A-D pass:

- 22,539/22,539 primary Slither-0.10 artifacts with zero unexpected failures;
- the single declared runtime exception filled under Slither 0.11.5;
- 22,540/22,540 accepted-V9 population equality and token-byte identity;
- binding digest
  `d9f925588913e66476cfbc097bace7daa7e673295fe2a243760313d0bef5ebdd`.

The initial V4 audit correctly found that the V2.6 population was not the old
311-case V2.5 population. The exact set changed by +52/-8 to 355 identities, so
the old proof was not reused as a waiver. Three fresh 355-case generations and
three semantic-evidence passes produced a stable current proof:

- 349 persistent-storage WRITE corrections;
- 3,517/3,517 target groups proven, covering 6,247 occurrences;
- 2,532 duplicate target groups handled with explicit multiplicity;
- 6 exact node-index-invariant equivalence identities;
- zero blockers and zero unexplained drift.

The final V4 audit passes all 22,540 transition mechanics and re-proves the
355-case evidence against the actual full candidate. Its report is
`data_module/data/v10-v26-full-candidate-attempt-2026-09-01-a/v10-transition-audit-v4.json`,
SHA-256
`c6ddc61b8005a688d422f4f8de28118fa3e644b9648d070ef53972ec9f2191ce`.

This completes the structural evidence tranche. The later refreshed binder and
current-commit V4 review support R4-D-011 / ADR-R4-011, which physically accepts
only the exact root and digest above. That decision grants no selector promotion
or training authority.

## Historical V2.5 starting point

The protected-local V2.5 candidate is fully constructed and mechanically
bound. Stages A-D pass for 22,540 identities, with exact accepted-V9 token
bytes, zero missing/extra/invalid artifacts, and the required 22,539 primary
Slither-0.10 + one identity-bound Slither-0.11.5 runtime split. The Stage-D
binding digest is
`17c5f334c75015fdaf89b1a9f77522af5185f2485c24df4e1e64917dc944f021`.

Stage E correctly fails closed with status
`PASS_BASE_MECHANICS_WITH_STRUCTURAL_EVIDENCE_BLOCKER`. The full population has
311 raw non-parse-only structural-drift identities. The audit re-proves the
exact old bounded classes (8 node-index equivalence and 12 storage-WRITE
corrections), but 298 identities remain outside those approved classes.
Physical acceptance and training authorization are false.

The diagnostic full-population probe classifies the 311 identities as:

- 298 `FEATURE_OR_METADATA_CLASSIFICATION_DRIFT`;
- 12 `NODE_ORDER_INDEX_NONDETERMINISM_PROVEN`;
- 1 `SEMANTIC_STRUCTURE_DRIFT`:
  `dive/bfa512a7a831999fa8140cd667e84524d3e01b09fb3cb258955f09b680863d62`.

It finds 895 uniquely identifiable semantic node differences across 183
contracts. Another 128 contracts have no uniquely matched node difference
under the first diagnostic because duplicate semantic node identities make
one-to-one matching ambiguous. This limitation is evidence missing, not proof
of equivalence.

## Why the old bounded closure cannot authorize this population

The 20-identity closure proved facts about exactly those 20 identities under
three exact-runtime repeats. It is not a class-wide waiver. The full audit
shows that the V2.5 deterministic storage-write guard affects a broader
population than the discovery sample. Expanding the decision by identity list,
corpus hash, or similarity assertion would convert missing evidence into an
unsupported acceptance claim.

## Bounded implementation tranche

1. Preserve the complete Stage A-D candidate, reports, V2.3 reference, V2.4
   diagnostics, accepted V9 root, and old bounded evidence byte-for-byte.
2. Implement a versioned full-population evidence collector with duplicate-safe
   semantic node matching. Matching must use stable semantic/context identity,
   explicit multiplicity, and graph-neighborhood constraints; it must not use
   corpus-specific hashes or identity allowlists.
3. Generate at least three fresh exact Slither-0.10 evidence repeats for all
   311 raw non-parse-only primary identities. Keep generation roots distinct
   from the candidate and structural reference.
4. For every candidate/reference WRITE difference, record expression-level
   lvalue evidence, persistent-storage resolution, node/contract context, and
   repeat stability. Absence of a uniquely matched node remains unresolved.
5. Prove candidate-repeat determinism. Classify reference-to-candidate changes
   only as exact labelled graph equivalence, semantically proven storage-WRITE
   correction, or unresolved semantic-structure drift.
6. Investigate the existing semantic-structure identity separately and treat
   any new topology/feature/metadata disagreement that lacks a proof as a
   blocker.
7. Define a new versioned evidence schema and validator. The validator must
   fail on population mismatch, duplicate ambiguity, unstable repeats, missing
   expression evidence, unexpected runtime, unexplained drift, or evidence
   linked to the wrong candidate binding digest.
8. Add focused synthetic tests for duplicate nodes/multiplicity plus real
   regressions for all observed classes, then run the full Phase-8 regression
   suite.
9. Only after the collector and validator pass, produce a new V4 transition
   audit. Do not modify or reinterpret the existing V3 audit report.

## Exit gate

This tranche exits only when a new audit is bound to the exact Stage-D candidate
and reports all of the following:

- 22,540 candidate identities and the exact Stage-D binding digest;
- all 311 raw drift identities accounted for;
- repeated exact-runtime evidence is stable;
- duplicate-node ambiguity is zero;
- every semantic change is supported by explicit evidence;
- unexplained non-parse-only drift is zero;
- `physical_acceptance=false` and `training_authorized=false` remain recorded
  until a separate reviewed decision changes them.

Passing this evidence tranche permits consideration of a separate physical
acceptance decision; it does not itself accept the candidate or authorize
training.

## 2026-08-30 execution result

The versioned collector, three exact-runtime repeat generations, and strict
full-population validator are complete. All three 311-identity generations
passed artifact/runtime/token validation. The three semantic-evidence reports
are byte-identical.

The duplicate-safe analysis found 3,374 changed WRITE semantic groups across
299 contracts, representing 6,020 graph occurrences. Of those groups, 2,479
have multiplicity greater than one. Expression-level persistent-storage roots
prove 3,373/3,374 groups. Exact graph comparison after canonicalizing only those
proven groups resolves 310/311 contracts:

- 298 duplicate-safe storage-WRITE corrections;
- 12 exact node-index-invariant equivalences;
- one unresolved contract,
  `dive/f7b02c8346e4bc62fa8797e644b42cd80edb08fb6c86d14b4219fc34c6c54ae2`.

The formerly apparent semantic-structure case
`dive/bfa512a7a831999fa8140cd667e84524d3e01b09fb3cb258955f09b680863d62`
is resolved. Its ten duplicate semantic groups account for 20 WRITE graph
occurrences, and the optimized exact isomorphism check proves equivalence with
zero permutation-search states.

The remaining `f7...` statement is
`player.withdrawals.push(PlayerWitdraw(...))`. Slither 0.10 exposes no
`variables_written_as_expression` lvalue for this method-call mutation. Its
classification is also unstable: one fresh repeat emits WRITE while two emit
ARITH. Therefore it cannot be admitted by the current V2.5 evidence contract.

No V4 transition audit was produced because the plan's zero-unexplained-drift
precondition is false. Physical acceptance and training authorization remain
false. The next action requires a separate reviewed, versioned extractor/evidence
decision for storage-mutating member calls; it is outside this evidence-only
tranche.

## Stop lines

- Do not overwrite or regenerate Stages A-D merely because Stage E is false.
- Do not add identity allowlists, corpus-hash special cases, or a blanket
  storage-WRITE waiver.
- Do not treat duplicate matching failure as equivalence.
- Do not patch accepted V9 or historical V10 roots.
- Do not launch training, fit thresholds/calibration, promote the selector, or
  introduce negative labels in this tranche.
