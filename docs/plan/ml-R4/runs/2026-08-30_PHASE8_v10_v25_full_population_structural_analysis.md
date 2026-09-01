# Phase-8 V10 V2.5/V2.6 full-population structural analysis

Date: 2026-08-30 through 2026-09-01
Decision: `V26_TRANSITION_EVIDENCE_RECONCILED_PENDING_PHYSICAL_DECISION`
Scope: historical 311-case V2.5 analysis plus complete 355-case V2.6 evidence

## Current V2.6 result

The versioned storage-collection mutator correction is complete under extractor
`v2.6-r4-call-semantics-deterministic-cfg-mutators`. The fresh staged candidate
contains 22,540/22,540 accepted-V9 identities and is bound by digest
`d9f925588913e66476cfbc097bace7daa7e673295fe2a243760313d0bef5ebdd`.
All token files are byte-identical to accepted V9; the required runtime split is
22,539 Slither-0.10 primary identities plus one Slither-0.11.5 identity-bound
exception.

The first V4 run failed closed because V2.6 changed the actual drift population:
52 identities were new and eight old V2.5 identities were absent, producing an
exact current census of 355 rather than 311. No identity was ignored. Three new
355-identity generations and three source-evidence passes were therefore run
against the new candidate binding.

The final evidence population contains:

- 349 contracts with proven persistent-storage WRITE drift;
- 3,517 proven semantic target groups and 6,247 graph occurrences;
- 2,532 duplicate target groups;
- 6 exact node-index-invariant equivalence contracts;
- zero unresolved WRITE groups or population mismatches;
- three byte-identical semantic-evidence reports, SHA-256
  `92afb95f5335226ee28c99969779af0dd5f69da4296cf400736ff4c4e75bce42`.

The complete V4 transition audit passes 22,540/22,540 mechanics and independently
re-proves all 355 identities against the actual full candidate: 349 storage-WRITE
corrections plus six index-equivalent graphs, with zero unexplained structural
drift. Its status is
`PASS_TRANSITION_EVIDENCE_RECONCILED_PENDING_PHYSICAL_DECISION`.

Protected-local roots and authority artifacts:

- primary attempt: `data_module/data/v10-v26-primary-attempt-2026-08-31-a`;
- full candidate: `data_module/data/v10-v26-full-candidate-attempt-2026-09-01-a`;
- 355-case evidence: `data_module/data/v10-v26-full-structural-proof-2026-09-01`;
- final full-population probe SHA-256
  `9a1cf96465613b61fae2d10ccaa81def0548663a4c4711ca745841f6354e7a55`;
- final V4 audit SHA-256
  `c6ddc61b8005a688d422f4f8de28118fa3e644b9648d070ef53972ec9f2191ce`.

This result is diagnostic evidence only. `physical_acceptance=false` and
`training_authorized=false`; explicit report review and a separate physical
acceptance decision remain required.

## Historical V2.5 result

### Result

The 311-case analysis is complete and bound to candidate digest
`17c5f334c75015fdaf89b1a9f77522af5185f2485c24df4e1e64917dc944f021`.
Three fresh exact Slither-0.10 generations each passed 311/311 artifact,
runtime, token-byte, and population checks.

The evidence population contains:

- 299 contracts with WRITE-classification drift;
- 3,374 changed semantic groups and 6,020 graph occurrences;
- 2,479 duplicate groups;
- 3,373 groups with stable expression-level persistent-storage proof;
- zero non-WRITE identity-population mismatches;
- three byte-identical semantic-evidence reports.

After duplicate-safe canonicalization of only the 3,373 proven groups, exact
labelled directed-multigraph comparison through edge type 10 reports:

- 298 `PROVEN_DUPLICATE_SAFE_STORAGE_WRITE_CORRECTION`;
- 12 `PROVEN_EXACT_NODE_INDEX_INVARIANT_EQUIVALENCE`;
- 1 `UNRESOLVED_STRUCTURAL_OR_SEMANTIC_DRIFT`.

### Resolved large-graph case

`dive/bfa512a7a831999fa8140cd667e84524d3e01b09fb3cb258955f09b680863d62`
is not semantic structure drift. Ten duplicate semantic groups cover 20 WRITE
occurrences. Once those independently evidenced writes are canonicalized, the
1,065-node reference, candidate, and repeats are exactly isomorphic. The
optimized verifier proves this without permutation-search states.

### Remaining blocker

The sole blocker is
`dive/f7b02c8346e4bc62fa8797e644b42cd80edb08fb6c86d14b4219fc34c6c54ae2`,
specifically:

`player.withdrawals.push(PlayerWitdraw({time: uint256(block.timestamp), amount: amount_withdrawable}))`

Slither 0.10 emits no expression-level written lvalue for this method-call
mutation. Its derived classification is unstable across the bound candidate
and three fresh repeats: candidate/repeats 2-3 emit ARITH, while repeat 1 emits
WRITE. SlithIR diagnostics show reference variables rooted in the storage local
`player`, but this mutable alias path is deliberately not accepted as a silent
replacement for the required expression-level evidence.

This is now a narrow classifier/evidence-contract gap, not a 311-case unknown.
A separate reviewed versioned change must decide how storage-mutating member
calls such as array `push` are recognized. The current evidence tranche does
not authorize that extractor change.

### Protected-local evidence

Root:
`data_module/data/v10-v25-full-structural-evidence-2026-08-30`

- full probe: `full-population-probe-v1.json`, SHA-256
  `8ef0d0d71f7311dc3287b77eccbdf4c687cda9f730ae5f8fe9f342da15ab0a20`;
- semantic evidence reports 1-3: byte-identical SHA-256
  `d12a3f7d19ef5e82132127b3a1b55859cdad3ed226d1e4f2473ed324ef10672c`;
- repeat report 1 SHA-256
  `b1ce31e17046c43d2a0a9cf655af60afb10f82fd870ce819099dd5baa2fd8f2c`;
- repeat report 2 SHA-256
  `85d03b02c705ac531ab6ff23c7dc51d7e3941dddf76abd90b9f82d7381351c4c`;
- repeat report 3 SHA-256
  `8fd1c5652d94aa45aaf203db07ce02164b832c17e2f9ac91167f85dc9e5be32b`.

### Historical gate

No V4 audit is generated while one identity remains unexplained.
`physical_acceptance=false` and `training_authorized=false`. No training,
selector promotion, threshold fitting, or candidate/root mutation is
authorized by this analysis.
