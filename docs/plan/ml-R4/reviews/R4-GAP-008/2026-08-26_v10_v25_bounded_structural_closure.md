# R4-GAP-008 V10 V2.5 bounded structural closure

Date: 2026-08-26
Status: BOUNDED 20-IDENTITY STRUCTURAL INVESTIGATION CLOSED
Scope: Phase-8 V10 structural evidence only; no physical acceptance or training authority

## Result

Three fresh regenerations of the exact 20 previously unexpected structural-drift
identities were produced under the exact primary runtime:

- `slither-analyzer = 0.10.0`;
- extractor `v2.5-r4-call-semantics-deterministic-cfg`;
- 20 / 20 requested records in each repeat;
- all records used `slither_full_analysis`;
- all three generations passed their mechanical regression checks.

The complete semantic WRITE evidence was merged from the original 12-contract
probe and the fail-closed eight-node expansion for
`dive/83c9d2d26dc19eaa2aee29fa7aedb4f4e208429a96cc7a0ffee7491b9830630d`.
The merged evidence still covers exactly 12 semantic-correction identities, and
that contract carries 13 positively evidenced persistent-storage WRITE nodes.

The final bounded V2.5 reproducibility verifier returned exit code 0 with:

- `unexpected_identities = 20`;
- `semantic_correction_identities = 12`;
- `index_equivalence_identities = 8`;
- `repeat_generations = 3`;
- `bounded_v25_reproducibility_passed = true`;
- `zero_unexplained_drift = true`;
- `blocking_identities = []`.

Decision census:

| Decision | Identities |
|---|---:|
| `V25_DETERMINISTIC_STORAGE_WRITE_CORRECTION_PROVEN` | 12 |
| `V25_NODE_ORDER_INDEX_EQUIVALENCE_REPRODUCED` | 8 |

## Meaning

The 20-identity root-cause tranche is closed. There is no remaining unexplained
repeat-to-repeat structural drift in this bounded set.

The 12 WRITE identities are not waived differences. Their relevant lvalues were
independently re-parsed under exact Slither 0.10.0 and shown to mutate persistent
storage. V2.5 deterministically emits `CFG_NODE_WRITE` for those evidenced nodes,
and after canonicalizing only those explicit nodes the frozen reference and V2.5
outputs remain exact node-index-invariant labelled multigraph equivalents through
unchanged edge type 10.

The 8 index-equivalence identities likewise are not raw-index waivers. Exact
labelled directed-multigraph isomorphism through unchanged edge type 10 was
reproduced across the three fresh V2.5 generations.

## Next gate

The bounded result does not physically accept V10. The next gate is a refreshed
complete V2.5 protected candidate lineage and a complete 22,540-identity
transition audit that:

1. preserves all V2 population, token-byte, schema/runtime, call-IR, call-edge,
   and binding checks;
2. re-proves the 8 index-equivalence identities against the actual full
   candidate;
3. re-proves the 12 deterministic WRITE identities against the actual full
   candidate using the merged semantic evidence;
4. rejects every additional non-parse-only structural difference;
5. keeps the historical accepted-V9 parse-only repair class separate;
6. leaves physical acceptance false until explicit review and decision;
7. leaves training authorization false as a separate later gate.

Repository support for this next gate is provided by:

- `p8_audit_v10_transition_v3.py`;
- `p8_validate_v10_v25_evidence_chain.py`;
- their focused fail-closed tests.
