# ADR-R4-009 — Correct leakage grouping with logical lineage V3

**Date:** 2026-08-15
**Status:** ACCEPTED FOR LOCAL VALIDATION
**Decision ID:** R4-D-009
**Scope:** Phase-8 leakage grouping, role partitioning, evaluation reservations, and future training population

## Context

Repaired-v2 physically validated 22,540 contracts and 67,620 representation files, but its logical grouping policy treated any Ethereum address literal shared by two artifacts from the same source as leakage-family evidence.

The full-population grouping audit showed that rule is not conservative in practice. A single DIVE connected component contains 10,327 contracts and is dominated by ubiquitous protocol/sentinel/constants rather than plausible source-family identity. Examples include the Uniswap V2 router address in 8,225 artifacts, the dead address in 1,240 artifacts, the zero address in 519 artifacts, and WETH in 504 artifacts. Transitive address overlap therefore makes unrelated contracts share one role boundary.

This does **not** invalidate the repaired source bytes, label/evidence semantics, graph/token artifacts, or repaired-v2 physical binding. It invalidates confidence in `r4-leakage-groups-v2` as the logical split authority and therefore in role/evaluation populations derived from it.

## Decision

Create a new immutable logical lineage:

- grouping: `r4-leakage-groups-v3`;
- role partition: `r4-vnext-roles-v3`;
- publication: `sentinel-r4-vnext-v3`;
- logical build: `r4-logical-lineage-v3`.

V3 reuses the accepted physical repaired-v2 artifacts and role-independent source-evidence ledger. It changes only logical grouping and downstream artifacts derived from grouping.

### Grouping authority in V3

The following may create grouping edges:

1. identical normalized-code identity, globally;
2. explicit source-provided family/project identifiers (`base_family_id`, `family_id`, `project_group_id`, `project_id`);
3. exact artifact identity is intrinsically one contract identity and needs no extra union edge.

The following **must not** create grouping edges:

- arbitrary Ethereum address literals;
- common protocol addresses;
- zero/dead/sentinel addresses;
- event/topic/constants that happen to parse as addresses;
- cross-source or same-source address coincidence without an explicit family relation.

Address overlap remains diagnostic evidence only.

## Required invariants

Local V3 acceptance must prove all of the following:

- contract population unchanged from repaired-v2;
- contract×class population unchanged;
- target counts unchanged;
- STRONG/WEAK/NONE semantic counts unchanged;
- confirmed-negative rows remain zero unless a later evidence decision changes them;
- no address-authority grouping edge exists;
- the V2 giant address-connected group is removed;
- physical representation binding passes for every required contract/file;
- the V3 physical binding digest equals repaired-v2 because graph/token/sidecar bytes are reused unchanged;
- frozen graph schema v9 and token tensor shape `[4,512]` remain unchanged;
- architecture `four_eye_v8` / `v8.1` remains frozen.

## Consequences

### V2 remains historical evidence

Do not rewrite or delete `r4-leakage-groups-v2`, `r4-vnext-roles-v2`, or `sentinel-r4-vnext-v2`. Their physical acceptance remains evidence of the repaired corpus and representations, but their grouping/role boundary is superseded for future model research.

### V2 research outputs become population-specific historical evidence

The following V2 outputs remain useful diagnostics but are not future decision authority after V3 is accepted:

- V2 confirmed-negative review queue;
- V2 selector population statistics;
- V2 representation sensitivity role sets;
- V2 selector CUDA comparison.

They must be regenerated against V3 roles before manual negative adjudication, selector promotion, exclusion/down-weighting decisions, or training-horizon binding.

### No physical rebuild

V3 must reuse:

- `sentinel-preprocessed-r4-v2`;
- `r4-provenance-v1` / repaired source claims;
- `evidence-ledger-r4-v2` as role-independent semantic evidence;
- `representations-r4-v2` / extractor `v2.2-r4-repaired`.

If V3 changes any physical representation hash, the logical-only migration assumption has failed and the attempt must stop for investigation.

### Training remains unauthorized

This decision repairs split/group semantics only. It does not:

- create confirmed negatives;
- approve Positive-Unlabeled learning;
- promote `target_aware_guarded_v1`;
- enable threshold/calibration/untouched acceptance roles;
- authorize the 100-epoch Phase-8 run.

G8 remains open.

## Migration

Execute the versioned V3 logical rebuild and acceptance sequence documented in:

`runs/2026-08-15_PHASE8_logical_v3_grouping_repair_handoff.md`.

After V3 acceptance, regenerate the negative-review queue, selector coverage experiment, representation sensitivity profile, and identical-initialization CUDA comparison with required worst-case probes.

## Rollback

Rollback is artifact selection, not reverse-editing:

- retain repaired-v2 physical artifacts as the physical reproducibility root;
- discard/archive an unsuccessful V3 generated attempt only;
- do not mutate V2 grouping, roles, publication, or evidence snapshots;
- do not start long training from V2 merely because V3 validation fails.

## Evidence

Primary triggering evidence:

`docs/plan/ml-R4/evidence/2026-08-15_phase8_research/grouping_breadth_audit_v1.json`

Key observed V2 condition: largest group = 10,327 contracts, with 18,213 address-derived edges and 999 address keys.
