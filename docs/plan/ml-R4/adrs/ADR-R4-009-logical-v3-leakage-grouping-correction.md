# ADR-R4-009 — Correct leakage grouping with logical lineage V3

**Date:** 2026-08-15
**Status:** ACCEPTED
**Decision ID:** R4-D-009
**Scope:** Phase-8 leakage grouping, role partitioning, evaluation reservations, and future training population

## Context

Repaired-v2 physically validated 22,540 contracts and 67,620 representation files, but its logical grouping policy treated any Ethereum address literal shared by two artifacts from the same source as leakage-family evidence.

The full-population grouping audit showed that rule was too broad. A single DIVE connected component contained 10,327 contracts and was dominated by ubiquitous protocol/sentinel/constants rather than plausible source-family identity. Examples included the Uniswap V2 router address in 8,225 artifacts, the dead address in 1,240 artifacts, the zero address in 519 artifacts, and WETH in 504 artifacts.

This did **not** invalidate repaired source bytes, label/evidence semantics, graph/token artifacts, or repaired-v2 physical binding. It invalidated `r4-leakage-groups-v2` as future split authority.

## Decision

Create immutable logical lineage:

- grouping: `r4-leakage-groups-v3`;
- role partition: `r4-vnext-roles-v3`;
- publication: `sentinel-r4-vnext-v3`;
- logical build: `r4-logical-lineage-v3`.

V3 reuses accepted repaired-v2 physical artifacts and role-independent source-evidence semantics. It changes logical grouping and downstream role/evaluation artifacts only.

### Grouping authority in V3

The following may create grouping authority:

1. identical normalized-code identity, globally;
2. exact artifact identity, globally by artifact identity/hash;
3. explicit source-provided family/project identifiers (`base_family_id`, `family_id`, `project_group_id`, `project_id`) **within the source namespace**.

Explicit family evidence keys must therefore use:

```text
<source>:<field>:<value>
```

Two unrelated sources that both use `project_id=1` must not be merged merely because the source-native identifier collides.

The following **must not** create grouping edges:

- arbitrary Ethereum address literals;
- common protocol addresses;
- zero/dead/sentinel addresses;
- event/topic/constants that happen to parse as addresses;
- same-source or cross-source address coincidence without an explicit family relation.

Address overlap remains diagnostic evidence only.

## Required invariants

Local V3 acceptance must prove:

- contract and contract×class populations unchanged from repaired-v2;
- target and STRONG/WEAK/NONE semantic counts unchanged;
- confirmed-negative rows remain zero unless a later evidence decision changes them;
- no address-authority grouping edge exists;
- the V2 giant address-connected group is removed;
- physical representation binding passes for every required contract/file;
- the V3 physical binding digest equals repaired-v2 because graph/token/sidecar bytes are reused unchanged;
- graph schema v9, token tensor `[4,512]`, and architecture `four_eye_v8` / `v8.1` remain unchanged.

## Validation outcome — 2026-08-16

Protected local V3 acceptance passed:

- contracts / contract×class rows: 22,540 / 225,400;
- positive / unknown / confirmed-negative: 1,080 / 224,320 / 0;
- STRONG / WEAK: 474 / 606;
- V3 groups: 22,394;
- maximum group size: 7;
- normalized-code edges: 146;
- address-authority edges: 0;
- V2 10,327-contract giant component removed;
- representation files checked: 67,620;
- physical binding digest: `16dd4a3f98c34e52e5c411b39268361881efede07e8f3f52d0c060dd1c5bb6dd`;
- physical rebuild performed: false.

The accepted population had **zero explicit-family edges**. Therefore the later source-namespacing hardening of explicit family IDs does not change the accepted V3 grouping population or group IDs; it closes a future correctness hole.

R4-D-009 remains **ACCEPTED**. Logical V3 is the current grouping/role authority for Phase-8 research.

## Post-acceptance evidence hardening — 2026-08-16

A protected-local audit of the research/reporting tranche found that:

- combined `MODEL_SELECTION` + `INTERNAL_AUDIT` outcome-metric rows had been mislabeled as model selection in one acceptance report;
- the final snapshot helper did not prove cross-report coherence before copying evidence;
- sensitivity evidence lacked sufficient immutable lineage metadata;
- confirmed-negative queue generation did not enforce group uniqueness across classes;
- explicit source-native family IDs were not source-namespaced.

These defects do **not** reverse V3 logical acceptance or repaired-v2 physical acceptance. Repository hardening now separates selection/audit reporting, binds research reports to the V3 manifest/physical digest/source commit, enforces globally unique queue groups, source-namespaces explicit family IDs, and makes final snapshotting fail closed on cross-report coherence.

The affected protected-local research reports must be regenerated before the final durable V3 evidence snapshot.

Active restart authority:

`runs/2026-08-16_PHASE8_v3_evidence_hardening_handoff.md`

The earlier acceptance/research checkpoint remains pre-hardening historical evidence.

## Consequences

### V2 remains historical evidence

Do not rewrite or delete `r4-leakage-groups-v2`, `r4-vnext-roles-v2`, or `sentinel-r4-vnext-v2`. Their physical evidence remains useful, but their grouping/role boundary is superseded for future model research.

### V3 physical lineage remains unchanged

V3 continues to reuse:

- `sentinel-preprocessed-r4-v2`;
- repaired source/provenance claims;
- `evidence-ledger-r4-v2`;
- `representations-r4-v2` / extractor `v2.2-r4-repaired`.

No physical rebuild is authorized by the evidence-hardening work.

### Research reports are version/source-bound evidence

Sensitivity, selector, queue, GPU and final-snapshot evidence must be mutually bound to the same V3 publication manifest and physical representation digest. A stale report must fail closed rather than be silently mixed with a newer lineage.

### Training remains unauthorized

This decision does not:

- create confirmed negatives;
- approve Positive-Unlabeled learning;
- promote `target_aware_guarded_v1`;
- enable threshold/calibration/untouched acceptance roles;
- authorize the 100-epoch Phase-8 run.

G8 remains open.

Before selector promotion, a separate decision must include full-population verification that the historical control selector reproduces the currently bound representation token tensors exactly.

## Rollback

Rollback is artifact selection, not reverse-editing:

- retain repaired-v2 physical artifacts as the physical reproducibility root;
- retain accepted V3 logical artifacts as current logical authority;
- preserve V2 grouping/roles as historical evidence;
- regenerate stale V3-derived research reports rather than editing them by hand;
- do not start long training from V2 or from pre-hardening V3 research evidence.

## Evidence

Primary triggering V2 grouping evidence:

`docs/plan/ml-R4/evidence/2026-08-15_phase8_research/grouping_breadth_audit_v1.json`

Pre-hardening V3 checkpoint:

`docs/plan/ml-R4/runs/2026-08-16_PHASE8_logical_v3_acceptance_and_research_checkpoint.md`

Current hardening/restart boundary:

`docs/plan/ml-R4/runs/2026-08-16_PHASE8_v3_evidence_hardening_handoff.md`
