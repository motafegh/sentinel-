# SENTINEL DATA Module Working Instructions

These instructions refine root `CLAUDE.md` for current DATA work. Root project authority and committed R4 machine-readable policy/manifests remain higher authority.

## Current DATA state

Historical `sentinel-r4-vnext-v1` / graph-schema-v9 evidence remains immutable G7 history.

R4-D-008 physically accepted the repaired-v2 source/representation lineage for bounded Phase-8 research:

- preprocessing: `sentinel-preprocessed-r4-v2`;
- provenance/source claims: `r4-provenance-v1`;
- role-independent evidence ledger: `evidence-ledger-r4-v2`;
- representation root: `representations-r4-v2`;
- extractor: `v2.2-r4-repaired`;
- graph schema: `v9`;
- token tensor: `[4,512]`;
- physical population: 22,540 contracts / 67,620 graph-token-sidecar files;
- physical binding digest: `16dd4a3f98c34e52e5c411b39268361881efede07e8f3f52d0c060dd1c5bb6dd`.

That physical acceptance still stands.

However, the 2026-08-15 full-population grouping audit proved `r4-leakage-groups-v2` over-connects unrelated contracts through arbitrary same-source Ethereum address literals. One DIVE component contains 10,327 contracts and is dominated by common protocol/sentinel addresses. Therefore R4-D-009 / ADR-R4-009 supersedes V2 **grouping/roles** for future research while preserving V2 physical artifacts as historical/physical evidence.

The active logical candidate is:

- grouping: `r4-leakage-groups-v3`;
- role partition: `r4-vnext-roles-v3`;
- publication: `sentinel-r4-vnext-v3`;
- logical build: `r4-logical-lineage-v3`.

Repository implementation exists; protected local V3 generation/acceptance is still required before V3 becomes accepted local authority.

## Mandatory semantics

1. **Never overwrite historical or accepted physical DATA artifacts.** Semantic/partition changes use new versioned roots.
2. **Unknown is not negative.** No target `0` without class-specific confirmed-negative evidence. Current policy v1 has none.
3. **Source record != contract identity != leakage group.** Preserve all three boundaries explicitly.
4. **Ethereum address coincidence is not identity or family authority.** In V3, address literals are diagnostic only and create zero grouping edges. Do not reintroduce V2 same-address union logic through thresholds or heuristics without a new evidence-backed decision.
5. **Normalized-code identity and explicit source family/project IDs may define leakage groups.** Exact artifact identity remains one contract identity.
6. **Compile the exact promoted normalized source.** No compile-before-normalize or regex-only semantic rewriting.
7. **File-level graph selection preserves label scope.** Provenance target is authoritative; otherwise use the documented inheritance-leaf/file-union rules rather than guessing one unrelated declaration.
8. **Long-contract truncation is visible.** Keep `[4,512]` for the frozen architecture tranche, record pre-subsampling telemetry, and do not equate shape validity with adequacy.
9. **Weak evidence stays weak.** DIVE Front Running→TransactionOrderDependence remains WEAK training-only under policy v1.
10. **Every failure is explicit.** Compile, graph, binding, grouping, and evidence failures must raise or be represented structurally; no silent skips.

## Current execution seam

Do **not** rerun the expensive repaired-v2 physical build for the grouping correction.

Use the logical-only V3 handoff:

`docs/plan/ml-R4/runs/2026-08-15_PHASE8_logical_v3_grouping_repair_handoff.md`

Primary driver:

```bash
PYTHONPATH=.:data_module ./ml/.venv/bin/python \
  docs/plan/ml-R4/scripts/p8_rebuild_logical_v3.py --help
```

Required local sequence is V3 grouping → V3 role/publication freeze → same-byte physical rebinding → V2→V3 acceptance audit → regenerated V3 research evidence.

The V3 binder must prove the physical representation binding digest is unchanged from repaired-v2. If that digest changes, stop: a supposedly logical-only change has modified the physical contract.

## Research/evaluation boundary

The V2 confirmed-negative queue and V2 role-dependent selector/sensitivity/GPU outputs are historical population-specific evidence after R4-D-009. Do not manually adjudicate or promote from them.

After local V3 acceptance, regenerate:

- V3 representation sensitivity;
- V3 selector population comparison;
- V3 confirmed-negative pilot queue;
- V3 identical-initialization CUDA selector comparison with mandatory worst-case probes.

No pseudo-negatives, selector promotion, PU objective, threshold/calibration fitting, or full training is authorized by those generation steps alone.

## Validation

Repository-safe validation is `.github/workflows/r4-phase8-data-repair.yml`, including V3 grouping/partition tests and frozen historical G6 validation.

Protected local acceptance additionally requires the existing repaired-v2 preprocessing/representation trees and V3 generated publication/binding reports. Repository CI cannot substitute for local physical hash verification.

## Training boundary

Full Phase-8 training remains prohibited. Current blockers are:

- local V3 logical acceptance pending;
- zero confirmed-negative evaluation evidence;
- target-aware selector not promoted;
- threshold/calibration/untouched acceptance unavailable.

**Current DATA status:** repaired-v2 physical source/representations accepted and immutable; V2 logical grouping/roles superseded for future research; corrected logical V3 implemented repository-side and awaiting local generation/acceptance; G8 open.
