# Phase-8 V10 v2.5 bounded reproducibility checkpoint

Date: 2026-08-25
Status: 19 / 20 bounded identities resolved; one deterministic-CFG blocker remains
Scope: R4-B008 structural blocker only; physical acceptance and training remain unauthorized

## Runtime and lineage

Protected-local evidence was produced with:

- `ml/.venv/bin/python`
- `slither-analyzer = 0.10.0`
- `crytic-compile = 0.3.11`
- `PYTHONPATH=.:data_module`
- extractor `v2.5-r4-call-semantics-deterministic-cfg`

The frozen v2.3 structural reference remains immutable at:

`data_module/data/representations-r4-v3-candidate-v2.3-structural-reference-6087dc6d`

with binding digest:

`6087dc6d76d781efbefe0c4984458d291790c38b1c55d852f48fd796222b0260`.

## Semantic-direction evidence

The exact 12 previously feature-classification-drifting contracts were re-parsed under exact Slither 0.10.0. All requested drifting nodes were found. Every relevant member/index lvalue was rooted in a `LocalVariable` with `location = storage` and `is_storage = true`; no memory-only false-positive direction was observed.

Therefore the intended deterministic semantic correction is `CFG_NODE_WRITE` for those explicitly evidenced nodes. The v2.5 guard retains negative controls for memory-local member writes and storage-reference declarations/rebindings.

Focused v2.5 source tests passed 27 / 27. The dedicated v2.5 reproducibility-verifier synthetic suite passed 3 / 3.

## Three fresh v2.5 bounded regenerations

The same 20 unexpected identities from transition audit v2 were regenerated three fresh times under exact Slither 0.10.0.

All three runs reported:

- `passed = true`;
- 20 / 20 records;
- 20 / 20 `slither_full_analysis`;
- extractor identity `v2.5-r4-call-semantics-deterministic-cfg` through the bound V10 generation path.

The dedicated verifier report is:

`docs/plan/ml-R4/reviews/R4-GAP-008/v10_v25_reproducibility_probe_v1.json`

The bounded result is currently:

| Decision | Count |
|---|---:|
| `V25_NODE_ORDER_INDEX_EQUIVALENCE_REPRODUCED` | 8 |
| `V25_DETERMINISTIC_STORAGE_WRITE_CORRECTION_PROVEN` | 11 |
| `BLOCKED_V25_STORAGE_WRITE_REPRODUCIBILITY` | 1 |

The sole remaining blocker is:

`dive/83c9d2d26dc19eaa2aee29fa7aedb4f4e208429a96cc7a0ffee7491b9830630d`

The other 19 identities are now boundedly resolved under the explicit evidence classes. Do not reopen them unless concrete contradictory evidence appears.

## Immediate next action

Do not regenerate the 20 identities again yet. First inspect the existing v2.5 verifier record for `83c9d2...` and distinguish which fail-closed condition fired:

1. one of its five evidenced storage-write nodes was not emitted as `CFG_NODE_WRITE` in a repeat;
2. repeat-to-repeat exact labelled graph equivalence failed outside the expected storage-write correction; or
3. canonicalized v2.5 still differs from the frozen reference outside those five explicitly reviewed nodes.

Only after that exact difference is identified should another extractor change or repeat generation be authorized.

Physical acceptance remains false. Full 22,540-identity transition audit has not yet been rerun for v2.5. Training and model-quality claims remain unauthorized.
