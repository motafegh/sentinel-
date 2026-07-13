# L02 — Inspect export, hashes, splits, and SentinelDataset

## Learning objective

Prove how an export admits or rejects a sample and how graph/token/label rows remain aligned.

## Prerequisites

Read [T02](../technical/02_data_representation_export.md). Use the ML environment and a local DATA v2 export or regenerate a miniature export.

## Source reading order

`export/export.py::SentinelDatasetExport` → export writers/chunker → `ml/src/datasets/sentinel_dataset.py::SentinelDataset` → `collate.py::sentinel_collate_fn` → `ml/tests/test_sentinel_dataset.py`.

## Setup and artifact requirements

Tier is module. The baseline export is ignored-local and absent in a fresh clone. Confirm the path through inventory; do not substitute a different export without recording its manifest/hash.

## Initial observation

```bash
python3 docs/handbook/tools/verify_handbook.py inventory
TMPDIR=/tmp TMP=/tmp TEMP=/tmp ml/.venv/bin/python -m pytest \
  ml/tests/test_sentinel_dataset.py -q
```

## Controlled edit

In a disposable worktree, add a test that copies the smallest test export to `tmp_path`, changes one byte in a shard, clears `.hash_cache.json` if present, and asserts `SentinelDataset` raises the artifact-hash mismatch. Add a second assertion that a split ID absent from `shard_index` is filtered, not misindexed.

## Expected success output

Normal fixture loads with graph feature width 12, token shape `[4,512]`, label shape `[10]`; the tampered copy is rejected before sampling.

## Expected failure output

Fresh clone: the real-export test setup reports the missing ignored-local artifact. A mismatched graph version raises a graph-schema error distinct from hash failure.

## Verification

Run the selected dataset tests and `python3 docs/handbook/tools/verify_handbook.py lab --check L02`.

## Reset and cleanup

Restore only the test file and let `tmp_path` cleanup remove copies. Never “repair” the original manifest hash.

## Completion rubric

Complete when you can map a contract ID to shard/position and explain all three admission gates.

## Review questions

Why is the manifest excluded from data hashing? Why filter missing representations? Which change bumps format versus graph schema?

## Classification

Module; local-artifact preflight; controlled test-copy edit.
