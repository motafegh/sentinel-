# L05 — Inspect proxy, witness signals, and architecture guards

## Learning objective

Validate the frozen proxy, little-endian output decoding, and 128+10 public-signal layout without confusing unit evidence with a live proof.

## Prerequisites

Read [T05](../technical/05_zkml_proof_lifecycle.md). Use the ML Python environment.

## Source reading order

`proxy_model.py::ProxyModel` → `zkml/ezkl/settings.json` → `run_proof.py::generate_proof` → `extract_calldata.py` → registry `INPUT_OFFSET`.

## Setup and artifact requirements

Tier is smoke for unit work. Tracked proxy/settings/circuit/VK support inspection. Live witness/proof requires teacher checkpoint, ignored proving key/SRS, EZKL, and compatible runtime.

## Initial observation

```bash
TMPDIR=/tmp TMP=/tmp TEMP=/tmp ml/.venv/bin/python -m pytest \
  zkml/tests/test_proxy_model.py zkml/tests/test_run_proof.py -q
```

## Controlled edit

Add a test vector with a known little-endian hex value and assert the decoded integer. Add a guard test constructing `ProxyModel(hidden2=31)` and assert `RuntimeError` mentions key invalidation. Edit tests only.

## Expected success output

Proxy output is `[B,10]`, parameter count remains 10,666, all architecture mutations fail fast, and public signals total 138 with class offset 128.

## Expected failure output

Big-endian decode produces a different integer. `verify_handbook.py live --ezkl` explicitly fails if proving key/SRS or service dependencies are absent.

## Verification

Run the observation command and `verify_handbook.py lab --check L05`.

## Reset and cleanup

Restore changed test files. A failed live proof may clean partial witness/proof; never delete setup keys as routine cleanup.

## Completion rubric

Complete when you can map all 138 positions and state what a passing unit suite does not prove.

## Review questions

Why little-endian? Which change invalidates keys? Why are teacher and verdict outside proof semantics?

## Classification

Smoke unit lab; safe preflight; live proof is separately classified live.
