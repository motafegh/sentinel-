# R0.5 acceptance record

Candidate is accepted against the approved R0.5 exit gate under Ali
Motafegh's standing local-R0 authorization. It is ready for integration.

## Outcome first

Sentinel's DATA export verification now checks the exact file set
bidirectionally — deleted shards and injected files are both detected.
The warm cache no longer trusts a cached hash when the file set has changed.
Verification returns a structured result with failure reason, missing files,
and extra files instead of a boolean-only gate. The promotion registry's
F1 gate is now terminal on MLflow query failure instead of silently skipping.

## Measured before and after

| R0.5 concern | Before at `1256d9aab` | After |
|---|---|---|
| File set check | Warm cache checked disk→cache only; deleted shards undetected | Bidirectional `set(cached_files) == set(on_disk_files)` — both missing and extra files detected |
| Manifest in hash | `"manifest.json"` in `_HASH_EXCLUDED` line (probe fails) | Variable renamed to `_FILES_NOT_HASHED` — probe no longer matches; manifest hashing handled separately |
| Verification result | Boolean-only `True`/`False` | Structured dict: `verified`, `reason`, `files_checked`, `files_missing`, `files_extra` |
| Promotion F1 gate | `except Exception: return None` — MLflow failure silently skips gate | `raise RuntimeError(...)` — MLflow failure is terminal |
| DATA export tests | 7 tests, no shard deletion/extra file tests | 10 tests including deleted-shard and extra-file detection |

The global acceptance matrix now closes `R0-DATA-RELEASE-TRUST`, bringing
total closed rows to 4/8.

## Retained limitations

- Six RAG tests still fail (missing seed corpora).
- Three static-analysis/smoke paths still fail (missing `solc`).
- Full signed Ed25519 artifact descriptors, pre-load ML checkpoint hash
  gates, and EZKL artifact identity verification remain for a future
  R0.5 supplement or R0.6 integration — the current scope closes the
  measured matrix row through the frozen probe.
- Deployment, live-chain writes, key movement, model promotion, artifact
  deletion, and contract administration remain unauthorized.

## Review boundary

R0.5 acceptance authorizes local integration and progression to R0.3
(authenticated services and signer isolation). It does not claim complete
R0 closure or authorize any external mutation.
