# T05 — Proxy distillation and EZKL proof lifecycle

## Learning outcome

You can trace teacher embedding → proxy logits → ONNX/EZKL witness → proof/calldata, decode the 138 public signals, and state precisely what the proof and provenance do not establish.

## Prerequisites

Read [ZKML](../07_zkml.md). Know logits, knowledge distillation, field elements, hashes, and the difference between a witness, proving key, verification key, and verifier contract.

## Source map and reading order

1. `zkml/src/distillation/{extract_features,train_proxy,export_onnx}.py`.
2. `zkml/src/distillation/proxy_model.py::ProxyModel` — frozen architecture guards.
3. `zkml/src/ezkl/setup_circuit.py`, `zkml/ezkl/settings.json`, then `run_proof.py::generate_proof`.
4. `zkml/src/ezkl/extract_calldata.py` — public instances and proof bytes.
5. `agents/src/mcp/servers/audit/_submit.py::_run_submit` — real-contract direct submission.
6. `contracts/src/IZKMLVerifier.sol::verifyProof` and `AuditRegistry.submitAuditV2`.

## Entry point and complete call chain

Offline distillation extracts 128-wide teacher fusion embeddings and trains the fixed `128→64→32→10` proxy. ONNX export and EZKL setup produce settings, compiled circuit, keys, SRS dependency, and Solidity verifier. Per audit, the direct MCP path requests `/fusion-embedding`, runs the local proxy, writes the actual embedding to proof input, generates a witness, proves, verifies off-chain, extracts proof/public instances, and calls the registry. The standalone `run_proof.py` chooses a corpus contract and is a workflow demonstration, not the gateway submission path.

## Important symbols and configuration

- Frozen dimensions: 128, 64, 32, 10; exact parameters: 10,666; circuit `v2.0`.
- Public instances are 128 input values followed by ten output values: 138 total. Contract `INPUT_OFFSET=128` selects class outputs.
- EZKL settings currently use public inputs, public outputs, fixed parameters, and `check_mode="UNSAFE"`. The last setting requires explicit security review before production claims.
- Witness output hex is decoded little-endian; human scores use scale 8192.
- Proving key and SRS are ignored/private; tracked compiled/settings/VK artifacts alone cannot create a proof.

## Annotated source excerpt

Source: `zkml/src/ezkl/run_proof.py::generate_proof`

```python
outputs = witness["outputs"][0]
for hex_str in outputs:
    felt = int.from_bytes(bytes.fromhex(hex_str), byteorder="little")
    class_score_felts.append(felt)
```

Endianness is part of the cross-language contract. Interpreting these bytes big-endian changes the claimed class values even if proof bytes remain unchanged.

## Worked example

A teacher returns embedding `e[0..127]`. The proxy produces ten raw logits; sigmoid score `0.75` becomes approximately `round(0.75×8192)=6144`. EZKL exposes input instances `0..127` and output instances `128..137`. Registry V2 checks the supplied ten scores against those output positions. It does not recompute the teacher, threshold classes, or evaluate `verdict_provable`.

## Success trace

Architecture guard passes; checkpoint and ONNX match; setup artifacts share a circuit/version; witness contains ten outputs; proof verifies off-chain; calldata has 138 instances; verifier accepts; stake and score guards pass; transaction receipt records V2 event/storage.

## Failure trace

Wrong dimensions raise before expensive setup. Missing proving key/SRS fails explicitly. Teacher/proxy disagreement may block the demonstration workflow. Wrong endianness or class offset causes score mismatch. Shared `proof_input.json`, `witness.json`, and `proof.json` create a concurrency hazard in the current submit implementation. Gateway audits do not invoke this path.

## Design reasoning and rejected alternatives

The small fixed proxy keeps circuit cost manageable; proving the full teacher was rejected as impractical here. Public embeddings simplify verifier integration but reveal the embedding. Fixed parameters bind the proxy weights. `UNSAFE` is a development configuration, not evidence of production security; review against the pinned EZKL binding behavior and official security guidance is mandatory.

## Safe change walkthrough

Any proxy dimension, activation, weight, visibility, scale, or EZKL-version change is a circuit migration: bump circuit identity, retrain/export, regenerate settings/compiled/keys/verifier, run architecture and signal-layout tests, deploy verifier, update registry/deployment config, and retain rollback artifacts. Never reuse keys across a changed circuit.

## Guided lab

Complete [L05 — proxy, witness, signals, and guards](../labs/05_zkml_witness_signals.md).

## Tests and expected results

```bash
TMPDIR=/tmp TMP=/tmp TEMP=/tmp ml/.venv/bin/python -m pytest \
  zkml/tests/test_proxy_model.py zkml/tests/test_run_proof.py -q
```

Expected: shape/freeze/parameter/endian/layout tests pass without claiming a live proof. `verify_handbook.py live --ezkl` must fail when private proving prerequisites are absent.

## Review questions

What are public signals 128 and 137? Why does a valid proof not prove teacher provenance? What invalidates keys? Why is `verdict_provable` outside this circuit?

## Ownership checklist

- I distinguish teacher, proxy, circuit, verifier, and registry claims.
- I can decode every public output position.
- I treat provenance as an off-chain operator assertion.
- I require review before changing `UNSAFE` or visibility settings.
