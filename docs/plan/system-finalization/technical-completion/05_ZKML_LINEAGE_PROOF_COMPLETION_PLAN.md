# ZKML Lineage / Proof Completion Plan

**Status:** ACTIVE — audit/research may proceed; rebuild waits for an accepted repaired teacher  
**Parent:** `00_MASTER_TECHNICAL_COMPLETION_ROADMAP.md`  

## 1. Objective

Reconcile SENTINEL’s ZKML subsystem with the eventual accepted operational teacher model while preserving exact proof-scope boundaries. Produce a versioned, lineage-bound proxy/ONNX/EZKL/verifier bundle whose security settings and compatibility are explicitly understood and validated.

This plan does not attempt to make EZKL prove the entire Solidity audit. The retained architecture proves a proxy computation over a teacher fusion representation; V3 context/provenance binding remains a separate authenticated statement.

## 2. Current executable baseline

Substantive sources already exist under:

- `zkml/src/distillation/`
- `zkml/src/ezkl/`
- `zkml/models/`
- `zkml/ezkl/`
- `zkml/tests/`
- generated verifier integration under `contracts/`

Current facts:

- teacher fusion input is 128-dimensional;
- proxy output is 10-class teacher-probability regression;
- distillation, ONNX export, calibration generation, circuit setup, proof generation/verification and bundle validation exist;
- the current proof generator is explicitly historical-Run12-oriented;
- the current proof scope is `legacy_proxy_only_unbound`;
- V3 context attestation does not upgrade what the ZK circuit itself proves;
- retained tracked EZKL settings contain `check_mode="UNSAFE"`;
- new ONNX identity is already treated by setup code as requiring a newly generated/validated bundle unless an explicit compatibility procedure proves reuse.

## 3. Immutable/history rules

- Preserve existing Run12/proxy/EZKL artifacts as historical evidence.
- Never overwrite a historical proof bundle in place to represent a new teacher.
- New teacher → new distillation identity → new ONNX/calibration identity → new setup/bundle identity unless compatibility is separately proven.
- Proof scope must stay explicit in every layer.
- Do not reuse verification/proving keys merely because dimensions are unchanged.

## 4. Primary source/artifact owners to audit

At minimum:

- `zkml/src/distillation/proxy_model.py`
- `zkml/src/distillation/train_proxy.py`
- `zkml/src/distillation/corpus_distill.py`
- `zkml/src/distillation/export_onnx.py`
- `zkml/src/distillation/generate_calibration.py`
- `zkml/src/ezkl/setup_circuit.py`
- `zkml/src/ezkl/run_proof.py`
- `zkml/src/ezkl/validate_bundle.py`
- `zkml/src/ezkl/export_foundry_fixture.py`
- tracked model/manifests/settings/proof artifacts;
- `contracts/src/ZKMLVerifier.sol` and verifier tests;
- agents V3 submission/policy-signing preparation code.

## 5. Work package Z0 — source and artifact lineage audit

**State:** `AUDITING`

Produce an exact current map:

```text
Run12 teacher checkpoint
→ fusion extraction
→ distillation corpus
→ proxy checkpoint
→ ONNX + manifest
→ calibration + manifest
→ EZKL settings
→ compiled circuit + SRS + PK/VK
→ proof/public signals
→ generated Solidity verifier
→ AuditRegistry V2/V3 verification boundary
```

For every artifact identify:

- path/version;
- SHA-256 where recorded/available;
- generating source commit;
- teacher/DATA identity;
- EZKL version/settings identity;
- whether tracked artifact is historical, reusable control, or current candidate;
- tests/evidence that validate it.

Any anonymous/unbound artifact is historical/research-only until rebound.

## 6. Work package Z1 — repaired-teacher handoff contract

**State:** `BLOCKED` until ML promotion decision

ZKML may begin a new distillation lineage only when ML provides:

- accepted teacher checkpoint SHA-256;
- architecture/class schema;
- exact DATA/representation lineage;
- source/config identity;
- raw-output semantics;
- supported/restricted classes;
- explicit statement that this checkpoint is eligible to become the ZKML teacher.

Do not distill from an intermediate training checkpoint simply because it exists.

## 7. Work package Z2 — new distillation corpus and proxy training

After Z1:

1. Audit how the current corpus extracts teacher fusion embeddings and probabilities.
2. Confirm the corpus uses the accepted teacher’s **inference-equivalent preprocessing contract**.
3. Define train/validation/test separation for proxy fidelity without contaminating R4 model evaluation roles.
4. Regenerate a versioned distillation corpus bound to teacher + DATA identities.
5. Train proxy from fresh initialization unless a specific warm-start policy is justified.
6. Track proxy config, seed, optimizer, epochs and checkpoint identity.

Proxy fidelity evaluation should include:

- per-class regression error;
- rank/correlation behavior where meaningful;
- threshold/tier disagreement against teacher only as a **fidelity** measure, not vulnerability ground truth;
- worst-case errors;
- stability on representative long/multi-contract examples;
- unsupported/restricted class behavior.

## 8. Work package Z3 — proxy acceptance

The proxy is accepted only if a predefined fidelity contract passes.

The acceptance record must distinguish:

- “proxy reproduces teacher outputs sufficiently for the intended proof protocol”

from

- “teacher predictions are correct vulnerability outcomes.”

The former may be proven by distillation evaluation; the latter belongs to ML evaluation and must not be inferred from ZK fidelity.

## 9. Work package Z4 — ONNX and calibration lineage

For the accepted proxy:

- export a fresh ONNX artifact;
- validate input/output dimensions and semantics;
- bind ONNX to proxy and teacher identities;
- generate fresh calibration data/manifests;
- validate external ONNX data if present;
- ensure all hashes/paths/manifests agree;
- reject stale mixed-lineage manifests.

Run native-PyTorch versus ONNX parity tests on a bounded corpus before circuit setup.

## 10. Work package Z5 — EZKL security/settings investigation

**State:** `INVESTIGATING`

The retained settings contain `check_mode="UNSAFE"`. Before making any stronger proof-assurance claim:

1. research the exact semantics of `check_mode` for the **pinned EZKL version used by SENTINEL**, using authoritative EZKL documentation/source where necessary;
2. determine why the existing bundle used `UNSAFE`;
3. determine which safer/production-suitable setting is appropriate, if one exists for this circuit;
4. identify resource/proving/verifier consequences;
5. test whether changing the setting alters circuit identity, public signals, keys or generated verifier;
6. document any residual limitations even after changing it.

Do not assume that replacing the string `UNSAFE` is sufficient security hardening. The entire setup/proof/verifier contract must be revalidated.

## 11. Work package Z6 — fresh EZKL setup bundle

Using accepted ONNX/calibration and the resolved settings policy:

- generate settings;
- calibrate settings;
- compile circuit;
- obtain/generate SRS according to the pinned protocol;
- generate PK/VK;
- record exact EZKL/runtime versions;
- hash every artifact;
- write a versioned setup manifest;
- validate bundle consistency with `validate_bundle.py` or an improved successor;
- preserve old bundle unchanged.

If any step is nondeterministic or environment-sensitive, record that explicitly rather than pretending byte reproducibility.

## 12. Work package Z7 — proof generation and verification

For representative contracts:

1. extract teacher fusion under the accepted preprocessing/model lineage;
2. compute proxy output;
3. generate witness;
4. generate proof;
5. verify off-chain;
6. decode/validate public signals;
7. prove output/public-signal agreement with proxy execution;
8. verify proof/bundle identity metadata.

Required negative tests:

- mutated witness/input;
- mutated proof;
- wrong VK/bundle;
- wrong public signals;
- stale/mismatched teacher/proxy manifest;
- stale circuit/key reuse attempt.

## 13. Work package Z8 — Solidity verifier export/integration

If the new setup requires a new Solidity verifier:

- export it as a new versioned/generated artifact;
- retain provenance/SPDX notice;
- compare interface/public-signal contract with `IZKMLVerifier`/registry expectations;
- build a real-proof Foundry fixture from the exact bundle;
- verify on-chain/off-chain agreement;
- hand off exact verifier/bundle identity to the contracts plan.

Do not manually patch a generated verifier to make tests pass without regenerating/verifying provenance.

## 14. Work package Z9 — V3 proof/context reconciliation

Reconfirm that V3 binds:

- proof hash;
- public signals;
- target code hash;
- agent/round;
- teacher model identity;
- proxy bundle identity;
- DATA version;
- class schema;
- chain + registry domain.

This attestation remains separate from neural proof scope. Public documentation/report fields must preserve both dimensions.

## 15. Stop/fail conditions

Stop and investigate if:

- teacher checkpoint is not accepted;
- distillation preprocessing differs materially from accepted inference/training semantics;
- proxy fidelity is insufficient;
- ONNX differs materially from PyTorch proxy;
- manifests mix teacher/DATA/proxy identities;
- `UNSAFE` setting implications remain unresolved but production-assurance claims are proposed;
- new ONNX is paired with stale PK/VK/circuit without explicit compatibility proof;
- proof public signals do not match proxy outputs;
- generated verifier/interface differs from registry assumptions;
- proof scope is described as proving the full audit.

## 16. Completion criteria

ZKML is `ACCEPTED` when a fresh bundle is bound to the accepted teacher/DATA lineage, proxy fidelity is validated, EZKL settings are understood and justified, proof generation/verification passes, verifier compatibility is proven, and the system still states precisely what the proof does and does not prove. If no repaired teacher is promoted, the current ZKML bundle remains historical and this plan may terminate as `DEFERRED` rather than forcing a rebuild.
