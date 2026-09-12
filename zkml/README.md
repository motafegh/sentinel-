# SENTINEL ZKML Module

`zkml/` owns the retained teacher-to-proxy distillation boundary, ONNX/EZKL circuit artifacts, proof generation/verification helpers, and proxy/circuit lineage.

> **Current trust statement:** the retained proof proves only the fixed 128→10 proxy computation. It does **not** prove the Solidity source, teacher execution, AGENTS verdict, DATA provenance, chain, target, or audit round. V3 binds that surrounding context separately with an EIP-712 policy attestation.

## Current retained proxy

```text
teacher fusion[128]
      ↓
Linear 128→64 → ReLU → Linear 64→32 → ReLU → Linear 32→10
      ↓
ten proxy scores
      ↓
EZKL proof
```

Current retained facts:

- proxy parameters: **10,666**;
- circuit version: `v2.0`;
- public signals: 128 inputs + 10 outputs = **138**;
- output slots: indices 128–137;
- fixed-point scale: `2^13 = 8192`;
- proxy output semantics: direct regression toward `sigmoid(teacher_logits)`; do not apply a second sigmoid;
- retained proof scope: `legacy_proxy_only_unbound`;
- EZKL settings retain `check_mode="UNSAFE"`.

The retained bundle is valuable for reproducibility/regression but is not by itself a fully context-bound production audit statement.

## V3 relationship

V3 is already merged into the current system. It deliberately keeps two independent checks:

1. EZKL verifier validates the proxy proof/public signals;
2. an authorized policy signer authenticates the exact V3 audit context with EIP-712.

The V3 request binds agent, target/runtime bytecode hash, chain/registry, round, teacher hash, proxy-bundle hash, DATA-version hash, class-schema hash, proof hash, public-signal hash, ten-score hash, and deadline.

See:

- `agents/src/security/policy_signer.py`
- `contracts/src/AuditRegistry.sol::submitAuditV3`

The policy signature authenticates context/provenance. It does not expand the ZK circuit statement.

## R4 / future regeneration

Run12 remains the historical operational teacher. Historical R4 G0–G7 remain PASSED and **Phase 8 is IN_PROGRESS**. The DATA/representation path has already advanced beyond historical G7:

- R4-D-011 accepts the exact V10 V2.6 physical representation lineage;
- R4-D-012 promotes `target_aware_guarded_v1` only for a fresh versioned successor candidate that still requires separate physical acceptance;
- full training remains unauthorized;
- no repaired teacher has been trained or selected;
- confirmed negatives remain zero and threshold/calibration/untouched-acceptance support remains unavailable.

Therefore **do not regenerate/promote a new proxy/circuit yet** merely because the DATA/representation lineage changed.

The correct future order is:

```text
R4-D-011 accepted V10 V2.6 physical representation
→ R4-D-012 guarded-selector successor candidate
→ binding + transition evidence + separate physical acceptance
→ objective/evaluation support + explicit training authorization
→ repaired teacher retrain
→ evaluate/select teacher candidate
→ redistill proxy
→ remeasure teacher/proxy agreement
→ regenerate ONNX/settings/circuit/keys/verifier as required
→ bind new proxy-bundle identity into V3
→ integration/deployment validation
```

Old proxy agreement cannot automatically transfer to a new teacher even when fusion remains 128 values.

## Main source/artifacts

```text
zkml/src/distillation/      proxy definition/training/export
zkml/src/ezkl/              setup/proof/calldata inspection helpers
zkml/src/validation/        bundle integrity validation
zkml/models/                retained proxy/ONNX artifacts
zkml/ezkl/                  retained settings/compiled/VK/proof artifacts
contracts/src/ZKMLVerifier.sol  canonical generated verifier
```

`extract_calldata.py` is read-only. ZKML helpers must not reintroduce direct private-key transaction submission around the isolated V3 policy/signing boundary.

## Verification

```bash
export TMPDIR=/tmp TMP=/tmp TEMP=/tmp
ml/.venv/bin/python -m pytest zkml/tests -q
cd contracts && forge test
cd .. && python3 docs/handbook/tools/verify_handbook.py static
```

Live proof regeneration additionally requires local cryptographic/data prerequisites that are not guaranteed in a fresh clone.

## Permanent boundaries

- fusion input dimension = 128;
- proxy output count = 10;
- public signals = 138;
- proof statement remains proxy-only;
- V3 attestation remains a separate context/authentication claim;
- `UNSAFE` check mode remains an explicit production-assurance limitation;
- new proxy/circuit artifacts require a selected repaired teacher and fresh lineage/proof/verifier validation;
- R4 physical representation or selector acceptance does not by itself authorize proxy regeneration.

For current detail, see [ZKML handbook](../docs/handbook/07_zkml.md), [contracts](../docs/handbook/08_contracts.md), [security/trust](../docs/handbook/12_security_and_trust.md), and [current status](../docs/handbook/16_current_status.md).
