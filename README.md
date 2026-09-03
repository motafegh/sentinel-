# SENTINEL

SENTINEL is a smart-contract security research/engineering system under active development. It combines a Solidity DATA pipeline, a four-eye graph/code ML teacher, multi-tool LangGraph auditing, a distilled EZKL proxy proof, and an upgradeable on-chain audit registry.

## Current system state

Historical R4 **G0–G7 remain PASSED and immutable**. **Phase 8 is IN_PROGRESS; G8 is open and full training remains unauthorized.**

The current DATA/ML authority has moved beyond the historical G7 publication:

- R4-D-008 accepts repaired-v2 physical DATA as immutable reproducibility evidence: **22,540 contracts**, **225,400 contract×class rows**, and **67,620 graph/token/sidecar files**;
- R4-D-009 accepts corrected logical V3 grouping/roles: **22,394 groups**, maximum group size **7**, **146 normalized-code edges**, and zero address-authority edges;
- R4-D-010 preserves graph schema v9 for historical reproduction but makes it ineligible for a new full training run;
- R4-D-011 accepts the exact **V10 V2.6** 22,540-identity physical representation lineage with binding digest `d9f925588913e66476cfbc097bace7daa7e673295fe2a243760313d0bef5ebdd`;
- R4-D-012 promotes `target_aware_guarded_v1` only for construction/evaluation of a **fresh versioned candidate**. The R4-D-011 root remains immutable/current physical authority until that new token lineage is separately generated, bound, reviewed, and accepted;
- confirmed negatives remain **zero**. Candidate #2 has primary-review support only and still requires genuinely independent agreement;
- threshold fitting, calibration fitting, untouched acceptance, model-quality promotion, and the 100-epoch/full training run remain unsupported or unauthorized;
- the existing Run12 teacher remains the **historical operational baseline** and is not repaired R4 truth.

The current work is therefore evidence and physical-lineage closure before any repaired teacher training—not a claim that a new model has already been trained or improved.

## Current architecture

```text
Historical / upstream Solidity
        ↓
repaired physical DATA + evidence/policy controls
        ↓
accepted logical V3 grouping / roles
        ↓
accepted V10 V2.6 physical representation (R4-D-011)
        ↓
[next pending] fresh guarded-selector token lineage + separate acceptance
        ↓
[later, only if authorized] repaired teacher retraining / evaluation

Historical operational runtime today:
Four-eye Run12 teacher → ML API :8001 ─→ AGENTS / LangGraph → gateway :8000 → off-chain report
        ↓
 fusion[128]
        ↓
retained proxy 128→64→32→10 / EZKL proof boundary
        ↓
AuditRegistry V3 protocol (context-attested submission contract)
```

Important runtime separation:

- the **gateway** runs the off-chain 14-node audit and stores a report;
- the live **audit MCP on :8012 is read-only** and exposes version-aware V1/V2/V3 registry queries;
- historical mutable `submit_audit` code remains for compatibility/history but is **not exposed by the live analysis MCP service**;
- V3 defines the current on-chain submission protocol, but signing/broadcast belongs outside the analysis MCP boundary and no production signer/broadcaster is claimed here;
- the retained EZKL proof proves the proxy computation only. V3 adds a separate EIP-712 policy/provenance attestation; it does not make the circuit prove teacher/source/AGENTS execution;
- retained EZKL settings still use `check_mode="UNSAFE"`, which remains a production-assurance limitation.

## Start here

- [Developer setup and environment contract](DEVELOPMENT.md)
- [Progressive developer handbook](docs/handbook/00_README.md)
- [Current status and gaps](docs/handbook/16_current_status.md)
- [Architecture](docs/handbook/01_architecture.md)
- [Runtime flows](docs/handbook/02_runtime_flows.md)
- [DATA pipeline](docs/handbook/03_data_pipeline.md)
- [DATA artifacts / ML seam](docs/handbook/04_data_artifacts.md)
- [Security and trust](docs/handbook/12_security_and_trust.md)
- [Security reporting policy](SECURITY.md)
- [R4 control plane](docs/plan/ml-R4/00_MASTER_PLAN.md)

## Repository map

| Path | Purpose |
|---|---|
| `data_module/` | ingestion, preprocessing, representations, historical labels/exports, and DATA vNext implementation work |
| `ml/` | four-eye teacher architecture, historical training/inference, repaired-training preparation, evaluation tooling, interpretation, MLOps |
| `agents/` | LangGraph orchestration, evidence, RAG, five MCP services, gateway, V3 observation/feedback boundaries |
| `zkml/` | proxy distillation, ONNX, retained EZKL circuit/proof lifecycle |
| `contracts/` | SentinelToken, verifier, UUPS AuditRegistry V1/V2 historical storage plus V3 context-attested protocol |
| `docs/plan/ml-R4/` | active DATA/ML repair plan, evidence ledger, policies, role manifests, gates, decisions, risks |
| `docs/handbook/` | canonical current system documentation; older learning/planning material is subordinate |

## Development model

SENTINEL is a **multi-environment monorepo**. ML, DATA, AGENTS, Contracts, and ZKML have distinct dependency/tooling boundaries; the root Poetry metadata is not a universal environment. Start with [`DEVELOPMENT.md`](DEVELOPMENT.md) before installing dependencies or attempting a full runtime.

## Documentation authority

Executable source is authoritative for behavior. Current machine-readable R4 governance/evidence is authoritative for DATA/ML semantic and gate state. The canonical handbook is the explanatory/navigation layer. Historical plans/reports/learning files remain in the repository for auditability but must not override current source, R4 decisions, or `docs/handbook/16_current_status.md`.

## Minimum documentation verification

```bash
export TMPDIR=/tmp TMP=/tmp TEMP=/tmp
python3 docs/handbook/tools/verify_handbook.py static
python3 docs/handbook/tools/verify_handbook.py inventory
python3 -m unittest discover -s docs/handbook/tools/tests -p 'test_*.py'
```

Large historical DATA/teacher/proving artifacts are not guaranteed in a fresh clone. Do not commit `.env` files, RPC credentials, private keys, mnemonics, or private endpoint values. For suspected vulnerabilities or accidental credential exposure, follow the [security reporting policy](SECURITY.md) rather than publishing sensitive details in an issue.
