# SENTINEL

SENTINEL is a smart-contract security research/engineering system under active development. It combines a Solidity DATA pipeline, a four-eye graph/code ML teacher, multi-tool LangGraph auditing, a distilled EZKL proxy proof, and an upgradeable on-chain audit registry.

## Current system state

The current stable `main` baseline is the R4 **G6-passed** state from merge commit `91f795885` plus later documentation-only reconciliation. R4 repaired the historical DATA/ML label assumptions before any new teacher retraining:

- 22,493 historical contracts were reconstructed as a 224,930-row contract×class evidence ledger;
- historical `0` is no longer treated as a confirmed negative;
- `data-vnext-policy-v1` separates outcome truth from training signal/strength;
- leakage-safe roles are frozen in `r4-vnext-roles-v1`;
- threshold-fit, calibration-fit, and untouched-acceptance roles are intentionally unsupported/empty because the recovered evidence does not justify them;
- the existing Run12 teacher remains the historical operational baseline and has **not yet been retrained on DATA vNext**.

Phase 7 implements the additive DATA vNext v2 semantic overlay on branch `r4/phase7-data-vnext-implementation`. Its remote semantic checks are green, but G7 is not complete until the existing local graph/token representations are physically bound and validated. Until G7 is merged, the v2 implementation branch is candidate work rather than canonical `main` runtime.

## Current architecture

```text
Historical / upstream Solidity
        ↓
DATA + R4 evidence/policy/role controls
        ↓
current historical representations + future DATA vNext v2 semantic overlay
        ↓
Four-eye teacher (Run12 historical baseline today; repaired retrain later)
        ↓
ML API :8001 ───────────────→ AGENTS / LangGraph → gateway :8000 → off-chain report
        ↓
 fusion[128]
        ↓
legacy proxy 128→64→32→10 / EZKL proof boundary
        ↓
AuditRegistry V3 protocol (context-attested submission contract)
```

Important runtime separation:

- the **gateway** runs the off-chain 14-node audit and stores a report;
- the live **audit MCP on :8012 is read-only** and exposes version-aware V1/V2/V3 registry queries;
- historical mutable `submit_audit` code remains for compatibility/history but is **not exposed by the live analysis MCP service**;
- V3 defines the current on-chain submission protocol, but signing/broadcast belongs outside the analysis MCP boundary and no production signer/broadcaster is claimed here;
- the retained EZKL proof proves the proxy computation only. V3 adds a separate EIP-712 policy/provenance attestation; it does not make the circuit prove teacher/source/AGENTS execution.

## Start here

- [Progressive developer handbook](docs/handbook/00_README.md)
- [Current status and gaps](docs/handbook/16_current_status.md)
- [Architecture](docs/handbook/01_architecture.md)
- [Runtime flows](docs/handbook/02_runtime_flows.md)
- [DATA pipeline](docs/handbook/03_data_pipeline.md)
- [DATA artifacts / ML seam](docs/handbook/04_data_artifacts.md)
- [Security and trust](docs/handbook/12_security_and_trust.md)
- [R4 control plane](docs/plan/ml-R4/00_MASTER_PLAN.md)

## Repository map

| Path | Purpose |
|---|---|
| `data_module/` | ingestion, preprocessing, representations, historical labels/exports, and DATA vNext implementation work |
| `ml/` | four-eye teacher architecture, historical training/inference, calibration tooling, interpretation, MLOps |
| `agents/` | LangGraph orchestration, evidence, RAG, five MCP services, gateway, V3 observation/feedback boundaries |
| `zkml/` | proxy distillation, ONNX, retained EZKL circuit/proof lifecycle |
| `contracts/` | SentinelToken, verifier, UUPS AuditRegistry V1/V2 historical storage plus V3 context-attested protocol |
| `docs/plan/ml-R4/` | active DATA/ML repair plan, evidence ledger, policies, role manifests, gates, decisions, risks |
| `docs/handbook/` | canonical current system documentation; older learning/planning material is subordinate |

## Documentation authority

Executable source is authoritative for behavior. The canonical handbook and R4 registers describe current architecture, limitations, and active decisions. Historical plans/reports/learning files may remain in the repository for auditability but must not override current source, R4 decisions, or `docs/handbook/16_current_status.md`.

## Minimum documentation verification

```bash
export TMPDIR=/tmp TMP=/tmp TEMP=/tmp
python3 docs/handbook/tools/verify_handbook.py static
python3 docs/handbook/tools/verify_handbook.py inventory
python3 -m unittest discover -s docs/handbook/tools/tests -p 'test_*.py'
```

Large historical DATA/teacher/proving artifacts are not guaranteed in a fresh clone. Do not commit `.env` files, RPC credentials, private keys, mnemonics, or private endpoint values.
