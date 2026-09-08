# SENTINEL

SENTINEL is a smart-contract security and ML/data research system built around a difficult question: **how can a security model or audit pipeline make trustworthy claims when the underlying labels, representations, splits, and evaluation evidence may themselves be wrong or incomplete?**

The project combines Solidity data/evidence work, graph/code ML research, multi-tool auditing, proof experiments, and an on-chain audit-record protocol. The important current result is not a new model-quality claim. It is an evidence-driven repair process that found serious data/grouping/representation problems and continues to block stronger training/promotion claims until the required evaluation evidence exists.

## Project eras and contribution boundary

Sentinel has two important development eras.

The **original project** was Ali's long-running AI-assisted learning/building work across Python, data preparation, repeated ML training, Linux, graph/agent experiments, and dataset/model-quality diagnosis.

The later **R4 continuation** is substantially AI-led research under Ali's direction. It should not be interpreted as independent ownership of the current ML/data pipeline, LangGraph, zkML, blockchain, or full-system implementation. Repository capability, Ali's original hands-on experience, and later AI-led research are intentionally kept separate.

## Current research state

Historical R4 **G0–G7 remain passed and immutable**. **Phase 8 is in progress; G8 is still open.** Run12 remains the historical operational ML baseline rather than current repaired training truth.

The current R4 line has established, among other things:

- repaired-v2 physical DATA accepted across **22,540 contracts** and **225,400 contract×class rows**;
- corrected logical V3 grouping after the earlier address-literal grouping produced an invalid 10,327-contract connected component;
- a hardened evidence snapshot with cross-report coherence checks;
- **zero accepted confirmed-negative examples** so far;
- a real graph-representation defect in historical v9 external-call semantics;
- an accepted V2.6 physical representation lineage for the future candidate path;
- guarded selector policy for a future candidate, while the corresponding new physical token lineage is still not built/accepted;
- threshold fitting, calibration, untouched acceptance, and the 100-epoch Phase-8 run remain unauthorized.

The core research discipline is:

```text
valid physical DATA
≠ valid leakage split
≠ coherent research evidence
≠ sufficient supervision
≠ trustworthy model quality
```

For the exact current state and restart boundary, use [`docs/handbook/16_current_status.md`](docs/handbook/16_current_status.md).

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
