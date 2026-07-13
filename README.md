# SENTINEL

SENTINEL is a decentralized smart-contract security oracle under active development. It combines a ten-stage Solidity data pipeline, a four-eye graph/code ML model, multi-tool LangGraph auditing, a distilled EZKL proxy proof, and an upgradeable on-chain audit registry.

Current architecture:

```text
DATA → teacher training → ML API :8001
                         ↓
client → gateway :8000 → 14-node AGENTS audit → off-chain report
                         ↓ selected MCP services :8010–8014

explicit audit MCP submission → ML fusion[128] → proxy/EZKL → AuditRegistryV2
```

The gateway audit and direct ZK/on-chain submission are currently separate flows. A gateway report’s initial on-chain block is an unsubmitted placeholder; see [runtime flows](docs/handbook/02_runtime_flows.md) and [current status](docs/handbook/16_current_status.md) before making deployment claims.

## Start here

- [Progressive developer handbook](docs/handbook/00_README.md)
- [Technical deep guides](docs/handbook/technical/01_data_pipeline_internals.md)
- [Guided ownership labs](docs/handbook/labs/01_data_fixture_representation.md)
- [Architecture](docs/handbook/01_architecture.md)
- [Operations](docs/handbook/14_operations.md)
- [Commit-bound status and known gaps](docs/handbook/16_current_status.md)

## Minimum verification setup

From the repository root:

```bash
export TMPDIR=/tmp TMP=/tmp TEMP=/tmp
python3 docs/handbook/tools/verify_handbook.py static
python3 docs/handbook/tools/verify_handbook.py inventory
python3 docs/handbook/tools/verify_handbook.py lab --list
python3 -m unittest discover -s docs/handbook/tools/tests -p 'test_*.py'
```

Runtime environments are module-specific: Poetry for `agents/`, the existing `ml/.venv` for ML/ZKML, the DATA environment for `data_module/`, and Foundry for `contracts/`. Large DATA/teacher artifacts and private proving/chain prerequisites are not guaranteed in a fresh clone; follow the acquisition classifications in [operations](docs/handbook/14_operations.md).

## Repository map

| Path | Purpose |
|---|---|
| `data_module/` | ingest through freshness, graph/token artifacts, labels/splits/exports |
| `ml/` | four-eye model, training, inference, calibration, interpretation, MLOps |
| `agents/` | LangGraph, evidence fusion, RAG, five MCP servers, gateway, evaluation |
| `zkml/` | proxy distillation, ONNX, EZKL circuit/proof lifecycle |
| `contracts/` | SentinelToken, verifier, UUPS AuditRegistry V1/V2 |
| `docs/handbook/` | canonical progressive system documentation |

No `.env`, RPC credential, private key, or private endpoint value should be committed or copied into documentation.
