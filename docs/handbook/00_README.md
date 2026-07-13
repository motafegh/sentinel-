# SENTINEL progressive developer handbook

**Read this when:** you are learning, operating, changing, reviewing, or handing over SENTINEL.

**Skip this if:** never on first use; it defines the handbook’s authority and shortest learning paths.

**Estimated reading time:** 6 minutes, plus the selected path.

## 30-second summary

This handbook explains SENTINEL progressively: a one-hour ownership path first, subsystem depth only when needed. Executable source is authoritative. Every page distinguishes implemented, degraded, planned, tracked, regenerated, ignored/private, and local-only behavior. Volatile test counts live only in [current status](16_current_status.md).

## Just-enough mental model

SENTINEL turns Solidity into versioned DATA artifacts, a four-eye ML assessment, multi-tool agent evidence and dual verdicts, an optional proxy ZK proof, and an on-chain audit record. The off-chain gateway and direct proof/submission paths are currently separate.

### Authority rules

1. Executable `.py`, `.sol`, `.sh`, and configuration behavior is current truth.
2. This handbook is the canonical learning/navigation layer and is checked against selected source facts.
3. ADRs explain decisions; reports/testing specs provide bound evidence; plans and experiments are historical unless explicitly active.
4. A local file is not a fresh-clone artifact unless Git tracks it.
5. A passing test proves the checked behavior, not product quality or end-to-end security.
6. No `.env`, RPC credential, private key, or private endpoint value belongs in documentation.

## Actual runtime/source walkthrough

Start at architecture, choose the runtime flow, then follow the producer-to-consumer boundary relevant to your task. Each chapter provides source-symbol anchors, current limitations, a change recipe, and smoke/module/live verification.

### Page index

| Page | Owns |
|---|---|
| [01 Architecture](01_architecture.md) | topology, processes, ports, trust boundaries |
| [02 Runtime flows](02_runtime_flows.md) | off-chain, direct ZK/on-chain, feedback |
| [03 DATA pipeline](03_data_pipeline.md) | ten stages and lineage |
| [04 DATA artifacts](04_data_artifacts.md) | schema, exports, splits, dataset seam |
| [05 ML model/inference](05_ml_model_inference.md) | four-eye teacher and HTTP API |
| [06 ML training/quality](06_ml_training_quality.md) | loss, calibration, MLOps, interpretation |
| [07 ZKML](07_zkml.md) | proxy, EZKL, proof semantics |
| [08 Contracts](08_contracts.md) | token, registry, verifier, UUPS |
| [09 AGENTS orchestration](09_agents_orchestration.md) | state, 14-node graph, evidence/verdicts |
| [10 AGENTS services](10_agents_services.md) | five MCPs, RAG, gateway, feedback |
| [11 Cross-module contracts](11_cross_module_contracts.md) | shapes, versions, hashes, boundaries |
| [12 Security and trust](12_security_and_trust.md) | injection, Rule 5C, secrets, ZK limits |
| [13 Evaluation](13_evaluation.md) | DATA/ML/AGENTS evidence and gates |
| [14 Operations](14_operations.md) | setup, startup, smoke/live, troubleshooting |
| [15 Change playbooks](15_change_playbooks.md) | blast-radius recipes |
| [16 Current status](16_current_status.md) | commit-bound tests, gaps, availability |
| [17 Reference](17_reference.md) | glossary, symbols, configs, artifacts, history |

### Learning paths

- Core ownership (~1 hour): `00 → 01 → 02 → 11 → 12 → 16`
- Run the system: `01 → 02 → 14 → 16`
- DATA/ML/ZK: `03 → 04 → 05 → 06 → 07 → 11 → 13`
- Agents: `09 → 10 → 11 → 12 → 13`
- Blockchain: `07 → 08 → 11 → 14`
- Maintainer: `15 → 17`

Read each page’s 30-second summary first. Stop after the mental model if you are not changing that subsystem. Read source walkthrough and interfaces before editing. Use deep references only when the task needs their depth.

### Technical guide and lab map

| Area | Deep guide | Ownership lab |
|---|---|---|
| DATA lifecycle | [T01](technical/01_data_pipeline_internals.md) | [L01](labs/01_data_fixture_representation.md) |
| DATA/ML artifact seam | [T02](technical/02_data_representation_export.md) | [L02](labs/02_export_dataset_seam.md) |
| Model/inference | [T03](technical/03_ml_model_inference_internals.md) | [L03](labs/03_ml_tensor_api_trace.md) |
| Training/quality | [T04](technical/04_ml_training_quality_mlops.md) | [L04](labs/04_training_calibration_promotion.md) |
| ZKML | [T05](technical/05_zkml_proof_lifecycle.md) | [L05](labs/05_zkml_witness_signals.md) |
| Contracts | [T06](technical/06_contracts_storage_upgrades.md) | [L06](labs/06_contract_registry_invariant.md) |
| AGENTS orchestration | [T07](technical/07_agents_orchestration_evidence.md) | [L07](labs/07_agents_state_fusion.md) |
| Services/RAG/gateway | [T08](technical/08_services_rag_gateway.md) | [L08](labs/08_services_rag_gateway_recovery.md) |
| Security/evaluation | [T09](technical/09_security_evaluation_trust.md) | [L09](labs/09_injection_rule5c_reliability.md) |
| End-to-end debugging | [T10](technical/10_end_to_end_debugging.md) | [L10](labs/10_end_to_end_capstone.md) |

## Interfaces, data shapes, and configuration

[`_meta/handbook.toml`](_meta/handbook.toml) is the documentation interface: page registry, required sections, source ownership, ports/routes, critical constants, test tiers, and artifact classifications. [`tools/verify_handbook.py`](tools/verify_handbook.py) checks it without third-party packages.

## Failure modes and current limitations

- Static checks cover declared critical facts, not every possible semantic drift.
- The handbook documents real failing tests and disconnected behavior; it does not repair product defects.
- Historical references may be stale even when useful.
- Some commands require local artifacts not supplied by Git; read prerequisites before running them.

## Common change recipe

When source changes, update the owning chapter, cross-module page if a boundary changed, status if evidence changed, and metadata if a declared fact/path changed. Run static and inventory checks before review.

## Verification commands

```bash
python3 docs/handbook/tools/verify_handbook.py static
python3 docs/handbook/tools/verify_handbook.py inventory
python3 docs/handbook/tools/verify_handbook.py lab --list
python3 docs/handbook/tools/verify_handbook.py lab --check-all-safe
python3 -m unittest discover -s docs/handbook/tools/tests -p 'test_*.py'
```

## Optional deep references

- [Reference registry](17_reference.md)
- [Current status](16_current_status.md)
- [`handbook.toml`](_meta/handbook.toml)

## Technical mastery layer

### Prerequisite knowledge

Basic Python, shell, Git, HTTP/JSON, tensors, and Solidity are assumed. Each technical guide refreshes the difficult graph, PyTorch, ZK, distributed-state, or upgrade concepts it uses.

### Source map and reading order

Use the one-hour core path first. Then choose a subsystem pair: canonical chapter → matching guide in [`technical/`](technical/) → matching lab in [`labs/`](labs/). The ten guide/lab pairs are registered in [`handbook.toml`](_meta/handbook.toml), so the validator can detect missing learning coverage.

### Execution trace and worked example

The capstone trace is [T10](technical/10_end_to_end_debugging.md): one Solidity contract becomes an off-chain report and, through a separately invoked path, a proxy proof/on-chain record. It identifies every shape, hash, port, and current disconnection.

### Implementation practice

Labs make controlled changes in tests or temporary fixtures, run focused checks, show an intentional failure, and restore the edit. Start with [L01](labs/01_data_fixture_representation.md); run `python3 docs/handbook/tools/verify_handbook.py lab --list` to choose prerequisites honestly.

### Review and ownership check

You own a topic when you can trace inputs to outputs, name the source symbols, predict success/failure behavior, implement a test-first change, identify regenerated artifacts, and roll back without weakening a gate.
