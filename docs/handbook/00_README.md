# SENTINEL progressive developer handbook

**Read this when:** you are learning, operating, changing, reviewing, or handing over SENTINEL.

**Skip this if:** never on first use; it defines documentation authority and the shortest safe learning paths.

**Estimated reading time:** 6 minutes, plus the selected path.

## 30-second summary

This handbook is the canonical current learning/navigation layer for SENTINEL. Historical R4 G0–G7 remain PASSED and immutable; Phase 8 is IN_PROGRESS and G8/full training remain unauthorized. The current DATA/ML authority includes accepted repaired-v2 physical DATA, accepted logical V3 grouping/roles, and the R4-D-011 V10 V2.6 physical representation lineage. R4-D-012 permits guarded selection only for a fresh versioned candidate that has not yet been physically accepted. Historical Run12/v1/v9 behavior is still preserved for reproducibility, but it must not be confused with current R4 training truth or the current V3 trust model. Volatile state belongs in [current status](16_current_status.md).

## Just-enough mental model

```text
Solidity/data
   ↓
repaired physical DATA + R4 evidence/policy controls
   ↓
accepted logical V3 grouping / roles
   ↓
accepted V10 V2.6 physical representation (R4-D-011)
   ↓
[next pending] fresh guarded-selector token lineage + separate acceptance
   ↓
[later, only if authorized] repaired teacher retraining / evaluation

historical operational runtime today:
Four-eye Run12 teacher / ML API
   ↓
AGENTS 14-node off-chain audit → gateway report

separate trust/protocol boundary:
ML fusion[128] → retained proxy/EZKL proof → V3 context-attested registry protocol
```

The live audit MCP is read-only. V3 submission signing/broadcast is outside the analysis MCP security domain. The retained proof proves the proxy computation only; the V3 policy signature separately binds context/provenance. Retained EZKL settings still use `check_mode="UNSAFE"`.

### Authority rules

1. Executable `.py`, `.sol`, `.sh`, configuration, and committed machine-readable policy/manifests are behavioral truth.
2. Current machine-readable R4 governance/evidence controls DATA/ML semantic and gate state.
3. This handbook is the canonical current explanatory/navigation layer.
4. `docs/plan/ml-R4/` is the controlling DATA/ML repair record for evidence, decisions, roles, risks, and gate state.
5. ADRs explain decisions; historical plans/reports/experiments remain evidence/history unless explicitly active.
6. Historical DATA v1 labels, Run12 thresholds, V1/V2 registry writes, v9 representations, and old submission code remain reproducibility artifacts—not current authority for new work.
7. A local file is not a fresh-clone artifact unless Git tracks it.
8. A passing test proves the checked behavior, not product quality or end-to-end security.
9. No `.env`, RPC credential, private key, mnemonic, or private endpoint value belongs in documentation.

## Actual runtime/source walkthrough

Start with architecture and current status. Then choose one of the two important tracks:

- **runtime/audit track:** gateway → LangGraph → evidence/report, plus read-only V1/V2/V3 registry observation;
- **DATA/ML repair track:** historical evidence → repaired-v2 physical DATA → accepted logical V3 → accepted V10 V2.6 physical representation → guarded-candidate construction/acceptance → only later, if authorized, retraining/evaluation.

### Page index

| Page | Owns |
|---|---|
| [01 Architecture](01_architecture.md) | topology, processes, ports, V3 trust boundaries |
| [02 Runtime flows](02_runtime_flows.md) | off-chain report, read-only registry observation, V3 protocol boundary, feedback |
| [03 DATA pipeline](03_data_pipeline.md) | historical lifecycle plus R4/DATA vNext authority |
| [04 DATA artifacts](04_data_artifacts.md) | historical vs current DATA/representation semantics, roles, ML seam |
| [05 ML model/inference](05_ml_model_inference.md) | four-eye teacher and current Run12 inference status |
| [06 ML training/quality](06_ml_training_quality.md) | historical training mechanics and repaired-retrain constraints |
| [07 ZKML](07_zkml.md) | retained proxy proof scope and V3 context binding |
| [08 Contracts](08_contracts.md) | token, V1/V2 history, V3 protocol, verifier, UUPS |
| [09 AGENTS orchestration](09_agents_orchestration.md) | state, 14-node graph, evidence/verdicts |
| [10 AGENTS services](10_agents_services.md) | five MCPs, read-only audit MCP, RAG, gateway, feedback |
| [11 Cross-module contracts](11_cross_module_contracts.md) | DATA/ML/ZK/V3 compatibility boundaries |
| [12 Security and trust](12_security_and_trust.md) | injection, Rule 5C, proof/attestation/signing boundaries |
| [13 Evaluation](13_evaluation.md) | R4 role limitations, ML/AGENTS evidence and gates |
| [14 Operations](14_operations.md) | safe startup, verification, artifact/local-only boundaries |
| [15 Change playbooks](15_change_playbooks.md) | versioned DATA/ML/V3 blast-radius recipes |
| [16 Current status](16_current_status.md) | canonical current gates, blockers, availability |
| [17 Reference](17_reference.md) | glossary, symbols, configs, artifacts, document classification |

### Learning paths

- Core ownership: `00 → 01 → 02 → 11 → 12 → 16`
- DATA/ML repair: `03 → 04 → 06 → 13 → 16 → docs/plan/ml-R4`
- Runtime/operator: `01 → 02 → 10 → 14 → 16`
- ZK/contracts: `07 → 08 → 11 → 12 → 16`
- AGENTS: `09 → 10 → 11 → 12 → 13`

Technical guides and labs remain useful for code-reading/practice, but they are **supplementary**. Some exercises were authored against the pre-R4/pre-V3 baseline. Where a guide/lab conflicts with a canonical chapter, current source, R4 policy, or current status, the canonical/current material wins.

## Interfaces, data shapes, and configuration

[`_meta/handbook.toml`](_meta/handbook.toml) is the machine-readable documentation interface: canonical pages, ports/routes/tools, critical shapes, source ownership, artifact classification, and test tiers. [`tools/verify_handbook.py`](tools/verify_handbook.py) checks declared facts against source.

## Failure modes and current limitations

- Static documentation checks cannot prove every semantic claim.
- Historical material can remain valuable while being operationally superseded.
- Phase 8 remains IN_PROGRESS; the accepted V10 V2.6 physical lineage does not authorize full training.
- R4-D-012's guarded selector still lacks a separately accepted physical token lineage.
- Confirmed negatives remain zero; candidate #2 still requires genuinely independent agreement.
- Run12 remains the historical operational teacher; no repaired retrain has been promoted yet.
- Threshold-fit, calibration-fit, and untouched-acceptance roles remain unsupported/empty.

## Common change recipe

When behavior changes, update the owning canonical chapter, cross-module/security pages if a boundary changed, `16_current_status.md`, and handbook metadata/validator assumptions. If the change is DATA/ML semantic, update the R4 decision/risk/gate artifacts first. Do not make old learning material silently authoritative again.

## Verification commands

```bash
python3 docs/handbook/tools/verify_handbook.py static
python3 docs/handbook/tools/verify_handbook.py inventory
python3 docs/handbook/tools/verify_handbook.py lab --list
python3 -m unittest discover -s docs/handbook/tools/tests -p 'test_*.py'
```

## Optional deep references

- [Current status](16_current_status.md)
- [Reference registry](17_reference.md)
- [R4 master plan](../plan/ml-R4/00_MASTER_PLAN.md)
- [`handbook.toml`](_meta/handbook.toml)

## Technical mastery layer

### Prerequisite knowledge

Basic Python, shell, Git, HTTP/JSON, tensors, Solidity, and evidence/provenance concepts are assumed. The handbook introduces the project-specific meanings of label state, training strength, proof scope, policy attestation, and dataset role.

### Source map and reading order

Read current status before acting on an older guide. For implementation work, follow canonical chapter → executable source → R4/ADR decision if applicable → focused tests. Technical guides/labs are practice aids, not a stronger source of current operational truth.

### Execution trace and worked example

A current DATA/ML trace is: historical evidence → repaired-v2 physical DATA → accepted logical V3 role/group authority → accepted V10 V2.6 physical representation → fresh guarded-selector candidate → separate physical acceptance → only then later objective/evaluation/training authorization work. A current chain trace is: proxy proof artifacts + fully bound V3 request → isolated policy attestation → `AuditRegistry.submitAuditV3`; the analysis MCP itself only reads registry state.

### Implementation practice

Before editing, identify whether you are changing historical compatibility, the repaired R4 path, live analysis runtime, or the V3 submission protocol. Preserve old artifacts in place and introduce versioned new behavior unless an approved migration says otherwise.

### Review and ownership check

Can you distinguish historical v1/Run12/v9/V1-V2 compatibility from the current R4/V3 direction, name which components are canonical today, and identify what remains blocked before retraining/promotion?
