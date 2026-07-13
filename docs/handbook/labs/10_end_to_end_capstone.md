# L10 — End-to-end ownership capstone

## Learning objective

Demonstrate the off-chain audit and direct ZK/on-chain preflight as separate flows, preserving every identity, artifact, and trust limitation.

## Prerequisites

Complete L01–L09 or equivalent. Read [T10](../technical/10_end_to_end_debugging.md) and [Operations](../14_operations.md).

## Source reading order

Gateway `submit_audit/_run_job` → graph `build_graph` → ML API `predict/fusion_embedding` → audit MCP `_run_submit` → EZKL proof generation/calldata → registry `submitAuditV2` → feedback ingester.

## Setup and artifact requirements

Tier is live. Required: configured ML/AGENTS environments, teacher checkpoint, proxy/circuit artifacts, private proving key and SRS, Foundry/Anvil, deployed token/verifier/registry, funded/staked operator, and non-secret environment configuration. Run preflight before starting.

## Initial observation

```bash
python3 docs/handbook/tools/verify_handbook.py inventory
python3 docs/handbook/tools/verify_handbook.py lab --check L10
```

## Controlled edit

Create a temporary Solidity fixture outside production corpora with one known pattern and record its SHA-256. Do not edit runtime code. If extending automation, add only a disposable smoke test that asserts gateway completion still has an unsubmitted placeholder and that direct submission returns a distinct transaction result.

## Expected success output

Off-chain: gateway 202 → queued/running → completed report, with no claim of chain submission. Direct-chain: fusion embedding has 128 values, proxy has ten outputs, calldata has 138 instances, off-chain verify succeeds, transaction receipt/event/query agree.

## Expected failure output

Missing service/artifact/operator requirement is reported at its boundary. Gateway success without transaction is expected current behavior. Missing proving key/SRS blocks proof; insufficient stake or score mismatch reverts; shared proof files make concurrent live submissions unsafe.

## Verification

Run service probes and Anvil only when launched:

```bash
python3 docs/handbook/tools/verify_handbook.py live --services --anvil
```

Save only non-secret hashes, versions, addresses, transaction IDs, and command results in your study notes.

## Reset and cleanup

Stop services/Anvil, remove the temporary fixture and transient proof input/witness/proof only if generated for this lab, and revert the disposable chain snapshot. Never remove proving/setup artifacts or copy operator secrets into docs.

## Completion rubric

Complete when you can produce two distinct traces, reconcile all shapes/hashes, identify unproved assertions, and explain rollback for each failed boundary.

## Review questions

Which path calls `submit_audit`? What does the proof bind? Which hash algorithm does each surface expose? Why is concurrent submission unsafe now?

## Classification

Live; not safe-preflight eligible; controlled temporary fixture only.
