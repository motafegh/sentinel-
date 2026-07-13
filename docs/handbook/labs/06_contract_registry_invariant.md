# L06 — Add a registry invariant and upgrade test

## Learning objective

Add one Foundry test that protects append-only history or upgrade state while preserving storage layout.

## Prerequisites

Read [T06](../technical/06_contracts_storage_upgrades.md). Install Foundry. Use a disposable worktree.

## Source reading order

`contracts/src/AuditRegistry.sol` → `SentinelToken.sol` → `IZKMLVerifier.sol` → tracked `contracts/test/AuditRegistry.t.sol` setup, guard, history, and upgrade tests.

## Setup and artifact requirements

Tier is module. Tracked contracts/tests suffice. The ignored local V2 test can be consulted but cannot become fresh-clone evidence unless separately unignored/fixed outside D1.

## Initial observation

```bash
cd contracts && forge test --match-path 'test/AuditRegistry.t.sol' -vv
```

## Controlled edit

Add a tracked V1 test that submits twice and asserts count, first record, latest record, caller, and proof hash remain consistent. Alternatively extend the upgrade test to create state before upgrade and assert it afterward. Do not modify storage or implementation code.

## Expected success output

The focused test passes and proves append-only history or state preservation across upgrade.

## Expected failure output

Reordering expected records, calling as an unstaked agent, or mismatching public score should revert/fail at the intended guard. Unauthorized upgrade must revert.

## Verification

Run the focused file, then full `forge test`, then `verify_handbook.py lab --check L06`.

## Reset and cleanup

```bash
git restore contracts/test/AuditRegistry.t.sol
```

No deployed chain state is involved unless you explicitly choose the live extension.

## Completion rubric

Complete when the test demonstrates a durable invariant and you can explain why it detects a storage/append regression.

## Review questions

Which guards run before storage append? What state survives proxy upgrade? Why does proof hash not authenticate provenance?

## Classification

Module; safe preflight; controlled tracked-test edit.
