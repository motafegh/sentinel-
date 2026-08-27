# Phase-8 V10 parse-only resolution working plan

Date: 2026-08-23
Status: **CLOSED — HISTORICAL EXECUTION PLAN; SUPERSEDED AS RESTART AUTHORITY**
Scope: R4-B008 parse-only remediation tranche only; no label, selector, objective, threshold, checkpoint, training, or physical-acceptance authority

> **Current-state pointer (2026-08-27):** Do not resume work from this plan. The 26-contract parse-only remediation described here is complete. The later V2.5 20-identity structural-drift investigation is also complete. The current restart authority is `runs/2026-08-27_PHASE8_v10_v25_current_restart_checkpoint.md`, followed by `runs/2026-08-26_PHASE8_v10_v25_full_candidate_staging.md`.

## Historical starting problem

The V10 diagnostic lineage originally contained 26 DIVE contracts using `slither_parse_only`, including 7 `TRAIN_WEAK` and 19 `TRAIN_UNLABELED` contracts. Their missing IR was material because the sources contained transfer/send/contract-creation syntax.

Root-cause analysis split those identities into:

- 24 contracts affected by Slither's singleton high-level-call destination type defect;
- one contract, `caa35c1a5906269bbe5e70de780d105c2968ece4fc038d7f7208efee681aeec9`, that requires the identity-bound Slither 0.11.5 runtime because primary Slither 0.10.0 cannot complete its full analysis;
- one state-initializer ternary case requiring an exact hash-bound, byte/line-preserving graph-only reconciliation.

Population-wide Slither 0.11.5 was rejected because it materially changed otherwise-stable graph structure. The accepted remediation direction therefore kept Slither 0.10.0 as the primary runtime and isolated the exact 0.11.5 exception.

## Historical remediation contract

The tranche required a new extractor identity rather than changing graph bytes under `v2.3-r4-call-semantics`. The V10-only recovery preserved accepted V9 token bytes and historical artifacts while adding narrowly recorded analyzer/source repairs. Parse-only remained diagnostic fallback only and could not satisfy physical acceptance.

Required invariants were:

- exact primary Slither 0.10.0 for ordinary identities;
- exact identity-bound Slither 0.11.5 for `caa35c...ec9` only;
- zero accepted parse-only outputs;
- zero unclassified call IR;
- zero call-mapping errors;
- classified/emitted/observed call-edge reconciliation;
- exact accepted-V9 token-byte equality;
- no mutation of accepted V9/repaired-v2 artifacts.

## Closure result

This plan's remediation goal **passed** in the V2.4 compatibility lineage:

- protected candidate population: 22,540 identities;
- exact accepted-V9 token bytes;
- zero parse-only outputs;
- zero unclassified call IR;
- required runtime split: 22,539 primary Slither-0.10.0 + one identity-bound Slither-0.11.5 exception;
- V2.4 binding digest: `bd907531a3e22b15d7b91552d15ef1f60c5fd59a109c4ef144ca62f3abab6950`.

The complete transition audit then exposed a **different** blocker: 20 unexpected structural-drift identities outside the historical parse-only-origin set. That later investigation must not be confused with failure of the 26-contract repair.

## Later superseding evidence

The 20-contract structural tranche subsequently advanced to extractor:

`v2.5-r4-call-semantics-deterministic-cfg`

and closed 20/20 under three fresh primary-runtime generations:

- 8 exact node-index-invariant labelled graph-equivalence identities;
- 12 deterministic persistent-storage `CFG_NODE_WRITE` corrections backed by positive semantic evidence;
- `bounded_v25_reproducibility_passed = true`;
- `zero_unexplained_drift = true`;
- blocking identities: none.

Durable closure record:

`reviews/R4-GAP-008/2026-08-26_v10_v25_bounded_structural_closure.md`

Current restart checkpoint:

`runs/2026-08-27_PHASE8_v10_v25_current_restart_checkpoint.md`

Current full-candidate staging protocol:

`runs/2026-08-26_PHASE8_v10_v25_full_candidate_staging.md`

## Historical evidence retained from this tranche

The frozen V2.3 structural reference remains immutable at:

`data_module/data/representations-r4-v3-candidate-v2.3-structural-reference-6087dc6d`

with binding digest:

`6087dc6d76d781efbefe0c4984458d291790c38b1c55d852f48fd796222b0260`

The protected V2.4 diagnostic candidate remains historical evidence and must not be overwritten while constructing the fresh V2.5 lineage.

## Current stop lines

- Do **not** restart the 26-contract parse-only repair unless concrete regression evidence invalidates its closure.
- Do **not** use this dated plan's former “next bounded tranche” as current instructions; that 20-contract work is complete.
- Do not mutate accepted V9/repaired-v2 artifacts, the frozen V2.3 structural reference, or protected V2.4 diagnostic history.
- Do not declare physical acceptance from this historical repair result.
- Do not authorize training. Phase-8 full training remains a separate later gate.
