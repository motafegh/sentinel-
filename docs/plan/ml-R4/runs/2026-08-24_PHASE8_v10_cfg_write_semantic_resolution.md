# Phase-8 V10 CFG WRITE semantic resolution

Date: 2026-08-24
Status: SEMANTIC DIRECTION RESOLVED; V2.5 REPRODUCIBILITY EVIDENCE NEXT
Scope: R4-B008 structural blocker only; no training or model-quality authority

## Inputs

This checkpoint follows:

- `2026-08-23_PHASE8_v10_structural_drift_probe_handoff.md`;
- the exact three-repeat structural probe over the 20 unexpected identities;
- the committed V10-only deterministic CFG guard (`v2.5-r4-call-semantics-deterministic-cfg`);
- the focused V2.5 test run; and
- the protected-local expression/write-evidence probe over the 12 feature/classification-drift identities.

The protected v2.3 structural reference and protected v2.4 candidate remain immutable diagnostic evidence. Training remains unauthorized.

## Focused V2.5 validation

The protected-local run at repository commit `a330cab16bbdf99c7951e1b4d1e0e4f17320f145` executed:

- `data_module/tests/test_representation/test_v10_cfg_determinism.py`;
- `data_module/tests/test_representation/test_graph_schema_v10.py`;
- `data_module/tests/test_representation/test_r4_v10_orchestrator.py`.

Result: **27 passed**, with only the existing `torch_geometric.distributed` deprecation warning.

The deterministic-guard tests explicitly cover the important negative controls:

- a member write rooted in a `memory` local is **not** persistent storage;
- declaring or rebinding a bare `storage` local reference is **not** itself a state mutation;
- V10-only guard activation restores the historical classifier after the guarded call;
- CALL priority remains above WRITE.

## Exact 12-node semantic evidence

`p8_probe_v10_cfg_write_evidence.py` completed under exact Slither 0.10.0 with:

- `contracts_requested = 12`;
- every requested drifting node located (`all_requested_nodes_found = true`);
- no missing target node.

For **every** requested feature/classification-drift statement, the written member/index lvalue is rooted in a `LocalVariable` whose stable Solidity data-location evidence is:

- `location = storage`;
- `is_storage = true`.

The corpus includes storage-rooted member/index mutations such as:

- `server.unregisterCaller`;
- `vpBound.timeLastUpdated`;
- `task.numVotes` / `task.rewardStatus`;
- `s.cashbackbalance` / `s.amountbalance`;
- `_roundID.leadPID` / `_roundID.team`;
- `dispute.*` fields and dynamic-array lengths;
- `p.*` fields and `p.minerCount[0]`;
- `result.requiredForResult`;
- `lot.processIndex`.

Some runs of Slither 0.10.0 expose the corresponding contract state variable through `state_variables_written`; other runs leave that derived list empty. The earlier expression-level lvalue/root evidence remains stable and independently proves that these statements mutate persistent storage.

Therefore the semantic direction is resolved:

> The deterministic V2.5 result for these 12 drifting nodes is `CFG_NODE_WRITE`.

This is **not** an acceptance waiver and is not a hash-specific corpus patch. It is a general semantic correction for member/index writes rooted in a true storage local.

## Consequences for the frozen reference

The frozen v2.3 structural reference was produced through the same unstable Slither 0.10.0 derived alias classification seam. Therefore a frozen row that labels one of these proven storage mutations as `CFG_NODE_OTHER`, `CFG_NODE_READ`, or `CFG_NODE_ARITH` is an under-classified historical analyzer state, not semantic authority that V2.5 must reproduce.

The complete transition decision must distinguish:

1. **8 node-order/index-only identities** — require exact labelled directed-multigraph equivalence modulo node index;
2. **12 deterministic storage-write corrections** — require stable V2.5 `CFG_NODE_WRITE` classification backed by this expression/data-location evidence;
3. any other drift — remains unexplained and blocking.

No broad weakening of structural comparison is authorized.

## Next evidence step

Regenerate the same 20 unexpected identities at least three times using current V2.5 under:

- `ml/.venv/bin/python`;
- exact `slither-analyzer = 0.10.0`;
- `crytic-compile = 0.3.11`;
- `PYTHONPATH=.:data_module`.

Use fresh disposable roots and do not overwrite the protected v2.3 reference or v2.4 candidate.

Acceptance for this bounded repeat requires:

- zero mechanical failures;
- zero repeat-to-repeat semantic/classification drift under V2.5;
- the 8 permutation-only identities remain exact node-identity-aware equivalents;
- the 12 storage-mutation identities remain deterministically WRITE;
- any additional state is blocking.

Only after this reproducibility evidence passes should the full 22,540-identity transition audit be revised to consume the explicit two-part decision above and the affected protected V2.5 lineage be regenerated.

Physical acceptance still requires zero **unexplained** drift plus an explicit decision record. Training remains unauthorized.
