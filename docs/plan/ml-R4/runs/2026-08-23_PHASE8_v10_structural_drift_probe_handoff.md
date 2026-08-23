# Phase-8 V10 structural-drift probe handoff

Date: 2026-08-23
Status: PROTECTED-LOCAL REPEAT EVIDENCE COMPLETE; DETERMINISTIC CFG REPAIR IN PROGRESS
Scope: R4-B008 structural blocker only; no label, selector, objective, threshold,
checkpoint, training, or model-quality authority

## Authority and inherited boundary

This tranche continues the v2.4 checkpoint in
`2026-08-23_PHASE8_v10_parse_only_resolution_working_plan.md` and the machine
record `reviews/R4-GAP-008/v10_transition_audit_v2.json`.

Do not restart the completed 26-contract parse-only repair. The protected v2.4
candidate already has 22,540 identities, exact accepted-V9 token bytes, zero
parse-only outputs, zero unclassified call IR, and the required runtime split.
Training remains unauthorized.

The exact frozen structural reference remains protected and immutable at:

`data_module/data/representations-r4-v3-candidate-v2.3-structural-reference-6087dc6d`

with binding digest:

`6087dc6d76d781efbefe0c4984458d291790c38b1c55d852f48fd796222b0260`

The current protected v2.4 candidate remains:

`data_module/data/representations-r4-v3-candidate`

Do not overwrite either root during bounded repair evidence.

## Repository-side diagnostic

`docs/plan/ml-R4/scripts/p8_probe_v10_structural_drift.py` is a fail-closed
structural probe. It does not alter `p8_audit_v10_transition.py` acceptance
semantics.

For edges through unchanged type 10 it compares a labelled directed multigraph
where every node label contains the complete persisted `node_metadata` record
and exact feature row. Edge direction, edge type, and multiplicity are preserved.
Search-limit exhaustion is `INCONCLUSIVE_FAIL_CLOSED`, never semantic equivalence.

The focused tests prove that the diagnostic:

- accepts exact labelled graph isomorphism when only indistinguishable node
  indices are permuted;
- rejects feature/classification drift such as
  `CFG_NODE_OTHER -> CFG_NODE_WRITE` even when raw unchanged-edge topology is
  identical;
- rejects genuine unchanged-edge topology changes.

## Protected-local environment facts

The primary repeat runtime is now explicitly resolved and recorded:

- interpreter: `ml/.venv/bin/python`;
- `slither-analyzer = 0.10.0`;
- `crytic-compile = 0.3.11`;
- invocation requires `PYTHONPATH=.:data_module` because the R4 orchestrator
  imports both `sentinel_data` and `ml.src`.

`data_module/.venv` intentionally carries Slither 0.11.5 and must not be
repurposed for the primary repeat population. Its stale `slither` console-script
shebang is unrelated to this tranche.

## 2026-08-23 protected-local repeat result

The exact 20 unexpected identities from transition audit v2 were regenerated
three independent times under `ml/.venv/bin/python`, exact Slither 0.10.0, with
`PYTHONPATH=.:data_module`.

All three runs reported:

- 20 / 20 requested records;
- `passed = true`;
- 20 / 20 `slither_full_analysis`;
- zero token-byte, unclassified-call, call-mapping, or classified/emitted-count
  mechanical failures.

The strict node-identity-aware report was written locally to:

`docs/plan/ml-R4/reviews/R4-GAP-008/v10_structural_drift_repeat_probe_v1.json`

and returned exit code 2 because semantic/classification blockers remain. Its
classification census is:

| Decision | Identities |
|---|---:|
| `NODE_ORDER_INDEX_NONDETERMINISM_PROVEN` | 8 |
| `SLITHER_FEATURE_CLASSIFICATION_NONDETERMINISM_PROVEN` | 8 |
| `CANDIDATE_ONE_OFF_DRIFT_REPEAT_MATCHES_REFERENCE` | 3 |
| `UNRESOLVED_MULTIPLE_REPEAT_STATES` | 1 |

The eight exact labelled-isomorphism identities are:

- `047f9d7c2db3d6ba43b62e9c1b35adb1ed5a6bd36d68da46e0f877b0974b73e4`
- `5a626b8baef72b243f1812118862af26ea796462c38e900f2f595ba73b55495e`
- `7e9bfccd7d3ed5076b7ea61fe444f33b50deb78c27582bcc413b7303422dc551`
- `8b1792cb3c0a40a4ebeec72ffe69d00920c80203213f24ec3e2d5a867eeae3d5`
- `a4068383ed30b56a39771e1dcbe835726242c164d125908207b4e616030aaa8c`
- `a7faec46ab38dbf5b87b1e1ef0e56fc5da743ac450535b1cc09f12922c86f46c`
- `beaa4d742f0b52b301fc2f143072b57ef8170540cdaaa096cdd2f51b047ab1ca`
- `dcf66533d7ee72d2a59ab07fd4117f03ea035489f53ac9184e1dcb4d82c6d823`

For these eight, exact node semantic labels and the complete directed typed
multigraph through edge type 10 are isomorphic despite raw index changes. This
proves index/order nondeterminism rather than semantic graph drift. Physical
acceptance is not yet changed; the complete audit must later consume an equally
strict node-index-invariant rule and an explicit decision record.

The remaining twelve identities contain feature/metadata classification drift:

- `1d9ce79b93c3a1bd7597a76204ae65027fc0471517b2a247c1e536a260c296fd`
- `42489184f712d85f392a47db45110b4406436bc8b648524300a18319111ab350`
- `48beaa23f916dfd3acbc86a799b0859709b29defa3796450693bab13f8e6e777`
- `6376d572b974fb2ba2c074bf7d43972b241a1000731563b44ee09ef72eeaca3e`
- `73cbc254caad8a7a6b8674125c029a530973459c294c6897dca01d219307c669`
- `83c9d2d26dc19eaa2aee29fa7aedb4f4e208429a96cc7a0ffee7491b9830630d`
- `85c5c0d173dbaed126f4bc5165c7453262b6ff50c89c52a458034f322a06a714`
- `95f7d52dff443cc825e20477a62de371cc4bbc31b6ba5aae653ff51caaaf974c`
- `af947a2b1a6d7c6fa500f5604bc7b3d3e8bbab6711c30b54f601b6db5db19464`
- `c159c57b830cb77686cb5a2a7b40f1452cae516ec688b55331adb61b1669064d`
- `c1d21cda50fb1f0c1194392080a2c7a21b3baed5edbe140caaec8c3f257f756b`
- `f8cd4abd34d1aa1ca5d3293ee286bfedfa39d0cab550a547ba22e798d5439b35`

The fluctuating statements are overwhelmingly assignments or updates to member
paths such as `server.unregisterCaller`, `vpBound.timeLastUpdated`, `task.*`,
`s.*`, `_roundID.*`, `dispute.*`, `p.*`, `result.*`, and `lot.*`. Observed
classes alternate among `CFG_NODE_WRITE` and lower-priority
`CFG_NODE_OTHER` / `CFG_NODE_READ` / `CFG_NODE_ARITH` states.

Eight identities alternate directly between reference-like and candidate-like
feature states. Three current-candidate drifts (`95f7d5...`, `af947a...`, and
`f8cd4a...`) disappeared in all three repeats, proving the current v2.4 bytes are
a one-off Slither classification state rather than a reproducible extractor
result. `c1d21c...` produced the reference state once and additional non-reference,
non-candidate states twice, proving more than two upstream classification states
can occur.

## Root-cause seam established from source

The current `_cfg_node_type()` in
`data_module/sentinel_data/representation/graph_extractor.py` gives WRITE
priority when `slither_node.state_variables_written` is non-empty, with only a
direct `StateVariable` IR-lvalue fallback.

Slither 0.10.0 computes `state_variables_written` after converting member/index
writes through mutable `ReferenceVariable.points_to_origin` chains. The repeated
real-corpus evidence shows this derived alias result is not stable for the
member-path statements above.

Slither's earlier expression analysis independently exposes
`variables_written_as_expression`. Its write visitor records lvalue identifiers,
member bases, and index bases before SlithIR reference resolution. A
`LocalVariable` also exposes `is_storage`, which distinguishes persistent-storage
references from `memory`/`calldata` locals.

This provides a source-backed deterministic repair seam: supplement, rather than
replace, Slither's state-write result with expression-level evidence that the
written lvalue is either a `StateVariable` or a `LocalVariable` whose
`is_storage` is true. The node's own storage-reference declaration must be
excluded so `Struct storage alias = stateStruct` is not falsely treated as a
state mutation; later `alias.field = ...` mutations are state writes.

## Versioned deterministic repair decision

Do not change emitted graph semantics under extractor identity
`v2.4-r4-call-semantics-compat`. The current v2.4 candidate is retained as
protected diagnostic evidence.

The next extractor identity must advance to a v2.5 deterministic-CFG revision.
The intended change is deliberately narrow:

1. preserve CALL as the highest CFG-node priority;
2. preserve every existing Slither-confirmed state write;
3. add a deterministic expression-level persistent-storage-write fallback;
4. exclude the local variable being introduced by the node's own storage
   declaration;
5. keep READ, ARITH, CHECK, and OTHER precedence unchanged after WRITE;
6. add focused tests for direct state writes, storage-reference member writes,
   memory-member writes, and storage-reference declarations;
7. do not hash-special-case the twelve corpus identities.

This is a semantic correction, not an acceptance waiver. Some frozen v2.3 nodes
may themselves be the under-classified nondeterministic state and can therefore
legitimately differ from the deterministic v2.5 result. Any such difference must
be explicitly enumerated and justified after regenerated v2.5 evidence; it must
not be silently ignored by the transition audit.

## Next evidence sequence

After the v2.5 source/tests are committed:

1. regenerate the same 20 identities at least three times under exact primary
   Slither 0.10.0 into disposable roots using `ml/.venv/bin/python` and
   `PYTHONPATH=.:data_module`;
2. use one v2.5 repeat as the candidate and the other repeats as reproducibility
   evidence against the same frozen v2.3 reference;
3. require zero unexplained repeat-to-repeat semantic drift;
4. explicitly enumerate any deterministic v2.5 correction relative to the
   frozen reference and bind it to source-level classification evidence;
5. then strengthen the complete transition audit to consume exact
   node-index-invariant graph equivalence for the eight proven permutation-only
   cases and the explicitly reviewed v2.5 classification correction rule;
6. regenerate only the affected protected candidate lineage as required, then
   rerun the complete 22,540-identity transition audit against the same frozen
   reference;
7. create a physical-acceptance decision record only after zero unexplained
   drift remains.

Training and model-quality claims remain unauthorized throughout this sequence.
