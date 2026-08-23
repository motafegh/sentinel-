# Phase-8 V10 parse-only resolution working plan

Date: 2026-08-23
Status: IN PROGRESS
Scope: R4-B008 only; no label, selector, objective, threshold, checkpoint, or
training authorization

## Accepted starting boundary

The committed V10 diagnostic candidate contains all 22,540 accepted logical-V3
identities and passes population, token-byte, schema, and call-count binding.
Physical acceptance is rejected because 26 DIVE contracts use
`slither_parse_only`, including 7 `TRAIN_WEAK` and 19 `TRAIN_UNLABELED`
contracts. Their missing IR is material: the sources contain 85 transfer, 14
send, and 14 contract-creation lexical hits.

The existing V10 candidate and its committed transition report remain
diagnostic history. This tranche must not mutate accepted repaired-v2/V9
artifacts or reinterpret parse-only absence as a clean semantic result.

## Read-only root-cause inventory

All 26 identities are compile-valid accepted preprocessed artifacts and all are
unique normalized-code groups.

| Full-analysis failure subgroup | Contracts | Finding |
|---|---:|---|
| Slither singleton destination type | 24 | Slither 0.10.0 and 0.11.5 represent a chained high-level-call destination as a singleton type list (for example `[uint256]`) and then use the list as a `using_for` dictionary key, raising `TypeError: unhashable type: 'list'`. |
| Stale Slither runtime only | 1 | `caa35c...ec9` fails in the generation runtime's Slither 0.10.0 but completes full analysis in the locked DATA runtime's Slither 0.11.5. |
| State-initializer ternary | 1 | `970340...235` triggers Slither's unsupported ternary IR conversion for `maxPerWallet < 50 ? maxPerWallet : 50`; the preceding construction-time initializer fixes `maxPerWallet` to `10`. |

The first full Slither-0.11.5 population regeneration exposed two additional
fail-closed findings before acceptance:

- three previously full-analysis contracts regressed to parse-only because
  Slither asserts while lowering `(bool success, ) = call.value(v)("")`;
  an exact hash-bound, byte/line-preserving LHS reconciliation recovered their
  full low-level-call IR;
- Slither 0.11.5 exposes internal calls as `InternalCall` operations whose
  `.function` points to the callee, while the extractor expected the historical
  direct `Function` object. The first candidate therefore had zero
  `CALL_ENTRY`/`RETURN_TO` edges versus 342,268/334,991 in accepted V9. Physical
  acceptance remains blocked until a fresh full candidate is generated with
  both representations supported and the lower edge kinds are reconciled.
- after that API repair, a 300-contract comparison still found unexpected
  structural drift in 284 contracts: 249 node-feature changes, 153 metadata
  changes, and 268 topology changes through unchanged edge type 10. The same
  sample against the Slither-0.10 V10 candidate had zero feature/metadata
  drift; its 180 topology differences were limited to the already-intended V10
  ICFG correction. A blanket 0.11.5 population upgrade is therefore rejected.

The root repository lock and DATA environment specify Slither 0.11.5, while the
ML environment and the structurally stable V9/V10 baseline use Slither 0.10.0.
Neither version alone repairs every contract without materially changing the
rest of the population.

Exploratory full-analysis evidence under Slither 0.11.5 established:

- exact singleton-list unwrapping recovered full SlithIR for all 24 matching
  contracts, with 1-5 recorded repairs per contract;
- `caa35c...ec9` completed full analysis without an analyzer repair;
- a graph-only replacement of the exact initializer ternary with its proven
  construction-time value `10`, padded to the same byte length, retained the
  line count and completed full analysis for `970340...235`.

These probes did not write canonical artifacts and do not constitute physical
acceptance.

## Versioned remediation decision

Keep graph schema V10 and its 17 edge kinds unchanged. Advance the extractor
identity from `v2.3-r4-call-semantics` to a new compatibility-repaired identity;
do not emit different graph bytes under the old extractor identity.

Implement the following ordered V10-only recovery:

1. require and record exact Slither 0.10.0 for the primary population so node,
   metadata, and unchanged-edge structure remain bound to the accepted
   baseline;
2. attempt normal full analysis;
3. only after the exact singleton-list failure, retry under a process-local,
   restored-after-use analyzer guard that unwraps only a one-element Solidity
   type list on `HighLevelCall.destination.type`; record every repair;
4. only for the exact accepted source hash `970340...235`, reconcile the exact
   initializer expression to `10` with byte- and line-preserving padding and
   record both source hashes and the replacement;
5. regenerate exact source `caa35c...ec9` alone under Slither 0.11.5, record it
   as the only identity-bound runtime exception, and fail binding if any other
   contract uses that runtime or if this contract remains on 0.10.0;
6. understand both historical direct-function and newer `InternalCall.function`
   internal-call representations, while requiring transition-audit structural
   preservation outside the historical 26 parse-only contracts;
7. retain parse-only as an explicit diagnostic fallback, never as accepted
   complete IR.

Historical V9 ordering and bytes remain unchanged.

## Required implementation evidence

- unit tests for exact failure gating, singleton-only repair, restoration of
  Slither globals, transform hash binding, byte/line preservation, and refusal
  of near-match sources;
- real-Slither bounded regeneration of all 26 identities under the new
  extractor identity;
- zero parse-only results, zero unclassified call IR, zero call-mapping errors,
  and exact classified/emitted/observed call-edge equality for all 26;
- exact accepted-V9 token-byte equality;
- exact runtime distribution: 22,539 primary Slither-0.10.0 contracts and one
  identity-bound Slither-0.11.5 exception;
- full candidate binding and V9-to-V10 transition audit after the affected
  artifacts are incorporated into a fresh protected candidate lineage;
- population preservation of internal-call `CALL_ENTRY`/`RETURN_TO` semantics
  under the bound Slither runtime;
- expanded DATA/representation/ML/handbook regression and workflow validation.

## Stop lines

Do not declare physical acceptance merely because all 26 contracts reach full
analysis. The refreshed transition report and explicit review must pass first.
Even physical acceptance would not authorize training: R4-GAP-007,
selector/objective design, and training authorization remain separate gates.

## 2026-08-23 local stop checkpoint

The v2.4 protected candidate now binds all 22,540 identities with exact V9
token bytes, zero parse-only outputs, zero unclassified call IR, and the exact
runtime split required above: 22,539 Slither-0.10.0 primary artifacts plus the
single identity-bound Slither-0.11.5 exception. Its binding digest is
`bd907531a3e22b15d7b91552d15ef1f60c5fd59a109c4ef144ca62f3abab6950`.

Physical acceptance is nevertheless still blocked. The strengthened complete
transition audit is saved at
`docs/plan/ml-R4/reviews/R4-GAP-008/v10_transition_audit_v2.json` (SHA-256
`5793b059e7e5149424e10a5361a5b0e420b1f86f3630920e36344c5737fd4f9b`).
It checked all 22,540 identities without operational errors, but reported 46
contracts with structural differences from the frozen passing Slither-0.10 V10
reference: the 26 historical V9 parse-only identities where recovery is
expected, plus 20 previously full-analysis identities where it is not yet
approved. Status is therefore
`PASS_DIAGNOSTIC_WITH_STRUCTURAL_BLOCKER`, physical acceptance is false, and
training authorization is false.

The first bounded classification of those 20 unexpected identities found:

- 10 have identical node features and metadata, identical unchanged-edge
  counts, but different node-index endpoints. A traced example swaps edges
  among nodes with indistinguishable metadata, so node-order nondeterminism or
  an overly index-sensitive comparison is plausible but not yet proven for all
  10;
- 10 have identical node and unchanged-edge counts and identical unchanged-edge
  topology, but one or more node feature/metadata classifications differ. A
  traced example changes the same source expression from `CFG_NODE_OTHER` to
  `CFG_NODE_WRITE`, which may be upstream Slither analysis nondeterminism and
  must not be waived without repeat evidence.

The next bounded tranche is to regenerate only these 20 identities repeatedly
under the exact primary runtime, compare reference/candidate/repeats with a
node-identity-aware diagnostic, and then either remove the nondeterminism or
prove a narrowly defined semantic equivalence rule. Do not weaken the
population audit merely to obtain a passing status. No ADR for physical
acceptance and no authority-surface promotion should be written before this
blocker is resolved.
