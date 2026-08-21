# Phase-8 v10 implementation and protected-local regression

**Date:** 2026-08-21–22
**Decision:** R4-D-010 / ADR-R4-010
**Status:** FULL DIAGNOSTIC PASS WITH COMPATIBILITY BLOCKER; NOT ACCEPTED
**Physical acceptance:** no
**Training:** not authorized

## Outcome

The versioned graph-schema-v10 repository tranche is implemented without
editing accepted repaired-v2/v9 artifacts. The implementation distinguishes
typed high-level calls, raw low-level calls, Ether `Transfer`, Ether `Send`,
`LibraryCall`, and contract creation using exact Slither IR operation classes.
It also makes unknown
Slither `Call` subclasses, missing CFG call-site maps, schema mismatches, and
out-of-range edge IDs fail closed.

The final protected-local run generated and bound all 22,540 candidate
representations with zero failures and byte-identical accepted-v9 token files.
The independent transition audit checked all 22,540 v9/v10 pairs with zero
errors. This is a complete mechanical candidate pass, not physical acceptance:
26 parse-only contracts lack complete call IR and materially block the semantic
acceptance required by R4-D-010.

The first protected-local regression found and prevented a real implementation
error: candidate #1 initially emitted its `Transfer` edge but lost the inherited
typed callback because canonical function-name maps were overwritten by a
parent-function clone. The bounded script's original mechanical checks were not
enough to catch that omission. The extractor now uses Slither function-object
identity for v10 caller maps and persists classified-IR counts, emitted-edge
counts, mapping errors, and unclassified call IR. Candidate binding requires
classified and emitted counts to agree and rejects either error category.

This is exactly why bounded source-reviewed regressions precede full generation.
The initial output was not accepted or used as positive evidence.

A later code review found a second fail-closed telemetry edge case: an unknown
`Call` subclass could escape the unclassified-IR list when its function or CFG
node map was itself missing. The first incomplete full-generation attempt was
interrupted and its candidate-only files were deleted. The extractor now scans
unknown call subclasses before either map lookup. Bounded evidence was
regenerated from the final code before restarting the full population. No
accepted-v9 file was edited or removed.

That restart then exposed a third real-data omission rather than a processing
failure: 137 of the first approximately 7,000 generated graphs contained 183
Slither `NewContract` operations that were reported as unclassified. The
incomplete candidate was stopped and removed. V10 now assigns
`CONTRACT_CREATION=16`, treats it as an external handoff for structural graph
consumers, keeps it out of `CallToUnknown` truth signals, and has 17 total edge
kinds. A protected-local regression on corpus identity `07e7b0e8...81b3d`
then emitted two contract-creation edges with exact IR-to-edge reconciliation,
zero mapping/unknown errors, and byte-identical accepted-v9 tokens.

## Versioned interface

| Field | Historical | Candidate |
|---|---|---|
| graph schema | `v9` | `v10` |
| extractor | `v2.2-r4-repaired` | `v2.3-r4-call-semantics` |
| edge kinds | ambiguous type-11 `EXTERNAL_CALL` | `HIGH_LEVEL_CALL=11`, `LOW_LEVEL_CALL=12`, `ETHER_TRANSFER=13`, `ETHER_SEND=14`, `LIBRARY_CALL=15`, `CONTRACT_CREATION=16` |
| representation root | `representations-r4-v2` | `representations-r4-v3-candidate` |
| token lineage | accepted `[4,512]` tensors | exact accepted-v9 byte copies |
| training authority | historical reproduction only | false |

Historical v9 remains the default import/config behavior. V10 must be selected
explicitly and is recorded in graph payloads, sidecars, generation manifests,
dataset contracts, and run bindings.

## Corrected consumers and stop lines

- v10 CEI/reentrancy treats high-level, low-level, transfer, send, and contract
  creation as external handoffs but excludes library calls;
- v9 `CallToUnknown` graph checking is now `NOT_EXTRACTABLE` because type 11 is
  ambiguous; v10 low-level/send edges are coarse corroboration only;
- `ExternalBug` is `NOT_EXTRACTABLE` until a class-specific source-backed signal
  exists;
- the GNN has exactly 17 v10 edge embeddings and raises on unknown edge IDs;
- v10 checkpoint edge-vocabulary mismatch cannot use historical automatic
  resizing;
- the v10 dataset/run path rejects a diagnostic candidate and requires a later
  physical-acceptance report plus a separate binding-specific training decision;
- the diagnostic binder and generation reports always retain
  `physical_acceptance=false` and `training_authorized=false`.

## Source-reviewed bounded regression

Committed expectation contract:

`docs/plan/ml-R4/reviews/R4-GAP-008/v10_gap007_regression_expectations_v1.json`

Expectation SHA-256:

`74e26c3f228c051323ebd72bf6eae646144cd965350afe27b5a9bc80fd42a94b`

Final local ignored report:

`data_module/data/r4-v10-regression-final3/v10_gap007_regression_report.json`

Report SHA-256:

`f6a895c96b4c17e119a279817236ca05779750399de4f0fc20d76369e232b366`

| Contract | Source-reviewed expected and observed v10 call edges | Token bytes | IR→edge mapping |
|---|---|---|---|
| candidate #1 `defe4690...a384` | high-level 1; transfer 1; all others 0 | exact accepted-v9 hash `e6801b3c...b4ef1` | exact; zero errors/unknowns |
| candidate #2 `f7afe9ff...1b93` | library 30; transfer 1; all others 0 | exact accepted-v9 hash `759a858c...bcfd` | exact; zero errors/unknowns |

Additional real-population contract-creation regression:

`data_module/data/r4-v10-contract-creation-regression/v10_contract_creation_regression_report.json`

Report SHA-256:

`2c37e48825e278bd14a87a71f11d661a6057677817a68869839f1d10b318bf67`

The diagnostic observed 2 contract-creation, 12 typed high-level, and 10
library edges. Classified IR, emitted edges, and graph counts agree exactly;
tokens are byte-identical and there are zero mapping or unknown-IR errors. This
is an extractor regression, not a complete source-reviewed semantic inventory.

Before the canonical restart, a final source review found that the historical
outer ICFG wrapper still logged and swallowed an unexpected generic exception.
That could have bypassed the explicit V10 telemetry. The partial candidate was
stopped and moved to trash; V10 now re-raises unexpected ICFG/call extraction
failures, while v9 retains its historical best-effort behavior. A dedicated
real-Slither test proves the V10 fail-closed path, and both bounded reports above
were regenerated afterward. The current canonical generation began only after
that revalidation.

The result fixes the concrete R4-GAP-008 regression: candidate #2 no longer
aliases 30 `SafeMath` library calls to unknown external calls and its real Ether
transfer is represented. Candidate #1's caller-selected typed callback is also
retained as high-level interaction. These are representation findings only;
candidate #1 remains `NOT_CONFIRMED`, candidate #2 remains UNKNOWN pending
independent review, and no label changes.

## Repository validation completed so far

Focused suites passed for:

- v9/v10 schema selection and exact call-kind classification;
- real Slither fixtures including imported `using for`, inherited typed
  callbacks, all four raw low-level forms across compatible compiler versions,
  `Transfer`, and `Send`;
- v9-preserved and v10-corrected CEI behavior;
- semantic checker behavior;
- generation root and accepted-token guards;
- diagnostic population binding;
- exact dataset/run authorization boundaries;
- GNN v10 vocabulary and OOB failure;
- inference checkpoint mismatch failure.

The complete affected regression suites and handbook validation are rerun in
the final Git-safe closeout below.

Final expanded validation evidence:

- Phase-8 DATA/representation/vnext/semantic plus CI-focused ML/audit suite:
  214 passed;
- complete representation suite, including v9 byte compatibility and real
  compiler fixtures: 142 passed, 2 expected environment/fixture skips;
- affected ML model/inference/run-binding suites: 49 passed;
- one stale HEAD model test exposed during development omitted the model's
  already-existing `fusion_embedding` auxiliary key; it was corrected to the
  actual six-key contract and is included in the passing ML suite;
- handbook: 145 static checks and 11 unit tests passed; inventory passed;
- workflow YAML parses, changed Python modules compile, and `git diff --check`
  passes.

## Full protected-local candidate result

Canonical local output:

`data_module/data/representations-r4-v3-candidate`

Generation used 16 workers, all 22,540 accepted preprocessed identities, and
accepted v9 token files as byte-copy inputs. It completed with:

- DIVE: 22,054 written / 0 failed;
- SmartBugs Curated: 143 written / 0 failed;
- SolidiFI: 343 written / 0 failed;
- candidate binding: 22,540 checked / 22,540 token-byte matches / 0 missing,
  extra, or invalid;
- binding digest:
  `6087dc6d76d781efbefe0c4984458d291790c38b1c55d852f48fd796222b0260`.

Local report hashes:

- full generation:
  `e5cfed34c6da8899251cfe9a37cdbbf1e057bef3d7f78dd95cda042e88844816`;
- candidate binding:
  `bc8fdfd4b0e62ce76fef331c9eaebf8da5b45aa0966fcb91cc096f05c6fcefcd`;
- committed transition audit:
  `6df5058596709f2140a00ab72277cd4c384ba255340a030027f440091d62d318`.

All nine mechanical/review gates were evaluated:

1. zero generation failures;
2. exact 22,540-identity equality with accepted v9;
3. exact token-byte equality for every identity;
4. v10 schema/extractor graph and sidecar checks;
5. zero unclassified call IR and zero call-to-CFG mapping errors;
6. classified-IR counts equal emitted and observed v10 call-edge counts;
7. deterministic binding digest creation;
8. full v9→v10 transition audit — diagnostic pass with compatibility blocker;
9. explicit human review — **reject physical acceptance for now**.

The independent transition census is:

| V10 call kind | Edges | Graphs containing kind |
|---|---:|---:|
| typed high-level | 77,562 | 15,587 |
| raw low-level | 7,561 | 4,311 |
| Ether transfer | 20,489 | 10,404 |
| Ether send | 3,203 | 819 |
| library | 183,826 | 14,913 |
| contract creation | 542 | 440 |

V9 contained 217,490 ambiguous type-11 edges. V10 changes the total call-edge
count in 15,242 graphs, demonstrating that this is a material representation
correction rather than a label-only rename. The durable machine-readable audit
is `reviews/R4-GAP-008/v10_transition_audit_v1.json`.

Even if every item passes, selector promotion, confirmed-negative acceptance,
objective changes, thresholds/calibration, checkpoint reuse, and training remain
separate decisions.

## Compatibility-mode acceptance boundary

The accepted v9 population contains 26 `slither_parse_only` graphs and two
`slither_full_analysis_constant_array_fold_v1` graphs. The latter retain full
IR analysis after a recorded byte/line-preserving source transform. The 26
parse-only graphs do not provide complete IR call semantics; observed examples
contain real `transfer`/`send` syntax even though their V10 classified-call
counts are necessarily zero. Exact identity reconciliation against accepted
logical V3 shows those 26 contracts are 7 `TRAIN_WEAK` and 19
`TRAIN_UNLABELED`; this is therefore an optimizer-input issue, not merely an
unused-tail caveat. A raw-source diagnostic screen finds explicit relevant
syntax in all 26: 85 `.transfer(...)` hits, 14 `.send(...)` hits, and 14
`new Contract(...)` hits (zero direct low-level-call hits). These are lexical
exposure counts, not source-reviewed semantic inventories, but they prove the
missing IR path is materially relevant to the V10 correction.

The transition audit now enumerates these identities, compares v9/v10 modes,
and records diagnostic raw-source hits for low-level calls, transfers, sends,
and contract creation. A parse-only graph is never interpreted as semantic
absence by the V10 semantic checker; it returns `NOT_EXTRACTABLE`. Any remaining
parse-only population is an explicit physical-acceptance blocker until resolved
through versioned extraction repair, an explicit exclusion/role decision, or
complete source-level reconciliation. A passing mechanical binder alone is not
enough.
