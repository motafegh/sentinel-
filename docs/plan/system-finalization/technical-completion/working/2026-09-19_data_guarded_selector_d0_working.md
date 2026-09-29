# DATA D0 — guarded-selector source/evidence reconstruction

**Date:** 2026-09-19  
**Last reconciled:** 2026-09-29  
**Branch:** `agent/data-target-aware-guarded-v1`  
**Base main:** `57f39d652cbec1092084b4efbf41bec6a117ba07`  
**Work package:** D0-D4 guarded-token technical-completion continuation  
**Status:** D0-D4 COMPLETE / D5 READY FOR PROTECTED-LOCAL GENERATION / D6 NOT STARTED

### Continuation checkpoint

The guarded-selector implementation was developed and D4-validated on `agent/data-target-aware-guarded-v1`, then promoted through PR #74 and merged to canonical `main` at `8bbe4f48edadf9e43ab448e7d935921116049858`.

From that merge onward, **canonical `main` is the continuation authority for D5 and later work**. The repository may retain `agent/data-target-aware-guarded-v1`, `technical-completion/data-target-aware-guarded-v1`, and `work/data-d0-selector-contract` as historical/alternate development lines; none should supersede `main` without a new explicit reconciliation.

The original implementation branch was based directly on main `57f39d652cbec1092084b4efbf41bec6a117ba07`. D0-D4 are now integrated; D5 is the first unexecuted work package.

## 1. Question

Reconstruct the exact executable contract authorized by R4-D-012 for
`target_aware_guarded_v1`, identify the safe production seam for a fresh
guarded-token physical candidate, and preserve all historical-control behavior
and R4-D-011 artifacts unchanged.

This record is an investigation/design aid. R4 decisions and executable
source/tests remain authoritative.

## 2. Authorities and source inspected

Controlling owners:

- `CLAUDE.md`
- `docs/plan/system-finalization/technical-completion/00_MASTER_TECHNICAL_COMPLETION_ROADMAP.md`
- `docs/plan/system-finalization/technical-completion/01_DATA_REPRESENTATION_COMPLETION_PLAN.md`
- `docs/plan/ml-R4/PLAN_STATUS_MATRIX.md`
- R4-D-011 run record and ADR
- R4-D-012 selector-promotion run record and ADR
- machine records under:
  - `docs/plan/ml-R4/evidence/2026-09-02_selector_promotion/`
  - `docs/plan/ml-R4/evidence/2026-09-02_selector_control_equivalence/`
  - `docs/plan/ml-R4/evidence/2026-08-15_phase8_logical_v3/`

Executable source/tests inspected:

- `ml/src/data_extraction/windowed_tokenizer.py`
- `ml/src/data_extraction/bounded_window_selector.py`
- `data_module/sentinel_data/representation/r4_target_spans.py`
- `data_module/sentinel_data/representation/r4_orchestrator.py`
- `data_module/sentinel_data/preprocessing/r4_versions.py`
- `data_module/sentinel_data/vnext/r4_binding.py`
- `data_module/sentinel_data/vnext/r4_v10_binding.py`
- `data_module/sentinel_data/vnext/representations.py`
- `data_module/sentinel_data/export/token_writer.py`
- Phase-8 selector comparison/control-equivalence scripts
- bounded-selector, tokenizer-coverage, and V10-orchestrator tests
- current ML dataset adapters that consume token payloads

## 3. Evidence identity

The current `bounded_window_selector.py` blob is identical on:

- current `main`;
- hardened evidence source commit
  `83bd566b9c4f4f653e530c2c0f5c990858dd759d`;
- control-equivalence source commit
  `735eda59dd02ab38ee5f14135f64b75a9a3a1111`.

The current `r4_target_spans.py` blob is also identical on all three refs.

Therefore the exact implementation retained in the repository is the accepted
research implementation; selector semantics do not need to be reconstructed
from prose or summary statistics.

R4-D-012 records the selector implementation SHA-256
`9eea0f837f77a512628efaa3dde444f039be81e98bf1828aa61b9099d2c87866`
and target-span implementation SHA-256
`7e6f2017e43425285fd51186cd0788ec2ba5f2fe56c90f6dc7b9ef190ee4a910`.

## 4. Current raw-source -> artifact path

### 4.1 Source and graph target

The repaired source is read from the immutable repaired preprocessing lineage.
`r4_orchestrator._select_targets()` resolves the requested file-graph targets
from explicit ingestion provenance plus the source declaration structure.

The accepted V10 sidecar persists:

- `requested_contract_names`;
- `actual_contract_names`;
- graph component count and graph/runtime provenance.

These requested names are the target-contract input used by the selector
research and by the full-population historical-control equivalence verifier.

### 4.2 Current production tokenization

`windowed_tokenizer.tokenize_windowed_contract_strict()` currently:

1. reads the Solidity source;
2. optionally strips comments lexically;
3. tokenizes the full source without truncation to count raw code tokens;
4. asks GraphCodeBERT for overlapping 512-token overflow windows with
   `stride=256`;
5. selects at most four windows using the historical linspace rule;
6. pads with all-pad windows until the tensor is exactly `[4,512]`;
7. returns `input_ids`, `attention_mask`, selected indices, token ranges,
   and retained-token coverage telemetry.

For repaired preprocessing through `r4_orchestrator`, comment stripping is
disabled at this seam because preprocessing has already performed the lexical
normalization.

### 4.3 R4-D-011 V10 path

The accepted V10 V2.6 builder does **not** retokenize. When
`accepted_tokens_dir` is supplied, it byte-copies the accepted historical token
artifact and writes a V10 graph plus sidecar. The sidecar records
`token_lineage="accepted_v9_byte_copy"`.

R4-D-011 therefore remains an immutable V10 graph parent with historical-control
token bytes.

## 5. Exact selector semantics

The authoritative selector names are:

- control: `historical_linspace_v1`;
- research greedy candidate: `target_aware_greedy_v1`;
- promoted candidate: `target_aware_guarded_v1`.

### 5.1 Source view and target spans

The accepted research path uses an offset-preserving, comment-stripped source
view. Comment bytes are replaced rather than removed, so source length and
newline positions are preserved.

Target character spans are computed from the original/preprocessed source using
the requested contract names. Every requested name must resolve exactly once.
Brace matching ignores comments and strings and requires balanced declaration
bodies.

Character spans are mapped to raw GraphCodeBERT token-index ranges using the
fast tokenizer's offset mapping. A non-empty character span that maps to zero
tokens is an error.

### 5.2 Window ranges

Let:

- `window_size = 512`;
- `special_tokens = tokenizer.num_special_tokens_to_add(pair=False)`
  (fallback 2);
- `content_capacity = window_size - special_tokens`;
- `stride = 256`;
- `step = content_capacity - stride`.

Window ranges are deterministic half-open raw-code-token intervals generated
from token 0 until the final interval reaches the token count.

The research path verifies that this computed range count is exactly equal to
the tokenizer overflow-window count before selecting tensors.

### 5.3 Historical control

If total windows are at or below the requested count, select every real window
in increasing index order.

If over cap, the exact historical rule is:

`[round(x) for x in np.linspace(0, total_windows - 1, count)]`.

For the frozen contract, `count=4`.

This is the behavior already proven equivalent to the accepted R4-D-011 token
population for all 22,540 identities.

### 5.4 Greedy candidate

The greedy candidate repeatedly chooses the not-yet-selected window with the
largest **marginal increase in union target-token coverage**.

Tie breaking is deterministic: maximize `(gain, -window_index)`, therefore the
smallest window index wins equal-gain ties.

Greedy selection stops if:

- the requested count is reached;
- no candidate remains; or
- the best marginal target-coverage gain is non-positive.

It then fills remaining slots first with historical-linspace indices that are
not already selected, then with the earliest still-unselected indices. The
final selected list is sorted and truncated to the requested count.

### 5.5 Guard

For `target_aware_guarded_v1`:

1. compute the historical-control target-token coverage;
2. compute the greedy candidate target-token coverage;
3. use the greedy candidate **only if its target-token coverage is strictly
   greater** than control;
4. if candidate coverage is equal to or below control, use the historical
   control and mark control fallback.

Therefore equality intentionally falls back. No retained-total-token criterion
participates in the guard.

This matches the accepted CPU evidence: 737 over-cap records, 476 improved,
261 equal/control-fallback, zero regressions, zero failures.

### 5.6 Under-cap behavior

For `total_windows <= 4`, historical control selects all real windows. The
greedy path ultimately contains the same complete set, so the guarded comparison
is a target-coverage tie and the effective output is the historical control.
Padding then restores the frozen `[4,512]` tensor shape when fewer than four
real windows exist.

## 6. Invalid/missing target evidence — resolved interpretation

There are two distinct cases and they must not be conflated:

1. **Valid target evidence, candidate does not strictly improve target
   coverage:** use the accepted guarded fallback to historical linspace.
2. **Target evidence cannot be established or validated:** fail closed as a
   structured target-evidence error.

The retained accepted research pipeline does not silently convert malformed,
ambiguous, or zero-token target spans into a clean guarded result:
`target_contract_char_spans()` and `char_spans_to_token_ranges()` raise.

The lower-level selector can return control for an explicitly empty
`target_ranges` list, but the accepted end-to-end research path requires
non-empty valid target spans before selection. Production integration will keep
that fail-closed boundary. This also reconciles the DATA plan's D2 requirement
that malformed/missing target evidence be an explicit fallback/error path rather
than an implicit clean result.

No new selector semantic is introduced by this interpretation.

## 7. Consumers and historical-control boundaries

### Must remain immutable / historical-control specific

- `windowed_tokenizer._selected_window_indices()` and its existing tests prove
  historical linspace behavior.
- R4-D-011 representation artifacts and sidecars remain byte/hash immutable.
- `r4_v10_binding.bind_v10_candidate()` is historical V10 acceptance machinery:
  it explicitly requires `token_lineage="accepted_v9_byte_copy"` and exact
  accepted-V9 token byte equality. It must **not** be loosened for the guarded
  lineage.
- `p8_verify_v10_bound_token_control_equivalence.py` is historical-control
  evidence and compares dynamic control indices/tensors with R4-D-011.
- existing V10 orchestrator tests that prove accepted-token byte-copy behavior
  stay historical-control tests.

### Generic consumers

- `r4_binding._validate_tokens()` binds token coverage fields in payload and
  sidecar and enforces shape/padding; a guarded-lineage binder can reuse the
  generic validation mechanics but needs new selector-aware identity checks.
- ML dataset adapters consume only `input_ids` and `attention_mask` from
  token payloads; they do not depend directly on selected-window indices.
- `token_writer.py` consumes only `input_ids` when creating legacy shards.
- downstream training identity is controlled by manifest/binding digest, not by
  selected-window metadata alone.

## 8. D1 design decision

Use a **fresh focused guarded-token candidate module** rather than modifying the
R4-D-011 V10 generation path.

Rationale:

- R4-D-012 requires a fresh token lineage while preserving accepted V10 graph
  semantics and bytes;
- copying the accepted R4-D-011 graph artifact is stronger and cheaper than
  re-running Slither merely to change token selection;
- the full-population control-equivalence verifier already proves the retained
  research tokenization mechanics reproduce the accepted token population;
- keeping the accepted builder/binder unchanged makes historical-control
  regression easier to prove.

Planned identities:

- selector requested: `target_aware_guarded_v1`;
- rollback/control selector: `historical_linspace_v1`;
- graph schema/extractor: unchanged V10 /
  `v2.6-r4-call-semantics-deterministic-cfg-mutators`;
- token tensor: unchanged `[4,512]`;
- guarded representation lineage: fresh versioned identity, never
  `accepted_v9_byte_copy`;
- fresh candidate root name, distinct from the R4-D-011 root.

The exact lineage/root constants will live in the existing R4 version-owner
module so builders, tests, binders, and later acceptance evidence cannot diverge
by string literal.

## 9. Required selector metadata

The fresh token payload and sidecar must bind enough evidence to answer:

- selector requested;
- selector actually effective;
- whether historical-control fallback occurred;
- explicit fallback reason;
- greedy candidate indices;
- historical-control indices;
- final selected indices;
- requested target contract names;
- target character spans and token ranges;
- target-token count and candidate/control/final coverage;
- pre-subsampling token/window counts;
- final retained-token coverage;
- tokenizer/window/stride configuration;
- graph-parent identity and graph hash;
- fresh representation/token lineage identity.

Malformed target evidence must never be labeled
`target_aware_guarded_v1` success.

## 10. D0 exit decision

D0 is complete:

- accepted selector semantics are reconstructable from unchanged executable
  source;
- historical-control behavior and immutable consumers are identified;
- the target-evidence error/fallback distinction is explicit;
- a safe fresh-lineage integration seam is identified;
- no contradiction was found between current source and R4-D-012 evidence.

## 11. D1/D2 implementation result

Implemented on `agent/data-target-aware-guarded-v1` without modifying the
historical selector implementation or the R4-D-011 builder/binder:

- version-owner constants for:
  - `historical_linspace_v1`;
  - `target_aware_guarded_v1`;
  - selector-decision schema;
  - fresh guarded token lineage;
  - fresh candidate root name;
- focused module
  `data_module/sentinel_data/representation/r4_guarded_token_candidate.py`;
- exact R4-D-011 acceptance/root validation before using a parent;
- graph byte-copy with post-copy SHA-256 equality verification;
- explicit target-span derivation from accepted requested graph targets;
- accepted guarded-selector execution with fail-closed target-evidence errors;
- token and sidecar selector decision metadata;
- bounded-candidate manifest with `physical_acceptance=false` and
  `training_authorized=false`;
- overwrite refusal and fresh-root enforcement.

The accepted V10 historical builder, historical V10 binder and selector
control-equivalence tooling remain unchanged.

The bounded builder intentionally rejects `identities=None` before creating an
output directory. This makes implicit full-population generation impossible
through the D4 API and preserves the D5 stop line in executable code.

## 12. D3 validation result

Focused tests were added under
`data_module/tests/test_representation/test_r4_guarded_token_candidate.py`
covering:

- under-cap complete-window selection and frozen padding;
- strict-improvement guarded selection;
- control fallback on equal target coverage;
- repeated-run tensor/metadata determinism;
- byte-identical graph-parent reuse;
- malformed/missing target evidence failure;
- requested/actual graph-target mismatch;
- overwrite refusal;
- exact accepted-parent-root validation;
- bounded manifest persistence;
- invalid output-root refusal;
- explicit D5/full-population refusal.

The Phase-8 repository workflow was extended to compile the new module and run
the new focused tests.

Exact code/test validation head:

`ccf49bfae504c82c192d72499ba8766f0f185376`

GitHub Actions run `35454628234` established:

- dependency installation: PASS;
- repaired DATA/ML/token/research module compilation: PASS;
- repaired and Phase-8 regression suite, including the new guarded tests: PASS;
- committed logical-V3 snapshot verification: PASS;
- frozen historical G6 validation: PASS.

The workflow-level result is still red only at the legacy
`git diff --check a10fae...HEAD` step. That check also fails on canonical
`main` (for example run `34870795452`) because its comparison range contains
pre-existing trailing whitespace in historical planning documents. This is
baseline CI debt, not a guarded-selector regression. The new working record's
own trailing whitespace was removed rather than using the baseline issue as an
excuse to add new debt.

D3 is therefore complete on substantive executable evidence.

### 12.1 2026-09-27 pre-D4 reproducibility and source-identity hardening

Before protected-local D4, the guarded candidate path was independently
re-audited against the physical-hash and immutable-parent requirements.

Two concrete gaps were found and closed on the canonical PR branch.

#### Token artifact byte reproducibility

Candidate manifests record `tokens_sha256`, so tensor equality alone is not
enough: repeated construction must produce the same physical token-file bytes.

An initial attempt to force PyTorch's legacy serialization stream was rejected
by the exact-head Phase-8 regression test: independently created but
semantically identical tensors can receive different legacy storage identities.

The final implementation instead serializes the token payload with the modern
PyTorch ZIP format into an in-memory `BytesIO` stream, then writes those exact
bytes to the candidate file. This removes candidate-root/path identity from the
serialization seam while retaining
`torch.load(..., weights_only=True)` compatibility.

The repository-safe determinism test now requires both:

- equal selected tensors/selector metadata; and
- exact repeated `.tokens.pt` byte equality / SHA-256 equality.

The protected-local D4 validator independently repeats the same physical
token-file SHA-256 comparison and reports
`token_artifact_repeat_equal`.

Relevant commits:

- `34f7495fe93b91a2f3d4e41c2277c73783ccd7cb` — first reproducibility
  hardening attempt;
- `a40a541354e73cc797fd87be0b11ae140ab6b735` — repository-safe byte
  determinism assertion;
- `a4e1b131c3558b361e474e7d78174e144bbecad6` — D4 physical-token repeat
  assertion;
- `10c102c515c9458821123df49e17cf0d8ee0a746` — corrected path-independent
  in-memory ZIP serialization.

#### Repaired-source identity binding

The builder previously validated that the preprocessing directory was the exact
R4-D-011 accepted parent path but did not re-prove that each persisted
`<contract_id>.sol` still contained the bytes named by that content identity.

The preprocessing owner proves that `contract_id` /
`normalized_text_sha256` is SHA-256 of the exact normalized Solidity bytes
copied into the repaired `.sol` artifact. The guarded builder now therefore:

1. reads the persisted repaired source as bytes;
2. requires `sha256(source_bytes) == contract_id`;
3. only then decodes and retokenizes it.

This prevents a stale accepted graph from being combined with tokens generated
from silently changed source bytes.

Relevant commits:

- `8e38813ebaf01bbf2883cd0a43d23c39cafbfea0` — source-byte identity guard;
- `54892954dc897e1d384bbab88cda5c49325bbd26` — regression coverage.

#### Exact-head repository evidence

Implementation head `10c102c515c9458821123df49e17cf0d8ee0a746` was
validated by Phase-8 repository-repair run `36341534405`.

Substantive results:

- repaired DATA/ML/token/research module compilation: PASS;
- main repaired + Phase-8 suite: **222 passed, 9 skipped**;
- focused ML compatibility suite: **15 passed**;
- deterministic semantic-evidence helper suite: **20 passed**;
- committed logical-V3 snapshot verification: PASS;
- frozen historical G6 validation: PASS.

The workflow conclusion remains red only at the inherited
`git diff --check a10fae...HEAD` step, which reports pre-existing trailing
whitespace in historical planning/portfolio documents already present in the
comparison baseline. No guarded-selector source/test failure remains.

Other exact-head PR checks are green, including Handbook, security hygiene,
Phase 3, Phase 4 G4/DIVE, Phase 5 policy/G5, Phase 6 G6 and Phase-8 vNext
training compatibility. The Phase-7 G7 workflow remains independently red
because that workflow installs pytest/pyarrow but then collects existing tests
that import `torch`; its failure is `ModuleNotFoundError: torch`, not a
guarded-candidate regression.

These hardenings do not advance D4. They strengthen the executable preconditions
for running it.

## 13. D4 bounded tranche selected from retained evidence

D4 must run only against the protected local physical roots. The initial tranche
is evidence-derived rather than arbitrary:

| Purpose | Source | Contract ID | Prior evidence |
| --- | --- | --- | --- |
| under-cap / fallback | `smartbugs_curated` | `85a6581669271b86cd58b837f216e6b140f726b1dce93270dcf6291995fbfe5d` | 1 window; control fallback; full target coverage |
| strong selector improvement | `solidifi` | `08378c9d432399d34e2f5a417e0b57e47b0ef63cc99a208f9efb67744d5e837f` | 11 windows; target coverage 0.5205566 -> 0.9854522 |
| over-cap equality/fallback | `solidifi` | `397813120698b5942a0168c339310bb57dcf2d8b4041b3590ad86ce3d3accfbd` | 8 windows; guarded equals control |
| class/shape fallback | `solidifi` | `9b8eb361195230fb9e7d8797c3c456fce60b169564f5484ae003814ee03a6e4c` | Reentrancy; 17 windows; control already covers target |
| long train-batch improvement | `dive` | `83c9d2d26dc19eaa2aee29fa7aedb4f4e208429a96cc7a0ffee7491b9830630d` | 62 windows; guarded target coverage improvement |
| worst-case CUDA forward probe | `dive` | `f50cd5d7df9ab644a02eb760ceab56548d327984db313015a66bca85513fa3c5` | 353 windows; retained CUDA worst-case; guarded target coverage improvement |
| additional improved train case | `dive` | `087f69b560460734f646e30aa9be314c7f9085289ba394677905d008cf3a7ae0` | 17 windows; guarded improvement |
| additional strong-train shape | `solidifi` | `d4b90b62c2ab33ce14d403f1d995b132e8178a09ca243db3be58b610c30cd297` | 12 windows; guarded improvement |

Also resolve and include the accepted V10 runtime-exception identity at runtime:

`caa35c1a5906269bbe5e70de780d105c2968ece4fc038d7f7208efee681aeec9`

Its source directory should be discovered from the protected R4-D-011 parent,
not guessed from memory.

An optional extra stress probe is the 403-window sensitivity identity
`dive/c74bbb7fbe8eda3e6d9404b08678e9eca476aa85831e7c23b578cfa089f77b8f`.
It is useful for token-window scale stress but remains optional. The required D4 tranche already includes both the retained 62-window long train-batch case and the 353-window worst-case CUDA forward probe above.

## 14. D4 protected-local result — reviewed 2026-09-29

D4 was executed at source commit `4efc0676e6966d9581aa6a10f882125dda5063fc` against the immutable R4-D-011 parent and repaired preprocessing roots.

Reviewed result:

- status: `PASS_BOUNDED_D4_REVIEW_REQUIRED`;
- 9 / 9 requested identities passed; zero identity failures and zero report failures;
- both fresh repeats produced identical selector decisions, sidecars, tensor digests, and exact token-artifact bytes;
- immutable R4-D-011 graph hashes matched in parent, repeat A, and repeat B for every identity;
- frozen token contract remained `[4,512]` / `torch.int64`;
- repeat A and repeat B each used `target_aware_guarded_v1` for 6 identities and `historical_linspace_v1` for 3 guarded fallbacks;
- the 353-window worst-case CUDA forward-probe identity passed;
- the declared Slither runtime-exception identity was resolved from the accepted parent and passed;
- no physical acceptance or training authority follows from D4.

The validator deliberately emitted `d5_authorized=false` because review is a separate governance step. This record performs that review: no D4 blocker remains, so D5 full protected-local generation is now authorized as the next DATA action. D6 remains the separate physical-acceptance gate.

## 15. D5 implementation design — 2026-09-29

D5 must not be enabled by weakening the bounded D4 API. Preserve
`build_guarded_token_candidate(..., identities=...)` as an explicitly bounded
construction seam whose `identities=None` call continues to fail closed.

Add a separate D5 full-population seam with these requirements:

- enumerate the exact accepted R4-D-011 parent inventory only after validating
  the accepted parent record/root;
- require the enumerated population to equal the accepted parent contract count;
- reuse the same already-D4-validated per-identity guarded builder;
- preserve the immutable graph bytes and fresh guarded-token lineage rules;
- emit a distinct full-candidate status and `full_population=true`;
- bind every sorted identity record (source, contract id, selector decision,
  selected indices, graph/token/sidecar SHA-256) into a deterministic candidate
  binding digest;
- keep `physical_acceptance=false` and `training_authorized=false`;
- provide a protected-local driver with periodic progress output and a compact
  D5 review report;
- finish successful generation at `PASS_FULL_D5_REVIEW_REQUIRED`, with D6
  still explicitly required before any physical-authority change.

A partial/failed full build is not resumable authority. It remains an
unaccepted attempt root; rerun from a fresh root after the cause is understood.

### D5 implementation checkpoint

Implemented on canonical `main` through `8c0e166b502908f4f801492f638c160d17c0a924`:

- a dedicated `build_guarded_token_full_candidate(...)` API that derives the
  exact population from R4-D-011 rather than accepting caller-supplied identities;
- the bounded D4 API still fails closed for `identities=None`;
- full manifests use `FULL_GUARDED_TOKEN_CANDIDATE` and
  `full_population=true`;
- sorted per-identity graph/token/sidecar hashes plus selector decisions are
  bound into a deterministic candidate digest;
- the protected-local D5 driver
  `docs/plan/ml-R4/scripts/p8_generate_guarded_token_candidate_d5.py`
  emits progress and a compact `PASS_FULL_D5_REVIEW_REQUIRED` report;
- D5 output remains `physical_acceptance=false`,
  `d6_authorized=false`, and `training_authorized=false`;
- focused repository-safe coverage was added for the exact-parent
  full-population derivation and manifest/digest boundary;
- CI compiles the D5 driver.

This implementation checkpoint was subsequently exercised against the full protected-local population; see the D5 review below.

## 16. D5 protected-local result — reviewed 2026-09-29

D5 was executed from source commit
`733f0c73eb76ab107751c30345d9a169a0429fdd` into the fresh attempt root
`data_module/data/r4-guarded-d5-full-2026-09-29-a`.

Reviewed generation result:

- status: `PASS_FULL_D5_REVIEW_REQUIRED`;
- 22,540 / 22,540 identities requested and written;
- `target_aware_guarded_v1`: 14,751 identities;
- `historical_linspace_v1` guarded fallback: 7,789 identities;
- candidate binding digest:
  `9885d7b88a46aff4102741d63eeaa0bd6968f0857f7bca6a389662bfaa158881`;
- candidate manifest SHA-256:
  `5aa48ce6eb218b742af3c99823db333272d65599039b6ec550d56f713b839dca`;
- accepted-parent manifest SHA-256:
  `5fc83eff39d4a28db9a5b6b5255a95ad64ee75ca88a948ba99dadb2bc03ee165`;
- graph schema remains `v10`;
- graph extractor remains
  `v2.6-r4-call-semantics-deterministic-cfg-mutators`;
- token lineage is `r4-v10-v26-guarded-tokens-v1`;
- frozen token shape remains `[4,512]`;
- Transformers runtime remains `4.46.3`;
- generation completed without an exception in approximately 585 seconds;
- generation report correctly retained
  `physical_acceptance=false`, `d6_authorized=false`, and
  `training_authorized=false`.

The long-sequence tokenizer warning is expected at the pre-windowing stage; it
does not imply that model-facing tensors exceed the frozen `[4,512]` contract.

This review closes D5 generation. It does **not** accept the candidate as
physical authority. D6 must independently reconstruct the full population
binding, verify every candidate artifact and selector contract, regenerate the
required deterministic probes, preserve R4-D-011 as immutable control, and
produce the explicit accept/reject/revise decision.

## 17. Current stop line

D5 is COMPLETE. D6 is READY FOR PROTECTED-LOCAL ACCEPTANCE REVIEW.

Therefore:

- the guarded full candidate exists but is **not yet physically accepted**;
- R4-D-011 remains the current accepted physical control;
- no ML training authority follows from D5;
- the 100-epoch Phase-8 run remains unauthorized;
- the next action is the independent D6 validator/review, not training.
