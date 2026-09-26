# DATA D0 — selector contract reconstruction working record

Date: 2026-09-26
Work package: D0 — source/evidence reconstruction
State: DESIGN_READY — D0 complete; D1 next
Branch: `work/data-d0-selector-contract`
Base: `main@57f39d652cbec1092084b4efbf41bec6a117ba07`

## Exact question

Reconstruct the executable contract required for the R4-D-012
`target_aware_guarded_v1` selector, identify the production integration seam,
and identify historical behavior that must remain immutable before any D1/D2
design or implementation.

This record is an investigation aid only. R4 decisions/ADRs and executable
source/config/tests remain authoritative.

## Governing boundary

- R4-D-011 accepts the exact V10 V2.6 physical root and digest as immutable.
- R4-D-012 promotes `target_aware_guarded_v1` only for a fresh versioned
  candidate.
- Graph schema remains `v10`; extractor remains
  `v2.6-r4-call-semantics-deterministic-cfg-mutators`.
- Token tensor shape remains `[4,512]`.
- Training remains unauthorized.
- Historical selector behavior must remain available as
  `historical_linspace_v1`.

## Evidence established so far

### 1. Current production selector

`ml/src/data_extraction/windowed_tokenizer.py` still uses the historical
selection rule:

- if `total_windows <= max_windows`, select `range(total_windows)`;
- otherwise select
  `[round(i) for i in np.linspace(0, total_windows - 1, max_windows)]`.

The live DATA adapter
`data_module/sentinel_data/representation/tokenizer.py` re-exports that ML
implementation. No production guarded-selector interface exists there today.

### 2. Current representation-build seam

`data_module/sentinel_data/representation/r4_orchestrator.py` is the current
strict graph/token/sidecar construction seam.

For the accepted V10 path it requires an accepted historical token source and
copies the accepted token payload. Its V10 sidecar records
`token_lineage = "accepted_v9_byte_copy"`. Therefore a guarded candidate
cannot be introduced by silently changing this accepted path; it needs a fresh
versioned build mode/lineage.

When tokenization is performed directly, the orchestrator writes:

- `<sha256>.tokens.pt`;
- `<sha256>.rep.json`;
- coverage fields including `selected_window_indices`;
- the frozen `[4,512]` token tensors.

### 3. Retained selector implementation

The retained research implementation is
`ml/src/data_extraction/bounded_window_selector.py`.

Named strategies:

- `historical_linspace_v1`;
- `target_aware_greedy_v1`;
- `target_aware_guarded_v1`.

The retained guarded semantics currently reconstruct as follows:

1. Build historical control indices with the exact linspace rule.
2. Convert requested target-contract character spans to token ranges.
3. Greedily choose windows by maximum **marginal union target-token coverage**.
4. Greedy ties are deterministic: the lowest window index wins because
   candidates are compared as `(gain, -index)`.
5. Stop greedy target selection when no candidate has positive marginal gain.
6. Fill any remaining slots first from historical linspace indices, then from
   remaining indices in ascending order.
7. Sort the final selected indices.
8. For `target_aware_guarded_v1`, use the greedy candidate only when its
   target-token coverage is **strictly greater** than the historical control.
   If coverage is equal or lower, use the historical control.
9. The research tokenizer pads fewer than four real windows to preserve exact
   `[4,512]` tensors.

This establishes the central improvement criterion and fallback rule without
inventing new semantics.

### 4. Target evidence and provenance — D0 item 2 CLOSED

The target chain is explicit and fail-closed:

`ingestion manifest entry -> repaired preprocessing source_records[].ingestion_entry -> _explicit_target_from_provenance() / meta.contract_names -> _select_targets() -> resolve_file_graph_targets() -> requested_contract_names sidecar -> target_contract_char_spans() -> token target ranges -> selector`.

#### Upstream provenance

`r4_pipeline._materialize()` preserves every contributing ingestion entry in
`meta["source_records"][*]["ingestion_entry"]`. It also persists
`meta["contract_names"]`, derived during preprocessing from the exact repaired
source.

`r4_orchestrator._explicit_target_from_provenance()` scans all preserved source
records for the explicit-target fields:

- `target_contract_name`;
- `contract_name`;
- `label_contract_name`.

If exactly one distinct non-empty name exists, it is authoritative. If multiple
distinct explicit names exist, representation construction fails with
`TargetSelectionError`; conflicting provenance is never guessed away.

#### File-graph target resolution

`_select_targets()` reads the repaired `.sol` and calls
`resolve_file_graph_targets()` with:

- the unique explicit target when available;
- otherwise the preprocessing `contract_names` as provenance names.

The resolver behavior is deterministic:

1. An explicit target must name an application contract. Unknown targets,
   libraries and interfaces fail closed.
2. Without an explicit target, provenance-matching contract declarations are
   preferred; if none match, all declared contracts are considered.
3. Inheritance parents are removed and all remaining inheritance leaves are
   retained. Therefore unrelated application leaves become a file-level union
   rather than one guessed target.
4. A library-only source retains executable libraries.
5. Interface-only/no-executable-target sources fail closed.

The resulting ordered tuple is passed to graph extraction and is persisted in
the representation sidecar as `requested_contract_names`. Actual extracted
targets are independently recorded as `actual_contract_names`.

#### Selector target spans

The durable selector research reads `requested_contract_names` from the
sidecar. Empty target lists are rejected.

`r4_target_spans.target_contract_char_spans()` then operates on the exact
repaired source, requires every requested name to resolve exactly once, masks
strings/comments while preserving offsets, and requires balanced declaration
braces. It returns one exact declaration/body character span per requested
target.

Those spans are converted against GraphCodeBERT offset mappings into token
ranges. Multi-target file unions remain multi-target: coverage is the union
across all requested target token ranges.

This means selector relevance is bound to the **same requested file-graph
target identity that produced the representation**, not to a later label guess
or vulnerability-class heuristic.

Malformed, missing or contradictory target evidence currently raises instead of
silently selecting windows. Whether D1 should expose any of those states as a
structured control fallback is intentionally deferred to D0 item 5; item 2 does
not invent that policy.

### 5. Production-token-view alignment — D0 item 1 CLOSED

The authoritative repaired source path is:

`raw Solidity -> flatten_contract() -> normalize(..., preserve_line_structure=True) -> compile exact normalized bytes -> stage -> content-addressed repaired .sol -> representation/tokenization`.

Source evidence:

- `r4_pipeline._prepare_one()` normalizes `flat.content` with
  `preserve_line_structure=True`.
- `normalizer.normalize()` calls `strip_comments_lexically()`, replaces
  non-newline comment characters with spaces, preserves newlines, strips
  trailing horizontal whitespace, preserves blank-line structure in R4 mode,
  and ensures a terminal newline.
- The SHA used as `normalized_text_sha256` and therefore the repaired artifact
  filename is computed from that exact normalized content.
- The exact normalized content is compiled before promotion.
- The same exact normalized content is written to staging and then copied into
  the repaired physical `<normalized_text_sha256>.sol` artifact.
- `r4_orchestrator._extract_one()` therefore correctly invokes
  `tokenize_windowed_contract_strict(..., strip_comments=False)` for direct
  repaired tokenization. Comment removal has already happened upstream.

The repaired `.sol` bytes, not the original raw or flattened bytes, are thus
the authoritative selector/tokenizer source view.

#### Research selector compatibility

`bounded_window_selector.prepare_source_for_tokenization()` calls
`strip_comments_lexically()` again. On a valid repaired R4 artifact this is
not a newly authorized preprocessing transform: the artifact is already
comment-free except for comment-like marker text protected inside Solidity
strings, which the lexical scanner preserves.

This compatibility call is byte-idempotent for the repaired source contract
established above. The preprocessing regression suite additionally proves
`normalize(..., preserve_line_structure=True)` is idempotent and preserves
comment markers inside strings.

D1/D2 should therefore treat **the persisted repaired `.sol` bytes as the
single authoritative input view**. A production selector interface should not
create an independent second normalization policy. If it retains a defensive
idempotence check, that check must prove byte equality or fail explicitly rather
than silently changing the source view.

No contradiction was found between preprocessing, historical tokenization,
selector research, or R4-D-012.



### 7. Lineage/version and binding boundary — D0 item 3 CLOSED

R4-D-012 requires a **new physical token/representation lineage**, not a mutation
of either the accepted V9 token population or the accepted R4-D-011 V10 root.

#### Identities that remain inherited/frozen

The guarded candidate must continue to bind to the accepted R4-D-011 parent
semantics:

- population identity: the same 22,540 source/contract identities unless a later
  evidence-backed blocker is separately decided;
- preprocessing parent: `sentinel-preprocessed-r4-v2`;
- graph schema: `v10`;
- graph extractor:
  `v2.6-r4-call-semantics-deterministic-cfg-mutators`;
- graph bytes for each identity: unchanged from the accepted R4-D-011 parent;
- graph target semantics: unchanged requested/actual file-graph targets;
- Slither/crytic runtime provenance and the exact 22,539 primary + 1
  identity-bound-exception partition;
- frozen token tensor shape: `[4,512]`;
- GraphCodeBERT tokenizer/model/window/stride contract.

R4-D-011's exact root, digest, token files, sidecars and machine-readable
acceptance record remain immutable historical authority.

#### Identities that must be new

The guarded candidate must introduce a fresh identity for the parts changed by
R4-D-012:

- physical candidate/root identity;
- token/representation lineage identity;
- selector policy/version identity:
  `target_aware_guarded_v1`;
- selector configuration identity where configuration affects semantics;
- per-artifact selector decision metadata;
- token payload hashes;
- sidecar hashes, because selector/coverage metadata changes;
- full candidate binding digest;
- source commit/runtime/build identity for the candidate generation;
- later physical-acceptance decision/evidence identity if the candidate passes.

Final constant/root names are deliberately **not fixed in D0**. D1 must choose
explicit names that cannot collide with the accepted R4-D-011 lineage.

#### Why the existing V10 binder cannot be reused unchanged

`sentinel_data.vnext.r4_v10_binding.bind_v10_candidate()` was designed for
R4-D-010/R4-D-011 graph remediation. It intentionally requires:

- `sidecar["token_lineage"] == "accepted_v9_byte_copy"`;
- candidate token bytes exactly equal accepted-V9 token bytes;
- a candidate root named by the current
  `V10_REPRESENTATION_ROOT_NAME`.

Those checks were correct for R4-D-011 and must remain correct for historical
control validation. A guarded candidate is expected to change token payloads on
some identities, so weakening these checks in place would destroy the old
binder's semantic meaning.

D1/D2 therefore require either a focused successor binder or an explicitly
versioned/parameterized binding interface whose **historical mode preserves all
current R4-D-011 checks exactly**.

#### Required guarded-candidate binding semantics

A guarded-lineage binder must prove two different classes of invariants:

**Parent-preservation invariants**

- exact population equality with R4-D-011;
- exact graph-byte equality with R4-D-011 for every identity;
- exact graph schema/extractor identity;
- exact requested/actual target semantics;
- exact graph/runtime provenance;
- no unexplained graph or population drift.

**Declared token-lineage changes**

- selector requested identity;
- selector actually used;
- whether control fallback occurred;
- explicit fallback reason;
- guarded and control selected indices as required by the D0/D1 contract;
- target-span/target-token evidence identity;
- token coverage/retention evidence;
- tokenizer/window/stride/config identity;
- token tensor shape/dtype;
- new token hashes and sidecar hashes.

The binding digest must therefore include enough selector/token lineage metadata
to distinguish the guarded candidate from both R4-D-011 and any future selector
version. Merely relying on a directory name is insufficient.

#### Existing binder precedent retained

The earlier `r4_binding.py` and `r4_v10_binding.py` establish useful
mechanical principles that should remain:

- logical source/contract IDs rather than machine-specific absolute roots are
  bound;
- graph/token/sidecar content hashes participate in the digest;
- malformed or missing triples fail closed;
- tensor shape and graph schema are validated, not inferred from filenames;
- requested/actual targets are checked;
- binder reports remain diagnostic and cannot self-grant physical acceptance or
  training authority.

For the guarded candidate, the V10/R4-D-011 root becomes the physical **parent
control**, replacing accepted V9 as the token-byte-equality authority. Token
byte equality to R4-D-011 is **not** a required invariant; graph byte equality
to R4-D-011 is.

No contradiction was found. The existing version/binding code reflects its
historical decisions correctly; a new lineage needs a new explicit binding
contract rather than edits that reinterpret those old decisions.




### 9. Executable consumer inventory — D0 item 4 CLOSED

The selector-related fields do not all have the same authority. The audit
separates the code that **produces** them, code that **validates** them, code
that uses them for **research diagnostics**, and the actual **model runtime**.

#### A. Candidate-build / metadata producers

`ml/src/data_extraction/windowed_tokenizer.py`

- produces historical `selected_window_indices`;
- produces pre-subsampling counts, selected token ranges and retained-token
  coverage;
- remains the historical-control implementation.

`data_module/sentinel_data/representation/r4_orchestrator.py`

- writes selector/coverage telemetry into both token payload and sidecar;
- writes `requested_contract_names` / `actual_contract_names`;
- writes historical V10 `token_lineage = "accepted_v9_byte_copy"`;
- is therefore a direct D2 integration surface for a fresh guarded build mode,
  but its accepted R4-D-011 behavior must remain intact.

#### B. Binding / validation consumers

`data_module/sentinel_data/vnext/r4_binding.py`

- validates the token tensor and the sidecar against each other;
- requires equality for the complete coverage field set, including
  `selected_window_indices`, selected ranges and retained-token telemetry;
- validates requested/actual graph target identity;
- historical repaired-v2/v9 binder; not a guarded-lineage semantic owner.

`data_module/sentinel_data/vnext/r4_v10_binding.py`

- reuses the token/sidecar coverage consistency checks;
- additionally consumes `token_lineage`;
- requires historical `accepted_v9_byte_copy` and exact accepted-V9 token
  bytes;
- historical R4-D-011 diagnostic binder; must remain unchanged in meaning.

A guarded candidate therefore needs a successor binding path that validates the
new selector metadata while preserving graph-byte identity to R4-D-011.

#### C. Historical-control evidence consumers

`docs/plan/ml-R4/scripts/p8_verify_v10_bound_token_control_equivalence.py`

- reads `requested_contract_names`;
- recomputes historical target spans and historical-control tokens;
- compares dynamic `selected_window_indices` against both the bound token
  payload and sidecar;
- is tied explicitly to the R4-D-011 acceptance record/root.

This verifier is immutable historical-control evidence and should not be
repurposed into the guarded-candidate binder.

#### D. Selector research / diagnostic consumers

`docs/plan/ml-R4/scripts/p8_compare_bounded_window_selector_v1.py` and the
older bounded-window comparison script:

- consume `requested_contract_names`;
- derive target spans/ranges;
- compare control/greedy/guarded selected indices, target coverage and retained
  coverage;
- are research evidence only.

`docs/plan/ml-R4/scripts/p8_run_selector_gpu_compare.py` and
`p8_run_selector_gpu_compare_v3.py`:

- dynamically replace only token tensors for bounded research;
- consume `requested_contract_names`;
- record selected/control indices, fallback flag, target coverage and retained
  coverage;
- verify control tensor identity where requested;
- do not persist a production physical lineage.

`data_module/sentinel_data/representation/r4_sensitivity.py`:

- consumes `pre_subsampling_window_count` to identify long/worst-case
  contracts;
- does not depend on selected indices or token-lineage identity;
- remains diagnostic-only.

These research consumers are evidence precedents, not the production metadata
contract by themselves.

#### E. Model/dataset runtime consumers

`VNextTrainingDataset`, `RepairedVNextTrainingDataset`, and the logical-V3
dataset path load the token payload but pass only:

- `input_ids`;
- `attention_mask`.

They do **not** interpret `selected_window_indices`, target spans, retained
coverage, fallback state, `token_lineage`, or
`requested_contract_names` during model forward/training.

Therefore the frozen model tensor API does not need to change for the guarded
selector. The model sees a different valid `[4,512]` token tensor, while the
selector semantics remain a representation-lineage concern.

The dataset/training boundary **does**, however, bind the representation digest.
A later ML handoff must therefore consume the newly accepted guarded-lineage
digest rather than silently pointing an old logical publication at new token
bytes.

#### F. Logical DATA / publication consumers

Logical grouping, roles and supervision are selector-independent. The existing
V3 logical publication can remain the semantic parent, but a future training
publication/manifest must explicitly bind the newly accepted guarded physical
digest and its acceptance evidence.

No role, target or label semantics should change merely because token windows
change.

#### G. Stale future-V10 training adapter discovered

`ml/src/datasets/vnext_logical_v3_v10_dataset.py` and its associated
`build_v10_run_binding()` path are fail-closed future-training surfaces, but
they predate R4-D-011 acceptance.

Current source still:

- describes the V10 population as only a candidate;
- expects physical-acceptance schema
  `sentinel-r4-v10-physical-acceptance-v1`;
- while the controlling R4-D-011 machine record is
  `sentinel-r4-v10-v26-physical-acceptance-v1`.

Its tests construct the same old synthetic schema. This is a concrete stale
downstream assumption relative to current R4 authority.

It does **not** block DATA D0/D1 because:

- training remains unauthorized;
- this adapter is not the current DATA build/binding path;
- D0 is reconstructing the new physical lineage before ML handoff.

It **must** be reconciled before a later accepted guarded lineage is handed to
ML. Do not treat its current constants/docstring as authority for D1 naming or
acceptance semantics.

#### Consumer-impact conclusion

D1/D2 changes are required in the representation construction and guarded
binding/validation surfaces. Focused tests must be added there.

The frozen model architecture, forward signature and collate tensor shape do
not require selector-specific fields. Dataset/run-control changes are downstream
lineage-rebinding work after physical DATA acceptance, not part of implementing
the selector itself.

No current executable consumer justifies mutating R4-D-011 metadata in place.




### 11. Guard fallback versus invalid-evidence failure — D0 item 5 CLOSED

R4-D-012 promotes a **guarded comparison policy over valid requested-target
evidence**. It does not authorize treating absent or malformed target evidence
as if target-aware selection succeeded.

#### Valid guard fallback

The exact retained condition is:

`greedy_target_coverage <= historical_control_target_coverage`.

When that condition is true, `target_aware_guarded_v1` selects the historical
control indices.

The safe semantic reason D1 may expose is therefore equivalent to:

`candidate_target_coverage_not_strictly_greater`.

The final spelling/schema is a D1 design choice, but its meaning must remain
exactly this condition. Do not split it into policy reasons that imply evidence
not present in R4-D-012.

The fallback is valid for:

- target-coverage ties;
- any theoretical greedy regression, which the guard converts to control;
- under-cap inputs.

#### Under-cap behavior

When `total_windows <= 4`, historical control selects every real window.

The greedy selector may choose target-overlapping windows first, but its fill
steps then add all remaining available windows. The sorted candidate therefore
contains the same complete real-window set as the control.

Consequences:

- candidate target coverage equals control target coverage;
- the guarded condition falls back to the historical control;
- all real windows are retained;
- output is padded only as needed to maintain exact `[4,512]`;
- under-cap is not a separate selection heuristic.

D1 may record `pre_subsampling_window_count <= max_windows` as descriptive
telemetry, but should not invent a distinct selector policy from it.

#### Invalid evidence is not an accepted fallback

The durable research/evidence path rejects before selection when it cannot
establish valid target evidence. Existing fail-closed cases include:

- missing/empty `requested_contract_names`;
- contradictory explicit target provenance;
- requested target absent from the repaired source;
- requested library/interface where an application contract is required;
- ambiguous/invalid target declaration resolution;
- a requested target resolving other than exactly once;
- missing opening or balanced closing declaration brace;
- empty/invalid character spans;
- a target span mapping to zero GraphCodeBERT tokens;
- empty repaired source or zero tokenizer code tokens;
- tokenizer overflow-window/range-count divergence;
- invalid selector/window configuration.

These are evidence/contract failures, not proof that the historical selector is
the correct target-aware outcome.

R4-D-012's accepted CPU evidence had 1,018/1,018 analyzed records and zero
failures. Its 261 control fallbacks are explicitly the **equal-coverage valid
cases**, not malformed-target cases.

#### Low-level empty-target permissiveness is not promoted semantics

`target_aware_greedy_indices()` returns historical linspace indices when
called directly with an empty `target_ranges` list, and
`char_spans_to_token_ranges()` can return an empty list if an internal caller
passes no spans.

However, the durable CPU/CUDA/control-equivalence paths first require non-empty
`requested_contract_names` and obtain spans through
`target_contract_char_spans()`, which rejects an empty target list.

Therefore the low-level empty-range behavior is an implementation convenience,
not R4-D-012 authority for production fallback. D1/D2 should put a validated
target-evidence boundary in front of the selector so missing/invalid evidence
cannot silently become a successful control selection.

#### D1 metadata/error implication

For a successfully emitted guarded token artifact:

- requested selector identity must be `target_aware_guarded_v1`;
- actual selector may be guarded candidate or historical control;
- `used_control_fallback` must be explicit;
- when fallback is true, the reason must mean
  `candidate target coverage was not strictly greater than control`;
- control and final selected indices and relevant coverage counts must be bound.

For invalid target evidence:

- no artifact may claim successful target-aware selection;
- the build must emit/propagate a structured error/failure record;
- silently substituting historical control would require a new explicit policy
  decision and is not authorized by D0/R4-D-012.

This resolves the DATA-plan phrase "structured fallback/error path": valid
coverage guards use fallback; invalid evidence uses the error path.




### 13. Consolidated selector contract and integration boundary — D0 item 6 CLOSED

The following table is the reconstructed D0 contract. It records only semantics
already established by executable source, tests, R4-D-011, or R4-D-012.

| Concern | Reconstructed contract | Authority / integration consequence |
|---|---|---|
| Authoritative source bytes | Persisted repaired R4 `.sol` bytes after line-preserving lexical normalization | Selector must not introduce a second mutable preprocessing policy |
| Tokenizer | `microsoft/graphcodebert-base`, current frozen tokenizer/runtime contract | Keep tokenizer identity bound in new lineage |
| Window shape | 512 tokens per window, stride 256, maximum 4 real windows | Model input remains exactly `[4,512]` after padding |
| Historical selector | `historical_linspace_v1`: all windows when under cap; otherwise rounded NumPy linspace across `[0,total_windows-1]` | Immutable control; full-population equivalence to R4-D-011 already proven |
| Guarded selector | `target_aware_guarded_v1` | Required selector policy for the new candidate under R4-D-012 |
| Target identity | Exact `requested_contract_names` that produced the file-graph representation | No vulnerability-label guess or later heuristic may substitute |
| Target spans | Exact declaration/body character spans for every requested target, offset-preserving and uniquely resolved | Missing/ambiguous/unbalanced target evidence fails closed |
| Target token ranges | Character spans mapped through GraphCodeBERT raw-token offset mapping | Zero-token target mapping fails closed |
| Candidate objective | Greedily maximize marginal **union requested-target token coverage** | No alternative score/heuristic is authorized |
| Greedy tie break | Maximum `(marginal_gain, -window_index)` | Lowest window index wins deterministic ties |
| Greedy stop | Stop target-directed additions when best marginal gain is `<= 0` | No zero-gain target heuristic |
| Fill order | First missing historical-linspace indices, then remaining indices ascending | Deterministic completion to at most four real windows |
| Final index order | Sort selected real-window indices ascending | Stable physical token ordering |
| Guard criterion | Use candidate only when candidate target coverage is **strictly greater** than historical control | Equal or worse coverage selects historical control |
| Valid fallback reason | Semantically: candidate target coverage not strictly greater than control | Exact metadata spelling is D1 schema design; meaning is fixed |
| Under-cap | All real windows are retained; candidate/control coverage tie; guarded path resolves to historical control | Pad only to `[4,512]`; under-cap is not a new policy |
| Missing/invalid target evidence | Structured build/selection failure | Must not silently become successful historical-control fallback |
| Determinism | Same repaired source + targets + tokenizer/config => same indices and token tensors | D3 must prove repeated-run identity |
| Graph parent | Exact R4-D-011 V10 graph bytes, schema, extractor, targets and runtime provenance | No graph regeneration/drift is authorized by selector work |
| Population parent | Exact R4-D-011 22,540 identities | Any population change needs separate evidence/decision |
| Historical token lineage | R4-D-011 `accepted_v9_byte_copy` remains immutable | Never relabel old token files as guarded |
| New token lineage | Fresh versioned physical identity using `target_aware_guarded_v1` | New root, token hashes, sidecar hashes and binding digest required |
| Per-artifact selector evidence | Requested selector, actual selector, fallback flag/reason, control/final indices, target evidence, coverage and tokenizer/config identity | D1 must define typed/schema fields; D2 must persist them |
| Binding | Parent graph/population identity must equal R4-D-011; token/sidecar changes allowed only in declared selector fields/payload | Existing R4-D-011 binder remains historical; guarded successor binder required |
| Model runtime | Consumes only resulting `input_ids` + `attention_mask`; no selector metadata in forward API | No architecture/forward-signature change |
| Logical DATA roles/labels | Remain logical-V3 authority and selector-independent | Later publication must rebind new physical digest; no role/label redefinition |
| Acceptance | Candidate generation/binding does not self-accept | Separate physical review/decision required before ML handoff |
| Training | Unauthorized | No training action follows from D0-D4 selector work |

#### Required implementation seams identified for D1/D2

The smallest **correct responsibility set**, without conflating unrelated
modules, is:

1. an explicit versioned selector interface/strategy surface around the
   historical and guarded selection logic;
2. a validated target-evidence input boundary using the existing repaired
   source/target semantics;
3. the token-build seam that emits guarded tensors plus structured selector
   metadata;
4. a fresh candidate construction mode/root that reuses the accepted R4-D-011
   graph parent without mutating it;
5. a guarded-lineage binder/validator that proves graph-parent identity and
   declared token-only differences;
6. focused historical-control, guarded-selector, metadata, lineage-isolation
   and determinism tests.

The model architecture, graph extractor semantics, logical role assignment,
training objective and training runner are outside this implementation boundary.

### 14. D0 exit assessment — PASSED

D0 exit criterion from the DATA plan:

> one unambiguous selector contract and versioning plan exists; no decision
> semantic is inferred ad hoc.

Assessment: **PASSED**.

Established without unresolved selector-policy ambiguity:

- exact historical control semantics;
- exact guarded candidate objective and deterministic tie-breaking;
- exact strict-improvement guard;
- exact under-cap behavior;
- exact target-evidence provenance and span/token mapping;
- valid-fallback versus invalid-evidence boundary;
- immutable R4-D-011 parent semantics;
- required new lineage/binding identity;
- executable consumer impact;
- production integration seams;
- historical tests/evidence that must remain controls.

The only stale downstream finding is the future logical-V3/V10 ML adapter's
pre-R4-D-011 acceptance schema. It is outside D0 selector semantics and must be
reconciled before ML handoff, not used to redefine the DATA candidate.

No source/R4 contradiction blocks selector construction. D0 is complete and
the DATA workstream may move to D1 design. No D1 implementation is implied by
this record.


### 15. Historical-control equivalence is already proven

The full-population verifier
`docs/plan/ml-R4/scripts/p8_verify_v10_bound_token_control_equivalence.py`
dynamically reconstructs `historical_linspace_v1` and compares it with
R4-D-011 bound tensors and indices.

Tracked R4 evidence records:

- 22,540 / 22,540 checked;
- exact `input_ids` match;
- exact `attention_mask` match;
- exact selected-window-index match;
- zero mismatches/failures.

This historical behavior is an immutable control requirement for later D1-D3.



### 13. Final D0 selector contract and versioning table — D0 item 6 CLOSED

| Contract dimension | Historical control | Guarded candidate requirement |
|---|---|---|
| selector identity | `historical_linspace_v1` | `target_aware_guarded_v1` |
| source view | persisted repaired R4 `.sol` bytes | exactly the same repaired bytes |
| tokenizer | `microsoft/graphcodebert-base` | unchanged |
| window size | 512 | unchanged |
| stride | 256 | unchanged |
| max real windows | 4 | unchanged |
| tensor shape | `[4,512]` IDs + mask | unchanged |
| under-cap | all real windows, then pad | same output; explicit under-cap/control-equivalence evidence |
| over-cap control | exact historical rounded linspace | retained as comparison/rollback |
| target evidence | not used for control | exact `requested_contract_names` -> declaration/body char spans -> token ranges |
| candidate selection | n/a | deterministic marginal union target-token coverage |
| greedy tie | n/a | lowest window index |
| fill after positive gains | n/a | historical control indices first, then ascending remaining indices |
| guard criterion | n/a | candidate only when target coverage is strictly greater than control |
| equal/lower coverage | n/a | historical control output with explicit reason |
| missing/invalid target evidence | n/a | explicit exceptional control fallback when control tokenization remains valid; preserve error; unexpected in full parent population |
| unsafe source/tokenizer/artifact failure | fail | fail; do not invent control artifacts |
| graph schema | V10 parent under R4-D-011 | unchanged V10 |
| graph extractor | V2.6 parent under R4-D-011 | unchanged |
| graph bytes | immutable R4-D-011 parent | exact byte equality required |
| population | 22,540 R4-D-011 identities | same 22,540 unless a separately governed blocker is discovered |
| runtime provenance | accepted R4-D-011 partition | unchanged |
| token bytes | immutable historical control | may differ only according to declared selector semantics |
| selector metadata | historical coverage/index evidence | requested/actual selector, fallback category/detail, candidate/control indices, target evidence, coverage/config identity |
| physical root/digest | immutable R4-D-011 root/digest | fresh root identity and fresh binding digest |
| training authority | false | false until later independent gates |

#### D0 versioning plan

D1 must create a fresh selector/token lineage without renaming or repurposing
existing R4-D-011 constants. The new lineage must:

1. reference R4-D-011 as its graph/physical parent;
2. retain explicit historical control identity;
3. assign a fresh token/representation lineage identity;
4. bind selector policy/config into token/sidecar identity;
5. bind graph-parent hashes and new token/sidecar hashes into a successor
   candidate digest;
6. keep physical acceptance and training authorization separate from the binder.

Exact constant and root-name spellings are a D1 interface-design decision. The
semantic version boundary itself is no longer ambiguous.

### 14. Historical controls D1-D3 must preserve

The following remain historical-control authorities and must not be rewritten
to make the guarded candidate pass:

1. `ml/src/data_extraction/windowed_tokenizer.py::_selected_window_indices()`
   — exact rounded linspace behavior.
2. `data_module/tests/test_representation/test_windowed_tokenizer_coverage.py`
   — exact over-cap `[0,6,13,19]`, under-cap `[0,1,2]`, frozen shape and
   coverage behavior.
3. `data_module/tests/test_representation/test_bounded_window_selector.py`
   — guarded no-regression, equal-coverage fallback, target-span and
   offset-preservation semantics.
4. `data_module/tests/test_vnext/test_r4_v10_binding.py`
   — R4-D-011-era V10 binder semantics, including accepted-V9 byte-copy lineage
   and token-drift rejection.
5. `docs/plan/ml-R4/scripts/p8_verify_v10_bound_token_control_equivalence.py`
   plus its accepted machine report — 22,540 / 22,540 exact tensor/index
   historical-control equivalence.
6. R4-D-011 / ADR-R4-011 and its machine-readable acceptance record — exact
   immutable graph/token parent authority.
7. R4-D-012 / ADR-R4-012 — authorization for
   `target_aware_guarded_v1` only in a fresh versioned candidate.

New guarded-lineage tests may be added beside these controls. They must not
replace or reinterpret them.

### 15. D0 exit decision

**D0 exit: PASS.**

The required executable semantics can now be reconstructed without ad-hoc
inference:

- source/token view is known;
- target provenance/span mapping is known;
- historical and guarded selection algorithms are known;
- deterministic tie/fill behavior is known;
- under-cap/over-cap guard behavior is known;
- exceptional fallback versus hard-failure boundary is known;
- production integration and consumer boundaries are known;
- immutable parent versus fresh lineage fields are known;
- historical controls are identified.

No source/evidence contradiction requiring a governance stop was found.

The remaining choices are interface/serialization choices, not unresolved
selector semantics. They belong to D1.


## Historical tests that must remain controls

At minimum preserve the intent of:

- `data_module/tests/test_preprocessing/test_r4_repair.py` for lexical
  normalization, line preservation and repaired-normalization idempotence;
- `data_module/tests/test_representation/test_windowed_tokenizer_coverage.py`
  for exact historical linspace indices and frozen-shape behavior;
- `data_module/tests/test_representation/test_bounded_window_selector.py`
  for target-aware coverage, tie fallback, multi-target spans, and
  offset-preserving token-source semantics;
- the full-population R4-D-011 control-equivalence verifier/evidence.

The older research-script test
`test_bounded_window_strategy.py` is historical evidence and must not become
the production semantic owner if the focused selector module already owns the
same behavior.

## D0 remaining audit, in order

1. **CLOSED** — trace the exact preprocessing source view that reaches
   tokenization and establish comment-removal/idempotence semantics.
2. **CLOSED** — trace target-name provenance:
   ingestion/preprocessing metadata -> `_select_targets()` ->
   `requested_contract_names` -> target spans and selector token ranges.
3. **CLOSED** — trace representation/version constants and binding utilities;
   establish inherited R4-D-011 identities versus new guarded-lineage identity
   and the successor-binding boundary.
4. **CLOSED** — enumerate and classify executable consumers of selected
   indices, coverage metadata, token-lineage identity and requested targets.
   The model runtime consumes only resulting tensors; build/binding and
   research/evidence surfaces own selector metadata. A stale future-v10 ML
   adapter is recorded for later handoff reconciliation.
5. **CLOSED** — valid target evidence falls back only when candidate target
   coverage is not strictly greater than control; under-cap naturally ties and
   falls back. Missing/malformed/unresolvable target evidence fails closed and
   must not silently become historical-control output.
6. **CLOSED** — consolidated selector contract and integration/versioning
   boundary recorded above.
7. **CLOSED / PASS** — D0 exit criterion is satisfied; no unresolved selector
   decision semantic remains. D1 design may begin; no production code has yet
   changed.

## Current D0 status

`AUDITING`.

D0 is complete and passes its exit criterion. No production source has been
modified. The working record now supplies the source/evidence reconstruction
needed to begin D1 without re-deriving selector policy.

## Next executable step

Begin D1 design only: define the explicit selector interface and fresh lineage
metadata/schema using the D0 contract. Preserve
`historical_linspace_v1` as an exact named control and do not implement the
candidate build until the D1 interface/versioning design is reviewable.
