# DATA / Representation Completion Plan

**Status:** ACTIVE — first critical workstream  
**Parent:** `00_MASTER_TECHNICAL_COMPLETION_ROADMAP.md`  
**Primary authority:** `docs/plan/ml-R4/` decisions/evidence, especially R4-D-011 and R4-D-012  

## 1. Objective

Construct, validate, bind and separately accept the fresh physical token/representation lineage authorized by R4-D-012, without modifying the accepted R4-D-011 V10 V2.6 root.

The immediate technical gap is explicit: the production representation path still uses the historical four-window linspace selector or byte-copies accepted historical token tensors. R4-D-012 has promoted `target_aware_guarded_v1` only as policy/evidence for a **new versioned candidate**. The new physical lineage does not yet exist.

## 2. Immutable inputs and constraints

Preserve unchanged:

- accepted V10 V2.6 graph lineage under R4-D-011;
- exact 22,540-identity population and accepted graph binding;
- accepted V10 graph schema and extractor semantics;
- accepted preprocessing identity;
- logical V3 leakage grouping/roles;
- frozen token tensor shape `[4,512]`;
- GraphCodeBERT model/revision contract where applicable;
- historical linspace selector as an explicit control;
- all historical token/graph roots and decision evidence.

This plan does **not** authorize:

- mutation of R4-D-011 artifacts;
- architecture changes;
- training;
- target-zero creation;
- threshold/calibration work;
- silent replacement of the historical selector in old artifact builders.

## 3. Primary source owners to audit

At minimum inspect before implementation:

- `ml/src/data_extraction/windowed_tokenizer.py`
- `data_module/sentinel_data/representation/tokenizer.py`
- `data_module/sentinel_data/representation/r4_orchestrator.py`
- `data_module/sentinel_data/representation/r4_target_spans.py`
- `data_module/sentinel_data/representation/r4_sensitivity.py`
- representation version registries/constants
- current R4 candidate/binding/validation utilities
- relevant `data_module` and `ml` tests
- R4-D-012 selector evidence and machine records under `docs/plan/ml-R4/`

If the exact research implementation of `target_aware_guarded_v1` exists outside the durable build path, locate and compare it before deciding whether to promote/refactor it or implement a clean production equivalent. Do not reconstruct semantics from memory when machine evidence/source is available.

## 4. Work package D0 — source/evidence reconstruction

**State:** `COMPLETE` — exact selector/source contract reconstructed and recorded

Tasks:

1. Trace the current four-window selection path from raw Solidity through tokenizer output to saved `.tokens.pt` and sidecar metadata.
2. Trace how R4 target spans/evidence are represented and where they can safely enter selector logic.
3. Reconstruct the exact `target_aware_guarded_v1` decision semantics from R4-D-012 evidence:
   - improvement criterion;
   - guard conditions;
   - control fallback conditions;
   - deterministic tie-breaking;
   - over-cap versus under-cap behavior;
   - expected selector metadata.
4. Identify every consumer that assumes historical selected-window indices or historical token lineage.
5. Identify current tests that prove linspace behavior and those that must remain as historical-control tests.
6. Record any ambiguity in a dated working record before coding.

**Exit:** one unambiguous selector contract and versioning plan exists; no decision semantic is inferred ad hoc.

## 5. Work package D1 — selector interface and lineage design

**State:** `COMPLETE` — fresh guarded selector/lineage interface defined

Design requirements:

- historical selector remains callable as a named/versioned control, e.g. `historical_linspace_v1`;
- new selector is explicitly named `target_aware_guarded_v1`;
- selection returns both selected window indices and structured decision evidence;
- output remains exactly four real/padded windows under the frozen tensor contract;
- any failure to establish valid target-aware evidence falls back according to the accepted guard policy, never to an invented heuristic;
- fallback reason is explicit;
- selection is deterministic for identical source, target evidence and config;
- selector policy/version becomes part of the artifact identity/binding;
- old artifact lineages retain their original selector identity.

Candidate metadata fields should be evaluated, not blindly fixed, but must be sufficient to answer:

- which selector was requested;
- which selector actually produced the output;
- whether fallback occurred and why;
- candidate/control selected indices;
- relevant target-span identity/evidence;
- pre-subsampling token/window counts;
- retained coverage diagnostics;
- tokenizer/model/config identity.

**Exit:** interface and lineage fields are reviewable and do not mutate existing artifact semantics.

## 6. Work package D2 — implementation

**State:** `COMPLETE` — bounded guarded-token candidate path implemented; historical paths unchanged

Implementation goals:

1. Refactor selection from `windowed_tokenizer.py` only as needed to support explicit selector strategies without changing historical behavior.
2. Implement `target_aware_guarded_v1` as a separate strategy/module with one clear responsibility.
3. Keep comment stripping/tokenization mechanics separate from selection policy.
4. Integrate target-span inputs through an explicit typed/validated boundary.
5. Make invalid/missing target evidence a structured fallback/error path according to D-012, not an implicit clean result.
6. Add a fresh candidate build mode to `r4_orchestrator.py` or a focused successor module rather than overloading the accepted D-011 path.
7. Ensure graph bytes/identity remain tied to the accepted V10 parent unless a separately discovered blocker requires a new decision.
8. Give the new token lineage a fresh version/name; do not reuse `accepted_v9_byte_copy` semantics.

## 7. Work package D3 — unit and property validation

**State:** `COMPLETE` — exact-head substantive repository CI passed at `ccf49bfae504c82c192d72499ba8766f0f185376`; inherited baseline `git diff --check` debt remains outside this tranche

Required tests include:

- historical linspace indices remain byte/behavior compatible for historical-control mode;
- under-cap inputs select all available windows consistently;
- over-cap inputs are deterministic;
- guarded selector improves/retains target coverage according to its accepted metric;
- every accepted guard/fallback branch is exercised;
- malformed/missing target evidence cannot silently claim target-aware selection;
- shape remains `[4,512]`;
- padding behavior remains stable;
- tokenizer identity and coverage evidence remain correct;
- repeated runs produce identical selected indices and token tensors;
- selector metadata round-trips through saved token artifacts/sidecars;
- no old candidate/root is overwritten.

Where practical, add property-style tests around index bounds, monotonic validity, duplicate-index prevention and deterministic tie handling.

**Exit:** local unit/property suite passes and historical-control compatibility remains intact.

## 8. Work package D4 — bounded candidate validation

**State:** `READY_FOR_PROTECTED_LOCAL_VALIDATION` — not yet executed; D5 remains blocked

Before full-population generation, run bounded tranches covering:

- under-cap contracts;
- over-cap contracts improved by the research selector;
- expected control-fallback contracts;
- worst-case long contracts used in prior CUDA safety evidence;
- multiple vulnerability classes/target-span shapes;
- the declared Slither runtime exception identity where relevant to complete representation assembly.

Compare candidate versus historical control on:

- selected indices;
- target-span coverage;
- retained token coverage;
- tensor shape/dtype;
- determinism;
- graph identity;
- artifact metadata;
- runtime/memory safety.

**Stop condition:** any regression outside the accepted D-012 semantics blocks full generation until explained.

## 9. Work package D5 — full physical candidate

**State:** `BLOCKED_ON_D4` — the bounded builder explicitly rejects implicit full-population execution

Generate the full protected-local 22,540-identity candidate only after D0–D4 pass.

Required outputs:

- 22,540 graph/token/sidecar identities or an explicitly justified successor population if a new evidence-backed blocker is discovered;
- exact candidate version names;
- selector identity for every token artifact;
- counts for target-aware selection versus control fallback;
- zero unexplained missing/duplicate identities;
- zero silent tokenizer failures;
- graph-parent identity proving the accepted V10 lineage was preserved as intended;
- full artifact hash/binding digest;
- source commit and runtime identities;
- reproducibility instructions.

Full generation must occur in a protected local artifact root; Git should contain the binding/evidence records, not unnecessary heavy artifacts.

## 10. Work package D6 — physical acceptance review

**State:** `NOT_STARTED`

The candidate is not authoritative merely because generation completed.

Acceptance must separately establish:

1. population completeness;
2. graph-parent compatibility;
3. selector-policy compliance;
4. deterministic regeneration on required probes;
5. artifact/schema validity;
6. no unexplained candidate/control regressions;
7. exact digest and source/runtime binding;
8. preservation of R4-D-011 as immutable control;
9. explicit decision to accept, reject or revise the new lineage.

If acceptance changes durable representation authority, record the decision in the existing R4 decision/ADR system rather than inventing a parallel governance mechanism.

## 11. Handoff to ML

The DATA plan may hand off to ML only with:

- accepted token/representation lineage name;
- exact physical digest;
- source commit;
- selector version/config identity;
- graph parent identity;
- logical V3 role/publication identities;
- validation/acceptance record;
- explicit statement of what DATA acceptance does **not** prove about model quality.

No training authorization is implied by this handoff.

## 12. Stop/fail conditions

Stop and investigate if any of the following occurs:

- D-012 semantics cannot be reconstructed exactly;
- historical-control mode changes unexpectedly;
- accepted V10 graph bytes/semantics drift without an approved reason;
- guarded selection regresses an evidence class that D-012 said must not regress;
- target evidence is ambiguous but selection proceeds anyway;
- output shape/schema changes;
- full population has unexplained missing/duplicate identities;
- binding cannot identify selector policy/config;
- candidate requires architecture change.

## 13. Completion criteria

This module is `ACCEPTED` when a fresh guarded-token physical lineage has been generated, bound, independently reviewed under the R4 process, and explicitly accepted or when the guarded candidate has been explicitly rejected with a documented successor decision. Merely implementing the selector is not completion.
