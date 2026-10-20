# ML Training / Evaluation Completion Plan

**Status:** ACTIVE — M0-M4 complete; R4-D-015 accepts the bounded pilot and keeps M5/full training on HOLD; R4-D-016 resolves `CallToUnknown` semantics; candidate #2 self-contained v2 independent review is the active R4-GAP-007 boundary  
**Parent:** `00_MASTER_TECHNICAL_COMPLETION_ROADMAP.md`  
**Primary authority:** R4 Phase 8–10 governance under `docs/plan/ml-R4/`  

## 1. Objective

Reach an evidence-honest terminal decision for the repaired SENTINEL ML model: train only under an explicitly justified objective/evaluation regime, produce a reproducible checkpoint candidate, evaluate only claims supported by independent evidence, and promote/restrict/reject the candidate without inventing negatives, thresholds, calibration, or acceptance populations.

This plan is deliberately broader than “run training.” A completed GPU run with no defensible evaluation/promotion contract is not ML completion.

## 2. Current executable baseline

Already implemented:

- frozen `four_eye_v8` / `v8.1` architecture;
- deterministic group sampler;
- mask-aware positive-only BCE optimization;
- strong/weak positive weighting;
- BF16 autocast for training and selection;
- gradient accumulation/clipping;
- optimizer/scheduler parameter grouping;
- checkpoint/resume and run binding;
- positive-only model-selection diagnostics;
- strict rejection of authorized supervision cells that are not target `1.0`.

Accepted physical input boundary:

- R4-D-013 accepts `r4-v10-v26-guarded-tokens-v1` with binding digest `9885d7b88a46aff4102741d63eeaa0bd6968f0857f7bca6a389662bfaa158881`;
- R4-D-011 remains the immutable V10 graph/control parent;
- physical DATA/representation completion no longer blocks ML execution, but it does not itself authorize full training.

Current evidence limitation:

- confirmed negatives remain zero;
- R4-D-016 resolves the `CallToUnknown` class boundary as the raw/dynamic low-level unknown-target interaction class while preserving class index/order and taxonomy schema v1;
- candidate #1's historical `NOT_CONFIRMED` rationale is definition-stale under R4-D-016; its queue state remains UNKNOWN/PENDING_REVIEW/target `None` and it requires a fresh primary review if reused;
- candidate #2 primary review remains semantically compatible with R4-D-016 but authoritative state remains UNKNOWN/PENDING_REVIEW/target `None` until admissible independent agreement exists;
- three 2026-10-20 external candidate #2 attempts remain non-authoritative because they were contaminated by prior candidate context and/or lacked sufficient self-contained governing material;
- blind-review v1 is retired for future adjudication;
- accepted negatives, if any, remain evaluation-only unless a later policy explicitly grants optimizer authority;
- threshold-fit, calibration-fit and untouched-acceptance roles remain unsupported/empty;
- R4-D-015 confirms that M4 bounded mechanics/recovery/dynamics pass but does not establish discrimination or false-positive behavior;
- G8 is open and M5/full training is unauthorized;
- Run12 weights, optimizer state, thresholds and calibration are historical operational state and must not seed repaired Phase-8 truth.

Closed semantic-alignment evidence:

- `docs/plan/ml-R4/adrs/ADR-R4-016-call-to-unknown-semantic-clarification.md`;
- `docs/plan/ml-R4/evidence/2026-10-20_call_to_unknown_semantic_alignment/decision.json`;
- `docs/plan/system-finalization/technical-completion/working/2026-10-20_ml_gap007_call_to_unknown_semantic_alignment.md`.

Current blind-review builder:

`docs/plan/ml-R4/scripts/p8_build_gap007_candidate2_independent_review_v2.py`

## 3. Primary source owners to audit

At minimum:

- `ml/src/datasets/` repaired/vNext adapters;
- `ml/src/training/vnext_losses.py`;
- `ml/src/training/vnext_epoch.py`;
- `ml/src/training/vnext_runner.py`;
- `ml/src/training/vnext_binding.py`;
- `ml/src/training/vnext_repaired_binding.py`;
- `ml/src/training/vnext_run_control.py`;
- `ml/src/training/vnext_checkpoint.py`;
- `ml/src/training/vnext_phase8_config.py`;
- `ml/src/training/vnext_param_groups.py`;
- `ml/src/training/vnext_model_factory.py`;
- model architecture/config owners;
- inference checkpoint-consumer requirements;
- R4 role/manifests and Phase-8 evidence records;
- relevant ML tests and micro-smokes.

For current R4-GAP-007 work, R4-D-016 is semantic authority. Do not reopen its definition merely to obtain a preferred candidate result. Review-package work must preserve source completeness, self-contained neutral policy, and reviewer isolation.

## 4. Work package M0 — current training-seam audit

**State:** `COMPLETE` — source audit closed 2026-10-01; canonical historical runner remains G7/v9-bound, R4-D-013 integration requirements were mapped for M3, and role/loss/checkpoint boundaries were verified fail-closed

Before changing objective or launching a run:

1. Trace exact current dataset adapter → collate → masks/strengths → loss → selection metrics → checkpoint binding.
2. Confirm which lineage/digest the durable runner currently accepts and identify every seam that must change to consume the accepted R4-D-013 guarded-token lineage.
3. Confirm no `INTERNAL_AUDIT` or later-role leakage reaches optimization/model selection.
4. Confirm frozen architecture compatibility with V10 graphs and `[4,512]` guarded tokens.
5. Audit checkpoint metadata needed by inference and later ZKML distillation.
6. Re-run or update bounded micro-smoke tests after any seam modification.

**Exit:** exact training inputs, role boundaries and binding requirements are known.

## 5. Work package M1 — supervision/evaluation evidence investigation

**State:** `COMPLETE_WITH_R4_GAP_007_OPEN` — positive-only limits quantified; PU/nnPU not justified under current labeling-mechanism/class-prior evidence; supported/unsupported evaluation roles mapped; R4-D-016 closes the semantic-definition sub-blocker; candidate #2 admissible independent review remains the active evidence continuation

This is the central unresolved ML design responsibility.

### M1-A — confirmed-negative evidence

- use R4-D-016 as the governing `CallToUnknown` definition;
- preserve candidate #1 queue truth as UNKNOWN/PENDING_REVIEW/target `None`; do not reuse its historical definition-stale `NOT_CONFIRMED` rationale as current semantic authority;
- preserve candidate #2 as UNKNOWN/PENDING_REVIEW/target `None` until admissible independent agreement exists;
- retain candidate #2's existing source-first primary review as compatible primary evidence;
- do not let the primary reviewer self-verify candidate #2;
- do not count the three 2026-10-20 external attempts as independent verification or disagreement votes;
- retire candidate #2 blind-review v1 for future adjudication;
- build/verify the self-contained candidate #2 v2 blind bundle containing complete source, manifest/hashes, exact neutral R4-D-016 definition/exclusions, generic class-specific confirmed-negative policy, and review task with no prior conclusion;
- use a genuinely isolated reviewer with no candidate-specific prior context; if such context is present, the reviewer must stop as procedurally contaminated rather than issue a substantive verdict;
- any accepted negative remains evaluation-only until a separate optimizer-authority decision exists;
- do not manufacture additional negatives from absence, tool silence, old all-zero labels, model confidence, or unlabeled data.

### M1-B — positive-only learning limits

Quantify what the existing positive-only BCE regime can and cannot establish:

- positive response/recall behavior;
- risk of unconstrained scores on unknown cells;
- collapse/saturation modes;
- class imbalance and weak-positive effects;
- whether selection by positive NLL/probability alone can choose a useful checkpoint without discrimination evidence.

M4-C supplies bounded evidence that positive fit can become near-saturated by epoch 8. That strengthens the need for negative-side observability; it does not establish classifier quality.

### M1-C — candidate alternative objectives

If evidence justifies exploring Positive–Unlabeled (PU) learning or another objective:

- research the statistical assumptions;
- identify what class priors or risk estimators would be required;
- determine whether SENTINEL data satisfies those assumptions;
- design bounded offline experiments before changing production training code;
- compare against positive-only control;
- explicitly reject the approach if assumptions cannot be defended.

PU is a candidate investigation, not current authority.

### M1-D — evaluation-role feasibility

Determine what can honestly support:

- model selection;
- discrimination metrics;
- threshold fitting;
- calibration;
- internal audit;
- untouched acceptance.

For every unsupported role, keep it unsupported. Do not reuse a role merely because another role is empty.

**Exit:** a written evidence-backed recommendation for objective and evaluation design exists.

## 6. Work package M2 — versioned objective/evaluation decision

**State:** `COMPLETE` — R4-D-014 accepts `masked_positive_bce_control_v1` for bounded control/pilot use only, rejects current PU/negative authority, fixes positive-only evaluation interpretation, and keeps full training on HOLD

The decision specifies:

- optimizer-authorized cell types;
- role eligibility;
- treatment of strong versus weak positives;
- treatment of unlabeled cells;
- whether any confirmed negatives are optimizer-authorized or evaluation-only;
- exact model-selection metric(s);
- whether threshold/calibration work is supported;
- class-specific unsupported/provisional states;
- stopping/selection rules;
- architecture remains frozen unless evidence forces a separately governed change.

If this changes R4 semantic policy, record it through the existing decision/ADR system.

## 7. Work package M3 — final training-lineage integration

**State:** `COMPLETE` — repository-safe guarded seam plus protected-local no-step preflight reviewed PASS; exact D9/D11/D13/D14 binding digest `5d2b913ecdd807cbf505a4a70b30e0ba4db6e28447d3e24a6122ad24fcfcf852`; zero optimizer steps executed.

After DATA acceptance and M2:

1. Update the repaired dataset/binding seam to consume the exact accepted guarded-token/V10 lineage.
2. Bind:
   - source commit;
   - DATA publication/logical role identities;
   - physical representation digest;
   - selector identity;
   - model architecture/config;
   - GraphCodeBERT revision;
   - objective version/config;
   - seed;
   - optimizer/scheduler config.
3. Make stale/historical lineage selection fail closed.
4. Preserve explicit historical reproduction paths separately.
5. Ensure checkpoint metadata is sufficient for inference and downstream distillation lineage checks.

## 8. Work package M4 — preflight and bounded pilot

**State:** `COMPLETE` — R4-D-015 accepts M4-A/B/C bounded mechanics, recovery and positive-fit dynamics evidence. M4-C completed 8 epochs / 120 optimizer steps under the exact guarded binding; no full-training authority follows.

Completed evidence includes:

- CPU/static contract tests;
- dataset-role/count assertions;
- deterministic sampler checks;
- one-step BF16 CUDA forward/backward smoke;
- gradient finiteness and optimizer-step accounting;
- checkpoint save/resume/RNG recovery proof;
- fail-closed changed-binding rejection;
- positive-only selection-metric generation;
- GPU memory/runtime measurement;
- bounded eight-epoch dynamics pilot;
- no Run12 weight/state contamination.

M4-C reached strong positive fit (`positive_nll≈0.00109`, mean positive probability `≈0.9989`) without negative-side observability. That is accepted as bounded dynamics evidence only.

**Exit:** R4-D-015 / `PASS_BOUNDED_PILOT_FULL_TRAINING_STILL_HELD`.

The dated `2026-10-03_ml_m4_bounded_pilot.md` working record contains historical in-progress stop lines from execution. It is no longer the current restart instruction; R4-D-015 and this plan control the post-M4 state.

## 9. Work package M5 — repaired training execution

**State:** `BLOCKED / NOT AUTHORIZED` — R4-D-015 keeps the full horizon on HOLD. R4-D-016 resolves `CallToUnknown` semantics, but R4-GAP-007 confirmed-negative evidence remains zero and candidate #2 still lacks admissible independent verification. R4-D-014 remains the bounded objective/evaluation authority and G8 remains open.

Only after a new explicit launch authority:

- run from fresh initialization under the frozen architecture unless a later decision says otherwise;
- retain deterministic seeds and exact environment/runtime identities;
- persist epoch/checkpoint/run-state evidence;
- make resume source/config compatible and fail closed on mismatches;
- record resource telemetry and anomalies;
- do not tune on internal-audit/acceptance roles;
- do not change thresholds/calibration during the training run unless the accepted design explicitly includes a separate eligible fitting role.

The nominal 100-epoch setting is a configuration value, not current authority. Do not start M5 merely because M4 passed.

### Re-entry condition for M5 review

1. Build and verify the self-contained R4-GAP-007 candidate #2 v2 blind-review bundle under R4-D-016.
2. Obtain genuinely independent candidate #2 verification from a reviewer with no candidate-specific prior context.
3. Formally adjudicate candidate #2 against the primary and independent evidence while preserving uncertainty.
4. Record any newly accepted negative as evaluation-only unless a separate decision grants optimizer authority.
5. Reassess objective/evaluation/full-training authority explicitly after new evidence exists.
6. Do not infer that one or a few negatives automatically support stable FPR estimation, threshold fitting or calibration.

## 10. Work package M6 — checkpoint selection and evidence-qualified evaluation

Evaluate only with eligible populations and only after an authorized candidate exists.

Potential evidence categories, depending on what later evidence establishes:

- positive NLL/probability/recall diagnostics;
- class-specific positive coverage;
- discrimination metrics only where true negatives exist;
- calibration metrics only on an eligible calibration population;
- thresholds only on an eligible threshold population;
- internal-audit behavior;
- workflow utility/regression against static/agent stages;
- stability across seeds/checkpoints where feasible;
- unsupported-class identification.

Every reported metric must name:

- population/role;
- class support;
- source commit;
- checkpoint identity;
- representation/data identity;
- objective/config identity;
- threshold/calibration artifact identity if applicable.

## 11. Work package M7 — promotion decision

Allowed terminal decisions:

- **FULL_PROMOTION** — only if all intended claims are supported;
- **PARTIAL_PROMOTION** — supported classes/capabilities promoted, others restricted;
- **RESEARCH_ONLY** — checkpoint useful for research but not runtime replacement;
- **REJECTED** — candidate fails evidence/quality/compatibility gates;
- **HOLD** — evidence remains insufficient.

Promotion must explicitly state:

- which classes are enabled;
- verdict language allowed;
- threshold/calibration status;
- operational checkpoint identity;
- rollback checkpoint;
- downstream ZKML consequence;
- whether Run12 is retired, retained as fallback, or remains operational baseline.

## 12. Handoff to inference

Inference receives a candidate only after M7 authorizes it.

Required handoff:

- exact operational checkpoint identity and digest;
- architecture/config/version;
- DATA/representation lineage;
- class-support matrix;
- threshold/calibration artifact identities where supported;
- verdict/policy restrictions;
- rollback checkpoint;
- downstream ZKML consequence.

Until that handoff exists, Run12 remains the operational baseline and M4 checkpoints remain bounded research evidence only.
