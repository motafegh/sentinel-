# ML Training / Evaluation Completion Plan

**Status:** ACTIVE — blocked on accepted guarded-token physical lineage and evidence/design resolution  
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

Current evidence limitation:

- confirmed negatives remain zero;
- candidate #2 has only primary-review support and requires genuinely independent agreement before authoritative negative truth can change;
- accepted negatives, if any, are evaluation-only unless a later policy explicitly grants optimizer authority;
- threshold-fit, calibration-fit and untouched-acceptance roles remain unsupported/empty;
- G8 is open and full training is unauthorized;
- Run12 weights, optimizer state, thresholds and calibration are historical operational state and must not seed repaired Phase-8 truth.

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

## 4. Work package M0 — current training-seam audit

**State:** `AUDITING`

Before changing objective or launching a run:

1. Trace exact current dataset adapter → collate → masks/strengths → loss → selection metrics → checkpoint binding.
2. Confirm which lineage/digest the durable runner currently accepts and identify every seam that must change after DATA guarded-token acceptance.
3. Confirm no `INTERNAL_AUDIT` or later-role leakage reaches optimization/model selection.
4. Confirm frozen architecture compatibility with V10 graphs and `[4,512]` guarded tokens.
5. Audit checkpoint metadata needed by inference and later ZKML distillation.
6. Re-run or update bounded micro-smoke tests after any seam modification.

**Exit:** exact training inputs, role boundaries and binding requirements are known.

## 5. Work package M1 — supervision/evaluation evidence investigation

**State:** `INVESTIGATING`

This is the central unresolved ML design responsibility.

Investigate separately:

### M1-A — confirmed-negative evidence

- preserve candidate #1 as `NOT_CONFIRMED`;
- do not let the primary reviewer self-verify candidate #2;
- if independent review is available, process it through the existing R4 evidence protocol;
- any accepted negative remains evaluation-only until a separate optimizer-authority decision exists;
- do not manufacture additional negatives from absence, tool silence, old all-zero labels, model confidence, or unlabeled data.

### M1-B — positive-only learning limits

Quantify what the existing positive-only BCE regime can and cannot establish:

- positive response/recall behavior;
- risk of unconstrained scores on unknown cells;
- collapse/saturation modes;
- class imbalance and weak-positive effects;
- whether selection by positive NLL/probability alone can choose a useful checkpoint without discrimination evidence.

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

**State:** `NOT_STARTED`

The decision must specify:

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

Before a long run:

- CPU/static contract tests;
- dataset-role/count assertions;
- deterministic sampler checks;
- one/few-batch forward/backward smoke;
- BF16 parity contract;
- gradient finiteness;
- optimizer-step accounting;
- checkpoint save/resume equivalence;
- selection-metric generation;
- GPU memory peak and runtime measurement;
- failure/restart behavior;
- no Run12 weight/state contamination.

If the objective itself is new, run a bounded pilot long enough to expose obvious collapse/saturation before authorizing the expensive horizon.

**Exit:** explicit launch/no-launch record.

## 9. Work package M5 — repaired training execution

Only after explicit launch authority:

- run from fresh initialization under the frozen architecture unless a later decision says otherwise;
- retain deterministic seeds and exact environment/runtime identities;
- persist epoch/checkpoint/run-state evidence;
- make resume source/config compatible and fail closed on mismatches;
- record resource telemetry and anomalies;
- do not tune on internal-audit/acceptance roles;
- do not change thresholds/calibration during the training run unless the accepted design explicitly includes a separate eligible fitting role.

The nominal 100-epoch setting is a configuration value, not an obligation to complete 100 epochs if the accepted stopping/selection policy chooses otherwise.

## 10. Work package M6 — checkpoint selection and evidence-qualified evaluation

Evaluate only with eligible populations.

Potential evidence categories, depending on what M1/M2 establish:

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

- checkpoint SHA-256/path/version;
- architecture/config metadata;
- class schema/order;
- graph/token/preprocessing lineage;
- raw-score semantics;
- approved threshold/tier artifact if any;
- calibration artifact if any;
- model hash/provenance requirements;
- unsupported/restricted classes;
- rollback identity.

## 13. Stop/fail conditions

Stop and investigate if:

- guarded-token physical lineage is not accepted;
- role leakage appears;
- target zero enters optimization without explicit authority;
- unlabeled cells are silently treated as negatives;
- objective assumptions cannot be defended;
- thresholds/calibration are fitted on reused/ineligible populations;
- checkpoint metadata cannot bind its training lineage;
- Run12 state contaminates repaired training;
- non-finite gradients/loss recur unexplained;
- selection metric rewards a degenerate solution with no independent way to detect it;
- proposed promotion exceeds available evidence.

## 14. Completion criteria

ML is complete only when SENTINEL reaches an explicit, reproducible terminal model decision and the operational checkpoint status is unambiguous. “Training finished” alone is not completion, and lack of sufficient evidence is allowed to end in `HOLD`, `RESEARCH_ONLY`, or `REJECTED` rather than a forced promotion.
