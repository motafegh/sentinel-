# Inference / Deployment Promotion Plan

**Status:** ACTIVE — audit may proceed; repaired-model migration waits for ML promotion authority  
**Parent:** `00_MASTER_TECHNICAL_COMPLETION_ROADMAP.md`  

## 1. Objective

Ensure that any repaired R4 checkpoint promoted from ML can be served with the same representation semantics, class schema, score semantics and policy artifacts it was evaluated under, without silently changing live behavior or overstating model quality.

The current operational baseline remains historical Run12. A new checkpoint must earn replacement through explicit compatibility and runtime evidence.

## 2. Current executable baseline

Primary inference sources include:

- `ml/src/inference/api.py`
- `ml/src/inference/predictor.py`
- `ml/src/inference/preprocess.py`
- `ml/src/inference/cache.py`
- `ml/src/inference/drift_detector.py`
- `ml/src/inference/execution_status.py`
- `ml/mlops_config.json` and environment overrides
- model/preprocessing schema owners under `ml/src/models/` and `ml/src/preprocessing/`

Current characteristics:

- FastAPI loads one checkpoint at startup and fails if it is absent;
- checkpoint architecture/class metadata is validated;
- graph schema/edge-embedding compatibility is guarded;
- raw probabilities, tiered outputs, model hash and execution provenance are exposed;
- per-class thresholds may be loaded from a companion artifact;
- default checkpoint remains historical Run12;
- inference preprocessing currently has its own source-to-graph/token path and must not be assumed equivalent to the future guarded-token training lineage without proof.

## 3. Critical parity question introduced by R4-D-012

R4-D-012 improves **requested-contract token coverage** while retaining four `[512]` windows. This is not vulnerability-label leakage, but deployment parity is still a hard requirement.

Before promoting a model trained on the new selector, establish:

1. how the training candidate identifies the requested contract in multi-contract Solidity sources;
2. what equivalent identity is available at live inference time;
3. whether the current API/preprocessor can reproduce the same selector deterministically;
4. what happens when the request contains one contract, inheritance trees, libraries, interfaces, or multiple deployable contracts;
5. whether the public API must add an optional/required target-contract identifier;
6. whether safe deterministic inference fallback is semantically identical to the training/evaluation policy;
7. whether model evaluation must include an inference-parity representation rather than only stored training artifacts.

If live inference cannot reproduce an evidence-equivalent selector, model promotion is blocked until the representation contract is reconciled.

## 4. Work package I0 — source/runtime audit

**State:** `AUDITING`

Trace:

```text
HTTP request
→ validation
→ ContractPreprocessor
→ graph extraction
→ token-window construction/selection
→ Predictor model construction
→ checkpoint load
→ scoring
→ threshold/tier policy
→ response schema
→ agent/MCP consumer
```

Audit especially:

- graph schema version used by live preprocessing;
- token model/window/stride/selection behavior;
- multi-contract target selection;
- comment/source transformations;
- long-source truncation/window behavior;
- checkpoint architecture registry/defaults;
- class order and enabled/disabled-class behavior;
- threshold loading and fallback;
- model-hash/provenance binding;
- cache key semantics;
- drift baseline compatibility;
- API/MCP/agent schema expectations.

Record discrepancies between training representation and live preprocessing even if current Run12 tolerates them.

## 5. Work package I1 — accepted-model compatibility contract

**State:** `NOT_STARTED`

When ML produces a promotion candidate, define an explicit inference compatibility manifest containing or binding:

- checkpoint SHA-256;
- architecture/version;
- class names/order;
- graph schema/extractor compatibility;
- tokenizer name/revision;
- token-window selection policy;
- requested-contract selection semantics;
- input shape;
- raw-output semantics;
- threshold artifact identity if supported;
- calibration artifact identity if supported;
- verdict/tier policy version;
- DATA/representation lineage;
- supported/restricted classes.

Startup must fail closed when the deployed checkpoint and required preprocessing/policy identities are incompatible.

## 6. Work package I2 — train/offline/live preprocessing parity

Build a parity corpus covering:

- single-contract files;
- files with interfaces/libraries plus target contract;
- inheritance-heavy files;
- multiple deployable contracts;
- under-cap and over-cap tokenization;
- contracts where D-012 uses target-aware selection;
- contracts where D-012 falls back to historical control;
- V10 external-call and persistent-storage edge cases.

For each case compare, as applicable:

- selected target contract(s);
- graph schema and meaningful graph structure;
- selected token indices/windows;
- tensor values/shapes;
- source transformation metadata;
- model logits/probabilities under the same checkpoint.

Define tolerances only where exact equality is impossible and justify them. Representation mismatch must never be hidden by accepting output-level similarity alone.

## 7. Work package I3 — score, threshold and verdict policy

Keep these layers separate:

```text
model logits
→ raw probabilities
→ optional calibration
→ threshold/tier policy
→ human-facing verdict language
```

Rules:

- no Run12 threshold/calibration reuse unless independently justified for the repaired model;
- if R4 cannot support calibrated thresholds, expose/restrict policy accordingly rather than fabricating them;
- unsupported classes must not be presented as equally validated outcome claims;
- threshold artifacts must be versioned and bound to the checkpoint/class schema/population used to fit them;
- backward-compatible fields must not silently change semantic meaning.

Audit existing `confirmed` / `suspicious` / legacy `vulnerabilities` behavior against the eventual R4 policy before deployment.

## 8. Work package I4 — service regression and resilience

Validate:

- startup success/failure behavior;
- CPU/GPU loading as intended;
- BF16-trained checkpoint normalization/inference precision;
- warmup catches graph/token/model shape mismatches;
- request-size limits;
- timeout and CUDA OOM behavior;
- concurrent request safety;
- cache correctness across model/version changes;
- drift detector baseline/version compatibility;
- Prometheus/health metadata accurately reports the deployed lineage;
- malformed/missing provenance cannot become eligible evidence downstream;
- no stale checkpoint survives a configuration change unnoticed.

## 9. Work package I5 — offline shadow comparison

Before changing the operational default, compare repaired candidate versus Run12 on a frozen, representative corpus.

This comparison is primarily for:

- runtime regressions;
- schema compatibility;
- representation differences;
- output distribution changes;
- agent-routing consequences;
- class-specific unsupported behavior.

Do not describe Run12 disagreement as proof that either model is correct unless outcome evidence supports it.

## 10. Work package I6 — deployment and rollback

Promotion procedure:

1. record exact accepted ML decision;
2. record inference compatibility/parity evidence;
3. version deployment config;
4. deploy candidate in a controlled environment;
5. execute service/MCP/agent integration smoke;
6. verify reported checkpoint/model hashes;
7. update the default checkpoint only after the above passes;
8. retain a tested rollback to the prior operational bundle.

A checkpoint file appearing in `ml/checkpoints/` is not deployment authorization.

## 11. Handoffs

### To Agents

Provide:

- exact response schema;
- supported/restricted classes;
- tier/threshold semantics;
- execution/provenance contract;
- model hash and availability behavior;
- any changed routing assumptions.

### To ZKML

Provide the exact **accepted teacher checkpoint identity and DATA/model metadata**, not merely the path used by the API. ZKML re-distillation begins only after the teacher itself is accepted; service deployment and ZKML rebuild may then proceed as separate controlled consequences of the same accepted model identity.

## 12. Stop/fail conditions

Block promotion if:

- live preprocessing cannot reproduce an evidence-equivalent representation contract;
- requested-contract selector semantics are unavailable/ambiguous at inference;
- graph schema/token shape differs unexpectedly from training;
- checkpoint architecture/class metadata requires silent fallback;
- thresholds/calibration are missing but response policy implies validated cutoffs;
- agent/MCP consumers misinterpret new output semantics;
- cache/drift identities can mix old and new model state;
- rollback cannot be executed cleanly.

## 13. Completion criteria

Inference is `ACCEPTED` when the operational checkpoint is explicit, its live preprocessing is evidence-compatible with its accepted training/evaluation lineage, score/policy semantics are versioned, integration tests pass, and rollback is tested. Until then Run12 remains the operational historical baseline even if a repaired checkpoint exists.
