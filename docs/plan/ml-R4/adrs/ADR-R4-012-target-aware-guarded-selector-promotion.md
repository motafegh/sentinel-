# ADR-R4-012 — Target-aware guarded selector promotion for a new lineage

Date: 2026-09-02
Status: ACCEPTED
Decision ID: R4-D-012
Scope: four-window token selection for the next versioned physical candidate

## Context

The historical linspace selector is physically reproducible but omits material
requested-contract code on long sources. The frozen architecture still accepts
exactly four `[512]` windows, so increasing window count is outside this
decision.

The hardened logical-V3 CPU study analyzed all 1,018 active training and
model-selection records. Of 737 records with more than four windows,
`target_aware_guarded_v1` improved requested-contract token coverage for 476,
fell back to the historical control for 261, regressed for zero, and failed for
zero. Median target coverage rose from 0.630063 to 0.879447. Median overall-code
retention fell from 0.601010 to 0.577922, so this is a targeted relevance policy,
not a claim that more total code is retained.

The bound CUDA study used identical initialization, no Run12 weights, and four
worst-case forward probes. It established shape/runtime safety only, not model
discrimination. The later R4-D-011 control-equivalence verifier dynamically
reproduced all 22,540 accepted token tensors and window indices with zero
failures, eliminating hidden control/tokenization drift.

## Decision

Promote `target_aware_guarded_v1` as the required selector for construction and
evaluation of a **new versioned token/representation candidate**. The historical
control remains the rollback and comparison policy.

The promotion does not alter the R4-D-011 root or digest. It authorizes only a
fresh candidate with a new lineage identity, token payloads, binding digest,
and physical acceptance review. Graph schema remains `v10`, graph extractor
remains `v2.6-r4-call-semantics-deterministic-cfg-mutators`, tensor shape remains
`[4,512]`, and model architecture remains frozen.

## Consequences

- Never rewrite or relabel R4-D-011 token files.
- The new candidate must retain complete population identity, graph bytes,
  sidecar graph semantics, and runtime provenance while changing only the
  explicitly versioned token-selector fields and token payloads.
- Binding and transition checks must distinguish expected token changes from
  graph drift; no blanket byte-equality waiver is allowed.
- Physical acceptance, model-quality claims, objective/evaluation semantics,
  threshold/calibration support, G8, and training remain unauthorized.
- R4-B006 remains open until the new physical candidate is generated, bound,
  reviewed, and accepted.

## Rollback

Select the immutable R4-D-011 root and `historical_linspace_v1` control. Do not
reverse-edit either lineage.

## Evidence

- CPU summary SHA-256 `9308fb5fc8970c9288d1e94c69ed2de7225fec7306846468055870cf51fa1b0a`;
- CUDA report SHA-256 `e8b0e9541d50032ad7de5c13616b2a5ef79236854c9d30ffa4fdd2e65dac19d8`;
- full-population control report SHA-256 `636838f376d8991e9ac07d26105aa2f907e535bbf90e4504e11d663f0c656021`;
- selector implementation SHA-256 `9eea0f837f77a512628efaa3dde444f039be81e98bf1828aa61b9099d2c87866`;
- `runs/2026-09-02_PHASE8_selector_promotion_review.md`.
