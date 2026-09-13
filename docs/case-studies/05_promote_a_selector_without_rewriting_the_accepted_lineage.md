# Case Study 05 — Promote a selector without rewriting the accepted lineage

## 1. Decision in one sentence

SENTINEL promoted a target-aware token selector only for a new versioned candidate, preserving the accepted R4-D-011 lineage unchanged and keeping physical acceptance and training as separate later gates.

## 2. Problem

The model architecture accepts exactly four 512-token windows. On long Solidity sources, the historical linspace selector is reproducible but can omit material code belonging to the requested contract.

That creates a representation-quality problem without automatically justifying an architecture change. Increasing the number of windows would change the model input contract, while silently replacing the selector inside an already accepted lineage would destroy reproducibility.

## 3. Evidence

The hardened logical-V3 CPU study analyzed 1,018 active training and model-selection records.

Among 737 records with more than four windows, `target_aware_guarded_v1`:

- improved requested-contract token coverage for 476 records;
- fell back to the historical control for 261;
- regressed for zero;
- failed for zero.

Median requested-target coverage increased from 0.630063 to 0.879447. Median overall-code retention decreased from 0.601010 to 0.577922, so the result was interpreted narrowly: the selector improves target relevance, not total-code retention.

A CUDA comparison with identical initialization and no Run12 weights passed four worst-case forward probes, establishing runtime/shape safety rather than model discrimination.

Finally, a full-population control-equivalence verifier dynamically reproduced all 22,540 R4-D-011 historical token tensors and selected-window indices with zero failures. This removed hidden tokenization/control drift from the comparison baseline.

## 4. Shortcut rejected

The project rejected several shortcuts:

- replacing token files inside the accepted R4-D-011 root;
- interpreting better token coverage as better vulnerability classification;
- increasing the number of windows without a separate architecture decision;
- treating a small CUDA smoke as model-quality evidence;
- allowing selector promotion to imply training authorization.

## 5. Decision

R4-D-012 promoted `target_aware_guarded_v1` as the required selector for construction and evaluation of a **new versioned token/representation candidate**.

The historical `historical_linspace_v1` selector remains the comparison and rollback policy. Graph schema V10, extractor V2.6, tensor shape `[4,512]`, and the frozen model architecture remain unchanged.

## 6. Implementation and validation boundary

The next candidate must preserve complete population identity, graph bytes, graph semantics, and runtime provenance while changing only explicitly versioned selector metadata and token payloads.

It must receive a new lineage identity, binding digest, transition review, and physical acceptance decision. Expected token changes must be distinguished from unexpected graph drift; there is no blanket byte-equality waiver.

## 7. Result

The project separated three questions that are easy to collapse:

1. **Is the selector policy better supported?** — yes, under R4-D-012.
2. **Does a physical candidate implementing that policy exist and pass acceptance?** — not yet at the recorded boundary.
3. **Does that authorize repaired model training or quality claims?** — no.

That separation allows the project to improve a representation policy without silently rewriting already accepted evidence.

## 8. Remaining limitation

R4-B006 remains open until the new guarded-token candidate is generated, bound, reviewed, and physically accepted.

The promotion does not provide negative evaluation evidence, threshold/calibration support, G8 passage, repaired-model quality evidence, or training authorization.

## 9. Evidence trail

Primary references:

- `docs/plan/ml-R4/adrs/ADR-R4-012-target-aware-guarded-selector-promotion.md`;
- `docs/plan/ml-R4/runs/2026-09-02_PHASE8_selector_promotion_review.md`;
- the full-population selector control-equivalence report;
- the hardened logical-V3 CPU selector and CUDA comparison evidence.

Exact evidence hashes and authority remain in R4-D-012 and its machine-readable decision record.
