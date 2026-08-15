# ADR-R4-008 — Repaired-v2 Physical DATA Acceptance and Phase-8 No-Launch

**Status:** Accepted
**Date:** 2026-08-15
**Deciders:** Ali Rajabi (routine technical/governance approval delegated), GPT-5.6 Sol
**Scope:** R4 Phase-8 repaired physical DATA acceptance, bounded-research authority, and full-training hold

## Context

The historical G7 publication `sentinel-r4-vnext-v1` remains immutable evidence, but the 2026-08-14 real-data readiness audit found material physical preprocessing and representation defects that made that historical physical lineage unsuitable for the first evidence-generating Phase-8 retrain.

A new repaired lineage was therefore built and validated locally without overwriting historical DATA or Run12 artifacts. The accepted repaired lineage is versioned separately as `sentinel-r4-vnext-v2` / `r4-vnext-roles-v2` with representation extractor `v2.2-r4-repaired`, graph schema `v9`, and the frozen `[4,512]` token tensor contract.

The local rebuild and acceptance evidence was generated from source commit `fb31326da4420c2289822c2a6db8a022ac25876a`. The final governing evidence is `runs/2026-08-15_PHASE8_repaired_data_acceptance_and_launch_decision.md`.

## Decision

Accept the repaired-v2 physical DATA lineage for **bounded local research and subsequent versioned objective/representation experiments**.

This acceptance means the physical DATA lineage is sufficiently complete and internally bound for controlled research:

- 22,540 published contract identities;
- 225,400 contract×class rows;
- 1,080 positive targets, 224,320 unknown targets, and zero confirmed-negative targets;
- 474 STRONG and 606 WEAK semantic cells;
- 11,551 leakage groups;
- 899 effective loss cells;
- 22,540 / 22,540 required representation triples validated;
- 67,620 graph/token/sidecar files validated with zero missing or invalid files;
- representation binding digest `16dd4a3f98c34e52e5c411b39268361881efede07e8f3f52d0c060dd1c5bb6dd`.

The compatibility tail is accepted as explicit, degraded provenance rather than silently equated with normal full Slither analysis:

- 22,512 contracts are classified as normal full-analysis inputs by the repaired binding lineage;
- 26 use Slither parse-only compatibility extraction;
- two use line/byte-preserving graph-only fixed-array constant folding before full analysis;
- seven of the 28 compatibility-recovered contracts carry optimizer signal, all weak; none is MODEL_SELECTION evidence.

Historical `sentinel-r4-vnext-v1`, `r4-vnext-roles-v1`, G7 artifacts, and Run12 remain immutable historical/rollback roots.

## Full-training decision

The fixed 100-epoch Phase-8 run is **not authorized** by this acceptance.

Every one of the 899 effective loss cells has target `1`. Unknown cells are masked and no class-specific confirmed-negative population exists. Therefore the current supervised objective can reward positive fitting but cannot identify or evaluate false-positive discrimination, specificity, calibrated threshold quality, or production security accuracy.

The current four-window selector is also not accepted as adequate. Physical evidence shows 19,451 / 22,540 contracts exceed four pre-subsampling windows. The target-aware bounded candidate materially improves target-contract coverage but still has regressions and has not passed an identical-initialization GPU comparison or worst-case large-graph review.

G8 therefore remains OPEN.

## Consequences and next authority

Before any full training authorization:

1. adopt an evidence-honest learning/evaluation contract: either acquire class-specific confirmed-negative evidence or approve a versioned positive-unlabeled learning policy with explicit assumptions and limitations;
2. version and evaluate the target-aware bounded selector with identical initialization, regression review, and worst-case large-graph GPU diagnostics;
3. bind any changed DATA/representation/objective lineage as a new version rather than mutating repaired-v2 evidence;
4. define a model-quality gate that is stronger than positive-only loss reduction;
5. recompute and bind the dataloader/scheduler horizon from the finally authorized active group population. Historical v1 values such as 88 micro-batches, 11 optimizer steps/epoch, and a 1,100-step horizon are not repaired-v2 authority.

No threshold fitting, calibration fitting, untouched-acceptance claim, or Run12 learned-state reuse is authorized by this ADR.

## Representation caveats retained as explicit risk

Physical validity is not equivalent to representation adequacy. In particular:

- file-level labels may be represented by a disconnected union of multiple unrelated inheritance leaves; 4,211 accepted samples are multi-component and the maximum is 28 components;
- the largest measured graph has 16,065 nodes and 166,459 edges;
- parse-only/graph-transform compatibility inputs require sensitivity analysis before their contribution is treated as interchangeable with normal full-analysis representations;
- source-scoped address literals are leakage-family evidence, not duplicate identity or label truth; grouping breadth must be profiled before a future role refreeze.

These caveats do not invalidate the physical DATA acceptance; they bound what can be claimed from subsequent modeling.

## Rollback

Rollback means selecting the immutable historical G7/v1 bundle or another previously hash-bound compatible lineage. Do not reverse-edit or overwrite repaired-v2 or historical artifacts.

## Evidence

- `runs/2026-08-14_PHASE8_real_data_readiness_audit.md`;
- `runs/2026-08-15_PHASE8_local_gate_reaudit_and_corrections.md`;
- `runs/2026-08-15_PHASE8_repaired_data_acceptance_and_launch_decision.md`;
- evidence ledger SHA-256 `5317aba94b9cdbe900bd90bd9b2fdf22d69c3810ec2b0a08d9be032f21658d6d`;
- repaired representation binding digest `16dd4a3f98c34e52e5c411b39268361881efede07e8f3f52d0c060dd1c5bb6dd`;
- final repository repair CI at `89059bfb0b9bf68447d96e0d416a7b4b78964209`: 108 repaired tests passed; historical G6 validator passed;
- handbook CI at the same commit: 145 static checks and 11 unit tests passed;
- bounded repaired-data CUDA smoke: PASS, no Run12 weights, no checkpoint, `full_training_authorized=false`.
