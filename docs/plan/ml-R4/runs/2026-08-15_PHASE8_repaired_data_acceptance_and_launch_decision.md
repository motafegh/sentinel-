# Phase 8 Repaired-DATA Acceptance and Training Launch Decision

**Date:** 2026-08-15
**Evidence source commit:** `fb31326da4420c2289822c2a6db8a022ac25876a`
**DATA decision:** repaired-v2 physical lineage ACCEPTED for bounded local research
**Training decision:** 100-epoch Phase-8 run NOT AUTHORIZED
**Gate:** G8 remains OPEN

## Outcome first

The repaired local corpus is now physically complete and internally consistent.
All 22,540 published contract identities have graph, token, and sidecar payloads;
all 67,620 files deserialize and bind to the same publication; the repaired
lineage acceptance profiler passes all 20 checks; and a two-step CUDA smoke is
finite. This closes the physical rebuild prerequisite.

It does not make the proposed 100-epoch run worthwhile yet. The publication has
1,080 positive targets, 224,320 unknown targets, and **zero confirmed negative
targets**. The effective training objective contains 899 cells and every one is
target `1`; unknown cells are masked. A long run can therefore learn to raise
positive logits, but it cannot learn or evaluate false-positive discrimination.
Positive-only model selection cannot repair that limitation. The defensible
decision is to preserve the accepted DATA evidence, keep G8 open, and resolve
the objective/evaluation design before spending the full training budget.

## Accepted physical evidence

### Source reconciliation and publication

| Measure | Repaired-v2 result |
|---|---:|
| Raw manifest records | 22,823 |
| Published contract identities | 22,540 |
| Contract x class rows | 225,400 |
| Positive / unknown / confirmed-negative targets | 1,080 / 224,320 / 0 |
| Strong / weak semantic cells | 474 / 606 |
| Leakage groups | 11,551 |
| Effective loss cells | 899 |
| Outcome metric cells | 176 |

Source outcomes:

- DIVE: 22,330 raw records, 22,308 prepared, 22 explicit
  `normalized_compile_failed` drops, 22,054 exact-identity artifacts;
- SmartBugs Curated: 143 raw/prepared/artifact identities, zero drops;
- SolidiFI: 350 raw/prepared records, 343 artifact identities after seven exact
  duplicates were aggregated, zero drops.

The repaired lineage adds 47 contract identities, 883 represented contracts,
71 strong semantic cells, and two weak semantic cells relative to the audited
historical baseline. No address-based deletion or target-zero synthesis occurs.

Evidence ledger SHA-256:
`5317aba94b9cdbe900bd90bd9b2fdf22d69c3810ec2b0a08d9be032f21658d6d`.

### Representation completeness and compatibility tail

The first complete DIVE representation attempt wrote 22,026 of 22,054 triples
and explicitly failed 28 (0.127%). It was rejected by the zero-failure gate.
The versioned recovery then reconciled the immutable attempt and recovered all
28 without changing promoted Solidity/token bytes:

- 22,512 contracts use normal Slither full analysis;
- 26 use provenance-visible Slither parse-only extraction;
- two use full analysis after a line/byte-preserving graph-only fold of a
  compile-time fixed-array expression.

Final physical binding:

- required/checked contracts: 22,540 / 22,540;
- checked graph/token/sidecar files: 67,620;
- missing or invalid: 0;
- binding digest:
  `16dd4a3f98c34e52e5c411b39268361881efede07e8f3f52d0c060dd1c5bb6dd`;
- DIVE representation manifest:
  `7bd7e798d9595e4f4c536706ade276e7017e5e3fb5f2fe23cbecb5a2a4d97a1f`;
- SmartBugs representation manifest:
  `28b3ce071903e731f515321489b3a7e1ef4b1c2b98f2e65adcdf16a6d4ccb0b5`;
- SolidiFI representation manifest:
  `e0c5a19f533b79e63f7eceb2065e12f7cf24861288a855b506bf3d037d0d0327`.

Parse-only graphs are accepted as explicit compatibility-degraded inputs, not
silently equated with normal IR analysis. Seven of the 28 recovered contracts
carry optimizer loss, all in `TRAIN_WEAK`; none enters `MODEL_SELECTION`.

### Graph and token population

The physical binder measured, rather than assumed, the generated population:

| Property | min | p50 | p95 | p99 | max |
|---|---:|---:|---:|---:|---:|
| Components | 1 | 1 | 2 | 5 | 28 |
| Nodes | 1 | 264 | 883 | 1,606 | 16,065 |
| Edges | 0 | 685 | 1,969 | 4,269 | 166,459 |
| Pre-subsampling token windows | 1 | 17 | 54 | 97 | 403 |
| Retained token ratio | 0.0195 | 0.4226 | 1.0 | 1.0 | 1.0 |

There are 4,211 multi-component contracts and 19,451 contracts (86.3%) exceed
four token windows. This is a material input-coverage limitation even though
the tensor shape is valid.

## Bounded-window evidence

The corrected comparison analyzed all 11,341 optimizer/model-selection role
records with zero failures; 10,996 exceed four windows. It accounts for every
target contract in multi-component files.

| Metric | Historical linspace control | Target-aware candidate |
|---|---:|---:|
| Median global retained ratio | 0.2759 | 0.2868 |
| Median target-contract coverage | 0.2760 | 0.5119 |
| Minimum target-contract coverage | 0.0000 | 0.0321 |

Target coverage improved for 10,208 records and regressed for 342. Among the
655 optimizer-active over-cap records, median target coverage improves from
0.6585 to 0.8738 (409 improve, 70 regress). Among 87 over-cap active
model-selection records, it improves from 0.4855 to 0.8295 (70 improve, seven
regress).

This is sufficient to reject an adequacy claim for the current selector. It is
not sufficient to promote the candidate: the selector needs its own versioned
representation lineage and an identical-initialization bounded GPU comparison,
including review of regressions and large-graph behavior.

## Bounded GPU smoke

The repaired-v2 CUDA micro-smoke passed on an NVIDIA GeForce RTX 3070 Laptop
GPU using BF16 autocast:

- two train batches and two optimizer steps;
- one model-selection batch;
- finite total/main/auxiliary/phase-two losses;
- peak allocated GPU memory: 964.46 MB;
- Run12 weights loaded: false;
- checkpoint written: false;
- `full_training_authorized: false` preserved.

The active optimizer population is 899 contracts / 831 groups: 298
`TRAIN_STRONG` and 601 `TRAIN_WEAK`. The active model-selection population is
103 contracts / 61 groups. One selection cell in a bounded smoke is a mechanics
check, not a quality estimate.

## Why full training remains a no-launch

The training code fails closed if an effective-loss target is not `1`, and the
accepted publication contains no target `0`. Consequently:

1. every supervised gradient rewards a higher positive logit;
2. unknown cells contribute no counter-signal;
3. model selection measures positive fit only;
4. false-positive rate, specificity, calibrated threshold quality, and
   production security accuracy are not identifiable from the current roles;
5. the current four-window control also omits material target code for many
   optimizer and model-selection examples.

Running 100 epochs now would produce a technically reproducible
**positive-fitting baseline**, not evidence that SENTINEL detects
vulnerabilities better or avoids broad overprediction. That distinction makes
the full compute spend premature.

## Next authorized work

1. Decide and implement an evidence-honest learning objective: acquire
   class-specific confirmed negatives, or adopt an explicit positive-unlabeled
   learning policy/objective with assumptions and evaluation limits recorded.
2. Version and evaluate the target-aware four-window selector with identical
   initialization and bounded GPU steps; include regression cases and a
   worst-case large-graph memory smoke.
3. Rebuild/rebind only artifacts affected by an accepted selector or policy
   change, preserving this accepted repaired-v2 lineage as evidence.
4. Define a credible quality gate before re-authorizing the fixed 100-epoch
   horizon. Do not infer model benefit from lower positive-only loss alone.

Repository-only assistance can implement and review objective/selector code,
tests, and governance. Protected Solidity bytes, regenerated physical
representations, full binding, and CUDA diagnostics remain local work.

## Local evidence artifacts

- `data_module/data/r4-v2-build/repaired_lineage_audit.json`
- `data_module/data/r4-v2-build/bounded_window_experiment.json`
- `data_module/data/r4-v2-build/repaired_gpu_smoke.json`
- `data_module/data/exports/sentinel-r4-vnext-v2/manifest.json`
- `data_module/data/exports/sentinel-r4-vnext-v2/representation_binding_report.json`

These generated files are Git-ignored/local-only. Their hashes and decisive
measurements are recorded here so a fresh clone does not falsely claim to
contain the physical artifacts.
