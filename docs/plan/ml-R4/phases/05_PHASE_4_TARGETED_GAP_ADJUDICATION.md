# Phase 4 — Targeted Evidence-Gap Adjudication

**Status:** PASSED — G4 PASS  
**Gate:** G4

## Objective

Fill only the evidence gaps that prevent a DATA vNext source/class/role decision.

## Authorization

Each work package must reference an approved `R4-GAP-*` entry.

No gap ID means no review.

Routine technical/governance approval authority was delegated by the human owner on 2026-08-11. `R4-GAP-002` was approved as the smallest decision-critical review. No unrelated gap was silently expanded.

## Review design

For each gap:

1. state the exact decision blocked;
2. summarize prior evidence searched;
3. define the smallest relevant population;
4. freeze class definition;
5. construct leakage-group-aware sample;
6. hide historical/model/tool conclusions during initial semantic review where possible;
7. reveal evidence in reconciliation pass;
8. use second review only where the intended role requires it;
9. adjudicate or retain conflict;
10. stop when the decision can be made with bounded uncertainty.

## No fixed global sample

Sample size is adaptive to the decision, prevalence, clustering, disagreement, and intended role. Do not create a universal 1,000–2,000 contract target.

## Outputs per gap

- authorization record;
- frozen sample manifest;
- review batch;
- evidence items;
- adjudication;
- role recommendation;
- uncertainty report;
- gap closure or mask/exclude decision.

## G3 entry result

G3 passed with 22,493 contracts × 10 classes = 224,930 unique contract-class rows. Historical positives entered Phase 4 as `NOT_REVIEWED`; historical zeros remained `UNKNOWN`.

## R4-GAP-002 execution

The approved gap covered five DIVE native classes that actively mapped to canonical targets but lacked source-specific semantic precision evidence:

- `DoS` → `DenialOfService`
- `Arithmetic` → `IntegerUO`
- `Time manipulation` → `Timestamp`
- `Front Running` → `TransactionOrderDependence`
- `Unchecked Return Values` → `UnusedReturn`

A deterministic initial sample was frozen from the committed Phase-3 ledger:

- sample version `r4-gap-002-sample-v1`;
- 20 contracts per stratum / 100 total;
- TRAIN-only;
- any review group touching validation or test was excluded;
- no review-group reuse across strata;
- sample SHA-256 `2899ad5a210ac6e2e2a4e6b43f31cd718afa3b1d603b659cdd6bf0918f34fbe9`.

The local source bundle was checksum-bound and published with SHA-256 `2b1ce12fdd96819c89bbb9fe1dfb2d9aa992ec0a05ce32f651c6b834b97ddf38`. CI verified safe extraction, exact task identity, and blind-state integrity.

The primary semantic review was source-only and blind to model outputs, tool votes, merger outcome, and non-target historical labels. It was explicitly recorded as an AI primary semantic review, not as a human/inter-rater or acceptance-grade review.

### Blind review results

| Stratum | Supports positive | Does not support | Unclear | Boundary conflict | First-baseline source-assertion role |
|---|---:|---:|---:|---:|---|
| DenialOfService | 0/20 | 20/20 | 0 | 0 | mask/exclude; retain structure as unlabeled |
| IntegerUO | 3/20 | 16/20 | 1/20 | 0 | mask/exclude; retain structure as unlabeled |
| Timestamp | 4/20 | 15/20 | 1/20 | 0 | mask/exclude; retain structure as unlabeled |
| TransactionOrderDependence | 12/20 | 5/20 | 0 | 3/20 | `TRAIN_WEAK` only; all outcome/selection/calibration/acceptance roles masked |
| UnusedReturn | 9/20 | 11/20 | 0 | 0 | mask/exclude; retain structure as unlabeled |

`DOES_NOT_SUPPORT_POSITIVE` never means `CONFIRMED_NEGATIVE`.

The detailed review is in:

- `findings/06_gap002_blind_semantic_review.md`
- `findings/06_gap002_blind_semantic_review_report.json`
- `reviews/R4-GAP-002/p4_gap002_blind_semantic_review_v1.jsonl`

## Adaptive stop decision

No second-review or larger sample is required for the first-baseline Phase-4 decision because no stratum is promoted to strong training, outcome metrics, model selection, threshold fitting, calibration fitting, or acceptance.

Four strata are conservatively demoted/masked. TOD is limited to weak training evidence only. Any future attempt to promote a DIVE stratum beyond these limits requires a new approved evidence gap and stronger/independent evidence.

## Other gap dispositions

- `R4-GAP-001` Web3Bugs: absent source explicitly excluded from the first baseline; acquisition is future source expansion, not a G4 requirement.
- `R4-GAP-003` provisional BCCC CTU/ExternalBug/GasException: remain deferred/masked unless Phase 5 explicitly proposes importing them.
- `R4-GAP-004` BCCC 2-tool consensus: non-blocking future benchmark evidence; tool intersection is not ground truth.
- `R4-GAP-005` fuzzing and `R4-GAP-006` exploit PoCs: non-blocking later evaluation/case-study evidence.

## G4 pass assessment

**G4 PASS.** Critical first-baseline DATA vNext decisions now have bounded evidence:

- active DIVE source assertions have explicit per-stratum role limits rather than inherited binary authority;
- unsupported DIVE populations are masked/excluded rather than forced into negatives;
- the only retained weak DIVE stratum is explicitly barred from outcome metrics and high-authority dataset roles;
- absent/provisional non-active source populations have explicit first-baseline exclusion/defer decisions;
- remaining proposed tool/fuzzing/PoC gaps are future non-blocking evidence opportunities, not hidden prerequisites.

## Next permitted action

Begin Phase 5 — DATA vNext Policy and Design. Convert Phase-0–4 evidence into the smallest versioned source/class/state/role contract, with ADRs for every semantic policy change. Do not make new label semantics silently in implementation code.
