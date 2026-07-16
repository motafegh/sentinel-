# Phase 4 — Targeted Evidence-Gap Adjudication

**Status:** WAITING FOR G3  
**Gate:** G4

## Objective

Fill only the evidence gaps that prevent a DATA vNext source/class/role decision.

## Authorization

Each work package must reference an approved `R4-GAP-*` entry.

No gap ID means no review.

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

## G4 pass criteria

Critical DATA vNext decisions are supported. Remaining unsupported populations are explicitly masked, excluded, or assigned a limited role.
