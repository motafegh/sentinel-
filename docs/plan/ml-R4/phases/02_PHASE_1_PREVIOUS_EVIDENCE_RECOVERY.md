# Phase 1 — Previous Evidence Recovery and De-duplication

**Status:** WAITING FOR G0  
**Gate:** G1

## Objective

Turn prior analyses into structured, traceable evidence that can be reused for DATA vNext decisions.

## Order of work

1. DIVE recovery workstream.
2. BCCC recovery workstream.
3. Other sources and manual evidence.
4. Model-run/export lineage.
5. Cross-investigation duplicate detection.

## Recovery method

For each prior evidence set:

- assign artifact IDs and hashes;
- identify contract-class coverage;
- identify class definitions used;
- retain raw review records where available;
- identify reviewer/tool/method;
- distinguish pilot, control, and final samples;
- reconstruct sampling probabilities where possible;
- record conclusions and limitations;
- reconcile internal count/wording contradictions;
- mark raw evidence missing;
- import evidence items without changing historical meaning.

## De-duplication

Detect:

- same contract reviewed under multiple IDs;
- same source copied across reports;
- same tool output cited as multiple independent signals;
- pilot samples included in later totals;
- reruns represented as independent evidence;
- project-family or injected-pair dependence.

## Prohibited

- no replacement broad audit;
- no new sample merely because a prior artifact is incomplete;
- no silent conversion of conclusion-only evidence into confirmed outcome;
- no implementation of DROP/KEEP yet.

## Outputs

- completed `PREVIOUS_EVIDENCE_REGISTER.md`;
- source-specific recovery reports;
- evidence inventory in Parquet/JSONL;
- contradiction report;
- unavailable-artifact report;
- proposed gaps, not review batches.

## G1 pass criteria

Every major prior claim is one of:

- recovered with raw evidence;
- recovered partially;
- conclusion-only;
- unavailable;
- contradicted and registered as a gap.

No duplicate review begins in Phase 1.
