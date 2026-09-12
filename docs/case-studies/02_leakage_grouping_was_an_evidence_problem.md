# Case Study 02 — Leakage grouping was an evidence problem

## 1. Decision in one sentence

SENTINEL rejected convenient address-based grouping as dataset authority and replaced it with versioned evidence-based identity rules before relying on ML evaluation or retraining.

## 2. Problem

The project needed dataset groups that prevent related contracts or duplicated evidence from leaking across training, evaluation, and other dataset roles.

An early grouping approach used address relationships as a strong signal. This looked useful because Ethereum addresses are easy to extract and compare, but it created a deeper problem: an observable relationship was treated as if it proved code or artifact identity.

That distinction matters. A shared address literal, library reference, or external interaction can indicate a relationship, but it does not automatically prove that two samples represent the same vulnerability-learning unit.

## 3. Evidence

The investigation showed that the historical grouping authority could create an oversized connected component containing 10,327 contracts through address-literal relationships.

The result was not accepted as a simple implementation bug. It was treated as an evidence-modeling failure: the project had not sufficiently separated observed relationships from the stronger claim of dataset identity.

The repaired logical V3 grouping model introduced explicit evidence semantics and acceptance checks. The accepted logical V3 population contains 22,394 groups, maximum group size 7, and zero address-authority edges.

## 4. Shortcut rejected

The project rejected the shortcut of keeping the old grouping because it was convenient, familiar, or produced cleaner-looking dataset partitions.

A smaller or easier pipeline is not valuable if its grouping authority silently changes what the evaluation result means.

## 5. Decision

Grouping became a versioned evidence contract rather than an incidental preprocessing detail.

Only relationships supported by the accepted identity model are allowed to influence dataset grouping decisions. Historical grouping outputs remain preserved as evidence of what happened, not as current authority.

## 6. Implementation and validation

The repair was governed through R4 decisions and validation artifacts rather than an informal code change.

The accepted logical V3 lineage established the corrected grouping boundary, preserved reproducibility, and separated historical observations from current dataset authority.

## 7. Result

SENTINEL gained a defensible boundary between:

- a relationship that was observed;
- evidence that supports identity equivalence;
- a dataset role decision.

This prevents future ML claims from depending on hidden assumptions inside preprocessing.

## 8. Remaining limitation

Correct grouping does not create missing labels, prove model quality, or authorize training. It only establishes a safer foundation for later evidence and evaluation decisions.

The project still requires separate decisions around negative evidence, evaluation design, selector promotion, and repaired training authorization.

## 9. Evidence trail

Primary authority remains:

- executable source, tests, and validation artifacts;
- R4 machine-readable evidence and ADR records;
- canonical handbook documentation.

This case study explains the engineering decision but does not replace those sources.
