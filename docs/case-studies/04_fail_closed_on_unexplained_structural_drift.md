# Case Study 04 — Fail closed on unexplained structural drift

## 1. Decision in one sentence

SENTINEL refused to accept a full V10 candidate while even one structural change remained unexplained, and regenerated evidence when the actual population changed rather than forcing the old proof to fit.

## 2. Problem

After the initial V10 remediation, bounded evidence had explained a small set of expected graph changes. That was not enough to establish that the complete 22,540-contract candidate contained only intended structural differences.

The risk was subtle: a candidate could pass unit tests and local examples while still containing population-wide graph drift that had never been classified.

## 3. Evidence

The historical V2.5 full-candidate gate correctly failed. It found 311 non-parse-only drift identities, while the earlier bounded evidence accounted for only a much smaller approved set.

Further analysis reduced the V2.5 population to:

- 298 duplicate-safe persistent-storage WRITE corrections;
- 12 exact node-index-invariant graph equivalences;
- 1 unresolved identity involving a storage collection `push` mutation.

That single unresolved case was sufficient to keep physical acceptance false.

A narrow V2.6 extractor change then added persistent-storage collection `push`/`pop` recognition. Crucially, the project did not assume that the old 311-case evidence still described the new candidate. The actual V2.6 drift population changed by +52/-8 to 355 identities.

## 4. Shortcut rejected

The project rejected several tempting shortcuts:

- accepting a candidate because almost every drift had an explanation;
- treating a storage alias heuristic as equivalent to the required evidence contract;
- reusing the V2.5 census after the extractor changed;
- ignoring identities that appeared only in the new candidate;
- manually editing generated evidence until the audit passed.

## 5. Decision

Every observed structural change in the exact candidate had to fall into an independently evidenced class or remain a blocker.

When V2.6 changed the population, the evidence process restarted against that exact binding rather than inheriting authority from V2.5.

## 6. Implementation and validation

Three fresh V2.6 generations and three semantic-evidence passes covered the exact 355-identity population.

The final evidence established:

- 349 contracts with proven persistent-storage WRITE corrections;
- 3,517 proven semantic target groups and 6,247 graph occurrences;
- 6 exact node-index-invariant equivalence contracts;
- zero unresolved WRITE groups or population mismatches.

The complete V4 transition audit then passed all 22,540 identities and independently re-proved all 355 structural differences with zero unexplained drift.

Only after refreshed binding and current-commit review did R4-D-011 grant physical acceptance to the exact V2.6 root and digest.

## 7. Result

The acceptance claim became stronger than “tests passed” or “the graphs looked reasonable.”

It became: for this exact bound population, every observed structural transition is accounted for by evidence accepted under the current contract.

That is a materially different assurance boundary from ordinary regression testing.

## 8. Remaining limitation

Structural reconciliation proves the bounded representation transition, not vulnerability discrimination or model quality.

It also does not mean future extractor changes can inherit the same proof. Any change that alters the physical lineage or drift population requires its own binding and evidence.

## 9. Evidence trail

Primary references:

- `docs/plan/ml-R4/runs/2026-08-30_PHASE8_v10_v25_full_population_structural_analysis.md`;
- `docs/plan/ml-R4/adrs/ADR-R4-011-v10-v26-physical-representation-acceptance.md`;
- V10 transition/evidence scripts and their committed artifact index entries;
- the machine-readable R4-D-011 physical acceptance record.

This case study summarizes the decision. Exact hashes and acceptance authority remain in the primary evidence chain.
