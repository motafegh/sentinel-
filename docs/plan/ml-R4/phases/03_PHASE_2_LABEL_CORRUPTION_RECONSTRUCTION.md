# Phase 2 — Label-Corruption Mechanism Reconstruction

**Status:** WAITING FOR G1  
**Gate:** G2

## Objective

Identify exactly how source-native assertions became misleading historical targets.

## Required traces

For every active source and class:

```text
source-native record
→ acquisition/pin
→ parser
→ crosswalk
→ absence/default behavior
→ source tier
→ merger
→ verification
→ split
→ export
→ ML target
```

## Mandatory semantic categories

Classify historical target origins:

- explicit source positive;
- explicit source negative;
- source absence;
- class unsupported by source;
- dropped source-native category;
- mapped-to-NonVulnerable category;
- parser default;
- merger conflict resolution;
- verification override;
- export all-zero;
- missing representation;
- other.

## Representative traces

Include:

- direct positive;
- true explicit negative where available;
- dropped-class-only contract;
- mapped-to-NonVulnerable contract;
- source/class not covered;
- multi-source conflict;
- all-zero;
- duplicate/project family;
- split row removed by representation filtering.

## Quantification

Produce counts by source, class, historical target, and corruption mechanism.

## Outputs

- source authority matrix;
- source semantics cards;
- crosswalk effect table;
- merger sensitivity table;
- all-zero decomposition;
- population reconciliation;
- end-to-end trace JSONL.

## G2 pass criteria

No historical positive or zero remains semantically unexplained at the category level. Individual unreviewed outcomes may remain unknown.
