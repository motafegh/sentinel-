# Phase 10 — Acceptance, Promotion, and Rollback

**Status:** WAITING FOR G9  
**Gate:** G10

## Objective

Make the final release decision using untouched acceptance evidence and tested migration/rollback.

## Acceptance bundle

- code commit;
- DATA vNext manifest;
- evidence ledger version;
- partition manifest;
- checkpoint;
- threshold sidecar;
- calibration sidecar;
- inference policy;
- claim-status matrix.

## Tests

- artifact integrity;
- class order/schema;
- no leakage;
- deterministic inference;
- evidence-qualified per-class metrics;
- calibration and threshold utility;
- abstention;
- API compatibility;
- downstream agent/ZK compatibility;
- latency/failure behavior;
- rollback rehearsal.

## Outcomes

- promote full bundle;
- promote selected classes;
- promote with restricted claims;
- retain current bundle temporarily with restrictions;
- reject repaired bundle and return to the relevant gate.

## G10 pass criteria

Promotion evidence, limitations, migration, and rollback are complete.
