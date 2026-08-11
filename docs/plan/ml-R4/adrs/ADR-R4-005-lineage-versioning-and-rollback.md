# ADR-R4-005 — DATA vNext Lineage, Versioning, Historical Compatibility, and Rollback

**Status:** Accepted  
**Date:** 2026-08-12  
**Deciders:** Ali Rajabi (technical/governance approval delegated), GPT-5.6 Sol  
**Scope:** R4 DATA vNext artifact identity and compatibility

## Context

R4 exists because historical labels and downstream model artifacts accumulated semantic mutations that were not always represented as explicit versioned policy. Repair must not overwrite those historical artifacts or make it impossible to reproduce what Run12 consumed.

The Phase-3 ledger and Phase-4 review are now hash-bound evidence. Phase 5 needs equivalent lineage rules for future DATA vNext artifacts.

## Decision

### Historical artifacts are immutable

The protected historical export remains an immutable lineage root, including historical `labels.parquet` SHA-256:

`26e739b5d82ba512e5a1830817d09609216e2184b79cf4ca7ec2d62ef34e32b5`

The Phase-3 evidence ledger remains an immutable repair-evidence root:

`3983cc2b3317515d546c784449b583ac9a7c23ac8da267ee10f5640857cd0ac7`

No Phase-5/6/7 operation rewrites those artifacts in place.

### Version identifiers are independent

DATA vNext manifests bind at least:

- export format version;
- class vocabulary/order identity;
- graph feature schema version;
- label-state schema version;
- source-policy version/hash;
- crosswalk/aggregation policy version/hash;
- evidence-ledger identity;
- Phase-4 adjudication identities where applicable;
- Phase-6 partition/role manifest identity;
- generation code commit;
- generated artifact hashes.

A graph schema version and a label-policy version are separate identities. R4 may change label semantics without pretending graph schema v9 changed.

### Explicit compatibility modes

Consumers must explicitly declare whether they read historical v1 or DATA vNext v2.

Forbidden behavior:

- detecting missing v2 fields and silently filling masks/targets from v1 labels;
- substituting a branch name or mutable path for an artifact hash;
- mixing v1 labels with v2 policy without a declared migration artifact;
- treating the Phase-3 historical target as the vNext target.

### Rollback

Rollback means selecting a previously hash-bound artifact set and compatible reader/model configuration. It never means reverse-mutating vNext or historical files in place.

A promoted vNext model/export must retain enough lineage to return to the historical bundle or an earlier vNext bundle without reconstructing identities from mutable paths.

### Publication order

When Phase 7 materializes vNext artifacts:

1. write candidates to staging;
2. validate schema/semantics/counts;
3. compute artifact hashes;
4. validate manifest bindings;
5. promote canonical files;
6. publish the final manifest last.

This follows the fail-closed publication discipline established in Phase 3.

## Consequences

- historical Run12 evidence remains reproducible and auditable;
- graph-schema stability is not confused with label-policy stability;
- v1/v2 incompatibilities fail visibly;
- rollback remains artifact selection rather than state mutation.

## Rejected alternatives

- **Reuse export schema v1 with extra optional columns:** rejected because old consumers could silently ignore the new semantics.
- **Overwrite labels under the existing export directory:** rejected because the path would no longer identify the historical bundle.
- **Version only by Git commit:** rejected because protected/generated artifact identity requires content hashes.

## Implementation contract

Phase 7 must produce new versioned artifact paths/manifests and use staged validation/promotion. Phase 8 checkpoint lineage must include the complete vNext export and training-config identities.
