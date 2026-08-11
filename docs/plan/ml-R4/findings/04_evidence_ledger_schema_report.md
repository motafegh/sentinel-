# 04 — Evidence Ledger Schema / Validator Report

- **Phase:** R4 Phase 3 — Contract-Class Evidence Ledger
- **Checkpoint:** remote framework complete through P3-W1/P3-W2/P3-W3
- **Gate:** G3 — **NOT YET PASSED**
- **Protected historical export changed:** NO

## What is implemented

### Versioned schemas

1. `schemas/evidence_ledger_row.v1.schema.json`
2. `schemas/evidence_item.v1.schema.json`
3. `schemas/evidence_ledger_manifest.v1.schema.json`

The row schema covers every field group required by the Phase-3 specification:

| Required field group | v1 representation |
|---|---|
| contract/source identity | contract_id, primary_source, source_record_id, source_tier |
| class identity | locked class_index + class_name |
| dedup/project/leakage identity | dedup_group_id, project_group_id, leakage_group_id |
| historical target/export identity | historical_state, historical_target, historical_export_artifact_id, historical_export_sha256, historical_split |
| representation eligibility | representation_available |
| source-native claims | source_native_state, source_native_label |
| parser/crosswalk decisions | parser_id, crosswalk_id, crosswalk_action, zero_origin_categories |
| merger/verification decisions | merger_action, verification_action |
| Phase-2 reconstruction linkage | phase2_trace_ids |
| evidence + independence | evidence_ids, independence_groups |
| prior review | prior_review_state |
| R4 outcome | outcome_state |
| uncertainty | limitations |
| supervised/metric masking | supervised_loss_masked, outcome_metrics_masked |
| role eligibility | role_eligibility |
| partition | partition |
| artifact identity | artifact_ids + protected export identity |
| historical/new provenance | provenance_kind |

### Conservative unknown handling

The schema explicitly supports:

`UNRESOLVED_WITHIN_KNOWN_MECHANISMS`

for a historical zero whose exact per-row origin cannot be recovered although Phase 2 bounded the possible mechanism set. This prevents artificial precision.

### Evidence-item schema

Evidence can be scoped as:

- CONTRACT_CLASS;
- CONTRACT;
- SOURCE_CLASS;
- CORPUS_CLASS;
- TRANSFORMATION_CATEGORY.

It records evidence strength, producer, independence group, polarity, artifact identity, raw-evidence availability, tool-only status, historical/new provenance and limitations.

### Manifest schema

The production manifest binds:

- ledger version;
- protected source-export artifact/hash;
- expected vs actual contract/row counts;
- exact canonical class order;
- Parquet/evidence/report identities;
- source/outcome/historical-state/role summary counts;
- generation commit;
- materialization/validation status;
- limitations.

A draft production manifest is committed with:

```text
expected_contracts = 22,493
expected_classes   = 10
expected_rows      = 224,930
actual_contracts   = 0
actual_rows        = 0
status             = DRAFT
```

It is intentionally impossible to mistake this checkpoint for a materialized ledger.

## Semantic validator

`scripts/p3_validate_evidence_ledger.py` validates JSONL directly and Parquet only when `pyarrow` is already available. Missing Parquet support fails explicitly; no substitute population is used.

Implemented rejection rules include:

1. duplicate canonical `(contract_id, class_index)` keys;
2. class index/name mismatch against the locked ten-class order;
3. incomplete 0–9 class coverage for any contract;
4. historical-state / numeric-target inconsistency;
5. missing or malformed protected export identity;
6. confirmed outcome without an evidence reference;
7. UNKNOWN / NOT_APPLICABLE / CONFLICTING_EVIDENCE / NOT_REVIEWED / INVALID_RECORD left unmasked for supervised loss or metrics;
8. invalid role/partition values;
9. EXCLUDE_OUTCOME_METRICS without metric mask;
10. historical zero without zero-origin provenance;
11. historical positive carrying zero-origin mechanisms;
12. unresolved evidence IDs;
13. evidence independence group omitted from the row;
14. contract/class/source scope mismatch between evidence and row;
15. UNTOUCHED_ACCEPTANCE supported only by tool evidence;
16. leakage group spanning incompatible partitions;
17. manifest expected-row arithmetic mismatch;
18. manifest/ledger actual-count mismatch;
19. incomplete production population unless explicitly running fixture mode with `--allow-partial-population`;
20. manifest class-order or export-hash mismatch.

## Deterministic fixtures/tests

Committed fixtures:

- `fixtures/p3_valid_ledger_fixture.jsonl` — one complete ten-class contract;
- `fixtures/p3_valid_evidence_fixture.jsonl` — direct injection evidence for the one confirmed positive;
- `fixtures/p3_valid_manifest_fixture.json`;
- `fixtures/p3_invalid_ledger_cases.jsonl` — ten targeted invalid cases.

Committed self-test:

- `scripts/test_p3_validate_evidence_ledger.py`

The valid fixture models a SolidiFI Reentrancy injection:

- Reentrancy: historical positive + evidence-backed CONFIRMED_POSITIVE, unmasked;
- other nine cells: historical zeros that remain UNKNOWN and masked;
- wholly unsupported SolidiFI classes use `CLASS_UNSUPPORTED` provenance.

The ten negative cases target duplicate keys, class order, evidence requirements, masking, unresolved evidence, evidence-scope mismatch, tool-only acceptance, leakage partition crossing, artifact hash identity and zero-origin requirements.

### Remote execution status

The repository's only tracked GitHub Actions workflow is `handbook.yml`; there is no general test workflow that executes this Phase-3 validator. The connector used for this checkpoint can author and inspect repository files but does not execute arbitrary repository Python.

Therefore the self-test harness is **implemented but not claimed as executed in GitHub CI** in this checkpoint. It must be run in a Python checkout before G3 closure, together with the production ledger validation.

## Phase-2 state import

`findings/04_phase3_state_mapping.md` defines conservative initialization from Phase-2 mechanisms.

Important defaults:

- unsupported source class → UNKNOWN, not NOT_APPLICABLE;
- source-native unknown → UNKNOWN + masks;
- dropped/mapped-to-NonVulnerable/default zero → UNKNOWN + masks;
- historical positive with source assertion only → NOT_REVIEWED unless stronger evidence resolves;
- direct June DoS positive→zero patch → CONFLICTING_EVIDENCE by default, not confirmed negative;
- missing representation affects ML eligibility, not vulnerability truth;
- conflicting duplicate-family evidence → CONFLICTING_EVIDENCE where class conflict is established.

## Seed evidence set

`manifests/evidence_items_v1.jsonl` contains only evidence the remote repository can support honestly:

- DIVE unknown→zero transformation;
- DIVE unsupported-class transformations;
- DIVE Bad Randomness drop;
- SmartBugs mapped-to-NonVulnerable behavior;
- SolidiFI non-target zero behavior;
- recovered historical DoS patch;
- missing-representation population filtering;
- ML zero→negative-loss behavior.

These are transformation/source/corpus-scoped evidence items. They are **not** presented as contract-specific confirmations.

## Production materialization blocker

The Phase-3 specification requires one row for every export-relevant contract-class pair. The protected identity is known:

- export labels artifact: `R4-P0-EXP-002`;
- SHA-256: `26e739b5d82ba512e5a1830817d09609216e2184b79cf4ca7ec2d62ef34e32b5`;
- contracts: 22,493;
- required ledger rows: 224,930.

However the actual protected row-identifying artifacts are not stored in ordinary GitHub contents. Direct GitHub reads return 404 for:

- `data_module/data/splits/v3/train.jsonl`;
- `data_module/data/splits/v3/split_manifest.json`;
- `data_module/data/exports/sentinel-v3-smartbugs-2026-06-13/labels.parquet`.

GitHub contains their historical hashes/counts in R4 documentation, but not the contract IDs/class cells needed for a truthful production ledger.

## G3 assessment

**G3 does not pass yet.**

The schema can represent the required states without forcing unknowns into binary negatives, but the Phase-3 gate also requires the production ledger artifact and its validation. That cannot be fabricated from aggregate counts.

### Exact remaining work to pass G3

1. make the protected v3 split/labels population available to an execution environment;
2. materialize all 224,930 canonical rows using the committed v1 schemas/state mapping;
3. attach only evidence whose scope resolves correctly;
4. write the production Parquet ledger;
5. populate the ledger manifest's actual counts/hashes;
6. run validator + self-tests;
7. produce the validation report with zero gate-breaking errors;
8. then, and only then, mark Phase 3 PASSED / G3 PASS.
