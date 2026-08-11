# R4 Phase 3 — Evidence Ledger Execution Plan

**Branch:** `r4/phase3-evidence-ledger`  
**Parent:** canonical `main` after R4 Phase 2 / G2 PASS  
**Phase:** 3 — Contract-Class Evidence Ledger  
**Gate:** G3  
**Initial status:** IN_PROGRESS

## Objective

Build a versioned sidecar ledger keyed by:

```text
ledger_version × contract_id × class_index
```

without modifying the protected historical labels/export. The ledger must preserve the Phase-2 distinction between historical numeric target and actual R4 evidence/outcome state.

## Governing invariants

1. Every export-relevant `contract_id × class_index` pair receives exactly one ledger row.
2. `historical_state` and `outcome_state` are separate fields.
3. `HISTORICAL_ZERO` never implies `CONFIRMED_NEGATIVE` without traceable evidence.
4. `UNKNOWN`, `NOT_REVIEWED`, and `CONFLICTING_EVIDENCE` are masked from supervised outcome loss and outcome metrics.
5. `CONFIRMED_POSITIVE` / `CONFIRMED_NEGATIVE` require evidence references.
6. Tool-only evidence cannot qualify a row for `UNTOUCHED_ACCEPTANCE` by default.
7. Historical and newly produced evidence are explicitly distinguished.
8. Artifact identity is required for historical target/export claims.
9. Leakage/dedup/project groups cannot cross incompatible dataset roles once partitions are assigned.
10. Phase 3 is a sidecar construction phase; it does not rewrite the protected v3 export.

## Work packages

### P3-W1 — Schema and manifest contract

Create versioned schemas for:

- evidence-ledger row;
- evidence item;
- ledger manifest.

The row schema must contain all field groups required by the Phase-3 specification:

- contract/source identity;
- class identity;
- dedup/project/leakage identity;
- historical target/export identity;
- source-native claim state;
- parser/crosswalk/merger transformation provenance;
- evidence IDs + independence groups;
- prior review outcome;
- final R4 outcome state;
- uncertainty/limitations;
- supervised mask and metric mask;
- role eligibility;
- partition;
- artifact hashes;
- historical/new provenance.

### P3-W2 — Validator and deterministic fixtures

Implement a read-only validator that rejects at minimum:

- duplicate canonical keys;
- invalid class order/name/index pairing;
- confirmed outcome without evidence reference;
- historical/new provenance ambiguity;
- acceptance eligibility from tool-only evidence;
- masked outcome included in supervised metrics;
- unknown/conflicting/not-reviewed row left unmasked for supervised outcome loss;
- incompatible role leakage within one leakage group;
- missing protected export/artifact identity;
- invalid evidence references;
- evidence item key mismatch with its ledger row.

Create small deterministic positive and negative fixtures covering every validation rule.

### P3-W3 — Phase-2 transformation import mapping

Define deterministic mapping from Phase-2 reconstructed mechanisms into ledger states, including:

- explicit source positive;
- explicit source negative;
- source-native unknown;
- unsupported class;
- dropped category;
- mapped-to-NonVulnerable;
- parser/default zero;
- direct post-export suppression;
- missing representation;
- conflicting duplicate family.

This mapping records provenance; it does **not** adjudicate unknown rows as negative.

### P3-W4 — Full population materialization

Materialize all protected export-relevant rows:

```text
22,493 contracts × 10 classes = 224,930 canonical ledger rows
```

Inputs required:

- protected v3 split contract IDs;
- protected labels/export identity;
- source identity per contract;
- dedup/leakage groups where available;
- Phase-1 evidence items and Phase-2 transformation mappings.

**Remote-only constraint:** GitHub contains hashes/counts and semantic evidence, but not the full protected split/export row population. Therefore P3-W4 cannot be truthfully completed until the protected contract-ID artifacts are made available to the execution environment. Schema/validator work may proceed remotely; G3 remains blocked until real population materialization and validation occur.

### P3-W5 — Evidence import

Import recovered evidence into JSONL with explicit independence groups and provenance. Historical conclusion-only evidence must remain distinguishable from retained raw evidence.

### P3-W6 — Validation report and G3 assessment

Validate:

- expected canonical row count = 224,930;
- unique key count = 224,930;
- exact class order 0–9 for every contract;
- all evidence references resolve;
- all masked states obey mask policy;
- no acceptance role is granted from tool-only evidence;
- artifact identity is present;
- partition/role leakage checks pass where partition fields are populated.

G3 passes only after the real ledger artifact satisfies these checks.

## Planned outputs

Under `docs/plan/ml-R4/`:

- `schemas/evidence_ledger_row.v1.schema.json`
- `schemas/evidence_item.v1.schema.json`
- `schemas/evidence_ledger_manifest.v1.schema.json`
- `scripts/p3_validate_evidence_ledger.py`
- `fixtures/p3_valid_ledger_fixture.jsonl`
- `fixtures/p3_invalid_ledger_cases.jsonl`
- `findings/04_phase3_state_mapping.md`
- `findings/04_evidence_ledger_schema_report.md`
- `manifests/evidence_items_v1.jsonl`
- production ledger Parquet + manifest once protected IDs are available

## Non-goals

Do not:

- change protected historical labels;
- regenerate the historical v3 export;
- make KEEP/DROP source-policy decisions reserved for later phases;
- start gap review without an approved gap ID;
- infer confirmed negatives from historical zeros;
- retrain the model or change architecture/thresholds/calibration.

## First authorized action

Implement and validate P3-W1/P3-W2/P3-W3 remotely. Keep Phase 3 `IN_PROGRESS` until P3-W4 can be executed against the real protected population.
