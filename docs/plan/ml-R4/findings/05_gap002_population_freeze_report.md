# 05 — R4-GAP-002 Population and Initial Sample Freeze

- **Phase:** R4 Phase 4 — Targeted Evidence-Gap Adjudication
- **Gap:** `R4-GAP-002`
- **Work package:** P4-WP1
- **Status:** POPULATION + SAMPLE IDENTITY FROZEN; LOCAL SOURCE BINDING PENDING
- **Validation environment:** GitHub Actions, Ubuntu 24.04, Python 3.12
- **Validated commit:** `3b3c2d87e0ec1ff7f32d8f4ed71c6228f9e40557`
- **Workflow run:** `31533355652`

## Frozen upstream identity

- Phase-3 ledger SHA-256: `3983cc2b3317515d546c784449b583ac9a7c23ac8da267ee10f5640857cd0ac7`
- Ledger rows: 224,930
- Sample version: `r4-gap-002-sample-v1`
- Initial sample SHA-256: `2899ad5a210ac6e2e2a4e6b43f31cd718afa3b1d603b659cdd6bf0918f34fbe9`
- Initial sample size: 100 contracts — 20 per approved stratum

## Selection invariants validated

- source = `dive`;
- historical state = `HISTORICAL_POSITIVE`;
- initial review uses TRAIN-only groups;
- any review group touching historical val/test is excluded;
- group key precedence = `project_group_id` → `dedup_group_id` → `contract_id`;
- no review group is reused across the five strata;
- deterministic SHA-256 ranking is bound to gap ID, class, group, contract, and committed ledger identity;
- model probabilities/tiers, tool votes, merger outcome, and non-target historical labels are excluded from the initial blind task manifest.

Five synthetic/adversarial sampling tests passed before the production freeze, including deterministic ordering, cross-split group exclusion, group-precedence behavior, fail-closed insufficient-population handling, and locked class-index mismatch rejection.

## Population results

| Canonical class | DIVE native label | DIVE positive rows | Unique review groups | Eligible TRAIN-only groups | Groups excluded because they touch val/test | Historical train / val / test positives |
|---|---|---:|---:|---:|---:|---|
| DenialOfService | `DoS` | 1,095 | 894 | 647 | 247 | 841 / 128 / 126 |
| IntegerUO | `Arithmetic` | 9,388 | 5,920 | 4,374 | 1,546 | 7,742 / 843 / 803 |
| Timestamp | `Time manipulation` | 6,272 | 4,443 | 3,256 | 1,187 | 5,028 / 635 / 609 |
| TransactionOrderDependence | `Front Running` | 604 | 492 | 340 | 152 | 447 / 81 / 76 |
| UnusedReturn | `Unchecked Return Values` | 5,859 | 3,998 | 2,958 | 1,040 | 4,757 / 549 / 553 |

## Interpretation

The approved strata all have enough TRAIN-only, group-isolated population for the initial 20-contract screening batch without exposing val/test groups. The smallest stratum, TransactionOrderDependence, still has 340 eligible TRAIN-only groups, so initial sampling is not population-constrained.

The initial batch is deliberately not treated as a final authority sample. A stratum that is clearly unsuitable for trusted use may stop at mask/exclude. A stratum that appears suitable for a higher-authority role must expand adaptively and receive second review before promotion.

## Remaining WP1 boundary

The exact 100 sampled IDs are frozen by `sample_sha256` and available from the successful CI artifact. Before WP2 semantic review begins, those IDs must be bound to the protected local files:

`data_module/data/preprocessed/dive/<contract_id>.sol`

and matching `.meta.json` sidecars. The local bundle builder records source/meta hashes and packages only the 100 sampled contracts; it does not publish the broader protected DIVE corpus.
