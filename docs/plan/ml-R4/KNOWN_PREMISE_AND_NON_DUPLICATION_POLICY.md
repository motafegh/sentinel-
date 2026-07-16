# Known Premise and Non-Duplication Policy

## Binding premise

For R4 planning and execution, the following is accepted as the working premise:

> The existing contract-vulnerability labels are materially unreliable for trustworthy supervised training and evaluation.

This premise is based on extensive prior analyses, including contract-level reviews and source-specific investigations. R4 does not need to repeat a broad audit merely to establish the same premise again.

## What remains uncertain

The following still require evidence-backed resolution:

- which source/class strata remain usable;
- which historical positives can be retained;
- which historical zeros are explicit negatives versus unknown;
- which labels can be corrected from existing evidence;
- which labels must be masked or excluded;
- which source transformations introduced corruption;
- whether enough trustworthy examples exist for each dataset role;
- what limited new reviews are required to fill critical gaps.

## Reuse-first rule

Before authorizing any new contract review, the agent must search and register:

- previous DIVE investigations;
- previous BCCC review batches and reports;
- previous SolidiFI and SmartBugs analyses;
- manual review CSVs;
- source snippets;
- exploit reproductions;
- Slither, Aderyn, Echidna, and other tool outputs;
- benchmark case studies;
- prior AI-review artifacts;
- commits and scripts that generated previous conclusions;
- local-only retained artifacts.

Previous work becomes usable by converting it into structured evidence items with artifact identity, method, reviewer, date, conclusion, limitations, and contract-class scope.

## No-duplicate-review rule

A contract-class pair must not be reviewed again merely because:

- the previous result is inconvenient;
- a new report format is preferred;
- the agent wants to independently “confirm everything”;
- the previous work was performed by another AI;
- the contract belongs to a newly named phase;
- the previous conclusion is already sufficient for the intended role.

## Authorization required for new review

Every new review batch must reference one or more entries in `EVIDENCE_GAP_REGISTER.md`.

Permitted gap reasons:

- `MISSING_RAW_EVIDENCE`
- `CONTRADICTORY_PRIOR_CONCLUSIONS`
- `UNREVIEWED_SOURCE_CLASS_STRATUM`
- `INSUFFICIENT_CONFIRMED_NEGATIVES`
- `INSUFFICIENT_ACCEPTANCE_SUPPORT`
- `AMBIGUOUS_CROSSWALK_EFFECT`
- `UNRESOLVED_DUPLICATE_OR_PROJECT_FAMILY`
- `DEFINITION_CHANGED_WITH_MATERIAL_EFFECT`
- `ARTIFACT_INTEGRITY_FAILURE`

A review batch without an approved gap ID is invalid R4 work.

## Historical versus new evidence

- `HISTORICAL_RECOVERED`: a retained artifact from prior work.
- `HISTORICAL_CONCLUSION_ONLY`: conclusion exists but supporting raw evidence is missing.
- `NEW_REPRODUCTION`: a new run using surviving historical inputs.
- `NEW_GAP_REVIEW`: a newly authorized contract review.
- `UNAVAILABLE`: referenced artifact cannot be found.

A new reproduction must never be described as the original historical output.

## Review priority

When new review is necessary, prioritize the smallest set that can resolve a specific dataset-role decision. Do not target a predetermined global “gold set” size.

## Enforcement

The agent must update `EVIDENCE_GAP_REGISTER.md` before creating a new sample manifest. CI or audit scripts should reject review manifests without gap IDs.
