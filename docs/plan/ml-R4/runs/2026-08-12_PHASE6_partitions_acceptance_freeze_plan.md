# R4 Phase 6 — Leakage-Safe Roles, Partition Freeze, and Acceptance Plan

**Phase:** 6 — Dataset Roles, Leakage-Safe Partitions, and Acceptance Freeze  
**Gate:** G6  
**Branch:** `r4/phase6-partitions-acceptance-freeze`  
**Entry condition:** G5 PASS / `data-vnext-policy-v1` accepted  
**Execution mode:** deterministic role manifests and support audit only; no DATA vNext implementation/training

## 1. Objective

Freeze one compatible role per leakage group before any DATA vNext artifacts are implemented or any retraining begins. Where a role lacks trustworthy evidence, publish an explicit unsupported/empty manifest rather than borrowing exposed, noisy, or semantically invalid data.

## 2. Governing constraints

- Group key precedence: `project_group_id` → `dedup_group_id` → `contract_id`.
- One leakage group may not cross incompatible roles.
- Phase-5 source/class authority is a ceiling; Phase 6 may only restrict it further.
- Historical binary labels/splits are lineage inputs, not vNext outcome truth or role assignments.
- Historical zeros never become negatives.
- DIVE Front Running→TOD may be weak training only; other DIVE source assertions are masked/unlabeled.
- SmartBugs Timestamp historical positives are row-level ambiguous because Phase-3 ledger cannot distinguish direct `time_manipulation` from historical `bad_randomness→Timestamp`; these rows remain unlabeled unless category identity is recovered safely.
- GasException and UnusedReturn supervision remain disabled.
- No role needing reliable negative outcomes may be populated from tool silence, BCCC `NonVulnerable`, historical all-zero records, or the corrupted quickstart `NonVulnerable` mapping.
- A corpus previously used to validate model/agent behavior cannot be called untouched acceptance.

## 3. Evidence audit before assignment

### Active Phase-3 population

The committed Phase-3 ledger is the partition population root:

`3983cc2b3317515d546c784449b583ac9a7c23ac8da267ee10f5640857cd0ac7`

For role eligibility, derive only these Phase-5-authorized signals:

- SolidiFI historical positive at an enabled injected class → strong positive candidate;
- SmartBugs Curated historical positive at an enabled class except Timestamp → strong positive candidate;
- SmartBugs Timestamp positive → ambiguous/unlabeled candidate;
- DIVE TOD positive → weak positive candidate;
- every other active row → unlabeled/no supervised outcome candidate.

Group-level role assignment applies to every contract in the group; row-level masks still follow policy.

### Historical/manual evaluation corpora

- `manual_hand_written_contracts/` has explicit `// expect:` labels but was deliberately created/used to validate ML and AGENTS. It is **exposed** and may be considered only for internal audit/case study, never untouched acceptance.
- `benchmark_v0.1_quickstart` is not trustworthy as a negative benchmark: its Tier-A builder maps SmartBugs `access_control` and SolidiFI `tx.origin` to `NonVulnerable`, contradicting current canonical ExternalBug semantics. The benchmark is therefore excluded from threshold/calibration/acceptance authority.
- Tier-E “known-safe” design uses BCCC NonVulnerable folder membership plus absence of Slither/Aderyn high/medium findings. Tool silence and BCCC folder labels do not establish class-specific confirmed negatives. No committed Tier-E manifest exists in the quickstart output.

## 4. Role strategy for the first vNext baseline

### TRAIN_STRONG

Eligible leakage groups containing at least one strong Phase-5 positive signal, unless the whole group is reserved for model selection/internal audit.

### MODEL_SELECTION

Reserve a deterministic subset of strong-positive groups only. This role supports positive-recall/positive-loss checkpoint diagnostics, **not** full discrimination/F1 because trustworthy negative support is absent. The manifest/support table must state this limitation.

### INTERNAL_AUDIT

Reserve a deterministic subset of strong-positive groups. An additional exposed manual-contract manifest may be published for qualitative/internal audit, clearly outside untouched acceptance.

### TRAIN_WEAK

Groups with no strong signal and at least one authorized DIVE TOD weak-positive signal.

### TRAIN_UNLABELED

Remaining active groups that contain structurally valid active-population contracts but no authorized supervised signal.

### THRESHOLD_FIT

**UNSUPPORTED / controlled empty manifest.** No trustworthy class-specific negative support exists for a discrimination threshold fit.

### CALIBRATION_FIT

**UNSUPPORTED / controlled empty manifest.** Calibration cannot be supported honestly from positive-only/unknown labels.

### UNTOUCHED_ACCEPTANCE

**UNSUPPORTED / controlled empty frozen manifest.** Existing evaluation/manual corpora are exposed, semantically invalid for negatives, unavailable, or deferred. Phase 6 must record this explicitly rather than rename an old test set as untouched.

### CASE_STUDY

May reference the exposed manual suite and Phase-4 reviewed examples, but case-study membership is not metric/selection authority.

## 5. Deterministic group assignment

Generate a group inventory from the Phase-3 ledger and `data-vnext-policy-v1`.

Group classification precedence:

1. any strong-positive candidate → `STRONG_ELIGIBLE_GROUP`;
2. else any weak-positive candidate → `WEAK_ELIGIBLE_GROUP`;
3. else → `UNLABELED_GROUP`.

For strong groups, deterministically rank:

`SHA256(partition_version | ledger_sha | policy_sha | group_id)`

and assign approximately:

- 70% `TRAIN_STRONG`
- 15% `MODEL_SELECTION`
- 15% `INTERNAL_AUDIT`

The implementation must verify every enabled class with strong support has nonzero train support and, where feasible, model-selection/internal-audit support. If a class has insufficient groups, fail closed and assign the scarce class to training/internal audit rather than inventing support.

Weak groups go to `TRAIN_WEAK`; unlabeled groups go to `TRAIN_UNLABELED` for the first baseline. No group is split because of historical train/val/test membership.

## 6. Required outputs

1. role/group manifest for the active 22,493-contract population;
2. per-role contract/group counts;
3. per-class strong/weak/unlabeled support table;
4. source composition and historical-split exposure table;
5. explicit unsupported-role manifest for threshold/calibration/untouched acceptance;
6. exposure register for manual/quickstart/other evaluation candidates;
7. leakage validator proving each group has exactly one role;
8. hashes and lineage binding to Phase-3 ledger + Phase-5 policy;
9. G6 report.

## 7. Acceptance freeze semantics

The untouched-acceptance manifest is frozen as **empty/unsupported** for this baseline unless a genuinely unexposed, semantically trustworthy corpus is discovered without new acquisition/review.

That is a valid G6 outcome. It means Phase 10 cannot make an untouched-acceptance promotion claim for the first repaired baseline; later work must add a separately protected acceptance corpus before such a claim becomes possible.

## 8. Non-goals

Do not in Phase 6:

- rewrite parsers/crosswalk/merger/export;
- create DATA vNext label-state artifacts beyond role/support manifests;
- change Phase-5 source authority;
- generate confirmed negatives from safe folders/tool silence;
- acquire Web3Bugs/BCCC/DeFiHackLabs;
- train or evaluate the teacher model;
- tune thresholds/calibration;
- expose a future acceptance corpus.

## 9. G6 pass criteria

G6 passes if:

- every active leakage group has one compatible role;
- strong/weak/unlabeled role assignments obey Phase-5 policy;
- model-selection limitations are explicit;
- threshold/calibration roles are explicitly unsupported rather than contaminated;
- untouched acceptance is hash-frozen as empty/unsupported with exposure rationale;
- no exposed manual/benchmark corpus is mislabeled untouched;
- all manifests/support tables are deterministic and hash-bound;
- no protected historical artifact changes.
