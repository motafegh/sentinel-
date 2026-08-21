# Phase-8 R4-GAP-007 candidate #1 local handoff

**Date:** 2026-08-21
**Canonical branch target:** `main`
**Physical DATA authority:** R4-D-008 / repaired-v2
**Logical authority:** R4-D-009 / accepted logical V3
**Evidence gap:** R4-GAP-007
**State:** PILOT IN PROGRESS — candidate #1 partial primary review only
**Training:** NOT AUTHORIZED
**G8:** OPEN

## Purpose

This is the current restart/handoff record for continuing R4-GAP-007 on the protected local Sentinel worktree (`~/projects/sentinel`). It records the exact state reached during the first real confirmed-negative pilot candidate review so a local AI assistant can continue without replaying the conversation or guessing at prior reasoning.

This record is additive. It does **not** supersede or mutate:

- repaired-v2 physical DATA;
- accepted logical V3 grouping/roles/publication;
- the coherent committed V3 evidence snapshot;
- the hardened 200-cell confirmed-negative review queue.

It only advances the execution state from **pilot ready** to **pilot in progress**.

## Required authority/read order

Before continuing, read:

1. `CLAUDE.md`;
2. `docs/plan/ml-R4/PLAN_STATUS_MATRIX.md`;
3. `docs/plan/ml-R4/runs/2026-08-16_PHASE8_v3_hardened_evidence_snapshot_closeout.md`;
4. `docs/plan/ml-R4/EVIDENCE_GAP_REGISTER.md`;
5. this handoff;
6. `data_module/sentinel_data/vnext/confirmed_negative_evaluation.py`;
7. `docs/plan/ml-R4/scripts/p8_validate_confirmed_negative_adjudications.py` before writing or validating adjudication artifacts.

The committed queue authority is:

`docs/plan/ml-R4/evidence/2026-08-15_phase8_logical_v3/confirmed_negative_review_queue_v1.json`

Do not use the obsolete V2 queue or any pre-hardening V3 queue.

## Unchanged accepted baseline

Current accepted physical DATA remains:

- 22,540 contracts;
- 225,400 contract×class rows;
- 67,620 graph/token/sidecar files;
- 1,080 positive targets;
- 224,320 UNKNOWN targets;
- 0 confirmed-negative targets;
- 474 STRONG and 606 WEAK semantic cells;
- physical binding digest `16dd4a3f98c34e52e5c411b39268361881efede07e8f3f52d0c060dd1c5bb6dd`.

Accepted logical V3 remains:

- dataset `sentinel-r4-vnext-v3`;
- grouping `r4-leakage-groups-v3`;
- partition `r4-vnext-roles-v3`;
- 22,394 leakage groups;
- maximum group size 7;
- 146 normalized-code edges;
- zero address-authority edges;
- 932 optimizer-active positive-only cells/groups.

The hardened queue remains exactly 200 `PENDING_REVIEW` cells, 25 for each of the eight enabled supervised classes, across 200 globally unique leakage groups. Queue membership is review reservation only and is **not negative truth**.

## Candidate #1 — exact committed queue identity

The first deterministic queue candidate is:

```text
Class:        CallToUnknown
Class index:  0
Candidate:    r4neg-f6a71e420a116cb4b9a334ba961ba1b6
Contract ID:  defe4690028dc863df4611176a4c35f0ffd0bbc90f61db2bd4f25f5ad7f2a384
Group:        r4grp-91091daa51a561493045bd21a5d321fa
Source:       dive
Current:      UNKNOWN
Target:       None
Role:         TRAIN_UNLABELED
Ordinal:      1
```

Queue publication manifest SHA-256:

`26b0a4b103010d3f2bcbb6ca08aaec98047dfeb8a4954440aa913c4a7053400f`

**Current candidate state remains unchanged:** `PENDING_REVIEW`, `UNKNOWN`, target `None`, `negative_truth_claim=false`.

No adjudication row has been accepted and no target `0` has been created.

## Local physical files already located

For the candidate contract ID, the protected local worktree contains:

- `data_module/data/sentinel-preprocessed-r4-v2/dive/<CID>.sol`;
- `data_module/data/sentinel-preprocessed-r4-v2/dive/<CID>.meta.json`;
- `data_module/data/representations-r4-v2/dive/<CID>.pt`;
- `data_module/data/representations-r4-v2/dive/<CID>.tokens.pt`;
- `data_module/data/representations-r4-v2/dive/<CID>.rep.json`.

Important metadata already observed:

- source: DIVE;
- pragma: `^0.4.18`;
- selected solc: `0.4.18`;
- compile status: `ok_normalized_bytes`;
- preprocessing artifact: `sentinel-preprocessed-r4-v2`;
- normalizer: `r4-lexical-v2`;
- 271 raw lines / 272 normalized lines;
- contract declarations include `owned`, `TokenERC20`, `CarbonExchangeCoinToken`;
- no address literals recorded in this candidate metadata.

Do not rebuild or overwrite these physical artifacts.

## Review work already completed

### 1. Whole-file external-interaction scan

The local source scan found these relevant locations:

```text
27: interface tokenRecipient { function receiveApproval(address _from, uint256 _value, address _token, bytes _extraData) public; }
134: tokenRecipient spender = tokenRecipient(_spender);
136: spender.receiveApproval(msg.sender, _value, this, _extraData);
254: msg.sender.transfer(amount * sellPrice);
```

No `.call`, `.delegatecall`, `.staticcall`, or `.send` site was found by the targeted scan.

### 2. Function inventory

The file contains the expected ownership/ERC20/exchange functions, including:

- `approveAndCall(...)`;
- `buy()`;
- `sell(uint256 amount)`;
- token transfer/allowance/burn functions;
- owner-only mint/freeze/price-management functions.

### 3. `approveAndCall()` inspected

The relevant source is:

```solidity
function approveAndCall(address _spender, uint256 _value, bytes _extraData)
    public
    returns (bool success) {
    tokenRecipient spender = tokenRecipient(_spender);
    if (approve(_spender, _value)) {
        spender.receiveApproval(msg.sender, _value, this, _extraData);
        return true;
    }
}
```

Interpretation reached so far:

- `_spender` is caller supplied;
- `spender.receiveApproval(...)` is therefore a real external interaction with a dynamic address;
- it is a typed/high-level Solidity interface call, not a raw low-level `.call(...)` or `.delegatecall(...)`;
- this fact alone establishes neither a positive nor a negative `CallToUnknown` outcome.

### 4. `sell()` interaction already identified

The source includes:

```solidity
msg.sender.transfer(amount * sellPrice);
```

This is also a real external-value interaction, but a Solidity `transfer` in the legacy compiler era is materially different from an unchecked raw low-level call. Again, this fact alone establishes neither a positive nor a negative class outcome.

### 5. Important class-semantics caution

Project test/recall logic for `CallToUnknown` is intentionally coarse and can recognize low-level/external-call-like syntax including `.transfer(...)`. Therefore:

- absence of `.call(...)` is **not** enough to declare this candidate negative;
- presence of `.transfer(...)` or a typed callback is **not** automatically vulnerability truth;
- the primary reviewer must evaluate the complete contract against the class-specific semantic intent and evidence policy, not merely grep syntax or graph edge presence.

The candidate is currently only a **plausible negative candidate**. No verdict has been made.

## Exact next local work

Continue locally rather than sending large source/graph outputs through chat.

### Step 1 — complete primary code-scope review

Review the entire 271-line Solidity source, not only the two call sites. At minimum determine:

- every externally controlled address/value involved in external interaction;
- all state changes before and after external interactions;
- whether any unchecked/raw low-level call behavior exists through syntax not caught by the first scan;
- whether callback/reentrancy/error-propagation semantics create a class-relevant concern;
- whether inheritance or duplicated `_transfer` logic changes the conclusion;
- whether any code path is missed by focusing only on `approveAndCall()` and `sell()`.

Do not set `code_scope_complete=true` until the whole file has actually been reviewed.

### Step 2 — review representation/graph components

Inspect the candidate `.rep.json` and relevant graph payload/sidecar evidence so `all_file_graph_components_reviewed=true` is supportable rather than ceremonial.

The graph schema is coarse (`v9`) and may expose an `EXTERNAL_CALL` relation without proving untrusted-target vulnerability semantics. Treat graph/tool evidence as corroboration unless it independently satisfies an approved primary evidence type.

### Step 3 — re-check project class semantics from source/tests

Use executable source/tests as authority for what the current model/taxonomy mechanically encodes, while keeping semantic truth distinct from a recall heuristic.

Do not equate a regex/checker pattern with ground truth.

### Step 4 — choose one explicit primary decision

Allowed adjudication decisions are:

- `CONFIRMED_NEGATIVE`;
- `NOT_CONFIRMED`;
- `EXCLUDE`.

If evidence remains ambiguous, use `NOT_CONFIRMED` or `EXCLUDE`. Ambiguity must never fail open into target `0`.

### Step 5 — if and only if primary review supports `CONFIRMED_NEGATIVE`

The adjudication must use:

- `negative_scope = CLASS_SPECIFIC_ONLY`;
- a real `primary_review` block with reviewer ID, timestamp, rationale, complete code scope, complete file/graph-component review, no contradictory positive evidence, and evidence independent of the training label;
- at least one primary evidence type:
  - `MANUAL_CLASS_SPECIFIC_REVIEW`,
  - `FORMAL_STATIC_ARGUMENT`, or
  - `TRUSTED_EXPLICIT_NEGATIVE_SOURCE`.

Then obtain **independent verification**:

- a different reviewer identity/context from the primary reviewer;
- `status = AGREES`;
- non-empty independent evidence;
- evidence independent of the training label.

One AI context must not self-confirm its own primary review by merely changing labels/names. Use a genuinely separate reviewer context/assistant or a human review.

### Step 6 — validate using explicit current paths

Before running the validator, inspect its CLI and provide explicit paths for the committed hardened V3 queue and the new adjudication/output locations. Do **not** rely blindly on historical default paths that may point at an older build root.

Use the project Python environment for project Python commands:

`./ml/.venv/bin/python`

Global `python` is not available in the known local shell; `python3` may exist but the project interpreter is the safer project-bound choice.

### Step 7 — record results without mutating accepted evidence

Do not edit the committed queue, snapshot JSON, repaired-v2 roots, or accepted V3 publication by hand.

Create/version new adjudication/evaluation evidence and a new run/working record for actual pilot results. If candidate #1 is not confirmed, record that result explicitly and proceed according to R4-GAP-007 rather than manufacturing a negative.

## Confirmed-negative authority remains narrow

If a candidate eventually passes dual review, the accepted cell is:

- `outcome_state = CONFIRMED_NEGATIVE`;
- `target_value = 0`;
- `usage_authority = EVALUATION_ONLY_NOT_TRAINING_AUTHORITY`.

It does **not** authorize:

- optimizer training on that target;
- threshold fitting;
- calibration fitting;
- selector promotion;
- full Phase-8 training.

The implementation explicitly returns all three authorizations as false.

## PU/objective-design note

Recent design discussion clarified the intended sequencing but created **no new learning-policy decision**.

Positive–Unlabeled (PU) learning is a serious future candidate because Sentinel has trustworthy positives plus a large UNKNOWN population, but it has **not been selected or implemented**.

Current intended order remains:

```text
confirmed-negative pilot
        ↓
measure negative yield / false-positive evaluation support
        ↓
versioned objective/evaluation design
        ├── ordinary supervised option
        ├── PU option
        └── hybrid/evidence-aware option
        ↓
small controlled comparisons
        ↓
possible later G8/full-training authorization
```

The current confirmed-negative queue and any accepted evaluation groups must remain outside a future PU/unlabeled optimizer population until a new role/objective policy explicitly reconciles the reservation. This rule is already enforced/documented in `confirmed_negative_evaluation.py`.

Do not implement PU as part of this candidate review.

## Secondary independent track remains unchanged

The guarded selector `target_aware_guarded_v1` remains unpromoted. Before any selector-promotion ADR, execute full-population verification that the historical control selector reproduces the token tensors currently bound to accepted representations.

This selector track is separate from candidate #1 review and does not authorize training.

## Hard stop lines

Do not:

- infer target `0` from DIVE zero/source silence/unlabeled state/tool silence;
- edit the 200-row queue to mark a result;
- overwrite repaired-v2 source/representation roots;
- mutate accepted V3 grouping/roles/publication;
- use V2/pre-hardening queue artifacts;
- treat `.transfer`/`EXTERNAL_CALL` pattern presence as vulnerability truth by itself;
- let the same reviewer self-verify a negative;
- add accepted evaluation negatives to optimizer supervision without a new versioned decision;
- implement PU prematurely;
- fit threshold/calibration roles;
- reuse Run12 learned state;
- silently promote the guarded selector;
- start the 100-epoch Phase-8 run.

## Local-worktree hygiene

Known unrelated local untracked files from the prior protected worktree must not be staged accidentally:

- `Data audit.md`;
- `Ml audit.md`;
- `Zk contracts audit.md`;
- `docs/plan/system-finalization/2026-07-15_SYSTEM_R0_PLAN_final_containment_recovery.md`;
- `docs/plan/system-finalization/2026-07-15_SYSTEM_R4_PLAN_data_ml_label_reality.md`.

Stage only files intentionally created/changed by the local continuation.

## Handoff checkpoint

At this handoff point:

```text
Physical DATA repair                    COMPLETE / ACCEPTED
Logical V3 correction                  COMPLETE / ACCEPTED
Evidence hardening + snapshot          COMPLETE / VERIFIED
R4-GAP-007 queue                       COMPLETE / COMMITTED
R4-GAP-007 pilot                       IN PROGRESS
Candidate #1 primary review            PARTIAL
Candidate #1 verdict                   NONE
Candidate #1 independent verification  NOT STARTED
Confirmed negatives                    0
PU/objective implementation            NOT STARTED / NOT AUTHORIZED
Selector promotion                     NOT AUTHORIZED
Full training                          NOT AUTHORIZED
G8                                     OPEN
```

The local AI assistant should continue from **candidate #1 complete primary review**, not restart DATA repair, regenerate the queue, or launch training.
