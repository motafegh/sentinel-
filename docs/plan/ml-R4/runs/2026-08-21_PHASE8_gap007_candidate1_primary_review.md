# Phase-8 R4-GAP-007 candidate #1 primary review

**Date:** 2026-08-21
**Candidate:** `r4neg-f6a71e420a116cb4b9a334ba961ba1b6`
**Contract:** `defe4690028dc863df4611176a4c35f0ffd0bbc90f61db2bd4f25f5ad7f2a384`
**Class:** `CallToUnknown` / index 0
**Primary decision:** `NOT_CONFIRMED`
**Target change:** none
**Training:** not authorized

## Outcome

Candidate #1 is **not a confirmed negative**. The complete primary review found no raw `.call`, `.delegatecall`, `.callcode`, `.send`, assembly call, or ignored low-level success value. It did find a typed external callback to a caller-selected address:

```solidity
tokenRecipient spender = tokenRecipient(_spender);
if (approve(_spender, _value)) {
    spender.receiveApproval(msg.sender, _value, this, _extraData);
    return true;
}
```

The project taxonomy defines `CallToUnknown` as unchecked low-level calls **or calls to unverified external addresses**. Because `_spender` is supplied by the caller and its implementation is not verified by this contract, the callback is contradictory or ambiguous positive evidence under the current class boundary. Absence of raw low-level syntax and zero targeted tool findings therefore cannot support `contradictory_positive_evidence_found=false`.

The fail-closed disposition is `NOT_CONFIRMED`. The committed queue remains immutable; the candidate remains `UNKNOWN`, `PENDING_REVIEW`, target `None`, and `negative_truth_claim=false`. No independent verification is required for a non-confirmed decision. Confirmed negatives remain zero.

## Authority and identity checks

The review used only the committed hardened V3 queue:

`docs/plan/ml-R4/evidence/2026-08-15_phase8_logical_v3/confirmed_negative_review_queue_v1.json`

The queue identity was rechecked before review:

| Field | Value |
|---|---|
| candidate | `r4neg-f6a71e420a116cb4b9a334ba961ba1b6` |
| contract | `defe4690028dc863df4611176a4c35f0ffd0bbc90f61db2bd4f25f5ad7f2a384` |
| leakage group | `r4grp-91091daa51a561493045bd21a5d321fa` |
| source | `dive` |
| queue ordinal | 1 within `CallToUnknown` |
| queue state | `UNKNOWN` / `PENDING_REVIEW` / target `None` |

The adjudication is additive evidence at:

`docs/plan/ml-R4/reviews/R4-GAP-007/candidate1_primary_adjudication_v1.jsonl`

It does not edit the queue, repaired-v2 roots, accepted logical V3 publication, or coherent pre-pilot snapshot.

## Physical and provenance evidence

All bound candidate artifacts exist and were hash-checked:

| Artifact | SHA-256 |
|---|---|
| normalized Solidity source | `defe4690028dc863df4611176a4c35f0ffd0bbc90f61db2bd4f25f5ad7f2a384` |
| preprocessing metadata | `1d660a7d9da985831a07cb21f511fc1cdf365c32d3964c2518f689ba87778d89` |
| graph tensor | `41349719d9d81fe728e02dae6f311c1917fa58b6ee6f7c5997fa7f931efaa548` |
| token tensor | `e6801b3ce66a71dcd4de5f449c550329648cf5f95893b4c17c73ee8ea62b4ef1` |
| representation sidecar | `a37f44be384750e08648ed6b5a7b1e670fb4a6ae6fd03fa19d56c79880b0eb55` |

The metadata binds the normalized source to raw DIVE record `repo/__source__/1470.sol`, pragma `^0.4.18`, selected compiler `0.4.18`, and compile status `ok_normalized_bytes`. The normalized source is the comment-stripped form of the raw source; the executable Solidity content relevant to this review is preserved.

## Complete code-scope review

All 271 normalized source lines and all declared contracts (`owned`, `tokenRecipient`, `TokenERC20`, and `CarbonExchangeCoinToken`) were reviewed.

The complete external-interaction inventory is:

1. `selfdestruct(owner)` in owner-gated `destruct()` (line 23). The destination is the stored owner and this is not a low-level call whose success is ignored.
2. `spender.receiveApproval(...)` in `approveAndCall()` (line 136). `_spender` is caller controlled, converted to the `tokenRecipient` interface, and called after `allowance[msg.sender][_spender]` is updated. This is the class-boundary conflict that prevents negative confirmation.
3. `msg.sender.transfer(amount * sellPrice)` in `sell()` (line 254). The function checks contract balance, transfers token state to the contract, then performs the Ether transfer. Under Solidity 0.4.18, `transfer` throws on failure; it is not an unchecked boolean-returning `send` or raw low-level call.

No hidden call form was found through inheritance, duplicated `_transfer` implementations, contract creation, inline assembly, fallback logic, or library dispatch. No `.call`, `.callcode`, `.delegatecall`, `.send`, or `new` expression exists.

Solidity 0.4.18 documentation supports the semantic distinction used here: failures in ordinary sub-calls bubble automatically, with `send` and low-level `call`/`delegatecall`/`callcode` as exceptions; `transfer` throws on failure. The same version warns that calls through an explicitly converted contract type are dangerous when the target implementation is not known in advance.

## Bound representation review

The representation sidecar selects the inheritance leaf `CarbonExchangeCoinToken` and records exactly one file-level graph component. That component was deserialized and reviewed:

- 205 nodes with 12 features each;
- 382 edges;
- `has_cei_path=0`;
- no edge of type 11 (`EXTERNAL_CALL`);
- three `CFG_NODE_CALL` nodes relevant to the reviewed interactions: two inherited copies of `spender.receiveApproval(...)` at source line 136 and one `msg.sender.transfer(...)` node at line 254.

The graph therefore contains both important source sites as CFG call nodes but does not emit the class pattern's type-11 external-call self-loop. This graph silence is a representation limitation, not negative evidence.

The bound token artifact has shape `[4,512]`, selected pre-subsampling windows `[0,2,5,7]`, and retains 1,813 of 2,065 unique code tokens (`0.8779661016949153`). Decoding/inspection confirmed that the selected windows retain both the callback and Ether-transfer sites. Token coverage is sufficient to establish that the model input sees the relevant text, but it does not adjudicate semantic truth.

## Targeted static corroboration

Local analysis used Slither `0.11.5` with the exact available compiler `0.4.18+commit.9cf6e910` and these detectors:

```text
low-level-calls
unchecked-lowlevel
unchecked-send
arbitrary-send-eth
reentrancy-eth
reentrancy-no-eth
reentrancy-unlimited-gas
```

Result: four contracts analyzed with seven detectors and zero findings. Slither classified `receiveApproval(...)` as a high-level call, while the Ether transfer is represented as a transfer IR operation. This corroborates the absence of raw/unchecked low-level-call behavior; it does not negate the taxonomy's broader unverified-target clause and is not negative truth by itself.

## Data-quality decision table

| Question | Evidence-backed answer |
|---|---|
| Correct queue candidate and physical binding? | Yes |
| Whole source reviewed? | Yes |
| Every file-level graph component reviewed? | Yes; one of one |
| Relevant call sites retained in graph/tokens? | Yes |
| Raw or unchecked low-level call found? | No |
| Contradictory or ambiguous class-positive evidence found? | Yes; caller-selected typed callback |
| Can `CONFIRMED_NEGATIVE` be supported? | No |
| Primary decision | `NOT_CONFIRMED` |
| Independent verifier needed? | No |
| Target or optimizer authority changed? | No |

## Validation and next controlled step

The new adjudication was validated against the explicit hardened queue path with the repository validator. The versioned report is:

`docs/plan/ml-R4/reviews/R4-GAP-007/candidate1_evaluation_v1.json`

It reports `status=PASS`, one adjudication, decision count `NOT_CONFIRMED=1`, zero accepted confirmed-negative cells, no errors, and training/threshold/calibration authorizations all false.

Validator scope is intentionally narrower for non-confirmed decisions: current executable code checks queue identity, allowed decision, and a non-empty rationale, but does not structurally validate the optional `primary_review` block unless the decision is `CONFIRMED_NEGATIVE`. Therefore `PASS` proves that this result fails closed and creates no authority; the tracked source/representation evidence in this record supports the separate claim that the primary review itself was complete.

The next queue item, if the pilot continues, is deterministic candidate #2:

```text
candidate_id = r4neg-bfe90ef82e33a324d612256a5d4053c6
contract_id  = f7afe9fff9f6c117c6cd9dd4730c0f12e3cc3c8ab98797911de091e240051b93
group_id     = r4grp-dc843217924fe207d2a658ada327615a
class        = CallToUnknown / index 0
ordinal      = 2
```

At this closeout point candidate #2 had not been reviewed. Its subsequent primary-review state is recorded separately in `2026-08-21_PHASE8_gap007_candidate2_primary_review.md`; do not use this historical sentence as the current restart boundary. Continuing the pilot does not authorize training, selector promotion, threshold/calibration fitting, or PU implementation.
