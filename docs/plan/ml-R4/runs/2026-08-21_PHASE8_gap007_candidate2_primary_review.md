# Phase-8 R4-GAP-007 candidate #2 primary review

**Date:** 2026-08-21
**Candidate:** `r4neg-bfe90ef82e33a324d612256a5d4053c6`
**Contract:** `f7afe9fff9f6c117c6cd9dd4730c0f12e3cc3c8ab98797911de091e240051b93`
**Class:** `CallToUnknown` / index 0
**Primary decision:** supports `CONFIRMED_NEGATIVE`, pending genuinely independent verification
**Current target/state:** `None` / `UNKNOWN` / `PENDING_REVIEW`
**Training:** not authorized

## Outcome

Complete primary review supports a class-specific negative conclusion for `CallToUnknown`, but this is **not an accepted adjudication yet**. The contract contains no raw or unchecked low-level call and no dynamic typed callback. Its one external interaction is:

```solidity
_customerAddress.transfer(_dividends);
```

`_customerAddress` is `msg.sender`; dividend/referral state is updated before the transfer, and Solidity 0.4.25 `transfer` reverts on failure. Recovered project evidence previously identified `.transfer()`-only behavior as a major false-positive root cause. The call does not satisfy the direct class intent of an unknown-target raw call or unchecked `send`.

The candidate remains UNKNOWN with target `None` because the confirmed-negative contract requires a genuinely distinct agreeing verifier. This assistant must not verify its own primary conclusion.

## Complete source review

All 343 source lines and both declarations (`ECT`, `SafeMath`) were reviewed. The complete interaction inventory is:

1. `withdraw()` updates `payoutsTo_`, adds and clears referral balance, then transfers Ether to the caller at line 109.
2. `transfer()` may call internal `withdraw()` before later token-accounting writes; Slither reports this as a separate `reentrancy-unlimited-gas` concern.
3. All `SafeMath.*` calls target the same-file library and are not external unknown-target interactions.

No `.call`, `.callcode`, `.delegatecall`, `.staticcall`, `.send`, assembly call, typed interface callback, `new`, `selfdestruct`, or `suicide` site exists.

The two Slither reentrancy findings are not ignored: they establish possible evidence for a different vulnerability class. Confirmed-negative scope is explicitly class-specific and never means that the contract is globally safe.

## Physical and provenance binding

| Artifact | SHA-256 |
|---|---|
| normalized source | `f7afe9fff9f6c117c6cd9dd4730c0f12e3cc3c8ab98797911de091e240051b93` |
| preprocessing metadata | `597fe21c9427fb6f56cf07e738e810aad35e9a0729c644a621670b11000735e9` |
| graph tensor | `351681cc774824799967985a56ea956e92b4d30fec69475144fb273c669b4b52` |
| token tensor | `759a858c2dca19ace5e5f3d8f4b7c3578993a3c546841c1a3efd4a106f8cbcfd` |
| representation sidecar | `6129a99de959f6fe68862ade77a096566424250db9cba6da9be7bb6fdba3c9eb` |

Metadata binds the contract to DIVE record `repo/__source__/14306.sol`, pragma `^0.4.25`, compiler `0.4.25`, and `compile_status=ok_normalized_bytes`.

## Representation review and R4-GAP-008

The one bound graph component has 205 nodes and 872 edges. It contains the Ether-transfer CFG node and reports `has_cei_path=1`.

However, the v9 graph signal is semantically distorted:

- 30 type-11 `EXTERNAL_CALL` edges exist;
- all 30 are attached to `SafeMath` `LibraryCall` nodes;
- the actual Ether `Transfer` node has no type-11 edge.

The token tensor has shape `[4,512]`, selects windows `[0,7,13,20]`, and retains 1,819 of 5,369 unique code tokens (`0.3387967964239151`). The selected text omits the `withdraw()` Ether-transfer site.

These limitations do not create positive or negative source truth. They make the candidate valuable for later false-positive evaluation, but they also triggered the separate full-population representation audit R4-GAP-008.

## Targeted static corroboration

Slither `0.11.5` with `solc 0.4.25+commit.59dbf8f1` ran:

```text
low-level-calls
unchecked-lowlevel
unchecked-send
arbitrary-send-eth
reentrancy-eth
reentrancy-no-eth
reentrancy-unlimited-gas
```

It returned two `reentrancy-unlimited-gas` results and no low-level/unchecked-send result. Tool output is corroborating only; the source review owns the class-specific primary conclusion.

## Current decision boundary

| Question | Answer |
|---|---|
| Whole source reviewed? | Yes |
| All file graph components reviewed? | Yes; one of one |
| Contradictory CallToUnknown-positive evidence? | No |
| Other-class concern present? | Yes; Slither reentrancy findings |
| Primary supports class-specific negative? | Yes |
| Accepted `CONFIRMED_NEGATIVE`? | No; independent verification pending |
| Target or role changed? | No |
| Training authority changed? | No |

The independent reviewer must work from the blind source bundle, use a distinct reviewer identity/context, and return `AGREES`, `DISAGREES`, or `INSUFFICIENT_EVIDENCE` with its own rationale and evidence. Only an agreeing sufficient review can be combined with this primary review and passed to the adjudication validator.

Deterministic blind bundle:

`docs/plan/ml-R4/review_bundles/r4_gap007_candidate2_independent_review_v1.zip`

Archive SHA-256:

`2e7f48c9648097624406d167266a42a31055f222a0f468a0453b2f353b343f1a`
