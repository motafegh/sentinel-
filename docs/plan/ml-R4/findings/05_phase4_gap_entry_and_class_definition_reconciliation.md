# 05 — Phase 4 Gap Entry and Class-Definition Reconciliation

- **Phase:** R4 Phase 4 — Targeted Evidence-Gap Adjudication
- **Status:** PRE-REVIEW / HUMAN APPROVAL REQUIRED
- **Purpose:** establish the smallest decision-critical gap and a review rubric without performing new contract adjudication

## 1. Entry conclusion

The first DATA vNext baseline does not require executing all six historically proposed gaps.

The only currently decision-critical new semantic review is **R4-GAP-002**, narrowed to five DIVE native classes that actively map into the locked SENTINEL 10-class target but lack source-specific semantic precision evidence:

1. `DoS` → `DenialOfService`
2. `Arithmetic` → `IntegerUO`
3. `Unchecked Return Values` → `UnusedReturn`
4. `Time manipulation` → `Timestamp`
5. `Front Running` → `TransactionOrderDependence`

DIVE `Bad Randomness` is not included in the first-baseline review population because the current crosswalk deliberately drops it rather than mapping it into a canonical class. It should remain unknown/masked unless a later ADR proposes a new mapping.

## 2. Why the other proposed gaps do not block G4

| Gap | Phase-4 entry disposition | Reason |
|---|---|---|
| R4-GAP-001 Web3Bugs | defer / propose explicit first-baseline exclusion | No executable/recovered contribution to the historical active population. Acquisition would be source expansion, not repair of an active label stratum. |
| R4-GAP-003 BCCC provisional classes | defer unless Phase 5 proposes importing BCCC | BCCC v1.4 is historical/deferred, not an active source in the protected export. |
| R4-GAP-004 BCCC 2-tool consensus | defer | Tool intersection is not ground truth and detector coverage is incomplete. |
| R4-GAP-005 fuzzing | defer | Complementary evidence; not required to decide active source/class roles. |
| R4-GAP-006 exploit PoCs | defer | Valuable for later evidence-qualified evaluation/case studies, but not the smallest evidence needed for DATA vNext source policy. |

## 3. Class-order reconciliation

The current canonical schema is `FEATURE_SCHEMA_VERSION=v9` and locks the class order as:

| Current index | Class |
|---:|---|
| 0 | CallToUnknown |
| 1 | DenialOfService |
| 2 | ExternalBug |
| 3 | GasException |
| 4 | IntegerUO |
| 5 | MishandledException |
| 6 | Reentrancy |
| 7 | Timestamp |
| 8 | TransactionOrderDependence |
| 9 | UnusedReturn |

Recovered BCCC Phase-5 definition documents contain older historical class numbers. Their semantic inclusion/exclusion prose may be reused, but **their numeric class IDs are not authoritative for R4**. Phase-4 review records must bind by current canonical class name and current v9 index.

## 4. Proposed review rubric for R4-GAP-002

These are semantic review definitions, not DATA-role decisions. Human approval is required before they are frozen for adjudication.

### 4.1 DenialOfService — current index 1

**Positive semantic core:** an attacker can cause a critical contract operation or state transition to become permanently or indefinitely unavailable, including forced-failure push/payment loops or attacker-controlled conditions that block essential progress.

**Exclude:** ordinary `require`/`revert` validation, intentional pausability, isolated transaction failure that does not create persistent loss of availability, or a pure unbounded-loop gas problem whose primary mechanism belongs to `GasException`.

**DIVE-specific caution:** folder name `DoS` alone is not proof. Review must identify a reachable attacker-controlled availability failure, not merely a loop or external call.

### 4.2 IntegerUO — current index 4

**Positive semantic core:** security-relevant integer overflow/underflow can wrap silently (pre-Solidity-0.8 arithmetic without effective protection) or occurs inside a reachable unsafe `unchecked` path in Solidity 0.8+.

**Exclude:** compiler-checked post-0.8 arithmetic outside `unchecked`, correctly applied SafeMath/manual guards, or arithmetic with no security-relevant effect.

**DIVE-specific caution:** `Arithmetic` is broader than integer overflow/underflow. Arithmetic-pattern presence alone is insufficient.

### 4.3 UnusedReturn — current index 9

**Positive semantic core:** a meaningful return status/value from an external interaction is discarded such that execution may proceed under an incorrect assumption of success.

**Exclude:** return values explicitly checked/required, calls whose failure necessarily reverts and has no ignored success flag, or irrelevant discarded values with no security consequence.

**Boundary:** low-level call/send exception-propagation issues may overlap `MishandledException`; record overlap rather than forcing an artificial exclusive label.

### 4.4 Timestamp — current index 7

**Positive semantic core:** `block.timestamp`/`now` materially influences a security-sensitive outcome, access decision, value transfer, ordering/gating rule, or other state transition where feasible timestamp manipulation changes the result.

**Exclude:** logging/record-keeping only, long-duration timing where small timestamp manipulation has no meaningful effect, and pure bad-randomness cases that exist only because timestamp is used as entropy.

**DIVE-specific boundary:** the separate DIVE `Bad Randomness` folder remains dropped. `Time manipulation` should not be broadened by silently importing that folder's semantics.

### 4.5 TransactionOrderDependence — current index 8

**Positive semantic core:** an attacker can gain a material advantage because a contract's security-relevant outcome depends on relative transaction ordering observable before finalization, including front-running/race-condition patterns where an attacker can submit a competing transaction to execute first.

**Exclude:** ordinary sequential state changes where ordering cannot be adversarially exploited, purely off-chain market competition with no vulnerable contract state transition, or timestamp-only behavior better classified as `Timestamp`.

**DIVE-specific caution:** the source-native label is `Front Running`; review must establish an exploitable transaction-order dependency rather than treating any externally callable state update as front-running.

## 5. Proposed adjudication record states

Initial semantic review should use only:

- `SUPPORTS_POSITIVE`
- `DOES_NOT_SUPPORT_POSITIVE`
- `UNCLEAR_INSUFFICIENT`
- `CLASS_BOUNDARY_CONFLICT`

These review states do **not** create confirmed negatives. A DIVE positive that fails semantic support becomes evidence against trusting that source assertion; it does not prove the contract is globally negative for the class.

## 6. Evidence reveal discipline

Initial review should hide, where practical:

- historical SENTINEL model probabilities/tiers;
- prior tool votes;
- downstream merger outcome;
- current historical target outside the source-native class being reviewed.

After the semantic verdict is recorded, reconciliation may reveal:

- DIVE source-native assertion;
- Slither/Aderyn/tool evidence;
- Phase-2 transformation trace;
- Phase-3 ledger provenance.

Tool agreement remains a correlated evidence source, not independent ground truth.

## 7. Authorization boundary

No contract source review has been performed in this Phase-4 entry work.

The next new-evidence action requires explicit human approval of **R4-GAP-002**, scoped to the five mapped DIVE strata and the semantic rubric above. On approval, the register can move R4-GAP-002 from `PROPOSED` to `APPROVED`, after which the deterministic population/sample freeze may begin.
