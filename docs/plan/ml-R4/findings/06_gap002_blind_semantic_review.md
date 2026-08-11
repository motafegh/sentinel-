# 06 — R4-GAP-002 Blind Semantic Review

- **Phase:** R4 Phase 4 — Targeted Evidence-Gap Adjudication
- **Gap:** `R4-GAP-002`
- **Sample:** `r4-gap-002-sample-v1`
- **Frozen sample SHA-256:** `2899ad5a210ac6e2e2a4e6b43f31cd718afa3b1d603b659cdd6bf0918f34fbe9`
- **Review kind:** AI primary semantic review
- **Review mode:** source-only blind review
- **Reviewer model:** GPT-5.6 Sol
- **Scope:** 100 checksum-bound DIVE contracts; 20 per authorized stratum

## 1. Review discipline

The first pass used only the normalized/flattened Solidity source in the checksum-verified blind bundle plus the frozen Phase-4 class rubric.

The following were intentionally unavailable during verdict formation:

- historical SENTINEL model probabilities/tiers;
- Slither/Aderyn/tool votes;
- downstream merger outcome;
- non-target historical labels.

The review states are:

- `SUPPORTS_POSITIVE`
- `DOES_NOT_SUPPORT_POSITIVE`
- `UNCLEAR_INSUFFICIENT`
- `CLASS_BOUNDARY_CONFLICT`

`DOES_NOT_SUPPORT_POSITIVE` means that the sampled DIVE source assertion is not semantically supported by the reviewed source under the frozen class definition. It **does not create a confirmed negative label** for the contract/class pair.

## 2. Results

| Canonical stratum | Supports | Does not support | Unclear | Boundary conflict | Observed support | Descriptive 95% Wilson interval | Phase-4 role recommendation |
|---|---:|---:|---:|---:|---:|---:|---|
| DenialOfService | 0 | 20 | 0 | 0 | 0% | 0–16.1% | `MASK_OR_EXCLUDE` |
| IntegerUO | 3 | 16 | 1 | 0 | 15% | 5.2–36.0% | `MASK_OR_EXCLUDE` |
| Timestamp | 4 | 15 | 1 | 0 | 20% | 8.1–41.6% | `MASK_OR_EXCLUDE` |
| TransactionOrderDependence | 12 | 5 | 0 | 3 | 60% | 38.7–78.1% | `TRAIN_WEAK` only |
| UnusedReturn | 9 | 11 | 0 | 0 | 45% | 25.8–65.8% | `MASK_OR_EXCLUDE` |

The intervals are descriptive summaries of the deterministic group-aware screening sample. They are not universal promotion thresholds and should not be read as if the sample were an unconstrained simple random sample.

## 3. Semantic findings by stratum

### 3.1 DenialOfService

No reviewed source established the frozen semantic core: an untrusted attacker able to make a critical operation or state transition persistently unavailable.

Recurring false-positive mechanisms included:

- loops without an attacker-controlled forced-failure condition;
- ordinary caller-local reverts;
- external dependencies without a demonstrated persistent attacker-controlled blocker;
- transfers to fixed/trusted recipients where recipient failure was not attacker-controlled.

**Decision:** DIVE `DoS` folder membership is not acceptable as a supervised positive assertion for the first DATA vNext baseline. Preserve the contracts and source assertion as provenance/unlabeled structure, but mask the class outcome from supervised loss and outcome metrics.

### 3.2 IntegerUO

Three contracts supported the class:

- pre-0.8 attacker-controlled raw multiplication used for token allocation;
- reversed balance/allowance guards followed by subtraction, permitting underflow;
- another pre-0.8 raw crowdsale multiplication before protected accumulation.

Most sampled contracts were modern Solidity with compiler-checked arithmetic, protected pre-0.8 arithmetic, or deliberate bounded unchecked optimization. One value-critical legacy arithmetic case remained unclear because exploitability depended on reachable economic bounds.

**Decision:** DIVE `Arithmetic` is materially broader than canonical `IntegerUO`. The blanket folder assertion must be masked/excluded as a supervised positive. A future repair may recover a narrower compiler/protection-aware sub-stratum, but Phase 4 does not invent that policy.

### 3.3 Timestamp

Four contracts supported the canonical class through short or materially security-sensitive timestamp boundaries, including:

- timestamp-dependent mint availability;
- short auction/game outcome timing;
- a crowdsale opening boundary;
- a 60-second punitive sniping/bot-classification window.

Most other positives were standard router deadlines, long-duration vesting/withdrawal windows, metadata timestamps, or timing whose feasible block timestamp skew was not materially security-sensitive. One governance timelock case remained configuration-dependent.

**Decision:** DIVE `Time manipulation` is too broad for blanket supervised-positive use. Mask/exclude the source assertion for the first baseline while preserving provenance.

### 3.4 TransactionOrderDependence

This stratum behaved differently from the other four. Twelve of twenty contracts supported the frozen semantic class.

The dominant positive mechanism was the classic ERC-20 allowance replacement race: `approve` directly overwrites a nonzero allowance, allowing an existing spender to observe the replacement transaction, front-run `transferFrom`, and then potentially consume the new allowance after replacement. One ERC-721 approval/revocation ordering race and one public unique-asset market ordering race also supported the class.

Three contracts were ordering-sensitive but belonged at a semantic boundary:

- one contract's root flaw was state/access overwrite rather than transaction ordering;
- two last-player/last-depositor games intentionally made transaction ordering the advertised mechanism, so ordering sensitivity alone was insufficient to call them vulnerabilities.

**Decision:** DIVE `Front Running` contains a real and materially stronger semantic signal, but 60% support with boundary-sensitive cases is not strong/metric-grade authority. It may be retained **only as `TRAIN_WEAK` evidence** in DATA vNext, with outcome metrics masked. It is not eligible from this review for model selection, threshold fitting, calibration fitting, untouched acceptance, or strong training authority.

No second-review expansion is required at Phase 4 because no high-authority role is being proposed. If a later ADR attempts to promote this stratum beyond weak training evidence, a new approved gap/second independent review is required.

### 3.5 UnusedReturn

Nine contracts supported the class. The recurring positive mechanisms were:

- ERC-20/token `transfer` return values used as bare statements;
- low-level ETH calls whose returned success flag was discarded entirely;
- low-level calls where `success` was assigned but never inspected before execution continued.

The false-positive group included native `transfer` (which reverts and has no boolean success return), checked low-level calls, void ERC-721 superclass methods, and files with no meaningful discarded external status.

**Decision:** the blanket DIVE `Unchecked Return Values` folder remains too noisy for supervised-positive authority in the first baseline. Mask/exclude it. A future policy may recover a narrower explicit ignored-bool/ignored-call-success sub-stratum.

## 4. Reconciliation with recovered DIVE evidence

Phase 1 had already established that DIVE folder labels are not ground truth: independent manual reviews found only roughly 4–5% support for the historically reviewed ExternalBug and Reentrancy strata, while DIVE+Slither and DIVE+Slither+Aderyn agreement did not improve precision. Those tools often agreed on the same superficial patterns and therefore cannot be counted as independent truth.

The new Phase-4 blind review extends the same lesson to previously unreviewed DIVE classes while also showing that quality is **class-specific**, not uniformly bad. In particular, `Front Running` contains substantially more semantic signal than DIVE EB/RE/DoS/Arithmetic/Timestamp.

This reconciliation does not change any blind verdict. It only determines the source/stratum role recommendation after the source-only decisions were locked.

## 5. First-baseline role recommendation

For Phase 5 design, recommend:

| DIVE native class | Canonical class | Recommended first-baseline role |
|---|---|---|
| `DoS` | DenialOfService | source assertion → `UNKNOWN`/masked; contract may remain `TRAIN_UNLABELED` |
| `Arithmetic` | IntegerUO | source assertion → `UNKNOWN`/masked; contract may remain `TRAIN_UNLABELED` |
| `Time manipulation` | Timestamp | source assertion → `UNKNOWN`/masked; contract may remain `TRAIN_UNLABELED` |
| `Front Running` | TransactionOrderDependence | at most `TRAIN_WEAK`; always exclude from outcome metrics and all acceptance/calibration/model-selection roles |
| `Unchecked Return Values` | UnusedReturn | source assertion → `UNKNOWN`/masked; contract may remain `TRAIN_UNLABELED` |

These are role recommendations for the **source assertion**, not final per-contract outcome labels. Phase 5 must encode the policy in versioned ADRs/schema/weights and must not silently convert review failures into confirmed negatives.

## 6. Adaptive-review stop decision

The initial 20-per-stratum batch is sufficient for the current Phase-4 decision because:

- four strata are being demoted/masked rather than promoted;
- the remaining TOD stratum is proposed only for weak training evidence;
- no stratum is proposed for strong training, outcome metrics, calibration, threshold fitting, model selection, or acceptance;
- expanding the review would therefore not change the minimum first-baseline authority decision enough to justify additional protected-data review.

A future attempt to recover stronger sub-strata or promote a source assertion must be separately authorized and evidence-bound rather than silently extending this sample.

## 7. Limitations

- This is a single AI semantic reviewer, not a second independent human or inter-rater study.
- The sample is deterministic, leakage-group aware, TRAIN-only, and excludes groups touching validation/test; the descriptive interval does not remove selection/design limitations.
- The review establishes source-assertion reliability for Phase-4 role decisions, not global vulnerability absence.
- No historical zero is promoted to a negative.
- No current model output was used as adjudication truth.
