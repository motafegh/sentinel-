# ADR-0002: Multi-label formulation (10-class multi-hot)

**Status:** Accepted
**Date:** 2026-06-06
**Deciders:** Ali Motafegh, Claude (assistant)
**Supersedes:** None
**Superseded by:** None (current)

## Context

SENTINEL classifies Solidity smart contracts for 10 vulnerability categories:
Reentrancy, Timestamp, IntegerUO, MishandledException, DenialOfService, ExternalBug,
UnusedReturn, GasException, CallToUnknown, and one implicit `safe` class (no label).

In the source dataset (BCCC-SCsVul-2024), labels are *not* mutually exclusive. A single
contract can have multiple true vulnerabilities — for example, a reentrant function that
uses `block.timestamp` for randomness has **both** Reentrancy and Timestamp labels. The
distribution of multi-labelled contracts is significant: the BCCC CSV has rows where
two, three, or more columns are simultaneously `1`.

Two co-occurrence facts drove the design:

1. **DenialOfService ↔ Reentrancy co-occurrence is 98.6%.** Of 12,395 contracts labeled
   DoS, 12,381 are also labeled Reentrancy (BCCC tags them as a paired pattern — gas-limit
   DoS via failed `send()` in a fallback). Splitting these into separate binary heads
   while merging them via a multi-hot output is essential, otherwise the model collapses
   to a single signal.
2. **Label space is dense at the high end.** ~60% of contracts have ≥2 positives; the
   mean positive count is 1.7 per contract. A `safe` single-class would dominate any
   single-label formulation.

Three reasonable formulations exist:

- **Single-label** (10-way softmax) — pick the one most likely vulnerability.
- **Multi-class** (4-way softmax over merged groupings like {Reentrancy,DoS},
  {IntegerUO}, {Timestamp}, {Other}) — collapses co-occurring classes.
- **Multi-label** (10-class multi-hot with independent sigmoids) — predict each class
  independently.

The formulation also dictates the loss function and the calibration strategy. Each
choice is downstream of this decision.

## Decision

We use a **10-class multi-hot multi-label** formulation:

- Each contract has a 10-dim binary label vector (one entry per vulnerability class).
- The model output is 10 independent sigmoid logits (no softmax across classes).
- The "no vulnerability" case is implicit: a contract with all-zero labels is `safe`.
- The loss is `AsymmetricLoss` (per-class sigmoid, see ADR-0006) — not cross-entropy
  over a softmax distribution.
- Per-class tuned thresholds (`ml/calibration/temperatures_runN.json`) replace the
  single "argmax" decision rule.

The 10 classes and their BCCC-derived mapping:

| Index | Class | BCCC tag (column) |
|-------|-------|-------------------|
| 0 | Reentrancy | `reentrancy` |
| 1 | Timestamp | `timestamp` |
| 2 | IntegerUO | `integer-overflow` |
| 3 | MishandledException | `unchecked-{send,transfer}` |
| 4 | DenialOfService | `dos-{gas-limit,uniswap}` |
| 5 | ExternalBug | `arbitrary-send-eth`, `controlled-delegatecall` |
| 6 | UnusedReturn | `unchecked-{return,lowlevel}` |
| 7 | GasException | `gas-exception` (BCCC-specific tag) |
| 8 | CallToUnknown | `low-level-calls` (BCCC-specific) |
| 9 | (reserved) | unused — see ADR followup |

The class list is **stable** across schema versions v4–v9. The mapping from BCCC
columns to class indices is `ml/scripts/build_multilabel_index.py`. Disambiguation
rules (e.g. `unchecked-send` → MishandledException, not UnusedReturn) live in that
script and in the audit doc `docs/pre-run9-fixes/05-slither-derived-labels.md`.

## Consequences

### Positive

- **Honest representation of multi-vulnerability contracts.** A contract that has both
  Reentrancy and Timestamp is reported as both, not as whichever is most likely.
- **DoS/Reentrancy separation is possible.** The model can learn "this is mostly
  Reentrancy with a DoS side-effect" rather than being forced to pick one.
- **Loss function is well-understood.** Per-class sigmoid + ASL is a standard
  multi-label formulation. Plenty of literature and tooling support it.
- **Tuning is per-class.** We can set `IntegerUO threshold=0.50` and
  `DoS threshold=0.45` independently. Run 7's calibration file
  (`ml/calibration/temperatures_run7.json`) does exactly this.

### Negative

- **No implicit calibration across classes.** A softmax would force "if Reentrancy is
  high, DoS is low" (competitive). Multi-label allows both to be high simultaneously,
  which is correct but requires per-class threshold tuning.
- **The `safe` class is implicit.** We cannot directly model "this is a safe contract"
  as a 10th class because the BCCC dataset has no `safe` column — safe contracts are
  those with all-zero labels. This makes "absence of evidence" indistinguishable from
  "evidence of absence" in the labels.
- **Class 9 is reserved.** We leave index 9 empty for future expansion (e.g. a
  ZK-circuit-specific vulnerability class from the M2 module). The classifier
  head's output is `[B, 10]` even if only 9 columns are populated. This wastes 1
  logit per forward pass but avoids re-bumping the schema if/when class 9 is filled.
- **DoS/Reentrancy disentanglement is hard.** Even with separate sigmoids, the
  per-class F1 on DoS remains low (0.15–0.30 across runs). The 98.6% co-occurrence
  means DoS positives have almost no DoS-only examples to learn from.

### Neutral

- The output is `[B, 10]` per batch, not `[B]`. The downstream predictor code must
  threshold per-class, not argmax.

## Alternatives Considered

**1. Single-label (10-way softmax).**

Rejected. Forces the model to pick *one* vulnerability per contract, which loses the
multi-label information that BCCC provides. The Reentrancy+DoS co-occurrence would be
collapsed into whichever class the softmax prefers, hiding the fact that ~12K contracts
have both.

**2. Multi-class with merged groupings (4-way softmax: {Reent,DoS}, {IntegerUO},
{Timestamp}, {Other}).**

Rejected during v4 design. Collapses too much signal. A 4-class softmax would treat
{Reentrancy, DoS} as a single class, which is acceptable but loses the per-class
probability we need for the agents module's 3-tier output (CONFIRMED / SUSPICIOUS /
NOTEWORTHY) — that tier system requires independent per-class scores.

**3. Hierarchical classifier (root: vulnerable? → leaf: which class?).**

Considered in v5 design. Rejected because: (a) the root classifier
"vulnerable or not" requires defining what counts as "vulnerable" when a contract has
both Reentrancy and DoS, (b) the architecture adds a 2-stage inference path that
complicates the ZK circuit later, and (c) the per-class F1 we get from multi-label is
already sufficient for downstream ranking.

**4. Binary-relevance (one binary head per class, no shared trunk).**

Considered for the interpretability angle. Rejected: a shared trunk exploits label
correlations better, and the agents module's "find evidence for THIS class" output
needs shared features.

## References

- Source: `ml/src/datasets/dual_path_dataset.py:90-403` (label loading, `NUM_CLASSES=10`)
- Source: `ml/src/training/losses.py:49-126` (`AsymmetricLoss` — multi-label)
- Source: `ml/src/inference/predictor.py:660-757` (per-class thresholding)
- Audit: `project_data_pipeline_audit.md` (co-occurrence table, 60% ≥2 positives)
- Audit: `docs/pre-run9-fixes/00-overview.md` (Finding C: 9/10 precision degenerate,
  Finding J: safe contracts fire 0.30-0.45)
- Calibration: `ml/calibration/temperatures_run7.json` (per-class tuned thresholds)
- Related: ADR-0006 (loss formulation, depends on multi-label)
- Related: ADR-0005 (dataset choice, defines what the 10 classes map to)
