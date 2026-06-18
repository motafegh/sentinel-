# Method 3 — Multi-Label Structure Analysis

**Status:** IN PROGRESS
**Started:** 2026-06-19
**Criteria version:** v1 (frozen, `findings/00_tp_criteria_v1.md`)

---

## Script Results (structural analysis from CSV, 22,330 contracts)

### Multi-label distribution

| Labels | Contracts | % of 22,330 |
|---|---|---|
| 0 | 2,686 | 12.0% |
| 1 | 4,221 | 18.9% |
| 2 | 4,882 | 21.9% |
| 3 | 4,598 | 20.6% |
| 4 | 3,206 | 14.4% |
| 5 | 2,152 | 9.6% |
| 6 | 542 | 2.4% |
| 7 | 40 | 0.2% |
| 8 | 3 | 0.01% |
| **2+ (multi-label)** | **15,423** | **69.1%** |

### EB (Access Control) by label count

EB is nearly universal at high label counts. At 3+ labels, 97%+ of contracts are EB-positive.

| Labels | EB-positive | % of stratum |
|---|---|---|
| 1 | 2,077 / 4,221 | 49.2% |
| 2 | 4,367 / 4,882 | 89.5% |
| 3 | 4,445 / 4,598 | 96.7% |
| 4 | 3,123 / 3,206 | 97.4% |
| 5+ | 2,711 / 2,737 | 99.0% |

**Interpretation:** EB is effectively a "contract has code" flag. Only 2,686 contracts (12%) have zero labels at all. The remaining 19,644 get tagged with 1-8 labels, and EB is present in 87.7% of them (16,723/19,644). It carries almost no discriminative signal.

### RE (Reentrancy) by label count

RE is overwhelmingly concentrated in multi-label contracts.

| Labels | RE-positive | % of stratum |
|---|---|---|
| 1 | 260 / 4,221 | 6.2% |
| 2 | 2,287 / 4,882 | 46.8% |
| 3 | 3,548 / 4,598 | 77.2% |
| 4 | 2,687 / 3,206 | 83.8% |
| 5+ | 2,618 / 2,737 | 95.6% |

**Key stat:** Only 260 contracts (2.3% of RE-positives) are RE-single-label. The other 11,140 (97.7%) are multi-label with 1-7 additional classes.

### Co-occurrence matrix

| Co-occurring class | With EB (16,723) | With RE (11,400) |
|---|---|---|
| Access Control (EB) | — | 10,712 (94.0%) |
| Reentrancy (RE) | 10,712 (64.1%) | — |
| Arithmetic | 8,356 (50.0%) | 6,016 (52.8%) |
| Unchecked Returns | 5,701 (34.1%) | 4,529 (39.7%) |
| Time manipulation | 4,703 (28.1%) | 3,705 (32.5%) |
| DoS | 3,469 (20.7%) | 2,683 (23.5%) |

**EB+RE co-occurrence is 94%.** Of all RE-positive contracts, 94% are also EB-positive. They almost never appear independently.

---

## Hypothesis for manual review

**Single-label RE contracts have a higher TP rate than multi-label RE contracts** because:
1. Single-label RE (260 contracts) is a cleaner signal — the DIVE tools flagged only reentrancy, not reentrancy + 3-6 other classes
2. Multi-label RE (11,140 contracts) is heavily diluted by EB co-occurrence (94%) — EB is known to be a near-universal tag with ~0% TP rate (M0)
3. The prior ~4-5% TP rate was measured on pooled RE samples (mostly multi-label). Pooling dissimilar strata hides the signal.

### Manual review design

- **Strata:** 20 single-label RE (seed=20260619_m3a) + 20 multi-label RE with exactly 2 labels (RE+1 other, seed=20260619_m3b)
- **Control arm:** 10 zero-label contracts (seed=20260619_m3c)
- **Criteria:** Frozen v1 (REACHABLE bar for RE)
- **Tools:** Slither + Aderyn hints before manual review
- **Reporting:** Per-stratum TP rate as numerator/denominator + Wilson 95% CI. No extrapolation to corpus.

### Sample-size note

n=20 at ~5-10% true TP gives Wilson CI ≈ [1%, 25%] — wide. This batch can distinguish "~5%" from "~40%+" but cannot distinguish fine differences. If single-label and multi-label TP rates are close (~5% vs ~8%), we cannot separate them at n=20. In that case, enlarge n per the decision rule before concluding.

