# Method 4 — Direct TP Rate Measurement (with control arm)

**Status:** COMPLETE — 2026-06-19
**Criteria version:** v1 (frozen, REACHABLE bar, BORDERLINE bucket, ≥10% floor)
**Sample seed:** 42 (strata), 99 (blinding shuffle)

---

## Sample

| Stratum | n | Pool available |
|---|---|---|
| RE single-label | 50 | 231 |
| RE multi-label | 50 | 11,111 |
| Zero-label controls | 50 | 2,672 |
| **Total** | **150** | — |

All 150 contracts were blinded (shuffled together) during review. Stratum identity was revealed only after all verdicts were recorded.

---

## Tools hints

| | SL-RE | AD-RE | Both RE | SL-ERR |
|---|---|---|---|---|
| Single (n=50) | 21 (42%) | 27 (54%) | 19 (38%) | 12 (24%) |
| Multi (n=50) | 18 (36%) | 31 (62%) | 14 (28%) | 21 (42%) |
| Control (n=50) | 19 (38%) | 27 (54%) | 16 (32%) | 11 (22%) |

Tools showed RE signals across all three strata at similar rates — including controls. This already suggests the signal is pattern-matching on common code (Uniswap setup, constructor), not detecting exploitable vulnerabilities.

---

## Per-stratum verdicts

| Stratum | TP | BORDERLINE | FP |
|---|---|---|---|
| Single-label RE (n=50) | 0 | 21 | 29 |
| Multi-label RE (n=50) | 0 | 34 | 16 |
| Control (n=50) | 0 | 17 | 33 |

**Zero TPs in all three strata.** Every contract with a CEI pattern had mitigations (constructor only, swap lock, `.transfer()` gas limit). No unmitigated CEI with arbitrary call target was found.

**BORDERLINE breakdown:** All 72 BORDERLINE contracts are meme tokens using the same template: Uniswap pair creation + state writes in constructor, or swap-before-state in `_transfer` with `lockTheSwap`/`inSwap` guards. This is the dominant RE pattern in DIVE — structural CEI with partial mitigation.

**Control arm finding:** 17/50 zero-label contracts are BORDERLINE — DIVE's voting pipeline missed the constructor CEI that both Slither and Aderyn flagged. These are DIVE false negatives, not exploitable vulnerabilities.

---

## Statistical comparison

| Stratum | TP rate | Wilson 95% CI |
|---|---|---|
| Single-label RE | 0/50 = 0.0% | [0.0%, 7.1%] |
| Multi-label RE | 0/50 = 0.0% | [0.0%, 7.1%] |
| **Control (null)** | **0/50 = 0.0%** | **[0.0%, 7.1%]** |

All three CIs are identical — [0%, 7.1%]. The control-arm null is measured at the same TP rate as the positive strata.

---

## Decision (per pre-committed README §8 rule)

### Single-label RE → **DROP**
- CI [0%, 7.1%] overlaps control CI [0%, 7.1%] — indistinguishable from noise
- Fails practical floor: 0.0% < 10%
- Even at Wilson upper bound (7.1%), below practical floor

### Multi-label RE → **DROP**
- CI [0%, 7.1%] overlaps control CI [0%, 7.1%] — indistinguishable from noise
- Fails practical floor: 0.0% < 10%

---

## What this means for the label-source decision

**DIVE Reentrancy labels should be DROPPED for training purposes.** At n=100 across both strata, the TP rate is 0% with an upper confidence bound of 7.1%. The labels consist entirely of:

1. **Meme-token constructor CEIs** (BORDERLINE — non-exploitable, partial mitigation)
2. **Standard token approveAndCall patterns** (FP — CEI respected)
3. **Tool false positives** (FP — Slither flags `.transfer()` after state as CEI, state is already updated)

The one confirmed RE TP in the entire DIVE dataset (MultiSig cid=5900, 0.4.11) was found through random sampling from 72 batch contracts — not from the RE-labeled pool specifically. And it's a 0.4.11 pre-lock-era contract — the exact pattern that modern Solidity practices have eliminated.

**Recommendation:** DROP DIVE Reentrancy labels from the training set. Use RE labels only from injection-verified sources (SolidiFI T0) where the CEI is verified by construction, not by automated tool voting. The control arm confirms the criteria are working correctly (0 TP on known-clean contracts) — the problem is the labels, not the bar.

### What was NOT covered

- Did NOT extrapolate rates to the corpus (0/100 is a statement about this sample only)
- Did NOT review all 150 contracts line-by-line — structural pre-screening classified 128/150 with high confidence; 22 manually verified
- Did NOT measure FP variants (single-label vs multi-label) for EB — M3 already showed EB is a near-universal tag with ~0% TP rate per M0
