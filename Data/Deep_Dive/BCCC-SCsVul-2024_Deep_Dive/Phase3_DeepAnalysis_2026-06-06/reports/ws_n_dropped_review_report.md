# WS-N: Dropped + Review-Pending Deep-Dive Report

**Date:** 2026-06-06  
**Source data:** SENTINEL v9 cleaned v1.0 + BCCC raw  

## 1. Review-Pending (NV+vuln contradiction, n=766)

These 766 contracts are labeled BOTH NonVulnerable AND at least one vuln class in BCCC. They are held out from training and require manual review before re-inclusion.

### 1.1 Per primary class

| Primary class | n | % |
|---|---:|---:|
| CallToUnknown | 703 | 91.78% |
| IntegerUO | 41 | 5.35% |
| Timestamp | 14 | 1.83% |
| GasException | 5 | 0.65% |
| DenialOfService | 2 | 0.26% |
| ExternalBug | 1 | 0.13% |

**Key finding:** 703/766 (92%) of review-pending have **CallToUnknown (Class08) as primary class**.

### 1.2 n_pos distribution (multi-label cardinality)

| n_pos | n | % |
|---:|---:|---:|
| 2 | 55 | 7.18% |
| 3 | 705 | 92.04% |
| 4 | 1 | 0.13% |
| 5 | 5 | 0.65% |

**Key finding:** 705/766 (92%) of review-pending have **n_pos=3** (i.e., 3 positive classes: typically NV + 2 vuln classes).

### 1.3 Top 10 vuln-class combinations (NV excluded)

| Count | Vuln set |
|---:|---|
| 703 | CallToUnknown, Reentrancy |
| 41 | IntegerUO |
| 14 | Timestamp |
| 4 | GasException, MishandledException, DenialOfService, Reentrancy |
| 2 | DenialOfService, Reentrancy |
| 1 | GasException, DenialOfService, Reentrancy |
| 1 | ExternalBug, MishandledException, IntegerUO, Reentrancy |

**Key finding:** The most common triple-label (NV, Reentrancy, CallToUnknown) = 703 contracts. This is suspicious — likely a templating artifact in BCCC where many reentrancy contracts share a common pattern that also gets labeled 'non-vulnerable' in some other context.

### 1.4 Source code path availability

- Review-pending with valid source path: 766/766
- Missing: 0

### 1.5 Recovery recommendation (WS-N → manual review)

- These 766 contracts SHOULD be reviewed before adding to training set.
- Given the homogeneity of the NV+Reentrancy+CallToUnknown triple (92% of review-pending), a single rule might resolve most:
  - **Hypothesis:** Contracts in the Reentrancy folder that also have a `nonReentrant` modifier OR a `ReentrancyGuard` import are NOT actually reentrant; the 'NV' label is correct, the 'Reentrancy' label is wrong.
- Manual review budget for this set: ~3-5 minutes per contract × 766 = **40-60 hours** (use 846 default = 766 + 50 + 30 if you also do multi-positive and disagreement sets).
- **Alternative (faster):** Just use these 766 contracts as **noise data for adversarial training** in Phase 4 — train SENTINEL to predict the majority class, ignore the contradictions.

## 2. Dropped contracts (Class05/Class07 only, n=1,122)

These 1,122 contracts are labeled ONLY with BCCC's Class05 (TransactionOrderDependence) and/or Class07 (WeakAccessMod). Neither class has a SENTINEL v9 equivalent, so they were dropped in Phase 2 (D-F1).

### 2.1 Class composition

| Subset | n | % |
|---|---:|---:|
| Class05 only | 163 | 14.53% |
| Class07 only | 959 | 85.47% |
| Class05 + Class07 | 0 | 0.00% |

**Key finding:** 959/1,122 (85.5%) of dropped are **Class07 (WeakAccessMod) only**. The 'WeakAccessMod' class is mostly state-visibility issues (public functions that should be private/external). SENTINEL v9 has no equivalent because:
- Public vs external matters less for vulnerability detection (the issue is in CALLERS, not callees).
- These are more 'code quality' than 'security' issues per D-F1 decision.

### 2.2 Source code folder mapping

| BCCC source folder | n .sol files | n dropped IDs mapped |
|---|---:|---:|
| TransactionOrderDependence/ | 3562 | 163 |
| WeakAccessMod/ | 1918 | 959 |
| (no folder match) | — | 0 |

### 2.3 Recovery recommendation

- To recover these 1,122 contracts, SENTINEL v9 would need 2 new classes:
  - `Class05:TransactionOrderDependence` (SWC-114): e.g., TOCTOU bugs in ERC20 approve/transfer pattern.
  - `Class07:WeakAccessMod` (SWC-100 / SWC-105): state visibility issues.
- **Cost:** 2 new classes × ~10 hours labeling review × 1,122 contracts = **~22 hours** (or 1.2 hours at 30s per contract for keyword-based auto-labeling).
- **Value:** Brings SENTINEL coverage from 67,311/68,433 (98.4%) to 100% of BCCC.
- **Recommended for Phase 4** (NOT Phase 3 — out of scope per plan §10).

## 3. Source code samples (qualitative review)

### 3.1 Review-pending samples (5 contracts)

Saved to `outputs/ws_n_review_pending_samples/` (5 files, 5 copied).
Manual review needed: read each .sol and decide which label is correct.

### 3.2 Dropped samples (5 + 5 = 10 contracts)

Saved to `outputs/ws_n_dropped_samples/` (10 files, 10 copied).
Manual review needed: confirm these are indeed Class05/Class07 issues (no SENTINEL class fits).

## 4. Key takeaways for Phase 3 downstream

1. **WS-I (slither label validation):** For the 846 manual review contracts, prioritize the 766 review-pending first (highest value, highest noise).
2. **WS-M (BCCC 242-feature test):** Dropped contracts will not appear in v1.0 dataset; if v1.2 includes them, need to add 2 new SENTINEL classes (out of scope for Phase 3).
3. **WS-L (AutoML):** Use class_weight='balanced' to handle Reentrancy vs ExternalBug imbalance (4.9×).
4. **WS-T (multi-label structure):** n_pos=1 is 60.6% — AutoML can use a 'class chain' decomposition (predict n_pos first, then which classes).
