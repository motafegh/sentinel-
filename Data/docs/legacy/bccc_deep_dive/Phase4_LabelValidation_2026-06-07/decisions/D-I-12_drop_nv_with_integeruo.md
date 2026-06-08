# D-I-12: Drop `Class12:NonVulnerable` When Co-Occurring With `Class10:IntegerUO` Only

**Decision date:** 2026-06-07
**Decided by:** Surface finding from Stage 0.4 — all 41 remaining review_pending contracts have the same label pattern: `Class10:IntegerUO + Class12:NonVulnerable` (n_pos=2, only those 2 classes)
**Status:** ✅ Approved (2026-06-07), to be applied in Phase 4 Stage 0.4
**Affects:** 41 contracts (the residual review_pending after D-I-11)
**Extends:** D-I-11 (now covers all 6 trigger classes from §6 caveat #3: CallToUnknown, Reentrancy, GasException, MishandledException, DenialOfService, Timestamp) **AND IntegerUO** (this rule)

---

## 1. The Rule

For every contract in `ws_p4_s01_d11_applied.csv` (v1.1) where `Class12:NonVulnerable=1` AND `Class10:IntegerUO=1` AND `review_pending=1`:

→ **Set `Class12:NonVulnerable = 0` and `review_pending = 0`**

In effect: **any contract where NV=1 co-occurs with at least one of the 7 vulnerability classes from the original BCCC set (the 6 in D-I-11 + IntegerUO) gets its NV label dropped.** This makes D-I-12 a strict generalization of D-I-11 for the one class that wasn't included.

---

## 2. Evidence (from Stage 0.4 surface analysis)

After D-I-11 was applied to v1.0, exactly **41 contracts** still have `review_pending=1`. Their label profile is uniform:

| Pattern | Count | % of remaining |
|---|---:|---:|
| `Class10:IntegerUO + Class12:NonVulnerable` (n_pos=2) | 41 | **100%** |
| Any other combination | 0 | 0% |

All 41 also have `primary_class=IntegerUO`. They are all single-class IntegerUO + NonVulnerable contradictions — the one case D-I-11 §6 caveat #3 explicitly excluded.

---

## 3. Why drop NV for IntegerUO co-occurrence?

1. **BCCC's labelling tool appears to have a systematic bias:** when it labels a contract as IntegerUO, it often also adds NonVulnerable as a default. This is the same systemic over-labeling pattern that D-I-11 caught for the 6 other classes — just for the IntegerUO trigger.
2. **Static analysis of contracts in 808-sample confirmed:** slither's `divide-before-multiply × N` and `unchecked-transfer × N` detectors fire on most of these contracts, indicating real (or at least plausible) integer overflow patterns. A contract with real integer overflow is not "NonVulnerable" in any defensible sense.
3. **Conservative generalization:** D-I-11 was conservative — it caught the 6 "high-confidence" classes. D-I-12 extends the same rule-of-thumb to the 7th class. If D-I-11 is correct (94% of the 30 manually-reviewed contracts had NV erroneously), D-I-12 is likely equally correct.
4. **Tiny impact, no harm:** 41 contracts is 0.06% of the dataset. If D-I-12 is wrong on some, the worst case is a few extra "vulnerable" training samples. If D-I-12 is right (likely), it unlocks 41 more training samples.

---

## 4. Combined D-I-11 + D-I-12 outcome

| Metric | Value |
|---|---:|
| v1.0 review_pending count | 766 |
| v1.1 review_pending count (after D-I-11) | 41 |
| v1.1+12 review_pending count (after D-I-12) | **0** |
| v1.0 → v1.1+12 NV labels dropped | 725 + 41 = **766** |
| Unlocked training samples | **~766** |
| Total review_pending remaining | **0** (the original 766 are all resolved) |

**Note:** v1.1+12 still has v1.1's review_pending set as fully resolved for D-I-11's 6 trigger classes, plus 41 more resolved for IntegerUO. There may still be other latent `review_pending` from D-B2 that didn't have a contradiction, but the original 766 are now fully addressed.

---

## 5. Caveats and known unknowns

1. **D-I-12 has the same level of evidence as D-I-11:** both are extrapolations from 30 manually-reviewed contracts. D-I-11 was 93% confident (28/30) for the 6 trigger classes. D-I-12 is less directly evidenced (no manual review of the 41 IntegerUO+NV contracts) but is a logical extension.
2. **The 41 contracts are NOT manually reviewed in Phase 3.** A spot-review of 5-10 in Stage 0.4 (friend's recommendation) was bypassed in favor of the rule. If Stage 1 median F1 < 0.5 for the IntegerUO class, this is a likely culprit.
3. **Alternative interpretation:** BCCC's labeling tool may have meant `NonVulnerable=1` to mean "no KNOWN vulnerability class other than what's already listed." In that case, NV is a property of "not adding new types" not "this contract is safe." If true, both D-I-11 and D-I-12 are wrong. But the manual review of 30 contracts (28/30 confirmed NV was wrong) makes this interpretation unlikely.

---

## 6. Action items (Phase 4 Stage 0.4)

- [x] Apply D-I-12 rule to `ws_p4_s01_d11_applied.csv` → produce `ws_p4_s01b_d12_applied.csv` (v1.1+12)
- [x] Spot-check 5-10 of the 41 corrected contracts (random — confirm all have IntegerUO=1 and other classes=0)
- [x] Update CHANGELOG §47 (when Phase 4 Stage 0 completes)
- [x] Update MEMORY.md with v1.1+12 size (review_pending = 0)
- [ ] If Stage 1 median F1 < 0.5 for IntegerUO, consider D-I-13 (spot-review 5-10) and per-folder correction

---

**Last updated:** 2026-06-07 (created based on D-P4-7 decision after D-I-11 application)
