# D-I-11: Drop `Class12:NonVulnerable` When Co-Occurring With Any Vulnerability Class

**Decision date:** 2026-06-07
**Decided by:** Manual review of 32 contracts in `Phase3_DeepAnalysis_2026-06-06/labels/ws_i_disagreement_inspections.md` (425 lines)
**Status:** ✅ Approved (2026-06-07), to be applied in Phase 4 Stage 0
**Affects:** All 67,311 contracts in `contracts_clean.csv` (Phase 2 v1.0)
**Trigger:** `Class12:NonVulnerable=1` AND any of {CallToUnknown, Reentrancy, GasException, MishandledException, DenialOfService, Timestamp}=1

---

## 1. The Rule

For every contract in `contracts_clean.csv` where `Class12:NonVulnerable=1` AND at least one of the following is also 1:
- `Class08:CallToUnknown`
- `Class11:Reentrancy`
- `Class02:GasException`
- `Class03:MishandledException`
- `Class09:DenialOfService`
- `Class04:Timestamp`

→ **Set `Class12:NonVulnerable = 0`**

After applying, also set `review_pending = 0` (the contract is no longer contradictory, it can be used for training).

---

## 2. Evidence (from manual review of 32 contracts)

| Bucket | Contracts | NonVulnerable + at least one vulnerability | Decision |
|---|---:|---:|---|
| review_pending (95% of sample) | 30 | 28/30 = **93%** | 28 MODIFY (drop NV), 2 MODIFY (reclassify entire contract) |
| nine_folder_maxing | 2 | 2/2 = **100%** | 2 MODIFY (reclassify entire contract) |
| **Total** | **32** | **30/32 = 94%** | — |

**Confidence:** Very High. Across 32 contracts, 28 (87.5%) had NonVulnerable co-occurring with confirmed vulnerabilities (slither reentrancy-eth × 4 in Oraclize contracts, ETH reentrancy in NamiPool/POOHMOX, missing-zero-check in DocSignature, etc.). The 2 remaining (HongZhangCoin, DocSignature) were reclassified entirely because their other 7-8 labels were also wrong (over-labeled templated contracts).

---

## 3. The 6 Template Clusters Identified

| Cluster | Contracts | BCCC labels | True labels (per manual review) |
|---|---:|---|---|
| **Oraclize API v1** | 16 | CallToUnknown, Reentrancy, **NonVulnerable** | CallToUnknown, Reentrancy |
| **Oraclize API v2** | 1 | CallToUnknown, Reentrancy, **NonVulnerable** | CallToUnknown, Reentrancy |
| **CentraSale ICO** | 3 | CallToUnknown, Reentrancy, **NonVulnerable** | CallToUnknown (Reentrancy marginal) |
| **NutzToken proxy** | 2 | CallToUnknown, Reentrancy, **NonVulnerable** | CallToUnknown, Reentrancy |
| **NamiPool DeFi** | 1 | CallToUnknown, Reentrancy, **NonVulnerable** | CallToUnknown, Reentrancy, Timestamp |
| **POOHMOX gambling** | 1 | CallToUnknown, Reentrancy, **NonVulnerable** | CallToUnknown, Reentrancy, Timestamp, GasException |
| **DEVCoin crowdsale** | 1 | CallToUnknown, Reentrancy, **NonVulnerable** | CallToUnknown (weak), Timestamp |
| **InkProtocol escrow** | 1 | CallToUnknown, Reentrancy, **NonVulnerable** | CallToUnknown, Reentrancy |
| **Peculium airdrop** | 1 | CallToUnknown, Reentrancy, **NonVulnerable** | CallToUnknown, Reentrancy, MishandledException |
| **ERC20 ICO crowdsale** | 1 | GasException, MishandledException, DenialOfService, Reentrancy, **NonVulnerable** | GasException, MishandledException, DenialOfService, Reentrancy |
| **SafeMath ERC20** (maxing) | 1 | 8 labels (all wrong) | NonVulnerable only |
| **Document Registry** (maxing) | 1 | 8 labels (7 wrong) | GasException only (unbounded array) |

---

## 4. Extrapolation to the 766 review_pending set

Per WS-N (Phase 3): **92% of the 766 review_pending contracts are NV + exactly {CallToUnknown, Reentrancy}**.

**Expected unlocks from D-I-11:**
- 766 review_pending × 92% = **~705 contracts** get NonVulnerable dropped and become training-eligible
- Estimated v1.1 training set: 67,311 + 705 = **~68,016 contracts** (vs 67,311 in v1.0)
- Net positive label rate should improve significantly (fewer contradictory NV=1)

**Sanity check (Stage 0.5):** Spot-check 10 random corrected contracts. All should still have at least one positive vulnerability class. If any becomes label-less (no positive class, no NV), it's an edge case — flag for manual review.

---

## 5. Should the rule apply to non-review_pending contracts?

The 67,311 contracts in v1.0 are split:
- 766 review_pending (held out for manual review)
- 66,545 non-review_pending (in training)

D-I-11 was derived from review_pending contracts (the ones BCCC flagged as contradictory). The same NonVulnerable error could exist in non-review_pending contracts — for example, the Oraclize cluster likely has hundreds of near-identical contracts scattered through training, not just in the 30 reviewed.

**D-P4-1 (TBD in Phase 4):** Apply D-I-11 broadly (all 67,311) or narrowly (only review_pending)?
- **Default:** Apply narrowly (review_pending only) to be conservative
- **Override:** If Stage 1.5 generalization check shows the rule applies to non-review_pending (e.g., 10%+ of non-review_pending contracts have the same NV+Reentrancy co-occurrence), apply broadly

---

## 6. Caveats and known unknowns

1. **D-I-11 is a rule-of-thumb, not a proof.** It's based on 32 contracts and extrapolation. Phase 4 Stage 1 (15% per folder sample) will validate or refute this on 1,400 contracts.
2. **The 2 nine_folder_maxing contracts (HongZhangCoin, DocSignature) need separate handling.** D-I-11 doesn't apply to them — they have 8 labels, none of which co-occurs with the standard 6 classes (because all 8 are positive). They need a different rule: "if n_pos ≥ 7, re-inspect for over-labeling."
3. **Class06:UnusedReturn and Class01:ExternalBug are not in the trigger list.** Reviewing the 30 contracts, NV did not co-occur with ONLY UnusedReturn or ONLY ExternalBug. If a contract has NV + UnusedReturn (and nothing else), D-I-11 won't fire. This is a conservative choice.
4. **The 5 vulnerability classes in the trigger list (CallToUnknown, Reentrancy, GasException, MishandledException, DenialOfService, Timestamp) are the ones confirmed by manual review as "NonVulnerable co-occurrence is clearly wrong."** IntegerUO and ExternalBug weren't in the trigger list because they didn't co-occur with NV in the 30 reviewed. Class06:UnusedReturn and Class12:NonVulnerable are the same class.

---

## 7. Related decisions

- **D-F1** (Phase 2): Drop Class05/07 — 1,122 contracts dropped, 10 SENTINEL classes
- **D-B2** (Phase 2): Hold out 766 NV+vuln contradictions as `review_pending=1`
- **D-I-1 to D-I-10** (Phase 3): Slither harness design + Stage 2 fixes
- **D-I-11** (Phase 3, this doc): Drop NV label when co-occurring with a vulnerability class
- **D-P4-1 to D-P4-6** (Phase 4, TBD): Decisions to make at the start of each Phase 4 session

---

## 8. Action items (Phase 4 Stage 0)

- [ ] Apply D-I-11 rule to `contracts_clean.csv` → produce `contracts_clean_v11.csv`
- [ ] Spot-check 10 random corrected contracts (all should have ≥1 positive vulnerability class)
- [ ] Update `split_assignments.csv` to include the ~705 newly-unlocked contracts
- [ ] Document in CHANGELOG §47 (when Phase 4 Stage 0 completes)
- [ ] Update MEMORY.md with v1.1 size
- [ ] If D-P4-1 → apply broadly (not just review_pending), re-run spot-check

---

**Last updated:** 2026-06-07 (created based on manual review of 32 contracts)
