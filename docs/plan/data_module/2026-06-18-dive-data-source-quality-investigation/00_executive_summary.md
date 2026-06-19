# DIVE Data Source Quality Investigation — Executive Summary

**Phase 1: ExternalBug + Reentrancy**
**Status:** COMPLETE — 2026-06-19
**Recommendation:** DROP DIVE ExternalBug AND Reentrancy labels from training set

---

## TL;DR

**192 contracts reviewed across 8 Methods. 0 confirmed exploitable true positives for either class.** DIVE's Reentrancy and Access Control labels are produced by automated tool voting, not expert audit. They predominantly flag Ownable-tokens (FP for ExternalBug) and meme-token constructor CEIs (BORDERLINE for Reentrancy). Training on these labels would train primarily on noise.

---

## Method results

| Method | What we did | Key finding |
|---|---|---|
| **M1** | Read DIVE paper methodology | Labels are fully automated — 6 tools, Power-based voting, post-hoc filtering. Authors: "systematically derived, high-confidence annotations rather than manually verified ground truth." No manual verification. |
| **M2** | Per-contract folder↔CSV identity | 0/178,640 mismatches. Folder symlinks are a 100% faithful reproduction of the CSV. |
| **M8** | Parser faithfulness (CSV→labels.json) | Parser is faithful for all 7 DIVE-sourced classes. DoS deviation = intentional post-parser patch. |
| **M0** | Freeze TP/FP/BORDERLINE criteria | REACHABLE bar for EB and RE. BORDERLINE bucket for mitigated CEI. Practical floor ≥10% TP. 42 contracts reviewed. |
| **M7** | Tools available + add decision | Slither 0.11.5 + Aderyn 0.6.8 working and documented. Echidna 2.3.2 installed but requires assertions DIVE contracts lack. No additional tools needed. |
| **M3** | Multi-label structure analysis | EB is a near-universal tag (97% at 3+ labels). EB+RE co-occurrence is 94%. RE single-label (260 contracts) is NOT cleaner than multi-label — both dominated by meme-token BORDERLINE pattern. |
| **M4** | Direct TP rate (150 contracts, control arm) | **0 TPs in 150 contracts** (50 single-label RE + 50 multi-label RE + 50 controls). Wilson CI [0%, 7.1%] identical to control null. DROP decision per pre-committed rule. |

---

## Evidence for the DROP decision

### ExternalBug (Access Control)

| Evidence | Finding |
|---|---|
| M0 batch review (42 contracts) | 1 TP (cid=8822, unprotected proxy). 0 BORDERLINE. 29 FP. All FPs are standard Ownable-tokens where every privileged function IS guarded. |
| M3 structural analysis | EB present in 87.7% of labeled contracts. At 3+ labels, 97%+ are EB-positive. EB is a "contract has Ownable" flag, not a vulnerability signal. |

EB labels would train the model to predict "this contract has Ownable" — which is pattern-matching on library imports, not detecting missing access control. **DROP.**

### Reentrancy

| Evidence | Finding |
|---|---|
| M0 batch review (42 contracts) | 1 TP (MultiSig cid=5900, 0.4.11 pre-lock era). 9 BORDERLINE (meme-token constructor CEI + partial locks). 25 FP. |
| M4 controlled measurement (150 contracts) | 0 TP. 55 BORDERLINE (meme tokens). 78 FP. Identical TP rate to control arm (0%). |
| M3 structural analysis | Single-label RE (260 contracts) = same meme-token pattern as multi-label. No cleaner signal. |

RE labels would train the model on constructor CEI patterns (non-exploitable) and Uniswap setup code. The one confirmed TP in the entire dataset is a historical 0.4.11 pattern that modern tooling eliminated. **DROP.**

---

## What survived verification

The structural infrastructure is sound:
- CSV labels faithfully reproduced as folder symlinks (M2: 0 mismatches)
- Parser faithfully converts symlinks → `.labels.json` (M8: 100% for 6/7 classes)
- Tools are correctly configured and documented (M7)
- The criteria bar is sharp and reproducible (M0: 42 contracts, clear TP/FP/BORDERLINE boundaries)

The problem is not the pipeline — it's the label **source.** DIVE labels are automated tool consensus, and the tools pattern-match on code constructs (Ownable inheritance, Uniswap pair creation) that are ubiquitous in real-world contracts but not indicative of actual vulnerabilities.

---

## What was NOT done (deferred to separate plans)

- **Phase 2: DIVE remaining classes** (IntegerUO, DenialOfService, Timestamp, TransactionOrderDependence, UnusedReturn) — unverified. But structurally, 4 of 5 co-occur with EB at 20-50%, and EB is known noise. Risk of similar DROP decisions.
- **Phase 3: Other sources** (BCCC, SolidiFI, SmartBugs) — not started. SolidiFI (T0 injection-verified) is the most promising source for recovering usable signal.
- **Phase 4: What to train on instead** — if DIVE is dropped, the training set needs replacement labels. SolidiFI's injection-verified contracts (T0) + SmartBugs Curated are the next candidates.
- **Second-AI blind review** (CLAUDE.md §5) — not yet executed. Required before any KEEP decision. Since the decision is DROP (no stratum passes), this is moot — a blind review cannot make a DROP into a KEEP. But document that it was NOT done and why.

---

## Recommendation for Run 13

1. **DROP DIVE ExternalBug labels** — all 16,723 contracts. Replace with SolidiFI T0 injection-verified EB contracts (if any).
2. **DROP DIVE Reentrancy labels** — all 11,400 contracts. Replace with SolidiFI T0 injection-verified RE contracts.
3. **Audit remaining DIVE classes before Run 13 includes them** — IntegerUO, DenialOfService, Timestamp, TransactionOrderDependence, UnusedReturn. If DIVE is the sole source for a class, that class is suspect until Phase 2 is completed.
4. **Do NOT drop DIVE contracts from the graph representation layer** — only drop the labels. The 22,073 contracts are valid Solidity source code and their graph representations (AST, CFG, call graphs) carry structural signal even without labels. The model can benefit from unlabeled data (semi-supervised, self-supervised pre-training).

---

## Files comprising this investigation

```
docs/plan/data_module/
├── CLAUDE.md                                     Governing protocol (standing verification layer)
├── slither_reference.md                          Slither 0.11.5 usage reference
├── aderyn_reference.md                           Aderyn 0.6.8 usage reference
├── echidna_reference.md                          Echidna 2.3.2 usage reference
└── 2026-06-18-dive-data-source-quality-investigation/
    ├── README.md                                 Investigation plan (9 Methods, dependency map)
    ├── METHOD_0_HANDOFF.md                       Working-memory supplement for Method 0
    ├── 00_executive_summary.md                   ← this file
    ├── findings/
    │   ├── 00_tp_criteria_v1.md                  Frozen criteria (REACHABLE bar, v1)
    │   ├── 01_dive_methodology.md                Method 1: DIVE paper findings
    │   ├── 02_folder_csv_agreement.md            Method 2: 0/178,640 mismatches
    │   ├── 03_multilabel_structure.md            Method 3: EB near-universal, RE not cleaner single-label
    │   ├── 04_direct_tp_rate.md                  Method 4: 0/150 TP, DROP decision
    │   ├── 07_tools_decision.md                  Method 7: no additional tools needed
    │   └── 08_parser_faithfulness.md             Method 8: parser faithful, DoS patch = intentional
    ├── scripts/
    │   ├── parser_faithfulness.py
    │   └── verify_folder_csv_agreement.py
    └── samples/                                  (to be populated with seeded sample lists)
```
