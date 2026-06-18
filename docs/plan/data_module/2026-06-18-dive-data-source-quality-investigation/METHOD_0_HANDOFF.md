# Method 0 — Progress Handoff (2026-06-18 → 2026-06-19)

**Status:** IN PROGRESS — Batch 1 complete, batch 2 pending
**Governing protocol:** `docs/plan/data_module/CLAUDE.md`

---

## What we are doing

**Method 0: Freeze TP/FP/BORDERLINE criteria for ExternalBug and Reentrancy** — the single highest-leverage step in the Phase 1 DIVE investigation. We are developing worked examples to sharpen the criteria, in progressive batches, before any TP measurement (Methods 3-6) begins.

No code changes, no label changes. Read-only manual review of DIVE `.sol` contracts.

---

## Proposed criteria (not yet frozen)

| Class | Bar | Definition |
|---|---|---|
| **ExternalBug (EB)** | REACHABLE | A privileged function is callable by an unauthorized caller — missing/misconfigured access control on a reachable path |
| **Reentrancy (RE)** | REACHABLE | External call followed by state writes where re-entry could alter behavior — genuine CEI violation with a re-entrant path |
| **BORDERLINE** | — | Vulnerability pattern exists but mitigating factors create genuine uncertainty: partially guarded function, unusual call sequence, compensating controls |

**Decision procedure** (pre-committed, per README §8): Keep a stratum only if its TP Wilson CI beats the control-arm CI (non-overlapping, stratum higher). DROP if CIs overlap. ENLARGE sample if CIs too wide.

**Proposed practical floor:** ≥10% TP rate (below this, training on noise). Not yet confirmed.

---

## Batch 1 results (10 contracts, seed=20260618)

| # | cid | DIVE labels | EB | RE | Key insight |
|---|---|---|---|---|---|
| 1 | 21797 | AC=1 | FP | — | OZ Ownable + OffsetOracle. All `onlyOwner`. |
| 2 | 19142 | AC=1 | FP | — | DDXP token. Owned, 2-step transfer. All guarded. |
| 3 | 6235 | AC=1, Arith=1 | FP | — | REDPILL. Pure OZ ERC20, no Ownable. Pattern-match on library imports. |
| 4 | 20724 | RE=1 | — | BORDERLINE | Meme token. `_transfer` CEI (swap before state), `lockTheSwap` mitigates. |
| 5 | 21559 | RE=1 | — | BORDERLINE | Same meme token pattern as 20724. |
| 6 | 19601 | RE=1, TS=1 | — | FP | ICO crowdsale. `.transfer()` after state updates. CEI respected. |
| 7 | 1607 | AC=1, RE=1, UR=1 | FP | BORDERLINE | ShibNet. All `onlyOwner` (EB). CEI with `swapLock` mitigation (RE). |
| 8 | 17235 | AC=1, RE=1, Arith=1 | FP | FP | NFT (vikingsgame). 18 `onlyOwner`, no CEI in mint. |
| 9 | 17404 | (zero) | FP | FP | THOG meme token. Correctly zero-label. |
| 10 | 19323 | (zero) | FP | FP | Legacy ERC20. Correctly zero-label. |

**Batch 1 summary:** EB: 5/5 FP. RE: 0 TP, 3 BORDERLINE, 2 FP, 2 correctly zero. Controls: 2/2 FP.

**Pattern found:** Meme tokens (20724, 21559, 1607) share a Uniswap-swap-before-state CEI with partial re-entrancy locks. Clear structural CEI but trusted target (Uniswap) + lock → BORDERLINE.

---

## Tooling foundation established

Before batch 1 was redone, we verified both analysis tools and documented their capabilities:

| | Slither 0.11.5 | Aderyn 0.6.8 |
|---|---|---|
| **RE detection** | Strong: finds CEI in all functions. Caught meme-token pattern and MultiSig CEI. | Limited: only public/external. Finds constructor CEIs (non-exploitable). Misses private-function CEI. |
| **EB detection** | No general missing-auth detector (confirmed: reviewed all 100 detectors). Only narrow cases: suicidal, unprotected-upgrade, arbitrary-send-eth. | No missing-auth detector. `centralization-risk` only flags Ownable PRESENCE, not auth ABSENCE. |
| **Pre-0.8 support** | Works with solc version matching. | Mostly works (8/10 0.4.x), rare parser failures. |
| **Reference doc** | `docs/plan/data_module/slither_reference.md` | `docs/plan/data_module/aderyn_reference.md` |

---

## Completed Methods

| Method | Status | Key finding |
|---|---|---|
| **M8** (parser faithfulness) | DONE | Parser faithful for all 7 DIVE-sourced classes. DoS mismatch = intentional patch (not bug). 257 dropped contracts (<1.7% per class). |
| **M2** (folder↔CSV identity) | DONE | 0/178,640 mismatches. 100% per-contract agreement. |
| **M1** (DIVE methodology) | DONE | DIVE labels are fully automated tool-derived consensus (6 tools, Power-based voting). No manual verification. Authors: "systematically derived, high-confidence annotations rather than manually verified ground truth." |

---

## What comes next

1. **Confirm batch 1 close** — Ali reviews and confirms the 10 judgments
2. **Batch 2** (20 contracts, already selected with seed=20260618_2) — run Slither + Aderyn hints, then manual review with the REACHABLE bar
3. **Batch 3+** (size determined by batch 2 edge case coverage) — iterative until criteria confirmed
4. After M0 frozen: **M7** (tools decision) then **M3** (multi-label structure) then **M4** (direct TP rate with control arm)

---

## Files produced

| File | Purpose |
|---|---|
| `docs/plan/data_module/CLAUDE.md` | Governing protocol (substantially updated during this session) |
| `docs/plan/data_module/aderyn_reference.md` | Aderyn 0.6.8 usage reference |
| `docs/plan/data_module/slither_reference.md` | Slither 0.11.5 usage reference |
| `findings/08_parser_faithfulness.md` | Method 8 findings |
| `findings/02_folder_csv_agreement.md` | Method 2 findings |
| `findings/01_dive_methodology.md` | Method 1 findings |
| `scripts/parser_faithfulness.py` | Method 8 script |
| `scripts/verify_folder_csv_agreement.py` | Method 2 script |

---

## Key learnings from this session

1. **WHAT ≠ WHY.** A correctly measured discrepancy (2,655 DoS mismatches) was misdiagnosed as a "parser bug" when it was a documented intentional patch. Rule: check MEMORY.md + git log + archived scripts before calling anything a bug.

2. **Neither tool detects missing access control.** Both Slither and Aderyn match known vulnerability patterns, not the absence of security controls. Manual review for EB is primary and non-optional.

3. **solc version matching is fragile.** `solc-select use` doesn't reliably update the symlink Slither uses. Manual `rm && ln -s` is the tested approach.

4. **Aderyn's `reentrancy_state_change` only analyzes public/external functions.** Constructor CEIs are the most common finding on DIVE contracts — they're non-exploitable but useful as clues that the contract uses Uniswap setup patterns.

5. **Progressive batch development works.** Batch 1 (10 contracts) revealed the meme-token CEI pattern that dominates BORDERLINE judgments. Batch 2 (20 contracts) is sized to find edge cases (actual TPs, CEI without locks, genuinely missing auth).
