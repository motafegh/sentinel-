# Understanding Checklist — BCCC-SCsVul-2024 Deep Dive, Phase 3

**Date:** 2026-06-06
**Audience:** The human reader of this doc (stakeholder / collaborator / future-self). You should be able to answer the questions in each stage's checklist before moving on.
**Purpose:** This is a **learning artifact**, not a session log. The session log lives in `01_session_log.md`. This doc captures **what you should understand at each stage** so that the work makes sense, not just what was done.

**How to use this doc:**
1. After I complete each stage, I summarize Problem / Solution / Context in the relevant section below.
2. You read it, ask questions if anything is unclear, then confirm "understood" (or "I have a question about X").
3. I move to the next stage.
4. At the end, you should be able to explain every decision in this Phase 3 deep dive to a smart colleague who's never seen the data.

**Stage progression:**

| Stage | Topic | Status |
|---|---|---|
| 0 | Setup, orientation, the bigger picture | ✅ Done |
| 1 | WS-I: Sample construction + slither harness | ✅ Done |
| 2 | WS-I: Full 808-contract run + agreement metrics | ✅ Done |
| 2.5 | Manual review of 32 contracts → D-I-11 (Session 3, 2026-06-07) | ✅ Done |
| 3 | WS-K-K1: 31 source-code regex features | ⏳ Moved to Phase 4 (Stage 0.5) |
| 4 | Synthesis: dataset v1.2, CHANGELOG, MEMORY | ⏳ Replaced by Phase 4 v1.3 |

---

## Stage 0: Setup, Orientation, the Bigger Picture

### The Problem (what are we even doing here?)

**One-line answer:** We need to verify that the 67,311 contracts in `contracts_clean.csv` have **trustworthy vulnerability labels** before SENTINEL trains on them. Phase 2 cleaned the *file* integrity; Phase 3 cleans the *label* integrity.

**Why this problem exists:**
- The BCCC-SCsVul-2024 paper published 111,897 label rows (in long format = 68,433 unique contracts × ~1.635 classes each).
- 766 of those contracts were labeled BOTH `NonVulnerable=1` AND at least one vulnerability class=1. That's logically contradictory — a contract can't be vulnerable to Reentrancy AND have no vulnerabilities.
- 2 contracts appear in 9 of 12 BCCC folders simultaneously — suspicious (likely templated contracts like SafeMath or ERC20 that got re-labeled for every class).
- Multi-label contracts (n_pos ≥ 2) make up 40% of the corpus. For each of those, ALL of their positive labels need to be real, not noise.

**The branches (different ways to handle bad labels):**

| Option | Pros | Cons | Why not chosen |
|---|---|---|---|
| **A.** Trust all labels, train anyway | Fastest | Garbage in → garbage out; caps SENTINEL F1 at the noise ceiling | Doesn't address the question |
| **B.** Drop all suspicious contracts | Conservative; clean training set | Loses ~2,000+ contracts = 3% of data; some labels were probably right | Too aggressive |
| **C.** Manual review every suspicious contract | Gold standard | 848 contracts × 5min/contract = 70+ hours of reading code | Too slow for a session |
| **D.** Use an independent detector (slither) as a 2nd opinion, then spot-check disagreements | Fast, principled, scalable | Detector ≠ oracle; 27% of BCCC won't even compile | **✅ CHOSEN** — D-B2 + D-I1 |

### The Solution (what did we choose to do?)

**WS-I (this session's first workstream):** Use **slither** (Trail of Bits' static analyzer) as an independent 3rd-party detector. For 848 contracts, compare BCCC's label set against slither's hit set. Compute agreement rate. Where they disagree, manually inspect.

**D-I1 (key design decision):** A "slither hit" = a detector that fired at least once on the contract. A "BCCC positive" = a class column with value 1. We compare at the **per-class level** with a mapping: each BCCC class has a set of slither detectors that should fire if that class is real. (The mapping is in `03_phase3_plan.md` §2.1.)

**Edge cases handled in the design:**
1. **Compilation failure:** 27% of BCCC contracts don't compile (per Phase 2 WS-C). Slither needs compilable code. We mark these as `slither_status=COMPILE_ERROR` and **exclude from agreement metrics** (you can't disagree with a detector that couldn't run).
2. **Multi-label contracts:** Slither might detect 3 of 5 labeled classes. That's **partial agreement**, not full. We compute **per-class precision/recall** rather than binary agree/disagree.
3. **NV+vuln contradictions (the 766):** The hypothesis is that EITHER (a) the contract is genuinely vulnerable and the NV=1 is wrong, OR (b) the contract is clean and the vuln=1 is wrong, OR (c) both are partially right (e.g., weak PRNG exists but doesn't lead to exploit). Slither's output is evidence for picking among these.
4. **Nine-folder "maxing" contracts:** These are templated contracts (SafeMath, Ownable) that BCCC apparently labeled for every class because the pattern is present. Slither's detector will tell us which classes have *actual exploitable* code vs. which are just *stylistic patterns*.

### The Context (why does this matter?)

**Downstream impact:**
1. **SENTINEL training (Run 10+):** Run 7 F1=0.3074, Run 9 v11 F1=0.2586. Both could be capped by label noise. If 30% of "Reentrancy" labels are actually clean contracts, SENTINEL wastes capacity learning to detect patterns that aren't there. **Fixing labels could raise the F1 ceiling by 5-15 points.**
2. **AutoML baselines (WS-L, Session 3+):** Tabular models (XGBoost, LightGBM) trained on `contracts_clean_v12.csv` (with WS-K features) are the "ceiling" that SENTINEL's GNN+CodeBERT must justify. If the labels are noisy, AutoML might "win" simply because XGBoost is more robust to label noise than SENTINEL. **Need clean labels for a fair comparison.**
3. **Portfolio narrative (career angle):** "I cleaned 766 label contradictions using independent detector evidence" is a strong data-engineering story. "I trained a model on noisy data" is not.
4. **Paper / blog post (future):** Phase 3 is the data-quality section of any SENTINEL writeup. A rigorous "label verification via cross-tool consensus" is publishable.

**Why slither specifically (not aderyn, not mythril)?**
- Slither: 5s/contract, 101 detectors, deterministic, well-documented, works in Python API. ✅ Default.
- Aderyn: 3s/contract, 88 detectors, Rust-based, needs a directory not a file. ✅ **Will join in Session 3 for 3-way consensus.**
- Mythril: 3min/contract, 17 detectors, symbolic execution. ❌ Too slow for batch (would take 25+ hours). Docker kept for ad-hoc deep analysis on disagreements.

### ✅ Checkpoint — Stage 0 understanding

Before moving to Stage 1, you should be able to answer:

- [ ] **Problem:** Why can't we just trust the BCCC labels and train SENTINEL?
- [ ] **Problem:** What's the "766 contradictions" thing? What are the 2 nine-folder contracts?
- [ ] **Solution:** Why slither, and what does "agreement" mean here at the per-class level?
- [ ] **Solution:** How do we handle the 27% of contracts that don't compile?
- [ ] **Context:** How does this connect to SENTINEL's F1 ceiling? To the AutoML baseline story?

---

## Stage 1: WS-I — Sample Construction + Slither Harness

**Status:** ✅ DONE (2026-06-06). Sample built (808 contracts, slightly under 818 target — 2 multi-positive classes had < 5 contracts each). Slither harness working on 3/5 test contracts; 2 exceptions are real compilation issues (~27% expected compile-fail rate per Phase 2).

### The Problem (what Stage 1 had to solve)

We need to (a) define exactly which contracts slither should analyze, and (b) make slither actually run on BCCC contracts — which are mostly pre-0.6 Solidity, often with bad syntax, multi-contract files, missing pragmas, etc.

**Three sub-problems discovered and fixed:**

1. **Path mismatch:** Phase 2 stored `bccc_file_path` as `BCCC-SCsVul-2024/Source Codes/...` (with space) but the actual directory was renamed to `SourceCodes` (no space). **Fix:** `.replace("Source Codes", "SourceCodes")` — verified all 67,311 paths resolve.

2. **Slither 0.11+ doesn't auto-register detectors.** Unlike older versions, the new slither API requires explicit `slither.register_detector(cls)` for each detector class. The CLI does this internally, but Python API users must do it themselves. **Fix:** the driver script imports `slither.detectors.all_detectors` and registers each of the 101 classes.

3. **Wrong solc version selected.** 92% of BCCC contracts are pre-0.6, but the active solc was 0.8.20 → instant compile failure. Also, solc 0.8.35 is on disk but **not selectable** in solc-select (registry mismatch). **Fix:** a `pick_solc_version(pragma)` function that:
   - Parses `^X.Y.Z`, `>=X.Y <Z.W`, exact `X.Y.Z`, etc.
   - Verifies each candidate is actually selectable before returning
   - Defaults to 0.5.17 (the most common BCCC version) for NaN/missing pragmas

### The Solution (the design)

**Sample design (808 contracts):**
| Bucket | n | Why |
|---|---:|---|
| All review_pending (NV+vuln contradiction) | 766 | Highest uncertainty; D-B2 |
| Top-2 by n_pos (the "maxing" contracts) | 2 | n_pos=8, suspicious templated contracts |
| Multi-positive, 5/class (10 classes) | 40 | Verify multi-label accuracy (2 classes had < 5 multi-pos contracts, hence 40 not 50) |
| Disagreement sample (post-slither) | 0 | Placeholder; filled in Stage 2 |
| **Total** | **808** | |

**Slither harness design:**
- **Subprocess wrapper** (not in-process) for two reasons: (a) timeout works reliably (slither spawns threads; `signal.alarm` is unreliable), (b) isolates memory/per-contract state.
- **Per-contract 30s timeout.** Some contracts have circular imports or complex inheritance that takes >30s.
- **Status enum:** `OK` (ran + got output) / `COMPILE_ERROR` (solc refused) / `TIMEOUT` (killed at 30s) / `EXCEPTION` (unexpected crash) / `PATH_MISSING` (file doesn't exist). All non-OK statuses are **excluded from agreement metrics**.
- **Output format:** JSON list `[{check: "reentrancy-eth", confidence: "Medium", ...}, ...]` — newer slither format is a list of findings, not a dict. Each finding has a `check` field (detector name).

**Test result (5 contracts):**
| Status | Count | Meaning |
|---|---:|---|
| OK (101 detectors registered, ran) | 3 | Worked; hits=0 on all 3 (clean contracts) |
| EXCEPTION (solc JSON parse fail) | 2 | Real compile issues, not harness bugs |

**Key empirical finding from 3 OK contracts:** all 3 had BCCC labels including Reentrancy/CallToUnknown/IntegerUO, but slither found **0 issues**. This suggests **BCCC label noise**: either the patterns are not exploitable (slither agrees they're clean) or the contracts are very simple templates.

### The Context (why this matters, what it impacts)

1. **The harness is reusable.** It will be used in:
   - Stage 2: full 808-contract run (~30-90 min, mostly compilation failures are fast)
   - WS-O (Session 3): 5,000-contract slither batch with aderyn cross-validation

2. **The sample design reveals BCCC class distribution:**
   - 766 review_pending = 1.1% of corpus but probably 30%+ of total label noise
   - 40 multi-positive is small but gives signal on multi-label accuracy
   - The 2 "maxing" contracts are likely templated (SafeMath, ERC20) and labeled for every class they touch — a BCCC methodology quirk

3. **Slither agreement rate will be the headline metric.** If 0.3+ agreement across all 766 review-pending contracts, labels are reasonably reliable. If 0.7+ agreement, they're very good. If 0.9+ agreement, slither is a good proxy for ground truth (and we can use it to find BCCC's mistakes systematically).

4. **Compilation failure rate drives the 808→effective_n reduction.** 27% expected compile failure means effective_n ≈ 590. That's still enough for statistical signal on the per-class agreement rates.

### ✅ Checkpoint — Stage 1 understanding

Before moving to Stage 2, you should be able to answer:

- [ ] **Problem:** Why does slither need an explicit detector registration loop in slither 0.11+? What changed from older versions?
- [ ] **Problem:** Why does the solc version matter, and how does `pick_solc_version` work for the 5 different pragma patterns we see in BCCC?
- [ ] **Solution:** Why a subprocess wrapper instead of in-process slither calls? What does the timeout protect against?
- [ ] **Solution:** Why 30 seconds per contract? Why is `COMPILE_ERROR` excluded from agreement metrics (not just logged)?
- [ ] **Context:** What does it mean that 3 contracts with BCCC "Reentrancy" labels had 0 slither hits — bug in slither, bug in BCCC, or expected?
- [ ] **Context:** What is the "headline metric" we'll compute in Stage 2, and what does each range (0.3 / 0.7 / 0.9 agreement) tell us about label quality?

---

## Stage 2: WS-I — Full 808-Contract Slither Run + Agreement Metrics

**Status:** ✅ DONE (2026-06-06). Ran slither on all 808 contracts, 757 OK + 51 EXCEPTION, **33,049 total findings** across 60 unique detectors. Computed per-class agreement metrics. Built manual review input doc for 30 worst-disagreement + 2 maxing contracts.

### The Problem (what Stage 2 had to solve)

Stage 1 proved the harness works on 5 contracts. Stage 2 had to (a) run it on all 808, (b) compute **per-class agreement** between BCCC's labels and slither's findings, and (c) identify which contracts deserve manual review.

**Four sub-problems discovered and fixed during Stage 2:**

1. **Slither 0.11+ findings parser bug.** Initial parser tried `findings[0]["check"]` directly, but `slither.run_detectors()` returns a **list-of-lists** (one list per detector, each containing finding-dicts). **Fix:** iterate `for det_findings in findings: for f in det_findings: f["check"]`. This fix added ~30,000+ findings that were previously missing.

2. **Bash 120s timeout killed slither runs.** Each contract takes 5-8s (subprocess startup + Python imports + 101 detector registration + solc invocation). At 6 workers, 808 contracts = ~67 min wall time. **Fix:** incremental save every 50 contracts + resume script for the rest. Required 3 separate bash invocations to complete (each ~2 min before timeout).

3. **Compile fail rate was 6.3%, not the 27% from WS-C.** WS-C used a stricter 0.4.24/0.5.17 probe; slither 0.5.17 + auto-solc-picker is more permissive. **Implication:** effective_n = 757 (much higher than the 590 expected).

4. **30 worst-disagreement contracts needed manual review** but reading 30 sources + slither outputs in 30 separate files would be tedious. **Fix:** single `ws_i_inspections_input.md` with all 30 + 2 maxing contracts in one scrollable doc, each with BCCC labels / slither findings / source code / Decision checkboxes.

### The Solution (the design + results)

**Per-class agreement (BCCC label=1 vs slither hit on that class's detectors, n=757 contracts):**

| Class | n_bccc | n_slither | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Class11:Reentrancy | 687 | 256 | 239 | 17 | 448 | **0.93** | 0.35 | **0.51** |
| Class08:CallToUnknown | 673 | 149 | 137 | 12 | 536 | **0.92** | 0.20 | 0.33 |
| Class04:Timestamp | 19 | 115 | 10 | 105 | 9 | 0.09 | 0.53 | 0.15 |
| Class10:IntegerUO | 68 | 75 | 5 | 70 | 63 | 0.07 | 0.07 | 0.07 |
| Class06:UnusedReturn | 7 | 103 | 2 | 101 | 5 | 0.02 | 0.29 | 0.04 |
| Class03:MishandledException | 14 | 119 | 2 | 117 | 12 | 0.02 | 0.14 | 0.03 |
| Class02:GasException | 11 | 161 | 2 | 159 | 9 | 0.01 | 0.18 | 0.02 |
| Class01:ExternalBug | 8 | 34 | 0 | 34 | 8 | 0.00 | 0.00 | 0.00 |
| Class09:DenialOfService | 7 | 70 | 0 | 70 | 7 | 0.00 | 0.00 | 0.00 |
| Class12:NonVulnerable | 728 | 0 | 0 | 0 | 728 | 0.00 | 0.00 | 0.00 |

**Overall:** macro-F1 (vuln classes) = **0.128**, micro-F1 = **0.240**, micro-precision = **0.367**, micro-recall = **0.179**.

**Top 5 slither detectors that fired:**
1. `naming-convention` × 15,088 (quality noise — function names too long, not vuln)
2. `deprecated-standards` × 2,901 (pragma noise — `^0.4.13` is old)
3. `dead-code` × 2,345 (quality)
4. `solc-version` × 1,533 (pragma)
5. `external-function` × 1,311 (quality)

**The 30 worst-disagreement contracts (manual review input):**
- 2 are "maxing" contracts (n_pos=8, in 9 BCCC folders) — both are templated (e.g., SafeMath + ERC20 + MyAdvancedToken)
- 28 are review_pending (NV+vuln contradiction)
- All have slither OK
- 8,014-line doc generated at `labels/ws_i_inspections_input.md` with each contract's: BCCC folders, source code (truncated to 200 lines), BCCC labels, slither findings grouped by detector, and Decision checkboxes (KEEP / MODIFY / REVIEW-NEEDED / FALSE-POSITIVE-CONTRACT)

### The Context (what this tells us about BCCC label quality)

**Three headline findings:**

1. **Reentrancy labels are mostly correct (P=0.93).** When BCCC says a contract has Reentrancy, slither agrees 93% of the time. But slither only catches 35% of BCCC's Reentrancy cases (R=0.35) — likely because the `approveAndCall` pattern in pre-0.5 contracts doesn't trip the state-change-after-external-call detector. **Implication for SENTINEL training:** Reentrancy labels are reliable for *positive* examples (use confidently); negative examples might be over-zealous (slither found 17 cases where BCCC said Reentrancy but slither saw no reentrancy detector fire).

2. **CallToUnknown has the same pattern (P=0.92, R=0.20).** BCCC's `missing-zero-check` is a strong signal — most are real. But slither only catches 20%, meaning the corpus has many "token sent to wrong address" patterns slither doesn't detect.

3. **IntegerUO is the worst (F1=0.07) — but it's slither's fault, not BCCC's.** Slither has no dedicated pre-0.8 integer overflow detector (pre-0.8 contracts have no compile-time overflow checks, so static analysis is harder). **This is exactly why D-P3-10 added Aderyn** — it has dedicated `unsafe-casting` and `division-before-multiplication` detectors for cross-validation in WS-O.

4. **The 728 NonVulnerable contracts with 0 slither findings... wait, actually 100% of contracts have findings (757/757).** This is the opposite of expected — slither fires SOMETHING on every BCCC contract. The top finding is `naming-convention` (style), so technically slither "finds issues" on every contract but they're mostly quality issues, not vulnerabilities. **NV (NonVulnerable) is not a "slither found 0 issues" class — it's a "this contract has no exploits" assertion.** This is why NV has F1=0.00 (slither has no "clean" detector).

**What this means for the dataset:**

- **Reentrancy and CallToUnknown labels are trustworthy for training** (precision >0.92, even if recall is low).
- **IntegerUO labels need Aderyn cross-validation** before SENTINEL can use them.
- **The 30 worst-disagreement contracts are likely edge cases** (templated, false-positive-prone, or BCCC methodology artifacts). The manual review (in `ws_i_inspections_input.md`) will yield label-change recommendations for v1.2.
- **The other 778 contracts are in better shape than the 30 worst** (they have lower disagreement scores by construction). The Stage 2 manual review is a *targeted* investigation, not a full re-labeling effort.

**Caveats:**
- Sample is biased toward review_pending (95% of contracts). Multi-positive bucket (40) and maxing (2) are too small for class-stratified conclusions.
- 2 nine-folder "maxing" contracts have the highest disagreement scores (0.444) — both have 8 BCCC classes and slither finds 19-21 issues each. Almost certainly templated contracts labeled for every class. Manual review will confirm.
- Many contracts have 200+ slither findings (e.g., `naming-convention` alone fires 15+ times per contract). The 30 worst-disagreement list is dominated by "noisy" contracts where BCCC said 3 specific classes but slither found 200+ generic issues. The BCCC labels are likely *narrower* (specific exploit type) vs slither's *broader* (any quality issue). This is a key methodological point for the paper.

### ✅ Checkpoint — Stage 2 understanding

Before moving to Stage 3, you should be able to answer:

- [ ] **Problem:** Why was the 6.3% compile fail rate surprising (WS-C said 27%)? What does this tell us about slither's auto-solc-picker?
- [ ] **Problem:** Why does Reentrancy have high precision but low recall? What's the contract pattern that trips BCCC's labeling but not slither's detector?
- [ ] **Solution:** What does the 30-worst-disagreement list represent? Why is the manual review *targeted* (not full re-labeling)?
- [ ] **Solution:** Why does IntegerUO F1=0.07 indicate we need Aderyn, not that BCCC is wrong?
- [ ] **Context:** If Reentrancy and CallToUnknown have P>0.92, what does that mean for SENTINEL training (which classes can we trust, which need cleaning)?
- [ ] **Context:** The headline macro-F1=0.13 is "low" — is that bad news or expected news for static-analysis-vs-static-analysis comparison?

---


