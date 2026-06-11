# Stage 4 — Verification (the BCCC-failure catcher)

**Date:** 2026-07-07
**Status:** NOT STARTED. Reading required before Stage 5.
**Reading time:** 25-35 minutes.
**Goal:** After this doc, you can answer all 7 items in `LEARNING_CHECKLIST.md` §"Stage 4" from memory.

---

## 1️⃣ The Problem

### What Stage 4 has to deliver

Stages 1–3 produced preprocessed, represented, labeled contracts. Stage 4 asks: **are these labels correct?** This is the question BCCC failed to ask for 14 days of work.

The 89% Reentrancy FP rate in BCCC would have been caught in minutes if someone had asked: "for each contract labeled Reentrancy, does the AST actually contain an external call + state change AFTER the call?" Stage 4 automates this question for all 10 classes.

### Why verification is per-class, not per-source (D-4.1)

A source can have 90% Reentrancy FP (BCCC), but other sources can recover the class. The verification gate looks at the **merged labels across all sources**, not per-source. A "good" Tier-1 source compensates for a "bad" Tier-4 source.

### The 6 verification components

| Component | What it does | Why it matters |
|---|---|---|
| **semantic_checker** | For each (class, contract) pair, checks if the AST contains the code pattern implied by the label | Would have caught 89% BCCC Reentrancy FP in minutes |
| **tool_validator** | Runs Slither on labeled positives, reports per-class agreement rate | Tool agreement is corroborative (not authoritative) |
| **fp_estimator** | Samples N positives per class, runs all tools, reports empirical FP rate | Sampling is stratified by source AND tier |
| **class_auditor** | Per-class count, per-source breakdown, co-occurrence matrix | Catches 99% DoS↔Reentrancy automatically |
| **negative_checker** | For NonVulnerable contracts, reports fraction with tool hits | Catches "41% of NonVulnerable had Slither hits" pattern |
| **probe_dataset** | 40 contracts per class where vulnerability is visually obvious | Model interpretability input |

### The BCCC Phase 5 regression test (D-4.8)

The BCCC Phase 5 verification report (`p5_s6_verification_report.md`) is the regression target. The new module must reproduce it to within ±0.5% per-class drop counts. This proves the module is **at least as good** as the 14-day ad-hoc Phase 5 work.

The regression test checks each Phase 5 stage's output individually:
- `p5_s1` → evidence table + coverage report
- `p5_s2` → automated verdict + dispute CSVs
- `p5_s3` → refined verdict + residual CSVs
- `p5_s4` → final verdict + gate results
- `p5_s6` → class size comparison + verification report

---

## 2️⃣ The Solution

### Semantic checks per class (D-4.2)

Each class has an AST-level pattern:

- **Reentrancy** → external call + state change **after** call (CEI ordering enforced)
- **CallToUnknown** → `.call{}` / `.delegatecall{}` / `.send()` / `.transfer()` + lvalue is `address` (not `bool`) + NOT in OZ SafeERC20 wrapper
- **Timestamp** → `block.timestamp` / `now` in a conditional (relies on `feat[2]=1.0` from A9 fix)
- **IntegerUO** → arithmetic op in pre-0.8 OR `unchecked{}` in 0.8+ (relies on `has_unchecked_block` from Stage 1)
- **DoS** → loop with external call or unbounded iteration
- **TOD** → `tx.origin` in a permission check
- **ExternalBug** → cross-contract call where target is not a known interface
- **GasException** → unchecked `send()` / `transfer()` / low-level call
- **MishandledException** → call with unused return (relies on `feat[7]` from return_ignored fix)
- **UnusedReturn** → internal function call with unused return

### Tool corroboration (D-4.3)

Slither agreement is a signal, not ground truth. A class with low Slither agreement is suspicious; high agreement is reinforced. The friend's research confirmed: Conkas + Slither + Smartcheck only detects 76.78% of actual vulnerabilities; Slither reentrancy precision is 51.97%.

### Hard gate vs soft gate (D-4.5)

| Verdict | Semantic check pass rate | Tool agreement | FP estimate | Action |
|---|---|---|---|---|
| **VERIFIED** | > 90% | > 70% | < 15% | Export allowed |
| **PROVISIONAL** | 60–90% | > 50% | < 30% | Export with warning |
| **BEST-EFFORT** | 30–60% | Documented | Documented | Export with warning |
| **FAIL** | < 30% | — | > 30% | Blocks export |

### Negative checker threshold (D-4.6, 5%)

If > 5% of NonVulnerable contracts have tool hits, warn. If > 10%, FAIL. Uses the canonical `CLASS_TO_DETECTORS` list, not generic Slither.

### SmartBugs Curated recall test (friend review)

The 143 SmartBugs Curated hand-labeled contracts are the ground-truth probe. The semantic_checker must retain ≥90% of confirmed positives. If < 90%, the checker is too strict and Run 11 is deferred.

Why 90%? Some valid reentrancies use intermediate state (not strict CEI ordering). The checker's CEI pattern is intentionally strict to drop BCCC FPs, but this may also drop some valid positives. 90% is the sweet spot.

---

## 3️⃣ The Broader Context

### What Stage 4 enables downstream

- **Stage 5 (splitting)** splits on verified labels, not raw labels
- **Stage 7 (export)** is blocked by FAIL classes
- **Stage 8 (Run 11)** trains on verified labels only

### What breaks if Stage 4 is wrong

- Missing semantic checker → 89% Reentrancy FP re-enters the corpus → same BCCC failure
- Missing co-occurrence matrix → 99% DoS↔Reentrancy goes undetected
- Missing SmartBugs recall test → checker is too strict → too many false negatives → model can't learn positive patterns

---

## 4️⃣ Verification — Stage 4 exit criteria

| # | Check | Status |
|---|---|---|
| 1 | 10 per-class pattern YAMLs exist | ⏳ |
| 2 | 6 verification components compile and run | ⏳ |
| 3 | BCCC regression test passes (±0.5% per-class) | ⏳ |
| 4 | Semantic checker catches 89% BCCC Reentrancy FPs | ⏳ |
| 5 | Semantic checker catches 86.9% BCCC CallToUnknown FPs | ⏳ |
| 6 | Probe dataset builds 420 contracts (40+1+1 per class) | ⏳ |
| 7 | Hard gate blocks export on FAIL classes | ⏳ |
| 8 | SmartBugs Curated 143-contract recall ≥ 90% | ⏳ |

---

## 5️⃣ The "got it" checklist

1. **What is the semantic checker?** For each (class, contract) pair, checks if the AST contains the code pattern implied by the label. The 86.9% CallToUnknown FP would have been caught by checking for `.call{}` existence.

2. **Why is tool corroboration not authoritative?** Slither reentrancy precision is 51.97% (friend research). Tool agreement is a signal among many, not ground truth.

3. **What's the BCCC Phase 5 regression test?** The new module must reproduce Phase 5's verification report to within ±0.5% per-class drop counts. Proves the module is at least as good as 14 days of ad-hoc work.

4. **What's the hard gate vs soft gate?** FAIL blocks export. PROVISIONAL/BEST-EFFORT export with warning. Override requires explicit config.

5. **Why 5% negative checker threshold (not 10%)?** 10% is too lax — by then the class is heavily contaminated. 5% is early warning.

6. **Why SmartBugs Curated ≥ 90% recall?** Some valid reentrancies use intermediate state. The strict CEI pattern drops BCCC FPs but may also drop some valid positives. 90% is the sweet spot.

7. **What's the probe dataset?** 40 contracts per class where vulnerability is visually obvious. Used by the model interpretability suite to verify the model learned the right patterns, not shortcuts.

If you can answer all 7, Stage 4 is mastered.

---

## 6️⃣ What to read next

- **LEARNING_CHECKLIST.md** §"Stage 4"
- **05_stage_4_verification.md** — the design + intent document
- **Sentinel_v2_Data_Module_Integration_Proposal.md** §3.5 (verification)
- **Reference:** `Data/docs/legacy/bccc_deep_dive/Phase5_LabelVerification_2026-06-08/` — the Phase 5 scripts that Stage 4 replaces

When you're ready, say **"Stage 4 is mastered — let's move to Stage 5."**
