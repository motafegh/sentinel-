# Plan — DIVE Data Source Quality Investigation (Phase 1: Understand the Data)

**Date:** 2026-06-18
**Module:** data_module
**Owner:** Ali
**Status:** PHASE 1 COMPLETE — DIVE EB+RE → DROP. Executive summary: `00_executive_summary.md`
**Parent plan:** `docs/plans/2026-06-18_data_module_dive_crosswalk_externalbug_reentrancy_fix_plan.md` (supersedes Step 1 / Step 2b / Step 2c conclusions; this plan is the fresh start)
**Governing protocol:** `docs/plan/data_module/CLAUDE.md` (the subfolder's instructions — applies to
any dataset/source, distrust-docs rule, manual-review batching + no-extrapolation rule, bullet-proof
gates, tooling policy)
— **every Method below is executed as its own gated step, not as a pre-approved batch.** Ali
does not approve "the whole plan" once; each Method gets explained immediately before it starts,
confirmed, executed without assumptions, then its findings are clarified and confirmed before the
next Method begins. The Methods below are a proposed *order*, not a pre-authorized work order — the
next Method to run is whatever Ali names next.

---

## 0. Scope & phasing — this is the FIRST instance of a reusable verification layer

This investigation is **not** a one-off DIVE/Run-12 check. The governing `CLAUDE.md` defines a
**standing verification layer** that every data source must pass before the expensive downstream
modules (training, ZK, on-chain) are allowed to trust it. **Completeness outranks speed — there is no
time pressure** (a week is fine); the "time budgets" per Method are planning estimates, not
constraints, and never a reason to cut a step (see `CLAUDE.md` top block).

**Phasing of the work (current state → expansion):**

| Phase | Scope | Status |
|---|---|---|
| **1 (NOW)** | **DIVE — ExternalBug + Reentrancy only** (the two classes BUG-2 / the prior audit flagged as broken) | the active target of this README |
| **2 (next)** | **DIVE — all remaining classes** (Access Control→EB already covered; Arithmetic→IntegerUO, Time→Timestamp, Unchecked Return→UnusedReturn, DoS, Front Running→ToD, Bad Randomness) | declared, not started |
| **3 (later)** | **Other sources** (BCCC, SolidiFI, SmartBugs, CGT, future) run through the *same* `CLAUDE.md` procedure | the layer is built to be reused; this README is its template |

So: do EB/RE first and well; the Methods, criteria format, gates, and decision procedure are
deliberately written to be re-pointed at the rest of DIVE and at any other source with minimal change.

**Plan structure (confirmed):** one main gated plan (this README) with **just-in-time** detail — each
Method's full spec is delivered at its EXPLAIN gate the moment before it runs, when earlier findings
can shape it. **No pre-written per-Method sub-plans** (that would re-introduce the over-planning the
protocol exists to prevent). The one exception is `00_tp_criteria_v1.md`, which is a *deliverable
produced during* Method 0, not a sub-plan written ahead of it.

---

## 1. Why this plan exists

Prior session (2026-06-18) concluded that DIVE's folder labels for ExternalBug and Reentrancy are ~95% false-positive and proposed dropping them entirely (Option B in the parent plan). The conclusion was based on:

- A 175-contract manual review of the DIVE+Slither-agreed subset
- A 30-contract spot check of Aderyn-only positives
- A 100-contract sample of raw DIVE folder positives
- A 75-contract sample of raw DIVE folder positives (separate run)

**Ali raised a fair challenge:** these conclusions came from one human reviewer (me) using strict TP criteria, and the multi-label structure of DIVE (15,423 / 22,330 = 69% of contracts are tagged with 2+ vulnerabilities simultaneously) was not examined. The parent plan treated DIVE as a per-class signal source; it is actually a multi-label signal source, and that has not been properly analyzed.

**This plan resets the investigation. It does not propose a fix. It proposes to understand the data first, with explicit methods and explicit deliverables, before any code or label changes.**

---

## 2. What we actually know (verified ourselves this session, not taken from docs)

> **Verification stamp (2026-06-18):** every structural count in this section was re-derived this
> session by running commands directly against the raw data (not copied from a prior doc), per the
> governing `CLAUDE.md` §2. `csv.reader` + `ls | wc -l` + a per-class counter. **All counts below
> matched exactly.** The non-structural claims in §2.3 (prior TP rates) are explicitly NOT verified
> and are marked as such.
>
> **Frame warning — read before using any number here:** this section describes **raw DIVE**
> (22,330 contracts, the source on disk). The **v3 export** that Run 12 trained on is a *subset*:
> **22,073 DIVE contracts** (= 22,330 raw − 257 compile failures, per `dive.yaml:26`). Therefore
> prior root-cause docs that cite **16,582 ExternalBug-positive / 75.1%** are computed on the
> **export**, whereas raw CSV "Access Control"=1 is **16,723**. The 141-contract gap is compile-
> failed Access-Control files dropped during preprocessing. **Never cross-compare a raw-DIVE number
> with an export number** — always state which frame a figure is on.

### 2.1 The raw DIVE directory structure

Source: `data_module/data/raw_staging/dive/` (symlinked from `data_module/data/raw/dive/repo`).

```
__source__/                  22,330 canonical .sol files (real Ethereum mainnet contracts)
Access Control/              16,723 SYMLINKS → __source__/<id>.sol
Arithmetic/                   9,542 symlinks
Bad Randomness/                 634 symlinks
DoS/                          3,781 symlinks
Front Running/                  606 symlinks
Reentrancy/                  11,400 symlinks
Time manipulation/            6,322 symlinks
Unchecked Return Values/      5,911 symlinks
```

Total symlink references: 54,919 across 8 folders. Each symlink points to one of the 22,330 canonical files.

### 2.2 The DIVE label source

`data_module/data/raw_staging/dive_labels/DIVE_Labels.csv` — **22,330 data rows** + header
(confirmed via `csv.reader`; note `tail -n+2 | wc -l` reports 22,329 because the file has no trailing
newline — the csv-parsed count of 22,330 is authoritative).

```
contractID,Reentrancy,Access Control,Arithmetic,Unchecked Return Values,DoS,Bad Randomness,Front Running,Time manipulation
1,0,0,0,0,1,0,0,0
2,1,0,0,0,1,0,0,0
...
```

**Per-class positive counts (CSV `=1`), verified this session — and they match the folder symlink
counts in §2.1 exactly:**

| Class column (CSV) | CSV `=1` | Folder symlinks (§2.1) | Match |
|---|---:|---:|:---:|
| Access Control | 16,723 | 16,723 | ✓ |
| Reentrancy | 11,400 | 11,400 | ✓ |
| Arithmetic | 9,542 | 9,542 | ✓ |
| Time manipulation | 6,322 | 6,322 | ✓ |
| Unchecked Return Values | 5,911 | 5,911 | ✓ |
| DoS | 3,781 | 3,781 | ✓ |
| Bad Randomness | 634 | 634 | ✓ |
| Front Running | 606 | 606 | ✓ |

→ At the **aggregate count level the folders and the CSV are identical.** This narrows Method 2 (see
§4): it is now a **per-contract identity check** ("is contract X in the AC folder iff its CSV
AccessControl=1?"), not a hunt for a count mismatch.

Multi-label distribution (frame: **all 22,330 raw contracts**, including the 2,686 with zero labels;
note `dive.yaml:21` quotes the same distribution over the 19,644 *labeled-only* files, i.e. excluding
the zero-label row — same data, different denominator):

| # of labels per contract | # of contracts | % of 22,330 |
|---|---|---|
| 0 | 2,686 | 12.0% |
| 1 | 4,221 | 18.9% |
| 2 | 4,882 | 21.9% |
| 3 | 4,598 | 20.6% |
| 4 | 3,206 | 14.4% |
| 5 | 2,152 | 9.6% |
| 6 |   542 | 2.4% |
| 7 |    40 | 0.2% |
| 8 |     3 | 0.01% |

**15,423 / 22,330 (69.1%) of contracts have 2+ vulnerability labels.**

### 2.3 What the prior session CLAIMED (unverified — treat as hypotheses, not facts)

> Per governing `CLAUDE.md` §2: every number below is a **prior-session claim that this investigation
> has NOT re-verified.** The cited audit files exist on disk (confirmed), but their numbers are
> hints/clues to be reproduced, not established results. Do not carry any of these forward as fact;
> each is a hypothesis a Method below may confirm or refute. The ✓/✗ marks denote only whether the
> prior session *attempted* the work, not whether we trust the result.

- ✓ attempted — Step 1: 75 random contracts from `Access Control/` → **claimed** 5.3% TP (`data_module/audit/2026-06-18_dive_crosswalk_sample_validation.md`)
- ✓ attempted — Step 2a: full Slither corroboration on **claimed** 15,920 EB + 11,018 RE positives (~15.5 min) — `data_module/audit/2026-06-18_dive_externalbug_reentrancy_slither_corroboration.json`
- ✓ attempted — Step 2b: 100 EB + 75 RE sample of the agreed set → **claimed** 4.0% / 2.7% TP
- ✓ attempted — Phase 3: Aderyn 0.6.8 cross-tool → **claimed** 3-way precision 3.0% / 1.7%
- ✓ attempted — Phase 3b: 30 Aderyn-only-positive (Slither-disagreed) → **claimed** 0/30 TP
- ✗ **NOT done: cross-tabulating agreed-set contracts with their multi-label structure**
- ✗ **NOT done: comparing TP rate between single-label and multi-label contracts**
- ✗ **NOT done: examining what a contract tagged with 5 vulnerabilities actually looks like**
- ✗ **NOT done: comparing TP rates between contracts tagged with only-1-class vs contracts tagged with many**
- ✗ **PARTIALLY done this session: CSV 0/1 vs folder symlinks agree at the AGGREGATE count level (§2.2 table, exact match). Per-contract identity check still outstanding (Method 2).**
- ✗ **NOT done: reading the DIVE source paper / methodology** to understand their labeling criteria

### 2.4 The factual gap in the prior conclusion

**My TP rate measurement of 4.0% (EB) is from a sample of the agreed set.** The agreed set is 6,804 contracts. But I do not know:

- How many of those 6,804 contracts are tagged ONLY as ExternalBug vs also tagged with 4 other classes
- Whether the 4 TPs I found are all single-label contracts (clean ExternalBug-only files) or all multi-label (Reentrancy + EB + Arithmetic + etc.)
- Whether the 96 FPs I found break down differently by multi-label count

If the agreed set is **mostly** multi-label contracts (likely, given 69% of DIVE is multi-label), then my sample is heavily contaminated by multi-label files where the EB "agreement" is a side effect of a different primary vulnerability. A contract tagged with 8 classes is by definition not a clean ExternalBug-only vulnerability.

**This is a real gap in the prior analysis. The TP rate of 4.0% may be wrong, or it may be right but for the wrong reason.**

### 2.5 Known intentional interventions on DIVE labels (verify — do NOT re-discover these as bugs)

Some DIVE labels were **deliberately modified** after parsing. These are documented, intended changes.
Any Method that finds one must treat it as such — not as a parser/data bug — per `CLAUDE.md` §1 (a
root cause is a claim, check history first) and §2 (history is a required lead). **Before diagnosing
any label discrepancy, check this table + `git log` + `MEMORY.md` first.**

| Intervention | Date | Effect (verified this session) | Signature in the data | Reference |
|---|---|---|---|---|
| **DoS + Reentrancy co-occurrence patch** | 2026-06-13 | Zeroed `DenialOfService` for the **2,655** contracts tagged DoS=1 **and** Reentrancy=1 from DIVE (single T2 source). Rationale: a reentrancy that incidentally blocks transfers is a reentrancy, not a DoS. | `DenialOfService.value=0` **with `tier: null`**, while CSV says DoS=1 and Reentrancy=1. All 2,655 confirmed to match this exact signature. | `~/.claude/.../memory/2026-06-13_project_dos_patch.md`; `data_module/temp/archive/2026-06-13_run12_prep/scripts/patch_dos_v3.py`; test `test_dive_dos_reentrancy_cooccurrence_finding` |

**Why this table exists (the lesson):** Method 8's first pass found exactly these 2,655 contracts and
misdiagnosed them as a "parser cache bug", recommending a `force=True` re-run that would have
**reverted** the deliberate, test-protected patch. The count (2,655), the date (2026-06-13), and the
`tier: null` field all pointed straight at the intentional patch — but the agent constructed a
plausible mechanism instead of checking history first. **Add new interventions to this table as they
are found, so the next Method never re-trips on them.**

---

## 3. Goal of this plan

**To produce an evidence-based understanding of:**

1. What the DIVE dataset's labels actually mean (semantics: how were contracts assigned to folders?)
2. Whether the folder symlinks and the DIVE_Labels.csv agree (consistency check)
3. Whether the v3 parsed labels the model **actually trained on** faithfully reflect the CSV
   (raw CSV → `dive.py` parser → `labels/dive/*.json`) — see Method 8
4. How multi-label structure relates to per-class quality
5. Whether the DIVE CSV's per-class column can be used as a more reliable signal than folder presence
6. The true-positive rate **against frozen, agreed criteria** (Method 0), measured **with a control
   arm** (labeled-negatives), so a TP rate is interpretable rather than free-floating
7. What an appropriate label-source decision is, given the data — **judged against a decision rule
   pre-committed before results are seen** (§8)

**Explicit non-goals (deferred to a separate plan after this one completes):**

- Any code change to `dive.yaml`, `dive.py`, the export pipeline, or `label_quality.py`
- Any decision on Option A / B / C from the parent plan
- Any change to Run 13 plans
- Any re-export of v3 → v3.1

---

## 4. Methods (explicit, in order — each gated individually per the governing protocol)

**How a Method actually proceeds (per protocol §1):** Claude explains the Method's data/tool/steps/
deliverable in full → Ali confirms explicitly → Claude executes exactly that, no substitutions, no
assumed values → Claude writes findings to the deliverable file as it goes → Claude explains the
findings in full (including caveats, sample size, what's not covered) → Ali confirms before the next
Method starts. If a Method's own steps are large enough to hide assumptions (e.g. Method 4's strata),
those steps get their own mini-gate too.

**Diagnosing a discrepancy (applies to EVERY Method — this is the lesson of the DoS misdiagnosis):**
When the data doesn't look as expected, the **WHAT** (which contracts, how many) and the **WHY** (root
cause) are *separate* claims (`CLAUDE.md` §1). Establish the WHAT with evidence. For the WHY, you must
**first** check: §2.5's known-interventions table, `git log` / `git blame` on the affected files,
`MEMORY.md` + `project_*.md`, archived/`temp/` scripts, **and the artifact's own provenance fields**
(`tier`, `source`, timestamps) — only *then* form an explanation, and label it a **HYPOTHESIS** until
evidenced. A plausible story is not a finding. Never call a discrepancy a "bug", and never propose a
fix (re-run/force/overwrite), without first ruling out that it is an intentional, documented change
and checking what the fix would revert or break.

**Dependency map (which Methods gate which — the listed order is NOT interchangeable):**

```
Method 0 (freeze TP criteria)  ─┐
Method 1 (label semantics)      ├─► REQUIRED before any manual TP judgement (M3 review, M4, M5)
Method 7 (tools available)      ─┘   (you cannot judge a TP without the bar, the label's meaning,
                                      and the tools)
Method 2 (folder↔CSV identity)  ──► independent; cheap; can run early
Method 8 (CSV→parser→v3 labels) ──► independent; should run early (validates the ACTUAL train labels)
Method 3, 4, 5, 6 (TP rates, strata, cross-source, agreement) ──► all DEPEND on M0 + M1 (+ M7)
```

Concretely: **Method 0, Method 1, and Method 7 are prerequisites.** No sampling/TP Method (3–6) may
start until the criteria are frozen (M0), the label semantics are understood (M1), and the tool set
is fixed (M7). Method 2 and Method 8 are independent structural checks and can run any time.

### Method 0 — Freeze the TP/FP/BORDERLINE criteria (prerequisite gate, per CLAUDE.md §4)

**Why:** Every prior "TP rate" is distrusted because the bar for "true positive" was never written
down and agreed. This Method fixes the bar **before** any contract is judged, so the later numbers
mean something. This is the single highest-leverage step in the plan.

**Steps:**

1. **The AI PROPOSES the bar per class — it does not ask Ali from a blank page (CLAUDE.md §1, §4).**
   For ExternalBug and Reentrancy, the AI presents a **recommended** TP definition (pattern-present
   vs reachable vs exploitable), the reasoning, the trade-off (looser → more recall + more noise;
   stricter → cleaner but may discard subtle real bugs), and a recommended choice. Ali confirms or
   corrects — starting from the AI's analysis, never a blank page.
2. Define the **BORDERLINE / UNSURE** bucket explicitly — what makes a contract borderline rather
   than TP or FP — so uncertainty is counted, not hidden. (Also AI-proposed, Ali-confirmed.)
3. Provide **worked examples**: for each contract, run Slither + Aderyn FIRST as investigative hints per CLAUDE.md §8 — their detector names and highlighted lines are clues to focus manual reading, never trusted as verdicts. Pre-judge ≥2 contracts TP, ≥2 FP, ≥1 BORDERLINE per class, each with the
   one-line reasoning, drawn from the actual DIVE corpus so the bar is concrete. The AI pre-judges
   them with its reasoning; Ali validates the AI applied its own bar correctly.
4. **Confirm the decision PROCEDURE too (not just the criteria):** the AI presents the calibrated-null
   keep/drop rule (CLAUDE.md §7 — keep a stratum only if its TP CI beats the control arm's CI; no
   hardcoded magic number) plus any proposed practical floor, with reasoning. Ali confirms.
5. **Worked examples are developed in PROGRESSIVE BATCHES, not a pre-set fixed number.** Start
    with a first batch (10), pre-judge them under the proposed bar, then assess with Ali what
    edge cases were missed or what the examples clarify. Propose the next batch size (20, 30, ...)
    based on what the first batch revealed — and repeat until Ali confirms the criteria are sharp
    enough to apply identically by a second reviewer. Never commit to "we'll do exactly N examples"
    before seeing the first batch. This is a standing rule for criteria development; do not repeat
    this protocol.
6. Get Ali's explicit sign-off. The result is **criteria version v1** + **decision procedure v1**,
    frozen. Any later change is a new version (v2, …); measurements record which version they used.

**Deliverable:** `findings/00_tp_criteria_v1.md` — the frozen, Ali-signed rubric + worked examples.

**Effort estimate (planning only — NOT a constraint; completeness outranks speed):** 1–2 hours (mostly discussion + a handful of example contracts).

**Gate:** No Method 3/4/5/6 may begin until `00_tp_criteria_v1.md` exists and is signed off.

### Method 1 — Verify the DIVE dataset origin and labeling methodology

**Why:** Before judging the labels, we need to know how the DIVE authors produced them. Are they (a) manual expert audit, (b) tool-derived, (c) automated pattern matching, (d) something else? Different methods have different reliability profiles.

**Steps:**

1. Search the DIVE dataset for any included README, methodology doc, paper reference, or citation. The raw `data_module/data/raw_staging/dive/` and `dive_labels/` dirs should be checked first.
2. The 2026-06-10 integration test doc (`data_module/docs/2026-06-10_data_module_integration_test_dive.md`) cites "Nature Sci. Data 2025" — find that paper. Likely URL: `https://www.nature.com/articles/s41597-025-XXXXX`. Look in the `ingestion_manifest.json` (`data_module/data/raw/dive/ingestion_manifest.json` has `url: ""` so no URL was captured, but the doc cites the source).
3. Read the paper's labeling methodology section. Pay attention to:
   - How were vulnerability labels assigned? (manual? tool? combination?)
   - What does each DASP class mean in their taxonomy?
   - How were multi-label contracts handled?
   - Is there a confidence/probability column we ignored?
4. Search the broader internet (web fetch from `nature.com`, `arxiv.org`, `github.com`) for "DIVE smart contract vulnerability dataset Nature Scientific Data 2025" to find the original repo or methodology.

**Deliverable:** A 1-page summary written to `docs/plan/data_module/2026-06-18-dive-data-source-quality-investigation/findings/01_dive_methodology.md` covering: source URL, labeling method, multi-label handling, confidence indicators (if any), and any explicit limitations the authors disclose.

**Effort estimate (planning only — NOT a constraint; completeness outranks speed):** 1-2 hours.

### Method 2 — Verify the DIVE folder symlinks match the DIVE_Labels.csv (per-contract)

**Why:** The folder layout was created by `label_folderize.py` from the CSV. We should verify that the symlinks faithfully reproduce the CSV — if there's a bug or mismatch, the prior analysis based on folder presence may be off.

**Already established this session (§2.2):** at the **aggregate count level the folders and CSV are
identical** (all 8 per-class counts match exactly). So this Method is **narrowed to a per-contract
identity check** — are the *same* contract IDs in the AC folder as have AC=1 in the CSV? — not a
search for a count discrepancy. A count match does not guarantee a per-contract match (a swap of two
contracts would preserve counts), so the check is still worth running, but the expected outcome is
"perfect per-contract agreement" and any deviation is a high-signal surprise.

**Steps:**

1. For each of the 22,330 contracts in the CSV, compute the set of folders it should be in (from CSV columns).
2. For each of the 22,330 contracts, compute the set of folders it actually IS in (by resolving symlinks in `__source__/`, `Access Control/`, `Arithmetic/`, etc.).
3. Compute agreement: for each (contract, class) pair, does CSV say 1 and folder exists? Does CSV say 0 and folder missing?
4. Report mismatch counts per class. If mismatches exist, decide whether to trust CSV (preferred) or folders (only if CSV is broken).

**Implementation:** Small Python script, ~100 lines. Read CSV with `csv.DictReader`, walk each folder with `pathlib.Path.glob('*.sol')`, build sets, diff.

**Deliverable:** Script at `docs/plan/data_module/2026-06-18-dive-data-source-quality-investigation/scripts/verify_folder_csv_agreement.py` + report at `findings/02_folder_csv_agreement.md`.

**Effort estimate (planning only — NOT a constraint; completeness outranks speed):** 1 hour.

### Method 3 — Multi-label structure analysis (the gap that triggered this plan)

**Why:** 69% of DIVE contracts are multi-label. The prior analysis ignored this. We need to understand: do multi-label contracts have higher, lower, or the same per-class TP rate as single-label contracts? If multi-label contracts are noisier, we may want to filter them out. If single-label contracts are cleaner, we may want to use them preferentially.

**Steps:**

1. For each contract, group by its multi-label count: 1-label, 2-label, 3-label, ..., 8-label.
2. For each class (Reentrancy, Access Control, etc.), compute per-class stats within each group:
   - How many contracts in the group are labeled with this class?
   - What is the positive rate for this class within the group?
3. Cross-tabulate: single-label contracts that are "Reentrancy-only" vs multi-label contracts that are tagged "Reentrancy + X". How do they differ in the source code patterns?
4. Sample 20 contracts from single-label "Reentrancy" only and 20 from multi-label "Reentrancy + others" and manually review: does the TP rate actually differ? **(Manual review requires Method 0
   criteria frozen; record verdict + reason + confidence per CLAUDE.md §5; report rates as
   numerator/denominator + Wilson CI; coordinate/share these samples with Method 4 to avoid double
   review.)**

The script half (steps 1–3, pure counting from the CSV) is **independent of Method 0** and can run
early; only the step-4 manual review is gated on the frozen criteria.

**Deliverable:** Report at `findings/03_multilabel_structure.md` with the per-class per-multi-label-count table and a manual sample comparison.

**Effort estimate (planning only — NOT a constraint; completeness outranks speed):** 4 hours (3 hours script + 1 hour manual review).

### Method 4 — Direct TP rate measurement on the DIVE CSV (not on the agreed subset)

**Why:** The prior manual review measured 5.3% (EB) and 4.2% (RE) on small samples with an unfrozen
bar. To make a defensible decision, we need a stratified sample judged against **frozen criteria
(Method 0)**, **with a control arm**, sampled directly from the DIVE CSV (not from the agreed subset).

**Prerequisite:** Method 0 criteria frozen + Method 1 semantics understood. Do not start otherwise.

**Steps:**

1. **Use the frozen `findings/00_tp_criteria_v1.md` rubric** — do not (re)define criteria here. Record
   the criteria version in the output.
2. Sample, by **seeded RNG (record the seed + exact command)**, stratified as:
   - **Positives** — 50 single-label "Access Control"; 50 multi-label "Access Control"+1 other;
     50 single-label "Reentrancy"; 50 multi-label "Reentrancy"+1 other.
   - **Controls (the §6 negative arm)** — 50 zero-label `__source__` contracts (CSV all-zero). Judged
     blind, mixed in with the positives so the reviewer doesn't know which arm a contract is from.
3. Review **in batches of ~25** (each batch its own mini-gate per CLAUDE.md §1/§3). For each contract
   record **verdict (TP/FP/BORDERLINE) + 1-line reason + confidence (H/M/L)**.
4. **Independent replication by a cold second AI (CLAUDE.md §5):** the first agent assembles a
   self-contained replication package — frozen criteria `00_tp_criteria_v1.md` + the contract list +
   a blank judging template + needed references, and **nothing revealing its own verdicts.** A fresh
   second agent (separate session handed the package by Ali) re-judges ≥10% from scratch under the
   same criteria. Diff the two verdict sets; report the disagreement rate. If >~15% disagree, stop —
   criteria too vague → bump to v2, restart the affected batch. **Ali arbitrates the disagreements**
   (shown both agents' reasoning, pre-analyzed), he is not a routine rater.
5. Compute, per stratum, **TP as numerator/denominator + Wilson 95% CI**. Report the control arm's
   "judged-vulnerable" rate next to the positives' TP rate (the confusion-style picture, §6).
6. **Sample-size note (state up front):** n=50 at a true ~5% gives a Wilson CI ≈ [1%, 14%] — this n
   can separate "≈5%" from "≈25%" but **cannot** separate 4% from 8%. That is sufficient *only if* the
   Method-0 / §8 decision threshold is coarse (e.g. keep ≥15% / drop <5%). If the decision hinges on a
   finer gap, enlarge n per the pre-committed rule before concluding — do not over-read a small sample.
7. Compare to the prior pooled estimate (5.3% / 4.2%) **as a hypothesis check**, with CI overlap
   stated — not as confirmation of a number we already distrust.

**Implementation:** Seeded sample lists → `samples/method4_*.txt` (seed + command recorded in the
finding). Review with `data_module/audit/scratch/smart_summary.py`. Append to
`findings/04_direct_tp_rate.md`. **Coordinate with Method 3:** reuse the same single/multi-label
Reentrancy samples across M3 and M4 so the two Methods are consistent and no contract is reviewed
twice under the same criteria.

**Effort estimate (planning only — NOT a constraint; completeness outranks speed):** 6–9 hours (250 contracts incl. controls × 1.5–2 min each + the blind re-check).

### Method 5 — Cross-source consistency check (DIVE vs SolidiFI vs SmartBugs Curated)

**Why:** If three independent security datasets agree on which contracts are vulnerable, that's a strong signal. If they disagree (DIVE says TP, SolidiFI says FP, etc.), we need to understand the disagreement.

**Steps:**

1. For each contract, look up the sha256 in:
   - `data_module/data/labels/merged/*.labels.json` (the existing merged labels for v3)
2. Per class, compute: how many DIVE-positive contracts are also positive in SolidiFI? SmartBugs Curated? Both?
3. For the contracts that are positive in 2+ sources (DIVE+SolidiFI, DIVE+SmartBugs, SolidiFI+SmartBugs), sample 30 per pair and manually review. If 2-source agreement has a much higher TP rate than 1-source, that's a usable signal.
4. For the contracts that are positive in only 1 source, sample 30 per source and manually review. Single-source TP rate per source.

**Deliverable:** Report at `findings/05_cross_source.md` with agreement matrices and per-source / per-source-pair TP rates.

**Effort estimate (planning only — NOT a constraint; completeness outranks speed):** 4 hours (3 hours script + 1 hour manual review).

### Method 6 — Re-examine the prior Slither+ Aderyn agreement numbers with multi-label stratification

**Why:** The prior agreement rates (42.7% EB, 75.0% RE) were computed on the entire agreed set, not stratified by multi-label count. If single-label "Reentrancy"-only contracts have a much higher agreement rate than multi-label "Reentrancy + 4 others", then Slither agreement is meaningful for the former and not for the latter.

**Steps:**

1. For each class (EB, RE), partition the agreed set by multi-label count: 1-label contracts, 2-label, 3-label, etc.
2. Compute agreement rate per stratum per class.
3. If single-label "Reentrancy" has, say, 95% agreement with Slither while 4-label "Reentrancy + X" has 60% agreement, that changes the picture significantly.
4. Decide: is there a "high-quality subset" we can extract from DIVE based on multi-label structure + Slither agreement?

**Deliverable:** Report at `findings/06_slither_by_multilabel.md` with stratified agreement rates.

**Effort estimate (planning only — NOT a constraint; completeness outranks speed):** 2 hours (script only, no new manual review — this re-uses the existing data).

### Method 7 — Tools available + which (if any) to add (prerequisite for cross-tool Methods)

**Why:** Cross-tool Methods (5, 6) need the tool set fixed first, and independence matters — two tools
that share a blind spot don't validate each other (CLAUDE.md §8). This is a **prerequisite**, not a
closing survey.

**Already verified this session (2026-06-18) — fold in, do not re-survey from scratch:**

| Tool | Status | Location | Note |
|---|---|---|---|
| Slither 0.11.5 | ✓ available | `.venv/bin/slither`, `ml/.venv/bin/slither` | **NOT on bare PATH** — `which slither` returns nothing; call the venv binary. The prior plan's `which slither` step was wrong. |
| Aderyn 0.6.8 | ✓ available | `~/.cargo/bin/aderyn` | Rust, fast. |
| Mythril / Manticore / Echidna / Semgrep / Securify | ✗ not installed | — | install is its own gated step. |

**Steps:**

1. Confirm the inventory above still holds (`.venv/bin/slither --version`, `aderyn --version`).
2. **Decide which 1–2 tools to ADD**, judged on *independence*: does the candidate fail differently
   from Slither/Aderyn (so its agreement is meaningful), or share their surface-pattern blind spot?
   Per CLAUDE.md §8, Mythril is allowed only in **fast/bounded mode on a tiny narrowed set**, never as
   a scanner. Document each candidate's license, detector set, independence rationale, install/run cost.
3. Any actual install is a **separate gated step** (CLAUDE.md §1) — this Method only decides and
   documents; it installs nothing.

**Deliverable:** `findings/07_tools_decision.md` — verified inventory + the add/no-add decision with
independence rationale.

**Effort estimate (planning only — NOT a constraint; completeness outranks speed):** 1 hour.

### Method 8 — v3 parser faithfulness: do the labels the model TRAINED ON match the CSV?

**Why:** Methods 2–6 reason about the raw DIVE CSV/folders. But Run 12 did **not** train on the raw
CSV — it trained on the **parsed v3 labels** emitted by `dive.py` → `data_module/data/labels/dive/`
(**22,073 `.labels.json`**, verified to exist this session). If the parser drops, mis-maps, or
mis-merges, then the entire raw-CSV analysis is beside the point — the model's actual labels could
differ. This Method closes that gap. It is an independent structural check (no Method-0 dependency)
and should run early.

**Steps:**

1. Read `data_module/sentinel_data/labeling/parsers/dive.py` + `crosswalks/dive.yaml` to state the
   intended transform: CSV column (e.g. "Access Control") → canonical class (e.g. ExternalBug).
2. For a **seeded random sample** of contracts (and ideally all 22,073, since it's a cheap file read),
   compare: CSV row → expected canonical labels (apply the crosswalk by hand in the script) **vs** the
   actual `labels/dive/<sha>.labels.json` content. Report per-class agreement as numerator/denominator.
3. Reconcile the counts: raw 22,330 → parsed 22,073 (257 dropped). **Resolve §9 Q4** here — derive
   the true drop count and confirm whether the integration-doc "67" or `dive.yaml` "257" is correct,
   and *which* contracts were dropped and why (compile failure? dedup?).
4. Flag any contract where the parser's emitted label ≠ the crosswalk-applied CSV label — that is a
   parser bug and a high-signal finding.

**Deliverable:** `findings/08_parser_faithfulness.md` — per-class CSV-vs-parsed agreement, the
resolved raw→export drop accounting, and any parser-vs-crosswalk mismatches.

**Effort estimate (planning only — NOT a constraint; completeness outranks speed):** 2 hours (mostly scripting; file reads are cheap).

---

## 5. Directories to investigate

All paths are inside `/home/motafeq/projects/sentinel/`:

| Path | What it contains | Why we need to look |
|---|---|---|
| `data_module/data/raw_staging/dive/` | Canonical DIVE repo (22,330 .sol + 8 class folders) | The actual data; symlink structure already mapped |
| `data_module/data/raw_staging/dive_labels/DIVE_Labels.csv` | Canonical DIVE label source | Authoritative labels per contract per class |
| `data_module/data/raw/dive/ingestion_manifest.json` | Ingest metadata (source, pin, contract_count) | Already inspected — `url: ""` so the source URL must be recovered from the methodology paper |
| `data_module/docs/2026-06-10_data_module_integration_test_dive.md` | Prior integration test doc, cites "Nature Sci. Data 2025" | Find the paper URL |
| `data_module/docs/2026-06-14_data_module_v2_v3_architecture.md` | Data module architecture overview | Background context |
| `data_module/sentinel_data/ingestion/label_folderize.py` | The script that built the folder symlinks from DIVE_Labels.csv | Method 2 — understand what transformation was applied |
| `data_module/sentinel_data/ingestion/connectors/manual_connector.py` | The connector that materialized the raw repo | Method 2 — understand the raw materialization |
| `data_module/data/labels/merged/*.labels.json` | v3 merged labels for all sources | Method 5 — cross-source analysis |
| `data_module/data/labels/dive/*.labels.json` | Per-contract DIVE-derived labels (current v3, **22,073 files** — what the model trained on) | **Method 8** — verify CSV → parser → these labels is faithful |
| `data_module/audit/2026-06-18_dive_*.json` and `.md` | Prior session's audit artifacts | Reference for what was already measured |
| `~/.claude/scratch/externalbug_datamodule_rootcause_20260618.md` | Prior session's working memory (Phase 1-3b) | Reference; do not re-do work already in here |
| `ml/checkpoints/v3_label_quality.json` | Run-12's per-class positive rates (already known) | Background — the 13 FAIL checks from Run 12's label quality gate |

**No file should be modified during this investigation.** Read-only.

---

## 6. References (other docs, papers, code)

| Reference | What it is | Used in |
|---|---|---|
| `docs/plans/2026-06-18_data_module_dive_crosswalk_externalbug_reentrancy_fix_plan.md` | Parent plan from prior session, now superseded by this fresh start | Background; do not extend, do not contradict |
| `data_module/docs/2026-06-10_data_module_integration_test_dive.md` | Prior integration test doc | Method 1 (find the paper) |
| `data_module/docs/2026-06-14_data_module_v2_v3_architecture.md` | Data module architecture | Background |
| `data_module/sentinel_data/labeling/parsers/dive.py` | Current DIVE label parser (in v3) | **Method 8** — verify CSV → parsed-label faithfulness; understand current logic |
| `data_module/sentinel_data/labeling/crosswalks/dive.yaml` | Current DIVE crosswalk YAML (in v3) | Method 8 — the intended class mapping to validate against |
| `data_module/sentinel_data/verification/aderyn_runner.py` | Aderyn runner (cached, per-file temp dir) | Method 7 — already available |
| `data_module/sentinel_data/verification/slither_runner.py` | Slither runner (cached, Python API) | Method 7 — already available |
| `data_module/sentinel_data/verification/negative_checker.py` | Corpus-level negative/control checker | **Method 4 control arm** (CLAUDE.md §6) |
| `data_module/audit/scratch/smart_summary.py` | Per-contract source summariser | Methods 3/4 manual review helper |
| `ml/testing_specs/label_quality.py` | Run 12's pre-launch label quality check | Background — already implements >50%/>1%/single-source gates |
| `~/.claude/projects/-home-motafeq-projects-sentinel/memory/MEMORY.md` | Project memory | Background |
| `~/.claude/scratch/externalbug_datamodule_rootcause_20260618.md` | Prior session's working memory | Reference; do not duplicate |
| Nature Sci. Data 2025 paper (URL TBD via Method 1) | DIVE methodology | Method 1 |

---

## 7. Deliverables (everything in this folder)

```
docs/plan/data_module/2026-06-18-dive-data-source-quality-investigation/
├── README.md                                      ← this file
├── 00_executive_summary.md                        ← written at the end, after all findings
├── findings/
│   ├── 00_tp_criteria_v1.md                       ← Method 0 (frozen, Ali-signed rubric)
│   ├── 01_dive_methodology.md                     ← Method 1
│   ├── 02_folder_csv_agreement.md                 ← Method 2
│   ├── 03_multilabel_structure.md                 ← Method 3
│   ├── 04_direct_tp_rate.md                       ← Method 4 (incl. control arm + blind re-check)
│   ├── 05_cross_source.md                         ← Method 5
│   ├── 06_slither_by_multilabel.md                ← Method 6
│   ├── 07_tools_decision.md                       ← Method 7 (inventory + add/no-add decision)
│   └── 08_parser_faithfulness.md                  ← Method 8 (CSV→parser→v3 labels + drop accounting)
├── samples/
│   └── (SEEDED sample lists generated by Methods 3, 4, 5, 6, 8 — each records seed + command)
├── scripts/
│   ├── verify_folder_csv_agreement.py             ← Method 2
│   ├── multilabel_structure.py                    ← Method 3
│   ├── direct_tp_rate_sampler.py                  ← Method 4
│   ├── cross_source_analysis.py                   ← Method 5
│   ├── slither_by_multilabel.py                   ← Method 6
│   └── parser_faithfulness.py                     ← Method 8
└── (any ad-hoc scratch files)
```

Every script writes a **provenance header** (CLAUDE.md §10): exact command, RNG seed, tool versions,
input corpus + count/hash + frame, output path, timestamp. A number without provenance is UNVERIFIED.

---

## 8. Success criteria (what "done" means for this plan)

This plan is complete when:

1. ✅ All 9 finding files (`00_tp_criteria_v1` … `08_parser_faithfulness`) exist, are non-empty, and
   cite the actual data + provenance (command/seed/tool-version/frame) that backs each claim
2. ✅ Every TP rate is reported as numerator/denominator + Wilson CI, **with its control-arm rate
   beside it**, and no batch rate is extrapolated to the corpus without Ali's explicit OK
3. ✅ The Method-4 independent replication by a cold second AI was run (self-contained package, no
   access to the first agent's verdicts) and the agent-vs-agent disagreement rate is recorded
4. ✅ Each finding includes a 1-paragraph "what this means for the label-source decision" conclusion
5. ✅ `00_executive_summary.md` ties the findings into a single recommendation, **evaluated against
   the decision rule pre-committed below** — not a post-hoc judgement
6. ✅ Total time spent is documented per method
7. ✅ The recommended next-step plan is sketched (NOT executed) — what comes after this investigation

**Pre-committed decision PROCEDURE (the procedure is fixed now; the numeric threshold is calibrated
from the data during the run — CLAUDE.md §7, dynamic-but-anti-post-hoc):**

No hardcoded magic number. Each stratum is judged against its **own empirically-measured null** — the
TP rate the *same frozen criteria + same tools* produce on the **control arm** (known-clean contracts,
§6). For each (class, stratum), using Method-4 Wilson 95% CIs:

- **KEEP** the stratum as a label source only if its TP CI is **statistically distinguishable from and
  above the control-arm (null) CI** — i.e. the two CIs **do not overlap** and the stratum is higher.
- **DROP** if the stratum's CI **overlaps the null** (indistinguishable from noise) or sits below it.
- **ENLARGE the sample** if the CIs are too wide to separate, then re-evaluate — never force a call on
  an underpowered sample.
- A stratum that is KEPT only survives if **Method 8** confirms the parsed v3 label faithfully reflects
  it — a KEEP on top of a buggy parser is not a KEEP.
- Any optional practical floor ("even if significant, a TP rate this low isn't worth the data") is
  **proposed by the AI with reasoning and confirmed by Ali at Method 0**, before results are seen, and
  recorded as part of the frozen procedure.

The numeric keep/drop boundary therefore **falls out of the measured null**, not a guess — dynamic, yet
fixed by a rule chosen before any number is seen, so the final call is arithmetic, not opinion.

This plan does NOT include: any code change to the data module, any re-export, any change to Run 13, any decision on Option A/B/C.

---

## 9. Open questions (to resolve before / during this plan)

1. **TP criteria — NOW PROMOTED TO Method 0** (no longer a free-floating question). The bar
   (pattern-present vs reachable vs exploitable, per class) + the BORDERLINE bucket + worked examples
   are frozen and Ali-signed in `findings/00_tp_criteria_v1.md` before any sampling. The §8 decision
   thresholds are confirmed at the same sign-off.
2. **What to do if multi-label structure changes the picture:** if Method 3 shows single-label
   contracts are clean and multi-label contracts are noise, do we want to filter multi-label out?
   Discuss before Method 4. (The §8 decision rule already handles this per-stratum.)
3. **What to do if the DIVE paper says "manual expert audit":** then the prior ~4% TP claim is much
   more concerning. If the paper says "tool-derived pattern matching", then a low TP rate may be
   expected. Method 1 will resolve this; it gates the interpretation of every later TP number.
4. **Doc-vs-doc drop-count conflict — NOW RESOLVED BY Method 8** (was "fold into Method 2"). The
   integration-test doc (`…_integration_test_dive.md:136`) says **67** dropped (0.3%); `dive.yaml:26`
   says **257** compile failures (22,330 → 22,073). The parsed-label count is **22,073** (verified this
   session), so the true drop is 257 and "67" is likely stale — Method 8 confirms which contracts
   dropped and why. Neither doc is trusted until Method 8 derives it (CLAUDE.md §2).

---

## 10. Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| The DIVE paper is paywalled / hard to find | Medium | Method 1 stalls | Use preprint search (arxiv, semanticscholar), GitHub search for "DIVE smart contract" |
| Method-4 review (vs frozen criteria) shows the same low TP rate, confirming the prior conclusion | Medium-high | Plan "wasted"? No. | Valid outcome — same conclusion now with frozen criteria + control arm + blind re-check, fully auditable. "Drop DIVE for EB+RE" may still be right, but now defensibly. |
| Method-4 review shows a high TP rate in some stratum, refuting the prior conclusion | Low-medium | Big — parent plan needs revision | Great outcome — we recover a usable DIVE signal (per the §8 KEEP rule). |
| **Reviewer disagreement >15% on the blind re-check** | Medium | Criteria too vague → every TP number is soft | Stop, bump criteria to v2 with sharper examples (Method 0), restart the affected batch. Built into Method 4. |
| **Control arm flags many known-clean contracts as vulnerable** | Medium | The criteria/tools are too loose — a "low TP" might be a detector artifact, not real mislabeling | The §6 control arm catches exactly this; report the confusion-style picture, don't read TP-on-positives alone. |
| **Method 8 finds the parser ≠ crosswalk** | Low | v3 train labels differ from the CSV we analyzed → raw-CSV analysis partly moot | Escalate immediately; the parsed labels (not the CSV) are what the model saw. |
| Aderyn / Slither agreement analysis (Method 6) reveals a useful high-quality subset | Low | Changes the recommendation | Document the subset precisely (with independence caveat, CLAUDE.md §8), propose a follow-up plan. |
| CSV/folder per-contract mismatch (Method 2) despite matching counts | Low | Folder-based prior analysis used wrong membership | Counts already match (§2.2); a per-contract swap would still surface here — escalate if found. |

---

## 11. Execution log — gated per the governing protocol

Each row is one gate cycle (explain → confirm → execute → clarify → confirm). Append rows as work
happens; do not batch multiple Methods into one row.

| # | Method | Explained | Ali confirmed start | Executed | Findings clarified | Ali confirmed close | Notes |
|---|---|---|---|---|---|---|---|---|
| 0a | Spec-file claim audit (README + CLAUDE.md) | ✓ | ✓ (Ali asked) | ✓ | ✓ | ✓ | Verified all §2 structural counts exact; found raw-vs-export frame trap + 67-vs-257 drop-count conflict; reframed §2.3 as unverified. Read-only. |
| 0b | Spec-file coverage/quality audit + fixes | ✓ | ✓ (Ali asked) | ✓ | ✓ | ✓ | Found 6 HIGH + MED/LOW coverage gaps; added Method 0 (criteria), Method 8 (parser faithfulness), control arm, blind re-check, pre-committed decision rule, dependency map, provenance; CLAUDE.md gained Concerns F/G/H + stat-reporting + provenance. Notes: `~/.claude/scratch/dive_specfiles_coverage_audit_20260618.md`. |
| 8 | Method 8 — Parser faithfulness | ✓ | ✓ (Ali asked) | ✓ | ✓ (corrected) | ✓ | Parser faithful for all 7 DIVE-sourced classes. DoS 2,655 mismatch = intentional patch. Self-critique: initial misdiagnosis corrected. Deliverable: `findings/08_parser_faithfulness.md` + `scripts/parser_faithfulness.py`. |
| 2 | Method 2 — Folder↔CSV per-contract identity | ✓ | ✓ (Ali asked) | ✓ | ✓ | ✓ | 0/178,640 mismatches. 100% per-contract agreement. Chain of trust validated. Deliverable: `findings/02_folder_csv_agreement.md` + `scripts/verify_folder_csv_agreement.py`. |
| 1 | Method 1 — DIVE methodology | ✓ | ✓ ("go on") | ✓ | ✓ | ✓ | DIVE = fully automated tool-derived consensus. Authors: "systematically derived, high-confidence annotations rather than manually verified ground truth." Deliverable: `findings/01_dive_methodology.md`. |
| 0 | Method 0 — Freeze TP/FP/BORDERLINE criteria v1 | ✓ | ✓ (Ali asked) | ✓ | ✓ | ✓ | REACHABLE bar. BORDERLINE bucket. Practical floor ≥10%. 42 contracts in 3 batches + Slither/Aderyn hints. Criteria v1 frozen. Deliverable: `findings/00_tp_criteria_v1.md`. |
| 7 | Method 7 — Tools available + add decision | ✓ | ✓ (implicit from "lets add Echidna") | ✓ | ✓ | ✓ | Slither 0.11.5 + Aderyn 0.6.8 working. Echidna 2.3.2 installed but requires assertions DIVE lacks. No additional tools justified. References: slither_reference.md, aderyn_reference.md, echidna_reference.md. Deliverable: `findings/07_tools_decision.md`. |
| 3 | Method 3 — Multi-label structure analysis | ✓ | ✓ (Ali asked) | ✓ | ✓ | ✓ | EB = near-universal flag (97% at 3+ labels). RE 97.7% multi-label. Single-label NOT cleaner — same meme-token BORDERLINE pattern. Single-label hypothesis REFUTED. Deliverable: `findings/03_multilabel_structure.md`. |
| 4 | Method 4 — Direct TP rate (150 contracts) | ✓ | ✓ (Ali asked) | ✓ | ✓ | ✓ | 0 TPs in 150 contracts (50+50+50). Wilson CI [0%, 7.1%] = control null. DROP both RE strata. Deliverable: `findings/04_direct_tp_rate.md`. |
| — | **PHASE 1 COMPLETE** | — | — | — | — | — | **Verdict: DROP DIVE EB+RE labels. See `00_executive_summary.md`.** M5, M6 skipped (moot after M4 DROP). Phase 2 (remaining 5 DIVE classes) and Phase 3 (other sources) deferred to separate plans. |

- **2026-06-18 ~20:00 UTC**: Plan written.
- **2026-06-18 (later)**: Governing protocol written as the subfolder's `docs/plan/data_module/CLAUDE.md` (generalized to any dataset/source; adds distrust-docs, manual-review batching + no-extrapolation, bullet-proof gates, tooling policy). This README updated to remove the "approve all 7 methods upfront" framing and replace it with per-Method gating.
- **2026-06-18 (spec claim audit)**: Audited both spec files against raw data (read-only). Verified every §2 count exact (folder symlinks, 22,330 CSV rows, multi-label distribution, per-class counts == folder counts). Found + fixed: (a) raw-DIVE (22,330) vs v3-export (22,073) frame trap, (b) integration-doc "67 dropped" vs `dive.yaml` "257 compile failures" conflict → §9 Q4, (c) reframed §2.3 prior numbers as unverified hypotheses. Working notes: `~/.claude/scratch/dive_specfiles_audit_20260618.md`.
- **2026-06-18 (spec coverage audit + fixes)**: Audited both files for *coverage*, not just claim accuracy. Applied all HIGH+MED+LOW fixes. **README** gained: Method 0 (freeze TP/FP/BORDERLINE criteria as a prerequisite gate), Method 8 (v3 parser-faithfulness — the 22,073 labels the model actually trained on), a control/negative arm + blind ≥10% re-check + sample-size justification in Method 4, a dependency map (M0/M1/M7 gate M3–M6), a pre-committed §8 decision rule, provenance requirements, and corrected Method-7 tool facts. **CLAUDE.md** gained: Concern F (criteria frozen before counting), Concern G (single-reviewer bias mitigation), Concern H (control arm), a stat-reporting law (n + Wilson CI + CI-overlap), pre-committed decision rules, and a provenance/reproducibility mandate. Working notes: `~/.claude/scratch/dive_specfiles_coverage_audit_20260618.md`.
- **2026-06-18 (Ali steering round)**: Applied 4 refinements. (1) **Reusable layer reframe** — this is a standing verification layer for ANY source feeding the expensive modules; completeness outranks speed, no time pressure; "time budgets" → effort estimates. (2) **Decisions pre-analyzed** — every decision routed to Ali arrives with the AI's explicit recommendation + reasoning, never a blank-page question (CLAUDE.md §1; applied to Method 0 criteria + decision procedure). (3) **Dynamic decision threshold** — replaced fixed 15%/5% with a calibrated-null rule: keep a stratum only if its TP CI beats the control-arm CI (procedure fixed in advance, number falls out of measured null). (4) **Cold second-AI replication** — Method-4 re-check is done by a fresh second agent given a self-contained package with no access to the first agent's verdicts; Ali arbitrates disagreements. (5) **Phased scope** — §0 added: Phase 1 EB/RE (now) → Phase 2 full DIVE → Phase 3 other sources.
- **2026-06-18 (causal-diagnosis hardening — triggered by the Method 8 DoS misdiagnosis)**: The DoS "parser bug" was actually the documented 2026-06-13 DoS+RE co-occurrence patch; the agent followed the protocol and still produced a confident wrong root cause, because the protocol verified the WHAT but not the WHY. Both files hardened against this whole class of failure. **CLAUDE.md** gained: §1 "WHAT vs WHY are separate claims / plausibility ≠ evidence / rule out intentional changes first / hedges survive into summary"; §2 "distrust ≠ ignore — git+MEMORY+archived-scripts are a REQUIRED lead" + "examine the full record (tier/source fields), not one value"; §7 "causal claims gated like numbers", "a recommended fix must be checked for what it reverts/breaks", "independently re-derive key numbers before close", "scripts read live config, no silent hardcoded copies"; §10 "route cross-system findings to their tracker", "progressive docs must be checkable"; +11 new checklist items. **README** gained: §2.5 known-intentional-interventions registry (DoS patch listed with its `tier:null` signature), and a standing "Diagnosing a discrepancy" rule in the Methods intro. **No investigation Method beyond the now-corrected Method 8 has been started.**
