# CLAUDE.md — Data Source Investigation & Audit Protocol (data_module)

**Scope:** This file governs **all** investigation, audit, verification, and quality-analysis work
under `docs/plan/data_module/` — for **any** dataset or label source (DIVE, BCCC, SolidiFI, SmartBugs,
CGT, future sources). It is not DIVE-specific. Read it at the start of any session that touches data
source quality. It OVERRIDES default behavior and complements (does not replace) the root `CLAUDE.md`.

**This is a STANDING VERIFICATION LAYER, not a one-off audit.** The procedure defined here is a
permanent, reusable quality gate that every data source must pass before it is trusted by the
expensive downstream modules (training, ZK, on-chain). A bad label that slips through here poisons a
multi-day training run, a benchmark, and every decision built on it (BUG-2 is the proof). Therefore:

> **Completeness and correctness outrank speed — always.** There is no time budget. A verification
> that takes 35 hours, or a week, is fine if that is what trustworthiness costs. Never shortcut a step,
> skip a control, or trim the method set to "save time." The "time budgets" in any plan are estimates
> for planning, **not constraints** and **not licence to cut corners**. If a choice is between "faster"
> and "more certain," choose more certain.

**Status:** ACTIVE since 2026-06-18.
**Origin:** Ali does not trust the prior DIVE/BCCC findings as presented (batched, summarized,
extrapolated, single-reviewer, criteria not frozen). This protocol changes *how* the work is done so
that trust is earned at each step.

---

## 0. The one-line spirit

**Nothing is true until we verified it ourselves, on the actual data, against criteria we froze in
advance, with explicit numbers Ali can see — and we earn that trust one small gated step at a time,
never in a batch.**

---

## 1. The non-negotiable per-step contract

Every investigation step (a "Method", a sub-task, a tool run, a sample batch) follows this cycle.
Never skip a phase, never merge two steps into one approval.

1. **EXPLAIN (before doing anything):** State in plain terms — what will be tested/analyzed/audited,
   on exactly what data (paths, counts), with what method and tool, what is expected, what the
   deliverable file will be, and roughly how long. Enough that Ali can judge the step before it runs.
2. **CONFIRM start:** Wait for an explicit go ("yes" / "go" / "looks good"). Silence, an ambiguous
   reply, or a follow-up question is **not** confirmation. Continued questions are not a nod.
3. **EXECUTE — no ghost assumptions:** Run exactly what was described. No silently swapping the
   method, no skipping sub-parts, no inventing/filling values, no "I assumed X". If the step cannot
   be done as described, **stop and surface it** — do not improvise around it silently.
4. **CLARIFY findings:** Explain what was found in full — the actual numbers, the sample size, the
   caveats, and **what was NOT covered or NOT verified**. Bring Ali to the same understanding the
   analysis produced. Not a tidy summary that hides nuance.
5. **CONFIRM close:** Wait for Ali's explicit confirmation that he's at the same point before the
   next step starts.

**One step at a time. Never jump or batch.** The failure mode we are correcting is exactly the pile
of bundled, pre-summarized findings that lost Ali's trust in the BCCC/DIVE work.

**Decisions routed to Ali arrive PRE-ANALYZED, with an explicit recommendation — never as a bare open
question.** Any point that needs Ali's judgement (criteria definitions, thresholds, scope, keep/drop,
tool choices) must first be worked through by the AI: lay out the options, the evidence for each, the
trade-offs, **and the AI's own explicit recommended answer with its reasoning.** Ali then either makes
a better-informed decision *or* simply confirms the AI's recommendation. Dumping an unanalyzed
question on Ali ("which TP definition do you want?") is forbidden — the AI does its homework first and
proposes ("here is the bar I recommend per class, and why; confirm or correct"). This applies to every
decision gate in every Method. It does **not** mean the AI decides unilaterally — Ali still approves;
it means Ali is never asked to think from a blank page.

**A finding's WHAT and its WHY are SEPARATE claims — both must be verified, and a plausible story is
NOT evidence.** Measuring *what* is true (counts, rates, which contracts) is not the same as
establishing *why* it is true (the root cause). The WHAT can be hand-checked; a WHY **cannot** be
confirmed by spot-checking the WHAT. This is the exact trap that produced a confident misdiagnosis on
this very investigation (a real label discrepancy was correctly counted, then assigned a fabricated
"parser cache bug" cause that was actually a documented, intentional patch). Therefore:
- A root-cause / "why" statement is a **claim that needs its own evidence** — or it is labeled an
  explicit **HYPOTHESIS** ("candidate explanation, not yet confirmed"), never stated as fact.
- **Plausibility is not evidence.** A coherent, internally-consistent, confidently-worded explanation
  is *exactly* what a wrong root cause looks like. The more fluent the story, the more suspicious you
  should be that it was constructed rather than found.
- **Prefer "I don't know yet — here are the candidate causes" over a confident guess.** An honest
  unknown is safe; a plausible fabrication poisons every decision built on it.
- **Before calling any discrepancy a bug/error, you must rule out that it is intentional and
  documented** (§2 history-as-lead). A discrepancy whose size/date/signature exactly matches a known
  prior intervention is almost certainly that intervention, not a new bug.
- **Uncertainty must survive into the summary.** If the finding file hedges ("likely", "probably"),
  the CLARIFY message to Ali carries the same hedge — never flatten a hedged finding into a flat
  assertion ("root-caused to X").
- **Never propose a fix without checking what it would revert or break** (see §7).

---

## 2. Concern A — Distrust all written docs and reports (verify ourselves)

This extends root `CLAUDE.md` Rule 4 to data and reports specifically.

- **Treat every written artifact as a hint or clue, not a fact** — until we re-verify it ourselves on
  the actual data. This includes: prior audit `.md` files, prior `.json` result dumps, MEMORY.md
  claims, the parent DIVE plan, the BCCC audit, dataset README/paper claims, docstrings, and any
  number quoted in a previous session (including by a previous Claude).
- A prior finding may be **cited as a hypothesis to test** ("the prior session reported 4% TP — let's
  see if that reproduces"), never **asserted as established** ("TP is 4%, so…").
- Canonical truth is: the raw data on disk, the source code that processes it, and the output of a
  tool/script we ran ourselves in this session and can show Ali. Nothing else.
- When a verified result **contradicts** a prior doc, that contradiction is itself a finding — record
  it explicitly (what the doc said, what we found, why they differ). Do not quietly overwrite.
- **Always know which *frame* a number is on before comparing it to another number.** A count can be
  computed on the **raw corpus**, on the **post-dedup / post-compile-filter subset**, or on the
  **final export** — and these differ. (Real example, DIVE: raw CSV "Access Control"=16,723, but the
  v3 export is 22,073 contracts and reports 16,582 ExternalBug-positive — same concept, different
  frame, 141-contract gap.) Before stating or comparing any figure, name its frame (raw / staged /
  filtered / exported) and its denominator. Two figures on different frames are not a discrepancy and
  must not be presented as one; conversely, a same-frame mismatch is a real finding.
- **Distrust ≠ ignore. The project's own history is a REQUIRED investigative lead, not optional.**
  "Treat docs as hints, not facts" means distrust their *conclusions* — it does **not** mean skip
  reading them. Before forming ANY explanation of why data looks the way it does, you **must** consult,
  as leads to verify against the data: `git log` / `git blame` on the affected files/directories,
  `MEMORY.md` + the `project_*.md` memory files, any archived or `temp/` patch and one-off scripts, and
  the changelogs. The reason data is shaped a certain way is *very often a documented, intentional
  change* — and the doc that records it is a clue to confirm against the data, not a claim to dismiss.
  Failing to read history is how a deliberate, recorded patch gets misdiagnosed as a fresh bug.
- **Examine the FULL record, not just the one field you're comparing.** When diagnosing a discrepancy,
  read every relevant field / metadata on the artifact — provenance fields (a label's `tier`,
  `source`), timestamps, flags — not only the single value that differs. The signature of an
  intentional change is usually sitting in a field you weren't looking at (e.g. a deliberately-zeroed
  label carries `tier: null`, which alone identifies it as a patch rather than an error).

---

## 3. Concern B — Manual `.sol` review is batched, findings are NOT extrapolated, rates are reported honestly

When manually reading `.sol` contracts to judge true-positive / false-positive:

- **Review in separate, explicit batches** (e.g. 20–30 at a time), each batch its own gated sub-step
  under §1. Do not review "200 contracts" as one opaque action — Ali cannot audit that.
- **Record a verdict + a 1-line reason per contract** (TP / FP / BORDERLINE + why), so every verdict
  is inspectable, not just a count. See §5 for the BORDERLINE bucket and §4 for the criteria.
- **DO NOT extrapolate a batch rate to the whole corpus.** The pattern *"100 samples agreed at 5%,
  therefore ~1,000 of the 20k agree"* is **forbidden** unless (a) it is genuinely necessary for a
  decision **and** (b) Ali has **explicitly confirmed** that extrapolation for that specific case.
  Until then, a batch result is a statement about **that batch only**: "in this batch of 30, 2 were
  TP" — full stop. No projected corpus totals, no implied percentages applied to the full set.
- **Stat-reporting law (always, even within a single batch):**
  - Report every rate as **numerator / denominator** with the raw **n**, not just a percent
    ("2 / 30", not "≈7%"). Always show the denominator.
  - When a percent is given, attach a **95% confidence interval** (Wilson interval — robust for small
    n and near-0% / near-100% rates). A bare point estimate from a small sample is misleading.
  - **Never compare two rates without checking whether their CIs overlap.** "Stratum A is 4%, B is 8%"
    is not a real difference if both CIs are [1%, 15%]. State overlap explicitly.
  - **Know what precision a sample size buys before sampling.** n=50 at a true 5% gives a Wilson CI of
    roughly [1%, 14%] — it **cannot** distinguish 4% from 8%. Pick n against the decision threshold
    (§7 decision rules): coarse decisions ("4% vs 15%") tolerate small n; fine ones do not.
  - Different strata get **separate** rates (single-label vs multi-label, per-source, per-tier).
    Never pool dissimilar strata into one headline number.

---

## 4. Concern F — Criteria are frozen BEFORE counting (this is the crux)

The entire reason the prior findings are distrusted is that **what counts as a "true positive" was
never pinned down and agreed.** Different implicit bars produce different "TP rates", and none can be
trusted. Therefore:

- **Before any sampling or TP/FP judgement begins, the criteria must be: (1) written down, (2) agreed
  with Ali, (3) frozen for that measurement.** Changing the bar mid-measurement (or after seeing
  results) is forbidden — that is moving the goalposts and it is the post-hoc trap we are escaping.
- The criteria document must state, **per vulnerability class**, the explicit definition of a true
  positive — and must resolve the hard question directly:
  - **Pattern-present** (the vulnerable code construct literally appears) vs
  - **Reachable** (the construct is on a path that can actually execute) vs
  - **Exploitable** (an attacker can profit / cause harm in context).
  These give very different TP rates. The chosen bar must be named per class and justified. (This is
  exactly where DIVE's ExternalBug breaks: "in the Access Control folder" ≠ "has an exploitable access
  control flaw".)
- **The AI proposes the bar; Ali confirms (per §1).** For each class, the AI does not ask Ali "what
  should the TP definition be?" — it presents a **recommended** definition (pattern/reachable/
  exploitable), the reasoning, the trade-off (looser bar → higher recall but more noise; stricter →
  cleaner but may discard real-but-subtle bugs), and worked examples. Ali confirms or corrects. Ali
  starts from the AI's analysis, never a blank page.
- **A third bucket, BORDERLINE / UNSURE, is mandatory.** Forcing every contract into TP-or-FP hides
  genuine uncertainty and biases the rate. Borderlines are counted and reported separately, never
  silently folded into TP or FP.
- Include **worked examples**: at least 2 contracts pre-labeled TP, 2 FP, 1 BORDERLINE, with the
  reasoning, so the bar is concrete and a second reviewer can apply it identically.
- The frozen criteria file is cited by every measurement that uses it. If the criteria genuinely must
  change, that is a **new, named criteria version**, and any prior measurement under the old version
  is re-labeled with its version — not retconned.

---

## 5. Concern G — Single-reviewer bias is actively mitigated

The prior conclusions came from **one reviewer** applying **unfrozen, strict** criteria. Even with
frozen criteria (§4), one rater is a single point of failure. Mitigations are mandatory for any
measurement that feeds a decision:

- **Independent replication by a COLD second AI agent (the primary anti-bias mechanism).** The first
  agent does NOT get to be the only judge. For any batch that feeds a decision, the first agent
  assembles a **self-contained replication package** and a second, fresh agent re-judges from scratch:
  - The package contains everything needed to judge independently and **nothing that reveals the first
    agent's verdicts:** the frozen criteria (`00_tp_criteria_vN.md`), the contract list (or sources),
    a blank judging template, and the references/context needed — but **not** the first agent's TP/FP
    calls, reasons, or counts.
  - The second agent judges the full batch (or a random ≥10% subsample for large batches) under the
    same frozen criteria, producing its own verdict + reason + confidence per contract, with no access
    to the first agent's output and no instruction to agree.
  - The two verdict sets are then diffed. **Disagreement rate is reported.** If they disagree on >~15%
    of contracts, the criteria (§4) are too vague → stop, bump to a new criteria version with sharper
    worked examples, restart the affected batch.
  - **Ali is the final arbiter**, not a routine rater: he is shown the *disagreements* (with both
    agents' reasoning, pre-analyzed per §1) and resolves them. He may also spot-check agreements.
  - Operationally: the second agent is a separate Claude session/instance handed the package by Ali,
    OR a clearly separate, context-isolated pass. The point is genuine independence — the second judge
    must not inherit the first's conclusions or context.
- **Record per-verdict confidence** (high / medium / low) alongside TP/FP/BORDERLINE. A rate built
  mostly from low-confidence calls is itself a finding.
- **Criteria-first, data-second:** apply the frozen rubric; do not invent a new reason to call
  something FP after the fact. If a contract reveals the rubric missed a case, that triggers a
  criteria-version bump (§4), not a silent one-off judgement.
- **No cherry-picking the sample.** Samples are drawn by a seeded RNG (§10 provenance), never
  hand-picked. The seed and selection command are recorded so the exact sample is reproducible.

---

## 6. Concern H — Every TP measurement has a control / negative arm

A true-positive rate measured **only on positives** is uninterpretable on its own — you cannot tell
"the dataset over-labels" from "our tools/criteria under-detect" without a baseline.

- For any class where we measure TP rate on labeled-positives, **also measure on a control set of
  labeled-negatives** (e.g. DIVE's zero-label `__source__` contracts, or SolidiFI-clean) using the
  **same frozen criteria and same tools**. This calibrates the tool/criteria false-positive and
  false-negative behavior.
- Report the **confusion-style picture**, not a lone TP rate: of N positives, how many we judged TP;
  of M controls, how many we (wrongly) judged vulnerable. A "5% TP on positives" reads very
  differently if the same criteria also flag 4% of known-clean contracts (criteria too loose) vs 0%
  (positives genuinely mostly mislabeled).
- The existing `negative_checker.py` (see §9) is the corpus-level control tool — use it, don't
  hand-roll a weaker one.

---

## 7. Concern C — Gates, self-verification, and pre-committed decision rules must be bullet-proof

Every step, plan, or analysis must carry its own gate / self-verification — and the gate must be
**evidence-showing, not box-ticking**.

- A gate that only prints **"tests passed" / "OK" / a green check is not acceptable.** A gate must
  surface, explicitly and in the open:
  - **what** was checked (the exact assertion),
  - the **actual numbers/stats** behind it (counts, rates, distributions — not just pass/fail),
  - **why** it passed or failed (the threshold and the measured value side by side),
  - **what it did NOT check** (so a pass is never mistaken for "everything is fine").
- **No workarounds that manufacture a pass.** Skipping a contaminated test set and calling it
  "verified", loosening a threshold to get green, sampling only the easy stratum, or substituting a
  proxy metric without saying so — all forbidden. If a check cannot be run honestly, the gate
  reports **UNVERIFIED**, not PASS. (This is exactly the C.2.1 failure: a 95.8%-contaminated set was
  treated as a clean benchmark and the gate was effectively skipped — never again.)
- **Self-verification of our own scripts:** before trusting a number our own script produced, sanity-
  check it (row counts reconcile, totals add up, a hand-checked example matches the script's verdict).
  Show that reconciliation, don't assume the script is right.
- Prefer **counts over percentages**, and always show the **denominator**.
- **Pre-commit the decision PROCEDURE before seeing results — but let the numeric threshold be
  calibrated from the data (dynamic, not hardcoded).** A fixed magic number ("keep if ≥15%") is both
  arbitrary and brittle. The rigorous and dynamic way is to **compare each stratum against its own
  empirically-measured null**, not against a guessed constant:
  - The **control arm (§6)** measures the null: what TP-rate do the *same frozen criteria + same
    tools* produce on KNOWN-CLEAN contracts? That is the noise floor for this exact setup.
  - The pre-committed *rule* (fixed in advance) is then: **a stratum is KEPT only if its TP-rate is
    statistically distinguishable from, and above, the measured null** — i.e. the stratum's TP Wilson
    CI and the control's CI **do not overlap**, and the stratum is the higher one. **DROP** if the
    stratum's CI overlaps the null (indistinguishable from noise) or sits below it. **ENLARGE the
    sample** if the CIs are too wide to separate.
  - This is **both dynamic and anti-post-hoc:** the *procedure* (beat the measured null with
    non-overlapping CIs) is frozen before any result is seen, while the *number* it resolves to is
    calibrated by the control arm and the observed distribution during the run. Nobody picks a
    threshold after seeing the answer; the threshold falls out of the data via a rule fixed in advance.
  - Any additional thresholds (e.g. a practical floor like "even if significant, a TP rate this low
    isn't worth keeping") are likewise **proposed by the AI with reasoning and confirmed by Ali**
    (§1) before results are seen, and recorded as part of the frozen procedure.
- **Causal claims are gated exactly like numeric claims (§1).** "It passed" for a *number* is not
  enough if the *explanation* attached to it is an unverified story. A "root cause: X" in a finding
  does not clear its gate unless it is evidenced — candidate alternatives considered and ruled out,
  project history checked (§2) — or explicitly stamped HYPOTHESIS.
- **A recommended fix must be checked for what it would REVERT or BREAK, before you propose it.**
  Before suggesting any remediation (re-run, `force=True`, overwrite, relabel, delete, re-export),
  verify it does not undo intentional prior work or break an existing test. (Concrete case from this
  investigation: a proposed `force=True` parser re-run would have *reverted* a deliberate, documented,
  test-protected label patch.) A fix proposed without this check is a liability, not help. State, with
  the recommendation, what you checked and what it would touch.
- **Independently re-derive the key numeric claims before close — standing, not optional.** For any
  Method whose result feeds a decision, the headline numbers must be reproduced by an *independent
  path* — a second script written from scratch, or a second agent — not merely re-read from the first
  agent's output. This is the numeric-claim analogue of §5's cold-second-AI for judgement calls.
- **Scripts read the live source of truth; they do not silently hardcode copies.** A script that bakes
  in a copy of an external config (e.g. the `dive.yaml` crosswalk) must either load it live at runtime,
  OR assert its hardcoded copy equals the live file at runtime, OR flag the copy explicitly as a known
  limitation in the finding. A silent stale copy drifts the instant the source changes and nothing
  catches it.
- **Re-use the existing gate machinery rather than reinventing a shallow one** (§9).

---

## 8. Concern D — Tooling: what we have, what to add

We need more independent signal than Slither alone. Independence matters: two tools that share a
blind spot (fire on the same surface pattern) do **not** validate each other — corroboration only
counts when tools fail *differently*.

**Verified available in this environment (2026-06-18, checked, not assumed):**

| Tool | Version | Location | Notes |
|---|---|---|---|
| Slither | 0.11.5 | `.venv/bin/slither` and `ml/.venv/bin/slither` | NOT on bare PATH. Uses `solc` via crytic_compile. |
| Aderyn | 0.6.8 | `~/.cargo/bin/aderyn` | Rust. **Input must be a directory** (single-file mode documented but returns `Not a directory` on v0.6.8). Compiles most 0.4.x (8/10 tested) and all 0.8.x contracts successfully using `foundry-compilers` + `solidity-ast-rs`. Rare parser failures on specific 0.4.x syntax (`throw`, early 0.4.11 — hits `BlockOrStatement` enum error). **0 issues found on all 17 DIVE contracts tested** (0.4.x and 0.8.x) — including confirmed CEI-vulnerable meme token patterns and unprotected proxy. `reentrancy_state_change` detector exists but fires on none of our contracts; detector signature (CFG + `is_extcallish()` + `ApproximateStorageChangeFinder`) is too narrow for DIVE patterns.
| solc | versions managed by `solc-select` | `.venv/bin/solc` (symlink managed by solc-select) | **Not on bare PATH.** Symlink managed by `solc-select use <version>`. Slither's crytic_compile resolves `solc` from PATH — must match contract pragma. See Slither operation below. |

**Tool capability matrix (verified 2026-06-18 against known-vulnerable DIVE contracts):**

| Vulnerability class | Slither 0.11.5 | Aderyn 0.6.8 |
|---|---|---|
| Reentrancy (CEI) | ✅ `reentrancy-eth` / `reentrancy-no-eth`. Confirmed on MultiSig (0.4.11). | ⚠️ `reentrancy_state_change` (High) fires on **constructor** CEIs (external `factory()`/`createPair()` before state writes). Confirmed on 5/14 contracts tested (20724, 21559, 19601, 2114, 1607). But: (a) constructor CEIs are non-exploitable (contract not yet deployed), (b) only analyzes `public`/`external` functions → misses CEI in `private` functions like `_transfer` → `swapTokensForEth`. The detector is working as designed but its scope limitation makes it a **constructor CEI flag** for DIVE contracts, not a re-entrancy detector. |
| Missing Access Control | ❌ No detector. | ❌ `centralization-risk` (Low) flags Ownable pattern, not missing auth. |
| Missing Access Control | ❌ No detector. | ❌ `centralization-risk` (Low) flags Ownable usage pattern, NOT missing auth. |
| Unchecked low-level call | ✅ `unchecked-low-level` | ✅ `unchecked-low-level-call` (High) |

**Blind spots: both tools miss the two vulnerability classes this Phase 1 investigation is about (EB/RE access control + meme-token CEI pattern). Their value for this investigation is primarily as supplementary clues (the detectors they DO fire at), not as independent verifiers.**

**Slither operation — see `docs/plan/data_module/slither_reference.md` for full usage guide.** Quick reference:

1. **solc version must match contract pragma.** Manual symlink (solc-select use is unreliable):
   ```
   rm .venv/bin/solc && ln -s ~/.solc-select/artifacts/solc-0.8.19/solc-0.8.19 .venv/bin/solc
   ```
2. **Run:** `slither <contract.sol> 2>&1 | grep -E 'Detector:|reentrancy|arbitrary|unchecked|suicidal|locked-ether' -i`
3. **For RE hints:** `reentrancy-eth` / `reentrancy-no-eth` → CEI pattern found. Check flagged function for guards (nonReentrant, lock, trusted target).
4. **For EB hints:** Slither has NO general missing-access-control detector. Only narrow cases: `suicidal`, `unprotected-upgrade`, `arbitrary-send-eth`. Manual review is primary.
5. **0 High/Medium findings with compilation OK** ≠ contract is clean for access control purposes.

**Tool capability summary:** See `docs/plan/data_module/slither_reference.md` and `docs/plan/data_module/aderyn_reference.md`.

**Aderyn operation — see `docs/plan/data_module/aderyn_reference.md` for full usage guide.** Quick reference:

1. Temp dir + `--stdout` (full report, NOT `--highs-only` — Low findings provide context): `td=$(mktemp -d) && cp <contract.sol> "$td/" && aderyn --stdout "$td" 2>&1; rm -rf "$td"`
2. Extract: `grep -E 'H-[0-9]:|Issue Summary' -A2`
3. `Reentrancy: State change after external call` = constructor CEI (non-exploitable) in DIVE contracts. Clue: contract uses Uniswap setup pattern. Check private `_transfer`/swap functions manually.
4. 0 High findings ≠ contract is clean. Only `public`/`external` functions are analyzed.

**Tool policy:**

**NOT currently installed (must be installed deliberately, as its own gated step, before use):**
Mythril, Manticore, Echidna, Semgrep, Securify.

**Tool policy:**
- **During manual `.sol` review (criteria development, TP/FP judgement): run Slither + Aderyn on the contract FIRST, as investigative hints — not as trusted verifiers.** Treat their output as clues: detector names, highlighted lines, severity levels can point you to code regions worth examining. If the hints surface the right area, you can focus manual reading there rather than reading every line. If the hints are unhelpful, suspicious, or silent, fall back to full manual reading. The tools' "vulnerable/not-vulnerable" verdict is never trusted — only the code regions they flag. This applies to every Method that involves manual contract review.
- **Mythril is too slow for corpus-scale runs** — do **not** run it across thousands of contracts.
  It is acceptable only in a **fast/bounded mode** (e.g. `--execution-timeout` low, a small loop
  bound, or `analyze` on a tiny targeted sample) and only on a **small, already-narrowed set** where
  symbolic execution adds signal a fast linter cannot. Treat it as a precision spot-check tool, never
  a scanner.
- Installing any new tool is **its own gated step** (§1): explain why, what *independent* signal it
  adds that Slither/Aderyn don't, install cost, run cost — then Ali confirms before install.
- For every tool used, record its **detector→class mapping** (which detectors count as evidence for
  which vulnerability class) before running it — an undocumented mapping is a ghost assumption.
- A static-analyzer "agreement" is only meaningful with its **independence** stated. Always note
  whether two agreeing tools could be sharing the same false-positive pattern (the DIVE+Slither
  co-fire on `sendValue`/`approveAndCall` is the cautionary case).

---

## 9. Concern E — Ground every analysis in the infra that already exists

Before writing a new script, check whether the data_module verification stack already does it. These
are real, on disk, and already emit explicit per-class/per-tier numbers (reuse them, extend them —
do not hand-roll a shallow replacement):

| Component | Path | What it gives you |
|---|---|---|
| Verification gate | `data_module/sentinel_data/verification/gate.py` | Per-class VERIFIED / PROVISIONAL / BEST-EFFORT / FAIL with **explicit thresholds** (semantic pass-rate bands, co-occurrence flag, fp_rate>30% → FAIL). The model for what a non-shallow gate looks like. |
| FP estimator | `…/verification/fp_estimator.py` | Empirical FP rate **stratified by source AND tier**, N≈50/class, via real tool runs. Already the right shape for §3 (no naive pooling). |
| Tool validator | `…/verification/tool_validator.py` | Tool-vs-label agreement per class. |
| Negative checker | `…/verification/negative_checker.py` | Corpus-level negative/sanity signal — **the control arm of §6.** |
| Class auditor | `…/verification/class_auditor.py` | Co-occurrence + per-class audit. |
| Semantic checker | `…/verification/semantic_checker.py` | AST-level semantic pass/fail per class. |
| FP estimator sampler | `…/verification/probe_dataset.py` | Probe/sample dataset construction. |
| Slither runner | `…/verification/slither_runner.py` | Cached Slither (content-addressed by sha256). `CLASS_TO_DETECTORS` mapping lives here. |
| Aderyn runner | `…/verification/aderyn_runner.py` | Cached Aderyn, per-file temp dir. `ADERYN_DETECTORS` mapping here. |
| Label quality gate | `ml/testing_specs/label_quality.py` | **F.1.0** — per-class positive rate (FAIL >50% or <1%), per-source rate (FAIL single source >80%), co-occurrence (FLAG >0.60). Built *because* of the DIVE ExternalBug failure. |
| Synthetic probes | `ml/testing_specs/synthetic_probes.py` | Behavioral probes — fixed contracts, fixed expected outputs. |
| Smart summary helper | `data_module/audit/scratch/smart_summary.py` | Per-contract source summariser used in prior manual review (cite full path, not bare name). |

**Tier semantics (already encoded in `gate.py` and `dive.yaml`) — know these before judging any source:**
- **T0** = injection-verified (SolidiFI) — ground truth *by construction*, trusted regardless of rep
  coverage.
- **T1** = high-confidence.
- **T2** = curated (DIVE) — "trusted curation, unverified at AST level" → only PROVISIONAL by default.
  This is precisely the tier whose blind trust produced the ExternalBug failure.
- **T3 / T4** = weaker.

When in doubt about what a prior run measured, the cache dirs (`slither_cache/`, `aderyn_cache/`)
and the existing audit JSONs are **clues to re-verify**, not facts (see §2).

---

## 10. Documentation, provenance & reproducibility (continuous, never batched)

- Every step's notes go into a dedicated file under the active investigation folder
  (`docs/plan/data_module/<dated-investigation>/findings/NN_<topic>.md`) or a scratch file per root
  `CLAUDE.md` Rule 3 — **updated as the step happens**, not reconstructed at the end.
- **"Incremental" means written progressively DURING execution, not composed once at the close gate.**
  A Method with multiple sub-steps (e.g. Method 8's: read parser → resolve ID mapping → write script →
  resolve drop-count → self-verify) appends to its finding file **after each sub-step finishes**, not
  once in a single pass after everything is done. If a Method is interrupted (context loss, session
  end) midway, the finding file should already show what was done up to that point — not be empty or
  all-or-nothing. A finding file written in one shot at the very end, even if accurate, does not
  satisfy this requirement.
- Every claim cites the **command, file, or data** that backs it. No uncited numbers.
- Every finding file states what it did **NOT** cover.
- **Provenance — every script run / measurement records, in its output or finding file:**
  - the **exact command** invoked,
  - the **RNG seed** (sampling must be seeded — §5),
  - the **tool name + version** (e.g. `slither 0.11.5`, `aderyn 0.6.8`),
  - the **input corpus + a hash or count** that identifies exactly which data it ran on, and the frame
    (§2),
  - the **output path** and a **timestamp**.
  A number whose seed/command/version isn't recorded is not reproducible and is treated as UNVERIFIED.
- **Route findings that matter outside this folder to where they're actually tracked.** If a finding
  is relevant to a system/tracker outside the active investigation folder — a model-affecting bug
  belongs in `ml/audit_docs/ISSUES.md`, a data-pipeline change in its changelog — name it and route it
  at the CLARIFY step; do not bury it in a `findings/NN_*.md` that only this investigation ever reads.
  BUT first confirm it actually *is* a finding and not a known intentional change (§1 causal rule, §2
  history-as-lead) — do not route a non-bug into a bug tracker.
- **Progressive documentation must be CHECKABLE, not merely claimed.** "Written incrementally" is
  itself a claim, and subject to the distrust rule. Make it verifiable: commit the finding file at each
  sub-step, OR keep a visible timestamped append-log inside the file (one dated line per sub-step as it
  completes). A single final write that *asserts* "(written progressively)" does not satisfy this — the
  trail has to be inspectable after the fact.
- The investigation's execution log is a **per-gate table** (one row per explain→confirm→execute→
  clarify→confirm cycle), not a single end-of-day summary line.
- An executive summary at the end ties findings together but **does not replace** the per-step trail.

---

## 11. What this protocol does NOT do

- It does not decide TP criteria, sample sizes, thresholds, or which Method runs next — those are
  decided step-by-step, under §1, with Ali. (It *requires* that the criteria and decision rule be
  frozen before measuring — §4, §7 — but does not fix their content.)
- It does not retroactively validate or invalidate prior findings. Whether DIVE/BCCC findings are
  re-derived, kept, or discarded is decided *through* this protocol, not asserted by it.
- It does not authorize any code change to the data pipeline, any re-export, or any Run 13 change.
  Those are separate, later, gated decisions.

---

## 12. Quick checklist (paste-able, per step)

```
[ ] EXPLAINED: data paths + counts + frame, method, tool+version, expected output, deliverable, time
[ ] Ali gave EXPLICIT go (not silence, not a question)
[ ] CRITERIA frozen + agreed + versioned BEFORE any TP/FP judgement (TP/FP/BORDERLINE, per-class bar)
[ ] DECISION RULE pre-committed before seeing results (keep/drop thresholds vs CI bounds)
[ ] Sample drawn by SEEDED RNG; seed + command recorded (no cherry-picking)
[ ] EXECUTED exactly as described — no method swap, no skipped sub-parts, no invented values
[ ] Manual review in batches; verdict + 1-line reason + confidence per contract; NO extrapolation
[ ] CONTROL/negative arm measured with the same criteria (not TP-on-positives alone)
[ ] Independent re-check of ≥10% blind subsample; disagreement rate reported
[ ] Rates as numerator/denominator + n + Wilson CI; CI-overlap checked before comparing
[ ] Gate shows ACTUAL numbers + threshold + what-was-NOT-checked (not just "passed")
[ ] Self-verified the script's own output (totals reconcile, hand-checked example matches)
[ ] Before calling any discrepancy a BUG: checked git log + MEMORY.md + archived/temp scripts for a documented intentional cause
[ ] Examined the FULL artifact record (all provenance fields: tier, source, timestamps) — not just the differing value
[ ] Every root-cause / WHY claim is evidenced (alternatives ruled out) OR stamped HYPOTHESIS; hedges preserved into the summary
[ ] Any recommended fix checked for what it would REVERT or BREAK (tests, intentional prior work) — stated explicitly
[ ] Key numeric claims independently RE-DERIVED (second script/agent), not just re-read from the first agent
[ ] Scripts read live config (or assert their copy == live source); no silent hardcoded copies
[ ] Findings relevant outside this folder ROUTED to their tracker (e.g. ISSUES.md) — after confirming they are real, not intentional
[ ] Progressive documentation is CHECKABLE (commits or in-file timestamped append-log), not just asserted
[ ] Next Method PROPOSED with reasoning at close (per §1 pre-analyzed rule)
[ ] PROVENANCE recorded: command, seed, tool version, input hash/count + frame, output, timestamp
[ ] Findings written to deliverable file AS work happened, with caveats + denominators
[ ] CLARIFIED to Ali in full; Ali confirmed before next step
```
