# Plan: Doc 02 — Evidence Model & Fuse(): How 6 Channels Become One Verdict

**Spec:** `docs/learning/LEARNING_DOCS_SPEC.md`
**Target:** `docs/learning/02_evidence_model_fuse.md`
**Session:** 1 of 5
**Prerequisite docs:** Doc 01 (Pipeline)

---

## Recall from previous docs

**From Doc 01 (Pipeline):** You learned that 14 nodes run in a LangGraph, producing state that accumulates via append-reducers. The pipeline has a fast path (3s) and deep path (60s). Routing is deterministic — never an LLM. The pipeline always produces a report (fail-soft).

**Connection to this doc:** Now we answer: *what does the pipeline produce?* Each node emits `Evidence` items into `state["evidence_list"]`. The `fuse()` function consumes that list and produces verdicts. This doc explains the Evidence model and how 6 independent channels (ML, Slither, Aderyn, RAG, debate, Halmos) all speak the same language.

**Key state fields from Doc 01 used here:** `evidence_list` (append-reducer), `verdict_provable`, `verdict_full`, `consensus_verdict`.

---

## Step 1: Read source files

- [ ] `agents/src/orchestration/verdict/evidence.py` (~180 lines) — Evidence dataclass, 7 constructors (ml, slither, aderyn, rag, debate, quick_screen, formal), Polarity enum, Kind enum
- [ ] `agents/src/orchestration/verdict/fuse.py` — fuse() function, weighted Bayesian combination, source→kind mapping, VerdictResult
- [ ] `agents/src/orchestration/verdict/emit.py` (~310 lines) — emit_ml_evidence, emit_static_evidence, emit_rag_evidence, emit_debate_evidence, emit_halmos_evidence, emit_consensus_evidence
- [ ] `agents/src/orchestration/verdict/reliability.py` — get_reliability(), L3→L1 fallback logic
- [ ] `agents/src/orchestration/verdict/verdict.py` — VerdictResult dataclass (if separate from fuse.py)

## Step 2: Read scratch files

- [ ] `~/.claude/scratch/system_finalization_statecheck_20260625.md` — P2 evidence model design, the pairwise→uniform decision, the 22.9% match rate finding, the 75 asymmetry violations
- [ ] `~/.claude/scratch/p2_plan_review_20260624.md` — fuse() vs legacy verdicts analysis, the decision to generalize before adding channels

## Step 3: Read eval reports

- [ ] `agents/eval/runs/20260624T133420Z_p0_honest_baseline/eval_report.md` — P0 baseline (F1=0.1958, Fbeta=0.2515)
- [ ] `agents/eval/runs/20260624T231228Z_p2_calibrated/eval_report.md` — P2 calibrated (F1=0.1998, Fbeta=0.2246)
- [ ] `agents/eval/runs/20260626T123145Z_p3_rule5c_v3/eval_report.md` — P3 data-derived (F1=0.3008, Fbeta=0.3821)

## Step 4: Read config

- [ ] `agents/configs/verdicts_default.yaml` — L1 config (thresholds, weights)
- [ ] `agents/configs/reliability_v3.yaml` — L3 fitted config (data-derived reliability values)

## Step 5: Write sections

- [ ] **TL;DR:** Uniform Evidence model (7 fields), 6 sources emit the same shape, fuse() combines them via weighted Bayesian, dual verdict (provable=deterministic only / full=all evidence)
- [ ] **The Problem:** Old `consensus_engine` + 8-case `_reconcile_verdicts` was pairwise and hand-cased. Adding 6 more channels would grow it to 28 cases. Needed generalization BEFORE scaling
- [ ] **How We Arrived at This Design:** invariant (FN/FP asymmetry — no flagged class may become SAFE) → constraint (ZK needs deterministic flag) → simplest model (Evidence dataclass + fuse function, no ML classifier) → stress-test (add Halmos in P8a without changing fuse()) → measure (F1 0.20→0.30 from L3 reliability)
- [ ] **The Solution:** Evidence dataclass field diagram. fuse() flow: collect evidence → group by class → weighted Bayesian per class → verdict. Dual verdict split diagram (provable vs full). The `deterministic` flag as the ZK boundary
- [ ] **Key Code:**
  - `Evidence` dataclass (evidence.py:29-38) — 7 fields: source, vuln_class, polarity, strength, reliability, kind, deterministic
  - `Evidence.formal()` (evidence.py:158-180) — the newest constructor, added in P8a
  - `fuse()` function (fuse.py) — the sole verdict producer
  - `emit_halmos_evidence()` (emit.py) — shows how a new channel plugs in
  - `get_reliability()` (reliability.py) — L3→L1 fallback chain
- [ ] **Design Decision:** Uniform Evidence vs pairwise rules vs ML classifier for fusion (tradeoff table: extensibility, interpretability, training data needed, determinism)
- [ ] **Technology Choice:** Weighted Bayesian vs Dempster-Shafer vs learned weights (5-question framework)
- [ ] **Anti-Patterns:**
  - ❌ Pairwise rules — "just 2 sources, keep it simple." Breaks: grows O(n²), 28 cases for 8 channels. Right: uniform Evidence + single fuse()
  - ❌ ML classifier for fusion — "let ML learn the weights." Breaks: no training data, uninterpretable, non-deterministic. Right: weighted Bayesian (interpretable, no training needed)
- [ ] **Mistakes & Fixes:**
  - 8-case `_reconcile_verdicts` would grow to 28 for 8 channels. Fix: generalize to Evidence + fuse() BEFORE adding channels (P2)
  - fuse() matched legacy only 22.9% — but legacy was wrong, not fuse(). 75 asymmetry violations (legacy flagged → fused SAFE). Lesson: measured truth > intuition
  - `deterministic` flag missing initially → couldn't separate ZK-anchorable evidence. Fix: add flag to Evidence dataclass
- [ ] **What Would Break Without This:** Remove `deterministic` flag → ZK boundary disappears, can't separate reproducible from non-reproducible. Remove fuse() → no verdict producer. Remove Evidence model → back to pairwise, can't add channels
- [ ] **At Scale:** 6 channels (current) / 15 / 50 / 100 — fuse() is O(n) in evidence count, scales linearly
- [ ] **Try It Yourself:**
  ```
  cd agents && source .venv/bin/activate
  python3 -c "from src.orchestration.verdict.evidence import Evidence, Kind, Polarity; e = Evidence.ml('Reentrancy', 0.85, 0.90); print(e)"
  python3 -c "from src.orchestration.verdict.evidence import Evidence, Polarity; e = Evidence.formal('halmos', 'Reentrancy', Polarity.SUPPORTS, 'reentrancy', False, '0xdead'); print(e)"
  ```
- [ ] **Limitations:** Fixed weights (not learned), reliability noisy with 61 contracts, no evidence conflict resolution beyond polarity, no uncertainty quantification on fuse() output
- [ ] **Transferable Patterns:** (1) Generalize before scaling — uniform interfaces absorb growth (2) Dual-tier architecture — provable vs advisory (3) Dataclass as contract — 7 fields, frozen, validated. Each with interview story + when wrong.

## Step 6: Verify

- [ ] Open `evidence.py` and verify all 7 constructors exist (ml, slither, aderyn, rag, debate, quick_screen, formal)
- [ ] Open `fuse.py` and verify the source→kind mapping includes `"halmos": "FORMAL"`
- [ ] Open `emit.py` and verify `emit_halmos_evidence()` exists
- [ ] Confirm eval numbers (0.1958 → 0.1998 → 0.3008) match the 3 eval reports
- [ ] Open `reliability_v3.yaml` and verify it exists with schema_version=1
