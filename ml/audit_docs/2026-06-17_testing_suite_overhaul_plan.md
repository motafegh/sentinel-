# Master Plan — Bulletproof Testing Spec Suite

**Date:** 2026-06-17
**Author:** Claude (build mode)
**Scope:** Overhaul `ml/testing_specs/` from manual/human-required to automated + AI-assisted + reusable

---

## Mission

Make the testing spec suite:
1. **Fully automated** — no human-required steps for routine checks
2. **Bullet-proof** — catches label noise, model gaming, contamination, false positives
3. **Reusable** — works for SENTINEL today, pluggable for other ML projects tomorrow

---

## Phase 1: Add the Missing Tests (THE GAPS)

These are the 3 specific gaps that allowed the ExternalBug FP to slip through:

### 1.1 Synthetic Behavioral Probes (the missing test)

**File:** `ml/testing_specs/synthetic_probes.py` (NEW)
**Spec:** Add `C.2.4 — Synthetic Behavioral Probes` to `C_diagnostic_checks.md`

**What it does:** Run the model on 30+ fixed synthetic contracts (3 per class) with hardcoded expected probability thresholds. Block promotion if any probe fails.

**Probes per class:**
- **Should trigger (high prob expected):** Real vulnerability pattern
- **Should NOT trigger (low prob expected):** Benign pattern that looks similar
- **Edge case (gray area):** Borderline contract

**Why:** The ExternalBug FP was a 0.822 on a safe_storage-style contract. The probe would block this with a single line: "ExternalBug < 0.3 on owner-pattern contract."

### 1.2 Label Quality Gate (pre-launch)

**File:** `ml/testing_specs/label_quality.py` (NEW)
**Spec:** Add `F.1.0 — Label Quality Gate` to `F_new_run_checklist.md`

**What it does:** Before training, check the training labels for:
- Per-class positive rate (FAIL if any class > 50% or < 1%)
- Class co-occurrence matrix (FLAG suspicious correlations)
- Source distribution per class (DIVE 75% vs Solidifi 14% is a red flag)

**Why:** Run 12 had ExternalBug at 75% positive in training. A pre-launch check would have flagged this and forced a label audit before training started.

### 1.3 Behavioral Gate in `promote_model.py`

**File:** `ml/scripts/promote_model.py` (MODIFY)
**Spec:** Update `I.2.2` to add the behavioral gate

**What it does:** Promote reads `<stem>_behavioral_probes.json` and BLOCKS promotion if any probe is FAIL.

**Why:** Currently, `promote_model.py` only checks F1. The behavioral gate adds a second hard check.

---

## Phase 2: Fix Weak Points

### 2.1 Automate the Manual Steps

| Manual step | Current spec | Automation |
|---|---|---|
| C.2.1 clean smoke inference | UNVERIFIED (95.8% contamination) | **F.1.0** detects contamination + **C.2.4** uses synthetic contracts as clean fallback |
| C.2.2 FP probe | Manual 9-contract inspection | **C.2.4** uses 30+ fixed contracts (3 per class) |
| I.3.1 behavior checks | "Not enforced by promote_model.py" | **I.2.2 update** — promoted model MUST pass behavioral probes |
| I.3.3 contamination check | Manual run | **`promote_model.py` auto-runs** before gate check |
| L.4 reproducibility | Manual verify | **NEW L.4.1** — auto-run hash compare |
| L.5 floating findings | Manual write | **NEW L.5.1** — auto-detect from run logs |

### 2.2 Adversarial Probes for All 10 Classes

**File:** `ml/testing_specs/synthetic_probes.py` (EXTEND)

Currently only ExternalBug has 3 probes. Add 3 probes for each of the other 9 classes:
- Reentrancy: 3 probes (real reentrancy, just transfer, fallback)
- IntegerUO: 3 probes (real overflow, simple math, uint256)
- ... etc

**Why:** Catches any future class with a broken feature, not just ExternalBug.

### 2.3 LLM-as-Judge for Narrative Quality (Phase A prep)

**File:** `ml/testing_specs/narrative_judge.py` (NEW)

**What it does:** After audit, use the LLM itself to evaluate the synthesizer's narrative:
- Does it mention the correct vulnerability?
- Is the fix actionable?
- Does it match the RAG evidence?

**Why:** The agents module's narrative quality is subjective. Currently judged only by manual inspection. The LLM can verify its own output at near-zero cost.

### 2.4 Cross-Tool Consistency Check

**File:** `ml/testing_specs/cross_tool.py` (NEW)

**What it does:** Run Slither + Aderyn + model on the same contracts. Verify:
- They don't agree 100% (model overfitting to tools)
- They don't disagree completely (model is broken)

**Why:** Strong agreement with tools = model is learning the tool's pattern, not the vulnerability. Strong disagreement = model is broken.

### 2.5 Threshold Sensitivity Check

**File:** `ml/testing_specs/threshold_sensitivity.py` (NEW)

**What it does:** For each class, run with thresholds 0.1, 0.3, 0.5, 0.7. Verify F1 doesn't swing wildly.

**Why:** ExternalBug had F1=0.88 at threshold 0.35. At 0.1 it would be near 0 (too many FPs). At 0.7 it would drop. Wide swings = bad calibration.

### 2.6 Auto-Detection of Stale Checkpoints

**File:** `ml/scripts/check_stale_checkpoints.py` (NEW)

**What it does:** Scan `ml/checkpoints/` for checkpoints that haven't been evaluated in 30+ days. Auto-archive or warn.

**Why:** Operational hygiene. Prevents old/broken checkpoints from being re-promoted.

---

## Phase 3: Make It Reusable

### 3.1 Project-Agnostic Framework

**File:** `ml/testing_specs/framework/` (NEW package)

```
framework/
├── __init__.py
├── gates.py            # Gate ABC (Always/Blocked/Conditional)
├── behavioral_probes.py  # Synthetic probe runner
├── label_quality.py    # Label quality checks
├── threshold_sensitivity.py
├── cross_tool.py
├── cli.py              # ml-validate entry point
├── config.py           # YAML config loading
├── reporters.py        # Output formats (text, JSON, HTML)
└── templates/          # Templates for new projects
    ├── gates.yaml
    ├── probes.yaml
    └── config.yaml
```

**Key design:** The framework takes a `ProjectConfig` (YAML) and runs all gates against the project's data + model. Works for SENTINEL, but also for vision/NLP/tabular projects.

### 3.2 Single CLI Entry Point

**File:** `ml/testing_specs/framework/cli.py`

```bash
# Run all gates on Run 12
ml-validate run --run GCB-P1-Run12-v3dospatched-20260613 --gate all

# Run a specific gate
ml-validate run --run Run12 --gate behavioral_probes

# Generate the next run's gates config
ml-validate init --template sentinel_v2 > ml/checkpoints/run13_gates.yaml

# Check if a model is promotable
ml-validate promote --checkpoint ml/checkpoints/Run12_best.pt \
    --gate all --dry-run
```

### 3.3 Templates for New Projects

**File:** `ml/testing_specs/framework/templates/`

For each project type:
- `sentinel_v2.yaml` — SENTINEL multi-label code vulnerability
- `image_classification.yaml` — Vision project
- `text_classification.yaml` — NLP project
- `tabular_regression.yaml` — Tabular regression

Each template has:
- List of gates
- Probe definitions (with placeholders for the user to fill in)
- Label quality thresholds
- Pass/fail criteria

### 3.4 Documentation Overhaul

**Files to update/create:**
- `ml/testing_specs/README.md` — rewrite to point at the framework
- `ml/testing_specs/QUICKSTART.md` (NEW) — 5-minute guide to add a new gate
- `ml/testing_specs/MIGRATION.md` (NEW) — how to apply this to a new project
- `docs/ml/testing/architecture.md` (NEW) — system diagram

### 3.5 Hooks for AI Assistance

**File:** `ml/testing_specs/framework/ai_helpers.py` (NEW)

When manual judgment is unavoidable (e.g., narrative quality):
- The framework logs the input/output
- An AI helper script can be run to assist: `ml-validate ai-review --checkpoint X --question "Is the synthesizer narrative high quality?"`
- Output: a scored assessment with reasoning

---

## Execution Plan (Concrete Steps)

### Step 1: Write the plan (this doc) ✓
### Step 2: Implement synthetic_probes.py
- 30+ probes (3 per class × 10 classes)
- Run on Run 12 checkpoint
- Verify Run 12 fails the ExternalBug probe
### Step 3: Implement label_quality.py
- Pre-launch check
- Run on Run 12 training data
- Verify Run 12 fails the 75% ExternalBug check
### Step 4: Update promote_model.py
- Add behavioral_probes.json check
- Add label_quality check
- Wire both into the gate sequence
### Step 5: Create framework/ package
- gates.py (the base Gate class)
- behavioral_probes.py (reuse synthetic_probes.py)
- label_quality.py (reuse)
- cli.py (ml-validate)
- config.py (YAML loader)
- templates/ (sentinel_v2.yaml first)
### Step 6: Update spec files
- C_diagnostic_checks.md: add C.2.4
- F_new_run_checklist.md: add F.1.0
- I_regression_guard.md: update I.2.2
- L_release_readiness.md: add L.4.1, L.5.1
- README.md: rewrite to point at framework
### Step 7: Test end-to-end
- Run ml-validate on Run 12
- Verify all gates fire
- Verify failures are correctly identified
- Run on a hypothetical Run 13 (passes everything)

---

## File-by-File Plan

| File | Action | Purpose |
|---|---|---|
| `ml/testing_specs/synthetic_probes.py` | CREATE | 30+ fixed probes (3 per class) |
| `ml/testing_specs/label_quality.py` | CREATE | Pre-launch label audit |
| `ml/testing_specs/threshold_sensitivity.py` | CREATE | Per-class threshold sweep |
| `ml/testing_specs/cross_tool.py` | CREATE | Cross-tool consistency check |
| `ml/testing_specs/framework/__init__.py` | CREATE | Package init |
| `ml/testing_specs/framework/gates.py` | CREATE | Gate base class |
| `ml/testing_specs/framework/cli.py` | CREATE | ml-validate CLI |
| `ml/testing_specs/framework/config.py` | CREATE | YAML config |
| `ml/testing_specs/framework/reporters.py` | CREATE | Output formats |
| `ml/testing_specs/framework/templates/sentinel_v2.yaml` | CREATE | SENTINEL template |
| `ml/scripts/promote_model.py` | MODIFY | Wire behavioral + label gates |
| `ml/scripts/check_stale_checkpoints.py` | CREATE | Operational hygiene |
| `ml/testing_specs/C_diagnostic_checks.md` | UPDATE | Add C.2.4 |
| `ml/testing_specs/F_new_run_checklist.md` | UPDATE | Add F.1.0 |
| `ml/testing_specs/I_regression_guard.md` | UPDATE | I.2.2 add behavioral gate |
| `ml/testing_specs/L_release_readiness.md` | UPDATE | Add L.4.1, L.5.1 |
| `ml/testing_specs/README.md` | UPDATE | Point at framework |
| `ml/testing_specs/QUICKSTART.md` | CREATE | 5-min guide |
| `ml/testing_specs/MIGRATION.md` | CREATE | Reuse guide |
| `docs/ml/testing/architecture.md` | CREATE | System diagram |

---

## Success Criteria

After this overhaul:
1. **No human required for routine checks** — all gates auto-run
2. **Catches label noise** — 75% positive rate would be flagged before training
3. **Catches false positives** — synthetic probes block broken models
4. **Reusable** — new projects can adopt the framework in <1 day
5. **Well-documented** — anyone can extend it without breaking things

---

## What I will NOT do (scope limits)

- Won't refactor the existing graph extractor or model code
- Won't change the loss function or training pipeline
- Won't add new ML features (the framework is the focus)
- Won't write tests for the framework itself in this pass (defer to a follow-up)

---

## Next Step

Start with Phase 1: write `synthetic_probes.py` with 30+ probes and verify Run 12 fails the ExternalBug probe. This is the most impactful change and proves the framework works.
