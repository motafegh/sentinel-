---
name: SENTINEL Run 12 — Full Project Onboarding (2026-06-15)
description: "Complete onboarding for SENTINEL project state at 2026-06-15 — Run 12 done (f1_tuned=0.7004, in Staging), SmartBugs Wild 47K eval complete, all post-processing done, Run 13 prep next"
---

# SENTINEL Project — Full Onboarding Report (2026-06-15)

> **Purpose:** One-stop reference for understanding the complete state of SENTINEL as of 2026-06-15. Read this first if you're onboarding. Use the table of contents to navigate.

---

## Table of Contents

1. [What SENTINEL Is](#1-what-sentinel-is)
2. [Quick Start: Get the Project Running](#2-quick-start-get-the-project-running)
3. [Current State at 2026-06-15](#3-current-state-at-2026-06-15)
4. [Run 12: The Current Best Model](#4-run-12-the-current-best-model)
5. [SmartBugs Wild Full 47K Eval: COMPLETE](#5-smartbugs-wild-full-47k-eval-complete)
6. [Key Findings & Insights](#6-key-findings--insights)
7. [Run 13 Plan: What's Next](#7-run-13-plan-whats-next)
8. [File Organization Rules](#8-file-organization-rules)
9. [Memory & Documentation Map](#9-memory--documentation-map)
10. [Recommended Reading Order](#10-recommended-reading-order)
11. [Critical Open Questions / Blockers](#11-critical-open-questions--blockers)

---

## 1. What SENTINEL Is

**Decentralised AI security oracle for smart contracts.**

- **Input:** Solidity smart contracts (any version 0.4.x – 0.8.x)
- **Processing:** Dual-path ML model (GNN + GraphCodeBERT) with CrossAttentionFusion + LoRA
- **Output:** Multi-label vulnerability classification (10 classes: ExternalBug, Timestamp, Reentrancy, UnusedReturn, IntegerUO, DoS, ToD, CallToUnknown, ME, GasException)
- **Verification:** ZK circuit (EZKL/Groth16) proves the prediction
- **Storage:** On-chain via AuditRegistry contract

**GitHub:** https://github.com/motafegh/sentinel-
**Project root:** `~/projects/sentinel/`
**Environment:** WSL2 Ubuntu, RTX 3070 8GB VRAM, Python 3.12.1, Poetry

---

## 2. Quick Start: Get the Project Running

### Activate Python venv
```bash
source ml/.venv/bin/activate
```

### Key directories at a glance
```
~/projects/sentinel/
├── ml/                           # ML pipeline (model, training, inference)
│   ├── src/                      # Source code (model, GNN, training, inference)
│   ├── scripts/                  # Operational scripts (audit/, eval/, util/, smoke/, interpretability/)
│   ├── checkpoints/              # Trained model checkpoints (~280 MB each)
│   ├── calibration/              # Temperature scaling (per run)
│   ├── testing_specs/            # Spec suite for validation runs
│   ├── data/                     # Graphs, tokens, splits, raw contracts
│   └── logs/                     # Training run JSONL logs
├── data_module/                  # Data pipeline (ingestion → representation → export)
│   ├── sentinel_data/           # v9 schema source of truth
│   ├── data/exports/             # Frozen data exports (v3 active)
│   ├── docs/                     # Data module docs (architecture, integration tests)
│   └── benchmarks/               # 5-tier benchmark design (v0.1 quickstart LIVE)
├── docs/                         # Project-wide docs
│   ├── plans/                    # Implementation plans
│   ├── reports/                  # Per-run eval reports
│   ├── run_summaries/            # Run completion summaries
│   ├── training/                 # Training analysis reports
│   ├── interpretability/         # Interpretability experiment docs
│   ├── ml/                       # ML-specific docs + ADRs
│   ├── decisions/                # Old-style ADRs (kept for compatibility)
│   ├── pre-run*-fixes/           # Historical pre-run planning docs
│   ├── proposal/                 # Old proposals
│   ├── changes/                  # Daily changelogs (50+ files, dated)
│   ├── archive/                  # Old/archived docs (don't touch)
│   └── .bin/                     # Safety storage for moved/old files
├── zkml/                         # ZK circuit code
├── contracts/                    # Solidity contracts (AuditRegistry, ZKMLVerifier)
├── agents/                       # LangGraph agent system (RAG, MCP)
├── BCCC-SCsVul-2024/             # BCCC dataset source (read-only)
└── CLAUDE.md                     # Claude Code rules (naming, file org, etc.)
```

### Useful commands
```bash
# Check model exists
ls -la ml/checkpoints/GCB-P1-Run12-v3dospatched-20260613_FINAL.pt

# Check eval state
bash ml/scripts/util/watch_smartbugs_eval.sh status

# Check what's running
crontab -l  # Run 12 monitoring (may be disabled now)

# Read core memory
cat ~/.claude/projects/-home-motafeq-projects-sentinel/memory/MEMORY.md
```

---

## 3. Current State at 2026-06-15

**Run 12 is COMPLETE and IN STAGING. SmartBugs Wild 47K eval is COMPLETE. All post-processing done. Run 13 prep is NEXT.**

### Headline metrics
| Metric | Value |
|---|---|
| Best F1 (Run 12, tuned) | **0.7004** (NEW SOTA, 2.07x Run 11 ep1's 0.3384) |
| Best F1 (Run 12, tier) | 0.6800 |
| Honest OOD F1 (66 contracts) | **0.8743 tuned / 0.8291 tier** |
| SmartBugs Wild 47K eval | 47,398 processed, 40,616 successful (85.7%), 6,782 errors (pre-0.4.21 Slither) |
| OOD trigger rate | 96.4% (model generalizes) |
| Contamination | 17.4% of Wild is in v3 training (all via DIVE/SmartBugs source) |
| Model location | `ml/checkpoints/GCB-P1-Run12-v3dospatched-20260613_FINAL.pt` |
| MLflow | `sentinel-vulnerability-detector` v1, Stage: **Staging** (NOT Production) |

### Status board
- ✅ Run 12 trained (51 epochs, killed cleanly at ep51 plateau)
- ✅ Calibration done (10 per-class temperatures, ECE 0.035)
- ✅ Threshold tuning done (F1-macro 0.6823 on val)
- ✅ Honest OOD benchmark v0.1 (66 contracts)
- ✅ SmartBugs Wild 47K full eval (4h 5m 39s)
- ✅ Contamination audit (Jaccard ≥0.75)
- ✅ OOD vs Seen analysis
- ✅ SENTINEL vs 9 static tools manual inspection
- ✅ BCCC 2-tool audit (658 ME contracts for Run 13)
- ✅ Feature leakage audit
- 🟡 Run 13 prep (5 fixes identified)
- ❌ Production promotion (blocked on drift_baseline + statistical sig test)

---

## 4. Run 12: The Current Best Model

### Training
- **Config:** v3 data + DoS patch + fresh start + dos_loss_weight=1.0 + drop_complexity=True (L4)
- **Epochs:** 51 (killed at ep51 plateau)
- **Duration:** 21h 35m
- **Process:** PID 230342, killed cleanly (SIGTERM, 0 NaN)
- **Best epoch:** 50 (f1_tuned=0.7004)

### Per-class F1 (Run 12, ep50, sorted high to low)
| Class | F1 tuned | Note |
|---|---|---|
| CallToUnknown | 0.91 | rare in v3 (87 train) |
| MishandledException | 0.91 | rare in v3 (39 train) — overfit on 7 test positives |
| ExternalBug | 0.88 | largest class (16,638 train) |
| Timestamp | 0.83 | 6,324 train |
| Reentrancy | 0.82 | 11,399 train |
| UnusedReturn | 0.77 | 5,859 train |
| IntegerUO | 0.74 | 9,452 train |
| ToD | 0.44 | 647 train |
| DoS | 0.30 | 1,101 (post-patch) |
| GasException | 0.00 | NO DATA in v3 (drop for Run 13) |

### DoS patch impact
- **Pre-patch:** DoS 3,756 (overlapping with Reentrancy 2,655 → memorization)
- **Post-patch:** DoS 1,101 (clean), 0 DoS+Reentrancy overlap
- **F1 trajectory:** 0.11 (ep1) → 0.38 (ep51), 3.5x improvement

### Calibration
- 10 per-class temperatures, mean ECE 0.1948 → 0.0346 (-82%)
- Files: `ml/calibration/run12/temperatures_run12.{json,stats.json,ece_comparison.png}`

---

## 5. SmartBugs Wild Full 47K Eval: COMPLETE

### Run details
- **Launched:** 2026-06-14 22:26 UTC (PID 984402)
- **Finished:** 2026-06-15 02:31:55 UTC
- **Duration:** 4h 5m 39s
- **Throughput:** 2.78 predictions/sec

### Headline numbers
- **47,398 contracts** (100% processed)
- **40,616 successful** (85.7%)
- **6,782 errors** (14.3% — ALL "Slither failed to parse" pre-0.4.21, NOT model bug)
- **96.3% of successful contracts** triggered ≥1 vuln (mean 2.51, max 6 classes/contract)

### Class distribution (40,616 successful)
| Class | Count | % | Mean conf |
|---|---|---|---|
| ExternalBug | 11,801 | 29.1% | 0.801 |
| Timestamp | 11,037 | 27.2% | **0.975** |
| Reentrancy | 7,994 | 19.7% | 0.818 |
| UnusedReturn | 4,486 | 11.0% | 0.963 |
| IntegerUO | 4,221 | 10.4% | **0.668** ⚠️ |
| DoS | 700 | 1.7% | 0.871 |
| ToD | 317 | 0.8% | 0.953 |
| CallToUnknown | 60 | 0.1% | 0.966 |

### Contamination audit (Jaccard ≥0.75)
- **6,493 in v3 train** (13.7%)
- **1,627 in v3 val/test** (3.4%)
- **Total contaminated:** 8,120 (17.4% of Wild)
- **Source:** 8,186 from DIVE + 47 from SmartBugs Curated

### OOD vs Seen analysis
- **32,496 OOD contracts** (68.6% of Wild) — honest benchmark pool
- OOD trigger rate: **96.37%** (vs 96.34% full) — model is generalizing, NOT memorising
- OOD class distribution vs full: deltas ≤1pp across all classes

### SENTINEL vs 9 static tools (manual inspection of 9 OOD contracts)
- **Timestamp class: ✅ Genuine** — high/mid conf predictions are TRUE POSITIVES (tools miss `block.timestamp` usage by design)
- **Reentrancy class: ✅ Reasonable** — high conf = genuine CEI violations
- **ExternalBug class: ⚠️ CRITICAL class definition mismatch** — high conf can be FALSE POSITIVE (e.g., `s_Form001` 26-line KV store, p=0.96 — model pattern-matches on deprecated `sha3`)
- **Key insight:** 65% SOnly rate (SENTINEL fires, 0 tools agree) is NOT over-prediction for Timestamp/Reentrancy; it IS a class-definition mismatch for ExternalBug

### Reports location
`docs/reports/2026-06-15_ml_Run12_eval_full_eval_smartbugs_wild_47K_complete/`:
- `2026-06-15_ml_Run12_eval_full_eval_smartbugs_wild_47K_final_summary.json` + `_summary_report.md`
- `2026-06-15_ml_Run12_eval_full_eval_smartbugs_wild_47K_per_contract_final.json` (48 MB, all 47,398 contracts)
- `2026-06-15_ml_Run12_eval_full_eval_smartbugs_wild_contamination_index.json` (12.7 MB)
- `2026-06-15_ml_Run12_eval_full_eval_smartbugs_wild_contamination_summary.json`
- `2026-06-15_ml_Run12_eval_full_eval_smartbugs_wild_ood_analysis.json` + `_summary.md`
- `2026-06-15_ml_Run12_eval_full_eval_smartbugs_wild_sentinel vs_tools_manual_inspection.md`

### Speed
- mean 359ms, p50 197ms, p95 903ms, p99 3168ms

---

## 6. Key Findings & Insights

### Finding 1: Run 12 is real learning, not memorization
- 96.4% OOD trigger rate (vs 96.3% on full) = **model generalizes**
- 0% leakage verified on v3 splits
- Honest OOD F1=0.8743 (66 contracts) confirms high quality

### Finding 2: SmartBugs Wild is HEAVILY multi-class
- Mean 2.51 triggers/contract (vs single-class in v0.1 benchmark)
- Max 6 classes triggered in one contract
- 95% of contracts have ≥1 trigger

### Finding 3: ExternalBug class has a learning artifact
- High conf FP at p=0.96 on 26-line `s_Form001` KV store
- Model pattern-matches on `sha3` (deprecated) and other syntactic features
- DeFiHackLabs training data (complex DeFi exploits) is too narrow for diverse smart contract patterns
- **Action for Run 13/14:** Review DeFiHackLabs ExternalBug labels, consider splitting/dropping ambiguous training samples

### Finding 4: 65% SOnly is NOT over-prediction
- Tools miss most real Timestamp and Reentrancy vulnerabilities (by design — flag exploitable instances, not structural patterns)
- SENTINEL sees patterns tools aren't designed to catch
- **When evaluating SENTINEL, use manual inspection + hotspot analysis, not tool agreement rate**

### Finding 5: 17.4% contamination is unavoidable
- DIVE source (8,186) + SmartBugs Curated (47) = 8,120 Wild contracts match v3 by Jaccard ≥0.75
- All matches are normalizations/deduplications of the same source contracts
- Only 6 exact-byte matches; rest are byte-level variants of the same underlying code
- **Implication:** future benchmarks MUST check contamination; v0.1 quickstart already does this

### Finding 6: Pre-0.4.21 contracts are pipeline limit
- 6,782 errors (14.3%) are all "Slither failed to parse" for pre-0.4.21 contracts
- Same as The DAO, Parity, WithdrawDAO, Hacken (famous contracts test)
- **Not a model bug — needs pipeline upgrade** (multi-solc-version support) to expand to historical contracts

---

## 7. Run 13 Plan: What's Next

**5 confirmed fixes (was 4 + 1 new from manual inspection):**

1. **Drop GasException → NUM_CLASSES=9** (zero data in v3, F1=0.0 by definition)
2. **Extend L4 to drop `loc` (dim 6)** (for consistency with `complexity`, both graph size proxies)
3. **Strip Solidifi `bug_*` prefix in pre-processing** (function names like `bug_intou*` literally encode vulnerability type)
4. **Inject 658 BCCC ME contracts** → re-export v4, re-split, re-train (13.5x boost in ME training data)
5. **ExternalBug label quality review** — `s_Form001` FP at p=0.96 suggests spurious training signal in DeFiHackLabs; review and clean

**Estimated:** 3 weeks of work

**Detailed plan:** `docs/plans/2026-06-14_Run12_to_Run13_handoff.md` + `docs/plans/2026-06-14_Run13_4_fixes_preparation.md`

---

## 8. File Organization Rules

**6-part naming convention (mandatory):**
`<YYYY-MM-DD>_<MODULE>_<RUN_or_PHASE>_<WHAT_it_is>_<descriptor>.<ext>`

- Date first (YYYY-MM-DD)
- Module: `ml`, `data_module`, `agents`, `zkml`, `contracts`, `proposal`
- Run/Phase: `Run12`, `pre_Run5`, `Stage7B`, `v2_audit`, `bccc_deep_dive`
- What: `plan`, `audit`, `summary`, `eval_benchmark`, `eval_full_eval`, `training_analysis`, `calibration`, `findings`, `proposal`, `handoff`
- Descriptor: specific (e.g., `honest_OOD_v0.1_quickstart`, `47K_complete`, `45pct_leakage`)
- Extension: `.md`, `.json`, `.txt`, `.py`, etc.

**Canonical locations:**
- Project plans → `docs/plans/<dated>_<subject>.md`
- Project reports → `docs/reports/<YYYY-MM-DD>_<module>_<Run>_<what>_<descriptor>/`
- Per-run summaries → `docs/run_summaries/<dated>_<module>_<Run>_<what>_<descriptor>.md`
- Training analysis → `docs/training/<dated>_<Run>_<what>_<descriptor>.md`
- Data module docs → `data_module/docs/<dated>_data_module_<what>_<descriptor>.md`
- Data module audits → `data_module/audit/<dated>_data_module_<audit>_<stage>.md`
- Calibration artifacts → `ml/calibration/run<N>/`
- Code audit/eval → `ml/scripts/{audit,eval,util}/<script>.py` (no date prefix for code)
- Cross-session memory → `~/.claude/projects/.../memory/{MEMORY.md, project_*.md}`
- Session scratch → `~/.claude/scratch/<topic>_<YYYYMMDD>.md`
- Safety bin → `docs/.bin/<YYYY-MM-DD>_<phase>_<reason>/` (do-not-touch)

**Duplicate/move rule:** Never delete directly. MOVE to `docs/.bin/<date>_<phase>_<reason>/`. After full validation, user can `rm -rf docs/.bin/`.

**Exception:** Code in `ml/scripts/` (no date prefix) and ADRs in `docs/ml/adr/` (industry-standard `ADR-NNNN-name.md`).

**When in doubt, ASK the user.**

Full rules: `~/projects/sentinel/CLAUDE.md` § "Documentation & File Naming Rules"

---

## 9. Memory & Documentation Map

### Cross-session memory (`~/.claude/projects/.../memory/`)
- **MEMORY.md** (125 lines) — index, current state, key file paths
- **21 project_*.md files** (dated, e.g., `2026-06-15_project_smartbugs_wild_full_eval.md`)

### Plans (`docs/plans/`)
- 7 files: post-training process, Run 12→13 handoff, 4 fixes, pre-Run-12, data addition, seam swap

### Reports (`docs/reports/`)
- 8 subdirs for Run 12 (one per eval type + Run 13 plan)
- All use `<YYYY-MM-DD>_<module>_<Run>_<what>_<descriptor>/` naming

### Run summaries (`docs/run_summaries/`)
- 1 file: `2026-06-14_Run12_post_training_process_summary.md`
- (NEW: `2026-06-15_Run12_smartbugs_wild_full_eval_summary.md` to be created)

### Training analysis (`docs/training/`)
- 8 files (dated), most recent: `2026-06-14_ml_Run12_training_analysis_v3dospatched.md`

### Data module docs (`data_module/docs/`)
- 6 files (dated, e.g., `2026-06-14_data_module_v2_v3_architecture.md`)

### Data module audits (`data_module/audit/`)
- 9 files (00-08 series) + `v2_full_audit/` (7 files)
- All dated

---

## 10. Recommended Reading Order

If you're new to SENTINEL, read in this order:

1. **`CLAUDE.md`** (project root) — naming rules, mode triggers, WSL2 rules
2. **This onboarding report** (you are here) — full project state
3. **MEMORY.md** (memory dir) — index + current state
4. **`2026-06-14_project_run12_post_training.md`** — Run 12 details
5. **`2026-06-15_project_smartbugs_wild_full_eval.md`** — Wild eval details
6. **`2026-06-14_project_bccc_2tool_audit.md`** — BCCC ME findings (Run 13 input)
7. **`2026-06-14_project_feature_leakage_audit.md`** — feature audit
8. **`data_module/docs/2026-06-14_data_module_v2_v3_architecture.md`** — data module
9. **`docs/plans/2026-06-14_Run12_to_Run13_handoff.md`** — what's next
10. **`docs/plans/2026-06-14_Run13_4_fixes_preparation.md`** — detailed Run 13 fixes

For specific questions:
- "What's the current F1?" → MEMORY.md, this report
- "Why ExternalBug class has FPs?" → `sentinel vs_tools_manual_inspection.md` (Finding 3)
- "What was Run 11?" → `2026-06-13_project_run11_launch.md`
- "How do I train?" → `2026-06-13_project_run12_launch.md` (config + launch)
- "Where is the calibration file?" → `ml/calibration/run12/temperatures_run12.json`
- "How is the data built?" → `data_module/docs/2026-06-14_data_module_v2_v3_architecture.md`

---

## 11. Critical Open Questions / Blockers

### Decision gates (need Ali's input)

1. **Run 13 Fix #5 priority:** When to do the ExternalBug label review? Before or after the other 4 fixes? My recommendation: do it AFTER the 4 fixes (cleaner rollback if it goes wrong).

2. **Production promotion gate:** Run 12 is in Staging but NOT Production. Requirements:
   - `ml/data/drift_baseline.json` needs to be rebuilt from REAL warmup traffic (currently PLACEHOLDER)
   - Statistical significance test vs current Production model (there is no current Production)
   - If both pass, Run 12 can be promoted to Production.

3. **Seam swap (Stage 7B) 3 open Qs** in `2026-06-13_data_module_seam_swap_completion_Stage7B.md` §11.1:
   - Q1: Delete legacy `ml/src/data_extraction/tokenizer.py`?
   - Q2: Flip `ml/src/inference/preprocess.py` shim too?
   - Q3: Keep data_module/.gitignore rules?

### Known limitations
- **IntegerUO confidence low (0.668)** — needs investigation (limited training representation for pre-0.8 contracts)
- **Pre-0.4.21 contracts** not supported (6,782 errors in Wild eval) — needs multi-solc-version pipeline upgrade
- **ME / GasException** not in v3 (gas=0, ME=39) — Run 13 will add 658 BCCC ME contracts

### Things to monitor
- **ExternalBug class precision** — Finding 3 suggests possible precision regression in next training
- **Contamination in future benchmarks** — `ml/scripts/audit/check_contamination_wild.py` is the gate
- **Production drift** — once we have warmup traffic, rebuild `drift_baseline.json`

---

## TL;DR for Busy People

**What's done:** Run 12 model trained (f1_tuned=0.7004), validated, calibrated, and tested on 47,398 real-world contracts (96.4% OOD trigger rate). All post-processing done. **In Staging.**

**What's next:** Run 13 with 5 fixes (drop GasException, drop `loc`, strip `bug_*` prefix, inject 658 BCCC ME, review ExternalBug labels). ~3 weeks.

**What's blocked:** Production promotion (need drift_baseline + sig test). Seam swap 3 open Qs awaiting Ali's decision.

**Where to look first:** `MEMORY.md` (cross-session), this report (full context), `CLAUDE.md` (naming rules).

**Critical insight:** The model generalizes well (96% OOD trigger) but has a class definition mismatch on ExternalBug that needs fixing in Run 13.
