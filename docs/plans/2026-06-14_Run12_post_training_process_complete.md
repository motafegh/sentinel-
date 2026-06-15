# Post-Training Process — Complete Plan (2026-06-14)

> **Purpose:** The complete workflow for **evaluating, validating, calibrating, benchmarking, and promoting a trained model**. This plan is **Run-N agnostic** — it applies to Run 12, Run 13, Run 14, or any future run.
>
> **When:** After any training run completes (or early-stops). Trigger: process death OR "Early stop" OR "Training complete" in `ml/logs/<run_name>.log`.
>
> **Out of scope:** Preparing the NEXT training run. That is documented in the two companion files in `docs/plans/`:
> - `2026-06-14_Run12_to_Run13_handoff.md` — the 6-step Run N → Run N+1 handoff workflow
> - `2026-06-14_Run13_4_fixes_preparation.md` — the 4-fixes details for Run 13
>
> **This plan does NOT cover "fix the data and retrain" — that is a separate concern in the companion files above.**
>
> **Supersedes:** `post_training_process_2026-06-14.md` (208-line overview). The companion docs in the same directory cover the NEXT-training work: `run_12_to_13_handoff_2026-06-14.md` (287 lines, Run N → Run N+1 handoff) and `run13_plan_2026-06-14.md` (336 lines, Run 13 specific fixes).

---

## 0. What "Post-Training" means

The 5 phases, each with a clear input → output:

| # | Phase | Input | Output | Time |
|---|---|---|---|---|
| 1 | **Run Validation** | `best.pt` checkpoint + `epoch_summary.jsonl` | Reproducibility confirmed + per-class analysis + Run N final report | ~30 min |
| 2 | **Calibration** | `best.pt` + val split | `<stem>_thresholds.json` + `<stem>_temperatures.json` (10-class, or 9-class post-Run-13) | ~1 hour |
| 3 | **External Benchmark** | `best.pt` + thresholds + benchmark datasets | Per-mode (tier + tuned) F1 on SmartBugs (137) + SolidiFI (283) | ~1 hour |
| 4 | **Behaviour & API Validation** | `best.pt` + FastAPI server | Round-trip tests, FP probe regression, hotspot sanity, cache invalidation check | ~1 hour |
| 5 | **Promotion** | All artifacts from phases 1-4 + drift baseline | MLflow Model Registry entry at Staging (always) or Production (if statistical sig + drift baseline) | ~30 min |

**Total time:** ~4-5 hours active work (parallelizable with next-training prep).

---

## 1. The 5-phase workflow diagram

```
Run N completes (PID gone, OR early-stop, OR epN reached)
    │
    ├─→ [Day 0] Phase 1: Run Validation (~30 min)
    │     • Reproducibility (L.1)
    │     • Performance analysis
    │     • Run N final report (NEW DOC)
    │
    ├─→ [Day 0] Phase 2: Calibration (~1 hour)  ← NEW: was missing from handoff
    │     • Threshold tuning (tune_threshold.py)
    │     • Temperature scaling (calibrate_temperature.py)
    │
    ├─→ [Day 0] Phase 3: External Benchmark (~1 hour)  ← CONCRETE: SmartBugs + SolidiFI
    │     • Contamination check (check_contamination.py) — Tier 1-4
    │     • SmartBugs Curated benchmark (137 real-world contracts)
    │     • SolidiFI benchmark (283 synthetic contracts)
    │     • OOD analysis (graph size, class distribution)
    │
    ├─→ [Day 0-1] Phase 4: Behaviour & API Validation (~1 hour)
    │     • Smoke suite (smoke/run_all.py)
    │     • Round-trip on known-positive (1 per class)
    │     • Round-trip on known-negative (5+ NonVulnerable)
    │     • FP probe regression (K.5.3)
    │     • Hotspot endpoint sanity (K.5.4)
    │     • Cache invalidation check
    │
    └─→ [Day 1] Phase 5: Promotion (~30 min)
          • Staging (always, per I.2.1)
          • Production (only if drift baseline rebuilt + statistical sig vs current Production)
                ↓
         HANDOFF to run_13_prep_plan_complete_2026-06-14.md (next training prep — separate file)
```

---

## 2. Phase 1: Run Validation (~30 min)

**Goal:** Confirm the run is reproducible + understand its per-class performance + write a final report.

### 2.1 — Reproducibility check (per `L_release_readiness.md` §L.1)

```bash
cd /home/motafeq/projects/sentinel

# L.1.1 — RNG seed
PYTHONPATH=. ml/.venv/bin/python -c "
import torch, json
ckpt = torch.load('ml/checkpoints/GCB-P1-Run<N>-<tag>-YYYYMMDD_best.pt',
                  map_location='cpu', weights_only=False)
print('seed:', ckpt.get('config', {}).get('seed'))
print('architecture:', ckpt.get('config', {}).get('architecture'))
print('num_classes:', ckpt.get('config', {}).get('num_classes'))
"
# Confirm seed matches MEMORY.md Training History (should be 42 unless ablation)
# Confirm num_classes matches the active export (10 for Run 12; will be 9 for Run 13+)

# L.1.2 — Tokenizer mode
grep "TRANSFORMERS_OFFLINE" ml/logs/<run_name>_launch_*.log
# Expected: present in the launch command (=1)

# L.1.3 — Data export hash (replaces retired DVC check)
PYTHONPATH=. ml/.venv/bin/python -c "
import json
m = json.load(open('data_module/data/exports/<active_export>/MANIFEST.json'))
print('export_artifact_hash:', m['artifact_hash'])
"
# Compare to run's recorded value in ml/logs/<run_name>/epoch_summary.jsonl

# L.1.4 — poetry.lock unchanged
git diff HEAD~1 poetry.lock --stat
# Expected: empty (or only dev deps)
```

**Pass criteria:** seed=42 (or documented ablation), `TRANSFORMERS_OFFLINE=1` confirmed, export hash matches, poetry.lock stable.

### 2.2 — Performance analysis (per `C_diagnostic_checks.md` C.2)

```bash
# Best epoch + f1_tuned
PYTHONPATH=. ml/.venv/bin/python -c "
import json
epochs = [json.loads(l) for l in open('ml/logs/<run_name>/epoch_summary.jsonl')]
tunes = [(e['epoch'], e.get('f1_macro_tuned')) for e in epochs if e.get('f1_macro_tuned') is not None]
print('Tune history:', [(e, f'{f:.4f}') for e, f in tunes])
best = max(tunes, key=lambda x: x[1])
print(f'BEST: ep={best[0]} f1_tuned={best[1]:.4f}')

best_epoch = next(e for e in epochs if e['epoch'] == best[0])
print('Per-class F1 at best:')
for c, v in best_epoch['per_class_f1'].items():
    print(f'  {c:30s} F1={v:.3f}')
print()
print('Alerts in run:',
      sum(1 for l in open(f'ml/logs/<run_name>/alerts.jsonl')))
"
```

**Document:** best epoch + f1_tuned, per-class F1, alert count, DoS_F1 trajectory (should climb if DoS patch is active).

### 2.3 — Run N final report (NEW DOC, ~45 min)

**File:** `docs/training/GCB-P1-Run<N>-<tag>-analysis-YYYYMMDD.md`

**Template:** mirror `docs/training/GCB-P1-Run8-v10-20260605-analysis.md` (Run 8 analysis, 150 lines). Sections:

1. **Config recap** (link to project file + MEMORY.md entry)
2. **Per-epoch metrics table** (best ep, every ep10, every tune)
3. **Per-class F1 + AUC-PR + AUC-ROC at best epoch** (with comparison to Run 9 baseline = 0.2965)
4. **Per-class trajectory plot** (use `ml/scripts/interpretability/exp_s3_feature_distribution.py`)
5. **Hypothesis verification** (H1-HN from launch plan)
6. **Contamination status** (CLEAN / N=0)
7. **Benchmark results** (filled in Phase 3)
8. **Calibration results** (filled in Phase 2)
9. **Lessons learned** (what worked, what didn't — e.g., ME F1=1.0 overfit)
10. **Recommendations for Run N+1** (linked to next-training prep plan)

---

## 3. Phase 2: Calibration (~1 hour)

**Goal:** Produce per-class decision thresholds and temperature scaling for probability calibration. Both are **required** per `I_regression_guard.md` §I.3.2 for any promotion. **This phase was MISSING from the original handoff plan** and has been added back.

### 3.1 — Threshold tuning (`tune_threshold.py`)

**Script:** `ml/scripts/tune_threshold.py` (verified exists)
**What it does:** Sweeps 19 candidate thresholds per class over [0.1, 0.9] on the val set, picks the per-class threshold that maximises each class's F1. Saves 10 (or 9) floats in `CLASS_NAMES` index order.

```bash
PYTHONPATH=. ml/.venv/bin/python -m ml.scripts.tune_threshold \
    --checkpoint ml/checkpoints/GCB-P1-Run<N>-<tag>-YYYYMMDD_best.pt
# Output: ml/checkpoints/GCB-P1-Run<N>-<tag>-YYYYMMDD_best_thresholds.json
```

**Output schema (per `ml/calibration/GCB-P1-Run9-v11-20260606_thresholds.json`):**
```json
{
  "checkpoint": "ml/checkpoints/<path>",
  "architecture": "four_eye_v8",
  "num_classes": 10,
  "class_names": ["CallToUnknown", "DenialOfService", ...],
  "threshold_grid": [0.05, 0.075, 0.1, ...],
  "thresholds": {"CallToUnknown": 0.45, "DenialOfService": 0.32, ...}
}
```

### 3.2 — Temperature scaling (`calibrate_temperature.py`)

**Script:** `ml/scripts/calibrate_temperature.py` (verified exists)
**What it does:** Fits one scalar temperature T_c per class by minimising BCE NLL on the val set. Calibrated logit for class c = `logit_c / T_c`.

```bash
# v3-aware calibration (use this for Run 12+):
PYTHONPATH=. ml/.venv/bin/python ml/scripts/audit/calibrate_temperature_v3.py \
    --checkpoint ml/checkpoints/GCB-P1-Run<N>-<tag>-YYYYMMDD_best.pt \
    --out ml/calibration/run<N>/temperatures_run<N>.json
# Outputs:
#   ml/calibration/run<N>/temperatures_run<N>.json      (10-class T values)
#   ml/calibration/run<N>/temperatures_run<N>_stats.json  (ECE before/after)
#   ml/calibration/run<N>/temperatures_run<N>_ece_comparison.png  (bar chart)
```

> ⚠️ **DO NOT use legacy `ml/scripts/calibrate_temperature.py`** — it uses v9/v10 paths (`cached_dataset_v9.pkl`, `multilabel_index_deduped.csv`). Use `ml/scripts/audit/calibrate_temperature_v3.py` instead.

**Output schema (per `ml/calibration/run12/temperatures_run12.json`):** `{class_name: T_c, ...}` — one T per class.

### 3.3 — Why BOTH are needed

- **Thresholds** control the **decision boundary** for the per-class classifier. Different classes have different positive/negative ratios; a single 0.5 threshold is wrong for all of them.
- **Temperatures** control the **calibration of probabilities** (independent of decision boundary). A high T_c means the model is over-confident for class c; a low T_c means under-confident. Calibrated probabilities are needed for:
  - The 3-tier output (`confirmed`/`suspicious`/`safe`) per `api.py:35-38` (uses 0.55/0.25/0.10)
  - ECE reporting (per H.2 row `[9.3.6d]`: Brier > 0.4 = severe miscalibration)
  - Honest uncertainty communication in production

**Run 9 baseline for ECE (per `temperatures_run9_stats.json`):** Run 4 ECE was 0.205-0.310 across classes (uncalibrated). After temperature scaling, expected ECE reduction of 30-50%.

### 3.4 — Calibration verification

```bash
# Verify both files exist and are dated after the checkpoint
ls -la ml/checkpoints/GCB-P1-Run<N>-*_thresholds.json
ls -la ml/calibration/run<N>/temperatures_run<N>.json
# Both must be dated AFTER the checkpoint was saved

# Quick sanity: 10 (or 9) entries each
PYTHONPATH=. ml/.venv/bin/python -c "
import json
t = json.load(open('ml/checkpoints/GCB-P1-Run<N>-*_thresholds.json'))
T = json.load(open('ml/calibration/temperatures_run<N>.json'))
assert len(t['thresholds']) == len(T), f'mismatch: {len(t[\"thresholds\"])} vs {len(T)}'
print(f'OK: {len(t[\"thresholds\"])} classes calibrated')
"
```

---

## 4. Phase 3: External Benchmark (~1 hour)

**Goal:** Evaluate the trained model on held-out, out-of-distribution contracts from SmartBugs Curated (real-world) and SolidiFI (synthetic) using BOTH tier and tuned decision modes.

### 4.0 — ⚠️ CRITICAL CONTAMINATION FINDING (2026-06-14, MUST READ)

**Legacy `ml/scripts/check_contamination.py` is STALE and only checks SmartBugs vs BCCC (using v9/v10 paths, not v3).** The script never compared the benchmark contracts against the actual v3 training set — so we have been reporting benchmark numbers on **mostly training data**. **Use the v3-aware replacement at `ml/scripts/audit/check_contamination_v3.py` for all future contamination audits.**

**Verified via the new v3-aware audit script `/mnt/c/Users/lenovo/AppData/Local/Temp/opencode/check_contamination_v3.py` (2026-06-14):**

| Benchmark | Total contracts | In v3 (any split) | In v3 train | In v3 val | In v3 test | **Honest OOD** (not in any v3 split) |
|---|---|---|---|---|---|---|
| **SmartBugs Curated** | 143 | **137 (95.8%)** | 101 | 13 | 23 | **6** |
| **SolidiFI-benchmark** | 350 | **290 (82.9%)** | 209 | 34 | 47 | **60** |

**Honest OOD subset = contracts NOT in v3 train/val/test (the only ones that count as true OOD):**
- **SmartBugs: 6 contracts** (mostly unchecked_low_level_calls)
- **SolidiFI: 60 contracts** (~8-9 per category)

**Implication:** **All prior benchmark F1 numbers (Run 9 = 0.2965/0.3081, etc.) were INFLATED** because the model was tested on 80-95% training data. The actual OOD performance of Run 9 on the 6+60 honest subset is unknown without re-evaluation.

**New HARD GATE for Phase 3:** the v3-aware contamination audit MUST be run BEFORE any benchmark claim. If the contamination ratio is > 0% in the reported subset, the result must be either:
- (a) Restricted to the honest OOD subset (6 SB + 60 SF) — small sample, statistical power limited
- (b) Run on a different OOD source (DeFiHackLabs wild contracts, BCCC held-out, or external test set)
- (c) Reported with explicit contamination disclosure (e.g., "F1=0.65 on SmartBugs, of which 95.8% are in v3 training; honest OOD F1=0.20 on 6 contracts")

**The post-training plan for Run 12 will use option (a) by default** — run the benchmark on the honest OOD subset only. The full contaminated benchmark is still reported for continuity with prior runs, but flagged explicitly.

### 4.0.1 — What benchmarks SENTINEL uses (and why)

| Benchmark | Where | Count | Type | Why | Contamination status |
|---|---|---|---|---|---|
| **SmartBugs Curated** (PRIMARY) | `ml/data/smartbugs-curated/dataset/` | 143 raw → 137 in v3 → **6 honest OOD** | **Real-world deployed Solidity** from the DASP-10 taxonomy paper (Durieux et al. 2020) | Real attack contracts, not synthetic | 95.8% in v3; honest OOD subset = 6 contracts |
| **SolidiFI** (SECONDARY) | `ml/data/SolidiFI-benchmark/buggy_contracts/` | 350 raw → 283 in v3 → **60 honest OOD** | **Synthetic** injected single-vulnerability contracts (SolidiFI paper) | Clean ground truth (one bug per contract, known injection point) | 82.9% in v3; honest OOD subset = 60 contracts |
| **DeFiHackLabs wild** (FUTURE) | `ml/data/smartbugs-wild/` | TBD (currently training source) | Real-world exploits | Could be split: half for training, half for OOD | TBD |
| ~~BCCC~~ | n/a | n/a | n/a | **NOT a benchmark** — BCCC is a TRAINING source (after 2-tool validation per `project_bccc_2tool_audit_2026-06-14.md`) | n/a |
| ~~test_contracts/~~ | n/a | n/a | n/a | **NOT a benchmark** — massively OOD (median 20 nodes vs v3 training median 295). Used for inference API smoke only. Per `project_run8_audit_findings.md`. | n/a |

### 4.0.2 — SmartBugs honest OOD subset (the 6 contracts that count as true OOD)

Per the contamination audit on 2026-06-14:
```
SmartBugs honest OOD: 6 / 143
  access_control/parity_wallet_bug_1.sol
  reentrancy/0x4320e6f8c05b27ab4707cd1f6d5ce6f3e4b3a5a1.sol
  unchecked_low_level_calls/0x78c2a1e91b52bca4130b6ed9edd9fbcfd4671c37.sol
  unchecked_low_level_calls/0x7a4349a749e59a5736efb7826ee3496a2dfd5489.sol
  unchecked_low_level_calls/0xd2018bfaa266a9ec0a1a84b061640faa009def76.sol
  unchecked_low_level_calls/0xf70d589d76eebdd7c12cc5eec99f8f6fa4233b9e.sol
```

**Note:** 5 of the 6 are in `unchecked_low_level_calls`. The other categories (arithmetic, denial_of_service, time_manipulation, etc.) have 0 honest OOD contracts. **SmartBugs alone is NOT a sufficient OOD benchmark** for Run 12 reporting — need SolidiFI honest OOD or another source.

### 4.0.3 — SolidiFI honest OOD subset (60 contracts, more useful)

The 60 honest OOD contracts are distributed roughly evenly across categories (~8-9 per category, plus tx.origin FP probe). For per-class F1 reporting on a contaminated-but-mostly-honest split, use these 60. For full per-class breakdown, combine with SmartBugs honest OOD (6 more contracts).

The honest OOD list is regenerated by the contamination script (saved in the JSON report); copy from `/tmp/contamination_v3_<date>.json` to the Run N report.

### 4.1 — V3-aware Contamination Audit (HARD GATE, must run FIRST)

**Script:** `/mnt/c/Users/lenovo/AppData/Local/Temp/opencode/check_contamination_v3.py` (NEW, 2026-06-14)
**What it does:** Builds the v3 training/val/test SHA-256 index from `data_module/data/splits/v3/*.jsonl` and audits benchmark contracts against it. 3 tiers: exact SHA-256, normalised SHA-256, Jaccard≥0.75.

**Run:**
```bash
ml/.venv/bin/python /mnt/c/Users/lenovo/AppData/Local/Temp/opencode/check_contamination_v3.py
# Output: console report + /tmp/contamination_v3_<date>.json
```

**Pass criteria:** Document the contaminated contracts and honest OOD subset. **This gate is NOT a pass/fail** — it's an information disclosure step. The decision rule is:
- If contamination = 0% (no benchmark contracts in v3) → safe to publish full benchmark numbers
- If contamination > 0% (e.g., 95.8% for SmartBugs) → must EITHER:
  - (a) Restrict reporting to honest OOD subset (preferred for honesty)
  - (b) Report both full (with contamination flag) and honest OOD numbers
  - (c) Document the inflation explicitly: "F1=0.65 on SmartBugs full set, 95.8% in v3 training; honest OOD F1=X on 6 contracts"

**Current state (2026-06-14):** SmartBugs 95.8% contaminated, SolidiFI 82.9% contaminated. Run 12 post-training should use option (a) — honest OOD subset only.

### 4.2 — SmartBugs Curated (PRIMARY, 137 in v3, 6 honest OOD)

**Location:** `ml/data/smartbugs-curated/dataset/<category>/*.sol`
- 10 categories: `reentrancy`, `arithmetic`, `unchecked_low_level_calls`, `denial_of_service`, `front_running`, `time_manipulation`, `access_control`, `bad_randomness`, `short_addresses`, `other`
- 6 categories map to SENTINEL classes; 4 are FP probes (no SENTINEL equivalent)

**Mapping (from `benchmark_run9_smartbugs.py:44-53`):**
```python
CATEGORY_TO_CLASS = {
    "reentrancy":                 "Reentrancy",
    "arithmetic":                 "IntegerUO",
    "unchecked_low_level_calls":  "CallToUnknown",
    "denial_of_service":          "DenialOfService",
    "front_running":              "TransactionOrderDependence",
    "time_manipulation":          "Timestamp",
}
UNMAPPED_CATEGORIES = {"access_control", "bad_randomness", "short_addresses", "other"}  # FP probes
```

**Script:** `ml/scripts/benchmark_run9_smartbugs.py` (337 lines, verified)
- Note: script name has "run9" but is checkpoint-agnostic (pass `--checkpoint` explicitly)
- Loads `best.pt` + companion `_thresholds.json` (auto-detected)
- Reports per-class P/R/F1 under BOTH modes:
  - **Tier mode:** prediction counts if class is in `confirmed` (p≥0.55) or `suspicious` (p≥0.25) — lenient recall
  - **Tuned mode:** applies per-class thresholds from companion JSON — precision/recall tradeoff
- Reports OOD graph-size stats (median nodes vs v3 training median=295)
- Reports FP probe rate on unmapped categories (should be < 30%)

**Run:**
```bash
# Full benchmark (CONTAMINATED — for continuity with prior runs)
PYTHONPATH=. ml/.venv/bin/python -m ml.scripts.benchmark_run9_smartbugs \
    --checkpoint ml/checkpoints/GCB-P1-Run<N>-<tag>-YYYYMMDD_best.pt \
    --verbose > docs/reports/Run<N>/$(date +%Y-%m-%d)_smartbugs_FULL/Run<N>_smartbugs_FULL_$(date +%Y%m%d).log 2>&1

# Honest OOD only (use the list from contamination_v3_<date>.json)
# The current benchmark script does NOT support --exclude-contaminated mode.
# TODO: add --exclude-list flag that takes a list of contract names to skip.
# For now: filter the JSON output post-hoc or manually exclude the 137 contaminated
# contracts by re-running with a filtered directory.
```

**Expected output:**
- Per-mode (tier + tuned) per-class F1 for 6 mappable categories
- FP probe rate for 4 unmapped categories
- OOD graph-size distribution (median, p25, p75, %<30 nodes)
- Final F1 (micro + macro)

**Historical baseline:** Run 9 (last honest pre-45%-leakage fix): tier F1 ~0.2965, tuned F1 ~0.3081. **These numbers were on the 95.8% contaminated SmartBugs set; honest OOD F1 is unknown.**

### 4.3 — SolidiFI (SECONDARY, 290 in v3, 60 honest OOD)

**Location:** `ml/data/SolidiFI-benchmark/buggy_contracts/`
- 350 raw contracts → 290 in v3 train/val/test (per contamination audit) → 60 honest OOD
- 7 categories (per `benchmark_run9_solidifi.py:8-10`): all `pragma >=0.4.22 <0.6.0`, use solc 0.5.17

**Mapping (from `benchmark_run9_solidifi.py:45-53`):**
```python
SOLIDIFI_TO_SENTINEL = {
    "Re-entrancy":          "Reentrancy",
    "Overflow-Underflow":   "IntegerUO",
    "TOD":                  "TransactionOrderDependence",
    "Timestamp-Dependency": "Timestamp",
    "Unchecked-Send":       "CallToUnknown",
    "Unhandled-Exceptions": "MishandledException",
}
UNMAPPED = {"tx.origin"}  # FP probe
```

**Script:** `ml/scripts/benchmark_run9_solidifi.py` (289 lines, verified)
- Same script name pattern; checkpoint-agnostic via `--checkpoint`
- Has additional `--include-near-dups` flag (9 Unchecked-Send contracts are BCCC near-dups; default excluded)
- Same tier + tuned evaluation modes

**Run:**
```bash
PYTHONPATH=. ml/.venv/bin/python -m ml.scripts.benchmark_run9_solidifi \
    --checkpoint ml/checkpoints/GCB-P1-Run<N>-<tag>-YYYYMMDD_best.pt \
    --verbose > docs/reports/Run<N>/$(date +%Y-%m-%d)_solidifi_FULL/Run<N>_solidifi_FULL_$(date +%Y%m%d).log 2>&1
```

**Honest OOD report:** filter the JSON output to the 60 contracts NOT in v3. The audit script saves the full honest OOD list.

### 4.4 — OOD analysis (post-contamination)

Per `A_benchmark_runs.md` §A.2.5 + §A.4.4:

```bash
# Per-benchmark median node count (on the HONEST OOD subset, not the full set)
PYTHONPATH=. ml/.venv/bin/python -c "
import json
report = json.load(open('/tmp/contamination_v3_<date>.json'))
for bm in ['smartbugs', 'solidifi']:
    results = report[bm]['results']
    honest = [r for r in results if r['tier1_exact'] is None]
    print(f'{bm} honest OOD: {len(honest)} / {len(results)} ({100*len(honest)/len(results):.1f}%)')
    # Per-category honest OOD
    from collections import Counter
    cat_count = Counter(r['category'] for r in honest)
    for cat, n in sorted(cat_count.items()):
        print(f'  {cat}: {n}')
"
```

**Interpretation:**
- SmartBugs contracts are typically SMALLER (median ~20 nodes per `A.2.1`)
- v3 training median is 295 (per `A.2.1:106` and confirmed in audit)
- A high `<30 nodes` percentage means many benchmark contracts are OOD-tiny
- This is a known distributional difference; **must be noted in any reported result** (per `A.4.4`)

### 4.5 — Reporting rules (per `A_benchmark_runs.md` §A.4)

When writing benchmark numbers in the Run N final report (Phase 1.3) or any external doc:

1. **Always include contamination status** (per §4.1): `Contamination check: CLEAN (N=143+350, all tiers, <date>)` OR `Contamination: PARTIAL (SmartBugs 95.8% in v3; honest OOD subset: 6 contracts; full + honest F1 reported)`
2. **Always specify evaluation mode** (tier vs tuned) — they are NOT comparable
3. **Always specify checkpoint name and epoch** (from sidecar `.state.json` or `promote_model.py` output)
4. **Always note OOD graph size distribution** if median differs materially from training (295)
5. **Do not aggregate P/R/F1 across tier + tuned modes** in a single number
6. **NEW: Always report honest OOD F1 separately** from contaminated-set F1, with the contamination ratio disclosed
7. **NEW: Do not report benchmark F1 as "OOD performance"** unless the contract is in the honest OOD subset

### 4.6 — Decision: what to do about the contamination

**Three options, in order of preference for Run 12:**

| Option | What | Effort | Honesty | Sample size |
|---|---|---|---|---|
| **A (recommended)** | Report **honest OOD F1 only** (6 SB + 60 SF = 66 contracts total). Mark all prior benchmark numbers as "inflated by contamination" in MEMORY. | 0 extra | HIGH | 66 contracts (small) |
| **B** | Report **both** full contaminated F1 (for continuity with Run 9 baseline) AND honest OOD F1. Disclose contamination ratio explicitly in every mention. | 0 extra | MEDIUM | 143 + 350 (full) |
| **C** | **Decontaminate** — remove SmartBugs + SolidiFI from v3 training → v3.1 export → retrain Run 12 v3.1 (~2-3 days). Then SmartBugs + SolidiFI become truly OOD. | 2-3 days | HIGHEST | 143 + 350 (all OOD) |

**Recommendation: Option A for Run 12** (cheap, honest, runs alongside current benchmark for continuity). Defer Option C to a future "v3.1 export" plan if Ali wants truly clean OOD benchmarks.

**For Run 13 (post-prep):** if Option A is chosen for Run 12, the same applies to Run 13 unless Option C is taken (in which case Run 13's post-training is on v3.1 and all benchmarks are OOD).

### 4.7 — Future: decontaminated benchmark (v3.1 export, separate plan)

If Ali wants truly clean OOD benchmarks, the work is:
1. Remove SmartBugs + SolidiFI from `data_module/data/raw/`
2. Re-run `data_module/sentinel_data/cli.py` (preprocess → represent → label → split → export)
3. Save as `data_module/data/exports/sentinel-v3.1-noSBnoSF-2026-06-XX/`
4. v3.1 will have ~22,073 contracts (DIVE only) + DeFiHackLabs = ~22,073 + ~?, 0% SB/SF overlap
5. Re-train Run 12 (or accept the current Run 12 trained on contaminated data)
6. New benchmark numbers are now truly OOD

This is a significant data export. **Document this in `run_13_prep_plan_complete_2026-06-14.md` §"Decontaminated benchmark option" if Ali wants to pursue.**

---

## 5. Phase 4: Behaviour & API Validation (~1 hour)

**Goal:** Confirm the trained model behaves correctly via the FastAPI server (not just as a raw `.pt` file). Tests: smoke suite, known-positive round-trip, known-negative, FP probe regression, hotspot endpoint, cache invalidation.

### 5.1 — Smoke suite

**Script:** `ml/scripts/smoke/run_all.py` (verified exists, orchestrates `smoke_fix1.py` through `smoke_fix8.py`)

```bash
PYTHONPATH=. ml/.venv/bin/python -m ml.scripts.smoke.run_all
# Expected: all 8 smoke tests PASS
# Covers: graph, token, label, split, dataset, model, train step, inference
```

### 5.2 — Known-positive round-trip (per `K_inference_api.md` §K.5.1)

For each of the 10 classes (or 9 post-Run-13), pick a contract from `ml/scripts/test_contracts/` (or the v3 export) confirmed positive for that class. Hit the API and verify:
- `probabilities[<class_name>]` > the per-class tuned threshold (from Phase 2.1)
- The class appears in `confirmed` or `suspicious` (depending on probability magnitude)
- `label` is not `"safe"` for a confirmed-positive contract

```bash
# Start the API server first
cd /home/motafeq/projects/sentinel
SENTINEL_CHECKPOINT=ml/checkpoints/GCB-P1-Run<N>-<tag>-YYYYMMDD_best.pt \
PYTHONPATH=. ml/.venv/bin/python -m ml.src.inference.api &
# Wait for /health to return 200

# Round-trip test (script to create: ml/scripts/round_trip_known_positive.py)
PYTHONPATH=. ml/.venv/bin/python ml/scripts/round_trip_known_positive.py \
    --contracts-dir ml/scripts/test_contracts/ \
    --thresholds ml/checkpoints/GCB-P1-Run<N>-*_thresholds.json
# Expected: 10/10 (or 9/9) classes correctly predicted
```

### 5.3 — Known-negative test (per `K_inference_api.md` §K.5.2)

Run at least 5 clean contracts with no known vulnerabilities. For each, verify:
- `confirmed` list is empty
- `probabilities` values are all below `tier_thresholds["confirmed"]` (0.55)
- `label` is `"safe"` or `"suspicious"` (not `"confirmed_vulnerable"`)

**Source for known-negative contracts:** DIVE's NonVulnerable pool (DISL NonVulnerable cap 3:1 per `data_module/docs/architecture.md` v2 section) OR a curated set in `ml/scripts/test_contracts/non_vulnerable/`.

### 5.4 — FP probe regression (per `K_inference_api.md` §K.5.3)

For any contract previously used as a False Positive probe in prior runs: confirm its prediction has not regressed. The 4 unmapped SmartBugs categories (`access_control`, `bad_randomness`, `short_addresses`, `other`) are the canonical FP probes. Run on all contracts in these categories and confirm:
- CONF+SUSP rate < 30% (per `A_benchmark_runs.md` §A.2.6)

### 5.5 — Hotspot endpoint sanity (per `K_inference_api.md` §K.5.4)

For one known-positive Reentrancy contract, call `/hotspots` and verify:
- `hotspots` list is non-empty
- `hotspot_stats["attention_source"]` is present
- `hotspot_stats["total_function_nodes"]` > 0
- The top-scored hotspot `fn_name` is a function name present in the contract source
- `label`, `probabilities`, `confirmed`/`suspicious` match the `/predict` response for the same contract

```bash
curl -s -X POST http://localhost:8000/hotspots \
    -H 'Content-Type: application/json' \
    -d '{"source_code": "<known_reentrancy_contract_source>"}' | python -m json.tool
```

### 5.6 — Inference API smoke (per `K_inference_api.md` §K.3-K.4)

```bash
# K.3.1 — /health
curl -s http://localhost:8000/health | python -m json.tool
# Verify:
#   "predictor_loaded": true
#   "thresholds_loaded": true  (per-class thresholds found and applied)
#   "architecture" matches expected (e.g., "four_eye_v8")
#   "model_epoch" and "model_f1_val" match the promoted checkpoint

# K.3.3 — Tier thresholds
PYTHONPATH=. ml/.venv/bin/python -c "
import requests
h = requests.get('http://localhost:8000/health').json()
print('tier_thresholds:', h['tier_thresholds'])
# Expected: {\"confirmed\": 0.55, \"suspicious\": 0.25, \"noteworthy\": 0.10}
"

# K.4.2 — Cache invalidation check
# Per preprocessor cache key: {content_md5}_{FEATURE_SCHEMA_VERSION}
# If schema changed since last cache, old entries are unreachable; this is automatic.
# Manual check: see if any eviction warnings in API logs.
```

### 5.7 — Phase 4 Pass criteria

ALL must be true:
- [ ] All 8 smoke tests PASS
- [ ] ≥ 90% (ideally 100%) of known-positive contracts predict the right class
- [ ] 100% of known-negative contracts have empty `confirmed` list
- [ ] FP probe rate on SmartBugs unmapped categories < 30%
- [ ] Hotspot endpoint returns non-empty `hotspots` for known Reentrancy contract
- [ ] `/health` reports `thresholds_loaded: true`
- [ ] Cache key reflects current schema version (no eviction warnings in logs)

---

## 6. Phase 5: Promotion (~30 min)

**Goal:** Move the validated checkpoint to MLflow Model Registry at Staging (always) or Production (if statistical sig + drift baseline).

### 6.1 — Pre-promotion checklist

| Gate | Required for | Verified by |
|---|---|---|
| Phase 1 reproducibility checks | Staging + Production | §2.1 above |
| Phase 2 calibration files present | Staging (WARNING only) + Production (HARD) | §3.1, §3.2 outputs |
| **Phase 3 v3-aware contamination audit** (NEW HARD GATE) | Staging + Production | §4.1 — contamination_v3_<date>.json report; honest OOD F1 reported |
| Phase 3 benchmark run (with honest OOD disclosure) | Staging + Production | §4.2, §4.3, §4.4, §4.5 |
| Phase 4 behaviour checks PASS | Staging + Production | §5.1-§5.6 |
| MEMORY.md Training History updated | Staging + Production | `~/.claude/projects/.../memory/MEMORY.md` row added |
| **Drift baseline `source=warmup`** | **Production only (HARD)** | `ml/data/drift_baseline.json` field |
| **Statistical significance vs current Production** | **Production only (HARD)** | Bootstrap CI script (see §6.3) |
| **Strict F1 improvement vs current Production** | **Production only (HARD)** | `promote_model.py` enforces `>` (not `>=`) |

**NEW contamination gate enforcement:** the Run N final report (Phase 1.3) MUST include a section "Contamination Status" linking to `/tmp/contamination_v3_<date>.json` and the honest OOD F1. If this section is missing, **promote_model.py dry-run should be extended to FAIL on missing section** (TODO).

### 6.2 — Staging promotion (always, per `I_regression_guard.md` §I.2.1)

```bash
# Dry-run first
PYTHONPATH=. ml/.venv/bin/python ml/scripts/promote_model.py \
    --checkpoint ml/checkpoints/GCB-P1-Run<N>-<tag>-YYYYMMDD_best.pt \
    --stage Staging \
    --val-f1-macro 0.XXXX \
    --note "Run N final: <tag>, <num_classes>-class" \
    --dry-run
# Verify output:
#   F1 gate: <new_f1> > Staging <none or prior> ✓
#   Thresholds JSON: found/missing
#   Architecture: <from checkpoint>

# Live
PYTHONPATH=. ml/.venv/bin/python ml/scripts/promote_model.py \
    --checkpoint ml/checkpoints/GCB-P1-Run<N>-<tag>-YYYYMMDD_best.pt \
    --stage Staging \
    --val-f1-macro 0.XXXX \
    --note "Run N final: <tag>, <num_classes>-class"
```

**Default experiment:** `sentinel-retrain-v2` (per `promote_model.py:50` constant `DEFAULT_EXPERIMENT`). Override with `--experiment <name>` if needed.

**Staging is for evaluation, not deployment.** No F1 regression gate enforced. **This is the only promotion done as part of post-training — Production is the next-training prep's job.**

### 6.3 — Production promotion (CONDITIONAL, per `I_regression_guard.md` §I.2.2)

**Only proceed if ALL of the following:**
1. Drift baseline is `source=warmup` (currently `PLACEHOLDER` per `ml/data/drift_baseline.json`)
2. f1_tuned is **strictly greater** than current Production's f1_tuned
3. Statistical significance: 95% CI lower bound > 0 (bootstrap script below)
4. No class F1 regression > 0.05 vs current Production (per `I_regression_guard.md` §I.4)

**Drift baseline rebuild (Production gate, required before any Production promotion):**
```bash
# 1. Wait for API to collect warmup traffic (production-like predictions)
# 2. Rebuild:
PYTHONPATH=. ml/.venv/bin/python ml/scripts/compute_drift_baseline.py --source warmup
# Verify: jq '.source' ml/data/drift_baseline.json → "warmup"
# Verify: jq '.status' ml/data/drift_baseline.json → "OK" (not PLACEHOLDER)
```

**Bootstrap CI for statistical significance (NEW):**

The original handoff plan didn't include a significance test. Adding this to make Production decisions data-driven:

```bash
# Script: /mnt/c/Users/lenovo/AppData/Local/Temp/opencode/bootstrap_f1_diff.py
PYTHONPATH=. ml/.venv/bin/python /mnt/c/Users/lenovo/AppData/Local/Temp/opencode/bootstrap_f1_diff.py \
    --run-a ml/checkpoints/Run<N-1>_FINAL.pt \
    --run-b ml/checkpoints/Run<N>_FINAL.pt \
    --splits data_module/data/splits/<active>/val.jsonl \
    --n-bootstrap 10000
# Output: 95% CI for (RunN - RunN-1) f1_tuned difference
# Decision rule: if lower bound > 0 → significant improvement
```

**Production dry-run:**
```bash
PYTHONPATH=. ml/.venv/bin/python ml/scripts/promote_model.py \
    --checkpoint ml/checkpoints/GCB-P1-Run<N>-<tag>-YYYYMMDD_best.pt \
    --stage Production \
    --val-f1-macro 0.XXXX \
    --require-baseline ml/data/drift_baseline.json \
    --note "Run N final: <tag>, <num_classes>-class, statistical sig vs Run N-1" \
    --dry-run
# Verify: F1 gate: <new_f1> > Production <current> ✓ (strictly greater)
# Verify: Drift baseline: source=warmup ✓
# Verify: Thresholds JSON: found

# Live
PYTHONPATH=. ml/.venv/bin/python ml/scripts/promote_model.py \
    --checkpoint ml/checkpoints/GCB-P1-Run<N>-<tag>-YYYYMMDD_best.pt \
    --stage Production \
    --val-f1-macro 0.XXXX \
    --require-baseline ml/data/drift_baseline.json \
    --note "Run N final: <tag>, <num_classes>-class, statistical sig vs Run N-1"
```

### 6.4 — Save artifacts (~15 min)

```bash
# Immutable checkpoint name (archival only; do NOT point cron at this)
cp ml/checkpoints/GCB-P1-Run<N>-<tag>-YYYYMMDD_best.pt \
   ml/checkpoints/GCB-P1-Run<N>-<tag>-YYYYMMDD_FINAL.pt

# Log archive
cd ml/logs
tar czf _archive/Run<N>_<date>.tar.gz <run_name>/
cd -

# MLflow experiment snapshot
mlflow runs list --experiment-name sentinel-multilabel > ml/reports/Run<N>_mlflow_runs.txt

# Update MEMORY.md Training History (add row) + Current State
```

---

## 7. Decision gates (where Ali sign-off is needed)

| # | After | Question to Ali | If NO |
|---|---|---|---|
| 1 | Phase 1.3 (Run N final report) | OK to proceed to Calibration + Benchmark? | Diagnose; skip to rollback |
| 2 | Phase 3 (Benchmark done) | Is the result good enough to keep in Staging? | Skip Promotion; investigate |
| 3 | Phase 5.2 (Staging promoted) | OK to mark post-training as DONE? | Halt; diagnose; possibly rollback |
| 4 | Phase 5.3 (if Production eligible) | Promote to Production (requires drift baseline + statistical sig)? | Keep in Staging; plan Run N+1 fixes |

**After Phase 5.3:** HANDOFF to `data_module/temp/live_plans/run_13_prep_plan_complete_2026-06-14.md` for the next-training prep (4 fixes + v4 + Run 13 launch + monitoring).

---

## 8. Scripts inventory (verified 2026-06-14)

| Phase | Script | Path | Output | Status |
|---|---|---|---|---|
| 1 | (none — analysis only) | — | Run N final report | NEW DOC each run |
| 2.1 | `tune_threshold.py` | `ml/scripts/tune_threshold.py` | `<stem>_thresholds.json` | exists |
| 2.2 | `calibrate_temperature.py` | `ml/scripts/calibrate_temperature.py` | `temperatures_run<N>.json` + `*_ece_comparison.png` + `*_stats.json` | exists |
| 3.0 (HARD GATE) | `check_contamination_v3.py` | `/mnt/c/Users/lenovo/AppData/Local/Temp/opencode/check_contamination_v3.py` | Console report + `/tmp/contamination_v3_<date>.json` | **NEW 2026-06-14** (replaces stale legacy) |
| 3.0 (legacy, supplemental) | `check_contamination.py` | `ml/scripts/check_contamination.py` | SmartBugs vs BCCC only (v9/v10 paths) | exists, INCOMPLETE for v3 |
| 3.1 | `benchmark_run9_smartbugs.py` | `ml/scripts/benchmark_run9_smartbugs.py` | Per-mode F1, OOD stats, FP probe | exists (Run 9 in name, checkpoint-agnostic); **TODO: add --exclude-list flag** |
| 3.2 | `benchmark_run9_solidifi.py` | `ml/scripts/benchmark_run9_solidifi.py` | Per-mode F1 on SolidiFI categories | exists; **TODO: add --exclude-list flag** |
| 4.1 | `smoke/run_all.py` | `ml/scripts/smoke/run_all.py` | 8 smoke tests PASS/FAIL | exists |
| 4.2-5.6 | `round_trip_known_positive.py` (TODO) | `ml/scripts/round_trip_known_positive.py` | N/10 (or N/9) classes correctly predicted | **TO CREATE** |
| 5.3 | `compute_drift_baseline.py` | `ml/scripts/compute_drift_baseline.py` | `ml/data/drift_baseline.json` | exists (currently PLACEHOLDER) |
| 6.3 | `promote_model.py` | `ml/scripts/promote_model.py` | MLflow Model Registry update | exists |
| 6.3 | `bootstrap_f1_diff.py` | `/mnt/c/Users/lenovo/AppData/Local/Temp/opencode/bootstrap_f1_diff.py` | 95% CI for f1 difference | **TO CREATE** |
| 4.5 (diagnostic) | `diag_per_eye_solidifi.py` | `ml/scripts/diag_per_eye_solidifi.py` | Per-class SolidiFI breakdown | exists |
| 4.5 (diagnostic) | `interpretability/exp_s3_feature_distribution.py` | `ml/scripts/interpretability/exp_s3_feature_distribution.py` | Per-feature distribution plot | exists |

### Calibration artifacts that EXIST currently (verified)

| File | Run | Classes |
|---|---|---|
| `ml/calibration/GCB-P1-Run9-v11-20260606_thresholds.json` | Run 9 | 10 |
| `ml/calibration/temperatures_run4.json` + `_ece_comparison.png` + `_stats.json` | Run 4 | 10 |
| `ml/calibration/temperatures_run7.json` | Run 7 | 10 |
| `ml/calibration/temperatures_run9.json` + `_ece_comparison.png` + `_stats.json` | Run 9 | 10 |
| `ml/calibration/run9_calibration_report.md` | Run 9 | 10 |
| `ml/checkpoints/GCB-P1-Run7-v10-20260603_best_thresholds.json` | Run 7 | 10 |
| `ml/checkpoints/GCB-P1-Run8-v10-20260605_best_thresholds.json` | Run 8 | 10 |

**Not yet present:**
- `ml/checkpoints/GCB-P1-Run12-v3dospatched-20260613_best_thresholds.json` (will be created in Phase 2.1)
- `ml/calibration/temperatures_run12.json` (will be created in Phase 2.2)
- `ml/calibration/temperatures_run13.json` + Run 13 thresholds (will be created in Phase 2 of Run 13's post-training)

### Tests that exist (must pass)

`ml/tests/`: 12 test files (test_api, test_cache, test_cfg_embedding_separation, test_drift_detector, test_fusion_layer, test_gnn_encoder, test_model, test_predictor, test_preprocessing, test_promote_model, test_sentinel_dataset, test_trainer)

`ml/scripts/smoke/`: 8 smoke_fix scripts + run_all.py orchestrator

---

## 9. Files to create / update (per Phase)

### Phase 1 (Run Validation)

**Create:**
- `docs/training/GCB-P1-Run<N>-<tag>-analysis-YYYYMMDD.md` (the Run N final report)

**Update:**
- `~/.claude/projects/-home-motafeq-projects-sentinel/memory/MEMORY.md` Current State (note Run N final complete)
- `~/.claude/projects/-home-motafeq-projects-sentinel/memory/project_run<N>_launch.md` (add final section)

### Phase 2 (Calibration)

**Create:**
- `ml/checkpoints/GCB-P1-Run<N>-*_thresholds.json` (auto-created by `tune_threshold.py`)
- `ml/calibration/temperatures_run<N>.json` (auto-created by `calibrate_temperature.py`)
- `ml/calibration/temperatures_run<N>_stats.json` (auto-created)
- `ml/calibration/temperatures_run<N>_ece_comparison.png` (auto-created)

**Update:** none (artifacts only)

### Phase 3 (Benchmark)

**Create:**
- `ml/reports/Run<N>_smartbugs_YYYYMMDD.log` (benchmark output)
- `ml/reports/Run<N>_solidifi_YYYYMMDD.log` (benchmark output)

**Update:**
- Run N final report (Phase 1.3) — add §7 Benchmark Results

### Phase 4 (Behaviour & API)

**Create:**
- `ml/scripts/round_trip_known_positive.py` (the round-trip test orchestrator) — **TODO if not exists**

**Update:** none

### Phase 5 (Promotion)

**Create:**
- `ml/checkpoints/GCB-P1-Run<N>-*_FINAL.pt` (immutable copy)
- `ml/logs/_archive/Run<N>_<date>.tar.gz` (log archive)
- `ml/reports/Run<N>_mlflow_runs.txt` (MLflow snapshot)

**Update:**
- `~/.claude/projects/-home-motafeq-projects-sentinel/memory/MEMORY.md` Training History + Current State
- MLflow Model Registry (via `promote_model.py`)

### One-time (across all runs)

**Create:**
- `/mnt/c/Users/lenovo/AppData/Local/Temp/opencode/bootstrap_f1_diff.py` (the bootstrap CI script for §6.3)

---

## 10. Effort summary

| Phase | Time | Owner |
|---|---|---|
| Phase 1: Run Validation | 30 min | Claude |
| Phase 2: Calibration | 1 hour | Claude |
| Phase 3: External Benchmark | 1 hour | Claude |
| Phase 4: Behaviour & API | 1 hour | Claude |
| Phase 5: Promotion | 30 min | Claude (Staging) / Claude + Ali (Production) |
| **Total** | **~4-5 hours** | Mostly Claude; Ali sign-off at 2 gates |

**This plan is Run-N agnostic.** The same 5 phases apply to Run 12 (now training), Run 13 (next), or any future run. The actual scripts, files, and commands are templates that take the run name + checkpoint path as inputs.

---

## 11. Out-of-scope (handled by separate plan)

The following are NOT part of post-training and are documented in **`data_module/temp/live_plans/run_13_prep_plan_complete_2026-06-14.md`**:

- **Data fixes** (e.g., drop GasException, strip Solidifi `bug_*`, inject BCCC ME)
- **Model architecture changes** (e.g., extend L4 to drop `loc`)
- **v4 export** (rebuild data with all fixes)
- **Run 13 launch** (next training run)
- **Run 13 monitoring** (cron + PowerShell toast for next run)

If Ali wants to do those things AFTER Run 12's post-training is done, switch to that plan.

---

## 12. What's still missing (per `post_training_process_2026-06-14.md` §6)

- `inference_hardening_2026-06-XX.md` (TODO) — Phase 4 partially covers this (the API smoke + round-trip tests)
- `autoML_baseline_2026-06-XX.md` (TODO, parked) — needs re-run on v3; not part of post-training
- `seam_swap_completion_2026-06-13.md` (open Q1-Q3) — independent of post-training; deferred
- `run14_cgt_ingestion_2026-06-XX.md` (TODO) — if Run 13 < 0.72 f1_tuned; covered in `data-source-addition-plan-2026-06-13.md`
- `runN_solidifi_promotion_2026-06-XX.md` (TODO) — Production promotion decision; covered in Phase 5.3 + 6.3 bootstrap CI

## 13. Critical findings (2026-06-14 audit)

### 13.1 — Benchmark contamination (CRITICAL, NEW)

**The legacy `ml/scripts/check_contamination.py` only checks SmartBugs vs BCCC, not SmartBugs vs v3 training. This is why we have been reporting benchmark F1 numbers that were 80-95% contaminated.**

**Verified contamination ratio (per the new v3-aware audit, 2026-06-14):**
- SmartBugs: 137/143 (95.8%) in v3 train/val/test → only **6 honest OOD contracts**
- SolidiFI: 290/350 (82.9%) in v3 train/val/test → only **60 honest OOD contracts**

**Implication:** All prior "benchmark F1" numbers (Run 9 tier 0.2965, tuned 0.3081, etc.) were testing on 80-95% training data. The actual OOD performance of prior runs is unknown without re-evaluation on the honest OOD subset (66 contracts total).

**Action items:**
- [x] New v3-aware contamination script: `/mnt/c/Users/lenovo/AppData/Local/Temp/opencode/check_contamination_v3.py`
- [x] Documentation: §4.0 in this plan + decision in §4.6 (Option A recommended for Run 12)
- [ ] TODO: Add `--exclude-list` flag to `benchmark_run9_smartbugs.py` + `benchmark_run9_solidifi.py` to skip contaminated contracts at the script level
- [ ] TODO: Extend `promote_model.py` to FAIL on missing "Contamination Status" section in the Run N final report
- [ ] TODO: Decide whether to pursue v3.1 export (decontaminated training) — covered in `run_13_prep_plan_complete_2026-06-14.md` §"Decontaminated benchmark option"
- [ ] TODO: Retroactively re-evaluate Run 9, 10, 11 on the honest OOD subset (66 contracts) to establish a true OOD baseline

### 13.2 — Calibration files MISSING for Run 12 (CRITICAL)

**Per `I_regression_guard.md` §I.3.2, promotion to Staging/Production requires both `<stem>_thresholds.json` and `<stem>_temperatures.json`.** Run 12 currently has neither (verified via `ls ml/checkpoints/GCB-P1-Run12*`). **Phase 2 of the post-training process MUST add these before any promotion.**
