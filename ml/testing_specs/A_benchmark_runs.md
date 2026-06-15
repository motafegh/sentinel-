# A — Benchmark Runs

> Always load `00_rules.md` before following this procedure.
> Apply Rule 2 (gate assertions + completion attestation) at every step.
>
> **Last revised: 2026-06-14** (post-Run-12 launch). Updated paths to v3 export (`data_module/data/exports/sentinel-v3-smartbugs-2026-06-13/`) and v3 splits (`data_module/data/splits/v3/`). Updated default checkpoint to point at Run 12 (current best, in flight). Added note about BCCC re-evaluation findings (callToUnknown/GasException noise — see `project_bccc_2tool_audit_2026-06-14.md`).

---

## When This File Applies

- Before presenting any benchmark result to an external audience
- When comparing two runs on an external dataset (SmartBugs, SolidiFI)
- When a contamination question arises about training / benchmark overlap
- After a new run completes and benchmark evaluation is next

Always load alongside: `C_diagnostic_checks.md` (C.2 smoke inference is a
pre-condition for A.2), `F_new_run_checklist.md` (F.3.2 contamination gate).

---

## A.1 — Contamination Check

Run `ml/scripts/check_contamination.py` before presenting any benchmark
result. Do not skip or defer this. Read the script header before running
to understand what it does.

### A.1.1 — What the Script Checks

Four detection tiers (from `check_contamination.py` docstring):

| Tier | Method | What it catches |
|---|---|---|
| 1a | SHA-256 of raw bytes vs BCCC index | Exact copies |
| 1b | SmartBugs content SHA-256 == BCCC filename stem | BCCC named-by-hash copies |
| 2 | SHA-256 of normalised content (comments stripped, whitespace collapsed, lowercased) | Reformatted / re-annotated copies |
| 3 | Token Jaccard ≥ 0.75 vs top-50 BCCC candidates (pre-filtered by length) | Near-duplicates |
| 4 | (num_nodes, num_edges, sorted function-name list) structural match | Same contract compiled differently |

Tier 4 loads all 41,576 training `.pt` files and takes ~3 minutes.
Pass `--no-tier4` only if you are doing a quick preliminary check
and plan to re-run with Tier 4 before final reporting.

### A.1.2 — Invoking the Check

```bash
# Full check (all four tiers)
PYTHONPATH=. python -m ml.scripts.check_contamination

# Quick check (Tier 1-3 only, ~2 min)
PYTHONPATH=. python -m ml.scripts.check_contamination --no-tier4

# Non-default Jaccard threshold
PYTHONPATH=. python -m ml.scripts.check_contamination \
    --jaccard-threshold 0.80 --top-k-candidates 100
```

Required paths (read from `check_contamination.py` constants):
- `ml/data/smartbugs-curated/dataset/` — SmartBugs Curated (143 real contracts)
- **`/home/motafeq/projects/sentinel/BCCC-SCsVul-2024/SourceCodes/`** — 111,897 BCCC training contracts (read-only)
- **Active v3 export: `data_module/data/exports/sentinel-v3-smartbugs-2026-06-13/`** — 22,493 contracts, 5 shards, ~21,657 graphs. (The legacy path `ml/data/graphs/` with 41,576 graphs was for v9/v10 — DO NOT USE for v3 runs.)
- **Active v3 splits: `data_module/data/splits/v3/`** — 18,596/1,983/1,914 (0% leakage). (The legacy path `ml/data/splits/deduped/` was for v9/v10 — DO NOT USE for v3 runs.)

If any required path is missing the script prints a clear error and exits 1.

### A.1.3 — Interpreting the Output

The summary block at the end of the run reports:

```
  SmartBugs contracts checked : 143
  Exact content match         : N
  Near-duplicate (J≥0.75)     : N
  Structural graph match      : N
  Total flagged (any tier)    : N
```

(Note: the **v3** export now has **137** SmartBugs contracts (the original 143 minus 6 that failed compilation in the v3 preprocess stage). Use 137 for v3+ benchmark contamination checks; 143 is the v9/v10 number.)

- `Total flagged = 0` — safe to publish benchmark numbers
- `Total flagged > 0` — inspect each flagged contract; determine whether
  it appears in the **training** split specifically (Tier 4 shows the split).
  A contract only in `val` or `test` is less severe than one in `train`,
  but all flagged contracts must be documented
- Jaccard distribution is printed for diagnostic use; a high median Jaccard
  (> 0.5) across all contracts suggests the datasets share a common Solidity
  codebase style and does not by itself indicate contamination

---

## A.2 — SmartBugs Curated Benchmark

Read `ml/scripts/benchmark_run9_smartbugs.py` before running. The script
has hardcoded paths for the Run 9 checkpoint and thresholds JSON — verify
these match the checkpoint you want to evaluate before invoking.

### A.2.1 — Paths and Pre-Conditions

Hardcoded in the script (read to confirm before each run):

| Constant | Value |
|---|---|
| `DEFAULT_CKPT` | **Historical: `ml/checkpoints/GCB-P1-Run9-v11-20260606_best.pt` (Run 9, last honest F1 before 45% leakage fix).** Current best: `ml/checkpoints/GCB-P1-Run12-v3dospatched-20260613_best.pt` (Run 12, f1_tuned=0.6941, in flight as of 2026-06-14). Always verify the active best from `MEMORY.md` Current State. |
| `THRESHOLDS_JSON` | Companion to active checkpoint: e.g., `ml/calibration/GCB-P1-Run12-v3dospatched-20260613_thresholds.json` (created when Run 12 finalizes). |
| `SMARTBUGS_DIR` | `ml/data/smartbugs-curated/dataset/` |
| `TRAINING_MEDIAN_NODES` | **v9/v10: 90. v3: 295** (mean, from Run 12 audit). SmartBugs contracts are typically smaller (median ~20). The v3 number reflects post-DoS-patch + L3-deduped data. |
| `TIER_CONFIRMED` | 0.55 |
| `TIER_SUSPICIOUS` | 0.25 |

If evaluating a new checkpoint, pass `--checkpoint` explicitly —
do not modify the hardcoded default without a corresponding script update.

### A.2.2 — Category → Class Mapping

Six SmartBugs categories map to SENTINEL classes; four are unmapped FP probes:

| SmartBugs category | SENTINEL class | Notes |
|---|---|---|
| `reentrancy` | Reentrancy | Largest category |
| `arithmetic` | IntegerUO | |
| `unchecked_low_level_calls` | CallToUnknown | |
| `denial_of_service` | DenialOfService | |
| `front_running` | TransactionOrderDependence | |
| `time_manipulation` | Timestamp | |
| `access_control` | *(FP probe)* | No SENTINEL equivalent |
| `bad_randomness` | *(FP probe)* | No SENTINEL equivalent |
| `short_addresses` | *(FP probe)* | No SENTINEL equivalent |
| `other` | *(FP probe)* | No SENTINEL equivalent |

### A.2.3 — Evaluation Modes

The script evaluates under two threshold modes simultaneously:

- **Tier mode** — a prediction counts as a hit if the class appears in either
  `confirmed` (p ≥ 0.55) or `suspicious` (p ≥ 0.25). This is the lenient recall mode
- **Tuned mode** — applies the per-class thresholds from the companion
  `_thresholds.json`. Default fallback if a class is not in the JSON: 0.325
  (read `classify_with_tuned_thresholds()` for the exact fallback)

For recall-focused external reporting, use tier mode (0.55/0.25) results.
For precision-recall tradeoff reporting, use tuned mode.

### A.2.4 — Running the Benchmark

```bash
# Standard run (summary only)
PYTHONPATH=. python -m ml.scripts.benchmark_run9_smartbugs

# Verbose (per-contract predictions)
PYTHONPATH=. python -m ml.scripts.benchmark_run9_smartbugs --verbose

# Explicit checkpoint
PYTHONPATH=. python -m ml.scripts.benchmark_run9_smartbugs \
    --checkpoint ml/checkpoints/<run-name>_best.pt
```

### A.2.5 — OOD Graph Size Interpretation

The script reports benchmark graph size vs training median (90 nodes).
SmartBugs contracts are typically smaller — a high `<30 nodes` percentage
means many benchmark contracts are OOD-tiny and may produce lower-confidence
probabilities. This is a known distributional difference and must be noted
when reporting results.

### A.2.6 — FP Probe Interpretation

The FP probe section reports what fraction of the four unmapped categories
receive a `confirmed` or `suspicious` flag. A `CONF+SUSP rate` above 30%
on unmapped categories indicates the model is over-triggering and the
`suspicious` threshold (0.25) may be too low for production use.

---

## A.3 — SolidiFI Benchmark

Read `ml/scripts/benchmark_run9_solidifi.py` before running. The SolidiFI
benchmark covers injected single-vulnerability contracts with known ground truth.

SolidiFI uses the same evaluation logic as SmartBugs but with different
category-to-class mapping (read the script header before running).
The contamination check (`A.1`) must also be run against the SolidiFI
contract set before reporting.

---

## A.4 — Reporting Rules

When writing benchmark numbers into any document:

1. Always include the contamination status: `Contamination check: CLEAN (N=143, all tiers, <date>)`
   or `Contamination status: NOT RUN` if it was not run — never omit
2. Always specify which evaluation mode produced the number (tier vs tuned)
3. Always specify the checkpoint name and its epoch (from the checkpoint
   sidecar `.state.json` or the `epoch` field from `promote_model.py` output)
4. Always note the OOD graph size distribution if the benchmark median node
   count differs materially from the training median (90 nodes)
5. Do not aggregate Precision/Recall/F1 across the two evaluation modes in
   a single number — they are not comparable

---

## A.5 — Completion Attestation

After completing this section, append to the relevant run / report doc:

```
## Procedure Attestation — A_benchmark_runs — <ISO date>
Steps completed:
  A.1 contamination check:            CLEAN / FLAGGED (N=X contracts) / NOT RUN
    Tiers run:                          1a / 1b / 2 / 3 / 4 (circle applied)
    Jaccard threshold:                  0.75 (or override)
    Result:                             total flagged = N
  A.2 SmartBugs benchmark:            RUN / NOT RUN
    Checkpoint:                         <path>
    Thresholds JSON:                    found / missing (uniform 0.325 fallback)
    Evaluation mode reported:           tier / tuned
    OOD graph size noted:               YES / NO
  A.3 SolidiFI benchmark:             RUN / NOT RUN
  A.4 reporting rules applied:         YES / NO
Steps skipped:     [any skipped + explicit reason]
Unverified items:  [anything not confirmable]
New findings:      [link to audit doc entry, or "none"]
Written to:        [path of this attestation]
```
