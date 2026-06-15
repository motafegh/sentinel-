# SmartBugs Wild — OOD Separation Plan (2026-06-15)

> **Purpose:** After the full 47K eval finishes, separate the results into (a) contracts
> the model already saw in training vs (b) genuine OOD contracts, and re-report stats
> honestly on each group.
>
> **Gate this satisfies:** `ml/testing_specs/A_benchmark_runs.md` §A.1 contamination check —
> mandatory before reporting any benchmark result to an external audience.
>
> **Canonical tracking:** this file. Update in-place as steps complete.
>
> **Related:** `2026-06-14_project_run12_post_training.md` (post-training context),
> `2026-06-15_project_smartbugs_wild_full_eval.md` (eval launch + live progress).

---

## Why This Matters

v3 training data is **real Ethereum mainnet contracts** (DIVE source confirmed).
SmartBugs Wild is also real Ethereum mainnet. There is a non-trivial probability
of overlap — we cannot know without checking.

If we report "Run 12 gets X% trigger rate on 47K real-world contracts" without
separating seen vs unseen, that number is inflated by memorisation, the same
category of bug that made Run 10 F1=0.683 meaningless. We have done this separation
before (SmartBugs Curated 95.8% contaminated, SolidiFI 82.9% — the v3 quickstart
benchmark honesty finding). This plan applies the same discipline to Wild.

Prior work already done (DO NOT REDO):
- `check_contamination_v3.py` — contamination logic for Curated + SolidiFI (template, reuse)
- `build_v3_index()` function — builds SHA-256 → split mapping, reuse directly
- v3 splits index (sha256 field in `data_module/data/splits/v3/{train,val,test}.jsonl`)

---

## Data Facts

| Dataset | Size | Contract IDs | Origin |
|---|---|---|---|
| SmartBugs Wild (target) | 47,398 | ETH address (0x…) → `.sol` file | Real Ethereum mainnet |
| v3 train | 18,596 | SHA-256 of source text | DIVE (22,073 real ETH) + SmartBugs Curated (137) + SolidiFI (283) |
| v3 val | 1,983 | SHA-256 of source text | Same sources |
| v3 test | 1,914 | SHA-256 of source text | Same sources |

Matching key: compute SHA-256 of each Wild `.sol` file → look up in v3 splits index.
Use 3 tiers to catch reformatted copies.

---

## Step 0 — Wait for Eval to Finish

**Status:** eval PID 984402 running, 50% done (~2h remaining as of 2026-06-15 00:33 UTC).

Do not start Step 1 until eval has finished and written its final report.
Final report will appear in `docs/reports/2026-06-14_ml_Run12_eval_full_eval_smartbugs_wild_47K_in_progress/`.

*Optionally:* Step 1 (building the contamination index) can run in parallel because it
only reads the `.sol` files and the v3 splits — it does NOT touch the eval state file.
This saves ~10-15 min. Your call.

---

## Step 1 — Build Contamination Index for Wild

**Goal:** For every Wild contract, determine: in-train / in-val-test / OOD.

**Script to write:** `ml/scripts/audit/check_contamination_wild.py`

Template: `ml/scripts/audit/calibrate_temperature_v3.py` for file conventions,
and the `build_v3_index()` + matching logic from `check_contamination_v3.py`
(currently at `/mnt/c/Users/lenovo/AppData/Local/Temp/opencode/check_contamination_v3.py`).

### Detection tiers

| Tier | Method | Notes |
|---|---|---|
| 1 | Exact SHA-256 of raw `.sol` bytes vs v3 sha256 | Fastest; catches verbatim copies |
| 2 | SHA-256 of normalised content (comments stripped, whitespace collapsed, lowercased) | Catches reformatted/re-annotated copies |
| 3 | Token Jaccard ≥ 0.75 vs v3 candidates (length-filtered ±50%) | Catches near-duplicates |

Tier 3 on 47K × 22K is expensive — pre-filter by length (±50% LOC) before Jaccard.
Cap at Tier 3 only against contracts that pass length filter.

### Outputs

```
ml/reports/Run12_smartbugs_wild_contamination_index.json  ← per-address verdict
ml/reports/Run12_smartbugs_wild_contamination_summary.json ← aggregate stats
```

Per-address record:
```json
{
  "address": "0x...",
  "tier_hit": 1,           // 0 = OOD, 1/2/3 = tier that matched
  "v3_split": "train",     // "train" | "val" | "test" | null
  "v3_sha256": "...",      // matched sha256, or null
  "v3_source": "dive"      // "dive" | "smartbugs_curated" | "solidifi" | null
}
```

### Expected runtime

- Tier 1+2: read 47,398 `.sol` files, hash each → ~5-10 min
- Tier 3 (Jaccard): only on non-matched contracts after Tier 2 → estimate 20-40 min
- Total: ~30-50 min on WSL2

### How to run

```bash
cd ~/projects/sentinel
source ml/.venv/bin/activate
python ml/scripts/audit/check_contamination_wild.py \
    --wild-dir ml/data/smartbugs-wild/contracts \
    --v3-splits data_module/data/splits/v3 \
    --output ml/reports/Run12_smartbugs_wild_contamination_index.json \
    --jaccard-threshold 0.75
```

---

## Step 2 — Tag Eval Results

**Goal:** Join the per-contract eval predictions (from the completed eval) with the
contamination index from Step 1.

**Script:** `ml/scripts/audit/tag_wild_eval_results.py`

Reads:
- `ml/data/smartbugs_wild_eval_state.json` (or the final per-contract JSON in docs/reports/)
- `ml/reports/Run12_smartbugs_wild_contamination_index.json`

Writes:
- `ml/reports/Run12_smartbugs_wild_eval_tagged.json` — each result with `ood_group` field:
  - `"in_train"` — in v3 train split
  - `"in_val_test"` — in v3 val or test split (tuning contamination)
  - `"ood"` — not in v3 at all (honest evaluation)
  - `"eval_error"` — slither failed (excluded from all stats)

---

## Step 3 — Re-analyze Per Group

**Goal:** Produce statistics separately for each group.

**Script:** `ml/scripts/audit/analyze_wild_ood_split.py`

For each group (`in_train`, `in_val_test`, `ood`), report:

| Metric | Why |
|---|---|
| N contracts | Group size |
| Error rate | Compilation failure rate per group |
| Trigger rate (≥1 class) | Does the model fire more on seen contracts? |
| Class distribution (top class) | Does seen data show different distribution? |
| Mean confidence per class | Are seen contracts higher confidence? |
| Mean triggers per contract | Multi-label density |

**Key hypothesis:** If DIVE overlaps heavily with Wild, `in_train` contracts will show
systematically higher confidence than `ood` contracts. This would confirm the evaluation
we've been running is partially inflated.

**Outputs:**
```
ml/reports/Run12_smartbugs_wild_ood_analysis.json    ← machine-readable
ml/reports/Run12_smartbugs_wild_ood_analysis.md      ← human-readable report
```

---

## Step 4 — Write Honest Report

**Goal:** Replace the raw eval stats with properly segmented numbers.

The current incremental reports (`incremental_N20000.json` etc.) show stats over ALL
47K contracts. After this plan is complete, we have:

- **Inflated stat (all contracts):** X% trigger rate, distribution over full 47K
- **Honest OOD stat:** Y% trigger rate, distribution over OOD-only subset

Both numbers belong in the report — the "all contracts" stat is still useful for
"how does the model behave on mainnet contracts in general" but must be labelled
as including seen contracts. The OOD number is the honest generalisation claim.

**Report location:** `docs/reports/2026-06-14_ml_Run12_eval_full_eval_smartbugs_wild_47K_in_progress/`
→ rename folder to `2026-06-15_ml_Run12_eval_full_eval_smartbugs_wild_47K_COMPLETE/`
→ add file `2026-06-15_ml_Run12_smartbugs_wild_47K_honest_ood_report.md`

**Attestation to add** (per `A_benchmark_runs.md` §A.5):
```
## Procedure Attestation — A_benchmark_runs A.1 (Wild) — 2026-06-15
A.1 contamination check:       CLEAN / FLAGGED
  Tiers run:                   1 / 2 / 3
  Jaccard threshold:           0.75
  N Wild contracts checked:    47,398 (minus eval errors)
  In-train:                    N (X%)
  In-val/test:                 N (X%)
  OOD (honest):                N (X%)
  Source of in-train matches:  dive / smartbugs_curated / solidifi (breakdown)
```

---

## Step 5 — Update MEMORY.md and Project Tracking

After Step 4 complete, update:
1. `MEMORY.md` — replace SmartBugs Wild eval entry with final numbers, add OOD-split numbers
2. `2026-06-15_project_smartbugs_wild_full_eval.md` — mark complete, add honest OOD stats
3. `docs/CHANGELOG.md` — add entry for Wild eval + honest OOD separation

---

## Decision Gate

After Step 3, if OOD-only trigger rate is materially different from all-47K rate:
→ all prior Wild stats (from incremental reports, from this session's briefing) must
   be marked "includes seen contracts" in any document that cites them.

If overlap is <5% (v3 has very few Wild contracts), stats are still valid as stated.

---

## Scripts Summary

| Script | Status | Inputs | Outputs |
|---|---|---|---|
| `check_contamination_wild.py` | TODO (write in session) | Wild `.sol` dir, v3 splits | contamination_index.json |
| `tag_wild_eval_results.py` | TODO (write in session) | eval state JSON, contamination_index.json | eval_tagged.json |
| `analyze_wild_ood_split.py` | TODO (write in session) | eval_tagged.json | ood_analysis.json + .md |

All scripts go in `ml/scripts/audit/` (existing audit scripts dir).
Follow the same pattern as `calibrate_temperature_v3.py` (argparse, explicit paths,
no side effects on production files, writes to `ml/reports/`).

---

## Checklist

- [ ] Step 0: Eval finishes (PID 984402)
- [ ] Step 1: `check_contamination_wild.py` written + run
- [ ] Step 2: `tag_wild_eval_results.py` written + run
- [ ] Step 3: `analyze_wild_ood_split.py` written + run
- [ ] Step 4: Honest report written, eval report folder renamed
- [ ] Step 5: MEMORY.md + CHANGELOG.md updated
