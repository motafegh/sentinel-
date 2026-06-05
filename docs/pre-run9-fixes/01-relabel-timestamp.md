# Fix #1 — Run --relabel-timestamp on v10 CSV

**Effort:** 10 minutes (no re-extract needed)
**Impact:** Timestamp (drop ~46% noise)
**Risk:** Very low — already production-tested in archive/
**Order:** Do this FIRST before anything else.

---

## Problem (Finding A + B from audit)

The v10 CSV `ml/data/processed/multilabel_index.csv` (timestamp: Jun 2 23:08) is the raw output of
`build_multilabel_index.py`. It was NEVER cleaned by the relabeling script that lives in archive/.

**Audit sample:** of 200 Timestamp=1 graphs, only 27.5% fire `uses_block_globals` (feat[2]) anywhere
in the graph. Baseline (Timestamp=0): 8% fire. So 72.5% of Timestamp=1 contracts have NO
block.timestamp/number/difficulty/basefee signal in the graph — they are demonstrably false positives.

**Root cause:** BCCC-SCsVul-2024 places the same SHA256 contract in multiple category folders (a
contract labeled as Timestamp may not actually read block.timestamp — BCCC's label is noisy).

---

## Source Code References

### The cleaner script (NEVER invoked on v10)

`ml/scripts/archive/dedup_multilabel_index.py:226` — `--relabel-timestamp` argument:
```python
p.add_argument("--relabel-timestamp", action="store_true",
    help="After dedup, verify Timestamp=1 labels against source "
         "patterns and graph features; remove labels neither confirms")
```

`ml/scripts/archive/dedup_multilabel_index.py:174-228` — `relabel_timestamp()` function:
- Checks `data.x[:, 2]` (uses_block_globals feature) for any node > 0.5
- Greps source `.sol` for patterns: `block.timestamp`, `block.number`, `now`, `block.difficulty`,
  `block.prevrandao`, `blockhash(`
- If NEITHER confirms → set Timestamp=0
- If EITHER confirms → keep Timestamp=1

### The label-loading path (where Timestamp is used)

`ml/src/datasets/dual_path_dataset.py` — loads `multilabel_index.csv` and exposes `labels` tensor
of shape `[N, 10]` where index 7 is Timestamp.

`ml/src/training/losses.py` — AsymmetricLoss reads these labels directly (no extra validation).

### FEATURE_SCHEMA_VERSION interplay

`ml/src/preprocessing/graph_schema.py:63` — `FEATURE_SCHEMA_VERSION = "v8"`. The cleaner script
embeds this version in each output row. No version bump needed for this fix.

---

## Execution Steps

```bash
cd /home/motafeq/projects/sentinel
source ml/.venv/bin/activate
export PYTHONPATH=.
export TRANSFORMERS_OFFLINE=1

# 1. Dry-run first to see the impact
python ml/scripts/archive/dedup_multilabel_index.py \
  --multilabel-csv ml/data/processed/multilabel_index.csv \
  --relabel-timestamp \
  --dry-run

# Expected output:
#   Removed (neither source nor graph confirms): 953
#   Confirmed by both source + graph:           525
#   Confirmed by graph only:                      0
#   Confirmed by source only:                   423
#   Timestamp=1 remaining: 948 / 1901  (49.9% kept)

# 2. If the numbers look right, run for real
python ml/scripts/archive/dedup_multilabel_index.py \
  --multilabel-csv ml/data/processed/multilabel_index.csv \
  --relabel-timestamp

# 3. Outputs:
#   ml/data/processed/multilabel_index_deduped.csv  ← use this for Run 9
#   ml/data/splits/deduped/                         ← new stratified splits
```

---

## Bug to Fix Before Running

The archived script has `PROJECT_ROOT = parents[2]` (wrong for the archive/ subdirectory).
It should be `parents[3]` to resolve to the actual project root.

**Location:** `ml/scripts/archive/dedup_multilabel_index.py:50`

```python
# WRONG (current):
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# CORRECT:
PROJECT_ROOT = Path(__file__).resolve().parents[3]
```

Without this fix, the script reports "0/41,576 rows mapped to content hash" and does nothing.
**Already applied on this branch (June 5, 2026) — verify with grep before running.**

---

## Validation After Fix

1. Confirm new file exists: `ml/data/processed/multilabel_index_deduped.csv`
2. Confirm Timestamp count dropped from 1,901 to ~948:
   ```bash
   python -c "
   import pandas as pd
   df = pd.read_csv('ml/data/processed/multilabel_index_deduped.csv')
   print(f'Timestamp=1 count: {int(df[\"Timestamp\"].sum())}')  # Expect ~948
   print(f'Row count: {len(df)}')                                # Expect 41,576
   "
   ```
3. Confirm new splits exist: `ml/data/splits/deduped/{train,val,test}_indices.npy`
4. Spot-check 5 removed contracts: open their source `.sol` files in `BCCC-SCsVul-2024/SourceCodes/`
   and confirm they have NO block.timestamp/now references.

---

## What This Does NOT Fix

- **Audit Finding E:** `now` keyword (Solidity 0.4.x alias) may not be caught by the source-grep
  patterns. The patterns include `\bnow\b` so it SHOULD be caught — but if Slither's source
  extraction is lossy, some may slip through. This is covered by Fix #2.
- **Audit Finding D:** External CALL_ENTRY missing. Not a Timestamp issue — covered by Fix #3.
- **Audit Finding F:** IntegerUO unlearnable. Separate schema gap — covered by Fix #4.

---

## Risk Assessment

**LOW.** The cleaner is conservative — it only removes labels where BOTH source grep AND graph
features fail. Confirmed labels (525 by both, 423 by source alone) are preserved. False negative
risk: a Timestamp contract that uses block.timestamp only inside a string literal would be
incorrectly removed (but these are vanishingly rare in vulnerability contracts).

**Side effect:** also deduplicates by content hash (54 records deduplicated from 41,576 → 41,576
in current state because the v10 CSV was already path-deduplicated during build). No regression.
