# B — Data Pipeline

> Always load `00_rules.md` before following this procedure.
> Apply Rule 2 (gate assertions + completion attestation) at every step.

---

## When This File Applies

- Rebuilding the label CSV from scratch after a graph re-extraction
- Adding new contracts to the training corpus
- Generating new splits (new run, new schema version)
- When a label quality question arises (Timestamp FP rate, WeakAccessMod absence)
- When investigating per-class positive counts that differ from expected

Always load alongside: `E_preprocessing_consistency.md` (E.6 re-extraction
trigger conditions) and `F_new_run_checklist.md` (F.1.1 label file gate).

---

## B.1 — Two Hash Systems (Never Mix)

Read `ml/scripts/build_multilabel_index.py` docstring before touching any
label or graph file. Two parallel hash systems exist:

| Hash type | What it hashes | Where it appears |
|---|---|---|
| **BCCC SHA256** | Raw `.sol` file content bytes | BCCC `SourceCodes/` filename stems; `BCCC-SCsVul-2024.csv` second column |
| **Internal MD5** | File path (not content) | `.pt` filename stems in `ml/data/graphs/`; `md5_stem` column in label CSV |

**Bridge:** each `.pt` graph stores `graph.contract_path` pointing back to
the originating BCCC `.sol` file. `Path(contract_path).stem` gives the
SHA256, which is then looked up in the BCCC CSV.

Never substitute one hash for the other. A SHA256 lookup in the MD5 index
or vice versa silently returns no match and produces an all-zeros label row.

---

## B.2 — Label CSV Construction (`build_multilabel_index.py`)

### B.2.1 — What the Script Does

1. Reads `BCCC-SCsVul-2024/BCCC-SCsVul-2024.csv` (111,897 rows — one per
   folder-file occurrence, NOT per unique SHA256)
2. Groups by SHA256, takes `max()` per class column — max of 0/1 == OR,
   producing a multi-hot label per unique contract
3. For each `.pt` file in `ml/data/graphs/`, extracts `graph.contract_path`,
   derives the SHA256 stem, and looks it up in the grouped table
4. Writes `ml/data/processed/multilabel_index.csv`

Output columns: `md5_stem`, then 10 class columns in this exact order:

```
CallToUnknown (0), DenialOfService (1), ExternalBug (2), GasException (3),
IntegerUO (4), MishandledException (5), Reentrancy (6), Timestamp (7),
TransactionOrderDependence (8), UnusedReturn (9)
```

This order is the training vector index 0–9. Any change to this order
requires rebuilding the CSV **and** retraining from scratch.

### B.2.2 — Excluded Classes

Two BCCC classes are intentionally excluded (read the `CLASS_NAMES` comment
block in the script before questioning this):

- **Class12:NonVulnerable** — not a vulnerability type; absence of all 10
  classes already encodes "safe"
- **Class07:WeakAccessMod** — 1,918 `.sol` files in BCCC but **zero** were
  extracted into `.pt` graph files during the original extraction pass.
  Adding a class with zero training examples produces undefined gradients
  and a permanently near-zero output node. Do not add it back until
  WeakAccessMod `.pt` files exist; if added back, append at index 9 (not
  inserted earlier) so existing trained indices 0–8 stay valid.

### B.2.3 — Invoking the Script

```bash
cd ~/projects/sentinel
PYTHONPATH=. python ml/scripts/build_multilabel_index.py
```

The script prints a per-class positive count table on completion — read
this before proceeding. The `pos_weight` values shown are diagnostic;
the actual training `pos_weight` is computed in `trainer.py` and may be
capped by `--pos-weight-min-samples` (read default from `train.py` args).

### B.2.4 — BUG-6 `contract_path` Correction

After re-extracting graphs with the `most_derived` contract selection
heuristic (BUG-6 fix, 2026-05-17), always re-run `build_multilabel_index.py`.
The old `most functions` heuristic picked the wrong contract 47.4% of the
time, meaning the SHA256 in `graph.contract_path` previously pointed to a
different contract than the one actually extracted. Re-running the script
propagates the corrected `contract_path` values into the label CSV.

### B.2.5 — Timestamp Label Garbage Rate

After rebuilding the label index, run `dedup_multilabel_index.py` with
`--relabel-timestamp` to remove false Timestamp labels. The script comment
notes 48.2% of `Timestamp=1` contracts had no block-global usage in source
or features. This pass is mandatory before generating splits for any run
that includes the Timestamp class.

---

## B.3 — Split Generation (`create_splits.py`)

### B.3.1 — How Splits Are Created

Read `ml/scripts/create_splits.py` before generating splits. Key facts:

- Split ratio: **70% train / 15% val / 15% test** (stratified by binary label)
- Binary label for stratification: `sum(class_cols) > 0` (any vulnerability)
- `label_index.csv` is intentionally ignored — `ast_extractor.py` hardcodes
  `graph.y = 0` for all contracts, making it unusable for stratification
- Random seed default: **42** — change the seed only for ablations, and
  document the change; different seeds produce non-comparable val sets
- Split files are saved as `.npy` index arrays: `train_indices.npy`,
  `val_indices.npy`, `test_indices.npy`

### B.3.2 — Zero-Overlap Verification

`create_splits.py` asserts zero overlap on completion:
```python
assert len(train_set & val_set)  == 0
assert len(train_set & test_set) == 0
assert len(val_set   & test_set) == 0
assert len(train) + len(val) + len(test) == len(df)
```
If any assertion fires, the CSV and graph directory are inconsistent.
Stop and diagnose before using the splits.

### B.3.3 — Freeze Mode (Adding New Contracts)

When adding augmented data to an existing corpus **without** changing the
val/test sets (to preserve comparability with previous experiments),
use `--freeze-val-test`:

```bash
PYTHONPATH=. python ml/scripts/create_splits.py \
    --splits-dir <splits-dir from MEMORY.md> \
    --freeze-val-test
```

Freeze mode behaviour (from `create_splits.py`):
- `val_indices.npy` and `test_indices.npy` are **not touched** — they stay
  on disk unchanged
- All indices not in val/test go to train (original train + any new rows)
- Requires existing `val_indices.npy` and `test_indices.npy` in the target
  directory; fails with `FileNotFoundError` if they are missing

Do not use freeze mode after a schema version change — the index positions
change when graphs are re-extracted, making old `.npy` index files invalid.

### B.3.4 — Invoking the Script

```bash
# Standard full split
PYTHONPATH=. python ml/scripts/create_splits.py \
    --splits-dir <splits-dir from MEMORY.md> \
    --multilabel-csv ml/data/processed/multilabel_index.csv

# Freeze val/test (augmentation run)
PYTHONPATH=. python ml/scripts/create_splits.py \
    --splits-dir <splits-dir from MEMORY.md> \
    --freeze-val-test

# Different seed (ablation — document the reason)
PYTHONPATH=. python ml/scripts/create_splits.py \
    --splits-dir <splits-dir from MEMORY.md>_seed<N> \
    --seed <N>
```

Read `MEMORY.md` Current State for the active splits directory path before
filling in `<splits-dir from MEMORY.md>`.

---

## B.4 — Full Pipeline Rebuild Order

When a full rebuild is required (new graph schema, BUG-6-style extraction
correction, new BCCC version), run steps in this exact order:

1. `reextract_graphs.py` — rebuild all `.pt` graph files (see E.6)
2. `build_multilabel_index.py` — rebuild label CSV from corrected `contract_path`
3. `dedup_multilabel_index.py --relabel-timestamp` — fix Timestamp garbage labels
4. `create_splits.py` — regenerate splits from the new CSV
5. `retokenize_windowed.py` — rebuild token `.pt` files if schema or tokenizer changed (see E.5)
6. Delete or invalidate the DataLoader RAM cache (`cached_dataset_v10.pkl`
   or equivalent) — it references stale graph/token file timestamps
7. Run smoke suite (D.1) before any training run

**Destructive warning:** steps 1, 2, 4, and 5 overwrite existing files.
Archive or DVC-snapshot before running if prior versions must be preserved.

---

## B.5 — Label Quality Checks

After rebuilding the label CSV, verify by reading the count table printed
by `build_multilabel_index.py` on completion:

- **Reentrancy positive count** — read the current expected count from
  `MEMORY.md` (recorded after the most recent verified build). If the
  rebuilt count differs from the MEMORY.md value by more than 5%, the
  graph extraction or CSV grouping changed — investigate before proceeding
- **WeakAccessMod** must not appear as a column — if it does, the exclusion
  in `build_multilabel_index.py` was overridden
- **All-zero rows** (safe contracts) should be present; if `safe_rows = 0`
  in the build output, the SHA256 lookup is failing for all contracts
- **Unknown count** (`SHA256 not in BCCC`) should be < 0.5% of total rows;
  a high unknown count indicates a mismatch between the graphs dir and the
  BCCC CSV version

---

## B.6 — Completion Attestation

After completing this section, append to the relevant run or data-rebuild doc:

```
## Procedure Attestation — B_data_pipeline — <ISO date>
Steps completed:
  B.1 hash system verified:                        PASS/FAIL
  B.2 label CSV rebuilt:                           YES/NO
    B.2.4 BUG-6 contract_path fix applied:         YES/NO/N/A
    B.2.5 --relabel-timestamp pass run:            YES/NO
  B.3 splits generated:                            YES/NO
    B.3.2 zero-overlap assertion passed:           PASS/FAIL
    B.3.3 freeze mode used:                        YES/NO
  B.4 full pipeline rebuild order followed:        YES/NO/PARTIAL
  B.5 label quality checks:                        PASS/FAIL
    Reentrancy positive count (from script):       N
    Expected count (from MEMORY.md):               N
    Delta within 5%:                               YES/NO
    Unknown count:                                 N (expected < 0.5%)
    WeakAccessMod absent:                          YES/NO
Steps skipped:     [any skipped + explicit reason]
Unverified items:  [anything not confirmable]
New findings:      [link to audit doc entry, or "none"]
Written to:        [path of this attestation]
```
