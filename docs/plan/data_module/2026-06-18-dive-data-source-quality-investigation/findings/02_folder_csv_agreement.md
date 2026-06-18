# Method 2 — Folder↔CSV Per-Contract Identity Check

**Status:** COMPLETE
**Started:** 2026-06-18
**Completed:** 2026-06-18
**Criteria version:** N/A (structural check — no manual TP/FP judgement)

**Provenance:**
- Command: `python3 scripts/verify_folder_csv_agreement.py`
- RNG seed: N/A (full corpus, 22,330 contracts × 8 classes)
- Tool: Python 3.12, stdlib only
- Input CSV: `data_module/data/raw_staging/dive_labels/DIVE_Labels.csv` (raw frame, 22,330 rows)
- Input folders: `data_module/data/raw_staging/dive/{8 class folders}` (raw frame, 54,919 symlinks)
- Script: `scripts/verify_folder_csv_agreement.py`
- Timestamp: 2026-06-18

---

## Result

### Per-class agreement (22,330 contracts × 8 classes = 178,640 pairs)

| Class | Match n / total | Status |
|---|---|---|
| Access Control | 22,330 / 22,330 | ✅ PERFECT |
| Arithmetic | 22,330 / 22,330 | ✅ PERFECT |
| Bad Randomness | 22,330 / 22,330 | ✅ PERFECT |
| DoS | 22,330 / 22,330 | ✅ PERFECT |
| Front Running | 22,330 / 22,330 | ✅ PERFECT |
| Reentrancy | 22,330 / 22,330 | ✅ PERFECT |
| Time manipulation | 22,330 / 22,330 | ✅ PERFECT |
| Unchecked Return Values | 22,330 / 22,330 | ✅ PERFECT |

**Total mismatches: 0 / 178,640.** Per-contract identity is perfect for all 8 classes.

### Per-folder symlink counts (match §2.1 exactly)

| Folder | Symlink count | Matches §2.1 |
|---|---|---|
| Access Control | 16,723 | ✓ |
| Arithmetic | 9,542 | ✓ |
| Bad Randomness | 634 | ✓ |
| DoS | 3,781 | ✓ |
| Front Running | 606 | ✓ |
| Reentrancy | 11,400 | ✓ |
| Time manipulation | 6,322 | ✓ |
| Unchecked Return Values | 5,911 | ✓ |

### Zero-label contracts

2,686 contracts have CSV all-zero and appear in no vulnerability folder — confirmed as NonVulnerable (source files exist in `__source__/`, no folder symlinks). Expected given multi-label distribution in §2.2 (2,686 zero-label out of 22,330).

---

## Self-verification

Hand-checked 3 contracts end-to-end:

| cid | CSV positives | Folder symlinks found | Match |
|---|---|---|---|
| 1 | {DoS} | {DoS} | ✓ |
| 100 | {AC, Arithmetic, Reentrancy, Time manip, Unchecked Return} | {AC, Arithmetic, Reentrancy, Time manip, Unchecked Return} | ✓ |
| 1000 | {AC, Arithmetic, Reentrancy} | {AC, Arithmetic, Reentrancy} | ✓ |

Also verified: the 2,686 "only-CSV" contracts are all zero-label (10 randomly sampled — all have empty CSV, all source files exist in `__source__/`, none have folder entries).

Script output reconciliation:
- CSV: 22,330 rows ✓
- Folder totals sum: 16,723 + 9,542 + 634 + 3,781 + 606 + 11,400 + 6,322 + 5,911 = 54,919 ✓
- All 22,330 cids have source files in `__source__/` (confirmed no gaps)
- Per-class agreement: 22,330/22,330 for all 8 classes ✓

---

## What this means for the label-source decision

**The folder symlinks are a 100% faithful reproduction of the CSV at the per-contract level.** `label_folderize.py` correctly built the symlink structure from the CSV. The parser's folder_index, which is built from these symlinks, therefore has a perfect signal to work from.

Combined with Method 8 (parser faithful for all 7 DIVE-sourced classes), the full chain of trust is now confirmed:

```
CSV (100% verified, §2.2) 
  → folder symlinks (100% per-contract identity, M2) 
  → parser crosswalk (100% faithful for 6 classes; DoS post-parser patch = intentional, M8) 
  → .labels.json (100% faithful to CSV + 1 documented policy deviation)
```

**The raw-CSV analysis in Methods 3–6 is fully valid.** Every CSV label survives through the symlink→parser→labels pipeline intact (with the documented single exception of DoS+RE co-occurrence suppression). There are no structural gaps between "what the CSV says" and "what the model trained on."

### What was NOT checked

- Whether symlink targets (`__source__/<N>.sol`) are valid/readable (they are — the 22,073 successful compilations prove this; the 257 compile failures are the dropped set, already accounted for in M8)
- Whether `label_folderize.py` has edge-case bugs (irrelevant — we validated the output, not the code)
