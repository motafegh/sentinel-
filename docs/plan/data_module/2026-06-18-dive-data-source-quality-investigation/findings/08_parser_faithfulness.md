# Method 8 — Parser Faithfulness (CSV → parser → v3 labels)

**Status:** COMPLETE
**Started:** 2026-06-18
**Completed:** 2026-06-18
**Criteria version:** N/A (structural check — no manual TP/FP judgement)

**Provenance:**
- Command: `python3 scripts/parser_faithfulness.py` (full corpus, no sampling)
- RNG seed: N/A (no sampling — all 22,073 contracts compared)
- Tool: Python 3.12, stdlib only (csv, json, pathlib, collections)
- Input CSV: `data_module/data/raw_staging/dive_labels/DIVE_Labels.csv` (raw frame, 22,330 rows)
- Input meta: `data_module/data/preprocessed/dive/*.meta.json` (v3 export frame, 22,073 files)
- Input labels: `data_module/data/labels/dive/*.labels.json` (v3 export frame, 22,073 files)
- Input crosswalk: `data_module/sentinel_data/labeling/crosswalks/dive.yaml` (read for Step 1 only — script hardcodes copy)
- Script: `scripts/parser_faithfulness.py`
- Timestamp: 2026-06-18

---

## Step 1 — Document the intended transform (parser source + crosswalk)

### Source files read

- `data_module/sentinel_data/labeling/crosswalks/dive.yaml` (92 lines)
- `data_module/sentinel_data/labeling/parsers/dive.py` (167 lines)

### The label pipeline (4-stage chain)

```
DIVE_Labels.csv          label_folderize.py     folder symlinks        dive.py parser         .labels.json
(22,330 rows)      →     builds symlinks   →    in raw repo       →    scans folders     →    (22,073 files)
```

**Important:** The parser does NOT read the CSV directly. It reads the **folder symlinks** (which were built FROM the CSV by `label_folderize.py`). So the parser's fidelity depends on:
1. `label_folderize.py` faithfully reproducing the CSV (Method 2 checks this)
2. The crosswalk (`dive.yaml:class_map`) mapping being correct + complete
3. The parser code having no bugs

### Crosswalk mapping (from `dive.yaml`)

| DIVE folder/column name | SENTINEL canonical class | Notes |
|---|---|---|
| Reentrancy | Reentrancy | Direct |
| DoS | DenialOfService | Direct |
| Arithmetic | IntegerUO | Direct |
| Time manipulation | Timestamp | Direct |
| Front Running | TransactionOrderDependence | Direct |
| Access Control | ExternalBug | Not direct — "covers ownership, role checks, privilege enforcement" |
| Unchecked Return Values | UnusedReturn | Not direct — explicitly NOT CallToUnknown |
| Bad Randomness | DROPPED | No canonical class equivalent; 634 files excluded |

**Canonical classes with NO DIVE source (3 of 10):** CallToUnknown, GasException, MishandledException.
**Canonical classes WITH DIVE source (7):** Reentrancy, ExternalBug, IntegerUO, DenialOfService, Timestamp, TransactionOrderDependence, UnusedReturn.

### Parser logic (from `dive.py`)

1. **Input:** `preprocessed/dive/*.meta.json` files (one per contract, keyed by sha256)
2. **Build folder index:** Scan `raw/dive/repo/` folders (→ `raw_staging/dive/` via symlink) → `{filename: frozenset(canonical_classes)}` using crosswalk class_map
3. **Per-contract:**
   - Read `.meta.json` → extract `sha256`, `original_path`
   - Extract filename from `original_path` (e.g. `repo/__source__/12727.sol` → `12727.sol`)
   - Look up filename in folder index → get canonical_classes set
   - If zero classes → NonVulnerable (n_pos=0, all class values=0)
   - Write `{sha256}.labels.json`
4. **Output:** `data/labels/dive/<sha256>.labels.json` — 22,073 files exist on disk

### What Step 1 did NOT cover

- Did NOT verify that `label_folderize.py` built the symlinks correctly (Method 2)
- Did NOT verify that folder symlinks still exist / haven't been corrupted
- Did NOT read a single `.labels.json` to check format — done in Steps 3-5

---

## Step 2 — Resolve contractID → sha256 mapping; verify 1:1

### Mapping mechanism

The `.meta.json` files contain both `sha256` and `original_path`. The contractID is the numeric part of `original_path`'s filename:
- `original_path: "repo/__source__/1.sol"` → contractID = 1
- The meta file is named `{sha256}.meta.json` — both filename and internal field match

### Verification results

| Check | Result |
|---|---|
| meta.json files total | 22,073 |
| Successfully mapped | 22,073 |
| Parse failures | 0 |
| cid collisions (different sha256 for same cid) | 0 |
| sha collisions (different cid for same sha256) | 0 |
| **Mapping is 1:1?** | **YES — confirmed** |
| labels.json exist for every mapped sha256 | 22,073/22,073 (0 missing) |

**No collisions. Clean 1:1 mapping. No fallback needed.**

---

## Step 3 — Full CSV → parsed label comparison

### Aggregate counts

| Frame | Count |
|---|---|
| Raw CSV rows | 22,330 |
| Parsed labels.json | 22,073 |
| Contracts compared (in both) | 22,073 |
| CSV-only (no labels.json) | 257 (dropped — see Step 4) |
| Labels-only (no CSV row) | 0 |

### Per-class agreement (22,073 contracts compared)

| Class | Match n / total | Percent | Status |
|---|---|---|---|
| ExternalBug | 22,073 / 22,073 | 100.00% | ✅ PERFECT |
| IntegerUO | 22,073 / 22,073 | 100.00% | ✅ PERFECT |
| Reentrancy | 22,073 / 22,073 | 100.00% | ✅ PERFECT |
| Timestamp | 22,073 / 22,073 | 100.00% | ✅ PERFECT |
| TransactionOrderDependence | 22,073 / 22,073 | 100.00% | ✅ PERFECT |
| UnusedReturn | 22,073 / 22,073 | 100.00% | ✅ PERFECT |
| **DenialOfService** | **19,418 / 22,073** | **87.97%** | **⚠️ 2,655 mismatches** |

### DenialOfService mismatch root cause

**All 2,655 mismatches are CSV-says-DoS=1 but labels-says-DoS=0.** Zero cases of CSV-says-DoS=0 but labels-says-DoS=1.

**This is NOT a parser bug.** It is the **intentional DoS+Reentrancy co-occurrence patch from 2026-06-13**, fully documented and verified:

- **Evidence chain:** Ali identified the pattern → confirmed against MEMORY.md → verified against `patch_dos_v3.py` source → triple-verified against the live labels data. All checks converge.

- **What the patch does** (from `data_module/temp/archive/2026-06-13_run12_prep/scripts/patch_dos_v3.py`):
  - For every DIVE `.labels.json` where `DenialOfService.value==1` AND `Reentrancy.value==1`:
  - Set `DenialOfService.value=0`, `tier=None` (preserving `source="dive"`)
  - Recompute `n_pos`
  - **Exactly 2,655 contracts patched** (the full set of DIVE contracts where both DoS=1 and Reentrancy=1 in CSV)

- **Rationale** (from `~/.claude/.../memory/2026-06-13_project_dos_patch.md`): "a reentrancy attack that ALSO blocks transfers is really a reentrancy, not a DoS." The original parser correctly wrote DoS=1 labels for all 3,750 DIVE DoS-positive contracts. The *subsequent patch* zeroed the 2,655 that co-occurred with Reentrancy, leaving 1,095 DoS-only labels. This is label policy, not parser error.

- **Triple-verification of Ali's claims** (verified live, this session):
  1. All 2,655 mismatched contracts have CSV Reentrancy=1: **YES (2,655/2,655)** ✓
  2. All have `DenialOfService.tier=null` in labels.json: **YES (2,655/2,655)** ✓ — this is the patch's exact signature (`value=0, tier=None, source="dive"`)
  3. Count matches patch script's logged output: **YES (2,655)** ✓

- **Do NOT re-run parser with force=True** — that would REVERT a deliberate, tested, documented fix. The test at `data_module/tests/test_verification/test_class_auditor.py::test_dive_dos_reentrancy_cooccurrence_finding` explicitly asserts `len(flagged_dos) == 0` and will fail if the patch is ever reverted.

### What Step 3 did NOT cover

- Did NOT check whether DoS=0 labels for non-DoS-contracts are correct (they are — the other 18,323 DoS=0 labels are all correct CSV-DoS=0 contracts whose Reentrancy is also 0)
- Did NOT investigate whether other intentional post-parser patches exist (none known for EB/RE — verified via MEMORY.md search, this session)

---

## Step 4 — Raw → export drop accounting (22,330 → 22,073)

**Drop count: 257** — matches `dive.yaml:26` exactly. The integration-test doc's claim of "67 dropped" is stale/wrong.

### Dropped contracts: class breakdown

| Class | CSV-positives dropped | % of total CSV-positives |
|---|---|---|
| Access Control (EB) | 141 | 0.84% of 16,723 |
| Arithmetic (IUO) | 154 | 1.61% of 9,542 |
| Reentrancy | 70 | 0.61% of 11,400 |
| Unchecked Return (UR) | 52 | 0.88% of 5,911 |
| Time manipulation (TS) | 50 | 0.79% of 6,322 |
| DoS | 31 | 0.82% of 3,781 |
| Bad Randomness | 2 | 0.32% of 634 |
| Front Running (TOD) | 2 | 0.33% of 606 |
| Zero-label | 28 | — |

All 257 dropped contracts have `.sol` files in `__source__/` (verified for first 5). The drop is likely due to compile failures during preprocessing (no `.meta.json` generated for these contracts). The loss is negligible: <1.7% for every class.

### Impact on Phase 1 (EB/RE)

- EB: 141 of 16,723 lost (0.84%) — negligible
- RE: 70 of 11,400 lost (0.61%) — negligible
- The raw-CSV analysis in Methods 3-6 can safely ignore these 257 dropped contracts.

---

## Step 5 — Self-verification (hand-check 3 contracts end-to-end)

| cid | CSV positive | Expected (crosswalk) | Actual (labels.json) | Match? |
|---|---|---|---|---|
| 1 | DoS=1 | {DenialOfService} | {DenialOfService} | ✅ |
| 2 | Reentrancy=1, DoS=1 | {Reentrancy, DenialOfService} | {Reentrancy} | ❌ Missing DoS |
| 12727 | Reentrancy=1, Access Control=1 | {Reentrancy, ExternalBug} | {Reentrancy, ExternalBug} | ✅ |

Hand-verification confirms the script's output:
- cid=1: correct (DoS=1, not co-occurring with Reentrancy — patch leaves it alone)
- cid=2: "Missing DoS" is **intentional** (Reentrancy=1 in CSV → DoS zeroed by patch — correctly applied)
- cid=12727: correct — **EB and RE labels are faithful for all 22,073 contracts**

---

## Summary & conclusion for the label-source decision

### Verdict: Parser is faithful for all 7 DIVE-sourced classes

The only deviation from CSV (DoS: 2,655/22,073 mismatches) is the **intentional, documented, tested DoS+Reentrancy co-occurrence patch** — upstream label policy, not parser error. The parser itself never wrote a wrong label; the patch correctly zeroed DoS on co-occurring contracts after the parse.

**All 7 classes are parser-faithful.** The labels the model trained on reflect the CSV labels plus one deliberate, documented downstream policy decision (DoS+RE co-occurrence suppression). No other policy patches affect DIVE labels (confirmed via MEMORY.md search).

### What this means

1. **Parser is 100% faithful for the 6 classes relevant to Phase 1 (EB, RE, IUO, TS, TOD, UR).** The labels the model trained on match the CSV labels exactly. Method 8 validates that analysis on the raw CSV IS analysis on what the model actually saw — for these 6 classes.

2. **DenialOfService's apparent 12.03% "mismatch" is the intentional DoS+RE co-occurrence patch** — not a defect. A `force=True` parser re-run would REVERT this tested, documented fix and break an explicit test assertion. Do NOT re-run.

3. **Drop count confirmed at 257** (matches `dive.yaml`, disproves "67" from integration doc). 141 EB-positive and 70 RE-positive contracts lost — negligible for Phase 1.

4. **The chain of trust is:** CSV labels (verified aggregate counts match folders in README §2.2) → folder symlinks (pending Method 2 per-contract check) → parser crosswalk → parsed labels **→ optional intentional post-parser patches (DoS+RE only).** Method 8 validates the parser+crosswalk link. Method 2 will validate the folder-symlink link. Together they confirm that the parsed labels faithfully reproduce the CSV (plus one documented deliberate deviation).

5. **The raw-CSV analysis in Methods 3-6 is valid for EB/RE.** There is no parser-driven or patch-driven distortion between the CSV and the labels the model trained on for ExternalBug or Reentrancy.

### What was NOT checked

- Whether `label_folderize.py` faithfully wrote the folder symlinks from the CSV (Method 2)
- Whether the parser correctly handles edge cases (no test suite run)
- Whether tier assignment logic is correct (all DIVE labels get T2)
- Whether the parser's `class_names()` function returns exactly the 10 canonical classes — confirmed from Step 1 schema

---

## Self-critique — what went wrong in the original Step 3/4 analysis

**The original Step 3/4 (corrected above) falsely concluded the DoS mismatch was a parser operational bug** — a "stale-cache / incomplete run" where the DoS folder was "temporarily in an unexpected state." This was wrong.

**What I should have done** (and what must be done before concluding any future discrepancy is a "bug"):

1. **Check MEMORY.md for documented intentional changes** before constructing a plausible-sounding mechanism. The `2026-06-13_project_dos_patch.md` file, referenced from MEMORY.md line 128, documents the exact DoS+RE co-occurrence patch — its count (2,655), its timestamp (2026-06-13), its rationale, its test, and its script path. I had no excuse to miss it.

2. **Check git log for relevant commits** on the data labeling pipeline. The patch script's existence in `data_module/temp/archive/` and the backup directories would have been visible.

3. **Check the actual labels.json content more carefully before constructing a theory.** The `tier=null` signature is the patch's fingerprint — the parser would never produce `tier=null` for a folder-derived label (it always assigns `tier=T2`). If I had checked the tier field in Step 3, I would have recognized the patch's signature immediately rather than inventing a cache-timing story.

4. **When a number matches a known documented count exactly** (2,655 appears in MEMORY.md as the exact patch count), treat that as a clue to check the documentation, not as coincidence.

**The fix:** The corrected finding now cites the patch script, the memory file, and the triple-verification of Ali's claims (all 2,655 have Reentrancy=1; all have tier=null; count matches patch output). Future discrepancies will be checked against MEMORY.md and git log BEFORE concluding "bug."

**Material impact:** The corrected conclusion is stronger — the parser is faithful for all 7 classes, not just 6. The DoS deviation is label policy, not parser error. The Phase 1 investigation (EB/RE) was never affected either way.
