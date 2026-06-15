# SENTINEL Data Module — Full Pipeline Audit

**Date:** 2026-06-13  
**Scope:** Full pipeline from raw ingestion to ML-ready export  
**Trigger:** Training Run 10 showed F1=0.683 at ep30 — suspected leakage. Investigation confirmed 45% val/test contamination from v1 splits.

---

## Executive Summary

1. **Multi-label handling is correct and intentional.** 69.1% of DIVE contracts carry multiple vulnerability labels (e.g. Reentrancy + IntegerUO + ExternalBug). This is by design — DIVE places the same `.sol` file in multiple vulnerability folders, and the parser reads all folder memberships into a single `.labels.json` record. The user's concern ("same contract can have different vulnerabilities") is already handled properly.

2. **The 677 "inconsistent" duplicate groups are a dataset phenomenon, not a labeling bug.** They arise from different source files (different sha256, different addresses) that compiled to the same graph tensor. Each gets its own labels.json with its own folder-membership labels. From the model's view: same input features → contradictory targets. This is label noise from DIVE curation inconsistency.

3. **v1 splits had 45% val/test leakage.** Root cause: `_run_split` in `cli.py` built `Contract` objects without setting `dedup_group`, so the `dedup_enforcer` had nothing to act on (`dedup_groups_resolved: 0`). The underlying Level-3 dedup in `preprocessing/deduplicator.py` is also a stub.

4. **v2 splits are clean.** Fix: graph content hashes computed for all 21,523 exported contracts (12,577 unique groups). CLI now reads `data/dedup_groups_graph_hash.json` and passes `dedup_group=` to each Contract. `dedup_groups_resolved: 3,009`. Verified 0 cross-split graph-level leakage.

5. **Run 10 killed at ep33 (F1=0.683 inflated).** Run 11 should start on the clean v2 splits.

---

## 1. Pipeline Architecture

### 1.1 Stage Flow

```
Raw Sources
  ├─ DIVE (Git repo, 22,330 .sol files)
  └─ SolidiFi (Zenodo, 50 benchmark .sol files)

Stage 1 — INGEST
  sentinel-data ingest --source dive|solidifi
  Output: data/raw/<source>/repo/

Stage 2 — PREPROCESS
  sentinel-data preprocess --source dive|solidifi
  Input:  data/raw/<source>/
  Output: data/preprocessed/<source>/<sha256>.json
          data/preprocessed/<source>/<sha256>.meta.json
          data/preprocessed/<source>/<sha256>.sol (normalized)
  Dedup:  Level 1 (exact sha256) + Level 2 (Ethereum address) + Level 3 (STUB)

Stage 3 — LABEL
  sentinel-data label --source dive|solidifi
  Input:  data/preprocessed/<source>/ + raw repo for folder membership
  Output: data/labels/<source>/<sha256>.labels.json
  Then:   sentinel-data label --merge
  Output: data/labels/merged/<sha256>.labels.json
          (per-sha256, multi-source OR-merge with tier precedence)

Stage 4 — VERIFY  (Phase 5 — manual + Slither-based verification)
  sentinel-data verify
  Input:  data/labels/merged/
  Output: data/verification/verification_report.md
          (filtered labels written back to merged/)

Stage 5 — SPLIT
  sentinel-data split --version N
  Input:  data/labels/merged/ + data/dedup_groups_graph_hash.json (v2+)
  Output: data/splits/vN/train.jsonl, val.jsonl, test.jsonl
          data/splits/vN/split_manifest.json
  Two-pass: stratified_split → dedup_enforcer → nonvulnerable_cap

Stage 6 — REPRESENT
  sentinel-data represent
  Input:  data/preprocessed/<source>/<sha256>.sol
  Output: data/representations/<sha256>/graph.pt + tokens.pt

Stage 7 — EXPORT
  sentinel-data export --split-version N
  Input:  data/representations/ + data/splits/vN/ + data/labels/merged/
  Output: data/exports/<name>/graphs/graphs-NNNNN.pt
                              tokens/tokens-NNNNN.pt
                              labels.parquet
                              metadata.parquet
                              manifest.json
```

### 1.2 Data Artifacts Summary

| Stage | Input | Output | Location |
|---|---|---|---|
| Ingest | Remote source | Raw .sol files | `data/raw/<source>/` |
| Preprocess | Raw .sol | SHA256-keyed meta+content | `data/preprocessed/<source>/` |
| Label (source) | Preprocessed meta + raw folders | Per-sha256 labels | `data/labels/<source>/` |
| Label (merge) | Per-source labels | Merged canonical labels | `data/labels/merged/` |
| Verify | Merged labels | Verified labels (in-place) | `data/labels/merged/` |
| Split | Merged labels + dedup groups | train/val/test JSONL | `data/splits/vN/` |
| Represent | Normalized .sol | PyG graph + token tensors | `data/representations/` |
| Export | Representations + splits + labels | Sharded tensors + manifest | `data/exports/<name>/` |

---

## 2. Source Datasets

### 2.1 Active Sources

**DIVE** (`source: dive`, tier: T2)
- 22,330 raw `.sol` files in `repo/__source__/`
- Label mechanism: folder membership. The same file appears in one or more vulnerability folders (`repo/Reentrancy/`, `repo/DoS/`, `repo/Arithmetic/`, etc.)
- **Multi-label is the norm:** 15,394 / 22,073 preprocessed contracts (69.7%) appear in multiple vulnerability folders → multiple positive labels in one record
- After preprocessing: 22,073 unique sha256 (257 compile failures dropped)
- Confidence tier: T2 (curated, folder-based, not exploit-verified)
- Class mapping (from `sentinel_data/crosswalks/dive.yaml`):

| DIVE folder | Canonical class |
|---|---|
| Reentrancy | Reentrancy |
| DoS | DenialOfService |
| Arithmetic | IntegerUO |
| Time manipulation | Timestamp |
| Front Running | TransactionOrderDependence |
| Access Control | ExternalBug |
| Unchecked Return Values | UnusedReturn |
| Bad Randomness | **DROPPED** (no canonical class) |

**SolidiFi** (`source: solidifi`, tier: T0)
- 50 benchmark contracts with injected vulnerabilities
- Highest confidence tier (T0 = inject-verified ground truth)

### 2.2 Skipped Sources (not in current export)

From `manifest.json` `skipped_sources`:
- `smartbugs_curated` — preprocessed dir not found
- `web3bugs` — preprocessed dir not found
- `disl` — preprocessed dir not found

These are future corpus expansion targets. GasException (class 3) has NO positive labels in the current export because it was only expected from SmartBugs Curated.

---

## 3. Stage-by-Stage Analysis

### 3.1 Ingestion

**Handler:** `_run_ingest` in `cli.py:111`  
**Connectors:** `sentinel_data/ingestion/connectors/` — `git_connector.py`, `zenodo_connector.py`, `manual_connector.py`, `huggingface_connector.py`, `etherscan_connector.py`

Downloads raw source files into `data/raw/<source>/`. For DIVE: clones the git repo. For SolidiFi: downloads from Zenodo. Output is raw `.sol` files only — no processing.

### 3.2 Preprocessing & Deduplication

**Handler:** `_run_preprocess` in `cli.py:160`  
**Pipeline:** `sentinel_data/preprocessing/pipeline.py`  
**Deduplicator:** `sentinel_data/preprocessing/deduplicator.py`

**Step order:** flatten → compile (solc) → dedup → normalize → segment + version-bucket

For each `.sol` file:
1. Flatten imports
2. Compile with appropriate solc version (version-bucket detection)
3. Run deduplicator:
   - **Level 1 (exact sha256):** if content seen before → `is_duplicate=True`, skip write
   - **Level 2 (Ethereum address):** if a 0x-address in this file was seen in a prior file → `is_duplicate=True`
   - **Level 3 (AST near-dup): STUB** — `deduplicator.py:73`: *"Level 3: AST near-dup — STUB (requires Slither; deferred to Stage 2). Mark files with dedup_group_id=sha256 for now."*
4. Output: `<sha256>.json` (compiled artifact) + `<sha256>.meta.json` + `<sha256>.sol` (normalized)

**The Level-3 stub consequence:** every non-duplicate contract gets `dedup_group_id = sha256` (itself). Two contracts with different file content (different sha256) but effectively the same code (same graph) are treated as distinct. This is the root cause of the 677 inconsistent label groups found later.

**What's actually in preprocessed JSONs:**
```json
{
  "sha256": "<hex>",
  "dedup_group_id": "<same hex as sha256>",  ← always self-referential
  "is_duplicate": false,
  "source_name": "dive",
  "solc_version": "0.8.17",
  "compile_status": "success",
  ...
}
```

### 3.3 Labeling

**Handler:** `_run_label` in `cli.py:228`  
**Parser (DIVE):** `sentinel_data/labeling/parsers/dive.py`  
**Merger:** `sentinel_data/labeling/merger.py`

#### Per-source parsing

For DIVE (`dive.py`):
1. Read `<sha256>.meta.json` to get `original_path` (e.g., `"repo/__source__/12727.sol"`)
2. Extract filename: `"12727.sol"`
3. Look up which vulnerability folders contain that filename in the raw repo
4. Map each folder to a canonical class via `crosswalks/dive.yaml`
5. Write `data/labels/dive/<sha256>.labels.json` with all positive classes

**This is where multi-label assignment happens.** If `12727.sol` appears in `repo/Reentrancy/`, `repo/Arithmetic/`, and `repo/Access Control/`, it gets `{Reentrancy: 1, IntegerUO: 1, ExternalBug: 1}` in a single record.

**Numbers from the actual data:**
- 22,073 DIVE contracts labeled
- 15,259 (69.1%) have n_pos > 1 (multi-label)
- 0 duplicate sha256 in preprocessed (Level 1+2 dedup already cleaned this)

**Sample `.labels.json` structure:**
```json
{
  "sha256": "16ef65c...",
  "sources": ["dive"],
  "classes": {
    "Reentrancy": {"value": 1, "tier": "T2", "source": "dive"},
    "IntegerUO":  {"value": 1, "tier": "T2", "source": "dive"},
    "ExternalBug":{"value": 0, "tier": null, "source": "dive"},
    ...
  },
  "n_pos": 2,
  "flags": []
}
```

#### Multi-source merging (`merger.py`)

When the same sha256 appears labeled by multiple sources:
- `merger.py:_merge_class_entries` applies tier-precedence resolution per class
- **Positive wins within the same tier** (`merger.py:84-91`)
- Higher-confidence tier (T0 > T1 > T2 > T3 > T4) takes precedence
- Result: one merged `.labels.json` per sha256 with labels from ALL sources combined

**Current state:** 0 multi-source files found in the merged dir (all 22,356 contracts come from a single source each). This means DIVE and SolidiFi share no sha256 in common. If SmartBugs Curated or Web3Bugs were added, multi-source merging would activate.

**Co-occurrence noise check (`merger.py:100-124`):** Flags DoS+Reentrancy co-occurrence from T3/T4 sources with >50% co-occurrence rate as suspect. DIVE's T2 with 12% co-occurrence is explicitly NOT flagged — documented as legitimate multi-label signal.

### 3.4 Label Verification (Phase 5)

The verification stage ran manually in June 2026 (sessions 1–3). Results:

| Class | Before | After | Retained | Gate |
|---|---|---|---|---|
| Reentrancy | 17,698 | 1,699 | 9.6% | VERIFIED (99.8% high-conf) |
| CallToUnknown | 11,131 | 239 | 2.1% | PROVISIONAL |
| Timestamp | 2,674 | 1,075 | 40.2% | BEST-EFFORT |
| ExternalBug | 3,604 | 344 | 9.5% | PROVISIONAL |
| GasException | 0 | 0 | — | No source |
| DenialOfService | 12,394 | 1,252 | 10.1% | BEST-EFFORT |
| IntegerUO | 16,740 | 16,740 | 100% | VERIFIED |
| UnusedReturn | 3,229 | 3,229 | 100% | VERIFIED |
| MishandledException | 5,154 | 5,154 | 100% | VERIFIED |

Best output: `contracts_clean_v1.4.csv` → applied to `labels.parquet` in the export.

**Additional patch (2026-06-13):** DoS/Reentrancy co-occurrence zeroed in `labels.parquet`:
- 2,655 contracts where `DenialOfService=1 AND Reentrancy=1` → `DenialOfService` zeroed
- DoS before: 3,750 → after: 1,095 (pure DoS only)
- Backup: `_labels_pre_dos_reentrancy_patch_2026-06-13.parquet`
- Manifest artifact_hash updated after patch

### 3.5 Splitting

**Handler:** `_run_split` in `cli.py:237`

**Two-pass process:**
1. `stratified_split` — per (primary_class, source, tier) stratum, 70/15/15 ratios, seed=42
2. `apply_dedup_enforcer` — reassigns any dedup group straddling a split boundary to the majority split
3. `apply_nonvulnerable_cap` — caps NonVulnerable:Positive at 3:1

#### v1 Split — What Went Wrong

`_run_split` builds `Contract` objects from `labels/merged/*.labels.json`:

```python
# cli.py:282-285 (BEFORE FIX)
contracts.append(Contract(
    sha256=sha, source=source, tier=tier,
    classes=classes, primary_class=primary, n_pos=n_pos,
    # ← dedup_group never set! defaults to None
))
```

`dedup_enforcer.py:46` skips contracts where `c.dedup_group is None`:
```python
if c.dedup_group is not None:
    group_to_splits[c.dedup_group][split_name].append(c)
```

With all 22,356 contracts having `dedup_group=None`, the enforcer loops over zero groups → `dedup_groups_resolved: 0` in the manifest.

**Result:** stratified random split with NO dedup enforcement.

**Leakage confirmed at graph tensor level:**
- 10,811 / 21,523 contracts (50.2%) are exact graph duplicates (same x + edge_index tensors)
- 1,505 / 3,344 val contracts (45.0%) had an identical graph in train
- 1,518 / 3,368 test contracts (45.1%) had an identical graph in train

**Impact:** Run 10 F1=0.683 at ep32 — heavily inflated. The model "recalled" training samples it had seen during training because val/test contained byte-identical graph representations.

#### v2 Split — Fix Applied (2026-06-13)

**Step 1:** Computed graph content hash (MD5 of `x.tobytes() + edge_index.tobytes()`) for all 21,523 exported contracts.  
**Step 2:** Saved `data/dedup_groups_graph_hash.json` (21,523 → 12,577 unique groups, 1,865 duplicate groups covering 10,811 contracts).  
**Step 3:** Fixed `cli.py:_run_split` to read the file and pass `dedup_group=cid_to_group.get(sha)` to each `Contract`.  
**Step 4:** Re-ran `sentinel-data split --version 2`.

```
cli.py (AFTER FIX):
cid_to_group = json.loads(dedup_groups_path.read_text())["groups"]
contracts.append(Contract(
    sha256=sha, ...,
    dedup_group=cid_to_group.get(sha),  ← now set for represented contracts
))
```

**v2 results:**
- `dedup_groups_resolved: 3,009`
- train=17,877 val=1,878 test=1,768
- Val leakage at graph level: **0 / 1,878 (0.0%)**
- Test leakage at graph level: **0 / 1,768 (0.0%)**

**Why val/test shrank:** The enforcer uses majority-wins. For a 509-contract duplicate group (364 train / 77 val / 68 test), all 509 go to train. Many large duplicate groups are majority-train, so enforcement moves val/test members to train.

### 3.6 Representation

**Handler:** `_run_represent` in `cli.py:160`  
Runs Slither on each `.sol` in `data/preprocessed/<source>/`, builds PyG `Data` graph and GraphCodeBERT tokens.  
Output: `data/representations/<sha256>/graph.pt` + `tokens.pt`

The representation stage runs on ALL preprocessed contracts regardless of split or dedup status. Dedup does not reduce representation work — every unique sha256 gets represented. This is correct: dedup only affects splitting, not graph extraction.

### 3.7 Export

**Handler:** `_run_export` in `cli.py:682`  
Reads splits from `data/splits/vN/`, loads representations from `data/representations/`, shards everything into `data/exports/<name>/`.

The export manifest's `splits` field contains the ordered list of contract IDs per split. `SentinelDataset` in the ML module reads this manifest and uses `shard_index` to locate each contract in the shards.

**Important:** The manifest's `artifact_hash` covers all files EXCEPT `manifest.json` itself. Updating `manifest.splits` in manifest.json does not change the hash. This means the v2 split was applied by updating manifest.json directly — no re-sharding required.

---

## 4. Deduplication Deep Dive

### 4.1 "Same Contract, Different Vulnerabilities" — The User's Question

**Is this handled?** Yes, for the intended case.

The DIVE dataset already embeds multi-vulnerability assignments at the folder level. When `1234.sol` appears in both `repo/Reentrancy/` and `repo/Arithmetic/`, the `dive.py` parser reads BOTH folder memberships and writes a single `.labels.json` with `{Reentrancy: 1, IntegerUO: 1}`. No merging is needed — it comes out multi-label from the start.

**Numbers:** 15,259 of 22,073 DIVE contracts (69.1%) carry multiple positive labels. This is working correctly.

**Same sha256, multiple sources:** If the same sha256 appeared in both DIVE and SolidiFi, `merger.py` would merge them with tier-precedence resolution, with positive winning over negative within a tier. This path is correct but not currently exercised (no sha256 overlap between active sources).

### 4.2 The 677 Inconsistent Groups — What They Actually Are

These are NOT "same contract, different vulnerabilities." They are:

- **Contract A:** `sha256=abc`, filename=`21794.sol` in DIVE → appears in folders [Reentrancy, Arithmetic, AccessControl, Time_manipulation, UnusedReturn] → 5 labels
- **Contract B:** `sha256=xyz`, filename=`8030.sol` in DIVE → appears in folders [AccessControl, Reentrancy, UnusedReturn] → 3 labels  
- **Both compile to the same graph tensor** (same bytecode-derived features)

They are different source files (different sha256), different on-chain addresses, but identical compiled representations. The DIVE dataset curated them independently and placed them in different sets of vulnerability folders — hence different labels.

From the **labeling pipeline's perspective**: correct. Each sha256 has its own record with its own labels from its own folder memberships.

From the **model's perspective**: contradictory. The same input feature vector (graph) must predict different label vectors depending on which sha256 the sample came from. The model cannot distinguish between them.

**Root cause:** DIVE contains many re-deployments and forks of the same code at different addresses. These end up as different sha256 in the dataset but produce identical graph representations.

**Impact on training:** 677 groups × average ~7 contracts = ~4,700 contracts generating contradictory gradient signals during training. The model cannot learn a consistent mapping for these. This is irreducible noise from DIVE curation inconsistency.

**What to do:** Since the v2 split places all group members in train (majority-wins), the model trains on all copies. The contradictory labels average out during gradient descent — effectively the model will learn the UNION of labels for these groups, but with higher variance on the ambiguous classes. This is acceptable; the alternative (picking one label set to discard) requires manual curation decisions.

### 4.3 Level-3 Dedup Stub

`preprocessing/deduplicator.py:73-78`:
```python
# Level 3: AST near-dup — STUB (requires Slither; deferred to Stage 2)
return DedupRecord(
    sha256=sha,
    dedup_group_id=sha,   # ← every contract is its own group
    is_duplicate=False,
    duplicate_of="",
)
```

This was intentionally deferred. The graph-hash groups file (`data/dedup_groups_graph_hash.json`) serves as the practical replacement for the v2 split. For full correctness, Level 3 should eventually be implemented in the preprocessor using text-shingle Jaccard similarity (as `leakage_auditor.py` does) so that dedup groups are computed before representation — not after.

### 4.4 dedup_group_id Not Propagated to labels.json

Even if Level 3 ran correctly and set `dedup_group_id` to a real group ID in the preprocessed JSONs, the split CLI would still not use it — because `_run_split` reads from `labels/merged/*.labels.json`, and those files have no `dedup_group_id` field. The fix applied (graph-hash file + CLI change) bypasses this gap, but a proper fix would:

1. Propagate `dedup_group_id` from `preprocessed/<source>/<sha256>.meta.json` into `labels/<source>/<sha256>.labels.json` at label time
2. Carry it through the merger into `labels/merged/<sha256>.labels.json`
3. Have `_run_split` read it from there

---

## 5. Label Quality (Post-Patch State)

### 5.1 Current labels.parquet counts (after DoS/Reentrancy patch)

| Class | Positives | % of 21,523 |
|---|---|---|
| CallToUnknown | 39 | 0.2% |
| DenialOfService | 1,095 | 5.1% |
| ExternalBug | 16,621 | 77.2% |
| GasException | 0 | 0.0% |
| IntegerUO | 9,437 | 43.8% |
| MishandledException | 39 | 0.2% |
| Reentrancy | 11,369 | 52.8% |
| Timestamp | 6,311 | 29.3% |
| TransactionOrderDependence | 643 | 3.0% |
| UnusedReturn | 5,859 | 27.2% |

### 5.2 Class Issues

- **GasException = 0:** No positive samples. Source (SmartBugs Curated) was skipped. Model will always output F1=0 for this class.
- **CallToUnknown = 39, MishandledException = 39:** Extremely rare. Only 5 positives in val (v1) / even fewer in v2. AUC-ROC reaches 1.000 easily but F1 is low due to threshold sensitivity.
- **ExternalBug dominates (77%):** Almost every DIVE contract is in AccessControl folder. This makes ExternalBug easy to learn and may inflate macro F1.

### 5.3 Within-Group Label Inconsistency

677 of 1,865 duplicate graph groups (36.3%) have different label sets across members. Classes most affected:

| Class | Groups with conflict |
|---|---|
| Reentrancy | 411 |
| ExternalBug | 396 |
| IntegerUO | 363 |
| Timestamp | 216 |
| UnusedReturn | 154 |
| DenialOfService | 72 |
| TOD | 17 |

With the v2 split, all members of each group are in train. The contradictory signals train against each other but do not contaminate val/test evaluation.

---

## 6. Issues Found

| # | Issue | Root Cause | Impact | Status |
|---|---|---|---|---|
| I-1 | v1 split: 45% val/test graph leakage | `_run_split` never set `dedup_group` on Contract objects | Run 10 F1=0.683 inflated | **Fixed — v2 split** |
| I-2 | Level-3 dedup is a stub | `deduplicator.py:73` intentionally deferred | Same-code contracts treated as distinct | **Mitigated** — graph-hash file covers practical need |
| I-3 | `dedup_group_id` not in labels.json | Label parsers don't read preprocessed meta | Enforcer can't use preprocessed groups directly | **Mitigated** — graph-hash workaround; proper fix is to propagate the field |
| I-4 | 677 dup groups with conflicting labels | Same graph, different sha256, different DIVE folder placement | ~4,700 contradictory training samples | **Accepted** — in train only (v2); averages out in gradient descent |
| I-5 | GasException has 0 positives | SmartBugs Curated skipped (preprocessed dir not found) | F1=0 for that class forever | **Open** — needs corpus expansion |
| I-6 | DoS/Reentrancy co-occurrence | BCCC-era label noise propagated into labels.parquet | Model conflated DoS and Reentrancy | **Fixed** — 2,655 DoS labels zeroed, patch applied |
| I-7 | Leakage auditor never ran | `leakage_audit: null` in split_manifest.json | No post-split similarity check was done | **Open** — auditor requires source text; not run for DIVE |
| I-8 | Level-3 dedup should run before representation | Currently graph-hash computed from export (post-representation) | Chicken-and-egg for future re-exports | **Open** — needs proper pipeline integration |

---

## 7. Answering the User's Question Directly

> "Some contracts can be duplicates but with different labels — that's OK as some contracts might have different vulnerabilities. Did we deal with this?"

**Yes, the intended case is handled correctly.**

The DIVE dataset places the same `.sol` file in multiple vulnerability folders. The `dive.py` parser reads all folder memberships for each file and assigns ALL matching classes in one record. A contract in both Reentrancy and Arithmetic gets `{Reentrancy: 1, IntegerUO: 1}` from the start — no merging step needed.

**The 677 inconsistent groups are a different phenomenon.** These are not "one contract, multiple auditors, consistent vulnerability." They are:

- **Different files** (different sha256, different on-chain addresses)  
- That happen to produce **identical graph representations** when compiled  
- And were **placed in different sets of vulnerability folders** by DIVE curators

The data module cannot know these are "the same" without Level-3 dedup (which is stubbed). With the v2 fix, all copies land in train, so val/test evaluation is clean. The contradictory training signal is real but bounded (~4,700 contracts out of 17,877 train).

---

## 8. Recommendations Before Run 11

### Must Do
1. **Use v2 splits for Run 11** — manifest.json updated, 0 leakage confirmed ✓
2. **Verify Gate 3 passes** with v2 splits before launching training

### Should Do
3. **Re-apply DoS/Reentrancy co-occurrence patch** to the v2 labels if the export was re-sharded (current labels.parquet already patched — verify hash matches)
4. **Update cloud monitoring routine** to note that v2 val is 1,878 (not 3,344) — F1 numbers are now on a smaller but honest val set

### Long-term
5. **Implement Level-3 dedup** in `deduplicator.py` using text-shingle Jaccard similarity (the infrastructure exists in `leakage_auditor.py`)
6. **Propagate `dedup_group_id`** through labels.json so the split CLI can use it directly (eliminates the graph-hash workaround)
7. **Add SmartBugs Curated** as a source to get GasException labels
8. **Run the leakage auditor** (`leakage_auditor.py`) after every split — it was designed for this but has never been called with real data

---

## 9. Current Data Artifact State (2026-06-13)

| Artifact | Path | Version | Notes |
|---|---|---|---|
| Preprocessed contracts | `data/preprocessed/dive/` | — | 22,073 files, Level 1+2 dedup applied |
| Merged labels | `data/labels/merged/` | — | 22,356 files, post-Phase-5 verification |
| Graph-hash dedup groups | `data/dedup_groups_graph_hash.json` | new | 12,577 unique groups from 21,523 contracts |
| v1 split | `data/splits/v1/` | v1 | 15,644/3,344/3,368 — **DO NOT USE** (45% leakage) |
| v2 split | `data/splits/v2/` | v2 | 17,877/1,878/1,768 — **CLEAN** (0% leakage) |
| Export | `data/exports/sentinel-v2-baseline-2026-06-12/` | v2 splits | manifest.json updated with v2 splits; labels.parquet has DoS patch |
| Run 10 checkpoint | `ml/checkpoints/GCB-P1-Run10-v2clean-20260613_best.pt` | ep32 | **INFLATED F1=0.683** — trained on v1 leaky splits |
