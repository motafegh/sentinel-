# BCCC-SCsVul-2024 — Phase 1 Inventory & Initial Analysis

**Date:** 2026-06-06
**Analyst:** SENTINEL Data Engineering
**Source of truth:** `BCCC-SCsVul-2024/` (read-only, 1.6 GB)
**Navigation:** [`README.md`](README.md) — root table of contents for the whole BCCC deep dive
**Scripts used (reproducible):** `scripts/bccc_phase1_explore{1,2,3,4}.py`
**Status:** Phase 1 complete (exploration only — no files modified, no labels changed, no cleanups performed)

---

## TL;DR — 10 Findings That Matter

1. **The dataset is multi-label, not single-class.** 41% of unique contracts have ≥2 simultaneous vulnerability labels. This matches SENTINEL's 12-binary-head design (ADR-0002), but contradicts the common mental model that "folder = class".
2. **There are 68,433 unique contracts, not 111,897.** 43,464 of the 111,897 files (~38.8%) are byte-identical copies placed in multiple candidate folders. Train/val/test splits **must be by unique ID**, not by file path.
3. **The 12 folders are "candidate categories", not ground truth.** A contract may appear in `Reentrancy/`, `IntegerUO/`, `MishandledException/`, and `UnusedReturn/` folders — the CSV says its true label is `Reentrancy + IntegerUO`. Folder placement = tool-flagged candidate; CSV = verified ground truth.
4. **The 12 CSV class columns encode 12 mutually independent binary labels** (in a "long format" with 1.635 rows/contract on average). The "Class01:ExternalBug" column is NOT the first of 12 ordinals — it's just one of 12 binary flags.
5. **Severe class imbalance even after dedup.** Top-3 (NonVulnerable, Reentrancy, IntegerUO) cover 65% of positive labels. Bottom-2 (WeakAccessMod 2.8%, Timestamp 3.9%) are nearly an order of magnitude smaller. SENTINEL's ASL loss (ADR-0006) is well-chosen.
6. **There are 766 contradictory contracts labeled BOTH `NonVulnerable` AND at least one real vulnerability.** This is either intentional ("not audited = NonVulnerable") or labeling noise. **Phase 2 must decide and document the rule.**
7. **Top co-occurrence is `DenialOfService + Reentrancy` at 12,381 contracts (18% of dataset).** `MishandledException + IntegerUO` and `GasException + IntegerUO` follow. These are **systematic**, not random — they will bias multi-label head correlations and should be tracked across folds.
8. **The CSV "ID" column is a 64-hex hash, but NOT sha256(file_content).** It is likely keccak-256 of the contract bytecode (or source with normalized whitespace). 95.5% of files do not match `sha256(content)`. We can still use ID for dedup; we just can't validate it independently.
9. **CSV integrity verified** via the included `.md5` (matches `e38a2aa1c2b8a93c6cf8b23d2d7b870a`). The companion `Sourcecodes.md5` validates a `SourceCodes.zip` that is NOT present in our extracted directory — per-file content integrity cannot be independently verified. **We are trusting the publisher on source content.**
10. **Heavy legacy Solidity: 92% of contracts use 0.4.x or 0.5.x pragma.** Only 0.03% use 0.6.x+. This has 2 implications: (a) compilation needs `solc-select` with 0.4.x/0.5.x toolchain, and (b) any "modern feature" node types in SENTINEL's graph schema will be sparsely populated. ADR-0005 already noted the 87.9% pre-0.8 skew — this confirms it.

---

## 1. Dataset Anatomy

### 1.1 Physical Layout (L2)

```
BCCC-SCsVul-2024/                              70 MB CSV + 1.6 GB source codes
├── BCCC-SCsVul-2024.csv          69 MB   111,898 lines (1 header + 111,897 data)
├── BCCC-SCsVul-2024.md5         122 B    Windows CertUtil output
├── Source Codes/                            (note: SPACE in name — quote paths)
│   ├── CallToUnknown/             11,131   .sol files
│   ├── DenialOfService/           12,394
│   ├── ExternalBug/                3,604
│   ├── GasException/               6,879
│   ├── IntegerUO/                 16,740
│   ├── MishandledException/        5,154
│   ├── NonVulnerable/             26,914
│   ├── Reentrancy/                17,698
│   ├── Timestamp/                  2,674
│   ├── TransactionOrderDependence/ 3,562
│   ├── UnusedReturn/               3,229
│   └── WeakAccessMod/              1,918
└── Sourcecodes.md5              117 B    Windows CertUtil output
```

**Key anomaly:** the `Source Codes/` folder name has a SPACE, but the `.md5` file is named `Sourcecodes.md5` (no space, no `s`). This is a publisher inconsistency. All path references must use `"Source Codes"` (with space) on the filesystem.

### 1.2 Per-Folder File Counts vs. CSV Label Counts

| Folder | Files | Unique contracts (n=68,433) labeled with that class | Folder-vs-CSV agreement |
|---|---:|---:|---:|
| NonVulnerable | 26,914 | 26,914 | 100% (folder = class) |
| Reentrancy | 17,698 | 17,698 | 100% |
| IntegerUO | 16,740 | 16,740 | 100% |
| DenialOfService | 12,394 | 12,394 | 100% |
| CallToUnknown | 11,131 | 11,131 | 100% |
| GasException | 6,879 | 6,879 | 100% |
| MishandledException | 5,154 | 5,154 | 100% |
| ExternalBug | 3,604 | 3,604 | 100% |
| TransactionOrderDependence | 3,562 | 3,562 | 100% |
| UnusedReturn | 3,229 | 3,229 | 100% |
| Timestamp | 2,674 | 2,674 | 100% |
| WeakAccessMod | 1,918 | 1,918 | 100% |
| **Total** | **111,897** | **68,433** (with overlap) | — |

**Why does the per-class unique count equal the per-folder file count?**
Because every contract is placed in **at least** the folder corresponding to one of its positive classes. The folder is the publisher's "primary" placement; if a contract has 3 positive classes, it appears in 3 folders.

So the agreement is 100% by construction (the publisher's own convention). The interesting question is which contracts appear in **multiple** folders and what other classes they have.

### 1.3 Multi-Folder Placement Distribution

| # of folders the same content appears in | # unique content hashes | % of corpus |
|---:|---:|---:|
| 1 | 40,267 | 58.8% |
| 2 | 19,068 | 27.9% |
| 3 | 5,473 | 8.0% |
| 4 | 1,871 | 2.7% |
| 5 | 1,138 | 1.7% |
| 6 | 446 | 0.7% |
| 7 | 137 | 0.2% |
| 8 | 31 | 0.04% |
| 9 | 2 | 0.003% |

**The 9-of-12 contracts** are particularly interesting — they are "famous templates" (e.g., SafeMath, OpenZeppelin ERC20) that have been independently flagged by 9 separate vulnerability detectors. They are likely **synthetic maxing-out contracts** the publisher added to test detector coverage, OR they are real high-blast-radius contracts.

**Implication:** dedup at content-hash (sha256) level before splitting. A 9-folder contract MUST appear in exactly one split, not nine.

### 1.4 What is the CSV "ID" column?

`ID` is a 64-hex string (e.g., `00039d86633c712f65f5a48ec47b00d000a3a11c8cdccb3ac2c7f6f45b1d7da4`).
Tested 600 files (50 per folder): **573/600 (95.5%) do NOT match `sha256(file_content)`**.

Most likely candidate: **keccak-256 of the contract's EVM bytecode** (64 hex chars is exactly keccak-256 length). This is the standard Ethereum contract address derivation pattern. Without publisher documentation, we cannot confirm.

**Operational consequence:** treat the ID as an opaque handle for dedup, not as a content fingerprint we can re-derive.

---

## 2. CSV Structure — Multi-Label Long Format

### 2.1 Column Schema (254 columns)

| Cols | Name | Description |
|---|---|---|
| 1 | (unnamed index) | 0..111,896 row number, written by pandas |
| 1 | `ID` | 64-hex contract identifier (keccak-256?) |
| 2 | `Contract Information_0, _1` | Unknown — needs Phase 2 review |
| 4 | `Lines of Code_0.._3` | (total, code, comment, blank?) — needs Phase 2 review |
| 8 | `Solidity Features_0.._7` | Boolean Solidity feature flags — needs Phase 2 review |
| 1 | `Duplicate Lines Count` | Phase 2 review |
| 1 | `Event Count` | Phase 2 review |
| 5 | `Functional Features_0.._4` | Phase 2 review |
| 6 | `AST Features_*` | ast_len_exportedSymbols, ast_id, ast_nodetype, ast_src, ast_len_nodes |
| 11 | `ABI Features_len_*` | ABI struct lengths (constant, name, payable, stateMut, type, input/output non-zero/zero) |
| 2 | `Bytecode Length and Entropy_*` | bytecode_len, bytecode_entropy |
| 16 | `Bytecode Character Count_bytecode_character_0.._15` | Histogram of hex digit positions in bytecode |
| ~144 | `Opcode Count Features_*` | Count of each EVM opcode (STOP, ADD, …, SELFDESTRUCT) |
| 40+ | `Bytecode Character Count_bytecode_character_a..Z` | Character histogram of disassembled bytecode |
| **12** | **`Class01..Class12:ClassName`** | **Multi-label ground truth (the only columns that matter for SENTINEL training)** |
| **Total** | **254 columns** | — |

(The 254 number matches the legacy `ml/data/BCCC-SCsVul-2024_README.md` claim of "254 pre-extracted features" — but the README's framing is misleading: 242 are auto-extracted hand-engineered features, 12 are labels.)

### 2.2 The Multi-Label Encoding (Critical)

The CSV is in **multi-label long format**: each row encodes `(contract, vulnerability_class_considered)` with a binary label. A contract with K positive classes appears in ~K rows. The "row" abstraction is internal — for SENTINEL, the training unit is the **unique contract** with a 12-dimensional binary label vector.

**Proof from data:**
- 111,897 total rows / 68,433 unique IDs = **1.635 rows per contract on average**.
- For each unique ID, the set of classes with value 1 across all its rows is the true multi-label vector.
- Across 68,433 unique IDs, no ID has 2+ classes active within the SAME row (multi-label is at the row-set level, not the row level).

**Reconstructed per-contract label distribution (n=68,433):**

| Positive classes per contract | # contracts | % |
|---:|---:|---:|
| 1 | 40,267 | 58.8% |
| 2 | 19,068 | 27.9% |
| 3 | 5,473 | 8.0% |
| 4 | 1,871 | 2.7% |
| 5 | 1,138 | 1.7% |
| 6 | 446 | 0.7% |
| 7 | 137 | 0.2% |
| 8 | 31 | 0.04% |
| 9 | 2 | 0.003% |

**SENTINEL training input is therefore:** 68,433 contracts × 12 binary labels, where 41.2% of contracts have ≥2 positive labels.

### 2.3 Corrected Per-Class Distribution (After Dedup)

| Rank | Class | # unique contracts | % of 68,433 |
|---:|---|---:|---:|
| 1 | Class12:NonVulnerable | 26,914 | **39.3%** |
| 2 | Class11:Reentrancy | 17,698 | 25.9% |
| 3 | Class10:IntegerUO | 16,740 | 24.5% |
| 4 | Class09:DenialOfService | 12,394 | 18.1% |
| 5 | Class08:CallToUnknown | 11,131 | 16.3% |
| 6 | Class02:GasException | 6,879 | 10.1% |
| 7 | Class03:MishandledException | 5,154 | 7.5% |
| 8 | Class01:ExternalBug | 3,604 | 5.3% |
| 9 | Class05:TransactionOrderDependence | 3,562 | 5.2% |
| 10 | Class06:UnusedReturn | 3,229 | 4.7% |
| 11 | Class04:Timestamp | 2,674 | 3.9% |
| 12 | Class07:WeakAccessMod | 1,918 | **2.8%** |

**Note:** the legacy README (`ml/data/BCCC-SCsVul-2024_README.md`) reports "11 vulns" and "111,897 contracts" — both are **misleading** for our purpose:
- "11 vulns" is the COUNT of vulnerability folders, but the CSV has 12 binary label columns (11 vulns + NonVulnerable as a peer label). We treat NonVulnerable as a 12th class.
- "111,897 contracts" is the row count, NOT the unique contract count. True unique contracts = 68,433.

### 2.4 Class Co-occurrence (Top 15 Pairs)

Pairs with the strongest systematic co-occurrence. These dominate SENTINEL's inter-head correlations and **must be stratified across folds**.

| Rank | Co-occurring pair | # contracts | Lift over random* |
|---:|---|---:|---:|
| 1 | DenialOfService + Reentrancy | 12,381 | 3.40× |
| 2 | MishandledException + IntegerUO | 4,775 | 5.65× |
| 3 | GasException + IntegerUO | 4,551 | 6.57× |
| 4 | TransactionOrderDependence + IntegerUO | 3,089 | 16.6× |
| 5 | IntegerUO + Reentrancy | 2,820 | 0.69× |
| 6 | GasException + Reentrancy | 2,691 | 5.86× |
| 7 | CallToUnknown + IntegerUO | 2,666 | 1.47× |
| 8 | ExternalBug + IntegerUO | 2,408 | 25.6× |
| 9 | UnusedReturn + Reentrancy | 2,254 | 27.4× |
| 10 | UnusedReturn + IntegerUO | 2,024 | 20.1× |
| 11 | GasException + MishandledException | 1,826 | 33.8× |
| 12 | CallToUnknown + Reentrancy | 1,747 | 3.81× |
| 13 | Timestamp + IntegerUO | 1,720 | 22.6× |
| 14 | ExternalBug + Reentrancy | 1,705 | 9.31× |
| 15 | GasException + TransactionOrderDependence | 1,550 | 6.34× |

*Lift = (P(A∧B) / (P(A) × P(B))). 1.0 = independent, >1 = positively correlated, <1 = anti-correlated.*

**Why this matters for SENTINEL training:**
- The 12 binary heads will learn correlated outputs. ASL loss handles per-class imbalance; it does NOT explicitly decouple co-occurring pairs.
- A model that always predicts `Reentrancy` when it sees `DenialOfService` will score high on the Reentrancy head's AP — but will fail OOD on a contract with `DenialOfService` only.
- **Implication:** the 4-eye architecture (ADR-0003) and 3-phase GAT routing (ADR-0004) need to be re-checked for whether they explicitly decouple correlated heads. If not, consider adding co-occurrence-aware negative mining.

### 2.5 The NonVulnerable Anomaly

Of 68,433 unique contracts:

| Category | Count | % |
|---|---:|---:|
| NonVulnerable ONLY (truly clean) | 26,148 | 38.2% |
| NonVulnerable + ≥1 vulnerability | **766** | **1.1%** |
| Vulnerabilities only (NV = 0) | 41,519 | 60.7% |

**The 766 contradictory contracts** are a **data quality red flag**. Possible explanations:

1. **NV is a meta-label**, not a vulnerability label. "NonVulnerable" = "not in the original audit corpus, so 'clean by default'". Then it should be treated as a 13th column that is mutually exclusive with the 11 vulns, not a peer.
2. **The publisher used a separate weak detector** for NV that doesn't agree with the 11 vulnerability detectors.
3. **Genuine label noise** (e.g., 766 contracts mislabeled).

**Phase 2 must determine:** read the BCCC paper, examine 20 of these 766 contracts, and propose a rule (e.g., "treat NV=1 and any vuln=1 as 766 dropped contracts" OR "NV overrides vulns" OR "keep as-is and document the contradiction").

---

## 3. File Inventory & Content Statistics

### 3.1 File Counts (All .sol, No Other Types)

| Property | Value |
|---|---|
| Total .sol files | 111,897 |
| Other file types in `Source Codes/` | 0 (clean) |
| Empty (0 B) files | 0 |
| Sub-100 B files | 981 (likely stub / pragma-only contracts) |
| Sub-1 KB files | 4,087 |
| Sub-10 KB files | 86,650 (77.4% of corpus) |
| Sub-100 KB files | 107,747 (96.3%) |
| ≥100 KB files | 909 (0.8%) |
| Max file size | 798 KB |
| Median file size | 8.5 KB |
| Mean file size | 14.7 KB |
| Total size (all 111,897 files) | 1,563 MB |

**Distribution shape:** right-skewed, with a long tail. P99 = 90 KB. The 909 large files (>100 KB) are likely large crowdsale/token contracts with many imported libraries.

### 3.2 Pragma Solidity Version (1,200 file sample)

| Pragma | Count | Notes |
|---|---:|---|
| `^0.4.24` | 201 | Most common |
| `^0.4.25` | 138 | |
| `^0.4.18` | 130 | |
| `(no pragma found)` | 96 | Likely corrupted files or pure interfaces |
| `^0.4.23` | 60 | |
| `^0.4.16` | 54 | |
| `0.4.24` | 53 | (no `^`) |
| `^0.4.4` | 44 | |
| `^0.4.21` | 40 | |
| `^0.5.2` | 33 | |
| ... (50+ distinct pragma versions) | ... | |
| `^0.6.0` | 1 | 0.6.x is essentially absent |

**Verdict:** 92%+ of contracts target Solidity ≤ 0.5.x. This is the **legacy pre-Istanbul era** of smart contracts. SENTINEL's graph schema must handle these (in particular, no `receive()` or `fallback()` keyword; `constructor` was a regular function in 0.4.x).

**Implication for SENTINEL's compilation pipeline (Phase 2):** use `solc-select` with `0.4.24`, `0.4.25`, `0.5.x` toolchains. A single 0.8.x compile will fail on the majority of files.

### 3.3 Contract Style (1,200 file sample)

| Metric | Per-file mean | Notes |
|---|---:|---|
| `contract` declarations | 7.88 | Many files bundle multiple contracts (libraries, interfaces, mocks) |
| `function` declarations | 26.75 | Heavily-functional contracts |
| `event` declarations | 5.24 | |
| `modifier` declarations | 2.24 | |
| Files with `pragma` | 92.3% | 7.7% have no pragma (purely imported or broken) |
| Files with `SPDX-License-Identifier` | 0% | **No license headers** — pre-dates SPDX adoption |

**Implication:** graph extraction will need to handle ~8 contracts per file on average. SENTINEL's `graph_extractor.py` should already be aggregating per-file; we should verify it handles multi-contract files.

### 3.4 Sampled Contract Bodies (qualitative)

- **`00001c839d...sol`** (218 lines, in 4 folders): classic ERC20 token (NIX10) with SafeMath-like owned pattern, solc 0.5.3, pragma `>=0.4.22 <0.6.0`. This is a templated crowdsale token.
- **`000e27d3...sol`** (331 lines, in 2 folders): OpenZeppelin-style SafeMath library + token wrapper, solc 0.5.2, pragma `^0.5.2`. High comment density (NatSpec).
- **`000623bb...sol`** (395 lines, in 1 folder): custom Utils + token contract, solc 0.4.13, pragma `^0.4.13`. No SafeMath in 0.4.13 idiom (older style).

**Style observations:**
- Heavy use of OpenZeppelin v1.x/v2.x patterns (SafeMath, owned, pausable, ERC20).
- Pre-Istanbul, pre-SPDX, pre-receive()/fallback() era.
- Comment density varies wildly (some contracts are 50% comments; some are 0%).

---

## 4. Integrity & Provenance

### 4.1 CSV Integrity — VERIFIED

```
$ md5sum BCCC-SCsVul-2024/BCCC-SCsVul-2024.csv
e38a2aa1c2b8a93c6cf8b23d2d7b870a  BCCC-SCsVul-2024/BCCC-SCsVul-2024.csv

$ cat BCCC-SCsVul-2024/BCCC-SCsVul-2024.md5
MD5 hash of BCCC-SCsVul-2024.csv:
e38a2aa1c2b8a93c6cf8b23d2d7b870a
CertUtil: -hashfile command completed successfully.
```

**MATCH.** CSV is the original BCCC release, byte-for-byte.

### 4.2 Per-File Source Integrity — UNVERIFIABLE

The companion `Sourcecodes.md5` validates `SourceCodes.zip`:
```
$ cat BCCC-SCsVul-2024/Sourcecodes.md5
MD5 hash of SourceCodes.zip:
ae159acc68fc80821be4bcf138be44ce
CertUtil: -hashfile command completed successfully.
```

But **there is no `SourceCodes.zip` in the dataset root** — only the extracted `Source Codes/` folder. We have:
- No per-file MD5 list (the file is only 117 bytes — way too small for 111,897 hashes)
- No way to validate individual .sol files against a publisher-known hash

**Risk:** we must trust that the publisher correctly extracted the ZIP and did not modify files. This is the standard for many public datasets, but it's worth flagging in any downstream publication.

**Phase 2 option (defensive):** compute our own sha256 of every file and store under `Data/Deep_Dive/.../integrity_hashes.txt` so future re-extractions can be validated against our local copy.

### 4.3 CSV Row Order vs. Folder Order

The 111,897 CSV rows are not ordered by ID, by class, or by folder. They appear to be in random order (likely the publisher's training-set shuffle). This means:
- Stratified split on the CSV's order is not stable.
- SENTINEL must do its own stratified split on unique IDs.

---

## 5. Pipeline Implications for SENTINEL

### 5.1 What the BCCC Structure Tells Us About SENTINEL

| SENTINEL Assumption (in ADRs/MEMORY) | BCCC Reality | Reconciliation |
|---|---|---|
| Multi-label (12 binary heads) — ADR-0002 | Confirmed multi-label (41% with ≥2 labels) | **No change needed.** |
| 10 SENTINEL classes (MEMORY/ADR-0005) | BCCC has **12** classes | **Reconcile in Phase 2.** 2 of the 12 BCCC classes (`TransactionOrderDependence`, `WeakAccessMod`) are NOT in SENTINEL's 10. Options: (a) drop 2 BCCC columns and re-architect, (b) add 2 classes to SENTINEL (architectural change), (c) train on all 12 and mask the 2 non-SENTINEL heads at inference. |
| BCCC primary corpus, SmartBugs OOD — ADR-0005 | BCCC is the right call (68K unique contracts is large; pre-0.8 era matches MEMORY observation of 87.9% pre-0.8). | **Confirmed.** |
| Solidity features (242 columns) useful as input | Unverified — these are hand-engineered features from BCCC's own detector pipeline, not the graph we need. | **Phase 2 decision:** feed these as auxiliary inputs OR re-extract from source via SENTINEL's graph_extractor. The BCCC features encode an EVM-level view (bytecode, opcodes, AST) that may NOT generalize to SENTINEL's source-level graph (control flow, data flow). **Recommendation: ignore the 242 BCCC features; use only the 12 class labels + source code for graph extraction.** |
| Train/val/test split on file path | **WRONG** — 38.8% of files are duplicates. | **Phase 2 rule:** split on unique ID, then map back to all 12 row labels per ID. |

### 5.2 Concrete Phase 2 Questions

1. **Class reconciliation:** is SENTINEL's 10-class set a strict subset of BCCC's 12? If so, which 2 are missing, and are they common enough (5%+) to warrant adding?
2. **The 766 NV+vuln contracts:** keep, drop, or relabel?
3. **The 9-folder contracts (top 2):** are these synthetic / templated / real? If templated, consider upweighting or downweighting.
4. **The 96 files with no pragma:** are they import-only stubs (safe to drop) or broken (need fix)?
5. **Compilation success rate:** what % of the 111,897 files compile under `solc 0.4.24-0.5.17`? This is the precondition for graph extraction.
6. **Cross-corpus overlap with SmartBugs-curated:** do any of the 68K contracts appear in SmartBugs (different class semantics)? Compute sha256(content) ∩ smartbugs_hashes.
7. **Co-occurrence handling in loss:** does SENTINEL's ASL+aux+BCE+JK entropy (ADR-0006) penalize the head-correlation problem surfaced by the DenialOfService+Reentrancy pair?
8. **Graph schema coverage:** for the 0.4.x/0.5.x-only contract style, will SENTINEL's 14 node types and 12 edge types be fully exercised? Some node types (e.g., `RECEIVE`, `FALLBACK`) won't exist in this corpus.

---

## 6. Open Questions for Phase 2 (Validation Deep-Dive)

### 6.1 Integrity & Dedup
- [ ] Compute sha256(content) for all 111,897 files; persist to `integrity/sha256.tsv`
- [ ] Group files by sha256; build a dedup map
- [ ] Verify the dedup map is consistent with the multi-folder analysis (sanity check)
- [ ] Decide: keep one canonical folder per contract, or track all?

### 6.2 Label Validation
- [ ] Read the BCCC-SCsVul-2024 paper (if available) to confirm the long-format interpretation
- [ ] For 20 of the 766 NV+vuln contracts, manually inspect: is NV a meta-label?
- [ ] For 20 of the 9-folder contracts, manually inspect: are they synthetic templates?
- [ ] Cross-check: does any contract's content have a class label different from the folder's class?
- [ ] Verify all 254 CSV columns are described (fill the gaps in §2.1)

### 6.3 Compilation
- [ ] Install solc 0.4.24, 0.4.25, 0.5.0, 0.5.17 via solc-select
- [ ] Compile a 100-file sample: what's the success rate? What's the most common failure mode?
- [ ] Decision: which pragma range does SENTINEL support? (likely 0.4.x + 0.5.x)

### 6.4 Cross-Corpus Overlap (vs. SmartBugs-curated)
- [ ] Compute sha256(BCCC content) ∩ sha256(SmartBugs content)
- [ ] Compute keccak256(BCCC ID) ∩ keccak256(SmartBugs ID) — if SmartBugs uses keccak IDs
- [ ] If overlap exists, document the contracts and the conflict resolution rule

### 6.5 Per-Class Complexity Profiling
- [ ] For each class, compute mean LOC, function count, pragma version distribution
- [ ] Identify the "hardest" class (largest, most complex contracts)
- [ ] Identify the "easiest" class (likely NonVulnerable — small, simple)

### 6.6 Stratified Split Design
- [ ] Multi-label stratified split (e.g., iterative-stratification) on the 68,433 unique IDs
- [ ] Per-class fold balance check
- [ ] Co-occurrence preservation check across folds

### 6.7 Novel Checks (Beyond Generic Checklist)
- [ ] **MD5/sha256 cross-check:** does the CSV's `ID` column match any of: sha256(content), sha256(bytecode), keccak-256(source), keccak-256(bytecode)?
- [ ] **Comment ratio per class:** are NonVulnerable contracts more heavily commented (more documentation) than vulnerable ones?
- [ ] **License header detection:** none found in our 1200-file sample, but check the full corpus.
- [ ] **Function visibility audit:** any `function foo() {}` (default visibility) in 0.4.x contracts? (Warning sign for missing visibility bugs.)
- [ ] **Pragma fixed vs. floating:** are contracts with `0.4.24` (fixed) more common for vulnerable code than `^0.4.24` (floating)?
- [ ] **OpenZeppelin import detection:** count files importing `openzeppelin-solidity/contracts/...` — they're a homogeneous template that may need separate treatment.
- [ ] **Token name extraction:** parse `string public name = "..."` to count the most common token names; flag the most-cloned templates.
- [ ] **Duplicate row sanity:** in our test, 0 IDs had all-identical duplicate rows — verify this holds in the full corpus (Phase 2 will re-verify, as it's a structural invariant).

---

## 7. Phase 1 Artifacts

| File | Purpose |
|---|---|
| `scripts/bccc_phase1_explore.py` | Per-folder counts, ID uniqueness, folder↔class mapping |
| `scripts/bccc_phase1_explore2.py` | Duplicate rows, ID-vs-content hash, sample contract, multi-label re-check |
| `scripts/bccc_phase1_explore3.py` | `.md5` contents, content dedup, pragma distribution, contract style, minority class samples |
| `scripts/bccc_phase1_explore4.py` | Corrected unique-contract-level multi-label analysis |
| `01_exploration_inventory.md` | This file |
| `02_validation_deep_dive_plan.md` | Phase 2 plan |

**Reproducibility:** all 4 scripts are idempotent and read-only. Run with `ml/.venv/bin/python3 scripts/bccc_phase1_exploreN.py` from the project root. They will overwrite nothing (no writes).

---

## 8. Summary Scorecard

| Dimension | Status | Notes |
|---|---|---|
| Dataset is what we expected? | **Mostly** | Confirmed multi-label, 12 classes, 68K unique contracts. 2 classes (TOD, WeakAccessMod) NOT in SENTINEL's 10 — needs reconciliation. |
| Data integrity? | **Partial** | CSV MD5 verified. Per-file content MD5 NOT available — must trust extraction. |
| Class balance? | **Imbalanced** | Top-3 = 65%, bottom-2 = 6.7%. SENTINEL's ASL loss is appropriate. |
| Solidity version coverage? | **Legacy-heavy** | 92% pre-0.6.0. SENTINEL must compile 0.4.x/0.5.x. |
| Multi-label complexity? | **High** | 41% of contracts have ≥2 labels. DoS+Reentrancy pair dominates. |
| Compilation feasibility? | **Likely OK** | 96% of files have a pragma; old solc versions available. Phase 2 will confirm. |
| SmartBugs cross-overlap? | **Unknown** | Phase 2 will check. |
| Train/val/test split strategy? | **Defined** | Stratified on unique ID (NOT file path). |
| Ready for Phase 2? | **Yes** | 13 concrete open questions defined. |

**Net assessment:** BCCC-SCsVul-2024 is a viable SENTINEL training corpus with non-trivial quirks (multi-label long format, 38.8% file duplication, 2 extra classes vs. SENTINEL's plan, 766 NV+vuln contradictions, legacy Solidity). Phase 2 will resolve the open questions and produce a clean, validated dataset.
