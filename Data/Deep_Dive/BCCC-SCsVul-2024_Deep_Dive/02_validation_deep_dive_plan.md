# BCCC-SCsVul-2024 — Phase 2 Validation Deep-Dive Plan

**Date:** 2026-06-06
**Prerequisite:** Phase 1 inventory complete (`01_exploration_inventory.md`)
**Goal:** Produce a validated, deduplicated, SENTINEL-ready dataset from `BCCC-SCsVul-2024/` with documented decisions, integrity manifest, and stratified train/val/test split. **No ML training in this phase** — this is data engineering only.
**Navigation:** [`README.md`](README.md) — root table of contents for the whole BCCC deep dive
**Phase 2 deliverable:** [`Phase2_Validation_2026-06-06/outputs/contracts_clean.csv`](Phase2_Validation_2026-06-06/outputs/contracts_clean.csv) (67,311 × 24, SENTINEL v1.0)

**Working directory:** `Data/Deep_Dive/BCCC-SCsVul-2024_Deep_Dive/`
**Mode:** Read-only on source data; writes only to `Data/Deep_Dive/.../outputs/` and `Data/Deep_Dive/.../scripts/`.

---

## 0. Output Directory Layout

```
Data/Deep_Dive/BCCC-SCsVul-2024_Deep_Dive/
├── 01_exploration_inventory.md      [DONE]
├── 02_validation_deep_dive_plan.md  [this file]
├── README.md                        [TO DO — top-level orientation for any future reader]
├── scripts/                         [Phase 1 + Phase 2 scripts]
│   ├── bccc_phase1_explore{1,2,3,4}.py
│   ├── a_integrity_dedup.py
│   ├── b_label_inspect.py
│   ├── c_compile_probe.py
│   ├── d_cross_corpus.py
│   ├── e_complexity_profile.py
│   ├── f_class_reconciliation.py
│   └── g_stratified_split.py
├── integrity/
│   ├── sha256_all_files.tsv          [id, folder, sha256, size]
│   ├── dedup_map.csv                 [content_sha -> canonical_id, list_of_folders]
│   └── manifest.md                   [what was checked, what passed, what didn't]
├── labels/
│   ├── bccc_paper_summary.md         [notes from paper, if obtained]
│   ├── nv_vuln_inspections.md        [20 manual reviews of the 766 contradictions]
│   ├── multi_folder_inspections.md   [20 manual reviews of the 9-folder contracts]
│   └── class_reconciliation_decision.md  [the 10-vs-12 classes decision]
├── compile/
│   ├── sample_100_results.csv        [id, pragma, solc, success, error]
│   └── compile_report.md             [success rate, failure modes, recommended toolchain]
├── cross_corpus/
│   ├── bccc_vs_smartbugs_overlap.csv [shared contracts with class conflict]
│   └── overlap_report.md
├── complexity/
│   ├── per_class_stats.csv           [class, n, mean_loc, mean_funcs, ...]
│   └── complexity_report.md
├── splits/
│   ├── train_ids.csv                 [unique IDs in train fold]
│   ├── val_ids.csv                   [unique IDs in val fold]
│   ├── test_ids.csv                  [unique IDs in test fold]
│   ├── fold_balance.csv              [per-class prevalence per fold]
│   └── split_manifest.md
├── outputs/
│   ├── contracts_clean.csv           [final per-contract label matrix (68,433 rows × 12 cols + meta)]
│   ├── contracts_clean.parquet       [same, columnar]
│   ├── metadata.json                 [schema version, hash of contracts_clean.csv, class mapping]
│   └── README.md                     [usage instructions for SENTINEL preprocessing]
└── CHANGELOG.md                      [Phase 2 session log]
```

---

## 1. Workstreams

### WS-A: Integrity & Dedup [BLOCKER for everything else]

**Objective:** assign a unique, verifiable identity to every contract in the corpus.

| Step | Action | Output | Est. time |
|---:|---|---|---|
| A1 | Compute sha256 of every `.sol` file in `Source Codes/`. | `integrity/sha256_all_files.tsv` (111,897 rows) | 1.0 h |
| A2 | Group files by sha256; emit `dedup_map.csv` with canonical ID + folder list. | `integrity/dedup_map.csv` (68,433 rows) | 0.5 h |
| A3 | Sanity-check: the dedup map should match Phase 1's "1.635 rows per contract" + the multi-folder distribution. | Inline assertions in script | 0.25 h |
| A4 | Compute sha256 of CSV; cross-check against `BCCC-SCsVul-2024.md5`. | Inline assertion (already verified, but record) | 0.1 h |
| A5 | Write `integrity/manifest.md` describing what's covered, what's not, and the trust assumption for source content. | `integrity/manifest.md` | 0.25 h |

**Decision points:**
- D-A1: should the canonical ID be (a) the existing 64-hex CSV `ID`, or (b) the sha256(content)?
  - **Recommendation: keep both.** Use `ID` for round-trip with the CSV (saves reverse-engineering the hash), and `sha256` for content-dedup. Track both in the manifest.

**Risk register:**
- R-A1: hashing 111K files is I/O-bound and takes ~3-5 min. Run on the WSL filesystem, not via PowerShell.
- R-A2: the `Source Codes/` path has a SPACE; quote everywhere.

**Done criteria:** `integrity/sha256_all_files.tsv` and `integrity/dedup_map.csv` exist, are deterministic, and pass the Phase 1 cross-checks.

---

### WS-B: Label Validation [DEPENDS on WS-A]

**Objective:** resolve the 5 most ambiguous label questions from Phase 1.

| Step | Action | Output | Est. time |
|---:|---|---|---|
| B1 | Acquire the BCCC-SCsVul-2024 paper or technical README. If unavailable, document the gap. | `labels/bccc_paper_summary.md` | 1.0 h (locating) + 2.0 h (reading) |
| B2 | Inspect 20 contracts from the 766 `NonVulnerable + vulnerability` set. For each, record: (a) does the source code contain the labeled vulnerability pattern, (b) is the `NonVulnerable` label plausible, (c) verdict. | `labels/nv_vuln_inspections.md` | 3.0 h |
| B3 | Inspect 20 contracts from the 9-folder set. For each, record: (a) is it a template (e.g., SafeMath, ERC20), (b) is it synthetic, (c) verdict. | `labels/multi_folder_inspections.md` | 2.0 h |
| B4 | Verify all 254 CSV columns are understood. The 12 class columns are confirmed; the 242 hand-engineered feature columns are still partially described. Fill in any gaps from the paper. | Inline docstring in `outputs/contracts_clean.csv` header | 1.0 h |
| B5 | Cross-check: for 100 random contracts, does the `ID` column match sha256(content)? (Should be ~0%, but document the actual rate.) | Inline table in `labels/bccc_paper_summary.md` | 0.5 h |

**Decision points (output of B2/B3):**
- D-B1: handling of 766 NV+vuln contracts.
  - **Recommendation:** read paper first. If NV is a meta-label, drop the 766 from training. If NV is peer-label, keep and document.
- D-B2: handling of 9-folder contracts.
  - **Recommendation:** they are valid (each is a real contract flagged by 9 separate detectors). Keep, but they will naturally upweight in stratified sampling if they're common templates. No special handling needed.

**Risk register:**
- R-B1: the BCCC paper may not be publicly available (this is a recent dataset). Fallback: reverse-engineer from the BCCC GitHub README (already partially captured in `ml/data/BCCC-SCsVul-2024_README.md`).
- R-B2: manual inspection is slow. Keep the sample small (n=20) to stay within session budget; document the sample selection as stratified random.

**Done criteria:** all 4 docs in `labels/` are written. The 766 NV+vuln handling is decided and recorded. The 9-folder contracts are characterized.

---

### WS-C: Compilation Probing [DEPENDS on WS-B for sample selection]

**Objective:** determine the actual solc toolchain and compilation success rate.

| Step | Action | Output | Est. time |
|---:|---|---|---|
| C1 | Install `solc-select` + solc 0.4.24, 0.4.25, 0.5.0, 0.5.17 (most common pragma versions per Phase 1 §3.2). | Shell history record | 0.5 h |
| C2 | Stratified random sample of 100 contracts (10 per class × 10 classes, 10 from NonVulnerable). Stratify on file size (small/medium/large). | `compile/sample_ids.csv` | 0.25 h |
| C3 | For each sample file: detect pragma, pick matching solc, compile with `--bin --abi` and standard-json (for errors). Record: success, error type, byte length. | `compile/sample_100_results.csv` | 2.0 h |
| C4 | Aggregate: success rate, top 10 error messages, byte length distribution. | `compile/compile_report.md` | 0.5 h |

**Decision points:**
- D-C1: which solc version is the canonical one for SENTINEL on this corpus?
  - **Recommendation:** `solc 0.4.25` (covers ~50% of files), with auto-fallback to 0.4.24, 0.5.0, 0.5.17. Do NOT support 0.6+ in this corpus.
- D-C2: how to handle compilation failures?
  - **Recommendation:** drop files that fail to compile. If failure rate is >30%, escalate to SENTINEL team for graph schema review.

**Risk register:**
- R-C1: solc 0.4.x is no longer maintained; may not install cleanly. Fallback: download solc static binaries from `binaries.soliditylang.org`.
- R-C2: some files may need their imports resolved (e.g., OpenZeppelin). Strategy: use `--allow-paths .` and inline imports via solc `remappings`.

**Done criteria:** `compile/sample_100_results.csv` exists with 100 rows; `compile/compile_report.md` has a concrete success rate and toolchain recommendation.

---

### WS-D: Cross-Corpus Overlap (BCCC vs. SmartBugs-curated) [DEPENDS on WS-A]

**Objective:** determine whether any contracts appear in BOTH BCCC and the SmartBugs-curated OOD set. If yes, document the conflict resolution rule.

| Step | Action | Output | Est. time |
|---:|---|---|---|
| D1 | Locate SmartBugs-curated in the SENTINEL project. Expected at `ml/data/smartbugs-curated/` or similar. If not present, find the source. | Path discovery log | 0.5 h |
| D2 | Compute sha256(content) for all SmartBugs `.sol` files. | SmartBugs sha256 list | 0.5 h |
| D3 | Set intersection: BCCC ∩ SmartBugs. | `cross_corpus/bccc_vs_smartbugs_overlap.csv` | 0.5 h |
| D4 | For each overlap, check class label: is the same contract labeled the same vulnerability in both? | Same file, +1 column "labels_agree" | 0.5 h |
| D5 | Write findings: how many overlap, agreement rate, what to do with them. | `cross_corpus/overlap_report.md` | 0.5 h |

**Decision points:**
- D-D1: how to handle overlap contracts?
  - **Recommendation:** drop from the OOD test set (the BCCC label is canonical for training; the SmartBugs label is canonical for OOD; having the same contract in both violates the OOD premise). If agreement is high (>80%), keep them in OOD and re-label to BCCC's classes (which is more granular). If agreement is low, drop.

**Risk register:**
- R-D1: SmartBugs uses a different file naming convention (e.g., `reentrancy/simple_dao.sol`). Need to use content-hash, not filename match.
- R-D2: SmartBugs label semantics differ from BCCC (e.g., SmartBugs has 4 broad classes, BCCC has 12 fine classes). Document the mapping.

**Done criteria:** `cross_corpus/overlap_report.md` is written with a concrete decision.

---

### WS-E: Per-Class Complexity Profile [DEPENDS on WS-A]

**Objective:** characterize each of the 12 classes by structural complexity.

| Step | Action | Output | Est. time |
|---:|---|---|---|
| E1 | For each unique contract, compute: LOC (lines, code, comment, blank), function count, event count, modifier count, contract count, SPDX presence. | Per-contract stats table | 1.5 h |
| E2 | Aggregate by primary class (or by positive-class membership for multi-label). | `complexity/per_class_stats.csv` | 0.5 h |
| E3 | Generate ASCII histogram (or matplotlib if env has it) for LOC per class. | Inline in report | 0.5 h |
| E4 | Write report: which classes are "hard" (large, complex) vs "easy" (small, simple). | `complexity/complexity_report.md` | 0.5 h |

**Decision points:**
- D-E1: which "primary class" for multi-label contracts?
  - **Recommendation:** report stats per (class, contract) pair, so a contract with 3 positive classes contributes 1 row to each of 3 classes. This is the multi-label idiom.

**Risk register:**
- R-E1: 68K contracts × 6 features = 408K computations. Use vectorized pandas. ~30 seconds.

**Done criteria:** `complexity/per_class_stats.csv` exists, 12 classes × 8 metrics; `complexity/complexity_report.md` is written.

---

### WS-F: Class Reconciliation [DEPENDS on WS-B]

**Objective:** resolve the 10-vs-12 classes question raised in Phase 1 §5.1.

| Step | Action | Output | Est. time |
|---:|---|---|---|
| F1 | Re-read ADR-0005 (BCCC dataset choice) and the 10 SENTINEL classes from MEMORY.md. Confirm which 2 BCCC classes (TransactionOrderDependence, WeakAccessMod) are NOT in SENTINEL's set. | Inline reference | 0.25 h |
| F2 | Compute the "loss" if we drop the 2 extra classes: how many contracts lose their only positive class? | Inline computation | 0.25 h |
| F3 | Present 3 options to user: (a) drop 2 BCCC classes, (b) add 2 classes to SENTINEL, (c) train on all 12 and mask 2 at inference. | `labels/class_reconciliation_decision.md` with 3-option matrix | 1.0 h |
| F4 | User picks option; document decision and update ADR-0005 if needed. | Decision recorded | 0.5 h |

**Decision points (user input required):**
- D-F1: which option to take? **Recommendation: (a) drop the 2 BCCC classes** for v1. Rationale: smaller architectural change; the 2 missing classes are uncommon (5.2% and 2.8%); can be added later in a v2. This keeps SENTINEL's class set stable.

**Risk register:**
- R-F1: if we drop 2 BCCC classes, the BCCC-NV+vuln contradictions reduce. (The 766 contracts are NOT in these 2 classes, so this WS doesn't resolve that — that's WS-B.)
- R-F2: adding 2 classes is a 4-eye / 3-phase GAT architecture change. Schedule for v2.

**Done criteria:** `labels/class_reconciliation_decision.md` is written, the user has approved one option, and (if ADR-0005 is updated) the update is committed.

---

### WS-G: Stratified Split Design [DEPENDS on WS-A, WS-F]

**Objective:** produce a deterministic, multi-label-stratified train/val/test split on the 68,433 unique contracts.

| Step | Action | Output | Est. time |
|---:|---|---|---|
| G1 | Choose split ratio: 70/15/15 (train/val/test) is the SENTINEL default; confirm with MEMORY. | Inline | 0.1 h |
| G2 | Use `iterative-stratification` (Python lib) to stratify on the 12 binary labels. If lib unavailable, implement greedy multi-label stratification manually. | `splits/{train,val,test}_ids.csv` | 1.0 h |
| G3 | Sanity-check per-fold balance: each fold should have ±2% of overall class prevalence. | `splits/fold_balance.csv` | 0.5 h |
| G4 | Sanity-check co-occurrence preservation: top-15 pairs should be present in similar proportions across folds. | Inline in `splits/split_manifest.md` | 0.5 h |
| G5 | Persist the split with a hash of the contract list so future re-runs are reproducible. | `splits/split_manifest.md` includes the hash | 0.25 h |

**Decision points:**
- D-G1: which stratification method?
  - **Recommendation:** `iterative-stratification` (https://github.com/trent-b/iterative-stratification) if installable; otherwise a manual greedy algorithm that picks the rarest class first and balances it.
- D-G2: random seed?
  - **Recommendation:** seed=42 (the canonical ML default), and record the seed in the manifest.

**Risk register:**
- R-G1: `iterative-stratification` may not be in `ml/.venv` or root `.venv`. Fallback: manual greedy algorithm (15-30 min to implement).
- R-G2: a single contract can appear in 9 folders; the split is by ID, not by file. The 9 file copies of one contract will all be in the same fold. Verify this in step G4.

**Done criteria:** `splits/{train,val,test}_ids.csv` exist with 47,903 / 10,265 / 10,265 contracts (70/15/15). `splits/fold_balance.csv` shows ±2% balance per class.

---

### WS-H: Final Cleaned Dataset [DEPENDS on WS-A through WS-G]

**Objective:** emit the final, validated, SENTINEL-ready dataset manifest.

| Step | Action | Output | Est. time |
|---:|---|---|---|
| H1 | Build `contracts_clean.csv` from the unique 68,433 contracts: (id, sha256, folder, loc, n_funcs, n_events, n_modifiers, n_contracts, pragma, has_spdx, then 12 class columns). | `outputs/contracts_clean.csv` | 1.0 h |
| H2 | Save parquet version for fast loading. | `outputs/contracts_clean.parquet` | 0.25 h |
| H3 | Build `metadata.json`: schema version, source CSV hash, class mapping (12 → 10 if WS-F = drop), split files, statistics. | `outputs/metadata.json` | 0.5 h |
| H4 | Write `outputs/README.md`: how to use the cleaned dataset in SENTINEL preprocessing. | `outputs/README.md` | 0.5 h |
| H5 | Update SENTINEL's `ml/src/preprocessing/` to consume `contracts_clean.csv` (deferred to a separate task — this is a "wiring" task, not data engineering). | Code change in `ml/src/preprocessing/` | 1.0 h |
| H6 | Commit Phase 2 outputs to git. | Single commit | 0.25 h |

**Decision points:**
- D-H1: should the 12 class columns or 10 be in the output?
  - **Recommendation:** keep all 12 in `contracts_clean.csv` (full information), but in `metadata.json` document which 10 are the "SENTINEL" classes. This is forward-compatible.

**Risk register:**
- R-H1: the cleaned dataset is ~5 MB (csv) or ~1 MB (parquet). Manageable.
- R-H2: if the schema changes, downstream code breaks. Document the schema version in `metadata.json` and pin in SENTINEL's preprocessing.

**Done criteria:** `outputs/contracts_clean.{csv,parquet}`, `outputs/metadata.json`, `outputs/README.md` exist. A git commit records Phase 2 completion.

---

## 2. Effort & Sequencing

| WS | Depends on | Est. (h) | Session fit |
|---|---|---:|---|
| A. Integrity & Dedup | — | 2.0 | 1 session |
| B. Label Validation | A | 8.0 | 2-3 sessions (manual inspection is the bottleneck) |
| C. Compilation Probing | B | 3.25 | 1 session |
| D. Cross-Corpus Overlap | A | 2.5 | 1 session |
| E. Per-Class Complexity | A | 3.0 | 1 session |
| F. Class Reconciliation | B | 2.0 | 1 session (light — needs user decision) |
| G. Stratified Split | A, F | 2.5 | 1 session |
| H. Final Cleaned Dataset | A-G | 3.5 | 1 session |
| **Total** | | **26.75 h** | **8-10 sessions** |

**Suggested ordering across sessions:**
- Session 1: WS-A (integrity/dedup) + start WS-B (paper lookup)
- Session 2: WS-B (manual inspections) + WS-E (complexity profile, parallelizable)
- Session 3: WS-C (compilation probe) + WS-D (cross-corpus) + WS-F (class reconciliation)
- Session 4: WS-G (stratified split) + WS-H (final dataset, commit)

**Parallelization opportunities (within a session):**
- WS-B and WS-E are independent; can be tackled in parallel.
- WS-C and WS-D are independent.
- WS-F is light and can be folded into any session where the user is available.

---

## 3. Risks & Mitigations

| ID | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| R1 | BCCC paper not publicly available; can't confirm label semantics. | Med | High | Fallback: reverse-engineer from the BCCC GitHub README; if still unclear, document the ambiguity and proceed with conservative assumptions. |
| R2 | Compilation success rate < 50% (esp. for 0.4.x). | Med | High | Escalate to SENTINEL schema review. Consider pragma auto-bump (solc 0.4 → 0.5 with `--evm-version` flag). |
| R3 | 766 NV+vuln contradictions are a real source of noise that hurts training. | High | Med | If WS-B confirms noise, drop them. Document the loss. |
| R4 | The 9-folder contracts are synthetic and create over-representation. | Low | Low | Confirm in WS-B. If confirmed, down-weight or drop duplicates. |
| R5 | SmartBugs overlap > 5%, breaking OOD premise. | Med | High | Drop overlapping contracts from OOD; retest. |
| R6 | `iterative-stratification` not installable. | Low | Low | Implement greedy stratification manually. |
| R7 | solc 0.4.x install fails (binary not available). | Low | Med | Use Docker image `ethereum/solc:0.4.25`. |
| R8 | SENTINEL's preprocessing code assumes the old data path; new contract matrix breaks it. | High | Med | Coordinate with a separate code-change task (WS-H5). Document the schema version bump. |
| R9 | The 2 SENTINEL-missing classes (TOD, WeakAccessMod) turn out to be high-value (e.g., frequently tested in interviews). | Med | Med | Document in WS-F decision. If user picks (a) drop, ensure the decision is reversible for v2. |

---

## 4. Decision Required Before Phase 2 Starts

The user must answer **D-F1** (class reconciliation) before WS-G and WS-H can finish. Everything else can proceed without user input.

**D-F1 prompt for the user (deferred to WS-F session):**

> BCCC has 12 classes; SENTINEL's ADR-0005 plans 10. Two BCCC classes (`TransactionOrderDependence`, `WeakAccessMod`) are missing from SENTINEL.
>
> Options:
> - (A) **Drop the 2 BCCC classes from training.** SENTINEL keeps its 10-class plan. The 2 BCCC columns are present in `contracts_clean.csv` but masked at training time. Easy to add back in v2.
> - (B) **Add the 2 classes to SENTINEL.** Architectural change: 12 binary heads, 12-fold output, ADR-0005 update. Higher SENTINEL coverage but bigger change.
> - (C) **Train on all 12, mask 2 at inference.** Hybrid: model learns 12 heads but only emits 10 at prediction time. Slightly wasteful but no architectural change.
>
> My recommendation: **(A)**. The 2 classes are uncommon (5.2% and 2.8%) and SENTINEL v1 stability is more valuable than the marginal coverage. We can revisit in v2.

---

## 5. Definition of Done (Phase 2)

- [ ] All 8 workstreams (A-H) have `Done criteria` satisfied.
- [ ] `outputs/contracts_clean.csv` + `.parquet` + `metadata.json` + `README.md` are present.
- [ ] A single git commit records Phase 2 completion (and any ADR-0005 update).
- [ ] `CHANGELOG.md` in `Data/Deep_Dive/.../` records what was done, in what session, and any unresolved questions.
- [ ] SENTINEL team has been informed the cleaned dataset is ready for consumption (next session can wire it into `ml/src/preprocessing/`).

---

## 6. What Phase 2 Does NOT Do (Out of Scope)

- **Does NOT train a model.** SENTINEL training is in Run 9 (and beyond); this phase is data engineering.
- **Does NOT modify the BCCC source files.** All writes are to `Data/Deep_Dive/.../outputs/`.
- **Does NOT decide the final SENTINEL graph schema.** That's a separate workstream (already in `ml/src/preprocessing/graph_schema.py`).
- **Does NOT re-validate the SmartBugs-curated corpus.** That's a separate future workstream (planned for post-Run-9).
- **Does NOT touch Run 9 / Run 10 / SENTINEL training code** beyond the minimal wiring in WS-H5.

---

## 7. Tracking & Updates

This plan is a living document. As workstreams complete, update the "Done criteria" checkboxes at the end of each workstream. Major decisions (especially D-F1) should be recorded with rationale in the relevant file (e.g., `class_reconciliation_decision.md`).

If a workstream reveals new questions, add them to §6 of `01_exploration_inventory.md` and the corresponding workstream in this plan.
