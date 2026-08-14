# Phase-8 Real-Data Readiness Audit

**Date:** 2026-08-14
**Status:** COMPLETE — SOURCE, PREPROCESSING, REPRESENTATION, ROLE, AND BOUNDED END-TO-END GATES EXECUTED
**Current launch decision:** **DO NOT START the 100-epoch run. Repair and re-freeze DATA first.**
**Scope:** Physical raw Solidity, ingestion manifests, preprocessing metadata, DATA vNext projection, representations, and the real Phase-8 training boundary.
**Primary decision:** Whether the current repaired dataset is sufficiently faithful and useful to justify the full Phase-8 GPU run.

## Technical summary

The current vNext overlay is structurally consistent with its physical representation population, and the raw ingestion manifests are byte-perfect. The active `.sol` files are real, self-contained Solidity inputs: all 22,493 retained contracts have paired metadata, normalization reproduces the stored source exactly, and no retained file contains an unresolved import. Every one of the 21,657 representation triplets also loads with valid graph/token shapes, finite features, in-range edges/token IDs, and exact sidecar parity. The live GPU path completes optimizer and model-selection steps.

Those passes prove execution integrity, not evidence adequacy. The live audit found multiple material data losses and representation distortions:

1. **65 compile-valid, content-distinct positive contracts were discarded only because they shared an Ethereum address literal with an earlier file**: 60 SolidiFI and 5 SmartBugs Curated. The current deduplicator treats a shared address as proof of duplication. Direct hashing and normalized-text hashing show these 65 files are not duplicates at either level.
2. **One additional SmartBugs access-control contract is valid but was excluded by a compiler-wrapper incompatibility.** Solidity 0.4.9 compiles it successfully without `--allow-paths`; the preprocessing wrapper fails because it supplies that unsupported option unconditionally.
3. **Five retained SmartBugs `time_manipulation` contracts are physically identifiable but intentionally masked in vNext.** The frozen Phase-3 ledger lost the category distinction between `time_manipulation` and the superseded `bad_randomness→Timestamp` mapping. Current `.meta.json` provenance safely distinguishes the five direct Timestamp records from eight bad-randomness records, but the frozen overlay does not consume that evidence.
4. **Normalization corrupts valid Solidity after the preprocessing compile gate.** Compilation occurs before regex comment stripping. The normalizer then treats comment markers inside strings or complex comments as real comments. On current retry, 790 excluded DIVE files fail with direct string/primary/declaration errors, and all seven excluded SolidiFI files begin with a normalization-created invalid top-level fragment. Two excluded SmartBugs Reentrancy contracts now graph successfully and are recoverable immediately.
5. **The GNN sometimes represents the wrong declaration.** Of 21,657 valid graph tensors, 341 select a library or other Slither declaration absent from preprocessing's contract list; `SafeMath` accounts for 312. This directly affects 16 effective weak TOD cells and one strong model-selection cell.
6. **The four-window token cap omits code for most contracts.** Exact offline re-tokenization finds 18,491 / 21,657 represented contracts exceed four windows. Median retained code-token coverage is 43.8% for DIVE and 52.9% for SolidiFI. Of 852 effective optimizer cells, 612 are attached to over-cap inputs.
7. **Normalized/cross-source deduplication is incomplete.** There are 120 byte-identical normalized-code groups containing 288 contract records. Ten groups retain multiple frozen group IDs; one identical DIVE/SmartBugs pair has different training roles and target states.

Recovering the already-identified raw/category records would add as many as **71 strong positive class cells** to the current 403-cell strong-positive semantic population, a **17.6% increase**, before representation generation and role re-freezing. The optimizer currently uses only **275 strong + 577 weak cells**; 10 strong + 27 weak target cells are excluded for missing representations, and 118 strong cells are reserved for model selection/internal audit. With supervision this small, the normalization, selection, truncation, and omitted-source findings are not marginal.

The 604 weak DIVE TOD targets are not affected by DIVE's exact-content label conflicts. DIVE remains useful as weak TOD plus unlabeled representation exposure, but it does not supply strong truth for the other classes.

## Audit frames and definitions

Counts in this report use four explicit frames:

| Frame | Meaning |
|---|---|
| raw manifest record | One path recorded in a source ingestion manifest |
| unique pipeline text | SHA-256 after Python universal-newline decoding, matching preprocessing's `read_text(errors="replace")` behavior |
| preprocessed contract | One retained content-addressed `.sol` plus `.meta.json` pair |
| vNext contract/class cell | One current `contract_id × class_index` semantic record |

"Address-only drop" means the raw file has a different exact content hash and a different comment/whitespace-normalized hash from its alleged duplicate, but both files contain at least one identical 40-hex-character Ethereum address. This is the exact branch by which the current deduplicator removes the file.

## Reproducible evidence

The read-only profilers are:

```bash
ml/.venv/bin/python docs/plan/ml-R4/scripts/p8_audit_real_data.py
ml/.venv/bin/python docs/plan/ml-R4/scripts/p8_audit_representations.py
TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 \
  ml/.venv/bin/python docs/plan/ml-R4/scripts/p8_audit_token_coverage.py
```

Together they verify manifest bytes, preprocessed pairs, metadata/source joins, normalized source identity, DIVE label/folder agreement, duplicate mechanisms, category attrition, vNext supervision, every graph/token/sidecar payload, current excluded-contract extraction behavior, SolidiFI injection-log coverage, and exact pre-subsampling token coverage. They write nothing and print deterministic JSON to stdout.

Primary live inputs:

- `data_module/data/raw/{dive,smartbugs_curated,solidifi}/ingestion_manifest.json`
- `data_module/data/raw_staging/dive_labels/DIVE_Labels.csv`
- `data_module/data/preprocessed/{dive,smartbugs_curated,solidifi}/`
- `data_module/data/exports/sentinel-r4-vnext-v1/{label_states,ml_targets}.parquet`
- `docs/plan/ml-R4/ledger/evidence_ledger_v1.parquet`
- `docs/plan/ml-R4/specs/data_vnext_policy_v1.json`

## Gate 1 — physical data and training-boundary inventory

The project contains approximately 7.0 GB under `data_module/data`, 34 GB under historical `ml/data`, and 9.5 GB under historical `mlruns`. Phase 8 does not consume all of that material. Its active input is the 3.2 MB vNext semantic overlay plus the 2.1 GB graph/token/sidecar representation root.

| Source | Raw manifest | vNext contracts | Represented | Excluded |
|---|---:|---:|---:|---:|
| DIVE | 22,330 | 22,073 | 21,247 | 826 |
| SmartBugs Curated | 143 | 137 | 134 | 3 |
| SolidiFI | 350 | 283 | 276 | 7 |
| **Total** | **22,823** | **22,493** | **21,657** | **836** |

The overlay has 22,493 unique contract IDs and 224,930 contract/class rows. The 21,657 represented contracts have exactly one graph, one token tensor, and one sidecar each: 64,971 required files. The 836 representation omissions exactly equal the frozen `EXCLUDED` population.

`mlruns.db` is not a Phase-8 input. It contains 114 historical runs, including 78 still marked `RUNNING`, and has zero MLflow dataset/input lineage records. The new vNext runner uses its own bound filesystem artifacts rather than this database.

## Gate 2 — raw Solidity, preprocessing, provenance, and source attrition

### What passed

| Check | Result |
|---|---|
| Manifest file existence, byte count, and SHA-256 | 22,823 / 22,823 exact |
| DIVE CSV grain | 22,330 rows / 22,330 unique IDs / IDs 1–22,330 |
| DIVE label domain | 178,640 / 178,640 cells are binary `0` or `1` |
| DIVE class-folder membership vs CSV-positive IDs | Exact for all 8 classes |
| DIVE label symlinks | 54,919 / 54,919 valid; 0 broken |
| Retained `.sol`/`.meta.json` pairing | 22,493 / 22,493 |
| Retained metadata JSON | 22,493 / 22,493 parseable |
| Metadata source/path/hash/line-count checks | 0 mismatches |
| Recomputed normalization vs stored `.sol` | 22,493 / 22,493 exact |
| Retained unresolved imports | 0 / 22,493 |
| Exact retained contract-ID overlap across active sources | 0 pairs |

All retained source files report `flatten_status=skipped_no_imports`; the corpus is already self-contained at the preprocessing boundary. This is appropriate for the current representation extractor.

### Source shape

| Source | Preprocessed | Raw lines p50 / p95 / max | Normalized lines p50 / p95 / max | Solidity era |
|---|---:|---:|---:|---|
| DIVE | 22,073 | 366 / 1,734 / 9,477 | 325 / 1,086 / 6,588 | 9,221 legacy; 743 transitional; 12,109 modern |
| SmartBugs Curated | 137 | 41 / 307 / 2,470 | 35 / 228 / 1,657 | 137 legacy |
| SolidiFI | 283 | 375 / 739 / 1,031 | 311 / 638 / 917 | 283 legacy |

DIVE contains 16 records with no `contract_names` metadata, but 15 are library/interface-only files rather than empty data. The single declarationless retained file is already `EXCLUDED` because it has no representation. Eight DIVE raw files contain 551 UTF-8 replacement characters in total; their manifest bytes are intact and the retained pipeline identity is internally consistent, but their exact original encoding is not preserved through text decoding.

### Finding RD-001 — address matching discarded 65 distinct positive contracts

**Severity:** High
**Confidence:** High — reproduced from raw bytes, normalized text, `dropped.csv`, and the current deduplicator branch.

| Source/class | Raw | Retained | Address-only drops | Exact drops | Other drop |
|---|---:|---:|---:|---:|---:|
| SmartBugs Reentrancy | 31 | 30 | 1 | 0 | 0 |
| SmartBugs unchecked low-level calls | 52 | 48 | 4 | 0 | 0 |
| SolidiFI Overflow-Underflow | 50 | 49 | 0 | 1 | 0 |
| SolidiFI Re-entrancy | 50 | 39 | 10 | 1 | 0 |
| SolidiFI TOD | 50 | 39 | 10 | 1 | 0 |
| SolidiFI Timestamp-Dependency | 50 | 39 | 10 | 1 | 0 |
| SolidiFI Unchecked-Send | 50 | 39 | 10 | 1 | 0 |
| SolidiFI Unhandled-Exceptions | 50 | 39 | 10 | 1 | 0 |
| SolidiFI tx.origin | 50 | 39 | 10 | 1 | 0 |

The 60 SolidiFI address-only drops are 17.1% of the 350-record raw source. Each file compiled before the dedup step and has distinct exact and normalized content. These are often variants of a shared base contract, which is a valid reason to place them in one leakage group, but not a valid reason to erase the injected-class variant and its positive label.

The earlier integration report noted the address heuristic's false-positive behavior but concluded that none of the final 67 SolidiFI drops were address duplicates. The live current files contradict that conclusion: **60 of 67 are address-only and only 7 are exact duplicates**.

**Impact:** The current vNext strong targets are systematically reduced by 10 examples in each SolidiFI class except IntegerUO, plus four SmartBugs CallToUnknown examples and one SmartBugs Reentrancy example. This is material for classes with only 39–87 total strong targets.

**Recommended remediation:** Disable address-as-duplicate removal. Retain distinct variants, calculate explicit exact/normalized/base-family group identities, and assign the entire family to one frozen role. Rebuild representations, ledger, roles, and vNext overlay as a new version; do not overwrite historical v1/v2 artifacts.

### Finding RD-002 — one valid SmartBugs contract was rejected by the compiler wrapper

**Severity:** Medium overall; High for the small ExternalBug class
**Confidence:** High — directly reproduced.

`repo/access_control/parity_wallet_bug_1.sol` is recorded as a compile failure. Direct execution shows:

- `solc-0.4.9 --bin <file>`: exit 0 and bytecode produced;
- the same command with `--allow-paths`: exit 1, `unrecognised option '--allow-paths'`.

The preprocessing compiler supplies `--allow-paths` to every Solidity version. This removes one of 18 raw SmartBugs access-control contracts, 5.6% of that source/category.

**Recommended remediation:** Make compiler flags version-aware, add a regression fixture for pre-`--allow-paths` solc, then recover this contract in a new DATA version.

### Finding RD-003 — five direct SmartBugs Timestamp positives are available but masked

**Severity:** Medium
**Confidence:** High; the masking is intentional and already documented, while physical recoverability is newly verified.

The current policy approves `smartbugs_curated:time_manipulation → Timestamp` and rejects `bad_randomness → Timestamp`. The frozen Phase-3 ledger has `source_native_label=null`, so vNext conservatively masks all historical SmartBugs Timestamp rows because it cannot distinguish those categories inside that ledger.

The current physical metadata does distinguish them exactly through `original_path`:

- 5 retained `repo/time_manipulation/*.sol` contracts;
- 8 retained `repo/bad_randomness/*.sol` contracts.

Thus five direct strong Timestamp cells can be recovered without manual semantic inference, provided a new evidence-ledger version binds the preprocessed metadata and raw source manifest. This would increase current strong Timestamp targets from 39 to 44 before recovering SolidiFI address-dropped variants, or to 54 if both repairs succeed.

### Finding RD-004 — DIVE has 190 silently collapsed duplicate paths, not 257 compile failures

**Severity:** Medium for provenance; Low for the current Phase-8 TOD target
**Confidence:** High.

DIVE reconciles as:

| Outcome | Count |
|---|---:|
| Raw manifest records | 22,330 |
| Unique preprocessing text hashes | 22,140 |
| Exact duplicate extra paths | 190 across 88 groups |
| Recorded compile failures | 67 |
| Retained content-addressed contracts | 22,073 |

The multiprocessing preprocessing path creates a fresh deduplicator per file. Exact duplicate source paths therefore converge silently on the same content-addressed output filename rather than being recorded in `dropped.csv`. Earlier documentation that attributes all 257 raw-to-export omissions to compilation is incorrect.

Eighteen of the 88 exact-content groups carry conflicting DIVE folder labels. Conflict groups by class are: Access Control 10, Arithmetic 9, Unchecked Return Values 8, Reentrancy 5, Time manipulation 4, and DoS 2. No duplicate group conflicts on Front Running, so the current 604 weak TOD targets are unaffected by this issue. All other DIVE labels are masked in vNext v1.

**Recommended remediation:** Preserve all source-record identities separately from content identity, retain their label claims, and adjudicate or explicitly aggregate conflicts before deriving one contract-level semantic row. Make parallel preprocessing aggregate duplicate provenance deterministically rather than relying on filename collision.

### Finding RD-005 — source recovery is locally valid but not portable

**Severity:** Medium
**Confidence:** High.

- SolidiFI is a clean detached checkout at the manifest's exact Git commit `4b0573e1…`.
- SmartBugs is a clean checkout at `230e6491…`, but the DATA manifest uses the date-like manual pin `2025-01-01` rather than that commit.
- DIVE uses an absolute local staging symlink, a date-like manual pin, and no source URL. Its manifest binds every byte but does not provide a self-sufficient recovery locator.
- All large physical data is Git-ignored.

This does not block the present local run, but a disk loss or fresh-machine rebuild cannot be proven reproducible from repository state alone.

**Recommended remediation:** Record content-addressed archive locations and source acquisition instructions; bind SmartBugs to its exact commit; avoid absolute workspace paths in the portable acquisition descriptor.

## Gate 3 — representation contents, exclusions, and code coverage

### What passed

Every required representation was deserialized and inspected, not merely counted:

| Check | Result |
|---|---|
| Graph/token/sidecar triplets loaded | 21,657 / 21,657 |
| Hard tensor/sidecar structural failures | 0 |
| Graph feature shape/dtype | all `float32 [N,12]` |
| Non-finite or out-of-range graph features | 0 |
| Invalid edge-index shape/dtype/bounds | 0 |
| Invalid edge-attribute length/range | 0 |
| Token shape/dtype | all `int64 [4,512]` |
| Non-binary/non-right-padded attention masks | 0 |
| Out-of-vocabulary token IDs | 0 |
| Sidecar graph/token count mismatches | 0 |
| Schema/extractor pair | 21,657 `v9 / v2.1-windowed-gcb` |

The graph path is not empty or uniformly degenerate. DIVE graphs have median 258 nodes / 638 edges; SmartBugs 15 / 31; SolidiFI 336 / 623. Twenty-seven graphs have zero edges, but all have at least one node and load correctly. Isolated declaration nodes are common and expected under the current graph schema.

### Finding RD-006 — compile-before-normalize accepts files that normalization corrupts

**Severity:** Critical for preprocessing correctness
**Confidence:** High — current extractor retry plus direct source/output inspection.

The preprocessing order is compile → deduplicate → normalize. The compile gate therefore validates the raw/flattened text, not the normalized `.sol` later given to Slither and the tokenizer. Comment removal uses regular expressions that do not understand Solidity string/comment lexical state. A value such as `'https://…'` can become `'https:`, and complex block-comment material can leave an orphan `/` at top level.

All 836 excluded rows have no graph, token, or sidecar component. Re-running their normalized Solidity through the current graph extractor produced:

| Source/outcome | Contracts |
|---|---:|
| DIVE — expected string end quote | 620 |
| DIVE — expected primary expression | 120 |
| DIVE — declaration expected | 50 |
| DIVE — top-level definition expected | 8 |
| DIVE — other Slither/compiler failure | 28 |
| SolidiFI — top-level definition expected | 7 |
| SmartBugs — still fails | 1 |
| SmartBugs — now succeeds | 2 |

The 620 + 120 + 50 direct DIVE syntax categories alone prove at least **790 normalization-corrupted DIVE outputs**. The seven excluded SolidiFI outputs visibly contain the same kind of orphan top-level fragment. Two excluded SmartBugs Reentrancy contracts (`reentrancy_bonus.sol` and `reentrancy_cross_function.sol`) now produce valid 17-node/40-edge and 15-node/48-edge graphs and can be recovered without source adjudication.

This changes the interpretation of the historical 836-row exclusion: it is not a harmless benchmark-availability limitation. Much of it is pipeline-created loss after a misleading `compile_status=ok` record.

**Recommended remediation:** Replace regex comment stripping with a Solidity-aware lexer that preserves strings, compile the exact normalized output before promotion, add URL/nested-comment regression fixtures, then regenerate all preprocessed IDs and downstream artifacts as a new DATA version.

### Finding RD-007 — 341 GNNs select a library or non-contract declaration

**Severity:** High for graph-label alignment
**Confidence:** High — graph `contract_name` joined to physical normalized source and preprocessing metadata.

The graph extractor selects one Slither declaration per file. Slither exposes libraries as non-interface candidates, and the fallback is last-defined. This selected a name absent from preprocessing's actual `contract` declarations for 341 graphs:

| Source | Mismatched graphs |
|---|---:|
| DIVE | 340 |
| SmartBugs | 1 |

`SafeMath` is selected in 312 cases. The affected roles are 185 `TRAIN_WEAK`, 155 `TRAIN_UNLABELED`, and one `MODEL_SELECTION`. Sixteen effective weak TOD cells are trained against these wrong-declaration graphs; the SmartBugs case is one strong CallToUnknown model-selection cell.

This is the confirmed lower bound. Multi-contract files are widespread—18,230 DIVE, 44 SmartBugs, and 189 SolidiFI represented files—and a selected name that exists in the contract list is not automatically the declaration owning the label.

SolidiFI provides a useful direct check through `BugLog_N.csv`: 275 / 276 represented files have at least one logged injection inside the contract selected for the GNN, but the selected contract covers only 4,335 / 7,203 logged injection sites (60.2%). One Overflow-Underflow file selects an empty `RaffleToken` declaration with none of its 19 logged injected sites. The token branch consumes the full normalized file, so this is partial multimodal loss rather than total sample loss.

**Recommended remediation:** Represent every relevant contract declaration or bind each label to an explicit target contract and use `by_name`; exclude libraries from target selection. Preserve full-file tokens only as a deliberate second view, not as compensation for an unverified graph target.

### Finding RD-008 — four selected token windows omit most long-contract code

**Severity:** High for DIVE/SolidiFI evidence coverage
**Confidence:** High — exact re-tokenization with the locally pinned GraphCodeBERT tokenizer and the production window-selection algorithm.

The saved token payload records four selected 512-token windows at most. Re-tokenizing all 21,657 represented contracts found zero saved-window-count mismatches but substantial pre-subsampling loss:

| Source | Represented | Over four windows | Median full tokens | Median retained coverage | Median omitted tokens |
|---|---:|---:|---:|---:|---:|
| DIVE | 21,247 | 18,206 | 4,374 | 43.8% | 2,534 |
| SmartBugs | 134 | 22 | 306 | 100% | 0 |
| SolidiFI | 276 | 263 | 3,646 | 52.9% | 1,772 |
| **Total** | **21,657** | **18,491** | **4,343** | **44.3%** | **2,534** |

The worst retained coverage is 1.9%; the maximum contract requires 399 pre-subsampling windows. Across the corpus, 78,566,016 code-token positions are omitted by the four-window selection. The start/middle/end sampling is deterministic, but it cannot guarantee that a sparse vulnerability site is among the selected windows.

This affects **612 / 852 effective optimizer cells**: 202 / 275 strong cells and 410 / 577 weak cells. It also affects 79 / 118 positive-only outcome-metric cells.

**Recommended remediation:** Measure label/injection-line inclusion, not only token percentage. For a first repair, use vulnerability-aware/contract-aware windows or a bounded hierarchical encoder and version the representation schema. The 100-epoch baseline should not silently treat a median 44.3% token view as full-contract evidence.

### Finding RD-009 — normalized duplicate identities survive grouping

**Severity:** Medium
**Confidence:** High — byte hashes over every stored normalized `.sol` joined to frozen roles/groups.

The active population contains 120 identical-normalized-code groups with 288 records, or 168 extra contract records. Two groups cross sources; ten have multiple frozen `group_id` values; one has conflicting target state. The only cross-role case is the same normalized contract assigned as:

- DIVE `TRAIN_WEAK`, weak TOD target;
- SmartBugs `TRAIN_UNLABELED`, no target.

Both roles are training-side, so this is not current model-selection/internal-audit leakage. It is nevertheless duplicate weighting and provenance drift, and it shows the frozen grouping key is not complete at the actual model-input boundary.

**Recommended remediation:** Derive leakage families after the final normalized representation unit is fixed, join exact normalized hashes across all sources, and then freeze roles. Keep conflicting source claims rather than erasing one.

## Gate 4 — bounded real end-to-end execution

The canonical GPU micro-smoke was re-run from current `main` with the real vNext overlay and representation root:

```bash
TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTHONPATH=.:data_module \
  ml/.venv/bin/python docs/plan/ml-R4/scripts/p8_run_micro_smoke.py \
  --train-batches 2 --selection-batches 1 --batch-size 1
```

It passed on the RTX 3070 Laptop GPU: two optimizer steps, finite masked losses, one positive-only model-selection cell, 970.04 MB peak allocated, no Run12 weights loaded, and no checkpoint written. This proves dataset loading, collation, model forward/backward, masked loss, and optimizer wiring are executable. It is not quality, calibration, negative discrimination, full-batch-memory, or long-horizon evidence.

## Current supervision and recoverable opportunity

Current vNext target cells by active source are:

| Class | DIVE weak | SmartBugs strong | SolidiFI strong | Current total |
|---|---:|---:|---:|---:|
| CallToUnknown | 0 | 48 | 39 | 87 |
| DenialOfService | 0 | 6 | 0 | 6 |
| ExternalBug | 0 | 17 | 39 | 56 |
| IntegerUO | 0 | 15 | 49 | 64 |
| MishandledException | 0 | 0 | 39 | 39 |
| Reentrancy | 0 | 30 | 39 | 69 |
| Timestamp | 0 | 0 | 39 | 39 |
| TransactionOrderDependence | 604 | 4 | 39 | 647 |
| GasException / UnusedReturn | 0 | 0 | 0 | 0 |

Those are semantic target counts, not optimizer counts. The exact current disposition is:

| Disposition | Strong | Weak | Total |
|---|---:|---:|---:|
| Effective training loss | 275 | 577 | 852 |
| Positive-only model selection | 56 | 0 | 56 |
| Positive-only internal audit | 62 | 0 | 62 |
| Excluded for no representation | 10 | 27 | 37 |
| **Semantic overlay total** | **403** | **604** | **1,007** |

Known existing-data repairs could change the strong-positive population as follows:

| Class | Current strong | Known recoverable additions | Candidate strong |
|---|---:|---:|---:|
| CallToUnknown | 87 | 14 | 101 |
| DenialOfService | 6 | 0 | 6 |
| ExternalBug | 56 | 11 | 67 |
| IntegerUO | 64 | 0 | 64 |
| MishandledException | 39 | 10 | 49 |
| Reentrancy | 69 | 11 | 80 |
| Timestamp | 39 | 15 | 54 |
| TransactionOrderDependence strong | 43 | 10 | 53 |
| **Total strong** | **403** | **71** | **474** |

These are upper-bound source cells before representation generation, conflict review, and role allocation. They are not a promise that all 71 will become effective training-loss cells. Separately, repairing normalized-output representation loss can recover some of the ten strong and 27 weak cells already present in the semantic overlay but currently excluded. These two recovery frames must not be added together without a new role freeze.

## Final launch decision

The data is real and the current overlay is executable, but **the full Phase-8 run should not start on `sentinel-r4-vnext-v1`**. A short diagnostic smoke has already served its purpose. A 100-epoch evidence-generating run is not worth its cost while:

- 65 compile-valid distinct positive contracts are mislabeled as duplicates and absent;
- one valid benchmark contract is blocked by a known wrapper flag bug;
- five direct Timestamp positives are physically recoverable but semantically masked;
- at least 790 DIVE outputs and seven SolidiFI outputs are corrupted after the misleading preprocessing compile pass;
- two excluded strong SmartBugs records already graph successfully under the current runtime;
- 341 GNN inputs select a library/non-contract declaration, including 16 effective weak cells;
- 18,491 contracts omit code tokens, including 612 / 852 optimizer cells;
- the actual effective supervision is only 275 strong + 577 weak positive cells, with no confirmed negatives;
- 120 identical normalized-code groups remain duplicated and ten are not grouped together.

Execution success does not compensate for missing or misaligned evidence. Training now would create a cleanly bound checkpoint for a corpus already known to be materially repairable, forcing a second expensive run and making comparisons harder to interpret.

## Required next tranche

1. Fix address deduplication, version-aware compiler flags, and Solidity-aware normalization; compile the exact promoted output.
2. Make representation targeting contract-aware/library-safe and design a measured long-contract token strategy.
3. Recover the known source records and currently recoverable exclusions; retain source claims and SolidiFI injection-log lineage.
4. Recompute exact normalized/base-family groups across sources, then freeze new roles and semantics under a new DATA version.
5. Rebuild representations and rerun these three profilers, G7-style physical binding, and the bounded GPU smoke.
6. Only then authorize a new fixed-horizon Phase-8 run. Preserve v1/v2 artifacts as historical comparison evidence.

## Remaining limitations

- No manual TP/FP semantic verdict was made. Such a rate requires frozen per-class criteria, controls, and independent review.
- The 67 DIVE compile failures were re-run through the current wrapper and all still failed, but their ultimate recoverability was not individually adjudicated. Forty lack a recognized pragma; several others fail from exact-version/range or stack-depth behavior.
- The 71-cell recovery estimate assumes content-distinct injected/category records remain valid positive authority under a new versioned DATA decision; that decision still needs a controlled rebuild and validation gate.
- SolidiFI graph injection coverage uses the benchmark's published raw line logs and declaration spans; it proves logged-location inclusion, not semantic correctness of every injected pattern or completeness of graph features.
- Exact token-position coverage is measured; whether omitted positions contain the decisive vulnerability signal still requires class/injection-location coverage tests in the repaired representation design.
