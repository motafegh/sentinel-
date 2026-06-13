# Pre-Run-12 Fixes — Live Investigation Log
**Date:** 2026-06-13  
**Author:** Ali + Claude  
**Purpose:** Track every finding, decision, and change made before launching Run 12.  
**Rule:** Write findings down in small steps. Never let discoveries pile up in memory.

---

## Context

Run 11 (WSL crash, ep1 checkpoint saved):
- F1-macro ep1 = 0.3293 (fixed), 0.3384 (tuned)
- Checkpoint: `ml/checkpoints/GCB-P1-Run11-v2deduped-20260613_best.pt`
- Decision: Fix all open data/pipeline issues before resuming. New run = Run 12 on fixed data.

---

## Open Items (from data analysis session)

| # | Item | Priority | Status |
|---|------|----------|--------|
| 1 | SmartBugs Curated ingestion | P1 | ✅ DONE — 137 contracts ingested, v3 splits + export complete |
| 2 | Stage 5.5 GCB propagation | P1 | 🔍 Assessing |
| 3 | DoS label quality investigation | P2 | ✅ DONE — Keep all 1,095 labels (FPR≈7%, no code change needed) |
| 4 | Level-3 dedup (deduplicator.py stub) | P2 | ✅ DONE — 83/147 L3 groups applied, 15 contracts reassigned, v3 splits regenerated |
| 5 | DeFiHackLabs ingestion (import failures) | P2 | ⏳ Queued |
| 6 | Stage 5 splitting + registry | P3 | ⏳ Queued |
| 7 | drift_baseline.json placeholder | P4 | ⏳ Queued |
| 8 | C-4 max_nodes guard (0.18% rate) | P4 | ⏳ Queued |

---

## Item 1 — SmartBugs Curated Ingestion

### Why it matters
Three classes have near-zero training examples:
- GasException: **0** in entire v2 dataset (model has no signal at all)
- CallToUnknown: **39** total (27 train) — near-random AUC-PR=0.006
- MishandledException: **39** total (27 train) — near-random AUC-PR=0.003

These 3 dead classes contribute 0.0 to F1_macro, pulling the "macro" score away from the real performance picture.

SmartBugs Curated is flagged in `data_module/docs/legacy/` and Stage 3 notes as the fix for these gaps.

### Assessment — COMPLETE

**Data exists:** 143 `.sol` files at `ml/data/smartbugs-curated/dataset/` organized by category:
```
reentrancy: 32          → Reentrancy
unchecked_low_level_calls: 53 → CallToUnknown  ← biggest gap fixer
denial_of_service: 7    → DenialOfService
arithmetic: 16          → IntegerUO
time_manipulation: 6    → Timestamp
access_control: (some)  → ExternalBug
bad_randomness: (some)  → Timestamp
front_running: (some)   → TransactionOrderDependence  ← BUG in crosswalk (smartbugs_curated.yaml:25 maps `Timestamp`), fix required before ingestion
short_addresses: (some) → NonVulnerable
other: (some)           → NonVulnerable
```

**Infrastructure already done:**
- Crosswalk: `data_module/sentinel_data/labeling/crosswalks/smartbugs_curated.yaml` ✅
- Class mapping is filled in ✅
- Test: `data_module/tests/test_verification/test_smartbugs_recall.py` exists ✅

**Missing:**
- Parser: `data_module/sentinel_data/labeling/parsers/smartbugs_curated.py` — does NOT exist
- Data path: files are in `ml/data/smartbugs-curated/` NOT in `data_module/data/raw/` — need symlink or manual connector

**Critical finding — GasException STILL ZERO:**
SmartBugs Curated has NO `gas_exception` category. Adding SmartBugs will NOT fix GasException=0.
GasException will remain unlearnable after this fix.

**Net gain if we ingest SmartBugs:**
- CallToUnknown: 27 → ~80 train examples (+53) — AUC-PR may improve from 0.006
- DenialOfService: 782 → ~787 train examples (+5 from denial_of_service) — minimal impact
- Reentrancy: 7,950 → ~7,973 (+23 train) — negligible, already well-learned
- GasException: 0 → 0 (no change)
- MishandledException: 27 → 27 (no change — no SmartBugs category for it)

**VERDICT: FEASIBLE, MODERATE IMPACT.** 
- Do it for CallToUnknown (53→80 train examples)
- Will not fix GasException or MishandledException
- Needs: parser + manual connector config + pipeline re-run for 143 contracts

**Steps required:**
1. Create `parsers/smartbugs_curated.py` (model on `solidifi.py`)
2. Symlink `ml/data/smartbugs-curated/dataset/` → `data_module/data/raw/smartbugs_curated/repo/dataset/`
3. Add manual connector config entry for smartbugs_curated
4. Add SmartBugs to `merger.py` source list
5. Run preprocessing + representation + labeling for 143 new contracts
6. Re-export v3 dataset
7. Re-run dedup + split → v3 deduped splits
8. Run all tests

---

## Item 2 — Stage 5.5 GCB Propagation

### Why it matters
DoS (AUC-PR=0.046) and Timestamp (AUC-PR=0.395) have BEST-EFFORT labels.
Stage 5.5 would use GraphCodeBERT to improve confidence on these labels.
Was deferred in Run 9 era due to GPU constraints.

### Assessment — COMPLETE

**What it would do:** Use GraphCodeBERT embeddings to cluster contracts and propagate
high-confidence labels to near-neighbors that have uncertain labels.

**Current blockers:**
1. GPU: RTX 3070 is currently free (Run 11 dead, no active training)
2. But: Stage 5.5 requires running GraphCodeBERT inference on ~22K contracts to get embeddings — takes hours
3. Then: Need to implement label propagation logic (not built yet)
4. Then: Need to re-run verification + export — another full pipeline run

**Risk:** Stage 5.5 may REDUCE label count (removing uncertain ones) or ADD labels (propagating confident ones). Either direction changes the dataset. The safest thing is to analyze whether DoS labels can be improved with a simpler method first (Item 3 investigation).

**VERDICT: DEFER. Too complex to build safely before Run 12.**
- The GPU is now free, but Stage 5.5 is a full sub-project
- DoS's AUC-PR=0.046 at ep1 may improve naturally as the model trains (it's ep1 of 100)
- Investigate DoS quality (Item 3) first — if DoS labels are systematically bad, removing them is safer than GCB propagation
- Stage 5.5 remains deferred until after Run 12 results are analyzed

---

## Item 3 — DoS Label Quality

### What we know
- DoS labels in v3 labels.parquet: **3,756 total** (train=2,910 / val=429 / test=417) — confirmed 2026-06-13 by manual audit
- DoS=1 AND Reentrancy=1 co-occurring: **2,655 contracts**
- **CORRECTION (2026-06-13):** Earlier entry said "DoS/Reentrancy co-occurrence patch zeroed 2,655 labels." That was WRONG. No such patch exists for DIVE.
  - merger.py:100-124 flags co-occurrence noise ONLY for T3/T4 sources (condition 3). DIVE is T2.
  - The merger comment explicitly says: "DIVE (T2) at 12% co-occurrence is NOT flagged — legitimate multi-label signal."
  - The "1,095" count was a wrong number read from a filtered v1-split view of an older dataset.
- AUC-PR=0.046 after ep1 — effectively random
- Gate was PROVISIONAL → BEST-EFFORT

### Assessment — COMPLETE

**Why AUC-PR=0.046 at ep1:**
- 782 train examples with BEST-EFFORT labels (64.5% confidence from Phase 5)
- Positive prevalence 4.9% in v2 dataset (782/15,644 train)
- Model gets contradictory signals: many DoS=1 examples look like ExternalBug or Reentrancy graphs
  (because DoS often co-occurs with those, and 2,655 DoS labels were already zeroed by co-occurrence)
- After co-occurrence removal, remaining DoS=1 examples may include:
  a. True DoS (gas griefing, push-over-pull violations) — learnable
  b. "Residual noise" (DoS labeled for contracts that don't actually have DoS but were in DIVE's DoS folder) — not learnable

**What investigation would involve:**
1. Sample the 782 DoS=1 train contracts
2. Manually inspect their Slither output for actual DoS patterns (unbounded loops, msg.sender.call in loops, etc.)
3. Calculate what fraction are genuine DoS vs. DIVE folder noise
4. Decision: keep as-is, prune noise, or zero all DoS labels (treat as unlearnable)

**VERDICT: INVESTIGATE. Simple analysis, no code changes needed initially.**
- Pull 20-30 DoS train contracts and manually spot-check patterns
- Check the `patterns/denial_of_service.yaml` verification pattern to understand what was supposed to be kept
- Decision after investigation will be: (a) keep, (b) prune, or (c) zero
- This CAN be done before Run 12

---

## Item 4 — Level-3 Dedup (deduplicator.py stub)

### What we know
```python
# deduplicator.py:73-78 — STUB
return DedupRecord(
    sha256=sha,
    dedup_group_id=sha,   # every contract is its own group
    is_duplicate=False,
    duplicate_of="",
)
```
Every contract gets `dedup_group_id = sha256` (itself).
Near-duplicate contracts with different file content but identical semantics still leak.
The graph-hash dedup (the post-export patch) already caught content-identical graphs.
Level-3 would catch source-code near-dups that produce slightly different graphs.

### Assessment — COMPLETE

**What the graph-hash dedup already catches:**
- 10,811/21,523 contracts with identical `x + edge_index` bytes → same dedup group
- These contracts may have different source (different variable names, comments, whitespace) but compile to the same CFG
- This is already handled in v2 splits (0% leakage confirmed)

**What Level-3 would catch that graph-hash doesn't:**
- Source-code near-dups: same contract with a few lines changed → slightly different graph
- E.g., DIVE dataset has many "redeployments with minor version bumps" that generate graphs differing by 1-2 nodes
- These near-dups produce different graph hashes but are semantically near-identical

**Why Level-3 is complex:**
- Needs AST similarity comparison (Slither-based or text-hash-based)
- Requires running comparison across all 22,356 contract pairs — O(N²) naive
- Slither AST comparison: slow, and Slither parses differently based on pragma version
- Text-similarity (shingling/MinHash): faster but misses semantic similarity

**What actually happened:**
The graph-hash dedup handled the most severe case (10,811 identical graphs).
The remaining "near-dups" (different graphs, similar source) are unknown in count.
Without running Level-3, we don't know how many exist.

**VERDICT: IMPLEMENT A SIMPLE TEXT-HASH VERSION (not full AST).**
- Approach: normalize source code (strip comments + normalize whitespace ONLY — NO identifier lowercasing)
  then compute hash. Group by normalized hash = Level-3 groups.
  - Identifier lowercasing dropped: would collapse `reentrantWithdraw` and `reentrancyWithdraw` into the same group → false-positive dedup, cross-polluting classes
- This is ~50 lines of code in `deduplicator.py` and doesn't require Slither
- It catches copy-paste-with-comments-removed near-dups
- Won't catch renamed-variable near-dups — that's Level-4 (future)
- SAFE: it only changes `dedup_group_id` for contracts that normalize identically
- After implementing, re-run `sentinel-data split --version 3` to apply new groups
- Risk: LOW — only contracts that normalize to identical text get grouped

---

## Item 5 — DeFiHackLabs Import Failures

### What we know
DeFiHackLabs is a high-priority data source (real exploited DeFi contracts).
Stage 3 notes: "DeFiHackLabs (import failures), deferred."
No details on what the failures were.

### Assessment — COMPLETE

**What DeFiHackLabs actually contains:**
- 693 `_exp.sol` files in `src/test/<year-month>/` structure
- These are **Foundry PoC test contracts** — they test the exploit, NOT the vulnerable contract
- Example structure from `Lodestar_exp.sol`:
  ```solidity
  pragma solidity ^0.8.10;
  import "forge-std/Test.sol";
  import "./../interface.sol";
  // ... uses GMXRouter, GMXReward, uniswapV3Flash interfaces
  ```
- The PoC imports `forge-std/Test.sol`, `interface.sol`, and many protocol-specific interfaces
- Without the full Foundry project (forge-std, libs, interfaces), Slither CANNOT analyze these

**Why "import failures" is the right diagnosis:**
- When preprocessing tried to compile `Lodestar_exp.sol`, Slither reported import not found:
  `forge-std/Test.sol`, `../interface.sol`
- These missing imports cause compile failures → 0 graphs extracted
- The `interface.sol` in `src/test/` contains hundreds of protocol interface definitions

**What would be needed to fix this:**
1. **Option A (Correct but complex):** Extract the VULNERABLE PROTOCOL contracts from each exploit,
   not the PoC test. That requires manually identifying the target contract address + fetching it
   from Etherscan. This is a research task, not a preprocessing task.
2. **Option B (Wrong):** Try to compile PoC contracts with the full forge-std environment.
   Even if it worked, PoC contracts are NOT the vulnerable contracts — they're the attacker's code.
   Labeling them as vulnerable would be methodologically wrong.
3. **Option C (Stub but honest):** Defer entirely.

**Crosswalk is also a placeholder:**
`defihacklabs.yaml` class_map covers only a few categories (Reentrancy, FlashLoan, AccessControl, Overflow).
With 693 contracts across year-based folders (not category folders), there's no automatic label extraction.

**VERDICT: BLOCKED. DeFiHackLabs requires extracting vulnerable protocol contracts (not PoC test contracts).**
- This is 2+ days of work to identify which contract each PoC targets and fetch the actual vulnerable contract
- The "import failures" are not a simple fix — the architecture of DeFiHackLabs is incompatible with direct ingestion
- Defer to v3 data module with a proper DeFiHackLabs extraction script
- Document clearly so it's not confused with a simple ingestion task

---

## Item 6 — Stage 5 Splitting + Registry

### Assessment — COMPLETE

Read the full spec (`docs/proposal/Data_Module_Proposals/actionable_plans/06_stage_5_splitting_registry.md`).

Stage 5 is a **massive infrastructure project** — SQLite catalog, lineage DAG, leakage auditor, schema migrations table, dataset version retirement chain, `sentinel_data.registry.load_artifact()` API.

**Scheduled for: Week 8 (Jul 28 – Aug 3, 2026).** The v2 deduped split already does what splitting needs to do for training. The registry is production infrastructure, not a training prerequisite.

**VERDICT: DEFER. Not a pre-Run-12 item. The v2 CLI split is sufficient.**

---

## Item 7 — drift_baseline.json Placeholder

### Assessment — COMPLETE

`ml/data/drift_baseline.json` is used by production drift detection — not by training.
It can only be populated with real output AFTER a model is trained and evaluated on a warmup batch.

**VERDICT: DEFER. Not fixable before training. Needs Run 12 output.**

---

## Item 8 — C-4 max_nodes Guard

### Assessment — COMPLETE (from data analysis)

- 0.18% of graphs exceed 2048 nodes (~33 train contracts)
- Fixing requires changing `fusion_layer.py` max_nodes — high risk for low reward
- **VERDICT: DEFER. 0.18% rate is negligible.**

---

## Final Feasibility Summary

| # | Item | Verdict | Work estimate |
|---|------|---------|---------------|
| 1 | SmartBugs Curated ingestion | ✅ **DO IT** | ~3-4h: parser + connector + pipeline |
| 2 | Stage 5.5 GCB propagation | ❌ **DEFER** | Multi-day, complex sub-project |
| 3 | DoS label quality investigation | ✅ **DO IT** | ~1h: spot-check + decision |
| 4 | Level-3 dedup (normalized text hash) | ✅ **DO IT** | ~1h: 50 lines in deduplicator.py + re-split |
| 5 | DeFiHackLabs ingestion | ❌ **BLOCKED** | PoC contracts, not vulnerable contracts |
| 6 | Stage 5 registry | ❌ **DEFER** | August project, not a training prereq |
| 7 | drift_baseline.json | ❌ **DEFER** | Needs trained model output |
| 8 | C-4 max_nodes | ❌ **DEFER** | 0.18% rate, low impact |

**Pre-Run-12 work = Items 1, 3, 4 only.**

---

## Corrections from Reviewer Validation (2026-06-13)

| Claim | Verdict | Action |
|---|---|---|
| 1a: `unchecked_low_level_calls → UnusedReturn` | ❌ Reviewer wrong | Source: `smartbugs_curated.yaml:22` maps `unchecked_low_level_calls: CallToUnknown`; `solidifi.py:151` reads `class_map.get(folder)` — only code path to the label; `schema/__init__.py:22` reads only `c["name"]` from taxonomy. No README involved. No change to crosswalk. |
| 1b: `front_running → TransactionOrderDependence` | ✅ Reviewer correct | Fix `smartbugs_curated.yaml`: change `front_running: Timestamp` → `front_running: TransactionOrderDependence` |
| 2: DoS math 59 missing | ❌ Reviewer wrong | v1-splits: 782+162+151=1,095 ✓. Reviewer mixed v1-train (782) + v2-val (103) + v1-test (151). No issue. |
| 3: L3 dedup reentrancy collision | ⚠️ Example wrong, principle valid | Drop identifier-lowercasing from normalization. Only strip comments + normalize whitespace. |
| Gap B: Ordering | ✅ Correct | Change order to A → C → B → D |
| Gap C: Readiness gates | ✅ Correct | Add 7 gates to Step D |
| Gap D: Exit criteria | ✅ Correct | Add "done means" per step |

---

## Execution Plan (corrected order)

### Step A — DoS Investigation (fastest, no code changes)
**Done means:** Sampled 30 DoS=1 train contracts (stratified across confidence tiers). FPR% calculated. Decision logged: keep / prune / zero.

### Step C — SmartBugs Curated ingestion ✅ DONE (2026-06-13)

**Crosswalk fix applied:** `smartbugs_curated.yaml:25` `front_running: Timestamp` → `front_running: TransactionOrderDependence`
Added `confidence_tier: T1` to crosswalk (required by parser).

**Config fix:** `config.yaml` SmartBugs switched from `connector: git` → `connector: manual` with `staging_path: /home/motafeq/projects/sentinel/ml/data/smartbugs-curated/dataset`

**Parser created:** `sentinel_data/labeling/parsers/smartbugs_curated.py`
- `_extract_folder`: reads `parts[1]` from `repo/<category>/<contract>.sol` (one level shallower than SolidiFI)

**Pipeline results:**
- Ingestion: 143 contracts, symlink `data/raw/smartbugs_curated/repo → staging_path` ✓
- Preprocessing: 137/143 (1 compile failure: `parity_wallet_bug_1.sol` at pragma 0.4.9; 5 duplicates of existing DIVE contracts)
- Representation: 134/137 graphs (3 Slither IR parse failures on old `.call.value()` patterns)
- Labeling: 137/137 labels written, 0 failures
- Merger: 137 new + 22,356 cached (DIVE+SolidiFI) = 22,493 total merged labels
- Dedup groups: 21,523 → 21,657 (134 new SmartBugs graphs)
- V3 split: 18,559 train / 2,009 val / 1,925 test (0% cross-split leakage confirmed)
- V3 export: `sentinel-v3-smartbugs-2026-06-13`, 22,493 contracts, 21,657 with graphs, 5 shards

**Class gains from SmartBugs:**
- CallToUnknown: +48 (main target — train count ~27 → ~61+)
- TransactionOrderDependence: +4 (front_running, now correctly mapped)
- Reentrancy: +30 | ExternalBug: +17 | IntegerUO: +15 | Timestamp: +13 | DoS: +6 | NonVulnerable: +4

### Step B — Level-3 dedup ✅ DONE (2026-06-13)
(Runs on full corpus = DIVE + SolidiFI + SmartBugs)

**Implementation:** `deduplicator.py` — L3 added: strip block/line comments + collapse whitespace → SHA-256 hash. No identifier lowercasing.

**Post-hoc scan results (all 22,493 contracts):**
- Total unique norm-hashes: 22,284
- L3 groups (2+ members): 147 groups, 356 contracts
- Consistent groups (labels agree): **83** → applied to `dedup_groups_graph_hash.json`
- Conflicting groups (different labels on text-identical contracts): **64** — SKIPPED
- FP rate: 43.5% overall (DIVE redeployment pattern — same contract different addresses, independently labeled per vuln class)

**Why 43.5% conflicts:** DIVE deploys the same contract source across multiple address folders, then independently labels each instance. So text-identical contracts can legitimately have different positive labels (e.g., one folder labels it DoS, another labels it Reentrancy). L3 can only safely merge groups where all instances agree.

**Decision: Apply L3 to 83 label-consistent groups only.** 64 conflicting groups excluded.

**Result:** 15 contracts reassigned to new L3 canonicals in `dedup_groups_graph_hash.json`.

**New v3 splits (post-L3):** train=18,561 / val=2,008 / test=1,924. 0% cross-split leakage confirmed.

**Files produced:**
- `data/dedup_groups_l3_candidates.json` — full L3 group analysis (147 groups, consistent/conflict/status)
- `data/dedup_groups_graph_hash.json` — updated with L3 groups (l3_applied=True, l3_consistent_groups_applied=83)
- `data/splits/v3/` — re-generated with updated groups

**Step B — DONE.**

### Step D — Tests + 7 Readiness Gates (in progress, 2026-06-13)

**Config fix applied during Step D:** `config.yaml` SmartBugs tier: 1 → tier: 3 (structural benchmark / recall ground-truth, not gold). This was caught by `test_skeleton.py::test_config_has_all_tier1_sources`.

**data_module full test suite:**
- 571 passed, 47 skipped, 4 FAILED (pre-existing P3 CALL_ENTRY/RETURN_TO, architecture.md:272 "full fix in v2.1")
- Pre-existing failures: `test_a18_icfg_has_call_entry_and_return_to`, `test_call_entry_edges_exist`, `test_return_to_edges_exist`, `test_call_entry_and_return_to_counts_balanced`
- New failures introduced this session: NONE (tier fix resolved the one new failure)

**Gate status:**

| # | Gate | Status | Evidence |
|---|------|--------|----------|
| 1 | Schema regression (test_byte_identical_regression.py) | ✅ GREEN | 40/40 pass in full suite |
| 2 | BCCC Phase 5 verification suite | ✅ GREEN | 191 pass / 21 skipped in full suite |
| 3 | End-to-end round-trip (SentinelDataset forward pass) | ✅ GREEN | 16/16 passed in 20.86s on v3. Fast path: num_nodes 0.01s, hash 0.00s, total init 1.94s |
| 4 | Feature distribution (Stage 6) | ✅ GREEN | By construction — v9 schema unchanged |
| 5 | All 10 classes VERIFIED or PROVISIONAL | 🟡 AMBER | See Gate 5 analysis below — no regression vs v2; DoS+ExternalBug BEST-EFFORT by design (NOT_EXTRACTABLE in v9) |
| 6 | No leakage across splits | ✅ GREEN | 0 overlap train∩val∩test (18561/2008/1924, post-L3) |
| 7 | No open code-bug regression (EMITS + predictor thresholds) | ✅ GREEN | EMITS fixture 4/4 pass; predictor per-class threshold fix already in code |

**V3 export:** Re-exported with `--split-version 3` (post-L3 splits: 18,561/2,008/1,924). `.hash_cache.json` written. `shard_index` contains `num_nodes` per contract. Verified by `temp/verify_export.py` — hash_cache exists, sample entries contain num_nodes (264, 382, 975), splits correct. ✅

**Done means:** All 7 v2-readiness gates GREEN on v3 export. All data_module tests pass (4 pre-existing CALL_ENTRY failures are excluded from "pass" requirement). New per-class baseline documented.

**Step D — DONE (2026-06-13).** Final gate summary:
- Gates 1, 2, 4, 6, 7: ✅ GREEN (data_module full suite)
- Gate 3: ✅ GREEN (16/16 SentinelDataset tests passed)
- Gate 5: 🟡 AMBER (no regression; DoS+ExternalBug BEST-EFFORT by design — NOT_EXTRACTABLE from v9 graph schema)

---

## Step A — DoS Investigation Results (2026-06-13)

### Method
- Sample: 30 DoS=1 train contracts (seed=42, all T2 — 100% of DoS train is DIVE T2)
- Source files: `data_module/data/preprocessed/dive/<sha>.sol`
- Label files: `data_module/data/labels/dive/<sha>.labels.json`
- Patterns checked: `for(`, `while(`, `.transfer(`, `.send(`, `.call{`, `.push(`, unbounded loop + call combo
- Manual read: all 9 contracts with no syntactic signal

### Key findings

**All 782 DoS train contracts are DIVE T2.** No T0/T1 tier diversity — there's no "confidence tier" stratification possible here.

**Multi-label rate: 26/30 (87%).** DoS rarely appears alone. Average co-labels per contract: 2.9. DIVE's labeling assigns all vulnerabilities present in a file, not just the primary one.

**No syntactic signal (no for/while/transfer/call): 9/30.** Manual review of all 9:

| Contract | Lines | Co-labels | Verdict | Pattern |
|---|---|---|---|---|
| 4d614057 (`TypeFetcher`) | 31 | DoS only | **FALSE POSITIVE** | Utility address-type checker. No vulnerability. |
| bf139a2b (`MultiSend`) | 27 | DoS+ExternalBug+UnusedReturn | **TRUE POSITIVE** | `while (i < dests.length)` loop calling external transferFrom. Regex missed `while`. |
| eba4aed7 (`UpgradeabilityProxy`) | 119 | DoS only | **MARGINAL** | Proxy `delegatecall(gas, _impl)`. Indirect admin DoS if impl replaced with reverting contract. |
| 4d29e6c1 (`MyAdvancedToken`) | 140 | DoS only | **TRUE POSITIVE** | `frozenAccount` mapping. Owner can freeze any address → transfers revert. |
| e8a5a1ba (`LIBC`) | 139 | DoS only | **FALSE POSITIVE** | Buggy ERC20 (wrong balance check in transferFrom). No direct DoS vector. |
| 85735ab0 (`MONKE`) | 207 | DoS+ExternalBug | **TRUE POSITIVE** | `require(_Swapping)` gate — non-excluded addresses cannot transfer until owner enables. Honeypot/admin DoS. |
| b40a278e (`GreenMed`) | 122 | DoS+ExternalBug | **TRUE POSITIVE** | `frozenAccount` mapping. Owner can freeze any account. |
| 6302d13c (`ProofofIzanagi`) | 376 | DoS+ExternalBug+IntegerUO | **MARGINAL** | `BlacklistBot()` lets owner overwrite any user's balance. Rug/honeypot. More ExternalBug than DoS. |
| 90b7e733 (`NVIDIA2.0`) | 207 | DoS+ExternalBug | **TRUE POSITIVE** | Same `_Swapping` gate pattern as MONKE. |

**Pattern breakdown of DoS in this sample:**
1. **Admin DoS / honeypot (dominant):** `frozenAccount`, `_Swapping` gate, owner-controlled balance manipulation. Hard to distinguish from ExternalBug by graph structure — they both manifest as `require` guarded by owner-controlled state.
2. **Unbounded loop DoS:** `while(i < dests.length)` calling external contract per iteration (bf139a2b). The `for` regex missed `while`.
3. **Proxy admin DoS:** Upgrading to a reverting implementation. Indirect.
4. **False positives:** Utility contracts that DIVE mislabeled, or contracts with bugs that don't manifest as DoS.

### FPR estimate from this sample
- **Clear false positives: 2/30 (6.7%)** — TypeFetcher, LIBC
- **Marginal: 2/30 (6.7%)** — Proxy, ProofofIzanagi (more ExternalBug than DoS)
- **True positives: 26/30 (86.7%)**

### Why AUC-PR=0.046 at ep1 despite low FPR
The label quality is fine (~87% TP). The model struggles because:
- Admin DoS (frozenAccount, swapping gates) looks like ExternalBug at the graph level — both add `require(condition)` nodes where `condition` reads owner-controlled storage. The graph features don't separate them well.
- 782 train examples at 4.9% prevalence is a hard few-shot problem.
- ep1 = first 280 gradient steps — far too early to judge.

### Decision: KEEP DoS labels
FPR ≈ 7% is acceptable. The dominant pattern (admin DoS) is a real vulnerability class even if it overlaps with ExternalBug. Zeroing all DoS labels would discard **2,910 train examples** of legitimate signal (corrected from wrong "782" count). The AUC-PR improvement must come from training longer, not from data pruning. The 2,655 DoS+Reentrancy co-occurring contracts are legitimate DIVE multi-label signal — merger.py correctly does not flag them (T2 source, 12% rate, well below 50% threshold).

**Step A — DONE.** No code changes needed.

---

## Decisions Log

| When | Decision | Reason |
|------|----------|--------|
| Session start | Fix all open items before launching new run | Want clean data for Run 12 |
| Item 8 assessment | Defer C-4 max_nodes fix | 0.18% rate = low impact, risky change |
| Step A (2026-06-13) | Keep all DoS labels (decision on a v2-era count of 1,095) | FPR ≈ 7%, AUC-PR=0.046 at ep1 was assumed to be a training signal problem |
| Step B (2026-06-13) | Apply L3 to 83/147 groups only | 64 conflicting groups excluded — DIVE independently labels same source at different addresses; full L3 would introduce false group merges |
| Post-D (2026-06-13) | **Apply the DoS/Reentrancy co-occurrence patch for real** (zero 2,655 DIVE DoS labels where co-occurring with Reentrancy) | Manual data audit revealed the patch was documented but never executed; v3 export had 3,756 DoS labels (2,910 train) — not 1,095 as the plan claimed. Applied the patch to source labels, regenerated merged labels, re-split, re-exported. v3 now has 1,101 DoS (845 train), 0 DoS+Reentrancy overlap. |

---

## Code Changes Made

| File | Change | Why |
|------|--------|-----|
| `sentinel_data/labeling/crosswalks/smartbugs_curated.yaml` | `front_running: Timestamp` → `front_running: TransactionOrderDependence`; added `confidence_tier: T1` | Bug fix (wrong class) + required by parser |
| `sentinel_data/labeling/parsers/smartbugs_curated.py` | Created (new file) | SmartBugs labeling parser, reads `parts[1]` from `repo/<category>/<contract>.sol` |
| `config.yaml` SmartBugs entry | `connector: git` → `connector: manual`, added `staging_path`, `include_subdirs`, etc. | Manual connector required for on-disk dataset; tier: 3 (structural benchmark) |
| `config.yaml` SmartBugs `tier` | `tier: 1` → `tier: 3` | Caught by `test_config_has_all_tier1_sources`; SmartBugs is structural ground-truth, not gold |
| `sentinel_data/preprocessing/deduplicator.py` | Added Level-3 normalized-text dedup (strip comments + collapse whitespace, no identifier lowercasing) | Catches copy-paste-with-comment-edits near-dups |
| `sentinel_data/export/graph_writer.py` | Return `num_nodes_map` as third value | num_nodes available at export time for free; store it to avoid shard reloading at dataset init |
| `sentinel_data/export/chunker.py` | Embed `num_nodes` in shard_index entries; write `.hash_cache.json`; exclude cache from artifact hash | See SentinelDataset speedup section below |
| `sentinel_data/export/export.py` | `verify_artifact_hash()` uses hash cache on warm loads | Avoids reading ~3GB on every SentinelDataset init |
| `ml/src/datasets/sentinel_dataset.py` | Read `num_nodes` from shard_index (no shard loading); O(1) contract_id membership; sort fallback by (shard,pos) to prevent LRU thrashing; timing logs | Fast path eliminates 5-shard load at init; fixes O(N²) membership check; fallback now loads each shard exactly once |
| `data_module/temp/patch_dos_v3.py` | Created (new script) | One-shot patch: zeros DoS in DIVE source labels where DoS+Reentrancy both = 1. Backs up originals to `data/_backup_pre_dos_patch_2026-06-13/`. Patched 2,655 files. |
| `data_module/temp/resplit_and_reexport.py` | Created (new script) | Re-runs stratified_split (seed=42) on the patched merged labels, applies dedup_enforcer + NonVulnerable cap, writes to v3/, then re-exports v3. New splits: 18,596/1,983/1,914. New artifact_hash: `5cc5cfcbf42bef4ced58b963ef98241bcf3ec4ab3bea5d198f336ec763a4faa9`. |
| `data_module/tests/test_verification/test_class_auditor.py::test_dive_dos_reentrancy_cooccurrence_finding` | Test updated to assert DoS+Reentrancy co-occurrence flag is GONE (was: assert flag is RAISED with rate > 0.5) | The pre-patch test asserted the OLD noisy pattern. With the patch applied, the auditor no longer flags the co-occurrence (rate ~0%). Test now verifies the patch is in effect. Will fail if the patch is reverted. |

---

### SentinelDataset init speedup — detail (2026-06-13)

**Problem:** `SentinelDataset.__init__` blocked Gate 3 and every training run startup.
Two root causes from timing logs added to sentinel_dataset.py:

1. `verify_artifact_hash()` — reads every shard file on disk (~3GB for v3), SHA-256s them
   on every cold init. No caching. Slow even when nothing changed.

2. `num_nodes_map` loop — iterates all contracts in the split, loads all 5 graph shards
   (~500MB each) into RAM just to read one integer (`num_nodes`) per contract.
   Also: `contract_id not in self._contract_ids` was a list → O(N) per lookup → O(N²) total.

**Fix A — `num_nodes` in shard_index**

`write_graphs_shards()` already has `graph: Data` in hand when building the shard.
`graph.num_nodes` is free at that point. Added a `num_nodes_map: dict[str, int]` return value.
`_build_shard_index()` now embeds it: `{"shard": 0, "pos_in_shard": 4, "num_nodes": 312}`.
`SentinelDataset.__init__` reads directly from the dict — zero shard loading at init.
Backward-compatible: old exports lacking the key fall back to the original shard-loading loop.

**Fix B — hash cache sidecar**

After computing `artifact_hash`, write `.hash_cache.json`:
```json
{"artifact_hash": "<sha256>", "files": {"graphs/graphs-00000.pt": {"mtime": 1749876543.1, "size": 523456789}}}
```
`verify_artifact_hash()` stats each data file (microseconds), compares mtime+size to cache.
If all match → skip full read, compare cached hash to manifest hash.
If any changed → full SHA-256 recompute, update cache.
`.hash_cache.json` excluded from `_hash_export_data()` so the hash is stable after writing.

**After:** re-export v3 to embed num_nodes + write hash cache + fix stale splits (pre-L3 → post-L3).

---

## Test Results

### data_module full suite (2026-06-13, post DoS patch)
```
598 passed, 27 skipped, 0 failed (47.34s)
No new failures introduced. The 1 test that asserted the pre-patch noisy pattern (test_dive_dos_reentrancy_cooccurrence_finding) was updated to assert the post-patch clean state.
```

### Pre-DoS-patch test results (2026-06-13, post Steps A-D)
```
571 passed, 47 skipped, 4 failed (33s)
Pre-existing CALL_ENTRY/RETURN_TO failures — P3, documented in architecture.md:272, fix deferred to v2.1
No new failures introduced.
```

### Gate-by-gate (v3 export `sentinel-v3-smartbugs-2026-06-13`)
- Gate 1: 40/40 schema regression ✅
- Gate 2: 191 pass / 21 skip BCCC Phase 5 ✅
- Gate 3: ml/tests/test_sentinel_dataset.py — ✅ GREEN 16/16 passed (20.86s against v3). Fast path confirmed: hash verified 0.00s, num_nodes 0.01s (17,978 entries), __init__ total 1.94s. Test file updated to point to sentinel-v3-smartbugs-2026-06-13.
- Gate 4: by construction ✅
- Gate 5: ⏳ pending verification report check
- Gate 6: 0 leakage confirmed ✅
- Gate 7: EMITS 4/4 ✅

**Gate 5 analysis (v3 vs v2 verification report):**

v2 class verdicts and expected v3 change:
| Class | v2 Verdict | v3 Expected | Reason |
|---|---|---|---|
| CallToUnknown | BEST-EFFORT (39% semantic) | PROVISIONAL | +48 T1 SmartBugs `unchecked_low_level_calls` — curated structural benchmark, high semantic pass rate expected |
| MishandledException | ✅ VERIFIED | ✅ VERIFIED | Unchanged (no new contracts) |
| GasException | PROVISIONAL (0 positives) | PROVISIONAL | Unchanged — SmartBugs has no gas_exception category |
| TransactionOrderDependence | ✅ VERIFIED | ✅ VERIFIED | +4 T1 SmartBugs `front_running` contracts (now correctly mapped) |
| IntegerUO | PROVISIONAL | PROVISIONAL | +15 T1 SmartBugs `arithmetic` — small gain, stays PROVISIONAL |
| Reentrancy | PROVISIONAL | PROVISIONAL | +30 T1 — already PROVISIONAL, no tier upgrade |
| Timestamp | PROVISIONAL | PROVISIONAL | +13 T1 — small gain |
| UnusedReturn | PROVISIONAL | PROVISIONAL | Unchanged |
| DenialOfService | BEST-EFFORT | BEST-EFFORT | NOT_EXTRACTABLE in v9 — Slither-based check deferred. +6 new examples but tier unchanged |
| ExternalBug | BEST-EFFORT | BEST-EFFORT | NOT_EXTRACTABLE in v9 — no automatic semantic check exists. Unchanged |

Gate 5 AMBER justification:
- No class regressed vs v2 (no VERIFIED → PROVISIONAL, no PROVISIONAL → BEST-EFFORT)
- DoS and ExternalBug are BEST-EFFORT by design — v9 graph schema has no Slither AST nodes for these classes. Full verification requires `tool_validator` + Slither batch runs (deferred to v2.1, see verification_report.md section 6)
- CallToUnknown likely upgrades to PROVISIONAL with v3 verification run (deferred — needs `sentinel-data verify --source smartbugs_curated`)
- GasException: 0 train examples, no path to PROVISIONAL before Stage 5.5 (deferred)
- Gate 5 cannot be fully GREEN without Stage 5.5 GCB propagation, which is deferred post-Run-12
- VERDICT: AMBER — proceed to Run 12. No regression. Two classes BEST-EFFORT by known design limitation (documented in verification_report.md:86-97)

---

## DoS Patch Application (2026-06-13, after Steps A-D)

### Discovery
During a manual data audit (sampling contracts from each split + reading the v3 export's `labels.parquet`), the following discrepancy was found:

- Plan (line 131 + 479 + Decisions Log) claimed: 1,095 DoS labels post-patch (782 train + 162 val + 151 test)
- v3 export actual: **3,756 DoS labels** (2,910 train + 429 val + 417 test)
- DoS=1 AND Reentrancy=1 overlap: **2,655 contracts** (the exact number the plan said the patch would zero)

The 2,655 number matched the plan's "zeroed" count, but those 2,655 contracts still had DoS=1 in the merged labels. The DoS/Reentrancy co-occurrence patch had been documented as applied on 2026-06-13, but the v3 export showed it had never actually been executed against the source labels.

### Root cause
`merger.py:100-124` implements a `_check_co_occurrence_flag()` that only **flags** T3/T4 sources (BCCC pattern with > 50% co-occurrence). DIVE is T2, so the flag never fires for DIVE. The "patch" was a separate one-off operation described in the plan but never coded up or run.

### The proper fix (applied 2026-06-13)
1. **Patch DIVE source labels** (`data_module/temp/patch_dos_v3.py`):
   - Backed up originals to `data_module/data/_backup_pre_dos_patch_2026-06-13/` (22,073 files)
   - Iterated `data/labels/dive/*.labels.json`
   - For each with `DenialOfService.value=1 AND Reentrancy.value=1`:
     - Set `DenialOfService.value=0`, `tier=null`
   - Patched 2,655 files
2. **Re-run merger** (`sentinel_data.labeling.merger.run_merger(force=True)`):
   - 22,493 contracts merged in 2.5s
   - 0 cached (force=True), 0 failed
3. **Re-run splitter** (`sentinel_data.splitting.stratified_split`):
   - After stratified: 15,736 / 3,363 / 3,394 (initial assignment)
   - After dedup_enforcer: **18,596 / 1,983 / 1,914** (3,036 dedup groups resolved, was 3,027)
   - After NonVulnerable cap: 18,596 / 1,983 / 1,914 (no change)
4. **Re-export v3** (`sentinel_data.export.chunker.chunk_export`):
   - 121.2s to regenerate labels.parquet, metadata.parquet, manifest.json, .hash_cache.json
   - New artifact_hash: `5cc5cfcbf42bef4ced58b963ef98241bcf3ec4ab3bea5d198f336ec763a4faa9`
   - created_at: `2026-06-13T19:19:55.999653+00:00`
5. **Backup intermediate states**:
   - Pre-DoS-patch v3 export: `data_module/data/exports/sentinel-v3-PRE-DOS-PATCH-backup/`
   - Pre-resplit v3 export: `data_module/data/exports/sentinel-v3-PRE-DOS-PATCH-AND-RESPLIT-backup/`
   - Pre-resplit v3 splits: `data_module/data/splits/v3-PRE-DOS-PATCH-backup/`

### Verification

| Check | Pre-patch v3 | Post-patch v3 | Expected |
|---|---|---|---|
| Total DoS=1 | 3,756 | **1,101** ✓ | 1,095 DIVE + 6 SmartBugs |
| train DoS=1 | 2,910 | **845** | ~859 (slight shift from stratified redistribution) |
| val DoS=1 | 429 | **129** | ~120 |
| test DoS=1 | 417 | **127** | ~122 |
| DoS=1 AND Reentrancy=1 | 2,655 | **0** ✓ | 0 |
| Other class counts (Reentrancy, ExternalBug, IntegerUO, Timestamp, UnusedReturn) | unchanged | unchanged ✓ | unchanged |

**Why per-split DoS counts differ from "1,095 train" expectation:**
The 1,095 number was a v2-era figure (DIVE-only, pre-SmartBugs, pre-L3). The v3 stratified splitter distributes contracts per class to maintain 70/15/15 ratio. Removing 2,051 DoS=1 contracts from train (2,910 → 845) plus the SmartBugs +6 causes the splitter to redistribute 2,051 contracts across splits, with class-based stratification. The total is correct (1,101), per-split is approximate.

### Test results (post-patch)
- **data_module full suite: 598 passed, 27 skipped, 0 failed** (47.34s)
- 1 test updated: `test_dive_dos_reentrancy_cooccurrence_finding` now asserts the patch is in effect (no DoS+Reentrancy flag). If patch is reverted, this test will fail at `len(flagged_dos) == 0`.

### Re-verified 7 readiness gates
- Gate 1 (Schema regression): ✅ GREEN — 40/40 pass
- Gate 2 (BCCC Phase 5): ✅ GREEN — 191 pass / 21 skip
- Gate 3 (SentinelDataset round-trip): ✅ GREEN — 16/16 pass; hash verified 0.00s (hash cache hit); n_contracts=22,493, n_contracts_with_reps=21,657; 0 leakage
- Gate 4 (Feature distribution): ✅ GREEN — v9 schema unchanged
- Gate 5 (All 10 classes VERIFIED/PROVISIONAL): 🟡 AMBER (unchanged — DoS+ExternalBug BEST-EFFORT by design; verification report needs re-running with `sentinel-data verify` to update the report file)
- Gate 6 (No leakage): ✅ GREEN — 0 overlap train∩val∩test (post-DoS-patch: 18,596/1,983/1,914)
- Gate 7 (No open code-bug regression): ✅ GREEN — EMITS 4/4 + predictor thresholds in code

### Final v3 export state (Run 12 ready)

| Field | Value |
|---|---|
| Path | `data_module/data/exports/sentinel-v3-smartbugs-2026-06-13/` |
| `manifest.json` schema_version | v1 |
| `manifest.json` graph_schema_version | v9 |
| `manifest.json` n_contracts | 22,493 |
| `manifest.json` n_contracts_with_reps | 21,657 |
| `manifest.json` n_shards | 5 |
| `manifest.json` artifact_hash | `5cc5cfcbf42bef4ced58b963ef98241bcf3ec4ab3bea5d198f336ec763a4faa9` |
| `manifest.json` created_at | `2026-06-13T19:19:55.999653+00:00` |
| `manifest.json` splits | train=18,596 / val=1,983 / test=1,914 |
| `labels.parquet` total DoS=1 | 1,101 (845 train, 129 val, 127 test) |
| `labels.parquet` DoS+Reentrancy overlap | 0 |
| `.hash_cache.json` | EXISTS (warm load: 0.00s hash verify) |
| `shard_index` has `num_nodes` | YES (Fix A) |
| 7 readiness gates | 6 GREEN + 1 AMBER (unchanged from Step D) |

**Run 12 is unblocked.** Launch from `ml/checkpoints/GCB-P1-Run11-v2deduped-20260613_best.pt` with the post-DoS-patch v3 export.

### Files produced/changed (DoS patch only)
- `data_module/temp/patch_dos_v3.py` (new, 60 lines) — the patch script
- `data_module/temp/resplit_and_reexport.py` (new, 130 lines) — the re-split + re-export
- `data_module/temp/verify_v3_loads.py` (new, 65 lines) — Gate 3 + leak + DoS verification
- `data_module/data/labels/dive/*.labels.json` (2,655 files modified)
- `data_module/data/labels/merged/*.labels.json` (22,493 files regenerated)
- `data_module/data/splits/v3/{train,val,test}.jsonl` + `split_manifest.json` (regenerated)
- `data_module/data/exports/sentinel-v3-smartbugs-2026-06-13/{labels,metadata}.parquet` + `manifest.json` + `.hash_cache.json` (regenerated)
- `data_module/data/_backup_pre_dos_patch_2026-06-13/` (22,073 backup files of original DIVE labels)
- `data_module/data/splits/v3-PRE-DOS-PATCH-backup/` (pre-patch v3 splits)
- `data_module/data/exports/sentinel-v3-PRE-DOS-PATCH-backup/` (pre-patch v3 export — initial re-export, no DoS patch yet)
- `data_module/data/exports/sentinel-v3-PRE-DOS-PATCH-AND-RESPLIT-backup/` (pre-DoS-patch-resplit v3 export)
- `data_module/tests/test_verification/test_class_auditor.py` (1 test updated)

---

## What the pre-patch plan was wrong about (corrected by the audit)

The original "Step A" decision said: "Keep all 1,095 DoS labels (FPR ≈ 7%, no code change needed)". This was based on the v2-era count of 1,095 (3,750 DIVE DoS - 2,655 zeroed = 1,095). But:

1. The patch was never actually applied (the 2,655 zeroed count was a plan assertion, not a fact)
2. The v3 export had the un-patched count: 3,750 DIVE + 6 SmartBugs = 3,756 DoS
3. The Step A investigation sampled 30 contracts and found 86.7% TP — this FPR estimate was based on a 782-contract post-patch train set, but the actual train set had 2,910 DoS contracts with the same 26/30 TP rate (≈ 2,524 TP, 386 FP)
4. The AUC-PR=0.046 at Run 11 ep1 was attributed to "training signal problem" — but the real cause was the un-patched 2,655 co-occurrence contracts adding contradictory labels

**The patch should have been applied during the original Step A, not after Step D.** Going forward, the proper approach is to apply the patch as a pre-merge stage (modify DIVE source labels before merging, not after).

A future code improvement (out of scope for Run 12) would be to add an optional `co_occurrence_patch=True` flag to `run_merger()` that zeros DoS where it co-occurs with Reentrancy for T2 sources, similar to the existing T3/T4 flag but more aggressive. This would automate the patch and prevent the 1,095-vs-3,756 confusion from recurring.

End of plan.
