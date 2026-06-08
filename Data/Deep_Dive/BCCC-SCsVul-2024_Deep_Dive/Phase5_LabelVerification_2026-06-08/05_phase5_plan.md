# BCCC-SCsVul-2024 Deep Dive — Phase 5 Plan

**Title:** Phase 5 — Comprehensive BCCC Label Verification
**Date:** 2026-06-08 (rev 2026-06-08 v1.1)
**Author:** SENTINEL Data Engineering
**Source dataset:** `BCCC-SCsVul-2024/` at repo root `~/projects/sentinel/BCCC-SCsVul-2024/` (1.6 GB, **read-only**)
**Previous deliverable (input):** [`../Phase4_LabelValidation_2026-06-07/outputs/ws_p4_s01b_d12_applied.csv`](../Phase4_LabelValidation_2026-06-07/outputs/ws_p4_s01b_d12_applied.csv) (67,311 contracts, v1.1+12, 24 cols, **no source_code column** — read source from disk via `bccc_file_path`)
**Previous critical findings:** [`../Phase4_LabelValidation_2026-06-07/CRITICAL_FINDINGS.md`](../Phase4_LabelValidation_2026-06-07/CRITICAL_FINDINGS.md)
**Status:** ⏳ **Design complete — execution pending**

---

## 1. Why Phase 5 Exists — The Pivot

Phase 4 Stage 1 applied static analysis (slither + aderyn) to 10,693 contracts (15% stratified sample) and measured agreement with BCCC labels. **Median F1 = 0.000 across 9 classes. All methods FAILED the 0.5 gate.**

Further investigation revealed the root cause was **not insufficient sample size** but **structural label noise** in the BCCC dataset:

| Class | FP Rate | Root Cause | Noise Type |
|---|---|---|---|
| Reentrancy | **89.4%** | BCCC definition: any external call + state change. Strict: `.call.value()` (pre-0.8) or `.call{value:}` (post-0.8) only | Broad definition |
| CallToUnknown | **91%** | Contracts with zero external calls labeled as calling unknown addresses | Mislabeling |
| ExternalBug | **100%** (tiny sample) | Simple library with no vulnerability labeled as ExternalBug | Unclear BCCC definition |
| GasException | **67%** | Definition mismatch — BCCC vs. actual costly-loop patterns | Broad definition |
| DenialOfService | **56%** | Similar definition mismatch | Broad definition |
| Timestamp | **50%** | Moderate noise — "uses timestamp" ≠ "timestamp-dependent critical logic" | Broad definition |
| IntegerUO | **0%** | Clean — confirmed by manual review + 2 tools | ✅ |
| UnusedReturn | **0%** | Clean — confirmed by manual review + 2 tools | ✅ |
| MishandledException | **0%** | Clean — confirmed by manual review + 2 tools | ✅ |

The original Phase 4 plan (Stages 2-7) assumed label noise was **sample-size limited** — more data would fix it. The critical findings proved it's **structural** — more data won't fix it.

**Phase 5 abandons that assumption.** Its sole mission: systematically verify every BCCC label using multiple methods, define ground truth per class, and produce a verified dataset before any AutoML or model training.

### Post-Cleanup Size Warning

Reentrancy (17,698 contracts) → ~1,875 after 89% noise removal. CallToUnknown (11,131) → ~1,000 after 91% noise removal. These two classes alone are 43% of all labeled contracts. After Phase 5 the training set for noisy classes shrinks ~10×. Phase 5's value is a cleaner dataset, not a larger one. AutoML and SENTINEL Run 10+ must be designed with this in mind.

### Contracts Whose ALL Labels Are Dropped

When every positive label for a contract is rejected by Phase 5 verification, that contract must be reclassified: **set `Class12:NonVulnerable=1`** and `n_pos=0`. These contracts are not removed from the dataset — they become additional NonVulnerable training examples. This rule is applied in Stage 5.6.

---

## 2. Scope Change — Everything Else Parked

All uncompleted work from Phases 1-4 is **postponed to after Phase 5**. Phase 5 is exclusively label verification.

### Parked (resume after Phase 5)

| Origin | Item | Reason |
|---|---|---|
| Phase 3 WS-P | Slither-Based Graph-Level Features | Needs verified labels first |
| Phase 3 WS-Q | SHAP Feature Importance | Depends on AutoML (WS-L), which needs labels |
| Phase 3 WS-R | 3-Way Model Comparison | Depends on WS-L + WS-O |
| Phase 3 WS-T | Multi-Label Structure Test | Depends on AutoML |
| Phase 3 WS-K2 | Slither-derived features (32 cols) | Depends on verified labels |
| Phase 4 Stage 2 | Escalation to 30% sample | Structural noise, not sample-size limited |
| Phase 4 Stage 3 | Escalation to 50% sample | Same reason |
| Phase 4 Stage 4 | Mythril tiebreaker (50 contracts) | Will fold into Phase 5.3 if needed |
| Phase 4 Stage 5 | Per-folder manual investigation | Will fold into Phase 5.4 |
| Phase 4 Stage 6 | AutoML (10 binary × 5 models × 50 trials × 5 folds) | Requires verified labels |
| Phase 4 Stage 7 | Synthesis — v1.3 + CHANGELOG | Will be produced by Phase 5.6 |
| D-P4-1 | Apply D-I-11 broadly or narrowly? | Deferred pending verification |
| D-P4-3 | Mythril: 50 or 100 contracts? | Deferred |
| D-P4-4 | AutoML: 50×5 or 25×5? | Deferred |
| D-P4-5 | Include LogReg in AutoML? | Deferred |
| D-P4-6 | Apply 3 hand-crafted features to v1.3? | Deferred |

---

## 3. Phase 5 Design Principles

1. **Multi-method per class** — No single verification method is sufficient. Each class gets the right combination of: static analysis, regex patterns, symbolic execution, structural analysis, manual review, and ML-assisted propagation.
2. **Incremental stages with gates** — Start cheap (existing evidence, fast automated checks), escalate only when needed. Each stage has a clear gate: "is this class verified enough to proceed?"
3. **Per-class verification, not dataset-wide** — Different classes have different noise profiles. Each class is verified independently and can pass its gate at different stages.
4. **Confidence scores, not binary verdicts** — Final output assigns each (contract, class) pair a confidence score (0.0–1.0), not just a corrected binary label. Weights defined in Stage 5.0.
5. **Traceable provenance** — Every label decision is traceable to the specific method(s) that produced it.
6. **Reuse all existing evidence** — 500 manually reviewed Reentrancy contracts, 43 + 199 manual reviews, slither + aderyn results on 10,693 contracts, 34 regex features on 67,311 — all counted before adding new work.

---

## 4. Verification Methods — Ordered by Cost

| # | Method | Speed | Coverage | Cost |
|---|---|---|---|---|
| **M1** | Existing evidence integration | Seconds | 10,693 contracts (15%) | None — data already exists |
| **M2** | Source-code regex patterns | ~2s/contract | 67,311 contracts (100%) | Low — 34 patterns already computed |
| **M3** | Static analysis (slither) | ~0.16s/contract | 10,693 contracts (15%) | Low — already done |
| **M4** | Static analysis (aderyn) | ~0.06s/contract | 10,693 contracts (15%) | Low — already done |
| **M5** | Targeted slither (remaining contracts) | ~0.16s/contract | ~56,618 contracts (85%) | Medium — ~2.5h; class-specific detectors only |
| **M6** | Targeted aderyn (remaining contracts) | ~0.06s/contract | ~56,618 contracts (85%) | Low — ~1h; parallel |
| **M7** | Structural analysis (surya, solc version) | ~1s/contract | Sampled subset | Medium |
| **M8** | Symbolic execution (mythril, Docker) | ~3min/contract | Top ~50 ambiguous only | High — use 5min timeout |
| **M9** | Manual review (human reads source) | ~1–2h/contract | ~200 sampled contracts | High — gold standard |
| **M10** | GraphCodeBERT embedding + clustering | ~0.5s/contract | 67,311 contracts (100%) | Medium — **requires GPU; cannot run concurrently with active SENTINEL training (VRAM conflict on RTX 3070 8GB)** |
| **M11** | Label propagation (from verified anchors) | ~1min | All contracts | Medium |
| **M12** | Consensus (2-of-3, 3-of-3 tool agreement) | Computational | Varies | Low |

**Note on M10:** Use **GraphCodeBERT** (`microsoft/graphcodebert-base`, already in `ml/` environment, `TRANSFORMERS_OFFLINE=1`) as default. Fall back to CodeBERT only if GraphCodeBERT fails. Do NOT start Stage 5.5 while SENTINEL training is running — confirm `ps aux | grep train.py` shows no active training job first.

---

## 5. Stage Plan — 6 Incremental Stages

### Stage 5.0: Ground Truth Definitions + Confidence Weight Table
**Type:** Design (no code execution)
**Goal:** Define what each vulnerability class actually means before verifying it; define how confidence scores are computed in Stage 5.6.

**For each of the 9 SENTINEL classes, produce a definition document covering:**
1. **Canonical vulnerability definition** — What constitutes a true positive? Cite sources (DASP-10, SWC registry, OWASP).
2. **Inclusion criteria** — Specific code patterns that make a contract vulnerable.
3. **Exclusion criteria** — Patterns that explicitly do NOT count (e.g., `.transfer()` is NOT reentrancy; `block.timestamp` in event logs is NOT a timestamp vulnerability).
4. **Pre-0.8 vs. post-0.8 distinctions** — BCCC is 92% pre-0.6; many patterns differ by compiler version.
5. **Edge cases** — Ambiguous patterns requiring manual judgment.
6. **Verification methods** — Which of M1–M12 apply to this class.
7. **Gate criteria** — How to know this class is "verified enough."

**Class-specific inclusion/exclusion notes to address:**
- **Reentrancy:** Include `.call.value(amount)()` (pre-0.8) and `.call{value: amount}(addr)` (post-0.8) with state-change before call. Exclude `.transfer()` (reverts on failure), `.send()` (limited gas), read-only calls.
- **CallToUnknown:** Include `.call()`, `.delegatecall()`, `.staticcall()` to externally-controlled addresses. Exclude library calls to known addresses, `address(this).call()`.
- **Timestamp:** Include `block.timestamp`/`now` controlling randomness, game outcomes, access control, or withdrawal amounts. Exclude `block.timestamp` used only for logging or informational displays.
- **ExternalBug:** Define explicitly — BCCC catch-all. Likely covers `selfdestruct` reachable by non-owner, `tx.origin` for auth, unauthorized `delegatecall`, `ecrecover` without replay protection. Must nail down inclusion criteria in Stage 5.0 before any verification.
- **GasException / DenialOfService:** Define the boundary. GasException = loop with unbounded iteration over external-controlled array. DoS = actions that permanently block a critical function (e.g., push-over-pull pattern blocking refund).

**Confidence weight table (to be finalized in Stage 5.0 and used in Stage 5.6):**

| Method | Weight | Rationale |
|---|---|---|
| M9 Manual review | **1.00** | Gold standard |
| M8 Mythril symbolic | **0.90** | Near-complete path coverage |
| M3 Slither (high-precision detector) | **0.75** | Conservative but reliable |
| M4 Aderyn | **0.65** | Good coverage, some FP |
| M2 Regex (precise pattern match) | **0.60** | Pattern present ≠ vulnerable, but strong signal |
| M11 Label propagation (cluster member) | **0.45** | Depends on cluster purity |
| M1 Existing evidence integration | **varies** | Inherits from underlying M2–M4 results |

Final confidence = weighted average of all methods that produced a verdict. If only one method applies, that method's weight IS the confidence score (not multiplied by 1.0). If no method produces a verdict, confidence = 0.0 (unverified).

**Deliverables:**
- `labels/p5_s0_class_definitions/` — 9 markdown files, one per class
- Confidence weight table (this document, Section 5.0, finalized in session)

**Gate:** All 9 definition files written AND reviewed against at least one external source (DASP-10, SWC, academic paper) before Stage 5.1 begins.

---

### Stage 5.1: Existing Evidence Integration
**Type:** Data assembly (fast, ~1h)
**Goal:** Harvest every piece of evidence already produced across Phases 1–4 into one unified per-contract per-class evidence table. Identify which classes already have sufficient evidence to pass verification without new runs.

**What we already have (free evidence):**
- Slither results on 10,693 contracts: `ws_p4_s1_slither_results.csv`
- Aderyn results on 10,693 contracts: `ws_p4_s1_aderyn_results.csv`
- 3-way agreement table: `ws_p4_s1_3way_agreement.csv`
- 34 regex + hand-crafted features on all 67,311 contracts: `ws_p4_s05_regex_features.csv` + `ws_p4_s06_handcrafted_features.csv`
- Manual review: 43 contracts (`ws_p4_s1_manual_review_50.csv`), 199 contracts (`ws_p4_s1_review_200.csv`)
- **500 Reentrancy audit contracts** (`ws_p4_s1_review_200.csv` extended; see CRITICAL_FINDINGS.md) — 10.6% true reentrancy rate
- 33 Phase 3 manual reviews (`Phase3_DeepAnalysis_2026-06-06/labels/ws_i_disagreement_inspections.md`)

**Steps:**
1. Load all existing evidence into a unified `p5_s1_evidence_table.csv` (67,311 rows × N evidence columns, indexed by `id`)
2. For each (contract, class) pair compute: BCCC label, slither verdict, aderyn verdict, regex features, manual verdict (if any)
3. Flag contracts with sufficient evidence for an immediate label decision (e.g., manual=DROP → confidence ≥ 0.9)
4. Per-class summary: % contracts with ≥ 1 verdict, % with high-confidence verdict, % with conflicting verdicts

**Important:** The v1.1+12 CSV has no `source_code` column. Source is read from disk:
`~/projects/sentinel/BCCC-SCsVul-2024/SourceCodes/<folder>/<file>.sol` via `bccc_file_path` column.
Any Stage 5.1 script that reads source code must construct this path explicitly.

**Output:** `outputs/p5_s1_evidence_table.csv` (67,311 × N evidence columns)
**Gate:**
- Produce per-class evidence coverage report
- Classes with ≥ 80% of contracts having a high-confidence verdict (confidence ≥ 0.75 from M1–M4) → **VERIFIED at Stage 5.1, skip Stages 5.2–5.4 for this class**
- Expected: IntegerUO, UnusedReturn, MishandledException pass here (0% noise, confirmed by 2+ tools)

---

### Stage 5.2: Bulk Automated Verification
**Type:** Fast automated (2–5h total, split by class)
**Goal:** For each class not verified at Stage 5.1, run the best automated method(s) on all contracts in that class. Run slither + aderyn on the remaining ~56,618 contracts not covered by the Phase 4 sample.

**Per-class verification plan:**

| Class | Primary Method | Secondary | Regex Pattern(s) | Est. Time | Expected Gate Outcome |
|---|---|---|---|---|---|
| **Reentrancy** | Regex: `.call\.value\s*\(` (pre-0.8) OR `.call\s*\{[^}]*value\s*:` (post-0.8) | Slither `reentrancy-eth` | f10_callvalue (existing col) | 2 min (regex) | 80–95% agreement → provisional |
| **CallToUnknown** | Regex: `.call()`, `.delegatecall()`, `.staticcall()` to non-fixed addr | Slither `controlled-delegatecall` | f09_delegatecall, f13_lowlevel_call | 1 min (regex) | 80–95% agreement → provisional |
| **IntegerUO** | Skip — **already verified (0% noise)** | — | — | 0 | **VERIFIED ✓** |
| **UnusedReturn** | Skip — **already verified (0% noise)** | — | — | 0 | **VERIFIED ✓** |
| **MishandledException** | Skip — **already verified (0% noise)** | — | — | 0 | **VERIFIED ✓** |
| **Timestamp** | Regex: `block\.timestamp` / `\bnow\b` in branch/assignment controlling critical logic | Slither `timestamp` detector | f06_block_timestamp, f07_now | 1 min (regex) | 50–80% agreement → **likely Stage 5.3** |
| **ExternalBug** | Regex: `selfdestruct`, `tx\.origin`, `delegatecall` to non-fixed | Slither `suicidal`, `tx-origin` | f08_tx_origin, f09_delegatecall, f17_selfdestruct | 1 min (regex) | **likely < 80% → Stage 5.3** (100% FP in sample; definition unclear until Stage 5.0) |
| **GasException** | Slither `costly-loop` (all remaining contracts) | No aderyn detector for this class | f19_unchecked_block proxy | ~25 min (slither, class-specific) | **Expect < 80% → Stage 5.3** (67% FP, no aderyn coverage) |
| **DenialOfService** | Slither `calls-loop`, `reentrancy-unlimited-gas` (all remaining) | No aderyn detector for this class | None | ~35 min (slither, class-specific) | **Expect < 80% → Stage 5.3** (56% FP, no aderyn coverage) |
| **NonVulnerable** | Cross-reference: if any verified vuln label → NV must be 0 | D-I-11 + D-I-12 already applied | — | Computational | Already corrected |

**Reentrancy regex note — MUST cover both Solidity eras (BCCC is 92% pre-0.6):**
- Pre-0.8: `\.call\.value\s*\([^)]*\)\s*\(` (e.g., `addr.call.value(amount)()`)
- Post-0.8: `\.call\s*\{[^}]*value\s*:` (e.g., `.call{value: amount}(data)`)
Using only one pattern silently misses the other era entirely.

**Slither run for remaining contracts:**
- Version-grouped processing (same approach as Phase 4 Slither V2)
- Pinned solc binary via `Slither(path, solc=solc_path)` — NOT `solc-select use` (global switch breaks parallelism)
- Class-specific detectors only (not all 101) to minimize run time
- 30s timeout per contract; parallel workers

**Output:** `outputs/p5_s2_automated_verdict.csv` (per-contract per-class automated verdict + confidence)

**Gate per class:**
- Agreement ≥ 95% with BCCC → class **VERIFIED**, skip Stages 5.3–5.4
- Agreement 80–95% → class **PROVISIONALLY VERIFIED**, flag edge-case contracts for Stage 5.3 only
- Agreement < 80% → class **UNVERIFIED**, all contracts proceed to Stage 5.3
- **Pre-expected failures at Stage 5.2:** GasException, DenialOfService, ExternalBug (budget Stage 5.3 time for these three)

---

### Stage 5.3: Discrepancy Resolution
**Type:** Targeted analysis (medium cost, 4–10h)
**Goal:** Resolve contracts where Stage 5.2 automated verdict disagrees with BCCC label. GasException, DenialOfService, and ExternalBug are expected to land here entirely.

**Tiered approach per disputed contract (cheapest first):**

| Tier | Method | When to use | Action |
|---|---|---|---|
| **T1** | Complementary tool | Tool A says no, BCCC says yes → run Tool B on just this contract | Add aderyn if slither disputed; run both on ExternalBug/GasException/DoS contracts |
| **T2** | Structural analysis | Mixed T1 signals | Check solc version (pre/post-0.8 changes overflow semantics), function signatures, call graphs via surya |
| **T3** | Mythril symbolic execution | T2 still ambiguous; top ~50 contracts by disagreement score | `docker run mythril/myth:0.24.8 analyze` with 5min timeout; skip on failure, document as no-mythril-verdict |
| **T4** | Manual review (human reads source) | Mythril inconclusive or class too ambiguous for tools | Sample disputed contracts (see Stage 5.4) |

**GasException / DenialOfService special handling:**
These classes have NO aderyn detector and slither's `costly-loop` / `calls-loop` are conservative (high FP in BCCC). For these two classes, T2 structural analysis is the primary resolution method:
- GasException: look for unbounded loops (`for`, `while`) iterating over arrays whose length is externally controlled
- DenialOfService: look for push-pattern (array append per user) without size cap, or conditional that can be forced to revert by attacker input

**Per disputed contract output:**
- Resolution verdict: KEEP / DROP / RELABEL
- Confidence score (0.0–1.0) using weight table from Stage 5.0
- Method code (T1/T2/T3/T4)
- Link to manual review notes if T4 used

**Output:** `outputs/p5_s3_resolution.csv` (all disputed contracts with resolution + confidence)

**Gate:**
- ≥ 90% of Stage 5.2 disputes have a resolution verdict → proceed to Stage 5.4
- < 90% resolved → expand T4 manual review sample
- For any class where Mythril is the only remaining tool and contracts keep timing out → document class as "unverified" with confidence = 0.0; flag for SENTINEL training to exclude or down-weight

---

### Stage 5.4: Manual Ground Truth
**Type:** Human review (gold standard, 10–20h)
**Goal:** Establish per-class ground truth through systematic human review of sampled contracts; derive extrapolation rules to extend findings to remaining contracts in the same class.

**Anchor data — reuse before sampling new contracts:**
- **Reentrancy has 500 contracts already reviewed** (CRITICAL_FINDINGS.md): 53 KEEP (`.call.value()`), 276 DROP (`.transfer()` / `.send()`), 171 DROP (no external call). These 500 contracts ARE the Stage 5.4 anchor for Reentrancy. Do NOT sample fresh Reentrancy contracts — use this existing data.
- 43 + 199 contracts from Phase 4 manual reviews cover multiple classes — cross-reference before sampling.
- 33 Phase 3 contracts (ws_i_disagreement_inspections.md) — use as additional anchors for Reentrancy/CallToUnknown.

**Sampling strategy for classes WITHOUT existing anchor data:**
Per noisy class (excluding Reentrancy which has its anchor), sample across four confidence tiers:
- **Tier A (high-confidence KEEP):** All tools + BCCC agree → spot-check 5
- **Tier B (high-confidence DROP):** No tool confirms, BCCC says yes → review 10
- **Tier C (ambiguous, mixed signals):** → review 15
- **Tier D (edge cases, unusual patterns):** → review 10

Total new reviews needed: ~40 per class × ~4 classes (Timestamp, ExternalBug, GasException, DoS) = ~160 new contracts
Reentrancy: use existing 500-contract anchor; derive rules only (no new sampling)

**Review process:**
1. Read full source code
2. Check against Stage 5.0 class definition (inclusion/exclusion criteria)
3. Document verdict: KEEP / DROP / RELABEL + one-sentence reasoning
4. Note any new patterns or rule candidates
5. After all contracts in a class: write extrapolation rules

**Extrapolation rules must be:**
- Explicit and automatable (e.g., "DROP any Reentrancy contract with no regex match for pre-0.8 OR post-0.8 call-value pattern")
- Traceable to specific reviewed contracts that established the rule
- Applied via automated script (not manual application to remaining contracts)

**Output:**
- `labels/p5_s4_manual_ground_truth.csv` — per-contract manual verdicts
- `labels/p5_s4_extrapolation_rules.md` — derived rules per class with contract citations
- `outputs/p5_s4_extrapolated_labels.csv` — rules applied to all remaining contracts in each class

**Gate:**
- Reentrancy: 500-contract anchor complete ✅ (already done) — just write extrapolation rules
- All other noisy classes: ≥ 20 contracts reviewed per class (across Tiers A–D)
- At least 1 extrapolation rule defined per noisy class before proceeding

---

### Stage 5.5: ML-Assisted Propagation
**Type:** Machine learning (medium cost, 8–12h)
**Prerequisite:** SENTINEL training must NOT be active on GPU. Verify: `ps aux | grep train.py`. If Run 9 (or any run) is still active, wait or run Stage 5.5 on CPU (expect ~4× time = 32–48h; use batched overnight run).
**Goal:** Use GraphCodeBERT embeddings to propagate verified labels from known-good anchors to structurally similar contracts.

**Model:** `microsoft/graphcodebert-base` (already in `ml/` SENTINEL environment, `TRANSFORMERS_OFFLINE=1`). Use this by default. Do NOT download a new CodeBERT model.

**Steps:**

1. **Embed all 67,311 contracts** using GraphCodeBERT
   - Input: read source from disk via `bccc_file_path` column in v1.1+12 CSV (no source_code in CSV)
   - Strip comments, normalize whitespace before embedding
   - Output: 768-dim embedding per contract saved as `outputs/p5_s5_embeddings.npy` (indexed by contract id)
   - Time: ~0.5s/contract on GPU = ~9h; ~2s/contract on CPU = ~37h (run overnight)
   - Batch size: 64; truncate at 512 tokens (GraphCodeBERT max)

2. **Cluster embeddings** (HDBSCAN, min_cluster_size=10, min_samples=5)
   - Expected: many clusters correspond to contract templates (Oraclize, ERC20, ICO crowdsale, etc.)
   - Save cluster assignments to `outputs/p5_s5_clusters.csv`

3. **Propagate labels from verified anchors:**
   - Anchor = contract with M9 (manual) or M3+M4 agreement verdict
   - For each cluster with ≥ 1 anchor, propagate labels to remaining cluster members
   - Propagation confidence = cluster_purity × anchor_confidence × 0.45 (M11 weight from Stage 5.0 table)
   - Do NOT propagate from provisional anchors (confidence < 0.75)

4. **Cross-validate propagation accuracy:**
   - Hold out 20% of manually reviewed contracts (from Stage 5.4 anchor data)
   - Measure: did propagation correctly predict the held-out verdicts?
   - If accuracy < 85% → restrict propagation to clusters with purity > 0.90 only (high-confidence subset)

**Output:**
- `outputs/p5_s5_embeddings.npy` — 67,311 × 768 float32
- `outputs/p5_s5_clusters.csv` — cluster assignments
- `outputs/p5_s5_propagated_labels.csv` — propagated labels with confidence scores

**Gate:**
- Propagation accuracy ≥ 85% on held-out manual reviews → use full propagation
- Propagation accuracy 70–85% → use only high-purity clusters (> 0.90 purity)
- Propagation accuracy < 70% → discard M11 propagation entirely; document class as "propagation failed"

---

### Stage 5.6: Synthesis — Final Verified Dataset
**Type:** Assembly and documentation (~2–3h)
**Goal:** Combine all verification stages into a single verified dataset with per-contract confidence scores, apply the all-labels-dropped disposition rule, run cross-class consistency pass, and produce v1.3.

**Steps:**

1. **Merge all evidence** from Stages 5.1–5.5 into unified label decisions per (contract, class) pair

2. **Compute final per-pair values:**
   - `label_<class>`: corrected binary label (0 or 1)
   - `confidence_<class>`: weighted average per Stage 5.0 weight table
   - `verification_method_<class>`: method code string (e.g., "M3+M4", "M9", "M11")
   - `verification_status_<class>`: `verified` / `provisional` / `unverified`

3. **Apply all-labels-dropped disposition rule:**
   - For each contract where ALL positive labels are rejected (all `label_<class>` = 0 after correction):
     - Set `Class12:NonVulnerable = 1`
     - Set `n_pos = 0`
     - Set `verification_status_NonVulnerable = provisional`
   - Log count of contracts reclassified this way

4. **Cross-class consistency pass:**
   - Flag any (contract, class) pair where `label_class = 1` and `confidence_class < 0.3` → downgrade to `label_class = 0`, log as `low_confidence_drop`
   - Check for new NV+vuln contradictions introduced by reclassification in step 3 → resolve via D-I-11/D-I-12 rules
   - Check for implausible co-occurrences (e.g., contract verified as IntegerUO-positive but no arithmetic ops in source) → flag for review

5. **Apply D-I-11 + D-I-12 final pass** (verify they still hold after all relabeling)

6. **Build `contracts_clean_v1.3.csv`** — output file name uses semver to avoid confusion with SENTINEL schema versions (v8/v9/v10):
   - All original columns from v1.1+12
   - New columns: `label_<class>` (corrected), `confidence_<class>`, `verification_method_<class>`, `verification_status_<class>` for all 9 SENTINEL vulnerability classes
   - `low_confidence_drop` flag column
   - `reclassified_to_nv` flag column (step 3)

7. **Generate per-class verification report** covering:
   - Original BCCC label count
   - Corrected label count
   - FP rate confirmed (% of original positives dropped)
   - FN rate (% newly found positives, if any)
   - Final class size (for SENTINEL training sizing)
   - Remaining uncertainty (% unverified)
   - Recommendation for SENTINEL training (use / use with confidence weighting / exclude class)

**Output:**
- `outputs/contracts_clean_v1.3.csv` — **MAIN DELIVERABLE**
- `outputs/p5_s6_verification_report.md` — per-class summary
- `outputs/p5_s6_confidence_distribution.csv` — histogram of confidence scores per class
- `outputs/p5_s6_class_size_comparison.csv` — before/after class sizes (for SENTINEL Run 10 planning)

**Gate:** Dataset completeness — all 67,311 contracts have labels + confidence scores for all 9 classes. Report completeness — all 9 classes documented. Cross-class consistency pass shows 0 unresolved NV+vuln contradictions.

---

## 6. Decision Gates Summary

| Stage | Class routing | Gate: PASS → | Gate: FAIL → |
|---|---|---|---|
| **5.0** | All classes | All 9 definition files written + checked vs. external source | Revise definition; do not proceed until approved |
| **5.1** | IntegerUO, UnusedReturn, MishandledException (expected pass) | ≥ 80% high-confidence verdicts from existing M1–M4 evidence → **VERIFIED, skip 5.2–5.4** | Unusual — treat as unexpected noise; re-examine existing evidence |
| **5.2** | All remaining classes | Agreement ≥ 95% → **VERIFIED, skip 5.3–5.4** | Agreement 80–95% → provisional, flag edge cases for 5.3 only; < 80% → all proceed to 5.3 |
| **5.3** | Classes failing 5.2 gate (expect: GasException, DoS, ExternalBug, Timestamp, Reentrancy, CallToUnknown) | ≥ 90% of disputes resolved with confidence ≥ 0.60 → proceed to 5.4 (for residual manual) | < 90% resolved → expand T4 manual sample; if still < 90% after 2 expansions → class is "unverified", confidence = 0.0 |
| **5.4** | Noisy classes needing manual ground truth | Reentrancy: use existing 500-anchor ✅; all others: ≥ 20 new reviews per class + ≥ 1 extrapolation rule | < 20 reviews → sample more; no rule derivable → document as "definition too ambiguous to extrapolate" |
| **5.5** | All classes (propagation only; runs after all class gates in 5.4) | Propagation accuracy ≥ 85% → full propagation; 70–85% → high-purity clusters only | < 70% → discard M11; skip propagation; document |
| **5.6** | All classes (synthesis) | Dataset complete (67,311 rows × all cols); 0 NV+vuln contradictions; all 9 classes in report | Missing data → backfill from lower stages before declaring done |

---

## 7. Per-Class Verification Flowchart

```
                     ┌─────────────────────┐
                     │ Stage 5.0:          │
                     │ Ground Truth Defs   │
                     │ + Confidence Weights│
                     └─────────┬───────────┘
                               │
                     ┌─────────▼───────────┐
                     │ Stage 5.1:          │
                     │ Existing Evidence   │
                     │ Integration         │
                     └─────────┬───────────┘
                               │
           ┌───────────────────┼─────────────────────────┐
           │                   │                         │
  ┌────────▼────────┐  ┌───────▼──────────┐  ┌──────────▼────────┐
  │ CLEAN CLASSES   │  │ MODERATE NOISY   │  │ HARD NOISY        │
  │ (IntegerUO,     │  │ (Reentrancy,     │  │ (GasException,    │
  │  UnusedReturn,  │  │  CallToUnknown,  │  │  DenialOfService, │
  │  MishandledEx)  │  │  Timestamp)      │  │  ExternalBug)     │
  │                 │  │                  │  │                   │
  │ → VERIFIED ✓   │  │ → Stage 5.2      │  │ → Stage 5.2 then  │
  │   at Stage 5.1 │  │   (regex primary)│  │   EXPECT Stage 5.3│
  └─────────────────┘  └───────┬──────────┘  └──────────┬────────┘
                               │                        │
                     ┌─────────▼────────────────────────▼────────┐
                     │ Stage 5.2: Automated Bulk Verification     │
                     │ - Reentrancy: regex pre-0.8 + post-0.8     │
                     │ - CallToUnknown: .call/.delegatecall regex  │
                     │ - Timestamp: block.timestamp in critical ctx│
                     │ - GasException/DoS: slither costly-loop/    │
                     │   calls-loop (no aderyn coverage)           │
                     │ - ExternalBug: selfdestruct/tx.origin regex │
                     └─────────┬──────────────────────────────────┘
                               │
                     ┌─────────▼──────────────────────────────────┐
                     │ Gate check per class:                       │
                     │ ≥ 95%  → VERIFIED, skip 5.3–5.4           │
                     │ 80-95% → PROVISIONAL, edge-cases to 5.3   │
                     │ < 80%  → proceed to Stage 5.3              │
                     │ (GasException/DoS/ExternalBug expected here)│
                     └─────────┬──────────────────────────────────┘
                               │
                     ┌─────────▼──────────────────────────────────┐
                     │ Stage 5.3: Discrepancy Resolution           │
                     │ T1: Complementary tool (aderyn/slither)     │
                     │ T2: Structural analysis (surya, solc ver)   │
                     │ T3: Mythril top ~50 ambiguous (5min timeout)│
                     │ T4: Manual sample (feeds Stage 5.4)         │
                     └─────────┬──────────────────────────────────┘
                               │
                     ┌─────────▼──────────────────────────────────┐
                     │ Stage 5.4: Manual Ground Truth              │
                     │ Reentrancy: use existing 500-anchor ✓       │
                     │ Others: ~40 contracts per class (4 classes) │
                     │ → derive extrapolation rules per class      │
                     │ → apply rules via script to full class      │
                     └─────────┬──────────────────────────────────┘
                               │
                     ┌─────────▼──────────────────────────────────┐
                     │ Stage 5.5: GraphCodeBERT Propagation        │
                     │ PREREQUISITE: no active SENTINEL training   │
                     │ Embed 67,311 contracts (~9h GPU / ~37h CPU) │
                     │ HDBSCAN cluster → propagate from anchors    │
                     │ Validate on held-out 20% of manual reviews  │
                     └─────────┬──────────────────────────────────┘
                               │
                     ┌─────────▼──────────────────────────────────┐
                     │ Stage 5.6: Synthesis                        │
                     │ 1. Merge all evidence                       │
                     │ 2. Apply all-labels-dropped → NV rule       │
                     │ 3. Cross-class consistency pass             │
                     │ 4. D-I-11/D-I-12 final verification        │
                     │ 5. → contracts_clean_v1.3.csv               │
                     │ 6. → per-class verification report          │
                     │ 7. → class size comparison (for Run 10)     │
                     └────────────────────────────────────────────┘
```

---

## 8. Estimated Timeline

| Stage | Description | Est. Time | Parallel? | Notes |
|---|---|---|---|---|
| **5.0** | Ground truth definitions + confidence weight table | 3–5h | No — design work | Nail down ExternalBug definition; it has no external reference |
| **5.1** | Evidence integration (script + run) | 0.5–1h | No | Fast — data already exists |
| **5.2** | Bulk automated verification | 2–5h | Yes — regex + slither in parallel | Add 2h buffer for GasException/DoS slither on full dataset |
| **5.3** | Discrepancy resolution | 5–10h | Partial — tools parallel, mythril sequential | GasException/DoS/ExternalBug will consume most of this time |
| **5.4** | Manual ground truth | 8–16h | No — human review, but per-class sequential | Reentrancy anchor free (500 contracts done); ~4 new classes × ~2h |
| **5.5** | GraphCodeBERT propagation | 9–12h GPU / 37–48h CPU | Yes — batched | **Cannot start while SENTINEL training is active** |
| **5.6** | Synthesis + report + CHANGELOG | 2–3h | No | Assembly only; no new analysis |
| **Total (GPU path)** | | **29–52h** | | |
| **Total (CPU path, Run 9 still training)** | | **57–88h** | | Stage 5.5 dominates |

**Expected session breakdown:**
- **Session 1:** Stage 5.0 (definitions, 3–5h) + Stage 5.1 (evidence, 1h) = 4–6h
- **Session 2:** Stage 5.2 (bulk automated, 2–5h) + Stage 5.3 start (discrepancy) = 6–10h
- **Session 3:** Stage 5.3 finish + Stage 5.4 (manual ground truth, 8–16h)
- **Session 4 (overnight if needed):** Stage 5.5 (GraphCodeBERT embedding) — start only when no active training
- **Session 5:** Stage 5.6 (synthesis, 2–3h) + README + CHANGELOG

---

## 9. Output Structure

```
Phase5_LabelVerification_2026-06-08/
├── 05_phase5_plan.md                          [This file — v1.1]
├── 06_handover_p1_to_p4.md                    [Full handover: Phases 1–4]
├── 00_actionable_checklist.md                 [Session tracker — created per stage]
├── 01_session_log.md                          [Session log — created per session]
├── labels/
│   ├── p5_s0_class_definitions/
│   │   ├── reentrancy.md
│   │   ├── integer_uo.md
│   │   ├── unused_return.md
│   │   ├── mishandled_exception.md
│   │   ├── timestamp.md
│   │   ├── call_to_unknown.md
│   │   ├── gas_exception.md
│   │   ├── denial_of_service.md
│   │   └── external_bug.md
│   ├── p5_s4_manual_ground_truth.csv          [new reviews only; anchor = existing 500 Reentrancy]
│   └── p5_s4_extrapolation_rules.md
├── outputs/
│   ├── p5_s1_evidence_table.csv               [67,311 × N evidence cols]
│   ├── p5_s2_automated_verdict.csv            [per-contract per-class automated verdict + confidence]
│   ├── p5_s3_resolution.csv                   [disputed contracts with resolution]
│   ├── p5_s4_extrapolated_labels.csv          [rules applied to full class]
│   ├── p5_s5_embeddings.npy                   [67,311 × 768 float32]
│   ├── p5_s5_clusters.csv
│   ├── p5_s5_propagated_labels.csv
│   ├── contracts_clean_v1.3.csv               [MAIN DELIVERABLE]
│   ├── p5_s6_verification_report.md           [per-class summary]
│   ├── p5_s6_confidence_distribution.csv
│   └── p5_s6_class_size_comparison.csv        [before/after sizes for SENTINEL Run 10 planning]
├── scripts/
│   └── (scripts created per stage)
├── reports/
│   └── (per-class investigation reports)
└── decisions/
    └── (Phase 5 decisions — D-P5-*)
```

---

## 10. Key Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Post-cleanup dataset too small for key classes (Reentrancy ~10×, CallToUnknown ~10×) | **High** | SENTINEL Run 10 may have insufficient training data for noisy classes | Document expected sizes in Stage 5.6 class-size comparison; SENTINEL Run 10 may need to drop or merge these classes |
| GasException/DenialOfService remain ambiguous after all stages | High | These classes stay "unverified" | Flag as `verification_status = unverified`; use confidence weighting in AutoML rather than binary labels |
| Manual review Stage 5.4 too slow (~200 contracts × 1–2h) | Medium | Session 3 expands to 2–4 weeks | Reduce to 100 contracts total if needed; prioritize extrapolation rule derivation over exhaustive coverage |
| GraphCodeBERT embedding takes > 12h on GPU | Medium | Stage 5.5 delays | Run overnight as background task; if GPU unavailable (Run 9 active), run CPU batches in parallel across multiple terminal sessions |
| Mythril hangs or crashes on ambiguous contracts | Medium | Stage 5.3 delays | 5min timeout hard limit; skip on failure; document as `no-mythril-verdict`; do NOT block Stage 5.4 waiting for Mythril |
| ExternalBug cannot be well-defined (BCCC catch-all with no standard reference) | Medium | Class may not be verifiable | If Stage 5.0 cannot produce inclusion criteria grounded in at least one external source, drop ExternalBug from Phase 5 entirely and document as "unverifiable, exclude from SENTINEL training" |
| Clean classes (IntegerUO, UnusedReturn, MishandledException) turn out to also be noisy in some subpopulation | Low | Major re-scope | Already confirmed by manual review + 2 tools; if Stage 5.1 shows unexpected noise for a subpopulation, investigate that subpopulation specifically rather than treating the whole class as noisy |
| All-labels-dropped contracts flood NonVulnerable class (skewing class balance) | Medium | Class imbalance worsens | Count in Stage 5.6 step 3; if reclassified NV count > 1,000 contracts, document as known imbalance for AutoML/SENTINEL SMOTE handling |

---

## 11. What Phase 5 Intentionally Does NOT Do

- ❌ **AutoML** — Parked until labels are verified
- ❌ **Feature engineering** — 34 features already exist; new features deferred to post-Phase-5
- ❌ **SENTINEL model training** — That is SENTINEL Run 10+, outside Phase 5 scope
- ❌ **Mythril on all contracts** — Only on top ~50 ambiguous cases in Stage 5.3
- ❌ **Full per-folder investigation** — Uses per-class investigation instead (more targeted)
- ❌ **WS-P/Q/R/T/K2 from Phase 3** — Parked after Phase 5
- ❌ **SmartBugs OOD re-run** — Keep existing SENTINEL v10 data unchanged
- ❌ **New dataset collection** — Phase 5 works only on existing 67,311 BCCC contracts

---

## 12. How This Differs From the Old Phase 4 Plan

| Dimension | Old Phase 4 (Stages 2–7) | Phase 5 (this plan) |
|---|---|---|
| **Assumption** | Label noise is sample-size limited | Label noise is structural |
| **Approach** | More samples + mythril on hardest cases | Multi-method verification + manual ground truth |
| **Gate** | Dataset-wide median F1 > 0.5 | Per-class gates at each stage |
| **Reentrancy** | Treat as BCCC defines it | Narrow to strict definition; reuse 500-contract audit as anchor |
| **Syntax coverage** | Single regex pattern | Pre-0.8 AND post-0.8 Solidity patterns explicitly |
| **Dropped contracts** | Removed from dataset | Reclassified as NonVulnerable |
| **Output** | contracts_clean_v13.csv with corrected labels | contracts_clean_v1.3.csv with confidence scores + verification provenance |
| **AutoML** | Run on v1.3 regardless of label quality | Deferred until verified labels exist |
| **Manual review** | 3–5 per folder (24–40 total) | ~200 new + 500 existing Reentrancy anchor = ~700 total |
| **ML propagation** | None | GraphCodeBERT (already in environment) |
| **Dataset size warning** | Not addressed | Explicit size estimate per class in Stage 5.6 output |

---

## 13. Immediate Next Steps (Session 1 Start)

1. [ ] Write 9 ground truth definition files — `labels/p5_s0_class_definitions/` (Stage 5.0)
   - Priority order: ExternalBug first (hardest to define), then Timestamp, GasException, DoS, Reentrancy, CallToUnknown, then the 3 clean classes
2. [ ] Finalize confidence weight table in this document (small adjustments allowed based on definitions written)
3. [ ] Write Stage 5.1 evidence integration script — load all 6 existing evidence sources into unified table
4. [ ] Run Stage 5.1 script; identify which classes pass at Stage 5.1
5. [ ] Start Stage 5.2 for classes that need automated verification
6. [ ] Update `Phase5_LabelVerification_2026-06-08/README.md` with Phase 5 status

---

**Last updated:** 2026-06-08 (v1.1 — comprehensive review pass; fixes: path, Timestamp noise level, ExternalBug routing, all-labels-dropped rule, post-cleanup size warning, GraphCodeBERT default, pre-0.8 regex coverage, confidence weights, VRAM prerequisite, 500-contract Reentrancy anchor, cross-class consistency pass, version naming v1.3)
