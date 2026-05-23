# Data Quality Findings — Root Cause Analysis

**Date:** 2026-05-23  
**Context:** After v7.0, v8.0-AB, and PLAN-3A all converged to the same ~0.287–0.288 tuned F1-macro  
**Conclusion:** The bottleneck is not architecture — it is data quality

---

## The Flat Ceiling Proof

| Run | Architecture change | Tuned F1 | Delta |
|-----|--------------------|---------:|------:|
| v7.0 | baseline (CF only) | 0.2875 | — |
| v8.0-AB | +ICFG + DEF_USE (11 edge types) | 0.2851 | −0.0024 |
| PLAN-3A | +ICFG only (drop DEF_USE) | 0.2877 | +0.0002 |

Three architecturally different models. Same ceiling. Architecture is ruled out as the bottleneck.

---

## Mislabeling Taxonomy: Two Tiers

### Tier 1 — Structural impossibility (fixable by label_cleaner.py)

A contract labeled as vulnerable when the graph literally cannot exhibit the vulnerability.

| Class | Check | Count removed | Status |
|-------|-------|--------------|--------|
| Reentrancy | Has external calls AND state WRITES | −611 | **Applied 2026-05-23** |
| Timestamp | Has block globals AND (external calls OR payable) | −568 | **Applied 2026-05-23** |
| UnusedReturn | Has return_ignored feature | −1,665 | Applied |
| CallToUnknown | Has untyped call or external call | −383 | Applied |
| MishandledException | Has return_ignored or untyped call | −632 | Applied |

**Total removed (2026-05-23 run):** 3,859 labels from 44,524-row deduped CSV.  
**Output:** `ml/data/processed/multilabel_index_cleaned.csv`  
**Audit log:** `ml/data/processed/multilabel_index_cleaned.audit.json`

### Tier 2 — Semantic mislabeling (NOT fixable by structural checks alone)

A contract labeled as vulnerable where the structure PERMITS the vulnerability but the specific contract is safe. These contracts have real external calls, real block.timestamp usage, real state writes — but the specific code pattern is benign. Slither's detectors are too aggressive.

**Primary example — Timestamp (BUG-H4):**  
Before structural cleaning: 48.2% of Timestamp=1 contracts had `uses_block_globals=0` — purely structural mislabeling. After the 2026-05-23 cleaning: structural impossibilities are removed. BUT: many remaining Timestamp=1 contracts use `block.timestamp` safely (e.g., for event timestamps, log entries, non-security-sensitive expiry checks). These pass the structural check but are semantically wrong. PLAN-3A confirmed this: Timestamp improved +0.038 when DEF_USE was dropped — the edges were amplifying noise from these mislabeled contracts.

**Estimated remaining Tier 2 noise:**
- Timestamp: after structural cleaning, likely 20–35% of remaining Timestamp=1 contracts are still semantically mislabeled
- Reentrancy: contracts that have external calls AND state writes but the specific call is inside a `nonReentrant` modifier or the write precedes the call (safe CEI order)
- IntegerUO: post-0.8 Solidity contracts (which have built-in overflow protection) were labeled based on folder association, not per-contract Solidity version check

---

## OR-Labeling Noise (BCCC Dataset Root Cause)

The BCCC dataset uses folder-level labeling: every contract in a "reentrancy" folder gets `Reentrancy=1`, regardless of whether that specific contract has a reentrancy bug. This produces systematic false positives for benign contracts that happen to share a folder with vulnerable ones.

**Structural consequence:** An ERC20 token contract with `transfer()` has an external call. It passes `check_reentrancy`. It is in a reentrancy folder. Result: benign ERC20 gets `Reentrancy=1, CallToUnknown=1, MishandledException=1, IntegerUO=1` simultaneously.

This is the root cause of the **safe-contract false-positive problem** observed in behavioral testing: `0/3 safe contracts clean` — the model fires 7/10 classes on a clean contract because it trained on benign ERC20s labeled as vulnerable across 7 classes.

**There is no structural fix for OR-labeling noise.** The only fixes are:
1. Add confirmed-clean contracts as explicit negatives (augmented safe contracts)
2. Source better-labeled data (per-contract Slither output, not folder-level)
3. Active learning: manually verify the highest-uncertainty predictions

---

## Per-Class Noise Analysis

### Timestamp — Primary bottleneck (BUG-H4)

**Pre-cleaning:** 538 total → ~259 structural impossibilities (uses_block_globals=0)  
**Post-cleaning (2026-05-23):** 568 removed via stricter check (block globals AND value-sensitive path)  
**Remaining noise:** ~20–35% of remaining Timestamp=1 are semantically safe uses  
**What fixes this:** Active learning (surface highest-uncertainty Timestamp preds for manual review); better Slither re-labeling using taint analysis

PLAN-3A key finding: Timestamp improved +0.038 by DROPPING DEF_USE, not by adding better edges. This confirms the model was being hurt by training noise amplified through DEF_USE def-use chains. Fixing the labels is more impactful than changing the edge types.

### Reentrancy — Secondary bottleneck (BUG-H5)

**Pre-PLAN-3A estimate:** ~14% of Reentrancy=1 contracts have no external calls — structural impossibility  
**Post-cleaning (2026-05-23):** −611 removed (no external calls AND no WRITES edges)  
**Remaining noise:** contracts that have external calls AND state writes but the reentrancy guard (`nonReentrant` modifier, CEI-ordered code) makes them safe  
**Evidence:** PLAN-3A H1 test showed only +0.005 Reentrancy improvement from ICFG edges — the architecture could not break through the label noise ceiling  
**What fixes this:** Per-contract Slither call analysis checking for re-entrant execution paths, not just structural presence of call+write

### DoS — Data starvation (D2)

**Training positives:** ~260 in current dataset (after augmentation injection)  
**Current treatment:** `dos_loss_weight=0.0` — excluded from gradient  
**Issue:** This setting was made when DoS had 7 samples. With ~260, it warrants re-evaluation.  
**Action:** Enable `dos_loss_weight=0.5` in next training run; monitor DoS F1 — if it rises while other classes hold, the loss weight was the bottleneck, not data volume.

### IntegerUO — Moderate noise

**Training positives:** 13,797 (largest class)  
**Noise source:** OR-labeling; post-0.8 Solidity contracts with built-in overflow protection labeled as IntegerUO from folder association  
**Observable evidence:** IntegerUO F1 has been stable across all runs (0.699–0.715) suggesting the class has enough clean signal to learn despite noise  
**No immediate action needed** — focus on Timestamp and Reentrancy first

---

## What label_cleaner.py Can and Cannot Do

**CAN do (structural impossibility checks):**
- Remove Reentrancy=1 from contracts with no external calls and no state writes
- Remove Timestamp=1 from contracts with no block globals and no value-sensitive path
- Remove UnusedReturn=1 from contracts with no ignored returns
- Remove CallToUnknown=1 from contracts with no external calls at all
- Remove MishandledException=1 from contracts with no unhandled return or untyped call

**CANNOT do (requires semantic analysis or external information):**
- Identify Timestamp=1 contracts where block.timestamp is used safely (event log, non-critical expiry)
- Identify Reentrancy=1 contracts protected by `nonReentrant` or CEI ordering
- Identify post-0.8 IntegerUO mislabelings (requires parsing Solidity version pragma)
- Fix OR-labeling noise for ERC20s that share folders with vulnerable contracts
- Distinguish confirmed-safe from unknown-safe contracts

---

## The Safe-Contract False-Positive Problem

Behavioral testing: `0/3 safe contracts clean` in all three models (v7, v8-AB, PLAN-3A).  
A clean ERC20 contract fires 7/10 vulnerability classes.

**Root cause:** The training data is dominated by the BCCC dataset which labels entire folders. Many benign contracts in vulnerability folders received positive labels across multiple classes. The model learned "this type of ERC20 structure → multiple vulnerabilities" because that's what the training data says.

**Fix path:**
1. Add 100+ confirmed-clean contracts with all labels explicitly set to 0 in training
2. These act as negative anchors — they teach the model what a truly safe contract looks like
3. The augmented `dos_safe_01..30.sol` contracts are a start (30 contracts)
4. Need more: standard ERC20/ERC721 contracts from OpenZeppelin (known clean, audited)

---

## Priority Action Plan (Updated 2026-05-23)

Steps in strict order — do not skip.

### Step 1 — DONE: Apply improved structural cleaning
- Improved `label_cleaner.py`: stricter Timestamp (requires value-sensitive path), stricter Reentrancy (requires WRITES edge)
- Output: `ml/data/processed/multilabel_index_cleaned.csv` (3,859 labels removed)
- No cache rebuild needed — labels are read from CSV at training time

### Step 2 — Re-enable DoS loss weight
- Edit `dos_loss_weight` from 0.0 to 0.5 in trainer config
- ~260 DoS training examples warrant a gradient signal
- Start with 0.5 to avoid dominating real-data gradients with augmented data

### Step 3 — Add clean negative contracts
- Source 100+ audited-clean Solidity contracts (OpenZeppelin ERC20/ERC721 base contracts)
- Inject with all-zero labels via `inject_augmented.py`
- Reduces safe-contract false positive rate in behavioral testing

### Step 4 — Retrain (next run = v8.0-B with cleaned labels)
- Use PLAN-3A configuration (best architecture found: CF+CALL_ENTRY+RETURN_TO)
- Use updated cleaned CSV
- Enable dos_loss_weight=0.5
- Expected improvements: Reentrancy (+0.01–0.02), Timestamp (small additional gain), DoS (non-zero F1)

### Step 5 — Active learning loop (after v8.0-B training)
- Collect val contracts where Timestamp prob ∈ (0.35, 0.65)
- Surface for manual review — these are the most uncertain labels
- Fix ~200–300 Timestamp labels manually
- Retrain → expected +0.03–0.05 Timestamp F1 from this iteration

### Step 6 — PLAN-3D: JK concatenation mode (after Step 4 results confirmed)
- Switch `gnn_jk_mode` from "attention" to "cat"
- Eliminates JK attention collapse by construction
- Requires fusion head resize (3×256=768 → projection)
- Run AFTER label quality is fixed so comparison is meaningful

---

## Diagnostics Added (2026-05-23)

| Script | Purpose | Output |
|--------|---------|--------|
| `ml/scripts/complexity_correlation.py` | Shortcut learning detection — Spearman r between CFG node count and predicted probability | `ml/logs/complexity_correlation_<run>.json` |

**Complexity correlation threshold:** r > 0.40 = strong shortcut evidence. If the model's predictions correlate with raw contract size rather than specific structural patterns, it means the model is using "large = vulnerable" as its primary heuristic — a shortcut that explains why all architecture variants hit the same ceiling.

**Results (PLAN-3A checkpoint, 1500 val contracts, 2026-05-23):**

| Metric | Highest r | Class | Alert? |
|--------|----------:|-------|--------|
| num_nodes | 0.396 | Timestamp | No (just under 0.40) |
| num_edges | 0.385 | Timestamp | No |
| num_cfg_edges | 0.390 | Timestamp | No |
| ext_calls_sum | **0.402** | MishandledException | **YES — r > 0.40** |
| max_complexity | 0.261 | Timestamp | No |

**Single shortcut alert:** `ext_calls_sum` vs `MishandledException`, r=0.402. Interpretation: MishandledException predictions positively correlate with how many external calls a contract makes (summed across all nodes). This makes sense — more external calls → more chances to mishandle a return value → model is partly using external-call density as a proxy for MishandledException risk.

**Important nuance:** r=0.402 is a weak shortcut. It confirms the model is using a legitimate structural correlate (external call count IS a necessary condition for MishandledException), not a spurious size shortcut. This is expected and not alarming.

**The critical finding:** `num_nodes` vs `Timestamp` at r=0.396 is the most concerning near-shortcut. All complexity metrics show moderate (0.25–0.40) correlations with Timestamp, which means the model is partly learning "large contracts are more likely to have timestamp dependencies" rather than detecting specific timestamp misuse patterns. This is consistent with BUG-H4 semantic mislabeling — if 48.2% of Timestamp labels are wrong, the model falls back to statistical correlates (contract size, external call count) rather than true structural patterns.

**Shortcut assessment:** The model shows MODERATE complexity shortcuts (r=0.25–0.40 across all classes, r=0.40 for one specific metric-class pair) but NOT strong shortcuts (r > 0.50). Interpretation B from R2 (size-as-feature proxy) is PARTIALLY correct but not dominant. Label quality improvements will reduce the moderate correlations by giving the model better structural patterns to learn.

---

## What Improving Architecture Cannot Fix

The following are data problems that architecture changes cannot address:

1. Timestamp semantic mislabeling → model trains on wrong signal; any architecture learns the wrong pattern
2. OR-labeling noise for ERC20s → model learns "ERC20 structure = multiple vulnerabilities"; safe-contract false positives persist
3. Reentrancy noise from safe CEI-ordered contracts → model confuses safe external-call-then-write with vulnerable write-after-call
4. DoS starvation → model has no signal to learn from regardless of how many edges or attention heads are added

All three training runs reached the same ceiling because the training signal is identical — the same noisy labels produce the same approximate F1. Label quality is the only lever that changes the ceiling.
