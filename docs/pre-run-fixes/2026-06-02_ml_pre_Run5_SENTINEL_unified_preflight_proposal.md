# SENTINEL Run 5 — Unified Pre-Flight Proposal

## Comprehensive Fix & Intervention Plan — ML Module

**Prepared:** 2026-06-02 (revised with review findings)
**Baseline:** GCB-P1-Run4-no-asl-pw_best.pt — epoch 32 — macro-F1 = 0.3362
**Consolidated from:** phase2_root_cause_analysis.md, validated_audition.md, CHANGELOG.md
**Scope:** All 36 confirmed code-level bugs (A1–A38, excluding A24 absent and A31 already fixed), 12 new findings (NF-1–NF-12), seven Phase 2 root causes (RC1–RC7), capacity ceiling analysis, and all recommended training interventions — structured as an ordered execution plan with explicit go/no-go gates and rollback decision trees at every critical phase boundary. Includes comprehensive data archival and migration requirements to ensure clean v9 dataset usage throughout.

> **Design principle for this document:** This proposal describes *what* must be fixed, *why*, and *in what order*. Detailed fix code is intentionally omitted to avoid constraining implementation choices — the same fix can often be realized in multiple valid ways, and the implementer should have the freedom to select the best approach given the surrounding context at the time of implementation.

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Fix Categories & Inventory](#2-fix-categories--inventory)
3. [Execution Order Rationale](#3-execution-order-rationale)
4. [Phase 0 — Critical Pre-Flight Safety Fixes](#4-phase-0--critical-pre-flight-safety-fixes)
5. [Phase 1 — Data & Schema Layer Fixes](#5-phase-1--data--schema-layer-fixes)
6. [Phase 2 — Graph Extraction Layer Fixes](#6-phase-2--graph-extraction-layer-fixes)
7. [Phase 3 — Model Architecture Fixes](#7-phase-3--model-architecture-fixes)
8. [Phase 4 — Training Loop Fixes](#8-phase-4--training-loop-fixes)
9. [Phase 5 — Training Interventions for Phase 2 Signal](#9-phase-5--training-interventions-for-phase-2-signal)
10. [Phase 6 — Calibration & Threshold Fixes](#10-phase-6--calibration--threshold-fixes)
11. [Phase 7 — Data Re-Extraction, Archival & Migration (IMP-D1)](#11-phase-7--data-re-extraction-archival--migration-imp-d1)
12. [Phase 8 — Run 5 Execution & Monitoring](#12-phase-8--run-5-execution--monitoring)
13. [Gate Summary — Complete Go/No-Go Matrix with Rollback Decision Trees](#13-gate-summary--complete-gono-go-matrix-with-rollback-decision-trees)
14. [Expected Outcomes & Risk Assessment](#14-expected-outcomes--risk-assessment)
15. [Out-of-Scope Items for Run 5 (Run 6+ Candidates)](#15-out-of-scope-items-for-run-5-run-6-candidates)
16. [Open Bugs From Changelog (Non-Proposal)](#16-open-bugs-from-changelog-non-proposal)

---

## 1. Problem Statement

### 1.1 The Ceiling

Run 4 plateaued at macro-F1 = 0.3362 (epoch 32) and refused to improve through epoch 44. This is not a data problem — the Phase 3.5 experiment (v8.0-B, strict label cleaning) confirmed that data quality alone produces F1 = 0.2460, *worse* than the architectural ceiling of ~0.287. The ceiling is purely architectural and training-configuration driven.

### 1.2 The Root Diagnosis

The interpretability study established a precise causal account across two compounding layers:

**Layer 1 — Phase 2 Signal Collapse (Structural):** Phase 2 (GNN layers L3+L4+L5 processing CFG / ICFG / DEF_USE edges) receives 72–91% of Phase 1's gradient norm during backprop but contributes less than 1.4% of the inference signal that Phase 1 does. Structural ablation confirmed: combined CFG drop = 0.0121 versus embedding-only drop = 1.11×10⁻⁶. Seven confirmed root causes (RC1–RC7) explain why Phase 2 receives gradient but cannot convert it into useful inference signal.

**Layer 2 — Code-Level Bugs (Accumulated):** 36 confirmed bugs spanning graph extraction, model architecture, and the training loop — three of which can corrupt training data or invalidate Adam optimizer state permanently. These bugs have been present across all previous runs and compound the structural problem by silently degrading graph quality, losing signal, or introducing incorrect data.

### 1.3 Why Both Layers Must Be Fixed Together

Fixing bugs alone will not break the ceiling (the Phase 2 signal collapse is structural). Fixing Phase 2 interventions alone will not work because bugs are silently poisoning data, losing extraction signal, and corrupting optimizer state. Run 5 cannot be expected to outperform Run 4 unless both classes of issue are fixed together and in the correct dependency order.

### 1.4 Historical Context

The project has progressed through multiple architecture iterations — v4 (CodeBERT, 4-layer GNN, leaky dataset F1=0.5422), v5.0-v5.2 (Three-Eye + JK + LoRA), v6 (schema patch 12→11 dims), v7 (full overhaul, 27 bugs fixed, F1=0.2875), v8 (cross-function edges: CALL_ENTRY, RETURN_TO, DEF_USE), v8-AB (joint ICFG+DEF_USE ablation), PLAN-3A (ICFG-only, F1=0.2877), Phase 3.5 (strict label cleaning, H5 refuted), Phase 3.6 (GraphCodeBERT + GNN prefix injection), IMP-G1/G2/G3 architectural fixes (layer-specific edge subsets, input projection skip, bidirectional Phase 3), and P1-TRAIN Runs 1–4. Run 4 (the current baseline at F1=0.3362) used an 8-layer GNN with GraphCodeBERT, prefix K=48, IMP-* fixes, and λ=0.005 JK entropy — but no auxiliary Phase 2 loss, no bug fixes, and still-incorrect extraction logic. Run 5 is the first attempt to fix everything simultaneously.

---

## 2. Fix Categories & Inventory

| Category | Source IDs | Count | Max Severity | Files Affected |
|----------|-----------|-------|-------------|----------------|
| **Critical safety — must fix before any training** | A20, A38 | 2 | High / Data Poisoning | ast_extractor.py, trainer.py |
| **Phase 2 root causes — architectural** | RC1, RC3, RC5, RC6 | 4 | Architecture | gnn_encoder.py, sentinel_model.py |
| **Phase 2 root causes — training** | RC2, RC4, RC7 | 3 | Training | trainer.py, TrainConfig |
| **Graph extraction bugs** | A3–A18 | 16 | Medium | graph_extractor.py |
| **Model architecture bugs** | A23–A34, A25b | 9 | Medium | gnn_encoder.py, transformer_encoder.py, sentinel_model.py |
| **Training loop bugs** | A35–A38 | 4 | Medium | trainer.py |
| **Schema / toolchain bugs** | A1–A2, A19–A22 | 6 | Medium | graph_schema.py, hash_utils.py, ast_extractor.py |
| **Performance improvements** | A26, A29, A33 | 3 | Low | gnn_encoder.py, transformer_encoder.py, sentinel_model.py |
| **Already fixed (excluded)** | A31 | 1 | — | fusion_layer.py (C2 fix) |
| **Total active fixes** | | **36 + 7 RC** | | |
| **New findings (NF-1–NF-12, validated 2026-06-02)** | NF-1–NF-12 | 12 | High–Medium | graph_extractor.py, gnn_encoder.py, sentinel_model.py, trainer.py, train.py |
| **New findings total** | | **48 + 7 RC** | | |

### Complete Bug Inventory (A1–A38)

| ID | File | Bug Summary | Severity | Validated |
|----|------|-------------|----------|-----------|
| A1 | graph_schema.py | Missing `max(NODE_TYPES)` range guard — count guarded but max value not | Medium | CONFIRMED |
| A2 | hash_utils.py | `validate_hash` accepts uppercase hex via `int(,16)` — deduplication breaks | Low | CONFIRMED |
| A3 | graph_extractor.py | Dynamic `_MAX_TYPE_ID` changes silently if NODE_TYPES gains entries | Medium | CONFIRMED |
| A4 | graph_extractor.py | `assert` for production invariant (node/metadata alignment) — silent under `-O` | Medium | CONFIRMED |
| A5 | graph_extractor.py | `except AttributeError` scope too broad in `_compute_return_ignored` | Low | CONFIRMED (not fully silent — logs warning) |
| A6 | graph_extractor.py | Bare `except Exception: pass` in `_compute_call_target_typed` — silent total loss | Medium | CONFIRMED |
| A7 | graph_extractor.py | Dead code `_compute_in_unchecked` — deprecated since v7 | Low | CONFIRMED |
| A8 | graph_extractor.py | `is True` identity check in `_compute_has_loop` — misses truthy non-booleans | Low | CONFIRMED |
| A9 | graph_extractor.py | String-based class check for `SolidityVariableComposed` — breaks on Slither rename | Medium | CONFIRMED |
| A10 | graph_extractor.py | Bare `except Exception: pass` in `_cfg_node_type` — all nodes become CFG_NODE_OTHER | Medium | CONFIRMED |
| A11 | graph_extractor.py | Hardcoded parent feature indices in `_build_cfg_node_features` — fragile to reordering | Medium | CONFIRMED |
| A12 | graph_extractor.py | `n.node_id` without fallback in sort key — crashes on synthetic nodes | Low | CONFIRMED |
| A13 | graph_extractor.py | Silently dropped CONTROL_FLOW edges not logged | Low | CONFIRMED |
| A14 | graph_extractor.py | RETURN_TO cartesian product includes revert paths — semantically incorrect | Medium | CONFIRMED |
| A15 | graph_extractor.py | DEF_USE `def_map` keyed by variable name only — scope collision (see §6.11 for two-tier fix) | **Medium** | CONFIRMED |
| A16 | graph_extractor.py | `assert` for sentinel range check — silent under `-O` | Low | CONFIRMED |
| A17 | graph_extractor.py | Exception routing by string keyword matching — fragile both directions | Medium | CONFIRMED |
| A18 | graph_extractor.py | Bare `except Exception: pass` for ICFG map construction — zeros all CALL_ENTRY/RETURN_TO | Medium | CONFIRMED |
| A19 | ast_extractor.py | `get_solc_binary` uses `Path.cwd()` instead of `get_project_root()` | Medium | CONFIRMED |
| A20 | ast_extractor.py | **`label=0` hardcoded in batch extraction — DATA POISONING** | **HIGH** | CONFIRMED |
| A21 | ast_extractor.py | Worker `print()` under concurrency — interleaved output | Low | CONFIRMED |
| A22 | ast_extractor.py | `torch.save` without error handling — disk-full aborts entire batch | Medium | CONFIRMED |
| A23 | gnn_encoder.py | `last_weight_stds` NaN for N=1 — comment claims 0, PyTorch returns NaN | Low | CONFIRMED |
| A25 | gnn_encoder.py | `edge_index.max()` O(E) scan every forward pass — belongs at data-load time | Low | CONFIRMED |
| A26 | gnn_encoder.py | `next(self.parameters())` called twice per forward pass — should cache | Low | CONFIRMED |
| A27 | gnn_encoder.py | `num_layers` stored but hardcoded to 8 — misleads checkpoint checks | Low | CONFIRMED |
| A28 | transformer_encoder.py | `except (ImportError, ValueError)` catches real BERT load errors | Medium | CONFIRMED |
| A29 | transformer_encoder.py | Python loop for prefix mask construction — vectorizable | Low | CONFIRMED |
| A30 | transformer_encoder.py | `_word_embeddings` fragile hardcoded PEFT path — breaks on version change | Low | CONFIRMED |
| A31 | fusion_layer.py | `_scatter_to_dense` truncation overwrites real node at max_nodes-1 | — | **ALREADY FIXED (C2)** |
| A32 | sentinel_model.py | Dynamic `_MAX_TYPE_ID` decoupled from encoded .pt files — same as A3 | Medium | CONFIRMED |
| A33 | sentinel_model.py | `select_prefix_nodes` Python loop over batch dimension — vectorizable | Low | CONFIRMED |
| A34 | sentinel_model.py | Secondary sort uses post-GAT embedding dim, not raw input feature | Medium | CONFIRMED |
| A25b | sentinel_model.py | `compute_prefix_attention_mean` discards `node_counts` — diagnostic understates attention | Low | CONFIRMED |
| A35 | trainer.py | `_FocalFromLogits` unpicklable local class — blocks DDP/checkpointing | Low | CONFIRMED |
| A36 | trainer.py | `compute_pos_weight` re-reads label CSV every call — redundant I/O | Low | CONFIRMED |
| A37 | trainer.py | Threshold sweep O(N×C×19) every validation epoch — known BUG-M8 | Low | CONFIRMED |
| A38 | trainer.py | **NaN loss `backward()` runs before NaN check — corrupts Adam state permanently** | **MEDIUM-HIGH** | CONFIRMED |

### New Finding Inventory (NF-1–NF-12) — Validated 2026-06-02

| ID | File | Bug Summary | Severity | Confirmed |
|----|------|-------------|----------|-----------|
| NF-1 | graph_extractor.py ~L1311 | EMITS fallback uses `ir.name` (short) but node_map stores `canonical_name` — all EMITS edges silently dropped for Solidity <0.4.21 fallback path | Medium | CONFIRMED |
| NF-2 | graph_extractor.py ~L1094 | `_add_node` reverse-normalization hardcodes `12` instead of `_MAX_TYPE_ID` — decode-side counterpart of A3/A32 | Medium | CONFIRMED |
| NF-3 | scripts/build_multilabel_index.py ~L193 | Downstream A20 propagation: fallback `binary_y = int(graph.y.item())` for non-BCCC contracts labels them all safe while A20 is unfixed. Auto-heals after Phase 7 re-extraction. | Medium | CONFIRMED (healed by A20 fix) |
| NF-4 | scripts/train.py ~L151 | **`--gnn-layers` CLI default = 7 but TrainConfig default = 8.** Running `python train.py` without `--gnn-layers` produces a 7-layer model instead of 8. Run 4 used 8 layers — this silently changes architecture. | **HIGH** | CONFIRMED |
| NF-5 | scripts/train.py | `aux_phase2_loss_weight` not exposed as CLI arg — the most important new Run 5 hyperparameter cannot be overridden from the command line | Medium | CONFIRMED |
| NF-6 | gnn_encoder.py ~L476–480 | Phase 2 Layer 3 and Layer 4 build edge subsets from raw `edge_attr` unconditionally, ignoring the `phase2_edge_types` ablation config. Only Layer 5 respects ablation. Ablation experiments silently include excluded edge types in Layers 3/4. | Medium | CONFIRMED |
| NF-7 | graph_extractor.py ~L404, ~L430 | `_compute_external_call_count` and `_compute_uses_block_globals` both catch all exceptions and return `0.0` with no warning — "0 calls" / "no block globals" instead of signaling failure. Creates silent false-negatives for DoS and Timestamp detection. | Medium | CONFIRMED |
| NF-8 | sentinel_model.py ~L422–426 | Empty-batch guard's `aux_zeros` dict has keys `{gnn, transformer, fused}` but the normal-path aux dict has keys `{phase2, jk_entropy}`. Inconsistent return contract — KeyError if trainer tries both key sets on the same aux dict. **Can crash Run 5 on any empty-batch epoch.** | **Medium** | CONFIRMED |
| NF-9 | trainer.py ~L1177 | `AdamW(fused=True)` raises `RuntimeError` on CPU-only machines. Should be `fused=(device.type == "cuda")`. Non-blocking on RTX 3070 but fragile. | Low | CONFIRMED |
| NF-10 | graph_extractor.py ~L1134–1148 | Duplicate function name (e.g., inherited + overriding) causes second function's CFG nodes to be attached via CONTAINS to the first function's node — conflating two distinct CFGs under one FUNCTION node. | Medium | CONFIRMED |
| NF-11 | graph_extractor.py ~L1274–1279 | `_add_edge` silently drops ALL edge types (CALLS, READS, WRITES, EMITS, INHERITS) when either endpoint is absent from node_map, with no counter or log. A13 covers only CONTROL_FLOW; this is the broader form of the same bug. | Medium | CONFIRMED |
| NF-12 | src/inference/predictor.py | Inference-only: predictor silently truncates to 4 windows with no warning when preprocessor produces more; old checkpoints get randomly-initialized embeddings for new edge types with no warning. **Out of scope for Run 5 training** — deferred to inference hardening. | Low | CONFIRMED (inference-only) |

> **NF-3 note:** NF-3 is not an independent bug fix — it auto-heals when A20 is fixed and Phase 7 re-extraction is done. However, it is an important **pre-extraction validation point**: build_multilabel_index.py must not be run on the un-fixed (A20-corrupted) graphs. Add NF-3 as a validation note in Gate 7.3.

> **NF-8 note:** This bug is elevated to Medium severity because Run 5 will access `aux["phase2"]` for auxiliary Phase 2 loss computation. If an empty batch occurs in any epoch, the trainer will crash with a `KeyError` that is extremely difficult to diagnose without knowing about this inconsistency. The fix is simple and must be applied in Phase 3 before Run 5.

> **NF-9 note (BF16 prefix, not added):** The finding about BF16 prefix tensor precision loss (Finding 10 in the source document) was not confirmed by code inspection. The prefix tensor is allocated with `dtype=node_embs.dtype`, and the projection/type_embedding operations share the same dtype, so no cross-dtype truncation occurs in the current code. The known BF16 quantization floor on `gnn_to_bert_proj` is a separate issue (already documented in §14.3 as a Run 6 candidate).

---

### Root Cause Inventory (RC1–RC7)

| RC | Root Cause | Category | Fix in Run 5? | Expected Impact |
|----|-----------|----------|---------------|----------------|
| RC1 | FUNCTION nodes get identity transform from Phase 2 (zero CF edges incident on FUNCTION nodes → residual passes them through unchanged) | Architecture | Partial (aux loss on CEI pooling may not fix pooling mismatch) | Partial |
| RC2 | `aux_phase2_loss_weight = 0.0` throughout all of Run 4 — no auxiliary supervision pushing Phase 2 toward class-discriminative structure | Training | **YES — `aux_phase2_loss_weight = 0.10`** (already in TrainConfig) | **Direct** |
| RC3 | 8× head capacity gap (Phase 2 heads=1 vs Phase 1 heads=8) — single head cannot learn selective message routing | Architecture | NO — out of scope for Run 5 | None |
| RC4 | JK entropy regularizer pushes Phase 2 weight below 1/3 — FUNCTION nodes have identical Phase 1 and Phase 2 embeddings, so linear attention cannot differentiate them, and entropy pressure pushes both to 1/3 | Training | Partial — aux loss may shift equilibrium; λ=0.005 unchanged | Partial |
| RC5 | DEF_USE edges get only 1 hop (single layer, Layer 5 only) — meaningful data-flow reasoning requires chaining | Architecture | NO — architecture fixed for Run 5 | None |
| RC6 | Phase 3 does Phase 2's job via REVERSE_CONTAINS — Phase 3 lifts CFG information to FUNCTION level, making Phase 2's contribution redundant by design | Architecture | NO — by design; aux loss targets CEI pooling instead | None |
| RC7 | Phase 2 learned Reentrancy suppression — structural ablation deltas are positive (removing Phase 2 edges *increases* Reentrancy prediction), meaning Phase 2 actively suppresses Reentrancy | Training | Unknown — aux loss may rebalance the suppression signal | Unknown |

---

## 3. Execution Order Rationale

Fixes must be applied in strict dependency order. Violating this order produces incorrect results that require re-work:

```
Phase 0 (safety — prevent data poisoning & optimizer corruption)
    → Phase 1 (schema correctness — validate normalization & toolchain)
        → Phase 2 (extractor correctness — fix graph structure bugs)
            → Phase 3 (model correctness — fix architecture & performance bugs)
                → Phase 4 (trainer correctness — fix training loop bugs)
                    → Phase 5 (Phase 2 signal interventions — aux losses & monitoring)
                        → Phase 7 (re-extraction + CEI labeling + full data archival/migration)
                            → Phase 8 (Run 5 — execute training with all fixes in place)
                                → Phase 6 (calibration — post-training inference fix)
```

> **Note on Phase 6 placement:** Phase 6 (Calibration & Threshold Fixes) is listed in numerical order for organizational clarity, but it is **executed after Phase 8 (Run 5 training)**. Temperature scaling is a post-training, inference-time fix that does not affect training dynamics. It must be re-fitted on Run 5's checkpoint — Run 4 temperatures must not be reused. Phases 7 and 8 do NOT depend on Phase 6.

**Why Phase 7 must come after Phase 5:** Re-extracting with a broken extractor (before Phase 2 fixes) would produce incorrect graphs, requiring re-extraction again. Phase 7 must happen after all graph extraction, model, and trainer fixes are in place. Additionally, CEI path labels (Intervention 2) must be computed on the re-extracted v9 graphs — not on the buggy v8 graphs — so CEI labeling is integrated into Phase 7. Similarly, `max_nodes` increase (IMP-D1, part of Phase 5) must be reflected in the extractor before re-extraction.

**Why Phase 0 must come before everything else:** A20 (label=0) can silently poison all training labels. A38 (NaN before check) can permanently corrupt Adam momentum buffers. Applying any other fix while these are active means the training data or optimizer state may already be corrupted, making all other fixes meaningless.

**Why all previous data must be archived before Phase 7:** After every code fix is applied, all v8-era artifacts (graphs, tokens, caches, checkpoints, label files, splits, index files) must be moved to a clearly-labeled archive directory. This prevents accidental use of stale, buggy data during or after re-extraction. Run 5 must train exclusively on v9 data produced after all fixes.

---

## 4. Phase 0 — Critical Pre-Flight Safety Fixes

These two bugs can corrupt training data or permanently corrupt optimizer state. They must be fixed before touching any other code. No other work should proceed until both are resolved and their gates pass.

### 4.1 A20 — `label=0` Hardcoded in Batch Extraction (DATA POISONING)

**Severity: HIGH — confirmed training data corruption risk**
**File:** `ast_extractor.py` lines 307–311

**Root Cause:** The multiprocessing worker uses `partial(self.contract_to_pyg, ..., label=0)`, which always assigns label 0 to every contract regardless of actual vulnerability status. This means that any batch extraction performed with the current code produces all-zero labels — effectively poisoning the entire dataset with "all safe" labels.

**Fix Approach:** Refactor the worker to accept `label` as a per-call argument rather than a `partial` fixed parameter. Construct an args list that pairs each contract path with its ground-truth label from a `label_map` populated from the ground-truth CSV before any pool worker is spawned.

**Gate 0.1 (CRITICAL — blocks all subsequent work):**
- Verify that `label_map` is populated from the ground-truth CSV before any pool worker is spawned
- Assert `len(label_map) == len(batch)` before the `pool.map` call
- After the first test extraction run, verify that labels in the output .pt files match the source CSV (spot-check at least 100 contracts)

**Rollback Decision Tree:**
- If `label_map` cannot be built (CSV missing/corrupted) → Stop. Restore CSV from version control. Do not proceed with any extraction.
- If spot-check reveals label mismatches after the fix → The fix implementation is wrong. Revert the worker change, debug the `label_map` construction, and re-test before proceeding.

### 4.2 A38 — NaN Loss `backward()` Runs Before NaN Check (CORRUPTS ADAM STATE)

**Severity: MEDIUM-HIGH — NaN gradients permanently corrupt Adam momentum buffers**
**File:** `trainer.py` line 650 (`loss.backward()`) vs line 713 (NaN check)

**Root Cause:** The `loss.backward()` call happens at line 650, but the `torch.isfinite(loss)` check does not happen until line 713. When a NaN loss occurs, `backward()` computes NaN gradients, which are then passed to `optimizer.step()`. This corrupts the `m1` (first moment) and `m2` (second moment) buffers in Adam for all affected parameters. The corruption is permanent within a run — even subsequent non-NaN batches will propagate corrupted moments because Adam's update rule uses exponential moving averages of these corrupted values.

**Fix Approach:** Move the finite check to *before* `backward()`. If loss is not finite, skip both `backward()` and `optimizer.step()` entirely, zero out any stale gradients, and increment a counter. Additionally, add a post-clipping, pre-step guard that checks whether any parameter has non-finite gradients despite a finite loss (this can happen with BF16 overflow in intermediate activations) — if so, zero the gradients and skip the step.

**Gate 0.2 (CRITICAL — blocks all training):**
- Add a `nan_loss_count` summary log at the end of each epoch
- If `nan_loss_count > 0.5% × steps_per_epoch`, halt training immediately — it indicates systematic instability that requires investigation before continuing
- Verify in a short smoke test (5 steps) that NaN losses are correctly caught and skipped without crashing

**Rollback Decision Tree:**
- If `nan_loss_count > 0` but < 0.5% threshold → Log warning, continue training, but investigate the root cause of NaN occurrences.
- If `nan_loss_count > 0.5% × steps_per_epoch` → **Halt training immediately.** This indicates systematic instability. Investigate: check learning rate, check for BF16 overflow in new code paths, check for data quality issues (infinite values in input features). Do not resume until the NaN source is identified and resolved.
- If `nan_loss_count > 0` and Adam state may already be corrupted → The safest action is to restart training from the last clean checkpoint (before NaN occurred), not to continue from the current checkpoint.

---

## 5. Phase 1 — Data & Schema Layer Fixes

These fixes address schema correctness, hash validation, toolchain reliability, and data persistence robustness. They must be applied before any extraction work.

### 5.1 A1 — Missing `max(NODE_TYPES)` Range Guard

**File:** `graph_schema.py`
**Bug:** `assert len(NODE_TYPES) == 13` guards the count but not the maximum value. A new node type with id=13 inserted without updating the normalization divisor (`/12.0`) would produce `type_id_norm > 1.0` silently.
**Fix Approach:** Add an assertion that `max(NODE_TYPES.values()) == 12` after the existing count assertion. This ensures any schema extension is caught immediately with an actionable error message directing the developer to update the normalization divisor.

### 5.2 A2 — Uppercase Hex Accepted in Hash Validation

**File:** `hash_utils.py`
**Bug:** `validate_hash` uses `int(hash_string, 16)` which accepts uppercase A–F. All hashes produced by `hashlib.md5().hexdigest()` are lowercase. This creates silent permissiveness where uppercase hex from an external source passes validation but fails deduplication.
**Fix Approach:** Replace the `int(,16)` approach with a strict lowercase hex regex match: `[0-9a-f]{32}`. Also enforce string type and length checks before the regex.

### 5.3 A3 + A32 — Dynamic `_MAX_TYPE_ID` in Two Files

**Files:** `graph_extractor.py` line 113, `sentinel_model.py` line 75
**Bug:** `_MAX_TYPE_ID = float(max(NODE_TYPES.values()))` is computed dynamically. The comment in sentinel_model.py says `# 12.0 for v2 schema (ids 0–12)` but no assertion enforces this. If NODE_TYPES gains a new entry, normalization silently changes across both files.
**Fix Approach:** Add an assertion immediately after the dynamic assignment in both files: `assert _MAX_TYPE_ID == 12.0`. This makes the "dynamic" assignment self-documenting and safe — if someone adds a node type, they get an immediate, actionable error rather than silent normalization drift.

### 5.4 A19 — Solc Binary Resolution CWD-Dependent

**File:** `ast_extractor.py` line 143
**Bug:** `venv_path = Path.cwd() / ".venv" / ...` makes solc binary resolution dependent on the current working directory. The `get_project_root()` helper exists in the same file but is not used.
**Fix Approach:** Replace `Path.cwd()` with `get_project_root()` for deterministic path resolution.

**Gate 1.1 — Toolchain Check (blocks extraction):**
- Verify `solc` availability: `which solc && solc --version`
- Verify `solc-select versions` shows the required range (0.4.0–0.8.35)
- If solc-select is absent, install and configure before proceeding

### 5.5 A21 — Worker `print()` Under Concurrency

**File:** `ast_extractor.py` lines 223, 228
**Bug:** Raw `print()` from `mp.Pool` workers produces interleaved, illegible output under concurrency.
**Fix Approach:** Replace both `print()` calls with `logger.warning()`. Worker processes share the Python logging subsystem through a `QueueHandler`/`QueueListener` pair (already in place), so log messages are serialized correctly.

### 5.6 NF-2 — `_add_node` Reverse-Normalization Hardcodes `12`

**File:** `graph_extractor.py` line 1094
**Bug:** `actual_type_id = int(round(x_list[-1][0] * 12))` hardcodes `12` as the normalization divisor on the decode side. This is the decode-side counterpart of A3/A32. If a new node type is added (max type_id becomes 13), the metadata type name for the new type will be silently misread — it will look up type_id 12 instead of 13.
**Fix Approach:** Replace the hardcoded `12` with `_MAX_TYPE_ID` (which is the same constant guarded by the assertion in A3). Since A1's fix adds `assert max(NODE_TYPES.values()) == 12.0`, the value is always 12 — but the decode must also use the named constant so that any future schema change triggers a clear error rather than silent corruption.

### 5.7 A22 — `torch.save` Without Error Handling

**File:** `ast_extractor.py` line 328
**Bug:** A disk-full or I/O error during `torch.save()` aborts the entire batch loop, losing up to 499 unwritten graphs.
**Fix Approach:** Wrap `torch.save()` in a `try/except (OSError, IOError)` block. On failure, log the error, append the file to a `failed_saves` list, and continue processing. At the end of the batch, if any saves failed, raise an exception with the full list so the caller can decide whether to retry or abort.

---

## 6. Phase 2 — Graph Extraction Layer Fixes

These fixes address correctness of the graph structure that the model trains on. They must all be applied before Phase 7 (re-extraction). The extraction layer has the highest bug density (16 confirmed bugs + 5 new findings) and the most impactful silent-failure points.

### 6.1 A4 + A16 — `assert` Used for Production Invariants (Silent Under `-O`)

**File:** `graph_extractor.py` lines 1253–1257 (A4), 856–857 (A16)
**Bug:** `assert` is silently removed by `python -O`. Critical invariants — node/metadata alignment (A4) and sentinel value range (A16) — are unguarded in production mode.
**Fix Approach:** Replace all production-critical `assert` statements with explicit `if`/`raise ValueError` checks that cannot be optimized away. A4 becomes a node-metadata-length vs. tensor-shape check. A16 becomes a sentinel value range validation for `return_ignored` and `call_target_typed`.

### 6.2 A5 — `except AttributeError` Scope Too Broad

**File:** `graph_extractor.py`
**Bug:** The `except AttributeError` block covers the entire function body of `_compute_return_ignored`. A refactoring-induced `AttributeError` from any inner expression would be swallowed, returning `-1.0`. (The catch does log a warning — it is not fully silent — but the scope is still wrong.)
**Fix Approach:** Narrow the `try` block to only the specific Slither API calls that are expected to raise `AttributeError` (specifically `func.calls_as_expression`). All other logic should execute outside the `try` block so that unexpected AttributeErrors propagate normally.

### 6.3 A6 + A10 + A18 — Bare `except Exception: pass` (Three Critical Silent-Failure Points)

**Files:** `graph_extractor.py` lines 312–313 (A6), 493–494 (A10), 1160–1173 (A18)

These are the three most impactful silent-failure points in the entire codebase:

- **A6 (call target resolution):** Bare `except Exception: pass` — Slither API changes silently fall through to the regex scan without any diagnostic.
- **A10 (CFG node type classification):** Bare `except Exception: pass` — Any Slither API change silently makes every CFG node `CFG_NODE_OTHER`, destroying all per-node type signal.
- **A18 (ICFG map construction):** Bare `except Exception: pass` — When ICFG map construction fails, all `CALL_ENTRY`/`RETURN_TO` edges for all callers of the affected function become absent. This is catastrophic for cross-function vulnerability detection.

**Fix Approach:** Replace all three `pass` statements with structured logging and metric counters:
- A6: Log at debug level (type resolution failure is expected and has a fallback to source scan)
- A10: Log at warning level with a per-contract fallback counter `_cfg_type_fallback_count`
- A18: Log at error level with a per-contract failure counter `_icfg_failure_count`

**Gate 2.1 — Extraction Health Check (blocks Run 5):**
After re-extraction (Phase 7), validate the new dataset against these criteria:
- `_icfg_failure_count == 0` (ICFG map construction never failed)
- `_cfg_type_fallback_count / total_cfg_nodes < 0.01` (less than 1% of CFG nodes falling back to OTHER)
- CALL_ENTRY edge presence rate ≥ 64.2% (baseline from cache audit)
- RETURN_TO edge presence rate ≥ 55.6% (baseline)

If any check fails, the extraction bugs are not fully resolved — do not proceed to Run 5.

### 6.4 A7 — Dead Code `_compute_in_unchecked`

**File:** `graph_extractor.py` lines 331–360
**Fix Approach:** Replace the function body with a `raise NotImplementedError` tombstone that documents when it was deprecated (v7, BUG-L2) and warns that any call site was not updated. This is safer than deletion because it surfaces forgotten call sites immediately rather than silently failing.

### 6.5 A8 — `is True` Identity Check in `_compute_has_loop`

**File:** `graph_extractor.py` line 376
**Bug:** `getattr(func, "is_loop_present", None) is True` misses integer `1` or any truthy non-boolean. Slither returns truthy integers for some AST properties.
**Fix Approach:** Replace `is True` with `bool(...)` coercion to handle all truthy values consistently.

### 6.6 A9 — String-Based Class Check for `SolidityVariableComposed`

**File:** `graph_extractor.py` line 424
**Bug:** `type(rv).__name__ == "SolidityVariableComposed"` breaks silently on any Slither class rename. This is the gating check for `uses_block_globals` — the feature that distinguishes Timestamp and TOD contracts. A Slither rename would zero out this feature for all contracts, destroying Timestamp/TOD discriminative signal entirely.
**Fix Approach:** At module import time, attempt to import `SolidityVariableComposed` directly from Slither. If the import succeeds, use `isinstance` checks against the imported class. If the import fails, log a prominent warning that `uses_block_globals` will always be 0.0 and that Timestamp/TOD detection will be severely degraded.

**Gate 2.2 — Feature Validation (blocks Run 5):**
After applying this fix, verify that `uses_block_globals` is non-zero for at least 80% of Timestamp-positive contracts in the validation split. If this threshold is not met, the feature is still broken.

### 6.7 A11 — Hardcoded Parent Feature Indices in `_build_cfg_node_features`

**File:** `graph_extractor.py` lines 542–547
**Bug:** Raw integer indices (`p[1]`, `p[3]`, `p[4]`, `p[5]`, `p[9]`) are used to copy parent FUNCTION features into CFG nodes. Any feature reordering silently inherits the wrong feature. The code comment even acknowledges an earlier index shift from v6 to v7.
**Fix Approach:** Replace hardcoded indices with a schema-driven lookup using `FEATURE_NAMES` (the canonical ordered list from `graph_schema`). Build a name→index mapping and look up features by name (visibility, view, payable, complexity, has_loop). This makes feature inheritance self-maintaining — any schema reorder is automatically reflected.

### 6.8 A12 — `n.node_id` Without Fallback in Sort Key

**File:** `graph_extractor.py` lines 606–611
**Bug:** `n.node_id` accessed directly in a `sorted()` key function — no `getattr` default. Synthetic nodes without `node_id` raise `AttributeError` inside the sort.
**Fix Approach:** Replace `n.node_id` with `getattr(n, "node_id", 0)` for a safe fallback.

### 6.9 A13 — Silently Dropped CONTROL_FLOW Edges Not Logged

**File:** `graph_extractor.py` lines 639–641
**Bug:** CF successors not in `node_index_map` are silently dropped with no logging or counting.
**Fix Approach:** Add an `else` branch that increments a per-contract counter and logs at debug level. Log a summary per contract with the total dropped count at info level.

### 6.10 A14 — RETURN_TO Cartesian Product Includes Revert Paths

**File:** `graph_extractor.py` lines 695–706
**Bug:** All `callee_terminals` (including revert/throw nodes) are connected to all `call_site_sons` via RETURN_TO edges. Revert terminals should not produce RETURN_TO edges because control never actually returns to the call site from a revert — it unwinds the call stack entirely. This injects spurious edges that tell the model "control returns here after a revert," which is semantically incorrect.
**Fix Approach:** Filter `callee_terminals` to normal-return terminals only, excluding THROW and RETURN node types. Only normal terminals should produce RETURN_TO edges.

**Important:** This changes the edge structure in extracted graphs. Re-extraction (Phase 7) is required before the model sees the corrected RETURN_TO edges.

### 6.11 A15 — DEF_USE `def_map` Keyed by Variable Name Only (Scope Collision)

**File:** `graph_extractor.py` line 752
**Bug:** `def_map.setdefault(lval.name, [])` uses variable name alone as the key. Variable name shadowing in nested scopes (e.g., `uint x` in two different functions, or a local `balance` shadowing a state variable) produces spurious cross-scope DEF_USE edges that connect nodes that have no actual data-flow relationship.
**Fix Approach (two-tier scope key):** Key by `(scope_id, variable_name)` where `scope_id` is determined by the variable's declaration level:
- **Local variables** (declared inside a function): Use the containing `Function` object as `scope_id`. This prevents cross-function name collisions — a local `balance` in function A won't be connected to a local `balance` in function B.
- **State variables** (declared at the contract level): Use the containing `Contract` object as `scope_id`. State variables *should* have cross-function DEF_USE edges — a state variable written in function A and read in function B is a legitimate cross-function data flow that the model needs to see.
- To determine whether a variable is local or state-level: check if the variable appears in `contract.state_variables` (Slither provides this). If it does, use contract scope; otherwise, use function scope.

**Implementation note:** When building the `use_map` (the lookup side), the same scope resolution must apply — a reference to a state variable from within a function must look up `(contract, variable_name)`, not `(function, variable_name)`.

**Important:** This changes the DEF_USE edge structure. Re-extraction (Phase 7) is required.

### 6.12 A17 — Exception Routing by String Keyword Matching

**File:** `graph_extractor.py` lines 1059–1067
**Bug:** Slither/solc error categorization via `kw in exc_lower` is fragile in both directions — new Slither error messages are mis-categorized as ParseErrors, and real compilation errors with unusual phrasing fall through to the wrong exception type.
**Fix Approach:** Restructure exception handling to use type-based checks first (catch `SlitherError` and `SolcError` by type), and only fall back to string keyword matching for untyped `Exception` instances. Add a tracked TODO to replace string matching with `isinstance` checks on Slither's exception hierarchy in a future iteration.

### 6.13 NF-1 — EMITS Edge Key Mismatch in Fallback Path

**File:** `graph_extractor.py` lines 1305–1317
**Bug:** The BUG-H7 fallback path (for Solidity <0.4.21 where `events_emitted` is empty or raises) retrieves event names via `getattr(ir, "name", None)` on `EventCall` IR objects. These are *short names* (e.g., `"Transfer"`). However, event nodes are registered in `node_map` using `canonical_name` (e.g., `"ERC20.Transfer"`). The `_add_edge(fn, key, EDGE_TYPES["EMITS"])` call therefore always fails to find the event node in `node_map` for the fallback path, silently dropping all EMITS edges for contracts using old-style event emission.
**Fix Approach:** In the fallback loop, resolve the full canonical key by looking up the event in the contract's `events` list and using `canonical_name`. A simple approach: build a `short_name → canonical_name` map from `contract.events` before the fallback loop and use it to translate `ir.name` to the canonical form before calling `_add_edge`.

**Important:** This changes EMITS edge presence for Solidity <0.4.21 contracts. Re-extraction required (Phase 7).

### 6.14 NF-7 — `_compute_external_call_count` and `_compute_uses_block_globals` Return `0.0` on Exception

**File:** `graph_extractor.py` lines 394–405 and 419–432
**Bug:** Both functions have a bare `except Exception: ... return 0.0` (implicit via `pass; return 0.0`). When Slither API changes cause the computation to raise (e.g., `func.high_level_calls` attribute renamed or `func.nodes` absent), the function returns `0.0` — indistinguishable from "zero external calls" or "no block globals". This creates silent false-negatives that suppress DoS and Timestamp/TOD features without any diagnostic.
**Fix Approach:** Add a per-contract warning log inside the `except` block for each function. Add per-extraction counters (`_ext_call_fail_count`, `_block_globals_fail_count`) and log the totals in the extraction summary. Do NOT change the return value to `-1.0` — the model is not trained to handle sentinel values for these features and doing so would break inference on all previously-extracted contracts.

### 6.15 NF-10 — Duplicate Function Name Attaches Second Function's CFG to First

**File:** `graph_extractor.py` lines 1134–1148
**Bug:** When `_add_node(func, ...)` returns `None` (duplicate canonical name — e.g., inherited function + overriding function share the same canonical_name), the code correctly looks up the existing `fn_idx`. However, it then builds CONTAINS edges from that `fn_idx` to the *new function's* CFG nodes, and passes `parent_features=x_list[fn_idx]` (the first function's features) to the CFG builder. The result is that both functions' CFG nodes are attached as children of the same FUNCTION node, with the first function's feature vector used as the parent context for the second function's CFG. This conflates two distinct CFGs and propagates wrong parent features.
**Fix Approach (synthetic key as primary, skip as fallback):** When `_add_node` returns `None` for a function, assign it a unique synthetic key (e.g., `canonical_name + "__override__" + str(index)`) and re-attempt `_add_node` with the synthetic key. This preserves both functions' CFGs, which is critical because overriding functions often introduce the vulnerability. If edge density analysis shows the second function's CFG is identical to the first (a degenerate case), fall back to skipping CFG construction for the second function. Increment a `_duplicate_func_count` counter logged at the end of extraction regardless of which path is taken.

**Rationale for synthetic key over skip:** Inherited/overriding functions frequently have *different implementations* — the override may introduce the vulnerability (e.g., an overridden `withdraw` that adds a reentrancy). Skipping the second function's CFG loses this signal entirely. The synthetic key approach preserves it at the cost of one extra FUNCTION node per duplicate, which is negligible.

**Important:** This changes graph structure for contracts with overloaded/inherited functions. Re-extraction required (Phase 7).

### 6.16 NF-11 — `_add_edge` Silently Drops ALL Edge Types

**File:** `graph_extractor.py` lines 1274–1279
**Bug:** The `_add_edge` helper silently drops any edge where either endpoint is absent from `node_map`. A13 already addresses CONTROL_FLOW edges by adding a counter and log. However, `_add_edge` is also called for ALL other edge types (CALLS, READS, WRITES, EMITS, INHERITS) with no per-type diagnostics. Dropped CALLS edges (functions calling unknown functions) and dropped READS/WRITES edges (access to unregistered state variables) are completely invisible.
**Fix Approach:** Extend the A13 fix: add per-type drop counters inside `_add_edge` itself. Add a `_edge_drop_counts: dict[int, int]` accumulator (keyed by edge type) in the outer scope, increment it inside `_add_edge` when an edge is dropped, and log the per-type summary at the end of extraction. This gives complete visibility into edge loss across all types.

---

## 7. Phase 3 — Model Architecture Fixes

These fixes address correctness and performance of the model's forward pass. They do not change the training dynamics but fix bugs that produce incorrect outputs, waste computation, or create fragile dependencies.

### 7.1 A25 — `edge_index.max()` O(E) Scan on Every Forward Pass

**File:** `gnn_encoder.py` lines 389–393
**Bug:** A full tensor scan (`edge_index.max()`) on every forward call to validate that no edge index exceeds the number of nodes. This is a safety check that belongs at data-loading time, not in the inference hot path.
**Fix Approach:** Move the integrity check to `DualPathDataset.__getitem__` or the collation function where graphs are loaded from disk. At inference time, disable this validation (or gate it behind a `validate_graph_integrity` flag defaulting to `False` in production).

### 7.2 A26 — `next(self.parameters())` Called Twice Per Forward Pass

**File:** `gnn_encoder.py` lines 398, 521
**Bug:** Two separate `next(self.parameters())` generator constructions per forward pass to determine the model's dtype.
**Fix Approach:** Cache `self._param_dtype` in `__init__`. In forward, read from the cached value. Add a `refresh_dtype_cache()` method that callers must invoke after any runtime dtype cast (`.float()`, `.half()`, `.bfloat16()`).

### 7.3 A27 — `num_layers` Stored but Hardcoded to 8

**File:** `gnn_encoder.py` line 196
**Bug:** The `num_layers` parameter is stored as metadata but has no effect on construction — the architecture is hardcoded to 8 layers. This misleads checkpoint compatibility checks and wastes a parameter.
**Fix Approach:** Make the hardcoding explicit with a constant `SENTINEL_GNN_NUM_LAYERS = 8`. If `num_layers` is passed with any value other than 8, raise `ValueError` explaining that the architecture is fixed and the parameter is for documentation only.

### 7.4 A23 — `last_weight_stds` NaN for N=1

**File:** `gnn_encoder.py` line 123
**Bug:** `.std(0)` without `unbiased=False` returns NaN for N=1 (single sample in JK weight batch). The comment claims it returns 0 — incorrect.
**Fix Approach:** Use `.std(0, unbiased=False)` followed by `.nan_to_num(0.0)` to produce the documented behavior (0 for single-sample batches).

### 7.5 A28 — `except (ImportError, ValueError)` Catches Real BERT Load Errors

**File:** `transformer_encoder.py` lines 142–147
**Bug:** `ValueError` from a corrupted `config.json` or missing model files silently falls through to the SDPA retry path, masking the real error and making debugging extremely difficult.
**Fix Approach:** Only catch `ImportError` (flash_attention_2 not installed) for the fallback to SDPA. Let `ValueError` propagate to the caller as a real configuration error that must be fixed.

### 7.6 A29 — Python Loop for Prefix Mask Construction (Vectorize)

**File:** `transformer_encoder.py` lines 241–242 and 284–285 (two occurrences)
**Bug:** A Python `for b in range(B)` loop constructs the prefix attention mask one batch element at a time. This is slow and unnecessary.
**Fix Approach:** Vectorize using a broadcast comparison: create an `arange` tensor and compare against `gnn_prefix_counts` to produce the full mask in a single tensor operation. Apply to both occurrences.

### 7.7 A30 — `_word_embeddings` Fragile Hardcoded PEFT Path

**File:** `transformer_encoder.py` lines 168–170
**Bug:** `self.bert.base_model.model.embeddings.word_embeddings` is a five-level path into PEFT internals that may change between PEFT versions. It is not validated at `__init__` time — failure surfaces at the first forward pass, which is the worst possible time.
**Fix Approach:** Replace the hardcoded path with a property that tries multiple known PEFT internal paths in order of precedence. If none produce an `nn.Embedding`, raise `AttributeError` with a clear message about PEFT version incompatibility. Additionally, validate the path at `__init__` time (call the property and catch the error) to surface problems at construction rather than at the first forward pass.

### 7.8 A32 — `_MAX_TYPE_ID` Decoupled in `sentinel_model.py`

Already covered by fix 5.3 (A3 + A32). Confirm the assertion is added to both files.

### 7.9 A33 — `select_prefix_nodes` Python Loop Over Batch Dimension

**File:** `sentinel_model.py` line 305
**Bug:** Python loop with `.tolist()`, list comprehension, and `sort()` inside each graph iteration. For a batch of 32 graphs with 256 nodes each, this is 32 Python iterations with nested sorting — a significant performance bottleneck.
**Fix Approach:** Pre-compute priority scores for all nodes in a single tensor operation, then use `torch.topk` per-graph using the PyG batch vector. A fully vectorized version requires a custom scatter-topk kernel; for Run 5, a hybrid approach (tensor scores + looped topk) is a significant improvement.

### 7.10 A34 — Secondary Sort Uses Post-GAT Embedding, Not Raw Feature

**File:** `sentinel_model.py` line 326
**Bug:** `g_embs[local_idx, _EXT_CALL_DIM]` uses the 256-dim post-GAT output. After 8 GAT layers, dimension 10 encodes a learned mixture of neighborhood features — not `external_call_count`. The prefix selection secondary sort is semantically wrong; it sorts by a meaningless learned representation instead of the actual feature it intends to use.
**Fix Approach:** Access raw input features (`graphs.x`) instead of post-GAT embeddings for the secondary sort. This ensures the secondary sort actually sorts by `external_call_count` as intended.

**Implementation note:** Raw features are available as `graphs.x` (the PyG Data object's node feature tensor). This must be passed to `select_prefix_nodes` as an additional argument, since the current code only receives `g_embs` (post-GAT). The function signature must be updated to accept both `g_embs` and `graphs` (or specifically `graphs.x`), and the secondary sort must read from the raw feature tensor.

**Important:** This changes the prefix node selection behavior. Re-run EXP-A4 (Aux Eye Contribution) after this fix to verify that prefix quality improves.

### 7.11 A25b — `compute_prefix_attention_mean` Discards `node_counts`

**File:** `sentinel_model.py` lines 544–546
**Bug:** `gnn_prefix, _ = gnn_prefix` discards `node_counts` — the diagnostic forward pass averages attention over all K=48 positions including padding, understating per-class prefix attention.
**Fix Approach:** Unpack the tuple to retain `node_counts`. When averaging attention, only average over real node positions (up to `node_counts[g]`) rather than over all K=48 positions. This is a diagnostic-only fix with no training impact.

### 7.12 NF-6 — Phase 2 Layers 3 and 4 Ignore `phase2_edge_types` Ablation

**File:** `gnn_encoder.py` lines 476–480
**Bug:** `cf_only_ei` (Layer 3) and `icfg_only_ei` (Layer 4) are computed directly from raw `edge_attr` comparisons, unconditionally. When `phase2_edge_types` ablation is active, only `phase2_ei` (used by Layer 5) respects the configured edge subset. Layers 3 and 4 always use the full CFG and ICFG edges regardless of ablation config.
**Fix Approach:** Compute `cf_only_ei` and `icfg_only_ei` from the ablated `phase2_ei` tensor (already masked by `phase2_edge_types`) rather than from raw `edge_attr`. Specifically: after computing `phase2_ei`, derive Layer 3 and Layer 4 subsets by applying type-specific masks to `phase2_ei` rather than to the unfiltered `edge_attr`.

> **Run 5 training impact:** Zero — `phase2_edge_types` is not set during normal training. This bug only manifests in ablation experiments. However, since interpretability experiments (EXP-E4 etc.) rely on ablation correctness, this must be fixed before any post-Run-5 interpretability measurements.

### 7.13 NF-8 — Empty Batch Guard Returns Inconsistent Aux Dict (ELEVATED TO MEDIUM)

**File:** `sentinel_model.py` lines 422–426
**Bug:** The empty-batch guard returns `aux_zeros = {"gnn": ..., "transformer": ..., "fused": ...}`, but the normal forward path returns `{"phase2": ..., "jk_entropy": ...}`. The two dicts have completely disjoint key sets. If the trainer accesses `aux["phase2"]` (expected for aux Phase 2 loss computation), it will get a `KeyError` on any epoch that happens to encounter an empty batch.

**Why this is MEDIUM (elevated from Low):** Run 5 will compute auxiliary Phase 2 loss by accessing `aux["phase2"]`. If even a single empty batch occurs during training — which can happen with certain sampler configurations, at epoch boundaries, or with small validation sets — the trainer will crash with a `KeyError`. This crash would be extremely difficult to diagnose without knowing about this key set inconsistency.

**Fix Approach:** Update `aux_zeros` to include `"phase2"` and `"jk_entropy"` keys (zero tensors of the appropriate shapes). The existing `"gnn"`, `"transformer"`, `"fused"` keys are not in the normal-path dict either — remove them from `aux_zeros` or add them to the normal-path dict (whichever matches the trainer's expected key set). Add a unit test that verifies the trainer can handle an empty batch without `KeyError`.

**Gate 3.1 — torch.compile Re-Validation (blocks Run 5):**
After all Phase 3 fixes are applied (A33 vectorization, A29 vectorization, NF-8 dict key changes, A25b unpacking changes, A34 signature change), run a 2-epoch smoke test with `torch.compile(model, dynamic=True)` enabled. If it raises `RuntimeError` or produces incorrect outputs, disable `torch.compile` for Run 5 and file as a Run 6 fix item. The forward pass structure has changed significantly and the compiled graph may be stale.

---

## 8. Phase 4 — Training Loop Fixes

These fixes address correctness and efficiency of the training loop. They do not change the model architecture but fix bugs that waste computation or create fragile code.

### 8.1 A35 — `_FocalFromLogits` Unpicklable Local Class

**File:** `trainer.py` lines 1066–1069
**Bug:** Locally-defined class inside `train()` cannot be pickled. Currently safe (never pickled), but incompatible with DDP or any checkpoint scheme that serializes the loss function.
**Fix Approach:** Move the class definition to module level (outside any function). This makes it picklable and compatible with distributed training frameworks.

### 8.2 A36 — `compute_pos_weight` Re-Reads Label CSV Every Call

**File:** `trainer.py` lines 378–388
**Bug:** Reads ~44K rows from CSV on every call. `DualPathDataset` already holds the same data in memory, making the CSV read entirely redundant.
**Fix Approach:** Change the function signature to accept the `DualPathDataset` directly and compute pos_weight from in-memory labels — no CSV I/O needed.

### 8.3 NF-4 — `--gnn-layers` CLI Default = 7, TrainConfig Default = 8

**Severity: HIGH — silently produces wrong architecture**
**File:** `scripts/train.py` line 151

**Root Cause:** `p.add_argument("--gnn-layers", type=int, default=7)` — the CLI default is 7. `TrainConfig.gnn_num_layers` defaults to 8. Run 4 used 8 layers. Running `python train.py` for Run 5 without an explicit `--gnn-layers 8` argument will produce a 7-layer GNN, silently changing the architecture and making the run incomparable to Run 4.
**Fix Approach:** Change the CLI default to `8`. Add an assertion in `train.py` that `args.gnn_layers == 8` (with an explicit override flag `--allow-gnn-layers-change`) so that any accidental architecture deviation is immediately visible.

**Gate NF-4 (blocks Run 5 launch):** Before launching Run 5, verify the launched process logs `gnn_num_layers=8` in its configuration summary. If the log shows `7`, halt immediately.

### 8.4 NF-9 — `AdamW(fused=True)` Crashes on CPU

**File:** `trainer.py` line 1177
**Bug:** `optimizer = AdamW(_param_groups, weight_decay=config.weight_decay, fused=True)` unconditionally passes `fused=True`. On a CPU-only machine this raises `RuntimeError: Fused AdamW requires CUDA`. Non-blocking on the RTX 3070 setup, but fragile.
**Fix Approach:** Replace `fused=True` with `fused=(device.type == "cuda")`. This is a one-line fix with zero training impact.

### 8.5 A37 — Threshold Sweep O(N×C×19) Every Validation Epoch

**File:** `trainer.py` lines 477–490 and 1493
**Bug:** `tune_thresholds=True` is hardcoded in the training loop, causing a full N×C×19 sweep every validation epoch. This is BUG-M8 (the author is already aware). The sweep is expensive and mostly redundant between adjacent epochs.
**Fix Approach:** Only tune thresholds at configurable intervals (default: every 10 epochs and once at the final epoch). During intermediate epochs, reuse the thresholds from the last tune run. This reduces overhead by ~90% while keeping thresholds current.

---

## 9. Phase 5 — Training Interventions for Phase 2 Signal

These are the interventions that directly address the seven confirmed Phase 2 root causes and target the F1 = 0.3362 ceiling. They are the core value proposition of Run 5 — all the bug fixes in Phases 0–4 remove impediments, but the interventions in Phase 5 are what should actually move the needle.

### 9.1 Intervention 1 — Phase 2 Auxiliary Loss (ALREADY IN TRAINCONFIG)

**Status:** `aux_phase2_loss_weight = 0.10` exists in `TrainConfig` as of commit 9310046. The fix is present in configuration but **must be verified as actually reaching `train_epoch()`**.

**Root Cause Addressed:** RC2 (aux_phase2_loss_weight = 0.0 throughout Run 4)
**Expected Impact:** Direct — provides explicit supervision signal pushing Phase 2 representations toward class-discriminative structure

**Implementation Verification:** The `train_epoch()` function signature must propagate `aux_phase2_loss_weight` from `TrainConfig` to the training loop. The default value in `train_epoch()` must match `TrainConfig`, not remain at 0.0. The auxiliary loss should be computed by passing Phase 2 embeddings through `aux_head_phase2` and adding the weighted BCE loss to the total loss.

**Gate 5.1 (blocks Run 5 continuation):**
- At epoch 1 of Run 5, log `aux_phase2_loss` separately
- If it is 0.0 or absent from logs, the loss weight is not reaching `train_epoch()` — **do not continue Run 5** until this is confirmed non-zero
- Verify that `aux_head_phase2` parameters have non-zero gradient norms by epoch 2
- Log `aux_head_phase2.weight.norm()` and `aux_head_phase2.bias.norm()` per epoch — if these stay at initialization values through epoch 5, the aux loss path has a connectivity bug

### 9.2 Intervention 2 — CEI Path Supervision for Reentrancy

**Status:** Proposed, not yet implemented.

**Root Causes Addressed:** RC1 (partially), RC7 (potentially)
**Expected Impact:** Direct for Reentrancy — provides explicit supervision on the Checks-Effects-Interactions pattern

**Rationale:** EXP-S4 confirmed that 69% of Reentrancy-positive val contracts have a complete `CALL_ENTRY → external_call → RETURN_TO` chain. EXP-L6 confirmed the model cannot detect the CEI pattern on minimal counterfactual contracts (safe score = 0.3032 > vuln score = 0.2962). The model needs explicit supervision to learn this pattern.

**Implementation Plan:**
1. **Generate CEI labels during Phase 7 (re-extraction):** CEI path labeling must be computed on the **re-extracted v9 graphs**, not on the current v8 graphs. The v8 graphs have incorrect RETURN_TO edges (A14), broken ICFG maps (A18), and other structural bugs that would produce wrong CEI labels. The CEI labeler should be implemented in `graph_extractor.py` and run as part of the re-extraction pipeline, not as a separate pre-training step on v8 data.

2. **Label computation:** For each contract, label `has_cei_path = 1` if a `CFG_NODE_CALL → (CONTROL_FLOW → CFG_NODE_WRITE)` path exists reachable via Phase 2 edges within 8 hops. Store as a scalar in the graph object during extraction.

3. **Add auxiliary BCE loss:** Expose Phase 2 CEI pooled embedding from `sentinel_model.py`, compute a binary cross-entropy loss on the CEI logit versus the `has_cei_path` label, and add it weighted by `aux_cei_loss_weight` to the total loss.

**Gate 5.2 — CEI Label Validation (blocks enabling `aux_cei_loss_weight`):**
This gate must be evaluated on the **re-extracted v9 graphs** (after Phase 7), not on v8:
- At least 60% of Reentrancy-positive training contracts should have `has_cei_path = 1`
- No more than 5% of Reentrancy-negative contracts should have `has_cei_path = 1`
- If CEI label coverage is below 40% for positives, the CEI path labeler has a bug — do not enable `aux_cei_loss_weight` until coverage is validated

### 9.3 Intervention 3 — Timestamp Size Confound Regularization

**Status:** Required for honest Timestamp F1 — EXP-S3 confirmed Cohen's d = 1.592 for `cfg_call_count` as a Timestamp shortcut.

**Root Causes Addressed:** RC7 (partially — Timestamp suppression is driven by size confound)
**Expected Impact:** Makes Timestamp F1 honest; may reduce overall macro-F1 but improves model reliability

**Implementation Options:**
- **Option A (recommended for Run 5 — zero code risk):** Size-stratified evaluation — report Timestamp F1 separately for small (total_nodes < 100), medium (100–300), and large (>300) contracts. This does not fix the shortcut but makes it visible in evaluation.
- **Option B (Run 6 candidate):** Adversarial size regularizer — gradient reversal layer that pushes the GNN embedding to be uninformative about contract size. Higher implementation risk.

**Recommendation:** Implement Option A for Run 5. Option B is a Run 6 candidate if Timestamp F1 remains inflated.

### 9.4 Intervention 4 — Raise `max_nodes` to 2048 (IMP-D1)

**Status:** Already documented as IMP-D1.

**Root Causes Addressed:** RC1 (partially — truncation removes vulnerability-relevant subgraphs)
**Expected Impact:** Direct for Timestamp — ~16% of Timestamp-positive contracts exceed the current 1024 limit and are silently truncated

**Implementation Plan:**
- Increase `MAX_NODES` from 1024 to 2048 in `graph_extractor.py` and all dataset configuration
- Update `max_nodes` parameter in `gnn_encoder.py` fusion layer call to match
- This must be done before Phase 7 (re-extraction)

**Memory Impact:** At max_nodes=2048 and batch_size=32, the dense attention matrix in CrossAttentionFusion is 32×2048×2048 = 128M entries. However, this is only the attention matrix — additional memory is consumed by:
- GNN intermediate activations: 8 layers × batch × 2048 × 256 (hidden_dim)
- Gradient storage: same size as activations
- The scatter-to-dense operation in fusion: `batch × 2048 × 256` dense tensor
- Optimizer states (Adam: 2× model parameters)
- Total VRAM usage with gradients and optimizer states can be ~3× the forward-only estimate.

**Gate 5.3 — Memory Gate (blocks Run 5 with max_nodes=2048):**
- Run a **realistic worst-case test**: `max_nodes=2048, batch_size=16, 8 GNN layers, gradient accumulation enabled, with a full training step (forward + backward + optimizer.step)**. This measures true peak VRAM including gradients and optimizer states — a forward-only test underestimates by ~3×.
- If VRAM exceeds 7.5GB with batch_size=16: reduce `batch_size` to 8 or keep `max_nodes=1536` as a compromise.
- Document the chosen configuration in the Run 5 log.
- Also test at batch_size=32 — if it fits, use it for speed; if not, use batch_size=16.

### 9.5 NF-5 — `aux_phase2_loss_weight` Not Exposed as CLI Arg

**File:** `scripts/train.py`
**Bug:** `aux_phase2_loss_weight` is the most important new hyperparameter for Run 5, but it is not exposed in `train.py`'s argument parser. It can only be modified by editing the `TrainConfig` dataclass directly. Any exploratory tuning (0.10 → 0.05 → 0.20) requires a code edit, creating unnecessary friction and making sweep logs ambiguous.
**Fix Approach:** Add `p.add_argument("--aux-phase2-loss-weight", type=float, default=0.10)` and wire it through to `TrainConfig`. Similarly expose `--aux-cei-loss-weight` (currently TBD) and `--jk-entropy-reg-lambda` which are also tunable for Run 5. This is a purely additive change with no training impact.

### 9.6 Monitoring Plan for Phase 2 Signal Recovery

These metrics must be logged each epoch to verify that the Phase 2 interventions are taking effect. Without this monitoring, a failed intervention can waste an entire multi-day training run.

| Metric | Expected Trend by ep10 | Alert Threshold | Action if Breached |
|--------|----------------------|-----------------|-------------------|
| Phase 2 JK weight (mean) | Rising toward 0.35 | Below 0.322 at ep10 | aux loss not working — investigate |
| Phase 2 gradient norm (mean) | Rising relative to Phase 1 | P2/P1 ratio below 72% at ep10 | Phase 2 not receiving gradient — check loss propagation |
| aux_phase2_loss | Decreasing over epochs | Not decreasing by ep5 | Phase 2 not learning — check aux head connectivity |
| **aux_head_phase2.weight.norm()** | **Increasing from init** | **Still at init value at ep5** | **Aux loss path broken — check aux head connectivity** |
| Reentrancy F1 | Rising from 0.169 | Below 0.15 at ep15 | Regression — may need to adjust aux_cei_loss_weight |
| ExternalBug F1 | Rising from 0.000 | Still 0.000 at ep20 | Phase 2 still unused — structural issue |

**Gate 5.4 (Run 5 go/no-go at epoch 10) with Rollback Decision Tree:**

If Phase 2 JK weight has not risen above 0.33 AND Reentrancy F1 has not risen above 0.18 by epoch 10, **pause Run 5**. Then follow this decision tree:

1. **Check if `aux_phase2_loss` is logged and non-zero at epoch 10:**
   - If absent or 0.0 → The loss weight is not reaching `train_epoch()`. Fix the propagation path (§9.1 verification), then resume from epoch 10 checkpoint.
   - If present and non-zero → Continue to step 2.

2. **Check if `aux_phase2_loss` is decreasing:**
   - If not decreasing by ep5 → The aux head may not be connected to the Phase 2 embeddings. Check `aux_head_phase2` weight norms — if still at init, the gradient path is broken. Debug the connectivity, then resume.
   - If decreasing → Continue to step 3.

3. **Check `aux_head_phase2.weight.norm()` trend:**
   - If still at initialization → Gradient is not reaching the head parameters despite the loss being computed. Check if the head is in `optimizer.param_groups`.
   - If increasing → The head is learning but Phase 2 embeddings aren't changing. Continue to step 4.

4. **Phase 2 head capacity (RC3) is likely the bottleneck:**
   - Try increasing `aux_phase2_loss_weight` from 0.10 to 0.20 and resume training for 5 more epochs.
   - If JK weight still doesn't rise → The single-head Phase 2 (RC3) is fundamentally insufficient. **Escalate to Run 6 with multi-head Phase 2 (4–8 heads).** Document the Run 5 partial results for Run 6 planning.
   - Do not continue training to ep60 if the signal is confirmed structurally blocked.

---

## 10. Phase 6 — Calibration & Threshold Fixes

> **CRITICAL NOTE:** This phase is listed here for organizational completeness but is **executed AFTER Phase 8 (Run 5 training) completes**. Temperature scaling is a post-training, inference-time fix. It does not affect training dynamics. Phases 7 and 8 do NOT depend on Phase 6. Do not attempt to calibrate before training.

### 10.1 Temperature Scaling (Post-Training Inference Fix)

**Status:** `ml/calibration/temperatures_run4.json` already computed. ECE: 0.249 → 0.028. Per EXP-B2: Individual eyes are well-calibrated (ECE 0.057–0.065). The main head is severely miscalibrated (ECE 0.249). Temperature scaling targets the main head only.

**Key Rule:** Temperature scaling is a post-training, inference-time fix. **Re-fit `temperatures_run5.json` after Run 5 training completes.** Do NOT reuse Run 4 temperatures on a Run 5 checkpoint — the temperature parameters are model-specific and will produce incorrect calibration if applied to a different model.

**Fitting Procedure:**
1. After Run 5 training completes, set the model to eval mode
2. Collect all validation logits and labels
3. Optimize per-class temperature parameters T via NLL minimization on the validation set
4. Save the resulting temperatures to `temperatures_run5.json`
5. Verify that ECE drops below 0.05 on the validation set

**Gate 6.1:** Post-fitting ECE on the main head must be below 0.05. If it is not, the temperature scaling may need per-eye calibration or a more complex calibration method (e.g., Platt scaling with binning).

### 10.2 Threshold Tuning Configuration

The threshold sweep frequency is already addressed by A37 (Phase 4). After implementing A37, thresholds will be tuned every 10 epochs instead of every epoch. The final epoch always tunes thresholds. Ensure the tuned thresholds from the final epoch are saved alongside the checkpoint for consistent inference behavior.

---

## 11. Phase 7 — Data Re-Extraction, Archival & Migration (IMP-D1)

This phase must come **after** all extractor, model, and trainer fixes are in place (Phases 0–5 complete and validated). Re-extracting with a broken extractor would produce incorrect graphs that would need to be re-extracted again. **This phase also includes comprehensive archival of all v8-era data and migration to v9 data to ensure no stale artifacts contaminate Run 5.**

### 11.1 Why Re-Extraction Is Required

Multiple fixes in Phases 0–2 change the graph structure:
- **A20 (label=0 fix):** Labels must be corrected in all extracted .pt files
- **A14 (RETURN_TO revert filtering):** Edge structure changes — revert paths must be excluded
- **A15 (DEF_USE scope key):** DEF_USE edges must be recomputed with two-tier scope-aware keys
- **A5 (narrowed except scope):** `return_ignored` values may change for some contracts
- **A6/A10/A18 (structured error handling):** Previously silently-failed extractions may now produce different (correct) results
- **IMP-D1 (max_nodes increase):** Previously truncated contracts must be re-extracted with the new limit
- **A9 (SolidityVariableComposed fix):** `uses_block_globals` values may change for Timestamp/TOD contracts
- **NF-1 (EMITS key mismatch fix):** EMITS edges for Solidity <0.4.21 contracts will be recovered
- **NF-10 (duplicate function synthetic key):** Overriding functions' CFGs will be preserved
- **NF-11 (per-type edge drop counters):** Extraction diagnostics will provide full visibility

### 11.2 Pre-Extraction Data Archival (MANDATORY — before any new extraction)

Before running re-extraction, **all v8-era artifacts must be moved to clearly-labeled archive directories**. This prevents accidental use of stale, buggy data during or after re-extraction. Run 5 must train exclusively on v9 data produced after all fixes.

**Archival checklist:**

| Artifact | Source Path | Archive Path | Notes |
|----------|-----------|--------------|-------|
| v8 graph .pt files | `ml/data/graphs/` | `ml/data/archive/graphs_v8_pre_run5/` | All 41,576 .pt files |
| v8 cached dataset | `ml/data/cached_dataset_v8.pkl` | `ml/data/archive/cached_dataset_v8.pkl` | 2.2 GB |
| v8 token files | `ml/data/tokens_windowed/` | `ml/data/archive/tokens_windowed_v8/` | Confirm tokens are schema-compatible with v9 (no retokenization needed if only graph structure changed) |
| v8 label CSV | `ml/data/processed/multilabel_index_cleaned.csv` | `ml/data/archive/multilabel_index_cleaned_v8.csv` | The "cleaned" CSV from Phase 3.5 |
| v8 splits | `ml/data/splits/deduped/` | `ml/data/archive/splits_v8_deduped/` | train/val/test .npy files |
| v8 multilabel index | `ml/data/processed/multilabel_index.csv` | `ml/data/archive/multilabel_index_v8.csv` | Pre-cleaning index |
| v8 build_multilabel_index output | Any index built from v8 graphs | `ml/data/archive/index_v8/` | **Do NOT reuse** — must be rebuilt from v9 (NF-3) |
| All pre-Run-5 checkpoints | `ml/checkpoints/*.pt` | `ml/data/archive/checkpoints_pre_run5/` | All Run 1–4 checkpoints |
| Run 4 training logs | `ml/logs/` | `ml/data/archive/logs_pre_run5/` | Keep for reference |

**Verification after archival:**
- Confirm the source directories are empty (or contain only a README pointing to the archive)
- Confirm the archive directories contain all expected files (count .pt files, verify CSV row counts)
- **Do NOT delete the archive** — it is the fallback if re-extraction fails or produces unexpected results
- Add a `ml/data/archive/v8_archive_manifest.txt` listing all archived files with counts and sizes

### 11.3 Re-Extraction Procedure

1. Confirm all Phase 0–2 fixes are applied to `graph_extractor.py`
2. Confirm `max_nodes=2048` is set (Intervention 4 from Phase 5)
3. Confirm A20 fix (Phase 0) is applied — labels will be correct in re-extracted graphs
4. Confirm Gate 1.1 has passed (solc toolchain is functional)
5. Run full extraction with all fixes applied (10 workers, ~30 min expected based on previous runs)
6. CEI path labeling (Intervention 2 from §9.2) runs as part of the extraction pipeline on the v9 graphs
7. Rebuild the cache: `ml/data/cached_dataset_v9.pkl`
8. Rebuild the multilabel index from v9 graphs only (do NOT use v8 index)
9. Regenerate train/val/test splits if label cleaning removes additional contracts

### 11.4 Post-Extraction Data Migration Verification

After re-extraction, verify that all code paths and configuration files point to v9 data:

| Check | Expected | Action if Wrong |
|-------|----------|----------------|
| `ml/data/graphs/` contains v9 .pt files | New extraction output | Re-run extraction |
| `ml/data/cached_dataset_v9.pkl` exists and is non-empty | Fresh cache from v9 graphs | Rebuild cache |
| `train.py --cache-path` defaults or is set to v9 cache | `ml/data/cached_dataset_v9.pkl` | Update CLI args / config |
| `train.py --label-csv` points to v9-cleaned CSV | `ml/data/processed/multilabel_index_cleaned_v9.csv` | Update CLI args / config |
| `train.py --splits-dir` points to v9 splits | `ml/data/splits/v9_deduped/` | Update CLI args / config |
| No Python imports or hardcoded paths reference v8 data | All paths point to v9 | Search codebase for "v8" references |
| `build_multilabel_index.py` run against v9 graphs only | Index built from v9 | Re-run if accidentally run on archived v8 |
| Old v8 cache file not accidentally loaded | `cached_dataset_v8.pkl` is in archive only | Delete from active directory if present |

### 11.5 Post-Extraction Validation

**Gate 7.1 — Extraction Completeness:**
- Total extracted graphs should be ≥ 41,000 (comparable to v8's 41,576)
- Skip count should not increase significantly vs v8 baseline
- Fail count must be 0

**Gate 7.2 — Extraction Quality (same as Gate 2.1):**
- `_icfg_failure_count == 0`
- `_cfg_type_fallback_count / total_cfg_nodes < 0.01`
- CALL_ENTRY edge presence rate ≥ 64.2%
- RETURN_TO edge presence rate ≥ 55.6%

**Gate 7.3 — Label Integrity:**
- Verify label distribution matches the cleaned CSV (no all-zero label batches)
- Spot-check at least 100 contracts: labels in .pt files match ground-truth CSV
- Verify `uses_block_globals` is non-zero for Timestamp-positive contracts (Gate 2.2 follow-up)
- **NF-3 validation:** Do NOT run `build_multilabel_index.py` against the archived (un-fixed) v8 graphs — it will propagate the A20 label-zero corruption into the index. Only run it against the re-extracted (Phase 7) v9 graphs. Verify that non-BCCC contracts (SolidiFI, SmartBugs) have correct labels in the re-extracted index.
- **Pre/post re-extraction label distribution comparison:** Compute and log the label distribution before and after re-extraction. Flag if any class's positive count changes by > 10% — this would indicate A20 had material impact on that class's training data. Log the comparison to `ml/data/v8_v9_label_comparison.txt` for the Run 5 report.

**Gate 7.4 — Schema Consistency:**
- All extracted graphs have `feature_schema_version` consistent with the new schema
- NODE_FEATURE_DIM = 11 in all graphs
- `max_nodes` increased to 2048 is reflected in extraction output

**Gate 7.5 — CEI Label Quality (same as Gate 5.2, evaluated on v9 data):**
- At least 60% of Reentrancy-positive training contracts have `has_cei_path = 1`
- No more than 5% of Reentrancy-negative contracts have `has_cei_path = 1`
- If positive coverage < 40%: the CEI path labeler has a bug — do not enable `aux_cei_loss_weight`

---

## 12. Phase 8 — Run 5 Execution & Monitoring

### 12.1 Run 5 Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| GNN layers | 8 | Unchanged from Run 4 |
| gnn_heads | 8 (Phase 1), 1 (Phase 2) | Phase 2 capacity gap (RC3) not changed |
| gnn_prefix_k | 48 | Unchanged |
| gnn_prefix_warmup_epochs | 15 | Unchanged |
| aux_phase2_loss_weight | **0.10** | NEW — was 0.0 in Run 4 |
| aux_cei_loss_weight | **TBD after Gate 7.5** | NEW — depends on CEI label validation on v9 data |
| jk_entropy_reg_lambda | 0.005 | Unchanged from Run 4 |
| max_nodes | **2048** (or 1536 per Gate 5.3) | NEW — was 1024 |
| dos_loss_weight | 0.5 | Unchanged from Run 4 |
| pos_weight | Not passed to ASL | Unchanged from Run 4 (NC-4 reverted) |
| Backbone | GraphCodeBERT + LoRA | Unchanged |
| Epochs | 100 with early-stop patience 30 | Unchanged |
| **Cache path** | **`ml/data/cached_dataset_v9.pkl`** | **v9 only — verify before launch** |
| **Label CSV** | **v9-cleaned CSV** | **v9 only — verify before launch** |
| **Splits dir** | **v9 splits** | **v9 only — verify before launch** |

### 12.2 Monitoring Schedule

| Epoch | Check | Expected Observation |
|-------|-------|---------------------|
| 1 | aux_phase2_loss logged and non-zero | Confirms aux loss reaches optimizer |
| 1 | `aux_head_phase2.weight.norm()` logged | Confirms head is receiving gradient |
| 1–5 | NaN loss count per epoch | Should be 0 — if not, Gate 0.2 triggers |
| 5 | Phase 2 JK weight | Should be rising from 0.322 baseline |
| 5 | `aux_head_phase2.weight.norm()` should be above init | If still at init, aux loss path is broken |
| 10 | **Gate 5.4 — go/no-go** | Phase 2 JK weight > 0.33 OR Reentrancy F1 > 0.18 |
| 15 | Prefix warmup ends | gnn_to_bert_proj starts receiving gradient |
| 16 | Loss spike expected | Brief loss increase from prefix activation |
| 20 | prefix_attention_mean | Should be > 0.005 |
| 20 | ExternalBug F1 | Should be > 0.000 |
| 30+ | Steady-state monitoring | F1 should be improving beyond 0.3362 |

### 12.3 Early Termination Criteria

**Immediate halt (investigate before continuing):**
- nan_loss_count > 0.5% × steps_per_epoch (Gate 0.2)
- Phase 2 JK weight falls below 0.25 at any point after ep5 (structural collapse)
- Loss becomes NaN or Inf for more than 3 consecutive steps

**Pause and evaluate at epoch 10 (Gate 5.4 — follow rollback decision tree in §9.6):**
- Phase 2 JK weight < 0.33 AND Reentrancy F1 < 0.18

**Normal early stopping:**
- F1-macro does not improve for 30 consecutive epochs (patience = 30)

### 12.4 Post-Training Checklist

1. Save the best checkpoint with full configuration metadata
2. Re-fit temperature scaling (Phase 6) — do not reuse Run 4 temperatures
3. Run behavioral test suite (20 contracts, 19 expected detections) — target >10/19
4. Run full evaluation on test split with tuned thresholds
5. Log per-class F1, precision, recall for comparison with Run 4
6. Run size-stratified Timestamp evaluation (Intervention 3, Option A)
7. Log JK weight distribution at the final epoch for Phase 2 analysis
8. Log Phase 2 gradient norm ratio (P2/P1) at the final epoch
9. Verify Run 5 checkpoint loads correctly with all v9 data paths
10. Archive Run 5 checkpoint and logs alongside the v8 archive

---

## 13. Gate Summary — Complete Go/No-Go Matrix with Rollback Decision Trees

| Gate | Phase | Checkpoint | Condition | Consequence of Failure | Rollback Action |
|------|-------|-----------|-----------|----------------------|----------------|
| **0.1** | 0 | After A20 fix | `label_map` populated; labels match CSV | Do not proceed — data is poisoned | Restore CSV from VC; debug `label_map` construction |
| **0.2** | 0 | Every training epoch | nan_loss_count < 0.5% × steps | Halt training — systematic instability | Investigate NaN source; restart from last clean checkpoint |
| **1.1** | 1 | Before extraction | solc available and versions correct | Do not extract — toolchain broken | Install/configure solc-select before proceeding |
| **2.1** | 2 | After re-extraction | ICFG failure=0; CF fallback<1%; edge rates at baseline | Do not proceed to Run 5 — extraction bugs not resolved | Re-examine Phase 2 fixes; re-extract |
| **2.2** | 2 | After A9 fix | uses_block_globals non-zero for 80%+ Timestamp+ contracts | Feature still broken — investigate Slither compatibility | Check Slither version; update isinstance import |
| **3.1** | 3 | After Phase 3 fixes | torch.compile produces correct 2-epoch output | Compiled graph stale from forward-pass changes | Disable torch.compile for Run 5; file Run 6 fix |
| **NF-4** | 4 | Before Run 5 launch | `gnn_num_layers=8` in training log | Halt — wrong architecture | Fix CLI default; relaunch |
| **5.1** | 5 | Run 5 epoch 1 | aux_phase2_loss logged and non-zero | Do not continue — aux loss not reaching optimizer | Debug loss propagation path; resume from ep1 |
| **5.2/7.5** | 5/7 | After re-extraction | CEI label coverage ≥60% positives, ≤5% negatives | Do not enable aux_cei_loss_weight | Debug CEI labeler; re-label on v9 data |
| **5.3** | 5 | Before Run 5 | VRAM < 7.5GB with max_nodes=2048 (full training step) | Reduce batch_size or max_nodes | Test batch_size=16 or max_nodes=1536 |
| **5.4** | 5 | Run 5 epoch 10 | Phase 2 JK weight > 0.33 OR Reentrancy F1 > 0.18 | Pause Run 5 — follow rollback decision tree (§9.6) | Try aux_weight=0.20; if still failing, escalate to Run 6 |
| **6.1** | 6 | After temperature fitting | Main head ECE < 0.05 | Consider per-eye calibration or Platt scaling | Try Platt scaling with binning |
| **7.1** | 7 | After re-extraction | Graph count ≥41K; fail count = 0 | Do not proceed — extraction incomplete | Re-examine extraction logs; re-extract |
| **7.2** | 7 | After re-extraction | Same as Gate 2.1 | Do not proceed — extraction quality insufficient | Re-examine Phase 2 fixes; re-extract |
| **7.3** | 7 | After re-extraction | Labels match CSV; no all-zero batches; label distribution comparison < 10% shift per class | Do not proceed — label integrity compromised | Verify A20 fix; re-extract |
| **7.4** | 7 | After re-extraction | Schema version consistent; correct dimensions | Do not proceed — schema mismatch | Verify schema fixes; re-extract |

---

## 14. Expected Outcomes & Risk Assessment

### 14.1 Expected Outcomes

**Pessimistic (fixes only, Phase 2 signal unchanged):** macro-F1 = 0.34–0.36
- Bug fixes (especially A20, A14, A15, A6/A10/A18) improve graph quality
- Temperature scaling reduces miscalibration
- But Phase 2 signal remains collapsed without successful interventions

**Base case (Phase 2 aux loss effective):** macro-F1 = 0.38–0.42
- aux_phase2_loss_weight = 0.10 provides enough gradient to make Phase 2 learn
- JK Phase 2 weight rises above 0.35
- Reentrancy and ExternalBug F1 improve materially
- CEI auxiliary loss further boosts Reentrancy if Gate 7.5 passes

**Optimistic (all interventions succeed):** macro-F1 = 0.44–0.48
- Phase 2 becomes genuinely informative
- max_nodes=2048 recovers truncated Timestamp-positive contracts
- CEI auxiliary loss provides strong Reentrancy supervision
- JK weight distribution becomes class-dependent rather than global constant

### 14.2 Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| A20 fix reveals that all previous extractions were poisoned | High | Severe — requires full re-extraction | Phase 7 is already planned; label verification in Gate 7.3 |
| Phase 2 single head (RC3) cannot learn from aux loss | Medium | High — Phase 2 signal stays collapsed | Gate 5.4 rollback decision tree; try aux_weight=0.20; escalate to Run 6 |
| CEI label coverage too low for useful supervision | Medium | Medium — Reentrancy improvement limited | Gate 7.5 validates coverage on v9 data before enabling aux_cei_loss_weight |
| max_nodes=2048 causes OOM on RTX 3070 | Medium | Low — fallback to 1536 or batch_size=16 | Gate 5.3 tests realistic worst-case VRAM before committing |
| Bug fixes interact unexpectedly (e.g., A14 + A15 + NF-10 change graph topology) | Low | Medium — graph structure changes significantly | Gate 7.2 validates extraction quality against baselines |
| Phase 2 learned suppression (RC7) persists despite aux loss | Medium | Medium — Reentrancy F1 does not improve | Monitor at Gate 5.4; may need Run 6 architectural changes |
| NaN loss from BF16 overflow in new code paths | Low | High — corrupts Adam state | Gate 0.2 provides early detection and halt; rollback to last clean checkpoint |
| torch.compile breaks after Phase 3 forward-pass changes | Medium | Low — disable torch.compile | Gate 3.1 validates; Run 5 without compile is slower but correct |
| Stale v8 data accidentally used in Run 5 | Low | High — trains on wrong data | Comprehensive archival (§11.2); migration verification (§11.4); Gate 7.3 label comparison |
| NF-8 empty-batch KeyError crashes Run 5 | Low | Medium — mysterious crash at epoch boundary | Fixed in Phase 3 (elevated to Medium); add unit test |

### 14.3 What Run 5 Cannot Fix

The following architectural limitations persist into Run 5 and will require Run 6+ to address:

1. **RC3 (8× head capacity gap):** Phase 2 uses 1 attention head vs Phase 1's 8 heads. This fundamentally limits Phase 2's ability to learn selective message routing. Increasing Phase 2 heads is an architectural change that requires careful testing.

2. **RC5 (DEF_USE 1-hop limitation):** DEF_USE edges are only processed in Layer 5, limiting data-flow chain propagation to 1 hop. Meaningful def-use reasoning requires multi-hop propagation, which would require either repositioning DEF_USE edges across multiple layers or adding dedicated def-use propagation passes.

3. **RC6 (Phase 3 does Phase 2's job):** Phase 3's REVERSE_CONTAINS edges lift CFG information to FUNCTION level, making Phase 2's contribution structurally redundant. This is by design in the current architecture — fixing it would require fundamentally restructuring how Phase 2 and Phase 3 interact.

4. **BF16 quantization floor on gnn_to_bert_proj:** Run 4 observed that the projection weight norm stagnated at ~16.0 (2 BF16 ULPs), preventing fine-grained gradient accumulation. This may persist in Run 5 unless the projection is kept in float32 or the learning rate is adjusted.

---

## 15. Out-of-Scope Items for Run 5 (Run 6+ Candidates)

| Item | Description | Reason for Deferment |
|------|-------------|---------------------|
| Phase 2 multi-head attention | Increase Phase 2 GAT heads from 1 to 4–8 | Architectural change; requires testing impact on JK weight dynamics |
| Multi-hop DEF_USE propagation | Add DEF_USE edges to multiple Phase 2 layers | Requires careful design to avoid dilution (v8-AB showed DEF_USE hurts some classes) |
| Adversarial size regularizer | Gradient reversal on size predictor for Timestamp | Option B from Intervention 3 — higher risk, defer to Run 6 |
| Phase 2/Phase 3 interaction redesign | Restructure how Phase 2 and Phase 3 share information | Major architectural change; requires separate proposal |
| gnn_to_bert_proj float32 preservation | Keep the projection in float32 to avoid BF16 quantization floor | May be needed if Run 5 shows same stagnation as Run 4 |
| Brainmab contract label cleanup (BUG-M5) | Standard ERC20 labeled with 4 vulnerability types | Open bug from changelog — not blocking for Run 5 |
| NF-12: predictor.py window truncation and random edge embeddings | Predictor silently truncates to 4 windows; old checkpoints get random embeddings for new edge types | Inference-only; deferred to post-Run-5 inference hardening |
| NF-6: Phase 2 ablation bypass (Layers 3/4 ignore `phase2_edge_types`) | Must fix before any post-Run-5 ablation experiments, but zero training impact | Affects interpretability experiments only, not Run 5 F1 |
| Token file schema version (BUG-M6) | Token files carry stale `feature_schema_version='v4'` | Auto-resolves on retokenize; not blocking |
| Empty contract_path in 8.5% of graphs (BUG-M7) | Cannot cross-reference source | Not blocking for training; affects post-hoc analysis |
| Hash-based graph-token pairing (BUG-L3) | Fragile to directory restructuring | Deferred — low priority |

---

## 16. Open Bugs From Changelog (Non-Proposal)

These bugs are documented in the CHANGELOG as of 2026-05-24 but are not included in this proposal's scope. They are tracked separately for future resolution.

| ID | Description | Status |
|----|-------------|--------|
| BUG-M5 | Brainmab contract: standard ERC20 labeled Reentrancy+CallToUnknown+IntegerUO+MishandledException | OPEN |
| BUG-M6 | Token files carry stale `feature_schema_version='v4'` metadata | OPEN (auto-resolves on retokenize) |
| BUG-M7 | 8.5% of graphs have empty `contract_path` (cannot cross-reference source) | OPEN |
| BUG-L3 | Hash-based graph-token pairing fragile to directory restructuring | DEFERRED |

---

*This proposal consolidates all findings from the interpretability audit, validated code audition, root cause analysis, and review feedback into a single authoritative execution plan for Run 5. All gates are mandatory — skipping a gate means proceeding with known-corrupt data, broken extraction, or ineffective training interventions. The order of execution is non-negotiable: each phase depends on the previous phase being complete and validated. All previous v8-era data must be archived before re-extraction, and Run 5 must train exclusively on verified v9 data.*
