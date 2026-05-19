# SENTINEL — Active Plan: v8 + v9 Roadmap

Last updated: 2026-05-19 (rev 7 — PLAN-1B gate PASSED; Phase 1 fully complete; Phase 2 next)

**Current state (2026-05-19):**
- **v7 training COMPLETE** — best F1=0.2651 at epoch 23, checkpoint `ml/checkpoints/v7.0_best.pt`
- **Schema v8 COMPLETE:** `NUM_EDGE_TYPES=11`, `FEATURE_SCHEMA_VERSION="v8"` — CALL_ENTRY(8) + RETURN_TO(9) + DEF_USE(10)
- **PLAN-1B gate PASSED (2026-05-19):** 2,000-contract sample validated
  - 1999/2000 structural parity (1 non-deterministic Slither failure, re-runs clean)
  - P99=1786 edges (limit 5,000) ✓ · max=3707 (limit 10,000) ✓
  - CALL_ENTRY=12,630 · RETURN_TO=11,311 · DEF_USE=55,680 (all non-zero) ✓
  - DataLoader batch_size=8: 8 graphs / 722 nodes batched cleanly ✓
- **Next:** Phase 2 full v8 re-extraction (PLAN-2A → PLAN-2I)

**Proposal source:** `docs/2026-18-05-SENTINEL — Graph Representation Extension Proposal.md` (v3 — Final Consolidated)

---

## Legend

| Status | Meaning |
|--------|---------|
| **OPEN** | Not started |
| **IN PROGRESS** | Started but not complete |
| **DONE** | Complete |
| **DEFERRED** | Known, accepted deferral with explicit reason |
| **BLOCKED** | Cannot start until dependency is resolved |

| Priority | Meaning |
|----------|---------|
| P0 | Blocking — must complete before next phase can start |
| P1 | High impact; do early in the phase |
| P2 | Important but can be done in any order within the phase |
| P3 | Low urgency; do before phase closes |

---

## Phase 0 — Pre-Implementation Cleanup
**Target:** Complete during v7 training. All items non-breaking (extractor not used during training).
**Status: DONE (2026-05-18)**

| ID | Item | File | Status |
|----|------|------|--------|
| P0-1 | Delete dead `in_unchecked = _compute_in_unchecked(obj)` assignment | `graph_extractor.py:699` | **DONE** |
| P0-2 | Mark `_compute_in_unchecked()` as deprecated (v7 BUG-L2) | `graph_extractor.py:318` | **DONE** |
| P0-3 | Update `_build_node_features` docstring: v4/12-dim → v7/11-dim | `graph_extractor.py:632` | **DONE** |
| P0-4 | Remove `[9] in_unchecked` from docstring feature layout table | `graph_extractor.py:635–659` | **DONE** |
| P0-5 | Remove dead `in_unchecked = 0.0` default variable | `graph_extractor.py:681` | **DONE** |

---

## Phase 1 — Extractor Refactor + Sample Validation
**Trigger:** v7 training completes (best checkpoint saved, F1 confirmed stable).
**Goal:** Implement v8 extractor changes; validate on sample before touching all 41,576 contracts.

---

### P0 — Blocking gates (must pass before full extraction)

#### PLAN-1A — Validate `node.internal_calls` at CFG-node level

- **Priority:** P0 (blocks all ICFG work)
- **File:** new scratch script (not committed)
- **Why:** `_add_icfg_edges()` uses `caller_node.internal_calls` per CFG node. The current extractor only uses `func.internal_calls` at function level (`graph_extractor.py:1077`). Node-level access is standard Slither API but is NOT exercised anywhere in the codebase — behaviour under Slither 0.11.x must be confirmed before full extraction.
- **Protocol:** Run on 10 contracts from `ml/data/augmented/` or any available `.sol` file:
  ```python
  for func in contract.functions:
      func_calls = set(f.canonical_name for f in func.internal_calls)
      node_calls = set()
      for node in func.nodes:
          for called in (node.internal_calls or []):
              node_calls.add(called.canonical_name)
      assert node_calls.issubset(func_calls)
  ```
- **If fails:** Implement `InternalCall` IR fallback (see Proposal §7.6):
  ```python
  from slither.slithir.operations import InternalCall
  called_funcs = [ir_op.function for ir_op in (node.irs or []) if isinstance(ir_op, InternalCall)]
  ```
- **Status:** **DONE** — `node.internal_calls` validated on synthetic contract; returns correct per-node callee canonical_names. IR fallback not needed.

#### PLAN-1B — 2,000-contract sample validation gate
- **Priority:** P0 (blocks full re-extraction)
- **Validation criteria (must ALL pass):**
  - P99 edge count per graph < 5,000
  - No single graph exceeds 10,000 edges
  - `edge_attr.max() <= 10` on all sampled graphs
  - CALL_ENTRY, RETURN_TO, DEF_USE all have non-zero total counts
  - DataLoader batch fits GPU memory at batch_size=8
  - **Structural comparison gate:** extract same 2,000 contracts with both v7 and v8 extractors; verify all v7 edges (CALLS/READS/WRITES/EMITS/INHERITS/CONTAINS/CONTROL_FLOW) are bit-for-bit identical in v8 output — new types are additive only, no existing edges may change or disappear. Any regression = bug in `global_cfg_node_map` refactor.
- **If gate fails:** Tighten ICFG depth or DFG categories before proceeding
- **Status:** **DONE (2026-05-19)** — 2,000-contract gate PASSED. 1999/2000 parity (0.05% Slither non-determinism, re-runs clean); P99=1786; max=3707; all new types fire; DataLoader batch clean.

---

### P1 — Core extractor changes

#### PLAN-1C — Accumulate `global_cfg_node_map` across function iterations
- **Priority:** P1
- **File:** `ml/src/preprocessing/graph_extractor.py`
- **Current:** `cfg_node_map: dict = {}` declared inside `for func in contract.functions` loop (line 989) — discarded after each iteration
- **Change:** Declare `global_cfg_node_map: dict = {}` before the loop; rename loop-local map to `func_cfg_map`; call `global_cfg_node_map.update(func_cfg_map)` after each iteration
- **Signature impact:** `_build_control_flow_edges()` signature UNCHANGED — still receives and populates its own local map; only the caller changes
- **Required by:** `_add_icfg_edges()` and `_add_def_use_edges()` (both need cross-function CFG node lookup)
- **Status:** **DONE** — `_func_entry_map` / `_func_terminal_map` / `_func_cfg_maps` accumulated in extraction loop (commit b9ba690)

#### PLAN-1D — Implement `_add_icfg_edges()`
- **Priority:** P1
- **File:** `ml/src/preprocessing/graph_extractor.py`
- **Status:** **DONE** — `_add_icfg_edges()` implemented and called from extraction loop; CALL_ENTRY(8) + RETURN_TO(9) edges validated on test contract (commit b9ba690)

#### PLAN-1E — Implement `_add_def_use_edges()`
- **Priority:** P1
- **File:** `ml/src/preprocessing/graph_extractor.py` (new helper function)
- **New edge type:** `DEF_USE(10)` — will bump `NUM_EDGE_TYPES` 10→11
- **Critical correctness rules (from audit):**
  - Use `isinstance(v, StateVariable)` NOT `v.is_storage` — `v.is_storage` is not a Slither API attribute
  - Use `isinstance(ir_op, Binary) AND ir_op.type in _ARITHMETIC` NOT just `isinstance(ir_op, Binary)` — Binary covers comparisons/logical/bitwise too
  - Definition categories: (1) HighLevelCall/LowLevelCall/Send return values, (2) arithmetic Binary results, (3) Assignment RHS reading StateVariable
  - Condition nodes are USE sites, NOT definition sites — Condition IR ops have no lvalue
  - Key by `lval.name` strings (consistent with BUG-M1 fix in `_compute_return_ignored`)
- **Status:** **DONE** — `_add_def_use_edges()` implemented; LocalVariable DEF→USE edges; NUM_EDGE_TYPES=11 (commit ce95e59)

---

### P2 — Schema updates

#### PLAN-1F — Update `graph_schema.py` for v8
- **Priority:** P2
- **File:** `ml/src/preprocessing/graph_schema.py`
- **Changes:**
  ```python
  FEATURE_SCHEMA_VERSION = "v8"          # was "v7"
  NUM_EDGE_TYPES         = 11            # was 8
  EDGE_TYPES["CALL_ENTRY"] = 8           # NEW
  EDGE_TYPES["RETURN_TO"]  = 9           # NEW
  EDGE_TYPES["DEF_USE"]    = 10          # NEW
  # NODE_FEATURE_DIM = 11  (UNCHANGED)
  # NUM_NODE_TYPES   = 13  (UNCHANGED)
  ```
- **Note:** Assertion guards at `graph_schema.py:391–402` enforce `NUM_EDGE_TYPES` consistency at import time — mismatch raises `AssertionError` on startup, catching any missed constant
- **Status:** **DONE** — `FEATURE_SCHEMA_VERSION="v8"`, `NUM_EDGE_TYPES=11`, CALL_ENTRY(8) + RETURN_TO(9) + DEF_USE(10) all added (commit ce95e59)

#### PLAN-1G — Update `gnn_encoder.py` Phase 2 mask for v8
- **Priority:** P2
- **File:** `ml/src/models/gnn_encoder.py`
- **Status:** **DONE** — `Embedding(11, 64)`; `cfg_mask` = CONTROL_FLOW(6)|CALL_ENTRY(8)|RETURN_TO(9)|DEF_USE(10) (commit ce95e59)

---

## Phase 2 — v8 Full Re-Extraction (~41,576 contracts)
**Trigger:** Phase 1 sample validation gate PASSES (PLAN-1B).
**Estimated time:** ~same as v7 extraction run.

| ID | Step | Status |
|----|------|--------|
| PLAN-2A | Archive current v7 graphs: `ml/data/archive/graphs_v7/` | OPEN |
| PLAN-2B | Run full v8 extraction | OPEN |
| PLAN-2C | Validate 100% graphs have `edge_attr.max() <= 10` | OPEN |
| PLAN-2D | Verify non-zero per-edge-type counts across full dataset | OPEN |
| PLAN-2E | Re-run `label_cleaner.py` on fresh v8 graphs | OPEN |
| PLAN-2F | Re-run `inject_augmented.py` | OPEN |
| PLAN-2G | Rebuild cache (`create_cache.py`) | OPEN |
| PLAN-2H | Regenerate splits from paired stems | OPEN |
| PLAN-2I | Log dataset statistics: mean/P99 node count, edge count, per-type distribution | OPEN |

---

## Phase 3 — v8 Training Ablations
**Trigger:** Phase 2 complete (v8 cache rebuilt, splits valid).
**Goal:** Attribute F1 gains to ICFG vs DFG separately, then jointly.

| ID | Run | Edges in Phase 2 mask | Purpose | Status |
|----|-----|-----------------------|---------|--------|
| PLAN-3A | v8-A | `CF(6) + CALL_ENTRY(8) + RETURN_TO(9)` | Isolate ICFG contribution | OPEN |
| PLAN-3B | v8-B | `CF(6) + DEF_USE(10)` | Isolate DFG contribution | OPEN |
| PLAN-3C | v8-AB | `CF(6) + CALL_ENTRY(8) + RETURN_TO(9) + DEF_USE(10)` | Joint effect | OPEN |

**After all three runs:**
- Compare per-class F1 delta from v7 baseline for each
- Check `jk.last_weights` per phase — Phase 2 weight < 0.10 or > 0.80 indicates Phase 2b sub-phase needed
- Run `return_ignored [7]` scalar ablation on best v8-AB checkpoint:
  - Zero out dim [7] in all graphs, retrain everything else equal
  - If per-class F1 delta for MishandledException + UnusedReturn < 0.5pp → deprecate in v9
  - If delta ≥ 0.5pp → keep scalar; it carries signal DEF_USE does not fully capture

| ID | Item | Status |
|----|------|--------|
| PLAN-3G | Fix stale `--run-name` default in `train.py:68` from `"multilabel-v5-fresh"` → `"sentinel-v8"` before any v8 run | OPEN |
| PLAN-3H | Apply optional S4 speed optimization (cap fusion tokens at 1024 in `fusion_layer.py`) — measure actual batch/s change in a 1-epoch comparison before committing | OPEN |
| PLAN-3D | Inspect `jk.last_weights` after each v8 run — verify Phase 2 weight is in 0.10–0.80 range | OPEN |
| PLAN-3I | Monitor fused eye loss as canary: it should decrease relative to GNN/TF eyes in v8 vs v7. If fused eye remains highest by same margin, flag as architectural concern in `CrossAttentionFusion` | OPEN |
| PLAN-3E | Run `return_ignored` ablation on best v8-AB checkpoint — 0.5pp F1 gate decides v9 deprecation | OPEN |
| PLAN-3J | Review `early_stop_patience=30` after first complete v8 run — if best F1 is near epoch 60–80 and still rising, increase to 40–50 | OPEN |
| PLAN-3F | Document per-class F1 deltas v7→v8-A/B/AB | OPEN |

---

## Phase 4 — v9 (Extension C — Control-Dependence Edges)
**Trigger:** Phase 3 ablation results show headroom; return_ignored decision made.
**Conditional:** Only proceed if v8 validation F1 ≥ target and remaining FNs are guard-scope-related.

### PLAN-4A — Implement `_add_control_dep_edges()`
- **Priority:** P1
- **File:** `ml/src/preprocessing/graph_extractor.py`
- **New edge type:** `CONTROL_DEP(11)`
- **Phase assignment:** Phase 1 (structural), NOT Phase 2 — control dependence is a scope relationship, not sequential execution order
- **check_types:** `{SNT.IF, SNT.IFLOOP, SNT.STARTLOOP, SNT.THROW}` — intentionally excludes `SNT.ENDLOOP` (convergence join point; post-loop code is not meaningfully control-dependent on the loop condition)
- **Algorithm:** Symmetric difference of reachable sets from true-branch vs false-branch successors
- **Full implementation:** see Proposal §4.C.2
- **Status:** OPEN (BLOCKED on Phase 3 results)

### PLAN-4B — Update schema for v9
- **Changes:**
  ```python
  FEATURE_SCHEMA_VERSION  = "v9"
  NUM_EDGE_TYPES          = 12         # +CONTROL_DEP(11)
  # NODE_FEATURE_DIM = 10 ONLY if return_ignored ablation confirms redundancy
  # NODE_FEATURE_DIM = 11 otherwise (keep unchanged)
  ```
- **Status:** OPEN (BLOCKED on PLAN-3E result)

### PLAN-4C — Update `gnn_encoder.py` Phase 1 mask for v9
- **Changes:**
  ```python
  nn.Embedding(11, 64)  →  nn.Embedding(12, 64)

  # Phase 1 struct_mask extended
  _CONTROL_DEP = EDGE_TYPES["CONTROL_DEP"]  # 11
  struct_mask  = (edge_attr <= _CONTAINS) | (edge_attr == _CONTROL_DEP)
  ```
- **Status:** OPEN (BLOCKED on PLAN-4B)

---

## Open Bugs (carried from ACTIVE_BUGS.md)

These were OPEN in v7 and remain unresolved. Address during v8 data preparation.

### BUG-H4 — Timestamp labels: 48.2% have no source evidence
- **File:** `ml/data/processed/multilabel_index_deduped.csv`
- **Impact:** ~463 Timestamp=1 contracts with `uses_block_globals=0` — model learns to associate "no timestamp signal" with timestamp vulnerability (inverted signal)
- **Fix:** Filter Timestamp=1 rows where `uses_block_globals=0.0` across all nodes — can be added to `label_cleaner.py` as a new precondition rule
- **When:** During PLAN-2E (re-run label_cleaner on v8 graphs)
- **Status:** OPEN

### BUG-H5 — ~14% of Reentrancy=1 contracts have no external calls
- **File:** `ml/data/graphs/*.pt`
- **Impact:** ~630 mislabeled training samples; Reentrancy requires external call by definition
- **Fix:** Filter Reentrancy=1 rows where `external_call_count=0` across all FUNCTION nodes — add to `label_cleaner.py`
- **When:** During PLAN-2E
- **Status:** OPEN

### BUG-M5 — Brainmab contract mislabeled across 4 classes
- **File:** `ml/data/processed/multilabel_index_deduped.csv`
- **Impact:** Standard ERC20 labeled Reentrancy=1, CallToUnknown=1, IntegerUO=1, MishandledException=1 — worst-case OR-label noise
- **Fix:** Identify contract hash and remove from CSV; audit any contract with ≥3 co-occurring labels and zero supporting features
- **When:** Before PLAN-2E
- **Status:** OPEN

### BUG-M6 — Token files carry stale `feature_schema_version='v4'`
- **Files:** `ml/data/tokens_windowed/*.pt` metadata
- **Impact:** Cosmetic — not checked at load time; will be resolved automatically when tokens are regenerated for v8
- **When:** Resolved automatically by v8 re-tokenization (runs as part of re-extraction)
- **Status:** OPEN (auto-resolved in Phase 2)

### BUG-M7 — 8.5% of graphs have empty `contract_path`
- **File:** `ml/data/graphs/*.pt` metadata
- **Impact:** Cannot manually inspect or cross-reference against source for those contracts
- **Fix:** Build `ml/scripts/backfill_contract_paths.py` — MD5→path map from BCCC-SCsVul-2024/, patch metadata in-place
- **When:** Before Phase 2 archive (or during extraction if path tracking is added to extractor)
- **Status:** OPEN

### BUG-L3 — Hash-based graph-token pairing fragile to directory restructuring
- **File:** `ml/src/datasets/dual_path_dataset.py`
- **Impact:** Any BCCC reorganization invalidates all hashes, requiring full re-extraction
- **Fix:** Store both path-hash (pairing) and content-hash (dedup)
- **Status:** DEFERRED — low immediate impact; reconsider at v9 if dataset is restructured

---

## Operational Items

### OPS-1 — Install CUDA toolkit + Flash Attention 2
- **Status:** **DONE (2026-05-19)** — CUDA toolkit installed; flash-attn 2.8.3 installed via pre-built wheel
- **Current speed stack:**
  - S2 Fused AdamW: `trainer.py:1029` — `fused=True` ✓
  - S3 SDPA: `transformer_encoder.py:131–143` ✓ (active — see note below)
  - flash-attn 2.8.3: installed ✓
- **Note:** CodeBERT (RoBERTa architecture) does NOT support `attn_implementation="flash_attention_2"` via HuggingFace Transformers — only decoder models (LLaMA, Mistral etc.) have it. The `TransformerEncoder` falls back to SDPA, which is the correct ceiling for this architecture. flash-attn is available for any future backbone swap.

### OPS-2 — Monitor v7 training through convergence
- **Epoch 9 watch point: PASSED** — F1 jumped 0.2102→0.2317 with no collapse, no guardrail fires.
- **FINAL RESULT:** Best F1=0.2651 at epoch 23 (2026-05-19 05:49). Killed at epoch 34, patience 10/30.
- **Plateau analysis:** Val F1 flat epochs 24–33 (0.2432–0.2618) while train loss still declining. Structural ceiling confirmed by JK Phase3 weight drifting 0.572→0.784 — model learned to distrust intra-function CFG signal. No cross-function edges available to rescue.
- **v7 vs v6:** +54.4% F1 (0.2651 vs 0.1717). Improvement attributed to: 3-hop CFG (conv3c), AsymmetricLoss, LR schedule, weighted sampler, correct VRAM headroom.
- **Status:** DONE — best checkpoint at `ml/checkpoints/v7.0_best.pt`

### OPS-3 — Commit Phase 0 cleanup to git
- **Files changed:** `ml/src/preprocessing/graph_extractor.py`
- **Status:** DONE — included in commit 907e442

---

## Summary Table

| ID | Item | Phase | Priority | Blocker | Status |
|----|------|-------|----------|---------|--------|
| P0-1..5 | Phase 0 dead-code + docstring cleanup | 0 | P0 | — | **DONE** |
| PLAN-1A | Validate `node.internal_calls` at node level (10 contracts) | 1 | P0 | v7 training done | **DONE** |
| PLAN-1C | Accumulate `global_cfg_node_map` in extractor loop | 1 | P1 | PLAN-1A | **DONE** |
| PLAN-1D | Implement `_add_icfg_edges()` (CALL_ENTRY, RETURN_TO) | 1 | P1 | PLAN-1C | **DONE** |
| PLAN-1F | Update `graph_schema.py` to v8 constants | 1 | P2 | PLAN-1D/1E | **DONE** |
| PLAN-1G | Update `gnn_encoder.py` Phase 2 mask + embedding size | 1 | P2 | PLAN-1F | **DONE** |
| PLAN-1E | Implement `_add_def_use_edges()` (DEF_USE) | 1 | P1 | PLAN-1C | **DONE** |
| PLAN-1B | 2,000-contract sample validation gate | 1 | P0 | PLAN-1D/1E | OPEN |
| PLAN-2A | Archive v7 graphs | 2 | P0 | Phase 1 done | OPEN |
| PLAN-2B–2I | Full v8 re-extraction + validation + cache rebuild | 2 | P1 | PLAN-2A | OPEN |
| PLAN-3G | Fix stale `--run-name` default in `train.py:68` | 3 | P0 | before any v8 run | OPEN |
| PLAN-3A | v8-A ablation run (ICFG only) | 3 | P1 | Phase 2 done | OPEN |
| PLAN-3B | v8-B ablation run (DFG only) | 3 | P1 | Phase 2 done | OPEN |
| PLAN-3C | v8-AB ablation run (both) | 3 | P1 | Phase 2 done | OPEN |
| PLAN-3H | Apply optional S4 (cap fusion tokens at 1024) — measure first | 3 | P2 | Phase 2 done | OPEN |
| PLAN-3D | Inspect `jk.last_weights` per run; check fused eye canary | 3 | P2 | PLAN-3A/B/C | OPEN |
| PLAN-3I | Monitor fused eye loss relative to GNN/TF eyes in v8 | 3 | P2 | PLAN-3A/B/C | OPEN |
| PLAN-3E | `return_ignored` scalar ablation on best v8-AB | 3 | P2 | PLAN-3C | OPEN |
| PLAN-3J | Review `early_stop_patience` after first v8 run | 3 | P3 | PLAN-3A | OPEN |
| PLAN-3F | Document per-class F1 deltas v7→v8-A/B/AB | 3 | P2 | PLAN-3A/B/C | OPEN |
| PLAN-4A | Implement `_add_control_dep_edges()` (CONTROL_DEP) | 4 | P1 | Phase 3 results | OPEN |
| PLAN-4B | Update `graph_schema.py` to v9 constants | 4 | P2 | PLAN-3E + 4A | OPEN |
| PLAN-4C | Update `gnn_encoder.py` Phase 1 mask + embedding size | 4 | P2 | PLAN-4B | OPEN |
| BUG-H4 | Filter Timestamp labels with no source evidence | 2 | P1 | v8 graphs ready | OPEN |
| BUG-H5 | Filter Reentrancy labels with no external calls | 2 | P1 | v8 graphs ready | OPEN |
| BUG-M5 | Remove Brainmab mislabeled contract | 2 | P2 | — | OPEN |
| BUG-M6 | Stale token schema version metadata | 2 | P3 | auto-resolved by retokenize | OPEN |
| BUG-M7 | 8.5% graphs have empty contract_path | 1 | P3 | — | OPEN |
| BUG-L3 | Path-hash pairing fragile to dir restructure | — | P3 | — | DEFERRED |
| OPS-1 | Install CUDA toolkit + Flash Attention 2 | — | P3 | — | **DONE** (SDPA is ceiling for CodeBERT; flash-attn ready for backbone swap) |
| OPS-2 | Monitor v7 training through convergence | — | P0 | — | **DONE** |
| OPS-3 | Commit Phase 0 cleanup | — | P1 | — | **DONE** |
