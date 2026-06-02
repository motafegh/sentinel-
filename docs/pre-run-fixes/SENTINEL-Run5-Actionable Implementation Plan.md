# SENTINEL Run 5 — Actionable Implementation Plan

> **Source document:** `SENTINEL_Run5_UNIFIED_PREFLIGHT_PROPOSAL.md` (revised 2026-06-02 with review findings)
> **Baseline:** `GCB-P1-Run4-no-asl-pw_best.pt` — epoch 32 — macro-F1 = 0.3362
> **Goal:** Break the F1 = 0.3362 ceiling by fixing all confirmed bugs (A1–A38 excl. A24/A31, plus NF-1–NF-11 from 2026-06-02 review) and 7 root causes (RC1–RC7) in strict dependency order before launching Run 5 training.
> **Key additions from review:** NF-8 elevated to Medium, A15 two-tier scope key, CEI labeling moved to Phase 7, realistic VRAM testing, NF-10 synthetic key primary, rollback decision trees, torch.compile gate, data archival/migration, aux_head monitoring.

> **Order is non-negotiable.** Each phase depends on the previous being complete and validated. Gates are mandatory blockers — do not skip them.

---

## Quick Reference — Files Touched

| File | Phases | Bugs |
|------|--------|------|
| `ast_extractor.py` | 0, 1 | A19, A20, A21, A22 |
| `graph_schema.py` | 1 | A1 |
| `hash_utils.py` | 1 | A2 |
| `graph_extractor.py` | 1, 2, 7 | A3–A18, NF-1, NF-2, NF-7, NF-10, NF-11, CEI labeler |
| `gnn_encoder.py` | 3 | A23, A25, A26, A27, NF-6 |
| `transformer_encoder.py` | 3 | A28, A29, A30 |
| `sentinel_model.py` | 3 | A25b, A32, A33, A34, NF-8 |
| `trainer.py` | 0, 4, 5 | A35, A36, A37, A38, NF-9 |
| `ml/src/training/training_logger.py` | 4 | — (new file per Training Log Specification) |
| `scripts/train.py` | 4, 5 | NF-4, NF-5 |
| `TrainConfig` | 5 | RC2, RC4 |
| `ml/calibration/` | 6 (post-training) | — |

---

## ⛔ Phase 0 — Critical Pre-Flight Safety Fixes

> **Proposal ref:** §4
> Fix these two bugs **before touching anything else**. Both can permanently corrupt training data or optimizer state, making all downstream work meaningless.

### 0.1 — Fix A20: `label=0` Hardcoded in Batch Extraction

- [x] **`ast_extractor.py` ~L307–311:** Remove `label=0` from the `partial(self.contract_to_pyg, ...)` call in the multiprocessing worker.
- [x] Build a `label_map` dict from the ground-truth CSV **before** any `pool.map` call.
- [x] Construct per-call args list pairing each contract path with its label from `label_map`.
- [x] Add `assert len(label_map) == len(batch)` immediately before the `pool.map` call.

**🚦 Gate 0.1 — DATA POISONING BLOCK (must pass before any other work)**
- [x] Verify `label_map` is populated from ground-truth CSV before pool spawn
- [ ] After a test extraction run, spot-check ≥ 100 contracts: labels in `.pt` files must match source CSV
- [ ] Zero all-zero-label batches confirmed absent
- [ ] **Rollback:** If `label_map` cannot be built → Stop. Restore CSV from VC. If spot-check reveals mismatches after fix → Revert worker change, debug `label_map` construction, re-test.

---

### 0.2 — Fix A38: NaN Loss `backward()` Runs Before NaN Check

- [x] **`trainer.py` ~L650 / L713:** Move `torch.isfinite(loss)` check to **before** `loss.backward()`.
- [x] If loss is not finite: skip `backward()`, skip `optimizer.step()`, zero stale gradients, increment `nan_loss_count`.
- [x] Add a post-gradient-clip, pre-step guard: if any parameter has non-finite gradients despite a finite loss (BF16 overflow), zero gradients and skip `optimizer.step()`.
- [x] Log `nan_loss_count` summary at end of each epoch.

**🚦 Gate 0.2 — ADAM STATE CORRUPTION BLOCK**
- [ ] Smoke test (5 steps with an injected NaN): confirm NaN is caught, `backward()` is skipped, and no crash occurs
- [x] Training epoch log includes `nan_loss_count` field
- [x] If `nan_loss_count > 0.5% × steps_per_epoch` at any point during Run 5 — **halt training immediately**
- [ ] **Rollback:** If `nan_loss_count > 0.5%` → Halt. Investigate LR, BF16 overflow, data quality. Restart from last clean checkpoint (before NaN occurred), not from current checkpoint.

---

## Phase 1 — Data & Schema Layer Fixes

> **Proposal ref:** §5
> Must be complete before any extraction work (Phases 2 or 7).

### 1.1 — Fix A1: Missing `max(NODE_TYPES)` Range Guard

- [x] **`graph_schema.py`:** After the existing `assert len(NODE_TYPES) == 13`, add:
  `assert max(NODE_TYPES.values()) == 12` with a message directing the developer to update the normalization divisor if this fires.

### 1.2 — Fix A2: Uppercase Hex Accepted in Hash Validation

- [x] **`hash_utils.py`:** Replace `int(hash_string, 16)` with a strict lowercase hex regex: `re.fullmatch(r'[0-9a-f]{32}', hash_string)`.
- [x] Add string type and length checks before the regex.

### 1.3 — Fix A3 + A32 + NF-2: Dynamic `_MAX_TYPE_ID` in Two Files + Decode-Side Hardcoding

- [x] **`graph_extractor.py` ~L113:** After `_MAX_TYPE_ID = float(max(NODE_TYPES.values()))`, add `assert _MAX_TYPE_ID == 12.0`.
- [x] **`sentinel_model.py` ~L75:** Same assertion — `assert _MAX_TYPE_ID == 12.0`.
- [x] Confirm both files have the assertion (the proposal cross-references §7.8 "A32 — already covered here").
- [x] **`graph_extractor.py` ~L1094 (NF-2):** Replace `int(round(x_list[-1][0] * 12))` with `int(round(x_list[-1][0] * _MAX_TYPE_ID))` — fix decode-side hardcoding, the mirror of A3/A32.

### 1.4 — Fix A19: Solc Binary Resolution Uses `Path.cwd()`

- [x] **`ast_extractor.py` ~L143:** Replace `Path.cwd()` with `get_project_root()` for deterministic solc binary resolution.

**🚦 Gate 1.1 — TOOLCHAIN CHECK (blocks extraction)**
- [x] `which solc && solc --version` returns successfully
- [x] `solc-select versions` shows range 0.4.0–0.8.35 available (97 versions present)
- [x] If `solc-select` is absent: install and configure before proceeding

### 1.5 — Fix A21: Worker `print()` Under Concurrency

- [x] **`ast_extractor.py` ~L223, L228:** Replace both `print()` calls with `logger.warning()` (workers already use `QueueHandler`/`QueueListener`).

### 1.6 — Fix A22: `torch.save` Without Error Handling

- [x] **`ast_extractor.py` ~L328:** Wrap `torch.save()` in `try/except (OSError, IOError)`.
- [x] On failure: log the error, append path to `failed_saves` list, continue the batch loop.
- [x] After the batch: if `failed_saves` is non-empty, raise an exception with the full list for the caller to handle.

---

## Phase 2 — Graph Extraction Layer Fixes

> **Proposal ref:** §6
> All `graph_extractor.py` bugs must be fixed here. **Re-extraction (Phase 7) must not happen until all of these are complete.** These bugs change graph structure — corrected graphs will differ from the v8 dataset.

### 2.1 — Fix A4 + A16: `assert` Used for Production Invariants

- [x] **`graph_extractor.py` ~L1253–1257 (A4):** Replace `assert` node/metadata-alignment check with `if condition: raise ValueError(...)`.
- [x] **`graph_extractor.py` ~L856–857 (A16):** Replace `assert` sentinel-range check with `if condition: raise ValueError(...)`.

### 2.2 — Fix A5: `except AttributeError` Scope Too Broad

- [x] **`graph_extractor.py` — `_compute_return_ignored`:** Narrow the `try` block to only the specific `func.calls_as_expression` Slither API call. Move all other logic outside the `try` block so unexpected `AttributeError`s propagate.

### 2.3 — Fix A6 + A10 + A18: Bare `except Exception: pass` (Three Silent-Failure Points)

- [x] **`graph_extractor.py` ~L312–313 (A6 — call target resolution):** Replace `pass` with debug-level log; add fallback counter.
- [x] **`graph_extractor.py` ~L493–494 (A10 — CFG node type):** Replace `pass` with warning-level log; add per-contract `_cfg_type_fallback_count` counter.
- [x] **`graph_extractor.py` ~L1160–1173 (A18 — ICFG map construction):** Replace `pass` with error-level log; add per-contract `_icfg_failure_count` counter.

**🚦 Gate 2.1 — EXTRACTION HEALTH (blocks Run 5, re-checked at Phase 7)**
After Phase 7 re-extraction:
- [ ] `_icfg_failure_count == 0`
- [ ] `_cfg_type_fallback_count / total_cfg_nodes < 0.01`
- [ ] CALL_ENTRY edge presence rate ≥ 64.2%
- [ ] RETURN_TO edge presence rate ≥ 55.6%

### 2.4 — Fix A7: Dead Code `_compute_in_unchecked`

- [x] **`graph_extractor.py` ~L331–360:** Replace function body with `raise NotImplementedError("Deprecated in v7 / BUG-L2 — any call site was not updated.")`.

### 2.5 — Fix A8: `is True` Identity Check in `_compute_has_loop`

- [x] **`graph_extractor.py` ~L376:** Replace `getattr(func, "is_loop_present", None) is True` with `bool(getattr(func, "is_loop_present", False))`.

### 2.6 — Fix A9: String-Based Class Check for `SolidityVariableComposed`

- [x] **`graph_extractor.py` — module import block:** Attempt to `from slither... import SolidityVariableComposed`. If import succeeds, use `isinstance(rv, SolidityVariableComposed)`. If import fails, log a prominent `WARNING: uses_block_globals will always be 0.0 — Timestamp/TOD detection severely degraded.`
- [x] Remove the `type(rv).__name__ == "SolidityVariableComposed"` string check at ~L424.

**🚦 Gate 2.2 — FEATURE VALIDATION (blocks Run 5)**
- [ ] After A9 fix and re-extraction: `uses_block_globals` is non-zero for ≥ 80% of Timestamp-positive contracts in the validation split.
- [ ] If below 80%: the feature is still broken — investigate Slither compatibility.

### 2.7 — Fix A11: Hardcoded Parent Feature Indices in `_build_cfg_node_features`

- [x] **`graph_extractor.py` ~L542–547:** Build a `name → index` map from `FEATURE_NAMES` (from `graph_schema`). Replace raw indices `p[1]`, `p[3]`, `p[4]`, `p[5]`, `p[9]` with named lookups (`visibility`, `view`, `payable`, `complexity`, `has_loop`).

### 2.8 — Fix A12: `n.node_id` Without Fallback in Sort Key

- [x] **`graph_extractor.py` ~L606–611:** Replace `n.node_id` in the `sorted()` key with `getattr(n, "node_id", 0)`.

### 2.9 — Fix A13: Silently Dropped CONTROL_FLOW Edges Not Logged

- [x] **`graph_extractor.py` ~L639–641:** Add an `else` branch that increments a per-contract `_dropped_cf_edges` counter and logs at debug level. Log a per-contract summary at info level.

### 2.10 — Fix A14: RETURN_TO Cartesian Product Includes Revert Paths

- [x] **`graph_extractor.py` ~L695–706:** Filter `callee_terminals` before building RETURN_TO edges — exclude `THROW` and `RETURN` (revert/unwind) node types. Only normal-return terminals produce RETURN_TO edges.
- [x] ⚠️ This changes edge structure — re-extraction is required (Phase 7 covers this).

### 2.11 — Fix A15: DEF_USE `def_map` Keyed by Variable Name Only (TWO-TIER SCOPE KEY)

- [x] **`graph_extractor.py` ~L752:** Change `def_map.setdefault(lval.name, [])` to use a two-tier scope key:
  - **Local variables** (declared inside a function, NOT in `contract.state_variables`): Key by `(containing_function, lval.name)` — prevents cross-function name collisions.
  - **State variables** (declared at contract level, found in `contract.state_variables`): Key by `(containing_contract, lval.name)` — state variables *should* have cross-function DEF_USE edges.
- [x] Determine variable scope by checking `contract.state_variables` (Slither provides this).
- [x] Apply the same scope resolution on the `use_map` lookup side: a reference to a state variable from within a function must look up `(contract, variable_name)`, not `(function, variable_name)`.
- [x] ⚠️ This changes DEF_USE edge structure — re-extraction is required (Phase 7 covers this).

### 2.12 — Fix A17: Exception Routing by String Keyword Matching

- [x] **`graph_extractor.py` ~L1059–1067:** Restructure exception handling to use type-based checks first — catch `SlitherError` and `SolcError` by type. Fall back to string keyword matching only for bare `Exception` instances. Add a tracked TODO comment to replace remaining string matching with `isinstance` checks on Slither's exception hierarchy.

### 2.13 — Fix NF-1: EMITS Edge Key Mismatch in Fallback Path

- [x] **`graph_extractor.py` ~L1305–1317:** Before the fallback loop, build a `short_name → canonical_name` map from `contract.events` (e.g., `{getattr(e, "name", None): (getattr(e, "canonical_name", None) or e.name) for e in contract.events}`).
- [x] In the fallback loop, translate `key = getattr(ir, "name", None)` using this map: `key = event_name_map.get(key, key)` before passing to `_add_edge`.
- [x] ⚠️ This changes EMITS edge presence for Solidity <0.4.21 contracts — re-extraction required (Phase 7 covers this).

### 2.14 — Fix NF-7: Silent 0.0 Return on Failure in `_compute_external_call_count` and `_compute_uses_block_globals`

- [x] **`graph_extractor.py` ~L394–405 (`_compute_external_call_count`):** Inside the `except Exception` block, add `logger.debug("_compute_external_call_count failed for %s: %s", getattr(func, 'canonical_name', '?'), exc)` and increment a module-level counter `_ext_call_fail_count`.
- [x] **`graph_extractor.py` ~L419–432 (`_compute_uses_block_globals`):** Same pattern — add debug log and increment `_block_globals_fail_count`. Do NOT change the return value from `0.0` to `-1.0`.
- [x] Log both counters in the per-contract extraction summary.

### 2.15 — Fix NF-10: Duplicate Function Name — Synthetic Key as Primary, Skip as Fallback

- [x] **`graph_extractor.py` ~L1134–1148:** When `_add_node` returns `None` (duplicate canonical name), **assign a unique synthetic key** (e.g., `canonical_name + "__override__" + str(function_index)`) and re-attempt `_add_node` with the synthetic key. This preserves the overriding function's CFG, which is critical because overrides often introduce vulnerabilities.
- [x] If the second function's CFG is identical to the first (verified by edge density analysis — a degenerate case), fall back to skipping CFG construction with `continue` after logging.
- [x] Add `_duplicate_func_count` as a per-contract accumulator and log it in the extraction summary.
- [x] ⚠️ This changes graph structure for contracts with overloaded/inherited functions — re-extraction required (Phase 7 covers this).

### 2.16 — Fix NF-11: `_add_edge` Silently Drops ALL Edge Types

- [x] **`graph_extractor.py` ~L1274–1279 (`_add_edge` inner function):** Add a `_edge_drop_counts: dict[int, int]` accumulator in the outer scope (keyed by edge type integer).
- [x] Inside `_add_edge`, when `si is None or di is None`, increment `_edge_drop_counts[etype]` and optionally log at debug level.
- [x] After the edge loop, log a per-type summary: `"Edge drop counts — CALLS: N, READS: N, WRITES: N, EMITS: N, INHERITS: N"`.
- [x] Note: CONTROL_FLOW drops are already addressed by A13 (step 2.9); this step covers ALL other types.

---

## Phase 3 — Model Architecture Fixes

> **Proposal ref:** §7
> These fix correctness and performance of the model forward pass. No training dynamics change, but incorrect outputs and fragile dependencies are removed.

### 3.1 — Fix A25: `edge_index.max()` O(E) Scan on Every Forward Pass

- [x] **`gnn_encoder.py` ~L389–393:** Move the `edge_index.max()` integrity check to `DualPathDataset.__getitem__` or the collation function.
- [x] Add a `validate_graph_integrity` flag (default `False` in production) to gate the check at inference time.

### 3.2 — Fix A26: `next(self.parameters())` Called Twice Per Forward Pass

- [x] **`gnn_encoder.py` ~L398, L521:** Cache `self._param_dtype` in `__init__`. In forward, read from the cached value.
- [x] Add `refresh_dtype_cache()` method that callers must invoke after any runtime dtype cast (`.float()`, `.half()`, `.bfloat16()`).

### 3.3 — Fix A27: `num_layers` Stored but Hardcoded to 8

- [x] **`gnn_encoder.py` ~L196:** Introduce `SENTINEL_GNN_NUM_LAYERS = 8`. If `num_layers != 8` is passed, raise `ValueError` with a message explaining the architecture is fixed at 8.

### 3.4 — Fix A23: `last_weight_stds` NaN for N=1

- [x] **`gnn_encoder.py` ~L123:** Replace `.std(0)` with `.std(0, unbiased=False)` followed by `.nan_to_num(0.0)`.

### 3.5 — Fix A28: `except (ImportError, ValueError)` Catches Real BERT Load Errors

- [x] **`transformer_encoder.py` ~L142–147:** Narrow the except to `ImportError` only (flash_attention_2 not installed → fallback to SDPA). Let `ValueError` propagate — a corrupted `config.json` or missing model file is a real error that must surface.

### 3.6 — Fix A29: Python Loop for Prefix Mask Construction

- [x] **`transformer_encoder.py` ~L241–242 and ~L284–285 (two occurrences):** Replace the `for b in range(B)` loop with a vectorized broadcast comparison using `torch.arange` vs. `gnn_prefix_counts`.

### 3.7 — Fix A30: `_word_embeddings` Fragile Hardcoded PEFT Path

- [x] **`transformer_encoder.py` ~L168–170:** Replace the five-level hardcoded path with a property that tries multiple known PEFT internal paths in order of precedence. If none yield an `nn.Embedding`, raise `AttributeError` with a PEFT version compatibility message.
- [x] Validate the path at `__init__` time (call the property, catch the error) so failures surface at construction, not at the first forward pass.

### 3.8 — Fix A32: `_MAX_TYPE_ID` in `sentinel_model.py`

- [x] Already covered in Phase 1 step 1.3 — confirm the assertion is present in both files.

### 3.9 — Fix A33: `select_prefix_nodes` Python Loop Over Batch Dimension

- [x] **`sentinel_model.py` ~L305:** Pre-compute priority scores for all nodes in a single tensor operation. Use `torch.topk` per-graph via the PyG batch vector. A hybrid approach (tensor scores + looped topk) is acceptable for Run 5.

### 3.10 — Fix A34: Secondary Sort Uses Post-GAT Embedding, Not Raw Feature

- [x] **`sentinel_model.py` ~L326:** Replace `g_embs[local_idx, _EXT_CALL_DIM]` (post-GAT 256-dim output) with `graphs.x[local_idx, _EXT_CALL_DIM]` (raw input feature) for the prefix selection secondary sort.
- [x] **Implementation note:** Raw features are available as `graphs.x` (the PyG Data object's node feature tensor). Update `select_prefix_nodes` function signature to accept both `g_embs` and `graphs` (or specifically `graphs.x`) so the secondary sort can access raw features.
- [x] ⚠️ This changes prefix node selection — re-run EXP-A4 (Aux Eye Contribution) after this fix to verify prefix quality improves.

### 3.11 — Fix A25b: `compute_prefix_attention_mean` Discards `node_counts`

- [x] **`sentinel_model.py` ~L544–546:** Unpack `gnn_prefix, node_counts = gnn_prefix` (do not discard with `_`). When averaging attention, average over only real node positions (indices 0 to `node_counts[g]`), not over all K=48 including padding. (Diagnostic-only fix — no training impact.)

### 3.12 — Fix NF-6: Phase 2 Layers 3/4 Ignore `phase2_edge_types` Ablation

- [ ] **DEFERRED to Run 6** — per Known Non-Fixes list (zero training impact; NF-6 comment added to `gnn_encoder.py` to mark the location). Required for correct post-Run-5 ablation experiments only.

### 3.13 — Fix NF-8: Empty Batch Guard Returns Inconsistent Aux Dict Keys (ELEVATED TO MEDIUM)

- [x] **`sentinel_model.py` ~L422–426:** Update `aux_zeros` to include `"phase2"` and `"jk_entropy"` keys matching the normal-path dict.
- [ ] Verify the trainer uses a consistent key set from the aux dict — confirm it does not raise `KeyError` on empty-batch epochs.
- [ ] Add a unit test: create a zero-size batch, run forward pass, verify no `KeyError` when trainer accesses `aux["phase2"]` and `aux["jk_entropy"]`.

**🚦 Gate 3.1 — torch.compile RE-VALIDATION (blocks Run 5)**
After all Phase 3 fixes are applied (forward pass structure has changed significantly):
- [ ] Run a 2-epoch smoke test with `torch.compile(model, dynamic=True)` enabled
- [ ] Verify no `RuntimeError` from compiled graph
- [ ] Verify output is numerically correct (compare compiled vs eager on same input)
- [ ] If compile fails: disable `torch.compile` for Run 5, file as Run 6 fix item

---

## Phase 4 — Training Loop Fixes

> **Proposal ref:** §8
> Correctness and efficiency fixes to `trainer.py`. No architecture or dynamics change.

### 4.1 — Fix A35: `_FocalFromLogits` Unpicklable Local Class

- [x] **`trainer.py`:** Moved `_FocalFromLogits` to module level as parameterised class `_FocalFromLogits(focal: FocalLoss)`. Picklable and DDP-safe.

### 4.2 — Fix A36: `compute_pos_weight` Re-Reads Label CSV Every Call

- [x] **`trainer.py`:** Changed signature to accept `DualPathDataset`; computes `pos_weight` from `dataset._label_map` and `dataset.paired_hashes` — no CSV I/O.

### 4.3 — Fix NF-4: `--gnn-layers` CLI Default = 7 vs TrainConfig Default = 8

**⚠️ HIGH — silently runs wrong architecture if not fixed before Run 5**

- [x] **`scripts/train.py`:** `default=7` → `default=8`.
- [x] Warning logged if `args.gnn_layers != 8`.
- [ ] Verify the launched Run 5 training log shows `gnn_num_layers=8` before continuing past epoch 1.

**🚦 Gate NF-4 (blocks Run 5 launch):**
- [ ] Confirm `gnn_num_layers=8` appears in the Run 5 configuration log at epoch 0.
- [ ] If it shows `7` — halt, apply this fix, relaunch.

### 4.4 — Fix NF-9: `AdamW(fused=True)` Crashes on CPU

- [x] **`trainer.py`:** `fused=True` → `fused=(device == "cuda" or device.startswith("cuda:"))`. Zero training impact.

### 4.5 — Fix A37: Threshold Sweep Every Validation Epoch

- [x] **`trainer.py`:** Added `threshold_tune_interval: int = 10` to `TrainConfig`. Sweep runs every N epochs and always at the final epoch; `_cached_tuned_thresholds` reused between sweeps.
- [x] **`scripts/train.py`:** `--threshold-tune-interval` CLI arg wired to `TrainConfig`.

### 4.6 — Implement Structured Training Logger (per Training Log Specification)

> **Spec doc:** `docs/pre-run-fixes/SENTINEL-Run5-Training-Log-Specification.md`

- [x] **`ml/src/training/training_logger.py` (new file):** `StructuredLogger` + `TrainingAbortError` per Spec §10.2. Three streams: `step_metrics.jsonl`, `epoch_summary.jsonl`, `alerts.jsonl`.
- [x] **Data integrity & label health (Spec §1):** `log_startup()` — data hash (§1.8), archive verification (§1.9); `check_batch()` — poisoned-label (§1.1) and NaN/Inf input (§1.5) checks with WARN_SKIP.
- [x] **NaN/Inf & gradient health (Spec §2):** `check_loss()` KILL on NaN/Inf (§2.1/9.1.1); `check_parameters()` KILL on param NaN (§2.2/9.1.2); `check_adam_state()` KILL on Adam state NaN (§2.6/9.1.3); `check_grad_norm()` rolling history + spike WARN (§2.4/2.8/9.3.2).
- [x] **AUC & probability quality (Spec §3B):** `compute_auc_metrics()` — per-label AUC-ROC/PR, macro/micro, epoch deltas (§3B.1–3B.6,12–13); `compute_brier()` — per-label + overall Brier (§3B.7–8); `compute_ece()` — ECE pooled (§3.9); `compute_prob_stats()` (§3B.10); `check_f1_auc_divergence()` WARN (§3B.15/9.3.6c); AUC-PR < 0.1 WARN (§9.3.6b); Brier > 0.4 WARN (§9.3.6d).
- [x] **Model-specific logs (Spec §4):** `check_aux_head()` norms + WARN if near-zero (§4.3/4.4/9.3.3); `check_jk_entropy()` + WARN if collapsed (§4.2/9.3.4).
- [x] **Resource & VRAM (Spec §6):** `check_vram()` peak MB + WARN > 7500 MB (§6.3/9.3.1).
- [x] **Epoch-level summary (Spec §8):** `build_epoch_summary()` — all 37 fields (§8.1–8.37).
- [x] **Alert tiers (Spec §9):** KILL raises `TrainingAbortError`; WARN_SKIP returns skip flag; WARN logs only.
- [x] **Wire logger into `trainer.py`:** `StructuredLogger` created at training start (after `mlflow.log_params`); `log_startup()` called before epoch loop; per-epoch summary + AUC/Brier/ECE computed and `log_epoch()` called; `close()` at training end. `_y_true`/`_y_probs` passed back from `evaluate()` for logger computations. `TrainConfig.log_dir` field added.
- [x] **`scripts/train.py`:** `--log-dir` CLI arg wired to `TrainConfig.log_dir`.

**🚦 Gate 4.6 — LOGGER STARTUP VERIFICATION (blocks Run 5 launch)**
- [ ] At Run 5 epoch 0 startup: `step_metrics.jsonl`, `epoch_summary.jsonl`, `alerts.jsonl` all created.
- [ ] Data integrity hash (item 1.8) logged before first training step.
- [ ] Archive verification (item 1.9) logged and confirmed.
- [ ] If any KILL-level alert fires in first epoch — do NOT continue. Investigate before restarting.

---

## Phase 5 — Training Interventions for Phase 2 Signal

> **Proposal ref:** §9
> These directly target the F1 = 0.3362 ceiling. All bugs in Phases 0–4 remove impediments; these interventions move the needle.

### 5.1 — Intervention 1: Enable Phase 2 Auxiliary Loss (RC2)

- [x] **`TrainConfig`:** Confirm `aux_phase2_loss_weight = 0.10` is set (already in commit 9310046).
- [x] **`trainer.py` — `train_epoch()` signature:** Confirmed `aux_phase2_loss_weight=config.aux_phase2_loss_weight` at call site (line 1567) — default 0.0 in signature but always overridden by TrainConfig value.
- [x] Verify `aux_head_phase2` connected: Phase 2 embeddings → `aux_head_phase2` → weighted BCE → summed into total loss at trainer.py:664.
- [ ] Add per-epoch logging of `aux_head_phase2.weight.norm()` and `aux_head_phase2.bias.norm()` — handled by StructuredLogger `check_aux_head()` (Phase 4.6); verify at Run 5 epoch 1.

**🚦 Gate 5.1 — AUX LOSS VERIFICATION (blocks Run 5 continuation)**
- [ ] At Run 5 epoch 1: `aux_phase2_loss` is logged separately and is **non-zero**
- [ ] By epoch 2: `aux_head_phase2` parameters have non-zero gradient norms
- [ ] By epoch 5: `aux_head_phase2.weight.norm()` is above initialization value
- [ ] If `aux_phase2_loss` is 0.0 or absent — **do not continue Run 5** until confirmed.

### 5.2 — Intervention 2: CEI Path Supervision for Reentrancy (RC1, RC7)

> **IMPORTANT:** CEI path labeling must be computed on **re-extracted v9 graphs** (during Phase 7), not on the current buggy v8 graphs. The v8 graphs have incorrect RETURN_TO edges (A14), broken ICFG maps (A18), and other structural bugs that would produce wrong CEI labels. Implementation is integrated into Phase 7.

- [ ] **`graph_extractor.py` (during Phase 7 re-extraction):** For each contract, compute and store `has_cei_path = 1` if a `CFG_NODE_CALL → (CONTROL_FLOW → CFG_NODE_WRITE)` path exists within 8 hops via Phase 2 edges. Store as a scalar in the graph `.pt` file.
- [ ] **`sentinel_model.py`:** Expose Phase 2 CEI-pooled embedding.
- [ ] **`trainer.py`:** Add `aux_cei_loss_weight` parameter. Compute BCE loss on CEI logit vs `has_cei_path` label. Add weighted to total loss. *Only enable after Gate 7.5 passes.*

**🚦 Gate 7.5 (was Gate 5.2 — now evaluated on v9 data after Phase 7):**
After Phase 7 re-extraction, validate CEI labels on v9 data:
- [ ] ≥ 60% of Reentrancy-positive training contracts have `has_cei_path = 1`
- [ ] ≤ 5% of Reentrancy-negative contracts have `has_cei_path = 1`
- [ ] If positive coverage < 40%: the CEI path labeler has a bug — do not enable `aux_cei_loss_weight`.

### 5.3 — Intervention 3: Timestamp Size Confound (Option A — Evaluation Only)

- [x] **Option A implemented (commit 4af3761):** `evaluate()` collects `num_nodes` per sample via `bincount(graphs.batch)`; reports `f1_Timestamp_{small,medium,large}` and `n_Timestamp_{stratum}` in metrics dict using EXP-L7 boundaries (<100 / 100–300 / >300). Wired into val_metrics → MLflow logging. Zero training code risk.
- [ ] Note: Option B (adversarial size regularizer) is deferred to Run 6.

### 5.4 — Intervention 4: Raise `max_nodes` to 2048 (IMP-D1)

- [x] **`--fusion-max-nodes` CLI arg added (commit 4af3761):** `scripts/train.py` — default 1024; wire to `TrainConfig.fusion_max_nodes`. Raise to 2048 after Gate 5.3 passes.
- [ ] **Before Phase 7:** Pass `--fusion-max-nodes 2048` on re-extraction run so graphs with 1025–1735 nodes are NOT truncated.
- [x] Verified: `fusion_max_nodes` already wires through `TrainConfig` → `SentinelModel` → `CrossAttentionFusion.max_nodes` (no code change needed in gnn_encoder.py).
- [ ] ⚠️ Raise `fusion_max_nodes` to 2048 **only after Gate 5.3 passes** and Phase 7 re-extraction is complete.

**🚦 Gate 5.3 — MEMORY GATE (blocks Run 5 with max_nodes=2048)**
- [x] Script created: `ml/scripts/vram_gate_test.py` — synthetic worst-case (max_nodes×batch_size full training step, AMP, fused AdamW, BF16). PASS<7500MB / WARN<8000MB / FAIL>8000MB.
- [ ] **Run before Phase 7:** `TRANSFORMERS_OFFLINE=1 PYTHONPATH=. python ml/scripts/vram_gate_test.py`
- [ ] Also test at `batch_size=32` — if it fits, use it for speed.
- [ ] If VRAM > 7.5 GB at batch_size=16: reduce `batch_size` to 8 or run `--max-nodes 1536`.
- [ ] Document the chosen configuration in the Run 5 log.

### 5.5 — Fix NF-5: Expose `aux_phase2_loss_weight` as CLI Arg

- [x] **`scripts/train.py` (commit 4af3761):** `--aux-phase2-loss-weight` (default 0.10) wired to `TrainConfig.aux_phase2_loss_weight`.
- [x] `--aux-cei-loss-weight` (default 0.0, Phase 7 placeholder) added as inert arg.
- [x] `--jk-entropy-reg-lambda` (default 0.005) already existed from Phase 4.

### 5.6 — Phase 2 Monitoring Configuration

> **Full monitoring requirements:** `docs/pre-run-fixes/SENTINEL-Run5-Training-Log-Specification.md` §4 (model-specific logs: JK weights, aux head norms, per-layer GNN output) and §3B (AUC/probability quality metrics, critical for agent module input). The table below is the Phase 2-specific subset — implement the full spec in Phase 4.6.

Set up the following metrics to be logged **every epoch** of Run 5:

| Metric | Epoch 10 Target | Alert Threshold | Action |
|--------|----------------|-----------------|--------|
| Phase 2 JK weight (mean) | Rising toward 0.35 | < 0.322 at ep10 | Aux loss not working — investigate |
| Phase 2 / Phase 1 gradient norm ratio | Rising | P2/P1 < 72% at ep10 | Phase 2 not receiving gradient |
| `aux_phase2_loss` | Decreasing | Not decreasing by ep5 | Phase 2 not learning |
| `aux_head_phase2.weight.norm()` | Increasing from init | Still at init at ep5 | Aux loss path broken — check connectivity |
| Reentrancy F1 | Rising from 0.169 | < 0.15 at ep15 | Regression — adjust `aux_cei_loss_weight` |
| ExternalBug F1 | Rising from 0.000 | Still 0.000 at ep20 | Phase 2 structurally unused |

**🚦 Gate 5.4 — RUN 5 GO/NO-GO AT EPOCH 10 (with Rollback Decision Tree)**
If Phase 2 JK weight < 0.33 **AND** Reentrancy F1 < 0.18 at epoch 10, **pause Run 5** and follow this decision tree:

1. [ ] **Check `aux_phase2_loss` logged and non-zero at ep10?**
   - No → Loss weight not reaching `train_epoch()`. Fix propagation (§5.1), resume from ep10 checkpoint.
   - Yes → Continue to step 2.

2. [ ] **Check `aux_phase2_loss` is decreasing?**
   - No → Aux head may not be connected. Check `aux_head_phase2` weight norms — if still at init, gradient path is broken. Debug connectivity, then resume.
   - Yes → Continue to step 3.

3. [ ] **Check `aux_head_phase2.weight.norm()` trend?**
   - Still at init → Gradient not reaching head. Check if head is in `optimizer.param_groups`.
   - Increasing → Head is learning but Phase 2 embeddings aren't changing. Continue to step 4.

4. [ ] **Phase 2 head capacity (RC3) is likely the bottleneck:**
   - Try `aux_phase2_loss_weight = 0.20` for 5 more epochs.
   - If JK weight still doesn't rise → **Escalate to Run 6 with multi-head Phase 2 (4–8 heads).**
   - Do not continue training to ep60 if the signal is confirmed structurally blocked.

---

## Phase 6 — Calibration & Threshold Fixes

> **⚠️ CRITICAL:** This phase is executed **AFTER Phase 8 (Run 5 training)** completes. It does NOT block Phases 7 or 8. Do not attempt to calibrate before training.

### 6.1 — Refit Temperature Scaling

- [ ] After Run 5 completes: set model to eval mode.
- [ ] Collect all validation logits and labels.
- [ ] Optimize per-class temperature parameters T via NLL minimization on the validation set.
- [ ] Save to `ml/calibration/temperatures_run5.json`.

**🚦 Gate 6.1**
- [ ] Post-fitting ECE on the main head < 0.05. If not: consider per-eye calibration or Platt scaling.

### 6.2 — Save Final Tuned Thresholds

- [ ] Ensure threshold tuning at the final epoch (from A37 fix, Phase 4) saves tuned thresholds alongside the checkpoint for consistent inference.

---

## Phase 7 — Data Re-Extraction, Archival & Migration

> **Proposal ref:** §11
> Must happen **after** all Phase 0–5 fixes are applied. Re-extracting with a broken extractor would require re-extraction again.
> **Also includes comprehensive archival of all v8-era data and migration to v9 data.**

### 7.0 — Pre-Extraction Data Archival (MANDATORY — before any new extraction)

Before running re-extraction, move **all** v8-era artifacts to clearly-labeled archive directories. Run 5 must train exclusively on v9 data.

- [ ] Archive v8 graph .pt files: `ml/data/graphs/` → `ml/data/archive/graphs_v8_pre_run5/`
- [ ] Archive v8 cached dataset: `ml/data/cached_dataset_v8.pkl` → `ml/data/archive/cached_dataset_v8.pkl`
- [ ] Archive v8 token files: `ml/data/tokens_windowed/` → `ml/data/archive/tokens_windowed_v8/`
  - Confirm tokens are schema-compatible with v9 (no retokenization needed if only graph structure changed)
- [ ] Archive v8 label CSV: `ml/data/processed/multilabel_index_cleaned.csv` → `ml/data/archive/multilabel_index_cleaned_v8.csv`
- [ ] Archive v8 pre-cleaning index: `ml/data/processed/multilabel_index.csv` → `ml/data/archive/multilabel_index_v8.csv`
- [ ] Archive v8 splits: `ml/data/splits/deduped/` → `ml/data/archive/splits_v8_deduped/`
- [ ] Archive v8 build_multilabel_index output: any index built from v8 graphs → `ml/data/archive/index_v8/`
  - **Do NOT reuse** this index — must be rebuilt from v9 (NF-3)
- [ ] Archive all pre-Run-5 checkpoints: `ml/checkpoints/*.pt` → `ml/data/archive/checkpoints_pre_run5/`
- [ ] Archive Run 1–4 training logs: `ml/logs/` → `ml/data/archive/logs_pre_run5/`
- [ ] Create `ml/data/archive/v8_archive_manifest.txt` listing all archived files with counts and sizes
- [ ] Verify source directories are empty (or contain only a README pointing to archive)
- [ ] Verify archive directories contain all expected files (count .pt files, verify CSV row counts)
- [ ] **Do NOT delete the archive** — it is the fallback if re-extraction fails

### 7.1 — Pre-Extraction Preparation

- [ ] Confirm all Phase 2 fixes (A3–A18, NF-1, NF-2, NF-7, NF-10, NF-11) are applied to `graph_extractor.py`.
- [ ] Confirm A15 two-tier scope key fix is applied (function scope for locals, contract scope for state vars).
- [ ] Confirm NF-10 synthetic key approach is applied for duplicate functions.
- [ ] Confirm `max_nodes=2048` is set (Intervention 4 from Phase 5).
- [ ] Confirm A20 fix (Phase 0) is applied — labels will be correct in re-extracted graphs.
- [ ] Confirm Gate 1.1 has passed (solc toolchain is functional).
- [ ] Confirm CEI labeler is implemented and integrated into the extraction pipeline.

### 7.2 — Run Full Extraction

- [ ] Run full extraction with 10 workers (~30 min expected based on previous runs).
- [ ] CEI path labeling runs as part of the extraction pipeline on the newly extracted graphs.
- [ ] Rebuild cache: `ml/data/cached_dataset_v9.pkl`
- [ ] Rebuild multilabel index from v9 graphs only (do NOT use v8 index).
- [ ] Regenerate train/val/test splits if label cleaning removes additional contracts.
- [ ] Generate v9-cleaned label CSV: `ml/data/processed/multilabel_index_cleaned_v9.csv`

### 7.3 — Post-Extraction Data Migration Verification

Verify that all code paths and configuration files point to v9 data:

- [ ] `ml/data/graphs/` contains v9 .pt files (new extraction output)
- [ ] `ml/data/cached_dataset_v9.pkl` exists and is non-empty
- [ ] `train.py --cache-path` defaults or is set to v9 cache: `ml/data/cached_dataset_v9.pkl`
- [ ] `train.py --label-csv` points to v9-cleaned CSV: `ml/data/processed/multilabel_index_cleaned_v9.csv`
- [ ] `train.py --splits-dir` points to v9 splits: `ml/data/splits/v9_deduped/`
- [ ] No Python imports or hardcoded paths reference v8 data (search codebase for "v8" references)
- [ ] `build_multilabel_index.py` run against v9 graphs only
- [ ] Old v8 cache file not accidentally loaded (confirm `cached_dataset_v8.pkl` is in archive only)
- [ ] All stale v8 references removed from active directories

### 7.4 — Post-Extraction Validation

**🚦 Gate 7.1 — EXTRACTION COMPLETENESS**
- [ ] Total extracted graphs ≥ 41,000 (v8 baseline: 41,576)
- [ ] Skip count not significantly higher than v8 baseline
- [ ] Fail count = 0

**🚦 Gate 7.2 — EXTRACTION QUALITY (same criteria as Gate 2.1)**
- [ ] `_icfg_failure_count == 0`
- [ ] `_cfg_type_fallback_count / total_cfg_nodes < 0.01`
- [ ] CALL_ENTRY edge presence rate ≥ 64.2%
- [ ] RETURN_TO edge presence rate ≥ 55.6%

**🚦 Gate 7.3 — LABEL INTEGRITY**
- [ ] Label distribution matches cleaned CSV — no all-zero-label batches
- [ ] Spot-check ≥ 100 contracts: labels in `.pt` files match ground-truth CSV
- [ ] `uses_block_globals` is non-zero for Timestamp-positive contracts (Gate 2.2 follow-up)
- [ ] **(NF-3 check):** Do NOT run `build_multilabel_index.py` against the archived v8 (A20-corrupted) graphs. Only run it against Phase 7 re-extracted v9 graphs. Verify non-BCCC contracts (SolidiFI, SmartBugs) have correct labels in the resulting index.
- [ ] **(Pre/post label distribution comparison):** Compute and log label distribution before (v8) and after (v9) re-extraction. Flag any class where positive count changes by > 10%. Log comparison to `ml/data/v8_v9_label_comparison.txt`.

**🚦 Gate 7.4 — SCHEMA CONSISTENCY**
- [ ] All extracted graphs have consistent `feature_schema_version`
- [ ] `NODE_FEATURE_DIM = 11` in all graphs
- [ ] `max_nodes=2048` reflected in extraction output metadata

**🚦 Gate 7.5 — CEI LABEL QUALITY (evaluated on v9 data)**
- [ ] ≥ 60% of Reentrancy-positive training contracts have `has_cei_path = 1`
- [ ] ≤ 5% of Reentrancy-negative contracts have `has_cei_path = 1`
- [ ] If positive coverage < 40%: the CEI path labeler has a bug — do not enable `aux_cei_loss_weight`

---

## Phase 8 — Run 5 Execution & Monitoring

> **Proposal ref:** §12
> Only begin Run 5 after all prior phase gates have passed.
>
> **Logging spec:** `docs/pre-run-fixes/SENTINEL-Run5-Training-Log-Specification.md` — exhaustive log checklist (11 sections, 100+ items) mapping every metric to a known bug or risk. The `StructuredLogger` from Phase 4.6 must be active before training starts. Gate 4.6 (logger startup verification) must pass before proceeding.

### 8.1 — Run 5 Configuration

Confirm the following parameters before launching:

| Parameter | Value | Status |
|-----------|-------|--------|
| GNN layers | 8 | Unchanged from Run 4 |
| `gnn_heads` | 8 (Phase 1), 1 (Phase 2) | RC3 gap unchanged — Run 6 candidate |
| `gnn_prefix_k` | 48 | Unchanged |
| `gnn_prefix_warmup_epochs` | 15 | Unchanged |
| `aux_phase2_loss_weight` | **0.10** | NEW — was 0.0 in Run 4 |
| `aux_cei_loss_weight` | **TBD after Gate 7.5** | NEW — conditional on v9 CEI labels |
| `jk_entropy_reg_lambda` | 0.005 | Unchanged |
| `max_nodes` | **2048 (or 1536 per Gate 5.3)** | NEW — was 1024 |
| `dos_loss_weight` | 0.5 | Unchanged |
| `pos_weight` | Not passed to ASL | Unchanged (NC-4 reverted) |
| Backbone | GraphCodeBERT + LoRA | Unchanged |
| Epochs | 100 with early-stop patience 30 | Unchanged |
| **Cache path** | **`ml/data/cached_dataset_v9.pkl`** | **v9 only** |
| **Label CSV** | **v9-cleaned CSV** | **v9 only** |
| **Splits dir** | **v9 splits** | **v9 only** |

### 8.2 — Monitoring Schedule

> **Full monitoring spec:** `docs/pre-run-fixes/SENTINEL-Run5-Training-Log-Specification.md` §9 (alert-grade anomalies) and §10.4 (post-run analysis checklist, 14 questions). The table below covers Phase-2-specific checkpoints — the spec covers all 100+ log items and their thresholds.

| Epoch | Check | Expected |
|-------|-------|----------|
| 1 | `aux_phase2_loss` logged, non-zero | Confirms aux loss reaches optimizer (Gate 5.1) |
| 1 | `aux_head_phase2.weight.norm()` logged | Confirms head is receiving gradient |
| 1–5 | `nan_loss_count` per epoch | Should be 0 (Gate 0.2) |
| 5 | Phase 2 JK weight | Rising from 0.322 baseline |
| 5 | `aux_head_phase2.weight.norm()` above init | If still at init, aux loss path is broken |
| 10 | **Gate 5.4 go/no-go** | JK weight > 0.33 OR Reentrancy F1 > 0.18 |
| 15 | Prefix warmup ends | `gnn_to_bert_proj` begins receiving gradient |
| 16 | Loss spike | Brief expected increase from prefix activation |
| 20 | `prefix_attention_mean` | Should be > 0.005 |
| 20 | ExternalBug F1 | Should be > 0.000 |
| 30+ | Steady-state | F1 improving beyond 0.3362 |

### 8.3 — Early Termination Criteria

**Immediate halt — investigate before continuing:**
- [ ] `nan_loss_count > 0.5% × steps_per_epoch` (Gate 0.2)
- [ ] Phase 2 JK weight falls below 0.25 after ep5 (structural collapse)
- [ ] Loss is NaN/Inf for > 3 consecutive steps

**Pause and evaluate at epoch 10 (Gate 5.4 — follow rollback decision tree):**
- [ ] Phase 2 JK weight < 0.33 AND Reentrancy F1 < 0.18

**Normal early stopping:**
- [ ] F1-macro does not improve for 30 consecutive epochs

---

## Post-Training Checklist

> **Proposal ref:** §12.4
> **Post-run analysis:** Use `docs/pre-run-fixes/SENTINEL-Run5-Training-Log-Specification.md` §10.4 (14-question checklist) with the structured logs from Phase 4.6 to answer: correct v9 data used? NaN/Inf appeared? AUC improving? F1-AUC divergence? Probabilities trustworthy for agent?

- [ ] Save best checkpoint with full configuration metadata
- [ ] Re-fit temperature scaling → `ml/calibration/temperatures_run5.json` (Phase 6, Gate 6.1)
- [ ] Run behavioral test suite (20 contracts, 19 expected detections) — target > 10/19
- [ ] Run full evaluation on test split with tuned thresholds
- [ ] Log per-class F1, precision, recall — compare with Run 4 baseline (0.3362)
- [ ] Run size-stratified Timestamp evaluation (Intervention 3, Option A)
- [ ] Log JK weight distribution at final epoch for Phase 2 analysis
- [ ] Log Phase 2 / Phase 1 gradient norm ratio at final epoch
- [ ] Verify Run 5 checkpoint loads correctly with all v9 data paths
- [ ] Archive Run 5 checkpoint and logs alongside the v8 archive

---

## F1 Outcome Expectations

> **Proposal ref:** §14.1

| Scenario | Expected macro-F1 | Condition |
|----------|-------------------|-----------|
| Pessimistic | 0.34–0.36 | Bug fixes only; Phase 2 signal unchanged |
| Base case | 0.38–0.42 | `aux_phase2_loss` effective; JK Phase 2 weight > 0.35 |
| Optimistic | 0.44–0.48 | All interventions succeed; CEI aux loss active |

---

## Known Non-Fixes (Run 6+ Candidates)

> **Proposal ref:** §14.3 and §15 — explicitly out of scope for Run 5.

| Item | Why Deferred |
|------|-------------|
| RC3: Phase 2 multi-head attention (1→4–8 heads) | Architectural change; requires JK weight dynamics testing |
| RC5: Multi-hop DEF_USE propagation | Risk of dilution (v8-AB showed DEF_USE can hurt) |
| RC6: Phase 2/Phase 3 interaction redesign | Major architectural change; separate proposal needed |
| Adversarial size regularizer (Option B, Timestamp) | Higher risk; defer if Option A makes confound visible |
| `gnn_to_bert_proj` float32 preservation | Needed only if Run 5 shows same BF16 stagnation as Run 4 |
| BUG-M5: Brainmab mislabeled contract | Non-blocking for training |
| BUG-M6: Stale `feature_schema_version='v4'` in token files | Auto-resolves on retokenize |
| BUG-M7: 8.5% of graphs have empty `contract_path` | Non-blocking for training; affects post-hoc analysis only |
| BUG-L3: Hash-based graph-token pairing | Low priority |
| NF-6: Phase 2 ablation bypass (Layers 3/4 ignore `phase2_edge_types`) | Zero training impact; fix before post-Run-5 ablation experiments |
| NF-12: predictor.py window truncation + random edge embeddings for old checkpoints | Inference-only; deferred to inference hardening post-Run-5 |
| torch.compile re-enablement (if Gate 3.1 fails) | Depends on Phase 3 forward-pass changes stabilizing |

---

## Gate Summary (Complete Reference with Rollback Actions)

| Gate | Phase | When | Pass Condition | Failure Action |
|------|-------|------|----------------|----------------|
| **0.1** | 0 | After A20 fix | `label_map` populated; labels match CSV | Stop — data is poisoned. Restore CSV, debug `label_map`. |
| **0.2** | 0 | Every training epoch | `nan_loss_count < 0.5% × steps` | Halt training. Investigate NaN source. Restart from last clean checkpoint. |
| **1.1** | 1 | Before extraction | `solc` available and versions correct | Do not extract — install solc-select. |
| **2.1** | 2 | After re-extraction | ICFG failure=0; CF fallback<1%; edge rates at baseline | Do not proceed to Run 5 — re-examine Phase 2 fixes, re-extract. |
| **2.2** | 2 | After A9 fix | `uses_block_globals` non-zero for ≥80% Timestamp+ contracts | Feature still broken — check Slither version and isinstance import. |
| **3.1** | 3 | After Phase 3 fixes | torch.compile 2-epoch smoke test passes | Disable torch.compile for Run 5; file Run 6 fix. |
| **4.6** | 4 | Before Run 5 launch | Logger files created; data integrity hash logged; archive verified | Do not launch Run 5 — logger not active; KILL alerts will be missed. |
| **NF-4** | 4 | Before Run 5 launch | `gnn_num_layers=8` in training log | Halt — fix CLI default, relaunch. |
| **5.1** | 5 | Run 5 epoch 1 | `aux_phase2_loss` logged and non-zero | Do not continue — debug loss propagation path. |
| **7.5** | 5/7 | After re-extraction | CEI label coverage ≥60% positives, ≤5% negatives | Do not enable `aux_cei_loss_weight` — debug CEI labeler. |
| **5.3** | 5 | Before Run 5 | VRAM < 7.5 GB with max_nodes=2048 (full training step) | Reduce `batch_size` or `max_nodes`. |
| **5.4** | 5 | Run 5 epoch 10 | Phase 2 JK weight > 0.33 OR Reentrancy F1 > 0.18 | Follow rollback decision tree (§5.6). Try aux_weight=0.20. If still failing, escalate to Run 6. |
| **6.1** | 6 | After temperature fitting | Main head ECE < 0.05 | Try per-eye calibration or Platt scaling. |
| **7.1** | 7 | After re-extraction | Graph count ≥41K; fail count=0 | Do not proceed — re-examine extraction logs. |
| **7.2** | 7 | After re-extraction | Same as Gate 2.1 | Do not proceed — extraction quality insufficient. |
| **7.3** | 7 | After re-extraction | Labels match CSV; no all-zero batches; label dist shift <10% per class | Do not proceed — verify A20 fix, re-extract. |
| **7.4** | 7 | After re-extraction | Schema version consistent; `NODE_FEATURE_DIM=11`; `max_nodes=2048` | Do not proceed — verify schema fixes. |

---

*Generated from `SENTINEL_Run5_UNIFIED_PREFLIGHT_PROPOSAL.md` (revised 2026-06-02). All gates are mandatory. Execution order is non-negotiable — each phase depends on the prior phase being complete and validated. All previous v8-era data must be archived before re-extraction, and Run 5 must train exclusively on verified v9 data.*
