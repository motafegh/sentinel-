# SENTINEL — Active Bug List

Last updated: 2026-05-18 (rev 8 — speed optimizations applied; 7-section data audit complete; no new bugs found)

**Pipeline status (2026-05-18 rev 8):**
- All 27 code bugs fixed (schema v7, NODE_FEATURE_DIM=11, gnn_layers=7)
- Re-extraction: 41,522 v7 graphs · 44,470 windowed tokens
- inject_augmented: +54 rows (+26 DoS vuln, 6 compile-fail skipped)
- **label_cleaner (rev 7):** −3,665 noisy labels (was −17,722; 4 heuristic bugs fixed — see below)
  → multilabel_index_cleaned.csv; audit trail in multilabel_index_cleaned.audit.json
- **create_cache (rev 7):** 41,577 pairs, 2.28 GB → cached_dataset_deduped.pkl (rebuilt with fixed labels)
- **Splits (rev 7):** regenerated from paired stems (41,576) not full CSV — fixes Index OOB crash
  → train=29,103 / val=6,236 / test=6,237; max_index=41,575 ✓
- **label_cleaner bugs fixed (rev 7):**
  - CRITICAL: IntegerUO removed from PRECONDITIONS — `has_loop` is unrelated to overflow; restored 9,897 labels
  - HIGH: Reentrancy check fixed — was testing CALLS edge (internal calls), not external calls; now uses x[10]>0; restored 601 labels
  - HIGH: CallToUnknown check fixed — Transfer/Send gap in extractor (not counted in call_target_typed); OR on x[10]>0; restored 1,815 labels
  - MEDIUM: MishandledException check improved — inherits Transfer/Send fix; OR on x[10]>0; restored 1,744 labels
  - KNOWN LIMITATION: Timestamp `now` alias (pre-Solidity 0.5) not captured by extractor — needs phase-2 re-extraction
- **inference fix (BUG-I1):** `three_eye_v7` added to predictor.py `_ARCH_TO_FUSION_DIM` — v7 checkpoints were unloadable
- **Full source audit (rev 6):** 12 config/default fixes in trainer.py + train.py + sentinel_model.py + fusion_layer.py
- All 36 fixes independently verified present in code (100% pass)
Sources: internal deep inspection + external reviewer cross-validation (docs/checking.md + docs/sentinel_remediation_plan.md, both verified against source)

Legend: **OPEN** = not fixed · **DEFERRED** = known, accepted for now · **FIXED** = resolved this session

---

## Severity Key

| Level | Meaning |
|-------|---------|
| CRITICAL | Actively corrupts training or produces wrong predictions |
| HIGH | Significantly degrades model performance; explains poor F1 |
| MEDIUM | Correctness risk or meaningful perf loss; fix before v7 |
| LOW | Minor inefficiency or cosmetic issue |

---

## CRITICAL

### BUG-C1 — Default `loss_fn="bce"` in TrainConfig
- **File:** `ml/src/training/trainer.py:246`
- **Code:** `loss_fn: str = "bce"`
- **Impact:** Any training run without `--loss-fn asl` silently falls back to BCEWithLogitsLoss, ignoring ASL entirely. The v6 plan specifies ASL as the primary loss. This is a silent regression — the flag must be passed every run.
- **Fix:** Change default to `"asl"`.
- **Source:** FM6.1.2 (external review, confirmed)
- **Status:** FIXED — default changed to `"asl"` in trainer.py:246

### BUG-C2 — No LayerNorm on token input to CrossAttentionFusion
- **File:** `ml/src/models/fusion_layer.py`
- **Code:** `tokens_proj = self.token_proj(token_embs)` — raw CodeBERT output applied directly
- **Impact:** CodeBERT hidden states have L2 norm ~10–15; GNN output after LayerNorm has norm ~1. Token key vectors dominate cross-attention dot products by 10-15×. Node→token attention attends to highest-norm tokens rather than semantically relevant ones. Fusion eye learns from amplitude, not semantics.
- **Fix:** Add `nn.LayerNorm(768)` applied to `token_embs` before `token_proj`.
- **Source:** FM5.3.1 (external review, confirmed)
- **Status:** FIXED — `self.token_norm = nn.LayerNorm(token_dim)` added to `__init__`; applied as `self.token_proj(self.token_norm(token_embs))` in `forward()`

### BUG-C3 — 72% of nodes carry near-zero features (CFG featureless problem)
- **File:** `ml/src/preprocessing/graph_extractor.py:511-524`
- **Code:** CFG node feature vector has dims 1-5, 7, 9-11 hardcoded to 0.0. Only type_id [0], loc [6], call_target_typed [8] carry signal.
- **Impact:** 48,175 of 66,288 nodes (72%) have 9 of 12 dimensions zero. GAT attention key/query dot products over these nodes are near-identical → attention weights collapse to uniform → Phase 2 message passing degrades to simple averaging. The directional CALL→WRITE signal (the entire motivation for Phase 2) is washed out. JK aggregation then learns to down-weight Phase 2, creating a cycle where CF signal gets ignored.
- **Fix:** Propagate parent FUNCTION node features (visibility, complexity, loc, return_ignored, has_loop, external_call_count) down to all CFG children during extraction. Requires schema change.
- **Source:** FM2.2.2 + FM5.1.1 + section 8.2 (external review, confirmed) + deep inspection
- **Status:** FIXED — `_build_cfg_node_features()` now accepts `parent_features`; inherits dims [1,3,4,5,9] (visibility/view/payable/complexity/has_loop) from parent FUNCTION; `_build_control_flow_edges()` passes `x_list[fn_idx]` as `parent_features`

### BUG-C4 — ASL γ⁻=4 causes all-zeros collapse with 60% zero-label rows
- **File:** `ml/src/training/trainer.py` (config) + `ml/src/training/losses.py`
- **Impact:** For a near-zero prediction p≈0.05, negative weight = (0.05)^4 = 0.000006 — effectively zero gradient. With 60.1% zero-label rows, the model finds the all-zeros basin immediately and has insufficient positive gradient to escape it. Observed in v6.0: Hamming rising (all-zeros improves Hamming) while F1 stagnates from epoch 9 onward. Compounded by BUG-H10 (no oversampling) — the model sees a 60/40 negative/positive split every batch with no rebalancing.
- **Fix:** Reduce γ⁻ from 4 to 2 for next run. Also add WeightedRandomSampler (BUG-H10) to change the batch-level ratio before the loss even fires. Gradient clipping (max_norm=1.0) and OneCycleLR scheduler are already implemented — no change needed there.
- **Source:** FM6.1.1 (external review) + observed training behavior
- **Status:** FIXED — `asl_gamma_neg` default reduced 4.0→2.0; `asl_clip` reduced 0.05→0.01 (also fixes BUG-M2)

---

## HIGH

### BUG-H1 — Phase 2 hop reach (2 hops) insufficient for full CEI pattern (4 hops)
- **File:** `ml/src/models/gnn_encoder.py`
- **Code:** Phase 2 = conv3 + conv3b = 2 CONTROL_FLOW hops
- **Impact:** Full reentrancy CEI pattern: ENTRY→CHECK→CALL→TMP→WRITE→RETURN = 5 nodes, 4 hops. Phase 2 reaches only 2 hops from any node. A WRITE 3 or 4 hops from a CALL never receives the CALL's signal. conv3b was added to extend reach but still only covers 2-hop patterns. The deeper CEI variants (CALL→TMP→ASSIGN→WRITE with 3 intermediate nodes) are invisible to Phase 2.
- **Fix:** Add conv3c (3rd CF hop) or restructure Phase 2 to 3 layers. Alternatively, use edge-conditioned message passing to carry the CALL signal further.
- **Source:** FM5.1.3 (external review, confirmed)
- **Status:** FIXED — `conv3c` added to `GNNEncoder.__init__()` and `forward()`; `gnn_layers` default updated 6→7 in TrainConfig

### BUG-H2 — Ghost graph fallback pools over STATE_VAR nodes
- **File:** `ml/src/models/sentinel_model.py:271-298`
- **Code:** `pool_mask = func_mask | fallback_mask` — fallback_mask selects ALL nodes for graphs with no function nodes
- **Impact:** 9% of graphs (≈4,002) have no FUNCTION/MODIFIER/FALLBACK/RECEIVE/CONSTRUCTOR nodes. These are interface-only contracts or Slither extraction failures. The fallback pools over STATE_VAR and CONTRACT nodes, producing a non-zero embedding dominated by variable-type features. The GNN eye "sees" these as meaningful contracts instead of degenerate inputs, injecting noise into the classifier for 9% of training samples.
- **Fix:** For graphs with no function nodes, return a zero vector directly instead of pooling non-function nodes.
- **Source:** FM5.4.2 (external review, confirmed) + deep inspection WARN-1
- **Status:** FIXED — `fallback_mask` removed; `pool_mask = func_mask` only; `global_max/mean_pool` naturally returns zero rows for graph indices with no contributing nodes

### BUG-H3 — pos_weight_min_samples disabled by default
- **File:** `ml/src/training/trainer.py:265`
- **Code:** `pos_weight_min_samples: int = 0`
- **Impact:** With 0, ALL classes get sqrt(neg/pos) pos_weight, including Reentrancy (4,498 samples) which gets 2.82× amplification. Reentrancy dominates gradient, suppressing learning for minority classes. The v6 plan recommended setting this to 3000 so that large classes don't get amplified.
- **Fix:** Change default to `3000`.
- **Source:** FM6.1.3 (external review, confirmed)
- **Status:** FIXED — default changed to `3000`

### BUG-H4 — Timestamp labels: 48.2% have no source evidence
- **Files:** `ml/data/processed/multilabel_index_deduped.csv`, BCCC source folders
- **Impact:** Of 961 Timestamp=1 contracts, ~463 have no block global access in source and `uses_block_globals=0` in their graph. Training on these teaches the model to associate "no timestamp signal" → "timestamp vulnerability" — the opposite of correct behavior. This systematically inverts the signal for nearly half the positive examples, making Timestamp F1 worse than not training on it at all.
- **Fix:** Filter out Timestamp=1 rows where `uses_block_globals=0.0` across all nodes. Estimated removal: ~463 rows.
- **Source:** Section 8.4 (external review) + deep inspection
- **Status:** OPEN

### BUG-H5 — ~14% of Reentrancy=1 contracts have no external calls
- **Files:** `ml/data/graphs/*.pt`, BCCC labeling
- **Impact:** Reentrancy requires a re-entrant external call by definition. Contracts with no CALL edges / external_call_count=0 cannot be genuinely reentrancy-vulnerable. These are BCCC OR-label noise (contract appeared in a Reentrancy folder but doesn't contain the pattern). Approximately 630 training samples are mislabeled, teaching the model wrong associations.
- **Fix:** Filter Reentrancy=1 rows where external_call_count=0 across all FUNCTION nodes. Requires verification pass.
- **Source:** Deep inspection WARN-2
- **Status:** OPEN

### BUG-H6 — DoS class is structurally unlearnable
- **Files:** `ml/data/processed/multilabel_index_deduped.csv`
- **Impact:** 346 total DoS samples; only 7 are pure DoS (no Reentrancy co-label) in training. 98.1% co-occur with Reentrancy. The model cannot learn to distinguish DoS from Reentrancy with 7 contradictory examples. Every gradient step on DoS is countermanded by 3,500 Reentrancy-only steps that teach the exact opposite. This wastes gradient and may suppress learning for other classes by adding noise.
- **Fix (options):** (a) **PREFERRED** — Drop DoS from loss computation: keep label in CSV but set `loss_mask[DoS_idx]=0` so gradients are never computed for that column; (b) Augment with 300+ pure-DoS contracts (Phase 4 plan); (c) Merge DoS into Reentrancy — **BLOCKED**: NUM_CLASSES=10 is LOCKED by ZKML proxy MLP shape (`Linear(128→64→32→10)`); changing to 9 classes breaks the ZK circuit.
- **Source:** Section 8.3 (external review) + prior audit + remediation plan (merge option ruled out)
- **Status:** FIXED — `dos_loss_weight: float = 0.0` added to TrainConfig; DoS logit column detached before loss computation in `train_one_epoch`; predictions still made normally at inference

### BUG-H7 — EMITS edges never created (old event syntax)
- **File:** `ml/src/preprocessing/graph_extractor.py`
- **Impact:** All 44,470 graphs have 0 EMITS edges. Event emissions — a key signal for GasException and ExternalBug classes — are missing from the graph entirely. Edge type slot 3 is allocated but always empty. BUG-7 from prior audit.
- **Fix:** Update event detection to use Slither's current IR syntax.
- **Source:** Prior audit BUG-7, confirmed in deep inspection
- **Status:** FIXED — `EventCall` IR scan added as fallback when `events_emitted` is empty; old-style Solidity 0.4.x events now detected via IR ops

### BUG-H8 — INHERITS edges never created
- **File:** `ml/src/preprocessing/graph_extractor.py`
- **Impact:** All 44,470 graphs have 0 INHERITS edges. Inheritance relationships — relevant for contract hierarchy and vulnerability propagation — are missing. Edge type slot 4 is allocated but always empty. BUG-8 from prior audit.
- **Fix:** Add inheritance edge extraction from `contract.inheritance`.
- **Source:** Prior audit BUG-8
- **Status:** FIXED — parent contracts added as CONTRACT nodes during extraction; INHERITS edges now fire via `_add_edge(contract_key, parent_key, INHERITS)`

### BUG-H9 — No structural label cleaning pipeline
- **Files:** `ml/data/processed/multilabel_index_deduped.csv`, `ml/data/graphs/*.pt`
- **Impact:** BCCC OR-labeling gives every contract in a folder all folder labels, regardless of whether the specific contract exhibits each vulnerability. The Brainmab case (clean ERC20 labeled across 4 classes) proves systematic noise. BUG-H4 and BUG-H5 are the two confirmed instances; there are estimated 2,000+ additional noisy labels across all 10 classes. Without a cleaning step the model trains on systematically wrong ground truth for a meaningful fraction of every batch.
- **Fix:** Build `ml/scripts/label_cleaner.py` — for each contract, check per-class structural preconditions against its graph features and zero out labels that fail. Preconditions confirmed valid: Timestamp (`uses_block_globals[dim2]>0`), Reentrancy (CALLS edge exists), MishandledException (`return_ignored[dim7]>0`), IntegerUO (`in_unchecked[dim9]>0` or `has_loop[dim10]>0`), CallToUnknown (`call_target_typed[dim8]==0`). Script must be idempotent and produce an audit JSON of every change.
- **Source:** Remediation plan section 2.1 (label_cleaner.py design validated against source — CLASS_NAMES order correct, dim indices correct, edge_attr shape handled)
- **Status:** FIXED — `ml/scripts/label_cleaner.py` built; covers Reentrancy/Timestamp/IntegerUO/MishandledException/CallToUnknown/UnusedReturn; `--dry-run` supported; run against fresh graphs after re-extraction

### BUG-H10 — No weighted sampler; 60% zero-label rows train at natural frequency
- **File:** `ml/src/training/trainer.py` (DataLoader construction)
- **Impact:** The DataLoader uses `RandomSampler` (uniform). Every batch has a ~60% chance of drawing a zero-label (all-clean) sample. The model sees a heavily skewed gradient signal even before the loss function fires. Combined with BUG-C4 (ASL γ⁻=4 suppressing negatives), the effective positive-to-negative gradient ratio per batch is extremely low. A WeightedRandomSampler that oversamples vulnerable contracts 3× would shift the in-batch ratio from 40/60 to ~60/40 without any label modification.
- **Fix:** Add `WeightedRandomSampler` to the training DataLoader. Weight = 3.0 for rows with any positive label, 1.0 for all-zero rows. Weights computed from the label CSV at dataset build time.
- **Source:** Remediation plan section 2.2.1 (strategy validated; sampler API confirmed compatible with existing DataLoader)
- **Status:** FIXED — `"positive"` mode added to `_build_weighted_sampler()`; `use_weighted_sampler` default changed from `"none"` to `"positive"`

---

## MEDIUM

### BUG-M1 — `id(lval)` identity comparison in `_compute_return_ignored()`
- **File:** `ml/src/preprocessing/graph_extractor.py:243,252`
- **Code:** `all_read_vars.add(id(rv))` / `if id(lval) not in all_read_vars`
- **Impact:** Relies on CPython object identity. If Slither ever interns or caches IR variable objects across functions (possible in future Slither versions), two distinct IR variables with the same identity will be conflated, producing false negatives for return_ignored. Currently works but is fragile to Slither internal changes.
- **Fix:** Replace `id(lval)` with `lval.name` (or a `(function_name, lval.name)` tuple for uniqueness).
- **Source:** FM2.2.3 (external review, confirmed)
- **Status:** FIXED — `all_read_names: set` now collects `rv.name` (stable string); `lval_name not in all_read_names` check replaces `id()` comparison

### BUG-M2 — Hard clip gradient boundary at p=0.05 in ASL
- **File:** `ml/src/training/losses.py`
- **Code:** `prob_neg = (prob - self.clip).clamp(min=0.0)` with `clip=0.05`
- **Impact:** Creates a sharp discontinuity: p<0.05 → zero negative gradient; p>0.05 → full negative gradient. Predictions oscillate at the boundary rather than converging smoothly. For minority classes where the model hovers around p≈0.03–0.06, this creates unstable training dynamics.
- **Fix:** Use a soft clip: `prob_neg = torch.sigmoid((prob - self.clip) * 10)` or reduce clip to 0.01.
- **Source:** FM6.2.1 (external review, confirmed)
- **Status:** FIXED — `asl_clip` default reduced 0.05→0.01 in TrainConfig (fixed alongside BUG-C4)

### BUG-M3 — No per-class gamma/clip in AsymmetricLoss
- **File:** `ml/src/training/losses.py`
- **Code:** `gamma_neg`, `gamma_pos`, `clip` are scalar floats shared across all 10 classes
- **Impact:** DoS (3 pure training samples) needs much more aggressive positive mining than IntegerUO (13,797 samples). A single γ⁻=4 is simultaneously too aggressive for large classes (causing all-zeros) and not aggressive enough for starved classes. Class-adaptive parameters would significantly improve minority class learning.
- **Fix:** Change parameters to accept `torch.Tensor` of shape [num_classes] alongside scalar fallback.
- **Source:** FM6.2.2 (external review, confirmed)
- **Status:** FIXED — `gamma_neg`, `gamma_pos`, `clip` accept `Union[float, Tensor]`; registered as buffers via `register_buffer()` for automatic device movement

### BUG-M4 — Aux loss warmup too short (3 epochs)
- **File:** `ml/src/training/trainer.py:225`
- **Code:** `aux_loss_warmup_epochs: int = 3`
- **Impact:** After 3 epochs, 3 auxiliary classification heads (GNN eye, TF eye, Fused eye) contribute full λ=0.3 gradient each. Combined aux gradient can exceed main loss gradient before the main classifier has learned anything useful. The 3-epoch window is insufficient for a 2.4M-parameter GNN with 100-epoch schedule.
- **Fix:** Increase to 8–10 epochs, or use a gradual ramp (linear warmup from 0 to 0.3 over 10 epochs).
- **Source:** FM6.1.6 (external review, confirmed)
- **Status:** FIXED — default increased 3→8 in trainer.py

### BUG-M5 — Brainmab contract confirmed mislabeled across 4 classes
- **File:** `ml/data/processed/multilabel_index_deduped.csv`
- **Impact:** One confirmed contract (standard ERC20, no vulnerabilities) is labeled Reentrancy=1, CallToUnknown=1, IntegerUO=1, MishandledException=1 in BCCC. This is the worst case of BCCC OR-label noise — a clean contract injecting wrong signal into 4 class heads simultaneously. Likely not unique; represents a broader label noise pattern in BCCC.
- **Fix:** Identify and remove this contract from the dataset. Consider a broader audit for contracts with ≥3 co-occurring labels and zero supporting features.
- **Source:** Deep inspection FIND-1
- **Status:** OPEN

### BUG-M6 — Token files carry stale `feature_schema_version='v4'`
- **File:** `ml/data/tokens_windowed/*.pt` metadata field
- **Impact:** Cosmetic only — schema version in token `.pt` files is not checked at load time. Graphs correctly report v6. But the mismatch could cause confusion during future validation scripts or if a version-check is added to the dataset loader.
- **Fix:** Re-run `retokenize_windowed.py` with updated metadata, or add a one-pass metadata patch script.
- **Source:** Deep inspection WARN-4
- **Status:** OPEN — low urgency

### BUG-M7 — 8.5% of graphs have empty `contract_path`
- **File:** `ml/data/graphs/*.pt` metadata
- **Impact:** Not used during training (labels come from CSV). But these graphs cannot be manually inspected, debugged, or cross-referenced against source without the path. Any future analysis script that uses `contract_path` will silently skip 8.5% of the dataset.
- **Fix:** Build `ml/scripts/backfill_contract_paths.py` — scan BCCC-SCsVul-2024/ to build an MD5→path map, then patch the affected graph `.pt` metadata in-place. Output a sidecar JSON for quick lookups without touching the full graph tensor.
- **Source:** Deep inspection WARN-3 + remediation plan section 5.3
- **Status:** OPEN — low urgency

### BUG-M8 — eval_threshold fixed at 0.35 during training; no per-epoch tuning
- **File:** `ml/src/training/trainer.py` (`evaluate()` function, `TrainConfig.eval_threshold`)
- **Impact:** Early stopping uses a single global threshold (0.35) applied to all 10 classes to compute validation F1. Optimal thresholds vary widely by class (IntegerUO ~0.45, Timestamp ~0.25, DoS ~0.15). A fixed threshold under-counts F1 for low-prevalence classes and over-counts for dominant ones, making the early-stopping metric noisy and potentially stopping training too early or too late for specific classes.
- **Fix:** After each eval epoch, sweep thresholds [0.1–0.9] per class on the validation predictions and use the resulting optimal thresholds to compute the early-stopping F1. This matches what `tune_thresholds.py` does post-training but wired into the training loop. Adds ~2s per eval epoch.
- **Source:** Remediation plan section 3.3 (confirmed: `tune_thresholds.py` exists but is post-training only, not wired into the training loop)
- **Status:** FIXED — `tune_thresholds=True` added to `evaluate()` call in training loop; sweeps 19 candidates per class; logs `val_f1_macro_tuned` to MLflow; `tuned_thresholds` list returned in metrics dict

### BUG-M9 — Uniform label smoothing applied to all classes equally
- **File:** `ml/src/training/trainer.py`
- **Code:** `label_smoothing = 0.05` applied uniformly across all 10 classes
- **Impact:** Uniform ε=0.05 tells the model "there is a 5% chance this clean contract has each vulnerability" regardless of actual noise rates. For Reentrancy (confirmed 14% noise) ε=0.05 is too low — the model learns to be over-confident about clean labels that are actually wrong. For IntegerUO (lower confirmed noise) ε=0.05 may be too high, smoothing away valid signal. Class-conditional smoothing calibrated to actual noise estimates would improve robustness for high-noise classes without harming clean-label classes.
- **Fix:** Replace uniform `label_smoothing` with a per-class tensor. Use confirmed noise estimates: Reentrancy=0.14, Timestamp=0.05 (structural check exists), DoS=0.18, others=0.10 (estimated). Applied as `labels[:, c] = labels[:, c] * (1 - eps[c]) + 0.5 * eps[c]` per class.
- **Source:** Remediation plan section 2.2.2 (design validated; smoothing formula correct)
- **Status:** FIXED — `class_label_smoothing` dict added to TrainConfig; `_class_eps` tensor built before training loop; applied in `train_one_epoch` via `class_eps` param; uniform `label_smoothing` set to 0.0

### BUG-M10 — No training guardrails; all-zeros collapse and GNN collapse go undetected
- **File:** `ml/src/training/trainer.py`
- **Impact:** v6.0 training collapsed to all-zeros by epoch 9 and ran until epoch 16 before being killed manually. There is no automated detection of: (1) all-zeros collapse (Hamming >0.85 for 3+ epochs), (2) class death (any class F1=0.0 for 5+ epochs with no recovery), (3) GNN eye collapse (gnn_share <10% for 5+ consecutive log intervals). Without guardrails the next training run can fail in the same way and only be caught by manual log inspection.
- **Fix:** Add three alert conditions to the training loop that log CRITICAL warnings (and optionally auto-adjust) when thresholds are crossed. Minimum viable: log-level alerts that print to stdout and the structured log so they are visible without tailing the full tqdm output.
- **Source:** Remediation plan section 8 (guardrail table validated; detection conditions are computable from metrics already being logged)
- **Status:** FIXED — three guardrail checks added after `evaluate()` each epoch: all-zeros collapse (Hamming>0.85 ×3), class death (F1=0.0 ×5), GNN collapse (gnn_share<0.10 ×5); all log CRITICAL/WARNING

---

## LOW

### BUG-L1 — `torch.isin()` O(N×K) for function-node pooling mask
- **File:** `ml/src/models/sentinel_model.py:269`
- **Code:** `func_mask = torch.isin(node_type_ids, _func_ids_tensor)` — K=5 type IDs
- **Impact:** Minor performance cost. For large graphs (N>1000), a range check `(ids >= 1) & (ids <= 6)` would be O(N) and faster, since FUNCTION=1 through CONSTRUCTOR=6 are contiguous.
- **Fix:** Replace with `func_mask = (node_type_ids >= 1) & (node_type_ids <= 6)`. Validate that no non-function types fall in range 1–6 first.
- **Source:** FM5.4.1 (external review, confirmed)
- **Status:** FIXED — `_FUNC_IDS_CPU` tensor moved to module level; `torch.tensor(list(...))` allocation eliminated from forward(); `torch.isin` still used but the per-call tensor allocation is gone

### BUG-L2 — `in_unchecked` feature (dim[9]) is dead
- **File:** `ml/src/preprocessing/graph_extractor.py`, `ml/src/preprocessing/graph_schema.py`
- **Impact:** Solidity 0.8.x `unchecked{}` blocks — the only source of in_unchecked=1 — account for ~0.1% of BCCC dataset (0.8.x contracts). 99.9% of training samples have in_unchecked=0. The feature dimension is permanently allocated but provides no discriminative signal.
- **Fix:** Drop dim[9] in next schema bump and re-extract — NODE_FEATURE_DIM would drop from 12 to 11. Not worth doing until v7.
- **Source:** Prior audit, confirmed in deep inspection
- **Status:** FIXED — dropped from `_build_node_features()` and `_build_cfg_node_features()`; `NODE_FEATURE_DIM` 12→11; `FEATURE_SCHEMA_VERSION` v6→v7; `has_loop` now dim[9], `external_call_count` now dim[10]; `label_cleaner.py` updated accordingly

### BUG-L3 — Hash-based graph-token pairing fragile to directory restructuring
- **File:** `ml/src/datasets/dual_path_dataset.py`, graph/token file naming
- **Impact:** Graph and token file names are MD5 hashes of source file paths. Any reorganization of BCCC-SCsVul-2024/ invalidates all hashes, requiring full re-extraction (hours). Content-based hashing would be more robust but risks collisions for duplicate contracts.
- **Fix:** Store both path-hash and content-hash; use content-hash for dedup, path-hash for current pairing.
- **Source:** Section 8.5 (external review)
- **Status:** DEFERRED — operational risk, low immediate impact

### BUG-L4 — No write-time feature validation in extraction pipeline
- **File:** `ml/src/preprocessing/graph_extractor.py`
- **Impact:** BUG-1/2/3 (fixed in-place) were only caught by a post-hoc validation scan. The extraction pipeline has no assertions or range checks that fire at write time. Future schema changes to the extractor can silently produce corrupt `.pt` files that must again be patched post-hoc.
- **Fix:** Add a `_validate_features(x: torch.Tensor)` call at the end of `_build_node_features()` that asserts all dims in [0,1] for normalized features.
- **Source:** Section 8.1 (external review) — "Patch-and-Pray Anti-Pattern"
- **Status:** FIXED — OOR check added after `x` tensor is assembled in `extract_contract_graph()`; logs WARNING (not raise) with node/dim details so a single bad contract doesn't abort a full extraction run

---

## Summary Table

| ID | Description | Severity | Status | File |
|----|-------------|----------|--------|------|
| BUG-C1 | Default loss_fn="bce" not "asl" | CRITICAL | **FIXED** | trainer.py:246 |
| BUG-C2 | No LayerNorm on token input to fusion | CRITICAL | **FIXED** | fusion_layer.py |
| BUG-C3 | 72% CFG nodes near-featureless (9/12 dims=0) | CRITICAL | **FIXED** | graph_extractor.py |
| BUG-C4 | ASL γ⁻=4 causes all-zeros collapse (60% zero rows) | CRITICAL | **FIXED** | trainer.py |
| BUG-H1 | Phase 2 2-hop limit misses 4-hop CEI pattern | HIGH | **FIXED** | gnn_encoder.py |
| BUG-H2 | Ghost graph fallback pools STATE_VAR nodes | HIGH | **FIXED** | sentinel_model.py |
| BUG-H3 | pos_weight_min_samples=0 disabled by default | HIGH | **FIXED** | trainer.py |
| BUG-H4 | Timestamp: 48.2% labels have no source evidence | HIGH | OPEN (run H9 script) | CSV + BCCC |
| BUG-H5 | 14% Reentrancy=1 have no external calls | HIGH | OPEN (run H9 script) | graphs + BCCC |
| BUG-H6 | DoS class unlearnable; merge to 9 classes BLOCKED by ZKML | HIGH | **FIXED** | trainer.py |
| BUG-H7 | EMITS edges never created | HIGH | **FIXED** | graph_extractor.py |
| BUG-H8 | INHERITS edges never created | HIGH | **FIXED** | graph_extractor.py |
| BUG-H9 | No structural label cleaning pipeline | HIGH | **FIXED** | ml/scripts/label_cleaner.py |
| BUG-H10 | No WeightedRandomSampler; 60% zero-label batches | HIGH | **FIXED** | trainer.py |
| BUG-M1 | id(lval) fragility in return_ignored | MEDIUM | **FIXED** | graph_extractor.py |
| BUG-M2 | Hard clip boundary at p=0.05 in ASL | MEDIUM | **FIXED** | trainer.py (clip→0.01) |
| BUG-M3 | No per-class gamma/clip in ASL | MEDIUM | **FIXED** | losses.py |
| BUG-M4 | Aux loss warmup 3 epochs too short | MEDIUM | **FIXED** | trainer.py |
| BUG-M5 | Brainmab contract mislabeled across 4 classes | MEDIUM | OPEN | CSV |
| BUG-M6 | Token files carry stale schema_version='v4' | MEDIUM | OPEN (resolved by re-extraction) | tokens_windowed/ |
| BUG-M7 | 8.5% graphs have empty contract_path | MEDIUM | OPEN (resolved by re-extraction) | graphs/ |
| BUG-M8 | Fixed eval_threshold=0.35; no per-epoch tuning | MEDIUM | **FIXED** | trainer.py |
| BUG-M9 | Uniform label smoothing; class-conditional not implemented | MEDIUM | **FIXED** | trainer.py |
| BUG-M10 | No training guardrails (all-zeros, class-death, GNN collapse) | MEDIUM | **FIXED** | trainer.py |
| BUG-L1 | torch.isin() O(N×K) pooling mask | LOW | **FIXED** | sentinel_model.py |
| BUG-L2 | in_unchecked feature dead (0.1% coverage) | LOW | **FIXED** | graph_schema.py + extractor |
| BUG-L3 | Path-hash pairing fragile to dir restructure | LOW | OPEN (Phase 4) | dual_path_dataset.py |
| BUG-L4 | No write-time feature validation in extractor | LOW | **FIXED** | graph_extractor.py |

---

## Previously Fixed (this session — 2026-05-17)

| ID | Description | Fix | Commit |
|----|-------------|-----|--------|
| BUG-1 | loc stored as raw line counts (max=2,167) | log1p normalization in-place patch | 8c8ce8c |
| BUG-2 | complexity stored as raw CFG block count (max=48) | log1p normalization in-place patch | 8c8ce8c |
| BUG-3 | visibility=2 for private (out of [0,1]) | VISIBILITY_MAP int→float {0.0,0.5,1.0} | 8c8ce8c |

---

## Fix Order

All bugs are tracked for resolution — none skipped. Fix order follows severity: CRITICAL → HIGH → MEDIUM → LOW. Within each tier, fix in summary table order. Deferred bugs (v7 scope) are addressed in the next schema/extraction cycle, not the current training run.
