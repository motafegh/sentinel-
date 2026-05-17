# SENTINEL — Bug Fix Implementation Plan

Last updated: 2026-05-17  
Source: docs/ACTIVE_BUGS.md (27 open bugs)  
Scope: All bugs fixed in full — none skipped, none deferred except v7-scope items already marked as such in the bug list.

---

## Fresh Extraction Strategy — IMPORTANT

**After all code fixes are complete, the entire data pipeline runs fresh from source:**

1. All existing `ml/data/graphs/*.pt` → moved to `ml/data/archive/graphs_v6/`
2. All existing `ml/data/tokens_windowed/*.pt` → moved to `ml/data/archive/tokens_v6/`
3. `ml/data/cached_dataset_windowed.pkl` → moved to `ml/data/archive/`
4. Full re-extraction: `python ml/scripts/reextract_graphs.py` against all 44,470 contracts
5. Full re-tokenization: `python ml/scripts/retokenize_windowed.py`
6. Cache rebuild: `python ml/scripts/create_cache.py`

**Consequences for the fix plan:**

- **BUG-M6** (token schema version) and **BUG-M7** (empty contract_path) are resolved automatically by re-extraction — the `fix_token_schema_version.py` and `backfill_contract_paths.py` scripts are no longer needed.
- **BUG-C3** (CFG featureless nodes), **BUG-H7** (EMITS edges), **BUG-H8** (INHERITS edges), **BUG-L2** (dead in_unchecked), **BUG-L3** (hash pairing) — all previously marked v7-deferred — are now fixed **in the extractor before re-extraction** since we pay the re-extraction cost anyway. There is no v7 deferral for extractor-level bugs.
- **BUG-M1** (`id(lval)`) is fixed in the extractor before re-extraction.
- **BUG-L4** (write-time validation) is added to the extractor before re-extraction.
- Phase 5 of this plan is therefore collapsed into Phase 4 — all extractor fixes happen together.

This means the extractor must be fully corrected before the archive-and-re-extract step runs.

---

## Phases Overview

| Phase | Bugs | Gate |
|-------|------|------|
| **Phase 1 — Data / CSV** | H9, H6, M5 | Cleaned CSV ready; M6/M7 resolved by re-extraction |
| **Phase 2 — Training code** | C1, C2, C4, H2, H3, H10, M4, M8, M9, M10, L1 | All model + trainer fixes in; smoke run passes |
| **Phase 3 — Loss / ASL** | M2, M3 | ASL changes in and tested |
| **Phase 4 — Extractor (all)** | C3, H1, H7, H8, M1, L2, L4 | Extractor fully corrected; then archive old data + full re-extraction |
| **Archive + Re-extract** | — | `mv` old graphs/tokens/cache to archive; run reextract → retokenize → create_cache |

Training restarts only after all four phases complete and the fresh cache is verified.

---

## Phase 1 — Data Fixes

### BUG-H9 — Build `ml/scripts/label_cleaner.py`

**What:** Structural precondition filter that zeroes out BCCC OR-label noise. Reads the multilabel CSV, loads each graph .pt, applies per-class checks, writes cleaned CSV + audit JSON.

**File to create:** `ml/scripts/label_cleaner.py`

**Per-class precondition logic (all dims verified against graph_schema.py):**

| Class | CSV column | Precondition | Graph check |
|-------|-----------|--------------|-------------|
| Reentrancy | col idx 6 | At least 1 CALLS edge | `(edge_attr == 0).any()` where CALLS=0 |
| Timestamp | col idx 7 | Any node uses block globals | `(x[:, 2] > 0.5).any()` |
| MishandledException | col idx 5 | Any node has return_ignored | `(x[:, 7] > 0.5).any()` |
| IntegerUO | col idx 4 | Has unchecked block or loop | `(x[:, 9] > 0.5).any() or (x[:, 10] > 0.5).any()` |
| CallToUnknown | col idx 0 | Has untyped external call | `(x[:, 8] == 0.0).any()` |
| UnusedReturn | col idx 9 | Any node has return_ignored | `(x[:, 7] > 0.5).any()` |

**Classes without reliable structural preconditions (leave as-is):** GasException, ExternalBug, TOD, DoS — preconditions would require deeper graph traversal or are too broad to filter safely.

**Script contract:**
- Input: `--csv ml/data/processed/multilabel_index_deduped.csv`, `--graphs-dir ml/data/graphs`, `--output ml/data/processed/multilabel_index_cleaned.csv`
- Output: cleaned CSV + `multilabel_index_cleaned.audit.json` with every change logged as `{hash, class, old, new, reason}`
- Idempotent: running twice produces the same result
- edge_attr shape: handle both [E] and [E,1] (squeeze if dim > 1) — existing graphs are [E] but script must be robust

**Verify:** `python ml/scripts/label_cleaner.py --dry-run` — prints counts of labels to be removed per class without writing. Check that Timestamp removals ≈ 460, Reentrancy removals ≈ 630.

**Docs:** Add entry to `docs/changes/INDEX.md`. Update `docs/STATUS.md` label counts table after running.

---

### BUG-H6 — DoS loss mask in trainer

**What:** DoS class (col idx 1 in CLASS_NAMES) has 3 pure training samples. Add a `loss_mask` tensor that zeros the DoS column gradient without removing it from the CSV or changing NUM_CLASSES.

**File:** `ml/src/training/trainer.py`

**Change:** In `TrainConfig`, add:
```python
dos_loss_weight: float = 0.0   # 0.0 = DoS column contributes no gradient
```
In the training loop where loss is computed, apply a per-class weight vector:
```python
# Build once before training loop:
class_loss_weights = torch.ones(NUM_CLASSES, device=device)
class_loss_weights[CLASS_NAMES.index("DenialOfService")] = config.dos_loss_weight
# In loss computation, multiply element-wise across the class dimension
```
This is compatible with both ASL and BCE — multiply the per-class loss tensor before reduction.

**Verify:** In a smoke run, log per-class gradient norms after first backward pass. DoS column should have grad norm = 0.

**Docs:** Note in `docs/STATUS.md` under training config decisions.

---

### BUG-M5 — Remove Brainmab mislabeled contract

**What:** One confirmed clean ERC20 contract is labeled Reentrancy=1, CallToUnknown=1, IntegerUO=1, MishandledException=1. Remove its row from the CSV.

**Steps:**
1. Identify the MD5 hash: scan `ml/data/graphs/` for contracts with all four labels = 1 and external_call_count=0 across all nodes and no CALLS edges. The Brainmab contract will have zero structural signals for all four classes simultaneously.
2. Remove the matching row(s) from `ml/data/processed/multilabel_index_deduped.csv` (and from the cleaned CSV produced by H9).
3. Re-run `ml/scripts/create_splits.py` to regenerate train/val/test splits after row removal.

**Verify:** Confirm row count drops by the number of removed contracts. Confirm no remaining rows have all four classes = 1 with zero call signals.

**Docs:** Document removed hash(es) in `docs/changes/2026-05-17-label-cleaning.md`.

---

### BUG-M6 — Build `ml/scripts/fix_token_schema_version.py`

**What:** Token `.pt` files carry `feature_schema_version='v4'`; graphs are v6. One-pass metadata patch.

**File to create:** `ml/scripts/fix_token_schema_version.py`

**Logic:** For each `.pt` in `ml/data/tokens_windowed/`, load with `weights_only=False`, check `data.get("feature_schema_version")`, if not `"v6"` set it and re-save. Print count of updated files.

**Verify:** After run, spot-check 5 random token files: `torch.load(f, weights_only=False)["feature_schema_version"] == "v6"`.

**Docs:** Update BUG-M6 status in `docs/ACTIVE_BUGS.md`.

---

### BUG-M7 — Build `ml/scripts/backfill_contract_paths.py`

**What:** 8.5% of graph `.pt` files have empty `contract_path` metadata. Recover paths by scanning BCCC source directory.

**File to create:** `ml/scripts/backfill_contract_paths.py`

**Logic:**
1. Scan `BCCC-SCsVul-2024/SourceCodes/**/*.sol`, compute `md5(file_path_string)` for each (matching the extraction hash scheme)
2. Build `{md5: path}` map
3. For each graph `.pt` where `metadata["contract_path"] == ""`, look up hash in map, patch metadata in-place
4. Write `ml/data/contract_path_map.json` as a persistent sidecar for future lookups

**Verify:** Before/after count of empty `contract_path` fields. Target: 0 remaining after run (or document which hashes are genuinely unresolvable).

**Docs:** Update BUG-M7 status in `docs/ACTIVE_BUGS.md`.

---

## Phase 2 — Training Code Fixes

### BUG-C1 — Default loss_fn

**File:** `ml/src/training/trainer.py`  
**Line:** `loss_fn: str = "bce"` → `loss_fn: str = "asl"`  
**Verify:** `grep "loss_fn" ml/src/training/trainer.py` — default must be `"asl"`.

---

### BUG-C2 — LayerNorm before token projection in CrossAttentionFusion

**File:** `ml/src/models/fusion_layer.py`

**Change:** In `CrossAttentionFusion.__init__()`, add:
```python
self.token_norm = nn.LayerNorm(token_dim)   # token_dim = 768
```
In `forward()`, before `tokens_proj = self.token_proj(token_embs)`, add:
```python
token_embs = self.token_norm(token_embs)
```

**Why:** CodeBERT hidden states have L2 norm ~10-15; GNN output after its own LayerNorm has norm ~1. Without normalization, token keys dominate cross-attention by 10-15×.

**Verify:** In a forward pass, assert `token_embs.norm(dim=-1).mean()` ≈ `node_embs.norm(dim=-1).mean()` after the norm is applied (both should be ~1).

---

### BUG-C4 — Reduce ASL γ⁻ from 4 to 2

**File:** `ml/src/training/trainer.py`  
**Line:** `asl_gamma_neg: float = 4.0` → `asl_gamma_neg: float = 2.0`  
**Verify:** Smoke run — check that loss value is higher than before (less suppression of negatives means more gradient signal).

---

### BUG-H2 — Ghost graph zero-vector fallback

**File:** `ml/src/models/sentinel_model.py`

**Change:** In the function-level pooling block (lines ~271-298), replace the `fallback_mask` logic:

Current behavior: graphs with no function nodes pool over ALL nodes.  
New behavior: graphs with no function nodes get a zero vector — the GNN eye contributes nothing for these contracts.

```python
# After computing func_mask and graph_has_func:
# Instead of: pool_mask = func_mask | fallback_mask
# Do:
pool_mask = func_mask
if not func_mask.any():
    # All graphs lack function nodes — return zero pooled output
    # global_max/mean_pool will handle this via scatter with no elements
    pass
# For per-graph fallback: use scatter_add with func_mask only;
# graphs with no function nodes will have a zero row in the output naturally
# because scatter never writes to those graph indices.
```

The key insight: `global_max_pool(node_embs[func_mask], batch[func_mask], size=num_graphs)` already returns a zero row for any graph index with no contributing nodes — that is the correct zero-vector behavior. The existing fallback_mask logic was added to prevent zero rows, but zero rows are exactly what we want for interface-only contracts.

**Verify:** Load a known ghost graph (no FUNCTION nodes), run forward pass, assert GNN eye output row is all zeros for that graph index.

---

### BUG-H3 — pos_weight_min_samples default

**File:** `ml/src/training/trainer.py`  
**Line:** `pos_weight_min_samples: int = 0` → `pos_weight_min_samples: int = 3000`  
**Effect:** Classes with ≥3000 positive samples (IntegerUO=13,797, GasException=4,957, Reentrancy=4,498, MishandledException=4,186) get pos_weight clamped to 1.0 (no amplification). Minority classes below 3000 retain sqrt amplification.  
**Verify:** After setting, print computed pos_weights in a dry run — Reentrancy should be 1.0, Timestamp should be sqrt(43,509/961) ≈ 6.7.

---

### BUG-H10 — WeightedRandomSampler for zero-label imbalance

**File:** `ml/src/training/trainer.py`

**Change:** In the DataLoader construction for training, add a weighted sampler:

```python
# After building train_dataset, before DataLoader:
train_labels = label_df.iloc[train_indices][CLASS_NAMES].values  # [N, 10]
has_any_vuln = train_labels.sum(axis=1) > 0                      # [N] bool
sample_weights = np.where(has_any_vuln, 3.0, 1.0).astype(np.float64)
sampler = WeightedRandomSampler(
    weights=torch.from_numpy(sample_weights),
    num_samples=len(train_indices),
    replacement=True,
)
# Pass sampler= to DataLoader; remove shuffle=True (incompatible with sampler)
```

**Effect:** Shifts effective positive/negative ratio per batch from ~40/60 to ~60/40 without modifying any label.

**Verify:** After one epoch, log fraction of positive-label samples per batch — should be ~0.55-0.65, up from ~0.40.

**Docs:** Add `use_weighted_sampler: bool = True` to `TrainConfig` so it can be disabled for debugging.

---

### BUG-M4 — Aux loss warmup

**File:** `ml/src/training/trainer.py`  
**Line:** `aux_loss_warmup_epochs: int = 3` → `aux_loss_warmup_epochs: int = 8`  
**Verify:** In training log, aux loss weight should reach full λ=0.3 only at epoch 9.

---

### BUG-M8 — Per-epoch threshold tuning in evaluate()

**File:** `ml/src/training/trainer.py`

**Change:** In `evaluate()`, after collecting all `all_probs` and `all_labels`, add:

```python
# Per-class threshold sweep for F1 computation
best_thresholds = []
for c in range(NUM_CLASSES):
    best_f1, best_t = 0.0, 0.5
    for t in np.arange(0.1, 0.91, 0.05):
        preds_c = (all_probs[:, c] >= t).float()
        f1 = f1_score(all_labels[:, c].cpu(), preds_c.cpu(), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    best_thresholds.append(best_t)
# Use best_thresholds for final F1 computation instead of fixed eval_threshold
```

Log the per-class thresholds to MLflow each epoch. Use the macro-F1 at optimal thresholds as the early-stopping metric.

**Note:** This adds ~2s per eval epoch. Only run the sweep every eval; keep `eval_threshold` as fallback for non-sweep passes.

---

### BUG-M9 — Class-conditional label smoothing

**File:** `ml/src/training/trainer.py`

**Change:** Replace uniform `label_smoothing` with a per-class tensor applied before loss computation:

```python
# In TrainConfig, add:
class_label_smoothing: dict = field(default_factory=lambda: {
    "Reentrancy":          0.14,
    "DenialOfService":     0.18,
    "CallToUnknown":       0.10,
    "ExternalBug":         0.10,
    "GasException":        0.12,
    "MishandledException": 0.12,
    "IntegerUO":           0.08,
    "Timestamp":           0.05,
    "TransactionOrderDependence": 0.10,
    "UnusedReturn":        0.10,
})

# In training loop, build smoothing tensor once:
eps = torch.tensor(
    [config.class_label_smoothing[c] for c in CLASS_NAMES],
    device=device
)  # [10]

# Apply per-class smoothing to labels before loss:
labels_smooth = labels * (1.0 - eps) + 0.5 * eps
# Pass labels_smooth instead of labels to loss function
```

**Keep:** Remove the existing uniform `label_smoothing` field or set it to 0.0 to avoid double-smoothing.

---

### BUG-M10 — Training guardrails

**File:** `ml/src/training/trainer.py`

**Change:** Add three alert conditions checked at the end of each eval epoch:

```python
# 1. All-zeros collapse detector
hamming = (preds == labels).float().mean().item()
if hamming > 0.85:
    consecutive_allzeros += 1
    if consecutive_allzeros >= 3:
        logger.critical(f"ALL-ZEROS COLLAPSE: Hamming={hamming:.4f} for {consecutive_allzeros} epochs. "
                        f"Consider reducing gamma_neg or increasing pos_weight.")
else:
    consecutive_allzeros = 0

# 2. Class death detector
for c, name in enumerate(CLASS_NAMES):
    if per_class_f1[c] == 0.0:
        class_death_counter[c] += 1
        if class_death_counter[c] >= 5:
            logger.warning(f"CLASS DEATH: {name} F1=0.0 for {class_death_counter[c]} epochs.")
    else:
        class_death_counter[c] = 0

# 3. GNN collapse detector (uses existing gnn_share logging)
if gnn_share < 0.10:
    consecutive_gnn_collapse += 1
    if consecutive_gnn_collapse >= 5:
        logger.critical(f"GNN COLLAPSE: gnn_share={gnn_share:.3f} for {consecutive_gnn_collapse} evals.")
else:
    consecutive_gnn_collapse = 0
```

All counters persist across epochs in the trainer state. Alerts are log-level only (no auto-adjustment) — human decides the response.

---

### BUG-L1 — torch.isin() → range check

**File:** `ml/src/models/sentinel_model.py`  
**Line:** `func_mask = torch.isin(node_type_ids, _func_ids_tensor)`

**Change:**
```python
# FUNCTION=1, MODIFIER=2, FALLBACK=4, RECEIVE=5, CONSTRUCTOR=6
# EVENT=3 is NOT a function type — gap at 3
# So cannot use simple range check 1-6; keep isin but pre-compute on CPU once.
# Move _func_ids_tensor to module-level constant (not recreated per forward pass):
_FUNC_IDS_CPU = torch.tensor(sorted(_FUNC_TYPE_IDS), dtype=torch.long)

# In forward():
_func_ids = _FUNC_IDS_CPU.to(node_type_ids.device)
func_mask = torch.isin(node_type_ids, _func_ids)
```

**Note:** EVENT=3 falls in the 1-6 range but is NOT a function type, so a simple `>=1 & <=6` range check would be wrong. The isin approach is correct; this fix eliminates the per-forward-pass `torch.tensor()` allocation.

**Verify:** Assert `func_mask` is identical before and after change for 10 sampled graphs.

---

### BUG-L4 — Write-time feature validation in graph_extractor.py

**File:** `ml/src/preprocessing/graph_extractor.py`

**Change:** Add a validation function and call it before saving each graph:

```python
def _validate_node_features(x: torch.Tensor, contract_path: str) -> None:
    """Assert all normalized feature dimensions are in [0, 1]."""
    # Dims that must be in [0,1]: all except type_id (dim0, already /MAX_TYPE_ID)
    normalized_dims = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    for d in normalized_dims:
        col = x[:, d]
        if col.min() < -0.001 or col.max() > 1.001:
            raise ValueError(
                f"Feature dim[{d}] out of [0,1] in {contract_path}: "
                f"min={col.min():.4f} max={col.max():.4f}"
            )
```

Call this immediately before `torch.save(graph, output_path)` in the extraction loop.

**Verify:** Temporarily inject a raw value (e.g., loc=500) and confirm the extractor raises ValueError rather than saving silently.

---

## Phase 3 — Loss Function Fixes

### BUG-M2 — Soften ASL clip boundary

**File:** `ml/src/training/losses.py`

**Change:** In `AsymmetricLoss.__init__()`, change default `clip` from `0.05` to `0.01`. This reduces the dead-gradient zone from p<0.05 to p<0.01, allowing more gradient flow for predictions near zero without eliminating the clip entirely.

**Alternative (softer):** Replace hard clamp with a soft transition:
```python
# Current:
prob_neg = (prob - self.clip).clamp(min=0.0)
# Soft alternative:
prob_neg = torch.sigmoid((prob - self.clip) * 20.0) * prob
```
Start with the simpler default change to 0.01; use soft version if oscillation persists.

**Verify:** Plot gradient magnitude vs. prediction probability for a range [0, 0.2] — confirm smooth(er) curve with no hard zero at 0.01.

---

### BUG-M3 — Per-class gamma/clip tensors in AsymmetricLoss

**File:** `ml/src/training/losses.py`

**Change:** `__init__()` accepts either scalar or 1-D tensor for `gamma_neg`, `gamma_pos`, `clip`:

```python
def __init__(self, gamma_neg=2.0, gamma_pos=1.0, clip=0.01, reduction="mean"):
    super().__init__()
    # Store as tensors; scalar → broadcast over [B, C] in forward
    self.register_buffer("gamma_neg", torch.as_tensor(gamma_neg, dtype=torch.float))
    self.register_buffer("gamma_pos", torch.as_tensor(gamma_pos, dtype=torch.float))
    self.register_buffer("clip",      torch.as_tensor(clip,      dtype=torch.float))
```

In `forward()`, the existing `prob_neg ** self.gamma_neg` already broadcasts correctly whether `gamma_neg` is scalar or [C].

**Suggested per-class defaults for next training run:**
```python
# gamma_neg: higher = more aggressive negative suppression
# Use higher values for large, clean classes; lower for starved/noisy ones
gamma_neg = torch.tensor([2.0, 0.5, 2.0, 2.0, 2.0, 2.0, 1.5, 1.0, 2.0, 2.0])
#            CU   DoS  EB   GE   IU   ME   Re   TS   TOD  UR
```

**Verify:** Forward pass with per-class tensor; confirm shapes are compatible. Check gradients flow correctly for each class.

---

## Phase 4 — Extractor Fixes

### BUG-M1 — `id(lval)` → `lval.name` in `_compute_return_ignored()`

**File:** `ml/src/preprocessing/graph_extractor.py`

**Change:**
```python
# Current:
all_read_vars.add(id(rv))
...
if id(lval) not in all_read_vars:

# Fix:
all_read_vars.add(rv.name)
...
if lval.name not in all_read_vars:
```

**Note:** If `lval.name` is not unique across functions (two different functions both have a temp var named `TMP_0`), this could create false negatives. Use a `(function.name, lval.name)` tuple if `lval` has a parent function reference available. Verify against Slither IR structure.

**Verify:** Run `_compute_return_ignored()` on 5 known contracts with ignored return values — confirm return_ignored=1 is still correctly detected.

---

## Phase 5 — v7 Architecture (Re-extraction Required)

These bugs require full re-extraction of all 44,470 graphs and are tracked for the v7 training cycle.

### BUG-C3 — CFG node feature propagation

**Plan:** After all other fixes are implemented and training results are evaluated, add a post-extraction step in `graph_extractor.py` that copies parent FUNCTION node features down to each CFG child:
- For each FUNCTION node F, find all CFG nodes reachable via CONTAINS edges from F
- Copy F's features for dims [1,2,3,4,5,7,9,10,11] into each CFG child
- This makes CFG nodes "aware" of the function context they belong to
- Bump FEATURE_SCHEMA_VERSION to v7
- Re-extract all 44,470 graphs

### BUG-H1 — Phase 2 third CF hop

**Plan:** In `gnn_encoder.py`, add `conv3c = GATConv(...)` as a third CONTROL_FLOW layer in Phase 2. Update forward() to apply conv3c after conv3b. Include conv3c output in JK buffer. Run full v7 training run to measure CEI detection improvement on Reentrancy.

### BUG-H7 / BUG-H8 — EMITS and INHERITS edges

**Plan:** In `graph_extractor.py`, update event detection to use Slither's current IR syntax for EMITS, and add `contract.inheritance` traversal for INHERITS. Both edges use existing type slots (3 and 4). No schema dimension change required — just populate currently empty edge types.

### BUG-L2 — Remove dead `in_unchecked` feature

**Plan:** In next schema bump, drop dim[9], shift dims [10,11] to [9,10], update NODE_FEATURE_DIM from 12 to 11. Re-extract all graphs.

### BUG-L3 — Content-based hash pairing

**Plan:** Store both path-MD5 (current) and content-MD5 in graph/token metadata during v7 re-extraction. Use content-MD5 as the primary dedup key.

---

## Verification Gate Before Training Restart

After Phase 1 and Phase 2 are complete, run these checks before launching training:

1. **Label counts:** `python ml/scripts/label_cleaner.py --dry-run` → confirm removal counts match estimates
2. **CSV integrity:** Row count, no NaN labels, all 10 class columns present, splits regenerated
3. **Feature validation:** `python ml/scripts/patch_graph_features.py --verify-only` → 0 OOR nodes
4. **Smoke run:** 2 epochs, batch_size=2, num_workers=0 — confirm loss > 0, no NaN, gnn_share > 5%
5. **Sampler check:** Log fraction of positive-label samples in first 10 batches — target ≥ 0.50
6. **DoS gradient check:** Log per-class grad norms after first backward — DoS column = 0
7. **Token schema:** Spot-check 5 token files — `feature_schema_version == "v6"`

---

## Documentation Updates Per Fix

| Bug | Doc to create/update |
|-----|---------------------|
| H9 (label cleaner) | `docs/changes/2026-05-17-label-cleaning.md` — new file with removal counts, per-class breakdown, audit JSON location |
| H6 (DoS mask) | `docs/STATUS.md` — add DoS loss_weight=0.0 to training config section |
| M5 (Brainmab) | `docs/changes/2026-05-17-label-cleaning.md` — include removed hash(es) |
| M6 (token version) | `docs/ACTIVE_BUGS.md` — mark FIXED |
| M7 (contract path) | `docs/ACTIVE_BUGS.md` — mark FIXED |
| C1, C2, C4, H2, H3 | `docs/STATUS.md` — update training config table |
| H10 (sampler) | `docs/STATUS.md` — add use_weighted_sampler=True to config |
| M4, M8, M9, M10 | `docs/STATUS.md` — update training config table |
| M1 (id → name) | `docs/ACTIVE_BUGS.md` — mark FIXED |
| M2, M3 (ASL) | `docs/STATUS.md` — update loss config section |
| L1, L4 | `docs/ACTIVE_BUGS.md` — mark FIXED |
| All Phase 5 | `docs/STATUS.md` v7 section (new), `docs/ACTIVE_BUGS.md` mark DEFERRED→IN_PROGRESS when started |
| After training restart | `docs/changes/INDEX.md` — new entry; `docs/STATUS.md` — update training status |
