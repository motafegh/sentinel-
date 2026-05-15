# SENTINEL Pre-Training Improvement Plan
## Everything We Fix Before the Next Training Run

**Version:** 1.0  
**Date:** 2026-05-14  
**Scope:** v5.1 → v5.2 (all code changes before `train.py` is launched again)

This document supersedes the partial roadmap in `project_next_steps.md` and incorporates the
findings from the v5.1-fix28 training analysis and the GNN Enhancement Proposal v2.0 audit.

**The key constraint:** No new training is launched until every item in Phases 0–3 is done
and the smoke run (Phase 4) gates pass.

---

## 1. Why We Fix Everything First

Training runs are expensive (8–10h per 60-epoch run on RTX 3070). The v5.1-fix28 run proved
that launching training with any one of these problems active wastes the entire compute budget:

- **Dead GNN:** collapsed to 10% gradient share by epoch 8; six classes never learned
- **Broken scheduler on resume:** LR frozen at wrong value for 28 epochs
- **Shared CONTAINS embedding:** Phase 3 aggregation semantically confused
- **No contrastive data for CEI:** behavioral gate can never pass without explicit CEI pairs

The fix-everything-first approach means one clean 60-epoch run produces a valid, trustworthy
checkpoint. The gate hierarchy is:

```
Phase 0 (quick fixes)
    → Phase 1 (architecture)
        → Phase 2 (training pipeline)
            → Phase 3 (data)
                → Phase 4 (2-epoch smoke)
                    → Phase 5 (60-epoch full run)
```

---

## 2. Complete Improvement Inventory

### 2.1 Architecture (GNN)

| ID | Change | Files | Addresses | Effort |
|---|---|---|---|---|
| A1 | JK Connections (attention mode) | `gnn_encoder.py` | L1, L3, L4 | 2h |
| A2 | Per-phase LayerNorm | `gnn_encoder.py` | Phase dominance | 30m |
| A3 | REVERSE_CONTAINS embedding (type 7) | `graph_schema.py`, `gnn_encoder.py` | L2 | 45m |
| A4 | TrainConfig validator relaxation | `trainer.py` | L4 | 10m |
| A5 | SentinelModel JK params | `sentinel_model.py` | — | 20m |
| A6 | Checkpoint version gate | `trainer.py` | compatibility | 30m |

### 2.2 Training Pipeline

| ID | Change | Files | Problem Solved | Effort |
|---|---|---|---|---|
| B1 | Separate LR groups (GNN/LoRA/other) | `trainer.py` | GNN LR starvation | 1h |
| B2 | Scheduler resume fix | `trainer.py` | LR frozen on resume | 1h |
| B3 | NaN loss counter + logging | `trainer.py` | Silent NaN batches | 20m |
| B4 | 24K orphan-token warning → DEBUG | `dual_path_dataset.py` | Log noise | 5m |

### 2.3 Monitoring

| ID | Change | Files | Purpose | Effort |
|---|---|---|---|---|
| C1 | JK attention weight logging | `trainer.py` | Phase dominance visibility | 30m |
| C2 | GNN collapse early warning | `trainer.py` | Catch L3 recurrence early | 20m |
| C3 | Phase dominance alert | `trainer.py` | Catch if Phase 1 > 80% | 15m |

### 2.4 Tests

| ID | Change | Files | Gate | Effort |
|---|---|---|---|---|
| T1 | Fix `test_gnn_encoder.py` dim bug | `ml/tests/test_gnn_encoder.py` | Correctness | 5m |
| T2 | `test_jk_gradient_flow()` — NON-NEGOTIABLE | `ml/tests/test_gnn_encoder.py` | Must pass before training | 30m |
| T3 | Test REVERSE_CONTAINS uses separate embedding | `ml/tests/test_gnn_encoder.py` | Correctness | 20m |
| T4 | Test JK output dimension unchanged | `ml/tests/test_gnn_encoder.py` | Compatibility | 10m |

### 2.5 Data Preparation

| ID | Change | Notes | Effort |
|---|---|---|---|
| D1 | CEI contrastive pairs (~50 pairs) | Reentrancy-vulnerable + safe (write-before-call) | 2h writing + 1h extract |
| D2 | DoS augmentation (~300 contracts) | SmartBugs SWC-128 + synthetic templates | 2h |
| D3 | Pos_weight recompute + cap | From deduped split; DoS cap ≤ 80.0 | 10m |

---

## 3. Implementation Details

### A1 + A2: JK Connections with Per-Phase LayerNorm

**File:** `ml/src/models/gnn_encoder.py`

**Critical background:** The existing `return_intermediates` infrastructure uses `.detach().clone()`
at lines 294, 303, 313. Using those for JK would silently break gradient flow — JK attention
weights would never update. The implementation MUST collect **live** (non-detached) tensors in
a **separate code path**.

**New imports:**
```python
from torch_geometric.nn.models import JumpingKnowledge
```

**New `__init__` parameters:**
```python
def __init__(
    self,
    ...,
    use_jk:   bool = True,
    jk_mode:  str  = 'attention',
) -> None:
```

**New modules in `__init__`:**
```python
self.use_jk = use_jk
if use_jk:
    self.jk         = JumpingKnowledge(mode=jk_mode, channels=hidden_dim, num_layers=3)
    self.phase_norm = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(3)])
else:
    self.jk         = None
    self.phase_norm = None
```

**Modified `forward()` — collect live intermediates and apply JK after Phase 3:**
```python
def forward(self, x, edge_index, batch, edge_attr=None, return_intermediates=False):
    # ... existing guards and edge setup unchanged ...

    _live: list | None = [] if self.jk is not None else None  # LIVE — for JK only
    _intermediates: dict = {}                                   # DETACHED — for inspection

    # ── Phase 1 ──────────────────────────────────────────────────────────────
    x  = self.conv1(x, struct_ei, struct_ea)
    x  = self.relu(x)
    x  = self.dropout(x)
    x2 = self.conv2(x, struct_ei, struct_ea)
    x2 = self.relu(x2)
    x  = self.dropout(x2 + x)

    if _live is not None:
        _live.append(x)                              # LIVE — grad flows through this
    _intermediates["after_phase1"] = x.detach().clone()  # DETACHED — for inspection only

    # ── Phase 2 ──────────────────────────────────────────────────────────────
    x2 = self.conv3(x, cfg_ei, cfg_ea)
    x2 = self.relu(x2)
    x  = x + self.dropout(x2)

    if _live is not None:
        _live.append(x)
    _intermediates["after_phase2"] = x.detach().clone()

    # ── Phase 3 ──────────────────────────────────────────────────────────────
    x2 = self.conv4(x, rev_contains_ei, rev_contains_ea)
    x2 = self.relu(x2)
    x  = x + self.dropout(x2)

    if _live is not None:
        _live.append(x)
    _intermediates["after_phase3"] = x.detach().clone()

    # ── JK aggregation ───────────────────────────────────────────────────────
    # Normalize per phase to prevent Phase 1 magnitude dominance, then aggregate.
    # Output shape: [N, hidden_dim] — identical to non-JK output.
    if self.jk is not None:
        normed = [norm(h) for norm, h in zip(self.phase_norm, _live)]
        x = self.jk(normed)   # grad flows through both jk and phase_norm

    if return_intermediates:
        return x, batch, _intermediates
    return x, batch
```

**Why this is correct:**
- `_live` holds the same `x` tensor that the forward pass uses — PyTorch autograd handles
  multiple references naturally. The tensor is not consumed by appending.
- `_intermediates` continues to use `.detach().clone()` — the inspection/testing infrastructure
  (`test_cfg_embedding_separation.py`) is fully backward compatible.
- `self.jk(normed)` replaces `x` with the JK-aggregated output. Output dim = `hidden_dim = 128`
  (JK attention mode preserves dimension). No downstream changes needed.

**Parameter budget addition:**
- JK attention: 3 × 128 = 384 params (+0.43% over ~90K GNN params)
- Per-phase LayerNorm: 3 × 2 × 128 = 768 params
- Total addition: **~1,152 params** — negligible

---

### A3: REVERSE_CONTAINS — No Re-Extraction Required

**Background:** Phase 3 currently flips CONTAINS edges at runtime (`edge_index.flip(0)`) but
reuses the type-5 embedding for the reversed direction. The GNN cannot distinguish "this function
contains this statement" from "this statement aggregates into this function."

**Key insight:** REVERSE_CONTAINS does NOT need to be stored in graph `.pt` files. The edge type
is assigned at runtime in the GNN forward pass. No schema version bump, no re-extraction.

**File 1: `ml/src/preprocessing/graph_schema.py`**

```python
# Change:
NUM_EDGE_TYPES = 7

EDGE_TYPES = {
    "CALLS":        0,
    "READS":        1,
    "WRITES":       2,
    "EMITS":        3,
    "INHERITS":     4,
    "CONTAINS":     5,
    "CONTROL_FLOW": 6,
}

# To:
NUM_EDGE_TYPES = 8

EDGE_TYPES = {
    "CALLS":            0,
    "READS":            1,
    "WRITES":           2,
    "EMITS":            3,
    "INHERITS":         4,
    "CONTAINS":         5,
    "CONTROL_FLOW":     6,
    "REVERSE_CONTAINS": 7,   # runtime-only — never stored in graph .pt files
}
```

`FEATURE_SCHEMA_VERSION` does NOT change — the graph file format is unchanged.

**File 2: `ml/src/models/gnn_encoder.py` — Phase 3 runtime embedding**

```python
# In the Phase 3 section, replace:
rev_contains_ea = e[contains_mask] if e is not None else None

# With:
if e is not None:
    _n_rev = contains_mask.sum()
    _rev_type = EDGE_TYPES["REVERSE_CONTAINS"]   # 7
    _rev_type_t = torch.full(
        (_n_rev,), _rev_type, dtype=torch.long, device=x.device
    )
    rev_contains_ea = self.edge_embedding(_rev_type_t)  # [E_rev, edge_emb_dim] — type 7
else:
    rev_contains_ea = None
```

The `nn.Embedding(NUM_EDGE_TYPES, edge_emb_dim)` table now has 8 rows instead of 7. Row 7
(REVERSE_CONTAINS) is randomly initialized and learned during training. Rows 0-6 are unaffected.

**Add to gnn_encoder.py imports:**
```python
# Already imported: EDGE_TYPES — no new import needed
```

**Note on checkpoint compatibility:** This changes `edge_embedding.weight` shape from [7,32] to
[8,32]. Old checkpoints will mismatch. The version gate (A6) handles this gracefully.

---

### A4: TrainConfig Validator Relaxation

**File:** `ml/src/training/trainer.py`

```python
# Replace (lines ~233-237):
def __post_init__(self) -> None:
    if self.gnn_layers != 4:
        raise ValueError(
            f"gnn_layers={self.gnn_layers} is not supported in v5.0. "
            "Only gnn_layers=4 is implemented."
        )

# With:
def __post_init__(self) -> None:
    if self.gnn_layers < 4:
        raise ValueError(
            f"gnn_layers={self.gnn_layers} is invalid. "
            "Minimum is 4 (three-phase architecture requires layers 1-4)."
        )
    if self.gnn_layers != 4:
        logger.warning(
            f"gnn_layers={self.gnn_layers} != 4. Experimental — verify GNNEncoder "
            "supports this depth. Proceeding."
        )
```

This unblocks future `gnn_layers=5` (second CONTROL_FLOW hop) without removing the safety
guard for accidental misconfiguration.

---

### A5: SentinelModel JK Parameters

**File:** `ml/src/models/sentinel_model.py`

Add to `__init__` signature and constructor body:

```python
def __init__(
    self,
    ...,
    gnn_use_jk:   bool = True,
    gnn_jk_mode:  str  = 'attention',
) -> None:
    ...
    self.gnn = GNNEncoder(
        hidden_dim    = gnn_hidden_dim,
        heads         = gnn_heads,
        dropout       = gnn_dropout,
        use_edge_attr = use_edge_attr,
        edge_emb_dim  = gnn_edge_emb_dim,
        num_layers    = gnn_num_layers,
        use_jk        = gnn_use_jk,      # NEW
        jk_mode       = gnn_jk_mode,     # NEW
    )
```

No changes to `forward()` — JK is entirely internal to `GNNEncoder.forward()`.

Add matching fields to `TrainConfig`:
```python
gnn_use_jk:  bool = True
gnn_jk_mode: str  = 'attention'
```

---

### A6: Checkpoint Version Gate

**File:** `ml/src/training/trainer.py`

Add helper function (module level):
```python
def _parse_version(v: str) -> tuple:
    """Parse 'v5.2' → (5, 2). Tuples compare correctly: (5, 10) > (5, 2)."""
    return tuple(int(x) for x in v.lstrip('v').split('.'))
```

When **saving** a checkpoint, add:
```python
torch.save({
    "model":           model.state_dict(),
    "optimizer":       optimizer.state_dict(),
    "scheduler":       scheduler.state_dict(),
    "epoch":           epoch,
    "best_f1":         best_f1,
    "patience_counter": patience_counter,
    "model_version":   "v5.2",      # NEW
    "total_steps":     total_steps,  # NEW — required for scheduler resume fix (B2)
    "config":          vars(cfg),
}, checkpoint_path)
```

When **loading** a checkpoint:
```python
ckpt_version = _parse_version(checkpoint.get("model_version", "v5.1"))
if ckpt_version < _parse_version("v5.2"):
    missing, unexpected = model.load_state_dict(checkpoint["model"], strict=False)
    jk_keys = [k for k in missing if "jk" in k or "phase_norm" in k]
    rev_keys = [k for k in missing if "edge_embedding" in k]
    if jk_keys:
        logger.warning(
            f"Loaded pre-v5.2 checkpoint; {len(jk_keys)} JK/LayerNorm keys "
            "initialised randomly — these will be learned from scratch."
        )
    if rev_keys:
        logger.warning(
            f"Loaded pre-v5.2 checkpoint; edge_embedding resized for REVERSE_CONTAINS. "
            "Row 7 initialised randomly."
        )
else:
    model.load_state_dict(checkpoint["model"], strict=True)
```

---

### B1: Separate LR Groups

**Root cause addressed:** The GNN (~90K params) and the TF LoRA (~590K params) use the same LR.
The LoRA is adapting a pretrained 125M model and converges fast at `lr=2e-4`. The GNN is trained
from scratch and needs a higher LR to compete for gradient influence. Equal LR leaves the GNN
underpowered.

**New TrainConfig fields:**
```python
gnn_lr_multiplier:  float = 2.5   # GNN LR = base_lr × 2.5
lora_lr_multiplier: float = 0.5   # LoRA LR = base_lr × 0.5
```

**File:** `ml/src/training/trainer.py` — optimizer construction:
```python
# Partition parameters into three groups
gnn_params    = list(model.gnn.parameters())
_gnn_ids      = {id(p) for p in gnn_params}
lora_params   = [p for n, p in model.named_parameters()
                 if 'lora_' in n and id(p) not in _gnn_ids]
_lora_ids     = {id(p) for p in lora_params}
other_params  = [p for n, p in model.named_parameters()
                 if id(p) not in _gnn_ids and id(p) not in _lora_ids]

optimizer = torch.optim.AdamW(
    [
        {'params': gnn_params,   'lr': cfg.lr * cfg.gnn_lr_multiplier,  'name': 'gnn'},
        {'params': lora_params,  'lr': cfg.lr * cfg.lora_lr_multiplier, 'name': 'lora'},
        {'params': other_params, 'lr': cfg.lr,                           'name': 'other'},
    ],
    weight_decay=cfg.weight_decay,
    betas=(0.9, 0.999),
    eps=1e-8,
)
logger.info(
    f"Optimizer — 3 param groups | "
    f"GNN lr={cfg.lr * cfg.gnn_lr_multiplier:.2e} ({len(gnn_params)} tensors) | "
    f"LoRA lr={cfg.lr * cfg.lora_lr_multiplier:.2e} ({len(lora_params)} tensors) | "
    f"Other lr={cfg.lr:.2e} ({len(other_params)} tensors)"
)
```

**Scheduler note:** The LR scheduler must be built using `max_lr` per group when using OneCycle,
or applied to the optimizer as a whole when using CosineAnnealingLR. Verify the current scheduler
type before finalising. If using CosineAnnealing, the per-group LR ratios are preserved by default.

**Smoke run validation:** At step 100, GNN gradient norm should be >= 0.3 (was 0.6 at epoch 1,
collapsed to 0.034 by epoch 24). If still <0.1, raise multiplier to 3.0-4.0.

---

### B2: Scheduler Resume Fix

**Root cause:** `total_steps` is recomputed on resume from current dataset size and remaining
epochs. If either changed (code edit, dataset change, epoch count), the scheduler state loaded
from checkpoint maps to a different position in the new schedule.

**Fix strategy:** Save `total_steps` in checkpoint. On resume:
- If `total_steps` matches → restore scheduler state normally
- If mismatch → rebuild scheduler and fast-forward `optimizer_step` steps to approximate
  the correct LR position rather than leaving LR frozen

```python
# In trainer.py resume block:
if "total_steps" in checkpoint and checkpoint["total_steps"] == total_steps:
    scheduler.load_state_dict(checkpoint["scheduler"])
    logger.info("Scheduler state restored — total_steps matches.")
else:
    steps_done = checkpoint.get("epoch", 0) * steps_per_epoch
    for _ in range(steps_done):
        scheduler.step()
    logger.warning(
        f"Scheduler fast-forwarded {steps_done} steps after total_steps mismatch. "
        f"LR approximated, not exact — recommend fresh run over long resume."
    )
```

---

### B3: NaN Loss Counter

**File:** `ml/src/training/trainer.py` — inside `train_one_epoch`:

```python
# Initialize before batch loop:
_nan_loss_count = 0

# Inside batch loop, after loss computation:
if torch.isnan(loss) or torch.isinf(loss):
    _nan_loss_count += 1
    if _nan_loss_count % 20 == 1:  # log every 20th occurrence
        logger.warning(
            f"NaN/Inf loss at batch {batch_idx} (total so far: {_nan_loss_count}). "
            "Scaler will skip this step."
        )
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    continue

# At end of epoch:
if _nan_loss_count > 0:
    logger.warning(f"Epoch {epoch}: {_nan_loss_count} NaN/Inf loss batches skipped.")
```

---

### B4: Orphan-Token Warning Downgrade

**File:** `ml/src/datasets/dual_path_dataset.py`

The message "N token files have no matching graph file — skipped" is logged at WARNING level
on every dataset creation (train + val, twice per epoch). Over 60 epochs = 120 warning lines
that obscure real warnings.

Change the log level from `logger.warning(...)` to `logger.debug(...)` (or `logger.info(...)` 
once at startup). This is a normal condition for the deduped dataset.

---

### C1: JK Attention Weight Logging

**File:** `ml/src/training/trainer.py` — inside per-epoch validate loop or after evaluate():

```python
# After validation, extract JK attention weights if available:
if hasattr(model.gnn, 'jk') and model.gnn.jk is not None:
    # JumpingKnowledge attention mode stores weights as model.gnn.jk.att
    # Each phase's attention logit is a learned parameter (not per-sample)
    try:
        jk_weights = torch.softmax(model.gnn.jk.att.squeeze(), dim=0)
        jk_log = {f"jk_phase{i+1}_attn": jk_weights[i].item() for i in range(len(jk_weights))}
        logger.info(
            f"Epoch {epoch} JK attention weights: "
            + " | ".join(f"phase{i+1}={v:.3f}" for i, v in enumerate(jk_weights.tolist()))
        )
        # Log to MLflow if active
    except Exception:
        pass  # JK may have different internal structure; don't crash training
```

**Note:** The exact attribute path for `JumpingKnowledge` attention weights depends on the PyG
version. Inspect `model.gnn.jk.named_parameters()` after first instantiation to confirm.

---

### C2 + C3: GNN Collapse and Phase Dominance Alerts

**File:** `ml/src/training/trainer.py`

```python
# Track consecutive low-GNN-share optimizer steps:
_low_gnn_share_count = 0
_GNN_COLLAPSE_THRESHOLD = 0.10   # < 10% share
_PHASE_DOMINANCE_THRESHOLD = 0.80  # > 80% for one JK phase

# Inside optimizer step logging:
total_norm = gnn_norm + tf_norm + fused_norm
if total_norm > 0:
    gnn_share = gnn_norm / total_norm
    if gnn_share < _GNN_COLLAPSE_THRESHOLD:
        _low_gnn_share_count += 1
        if _low_gnn_share_count >= 3:  # 3 consecutive log intervals
            logger.warning(
                f"GNN COLLAPSE WARNING: gnn_share={gnn_share:.3f} < "
                f"{_GNN_COLLAPSE_THRESHOLD} for {_low_gnn_share_count} consecutive "
                "log intervals. Consider stopping — architecture fix needed."
            )
    else:
        _low_gnn_share_count = 0

# JK phase dominance check (in epoch-end monitoring):
if jk_weights is not None:
    dominant_phase = jk_weights.max().item()
    if dominant_phase > _PHASE_DOMINANCE_THRESHOLD:
        logger.warning(
            f"JK PHASE DOMINANCE: one phase at {dominant_phase:.3f} attention. "
            "Consider increasing LayerNorm or adding per-phase dropout."
        )
```

---

### T1: Fix test_gnn_encoder.py Dimension Bug

**File:** `ml/tests/test_gnn_encoder.py`

```python
# Line 23 — change:
x = torch.randn(n_nodes, 8)
# To:
from ml.src.preprocessing.graph_schema import NODE_FEATURE_DIM
x = torch.randn(n_nodes, NODE_FEATURE_DIM)   # 12, not 8
```

This test currently creates 8-dim features but `GNNEncoder` defaults to `in_channels=12`.
Every test that creates an encoder with defaults and calls forward() would fail at runtime.

---

### T2: Non-Negotiable Gradient Flow Test

**File:** `ml/tests/test_gnn_encoder.py`

This test MUST pass before any training is launched. It would have caught the detach bug.

```python
def test_jk_gradient_flow():
    """All JK attention params must receive non-zero gradients after backward."""
    from ml.src.preprocessing.graph_schema import NODE_FEATURE_DIM, NUM_EDGE_TYPES, EDGE_TYPES
    gnn = GNNEncoder(use_jk=True, jk_mode='attention')

    # Minimal graph: 5 nodes, 4 edges with mixed types
    x          = torch.randn(5, NODE_FEATURE_DIM, requires_grad=True)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
    batch      = torch.zeros(5, dtype=torch.long)
    # edge_attr: one of each relevant type
    edge_attr  = torch.tensor(
        [EDGE_TYPES["CALLS"], EDGE_TYPES["CONTAINS"],
         EDGE_TYPES["CONTROL_FLOW"], EDGE_TYPES["CONTAINS"]],
        dtype=torch.long,
    )

    out, _ = gnn(x, edge_index, batch, edge_attr)
    loss = out.sum()
    loss.backward()

    # Verify JK parameters have non-zero gradients
    for name, param in gnn.jk.named_parameters():
        assert param.grad is not None, (
            f"JK param '{name}' has None gradient — detach bug present!"
        )
        assert param.grad.abs().sum() > 0, (
            f"JK param '{name}' has zero gradient — JK is not learning!"
        )

    # Verify LayerNorm parameters have gradients
    for i, ln in enumerate(gnn.phase_norm):
        for name, param in ln.named_parameters():
            assert param.grad is not None, (
                f"phase_norm[{i}].{name} has None gradient!"
            )
```

---

### T3: REVERSE_CONTAINS Uses Separate Embedding

**File:** `ml/tests/test_gnn_encoder.py`

```python
def test_reverse_contains_separate_embedding():
    """Phase 3 must use REVERSE_CONTAINS (type 7) embedding, not CONTAINS (type 5)."""
    from ml.src.preprocessing.graph_schema import EDGE_TYPES
    gnn = GNNEncoder(use_jk=False)  # isolate edge embedding test

    contains_type = EDGE_TYPES["CONTAINS"]          # 5
    reverse_type  = EDGE_TYPES["REVERSE_CONTAINS"]  # 7

    emb_contains = gnn.edge_embedding(
        torch.tensor([contains_type], dtype=torch.long)
    )
    emb_reverse = gnn.edge_embedding(
        torch.tensor([reverse_type], dtype=torch.long)
    )

    # After random init, embeddings are different with probability ~1.0
    assert not torch.allclose(emb_contains, emb_reverse, atol=1e-6), (
        "CONTAINS and REVERSE_CONTAINS embeddings are identical — "
        "Phase 3 is not using type 7!"
    )
```

---

### T4: JK Output Dimension Unchanged

**File:** `ml/tests/test_gnn_encoder.py`

```python
def test_jk_output_shape_unchanged():
    """JK must not change the output dimension — downstream is hardcoded to 128."""
    gnn = GNNEncoder(hidden_dim=128, use_jk=True, jk_mode='attention')
    x, edge_index, batch, edge_attr = _make_graph(n_nodes=10, n_edges=8)
    out, returned_batch = gnn(x, edge_index, batch, edge_attr)
    assert out.shape == (10, 128), f"Expected [10, 128], got {out.shape}"
    assert returned_batch.shape == (10,)

def test_jk_disabled_output_shape():
    """use_jk=False must produce identical output shape (backward compat)."""
    gnn = GNNEncoder(hidden_dim=128, use_jk=False)
    x, edge_index, batch, edge_attr = _make_graph(n_nodes=10, n_edges=8)
    out, _ = gnn(x, edge_index, batch, edge_attr)
    assert out.shape == (10, 128)
```

---

### D1: CEI Contrastive Pairs (~50 pairs)

**Purpose:** The behavioral gate "CEI-A fires, CEI-B silent" is an absolute requirement. Without
explicit vulnerable (call-before-write) and safe (write-before-call) contract pairs in training
data, the model cannot reliably learn this distinction.

**What to write:**
- `ml/data/augmented/cei_pairs/` directory
- ~25 vulnerable contracts: `withdraw_reentrancy_*.sol` — external call BEFORE state update
- ~25 safe contracts: `withdraw_safe_*.sol` — state update BEFORE external call (CEI pattern)
- Each pair should test a variation: with/without ReentrancyGuard, different call targets,
  ETH vs token transfers, modifier vs inline guard

**Graph extraction after writing contracts:**
```bash
source ml/.venv/bin/activate
PYTHONPATH=. python ml/scripts/inject_augmented.py --source ml/data/augmented/cei_pairs/
```

This script appends to `splits/deduped/train.csv` only (never val or test).

**Behavioral test mapping:** `ml/scripts/test_contracts/reentrancy_*.sol` — the manual test
contracts must include at least one CEI-A (vulnerable) and one CEI-B (safe) for gate validation.

---

### D2: DoS Augmentation (~300 contracts)

**Purpose:** Train split has only ~257 DoS (DenialOfService) contracts. The model consistently
scores F1=0.07-0.09 on DoS — it barely fires. Even with pos_weight=10.96, the class is too
rare for reliable learning.

**Sources:**
- SmartBugs curated: `ml/data/smartbugs-curated/` — filter for SWC-128 (DoS with block gas limit)
- SmartBugs wild: `ml/data/smartbugs-wild/` — grep for `gas` vulnerability labels
- Synthetic: loops unbounded by user-controlled input, unbounded token distribution patterns

**Target:** DoS train count 257 → ~550-600 (roughly doubling). Re-run inject_augmented.py.

---

### D3: Pos_Weight Recomputation

The trainer already computes pos_weights from the deduped training split at startup. This works
correctly — the warning "Fix #13: pos_weight recomputed from current training split" is expected.

Before the full run, confirm the DoS cap:
- DoS pos_weight must be ≤ 80.0 after augmentation (currently 10.96 sqrt-scaled — fine)
- If DoS augmentation brings count to ~550, pos_weight drops to ~7.5 sqrt-scaled — healthier

No code change needed. Just re-verify the output log line at training start.

---

## 4. Phased Implementation Plan

### Phase 0 — Immediate (no compute, ~1h)
Do these first, they're unconditional:

```
T1: fix test_gnn_encoder.py dim=8 → NODE_FEATURE_DIM
A4: relax TrainConfig validator (raise → warning for gnn_layers > 4)
```

Verify: run existing test suite, confirm no regressions.

---

### Phase 1 — Architecture (~4-5h total)

Order matters — do in sequence:

```
A3: REVERSE_CONTAINS in graph_schema.py + gnn_encoder.py Phase 3
    → verify: EDGE_TYPES["REVERSE_CONTAINS"] == 7, NUM_EDGE_TYPES == 8

A1+A2: JK + LayerNorm in gnn_encoder.py
    → verify: module creates without error, parameters exist

A5: SentinelModel + TrainConfig gnn_use_jk/gnn_jk_mode params
    → verify: SentinelModel forward() produces [B, 10] logits unchanged

A6: Checkpoint version gate (_parse_version, model_version field)
    → verify: loading old checkpoint logs warning, doesn't crash

T2: test_jk_gradient_flow — RUN THIS BEFORE ANYTHING ELSE TRAINS
T3: test_reverse_contains_separate_embedding
T4: test_jk_output_shape_unchanged
```

**Gate to proceed to Phase 2:** T2 PASSES. No exceptions.

---

### Phase 2 — Training Pipeline (~3-4h)

```
B2: scheduler resume fix (save total_steps, fast-forward on mismatch)
B1: separate LR groups (GNN/LoRA/other) in optimizer construction
B3: NaN loss counter
C1: JK attention weight logging
C2+C3: GNN collapse alert + phase dominance alert
B4: orphan-token warning → DEBUG
```

Verify: `pytest ml/tests/test_trainer.py` passes (add a test that confirms 3 param groups exist
in optimizer with correct LR ratios).

---

### Phase 3 — Data Preparation (~compute hours)

```
D3: verify pos_weights from deduped split (run once, inspect log output)
D1: write ~50 CEI contrastive pairs + inject into train split
D2: collect + inject ~300 DoS contracts into train split
```

After D1+D2:
- Verify: `wc -l ml/data/splits/deduped/train.csv` should increase by ~350+
- Verify: CEI-A contract passes `slither` with reentrancy detector firing
- Verify: CEI-B contract passes `slither` with reentrancy detector silent

---

### Phase 4 — Smoke Run (2 epochs, 10% data)

```bash
source ml/.venv/bin/activate
PYTHONPATH=. TRANSFORMERS_OFFLINE=1 python ml/scripts/train.py \
  --run-name v5.2-smoke --experiment-name sentinel-v5.2 \
  --epochs 2 --batch-size 16 --lr 2e-4 \
  --gnn-lr-multiplier 2.5 --lora-lr-multiplier 0.5 \
  --gnn-use-jk true --gnn-jk-mode attention \
  --lora-r 16 --lora-alpha 32 \
  --gnn-hidden-dim 128 --gnn-layers 4 --gnn-heads 8 --gnn-dropout 0.2 \
  --aux-loss-weight 0.3 --warmup-pct 0.06 \
  --label-csv ml/data/processed/multilabel_index_deduped.csv \
  --splits-dir ml/data/splits/deduped \
  --sample-frac 0.10
```

**Smoke gates (MUST ALL PASS before Phase 5):**

| Gate | Threshold | Measurement |
|---|---|---|
| GNN gradient share @ step 100 | ≥ 15% | Training log |
| GNN gradient share @ step 200 | ≥ 15% | Training log |
| JK Phase 2 attention weight | ≥ 5% | Training log |
| JK Phase 3 attention weight | ≥ 5% | Training log |
| Loss @ epoch 2 < loss @ epoch 1 | required | Training log |
| No NaN loss batches in epoch 1 | required | Training log |
| CUDA OOM | must not occur | Process exit |

If GNN share < 15%: raise `gnn_lr_multiplier` to 3.5 and re-smoke.
If JK attention collapsed to one phase: check LayerNorm init, re-smoke.

---

### Phase 5 — Full 60-Epoch Run (v5.2)

```bash
source ml/.venv/bin/activate
PYTHONPATH=. TRANSFORMERS_OFFLINE=1 nohup python ml/scripts/train.py \
  --run-name v5.2-jk --experiment-name sentinel-v5.2 \
  --epochs 60 --batch-size 16 --lr 2e-4 \
  --gnn-lr-multiplier 2.5 --lora-lr-multiplier 0.5 \
  --gnn-use-jk true --gnn-jk-mode attention \
  --lora-r 16 --lora-alpha 32 \
  --gnn-hidden-dim 128 --gnn-layers 4 --gnn-heads 8 --gnn-dropout 0.2 \
  --aux-loss-weight 0.3 --warmup-pct 0.06 --early-stop-patience 10 \
  --label-csv ml/data/processed/multilabel_index_deduped.csv \
  --splits-dir ml/data/splits/deduped \
  > ml/logs/train_v5.2_jk.log 2>&1 &
```

Fresh run only — no `--resume-from`. The scheduler fix means future resumes will work correctly,
but this first run should go clean end-to-end.

Monitor: `tail -f ml/logs/train_v5.2_jk.log | grep -E "Epoch|gnn_eye|F1|JK|collapse|dominance"`

---

## 5. Dependencies Map

```
T1 (test dim fix) ─────────────────────────────────┐
A4 (TrainConfig relax) ──────────────────────────┐  │
                                                  │  │
A3 (REVERSE_CONTAINS schema) ──────────┐          │  │
A1+A2 (JK + LayerNorm) ────────────────┤─ A5 ─── B1, B2, B3 ─── D3
A6 (version gate) ─────────────────────┘     │        │
                                              │        │
T2 (gradient flow) ← must verify A1+A2 ──────┘        │
T3 (REVERSE_CONTAINS) ← must verify A3                │
T4 (output shape) ← must verify A1+A2                 │
                                                       │
D1 (CEI pairs) ────────────────────────────────────────┤
D2 (DoS augment) ──────────────────────────────────────┘
                    ↓
              Phase 4 smoke
                    ↓
              Phase 5 full run
```

---

## 6. What We Are Explicitly NOT Doing Before Training

These items were considered and deferred:

| Item | Reason Deferred |
|---|---|
| Second CONTROL_FLOW hop (gnn_layers=5) | Needs JK validated first; TrainConfig now allows it for v5.3 |
| DFG edges | High value but medium-high effort; v5.3 after JK proven |
| R-GAT | Phase masking may be sufficient with JK; reconsider post-v5.2 |
| Focal loss | Different training objective; needs separate experiment, not pre-training fix |
| LoRA K projection | Changes trainable params significantly; not a pre-training fix |
| Graph re-extraction for REVERSE_CONTAINS | Not needed — runtime embedding swap in A3 |
| PDG / SSA | Very high effort; JK+DFG covers most of PDG's benefit at lower cost |

---

## 7. Success Criteria for v5.2

The v5.2 run is considered successful when ALL of these are met:

**Smoke gates (Phase 4):** All must pass  
**Training health (Phase 5 early):** GNN share ≥ 15% sustained through epoch 5  
**JK health:** All 3 phases maintain ≥ 5% attention weight through training  
**Behavioral test (post-training):**
- ≥ 70% detection rate on `ml/scripts/test_contracts/` vulnerable contracts
- ≥ 66% safe specificity on clean contracts
- CEI-A fires (absolute)
- CEI-B silent (absolute)

There is no numerical F1 gate derived from previous runs. The behavioral gates are the standard.
