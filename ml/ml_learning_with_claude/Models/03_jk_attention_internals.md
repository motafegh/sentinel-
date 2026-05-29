# Models — GNN Encoder Chunk 3: `_JKAttention` Internals & Diagnostics

> **File:** `ml/src/models/gnn_encoder.py` — **lines 76–131**
> **What you'll learn:** How `_JKAttention` is implemented, `register_buffer` (what it is and why it matters), per-phase vs per-node weight tracking, the JK entropy regularizer (C-3), and how training vs eval mode differ.
> **Time:** ~20 minutes
> **Interview relevance:** ML (deep GNN design), AI (attention diagnostics), MLOps (PyTorch state management, DDP)

---

## 1. The Full `_JKAttention` Class (lines 76–131)

```python
class _JKAttention(nn.Module):
    def __init__(self, channels: int, num_phases: int = 3) -> None:
        super().__init__()
        self.attn = nn.Linear(channels, 1, bias=False)    # [256] → [1]
        self.register_buffer("last_weights",      torch.zeros(num_phases))     # [3]
        self.register_buffer("last_weight_stds",  torch.zeros(num_phases))     # [3]
        self.last_node_weights: "torch.Tensor | None" = None   # not a buffer

    def forward(self, xs: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        stacked = torch.stack(xs, dim=1)        # [N, 3, 256]
        scores  = self.attn(stacked)             # [N, 3, 1]
        weights = torch.softmax(scores, dim=1)   # [N, 3, 1] — sum to 1 over dim 1
        w_nk    = weights.squeeze(-1)            # [N, 3]

        # Update buffers in-place
        self.last_weights.copy_(w_nk.mean(0).detach())      # [3] — mean over nodes
        self.last_weight_stds.copy_(w_nk.std(0).detach())   # [3] — std over nodes

        # Per-node weights: only in eval mode
        if not self.training:
            self.last_node_weights = w_nk.detach().cpu()

        output = (weights * stacked).sum(dim=1)              # [N, 256]

        # C-3: entropy regularizer
        jk_entropy = -(w_nk * (w_nk + 1e-8).log()).sum(dim=1).mean()  # scalar
        return output, jk_entropy
```

One linear layer, three buffers, one entropy computation. Let's go through each piece.

---

## 2. `register_buffer` — What It Is and Why It Matters

```python
self.register_buffer("last_weights",     torch.zeros(num_phases))
self.register_buffer("last_weight_stds", torch.zeros(num_phases))
```

`register_buffer` registers a tensor as part of the module's **state** but **not** as a trainable parameter. Buffers:

1. **Appear in `state_dict()`** — saved and loaded with `torch.save(model.state_dict())`
2. **Move with `.to(device)`** — calling `model.to("cuda")` moves all buffers too
3. **Work with DDP** — DistributedDataParallel handles buffer synchronization automatically
4. **Have `requires_grad=False`** — they never participate in backward pass

**Contrast with a regular Python attribute:**
```python
# NOT a buffer:
self.last_weights = torch.zeros(3)    # dies on model.to("cuda")
                                       # not saved in state_dict
                                       # DDP ignores it

# A buffer:
self.register_buffer("last_weights", torch.zeros(3))  # all of the above handled
```

If you stored `last_weights` as a plain Python attribute, it would stay on CPU even after `model.to("cuda")`. Then `self.last_weights.copy_(w_nk.mean(0).detach())` would fail with a device mismatch error because `w_nk` is on CUDA but `last_weights` is on CPU.

> 🎯 **INTERVIEW FOCUS:** "What's the difference between `register_buffer`, `register_parameter`, and a plain Python attribute on an `nn.Module`?" — `register_parameter`: trainable, in state_dict, moves with .to(). `register_buffer`: not trainable, in state_dict, moves with .to(). Plain attribute: not trainable, NOT in state_dict, does NOT move with .to(). Use buffers for running statistics, masks, or any non-trainable state you need to persist and move with the model.

---

## 3. `last_weights` and `last_weight_stds` — Batch-Level Monitoring

```python
self.last_weights.copy_(w_nk.mean(0).detach())     # [3]
self.last_weight_stds.copy_(w_nk.std(0).detach())  # [3]
```

**`last_weights`**: mean attention weight per phase, averaged over all N nodes in the batch. Shape `[3]` — one value per phase. After training step i, `last_weights[0]` tells you: "on average, how much weight did Phase 1 get across all nodes in this batch?"

**`last_weight_stds`**: standard deviation of the per-node weights across nodes, per phase. This is the more informative diagnostic.

| `last_weight_stds` value | What it means |
|--------------------------|--------------|
| ≈ 0.0 for all phases | JK assigns similar weights to every node — behaving as a constant (not content-dependent). Phase attention is collapsed. |
| > 0.10 for some phase | Different nodes get meaningfully different weights for that phase. JK is genuinely routing. |

**Why `.copy_()` instead of `=`?**

`self.last_weights = w_nk.mean(0).detach()` would create a new tensor and replace the buffer reference — the buffer might lose its device registration. `.copy_()` writes into the existing buffer in-place, preserving the device, dtype, and registration. This is the correct way to update registered buffers.

**The trainer reads these values:**
```python
# In trainer.py (training loop):
logger.info(f"JK weights: {model.gnn.jk.last_weights.tolist()}")
logger.info(f"JK weight stds: {model.gnn.jk.last_weight_stds.tolist()}")
```

This gives real-time visibility into whether the JK mechanism is working as intended.

---

## 4. `last_node_weights` — Why It Can't Be a Buffer

```python
# Per-node weights: eval mode only
if not self.training:
    self.last_node_weights = w_nk.detach().cpu()   # [N, 3]
```

Unlike `last_weights` (shape `[3]` — always the same), `last_node_weights` has shape `[N, 3]` where **N varies per batch**. Registered buffers must have a fixed shape (they're pre-allocated with `torch.zeros`). A variable-size buffer can't be registered.

So `last_node_weights` is a plain Python attribute. Its drawbacks (not in state_dict, not device-aware) don't matter here:
- It's not used in training — only for post-hoc diagnostic analysis
- The `.cpu()` call explicitly moves it to CPU for analysis scripts
- It's set to `None` between evaluations

**Why only in eval mode?**

During training, N nodes in a batch can be tens of thousands — storing a full `[N, 3]` tensor every step would consume significant memory with no training benefit. In eval mode, you're typically running a single batch for diagnosis, not the full training loop.

**Who reads it:**

`jk_weight_hist.py` diagnostic script calls `model.eval()`, runs a forward pass, then reads `model.gnn.jk.last_node_weights` to build histograms showing which node types (CONTRACT, FUNCTION, CFG_CALL, etc.) get high weight for which phases.

---

## 5. JK Entropy — C-3 Regularizer (lines 129–131)

```python
jk_entropy = -(w_nk * (w_nk + 1e-8).log()).sum(dim=1).mean()
```

**Shannon entropy** of the per-node weight distribution. For K=3 phases:
```
H(node) = -Σ_{k=1}^{3} w_k * log(w_k)
```

- `H = 0`: all weight on one phase (e.g., `[0, 0, 1]`). Collapsed — JK picked one phase and ignored the others.
- `H = log(3) ≈ 1.099`: uniform weights (`[1/3, 1/3, 1/3]`). JK is ignoring content entirely.
- `H ∈ (0, log(3))`: some diversity — different nodes weight phases differently.

**The `+ 1e-8` inside `log()`** prevents `log(0)` for nodes with near-zero weight on some phase (which would give `-inf` and NaN gradients).

**`.sum(dim=1)` then `.mean()`:**
- `.sum(dim=1)`: compute entropy for each of the N nodes → `[N]`
- `.mean()`: average over all nodes → scalar

**C-3: gradient-attached:**

`jk_entropy` is returned without `.detach()`. The trainer adds it to the loss:
```python
# In trainer.py:
loss = main_loss + λ_entropy * (-jk_entropy)   # maximize entropy = minimize negative entropy
```

The negative entropy term encourages JK to spread attention across phases rather than collapsing to one. The gradient flows back through `w_nk` → `softmax` → `self.attn` — directly training the attention linear layer to produce more diverse phase weights.

> 🎯 **INTERVIEW FOCUS:** "How would you prevent an attention mechanism from collapsing to always attending to one input?" — Entropy regularization: add `λ * H(attention_weights)` to the loss (or `-λ * H` if you're minimizing). This penalizes low-entropy (peaked) distributions and rewards diversity. The gradient flows through the softmax weights back to the scoring function.

---

## 6. The Attention Computation in Detail

```python
stacked = torch.stack(xs, dim=1)        # [N, 3, 256]
scores  = self.attn(stacked)             # [N, 3, 256] → [N, 3, 1]
weights = torch.softmax(scores, dim=1)   # [N, 3, 1]  ← softmax over 3 phases
w_nk    = weights.squeeze(-1)            # [N, 3]
output  = (weights * stacked).sum(dim=1) # [N, 256]
```

**Step by step:**

1. Stack the 3 phase outputs: `[N, 256]` × 3 → `[N, 3, 256]`
2. Apply linear: `nn.Linear(256, 1, bias=False)` maps each 256-dim vector to a scalar score → `[N, 3, 1]`
3. Softmax over the phase dimension (dim=1): for each node, scores across 3 phases become probabilities summing to 1 → `[N, 3, 1]`
4. Weighted sum: multiply each phase embedding by its weight, sum over phases → `[N, 256]`

**`bias=False` on `self.attn`:**

A bias term would add a constant to each phase score independently of the node's content. With bias, the model could learn "Phase 2 always gets +0.3 regardless of which node we're scoring." This creates a **content-independent phase preference** — the attention would be biased toward one phase for all nodes. `bias=False` forces the attention to be purely content-dependent: "does this phase's 256-dim embedding, when projected to a scalar, produce a higher score than the others?"

---

## 7. Training vs Eval Mode — Summary

| Behavior | Training mode | Eval mode |
|----------|--------------|-----------|
| `last_node_weights` updated | No (set to None) | Yes (`[N, 3]` on CPU) |
| `last_weights` updated | Yes (every forward) | Yes (every forward) |
| `last_weight_stds` updated | Yes (every forward) | Yes (every forward) |
| `jk_entropy` attached to grad | Yes | Yes (but `.backward()` not called) |
| Dropout in GNNEncoder | Active | Disabled |

The `last_node_weights` is the only behavior that differs between modes. Everything else runs identically.

---

## 8. `use_jk=False` — Backward Compatibility (lines 570–578 of `forward`)

```python
if self.use_jk and self.jk is not None:
    x, _jk_entropy = self.jk(_live)         # JK aggregation
else:
    _jk_entropy = x.new_zeros(1).squeeze()  # scalar 0.0, no gradient
    # x is already phase3_out — the last assignment in Phase 3
```

When `use_jk=False`, `_live` is populated but never used. `x` at this point is the Phase 3 LayerNorm output — the model returns only the final phase's representation, identical to SENTINEL v5 behavior.

`x.new_zeros(1).squeeze()` creates a scalar zero tensor on the same device as `x`. This ensures the trainer's `loss += λ * jk_entropy` doesn't produce NaN and stays at 0 even when JK is disabled — no special-casing needed in the training loop.

---

## 9. Complete `_JKAttention` Mental Model

```
Input: _live = [phase1_out [N,256], phase2_out [N,256], phase3_out [N,256]]
                    ↓ torch.stack(dim=1)
             stacked [N, 3, 256]
                    ↓ Linear(256→1, bias=False) applied to last dim
             scores  [N, 3, 1]
                    ↓ softmax(dim=1) — over the 3 phases
             weights [N, 3, 1]
                    ↓ squeeze(-1)
             w_nk    [N, 3]   ← per-node phase probabilities

Side effects:
  last_weights     = w_nk.mean(0)    [3]  (buffer, always)
  last_weight_stds = w_nk.std(0)     [3]  (buffer, always)
  last_node_weights = w_nk.cpu()  [N,3]  (plain attr, eval only)

Output:
  output     = (weights * stacked).sum(dim=1)   [N, 256]
  jk_entropy = mean(-Σ w_k log(w_k))            scalar, gradient-attached
```

---

## Interview Questions

1. **"What is `register_buffer` in PyTorch and when do you use it?"**
   → Registers a non-trainable tensor as part of the module's state. Included in `state_dict()`, moved by `.to(device)`, handled by DDP. Use for: running statistics (BatchNorm's running mean), masks, positional encodings, and any non-learned state that must persist with the model.

2. **"How do you track whether your attention mechanism is working correctly during training?"**
   → Monitor the entropy of attention weights. Low entropy → collapsed to one input (degenerate). Uniform distribution → content-independent (also degenerate). Track mean weights per phase via a buffer to log to MLflow each epoch. Track std to check if the routing is node-content-dependent.

3. **"Why would you add entropy as a regularizer to an attention mechanism?"**
   → To prevent attention collapse — the tendency for a learned attention to assign all weight to one input and ignore others. Entropy regularization adds `λ * H(weights)` to the loss, making high-entropy (diverse) weight distributions preferred. Gradient flows through the softmax back to the scoring network.

4. **"Why does `_JKAttention` compute per-node weights only in eval mode?"**
   → Memory efficiency during training. Per-node weights have shape `[N, 3]` where N can be tens of thousands. Storing this tensor every training step wastes memory with no training benefit — it's only useful for post-hoc diagnostic analysis. Eval mode typically runs a single diagnostic batch, so the overhead is acceptable.

---

**GNN Encoder module complete.** ✅

Check your understanding:
- Can you explain what IMP-G1 changes about Phase 2 and why it produces better JK inputs?
- Can you explain how REVERSE_CONTAINS edges are created and why they use type-7 embeddings?
- Can you explain `register_buffer` vs a plain attribute and when each is appropriate?
- Can you explain what JK entropy measures and what values indicate healthy vs collapsed attention?

**Next:** `04_transformer_lora_and_prefix_injection.md` → already written as `02_transformer_lora_and_prefix_injection.md`. Continue from there.
