# SENTINEL Model Evolution Analysis Plan
**Date:** 2026-05-20  
**Purpose:** Understand not just *which model scores higher*, but *what each model sees, how it processes it, why each design decision was made, and whether it achieved its intended effect.*

This plan supersedes and extends `v8-vs-v7-comparison-plan.md`. That plan answers "which is better." This one answers "why."

---

## Mental model: three layers of analysis

```
Layer 1 — INPUT REPRESENTATION
    What can each model see?
    What does each node feature encode?
    Which graph edges are traversable?
    What bugs distorted what the model learned from?

Layer 2 — COMPUTATION
    How does each model process what it sees?
    Which parts of the graph does it attend to?
    Which phases dominate the aggregation?
    Where does the signal come from: GNN, transformer, or fusion?

Layer 3 — LEARNING OUTCOME
    What did the model actually learn?
    Which classes can it distinguish and which can't it?
    Does the architecture match the vulnerability structure?
    What's the irreducible ceiling and why?
```

---

## Part 1 — Input representation evolution: what each model sees

### 1.1 Node feature vector across versions

| Feature | v4/v5 (12-dim) | v6/v7/v8 (11-dim) | Bug that changed it |
|---------|----------------|-------------------|---------------------|
| `type_id` | dim 0, `/12.0` | dim 0, `/12.0` | — |
| `visibility` | dim 1, raw int (0/1/2) | dim 1, float {0.0, 0.5, 1.0} | BUG-3: visibility=2 dominated embeddings |
| `uses_block_globals` | dim 2 (but SolidityVariableComposed not checked) | dim 2 (fixed) | BUG: `block.timestamp` was invisible (SolidityVariableComposed not in `state_variables_read`) |
| `view` | dim 3 | dim 3 | — |
| `payable` | dim 4 | dim 4 | — |
| `complexity` | dim 5, `log1p(count)/log1p(100)` raw (BUG-2) | dim 5, correct | BUG-2: raw count instead of log1p |
| `loc` | dim 6, raw lines (BUG-1) | dim 6, `log1p(lines)/log1p(1000)` | BUG-1: raw LOC dominated features |
| `return_ignored` | dim 7 (always 0 — lvalue bug) | dim 7 (fixed: lval.name not id(lval)) | BUG-M1: return_ignored always 0 in v5.x |
| `call_target_typed` | dim 8 | dim 8 | — |
| `has_loop` | dim 9 | dim 9 | — |
| `external_call_count` | dim 10, `log1p/log1p(20)` | dim 10, same | — |
| `in_unchecked` | dim 11 | **DROPPED** | BUG-L2: removed as unreliable |
| **CFG inheritance** | CFG nodes had zero dims 1,3,4,5,9 | CFG nodes inherit from parent FUNCTION | BUG-C3: CFG nodes were essentially feature-less |

**Key insight:** In v4 and v5.x, `return_ignored` was always 0 (the bug using `id(lval)` instead of `lval.name` caused every lvalue to look distinct, so nothing was flagged as re-used). `block.timestamp` was invisible because `SolidityVariableComposed` was not included in `state_variables_read`. CFG nodes had no meaningful features. The model was essentially working with 7-8 reliable dimensions in practice, not 12.

In v7+: all 11 dimensions are reliable. CFG nodes inherit function-level features. The model can actually distinguish visibility (public=0.0 vs private=1.0 vs internal=0.5), see timestamp usage, track return ignored, and count external calls.

### 1.2 Edge type coverage across versions

| Edge type | ID | v4/v5.0 | v5.2 | v7 | v8 | Purpose |
|-----------|----|---------|------|-----|-----|---------|
| CALLS | 0 | ✅ | ✅ | ✅ | ✅ | Contract→called function |
| READS | 1 | ✅ | ✅ | ✅ | ✅ | Node reads state variable |
| WRITES | 2 | ✅ | ✅ | ✅ | ✅ | Node writes state variable |
| EMITS | 3 | ✅ (but never fired) | ✅ (fixed) | ✅ | ✅ | Node emits event; BUG-H7 fixed v7 |
| INHERITS | 4 | ✅ (but never fired) | ✅ (fixed) | ✅ | ✅ | Contract inheritance; BUG-H8 fixed v7 |
| CONTAINS | 5 | ✅ | ✅ | ✅ | ✅ | Contract/function contains node |
| CONTROL_FLOW | 6 | ❌ (v5 only has 5 types) | ✅ (added v5.2) | ✅ | ✅ | CFG edges within function |
| REVERSE_CONTAINS | 7 | ❌ | ✅ (added v5.2) | ✅ | ✅ | Node→function→contract (upward) |
| CALL_ENTRY | 8 | ❌ | ❌ | ❌ | ✅ | Cross-function call entry (ICFG) |
| RETURN_TO | 9 | ❌ | ❌ | ❌ | ✅ | Cross-function return (ICFG) |
| DEF_USE | 10 | ❌ | ❌ | ❌ | ✅ | Intra-function data flow |

**Key insight:** v4/v5.0 had no CFG edges — the model couldn't see control flow within functions at all. v5.2 added CONTROL_FLOW(6) and REVERSE_CONTAINS(7). v7 is the first version where EMITS and INHERITS actually fire (parent contract nodes were missing before BUG-H8). v8 adds cross-function ICFG edges, but DEF_USE is intra-function only.

**What this means for each vulnerability class:**

| Class | What the model needs to see | Available since | Still missing |
|-------|----------------------------|-----------------|---------------|
| Reentrancy | CEI violation: Call before Write in same function | v5.2 (CF edges) | Cross-contract reentrancy via CALL_ENTRY in v8 (sparse) |
| IntegerUO | `has_loop`, `external_call_count`, no explicit overflow check | v7 (correct features) | Nothing structural; data-limited |
| GasException | Loop presence, complexity | v7 | Nothing critical |
| Timestamp | `uses_block_globals=1` | v7 (SolidityVariableComposed fix) | DEF_USE from timestamp to condition |
| DenialOfService | Loop + external call pattern | v5.2 | Detached from loss (dos_loss_weight=0); 372 positives |
| MishandledException | `return_ignored=1` on Send/Transfer calls | v7 (BUG-M1 fix) | Nothing critical |
| UnusedReturn | `return_ignored=1` | v7 (BUG-M1 fix) | Nothing critical |
| TOD | State write before external call | v5.2 (CF) | Ordering via DEF_USE (v8 intra-function) |
| ExternalBug | CALLS + READS/WRITES pattern | v7 (INHERITS/EMITS fix) | Cross-contract data flow |
| CallToUnknown | Dynamic call target | v7 (call_target_typed) | Nothing critical |

### 1.3 Training data quality per version

| Version | Rows | Labels | Leakage | Feature bugs | Effective quality |
|---------|------|--------|---------|--------------|-----------------|
| v4 | 44,420 | Noisy (no cleaner) | **34.9% cross-split** | BUG-1/2/3, BUG-M1, dim-12 | Very low |
| v5.0/v5.2 | 44,470 | Noisy | Cleaned | BUG-1/2/3, BUG-M1, dim-12 | Low |
| v6.0 | 44,470 | −972 Timestamp (partial) | Clean | BUG-1/2/3, BUG-M1 patched in-place | Medium-low |
| v7.0 | 41,576 | −17,722 via label_cleaner | Clean | All bugs fixed | High |
| v8.0-AB | 41,576 | Same as v7 | Clean | Same as v7 | High |

**The critical observation:** v4's F1=0.5422 is meaningless for comparison because 34.9% of its validation set was seen during training (cross-split contamination found in v5.1). v5.2's tuned 0.3373 was measured on the clean deduped split — that is the actual v5.2 performance on held-out data. v7/v8 at ~0.26 is on the same clean split but with 17,722 *fewer* noisy labels — the task is genuinely harder.

---

## Part 2 — Architecture evolution: how each model processes

### 2.1 GNN architecture timeline

| Component | v4 (pre-v5) | v5.0 | v5.2 | v7 | v8 |
|-----------|-------------|------|------|-----|-----|
| in_channels | 12 | 12 | 12 | 11 | 11 |
| hidden_dim | 128 | 128 | 128 | 256 | 256 |
| gnn_layers | 4 | 4 | 4 | **7** | 7 |
| edge_emb | None (raw) | Embedding(5,32) | Embedding(8,64) | Embedding(8,64) | **Embedding(11,64)** |
| Phase 1 | L1+L2: structural+CONTAINS | same | same | same | same |
| Phase 2 | L3: CONTROL_FLOW directed | same | L3+L3b+L3c: 3-hop CF | 3-hop CF | **3-hop CF+CE+RT+DU** |
| Phase 3 | L4: REVERSE_CONTAINS | same | L4+L5: RC (2 layers) | **L6+L7: RC (2 layers)** | same as v7 |
| JK aggregation | None (last layer) | None | **✅ _JKAttention** | ✅ | ✅ |
| Per-phase LayerNorm | None | None | ✅ | ✅ | ✅ |
| Separate LR groups | None | None | ✅ GNN×2.5 LoRA×0.5 | ✅ (lora_lr_mult=0.3) | ✅ |

### 2.2 Transformer and fusion timeline

| Component | v4 | v5.x | v7/v8 |
|-----------|-----|------|-------|
| CodeBERT LoRA | r=8, α=16, Q+V | r=8, α=16, Q+V | **r=16, α=32**, Q+V all 12 layers |
| Windowing | Single 512-token window | Single window | **MAX_WINDOWS=4, [4,512]** |
| Fusion attn_dim | 256 | 256 | 256 |
| Fusion output | 128 (LOCKED) | 128 | 128 |
| CrossAttn LayerNorm | None | None | **LayerNorm(768) on token input** (BUG-C2) |

### 2.3 Classifier timeline

| Component | v4 | v5.0 | v5.2+ |
|-----------|-----|------|-------|
| Structure | GNN→Linear(128,10) | **Three-eye**: GNN+TF+Fused→[384]→192→10 | same |
| Aux loss | None | ✅ per-class binary CE | ✅ |
| Loss function | BCE | BCE + aux | **ASL**(γ⁻=2.0, γ⁺=1.0) + aux (v7+) |
| Weighted sampler | None | None | **✅ "positive" mode** (3× for vuln rows) |
| DoS detach | None | None | **✅ dos_loss_weight=0.0** |
| Label smoothing | None | None | ✅ per-class class_label_smoothing |

---

## Part 3 — Why each change was made: the failure→fix chain

Each major architectural or data change was motivated by a specific observed failure. Tracing this chain explains what each version was designed to fix and what it actually fixed.

| Failure observed | Version | Root cause | Fix applied | Version fixed | Did it work? |
|-----------------|---------|-----------|------------|---------------|-------------|
| Reentrancy fires on every external call (0.97 prob on safe contracts) | v5.0/v5.2 | No CFG edges → model can't see CALL→WRITE ordering | Add CONTROL_FLOW(6) edges, conv3b/c for CEI | v5.2 | Partially — F1 improved but behavioral still fails |
| block.timestamp invisible (Timestamp F1=0.174) | v5.2 | SolidityVariableComposed not checked in `state_variables_read` | Fix graph_extractor.py | v7 | Yes — Timestamp F1 rose |
| return_ignored always 0 (MishandledException/UnusedReturn weak) | v5.2 | `id(lval)` instead of `lval.name` in return_ignored check | BUG-M1 fix | v7 | Yes — both classes improved |
| GNN collapsed to near-zero share after ep8 (v5.1) | v5.1 | No JK → gradient vanishes through 4 sequential layers | JK attention aggregation | v5.2 | Yes — GNN share stable 50-80% |
| All 12 eye losses identical (JK useless) | v5.1 first try | JK used `.detach()` before collecting live intermediates | Fix: collect without detach | v5.2 | Yes — each eye diverges |
| CFG nodes feature-less (dim 1,3,4,5,9 all zero) | v5.2 | CFG nodes not inheriting from parent FUNCTION | BUG-C3: inherit dims from parent | v7 | Yes — enables complexity/payable on CFG |
| EMITS/INHERITS never fire (zero edges) | v5.2 | Parent contract nodes not added to graph | BUG-H7/H8: add parent nodes explicitly | v7 | Yes — INHERITS now fires on inherited contracts |
| loc dominates (raw lines, range 1–2538) | v5.x | Raw value instead of log1p normalisation | BUG-1: `log1p(lines)/log1p(1000)` | v7 | Yes — feature balanced |
| Reentrancy pos_weight=2.82 → overfit | v5.2 | High positive class weight pushed extreme probs | pos_weight_min_samples=3000 cap | v7 | Partially — no pos_weight anymore with label cleaning |
| v6 collapsed (F1=0.1717) | v6 | Label cleaning broke the label distribution mid-pipeline; 12→11 dim mismatch; schema version gap | Full pipeline restart with schema v7 | v7 | Yes — v7 launches clean |
| DoS co-occurrence with Reentrancy (99%) | All | 372 positives, all also Reentrancy; model can't separate | dos_loss_weight=0.0 (detach) | v7 | Yes — DoS gradient removed; not learning it |
| ICFG edges absent → model can't follow cross-function calls | v7 | graph_extractor only built intra-function CFG | Add CALL_ENTRY(8)/RETURN_TO(9)/DEF_USE(10) | v8 | Unknown — this is what we're investigating |
| Phase 2 only trained on CF(6); other edge types now compete | v8 | Adding types 8,9,10 to same conv hops | PLAN-3A: isolate each type | after v8-AB | Not yet run |

---

## Part 4 — Evidence of change: what each version actually achieved

### 4.1 Per-class F1 trajectory across versions

*Note: v4 and v5.0/v5.2 numbers were measured on the leaky or partially leaky dataset. v7/v8 are on the clean deduped split with cleaned labels — these are not directly comparable to the earlier numbers. Use as directional evidence only.*

| Class | v4 (leaky) | v5.2-r3 (clean) | v7.0 (clean) | v8-AB (clean) | Key driver of change |
|-------|-----------|-----------------|--------------|--------------|----------------------|
| IntegerUO | 0.776 | 0.732 | 0.583 | 0.592 | −0.15 from v4→v7: label cleaning removed many easy positives |
| Reentrancy | 0.519 | 0.322 | ~0.24 | 0.261 | v5.2: Reentrancy overfit RC1+RC2 inflated; v7: correctly bounded |
| GasException | 0.507 | 0.407 | 0.299 | 0.299 | Plateau; needs complexity/loop features (now correct in v7) |
| DenialOfService | 0.384 | 0.329 | 0.019 | 0.019 | Detached from loss (dos_loss_weight=0.0); intentional |
| MishandledException | ~0.3 | ~0.2 | 0.276 | ~0.27 | BUG-M1 fix in v7 (return_ignored now works) |
| Timestamp | 0.174 | 0.174 | 0.227 | ~0.22 | block.timestamp fix in v7 (+0.05 gain) |
| CallToUnknown | ~0.28 | ~0.2 | ~0.18 | ~0.18 | INHERITS/EMITS fix helped; still weak |
| UnusedReturn | 0.238 | ~0.2 | ~0.15 | ~0.15 | return_ignored fix in v7 but low positive count |
| TOD | ~0.28 | ~0.2 | 0.228 | ~0.22 | CFG edges help; still ordering-dependent |
| ExternalBug | 0.262 | ~0.2 | ~0.15 | ~0.15 | INHERITS fix helped; still weak |
| **Macro avg** | **~0.34** | **0.3373** | **0.2651** | **0.2593** | Lower because cleaner labels = harder task |

**Reading this correctly:** The macro average *falling* from 0.5422 (v4) to 0.2651 (v7) does NOT mean the model got worse. It means:
1. Data leakage removed → no more "seen-at-training" shortcuts
2. 17,722 noisy labels removed → model must learn real signal, not noise patterns
3. Feature bugs fixed → model can no longer rely on systematically wrong features it had learned to exploit

The v7 model at 0.2651 on clean data is almost certainly better than the v4 model at 0.5422 on leaky data for real-world contracts.

### 4.2 JK attention weight evolution: what the model learned to trust

| Version | Phase 1 (structural) | Phase 2 (CFG/ICFG) | Phase 3 (reverse containment) | Interpretation |
|---------|---------------------|---------------------|-------------------------------|----------------|
| v7, early (ep1) | ~0.41 | ~0.26 | ~0.33 | Model starts evenly distributed |
| v7, ep11 | 0.05 | ~0.35 | ~0.60 | Phase 2 (CF) temporarily rises as model learns CEI |
| v7, final (ep33) | **0.050** | **0.182** | **0.768** | Phase 3 dominates; CF edges trusted only 18% |
| v8, early (ep1) | 0.185 | 0.329 | 0.486 | Phase 2 starts HIGHER — new edges attract weight |
| v8, ep3 | 0.084 | **0.362** | 0.554 | Phase 2 peaks: model explores new ICFG edges |
| v8, ep22 (best) | 0.072 | 0.289 | 0.639 | Phase 2 sliding, Phase 3 growing |
| v8, ep26 | 0.072 | 0.263 | 0.665 | Phase 2 at 0.263 — 45% more than v7 final |

**What this tells us:**
- v7 converged to heavy Phase 3 reliance (0.768): the REVERSE_CONTAINS hierarchy (node→function→contract) is the dominant signal. The model learned: "which contract/function *contains* this node" is more predictive than "what control flow edges exist."
- v8 Phase 2 starts higher (0.329 vs v7's ~0.26 at ep1) and stays higher (0.263 at ep26 vs v7's 0.182 final). The ICFG edges are genuinely attracting weight.
- But: Phase 2 in v8 is *declining* while Phase 3 *grows* — the model is learning to trust Phase 3 more as training continues, same as v7.
- The gap between v8 Phase 2 (0.263) and v7 Phase 2 (0.182) represents the marginal information in CALL_ENTRY/RETURN_TO/DEF_USE edges — real but not enough to break the ceiling.

### 4.3 Training dynamics comparison

| Metric | v7.0 ep1→33 | v8-AB ep1→26 | Interpretation |
|--------|-------------|-------------|----------------|
| Step loss final | ~0.135 | 0.142 | v8 slightly higher — still converging |
| Eye loss convergence epoch | ~ep8 | ~ep10 | v8 takes 2 extra epochs — more complex landscape |
| JK Phase 2 peak | 0.350 (ep11) | 0.362 (ep3) | v8 peaks earlier and higher — new edges novel early |
| JK Phase 2 final | 0.182 | 0.263 | v8 retains more Phase 2 weight at convergence |
| Fused grad spikes | ep ~5, ~14 | ep6, ep14, ep20 | Similar pattern — fusion is the breakthrough mechanism in both |
| F1 plateau onset | ep10 | ep10 | Identical — same structural ceiling |
| Best F1 | 0.2651 | 0.2593 | v8 is 0.006 below v7 |

---

## Part 5 — Live model analysis: what are the models computing?

This is the hands-on analysis run after killing v8-AB. Two categories: **inspecting saved state** (no GPU needed) and **inference-time analysis** (requires GPU, ~15 min each).

### 5.1 State inspection (no GPU required, run first)

#### 5.1.1 JK weights from checkpoint state dict

```python
import torch

v7 = torch.load("ml/checkpoints/v7.0_best.pt", weights_only=False)
v8 = torch.load("ml/checkpoints/v8.0-AB_best.pt", weights_only=False)

# JK last_weights is a buffer, lives in state_dict
v7_jk = v7["model_state_dict"].get("gnn.jk.last_weights")
v8_jk = v8["model_state_dict"].get("gnn.jk.last_weights")
print(f"v7 JK: {v7_jk}")  # should be ~[0.05, 0.18, 0.77]
print(f"v8 JK: {v8_jk}")  # should be ~[0.07, 0.29, 0.64]
```

Cross-check against logged values. If they differ significantly, it means the JK `last_weights` buffer was not saved in the same state as the final epoch (it's a running buffer, updated during forward passes — the checkpoint saves the last forward pass's weights, not the epoch-average).

#### 5.1.2 Edge embedding weight inspection

```python
# The edge embedding table learns which edge types are discriminative
v7_emb = v7["model_state_dict"]["gnn.edge_embedding.weight"]  # [8, 64]
v8_emb = v8["model_state_dict"]["gnn.edge_embedding.weight"]  # [11, 64]

# Compare L2 norm of each edge type's embedding row
for i, name in enumerate(["CALLS","READS","WRITES","EMITS","INHERITS","CONTAINS","CF","RC"]):
    v7_norm = v7_emb[i].norm().item()
    v8_norm = v8_emb[i].norm().item() if i < 11 else None
    print(f"  {name}[{i}]: v7={v7_norm:.3f}  v8={v8_norm:.3f}")

for i, name in enumerate(["CALL_ENTRY","RETURN_TO","DEF_USE"], start=8):
    v8_norm = v8_emb[i].norm().item()
    print(f"  {name}[{i}]: v8={v8_norm:.3f} (new in v8)")
```

**What to look for:** If CALL_ENTRY(8)/RETURN_TO(9)/DEF_USE(10) have significantly lower norm than the other edge types, the model has effectively learned to downweight them — confirming H1 (dilution). If they have comparable norm, the model is using them, but they may still be hurting if they interfere with the CF(6) signal.

#### 5.1.3 Layer weight norm comparison

```python
# Per-layer GATConv lin_src/lin_dst/att weight norms
# Compare v7 vs v8 for layers conv3/conv3b/conv3c (Phase 2 hops)
for name in ["gnn.conv3.lin_src.weight", "gnn.conv3b.lin_src.weight", "gnn.conv3c.lin_src.weight"]:
    v7w = v7["model_state_dict"].get(name)
    v8w = v8["model_state_dict"].get(name)
    if v7w is not None: print(f"v7 {name}: norm={v7w.norm():.3f}")
    if v8w is not None: print(f"v8 {name}: norm={v8w.norm():.3f}")
```

---

### 5.2 Inference-time analysis (requires GPU)

All run via `ml/scripts/compare_checkpoints.py` (to be implemented). The script:
- Reuses `load_model_from_checkpoint()` and `build_val_loader()` from `tune_threshold.py`
- Registers forward hooks to collect internal states
- Runs on 500 test-split samples (fast, representative)

#### 5.2.1 Per-class F1 / precision / recall / AUC-ROC

The baseline comparison. Side-by-side table showing where v8 wins vs loses per class. AUC-ROC is threshold-independent — if v8 has better AUC but worse F1 at 0.5, calibration (H4) is the problem.

**Run:**
```bash
TRANSFORMERS_OFFLINE=1 PYTHONPATH=. python ml/scripts/compare_checkpoints.py \
    --ckpt-a ml/checkpoints/v7.0_best.pt  --label-a "v7.0" \
    --ckpt-b ml/checkpoints/v8.0-AB_best.pt --label-b "v8-AB" \
    --cache ml/data/cached_dataset_v8.pkl \
    --splits-dir ml/data/splits/deduped --split test \
    --mode all --out ml/logs/comparison_v7_v8AB_full.json
```

**Important:** v7 uses `edge_emb(8,64)` — will crash on v8 cache graphs which contain edge type IDs 8/9/10. The script must clamp `graph.edge_attr` to `[0,7]` when running v7. Detect from the checkpoint config: `ckpt["config"]["num_edge_types"]`.

#### 5.2.2 Feature importance via gradient (GradCAM-style)

For each vulnerability class, identify which node feature dimensions have the largest gradient × activation signal. This tells us what the model *uses* from the 11 features.

**Method:**
```python
# For each test sample:
graph.x.requires_grad_(True)
logits = model(graph, tokens)                    # forward
logits[0, class_idx].backward()                  # backward from class logit
importance = (graph.x.grad * graph.x).abs()     # GradCAM: grad × activation
# Average across nodes, accumulate across samples
# Result: [11] importance vector per class, for each model
```

**Expected output:** A 10×11 matrix per model (class × feature dim). 

**What to look for:**
- **Reentrancy**: Does v7 weight `external_call_count` (dim 10) and `type_id` (CALL nodes, dim 0)? Does v8 weight the same or shift to something else?
- **Timestamp**: Does v7 weight `uses_block_globals` (dim 2) more than v8? Or equally?
- **MishandledException/UnusedReturn**: Both depend on `return_ignored` (dim 7). Do both models use it?
- **IntegerUO**: Primarily `has_loop` (dim 9) and `external_call_count` (dim 10). Do both agree?

#### 5.2.3 Edge type activation rates (v8 only)

For v8, measure how often each of the 11 edge types appears in the test set graphs, and what fraction of total edges they represent. This is not a model analysis — it's a data analysis — but it directly tests H2.

```python
from collections import Counter
edge_type_counts = Counter()
total_graphs = 0
for batch in test_loader:
    for graph in batch.to_data_list():
        edge_type_counts.update(graph.edge_attr.tolist())
        total_graphs += 1

total_edges = sum(edge_type_counts.values())
for type_id, name in enumerate(EDGE_TYPES):
    count = edge_type_counts[type_id]
    print(f"  {name}({type_id}): {count:,} edges ({100*count/total_edges:.1f}%)")
```

**Critical threshold:** If CALL_ENTRY(8) + RETURN_TO(9) + DEF_USE(10) together represent less than 15% of all edges in the test graphs, the Phase 2 convolutions (3 hops, all edge types combined) are dominated by CF(6) signal, and the new types are "passengers." If they're above 20%, they're substantial contributors.

#### 5.2.4 Attention weight extraction per phase

Register forward hooks on Phase 2 GATConv layers to collect the attention coefficients (alpha) during the forward pass. Average across test samples to see which edge types receive high attention in Phase 2.

```python
phase2_alpha = []

def hook_fn(module, input, output):
    # GATConv output = (node_embs, attention_weights) when return_attention_weights=True
    # But hooks on GATConv receive output = node_embs only
    # Alternative: patch forward to log internally
    pass

# Better approach: temporarily modify GATConv.forward to log alpha
# Or: use torch.fx tracing
```

**Simpler alternative:** Instead of attention weights, compare the edge-type-conditional gradient flowing through Phase 2 — compute gradient of loss w.r.t. each edge embedding row. Higher gradient = more learning signal flowing through that edge type.

```python
model.gnn.edge_embedding.weight.requires_grad_(True)
loss = compute_batch_loss(model, batch)
loss.backward()
# grad shape: [11, 64]
# L2 norm of grad per row = how much each edge type contributed
per_type_grad = model.gnn.edge_embedding.weight.grad.norm(dim=1)
```

#### 5.2.5 Prediction disagreement analysis

The most diagnostic analysis: find contracts where v7 is correct and v8 is wrong (and vice versa), then characterize what those contracts have in common.

**Structure of disagreement buckets:**
```
Both right     → not diagnostic for the gap
v7 right only  → v8 is regressing on these; what do they share? (H1, H6)
v8 right only  → v8 improves on these; what do they share? (H2)
Both wrong     → irreducible ceiling (H3)
```

For the "v7 right only" bucket: check their edge type distribution. If they have high CF(6) density and low CALL_ENTRY(8)/DEF_USE(10) density, v8's Phase 2 is being distracted by the new types at the expense of CF pattern recognition.

For the "v8 right only" bucket: check if they have meaningful CALL_ENTRY/RETURN_TO edges. If yes, the cross-function ICFG edges helped specifically on these contracts.

#### 5.2.6 Probability calibration curves

For each class, bin the predicted probabilities into 10 buckets (0–0.1, 0.1–0.2, …) and compute the empirical positive rate in each bucket.

A well-calibrated model: bucket [0.7–0.8] should have ~75% positive rate.  
An overconfident model: predicted probabilities are too high for actual positive rate.  
An underconfident model: predicted probabilities are too low → threshold should be lower than 0.5.

**Tells us (H4):** If v8 is systematically underconfident (predicts 0.4 for true positives that v7 predicts 0.6 for), then threshold tuning will largely close the F1 gap. If the calibration curves overlap and both are reasonable → calibration is not the issue.

#### 5.2.7 Behavioral test on canonical contracts

```bash
# v7
TRANSFORMERS_OFFLINE=1 PYTHONPATH=. python ml/scripts/manual_test.py \
    --checkpoint ml/checkpoints/v7.0_best.pt \
    --contracts ml/scripts/test_contracts/

# v8-AB
TRANSFORMERS_OFFLINE=1 PYTHONPATH=. python ml/scripts/manual_test.py \
    --checkpoint ml/checkpoints/v8.0-AB_best.pt \
    --contracts ml/scripts/test_contracts/
```

**Specific contracts to write (if not already in test_contracts/):**
```
cross_function_reentrancy.sol  — Reentrancy via two separate functions (fallback calls withdraw())
                                  v7 might miss this; v8's CALL_ENTRY edges could help
reentrancy_classic.sol         — Simple CEI violation in one function (both should detect)
tod_simple.sol                 — State write before external call (CF edges should help both)
timestamp_check.sol            — block.timestamp in require() (uses_block_globals=1)
safe_complex.sol               — Checks-effects-interactions pattern (should be safe)
```

---

## Part 6 — Synthesis and decision matrix

After all analyses, use this framework:

### 6.1 Hypothesis resolution

| Hypothesis | Test that resolves it | Resolution signal |
|-----------|----------------------|-------------------|
| H1: Phase 2 hop dilution | Edge embedding grad norms; "v7 right only" bucket | CE/RT/DU grads low + "v7 right" contracts have high CF density |
| H2: DEF_USE intra-function only | Edge type activation rates; "v8 right only" bucket | CE/RT/DU <15% of edges; "v8 right" not cross-function |
| H3: Label ceiling | Both-wrong bucket analysis | Both-wrong bucket is large (>40% of errors) and contracts are ambiguous |
| H4: Calibration shift | AUC-ROC comparison; probability calibration curves | v8 AUC ≥ v7 AUC but F1 worse → calibration |
| H5: Class-specific tradeoff | Per-class F1 table | v8 wins on some classes, loses on others |
| H6: Edge type spread | Phase 2 attention/grad; edge embedding norms | CE/RT/DU norm ≈ 0 but CF lower than v7 |

### 6.2 Decision table

| Observation | Next action |
|-------------|-------------|
| H4 confirmed: v8 tuned F1 ≥ v7 tuned F1 | v8-AB is the better model; tune thresholds and proceed to ZKML |
| H1 confirmed: CE/RT/DU dilute CF(6) | PLAN-3A (CF+CE+RT only) to see if removing DU recovers CEI pattern |
| H2 confirmed: CE/RT/DU too sparse (<15%) | PLAN-1D: true cross-function CFG extraction before next training |
| H3 confirmed: large both-wrong bucket | Label re-audit before further training; check SmartBugs labels for the stuck classes |
| H5 confirmed: class tradeoff | Per-class weighted training: different phase2_edge_types per class (complex) |
| H6 confirmed: CE/RT/DU near-zero norm | v7 architecture is correct; v8 schema change has no effect; stay with v7 config |
| None confirmed cleanly | Full ablation matrix: 3A + 3B + v7-config-on-v8-data (3C) to triangulate |

---

## Implementation: what scripts to build

| Script | Purpose | Reuses | Est. time to write |
|--------|---------|--------|-------------------|
| `ml/scripts/compare_checkpoints.py` | Master comparison driver | `load_model_from_checkpoint`, `build_val_loader` from tune_threshold.py | 3–4 hours |
| `ml/scripts/inspect_checkpoint.py` | State dict analysis without GPU | torch.load only | 30 min |
| `ml/scripts/edge_activation.py` | Edge type frequency in test split | Dataset loaders | 30 min |

`compare_checkpoints.py` should support `--mode` flags:

| Mode | Produces | GPU needed |
|------|---------|-----------|
| `metrics` | Per-class F1/precision/recall/AUC, macro avgs | Yes |
| `calibration` | Reliability curves per class per model | Yes |
| `overlap` | Both-right/v7-only/v8-only/both-wrong counts | Yes |
| `errors` | Sample indices + metadata for disagreement buckets | Yes |
| `importance` | GradCAM feature importance per class | Yes |
| `edge-grad` | Gradient norm per edge type embedding row | Yes |
| `all` | Everything above | Yes |

`inspect_checkpoint.py` runs without GPU:

| Output | Method |
|--------|--------|
| JK last_weights buffer | state_dict lookup |
| Edge embedding norms | state_dict weight.norm(dim=1) per row |
| Config comparison (v7 vs v8 architecture params) | ckpt["config"] diff |
| Parameter count per component | sum of param tensors by prefix |

---

## Execution order and time estimates

| Step | What | Time | Insight unlocked |
|------|------|------|-----------------|
| 0 | Kill v8-AB, verify checkpoints | 5 min | — |
| 1 | `inspect_checkpoint.py` | 10 min | JK weights, edge embedding norms, config diff |
| 2 | `edge_activation.py` | 15 min | H2: are ICFG edges sparse? |
| 3 | `compare_checkpoints.py --mode metrics` | 20 min | H4, H5: per-class F1 + AUC |
| 4 | `tune_threshold.py` × 2 + re-run metrics | 50 min | H4: does calibration close the gap? |
| 5 | `compare_checkpoints.py --mode importance` | 30 min | Feature importance per class |
| 6 | `compare_checkpoints.py --mode errors` | 20 min | H1, H2, H3: disagreement characterisation |
| 7 | `compare_checkpoints.py --mode edge-grad` | 20 min | H1, H6: CE/RT/DU gradient flow |
| 8 | `manual_test.py` × 2 | 15 min | H2: cross-function reentrancy detection |
| 9 | Synthesise findings + write results doc | 60 min | Full diagnosis |
| **Total** | | **~4 hours** | |

**Output:** `docs/ml/v8-vs-v7-comparison-results.md` — the findings document written after running all steps. That document drives the decision on what v8-C (the next training configuration) should be.

---

## Files produced

| File | Content |
|------|---------|
| `ml/scripts/compare_checkpoints.py` | Master GPU comparison script |
| `ml/scripts/inspect_checkpoint.py` | CPU-only checkpoint inspection |
| `ml/scripts/edge_activation.py` | Edge type frequency counter |
| `ml/logs/inspect_v7_v8.txt` | JK weights, embedding norms, config diff |
| `ml/logs/edge_activation_v8.json` | Edge type counts in test split |
| `ml/logs/comparison_v7_v8AB_full.json` | All comparison metrics |
| `ml/logs/comparison_v7_v8AB_tuned.json` | Post-threshold-tuning metrics |
| `docs/ml/v8-vs-v7-comparison-results.md` | Synthesised findings (written after execution) |
