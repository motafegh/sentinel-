The #1 Problem: torch.compile is Doing NOTHING
From your training logs, the two critical errors tell the whole story:

text

GuardOnDataDependentSymNode: Eq(8*u0, 0)  
  → _to_dense_batch.py line 127: out.repeat(size)
  
torch._dynamo hit config.cache_size_limit (8)
What's happening: torch.compile(model, dynamic=True) tries to compile, but to_dense_batch() in fusion_layer.py:168 produces data-dependent shapes — the max_nodes dimension varies per batch (740 nodes, 827 nodes, etc.). This causes Dynamo to recompile for every unique shape, hitting the cache limit after ~8 batches. After that, every single forward pass runs in pure eager mode — you're getting zero benefit from torch.compile.

The 20-40% speedup that --compile promises? You're getting 0%.

OPTIMIZATION PLAN (Ranked by Impact)
🔴 TIER 1 — CRITICAL (Expected: 30-50% total speedup)
1. Fix to_dense_batch to make torch.compile actually work

The root cause is fusion_layer.py:168:

python

padded_nodes, node_real_mask = to_dense_batch(nodes_proj, batch)
This creates a [B, max_nodes, 256] tensor where max_nodes changes every batch. Dynamo can't handle this.

Fix A — Pad to a FIXED max_nodes (recommended):

python

# In fusion_layer.py forward(), replace:
padded_nodes, node_real_mask = to_dense_batch(nodes_proj, batch)

# With:
MAX_NODES_PAD = 200  # 95th percentile of your graph sizes; tune this
padded_nodes, node_real_mask = to_dense_batch(
    nodes_proj, batch, max_num_nodes=MAX_NODES_PAD
)
This forces every batch to have the same [B, 200, 256] shape, making Dynamo happy. You'll waste some memory on padding but gain compilation. The zeroing-out step (enriched_nodes * node_real_mask) already handles the padded positions correctly.

Fix B — Increase Dynamo cache limit (quick band-aid):

python

# In trainer.py train(), BEFORE torch.compile():
import torch._dynamo
torch._dynamo.config.cache_size_limit = 256
torch._dynamo.config.accumulated_cache_size_limit = 256
This lets Dynamo cache more unique shapes. It's not as good as Fix A (still recompiles per unique shape), but it prevents falling back to eager mode entirely.

Best: Apply BOTH fixes. Fix A gives you static shapes for the fusion layer, and Fix B gives Dynamo room for the GNN's minor shape variations.

2. Compile submodules separately instead of the whole model

CodeBERT (125M params, frozen) doesn't benefit from compilation — it's a giant HuggingFace model with complex control flow. Compiling it wastes compile time and memory. Instead:

python

# In trainer.py train(), replace:
if config.use_compile:
    model = torch.compile(model, dynamic=True)

# With:
if config.use_compile:
    # Only compile the parts that benefit
    model.gnn = torch.compile(model.gnn, dynamic=True)
    model.fusion = torch.compile(model.fusion, dynamic=True)
    model.classifier = torch.compile(model.classifier)
    model.gnn_eye_proj = torch.compile(model.gnn_eye_proj)
    model.transformer_eye_proj = torch.compile(model.transformer_eye_proj)
    model.window_pooler = torch.compile(model.window_pooler)
    # Do NOT compile model.transformer — CodeBERT+LoRA is too complex for Dynamo
This gives you the benefit on the GNN (7-layer GAT, heavy compute) and fusion (MHA operations), while skipping the HuggingFace model that Dynamo struggles with.

3. Install torch-scatter for PyG acceleration

Your logs show:

text

UserWarning: torch-scatter not installed
PyG's global_max_pool, global_mean_pool, and scatter operations fall back to slow pure-PyTorch implementations without torch-scatter. On CUDA, the custom CUDA kernels in torch-scatter are 2-5× faster for scatter/gather operations that happen every forward pass (pooling, edge masking, etc.).

bash

pip install torch-scatter -f https://data.pyg.org/whl/torch-2.x.x+cu121.html
# Match your CUDA version
🟡 TIER 2 — HIGH IMPACT (Expected: 15-25% additional speedup)
4. Increase num_workers to 4 and prefetch_factor to 4

Current: num_workers=2, prefetch_factor=2. Your dataset uses a shared RAM cache (2.28 GB, fork CoW), so workers are essentially free in memory. With 4 workers and prefetch 4, the GPU won't stall waiting for batches:

python

# In TrainConfig:
num_workers: int = 4   # was 2

# In trainer.py DataLoader kwargs:
prefetch_factor=4       # was 2
Given you have 3638 batches/epoch at ~2.2 batch/s, each batch takes ~450ms. If GPU compute is ~300ms and data loading is ~150ms, 4 workers should fully overlap.

5. Use torch.amp.autocast with dtype=torch.bfloat16 instead of float16

RTX 3070 (Ampere) supports BF16 natively. BF16 has the same exponent range as FP32 (8 bits) so it never overflows/underflows like FP16 (5-bit exponent). This means:

No GradScaler needed — eliminates the scale/unscale/step/update overhead
Simpler training loop — fewer CUDA synchronization points
No FP16→FP32 casting overhead in loss computation
python

# In trainer.py train():
# Replace:
use_amp: bool = True  # uses float16

# With using bf16:
# In train_one_epoch and evaluate:
with torch.amp.autocast(device, dtype=torch.bfloat16, enabled=use_amp):
    ...

# And remove GradScaler entirely:
# scaler.scale(loss).backward()  → loss.backward()
# scaler.unscale_(optimizer)     → (remove)
# scaler.step(optimizer)         → optimizer.step()
# scaler.update()                → (remove)
BF16 on Ampere GPUs is hardware-accelerated and doesn't need loss scaling. This removes ~4 CUDA synchronizations per optimizer step.

6. Enable CUDA graphs for the compiled GNN subgraph

Once torch.compile is working (after Fix #1), you can go further with CUDA graphs for the GNN path, which eliminates Python overhead and kernel launch latency:

python

# After torch.compile on model.gnn:
if config.use_compile and device == "cuda":
    model.gnn = torch.cuda.make_graphed_callables(
        model.gnn, sample_inputs=(sample_x, sample_edge_index, sample_batch, sample_edge_attr)
    )
CUDA graphs record the entire execution trace and replay it. This is particularly effective for the GNN where the same 7 GAT layers run every forward pass.

🟢 TIER 3 — MEDIUM IMPACT (Expected: 5-15% additional speedup)
7. Gradient checkpointing to enable larger batch size

Currently batch_size=8 fits ~6.9/8.0 GB. Gradient checkpointing trades compute for memory, allowing you to fit batch_size=16 or even 32:

python

# In sentinel_model.py __init__():
from torch.utils.checkpoint import checkpoint

# In forward(), replace:
node_embs, batch = self.gnn(graphs.x, graphs.edge_index, graphs.batch, edge_attr)
# With:
node_embs, batch = checkpoint(self.gnn, graphs.x, graphs.edge_index, graphs.batch, edge_attr, use_reentrant=False)

# Same for transformer:
token_embs = checkpoint(self.transformer, input_ids, attention_mask, use_reentrant=False)
With batch_size=16, you cut gradient_accumulation_steps from 8 to 4 (same effective batch=64), reducing optimizer steps by 2×. Optimizer steps are expensive (AdamW requires 2 passes over all parameters).

8. Optimize the GNN edge masking — precompute on CPU/worker

In gnn_encoder.py, every forward pass computes:

python

struct_mask = edge_attr <= _CONTAINS       # line 379
cfg_mask    = edge_attr == _CONTROL_FLOW   # line 380
contains_mask = edge_attr == _CONTAINS     # line 381
These create 3 boolean masks every forward pass. Since edge_attr values are integers 0-7, you can precompute these masks in the dataset/collate function and store them as graph attributes:

python

# In dual_path_collate_fn, before Batch.from_data_list():
for g in graphs:
    ea = g.edge_attr
    g.struct_mask = (ea <= 5)
    g.cfg_mask = (ea == 6)
    g.contains_mask = (ea == 5)

# Then in GNNEncoder, read from the batched graph instead of computing:
struct_mask = getattr(graphs, 'struct_mask', edge_attr <= _CONTAINS)
This moves work from GPU forward pass to CPU data loading (workers).

9. Avoid .clone() for DoS masking — use detach() strategically

Current code in train_one_epoch:

python

_logits_for_loss = logits.clone()          # copies ENTIRE [B, 10] tensor
_logits_for_loss[:, _dos_idx] = logits[:, _dos_idx].detach()
Replace with:

python

# Create a view that detaches only the DoS column — no clone needed
_logit_parts = list(logits.split(1, dim=1))  # list of [B,1] tensors
_logit_parts[_dos_idx] = _logit_parts[_dos_idx].detach()
_logits_for_loss = torch.cat(_logit_parts, dim=1)
Or even better, use a mask tensor that's created once:

python

# In train() setup (once):
_dos_mask = torch.ones(NUM_CLASSES, device=device)
_dos_mask[_dos_idx] = 0.0

# In train_one_epoch:
_logits_for_loss = logits * _dos_mask + logits[:, _dos_idx].detach().unsqueeze(1) * (1 - _dos_mask).unsqueeze(0)
Same for aux masking — currently clones all 3 aux tensors per batch.

10. Pre-compute node_type_ids and func_mask in collate

In sentinel_model.py:267-268:

python

node_type_ids = (graphs.x[:, 0].float() * _MAX_TYPE_ID).round().long()
func_mask = torch.isin(node_type_ids, _FUNC_IDS_CPU.to(node_embs.device))
This runs every forward pass. The node_type_ids don't change — they're determined by the graph. Pre-compute them in the collate function and store as a graph attribute:

python

# In collate:
for g in graphs:
    g.node_type_ids = (g.x[:, 0].float() * _MAX_TYPE_ID).round().long()
    g.func_mask = torch.isin(g.node_type_ids, _FUNC_IDS_CPU)

# In sentinel_model.py forward():
node_type_ids = graphs.node_type_ids
func_mask = graphs.func_mask
11. Fuse the Phase 1+2+3 edge index slicing into a single operation

In gnn_encoder.py, three separate boolean index operations:

python

struct_ei = edge_index[:, struct_mask]
cfg_ei    = edge_index[:, cfg_mask]
rev_contains_ei = edge_index[:, contains_mask].flip(0)
Each creates a new tensor. You can pre-sort edges by type at extraction time so edge_index is already partitioned:

text

[struct_edges | cfg_edges | contains_edges]
Then you just need offset+length, not boolean masking.

12. Remove mid-epoch VRAM cleanup overhead

The gc.collect() + torch.cuda.empty_cache() every log_interval steps is expensive — empty_cache() forces a CUDA synchronization. Once you fix torch.compile (Fix #1), VRAM usage should be more stable. Remove the mid-epoch cleanup and only do it between epochs:

python

# Remove the VRAM check inside train_one_epoch's logging block
# Keep only the between-epoch cleanup in train()
Summary: Expected Speedup Stack
Fix
What It Does
Expected Speedup
Difficulty
1A	to_dense_batch(max_num_nodes=200) — static shapes	20-30%	Easy (1 line)
1B	dynamo cache_size_limit=256	5-10%	Easy (2 lines)
2	Compile submodules, not whole model	10-15%	Medium
3	Install torch-scatter	5-10%	Easy (pip install)
4	num_workers=4, prefetch_factor=4	5-10%	Easy (config change)
5	BF16 autocast, remove GradScaler	5-10%	Medium
6	CUDA graphs for GNN	5-8%	Medium
7	Gradient checkpointing → batch=16	10-20%	Medium
8	Precompute edge masks in collate	2-5%	Easy
9	Avoid .clone() for DoS masking	1-3%	Easy
10	Precompute func_mask in collate	1-2%	Easy
11	Pre-sort edges by type	2-4%	Hard
12	Remove mid-epoch VRAM cleanup	1-3%	Easy

Total estimated improvement: 2-3× faster training (from ~2.2 batch/s to ~5-6 batch/s, from ~25 min/epoch to ~10-12 min/epoch).