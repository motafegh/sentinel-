# Phase 5 — `models/` Teaching Roadmap

**Files:** `gnn_encoder.py`, `transformer_encoder.py`, `fusion_layer.py`, `sentinel_model.py`
**Total lines:** 1,769
**Sessions planned:** 8–13 (6 sessions, 7 chunks)
**Status:** Not started

---

## The Big Picture

Sentinel is a **dual-path + fusion** architecture. Every contract is processed through two independent
encoders, then fused:

```
Smart contract .sol file
        │
        ├──── Graph (PyG .pt) ──────► GNNEncoder          ──► node_embs [N, 256]
        │                              (3 phases, 8 layers,         │
        │                               GAT, edge masking)          │
        │                                                            │
        └──── Tokens (BERT .pt) ──► TransformerEncoder  ──► token_embs [B, 512, 768]
                                    (GraphCodeBERT frozen,           │
                                     LoRA adapters,                  │
                                     GNN prefix injection ◄──────────┘
                                     optional)
                                                                     │
                                              CrossAttentionFusion ◄─┘
                                              (node↔token cross-attn)
                                                     │
                                               [B, 128] fused_eye
                                                     │
                                    ┌────────────────┼────────────────┐
                               gnn_eye           fused_eye     transformer_eye
                              [B, 128]           [B, 128]         [B, 128]
                                    └────────────────┼────────────────┘
                                                concat [B, 384]
                                          Linear(384→192→10) → logits [B, 10]
```

**Three eyes, not two.** SentinelModel doesn't just fuse — it preserves three independent signals:
- **GNN eye** — structural opinion (pooled over function-declaration nodes only)
- **Transformer eye** — semantic opinion (pooled CLS token)
- **Fused eye** — joint structural+semantic opinion (CrossAttentionFusion output)

All three are concatenated and classified together. Each has its own auxiliary loss during training
to prevent any one eye from going dormant.

---

## File Summary

| File | Lines | Class(es) | Role |
|------|-------|-----------|------|
| `gnn_encoder.py` | 581 | `_JKAttention`, `GNNEncoder` | Contract graph → structural node embeddings `[N, 256]` |
| `transformer_encoder.py` | 350 | `TransformerEncoder`, `WindowAttentionPooler` | Source tokens → semantic embeddings `[B, 512, 768]` |
| `fusion_layer.py` | 277 | `CrossAttentionFusion`, `_scatter_to_dense` | Bidirectional node↔token cross-attention → `[B, 128]` |
| `sentinel_model.py` | 561 | `SentinelModel` | Three-eye orchestrator + classifier → logits `[B, 10]` |

---

## Session Plan

Teaching order follows the dependency graph: sub-modules before orchestrator.

### Session 8 — `gnn_encoder.py` Chunk 1
**Lines:** ~1–337 (`_JKAttention` + `GNNEncoder.__init__`)
**Covers:**
- `_JKAttention`: learned attention aggregation over K phase outputs, entropy tracking,
  diagnostics buffers (`last_weights`, `last_node_weights`)
- `GNNEncoder.__init__`: 8 conv layers by name and purpose, edge type embedding,
  IMP-G2 input_proj skip connection, per-phase LayerNorm, JK aggregator init
- Phase map: Phase 1 (conv1+conv2), Phase 2 (conv3+conv3b+conv3c), Phase 3 (conv4+conv4b+conv4c)
- Why `heads=8` for Phase 1 only; why `concat=False` for Phases 2+3
- Why `add_self_loops=True` Phase 1 / `add_self_loops=False` Phase 2+3

**New concepts introduced:** GAT (Graph Attention Network), multi-head attention in GNNs,
JK (Jumping Knowledge) connections, `nn.Embedding` for edge types

---

### Session 9 — `gnn_encoder.py` Chunk 2
**Lines:** ~338–582 (`GNNEncoder.forward`)
**Covers:**
- Input guards: feature dim check, `use_edge_attr` + `edge_attr=None` guard, OOB index check
- Edge mask computation: `struct_mask`, `cfg_mask`, `contains_mask`, per-layer Phase 2 subsets
- Phase 1 forward: IMP-G2 skip (`input_proj`), residual after Layer 2, LayerNorm
- Phase 2 forward: IMP-G1 distinct subsets per layer (CF-only, ICFG-only, joint)
- Phase 3 forward: `fwd_contains_ei` / `rev_contains_ei` flip, type-7 embeddings for up direction,
  type-5 embeddings for down direction (IMP-G3), LayerNorm
- JK aggregation: `_live` list vs `_intermediates` (detached), entropy term `_jk_entropy`
- `return_intermediates` diagnostic mode

**New concepts introduced:** Directed edge masking at runtime, residual connections in GNNs,
over-smoothing and why JK fixes it, `tensor.flip(0)` for reversing edges

---

### Session 10 — `transformer_encoder.py` (single chunk)
**Lines:** 1–350
**Covers:**
- Module-level peft hard-requirement check (raise vs warn rationale)
- `TransformerEncoder.__init__`: Flash Attention 2 → SDPA fallback, global dtype pollution fix,
  `get_peft_model()` mechanism — freezes backbone, injects LoRA A/B matrices
- Why all 512 tokens returned (not CLS only) — cross-attention needs token-level granularity
- Standard path (no prefix): single-window `[B, L]` and multi-window `[B, W, L]` → `[B, W*L, 768]`
- Prefix injection path: `inputs_embeds` construction, position IDs design, IMP-M3 count masking,
  `output_attentions=True` prefix diagnostic
- Multi-window with prefix: flattened batch dim, prefix expanded across windows
- `WindowAttentionPooler`: CLS extraction at `i*window_size + prefix_k`, learned attention over W CLS tokens

**New concepts introduced:** LoRA (Low-Rank Adaptation), parameter freezing + gradient split,
Flash Attention 2 / SDPA, `inputs_embeds` vs `input_ids`, RoBERTa position IDs

---

### Session 11 — `fusion_layer.py` (single chunk)
**Lines:** 1–278
**Covers:**
- Why concat+MLP was replaced: pooled summaries lose node/token detail before fusion
- `_scatter_to_dense`: why `to_dense_batch` caused `GuardOnDataDependentSymNode` compile break,
  static `max_nodes` fix, truncation warning (C-4)
- `CrossAttentionFusion.__init__`: projection layers, `token_norm` (BUG-C2), two MHA modules
- Forward: Step-by-step — project, pad nodes, Node→Token cross-attn, zero-out padding (Fix #8),
  Token→Node cross-attn, masked mean pool both sides, concat+project
- Mask convention: MHA `key_padding_mask` uses `True=ignore` (inverse of attention_mask)
- Fix #26: `need_weights=False` — skips 12.6 MB attention weight materialization

**New concepts introduced:** Cross-attention vs self-attention, padded batching in PyG graphs,
masked mean pooling, `torch.compile` graph breaks and how to avoid them

---

### Session 12 — `sentinel_model.py` Chunk 1
**Lines:** ~1–332 (module constants + `__init__` + `select_prefix_nodes`)
**Covers:**
- Module-level constants: `_MAX_TYPE_ID` (dynamic computation — same issue as A3/A20),
  `_FUNC_TYPE_IDS` frozenset, `_FUNC_IDS_CPU` pre-built tensor (BUG-L1 perf fix),
  `_PREFIX_NODE_PRIORITY`, `_PREFIX_TYPE_IDX`
- `SentinelModel.__init__`: sub-module wiring, GNN prefix injection modules (`gnn_to_bert_proj`,
  `prefix_type_embedding`), three eye projections, classifier, auxiliary heads
- `select_prefix_nodes()`: Python loop over graphs, priority sort (type priority + ext_call_count
  secondary for FUNCTION nodes), zero-padding, IMP-M3 count tracking
- Warmup suppression: `_current_epoch` field, `gnn_prefix_warmup_epochs`

**New concepts introduced:** Auxiliary losses and eye independence, GNN prefix injection rationale,
warmup suppression design (train GNN first, then align projection), `_current_epoch` training hook

---

### Session 13 — `sentinel_model.py` Chunk 2
**Lines:** ~334–562 (`forward` + `compute_prefix_attention_mean` + `parameter_summary`)
**Covers:**
- `forward()`: multi-window mask flatten, GNN path (`use_edge_attr` guard, `getattr` fallback),
  `node_type_ids` decode from `feature[0]` (multiply by `_MAX_TYPE_ID`, round, cast to long)
- Empty batch guard (`batch.numel() == 0`) — returns zero logits without crashing
- `graph_has_func` computation, ghost graph fix (BUG-H2): zero GNN embedding for no-function graphs
  (old: fallback pooled STATE_VARs — misleading signal for 9% of samples)
- Function-level pool: `global_max_pool` + `global_mean_pool` over func nodes only → `gnn_eye_proj`
- Prefix gating: `_current_epoch >= warmup` check, calling `select_prefix_nodes`
- Transformer path + `window_pooler` + `transformer_eye_proj`
- Fusion path: `flat_mask` for multi-window, `CrossAttentionFusion` call
- Three-eye concat + classifier, `num_classes==1` squeeze for binary
- `return_aux` pattern: auxiliary logits dict + `jk_entropy` key
- `compute_prefix_attention_mean()`: diagnostic forward with `output_attentions=True`,
  IMP-M3 gap (passes `None` for counts — potential audit flag)
- `parameter_summary()`: trainable vs frozen per sub-module

---

## New Concepts Introduced This Phase

| Concept | First appears | Why it matters |
|---------|--------------|----------------|
| GAT (Graph Attention Network) | Session 8 | Core GNN primitive — weighted neighborhood aggregation |
| Multi-head attention in GNNs | Session 8 | Differs from transformer attention — operates on graph neighborhoods |
| JK (Jumping Knowledge) connections | Session 8 | Prevents Phase 1 signal from being over-smoothed by Phases 2+3 |
| Edge type masking | Session 9 | Each phase uses a different edge subset — structural intent encoded per phase |
| LoRA (Low-Rank Adaptation) | Session 10 | Adapts frozen 125M-param model with ~590K extra trainable params |
| Flash Attention 2 / SDPA | Session 10 | Memory-efficient attention kernels — critical for 512-token sequences |
| Cross-attention (bidirectional) | Session 11 | Nodes query tokens AND tokens query nodes — fine-grained fusion |
| Masked mean pooling | Session 11 | Plain mean includes PAD positions — wrong for variable-length sequences |
| `torch.compile` graph breaks | Session 11 | Data-dependent shapes force eager mode — static shapes avoid it |
| Auxiliary losses | Session 12 | Keep all three eyes alive when main classifier could learn to ignore two |
| GNN prefix injection | Session 12 | GNN→Transformer info flow — most architecturally unique idea in Sentinel |
| Ghost graph fallback | Session 13 | Interface-only contracts have zero function nodes — pooling must not crash |

---

## Anticipated Audit Flags

Issues spotted during roadmap research (full audit inline during teaching):

| ID | File | Issue |
|----|------|-------|
| A23 | `sentinel_model.py:75` | `_MAX_TYPE_ID` reintroduces dynamic computation (same as A3) |
| A24 | `sentinel_model.py:select_prefix_nodes` | Python loop over num_graphs — not vectorized, O(B) eager |
| A25 | `sentinel_model.py:compute_prefix_attention_mean` | Passes `gnn_prefix_counts=None` — IMP-M3 mask bypassed in diagnostic |
| A26 | `transformer_encoder.py:142` | `except (ImportError, ValueError)` catches broken-weights ValueError silently |

*Full flag entries will be added to `audit_flags.md` during the teaching sessions when confirmed by code.*

---

## Cross-File Dependencies

```
graph_schema.py  ──► gnn_encoder.py        (NODE_FEATURE_DIM, NUM_EDGE_TYPES, EDGE_TYPES)
graph_schema.py  ──► sentinel_model.py     (NODE_TYPES → _MAX_TYPE_ID, _FUNC_TYPE_IDS)

gnn_encoder.py          ──► sentinel_model.py  (GNNEncoder)
transformer_encoder.py  ──► sentinel_model.py  (TransformerEncoder, WindowAttentionPooler)
fusion_layer.py         ──► sentinel_model.py  (CrossAttentionFusion)

sentinel_model.py  ──► trainer.py     (SentinelModel instantiation + training loop)
sentinel_model.py  ──► predictor.py   (SentinelModel load + inference)
```
