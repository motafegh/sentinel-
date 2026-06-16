# Deep Dive: Graph Contrastive Pre-training for SENTINEL

**Date:** 2026-06-16
**Module:** `ml`
**Phase:** Run 14 preparation
**Tier:** 1 (biggest F1 lever)
**Proposal Reference:** §5.2 of `2026-06-15_ml_Run13_prep_proposal_next_gen_ml_methods.md`

---

## 1. Full Method Explanation

### 1.1 What Is Graph Contrastive Learning?

Graph contrastive learning (GCL) is a self-supervised pre-training method that learns node/graph representations by maximising agreement between differently augmented views of the same graph. The core idea:

1. Create two augmented views of each graph (positive pair)
2. Create negative pairs from different graphs
3. Train the encoder to distinguish positive pairs from negative pairs
4. The learned representations capture structural regularities without labels

### 1.2 InfoNCE Loss

The standard contrastive objective:

```
L = -log(exp(sim(h_i, h_j) / τ) / Σ_k exp(sim(h_i, h_k) / τ))
```

Where:
- `h_i`, `h_j` are embeddings of two augmented views of the same graph
- `h_k` are embeddings of other graphs (negatives)
- `τ` is temperature (typically 0.07–0.1)
- `sim` is cosine similarity

### 1.3 Augmentation Strategies for Graphs

| Strategy | Description | Semantic Preservation |
|----------|-------------|----------------------|
| Node feature masking | Randomly zero out feature dimensions | ✅ High |
| Edge dropout | Remove edges with probability p | ✅ High (if p < 0.3) |
| Subgraph sampling | Extract random connected subgraph | ✅ High |
| Node dropout | Randomly remove nodes + incident edges | ⚠️ Medium |
| Graph perturbation | Add sparse noise edges | ⚠️ Medium |
| Feature permutation | Shuffle feature dimensions | ❌ Low |

### 1.4 Pre-training vs Fine-tuning

The standard workflow:
1. **Pre-train** encoder on unlabeled data with contrastive objective
2. **Fine-tune** encoder on labeled data with downstream loss (BCE for multi-label)
3. The pre-trained encoder starts from a better initialization → faster convergence, better generalization

### 1.5 Why This Works

Contrastive learning forces the encoder to learn:
- **Structural patterns** — which graph motifs are common
- **Feature correlations** — which node features co-occur
- **Invariance** — which augmentations don't change the graph's identity

For Solidity contracts, this means the GNN learns general CFG patterns (function call chains, state variable access patterns) before seeing any vulnerability labels.

---

## 2. SENTINEL-Specific Audit

### 2.1 Data Availability

**Unlabeled data:** ~25K SmartBugs Wild contracts not in v3 training set
- Total Wild: 47,398 contracts
- In v3 (contamination): 8,233 contracts (17.4%)
- **Truly unlabeled: ~39,165 contracts**

**Current labeled data:** 22,493 contracts (v3 split)

**Pre-training opportunity:** Nearly 2× more unlabeled data than labeled data.

### 2.2 GNN Architecture Compatibility

From `gnn_encoder.py`:
- `GNNEncoder` returns node-level embeddings: `[N, hidden_dim]`
- For contrastive pre-training, we need **graph-level embeddings** → pool node embeddings
- `global_mean_pool` and `global_max_pool` already imported in `sentinel_model.py`

**Contrastive head needed:**
```python
contrast_head = nn.Sequential(
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
)
```

### 2.3 Pre-training Pipeline

```
Unlabeled Contracts (39K)
    ↓
Graph Extraction (existing pipeline)
    ↓
Augmentation (2 views per graph)
    ↓
GNNEncoder + Contrastive Head
    ↓
InfoNCE Loss
    ↓
Pre-trained GNN Weights
    ↓
Load into SentinelModel.gnn
    ↓
Fine-tune with labeled data (BCE loss)
```

### 2.4 Critical Considerations

1. **Augmentation choice matters:** Node feature masking and edge dropout are safe for Solidity CFGs. Random subgraph sampling risks breaking execution paths — use connected subgraph sampling only.

2. **Pre-training epochs:** Too few = underfitting; too many = overfitting to unlabeled distribution. Start with 2-4 epochs, monitor validation loss.

3. **Contrastive head removal:** After pre-training, the contrastive head is discarded. Only the GNN encoder weights are transferred to SentinelModel.

4. **Batch size:** Contrastive learning benefits from large batches (more negatives). With 39K unlabeled graphs and batch=64, each batch has 63 negatives. Consider gradient accumulation to effective batch=256.

5. **VRAM:** Pre-training the GNN alone (without transformer) uses ~1.5GB VRAM. Can batch larger than full 4-eye model.

### 2.5 Expected Impact

- **F1 gain:** +3-8% macro F1 (based on literature: GCL typically gives +2-10% on graph tasks)
- **Rare class improvement:** DoS, ExternalBug benefit most — pre-training teaches structural patterns without label noise
- **Convergence speed:** Fine-tuning converges 2-3× faster from pre-trained initialization
- **Risk:** Low — if pre-training doesn't help, fine-tuning from scratch still works

### 2.6 Relationship to CGBC

CGBC (§5.4) also does contrastive pre-training but for **label correction**. The two methods are complementary:
- **GCL pre-training:** Learns structural patterns from unlabeled data
- **CGBC:** Corrects noisy labels in labeled data

They can be combined: pre-train with GCL → fine-tune with CGBC-corrected labels.

---

## 3. Implementation Plan

### 3.1 Phase B1: Unlabeled Dataset Preparation (~1 day)

**New file:** `ml/scripts/pretraining/build_unlabeled_dataset.py`

1. Load all SmartBugs Wild contracts (47K)
2. Remove contracts in v3 training/val/test splits (8,233)
3. Run graph extraction on remaining ~39K contracts
4. Save as `data_module/data/exports/sentinel-unlabeled-wild/`
5. Create `UnlabeledGraphDataset` class for loading

### 3.2 Phase B2: Contrastive Pre-training (~2 days)

**New file:** `ml/src/training/contrastive_trainer.py`

1. Implement 6 augmentation strategies for Solidity graphs
2. Create `ContrastiveGNNEncoder` wrapper with projection head
3. Implement InfoNCE loss
4. Train 2-4 epochs on unlabeled data
5. Save pre-trained GNN checkpoint

### 3.3 Phase B3: Fine-tuning Integration (~1 day)

**Modify:** `ml/src/training/trainer.py`

1. Add `--pretrained_gnn` flag to load pre-trained GNN weights
2. Verify fine-tuning works from pre-trained initialization
3. Compare F1 with/without pre-training on 66 OOD contracts
4. Document results

### 3.4 Augmentation Implementation

```python
import random

def augment_graph(graph, strategy="mixed"):
    """Apply random augmentation to a PyG Data object."""
    if strategy == "mixed":
        strategy = random.choice(["feature_mask", "edge_dropout", "node_dropout"])

    if strategy == "feature_mask":
        # Mask 15% of feature dimensions
        mask = torch.bernoulli(torch.full(graph.x.shape, 0.85))
        graph.x = graph.x * mask

    elif strategy == "edge_dropout":
        # Drop 20% of edges
        keep = torch.bernoulli(torch.full((graph.edge_index.shape[1],), 0.8))
        graph.edge_index = graph.edge_index[:, keep.bool()]
        if graph.edge_attr is not None:
            graph.edge_attr = graph.edge_attr[keep.bool()]

    elif strategy == "node_dropout":
        # Drop 10% of nodes (keep connected components)
        keep = torch.bernoulli(torch.full((graph.num_nodes,), 0.9))
        # Remap edge indices
        ...

    return graph
```

### 3.5 Expected Outcomes

- **Pre-training time:** ~4 hours (2 epochs on 39K graphs, RTX 3070)
- **Fine-tuning convergence:** 2-3× faster than from scratch
- **F1 improvement:** +3-8% macro F1 expected
- **VRAM:** ~1.5GB for GNN-only pre-training (well within 8GB)

### 3.6 Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Augmentations break CFG semantics | Low | Medium | Use only feature_mask + edge_dropout |
| Pre-training doesn't transfer | Low | High | Evaluate after 2 epochs, abandon if no gain |
| VRAM OOM with large batches | Low | Low | Use grad accum, batch=32 effective=128 |
| Unlabeled data too noisy | Medium | Low | Filter contracts < 10 nodes, > 10000 nodes |

---

## 4. References

- **GraphCL** (2020): "A Simple Contrastive Learning Framework for Graph Neural Networks" — foundational GCL
- **GRACE** (2020): "Deep Graph Contrastive Representation Learning" — node-level contrastive
- **GraphMAE** (2022): "Masked Graph Autoencoder" — complementary masked reconstruction
- **Jakiro** (2026): Cross-modal contrastive for smart contracts — same domain
- **CGBC** (2026): `arXiv:2603.27734` — contrastive pre-training for label correction
