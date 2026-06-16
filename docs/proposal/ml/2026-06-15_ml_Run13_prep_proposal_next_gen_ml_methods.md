# SENTINEL Next-Generation ML Methods — Final Merged Proposal

**Date:** 2026-06-16  
**Authors:** Ali (SENTINEL) + AI Research Collaborator  
**Module:** `ml`  
**Phase:** Run 13 preparation + Run 14/15/16 roadmap  
**Status:** Agreed proposal — ready for Run 13 gating  
**Version:** 2.0 (merged from Ali v1.0 + AI v1.0)  

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [SENTINEL Current Architecture Baseline](#2-sentinel-current-architecture-baseline)
3. [Known Weaknesses & Bottlenecks](#3-known-weaknesses--bottlenecks)
4. [ML Method Taxonomy (2025–2026 SOTA)](#4-ml-method-taxonomy-20252026-sota)
5. [Method Deep Dives](#5-method-deep-dives)
   - 5.1 Conformal Prediction
   - 5.2 Graph Contrastive Pre-training
   - 5.3 Structured Ensembling (BreachT5-style)
   - 5.4 Granular-ball Label Correction (CGBC)
   - 5.5 CPPO Constrained Fine-Tuning
   - 5.6 SAC Adaptive Threshold Agent
   - 5.7 Evidential Deep Learning on Aux Heads
   - 5.8 UniXcoder / CodeSage Encoder Upgrade
   - 5.9 SigGate-GT / GraphGPS
   - 5.10 MAML Few-Shot Adaptation
   - 5.11 Cross-Modal Contrastive Alignment
   - 5.12 MoE Classifier Head
   - 5.13 Causal Attention / Information Bottleneck
   - 5.14 Multi-Task Contract + Function-Level
   - 5.15 Graph-Mamba for Phase 2
   - 5.16 Additional Quick-Win Methods
6. [SENTINEL-Specific Ranking](#6-sentinel-specific-ranking)
7. [Implementation Roadmap](#7-implementation-roadmap)
8. [Risk & Trade-off Analysis](#8-risk--trade-off-analysis)
9. [What NOT to Do](#9-what-not-to-do)
10. [References](#10-references)
11. [Appendix A: Cost Estimate](#appendix-a-cost-estimate)
12. [Appendix B: Expected F1 Trajectory](#appendix-b-expected-f1-trajectory)

---

## 1. Executive Summary

SENTINEL is a dual-path GNN+GraphCodeBERT multi-label vulnerability detector for Solidity smart contracts, achieving **F1_tuned=0.7004** (Run 12, 10 classes). This merged proposal surveys the 2025–2026 state of the art across graph neural networks, code transformers, reinforcement learning, multi-modal fusion, self-supervised learning, uncertainty quantification, and production ML — then ranks each method by **impact ÷ complexity ÷ risk** for SENTINEL specifically.

**Two key findings drive this proposal:**

1. **The single biggest untapped lever is graph contrastive pre-training.** SENTINEL has ~47K SmartBugs Wild contracts but only ~22K are labelled. The ~25K unlabeled contracts can pre-train the GNN via self-supervised contrastive objectives, learning general Solidity structural patterns before any fine-tuning. Every 2026 SOTA smart contract paper uses some form of pre-training; SENTINEL currently trains from scratch every run.

2. **Reinforcement learning (CPPO specifically) is the safest way to push rare-class F1 without regression.** After supervised pre-training converges, CPPO fine-tuning with Lagrangian constraints can directly optimize a reward that emphasizes DoS (243 positives), ExternalBug, and TOD — while guaranteeing no regression on Reentrancy, IntegerUO, or calibration metrics. This is fundamentally different from ASL weighting: RL operates on task-level rewards, not loss-level adjustments.

**Top 6 recommendations for Run 14 (in priority order):**

| # | Method | Why | Timeline | Risk |
|---|--------|-----|----------|------|
| 1 | Conformal Prediction | Post-hoc, zero retraining, formal 95% coverage | ~1 day | None |
| 2 | Graph Contrastive Pre-training | Biggest F1 lever, uses 25K unlabeled contracts | ~3 days | Low |
| 3 | Structured Ensembling | Cascade fast→full, rare class coverage | ~2 days | Low |
| 4 | Granular-ball (CGBC) | Noisy label correction, ExternalBug FP fix | ~4 days | Low-Med |
| 5 | Quick Wins (SWA + Mixup + Laplacian PE + CC Label Smoothing) | ~50 lines total, 1–3% F1 each | ~1 day | None |
| 6 | CPPO Fine-Tuning | RL reward optimization for rare classes | ~3 days | Medium |

**Expected cumulative F1 trajectory:** Run 13 (baseline) → Run 14 (+5–12pp) → Run 15 (+2–8pp) → Run 16+ (target F1 ≥ 0.80).

---

## 2. SENTINEL Current Architecture Baseline

### Model: Four-Eye Dual-Path v8.1

| Component | Architecture | Output Dim | Parameters |
|-----------|-------------|------------|------------|
| **GNN Encoder** | 8-layer GAT (2+3+3 phases), JK attention, type embeddings | [N, 256] | ~2.5M |
| **Transformer** | GraphCodeBERT (125M frozen) + LoRA (r=16, Q+V) | [B, W×L, 768] | ~590K trainable |
| **Cross-Attention Fusion** | Bidirectional MHA (node↔token), masked mean pool | [B, 128] | ~1.5M |
| **4 Eyes** | GNN / Transformer / Fused / CFG — each [B, 128] | [B, 512] concat | ~200K |
| **Classifier** | Linear(512→256→C), no sigmoid | [B, C] | ~130K |
| **Aux Heads** | 4 auxiliary linear heads (training only) | [B, C] each | ~5K |
| **Total** | | | **~5M trainable / ~129M total** |

### Training

| Hyperparameter | Value |
|----------------|-------|
| **Loss** | AsymmetricLoss (γ_neg=2.0, γ_pos=1.0, clip=0.01) |
| **Optimizer** | AdamW, lr=2e-4, weight_decay=1e-2 |
| **Schedule** | OneCycleLR, warmup 10% |
| **Batch** | 8 (effective 64 with grad accum ×8) |
| **Precision** | AMP BF16 |
| **Hardware** | RTX 3070 8GB VRAM |
| **Data** | v3 export: 22,493 contracts, splits 18,596/1,983/1,914 |

### Inference

| Component | Detail |
|-----------|--------|
| **API** | FastAPI with Prometheus metrics |
| **Thresholds** | 3-tier: confirmed ≥0.55, suspicious ≥0.25, safe <0.25 |
| **Uncertainty** | None beyond raw sigmoid probability |
| **Drift Detection** | KS test every 50 requests (placeholder baseline) |

---

## 3. Known Weaknesses & Bottlenecks

### 3.1 ExternalBug False Positives (65% S_only)
Run 12 manual inspection found s_Form001 (26-line KV store) predicted as ExternalBug at p=0.96. The model learned **spurious features** from training data — not generalizable vulnerability indicators. Root cause: **noisy automated-tool labels**. This is a label noise problem, not just an imbalance problem, and requires label correction (CGBC) not just loss reweighting.

### 3.2 DoS Extreme Class Imbalance
Only 243 training positives vs 18,596 total. Standard ASL handles imbalance partially but the model rarely sees DoS patterns during training. This is a **few-shot learning** problem — both CGBC (label correction) and MAML/CPPO (few-shot adaptation / reward optimization) address it from different angles.

### 3.3 Local-Only GNN Reasoning
Phase 2 has only 3-hop CFG reach (L3–L5). Cross-function, long-range patterns (recursive call chains, multi-contract interactions) cannot be captured. APPNP teleport (α=0.2) helps but does not fundamentally solve this. GraphGPS (global attention + local MPNN) or Graph-Mamba (linear-complexity global state) are the architectural solutions.

### 3.4 No Uncertainty Quantification
SENTINEL outputs raw sigmoid probabilities with no distinction between:
- **Epistemic uncertainty** (model doesn't know — lack of training data for this pattern)
- **Aleatoric uncertainty** (data is inherently ambiguous — the vulnerability is borderline)

A p=0.55 prediction could be either a confident boundary case or a wild guess on an OOD contract. Conformal Prediction (post-hoc, guaranteed coverage) and Evidential Deep Learning (per-eye decomposition) address this from different angles.

### 3.5 CodeBERT Capacity Ceiling
GraphCodeBERT (125M) is the smallest code PLM. UniXcoder (same 125M size) shows +3.2pp CSN MRR improvement with zero VRAM change. CodeSage-1.3B shows further gains at the cost of ~1GB additional VRAM.

### 3.6 Contamination
17.4% of SmartBugs Wild is in v3 training data (normalised-SHA-256 match). Only 39,165 (82.6%) are truly OOD. All future benchmarks require 0% contamination gate.

### 3.7 GasException Dropped (Run 13)
Zero training data for GasException in v3 — class removed for Run 13 (9 classes).

### 3.8 No Self-Supervised Signal
The 41K+ graphs are only seen with their labels; structural regularities are not explicitly exploited. This is the biggest untapped data resource.

### 3.9 Static Thresholds
Per-class thresholds tuned post-hoc via grid search — no adaptive threshold optimization during training. RL threshold agents (SAC/DQN) can dynamically optimize thresholds.

---

## 4. ML Method Taxonomy (2025–2026 SOTA)

### 4.1 Graph Neural Network Architectures

| Method | Key Idea | Year | Performance | Complexity |
|--------|----------|------|-------------|------------|
| **GAT** (current) | Attention over neighbors | 2018 | Baseline | O(E) |
| **GIN** | MLP-based, max 1-WL expressive | 2019 | +0–3% vs GCN | O(E) |
| **GraphGPS** | Hybrid: local MPNN + global attention | 2022 | SOTA on OGB/LRGB | O(N²) full / O(N) sparse |
| **SigGate-GT** | Sigmoid gating replaces softmax → no attention sinks | 2026 | **SOTA on ogbg-molhiv (82.47% ROC-AUC)**, -30% over-smoothing | O(N²) |
| **Graph-Mamba** | Selective SSM replaces attention, O(n) complexity | 2024 | Matches Transformers at 74% less GPU memory | **O(N)** |
| **MbaGCN** | Mamba-based GCN with adaptive aggregation | 2025 | Avg rank 1.71 across 8 datasets, excels on heterophily | O(N) |
| **HOGAT** | Multi-hop attention on unified contract graph | 2026 | **89.8% F1 on smart contracts**, +3–8% over baselines | O(K·E) |
| **SBP** | Signed graph propagation — positive + negative edges | 2025 | **300 layers** without over-smoothing | O(E) |
| **Multi-Track MPNN** | Separate message channels per category | 2024 | 86.4% on Cora | O(K·E) |

### 4.2 Code Transformer Models

| Model | Type | Size | Code Search (CSN MRR) | Clone Detection (MAP) | LoRA VRAM |
|-------|------|------|----------------------|----------------------|-----------|
| **GraphCodeBERT** (current) | Encoder | 125M | 72.08% | 85.50% | ~0.5GB |
| **UniXcoder** | Encoder | 125M | **74.40%** | **89.56%** | ~0.5GB |
| **CodeSage** | Encoder | 1.3B | **75.80%** | 87.70% | ~3GB |
| **DeepSeek-Coder-1.3B + CL4D** | Decoder (adapted) | 1.3B | **77.57%** | 89.71% | ~3GB |
| **Qwen2.5-Coder-7B** | Decoder | 7B | Beats GPT-4o on coding benchmarks | | ~4GB (Q4) |

*Encoder-only models remain optimal for classification. Decoder models excel at generation and should not replace the primary encoder (see §9).*

### 4.3 Self-Supervised & Contrastive Learning

| Method | Domain | Objective | Relevance |
|--------|--------|-----------|-----------|
| **Graph Contrastive Learning (GCL)** | Graphs | InfoNCE over augmented views | Pre-train GNN on unlabeled contracts |
| **MGCL** | Graphs | Graphon-informed augmentations | Data-adaptive, principled augmentations |
| **FDAGCL** | Graphs | Feature discrepancy-aware multi-view | Handles weak vs strong features |
| **FOSSIL** | Graphs | Fused Gromov-Wasserstein subgraph contrastive | Homophilic + heterophilic |
| **IFL-GCL** | Graphs | InfoNCE as "free lunch" for semantic sampling | Corrects sampling bias in GCL |
| **GraphMAE** | Graphs | Masked feature reconstruction | Forces node semantic understanding |
| **CL4D** | Code | Contrastive learning for decoder-only models | +75.9pp on zero-shot code search |
| **Jakiro** | Smart contracts | CFG + source code cross-modal contrastive | **+6.49% precision, +4.62% F1 on 38K contracts** |
| **Granular-ball (CGBC)** | Smart contracts | Clustering + contrastive + symmetric CE | **Noisy label correction for automated tool labels** |

### 4.4 Reinforcement Learning Methods

| Method | Type | Action Space | Key Feature | SENTINEL Application |
|--------|------|-------------|-------------|---------------------|
| **PPO** | On-policy policy gradient | Continuous | Clipped surrogate objective | Reward-guided fine-tuning after supervised pre-training |
| **CPPO** | Constrained PPO | Continuous | Lagrangian constraints | Same as PPO with **guaranteed non-regression** on protected metrics |
| **REINFORCE** | Policy gradient | Discrete | Simplest RL method | Graph traversal agent for vulnerability path discovery |
| **A2C/A3C** | Actor-Critic | Discrete | Critic baseline reduces variance | Same as REINFORCE with faster convergence |
| **DQN** | Off-policy value-based | Discrete | Experience replay + target network | Adaptive per-class threshold optimization |
| **SAC** | Off-policy actor-critic | Continuous | Entropy regularization | Smooth continuous threshold optimization |
| **TD3** | Off-policy actor-critic | Continuous | Twin critics, delayed updates | Robust threshold control for noisy F1 reward |
| **Contextual Bandit** | Bandit | Discrete | Single-step feedback | Active learning sample selector (replaces weighted sampler) |

### 4.5 Multi-Modal Fusion

| Paper | Method | Performance | Key Innovation |
|-------|--------|-------------|----------------|
| **SENTINEL (current)** | Bi-directional cross-attention MHA | F1=0.7004 | Node↔token fusion |
| **ContractShield** (2026) | 3-level: self-attn → cross-modal → adaptive weighting | **91% F1, -1–3% under obfuscation** | xLSTM for opcodes, GATv2 for CFG, adaptive obfuscation-robust |
| **ORACAL** (2026) | RAG-enriched + causal attention + PGExplainer | **91.28% Macro F1, +39.6pp** | Causal disentanglement, RAG-augmented expert context |
| **Hierarchical Graph Transformer** (2026) | Multi-resolution + community integration + uncertainty multi-task | **92.18% F1, 6.12% FNR** | Same domain — contract + function-level jointly |
| **BreachT5** (2025) | CodeT5+ ensemble (220M + 770M) | Micro 0.6122, Macro 0.5114 | Structured ensembling reconciles rare vs frequent |

### 4.6 Uncertainty Quantification

| Method | Type | Forward Passes | Guarantee | Overhead |
|--------|------|---------------|-----------|----------|
| **Softmax probability** (current) | Heuristic | 1 | None | Zero |
| **MC Dropout** | Approx Bayesian | N | None | N× |
| **Deep Ensembles** | Ensemble | N | None | N× training + N× inference |
| **Conformal Prediction** | Distribution-free | 1 | **95% coverage guarantee** | Minimal (calibration set) |
| **Evidential Deep Learning (EDL)** | Single-pass | 1 | None (Dirichlet assumptions) | Architectural change |
| **F-EDL** (Flexible EDL) | Single-pass | 1 | None (weaker assumptions) | Architectural change |
| **ETN** (Evidential Transformation Network) | Post-hoc module | 1 | None | Very low |
| **Laplace Approximation** | Post-hoc | 1 | None (local posterior) | Low |
| **LoRA Ensemble** | Ensemble (shared backbone) | N | None | N× LoRA forward |

### 4.7 Ensemble & Multi-Model Methods

| Method | Structure | Training Cost | Inference Cost | Benefit |
|--------|-----------|---------------|----------------|---------|
| **Deep Ensembles** | N independent models | N× | N× | Best uncertainty |
| **Snapshot Ensembles / SWA** | N checkpoints from 1 run | 1× | N× (SWA: 1×) | Cheap training |
| **LoRA Ensemble** | 1 backbone + N adapters | ~1.2× | N× | Very cheap |
| **Structured Ensembling** | Specialized models per class | C× | C× | Best per-class |
| **BreachT5** | 2 models (220M + 770M) | 2× | 2× | Rare + frequent class balance |
| **MC Dropout** | 1 model, N stochastic passes | 1× | N× | Approximate Bayesian |

### 4.8 Over-smoothing / Deep GNN Methods

| Method | Mechanism | Max Depth | Already in SENTINEL? |
|--------|-----------|-----------|---------------------|
| **Residual connections** | Identity skip | ~10 layers | ✅ (Phase 1, Phase 2, Phase 3) |
| **APPNP teleport** | Blend with input | ~10 layers | ✅ (α=0.2 for Phase 2) |
| **JKNet** | Aggregate all layers | ~10 layers | ✅ (attention mode) |
| **LayerNorm per phase** | Normalize stats | ~8 layers | ✅ |
| **SigGate-GT** | Sigmoid gating | ~16+ layers | ❌ |
| **SBP** | Structural balance theory | **300 layers** | ❌ |
| **Mamba (SSM)** | Selective state space | Very deep | ❌ |
| **PairNorm** | Normalization pair | ~50 layers | ❌ |
| **DropEdge** | Stochastic edge dropping | ~10 layers | ❌ |

---

## 5. Method Deep Dives

### 5.1 Conformal Prediction (Tier 1 — Highest Priority)

**What:** Distribution-free uncertainty quantification that constructs prediction sets with finite-sample coverage guarantee: `P(Y ∈ C(X)) ≥ 1-α`. Split conformal prediction uses a held-out calibration set to find a non-conformity score threshold. At inference, includes all classes whose score falls below the threshold.

**Why for SENTINEL:**
- Currently outputs raw sigmoid probabilities with no reliability guarantee
- A p=0.55 prediction could be either confident or lucky
- CP turns probability into: *"95% confident the true vulnerability class is among {Reentrancy, Timestamp}"*
- Post-hoc: ~20 lines of code, zero retraining, no model changes
- Complements the 3-tier threshold system with formal guarantees

**Implementation sketch:**
```python
# 1. Calibration: run model on held-out calibration set
cal_scores = []
for x, y in calibration_loader:
    logits = model(x)  # [B, C]
    probs = torch.sigmoid(logits)
    # Non-conformity: 1 - max probability of any TRUE positive class
    score = 1.0 - probs[y == 1].max()
    cal_scores.append(score)
# 2. Find threshold at quantile (1-α) * (n+1)/n
q = np.quantile(cal_scores, (1 - 0.05) * (len(cal_scores) + 1) / len(cal_scores))
# 3. At inference: prediction set = {class | 1 - prob <= q}
pred_set = set(np.where(1.0 - sigmoid(logits) <= q)[0])
```

**References:**
- CONFIDE (2026): conformal prediction for fine-tuned transformers — `arXiv:2604.08885`
- ECP (2024): evidential conformal prediction with Dirichlet non-conformity — PMLR 230:466-489

---

### 5.2 Graph Contrastive Pre-training (Tier 1 — Biggest F1 Lever)

**What:** Self-supervised pre-training of the GNN encoder on **unlabeled** contract graphs. The model learns to maximise agreement between differently augmented views of the same graph (positive pairs) and minimise agreement between views of different graphs (negative pairs). Standard InfoNCE loss.

**Why for SENTINEL:**
- Biggest untapped data resource: **~25K unlabeled SmartBugs Wild contracts** not used in v3
- SENTINEL's GNN (2.5M params) is small enough that pre-training converges fast (a few hours)
- The GNN learns general Solidity structural patterns — CFG topology, function boundaries, call patterns — before seeing any vulnerability labels
- Reduces reliance on synthetic training signals → **directly addresses ExternalBug FP**
- Improves few-shot capability for rare classes like DoS

**Augmentation strategies for Solidity contract graphs:**

| Augmentation | Description | Semantics-Preserving? |
|-------------|-------------|----------------------|
| Node feature masking | Mask random feature dimensions | ✅ |
| Edge dropout | Randomly remove edges (p=0.2) | ✅ (noise-tolerant) |
| Subgraph sampling | Extract random connected subgraph | ✅ |
| Node type masking | Mask type ID in feat[0], model must predict | ✅ |
| CFG path dropout | Remove random CFG edges | ⚠️ (may break execution semantics) |
| Graph perturbation | Add/remove sparse noise edges | ✅ |

**Implementation approach:**
```python
# Pre-training phase (before any fine-tuning)
gnn = GNNEncoder(...)  # same architecture as main model
# Contrastive head: projection from [N, 256] → [N, 64] for contrastive loss
contrast_head = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 64))

# InfoNCE loss on pooled graph embeddings
for batch in unlabeled_dataloader:
    x1 = augment(batch.graph)  # view 1
    x2 = augment(batch.graph)  # view 2
    h1 = contrast_head(global_mean_pool(gnn(x1), batch.batch))
    h2 = contrast_head(global_mean_pool(gnn(x2), batch.batch))
    loss = info_nce(h1, h2, temperature=0.1)
    loss.backward()

# Fine-tuning: discard contrast_head, load pre-trained GNN weights into SENTINEL
model.gnn.load_state_dict(gnn.state_dict())
```

**Also consider:** GraphMAE (masked feature reconstruction) as an alternative or complement. SENTINEL's node features have semantic meaning (type_id, visibility, payable, complexity) — reconstructing them forces the GNN to understand what each feature represents. GraphMAE and contrastive learning can be combined: GraphMAE for node-level understanding, contrastive for graph-level understanding.

---

### 5.3 Structured Ensembling (Tier 1 — BreachT5-style)

**What:** Train two complementary models:
1. **Small fast model** (GNN-only, no transformer, ~2.5M params) — excels at structural patterns, runs in <50ms
2. **Full 4-eye model** (current SENTINEL) — adds semantic depth via CodeBERT + fusion

At inference, the small model runs first. If its confidence is high (>0.8 for any class), return early. Otherwise, run the full model. Meta-classifier learns weighting between the two.

**Why for SENTINEL:**
- Research shows **rare classes benefit from smaller, specialized models** while frequent classes benefit from larger ones (BreachT5, 2025)
- DoS (243 positives), MishandledException (~525 after BCCC injection) would get dedicated structural reasoning
- Reduce average inference latency by ~60% (80% of confident predictions from small model)
- The GNN-only model shares 100% of GNNEncoder code — trivial to train

**Implementation:**
```python
class GNNOnlyModel(nn.Module):
    """
    Lightweight GNN-only model for cascade inference.
    Reuses the exact same GNNEncoder architecture.
    """
    def __init__(self, gnn_config, num_classes=9):
        super().__init__()
        self.gnn = GNNEncoder(**gnn_config)  # Same as full model
        self.pool = VulnerabilityAttentionPooler(hidden_dim=256, num_queries=4)
        self.classifier = nn.Sequential(
            nn.Linear(4 * 256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, graph_data):
        node_embs = self.gnn(graph_data)
        pooled = self.pool(node_embs, graph_data.batch, graph_data.num_graphs)
        return self.classifier(pooled)


class CascadeInference:
    """
    Two-stage inference: fast model → confidence gate → full model.
    """
    def __init__(self, fast_model, full_model, confidence_threshold=0.8):
        self.fast = fast_model
        self.full = full_model
        self.threshold = confidence_threshold

    def predict(self, graph_data, token_data):
        # Stage 1: fast model
        fast_logits = self.fast(graph_data)
        fast_probs = torch.sigmoid(fast_logits)
        if (fast_probs > self.threshold).any():
            return fast_probs  # High confidence → return early

        # Stage 2: full 4-eye model
        return self.full(graph_data, token_data)
```

---

### 5.4 Granular-ball Label Correction — CGBC (Tier 1)

**What:** Method specifically designed for smart contract vulnerability detection with **noisy labels**. Groups similar contracts into "granular balls" (clusters), computes consensus labels per cluster, and trains with:
1. Contrastive pre-training with semantic-consistent augmentation
2. Inter-GB compactness loss + intra-GB looseness loss
3. Symmetric cross-entropy as the final loss (noise-robust)

**Why for SENTINEL:**
- SENTINEL's labels come from automated tools (Slither, SmartBugs static analyzers) — **inherently noisy**
- The ExternalBug FP (s_Form001 at p=0.96) is a direct symptom of spurious training signals from noisy labels
- CGBC corrects label noise BEFORE training: contracts clustered by structural+semantic similarity, labels aggregated by majority within each cluster
- Contrastive pre-training component provides additional robustness

**Implementation approach:**
```python
class GranularBallCorrector:
    """
    Cluster contracts into granular balls, compute consensus labels,
    and identify/correct noisy labels before training.
    """

    def __init__(self, embedding_dim=512, purity_threshold=0.7):
        self.purity_threshold = purity_threshold

    def fit(self, embeddings, labels):
        """
        embeddings: [N, 512] — four-eye concatenated embeddings from a pre-trained model
        labels: [N, C] — original (potentially noisy) multi-hot labels
        """
        from sklearn.cluster import DBSCAN
        # Cluster in embedding space
        clustering = DBSCAN(eps=0.5, min_samples=3).fit(embeddings)
        self.corrected_labels = labels.clone()
        self.noise_mask = torch.zeros(len(labels), dtype=torch.bool)

        for cluster_id in set(clustering.labels_):
            if cluster_id == -1:
                continue  # noise points
            mask = clustering.labels_ == cluster_id
            cluster_labels = labels[mask]
            # Majority vote per class
            consensus = (cluster_labels.sum(dim=0) / mask.sum()) > self.purity_threshold
            # Mark labels that disagree with consensus as potentially noisy
            for i in torch.where(mask)[0]:
                disagreement = (labels[i] != consensus.float()).any()
                if disagreement:
                    self.corrected_labels[i] = consensus.float()
                    self.noise_mask[i] = True
```

**Reference:** `arXiv:2603.27734` — CGBC achieves significant gains over baselines on smart contract vulnerability detection with label noise.

---

### 5.5 CPPO Constrained Fine-Tuning (Tier 1 — RL for Rare Classes)

**What:** After supervised pre-training converges, switch to CPPO (Constrained Proximal Policy Optimization) fine-tuning where the model is treated as a policy that chooses class assignments, and a reward signal encourages improvements on rare classes — with Lagrangian constraints that guarantee no regression on protected metrics.

**Algorithm:**
```
maximize   E[L^CLIP(θ)]
subject to F1(Reentrancy) ≥ 0.70       (Constraint 1: no regression on best class)
           Precision(any) ≥ 0.60        (Constraint 2: no hallucination increase)
           ECE ≤ 0.10                   (Constraint 3: calibration must not worsen)
           F1(DoS) improvement ≥ 0.02   (Constraint 4: must help rare class)
```

The constraints are enforced via Lagrangian multipliers updated concurrently with the policy:

```
L = L_ppo + Σ_i λ_i * max(0, c_i(θ) - d_i)
```

**Why for SENTINEL:**
- ASL loss weights classes but operates at the loss level — it cannot directly optimize F1, which is the actual evaluation metric
- CPPO directly optimizes a **task-level reward** that combines F1 improvements on rare classes
- The constraints make it **safe**: if any protected metric starts regressing, the Lagrangian penalty increases and pulls the optimization back
- This is fundamentally different from increasing `dos_loss_weight` — RL reward can capture non-differentiable metrics (F1 is non-differentiable at the decision threshold)

**Concrete integration:**
```python
class CPPOFineTuner:
    """
    Constrained PPO fine-tuning for rare-class F1 optimization.
    Applied after supervised pre-training converges (epoch 50+).
    """

    def __init__(self, model, config):
        self.model = model
        self.clip_epsilon = 0.2
        self.gamma = 0.99
        self.lam = 0.95  # GAE lambda
        self.ppo_epochs = 4
        self.value_coeff = 0.5
        self.entropy_coeff = 0.01

        # Protected metrics with floors
        self.constraints = {
            'reentrancy_f1_floor': 0.70,
            'min_precision': 0.60,
            'max_ece': 0.10,
            'dos_f1_improvement': 0.02,
        }
        # Lagrangian multipliers (learned)
        self.lambdas = {k: 0.0 for k in self.constraints}
        self.lagrangian_lr = 1e-3

    def compute_reward(self, logits, labels, class_weights):
        """
        Per-sample reward emphasizing rare-class improvement.
        R = Σ_c w_c * (TP_c - FP_c - 0.5*FN_c)
        """
        preds = (torch.sigmoid(logits) > 0.5).float()
        tp = (preds * labels * class_weights).sum(dim=1)
        fp = ((1 - labels) * preds * class_weights).sum(dim=1)
        fn = (labels * (1 - preds) * class_weights).sum(dim=1)
        return tp - fp - 0.5 * fn

    def constrained_objective(self, ppo_loss, constraint_vals):
        """
        L = L_ppo + Σ_i λ_i * max(0, c_i - d_i)
        Lagrangian multipliers updated via gradient ascent on violation.
        """
        lagrangian_penalty = 0.0
        for name, (val, limit) in constraint_vals.items():
            violation = max(0.0, val - limit)
            lagrangian_penalty += self.lambdas[name] * violation
            # Dual update: increase λ if constraint is violated
            self.lambdas[name] = max(0, self.lambdas[name] + self.lagrangian_lr * violation)
        return ppo_loss + lagrangian_penalty
```

**Expected benefit:** 2–5% F1 improvement on rare classes (DoS, ExternalBug, TOD) without any regression on well-performing classes.

**Key insight:** CPPO is the **safest RL method** for SENTINEL because:
1. It clips policy updates (PPO's core innovation) — prevents large, destructive updates
2. It adds hard constraints (CPPO's extension) — guarantees no regression
3. It operates as a fine-tuning phase — doesn't replace supervised training

---

### 5.6 SAC Adaptive Threshold Agent (Tier 2)

**What:** Train a Soft Actor-Critic agent that continuously adjusts per-class classification thresholds based on the model's current prediction distribution. Replaces the static post-hoc grid search in `tune_threshold.py` with dynamic optimization.

**State:** `[per_class_prec, per_class_rec, per_class_f1, per_class_pred_mean, per_class_pred_std]` (50-dim)  
**Action:** `[Δ_threshold_1, ..., Δ_threshold_9]` ∈ R^9, clamped to [-0.1, +0.1]  
**Reward:** `Δ(macro_F1) + 0.1 × Δ(micro_F1) - 0.01 × ||Δ_thresholds||²`

**Why SAC over DQN:** Continuous action space (smooth threshold adjustments), entropy regularization (prevents premature convergence to local optima — F1 is non-convex in threshold space), and off-policy learning (sample efficient).

**Why for SENTINEL:** The current grid search (0.05–0.95, step 0.05) is coarse and treats each class independently. SAC can discover non-obvious threshold interactions between classes (e.g., lowering Reentrancy threshold may improve Timestamp F1 via cross-class signal in the four-eye fusion).

**Risk:** Low — operates on thresholds only, not model weights. Worst case is suboptimal thresholds (grid search fallback available).

---

### 5.7 Evidential Deep Learning on Aux Heads (Tier 2)

**What:** Replace the 4 auxiliary linear heads (`Linear(128, C)`) with evidential heads producing Dirichlet distribution parameters. Each head outputs evidence vector `e ∈ ℝ^C_+` where `α_c = e_c + 1` are the Dirichlet concentration parameters.

**Why for SENTINEL:**
- Current aux heads produce point-estimate logits → no uncertainty signal
- Evidential heads give **per-eye uncertainty decomposition**:
  - Which eye is most uncertain for this prediction? → Tells you which modality is confused
  - High epistemic uncertainty on all 4 eyes → OOD contract
  - High aleatoric uncertainty on all 4 eyes → genuinely ambiguous vulnerability
- The 4-eye architecture is **uniquely suited**: each eye represents a distinct information modality
- F-EDL (NeurIPS 2025) extends EDL with flexible Dirichlet — handles complex distributions

**Implementation sketch:**
```python
class EvidentialHead(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)
        self.softplus = nn.Softplus()

    def forward(self, x):
        evidence = self.softplus(self.fc(x))  # e ≥ 0
        alpha = evidence + 1.0                 # Dirichlet params α_c = e_c + 1
        # Strength: S = sum(α_c)
        # Predicted prob: p_c = α_c / S
        # Uncertainty: u = C / S (higher = more uncertain)
        return alpha

def evidential_loss(alpha, targets):
    S = alpha.sum(dim=1, keepdim=True)
    loss = ((alpha - targets) ** 2).sum(dim=1) / (S ** 2)
    return loss.mean()
```

**Complement with Conformal Prediction:** CP provides formal coverage guarantees on the main classifier output; EDL provides per-eye uncertainty decomposition for interpretability. Use both.

---

### 5.8 UniXcoder / CodeSage Encoder Upgrade (Tier 2)

**What:** Replace GraphCodeBERT with a stronger code encoder:

**Phase 1 — UniXcoder (125M):**
- Same size, same VRAM (~0.5GB LoRA), 1-day swap
- CSN MRR: 74.40% vs 72.08% (+3.2pp), Clone MAP: 89.56% vs 85.50% (+4.1pp)
- UniXcoder also outputs [B, L, 768] → no architecture change in fusion_layer.py
- LoRA config unchanged (r=16 still works on Q+V)

**Phase 2 — CodeSage-1.3B (optional):**
- Larger but uses LoRA (r=8) to fit in ~3GB
- CSN MRR: 75.80% (best encoder-only)
- Same output dimension → fusion layer unchanged
- Requires ~1GB additional VRAM over current setup

**Migration path:**
```python
# transformer_encoder.py — minimal change
# Before:
self.bert = AutoModel.from_pretrained("microsoft/graphcodebert-base", ...)
# After (Phase 1):
self.bert = AutoModel.from_pretrained("microsoft/unixcoder-base", ...)
# After (Phase 2):
self.bert = AutoModel.from_pretrained("Salesforce/codeSage-large", ...)
```

---

### 5.9 SigGate-GT / GraphGPS (Tier 3)

**What:** Replace the 8-layer GAT with a GraphGPS framework: each layer = local MPNN (GAT) + global attention (Transformer) + sigmoid gating. SigGate-GT applies element-wise sigmoid gates to attention outputs, eliminating attention sinks.

**Why for SENTINEL:**
- Current GNN has **local-only** message passing (8 GAT layers, CFG reach limited to 3 hops)
- Contract vulnerabilities can span **arbitrarily distant** nodes
- GraphGPS adds **global attention**: every node can attend to every other node
- SigGate-GT solves over-smoothing — enables deeper graphs without representation collapse
- Only ~1% parameter overhead for sigmoid gating

**Architecture change:**
```python
class GPSBlock(nn.Module):
    def __init__(self, ...):
        self.local_mpnn = GATConv(...)
        self.global_attn = MultiheadAttention(...)
        self.sigmoid_gate = nn.Parameter(torch.ones(num_heads))
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, edge_index, batch):
        x_local = self.local_mpnn(x, edge_index)
        x_global = self.global_attn(x, batch)  # Performer-style linear attention
        # SigGate: element-wise sigmoid on global output
        x_gated = torch.sigmoid(self.sigmoid_gate) * x_global
        x = self.norm(x_local + x_gated)
        return x
```

**Cost:** ~20% more VRAM for attention. Expected F1 gain: +2–5% from global reasoning.

---

### 5.10 MAML Few-Shot Adaptation (Tier 3)

**What:** Model-Agnostic Meta-Learning learns initializations that can quickly adapt to new tasks with few gradient steps. Treat each vulnerability class as a separate "task". MAML learns an initialization that can adapt to rare classes (DoS, ExternalBug) with only a few labeled examples.

**Why for SENTINEL (in addition to CGBC):**
- CGBC fixes **label noise** (a data quality problem)
- MAML fixes **few-shot adaptation** (a learning problem)
- These are complementary: even with perfect labels, 243 DoS positives is still few-shot
- MAML's task structure (K=5 support examples per class) directly matches the rare-class regime
- First-order MAML approximation fits in 8GB VRAM

**Concrete integration:**
```python
class MAMLVulnLearner:
    def __init__(self, model, inner_lr=1e-3, inner_steps=3):
        self.model = model
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps

    def inner_loop(self, support_x, support_y, class_idx):
        """Adapt model to a specific vulnerability class (K=5 examples)."""
        fast_weights = OrderedDict(self.model.named_parameters())
        for step in range(self.inner_steps):
            logits = self.model.functional_forward(support_x, fast_weights)
            loss = F.binary_cross_entropy_with_logits(
                logits[:, class_idx], support_y[:, class_idx].float()
            )
            grads = torch.autograd.grad(loss, fast_weights.values(), create_graph=True)
            fast_weights = OrderedDict(
                (name, param - self.inner_lr * grad)
                for (name, param), grad in zip(fast_weights.items(), grads)
            )
        return fast_weights
```

**Risk:** Medium. MAML requires second-order gradients (memory-intensive). Use first-order approximation for 8GB GPU. Requires `functional_forward` support in SentinelModel.

---

### 5.11 Cross-Modal Contrastive Alignment (Tier 2)

**What:** Explicitly align the GNN and Transformer embedding spaces using contrastive objectives. The same contract's graph and code embeddings should be close; different contracts' should be far.

**Why for SENTINEL:** Currently, the GNN and Transformer embeddings are only connected through `CrossAttentionFusion`. Adding an explicit alignment loss ensures the two modalities share a meaningful common space before fusion:

```python
def cross_modal_alignment_loss(gnn_graph_emb, transformer_graph_emb, temperature=0.07):
    """
    Align GNN and Transformer graph-level embeddings.
    gnn_graph_emb:        [B, 128] from gnn_eye_proj
    transformer_graph_emb: [B, 128] from transformer_eye_proj
    """
    g = F.normalize(gnn_graph_emb, dim=-1)
    t = F.normalize(transformer_graph_emb, dim=-1)
    sim = torch.mm(g, t.t()) / temperature
    labels = torch.arange(g.size(0), device=g.device)
    loss_g2t = F.cross_entropy(sim, labels)
    loss_t2g = F.cross_entropy(sim.t(), labels)
    return (loss_g2t + loss_t2g) / 2
```

Add as auxiliary loss (weight 0.1–0.3) alongside existing aux BCE losses. Complementary to Jakiro's cross-modal contrastive approach.

**Complexity:** Very Low. ~50 lines. No architecture changes.

---

### 5.12 MoE Classifier Head (Tier 3)

**What:** Replace the single `Linear(512, 9)` classifier with a Mixture of Experts layer where different experts specialize in different vulnerability types.

**Why for SENTINEL:** The single classifier must learn all 9 vulnerability patterns simultaneously. MoE allows dedicated expert capacity for each pattern type — one expert can focus on reentrancy (call-value loops) while another focuses on access control (modifier patterns).

```python
class MoEClassifier(nn.Module):
    def __init__(self, input_dim=512, num_classes=9, num_experts=8, top_k=2):
        super().__init__()
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, 256), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(256, num_classes),
            ) for _ in range(num_experts)
        ])
        self.gate = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(), nn.Linear(64, num_experts),
        )
        self.top_k = top_k

    def forward(self, x):
        gate_weights = torch.softmax(self.gate(x), dim=-1)
        topk_vals, topk_idx = gate_weights.topk(self.top_k, dim=-1)
        topk_vals = topk_vals / topk_vals.sum(dim=-1, keepdim=True)
        output = torch.zeros(x.size(0), self.experts[0][-1].out_features, device=x.device)
        for i in range(self.top_k):
            for e in range(len(self.experts)):
                mask = (topk_idx[:, i] == e)
                if mask.any():
                    output[mask] += topk_vals[mask, i].unsqueeze(-1) * self.experts[e](x[mask])
        return output, gate_weights  # gate_weights for load-balancing loss
```

**Risk:** Medium. Routing collapse possible — requires load-balancing auxiliary loss. Increases classifier parameters by ~4–8×.

---

### 5.13 Causal Attention / Information Bottleneck (Tier 4)

**What:** Replace standard attention with causal attention that learns to **disentangle true vulnerability indicators from spurious correlations**. ORACAL (2026) achieves this with causal attention + PGExplainer for subgraph-level explanations. ContractGIB (2026) uses Hilbert-Schmidt Information Bottleneck (HSIC) for stable dependence measurement.

**Why for SENTINEL:** The ExternalBug FP problem IS a spurious correlation. A causal model learns: "what makes a contract vulnerable is the CFG pattern, not the presence of `s_Form` in the identifier."

**ORACAL architecture (adapted):**
```
1. Extract CFG, DFG, Call Graph (already have similar)
2. RAG enrichment: query vector DB of known audit findings
3. Causal attention over heterogeneous graph
4. PGExplainer for subgraph-level explanation
5. Training with causal intervention objective
```

**Complexity:** Very High. Requires new causal attention module, RAG pipeline, PGExplainer integration. Defers to Tier 4.

---

### 5.14 Multi-Task Contract + Function-Level (Tier 3)

**What:** Jointly predict vulnerability at the **contract level** (current task) and **function level** (new). The Hierarchical Graph Transformer (2026) with uncertainty-weighted multi-task learning achieves **92.18% F1** — the best reported on smart contract benchmarks.

**Why for SENTINEL:**
- Function-level detection provides **localized explanation**: "this vulnerability is in function withdraw()"
- Multi-task learning regularises the contract-level task
- Homoscedastic uncertainty weighting: `L = L_c/σ_c² + L_f/σ_f² + log(σ_c·σ_f)`

---

### 5.15 Graph-Mamba for Phase 2 (Tier 4)

**What:** Replace the Phase 2 GAT layers (L3–L5) with Graph-Mamba (selective state space model) for O(N) global reasoning instead of O(N²) attention.

**Why for SENTINEL:** Phase 2 currently processes CFG edges with 3-hop GAT. Graph-Mamba would provide **unbounded-range** state propagation at linear complexity — every CFG node sees the entire execution path, not just 3 hops. This directly addresses the "local-only GNN reasoning" bottleneck.

**Complexity:** High. Requires new SSM-based layer implementation. Research phase.

---

### 5.16 Additional Quick-Win Methods

These methods require <1 day each and <100 lines of code:

| Method | LOC | Expected F1 Gain | Description |
|--------|-----|-------------------|-------------|
| **SWA** | ~20 | +1–3% | Stochastic Weight Averaging — average model weights over last N epochs |
| **Manifold Mixup** | ~10 | +1–2% | Interpolate four-eye embeddings before classifier |
| **Laplacian PE** | ~100 | +1–3% | Positional encodings for global graph topology |
| **Class-Conditional Label Smoothing** | ~30 | +0.5–2% | Per-class smoothing rates based on class frequency |
| **Focal Calibration** | ~50 | +0.5–1% | Class-aware temperature scaling |
| **MC Dropout** | ~30 | N/A (uncertainty) | Multiple stochastic forward passes at inference |
| **Curriculum Learning** | ~200 | +2–5% convergence | Easy→hard sample ordering |
| **Adversarial Training** | ~150 | +1–3% robustness | FGSM perturbations on graph node features |

---

## 6. SENTINEL-Specific Ranking

### Scoring Rubric

| Dimension | Scale | Weight |
|-----------|-------|--------|
| **F1 Impact** | 1–10 (estimated macro-F1 gain in pp) | 3× |
| **Complexity** | 1–10 (1=trivial, 10=months) | 2× (lower is better) |
| **Risk** | 1–10 (regression probability) | 2× (lower is better) |
| **VRAM Fit** | 1–10 (higher = fits 8GB better) | 1× |
| **Maintenance** | 1–10 (1=high ongoing cost) | 1× |

### Ranked List

| Rank | Method | F1 Impact | Complexity | Risk | VRAM | Maint. | **Score** |
|------|--------|-----------|------------|------|------|--------|-----------|
| 1 | Conformal Prediction | 3 | 1 | 1 | 10 | 10 | **44** |
| 2 | Quick Wins (SWA+Mixup+PE+CC-Smooth) | 5 | 1 | 1 | 10 | 10 | **44** |
| 3 | Graph Contrastive Pre-training | 8 | 3 | 3 | 9 | 8 | **40** |
| 4 | Structured Ensembling | 6 | 3 | 2 | 8 | 7 | **37** |
| 5 | Granular-ball (CGBC) | 7 | 5 | 3 | 8 | 6 | **31** |
| 6 | CPPO Fine-Tuning | 6 | 4 | 3 | 8 | 7 | **30** |
| 7 | UniXcoder upgrade | 4 | 2 | 2 | 10 | 9 | **30** |
| 8 | Cross-Modal Contrastive Alignment | 3 | 1 | 1 | 10 | 10 | **29** |
| 9 | F-EDL on aux heads | 4 | 4 | 3 | 9 | 7 | **28** |
| 10 | SAC Threshold Agent | 3 | 2 | 1 | 10 | 9 | **27** |
| 11 | SigGate-GT / GraphGPS | 6 | 7 | 5 | 6 | 6 | **23** |
| 12 | ETN (Evidential Transformation) | 3 | 3 | 2 | 10 | 9 | **23** |
| 13 | CodeSage-1.3B upgrade | 5 | 4 | 4 | 6 | 8 | **22** |
| 14 | Multi-task (contract + function) | 5 | 6 | 5 | 7 | 6 | **22** |
| 15 | MAML Few-Shot | 5 | 5 | 4 | 7 | 6 | **22** |
| 16 | MoE Classifier | 4 | 4 | 3 | 8 | 7 | **22** |
| 17 | Causal Attention (ORACAL) | 7 | 8 | 6 | 5 | 4 | **20** |
| 18 | Graph-Mamba for Phase 2 | 4 | 8 | 6 | 8 | 5 | **20** |
| 19 | Information Bottleneck (ContractGIB) | 5 | 7 | 5 | 7 | 5 | **20** |
| 20 | Adversarial Training | 2 | 2 | 2 | 10 | 8 | **18** |
| 21 | Knowledge Distillation | 2 | 4 | 2 | 9 | 7 | **15** |
| 22 | Qwen2.5-Coder-7B upgrade | 6 | 6 | 6 | 4 | 6 | **17** |
| 23 | RAG-augmented auditing | 5 | 9 | 7 | 4 | 3 | **14** |
| 24 | Continual Learning (EWC) | 2 | 3 | 2 | 8 | 7 | **15** |

### Tier Summary

| Tier | Methods | Total Effort | Expected F1 Gain | Decision Gate |
|------|---------|-------------|------------------|---------------|
| **🔥 Tier 1 (Run 14)** | CP + Quick Wins + Contrastive + Ensemble + CGBC + CPPO | ~14 days | +5–12 pp | After Run 13 confirms baseline |
| **🟡 Tier 2 (Run 15)** | UniXcoder + F-EDL + SAC Thresholds + Cross-Modal CL + ETN | ~8 days | +2–5 pp | After Tier 1 eval |
| **🟠 Tier 3 (Run 15–16)** | GraphGPS + Multi-task + CodeSage + MAML + MoE | ~18 days | +4–8 pp | After Tier 1+2 |
| **🔴 Tier 4 (Run 16+)** | Causal + Mamba + RAG + Qwen + Distillation + EWC | ~30 days | +5–15 pp | Research phase |

---

## 7. Implementation Roadmap

### Run 13 (Current — 4 Confirmed Fixes)

```
Goal: Establish clean baseline for Run 14+

1. Drop GasException → NUM_CLASSES=9
2. Extend L4 to drop `loc` feature
3. Strip Solidifi `bug_*` prefix
4. Inject 658 BCCC ME contracts
5. ExternalBug label quality review

Expected: F1 baseline after fixes
```

### Run 14 (Tier 1 — 6 Phases, ~14 days)

```
Phase 0: Quick Wins (Day 1)
  - Add SWA to trainer.py (~20 lines)
  - Add Manifold Mixup to sentinel_model.py (~10 lines)
  - Add Laplacian PE to gnn_encoder.py (~100 lines)
  - Add class-conditional label smoothing to trainer.py (~30 lines)
  - Add focal calibration to predictor.py (~50 lines)
  - Gate: No regression on Run 13 baseline F1

Phase A: Conformal Prediction (Day 2)
  - Implement split conformal prediction in predictor.py
  - Calibration set from validation split
  - Integrate into api.py (return prediction sets alongside probabilities)
  - Gate: CP coverage ≥ 95% on held-out test set

Phase B: Graph Contrastive Pre-training (Days 3–6)
  - Build unlabeled dataset (~25K contracts not in v3)
  - Implement 6 augmentation strategies for Solidity graphs
  - Pre-train GNNEncoder with InfoNCE (2 epochs ~4 hrs on RTX 3070)
  - Fine-tune full SENTINEL with pre-trained GNN weights
  - Gate: GNN share ≥ 15%, compare F1 with Run 13 baseline

Phase C: Structured Ensembling (Days 7–9)
  - Train GNN-only model (same GNN, no transformer)
  - Implement cascade: fast model → confidence check → full model
  - Train meta-classifier on validation set
  - Gate: Avg inference latency reduction ≥ 40%, no F1 regression

Phase D: Granular-ball Label Correction (Days 10–13)
  - Implement CGBC clustering on v3 training set
  - Generate corrected labels per granular ball
  - Re-train Run 13 with corrected labels
  - Gate: ExternalBug FP rate reduction ≥ 20% on OOD benchmark
  - Gate: No drift in other class F1 beyond ±0.02

Phase E: CPPO Fine-Tuning (Days 14–16)
  - Implement CPPOFineTuner with constrained optimization
  - Reward: rare-class F1 improvement; Constraints: protected metrics
  - Fine-tune from Run 14 best supervised checkpoint
  - Gate: F1(DoS) improvement ≥ 2pp, no regression on other classes
```

### Run 15 (Tier 2 — ~8 days + optional Tier 3)

```
Phase A: Upstream Models (Days 1–3)
  - Swap GraphCodeBERT → UniXcoder (Day 1)
  - Full eval: compare with Run 14 best checkpoint
  - Add cross-modal contrastive alignment loss (~50 lines)
  - Optional: CodeSage-1.3B with LoRA r=8

Phase B: Evidential Uncertainty (Days 4–6)
  - Replace aux heads with F-EDL evidential heads
  - Tune evidential loss weight
  - Add SAC threshold agent for dynamic threshold optimization
  - Gate: Uncertainty decomposition consistent with manual inspection

Phase C: SigGate-GT (Days 7–12, optional)
  - Replace GNNEncoder layers with GPS blocks + sigmoid gating
  - Full re-train required (architecture change)
  - Gate: F1 improvement ≥ 3pp over Run 14 best
```

### Run 16+ (Tier 3–4)

```
Phase A: MAML + MoE (Days 1–8)
  - Implement first-order MAML for few-shot rare-class adaptation
  - Replace Linear classifier with MoE classifier
  - Gate: F1(DoS) ≥ 0.50, no routing collapse

Phase B: Multi-task Learning (Days 9–14)
  - Add function-level vulnerability prediction
  - Uncertainty-weighted multi-task loss
  - Gate: Function-level F1 ≥ 0.60

Phase C: Causal Attention (Days 15–25, research)
  - ORACAL-style causal disentanglement
  - PGExplainer for subgraph explanations
  - RAG enrichment pipeline
  - Gate: ExternalBug precision ≥ 0.80, interpretable subgraph explanations
```

---

## 8. Risk & Trade-off Analysis

### 8.1 Key Risks

| Risk | Mitigation |
|------|-----------|
| **Contrastive pre-training may not transfer** | If pre-training on unlabeled Wild contracts produces task-agnostic features, the fine-tuned F1 may not improve. **Mitigation:** evaluate on the 66 honest OOD contracts after 2 epochs. If no improvement, abandon and move to other methods. |
| **CPPO reward shaping is wrong** | Bad reward → model optimizes wrong objective. **Mitigation:** start with simple reward (ΔF1 on rare classes), validate on held-out set, iterate. The constraints prevent catastrophic outcomes even with imperfect reward. |
| **Ensembling increases inference latency** | Cascade architecture mitigates: 80% of confident predictions use fast model only. |
| **CGBC over-corrects labels** | Validate corrected labels against 66 manually inspected OOD contracts. If correction changes a confirmed TP to negative, adjust clustering hyperparameters. |
| **SigGate-GT VRAM OOM** | Test on single batch first. If OOM at batch=8, reduce to batch=4 with grad accum ×16. If still OOM, defer to Run 16. |
| **UniXcoder worse than GraphCodeBERT** | Run A/B comparison on 66 OOD contracts. If worse, revert (1-line change). |
| **MAML second-order gradients OOM** | Use first-order MAML approximation. Accept ~10% quality loss for ~40% VRAM reduction. |

### 8.2 Trade-off Matrix

| Decision | Upside | Downside | Verdict |
|----------|--------|----------|---------|
| Pre-train vs from scratch | +3–8% F1, 2 extra days | May not transfer | **Pre-train** — the upside is too large to skip |
| CPPO after supervised vs ASL-only | +2–5% on rare classes | +3 days, reward shaping risk | **CPPO** — constraints make it safe |
| Small ensemble vs big single model | Better rare class F1, faster avg inference | 2× training, cascade complexity | **Ensemble** — production inference benefit justifies it |
| EDL vs CP for uncertainty | Per-eye decomposition (interpretability) | Architectural change vs post-hoc | **Both** — CP for guarantees, EDL for interpretability |
| GraphGPS vs Mamba | GPS is more proven, better integration | Mamba is O(n), more efficient, but newer | **GPS first** (proven), Mamba later (research) |
| UniXcoder vs CodeSage | UniXcoder = zero VRAM change | CodeSage gives +3–5% but needs ~1GB VRAM | **UniXcoder first**, CodeSage in Tier 3 |
| CGBC vs MAML for rare classes | CGBC fixes labels, MAML fixes learning | Different problems, complementary | **Both** — CGBC in Tier 1, MAML in Tier 3 |

---

## 9. What NOT to Do

1. **Don't replace the entire model architecture at once** — incremental changes allow attribution of F1 gains. Each Tier 1 method must be evaluated independently before combining.

2. **Don't adopt decoder-only LLMs as the primary encoder** — they are 10–100× larger for marginal understanding gains; encoder-only models remain optimal for classification. Decoder LLMs are Tier 4 experimental only.

3. **Don't add more than 1–2GB VRAM** — 8GB RTX 3070 is already near capacity with the full 4-eye model (~5.5GB during training). CodeSage-1.3B is the maximum acceptable model size.

4. **Don't use pure attention Graph Transformers without local MPNN** — GraphGPS shows hybrid (local+global) outperforms pure global. The three-phase GAT already encodes valuable local structure.

5. **Don't skip the contamination gate** — all future benchmarks must pass 0% contamination to avoid Run 10-style F1 inflation. The 17.4% contamination rate must be eliminated.

6. **Don't apply RL before supervised convergence** — PPO/CPPO must come AFTER the supervised model has converged. RL fine-tuning on an untrained model will diverge.

7. **Don't use MAML without first-order approximation** — second-order MAML requires ~3× the VRAM of normal training and will OOM on 8GB. Use first-order MAML.

8. **Don't mix Tier 1 and Tier 3 methods in the same run** — each run must isolate variables for clean F1 attribution.

---

## 10. References

### Smart Contract Vulnerability Detection (2025–2026)
- **ContractShield** (2026): `arXiv:2604.02771` — Hierarchical cross-modal fusion, obfuscation-robust
- **ORACAL** (2026): `arXiv:2603.28128` — Causal graph enrichment + RAG, 91.28% Macro F1
- **Hierarchical Graph Transformer + Community** (2026): `Expert Systems with Applications` — 92.18% F1
- **BugSweeper** (2026): `AAAI-26` — Function-Level Abstract Syntax Graphs, two-stage GNN
- **HOGAT** (2026): `Iran J Sci Technol Trans Electr Eng` — Higher-order attention, 89.8% F1
- **Smart-LLaMA-DPO** (2026): `arXiv:2506.18245` — LLM + DPO for vulnerability detection
- **SAGE-Prompt** (2026): `Expert Systems with Applications` — Graph-enhanced prompting for LLMs
- **CGBC** (2026): `arXiv:2603.27734` — Contrastive granular-ball training, noisy label correction
- **Jakiro** (2026): `ISC 2025` — Cross-modal contrastive learning (CFG + source)
- **ContractGIB** (2026): `CMC 87(2)` — HSIC Information Bottleneck + CodeBERT
- **BreachT5** (2025): TU Delft — CodeT5+ ensemble, Micro/Macro tradeoff

### Graph Neural Networks
- **GraphGPS** (2022): `arXiv:2205.12454` — Hybrid MPNN + Transformer
- **SigGate-GT** (2026): `arXiv:2604.17324` — Sigmoid gating to eliminate attention sinks
- **Graph-Mamba** (2024): `arXiv:2402.00789` — Selective SSM for graphs, O(n) complexity
- **MbaGCN** (2025): `arXiv:2501.15461` — Mamba-based GCN, avg rank 1.71
- **SBP** (2025): `arXiv:2502.11394` — Signed graph propagation, 300 layers
- **Multi-Track MPNN** (2024): `PMLR 235` — Separate channels per category semantics
- **GraphBFF** (2026): `arXiv:2602.04768` — Billion-scale graph foundation models

### Graph Contrastive Learning
- **MGCL** (2026): `arXiv:2506.06212` — Model-driven graphon-informed augmentations
- **FDAGCL** (2026): `Neural Processing Letters` — Feature discrepancy-aware
- **FOSSIL** (2025): `arXiv:2502.20885` — Fused Gromov-Wasserstein subgraph contrastive
- **IFL-GCL** (2025): `arXiv:2505.06282` — InfoNCE as free lunch, semantic sampling
- **GraphMAE** (2022): `KDD 2022` — Masked graph autoencoder

### Reinforcement Learning
- **PPO** (2017): `arXiv:1707.06347` — Proximal Policy Optimization
- **CPO/CPPO** (2017): `ICML 2017` — Constrained Policy Optimization
- **SAC** (2018): `ICML 2018` — Soft Actor-Critic
- **MAML** (2017): `ICML 2017` — Model-Agnostic Meta-Learning
- **DQN** (2015): `Nature 2015` — Deep Q-Network

### Code Models
- **CL4D** (2026): `AAAI-26` — Contrastive learning for decoder-only code models
- **UniXcoder** (2022): `ACL 2022` — Unified cross-modal pre-training for code
- **CodeSage** (2023): `arXiv:2302.05012` — Sparse attention code encoder
- **Qwen2.5-Coder** (2025): `arXiv:2509.12190` — 7B, beats GPT-4o on coding

### Uncertainty Quantification
- **CONFIDE** (2026): `arXiv:2604.08885` — Conformal prediction for transformers
- **ECP** (2024): `PMLR 230` — Evidential conformal prediction
- **F-EDL** (2025): `NeurIPS 2025` — Flexible evidential deep learning
- **ETN** (2026): `CVPR 2026` — Evidential transformation network (post-hoc)
- **CreDRO** (2026): `arXiv:2602.08470` — Distributionally robust credal prediction
- **Laplace-LoRA** (2024): `arXiv:2402.09368` — Laplace approximation for LoRA

### Training Paradigms
- **Mixup** (2018): `ICLR 2018` — Beyond empirical risk minimization
- **SWA** (2018): `UAI 2018` — Averaging weights leads to wider optima
- **Curriculum Learning** (2009): `ICML 2009` — Easy→hard sample ordering
- **EWC** (2017): `PNAS 2017` — Overcoming catastrophic forgetting
- **Prototypical Networks** (2017): `NeurIPS 2017` — Few-shot classification

---

## Appendix A: Cost Estimate

| Method | Engineer-Days | GPU Hours | Data Prep | Total $ |
|--------|--------------|-----------|-----------|---------|
| Quick Wins (SWA+Mixup+PE+CC-Smooth+Calib) | 1 | 2 | None | ~$100 |
| Conformal Prediction | 1 | 0 | None | ~$100 |
| Graph Contrastive Pre-training | 3 | 8 | 1 day | ~$500 |
| Structured Ensembling | 2 | 4 | None | ~$300 |
| Granular-ball (CGBC) | 4 | 6 | 2 days | ~$700 |
| CPPO Fine-Tuning | 3 | 6 | None | ~$400 |
| UniXcoder upgrade | 1 | 4 | None | ~$200 |
| F-EDL on aux heads | 2 | 4 | None | ~$300 |
| SAC Threshold Agent | 2 | 2 | None | ~$200 |
| Cross-Modal Contrastive | 1 | 2 | None | ~$150 |
| SigGate-GT / GraphGPS | 5 | 10 | None | ~$800 |
| CodeSage-1.3B upgrade | 2 | 6 | None | ~$400 |
| Multi-task learning | 4 | 8 | 1 day | ~$700 |
| MAML Few-Shot | 5 | 8 | None | ~$700 |
| MoE Classifier | 3 | 4 | None | ~$400 |
| Causal Attention (ORACAL) | 8 | 12 | 3 days | ~$1,500 |
| Graph-Mamba for Phase 2 | 6 | 8 | None | ~$900 |
| **Total (Tier 1 only)** | **14** | **26** | **3 days** | **~$2,100** |
| **Total (Tier 1+2)** | **22** | **42** | **3 days** | **~$3,200** |
| **Total (all tiers)** | **53** | **94** | **7 days** | **~$8,450** |

---

## Appendix B: Expected F1 Trajectory

```
F1 (tuned, macro)
0.80 │                                                        ★ (Run 16+ causal + Mamba)
     │                                                     ╱
0.75 │                                               ★ (Run 15 with GraphGPS + multi-task)
     │                                            ╱
0.70 │────★ (Run 12: 0.7004)                     ╱
     │        ╲                                 ╱
0.65 │         ★ (Run 13: 9-class, fixes)       ╱
     │              ╲                         ╱
0.60 │               ★───★ (Run 14: CP + contrastive + ensemble + CGBC + CPPO)
     │                    ╲                 ╱
0.55 │                     ★ (conservative) ╱
     │
     │    Run 12   Run 13   Run 14         Run 15   Run 16+
     │             (fixes)  (6 methods)    (arch)   (research)
```

**Confidence bands:**
- Run 13: 0.66–0.70 (fixes may shift baseline; 9 classes vs 10)
- Run 14 (conservative): 0.72–0.75 (Tier 1 methods deliver)
- Run 14 (optimistic): 0.75–0.78 (if contrastive pre-training transfers strongly)
- Run 15: 0.76–0.82 (with architecture upgrades)
- Run 16+: 0.80–0.85 (with causal + Mamba + all methods combined)

---

*End of Final Merged Proposal v2.0. Agreed by both authors as the definitive roadmap for SENTINEL next-generation ML methods.*
