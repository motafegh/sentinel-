
***

## PART 1: Training Dynamics

### 1.1 Self-Paced Learning

**The core idea:** In curriculum learning, *you* decide what's easy. In self-paced learning, *the model* decides what it's ready to learn, based on its own current loss per sample.

**Intuition:** Imagine a student who tells the teacher "I'm not ready for this problem yet — my confusion on it is too high." The teacher sets it aside and comes back later. That's self-paced learning.

**How the loss signals readiness:**

```python
# After each epoch, compute per-sample loss on the training set
per_sample_losses = []
model.eval()
with torch.no_grad():
    for batch in train_loader:
        logits = model(batch)
        losses = F.binary_cross_entropy_with_logits(
            logits, batch.labels, reduction='none'
        ).mean(dim=1)                    # [B] — one loss per sample
        per_sample_losses.extend(losses.tolist())

# Self-paced threshold: starts tight, relaxes over epochs
# λ (lambda) = the "patience" parameter — how much loss is acceptable
λ = base_λ * (1 + epoch * growth_rate)  # grows each epoch

# Only train on samples the model is "ready" for
eligible_mask = [i for i, loss in enumerate(per_sample_losses) if loss < λ]
```

**The λ schedule is everything.** If it grows too fast, you include hard samples too early (same as random). Too slow, and you never see hard samples. Typical: start λ at the 40th percentile of losses, grow to 100th percentile by epoch 60% of training.

**For Sentinel specifically:** Your rarest classes (suicide, tx.origin misuse) have very few positive examples. Random ordering means the model sees them occasionally and gets high loss on them — but those high-loss samples get *excluded* by self-paced learning early on, which is exactly right. The model builds competence on the common classes first, then self-selects the rare ones when it's ready. This is the opposite of what happens now, where rare-class samples create noisy gradients throughout training.

**Difference from curriculum:**

```
Curriculum:       YOU rank samples by complexity score (static, pre-computed)
Self-paced:       MODEL ranks samples by its own current loss (dynamic, per epoch)
Combined (best):  Use complexity as initial order, then switch to loss-based 
                  self-pacing after epoch 10
```

***

### 1.2 GraphMixup

**The core idea:** Create synthetic training examples by interpolating between two real contracts in embedding space, with interpolated labels.

**Why interpolation works:** Neural networks learn smoother decision boundaries when they've seen "in-between" examples. A contract that's 60% like a reentrancy-vulnerable contract and 40% like a safe contract should produce a prediction of `[0.6, 0, 0, ...]`. If the model has only seen hard 0/1 examples, it has no reason to produce smooth probabilities.

**Where to interpolate in Sentinel:** NOT on raw graphs (you can't average two graphs directly — different node counts, different topology). Instead, after Phase 1 of the GNN when you have node embeddings:

```python
# In forward() during training, after GNN Phase 1 completes:
if self.training and use_mixup:
    # Sample a mixing coefficient from Beta distribution
    # Beta(0.4, 0.4) gives values close to 0 or 1 most of the time,
    # with occasional values near 0.5 — keeps most samples "pure"
    lam = np.random.beta(0.4, 0.4)
    
    # Shuffle batch to get mixing partner for each sample
    B = graphs.num_graphs
    perm = torch.randperm(B)
    
    # Mix graph-level embeddings (after pooling, before eye projection)
    # gnn_pooled: [B, 512] — max+mean pooled GNN output
    gnn_pooled_mixed = lam * gnn_pooled + (1 - lam) * gnn_pooled[perm]
    
    # Mix labels
    labels_mixed = lam * labels + (1 - lam) * labels[perm]
    
    # Continue forward pass with mixed embeddings
    gnn_eye = self.gnn_eye_proj(gnn_pooled_mixed)
```

**For multi-label specifically:** Mixup is particularly powerful for multi-label because soft mixed labels `[0.6, 0, 0.4, 0, ...]` are more honest than hard labels for contracts that exhibit partial vulnerability patterns. A contract that almost has a reentrancy bug (has CALL but has a check that prevents exploitation) gets a mixed label that reflects this ambiguity.

**What it regularizes:** Mixup penalizes the model for being overconfident between training examples. The embedding space between two different contracts must also be mapped to sensible intermediate predictions — this prevents the sharp, arbitrary decision boundaries that cause poor generalization.

***

### 1.3 Stochastic Weight Averaging (SWA)

**The core idea:** Your optimizer (AdamW) finds a single point in weight space. SWA instead maintains a running average of weights across many checkpoints, which typically lands in a flatter, wider region of the loss landscape.

**Why flat minima generalize better:** Sharp minima (narrow valleys in loss space) are sensitive to small distribution shifts — a contract from a slightly different Solidity version produces slightly different features and falls outside the sharp valley. Wide flat minima are insensitive to small perturbations — the prediction barely changes.

```
Sharp minimum:          Flat minimum (SWA):
    │                       ─────────────
    ▼                       
   ───                   
Loss spikes if you       Loss stays low even if 
move slightly            features shift slightly
```

**Implementation — fits perfectly into your existing trainer:**

```python
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn

# Wrap your model once at training start
swa_model = AveragedModel(sentinel_model)

# Use a cyclic LR schedule — SWA works by cycling LR to explore weight space
swa_scheduler = SWALR(optimizer, swa_lr=1e-5, anneal_epochs=5)

# In your epoch loop, after the warmup phase (e.g., after epoch 20):
for epoch in range(swa_start_epoch, total_epochs):
    train_one_epoch(sentinel_model, ...)
    
    swa_model.update_parameters(sentinel_model)  # accumulate weights
    swa_scheduler.step()

# After training: update BatchNorm statistics for the averaged model
update_bn(train_loader, swa_model)

# Save swa_model instead of sentinel_model
torch.save(swa_model.state_dict(), "sentinel_swa.pt")
```

**The cyclic LR is key:** SWA works by running the LR in cycles (low → high → low repeatedly). Each cycle's end point is a different local minimum — SWA averages them all. You're already using a warmup + decay schedule; you'd add cycling in the final 20–30% of training.

**Expected gain:** +1–3 F1 points on out-of-distribution contracts (contracts from DeFi protocols not well-represented in training). Near-zero implementation risk.

***

## PART 2: Data & Label Quality

### 2.1 Label Smoothing

**The core problem it solves:** Your Slither-generated labels are not ground truth. Slither has false negatives — it misses vulnerabilities. So a label of `0` for reentrancy doesn't mean "definitely not reentrancy," it means "Slither didn't detect reentrancy." Your model trains on these hard zeros with full confidence, learning to be certain about things that are uncertain.

**How smoothing encodes this uncertainty:**

```python
# Hard label: contract is NOT vulnerable to reentrancy
y = torch.tensor([0., 1., 0., 0., 0., 0., 0., 0., 0., 0.])

# Smoothed label (ε=0.05): "probably not, but 5% chance we're wrong"
ε = 0.05
y_smooth = y * (1 - ε) + ε / num_classes
# y_smooth = [0.005, 0.955, 0.005, 0.005, ...]
#             ↑ no longer exact zero — Slither might have missed it
```

**In your loss function — one line change:**

```python
# Current:
loss = F.binary_cross_entropy_with_logits(logits, labels)

# With label smoothing:
ε = 0.05
smooth_labels = labels * (1 - ε) + ε * 0.5  # 0.5 = uniform for binary case
loss = F.binary_cross_entropy_with_logits(logits, smooth_labels)
```

**Important tuning note:** Use different ε values per class. For classes where Slither is highly reliable (integer overflow — easy to detect), use `ε=0.02`. For classes where Slither misses a lot (access control subtleties), use `ε=0.10`. This encodes your domain knowledge about tool reliability directly into training.

***

### 2.2 Pseudo-Labeling

**The core idea:** Your trained model is reasonably good. Use it to label contracts it's very confident about, then add those as new training data.

**The self-training loop:**

```python
# Round 1: Train on your labeled dataset D_L → Model M1
# Round 2: Use M1 to label unlabeled dataset D_U
# Round 3: Train on D_L + high-confidence subset of D_U → Model M2
# Round 4: Repeat (each round improves)

# Step 1: Get predictions on unlabeled contracts (e.g., from Etherscan)
model.eval()
all_probs = []
with torch.no_grad():
    for batch in unlabeled_loader:
        probs = torch.sigmoid(model(batch))   # [B, 10]
        all_probs.append(probs)
all_probs = torch.cat(all_probs)              # [N_unlabeled, 10]

# Step 2: Filter by confidence
# Only use samples where max class probability is very high
# OR where all probabilities are very low (confident negative)
max_probs = all_probs.max(dim=1).values
confident_positive = max_probs > 0.85         # model is sure it's vulnerable
confident_negative = (all_probs < 0.1).all(dim=1)  # model is sure it's safe

pseudo_mask   = confident_positive | confident_negative
pseudo_labels = all_probs[pseudo_mask].round()   # binarize
pseudo_data   = unlabeled_contracts[pseudo_mask]

# Step 3: Add to training set at 20% mixing ratio
# Use a combined DataLoader that draws 80% real, 20% pseudo
```

**The danger:** If your model has systematic errors (e.g., consistently misses tx.origin misuse), pseudo-labeling amplifies those errors. The 0.85 confidence threshold mitigates this but doesn't eliminate it. Mitigation: never pseudo-label rare classes — only use pseudo-labels for classes where your model already has F1 > 0.75.

***

### 2.3 Contrastive Learning (Graph Contrastive Pre-training)

**The core idea:** Before any supervised training, teach the GNN to produce similar embeddings for augmented versions of the same contract, and dissimilar embeddings for different contracts. This is **unsupervised pre-training** — no labels needed.

**Why it helps:** Your GNN currently initializes randomly and learns purely from supervised loss. Contrastive pre-training gives it a strong inductive bias about what makes two contracts "the same thing" — same vulnerability structure, same overall shape — before it ever sees a label.

**Two augmentation strategies for Solidity graphs:**

```python
def augment_graph(graph, strategy):
    if strategy == "edge_drop":
        # Drop 10% of non-CFG edges (AST edges, CALLS edges)
        # CFG edges are preserved — they carry vulnerability-critical ordering
        non_cfg_mask = ~is_cfg_edge(graph.edge_attr)
        drop_mask = torch.rand(non_cfg_mask.sum()) > 0.10
        keep_mask = non_cfg_mask.clone()
        keep_mask[non_cfg_mask] = drop_mask
        keep_mask |= is_cfg_edge(graph.edge_attr)  # always keep CFG
        return graph.edge_subgraph(keep_mask)
    
    elif strategy == "feature_mask":
        # Zero out 15% of node features (not feat[0] = type_id)
        mask = torch.rand_like(graph.x) > 0.15
        mask[:, 0] = True    # always preserve type_id
        return graph._replace(x=graph.x * mask)
```

**The InfoNCE contrastive loss:**

```python
# For a batch of B contracts:
# aug1[i] and aug2[i] are two views of the SAME contract i
# aug1[i] and aug2[j] (i≠j) are views of DIFFERENT contracts

z1 = gnn_encoder(aug1_batch)   # [B, gnn_hidden_dim] — graph embeddings
z2 = gnn_encoder(aug2_batch)   # [B, gnn_hidden_dim]

# Normalize
z1 = F.normalize(z1, dim=1)
z2 = F.normalize(z2, dim=1)

# Similarity matrix [B, B]
sim_matrix = torch.mm(z1, z2.T) / temperature   # temperature=0.07

# Positive pairs are on the diagonal (same contract)
labels = torch.arange(B)
loss = F.cross_entropy(sim_matrix, labels)
```

**Pre-training workflow for Sentinel:**

```
Phase 0 (new): Contrastive pre-training on 100k+ unlabeled contracts
               GNN only, no transformer, no labels
               ~10 epochs, saves gnn_pretrained.pt

Phase 1 (existing): Load gnn_pretrained.pt as GNN init
                    Full supervised training with all four eyes
                    GNN starts from informed weights, not random
```

***

## PART 3: Architecture Improvements

### 3.1 Label Dependency Graph (LGNN)

**The core problem:** Your classifier treats the 10 vulnerability classes as completely independent. But reentrancy and unchecked-calls co-occur. Access-control and tx.origin co-occur. The classifier has no way to express "if I think this contract has reentrancy, I should also look harder for unchecked-calls."

**Computing the co-occurrence matrix:**

```python
# From your training labels [N, 10]
co_occur = torch.mm(train_labels.T, train_labels).float()  # [10, 10]
# Normalize to probabilities
co_occur = co_occur / co_occur.diagonal().unsqueeze(1)     # P(j | i)
# co_occur[i,j] = P(vulnerability j present | vulnerability i present)

# Example result:
# co_occur[reentrancy, unchecked_calls] = 0.73   (73% of reentrancy contracts also have unchecked calls)
# co_occur[timestamp, overflow]         = 0.12   (weakly correlated)
```

**The label GCN layer — added after the main classifier:**

```python
class LabelGCN(nn.Module):
    def __init__(self, num_classes, label_adj):
        super().__init__()
        self.register_buffer("adj", label_adj)   # [10, 10] fixed co-occurrence matrix
        self.W = nn.Linear(num_classes, num_classes, bias=False)
    
    def forward(self, logits):
        # logits: [B, 10]
        # Propagate label signals through co-occurrence graph
        # Each class aggregates evidence from correlated classes
        refined = torch.mm(logits, self.adj)     # [B, 10] — neighbor aggregation
        refined = self.W(refined)                # [B, 10] — learned transformation
        return logits + refined                  # residual — don't override, supplement

# In SentinelModel.forward():
logits = self.classifier(combined)              # [B, 10] — four-eye output
logits = self.label_gcn(logits)                 # [B, 10] — dependency-refined
```

**Tiny parameter count, meaningful semantic addition.** This is one of the highest ROI changes available because it fixes a known architectural limitation (independence assumption) with ~100 parameters.

***

### 3.2 Hierarchical Classification Head

**The core idea:** Group the 10 classes into vulnerability families, classify at family level first, then refine within family.

**Your natural groupings:**

```
Family 1 — Reentrancy:        reentrancy, cross-function reentrancy
Family 2 — Access Control:    access_control, tx.origin, ownership
Family 3 — Arithmetic:        integer_overflow, integer_underflow  
Family 4 — Dangerous Actions: suicide, unchecked_calls, timestamp_dependence
```

**The architecture:**

```python
class HierarchicalHead(nn.Module):
    def __init__(self, eye_dim=512, num_families=4, num_classes=10):
        super().__init__()
        # Level 1: classify into families
        self.family_head = nn.Linear(eye_dim, num_families)
        
        # Level 2: per-family classifiers
        # Each takes combined embedding + family logit as input
        self.class_heads = nn.ModuleList([
            nn.Linear(eye_dim + num_families, classes_in_family[f])
            for f in range(num_families)
        ])
    
    def forward(self, combined_embedding):
        # Level 1
        family_logits = self.family_head(combined_embedding)    # [B, 4]
        
        # Level 2: each family head sees the embedding + family-level context
        enriched = torch.cat([combined_embedding, family_logits], dim=1)  # [B, 516]
        
        class_outputs = []
        for head in self.class_heads:
            class_outputs.append(head(enriched))               # [B, n_classes_in_family]
        
        return torch.cat(class_outputs, dim=1)                  # [B, 10]
```

**Why rare classes benefit most:** `suicide` (rare) is in the same family as `unchecked_calls` (more common). When the family head learns "Dangerous Actions," it propagates gradient to the `suicide` head even on samples that only have `unchecked_calls` labels. The rare class gets indirect supervision it wouldn't get with a flat classifier.

***

### 3.3 Monte Carlo Dropout Uncertainty

**The core idea:** During inference, run the model N times with dropout randomly active each time. The variance across runs measures the model's uncertainty.

**Two kinds of uncertainty this captures:**

- **Aleatoric uncertainty** (irreducible): The contract itself is ambiguous — even a perfect model would be unsure. E.g., a pattern that's only vulnerable in specific deployment contexts.
- **Epistemic uncertainty** (reducible): The model hasn't seen enough contracts like this one. High epistemic uncertainty = "flag for human review, this is out-of-distribution."

```python
def predict_with_uncertainty(model, graphs, input_ids, mask, n_samples=30):
    model.train()   # IMPORTANT: activate dropout
    
    predictions = []
    with torch.no_grad():
        for _ in range(n_samples):
            logits = model(graphs, input_ids, mask, return_aux=False)
            probs  = torch.sigmoid(logits)         # [B, 10]
            predictions.append(probs)
    
    predictions = torch.stack(predictions)         # [30, B, 10]
    
    mean_probs   = predictions.mean(0)             # [B, 10] — final prediction
    uncertainty  = predictions.std(0)              # [B, 10] — per-class uncertainty
    
    # Flag high-uncertainty predictions for human review
    needs_review = (uncertainty > 0.15).any(dim=1) # [B] boolean
    
    return mean_probs, uncertainty, needs_review
```

**For a security tool, this is transformative.** Instead of just "vulnerable / not vulnerable," Sentinel can output: "reentrancy: 0.87 (±0.03 — high confidence)" vs "access_control: 0.61 (±0.19 — uncertain, recommend human review)." This is the difference between a research model and a professional security tool.

***

## PART 4: Regularization

### 4.1 R-Drop

**The core idea:** Every forward pass through a model with dropout produces a slightly different probability distribution (because different neurons are dropped). R-Drop says these two distributions should be consistent — the model shouldn't change its mind just because of which neurons dropout happened to zero out.

**The KL divergence loss:**

```python
# In training loop — two forward passes on SAME batch:
logits1 = model(graphs, input_ids, mask)      # dropout mask A
logits2 = model(graphs, input_ids, mask)      # dropout mask B (different)

p1 = torch.sigmoid(logits1)   # [B, 10]
p2 = torch.sigmoid(logits2)   # [B, 10]

# Symmetric KL divergence
# KL(p1 || p2): how much info is lost when using p2 to approximate p1
kl_1_2 = F.kl_div(p1.log(), p2, reduction='batchmean')
kl_2_1 = F.kl_div(p2.log(), p1, reduction='batchmean')
rdrop_loss = (kl_1_2 + kl_2_1) / 2.0

# Combine with task loss (average logits for task loss to use both passes)
logits_avg = (logits1 + logits2) / 2.0
task_loss  = criterion(logits_avg, labels)

total_loss = task_loss + α * rdrop_loss   # α=0.5 is typical
```

**Why it's so effective for Sentinel:** Your transformer eye (CodeBERT + LoRA) has dropout in the LoRA paths AND in the window attention pooler. With standard training, the model learns to rely on whichever neurons happen to survive dropout most often. R-Drop forces every neuron to contribute robust signal independently. The transformer eye specifically benefits because it's the most complex path with the most dropout stochasticity.

**Cost:** ~40% more training time per epoch (two forward passes). Worth it — R-Drop has shown consistent +1–3% improvements on classification tasks across many domains.

***

### 4.2 Adversarial Training on Token Embeddings

**The core idea:** Add small adversarial perturbations to CodeBERT's token embeddings during training. The model learns to be robust to small embedding-space perturbations, which correspond to code style variations in the real world.

**What "perturbation" means in code space:** If you perturb the embedding of the token `transfer` by a small vector, you're asking "what if this token was slightly semantically different?" The model that's trained to be robust to this perturbation will give consistent predictions regardless of whether the function is named `transfer`, `Transfer`, `sendTokens`, or `doTransfer` — which are all semantically identical but look different to CodeBERT's tokenizer.

```python
def fgsm_attack(model, graphs, input_ids, mask, labels, epsilon=1e-3):
    # Get token embeddings (requires accessing CodeBERT's embedding layer)
    token_embs = model.transformer.bert.embeddings(input_ids)  # [B, L, 768]
    token_embs.requires_grad_(True)
    
    # Forward pass with current embeddings
    logits = model.forward_from_embeddings(graphs, token_embs, mask)
    loss   = criterion(logits, labels)
    loss.backward()
    
    # FGSM: step in gradient direction (worst case perturbation)
    delta = epsilon * token_embs.grad.sign()   # [B, L, 768]
    
    # Forward pass with perturbed embeddings
    adv_logits = model.forward_from_embeddings(
        graphs, (token_embs + delta).detach(), mask
    )
    adv_loss = criterion(adv_logits, labels)
    
    return adv_loss
```

**The training loop adds this as an auxiliary loss:**

```python
normal_loss = criterion(model(graphs, input_ids, mask), labels)
adv_loss    = fgsm_attack(model, graphs, input_ids, mask, labels)
total_loss  = normal_loss + 0.3 * adv_loss
```

***

## PART 5: Evaluation & Production

### 5.1 Temperature Scaling

**The core problem:** `torch.sigmoid(logit)` gives you a number between 0 and 1, but that number is not a calibrated probability. A logit of 2.0 → sigmoid → 0.88, but that doesn't mean the model is right 88% of the time when it outputs 0.88. Your model is likely overconfident — when it says 0.9, it's right maybe 75% of the time.

**Temperature scaling fixes this in 5 minutes:**

```python
class TemperatureScaler(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1))  # starts at T=1 (no change)
    
    def forward(self, *args, **kwargs):
        logits = self.model(*args, **kwargs)
        return logits / self.temperature   # scale logits by learned T

# Post-training calibration (validation set only, model weights FROZEN):
calibration_model = TemperatureScaler(trained_sentinel)
optimizer = torch.optim.LBFGS([calibration_model.temperature], lr=0.01, max_iter=100)

def calibration_step():
    optimizer.zero_grad()
    logits = calibration_model(val_graphs, val_input_ids, val_mask)
    loss   = F.binary_cross_entropy_with_logits(logits, val_labels)
    loss.backward()
    return loss

optimizer.step(calibration_step)
# T > 1: model was overconfident (probabilities compressed toward 0.5)
# T < 1: model was underconfident (probabilities pushed toward extremes)
print(f"Learned temperature: {calibration_model.temperature.item():.3f}")
```

**Takes 10 minutes to implement, improves every downstream metric that uses probabilities.**

***

### 5.2 GNNExplainer / Attention Visualization

**The core idea:** After Sentinel predicts "reentrancy vulnerability," it should be able to say *why* — which specific nodes and edges in the contract graph drove that prediction.

**GNNExplainer learns a mask over edges and node features:**

```python
from torch_geometric.explain import Explainer, GNNExplainer

explainer = Explainer(
    model=sentinel_model,
    algorithm=GNNExplainer(epochs=200),
    explanation_type='model',
    node_mask_type='attributes',
    edge_mask_type='object',
    model_config=dict(
        mode='multiclass_classification',
        task_level='graph',
        return_type='log_probs',
    ),
)

# For a specific contract prediction:
explanation = explainer(
    x=graphs.x,
    edge_index=graphs.edge_index,
    batch=graphs.batch,
    # Additional args for your four-eye model
)

# explanation.node_mask:  [N, feat_dim] — which node features mattered
# explanation.edge_mask:  [E] — which edges mattered (0=irrelevant, 1=critical)

# Visualize: highlight the subgraph that caused the prediction
important_edges = graphs.edge_index[:, explanation.edge_mask > 0.5]
# Map back to Solidity: these edges correspond to specific function calls,
# state reads/writes, or CFG transitions in the original source code
```

**What this produces for reentrancy:** The explainer should highlight the path: `CHECK_NODE → CALL_NODE → WRITE_NODE` where the check comes after the external call — the classic CEI violation pattern. If it does, your model has genuinely learned the vulnerability pattern. If it highlights random nodes, your model is using spurious correlations.

**This is the most important long-term addition for Sentinel as a professional tool.** Auditors don't trust black boxes. A model that can say "line 47's external call happens before line 52's state update, creating a reentrancy window" is a tool that gets used.

***

## Complete  Roadmap

```
TODAY (no retraining)
├── Temperature scaling          → better calibrated probabilities
├── MC Dropout uncertainty       → uncertainty scores on current model
└── Optuna thresholds            → per-class threshold optimization

NEXT RUN (architecture additions, low risk)
├── Label smoothing              → honest training signal
├── Label dependency graph       → capture class co-occurrence
├── R-Drop                       → stronger transformer eye regularization
└── SWA                          → flatter loss minima, better generalization

DEDICATED RUNS (higher effort, high payoff)
├── Self-paced learning          → model-driven curriculum
├── GraphMixup                   → synthetic augmentation in embedding space
├── Adversarial token training   → robust to code style variations
└── Hierarchical class head      → rare class indirect supervision

LONGER HORIZON (infra investment needed)
├── Contrastive pre-training     → unlabeled Etherscan contracts
├── Pseudo-labeling              → self-training expansion
└── GNNExplainer integration     → production-grade audit trails
```

Every single one of these is either a well-established technique in the ML literature or directly analogous to methods proven on graph-based security tasks. None require changing your core four-eye architecture.