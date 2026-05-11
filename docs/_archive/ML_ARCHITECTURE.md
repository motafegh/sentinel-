# SENTINEL ML Architecture — Technical Reference

## Overview

SENTINEL is a dual-path neural network that scores Solidity smart contracts for vulnerabilities.
The two paths process complementary representations of the same source code:

- **GNN path** — captures the structural relationships in the contract's AST/CFG graph
- **CodeBERT path** — captures the semantic meaning of the code as text

Both representations are fused into a 64-dimensional vector and classified with a linear head that
produces a single scalar: the probability that the contract contains **any** vulnerability (binary
classification, not per-type).

---

## Data Source

**Primary dataset:** BCCC-SCsVul-2024 — 111,798 Solidity contracts across 12 vulnerability
categories (11 types + NonVulnerable). The contracts are organised as subfolders under
`BCCC-SCsVul-2024/SourceCodes/`: CallToUnknown, DenialOfService, ExternalBug, GasException,
IntegerUO, MishandledException, Reentrancy, Timestamp, TransactionOrderDependence, UnusedReturn,
WeakAccessMod, and NonVulnerable.

**Critical property:** 41.2% of contracts appear in **multiple** vulnerability folders — the same
file is labelled with 2 to 9 simultaneous vulnerability types. BCCC is genuinely a multi-label
dataset. The current training pipeline simplifies this to binary (safe / vulnerable) by picking one
label per unique hash. See ML_DATASET_PIPELINE.md for full details.

**Training set size after preprocessing:** 68,555 paired samples (graph + token files), after
~24,000 contracts were dropped due to Slither parse failures.

---

## Why Binary Classification (Not Multi-Class or Multi-Label)

The model outputs one number `∈ [0, 1]`: the probability that the contract is vulnerable at all.
It does **not** predict which of the 11 vulnerability types is present. This was a deliberate
choice made during the preprocessing design phase:

1. **The preprocessing simplified multi-label to binary from the start.** When the label-building
   script encountered the same contract hash in multiple folders, it kept only the first folder's
   label and discarded the rest. This decision collapsed a rich multi-label problem into a single
   primary label per contract, and then further to binary.

2. **The severe per-type class imbalance makes multi-class hard.** After deduplication: Reentrancy
   has ~122 samples, DenialOfService ~138, while IntegerUO has ~16K. A 12-class model would
   require aggressive per-class oversampling or loss weighting to avoid the rare classes being
   completely ignored.

3. **"Is this contract vulnerable at all?" is the primary oracle question.** For on-chain audit
   registries and quick deployment checks, a binary risk score is immediately actionable.

4. **Binary kept the ZK proof simple.** A scalar output `[B,1]` maps to a single public signal in
   the EZKL circuit. Multi-class output would require K public signals and a larger verifier
   contract. (Note: this was a consequence of the binary decision, not its cause — the ZKML module
   was built after the ML pipeline.)

**Upgrade path:** The CSV (`contract_labels_correct.csv`) retains `class_label` (0–11) and
Class01–12 one-hot columns for all 44,442 samples. Upgrading to 12-class requires rebuilding
`graph.y` from the CSV, changing the classifier head to `Linear(64→12)`, and rebuilding the EZKL
circuit. See ML_DATASET_PIPELINE.md → "Upgrade Path" section.

---

## End-to-end data flow

```
Solidity source (.sol file)
    │
    ├──── AST extraction (Slither) ──► PyG Data(x=[N,8], edge_index=[2,E])
    │                                          │
    │                                    GNNEncoder
    │                                    3×GAT layers
    │                                    global_mean_pool
    │                                          │ [B, 64]
    │                                          ▼
    └──── CodeBERT tokenizer ──────► [B, 512] input_ids
                                              │
                                       TransformerEncoder
                                       (frozen CodeBERT)
                                       CLS token extraction
                                              │ [B, 768]
                                              ▼
                                         FusionLayer
                                         concat → MLP(832→256→64)
                                              │ [B, 64]
                                              ▼
                                         Classifier
                                         Linear(64→1) → Sigmoid
                                              │ [B] float in [0, 1]
                                              ▼
                                    threshold (0.50) → "vulnerable" | "safe"
```

---

## Model architecture (LOCKED — do not change)

The architecture is frozen post-training. Changing any layer size:
1. Changes the circuit structure if the proxy model is involved
2. Invalidates the EZKL proving/verification keys
3. Requires full retraining from scratch

### GNNEncoder

| Component | Configuration |
|---|---|
| Input | `x: [N, 8]` — 8 float features per AST node |
| Layer 1 | `GATConv(8 → 64, heads=8, concat=True)` |
| Layer 2 | `GATConv(64 → 64, heads=8, concat=True)` |
| Layer 3 | `GATConv(64 → 64, heads=1, concat=False)` |
| Pooling | `global_mean_pool(x, batch)` — one vector per graph |
| Output | `[B, 64]` |
| Dropout | `p=0.2` between layers (disabled during eval) |

**Why GAT (Graph Attention Network)?**
GAT's attention mechanism learns which neighboring nodes to pay attention to when aggregating. In a function call graph, this means the model can learn that state variable reads matter more than events for predicting reentrancy, for example — something a fixed-weight GCN cannot do.

**Why global_mean_pool?**
We need one vector per graph (one per contract), not one per node. `global_mean_pool` averages all node embeddings, weighted equally. Alternative: `global_max_pool` (takes the most activated feature), `global_add_pool` (sums, scales with graph size). Mean pool is the most stable across variable-size graphs.

**Node feature vector (8 dims):**
```
Index  Feature      Values
─────  ──────────── ─────────────────────────────────────
  0    type_id      0=STATE_VAR, 1=FUNCTION, 2=MODIFIER,
                    3=EVENT, 4=FALLBACK, 5=RECEIVE, 6=CONSTRUCTOR, 7=CONTRACT
  1    visibility   0=public/external, 1=internal, 2=private
  2    pure         1.0 if pure function, 0.0 otherwise
  3    view         1.0 if view function, 0.0 otherwise
  4    payable      1.0 if payable, 0.0 otherwise
  5    reentrant    1.0 if is_reentrant (Slither flag), 0.0 otherwise
  6    complexity   float(len(func.nodes)) — CFG node count
  7    loc          float(len(source_mapping.lines)) — lines of code
```

**Critical invariant:** This exact 8-dim vector was used to build all 68,555 training `.pt` files. Any change requires rebuilding the dataset and retraining from scratch.

### TransformerEncoder

| Component | Configuration |
|---|---|
| Backbone | `microsoft/codebert-base` (RoBERTa, 12 layers) |
| Input | `[B, 512]` token ids + attention mask |
| Output | CLS token: `[:, 0, :]` from last hidden state |
| Shape | `[B, 768]` |
| Training | Fully **frozen** — no gradient flows through CodeBERT |

**Why frozen?**
- CodeBERT has 124.6M parameters. Fine-tuning it would require ~10× more GPU memory and training time.
- CodeBERT was pre-trained on millions of code files — its representations are already rich for Solidity. The task is discrimination, not semantic understanding.
- Freezing prevents catastrophic forgetting of CodeBERT's general code understanding.

**What the CLS token represents:**
BERT/RoBERTa is trained with a pooled representation at `[CLS]` (position 0) that captures the "global" meaning of the sequence. For classification tasks, extracting this token is standard practice and equivalent to using the model as a feature extractor.

**512-token limit:**
CodeBERT's positional embeddings are fixed at 512 positions. Contracts longer than 512 tokens are truncated. The first 512 tokens (beginning of file) are kept. Long contracts lose their tail — inference marks these with `truncated=True`.

### FusionLayer

| Component | Configuration |
|---|---|
| Input | `gnn_out: [B, 64]` + `transformer_out: [B, 768]` |
| Concat | `[B, 832]` |
| Layer 1 | `Linear(832 → 256)` → `ReLU` → `Dropout(0.3)` |
| Layer 2 | `Linear(256 → 64)` → `ReLU` |
| Output | `[B, 64]` |

**Why concat + MLP?**
The simplest fusion method that allows the model to learn cross-modal interactions. Alternative: attention-based fusion (more complex, marginal improvement for this task). The Dropout(0.3) regularises the larger first layer and prevents overfitting to one modality.

### Classifier

```python
Linear(64 → 1) → Sigmoid → squeeze(1) → [B] float in [0, 1]
```

Sigmoid produces a calibrated probability. **BCELoss is used, not BCEWithLogitsLoss** — because the sigmoid is already inside the model. Do not apply sigmoid again in the loss function.

### Parameter counts

| Component | Trainable | Frozen |
|---|---|---|
| GNNEncoder | 124,928 | 0 |
| TransformerEncoder | 0 | 124,645,632 |
| FusionLayer | 111,168 | 0 |
| Classifier | 65 | 0 |
| **Total** | **239,041** | **124,645,632** |

---

## Loss function: Focal Loss

```python
focal_loss = alpha_t × (1 - pt)^gamma × BCE
```

**Parameters:**
- `gamma = 2.0` — original paper default. Multiplies loss by `(1 - pt)^2`, where `pt` is the probability of the true class. Easy examples (high `pt`) get a near-zero loss, forcing the model to focus on hard examples.
- `alpha = 0.25` — confirmed correct. Applied to `label=1` (vulnerable, **majority** at 64.33%). `1 - alpha = 0.75` applied to `label=0` (safe, minority at 35.67%).

**Why alpha=0.25 when vulnerable is the majority?**
Higher `alpha` means higher weight. We want to down-weight the vulnerable class (majority) and up-weight the safe class (minority). So:
- `vulnerable: weight = alpha = 0.25` (down-weighted)
- `safe: weight = 1 - alpha = 0.75` (up-weighted)

The weight ratio 0.75/0.25 = 3.0× is deliberately stronger than the class imbalance ratio of 1.8×. Combined with `gamma=2.0`, this pushes the model to pay special attention to safe contracts (the harder minority class).

---

## Training

### Data splits

| Split | Samples | Class ratio preserved |
|---|---|---|
| Train | 47,988 (70%) | Yes (stratified) |
| Val | 10,283 (15%) | Yes |
| Test | 10,284 (15%) | Yes |
| **Total** | **68,555** | — |

Class distribution: 64.33% vulnerable, 35.67% safe. The 70/15/15 split preserves this ratio in all three splits (stratified sampling, seed=42).

### Optimizer

```python
AdamW(
    params=filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-4,
    weight_decay=1e-2,
)
```

`filter(requires_grad)` excludes the 124.6M frozen CodeBERT parameters — we don't want AdamW maintaining gradient statistics for parameters that never update.

### Training loop

```
for epoch in 1..N:
    model.train()
    for batch in train_loader:
        graphs, tokens, labels = batch          # dual_path_collate_fn output
        labels = labels.float().view(-1)        # [B] float — required by FocalLoss
        predictions = model(graphs, ...)        # [B] sigmoid output
        loss = focal_loss(predictions, labels)
        loss.backward()
        optimizer.step()

    model.eval()
    val_metrics = evaluate(model, val_loader)

    if val_metrics["f1_macro"] > best_f1:
        save_checkpoint(...)
```

### Checkpoint format (new format, April 2026+)

```python
{
    "model":     model.state_dict(),   # weights — used by predictor.py
    "optimizer": optimizer.state_dict(), # AdamW state — for resume
    "epoch":     epoch,                  # which epoch was saved
    "best_f1":   best_f1,               # best F1-macro achieved
    "config":    dataclasses.asdict(config),  # all hyperparameters
}
```

Old checkpoints (pre-April 2026) are plain `state_dict()` tensors. Both formats are supported by `predictor.py` and `tune_threshold.py`.

### Resuming training

```bash
poetry run python ml/scripts/train.py \
    --resume ml/checkpoints/run-alpha-tune_best.pt \
    --run-name run-alpha-tune-resumed \
    --epochs 40
```

Resume loads the model weights, AdamW optimizer state (momentum + variance), epoch counter, and best F1 achieved. Training continues from `saved_epoch + 1`.

**Important:** only new-format checkpoints (dict with `"model"` key) can be resumed. Old plain state dicts cannot — they don't contain optimizer state.

---

## Evaluation metrics

| Metric | Definition | Why it matters |
|---|---|---|
| `f1_macro` | Unweighted average of F1-safe and F1-vulnerable | Primary checkpoint metric. Not gameable by mass-flagging. |
| `f1_vulnerable` | F1 for label=1 (vulnerable) | Measures detection quality for the main target class |
| `f1_safe` | F1 for label=0 (safe) | Measures false positive rate |
| `precision_vulnerable` | TP/(TP+FP) for vulnerable | Of flagged contracts, how many are truly vulnerable? |
| `recall_vulnerable` | TP/(TP+FN) for vulnerable | Of truly vulnerable contracts, how many were caught? |

**Why F1-macro as the checkpoint criterion?**
Recall_vulnerable can be gamed: a model that predicts everything as vulnerable achieves recall=1.0. F1-macro averages F1 over both classes — mass-flagging collapses F1-safe, pulling down F1-macro. The optimal threshold under F1-macro is genuinely balanced.

### Production checkpoint results

| Run | Epoch | Val F1-macro | Threshold |
|---|---|---|---|
| `baseline` | 16 | 0.6515 | 0.50 |
| `run-alpha-tune_best.pt` | ~26 | **0.6686** | **0.50** ✓ production |
| `run-more-epochs_best.pt` | 22 | 0.6584 | pending sweep |

---

## Decision threshold

The default threshold of 0.50 was selected by `tune_threshold.py` using F1-macro on the val set:

```
Threshold |  F1-vuln | Precision |   Recall | F1-macro
---------------------------------------------------------
     0.45 |   0.7936 |    0.7160 |   0.8901 |   0.6294
     0.50 |   0.7458 |    0.7797 |   0.7147 |   0.6686  ← best
     0.55 |   0.6446 |    0.8543 |   0.5176 |   0.6325
```

At 0.50: 78% of flagged contracts are truly vulnerable (precision), and 71% of vulnerable contracts are caught (recall). This is the production threshold in `predictor.py`.

---

## Dataset pairing

Training data is stored as paired `.pt` files, matched by MD5 hash:

```
ml/data/graphs/<md5_hash>.pt  →  PyG Data(x=[N,8], edge_index=[2,E], y=[label])
ml/data/tokens/<md5_hash>.pt  →  dict(input_ids=[512], attention_mask=[512])
```

`DualPathDataset.__init__` builds the intersection of available hashes. 13 token files with no matching graph are silently skipped. 1 graph file with no matching token is skipped. **68,555 paired samples** remain.

---

## Critical rules

| Rule | Reason |
|---|---|
| `dual_path_collate_fn` must use `.squeeze(1)` not `.squeeze()` | `.squeeze()` collapses `[1, 1]` → scalar when batch_size=1 |
| Graphs load with `weights_only=False` | Graph `.pt` files contain PyG `Data` objects (not plain tensors) |
| Tokens load with `weights_only=True` | Token `.pt` files are plain dicts of tensors — stricter loading is safe |
| `focal_alpha=0.25` is correct | Vulnerable is majority — alpha down-weights it. The "4PM handover bug report" was wrong; confirmed at 10:18PM. |
| Node features must be exactly 8-dim | GNNEncoder has `in_channels=8` hardcoded; changing requires dataset rebuild + retrain |
| Model output is sigmoid-activated | Use `BCELoss`, NOT `BCEWithLogitsLoss`. Applying sigmoid twice produces wrong gradients. |

---

## Known Limitations

| Limitation | Description | Consequence |
|---|---|---|
| Binary output only | Model predicts "vulnerable or not" — cannot identify which vulnerability type | Users must rely on Slither/agents for type-specific findings |
| Multi-label data simplified to binary | 41.2% of BCCC contracts have multiple vulnerability types; only one label was kept per hash | Some contracts may be mislabelled relative to their ground-truth multi-label identity |
| ~24K contracts dropped from training | Slither failed to parse them during graph extraction | Training data is a subset of BCCC, biased toward parseable contracts |
| 512-token CodeBERT limit | Contracts longer than 512 tokens are truncated; the tail is lost | Long complex contracts (e.g. large DeFi protocols) may produce less accurate scores |
| Frozen CodeBERT | CodeBERT weights are never updated during training | Model cannot adapt to Solidity-specific idioms that differ from CodeBERT's pre-training distribution |
| F1-macro 0.6686 on validation | Best production checkpoint performance | Not production-grade for high-stakes decisions; false negatives and false positives both around 28% |
| Test set evaluation pending | `run-alpha-tune_best.pt` has only been evaluated on val set | True generalisation performance not yet confirmed (test set: 10,284 never-touched samples) |
| Class imbalance: 64% vulnerable | Majority class is vulnerable (due to SolidiFI synthetic injection in BCCC data pipeline) | Model trained with Focal Loss alpha=0.25 to compensate; if real-world distribution is different, recalibrate threshold |
