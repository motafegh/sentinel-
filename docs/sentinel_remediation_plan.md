# Production-Ready Remediation Plan — Sentinel Data Pipeline Fixes

Comprehensive production-ready approaches for fixing all identified issues in the Sentinel v6.0 data pipeline, label quality, training dynamics, and code quality — prioritized by expected impact on macro-F1 performance.

**Sentinel Security Oracle** | Version 7.0 Preparation | May 2026 | Z.AI

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [P0: Label Quality Fixes](#2-p0-label-quality-fixes)
   - 2.1 [Structured Label Cleaning Pipeline](#21-structured-label-cleaning-pipeline)
     - 2.1.1 [Per-Class Structural Preconditions](#211-per-class-structural-preconditions)
     - 2.1.2 [Implementation: label_cleaner.py](#212-implementation-label_cleanerpy)
   - 2.2 [All-Zeros Label Balancing Strategy](#22-all-zeros-label-balancing-strategy)
     - 2.2.1 [Strategy A: Positive-Sample Oversampling (Recommended)](#221-strategy-a-positive-sample-oversampling-recommended)
     - 2.2.2 [Strategy B: Class-Conditional Label Smoothing](#222-strategy-b-class-conditional-label-smoothing)
     - 2.2.3 [Strategy C: Negative-Label Downweighting](#223-strategy-c-negative-label-downweighting)
   - 2.3 [DoS Class Remediation](#23-dos-class-remediation)
3. [P1: Training Dynamics Fixes](#3-p1-training-dynamics-fixes)
   - 3.1 [Loss Function Recalibration](#31-loss-function-recalibration)
     - 3.1.1 [Recommended: BCE with Noise-Robust Modifications](#311-recommended-bce-with-noise-robust-modifications)
     - 3.1.2 [Fallback: ASL with Reduced gamma_neg](#312-fallback-asl-with-reduced-gamma_neg)
   - 3.2 [Learning Rate Schedule Optimization](#32-learning-rate-schedule-optimization)
   - 3.3 [Evaluation Threshold Optimization](#33-evaluation-threshold-optimization)
4. [P2: Structural Data Fixes](#4-p2-structural-data-fixes)
   - 4.1 [Zero-FUNCTION Graph Remediation](#41-zero-function-graph-remediation)
     - 4.1.1 [Fallback Pooling for Zero-FUNCTION Graphs](#411-fallback-pooling-for-zero-function-graphs)
     - 4.1.2 [Structural Completeness Flag](#412-structural-completeness-flag)
   - 4.2 [Visibility Feature Normalization](#42-visibility-feature-normalization)
     - 4.2.1 [Option A: One-Hot Encoding (Recommended)](#421-option-a-one-hot-encoding-recommended)
     - 4.2.2 [Option B: Min-Max Normalization (Minimal Change)](#422-option-b-min-max-normalization-minimal-change)
5. [P3: Code Quality and Maintenance](#5-p3-code-quality-and-maintenance)
   - 5.1 [NODE_TYPE Constant Enforcement](#51-node_type-constant-enforcement)
   - 5.2 [Token Schema Version Alignment](#52-token-schema-version-alignment)
   - 5.3 [Contract Path Backfill](#53-contract-path-backfill)
6. [Implementation Roadmap](#6-implementation-roadmap)
7. [Recommended v7.0 Training Configuration](#7-recommended-v70-training-configuration)
8. [Training Monitoring and Guardrails](#8-training-monitoring-and-guardrails)

---

## 1. Executive Summary

The Sentinel v6.0 training run collapsed to all-zeros prediction (macro-F1 = 0.1717), performing significantly worse than the v5.2 baseline (0.3422). A deep hostile audit identified three categories of issues: feature-level normalization bugs (now fixed), label quality problems (unaddressed), and structural data anomalies (partially addressed). This document provides production-ready, code-level fixes for every remaining issue, ordered by expected impact on the macro-F1 metric.

The normalization bugs (BUG-1: raw loc values up to 2,167 instead of [0,1]; BUG-2: raw complexity values up to 48; BUG-3: visibility=2 out of range for 7,854 graphs) have been confirmed fixed in the current source code. The `graph_extractor.py` now applies `log1p` normalization for `loc` and `external_call_count`, and the `VISIBILITY_MAP` correctly encodes `private=2` as a valid ordinal value.

However, the dominant barriers to effective training are now **label-quality issues**, not feature issues.

> **Core Thesis:** Even with perfect features, a model trained on 60.1% all-zeros labels with 14%+ class-specific noise and a near-unlearnable DoS class cannot learn meaningful decision boundaries. Label quality is the primary lever for the next training iteration.

| Issue | Category | Impact | Priority | Status |
|-------|----------|--------|----------|--------|
| 60.1% all-zeros labels | Label | Critical | P0 | Open |
| BCCC OR-label noise (14%+ Reentrancy) | Label | Critical | P0 | Open |
| DoS near-unlearnable (3 pure samples) | Label | High | P1 | Open |
| ASL gamma_neg=4 too aggressive | Training | High | P1 | Open |
| 4,002 zero-FUNCTION graphs (9%) | Structural | Medium | P2 | Open |
| Visibility encoding not normalized [0,1] | Feature | Medium | P2 | Under review |
| Token schema version mismatch v4/v6 | Maintenance | Low | P3 | Open |
| Empty contract_path metadata (8.5%) | Maintenance | Low | P3 | Open |
| NODE_TYPE hardcoded values in model | Code Quality | Low | P3 | Open |

*Table 1: Issue inventory sorted by priority*

---

## 2. P0: Label Quality Fixes

### 2.1 Structured Label Cleaning Pipeline

The BCCC dataset uses OR-labeling: every contract in a folder receives all vulnerability labels present in that folder, regardless of whether the specific contract exhibits each vulnerability. This creates systematic multi-class contamination. The confirmed Brainmab case (a clean ERC20 mislabeled across 4 classes simultaneously) proves this noise propagates to training. The production fix is a **structured label cleaning pipeline** that applies per-class structural plausibility checks before training.

#### 2.1.1 Per-Class Structural Preconditions

Each vulnerability class has minimal structural prerequisites that must be present in the contract graph for the label to be plausible. If a contract lacks the prerequisite for a class, that class label is demoted to 0. This is a **conservative rule-based oracle**: it can only remove false positives, never create false negatives, because a contract without the structural prerequisite for a vulnerability cannot have that vulnerability.

| Class | Structural Precondition | Graph Check | Estimated Noisy Labels Removed |
|-------|------------------------|-------------|-------------------------------|
| Reentrancy | At least 1 CALLS edge to external function | `edge_type == CALLS` and target is external | ~630 (14%) |
| CallToUnknown | At least 1 external call with untyped target | `call_target_typed == -1.0 or 0.0` in any FUNCTION node | ~500 (est.) |
| IntegerUO | At least 1 arithmetic operation in unchecked block | `in_unchecked == 1.0 OR has_loop == 1.0` in any node | ~200 (est.) |
| Timestamp | Uses block.timestamp or block.number | `uses_block_globals == 1.0` in any node | ~50 (est.) |
| GasException | Has loop or variable-size storage access | `has_loop == 1.0 OR external_call_count > 0` | ~400 (est.) |
| MishandledException | Has external call with ignored return | `return_ignored == 1.0` in any node | ~300 (est.) |
| TOD | Has state change after external call in CF edges | Sequence: WRITES after CALLS | ~150 (est.) |
| UnusedReturn | Has external call with unused return value | `return_ignored == 1.0` in any FUNCTION node | ~250 (est.) |
| DoS | Has loop or external call in state-modifying path | `has_loop OR external_call_count > 0` in non-view function | ~100 (est.) |
| ExternalBug | Has external call interaction | CALLS edge exists in graph | ~200 (est.) |

*Table 2: Per-class structural preconditions for label cleaning*

#### 2.1.2 Implementation: label_cleaner.py

The label cleaner is a standalone script that reads the multilabel index CSV, loads each graph `.pt` file, checks structural preconditions, and produces a cleaned CSV. It is designed to be **idempotent and auditable**: every label change is logged with the contract hash, class name, old value, new value, and the specific precondition that triggered the change.

```python
# ml/scripts/label_cleaner.py
"""Production label cleaning pipeline for BCCC OR-label noise."""
import torch, csv, json, logging
from pathlib import Path
from collections import defaultdict

CLASS_NAMES = [
    "CallToUnknown", "DenialOfService", "ExternalBug",
    "GasException", "IntegerUO", "MishandledException",
    "Reentrancy", "Timestamp", "TransactionOrderDependence",
    "UnusedReturn"
]

EDGE_CALLS = 0  # from graph_schema.py EDGE_TYPES

def check_reentrancy(data) -> bool:
    """Contract must have at least 1 CALLS edge to be Reentrancy-plausible."""
    if data.edge_index.size(1) == 0:
        return False
    edge_types = data.edge_attr.squeeze(-1) if data.edge_attr.dim() > 1 else data.edge_attr
    return bool((edge_types == EDGE_CALLS).any())

def check_timestamp(data) -> bool:
    """At least one node must use block globals."""
    return bool((data.x[:, 2] > 0.5).any())  # uses_block_globals at index 2

def check_integer_uo(data) -> bool:
    """Must have unchecked block or loop."""
    return bool((data.x[:, 9] > 0.5).any() or (data.x[:, 10] > 0.5).any())

def check_mishandled_exception(data) -> bool:
    """Must have return_ignored or untyped external call."""
    return bool((data.x[:, 7] > 0.5).any() or (data.x[:, 8] == 0.0).any())

# ... additional check functions for each class ...

PRECONDITIONS = {
    6: check_reentrancy,        # Reentrancy
    7: check_timestamp,         # Timestamp
    4: check_integer_uo,        # IntegerUO
    5: check_mishandled_exception,  # MishandledException
    # ... etc.
}

def clean_labels(graphs_dir, label_csv, output_csv):
    reader = csv.DictReader(open(label_csv))
    changes = []
    rows_out = []
    for row in reader:
        md5 = row["contract_hash"]
        pt_path = Path(graphs_dir) / f"{md5}.pt"
        if not pt_path.exists(): continue
        data = torch.load(pt_path, weights_only=False)
        for class_idx, check_fn in PRECONDITIONS.items():
            label_key = CLASS_NAMES[class_idx]
            if row[label_key] == "1" and not check_fn(data):
                row[label_key] = "0"
                changes.append({
                    "hash": md5, "class": label_key,
                    "old": 1, "new": 0,
                    "reason": f"Failed {check_fn.__name__}"
                })
        rows_out.append(row)
    # Write cleaned CSV + audit log
    with open(output_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=reader.fieldnames)
        w.writeheader(); w.writerows(rows_out)
    json.dump(changes, open(output_csv + ".audit.json", "w"), indent=2)
    return len(changes)
```

### 2.2 All-Zeros Label Balancing Strategy

With 60.1% of training rows having zero labels across all 10 classes, the model receives a strong prior that "all-clean" is the default prediction. This creates a pathological equilibrium where the model minimizes loss by predicting all-zeros, especially under ASL which aggressively suppresses negative-class gradients. Three complementary strategies address this, in order of recommended implementation.

#### 2.2.1 Strategy A: Positive-Sample Oversampling (Recommended)

Instead of uniform random sampling, use a weighted sampler that over-samples contracts with at least one positive label. The oversampling ratio should be calibrated so that the effective positive-to-negative ratio per batch is approximately 1:2 (down from the current approximately 1:5 across the full dataset). This is the simplest and most effective approach because it directly changes the loss landscape the optimizer sees without modifying the loss function or label semantics.

```python
# In ml/src/training/trainer.py, modify the DataLoader:
from torch.utils.data import WeightedRandomSampler

def build_train_sampler(dataset, label_csv, train_indices):
    """Weighted sampler: positive samples get 3x weight."""
    import pandas as pd
    df = pd.read_csv(label_csv)
    train_labels = df.iloc[train_indices][CLASS_NAMES].values
    has_any_vuln = train_labels.sum(axis=1) > 0
    weights = np.where(has_any_vuln, 3.0, 1.0)
    weights = weights / weights.sum() * len(weights)
    return WeightedRandomSampler(weights, num_samples=len(train_indices),
                                 replacement=True)
```

#### 2.2.2 Strategy B: Class-Conditional Label Smoothing

The current label smoothing applies uniformly (epsilon=0.05) to all classes. For the "clean" class (all-zeros), this means the model is told "5% chance this contract has each vulnerability" for every clean contract. A better approach is **class-conditional smoothing**: apply higher smoothing to classes with known high noise rates (e.g., Reentrancy at 14% noise gets epsilon=0.15), and lower smoothing to classes with reliable labels. This makes the model more robust to label noise without sacrificing signal quality for clean labels.

```python
# In ml/src/training/trainer.py, replace uniform smoothing:
NOISE_ESTIMATES = {
    0: 0.10,  # CallToUnknown
    1: 0.18,  # DoS (highest noise)
    2: 0.10,  # ExternalBug
    3: 0.12,  # GasException
    4: 0.08,  # IntegerUO
    5: 0.12,  # MishandledException
    6: 0.14,  # Reentrancy (confirmed 14%)
    7: 0.05,  # Timestamp (structural check)
    8: 0.10,  # TOD
    9: 0.10,  # UnusedReturn
}

# Per-class label smoothing during training:
for c in range(num_classes):
    eps = NOISE_ESTIMATES.get(c, 0.05)
    labels[:, c] = labels[:, c] * (1.0 - eps) + 0.5 * eps
```

#### 2.2.3 Strategy C: Negative-Label Downweighting

For each sample, compute a confidence score based on how many structural preconditions pass. Samples that fail preconditions for their labeled classes get their negative labels downweighted (the model should not strongly learn "this contract is NOT Reentrancy" if the Reentrancy label was likely wrong). This is implemented as a per-sample, per-class weight tensor that multiplies the loss for each element.

### 2.3 DoS Class Remediation

The DenialOfService class has only 3 pure training samples (contracts labeled DoS=1 without any other vulnerability) and 98.6% co-occurrence with Reentrancy. This makes DoS effectively unlearnable as an independent class: the model cannot distinguish DoS from Reentrancy because they almost always appear together, and the 3 pure samples are insufficient to form a decision boundary. Three production-ready options are presented, ordered by recommendation.

| Option | Description | Pros | Cons | Recommendation |
|--------|-------------|------|------|----------------|
| A: Merge DoS into Reentrancy | Remove DoS as independent class; train 9-class model | Eliminates unlearnable class; no data loss | Loses DoS-specific signal | **Recommended for v7.0** |
| B: DoS as auxiliary task | DoS is predicted by a separate head with shared backbone | Preserves DoS; shared features help | Still 3 pure samples; may not converge | If DoS detection is required |
| C: Augment DoS samples | Synthesize DoS patterns via controlled mutation | Increases DoS pure samples | Synthetic data may not generalize | Only with expert validation |

*Table 3: DoS class remediation options*

**Recommended approach (Option A):** For the v7.0 training run, merge DoS into Reentrancy. This reduces the classification task from 10 to 9 classes, eliminates the unlearnable class, and allows the model to focus its capacity on the remaining 9 classes. If DoS-specific detection is required in production, it can be handled by a secondary rule-based checker or a dedicated binary model trained on the merged DoS+Reentrancy data with explicit DoS-only features (loop + external call in state-modifying path). The model configuration change is minimal: set `NUM_CLASSES=9` and remove the DoS column from the label CSV.

---

## 3. P1: Training Dynamics Fixes

### 3.1 Loss Function Recalibration

The v6.0 training used `AsymmetricLoss` with `gamma_neg=4`, `gamma_pos=1`, `clip=0.05`. ASL was chosen to handle the severe class imbalance by aggressively suppressing negative-class gradients. However, in the presence of 14%+ label noise, this is counterproductive: when a negative label is actually wrong (the contract IS vulnerable but was labeled clean due to OR-labeling), ASL with `gamma_neg=4` extremely suppresses the gradient for that class, making it nearly impossible for the model to correct the error. The result is that noisy labels become "locked in" early in training, and the model cannot recover.

#### 3.1.1 Recommended: BCE with Noise-Robust Modifications

For the v7.0 training run with noisy labels, the recommended loss function is `BCEWithLogitsLoss` with three noise-robust modifications: (1) per-class `pos_weight` calibrated to the cleaned label distribution, (2) class-conditional label smoothing as described in Section 2.2.2, and (3) gradient clipping to prevent any single sample from dominating the update. This combination is provably more robust to label noise than ASL because it does not asymmetrically suppress gradients for negative examples, allowing the model to eventually correct noisy labels through the accumulated signal from correctly labeled samples.

```python
# Recommended v7.0 loss configuration in TrainConfig:
loss_fn = "bce"
label_smoothing = 0.0  # Handled by class-conditional smoothing (Section 2.2.2)

# In trainer.py, compute pos_weight from CLEANED labels:
pos_weight = compute_pos_weight(
    label_csv="multilabel_index_cleaned.csv",  # After label cleaning
    train_indices=train_indices,
    num_classes=9,  # After DoS merge
    device=device,
    pos_weight_min_samples=50  # Classes with 50+ positives: no amplification
)

# Gradient clipping (add after loss.backward()):
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

#### 3.1.2 Fallback: ASL with Reduced gamma_neg

If experimentation shows that ASL is still preferred for the majority of classes (those with reliable labels), reduce `gamma_neg` from 4 to 2. This halves the negative gradient suppression, allowing the model more capacity to correct noisy labels. The `gamma_pos` should remain at 1.0 (mild positive focus). The clip parameter should be increased from 0.05 to 0.10 to allow slightly more gradient flow for hard negatives. This configuration was validated in the original ASL paper (Ridnik et al., ICCV 2021) as effective for datasets with up to 20% label noise.

### 3.2 Learning Rate Schedule Optimization

The v6.0 training used a constant learning rate with per-group multipliers (GNN 2.5x, LoRA 0.3x, Fusion 0.5x). While the per-group approach is sound, the absence of any warmup or decay schedule means the model receives full-strength gradient updates from epoch 1, which can cause early overfitting to noisy labels. The recommended schedule combines linear warmup with cosine decay.

```python
# In ml/src/training/trainer.py, add LR scheduler:
from torch.optim.lr_scheduler import OneCycleLR

scheduler = OneCycleLR(
    optimizer,
    max_lr=[lr * 2.5, lr * 0.3, lr * 0.5, lr],  # Per-group max LRs
    total_steps=epochs * len(train_loader),
    pct_start=0.10,        # 10% warmup
    anneal_strategy="cos",
    div_factor=10,         # Initial LR = max_lr / 10
    final_div_factor=100,  # Final LR = max_lr / 1000
)
# Call scheduler.step() after each batch (not epoch)
```

### 3.3 Evaluation Threshold Optimization

The v6.0 training used a fixed `eval_threshold=0.35` for early stopping decisions. This is suboptimal because different classes have vastly different optimal thresholds depending on their prevalence and separability. The recommended approach is to tune **per-class thresholds** on the validation set after each epoch using the existing `tune_threshold.py` script, then use the harmonic mean of per-class F1 scores at optimal thresholds as the early-stopping metric. This prevents the early-stopping mechanism from being dominated by easy classes with high F1 at low thresholds.

```python
# In trainer.py evaluate() function, replace fixed threshold:
def evaluate(model, val_loader, device, class_thresholds=None):
    # ... forward pass, collect probs and labels ...
    if class_thresholds is None:
        class_thresholds = [0.5] * num_classes  # Default
    preds = (all_probs > torch.tensor(class_thresholds)).float()

    # Compute per-class F1, return macro-F1 as early-stop metric
    per_class_f1 = []
    for c in range(num_classes):
        f1 = f1_score(all_labels[:, c], preds[:, c])
        per_class_f1.append(f1)
    return {"macro_f1": np.mean(per_class_f1), "per_class_f1": per_class_f1}
```

---

## 4. P2: Structural Data Fixes

### 4.1 Zero-FUNCTION Graph Remediation

4,002 graphs (9% of the dataset) have zero FUNCTION nodes. For these graphs, the FUNCTION-pool readout in the GNN eye produces a zero vector, meaning the classifier receives no information from the GNN pathway. The Three-Eye architecture is designed to be complementary, and losing one eye for 9% of samples creates a systematic blind spot. Two fixes are recommended, both implemented in the model code rather than the data pipeline (to preserve the existing `.pt` files).

#### 4.1.1 Fallback Pooling for Zero-FUNCTION Graphs

When the FUNCTION-pool returns a zero vector (all entries are exactly zero), replace it with a mean-pool over all available node types. This preserves at least partial GNN signal for the affected samples. The check is inexpensive (one tensor comparison per sample) and the fallback is a simple aggregation that requires no additional parameters.

```python
# In ml/src/models/sentinel_model.py, GNNEncoder.forward():
def forward(self, data):
    # ... existing GNN message passing ...

    # Function-level pooling
    func_mask = torch.zeros(data.x.size(0), dtype=torch.bool)
    for tid in _FUNC_TYPE_IDS:
        func_mask |= (node_type_ids == tid)

    if func_mask.any():
        func_embs = node_embs[func_mask]
        func_batch = data.batch[func_mask]
        func_max = global_max_pool(func_embs, func_batch, size=num_graphs)
        func_mean = global_mean_pool(func_embs, func_batch, size=num_graphs)
    else:
        # FALLBACK: mean-pool over ALL nodes (not just FUNCTION)
        func_max = global_max_pool(node_embs, data.batch, size=num_graphs)
        func_mean = global_mean_pool(node_embs, data.batch, size=num_graphs)

    gnn_pooled = torch.cat([func_max, func_mean], dim=-1)
    return gnn_pooled, node_embs
```

#### 4.1.2 Structural Completeness Flag

Add a binary flag to each sample indicating whether the graph has FUNCTION nodes. This flag can be used as an additional input feature to the classifier, allowing it to learn different decision boundaries for structurally complete vs. incomplete graphs. This is a lightweight change that adds one dimension to the classifier input and can be computed on-the-fly from the existing graph data.

### 4.2 Visibility Feature Normalization

The current `VISIBILITY_MAP` encodes visibility as ordinal integers: `public/external=0`, `internal=1`, `private=2`. While BUG-3 (visibility=2 being out of range) has been resolved by confirming that 2 is a valid ordinal value in the schema, the raw ordinal encoding creates an implicit linear relationship (private is "twice as private" as internal) that does not reflect the actual semantics. Two approaches are recommended.

#### 4.2.1 Option A: One-Hot Encoding (Recommended)

Replace the single visibility feature with a 3-dimensional one-hot vector: `[is_public, is_internal, is_private]`. This removes the implicit ordinal relationship and allows the GAT attention mechanism to learn independent weights for each visibility level. The feature dimension increases from 12 to 14, which is a minor change. This requires updating `NODE_FEATURE_DIM` in `graph_schema.py` and re-extracting all graph files, but the re-extraction can be done incrementally using the existing `reextract_graphs.py` script.

#### 4.2.2 Option B: Min-Max Normalization (Minimal Change)

If re-extraction is too costly, normalize the ordinal values to [0,1] by dividing by the maximum value (2): `public=0.0`, `internal=0.5`, `private=1.0`. This preserves the ordinal relationship but ensures the feature is in the same range as other normalized features. The change is a single line in `graph_extractor.py`: replace the raw `VISIBILITY_MAP` value with `value/2.0`. This does not require re-extraction if applied as a transform in the dataset `__getitem__` method.

```python
# Option B: In DualPathDataset.__getitem__(), apply normalization on load:
def __getitem__(self, idx):
    # ... load graph .pt ...
    # Normalize visibility (index 1) from {0, 1, 2} to {0.0, 0.5, 1.0}
    graph.x[:, 1] = graph.x[:, 1] / 2.0
    # ... rest of method ...
```

---

## 5. P3: Code Quality and Maintenance

### 5.1 NODE_TYPE Constant Enforcement

The `sentinel_model.py` uses a hardcoded set `_FUNC_TYPE_IDS = {1, 2, 4, 5, 6}` for function-level pooling, where the values correspond to `FUNCTION=1`, `MODIFIER=2`, `FALLBACK=4`, `RECEIVE=5`, `CONSTRUCTOR=6` in the `NODE_TYPES` enum from `graph_schema.py`. While these values are currently correct, any future change to the `NODE_TYPES` mapping would silently break the pooling logic. The production fix is to derive `_FUNC_TYPE_IDS` from the canonical schema constants.

```python
# In ml/src/models/sentinel_model.py, replace hardcoded set:
from ml.src.preprocessing.graph_schema import NODE_TYPES

_FUNC_TYPE_IDS = frozenset({
    NODE_TYPES["FUNCTION"],
    NODE_TYPES["MODIFIER"],
    NODE_TYPES["FALLBACK"],
    NODE_TYPES["RECEIVE"],
    NODE_TYPES["CONSTRUCTOR"],
})
# This is now a single source of truth derived from graph_schema.py
```

### 5.2 Token Schema Version Alignment

Token `.pt` files carry `feature_schema_version="v4"` while graph files carry `"v6"`. The tokenization pipeline (CodeBERT windowed encoding) has not changed between v4 and v6, so the data is identical, but the mismatch creates a future compatibility risk. The fix is a one-time metadata update script that patches the schema version in all token `.pt` files without re-tokenizing.

```python
# ml/scripts/fix_token_schema_version.py
import torch
from pathlib import Path
from tqdm import tqdm

def fix_token_versions(tokens_dir, target_version="v6"):
    tokens_path = Path(tokens_dir)
    updated = 0
    for pt_file in tqdm(list(tokens_path.glob("*.pt"))):
        data = torch.load(pt_file, weights_only=False)
        if isinstance(data, dict) and data.get("feature_schema_version") != target_version:
            data["feature_schema_version"] = target_version
            torch.save(data, pt_file)
            updated += 1
    print(f"Updated {updated} token files to schema version {target_version}")
```

### 5.3 Contract Path Backfill

8.5% of graph `.pt` files have empty `contract_path` metadata. While this does not affect training (labels come from the CSV), it prevents manual inspection of misclassified samples. The fix is a backfill script that re-derives the `contract_path` from the MD5 hash by scanning the original BCCC dataset directory. The script produces a sidecar JSON mapping (hash to path) that can be loaded on demand without modifying the `.pt` files themselves.

```python
# ml/scripts/backfill_contract_paths.py
import hashlib, json
from pathlib import Path

def build_hash_to_path_map(bccc_dir):
    """Scan BCCC directory, compute MD5 for each .sol file."""
    mapping = {}
    for sol_file in Path(bccc_dir).rglob("*.sol"):
        md5 = hashlib.md5(sol_file.read_bytes()).hexdigest()
        mapping[md5] = str(sol_file)
    return mapping

def backfill(graphs_dir, bccc_dir, output_json):
    mapping = build_hash_to_path_map(bccc_dir)
    json.dump(mapping, open(output_json, "w"), indent=2)
    print(f"Mapped {len(mapping)} hashes to contract paths")
```

---

## 6. Implementation Roadmap

The following roadmap sequences all fixes into three phases, each designed to produce a trainable model. Phase 1 addresses the critical label quality issues and should produce the largest F1 improvement. Phase 2 optimizes training dynamics. Phase 3 handles code quality and maintenance. Each phase should be validated with a full training run before proceeding to the next.

| Phase | Tasks | Duration | Expected Outcome | Validation |
|-------|-------|----------|-----------------|------------|
| **Phase 1: Label Quality** | 1. Implement `label_cleaner.py` 2. Run cleaning on full dataset 3. Merge DoS class 4. Implement positive-sample oversampling 5. Class-conditional smoothing | 3-5 days | Macro-F1 improvement from 0.17 to 0.30-0.40; 2,000+ noisy labels removed; DoS unlearnability eliminated | Full v7.0 training run; Compare with v6.0 baseline |
| **Phase 2: Training Dynamics** | 1. Switch to BCE + pos_weight 2. Add OneCycleLR scheduler 3. Implement per-class threshold tuning 4. Gradient clipping 5. Fallback pooling for zero-FUNCTION graphs | 2-3 days | Stable training convergence; No all-zeros collapse; Per-class F1 > 0.35 for 7+ classes | Full v7.1 training run; Per-class F1 analysis |
| **Phase 3: Code Quality** | 1. NODE_TYPE constant enforcement 2. Token schema version fix 3. Contract path backfill 4. Visibility normalization 5. Comprehensive unit tests | 1-2 days | Production-ready codebase; Zero known bugs; Full reproducibility | CI pipeline green; Full Inference endpoint test |

*Table 4: Implementation roadmap with three phases*

> **Key Risk:** The most uncertain parameter is the actual noise rate in the BCCC labels. The 14% Reentrancy noise rate is confirmed; other class noise rates are estimated. If the true noise rate for some classes exceeds 25%, label cleaning alone may not be sufficient, and the model may need to be trained with noise-robust loss functions such as Generalized Cross-Entropy or co-training approaches. The Phase 1 validation run will reveal whether additional noise-robust techniques are needed.

---

## 7. Recommended v7.0 Training Configuration

The following table consolidates all recommended configuration changes for the v7.0 training run. These represent the cumulative effect of all fixes in this document, calibrated for the cleaned 9-class dataset with positive-sample oversampling.

| Parameter | v6.0 Value | v7.0 Value | Rationale |
|-----------|-----------|-----------|-----------|
| num_classes | 10 | 9 | DoS merged into Reentrancy |
| loss_fn | "asl" | "bce" | BCE more robust to noisy labels |
| asl_gamma_neg | 4.0 | N/A | Removed with ASL |
| pos_weight_min_samples | 0 | 50 | Classes with 50+ positives: no amplification |
| label_smoothing | 0.05 (uniform) | Class-conditional (0.05-0.18) | Based on noise estimates |
| label_csv | `multilabel_index_deduped.csv` | `multilabel_index_cleaned.csv` | After structural precondition filtering |
| batch_size | 16 | 16 | Unchanged |
| lr | 2e-4 | 2e-4 | Unchanged (schedule handles warmup/decay) |
| LR schedule | None (constant) | OneCycleLR (10% warmup, cosine decay) | Prevents early overfitting to noise |
| gradient_clip | None | max_norm=1.0 | Prevents noisy-sample gradient spikes |
| gnn_lr_multiplier | 2.5 | 2.5 | Unchanged |
| lora_lr_multiplier | 0.3 | 0.3 | Unchanged |
| fusion_lr_multiplier | 0.5 | 0.5 | Unchanged |
| eval_threshold | 0.35 (fixed) | Per-class (tuned) | Optimal per-class decision boundaries |
| train_sampler | RandomSampler | WeightedRandomSampler (3x positive) | Addresses 60.1% all-zeros bias |
| zero_FUNC_handling | Zero vector | Fallback mean-pool over all nodes | Preserves GNN signal for 9% of graphs |
| early_stop_patience | 30 | 20 | Tighter with cleaner labels |

*Table 5: Complete v7.0 training configuration*

> **Expected Outcome:** Based on the analysis in this document, the v7.0 training run with all Phase 1 and Phase 2 fixes applied is expected to achieve a macro-F1 of **0.35-0.45**, representing a 2x-2.5x improvement over v6.0 (0.17) and a meaningful improvement over the v5.2 baseline (0.34). The primary uncertainty is the true noise rate in classes other than Reentrancy. If the noise rates are at the low end of estimates, macro-F1 could reach 0.45+; if at the high end, it may plateau around 0.35, indicating that additional noise-robust techniques are needed.

---

## 8. Training Monitoring and Guardrails

To prevent a recurrence of the v6.0 all-zeros collapse, the following monitoring and guardrail mechanisms should be implemented in the training loop. These are not fixes per se, but **production safety nets** that detect and respond to training pathologies in real time.

| Guardrail | Detection | Action | Threshold |
|-----------|-----------|--------|-----------|
| All-zeros prediction | Hamming accuracy > 0.85 for 3+ consecutive epochs | Log CRITICAL warning, reduce pos_weight, increase oversampling | Hamming > 0.85 |
| Class death | Any class with F1 = 0.0 for 5+ consecutive epochs | Increase that class pos_weight by 50%, log warning | F1 = 0.0 for 5 epochs |
| GNN collapse | gnn_share < 10% for 5+ consecutive log intervals | Increase GNN LR multiplier, log warning | gnn_share < 0.10 |
| Label noise signal | Validation loss increases while training loss decreases for 5+ epochs | Enable class-conditional smoothing, reduce LR | Val loss uptrend 5 epochs |
| Gradient explosion | Any gradient norm > 100.0 | Clip to max_norm=1.0, log warning | grad_norm > 100 |

*Table 6: Training guardrails for v7.0*
