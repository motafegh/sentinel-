# Deep Dive: Structured Ensembling for SENTINEL

**Date:** 2026-06-16
**Module:** `ml`
**Phase:** Run 14 preparation
**Tier:** 1
**Proposal Reference:** §5.3 of `2026-06-15_ml_Run13_prep_proposal_next_gen_ml_methods.md`

---

## 1. Full Method Explanation

### 1.1 What Is Structured Ensembling?

Structured ensembling trains **complementary models** that specialize in different aspects of the prediction task, then combines them at inference. Unlike standard ensembles (N copies of the same model), structured ensembles have architectural diversity:

- **Small fast model** — captures structural patterns, runs in <50ms
- **Large slow model** — adds semantic depth via transformer + fusion
- **Cascade inference** — fast model runs first; if confident, skip slow model

### 1.2 Cascade Inference

```
Input → Fast Model (GNN-only)
        ↓
        Confidence > threshold?
        ├─ Yes → Return fast model prediction
        └─ No  → Run Full Model (4-eye) → Return full prediction
```

This reduces average inference latency by ~60% (80% of predictions are high-confidence from the fast model).

### 1.3 BreachT5 Paradigm

BreachT5 (2025) demonstrated that for smart contract vulnerability detection:
- Small models (220M) excel at **frequent classes** (Reentrancy, Timestamp)
- Large models (770M) excel at **rare classes** (DoS, ExternalBug)
- Combining them via structured ensembling improves both frequent AND rare class F1

### 1.4 Key Insight

The GNN already captures structural patterns (CFG topology, call chains). The transformer adds semantic depth (variable naming, function purpose). For many contracts, structural patterns alone are sufficient — the full 4-eye model is overkill.

---

## 2. SENTINEL-Specific Audit

### 2.1 Current Model Size and Latency

From verified parameters:
- **Full model:** 127M total, 2.5M trainable
- **GNN-only:** 615K params (0.6M)
- **Transformer:** 589K LoRA + 125M frozen

**Inference latency** (estimated):
- GNN-only: ~10-20ms (single forward pass, no transformer)
- Full 4-eye: ~50-100ms (includes transformer forward pass)

### 2.2 GNN-Only Model Architecture

Reuses the exact same `GNNEncoder` from `gnn_encoder.py`:

```python
class GNNOnlyModel(nn.Module):
    def __init__(self, gnn_config, num_classes=9):
        super().__init__()
        self.gnn = GNNEncoder(**gnn_config)  # Same 615K-param encoder
        self.pool = global_max_pool + global_mean_pool → [B, 512]
        self.classifier = Linear(512, 128) → ReLU → Dropout → Linear(128, num_classes)

    def forward(self, graph_data):
        node_embs = self.gnn(graph_data)  # [N, 256]
        pooled = pool(node_embs, graph_data.batch)  # [B, 512]
        return self.classifier(pooled)  # [B, num_classes]
```

### 2.3 Cascade Gate Logic

From `predictor.py:583` (`_score_windowed`):
- Current inference already has tier thresholds (CONFIRMED ≥ 0.55, SUSPICIOUS ≥ 0.25)
- Cascade gate uses a **higher threshold** (e.g., 0.8) for fast model confidence
- If fast model has any class > 0.8, return early (high confidence)
- Otherwise, run full model

### 2.4 Expected Performance Distribution

Based on Run 12 data:
- **~60-70% of contracts:** Fast model confident (structural patterns clear)
- **~20-30% of contracts:** Need full model (semantic ambiguity)
- **~5-10% of contracts:** Both models uncertain (genuinely ambiguous)

### 2.5 Training Strategy

**Phase 1:** Train GNN-only model on labeled data
- Same training loop as full model, but only GNN encoder + classifier
- Expected F1: ~0.55-0.65 (structural patterns only)

**Phase 2:** Train full 4-eye model (existing)
- Already done (Run 12: F1=0.7004)

**Phase 3:** Train cascade meta-classifier (optional)
- Learns weighting between fast and full model predictions
- Can be as simple as: if fast_model.confidence > 0.8, use fast; else use full

### 2.6 Critical Considerations

1. **GNN-only F1 ceiling:** The GNN alone cannot match the full model's F1. The cascade is about **average latency**, not peak F1.

2. **Rare classes suffer:** GNN-only will be worse at rare classes (DoS, ExternalBug) — that's why the cascade falls back to the full model.

3. **Training cost:** Training the GNN-only model adds ~2 hours (GNN is small, converges fast). The full model training is unchanged.

4. **Inference cost:** The cascade adds ~1ms overhead (confidence check). For 80% of contracts, it saves ~50-80ms (skipping transformer).

5. **Deployment:** Two models in memory: GNN-only (~2.5MB) + full model (~500MB). Total: ~502MB — well within production limits.

---

## 3. Implementation Plan

### 3.1 Phase C1: GNN-Only Model (~1 day)

**New file:** `ml/src/models/gnn_only_model.py`

1. Implement `GNNOnlyModel` class
2. Reuse `GNNEncoder` from `gnn_encoder.py`
3. Add `VulnerabilityAttentionPooler` for graph-level pooling
4. Train on v3 labeled data (same splits, same ASL loss)

### 3.2 Phase C2: Cascade Inference (~1 day)

**New file:** `ml/src/inference/cascade.py`

1. Implement `CascadeInference` class
2. Load both models (fast + full)
3. Implement confidence gate logic
4. Track hit rate (% of contracts using fast model only)

### 3.3 Phase C3: Integration and Validation (~1 day)

**Modify:** `ml/src/inference/predictor.py`

1. Add `--cascade` flag to enable cascade inference
2. Integrate `CascadeInference` into `Predictor`
3. Validate on 66 OOD contracts
4. Measure: F1 (should match full model), average latency (should reduce 40-60%)

### 3.4 Expected Outcomes

- **Average latency reduction:** 40-60% (80% of contracts use fast model)
- **F1 maintenance:** Cascade F1 = full model F1 (fallback for uncertain contracts)
- **Training cost:** +2 hours for GNN-only model
- **Memory:** +2.5MB (GNN-only model weights)

### 3.5 Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| GNN-only F1 too low | Expected | Low | Cascade fallback ensures full model used |
| Fast model confidence miscalibrated | Medium | Medium | Calibrate on validation set |
| Memory increase | Low | Low | GNN-only is 2.5MB, negligible |
| Cascade gate overhead | Low | Low | ~1ms, negligible vs 50ms saved |

---

## 4. References

- **BreachT5** (2025): TU Delft — structured ensembling for smart contract vulnerability detection
- **Cascade Models** (2019): "Cascade CNN" — confidence-based early exit
- **Early Exit** (2020): "BranchyNet" — adaptive computation for neural networks
