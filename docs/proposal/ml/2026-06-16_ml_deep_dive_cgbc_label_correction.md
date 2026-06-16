# Deep Dive: Granular-ball Label Correction (CGBC) for SENTINEL

**Date:** 2026-06-16
**Module:** `ml`
**Phase:** Run 14 preparation
**Tier:** 1
**Proposal Reference:** §5.4 of `2026-06-15_ml_Run13_prep_proposal_next_gen_ml_methods.md`

---

## 1. Full Method Explanation

### 1.1 What Is CGBC?

Contrastive Granular-ball Learning with Boundary Correction (CGBC) is a method designed for smart contract vulnerability detection with **noisy labels**. It addresses a fundamental problem: automated static analysis tools (Slither, SmartBugs) produce labels that are inherently noisy.

### 1.2 The Noisy Label Problem

In SENTINEL:
- Labels come from automated tools (Slither, SmartBugs static analyzers)
- These tools have known false positive patterns
- Example: `s_Form001` (26-line KV store) labeled as ExternalBug at p=0.96 — the model learned **spurious features** from noisy training signals

CGBC corrects label noise BEFORE training by:
1. Clustering contracts into "granular balls" (clusters)
2. Computing consensus labels per cluster
3. Identifying and correcting labels that disagree with consensus

### 1.3 Three-Phase CGBC Algorithm

**Phase 1: Contrastive Pre-training**
- Pre-train encoder with semantic-consistent augmentation
- Learn initial embeddings that capture contract structure

**Phase 2: Granular-ball Construction**
- Cluster contracts in embedding space
- Each cluster = one "granular ball"
- Compute inter-GB compactness loss + intra-GB looseness loss

**Phase 3: Label Correction + Training**
- Majority vote within each granular ball
- Contracts with label disagreement → corrected labels
- Train with symmetric cross-entropy (noise-robust)

### 1.4 Why Symmetric Cross-Entropy?

Standard BCE is sensitive to label noise: if 30% of ExternalBug labels are wrong, BCE forces the model to learn incorrect patterns. Symmetric cross-entropy (SCE) balances:
- Forward CE: p * log(p) — standard learning
- Reverse CE: q * log(p) — prevents overconfidence on noisy labels

### 1.5 Granular Ball Properties

- **Size:** Each ball contains 5-50 similar contracts
- **Purity:** % of contracts in ball with same label
- **Boundary:** Contracts near ball boundaries are most likely noisy
- **Consensus:** Majority vote within ball → corrected label

---

## 2. SENTINEL-Specific Audit

### 2.1 Label Noise Quantification

From Run 12 manual inspection:
- **ExternalBug:** 65% S_only rate — high false positive rate
- **s_Form001:** Predicted as ExternalBug at p=0.96 — clear false positive
- **Root cause:** Spurious training signals from automated tool labels

**Estimated noise rates by class:**
| Class | Estimated Noise | Evidence |
|-------|----------------|----------|
| ExternalBug | 30-40% | 65% S_only rate, manual inspection |
| Timestamp | 5-10% | Mostly true positives (block.timestamp in vesting) |
| Reentrancy | 5-10% | Mostly true positives (classic CEI violations) |
| DoS | 10-20% | Small sample size (243), hard to verify |
| Others | 10-15% | Unknown — needs manual audit |

### 2.2 Embedding Space for Clustering

CGBC needs embeddings for clustering. SENTINEL has two options:

**Option A: GNN embeddings** (faster)
- Use `GNNEncoder` output: `[N, 256]` → pool to `[B, 256]`
- Captures structural patterns only
- ~615K params, fast inference

**Option B: Four-eye embeddings** (better)
- Use concatenated eye projections: `[B, 512]`
- Captures structural + semantic patterns
- ~2.5M params, includes transformer signal

**Recommendation:** Start with GNN embeddings for speed. If clustering quality is poor, switch to four-eye embeddings.

### 2.3 Clustering Algorithm

From the proposal:
```python
from sklearn.cluster import DBSCAN

clustering = DBSCAN(eps=0.5, min_samples=3).fit(embeddings)
```

**Issues with DBSCAN:**
- `eps=0.5` assumes normalized embeddings — SENTINEL embeddings are not normalized
- `min_samples=3` may be too small for 22K contracts
- DBSCAN struggles with varying density clusters

**Better alternatives:**
- **K-Means:** Faster, assumes spherical clusters (reasonable for normalized embeddings)
- **HDBSCAN:** Density-based, handles varying density, no `eps` parameter
- **Spectral clustering:** Best for non-convex clusters, but O(N³)

**Recommendation:** Use HDBSCAN — handles varying density, automatically finds number of clusters.

### 2.4 Label Correction Logic

```python
def correct_labels(embeddings, labels, purity_threshold=0.7):
    """Majority vote within each cluster, correct disagreeing labels."""
    clusters = HDBSCAN(min_cluster_size=5).fit_predict(embeddings)
    corrected = labels.clone()
    noise_mask = torch.zeros(len(labels), dtype=torch.bool)

    for cluster_id in set(clusters):
        if cluster_id == -1:
            continue  # noise points
        mask = clusters == cluster_id
        cluster_labels = labels[mask]
        # Majority vote per class
        consensus = (cluster_labels.sum(dim=0) / mask.sum()) > purity_threshold
        # Correct disagreeing labels
        for i in torch.where(mask)[0]:
            disagreement = (labels[i] != consensus.float()).any()
            if disagreement:
                corrected[i] = consensus.float()
                noise_mask[i] = True

    return corrected, noise_mask
```

### 2.5 Critical Considerations

1. **Purity threshold tuning:** Too high (>0.8) → few corrections; too low (<0.5) → over-correction. Start with 0.7, tune on validation set.

2. **Cluster size:** Too small (<5) → noisy consensus; too large (>50) → mixed clusters. HDBSCAN's `min_cluster_size=10` is a good starting point.

3. **Multi-label complexity:** SENTINEL is multi-label (9 classes). Majority vote must be per-class, not per-contract. A contract can have multiple vulnerabilities.

4. **Validation:** Corrected labels must be validated against:
   - 66 honest OOD contracts (manually inspected)
   - Expert review of corrected ExternalBug labels
   - No drift in other class F1 beyond ±0.02

5. **One-time cost:** CGBC is a pre-processing step, not a training modification. Run once, save corrected labels, train as usual.

---

## 3. Implementation Plan

### 3.1 Phase D1: Embedding Extraction (~0.5 day)

**New file:** `ml/scripts/pretraining/extract_embeddings.py`

1. Load Run 12 checkpoint
2. Run GNN encoder on v3 training set (18,596 contracts)
3. Extract graph-level embeddings: `[B, 256]`
4. Save to `ml/data/v3_train_embeddings.pt`

### 3.2 Phase D2: CGBC Label Correction (~2 days)

**New file:** `ml/src/training/cgbc_corrector.py`

1. Implement `CGBBCorrector` class
2. Cluster embeddings with HDBSCAN
3. Compute consensus labels per cluster
4. Identify and correct noisy labels
5. Save corrected labels to `ml/data/v3_train_labels_corrected.json`

### 3.3 Phase D3: Validation (~1 day)

**New file:** `ml/scripts/audit/validate_cgbc.py`

1. Load corrected labels
2. Compare with original labels (how many corrected?)
3. Manual inspection of corrected ExternalBug labels
4. Verify no drift in other classes
5. Document in audit report

### 3.4 Phase D4: Training with Corrected Labels (~0.5 day)

**Modify:** `ml/src/training/trainer.py`

1. Add `--corrected_labels` flag
2. Load corrected labels instead of original labels
3. Train on corrected data
4. Compare F1 with/without correction on 66 OOD contracts

### 3.5 Expected Outcomes

- **ExternalBug FP reduction:** 20-30% (from 65% S_only rate)
- **Other class stability:** ±1-2% F1 (no significant drift)
- **Total corrected labels:** ~5-10% of training set (1,000-2,000 contracts)
- **Training time:** Unchanged (same model, same loss, different labels)

### 3.6 Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Over-correction of valid labels | Medium | High | High purity threshold (0.7), manual validation |
| Clustering quality poor | Low | Medium | Use HDBSCAN, tune min_cluster_size |
| Multi-label consensus tricky | Medium | Medium | Per-class majority vote, not per-contract |
| Corrected labels leak information | Low | High | Use only training split, never val/test |

---

## 4. References

- **CGBC** (2026): `arXiv:2603.27734` — Contrastive Granular-ball Learning with Boundary Correction
- **Symmetric CE** (2019): "Symmetric Cross Entropy for Robust Learning with Noisy Labels"
- **CleanLab** (2021): "Data-centric AI" — label quality estimation
- **HDBSCAN** (2017): "Clustering with Noise Nodes" — density-based clustering
