
## What's Missing

### 1. Logit Adjustment (highest priority, almost free)

Not mentioned anywhere. Menon et al. 2021: at inference, subtract `τ × log(π_y)` from each class logit, where `π_y` is the class prior from training data. This directly corrects the model's systematic over-prediction of IntegerUO and under-prediction of DoS without any retraining. It's theoretically motivated for imbalanced classification (recovers Bayes-optimal classifier under shifted test distribution). Implementation is literally 5 lines at inference time. The proposals spend paragraphs on temperature scaling, which is a weaker version of the same idea.

### 2. Decoupled Training / Classifier Re-Balancing

Completely absent from all proposals. The idea: after Run 10 completes, **freeze the entire backbone** (GNN + transformer + fusion) and retrain only the four-eye classifier head for 5-10 epochs on class-balanced data (oversampled or re-weighted). This explicitly separates representation learning (where IntegerUO's dominance is fine) from decision boundary learning (where it's catastrophic). Has strong empirical support on long-tail benchmarks. Zero risk to existing representations. Could be run post-Run 10 without any new training run.

### 3. F-beta Threshold Tuning (β > 1)

The P0-B Optuna search uses F1 (β=1) as objective. For a security oracle, **missing a real vulnerability is worse than a false alarm**. F2 or F3 as the Optuna objective would systematically shift all thresholds toward better recall — which is exactly what you want for DoS, TOD, and Timestamp (currently recall=0 or near-0). One-line change to the objective function, directly aligns the threshold optimization with the actual application requirement.

### 4. Graph Augmentation for Rare Classes

Never discussed. For DoS=246 and TOD=~135 samples, this is fundamentally a few-shot problem. Graph-domain augmentations:
- **Node dropping** (randomly remove non-critical non-call nodes, preserving the vulnerability-relevant structure)
- **Edge perturbation** (swap a small fraction of non-CFG edges)
- **Feature masking** (randomly zero node features during training, forces robustness)

Applied only to the rare class examples, this effectively multiplies DoS and TOD samples 3-5× without synthetic generation. Analogous to augmentation for long-tail image classification, which is well established. Moderate implementation effort but high potential upside for the classes where the model has essentially no signal.

### 5. Per-Class Platt Scaling Instead of Temperature Scaling

P0-C proposes a single scalar T. The adversarial review is right that single-T is dominated by safe contracts. But **Platt scaling** (fit a logistic regression per class on val logits → calibrated probabilities) is strictly better than temperature scaling with only 2 parameters per class (slope + bias vs 1 shared T). It can correct both over- and under-confidence independently per class, and it can shift the operating point of near-zero DoS logits upward while compressing IntegerUO's confident outputs. Implementation is sklearn-level simple.

### 6. Contrastive Loss Head for Embedding Quality

Absent from proposals. Add a supervised contrastive loss (SupCon) on the [B,512] fusion embeddings alongside the main ASL classification loss. The contrastive term pulls same-class embeddings together and pushes different-class embeddings apart, independent of class frequency — DoS embeddings will be pulled together regardless of there being only 246 of them. This could directly fix the "DoS logit ≈ 0" problem by creating a distinct region of embedding space for DoS contracts. The cost is ~10-15% training time overhead. Can be added with a separate loss weight (start at 0.1).

### 7. Loop Nesting Depth and Call Sequence Index as Node Features

The schema additions nobody has proposed explicitly:
- **Loop nesting depth** per basic block: the most direct feature for DoS via unbounded gas consumption. The CFG already captures loops structurally but the node feature doesn't quantify depth.
- **Call sequence index**: position of each `CALL` in the execution flow (1st call, 2nd call, etc.). Directly encodes temporal ordering, which is the core signal for TOD (Transaction Order Dependence). Currently absent from the v9 schema.

These are 2 new node features, both extractable from Slither's CFG output. The re-extraction cost is non-trivial but both features target exactly the classes with F1=0.

### 8. Error Analysis Tooling (Diagnostic, Not a Training Method)

There's no mention anywhere of systematically examining which contracts are consistently wrong. For DoS specifically: are all 246 DoS contracts getting near-zero logits? Or are some being detected and others not? If zero are being detected, that's a feature representation failure. If some are being detected, that's a threshold/calibration failure. These require different fixes. A 1-hour analysis script (forward pass on val/test, group wrong predictions by class and contract size) would clarify which proposals are actually attacking the right problem.

### 9. Hierarchical Safe/Vulnerable Pre-Filter in Loss

57.9% safe contracts dominate every loss computation. An auxiliary binary safe/vulnerable loss alongside the 10-class ASL loss forces the model to maintain a clean separation in the first place. The binary signal is much stronger (safe vs. any vulnerability is an easier problem) and bootstraps the harder multi-class task. Different from the existing aux heads (which are per-phase GNN heads), this is a semantic-level auxiliary task.

---

## What I'd Actually Prioritize

**Do before Run 10 launches:**
- Logit Adjustment at inference (1 day, highest expected value for DoS/TOD recall)
- F-beta (β=2) Optuna objective (30 min)
- Per-class Platt scaling instead of single-T (1 day)

**Add to Run 10 training:**
- Supervised contrastive loss head (2-3 days, addresses root cause for DoS)
- Loop nesting depth + call sequence index features (depends on re-extraction cost)
- Graph augmentation for DoS/TOD (2-3 days)

**Post-Run 10:**
- Decoupled classifier re-balancing pass (1 day, test before deciding on Run 11)
- Error analysis tooling (should have been done before Run 9 honestly)

The logit adjustment + F-beta tuning combination is probably the highest ROI item on this entire list. Both are almost free, both directly target the recall=0 classes, and both have strong theoretical grounding. I'd be surprised if they don't move DoS and TOD recall measurably.