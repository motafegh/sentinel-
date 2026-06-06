# ADR-0006: Loss formulation (AsymmetricLoss γ⁻=2, γ⁺=1, plus aux BCE pathway)

**Status:** Accepted
**Date:** 2026-06-06
**Deciders:** Ali Motafegh, Claude (assistant)
**Supersedes:** None (current)
**Superseded by:** None (current)

## Context

SENTINEL's loss formulation must support:

1. **Multi-label classification** (ADR-0002: 10-class sigmoid, not softmax)
2. **Class imbalance** (BCCC has 1.7% to 56.7% per-class positive rate; see
   ADR-0005)
3. **Multi-eye training signal** (ADR-0003: GNN, TF, Fused, CFG eyes must all
   contribute gradient)
4. **Per-class calibration** (the agent's 3-tier CONFIRMED / SUSPICIOUS /
   NOTEWORTHY output requires the model to output *meaningful* probabilities
   per class, not just argmax)

Three loss families were considered:

- **Binary cross-entropy (BCE):** per-class sigmoid, no class weighting, sum over
  classes. Simple. Ignores imbalance.
- **Focal loss:** downweights easy negatives via `(1-pt)^γ`. Standard for imbalanced
  multi-class. Asymmetric: treats positives and negatives equally.
- **AsymmetricLoss (ASL):** Ben-Baruch et al. 2020. Extends focal with separate
  focusing parameters γ⁻ (for negatives) and γ⁺ (for positives). Designed for
  multi-label with imbalance.

## Decision

We use **AsymmetricLoss with γ⁻=2.0, γ⁺=1.0, clip=0.05**, applied as follows:

- **Per-eye loss:** the four eyes (GNN, TF, Fused, CFG) each produce `[B, 10]`
  logits. Each eye gets its own ASL loss against the same target. The four
  per-eye losses are summed.
- **Aux BCE pathway supervision:** the Phase 2 GNN output (`_phase2_x`) is fed
  into a small `[128 → 10]` linear classifier that produces `_phase2_logits`.
  An auxiliary BCE loss is applied against the target. Weight: `0.20` baseline,
  `0.30` warmup target after 8 epochs (linear 0→0.30 over ep1–ep8).
- **Pathway weight warmup:** the aux pathway weight starts at 0 and is linearly
  ramped to `aux_phase2_max=0.30` over `aux_warmup_epochs=8`. This prevents the
  aux pathway from dominating early training.

The total loss is:
```
L_total = ASL(GNN) + ASL(TF) + ASL(Fused) + ASL(CFG) + w(t) * BCE(aux)
        + 0.005 * H(JK_attn)                          # entropy regularizer
        + 1e-5 * (||theta||^2)                        # weight decay
where:
  w(t) = clip(t / 8, 0, 1) * 0.30    # aux pathway warmup
  H(JK_attn) = -sum(p_i * log(p_i))   # JK phase attention entropy
```

**γ⁻=2.0** down-weights easy negatives aggressively — a contract that is
definitely *not* Reentrancy (the "easy negatives" in BCCC) contributes ~1/16th
of a normal BCE gradient.

**γ⁺=1.0** (not 0) means positives get no down-weighting at all. This is
deliberate: positives are sparse, we cannot afford to suppress them.

**clip=0.05** (the `eps` parameter) prevents the loss from going to 0 on
high-confidence easy negatives — a known issue with focal/ASL where
`(1-pt)^γ` for `pt≈1` is essentially 0 gradient.

The **aux BCE** is required because in early runs (Run 5, Run 6) the GNN eye
collapsed to a constant (the loss on the GNN eye stopped contributing
gradient). The aux BCE on the Phase 2 output forces that pathway to keep
producing meaningful features. The warmup prevents it from dominating.

## Consequences

### Positive

- **Class-imbalance handling without over-sampling.** ASL with γ⁻=2 achieves
  the same effective re-weighting as 8× over-sampling the minority classes
  but does not duplicate samples in the training set.
- **Per-eye interpretability.** Each eye has its own loss, so we can log
  `loss/{gnn,tf,fused,cfg}_eye` to MLflow and watch for collapse.
- **Aux pathway forces structural learning.** The CFG eye (consuming
  `_phase2_x`) has 0.20 baseline weight on the aux BCE. If the Phase 2 GNN
  collapses, the aux BCE loss explodes — this is a **leading indicator** of
  training failure that surfaces 2-3 epochs before F1-macro degrades.
- **JK entropy regularizer prevents collapse.** The 0.005 entropy term on
  JK attention weights was added in v5.2 to combat Run 4's `jk_phase3_attn
  → 0.95` collapse. Run 7 still showed drift (A3 finding); Run 8 raised
  entropy to 0.0075.
- **Calibration is preserved.** ASL with γ⁺=1 does not over-suppress positive
  gradient, so the model retains the ability to output *high* probabilities
  for high-confidence positives. This is critical for the 3-tier output.

### Negative

- **ASL is more sensitive to hyperparameters than BCE.** Wrong `γ⁻` (e.g.
  0.5 instead of 2.0) over-suppresses negatives to the point where the model
  cannot distinguish 0.0 from 0.1 logits. Locked at 2.0; do not change.
- **Aux loss warmup is fragile.** If `aux_warmup_epochs` is set wrong (too
  short or too long), the model can either:
  - Under-train the GNN eye (warmup too long)
  - Over-train the GNN eye and starve the TF eye (warmup too short)
  Empirically, 8 epochs is the sweet spot for a 30-50 epoch run.
- **Four per-eye losses increase tensorboard noise.** With 4 eyes × 4 metrics
  per eye (loss, acc, precision, recall) = 16 series, plus aux, plus JK
  entropy, the MLflow UI is busy. Aggregation is required.
- **The aux BCE pathway shares parameters with the Fused eye.** The
  `_phase2_x` output is consumed by both the CFG eye classifier and the aux
  pathway linear. This is intentional (gradient sharing) but means the aux
  pathway's loss can dominate if weight is too high.

### Neutral

- The four-eye sum (not mean) is a choice. Sum preserves the magnitude of
  the gradient; mean would be more "balanced" but with γ⁻=2 down-weighting,
  the per-eye losses have different scales, so sum is what we use.

## Alternatives Considered

**1. Plain BCE with pos_weight.**

Considered. Run 5 used `pos_weight = (1 - π) / π` per class (where π is the
class prior). Result: F1 plateaued at 0.27 and the high-resource classes
(Reentrancy, IntegerUO) collapsed because the pos_weight was so high for
low-resource classes that the gradients were dominated by the low-resource
class errors. Abandoned after Run 5 (see `project_run5_training.md`).

**2. Focal loss with γ=2 (symmetric, both γ⁻ and γ⁺).**

Considered. The symmetric version treats positives and negatives equally
aggressive. For multi-label with sparse positives, this is too aggressive
on the positive side — the model cannot learn "this is a strong positive"
because the gradient is suppressed. ASL's asymmetric formulation was
designed for exactly this case (Ben-Baruch et al. 2020).

**3. Class-weighted CE with weight = log(1/π).**

Considered. Log-inverse-frequency weighting is a classical approach. The
issue is that extreme minority classes (GasException at 1.7%, ExternalBug
at 2.1%) get weights 4-5x larger than majority classes, creating the same
gradient domination problem as `pos_weight`. ASL achieves the same effect
with a smoother gradient.

**4. Contrastive / triplet loss (CodeBERT pretraining-style).**

Considered in v4 design. Rejected because the labels are categorical
(multi-label), not relative similarity. Triplet loss would require
defining "similar" contracts in a way that's not directly aligned with
the vulnerability classification task.

**5. Per-eye mean instead of sum.**

Considered. Mean would be 0.25× the magnitude of sum, requiring a 4×
learning rate. Tested in v6 alpha runs; both formulations train
identically, but sum is more "honest" because each eye's loss is
unweighted in the gradient. Sum is what we use.

**6. Per-eye ASL weights (e.g. GNN eye = 1.5, TF eye = 1.0).**

Considered. Rejected because per-eye weighting would require tuning a
4-dimensional hyperparameter on a 41K-contract dataset, which we cannot
afford. Equal weights are the simpler choice and the four-eye design
inherently compensates for per-eye weakness via the eye-sum at inference.

## References

- Source: `ml/src/training/losses.py:49-126` (AsymmetricLoss implementation)
- Source: `ml/src/training/trainer.py:1389-1672` (per-eye loss + aux pathway + JK
  entropy)
- Source: `ml/src/training/trainer.py:1467-1491` (aux warmup: `aux_warmup_epochs=8`,
  `aux_phase2_max=0.30`)
- Source: `ml/src/models/sentinel_model.py:440-461` (per-eye classifier heads, each
  Linear(hidden → 256 → 10))
- Paper: Ben-Baruch et al., "Asymmetric Loss For Multi-Label Classification"
  (2020), arXiv:2009.14119
- Run history: `project_run5_training.md` (BCE + pos_weight abandoned)
- Run history: `project_run6_training.md` (ASL adopted, γ⁻=2.0 γ⁺=1.0)
- Run history: `project_run8_audit_findings.md` (A1 finding: per-eye loss collapse,
  A3: JK weight drift, both fixed in v8)
- Audit: `docs/pre-run8-fixes/RUN8-ULTRACODE.md` (L2: aux pathway added,
  aux_warmup_epochs=8, JK entropy 0.005→0.0075)
- Related: ADR-0002 (multi-label formulation defines the target shape)
- Related: ADR-0003 (Four-Eye architecture defines the per-eye loss structure)
- Related: ADR-0004 (three-phase GAT routing; the Phase 2 output is what the aux
  pathway supervises)
