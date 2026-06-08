Adversarial Technical Review — Pre-Run 10 Final Proposal
Role: Technical Lead Engineering review — evidence-based challenge of every proposed solution. All claims backed by source code or actual training data.

Data basis:

Training split: 29,101 samples (confirmed from splits/deduped/train_indices.npy)
Label distribution: 57.9% safe, IntegerUO dominates at 32.6% (9,486 samples)
DoS: only 246 samples, 38.6× fewer than IntegerUO
Co-occurrence matrix: computed from actual multilabel_index.csv
P0-A: Fix Three-Tier Predictor Thresholding
FOR — why this genuinely helps
The fix is real. self.thresholds (loaded at predictor.py:298-336) are never consulted in the tiering loop (predictor.py:698-715) — the code reads conf_thr = self.tier_confirmed_threshold (0.55) for every class. This means threshold tuning work is partially wasted: the Optuna result changes what *_thresholds.json contains, but the predictor's tier output doesn't use it. Classes whose Optuna threshold is 0.35 need a score of 0.55 to appear as CONFIRMED. The fix creates a coherent loop between calibration effort and deployed behavior.

AGAINST — why it may not deliver what's expected
The fix changes what the API reports, not what the model predicts. The macro-F1 metric computed by trainer.py:evaluate() uses the same per-class grid sweep (linspace 0.1-0.9) that tune_threshold.py uses — neither of these is affected by the predictor tier design. The _format_result() path is only exercised in inference (predict_source()), not during training or validation. So this bugfix improves deployment behavior but contributes zero to the core training metric that determines whether Run 10 beats Run 9.

Second concern: the proposed SUSPICIOUS boundary self.thresholds[i] * 0.7 is arbitrary. If Optuna sets threshold[DoS] = 0.15 (because DoS is rare and the model barely fires on it), then suspicious_boundary[DoS] = 0.105 — essentially identical to the NOTEWORTHY floor of 0.10. DoS would almost never show as SUSPICIOUS even when there's real signal. The multiplier 0.7 has no grounding in the data.

Verdict: Do it — the bug is real. But don't expect F1 improvement. Expect better-calibrated API output, nothing more.

P0-B: Optuna Joint Threshold Search
FOR — why this genuinely helps
The independent per-class grid misses cross-class interactions that are empirically strong. From the actual data:

P(Reentrancy | DoS) = 0.985 — these two classes are nearly always co-present
P(IntegerUO | MishandledException) = 0.961
P(IntegerUO | TOD) = 0.865
The current grid sets threshold[DoS] and threshold[Reentrancy] independently. It can't capture: "if DoS score is 0.40 AND Reentrancy score is 0.55, predict both." Optuna's TPE sampler searches the joint 10-dimensional space and naturally finds configurations that exploit these correlations. The observed gain from threshold tuning in Run 7 (+0.0349 fixed→tuned F1) suggests the model's raw outputs have real calibration headroom.

AGAINST — why it may fail to deliver
The DoS co-occurrence trap is a precision-recall fraud. DoS has 246 training samples. P(Reentrancy | DoS) = 0.985 means Optuna will find: set threshold[DoS] very low (≈0.15), so that whenever the model detects Reentrancy (which it does reasonably well), DoS is also triggered. This maximizes DoS recall (recovers ~98% of DoS samples via Reentrancy proxy) but produces massive DoS FPs on the ~96% of Reentrancy-labeled contracts that do NOT have DoS. The joint macro-F1 on the validation set improves because DoS recall goes from ~0 to ~0.8, but precision for DoS collapses.

Overfitting to the validation distribution. In Run 8, threshold tuning gave val tuned F1 = 0.2851 but test tuned F1 = 0.2307 — a 0.054 degradation. That was with a 19-point grid. Optuna's 2000 trials will fit the val distribution more aggressively. The gap between val-tuned and test-tuned F1 will likely widen further.

Verdict: Do it — it's computationally cheap and the joint search is conceptually better. But track test F1 alongside val F1. If val-tuned gains more than +0.02 over the grid result, treat it with suspicion.

P0-C: Temperature Scaling
FOR — why this genuinely helps
The empirical evidence from Run 9 is direct: tuned F1 peaked at ep20 (0.2907) and declined to ep30 (0.2875) while fixed F1 continued improving. This divergence is the textbook signature of overconfidence accumulation during training. Temperature scaling is theoretically sound: a single scalar T > 1.0 compresses all logits toward zero, pushing probabilities away from 0.40-0.67 clustering toward more separable values. It costs nothing at training time and the L-BFGS optimization converges in seconds.

AGAINST — why it may not work for this specific model
A single temperature T is wrong for a 10-class multi-label problem with 38.6× imbalance. Standard temperature scaling was designed for single-label classification where all classes share the same confidence regime. In this dataset, IntegerUO has 9,486 samples and the model has seen enough examples to be well-calibrated on it. DoS has 246 samples and the model barely fires on it — its logits hover near zero (underconfident, or uncalibrated in the opposite direction). A global T optimized on the val NLL is dominated by the majority classes: 57.9% safe samples + 32.6% IntegerUO examples. The resulting T rescales DoS and TOD logits by the same factor as IntegerUO, even though their calibration needs are completely different.

Evidence: the val NLL loss (which T is optimized against) at 57.9% safe samples is dominated by the model's confidence on safe predictions. If the model is overconfident that safe contracts are safe (low probability for all 10 classes → entropy is low), T will be pushed above 1.0 to soften those confident-safe predictions. This helps nothing — the model is correct that safe contracts are safe. T > 1.0 just spreads the already-correct-zero probabilities closer to 0.5.

Verdict: Low implementation cost, worth doing. But manage expectations: it will improve the FP explosion on complex contracts (the 6-8 simultaneous class firing). It will not materially help the missed classes (DoS, TOD, GasException). Consider class-specific temperature scaling as a refinement after seeing the single-T result.

P0-D: MC Dropout Uncertainty
FOR — why this genuinely helps
For the ZK audit deployment context, uncertainty quantification is a genuine value-add. An auditor needs to know not just "is this vulnerable?" but "how confident is the model?" The implementation is provably correct — model.train() activates dropout, 30 forward passes with different masks, std captures epistemic uncertainty. The API design (separate /predict/uncertainty endpoint, not default) correctly isolates the 30× cost.

AGAINST — why it will not catch the specific FPs you care about
The FP explosion on contracts 09/11/20 is CONSISTENT, not uncertain. MC Dropout measures variance across dropout masks. The bad predictions on complex contracts (6-8 classes simultaneously) occur because the model has learned "contract complexity → vulnerability" as a stable feature — stable meaning both dropout paths encode it the same way. If both paths agree "complex contract → CallToUnknown", std([0.65, 0.63, 0.66...]) ≈ 0.01 — well below the 0.15 review threshold. MC Dropout flags inconsistent predictions, not confidently-wrong ones.

The uncertainty source is smaller than expected. The model.train() call activates dropout in: LoRA layers (if dropout was set, typically 0.1), GAT attention dropout, and the classifier MLP dropout. But CodeBERT's 12 transformer layers are frozen — their weights don't vary across passes. The GNN uses LayerNorm (no running stats), so its behavior is identical in train() and eval(). The actual variation between 30 passes is concentrated in LoRA dropout paths (r=16 adapters) and GAT attention dropout. At LoRA dropout=0.1, the variation on Q/V projections is small. In practice, std across 30 passes may be < 0.03 for most predictions — the 0.15 threshold will almost never trigger.

Verdict: Implement it for the deployment story (ZK audit tool with quantified confidence), not for FP detection. The useful output is probability mean, not std. Don't claim it will catch the FP explosion — it won't.

P1-A: Label Dependency Graph
FOR — why this genuinely helps
The LabelDependencyLayer has two properties that make it theoretically sound: (1) identity initialization ensures it starts as a no-op — it cannot make things worse from epoch 0, and (2) if co-occurrence signal is noise, backprop drives W toward zero (learned suppression). The co-occurrence matrix has real structure: P(Reentrancy|DoS) = 0.985. If the model predicts DoS at probability 0.40, the layer should learn to also amplify Reentrancy — and this is a structurally valid relationship in the data.

AGAINST — the IntegerUO dominance problem
This is the strongest objection and it's grounded in the actual co-occurrence numbers:


P(IntegerUO | GasException)          = 0.766
P(IntegerUO | TOD)                   = 0.865  
P(IntegerUO | MishandledException)   = 0.961
P(IntegerUO | Timestamp)             = 0.716
P(IntegerUO | CallToUnknown)         = 0.683
P(IntegerUO | ExternalBug)           = 0.664
The adjacency matrix encodes: "for almost every class, IntegerUO co-occurs strongly." When the layer computes neighbor_signal = logits @ adj.T, the dominant signal everywhere is "IntegerUO should be higher." IntegerUO already achieves F1=0.698 in Run 7. The layer will spend most of its learned capacity on a class that doesn't need help.

The inverse direction is worse. The layer also learns from P(X | IntegerUO). IntegerUO's strongest neighbors are:

Reentrancy: P(Reentrancy|IntegerUO) = 0.499
MishandledException: P(MishandledException|IntegerUO) = 0.290
So when IntegerUO fires (which it does for 32.6% of all contracts), the layer amplifies Reentrancy and MishandledException. These classes don't need amplification — Reentrancy has F1=0.4+ in Run 7. The layer provides no uplift for DoS (246 samples), TOD (135 single-label samples), or the classes actually at F1=0.

The layer cannot create signal where none exists. DoS has F1≈0 because the model's logit for DoS is near zero. If logits[DoS] ≈ 0.0, then neighbor_signal[DoS] = sum(adj[DoS, j] * logits[j]) = mostly IntegerUO's signal scaled by P(IntegerUO|DoS)=0.009. The layer can only amplify existing signal — zero input produces zero output regardless of W.

Verdict: This will help already-good classes marginally and do nothing for the F1=0 classes. Consider modifying the adjacency construction to exclude IntegerUO as a neighbor (cap its outgoing edges at 0.0) and instead focus the graph on the missed-class relationships: DoS↔Reentrancy (0.985), GasException↔MishandledException (0.275).

P1-B: R-Drop Regularization
FOR — why this genuinely helps
The intuition targets a real problem: if different dropout paths produce correlated wrong predictions, the model has learned a spurious feature that's robust to dropout. R-Drop's KL penalty forces the model to find features that are consistent regardless of dropout mask — which means the stable features must come from the graph structure, not from spurious size/complexity correlations.

AGAINST — the FP cause is not dropout inconsistency
The FP explosion is not caused by dropout inconsistency. The audit finding H3 confirmed the label ceiling comes from data quality — the model learned "complexity proxy" for all 10 classes. A spurious complexity feature is stable across dropout masks (both masks propagate the same graph structure). R-Drop penalizes KL(p1 || p2) where p1, p2 are stochastic forward passes. If the model consistently predicts "complex contract → CallToUnknown" for both masks, KL(p1||p2) ≈ 0 and the R-Drop term provides no gradient. The pattern you're trying to break is precisely the pattern R-Drop can't break.

The DoS gradient scaling entanglement is non-trivial. The trainer applies _logits_for_loss at line 672-678 to scale DoS gradients by 50% (BUG-H6). With R-Drop:

logits_1, aux_1 = model(...) → apply DoS scaling → task_loss_1
logits_2, aux_2 = model(...) → apply DoS scaling → task_loss_2
kl_loss = KL(sigmoid(logits_1) || sigmoid(logits_2)) — computed on unscaled logits
The KL term uses the raw logit-derived probabilities, not the DoS-scaled version. This creates a gradient path through DoS that bypasses the 50% scaling. The intent of dos_loss_weight=0.5 was to prevent DoS from dominating the gradient — R-Drop partially undoes this.

The aux loss entanglement is also non-trivial. aux_loss uses aux["jk_entropy"] from both passes. JK entropy regularization (λ=0.005) appears in both loss computations. With two passes, JK entropy is computed twice and accumulated — effective lambda doubles. This may over-regularize the JK aggregation in Run 10 vs Run 9.

Training time. 8 grad accum steps × 2 forward passes × 48 min/ep baseline = ~67 min/ep. 80 epochs = ~89h. If the experiment fails (no F1 improvement), you've spent nearly 4 days. The proposal doesn't specify a kill condition.

Verdict: The motivation is partially correct but targets the wrong mechanism. R-Drop may still provide regularization benefits unrelated to the stated goal. But implement with explicit kill condition: if val F1 after ep20 smoke test is < Run 9's ep20 (0.2907), abort and run without R-Drop.

P1-C: SWA (Post-hoc Averaging)
FOR — why this genuinely helps
Checkpoint averaging finds flat loss minima. This is the most relevant property for this project given the audit finding: test_contracts are massively OOD (median 20 nodes vs 90 training). Flat minima generalize better to OOD distributions than sharp minima. Post-hoc averaging is scheduler-safe — no interaction with OneCycleLR, no torch.compile complications.

AGAINST — specific failure modes for this architecture
update_bn() is a no-op. The entire model uses LayerNorm, not BatchNorm. GNN uses LayerNorm after each phase. CodeBERT uses LayerNorm in all 12 transformer layers. LayerNorm doesn't track running statistics — it normalizes per-sample at forward time. update_bn(train_loader, swa_model) traverses the model, finds zero BatchNorm modules, and returns immediately. The 35-minute "one full pass" the proposal budgets for is dead compute. This isn't a reason to skip SWA, but the estimate should be corrected.

"Top-5 by val F1" averaging mixes different phases of training. Run 9 shows clear phase behavior: ep10 tuned F1=0.2836, ep20 peak=0.2907, ep30=0.2875. The "best" checkpoints by val F1 may be from different training phases where the model has different calibration states. Averaging a peak-recall ep20 checkpoint with a better-precision ep50 checkpoint gives you a model that's neither. The resulting average may have lower effective F1 than either individual checkpoint.

The intermediate checkpoint save is 20GB. On a run that takes 64h, each checkpoint save at epoch end adds ~30 seconds of I/O. 8 saves = ~4 minutes total. Acceptable. But 20GB of intermediate checkpoints will persist after training unless explicitly cleaned up — the proposal should include cleanup logic in swa_average.py.

Verdict: Valid technique, wrong rationale (update_bn cost is not real). The real question is whether Run 10's later checkpoints will be worth averaging. If Run 10 converges cleanly to a stable val F1 (like Run 7's ep32-39 plateau), SWA will help. If Run 10 oscillates (like Run 9's ep35-37 dip), the averaged result degrades.

P2-A: Curriculum Sampler
FOR — why this genuinely helps
The warmup composition argument is sound. During gnn_prefix_warmup_epochs (ep1-15), the GNN is training while CodeBERT is insulated from untrained prefix tokens. If these early epochs also use simple-contract curriculum, the GNN learns stable Phase 1 representations on clean examples before Phase 2 CF analysis introduces noise. The mixing floor (15% hard samples) prevents the safe-contracts-only collapse.

AGAINST — the label-complexity correlation is likely to fail the gate
The gate condition will likely fail. Look at the actual label density distribution from the real data:

40th percentile of label density = 0.0 (safe contracts)
60th percentile = 0.1 (one class)
80th percentile = 0.2 (two classes)
57.9% of training samples have label density = 0.0 (safe). Safe contracts are structurally simpler by design — they don't have reentrancy patterns, integer overflow paths, or external call chains. The complexity score (log(nodes) + log(edges) + ...) will be lower for safe contracts almost by construction. np.corrcoef(complexity_scores, label_density) will very likely exceed 0.4.

Even the decorrelation fallback has a problem. Residualizing complexity against label density (scores_decorr = scores - reg.predict(label_density)) removes the linear correlation but not higher-order correlations. A contract with label_density=0.1 (one vulnerability) but very high complexity is treated the same as a contract with label_density=0.1 but low complexity. The decorrelated score is a complexity measure conditioned on having the same label density — this is statistically valid but the residuals may not carry useful curriculum structure anymore.

Safe contracts clustering at complexity 40th percentile means the curriculum samples mostly safe contracts early. The model trains primarily on safe examples (easy negatives) for the first 15 epochs. This may lead to class imbalance in early gradient signal: the four eyes become very good at predicting "safe" but the positive-class heads starve. The mixing floor (15% hard = 15% of complex vulnerable contracts) may be insufficient to balance this.

Verdict: Run the correlation gate analysis first (2-4h, cheap). If rho > 0.4 (likely), the decorrelation attempt will probably fail too given the dataset structure. Be prepared to abandon P2-A entirely rather than force it through a failed gate.

P2-B: Self-Paced Learning
FOR — why this genuinely helps
Per-sample loss as difficulty signal is more adaptive than static complexity. If the model truly can't learn from GasException samples in early epochs (high loss), excluding them initially and including them when the threshold λ rises is theoretically sound. The model builds capacity on easier classes first, creating better representations for Phase 2 where GasException detection matters.

AGAINST — the feature bottleneck is not a scheduling problem
GasException has 3,392 training samples and F1=0 in Run 7. 3,392 is not a data scarcity problem — that's more samples than UnusedReturn (1,837) which achieves F1=0.238. The model fails on GasException not because it doesn't see enough examples but because the current node features don't encode the GasException-specific graph motif (external call in a gas-limited context). Self-paced learning changes when the model sees GasException examples, not what features it uses to detect them. Scheduling cannot overcome a feature representation gap.

TOD has 135 single-label contracts. That's the cleanest possible training signal — TOD-only contracts, no co-occurrence noise. Yet Run 7 achieves F1=0 on TOD. If 135 pure-TOD examples don't teach the model to detect TOD over 41 epochs, including them earlier or later changes nothing. The problem is that TOD detection requires recognizing state-read/state-write ordering across transactions — a temporal graph pattern that the current CFG edges don't capture.

The λ growth schedule is decoupled from training dynamics. λ = 2.0 * (1 + epoch * 0.05). At epoch 40, λ = 6.0. If the model's average loss at epoch 40 is 0.25 (well-converged), then λ=6.0 includes every sample (no exclusion). The self-pacing mechanism has effectively turned off by the midpoint of training. The proposal notes "compute every 5 epochs" — but if the schedule grows unconditionally, you're paying the computational cost of a mechanism that becomes inert after ep25-30.

Verdict: Theoretically appealing but the core premise is wrong for this dataset. GasException and TOD fail due to feature representation, not training schedule. Self-paced learning is worth trying only after a feature improvement (new edge types, new node features) makes the missed classes more learnable.

P3-A: Static Calibrator (SCsVulLyzer Impossibility Rules)
FOR — why this genuinely helps
The logic is deterministic and sound. If compiled bytecode has zero CALL opcodes, Reentrancy is bytecode-impossible. This isn't a probabilistic claim — it's a fact about EVM execution semantics. Applied post-inference, it removes FPs that the ML model cannot self-correct because it learned statistical correlations rather than structural impossibilities.

AGAINST — the rules may rarely fire on real-world contracts
Most contracts have non-zero CALL counts. The rule for Reentrancy fires only when CALL=0 AND DELEGATECALL=0. Real-world ERC20 contracts transfer tokens (token.transfer() = CALL), governance contracts call voting modules (CALL), most non-trivial contracts use at least one external call. The contracts where CALL=0 are likely: pure storage contracts, trivial math contracts, or view-only contracts. These are exactly the contracts the model is unlikely to predict Reentrancy on in the first place (because they have simple graph structure and low complexity score). The rule fires where it's not needed and doesn't fire where it would help.

The DenialOfService rule requires CALL=0 AND num_loops=0. This doubly restricts the condition. Contracts 09/11/20 (the FP explosion contracts) likely have both CALL > 0 (they're complex contracts) and loops > 0. The impossibility rules don't apply to them.

Compilation failures in production. The proposal notes "2-5s per contract for solc compilation." But many real-world contracts require specific solc versions (pragma directives), have multi-file imports, or are proxy patterns that reference deployed contracts. If analyze_solidity_contract() fails, the calibrator must fall back silently — and the fallback is "pass through without calibration," which is the current behavior. The operational complexity is non-trivial for near-zero improvement on the contracts that actually matter.

Verdict: Correct in principle, limited in practice for this dataset. The most actionable impossibility rule (CALL=0 → no Reentrancy) fires only on contracts where the model wouldn't predict Reentrancy anyway. Defer until SCsVulLyzer is installed and profiled on the real val set to measure actual coverage.

Optuna Hyperparameter Sweep (§7)
FOR
lora_r=16 was chosen as a round number. For a 124M frozen model where only Q and V projections are adapted, r=8 halves LoRA parameters. If Run 7's F1=0.3074 was achieved with frozen backbone dominating, smaller LoRA rank might reduce overfitting on minority classes.

AGAINST — 15-epoch proxies are unreliable for this model
The model doesn't converge before ep25. Run 4's best was ep32, Run 7's ep39. A 15-epoch proxy measures early dynamics (is the training stable? is there JK collapse?) not final quality. lora_r=8 vs lora_r=16 may show indistinguishable F1 at ep15 but diverge at ep35. The 27h sweep of 15-epoch runs will produce a ranking that doesn't transfer to 80-epoch runs.

asl_gamma_neg=6 risks all-zeros collapse. Run 3 collapsed with gamma_neg=4 plus the double-amp bug. The double-amp bug is fixed, but γ_neg=6 is even more aggressive at suppressing easy negatives. In this dataset, 57.9% of training samples are safe (easy negatives). γ_neg=6 suppresses these even more heavily, concentrating gradient on the 42.1% positive examples. For a model that already struggles with class imbalance (DoS 246 vs IntegerUO 9,486), this further amplifies the imbalance effect — IntegerUO positive examples get disproportionate gradient.

Verdict: Defer until after Run 10 establishes a clean baseline. The 15-epoch proxy issue is the main concern — either use 30-epoch proxies (doubling the ~81h estimate) or accept that the sweep may rank configurations incorrectly.