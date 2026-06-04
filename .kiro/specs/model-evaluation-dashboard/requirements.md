# Requirements Document

## Introduction

Model Evaluation Dashboard is a web-based API suite for the Sentinel smart contract vulnerability detection system. It enables users to load models, run evaluations on test/validation datasets, view comprehensive metrics, analyze errors, and tune classification thresholds—all via RESTful endpoints.

This specification is informed by Run 7 training analysis (GCB-P1-Run7-v10-20260603) and Phase 2 interpretability experiments (2026-06-04), which revealed:
- Significant gap between fixed-threshold F1 (0.3074) and tuned F1 (0.3329) at ep39
- Structural ceilings for UnusedReturn (~0.234) and Timestamp (~0.145) due to missing graph edges
- DoS class highly noisy due to only 65 positive samples in validation (1.04% prevalence)
- JK phase weights show Phase 3 dominance (0.395 by ep40) but near-uniform per-sample activation (entropy 1.094/1.099 = 99.5% of max)
- GNN gradient share settled at ~30% with transformer taking ~70% of gradient
- **Ensemble calibration gap:** 4-eye Main ECE=0.233 is 5.8× worse than individual eyes (GNN=0.046, TF=0.040, Fused=0.040); temperature scaling must target Main output logits
- **Complexity dominance:** `complexity` feature (dim 5) dominates all 10 classes at 34–36% of gradient signal; class-specific features (`return_ignored`, `external_call_count`, `uses_block_globals`) are not used discriminatively
- **Edge ablation near-zero:** Removing any single edge type changes predictions by ≤0.013; model uses node-level proxy (complexity), not graph topology
- **DoS structurally detected:** Transformer AUC=0.559 (near random), GNN AUC=0.726, Fused AUC=0.803
- **fusion_max_nodes=1024 truncation:** 227 contracts exceed limit, silently dropping nodes from fusion cross-attention

## Glossary

- **SentinelModel**: The GNN + Transformer (CodeBERT) dual-path model for vulnerability detection (v7+/v8.1 Four-Eye architecture).
- **Checkpoint**: A serialized model file (.pt) stored in ml/checkpoints/ containing weights, optimizer state, config, RNG states, and tuned thresholds.
- **Evaluation Result**: Cached inference outputs (predictions, probabilities, ground truth) for a specific checkpoint/dataset split combination.
- **Threshold**: Per-class decision boundary (0.0-1.0) used to convert probabilities to binary predictions.
- **Metrics**: Quantitative measures including precision, recall, F1 score, ROC-AUC, confusion matrix, and calibration metrics.
- **Vulnerability Class**: One of the 10 supported vulnerability types: CallToUnknown, DenialOfService, ExternalBug, GasException, IntegerUO, MishandledException, Reentrancy, Timestamp, TransactionOrderDependence, UnusedReturn.
- **Test Split**: A held-out portion of the ~41K Solidity contracts used for model evaluation.
- **Validation Split**: A portion used during training for hyperparameter tuning.
- **Cache**: Persistent storage of evaluation results for fast subsequent retrieval.
- **JK Phase Weights**: Learned attention weights over three GNN phases (structural, CFG+ICFG, containment) indicating which phase contributes most to predictions. Run 7 shows near-uniform distribution (entropy 99.5% of max) with mild Phase 3 drift.
- **Structural Ceiling**: A class where performance cannot improve without architectural changes (e.g., DEF_USE edges for UnusedReturn). Run 7 confirmed: UnusedReturn ~0.234, Timestamp ~0.145.
- **Complexity Dominance**: The phenomenon where `complexity` feature (node dim 5) captures 34–36% of gradient signal across all 10 classes, suppressing class-specific feature learning. Identified in L4 gradient saliency experiment.
- **Ensemble Calibration Gap**: The observation that concatenating four well-calibrated eyes (ECE 0.040–0.046) through Linear(512,256)→Linear(256,10) produces a 5.8× worse calibrated output (ECE 0.233). Requires post-training temperature scaling on Main logits.

## Requirements

### Requirement 1: Checkpoint Listing and Loading

**User Story:** As a model evaluator, I want to list available model checkpoints and load one for evaluation, so that I can compare different model versions.

#### Acceptance Criteria

1. THE CheckpointManager SHALL list all files with extensions .pt, .pth, or .state.json from the ml/checkpoints/ directory.
2. IF the ml/checkpoints/ directory does not exist, THE CheckpointManager SHALL return an empty list; IF the directory exists but contains no qualifying files, THE CheckpointManager SHALL return an empty list.
3. WHEN a checkpoint filename is provided, THE CheckpointManager SHALL load the model weights and config from the checkpoint file into memory.
4. IF the provided checkpoint filename does not exist in the checkpoint directory, THE CheckpointManager SHALL return an error indicating the file was not found.
5. THE CheckpointManager SHALL support loading up to 5 checkpoints simultaneously for side-by-side comparison.
6. WHERE a checkpoint includes per-class threshold definitions stored as a 'thresholds' key in the checkpoint file, THE CheckpointManager SHALL load those thresholds; OTHERWISE it SHALL use a default threshold of 0.50 for all classes.
7. THE CheckpointManager SHALL validate checkpoint integrity by verifying the checkpoint file can be loaded and contains required model weights.
8. IF checkpoint validation fails due to file corruption, THE CheckpointManager SHALL return an error indicating the checkpoint is corrupted.
9. IF checkpoint validation fails because the checkpoint format is incompatible with the current system, THE CheckpointManager SHALL return an error indicating incompatibility.

### Requirement 2: Dataset Evaluation

**User Story:** As a model evaluator, I want to run inference on test or validation dataset splits, so that I can measure model performance.

#### Acceptance Criteria

1. WHEN a model checkpoint path and dataset split name are provided, THE Evaluator SHALL run inference on all samples in that split, and IF the split contains zero samples, THE Evaluator SHALL return an empty result set.
2. THE Evaluator SHALL return raw logits, probabilities, and binary predictions for each sample, WHERE the output for binary mode SHALL contain arrays of shape [num_samples, 2] for logits and probabilities, and shape [num_samples] for binary predictions, AND WHERE multi-label mode is enabled, the output SHALL contain arrays of shape [num_samples, 10] for logits and probabilities, and shape [num_samples, 10] for binary predictions.
3. THE Evaluator SHALL support both binary mode (single class) and multi-label mode (10 classes), WHERE the mode is determined by a configuration parameter.
4. WHERE evaluation results for a specific checkpoint and split already exist in cache, THE Evaluator SHALL return cached results instead of re-running inference, WHERE cache lookup uses a SHA-256 hash of the checkpoint file path concatenated with the split name as the cache key.
5. THE CacheManager SHALL store evaluation results keyed by checkpoint hash and dataset split identifier, WHERE the checkpoint hash is computed using SHA-256 and the split identifier is the split name string.
6. WHEN a model checkpoint is re-evaluated (same checkpoint path and split), THE CacheManager SHALL invalidate the existing cached results before storing new evaluation results.

### Requirement 3: Metrics Computation and Display

**User Story:** As a model evaluator, I want to view comprehensive performance metrics, so that I can understand model strengths and weaknesses.

#### Acceptance Criteria

1. WHEN evaluation results are available, THE MetricsEngine SHALL compute per-class precision, recall, and F1 score for each of the 10 vulnerability classes.
2. THE MetricsEngine SHALL compute overall accuracy, macro-averaged F1 (mean of per-class F1), and micro-averaged F1 (aggregate TP, FP, FN across all classes).
3. THE MetricsEngine SHALL compute ROC-AUC for each vulnerability class and the macro-averaged ROC-AUC.
4. THE MetricsEngine SHALL compute the confusion matrix for binary mode; FOR multi-label mode, THE MetricsEngine SHALL compute per-class confusion matrix entries (TP, FP, TN, FN).
5. THE MetricsEngine SHALL compute precision-recall curve data points for each class at threshold values [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9].
6. THE MetricsEndpoint SHALL return all computed metrics in a structured JSON response.
7. IF predictions and labels have mismatched dimensions, THE MetricsEngine SHALL return an error indicating dimension mismatch.
8. THE MetricsEngine SHALL derive classification mode from label array shape, WHERE shape [N] indicates binary mode and shape [N, 10] indicates multi-label mode.
9. THE MetricsEngine SHALL compute valid metrics for classes with data, and SHALL report "N/A" for classes with zero positive samples in ground truth.
10. IF all samples in ground truth are negative for a given class, THE MetricsEngine SHALL report ROC-AUC as "N/A" for that class.

### Requirement 4: Error Analysis

**User Story:** As a model evaluator, I want to examine misclassified samples with context, so that I can identify patterns in model failures.

#### Acceptance Criteria

1. THE ErrorAnalyzer SHALL identify all samples where the predicted class differs from the true label.
2. FOR each misclassified sample, THE ErrorAnalyzer SHALL include the predicted class, true class, confidence score in the range 0.0 to 1.0, and sample identifier.
3. THE ErrorAnalyzer SHALL group errors by vulnerability type (true class) and include the error count per group, sorted in descending order by count.
4. THE ErrorAnalyzer SHALL include contract source code context for each error case, showing a code snippet of up to 20 lines centered on the line where the vulnerability exists, or the entire function if fewer than 20 lines.
5. IF source code cannot be retrieved for a misclassified sample, THE ErrorAnalyzer SHALL indicate that source code is unavailable for that sample.
6. WHERE filtering parameters are provided, THE ErrorAnalyzer SHALL filter errors by predicted class, true class, or confidence range (min/max probability where min is 0.0 and max is 1.0). IF no errors match the filter criteria, THE ErrorAnalyzer SHALL return an empty error list.

### Requirement 5: Threshold Tuning

**User Story:** As a model evaluator, I want to adjust per-class thresholds and see real-time metric updates, so that I can optimize classification performance.

#### Acceptance Criteria

1. THE ThresholdTuner SHALL accept a dictionary of per-class threshold values, WHERE each value is in the range 0.0 to 1.0 inclusive.
2. WHEN thresholds are adjusted, THE ThresholdTuner SHALL recompute all binary predictions and metrics within 1000 milliseconds.
3. THE ThresholdTuner SHALL compute optimal thresholds by maximizing per-class F1 score across candidate values [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9].
4. THE ThresholdTuner SHALL return both the original metrics and the metrics after threshold adjustment for comparison.
5. WHERE optimization is requested, THE ThresholdTuner SHALL suggest optimal thresholds that maximize macro-averaged F1 score, WHERE the threshold values for each class are independently optimized and the binary prediction rule is "probability >= threshold".
6. IF the threshold input dictionary contains a class key that does not exist in the model, THE ThresholdTuner SHALL return an error indicating class mismatch.

### Requirement 6: Caching and Performance

**User Story:** As a system operator, I want evaluation results cached for fast retrieval, so that repeated analysis is efficient.

#### Acceptance Criteria

1. THE CacheManager SHALL store evaluation results in a persistent cache directory (ml/cache/evaluation/).
2. THE CacheManager SHALL generate cache keys using SHA-256 hash of the checkpoint file path, with format "<hash>_<split_name>".
3. THE CacheManager SHALL automatically invalidate cached results when the checkpoint file hash changes (detected by file modification time or content hash mismatch).
4. THE CacheManager SHALL explicitly invalidate cached results when an invalidation request is received for a specific checkpoint and split combination.
5. THE CacheManager SHALL support cache inspection by returning a list of all cached entries, WHERE each entry includes checkpoint identifier, split name, timestamp (ISO 8601 format), and size in bytes.
6. WHERE a cached entry exists for the given checkpoint and split, THE CacheManager SHALL return the cached results.

### Requirement 7: JK Phase Weight Analysis

**User Story:** As a model researcher, I want to visualize JK phase attention weights, so that I can understand which GNN phase contributes most to predictions.

#### Acceptance Criteria

1. THE MetricsEngine SHALL compute JK phase attention weights (Phase1, Phase2, Phase3) from the model during evaluation.
2. THE MetricsEngine SHALL compute the mean and standard deviation of each phase weight over the evaluation dataset.
3. THE MetricsEngine SHALL track phase weight drift over training epochs, showing how each phase's contribution changes.
4. THE MetricsEndpoint SHALL return phase weights in the format: {"phase1": {"mean": float, "std": float}, "phase2": {...}, "phase3": {...}}.
5. IF phase weights indicate Phase3 > 0.40, THE MetricsEngine SHALL emit a warning about potential Phase3 dominance.
6. THE MetricsEngine SHALL compute JK entropy (Shannon entropy over 3 phase weights) per sample and report mean entropy across the dataset. Entropy range is [0, log(3)=1.099]; values below 0.50 indicate phase collapse.
7. THE MetricsEngine SHALL flag JK entropy decline > 0.01 over 10 epochs as a Phase 3 drift warning, per Run 7 finding (entropy declined −0.0082 over 40 epochs).
8. THE MetricsEngine SHALL compute per-class JK phase weight distributions to identify whether specific vulnerability classes preferentially use specific GNN phases. Run 7 L1 showed Phase 3 dominance is uniform across all classes (spread = 0.009), falsifying the per-class routing hypothesis.

### Requirement 8: Structural Ceiling Detection

**User Story:** As a model researcher, I want to identify when a class has hit a structural ceiling, so that I know when architectural changes are needed.

#### Acceptance Criteria

1. THE MetricsEngine SHALL track per-class F1 trend over the last N evaluation runs (configurable, default 10).
2. THE MetricsEngine SHALL identify a class as at "structural ceiling" WHEN the F1 delta over the last N runs is less than 0.01 AND the class requires graph edges not currently in the schema.
3. THE MetricsEngine SHALL flag UnusedReturn as having a structural ceiling at F1 ≈ 0.234 due to `return_ignored` feature not being used discriminatively (L4/B4: same feature ranking for positive and negative samples; DEF_USE edges needed to connect flag to vulnerability pattern).
4. THE MetricsEngine SHALL flag Timestamp as having a structural ceiling at F1 ≈ 0.145 due to `uses_block_globals` being only marginally elevated (10.7% vs 10.0% baseline) — the model treats it as background noise.
5. THE MetricsEngine SHALL flag TransactionOrderDependence as having a structural ceiling due to ensemble noise: Main AUC=0.794 vs TF-only AUC=0.813 — the GNN and CFG eyes hurt TOD detection.
6. THE MetricsEndpoint SHALL include ceiling flags in the per-class metrics response.

### Requirement 9: Micro vs Macro F1 Analysis

**User Story:** As a model evaluator, I want to understand the class imbalance impact, so that I can quantify the macro penalty.

#### Acceptance Criteria

1. THE MetricsEngine SHALL compute both micro-F1 and macro-F1 for each evaluation.
2. THE MetricsEngine SHALL compute the spread as micro_F1 - macro_F1, quantifying the class imbalance penalty.
3. THE MetricsEngine SHALL estimate the theoretical macro-F1 ceiling as micro_peak - 0.04 to 0.05 based on class imbalance analysis.
4. THE MetricsEndpoint SHALL return micro-F1, macro-F1, spread, and estimated ceiling in the metrics response.

### Requirement 10: DoS Noise Analysis

**User Story:** As a model evaluator, I want to understand DoS class variance, so that I can distinguish real learning from noise.

#### Acceptance Criteria

1. THE MetricsEngine SHALL compute the positive sample count for DenialOfService in the evaluation set.
2. THE MetricsEngine SHALL compute DoS F1 variance across consecutive evaluations to identify noise levels.
3. THE MetricsEngine SHALL flag DoS as "noisy" WHEN the positive sample count is less than 100 OR F1 variance exceeds 0.05 between evaluations.
4. THE MetricsEndpoint SHALL include DoS-specific metrics: positive_count, f1_variance, is_noisy flag.
5. THE MetricsEngine SHALL report DoS per-eye AUC to distinguish structural vs semantic detection. Run 7 A4 confirmed: Transformer AUC=0.559 (near random), GNN AUC=0.726, Fused AUC=0.803 — DoS detection is entirely structural, not semantic.
6. THE MetricsEngine SHALL flag DoS as "structurally detected" when GNN AUC exceeds Transformer AUC by >0.15, indicating that improving the transformer will not help DoS — only better structural representations (Phase 2 ICFG paths showing loop depth, call graph cycles) will.

### Requirement 11: Gradient Flow Analysis

**User Story:** As a model researcher, I want to see gradient flow between GNN and transformer components, so that I can diagnose training dynamics.

#### Acceptance Criteria

1. THE Evaluator SHALL compute GNN gradient share as a fraction of total trainable parameter gradient magnitude.
2. THE Evaluator SHALL compute the Phase2/Phase1 gradient ratio to verify Phase2 is receiving adequate gradients.
3. THE Evaluator SHALL return gradient metrics in the format: {"gnn_share": float, "ph2_ph1_ratio": float}.
4. IF GNN share drops below 0.20, THE Evaluator SHALL emit a warning about potential GNN under-training.
5. IF Phase2/Phase1 ratio is below 0.40, THE Evaluator SHALL emit a warning about Phase2 under-training.
6. THE Evaluator SHALL compute per-class Phase2/Phase1 and Phase3/Phase1 gradient ratios. Run 7 B1 confirmed: all classes show Ph2/Ph1 in 0.777–0.917 range (5–8× improvement over Run 4's 0.10–0.18). DoS shows highest Ph3/Ph1=0.931, indicating contract-identity shortcut.

### Requirement 12: Checkpoint Comparison

**User Story:** As a model evaluator, I want to compare multiple checkpoints side-by-side, so that I can track improvement across runs.

#### Acceptance Criteria

1. THE CheckpointManager SHALL support loading up to 5 checkpoints simultaneously as specified in Requirement 1.5.
2. THE MetricsEngine SHALL compute comparative metrics when multiple checkpoints are loaded, showing delta F1 per class.
3. THE MetricsEngine SHALL compute training epoch-level metrics (JK weights, gradient share) from checkpoint metadata where available.
4. THE MetricsEndpoint SHALL return comparison table with checkpoint names, F1-macro, F1-micro, and per-class deltas.

### Requirement 13: Edge Ablation Analysis

**User Story:** As a model researcher, I want to know which edge types are most important for each vulnerability class, so that I can prioritize graph schema improvements.

#### Acceptance Criteria

1. THE MetricsEngine SHALL run edge ablation experiments, WHERE each experiment removes one edge type and measures the F1 delta.
2. The edge types to test SHALL include: CALLS(0), READS(1), WRITES(2), EMITS(3), INHERITS(4), CONTAINS(5), CONTROL_FLOW(6), REVERSE_CONTAINS(7), CALL_ENTRY(8), RETURN_TO(9), DEF_USE(10).
3. THE MetricsEngine SHALL report the F1 delta per class for each edge type removal.
4. THE MetricsEndpoint SHALL return ablation results in the format: {"edge_type": {"class_name": f1_delta, ...}, ...}.
5. THE MetricsEngine SHALL identify edge types as "critical" (|delta| > 0.02), "moderate" (0.01 < |delta| ≤ 0.02), or "low-impact" (|delta| ≤ 0.01).
6. THE MetricsEngine SHALL detect the **topology-agnostic pattern** when all edge ablation deltas are below 0.015. Run 7 L2 confirmed: largest delta = 0.013 (RETURN_TO for DoS), all others ≤ 0.005. This indicates the model uses node-level feature proxies (complexity) rather than graph topology.
7. THE MetricsEngine SHALL flag DEF_USE edges as having paradoxical effect when removal improves a class (Run 7: DEF_USE removal helped IntegerUO by +0.010, indicating noisy sparse edges add confusion rather than signal).

### Requirement 14: Node Feature Saliency Analysis

**User Story:** As a model researcher, I want to know which node features (of the 11) drive predictions for each class, so that I can validate graph extraction quality.

#### Acceptance Criteria

1. THE MetricsEngine SHALL compute gradient-based saliency for each of the 11 node features.
2. THE features SHALL include: type_id, visibility, uses_block_globals, view, payable, complexity, loc, return_ignored, call_target_typed, has_loop, external_call_count.
3. THE MetricsEngine SHALL compute per-class feature importance by aggregating gradients over samples where that class is positive.
4. THE MetricsEngine SHALL identify feature shortcuts as features with Cohen's d > 1.5 between positive and negative samples.
5. THE MetricsEndpoint SHALL return saliency in the format: {"class_name": {"feature_name": saliency_score, ...}, ...}.
6. THE MetricsEngine SHALL detect **complexity dominance** when the `complexity` feature (dim 5) captures >30% of total gradient signal across all classes. Run 7 L4 confirmed: complexity dominates at 34–36% for all 10 classes, while class-specific features (`return_ignored` at 7.7%, `external_call_count` at 7.5%, `uses_block_globals` at 9.7–10.7%) are not used discriminatively.
7. THE MetricsEngine SHALL compute per-class feature ranking and flag when the ranking is identical across all classes (indicating no class-specific feature learning). Run 7 L4 showed identical ranking: complexity → visibility → uses_block_globals for every class.

### Requirement 15: JK Phase Preference by Class

**User Story:** As a model researcher, I want to know which GNN phase each vulnerability class relies on most, so that I can interpret model behaviour.

#### Acceptance Criteria

1. THE MetricsEngine SHALL compute per-class JK phase weight distributions from the evaluation run.
2. THE MetricsEngine SHALL identify the dominant phase (Phase1, Phase2, or Phase3) for each vulnerability class.
3. THE MetricsEngine SHALL flag classes where Phase3 dominance exceeds 0.40 as potentially using contract-hierarchy shortcuts.
4. THE MetricsEndpoint SHALL return phase preferences in the format: {"class_name": {"phase1": weight, "phase2": weight, "phase3": weight, "dominant": "phaseN"}, ...}.

### Requirement 16: UnusedReturn Def-Use Analysis

**User Story:** As a model researcher, I want to understand why UnusedReturn has hit a structural ceiling at 0.234, so that I can plan graph schema improvements.

#### Acceptance Criteria

1. THE MetricsEngine SHALL compute the return_ignored feature activation rate for UnusedReturn-positive samples.
2. THE MetricsEngine SHALL compare return_ignored activation between UnusedReturn-positive and UnusedReturn-negative samples.
3. IF return_ignored activation is not significantly higher for positive samples, THE MetricsEngine SHALL confirm DEF_USE edges are needed.
4. THE MetricsEndpoint SHALL return analysis results including: positive_sample_rate, negative_sample_rate, statistical_significance.
5. THE MetricsEngine SHALL compute gradient saliency comparison between top-scored and bottom-scored UnusedReturn contracts. Run 7 B4 confirmed: feature ranking is identical in both groups (complexity→visibility→uses_block_globals→external_call_count→return_ignored), with return_ignored at rank 5 in both cases. Ratio return_ignored/complexity = 22.8% (top) vs 21.2% (bottom) — the model uses the same features for positive and negative decisions.
6. THE MetricsEngine SHALL flag UnusedReturn as "feature-not-discriminative" when return_ignored gradient saliency is within ±20% of its saliency for non-UnusedReturn classes. Run 7 L4 confirmed: return_ignored saliency = 0.077 for UnusedReturn, 0.077–0.081 for all other classes — zero class-specific elevation.

### Requirement 17: Calibration Analysis

**User Story:** As a model evaluator, I want to understand probability calibration quality, so that I can trust the confidence scores.

#### Acceptance Criteria

1. THE MetricsEngine SHALL compute Expected Calibration Error (ECE) for each class and overall.
2. THE MetricsEngine SHALL compute Brier scores per class.
3. THE MetricsEngine SHALL compute per-eye ECE to compare calibration quality across the four model paths.
4. THE MetricsEndpoint SHALL return calibration metrics in the format: {"ece_overall": float, "ece_per_class": {...}, "brier_per_class": {...}, "ece_per_eye": {...}}.
5. THE MetricsEngine SHALL detect the **ensemble calibration gap** when Main ECE exceeds individual eye ECE by more than 3×. Run 7 B2 confirmed: Main ECE=0.233 vs GNN=0.046, TF=0.040, Fused=0.040 — the 4-eye concat through Linear(512,256)→Linear(256,10) produces 5.8× worse calibration than any individual eye.
6. THE MetricsEngine SHALL recommend temperature scaling when ensemble calibration gap exceeds 3×, specifying that temperature should be applied to Main output logits (not individual eyes). Target: ECE ~0.027 (matching Run 4 post-calibration level).
7. THE MetricsEngine SHALL report per-class calibration quality and flag DoS as worst-calibrated (ECE=0.307 in Run 7) due to 1.04% prevalence causing absolute probability miscalibration.

### Requirement 18: Contract Size Analysis

**User Story:** As a model evaluator, I want to understand how model performance varies with contract size, so that I can identify truncation issues.

#### Acceptance Criteria

1. THE MetricsEngine SHALL stratify evaluation results by contract size (node count: small <100, medium 100-300, large >300).
2. THE MetricsEngine SHALL compute per-class F1 for each size stratum.
3. THE MetricsEngine SHALL flag classes where F1 significantly degrades for large contracts, indicating potential truncation.
4. THE MetricsEndpoint SHALL return size-stratified metrics in the format: {"size_stratum": {"class_name": f1, ...}, ...}.

### Requirement 19: Complexity Dominance Detection

**User Story:** As a model researcher, I want to detect when the complexity feature dominates all classes, so that I can identify when the model is using a proxy rather than class-specific signals.

#### Acceptance Criteria

1. THE MetricsEngine SHALL compute per-class feature importance distribution across all 11 node features.
2. THE MetricsEngine SHALL detect complexity dominance when the `complexity` feature (dim 5) captures >30% of total gradient signal for >80% of classes.
3. WHEN complexity dominance is detected, THE MetricsEngine SHALL emit a warning that class-specific features (`return_ignored`, `external_call_count`, `uses_block_globals`) are likely not being used discriminatively.
4. THE MetricsEngine SHALL compute per-class feature ranking and compare across classes. IF the ranking is identical for >80% of classes, THE MetricsEngine SHALL flag "uniform feature usage" — indicating the model has not learned class-specific discriminative features.
5. THE MetricsEndpoint SHALL return complexity dominance metrics: {"complexity_share": float, "is_dominant": bool, "class_specific_feature_shares": {...}, "ranking_uniformity": float}.

### Requirement 20: Ensemble vs Individual Eye Comparison

**User Story:** As a model researcher, I want to compare the ensemble (4-eye Main) against individual eyes, so that I can identify when the ensemble hurts specific classes.

#### Acceptance Criteria

1. THE MetricsEngine SHALL compute per-class AUC-ROC for each individual eye (GNN, TF, Fused, CFG) and the Main ensemble.
2. THE MetricsEngine SHALL compute the ensemble advantage as Main_AUC - max(individual_eyes_AUC) per class.
3. WHEN ensemble advantage is negative for a class, THE MetricsEngine SHALL flag that class as "ensemble-hurt" — the concatenation + linear layers add noise for that class.
4. THE MetricsEngine SHALL compute per-eye ECE to compare calibration quality. Run 7 B2 confirmed: individual eyes ECE=0.040–0.046, Main ECE=0.233 (5.8× worse).
5. THE MetricsEndpoint SHALL return per-eye metrics in the format: {"eye": {"class_name": {"auc": float, "ece": float}, ...}, "ensemble_advantage": {"class_name": float, ...}, "ensemble_hurt_classes": [str, ...]}.

### Requirement 21: Fusion Max Nodes Truncation Analysis

**User Story:** As a model evaluator, I want to understand how fusion_max_nodes truncation affects performance, so that I can decide whether to increase the limit.

#### Acceptance Criteria

1. THE MetricsEngine SHALL compute the distribution of graph node counts across the evaluation set.
2. THE MetricsEngine SHALL count how many graphs exceed fusion_max_nodes (default 1024) and report the fraction.
3. THE MetricsEngine SHALL compute per-class F1 for graphs above and below fusion_max_nodes. IF F1 significantly degrades for large graphs, THE MetricsEngine SHALL flag truncation impact.
4. THE MetricsEndpoint SHALL return truncation metrics: {"max_nodes": int, "graphs_exceeding": int, "fraction_exceeding": float, "per_class_f1_by_size": {...}}.

---

*This document defines the requirements for the Model Evaluation Dashboard feature. All acceptance criteria follow EARS patterns and INCOSE quality rules for testable, unambiguous requirements.*