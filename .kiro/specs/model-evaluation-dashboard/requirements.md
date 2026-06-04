# Requirements Document

## Introduction

Model Evaluation Dashboard is a web-based API suite for the Sentinel smart contract vulnerability detection system. It enables users to load models, run evaluations on test/validation datasets, view comprehensive metrics, analyze errors, and tune classification thresholds—all via RESTful endpoints.

## Glossary

- **SentinelModel**: The GNN + Transformer (CodeBERT) dual-path model for vulnerability detection (v7+ architecture).
- **Checkpoint**: A serialized model file (.pt) stored in ml/checkpoints/ containing weights, optimizer state, and config.
- **Evaluation Result**: Cached inference outputs (predictions, probabilities, ground truth) for a specific checkpoint/dataset split combination.
- **Threshold**: Per-class decision boundary (0.0-1.0) used to convert probabilities to binary predictions.
- **Metrics**: Quantitative measures including precision, recall, F1 score, ROC-AUC, and confusion matrix.
- **Vulnerability Class**: One of the 10 supported vulnerability types: CallToUnknown, DenialOfService, ExternalBug, GasException, IntegerUO, MishandledException, Reentrancy, Timestamp, TransactionOrderDependence, UnusedReturn.
- **Test Split**: A held-out portion of the ~41K Solidity contracts used for model evaluation.
- **Validation Split**: A portion used during training for hyperparameter tuning.
- **Cache**: Persistent storage of evaluation results for fast subsequent retrieval.

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

---

*This document defines the requirements for the Model Evaluation Dashboard feature. All acceptance criteria follow EARS patterns and INCOSE quality rules for testable, unambiguous requirements.*