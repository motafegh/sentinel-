# ml/analysis — Data Quality Validation

Comprehensive validation and analysis tools for the SENTINEL dataset.

## Purpose

This module contains scripts and reports for validating the structural, statistical, and semantic quality of the graph-token paired dataset used for training and inference.

## Contents

### `data_quality_validation.py`

Main validation script that performs comprehensive checks on the dataset:

- **Structural Validation**: Verifies file integrity, graph-token pairing, and schema consistency
- **Statistical Analysis**: Analyzes feature distributions, label balance, and data statistics
- **Semantic Validation**: Checks for logical consistency and data quality metrics

### `validation_report.json`

Static output from validation runs containing:
- Dataset statistics
- Quality metrics
- Any detected anomalies or issues

## Usage

```bash
# Run full validation suite
poetry run python ml/analysis/data_quality_validation.py \
    --graph-dir ml/data/graphs \
    --token-dir ml/data/tokens_windowed \
    --metadata-path ml/data/processed/metadata.parquet
```

## Validation Phases

1. **Phase 1: Structural Validation**
   - File existence checks
   - Graph-token pairing verification
   - Schema version compatibility

2. **Phase 2: Statistical Analysis**
   - Feature distribution analysis
   - Label balance assessment
   - Outlier detection

3. **Phase 3: Semantic Validation**
   - Logical consistency checks
   - Data quality scoring
   - Anomaly detection

## Output

The validator generates a comprehensive report (`validation_report.json`) with:
- Overall dataset health score
- Per-metric validation results
- Recommendations for data improvements
- Detected issues with severity levels

## Integration

This module is used during:
- Initial dataset preparation
- After data pipeline changes
- Before training new model versions
- Periodic data quality checks
