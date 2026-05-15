# ml/_archive — Legacy and Superseded Files

This directory holds files that were part of earlier development phases.
They are preserved here for reference and reproducibility but are NOT
part of the active pipeline.

Use `git log -- ml/_archive/<file>` to see the full history of any file.

## Subdirectories

### src_data/
Old data-loading modules that predate the current pipeline:
- `graphs/ast_extractor.py` — 17-dim AST extractor (superseded by ml/data_extraction/ast_extractor.py)
- `graphs/graph_builder.py` — early graph construction script
- `bccc_dataset.py`        — BCCC CSV loader (reference for Track 3 multilabel work)
- `solidifi_dataset.py`    — SolidiFI dataset (not used in production pipeline)
- `validate_*.py`          — one-time dataset validation scripts (completed)

### src_tools/
Old Slither wrapper scripts (superseded by ml/src/inference/preprocess.py):
- `slither_wrapper.py`        — 62K synchronous Slither wrapper
- `slither_wrapper_turbo.py`  — async/parallel variant

### src_validation/
One-time statistical validation scripts run during development:
- All results were confirmed; scripts are no longer needed in active src/

### scripts/
Legacy runnable scripts:
- `test_*.py`                     — early unit tests for individual components
- `comprehensive_data_validation.py` — full-dataset validation (completed once)
- `fix_labels_from_csv.py`        — one-time label correction utility (completed)
- `analyze_token_stats.py`        — token distribution analysis (completed)
- `run_full_dataset_overnight.py` — root-level legacy overnight script
- `validation_report.json`        — static output from validation run

## What to Do If You Need One of These Files
Either import directly using the full path, or restore it with:
  git restore --source=v1.0-binary-baseline ml/src/data/...
