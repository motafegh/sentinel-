# _legacy_data_pipeline

Scripts in this directory were the v1 data pipeline for SENTINEL. They operated on
`ml/data/` (MD5-keyed graphs + tokens) and produced the v1 cache used in Runs 1-9.

They were superseded by the v2 data module (`data_module/sentinel_data/`) as part of
the Stage 7B seam swap (2026-06-12). All new pipeline work goes through:

    sentinel-data ingest → preprocess → represent → label → verify → split → export

**Do not use these scripts for new training runs.** They are kept for reference only.

| Script | What it did | Replacement |
|---|---|---|
| `reextract_graphs.py` | Re-ran graph extraction on all `ml/data/preprocessed/` contracts | `sentinel-data represent` |
| `retokenize_windowed.py` | Re-ran windowed tokenization for GraphCodeBERT | `sentinel-data represent` |
| `build_multilabel_index.py` | Built `ml/data/processed/multilabel_index.csv` | `sentinel-data label` + `sentinel-data export` |
| `create_splits.py` | Created `ml/data/splits/v10_deduped/` train/val/test split files | `sentinel-data split` |
| `create_cache.py` | Pickled all (graph, token) pairs into `ml/data/cached_dataset_v10.pkl` | `sentinel-data export` (shard format) |
| `validate_graph_dataset.py` | Validated all `.pt` graph files for shape/dtype correctness | `sentinel-data verify` |
| `archive_v8_data.py` | Moved v8 graphs to `ml/data/archive/` | historical, no replacement needed |
