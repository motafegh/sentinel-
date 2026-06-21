# ml/data — Training Data and Benchmarks

Training data, evaluation benchmarks, and derived artifacts. **Not committed to git** (`.gitignore`).

---

## Directory Structure

```
ml/data/
|-- graphs/                        .pt graph files (v9, 12-dim features)
|-- tokens_windowed/               .pt token files ([4,512] windows, stride=256)
|-- processed/                     CSV label files (multilabel_index.csv)
|-- splits/
|   |-- v3/                        Active splits (18,596/1,983/1,914)
|-- augmented/                     DoS and CEI augmented .sol files
|-- smartbugs-curated/             SmartBugs Curated dataset
|-- smartbugs-wild/                SmartBugs Wild dataset
|-- SolidiFI/                      SolidiFI benchmark
|-- SolidiFI-processed/            Processed SolidiFI
|-- slither_results/               Slither analysis results
|-- reports/                       Data quality reports
|-- archive/                       Archived/legacy data files
|-- cached_dataset_v9.pkl          Paired (graph, tokens) cache (legacy)
|-- drift_baseline_run12.json      Run 12 drift baseline (4 stats x 500 samples)
|-- drift_baseline.json            Legacy drift baseline
|-- smartbugs_wild_eval_state.json  Wild evaluation progress tracker
|-- validation_report.json         Data validation results
|-- warmup_run12.jsonl             Warmup inference data for drift baseline
```

---

## Active Data (v3)

**Export:** `data_module/data/exports/sentinel-v3-smartbugs-2026-06-13/` (artifact_hash `5cc5...`)
- 22,493 contracts, 5 shards, 0% leakage
- Splits: 18,596 train / 1,983 val / 1,914 test

**Splits:** `data_module/data/splits/v3/`

**Benchmarks:** `data_module/benchmarks/` — 66 contracts, 5-tier OOD design (v0.1)

---

## Drift Baseline

`drift_baseline_run12.json` — real synthetic warmup baseline for KS drift detection:
- Stats: `num_nodes`, `num_edges`, `confirmed_count`, `suspicious_count`
- 500 samples from synthetic warmup
- Replace with real production warmup traffic when available

---

## Legacy Data

- `cached_dataset_v9.pkl` — legacy paired cache (v2 export era)
- `drift_baseline.json` — older baseline (may be placeholder)
- `smartbugs-results-master/` — raw SmartBugs results archive
- `SolidiFI-benchmark/` — raw SolidiFI benchmark archive
