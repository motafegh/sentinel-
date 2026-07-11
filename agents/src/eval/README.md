# Eval — Evaluation Framework

Measures SENTINEL's audit quality against ground-truth labels. Provides Fbeta(β=2)
scoring, per-gate assertions, Bayesian reliability fitting, and a reproducible benchmark
runner. All eval runs are saved to `eval/runs/` with a timestamped directory.

## Files

| File | Purpose |
|------|---------|
| `pipeline_metrics.py` | `PipelineMetrics` — Fbeta(β=2), macro/class confusion matrix |
| `gates.py` | 9 gate assertions that must all pass for a run to count as "PASS" |
| `benchmarks.py` | Benchmark definitions: contract → expected vulnerability verdict |
| `run_benchmark.py` | CLI benchmark runner — runs pipeline on corpus, scores output |
| `reliability_matrix.py` | P3: builds TP/FP/FN/TN per tool per class from eval run data |
| `reliability_fit.py` | P3: Bayesian shrinkage fitter — produces `reliability_v3.yaml` |
| `regression.py` | Regression harness — asserts no gate regresses vs. a saved baseline |

## Metrics

### Why Fbeta(β=2)

Security auditing is FN-critical: a missed vulnerability (false negative) is far worse
than a false alarm (false positive). β=2 weights recall twice as heavily as precision.

```
Fbeta(β=2) = (1 + β²) × precision × recall / (β² × precision + recall)
           = 5 × precision × recall / (4 × precision + recall)
```

`PipelineMetrics` computes both macro_F1 (β=1) and macro_Fbeta (β=2) across all 10
vulnerability classes.

### Baseline History

| Phase | macro_F1 | macro_Fbeta | Notes |
|-------|----------|-------------|-------|
| P0 honest baseline | 0.1958 | 0.2515 | First real eval, no calibration |
| P2 calibrated | 0.1998 | 0.2246 | fuse() sole verdict producer |
| P3 L3 reliability | 0.3008 | 0.3821 | +0.10 F1, +0.16 Fbeta vs P2 |

The P3 jump came from replacing hand-set L1 reliability weights with L3 Bayesian-fitted
weights computed from actual tool TP/FP rates on the corpus.

## `gates.py` — Gate Assertions

Every benchmark run must pass all 9 gates to be "PASS". A failing gate blocks
promotion. Current gates:

| Gate | Condition | Rationale |
|------|-----------|-----------|
| `macro_f1_vs_baseline` | macro_F1 ≥ previous baseline − 0.01 | No F1 regression |
| `macro_fbeta_vs_baseline` | macro_Fbeta ≥ previous baseline − 0.01 | No recall regression |
| `reentrancy_recall` | Reentrancy recall ≥ 0.70 | Highest-value class |
| `no_all_safe` | ≥1 contract flagged per run | System is not trivially predicting SAFE |
| `tool_ran_rate` | tool_status `ran=True` rate ≥ 0.80 | Tools actually executing |
| `no_silent_failures` | No `ran=False` with empty static_findings | Rule 5C compliance |
| `audit_report_complete` | All required final_report keys present | Schema compliance |
| `model_hash_present` | `model_provenance.model_hash` present | P5 provenance |
| `injection_detections_serialized` | `security.injection_detections` key present | P4 compliance |

## `reliability_matrix.py` — P3 Tool Reliability

Reads saved eval run data (JSON files in `eval/runs/`) and builds a per-tool, per-class
confusion matrix by comparing `tool_status` against ground-truth labels. Contracts
where a tool did not run (`ran=False`) are excluded from that tool's TP/FP/FN/TN counts
— per Rule 5C, an absent run cannot be treated as "found nothing."

```python
from src.eval.reliability_matrix import build_reliability_matrix

matrix = build_reliability_matrix("eval/runs/20260626T123145Z_p3_rule5c_v3/")
# → {tool: {class: {TP, FP, FN, TN}}}
```

## `reliability_fit.py` — Bayesian Shrinkage

Takes the reliability matrix and fits per-tool, per-class reliability weights using
Bayesian shrinkage (α=5 pseudo-observations toward the prior). Small sample sizes are
pulled toward the class-level mean; large sample sizes converge to the empirical rate.

A ±5pp change in a fitted weight triggers the drop-gate: the weight is not updated
unless the delta is significant.

Output is `configs/reliability_v3.yaml` with `schema_version=1`.

```bash
cd agents
poetry run python scripts/build_reliability_matrix.py \
    --run-dir eval/runs/20260626T123145Z_p3_rule5c_v3 \
    --output configs/reliability_v3.yaml
```

## `run_benchmark.py` — CLI Runner

```bash
cd agents
# Run full benchmark (all contracts in corpus, --no-llm mode)
poetry run python src/eval/run_benchmark.py --no-llm --output eval/runs/

# Run with a specific contract set
poetry run python src/eval/run_benchmark.py --contracts test_contracts/ --no-llm
```

Produces a timestamped directory under `eval/runs/` containing:
- `eval_report.md` — human-readable summary with gate pass/fail status
- `results.json` — per-contract verdicts + tool_status + final_report
- `metrics.json` — macro_F1, macro_Fbeta, per-class precision/recall/F1

## Eval Run Layout

```
eval/runs/
  20260626T123145Z_p3_rule5c_v3/
    eval_report.md       human-readable summary
    results.json         per-contract full output
    metrics.json         aggregate scores
  20260624T133420Z_p0_honest_baseline/
    ...                  baseline for comparison
```

## Running Tests

```bash
cd agents
poetry run pytest tests/test_eval_fbeta.py tests/test_eval_framework.py \
    tests/test_run_benchmark.py -v
```
