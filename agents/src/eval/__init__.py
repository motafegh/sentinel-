"""
eval — Pipeline evaluation framework (WS6a Phase C.2, 2026-06-22).

Reusable library that lifts the comparator logic from
`scripts/eval_benchmark.py` into a proper module. Other callers
(notebooks, the C.1 FastAPI gateway, CI jobs) can import the
`PipelineMetrics` + `Benchmark` + `RegressionBaseline` classes
instead of re-implementing or shelling out to the script.

Modules:
  - pipeline_metrics: ClassMetrics + ContractMetrics + PipelineMetrics.
                      P/R/F1, macro/micro, per-class aggregation.
  - benchmarks: Benchmark class — loads contracts from a corpus dir
                (json sidecar + // expect: header fallback), pairs them
                with reports from a runs dir, returns ContractMetrics rows.
  - regression:    load/save a stored baseline + diff against current run.
                   Used by the comparator's --baseline flag and the
                   "regression" gate.
  - gates:         the 9 workstream gate assertions (WS1a/b/c/d/e, WS2,
                   WS3, D4, macro_f1). Reused by scripts/eval_benchmark.py.

Dataset decision (Ali, 2026-06-22): use the existing 88-contract WS0 corpus
for now. The classes here are written against the existing format so
swapping in a larger corpus later is a data change, not a code change.
"""

from src.eval.pipeline_metrics import (
    ClassMetrics,
    ContractMetrics,
    PipelineMetrics,
    DEFAULT_POSITIVE_VERDICTS,
    VALID_VERDICTS,
    BORDERLINE_BAND,
    metrics_from_contracts,
)
from src.eval.benchmarks import Benchmark
from src.eval.regression import (
    RegressionBaseline,
    RegressionResult,
)

__all__ = [
    # pipeline_metrics
    "ClassMetrics",
    "ContractMetrics",
    "PipelineMetrics",
    "DEFAULT_POSITIVE_VERDICTS",
    "VALID_VERDICTS",
    "BORDERLINE_BAND",
    "metrics_from_contracts",
    # benchmarks
    "Benchmark",
    # regression
    "RegressionBaseline",
    "RegressionResult",
]
