"""
pipeline_metrics.py — Core P/R/F1 + per-class + macro/micro aggregation.

Lifted from scripts/eval_benchmark.py (WS0 comparator) so it's importable
from anywhere — notebooks, the C.1 gateway, CI, or the script itself.

The semantics are unchanged from the WS0 version: a "positive" verdict is
in DEFAULT_POSITIVE_VERDICTS (CONFIRMED + LIKELY). A contract is a true
positive if the predicted class set intersects the labelled class set;
false positive if predicted but not labelled; false negative if labelled
but not predicted; true negative otherwise.

INCONCLUSIVE is deliberately NOT positive — it means "we couldn't check",
which is a different failure mode than a false positive. The WS1.5
reconciliation explicitly keeps it distinct from SAFE.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Any


def __getattr__(name: str):
    from src.config import get_config as _get_cfg

    _map = {
        "DEFAULT_POSITIVE_VERDICTS": lambda c: frozenset(c.eval.positive_verdicts),
        "BORDERLINE_BAND":           lambda c: (c.eval.borderline_band_low, c.eval.borderline_band_high),
    }
    if name in _map:
        return _map[name](_get_cfg())
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# All valid verdict strings (6 of them per WS1.5).
VALID_VERDICTS: frozenset[str] = frozenset({
    "CONFIRMED", "LIKELY", "DISPUTED", "WATCH", "SAFE", "INCONCLUSIVE",
})


# ---------------------------------------------------------------------------
# Per-class metrics
# ---------------------------------------------------------------------------

@dataclass
class ClassMetrics:
    """
    Per-vulnerability-class confusion matrix + derived P/R/F1.

    Support = number of contracts that have THIS class in their ground
    truth labels (i.e. positive class). The 2x2 confusion matrix is
    computed over ALL contracts in the benchmark, with "positive" being
    "this class is in the predicted positive set".
    """
    cls: str
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0
    precision: float = 0.0
    recall:    float = 0.0
    f1:         float = 0.0
    fbeta:      float = 0.0
    support:   int = 0   # number of contracts with this class in labels

    def compute(self, beta: float | None = None) -> None:
        """
        Derive precision/recall/F1/F-beta from the raw counts in place.

        Args:
            beta: F-beta parameter (default from config.eval.fbeta_beta).
        """
        if beta is None:
            from src.config import get_config as _get_cfg
            beta = _get_cfg().eval.fbeta_beta
        self.precision = (
            self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else float("nan")
        )
        self.recall = (
            self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else float("nan")
        )
        p = self.precision
        r = self.recall
        if not math.isnan(p) and not math.isnan(r) and (p + r) > 0:
            self.f1 = 2 * p * r / (p + r)
            b2 = beta * beta
            denom = b2 * p + r
            self.fbeta = (1 + b2) * p * r / denom if denom > 0 else 0.0
        else:
            self.f1 = 0.0
            self.fbeta = 0.0

    def as_dict(self) -> dict[str, Any]:
        """JSON-serialisable view (NaN → 0.0 since JSON has no NaN)."""
        return {
            "cls":       self.cls,
            "tp":        self.tp,
            "fp":        self.fp,
            "fn":        self.fn,
            "tn":        self.tn,
            "precision": 0.0 if math.isnan(self.precision) else self.precision,
            "recall":    0.0 if math.isnan(self.recall)    else self.recall,
            "f1":        self.f1,
            "fbeta":     self.fbeta,
            "support":   self.support,
        }


# ---------------------------------------------------------------------------
# Per-contract metrics
# ---------------------------------------------------------------------------

@dataclass
class ContractMetrics:
    """
    One contract's evaluation row.

    The verdict set comes from the report JSON (`final_report.verdicts`).
    The label set comes from the benchmark sidecar / `// expect:` header
    in the .sol file itself.
    """
    stem: str
    report_path: str
    labels: list[str]                 # ground truth classes
    ground_truth: str                 # "vulnerable" | "safe"
    verdicts: dict[str, str]          # class -> verdict
    probabilities: dict[str, float]   # class -> ML prob
    overall_verdict: str | None = None
    path_taken: str = "unknown"       # "deep" | "fast"
    error: str | None = None

    # Derived (filled in by compute_per_contract)
    predicted_positive_classes: list[str] = field(default_factory=list)
    true_positive_classes:      list[str] = field(default_factory=list)
    false_positive_classes:     list[str] = field(default_factory=list)
    false_negative_classes:     list[str] = field(default_factory=list)
    contract_correct: bool = False  # loose: safe→no flag OR vuln→≥1 correct flag
    contract_exact:   bool = False  # strict: predicted set == label set


# ---------------------------------------------------------------------------
# Aggregate pipeline metrics
# ---------------------------------------------------------------------------

class PipelineMetrics:
    """
    The full evaluation result for one run.

    Built from a list of `ContractMetrics` (one per contract in the
    benchmark). Exposes:
      - per-class P/R/F1 (ClassMetrics dict keyed by class name)
      - macro-F1 (mean of per-class F1, NaN-aware)
      - micro-F1 (computed from the sum TP/FP/FN)
      - contract-level accuracy (loose + strict)
      - JSON / dict export for baselines and reports
    """

    def __init__(
        self,
        contracts: list[ContractMetrics],
        positive_verdicts: frozenset[str] | set[str] | None = None,
    ):
        if positive_verdicts is None:
            from src.config import get_config as _get_cfg
            positive_verdicts = frozenset(_get_cfg().eval.positive_verdicts)
        self.contracts = list(contracts)
        self.positive_verdicts = frozenset(positive_verdicts)

        # Will be populated by `compute()`.
        self.class_metrics: dict[str, ClassMetrics] = {}
        self.macro_f1:      float = 0.0
        self.macro_fbeta:   float = 0.0
        self.micro_f1:      float = 0.0
        self.contract_accuracy_loose: float = 0.0
        self.contract_accuracy_exact: float = 0.0

    # ----- Per-contract derived fields -----

    @staticmethod
    def positive_classes(verdicts: dict[str, str], positive_set: set[str]) -> list[str]:
        """Return the classes whose verdict is in the positive set."""
        return [c for c, v in verdicts.items() if v in positive_set]

    def derive_per_contract(self) -> None:
        """Fill in TP/FP/FN + loose/strict accuracy on each contract."""
        for row in self.contracts:
            predicted = self.positive_classes(row.verdicts, self.positive_verdicts)
            label_set = set(row.labels)

            row.predicted_positive_classes = predicted
            row.true_positive_classes      = [c for c in predicted if c in label_set]
            row.false_positive_classes     = [c for c in predicted if c not in label_set]
            row.false_negative_classes     = [c for c in label_set if c not in predicted]

            # Loose: safe→no flag OR vuln→≥1 correct flag.
            if row.ground_truth == "safe":
                row.contract_correct = not predicted
            else:
                row.contract_correct = bool(row.true_positive_classes)

            # Strict: predicted positive set == label set.
            row.contract_exact = set(predicted) == label_set

    # ----- Per-class aggregation -----

    def compute_class_metrics(self) -> None:
        """
        Aggregate per-contract predictions into per-class ClassMetrics.
        Must be called AFTER derive_per_contract().
        """
        # Index predicted-positive classes per contract for fast TP/FP lookup.
        per_contract_predicted: dict[str, set[str]] = {
            row.stem: set(row.predicted_positive_classes) for row in self.contracts
        }
        per_contract_labels: dict[str, set[str]] = {
            row.stem: set(row.labels) for row in self.contracts
        }

        # Collect every class that appears anywhere.
        all_classes: set[str] = set()
        for row in self.contracts:
            all_classes.update(row.labels)
            all_classes.update(row.predicted_positive_classes)

        self.class_metrics = {cls: ClassMetrics(cls=cls) for cls in sorted(all_classes)}

        for cls, m in self.class_metrics.items():
            tp = fp = fn = tn = 0
            for row in self.contracts:
                pred = row.stem in per_contract_predicted and cls in per_contract_predicted[row.stem]
                lab  = row.stem in per_contract_labels    and cls in per_contract_labels[row.stem]
                if pred and lab:
                    tp += 1
                elif pred and not lab:
                    fp += 1
                elif not pred and lab:
                    fn += 1
                else:
                    tn += 1
            m.tp = tp
            m.fp = fp
            m.fn = fn
            m.tn = tn
            m.support = sum(1 for row in self.contracts if cls in per_contract_labels.get(row.stem, set()))
            m.compute()

    # ----- Macro / micro / contract-level -----

    def compute_aggregates(self) -> None:
        """Compute macro-F1, micro-F1, and contract-level accuracy."""
        if not self.class_metrics:
            self.compute_class_metrics()

        # Macro-F1 and macro-Fbeta: NaN-aware mean over classes with support>0.
        per_class_f1s = [m.f1 for m in self.class_metrics.values() if m.support > 0]
        self.macro_f1 = sum(per_class_f1s) / len(per_class_f1s) if per_class_f1s else 0.0
        per_class_fbetas = [m.fbeta for m in self.class_metrics.values() if m.support > 0]
        self.macro_fbeta = sum(per_class_fbetas) / len(per_class_fbetas) if per_class_fbetas else 0.0

        # Micro-F1: from the sum TP/FP/FN.
        total_tp = sum(m.tp for m in self.class_metrics.values())
        total_fp = sum(m.fp for m in self.class_metrics.values())
        total_fn = sum(m.fn for m in self.class_metrics.values())
        micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        self.micro_f1 = (
            2 * micro_p * micro_r / (micro_p + micro_r)
            if (micro_p + micro_r) > 0 else 0.0
        )

        # Contract-level accuracy.
        n = len(self.contracts)
        if n > 0:
            self.contract_accuracy_loose = sum(1 for r in self.contracts if r.contract_correct) / n
            self.contract_accuracy_exact = sum(1 for r in self.contracts if r.contract_exact) / n

    # ----- Public entry point -----

    def compute(self) -> None:
        """Compute everything. Call once after construction + adding contracts."""
        self.derive_per_contract()
        self.compute_class_metrics()
        self.compute_aggregates()

    # ----- Serialisation -----

    def as_dict(self) -> dict[str, Any]:
        """JSON-serialisable view (for baselines + reports)."""
        return {
            "contract_count":           len(self.contracts),
            "positive_verdicts":        sorted(self.positive_verdicts),
            "macro_f1":                 self.macro_f1,
            "macro_fbeta":              self.macro_fbeta,
            "micro_f1":                 self.micro_f1,
            "contract_accuracy_loose":  self.contract_accuracy_loose,
            "contract_accuracy_exact":  self.contract_accuracy_exact,
            "per_class": {
                cls: m.as_dict() for cls, m in sorted(self.class_metrics.items())
            },
        }


# ---------------------------------------------------------------------------
# Convenience constructors
# ---------------------------------------------------------------------------

def metrics_from_contracts(
    contracts: list[ContractMetrics],
    positive_verdicts: frozenset[str] | set[str] | None = None,
) -> PipelineMetrics:
    """
    Build + compute a PipelineMetrics in one call. The most common entry
    point for callers that already have ContractMetrics rows in hand.
    """
    pm = PipelineMetrics(contracts, positive_verdicts=positive_verdicts)
    pm.compute()
    return pm
