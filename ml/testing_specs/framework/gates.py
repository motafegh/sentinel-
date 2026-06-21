"""
framework/gates.py — Generic gate definitions.

A "gate" is any check that produces PASS / FAIL / WARN / UNVERIFIED + a
written result. Gates compose into larger checks (e.g., a "promotion gate"
runs multiple individual gates).

The original spec files (A, B, C, etc.) in ml/testing_specs/ are SENTINEL-
specific. The gates here are project-agnostic — any project can subclass
Gate and implement custom gates.

Example:
    from ml.testing_specs.framework.gates import Gate, GateResult, GateStatus

    class FileExistsGate(Gate):
        def __init__(self, path):
            self.path = path

        def run(self):
            if self.path.exists():
                return GateResult(
                    gate_name=f"file_exists:{self.path}",
                    status=GateStatus.PASS,
                    message=f"{self.path} exists",
                )
            return GateResult(
                gate_name=f"file_exists:{self.path}",
                status=GateStatus.FAIL,
                message=f"{self.path} missing",
            )

    gate = FileExistsGate(Path("ml/checkpoints/run.pt"))
    result = gate.run()
    print(result.status, result.message)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Optional


class GateStatus(str, Enum):
    """Standard gate outcomes."""

    PASS = "PASS"
    FAIL = "FAIL"
    WARN = "WARN"
    UNVERIFIED = "UNVERIFIED"


@dataclass
class GateResult:
    """Result of running one gate.

    Fields:
        gate_name: Unique name for the gate (e.g., "behavioral_probes:ext_owner_only")
        status: One of PASS / FAIL / WARN / UNVERIFIED
        message: Human-readable description
        value: Optional machine-readable result (for programmatic use)
        duration_s: How long the gate took
        metadata: Arbitrary additional context
    """

    gate_name: str
    status: GateStatus
    message: str
    value: Any = None
    duration_s: float = 0.0
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def passed(self) -> bool:
        return self.status == GateStatus.PASS

    @property
    def failed(self) -> bool:
        return self.status == GateStatus.FAIL


class Gate:
    """Base class for all gates.

    Subclass and implement `run()`. The base class provides:
    - `run_with_timing()` — wraps `run()` with duration measurement
    - `to_json()` — serialise result to JSON
    - Standard `__str__` for console output
    """

    def __init__(self, name: str):
        self.name = name

    def run(self) -> GateResult:
        """Execute the gate. Subclasses MUST override this."""
        raise NotImplementedError("Subclasses must implement run()")

    def run_with_timing(self) -> GateResult:
        """Run the gate, measure duration, return result."""
        t0 = time.time()
        try:
            r = self.run()
        except Exception as e:
            return GateResult(
                gate_name=self.name,
                status=GateStatus.FAIL,
                message=f"Gate raised exception: {type(e).__name__}: {e}",
                duration_s=time.time() - t0,
            )
        r.duration_s = time.time() - t0
        r.gate_name = self.name
        return r

    def to_json(self, indent: int = 2) -> str:
        """Serialise the last result to JSON."""
        return json.dumps(self._last_result.to_dict(), indent=indent, default=str)

    def __str__(self) -> str:
        if not hasattr(self, "_last_result"):
            return f"<{type(self).__name__} name={self.name}>"
        r = self._last_result
        icon = {"PASS": "✓", "FAIL": "✗", "WARN": "⚠", "UNVERIFIED": "?"}[r.status.value]
        return f"[{icon}] {r.gate_name}: {r.message}"


class CompositeGate(Gate):
    """A gate that runs multiple sub-gates and aggregates results.

    Aggregation rules:
    - Any FAIL → overall FAIL
    - Any UNVERIFIED (and no FAIL) → overall UNVERIFIED
    - All WARN (no FAIL) → overall WARN
    - All PASS → overall PASS
    """

    def __init__(self, name: str, sub_gates: list[Gate], require_all: bool = True):
        super().__init__(name)
        self.sub_gates = sub_gates
        self.require_all = require_all

    def run(self) -> GateResult:
        results: list[GateResult] = []
        for g in self.sub_gates:
            r = g.run_with_timing()
            results.append(r)

        # Aggregate
        statuses = [r.status for r in results]
        n_pass = sum(1 for s in statuses if s == GateStatus.PASS)
        n_fail = sum(1 for s in statuses if s == GateStatus.FAIL)
        n_warn = sum(1 for s in statuses if s == GateStatus.WARN)
        n_unverified = sum(1 for s in statuses if s == GateStatus.UNVERIFIED)

        if n_fail > 0:
            overall = GateStatus.FAIL
        elif n_unverified > 0:
            overall = GateStatus.UNVERIFIED
        elif n_warn > 0 and self.require_all:
            overall = GateStatus.WARN
        else:
            overall = GateStatus.PASS

        # Compose message
        failed = [r for r in results if r.status == GateStatus.FAIL]
        warned = [r for r in results if r.status == GateStatus.WARN]
        msg = f"{n_pass} PASS, {n_fail} FAIL, {n_warn} WARN, {n_unverified} UNVERIFIED"
        if failed:
            msg += f" (failures: {[r.gate_name for r in failed[:5]]})"

        return GateResult(
            gate_name=self.name,
            status=overall,
            message=msg,
            value={"results": [r.to_dict() for r in results]},
            metadata={"counts": {
                "passed": n_pass, "failed": n_fail, "warned": n_warn, "unverified": n_unverified,
            }},
        )


# ---------------------------------------------------------------------------
# Built-in gates (project-agnostic)
# ---------------------------------------------------------------------------


class FileExistsGate(Gate):
    """PASS if file exists, FAIL otherwise."""

    def __init__(self, name: str, path: Path):
        super().__init__(name)
        self.path = path

    def run(self) -> GateResult:
        if self.path.exists():
            size = self.path.stat().st_size
            return GateResult(
                gate_name=self.name,
                status=GateStatus.PASS,
                message=f"{self.path} exists ({size} bytes)",
                value={"path": str(self.path), "size": size},
            )
        return GateResult(
            gate_name=self.name,
            status=GateStatus.FAIL,
            message=f"{self.path} not found",
            value={"path": str(self.path)},
        )


class JSONFileGate(Gate):
    """Load a JSON file and check a key against expected value."""

    def __init__(self, name: str, path: Path, key: str, expected: Any):
        super().__init__(name)
        self.path = path
        self.key = key
        self.expected = expected

    def run(self) -> GateResult:
        if not self.path.exists():
            return GateResult(
                gate_name=self.name,
                status=GateStatus.FAIL,
                message=f"{self.path} not found",
            )
        try:
            with self.path.open() as f:
                data = json.load(f)
        except Exception as e:
            return GateResult(
                gate_name=self.name,
                status=GateStatus.FAIL,
                message=f"Could not parse {self.path}: {e}",
            )
        # Get the value at the key (supports nested keys with dots)
        keys = self.key.split(".")
        v = data
        for k in keys:
            if isinstance(v, dict) and k in v:
                v = v[k]
            else:
                return GateResult(
                    gate_name=self.name,
                    status=GateStatus.FAIL,
                    message=f"Key {self.key!r} not found in {self.path}",
                    value={"data": data},
                )
        if v == self.expected:
            return GateResult(
                gate_name=self.name,
                status=GateStatus.PASS,
                message=f"{self.key} == {self.expected!r}",
                value={"actual": v, "expected": self.expected},
            )
        return GateResult(
            gate_name=self.name,
            status=GateStatus.FAIL,
            message=f"{self.key} == {v!r}, expected {self.expected!r}",
            value={"actual": v, "expected": self.expected},
        )


class F1Gate(Gate):
    """F1 gate: PASS if val_f1 > threshold, FAIL if ≤ threshold, UNVERIFIED if can't compute."""

    def __init__(self, name: str, current_f1: float, prior_f1: Optional[float] = None,
                 min_improvement: float = 0.0):
        super().__init__(name)
        self.current_f1 = current_f1
        self.prior_f1 = prior_f1
        self.min_improvement = min_improvement

    def run(self) -> GateResult:
        if self.prior_f1 is None:
            return GateResult(
                gate_name=self.name,
                status=GateStatus.UNVERIFIED,
                message=f"No prior F1 to compare against (current={self.current_f1:.4f})",
                value={"current": self.current_f1},
            )
        if self.current_f1 <= self.prior_f1:
            return GateResult(
                gate_name=self.name,
                status=GateStatus.FAIL,
                message=f"current_f1={self.current_f1:.4f} ≤ prior_f1={self.prior_f1:.4f}",
                value={"current": self.current_f1, "prior": self.prior_f1},
            )
        delta = self.current_f1 - self.prior_f1
        if delta < self.min_improvement:
            return GateResult(
                gate_name=self.name,
                status=GateStatus.WARN,
                message=f"current_f1={self.current_f1:.4f} > prior_f1={self.prior_f1:.4f} but delta {delta:.4f} < min_improvement {self.min_improvement:.4f}",
                value={"current": self.current_f1, "prior": self.prior_f1, "delta": delta},
            )
        return GateResult(
            gate_name=self.name,
            status=GateStatus.PASS,
            message=f"current_f1={self.current_f1:.4f} > prior_f1={self.prior_f1:.4f} (delta={delta:+.4f})",
            value={"current": self.current_f1, "prior": self.prior_f1, "delta": delta},
        )


class JSONKeyGate(Gate):
    """A more flexible JSONFileGate that supports comparison operators.

    Args:
        key: Dot-separated key path in the JSON.
        op: One of "==", "!=", ">", ">=", "<", "<=".
        expected: Value to compare against.
    """

    OPS = {
        "==": lambda a, b: a == b,
        "!=": lambda a, b: a != b,
        ">":  lambda a, b: a > b,
        ">=": lambda a, b: a >= b,
        "<":  lambda a, b: a < b,
        "<=": lambda a, b: a <= b,
    }

    def __init__(self, name: str, path: Path, key: str, op: str, expected: Any):
        super().__init__(name)
        self.path = path
        self.key = key
        self.op = op
        self.expected = expected

    def run(self) -> GateResult:
        if self.op not in self.OPS:
            return GateResult(
                gate_name=self.name,
                status=GateStatus.FAIL,
                message=f"Unknown op {self.op!r}; must be one of {list(self.OPS)}",
            )
        if not self.path.exists():
            return GateResult(
                gate_name=self.name,
                status=GateStatus.FAIL,
                message=f"{self.path} not found",
            )
        try:
            with self.path.open() as f:
                data = json.load(f)
        except Exception as e:
            return GateResult(
                gate_name=self.name,
                status=GateStatus.FAIL,
                message=f"Could not parse {self.path}: {e}",
            )
        keys = self.key.split(".")
        v = data
        for k in keys:
            if isinstance(v, dict) and k in v:
                v = v[k]
            else:
                return GateResult(
                    gate_name=self.name,
                    status=GateStatus.FAIL,
                    message=f"Key {self.key!r} not found in {self.path}",
                )
        fn = self.OPS[self.op]
        if fn(v, self.expected):
            return GateResult(
                gate_name=self.name,
                status=GateStatus.PASS,
                message=f"{self.key} {self.op} {self.expected!r} (actual={v!r})",
                value={"actual": v, "expected": self.expected, "op": self.op},
            )
        return GateResult(
            gate_name=self.name,
            status=GateStatus.FAIL,
            message=f"{self.key} {self.op} {self.expected!r} FAILED (actual={v!r})",
            value={"actual": v, "expected": self.expected, "op": self.op},
        )


class ReproducibilityGate(Gate):
    """Gate that checks the output of `auto_reproducibility_check.py`.

    The JSON output has shape:
        {
          "checkpoint": "...",
          "model_state_hash": "...",
          "model_file_hash": "...",
          "git_commit": "...",
          "poetry_lock_hash": "...",
          "result": "PASS" | "FAIL"
        }
    """

    def __init__(self, name: str, path: Path, expected_result: str = "PASS"):
        super().__init__(name)
        self.path = path
        self.expected_result = expected_result

    def run(self) -> GateResult:
        if not self.path.exists():
            return GateResult(
                gate_name=self.name,
                status=GateStatus.UNVERIFIED,
                message=f"{self.path} not found — reproducibility not checked",
            )
        try:
            with self.path.open() as f:
                data = json.load(f)
        except Exception as e:
            return GateResult(
                gate_name=self.name,
                status=GateStatus.FAIL,
                message=f"Could not parse {self.path}: {e}",
            )
        result = data.get("result")
        if result == self.expected_result:
            return GateResult(
                gate_name=self.name,
                status=GateStatus.PASS,
                message=f"Reproducibility result is {result!r}",
                value=data,
            )
        return GateResult(
            gate_name=self.name,
            status=GateStatus.FAIL,
            message=f"Reproducibility result is {result!r}, expected {self.expected_result!r}",
            value=data,
        )


class StaleCheckpointsGate(Gate):
    """Gate that checks the output of `check_stale_checkpoints.py`.

    The JSON output has shape:
        {
          "summary": {
            "total": int,
            "stale_or_orphan": int,
            "all_clean": bool
          }
        }
    """

    def __init__(self, name: str, path: Path, max_stale: int = 0):
        super().__init__(name)
        self.path = path
        self.max_stale = max_stale

    def run(self) -> GateResult:
        if not self.path.exists():
            return GateResult(
                gate_name=self.name,
                status=GateStatus.UNVERIFIED,
                message=f"{self.path} not found — stale checkpoints not checked",
            )
        try:
            with self.path.open() as f:
                data = json.load(f)
        except Exception as e:
            return GateResult(
                gate_name=self.name,
                status=GateStatus.FAIL,
                message=f"Could not parse {self.path}: {e}",
            )
        summary = data.get("summary", {})
        n_stale = summary.get("stale_or_orphan", 0)
        if n_stale <= self.max_stale:
            return GateResult(
                gate_name=self.name,
                status=GateStatus.PASS,
                message=f"{n_stale} stale/orphan checkpoints (≤ max_stale={self.max_stale})",
                value=summary,
            )
        return GateResult(
            gate_name=self.name,
            status=GateStatus.FAIL,
            message=f"{n_stale} stale/orphan checkpoints (> max_stale={self.max_stale})",
            value=summary,
        )
