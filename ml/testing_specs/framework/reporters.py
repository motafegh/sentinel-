"""
framework/reporters.py — Output formatters for gate results.

The default CLI prints to stderr. This module provides:
  - JSON reporter: machine-readable, suitable for CI
  - Markdown reporter: human-readable, suitable for PR comments
  - HTML reporter: for dashboards (future)
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .gates import GateResult, GateStatus


def report_json(results: list[GateResult], path: Path) -> None:
    """Write results as JSON to `path`."""
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total": len(results),
        "passed": sum(1 for r in results if r.status == GateStatus.PASS),
        "failed": sum(1 for r in results if r.status == GateStatus.FAIL),
        "warned": sum(1 for r in results if r.status == GateStatus.WARN),
        "unverified": sum(1 for r in results if r.status == GateStatus.UNVERIFIED),
        "results": [r.to_dict() for r in results],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))


def report_markdown(results: list[GateResult], path: Path) -> None:
    """Write results as Markdown to `path`. Suitable for PR comments."""
    lines = [
        "# Testing Suite Gate Report",
        "",
        f"**Generated:** {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Summary",
        "",
    ]
    n_pass = sum(1 for r in results if r.status == GateStatus.PASS)
    n_fail = sum(1 for r in results if r.status == GateStatus.FAIL)
    n_warn = sum(1 for r in results if r.status == GateStatus.WARN)
    n_unverified = sum(1 for r in results if r.status == GateStatus.UNVERIFIED)
    n_total = len(results)
    lines.append(f"- **Total gates:** {n_total}")
    lines.append(f"- **Passed:** {n_pass} ✓")
    lines.append(f"- **Failed:** {n_fail} ✗")
    lines.append(f"- **Warned:** {n_warn} ⚠")
    lines.append(f"- **Unverified:** {n_unverified} ?")
    lines.append("")
    lines.append("## Results")
    lines.append("")
    lines.append("| Status | Gate | Message |")
    lines.append("|--------|------|---------|")
    for r in results:
        icon = {"PASS": "✓", "FAIL": "✗", "WARN": "⚠", "UNVERIFIED": "?"}[r.status.value]
        msg = r.message.replace("|", "\\|")
        lines.append(f"| {icon} {r.status.value} | `{r.gate_name}` | {msg} |")
    lines.append("")
    if n_fail > 0:
        lines.append("## Failures (must fix)")
        lines.append("")
        for r in results:
            if r.status == GateStatus.FAIL:
                lines.append(f"### {r.gate_name}")
                lines.append(f"")
                lines.append(f"{r.message}")
                lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def report_console(results: list[GateResult], verbose: bool = False) -> None:
    """Print results to console. If verbose, print all; else just FAILs."""
    for r in results:
        if r.status == GateStatus.FAIL or verbose:
            icon = {"PASS": "✓", "FAIL": "✗", "WARN": "⚠", "UNVERIFIED": "?"}[r.status.value]
            print(f"  [{icon}] {r.gate_name}: {r.message}")
