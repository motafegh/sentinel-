"""Freshness checker — compare pinned source versions to upstream HEAD.

Also checks the installed slither-analyzer version against the latest PyPI release.
A stale Slither is an early-warning for the extractor-broke-silently failure mode
(Slither API changes broke graph_extractor.py in Run 9 — see ADR-0002).

Output: data/analysis/freshness_report.md (informational, not blocking).
"""

from __future__ import annotations

import subprocess
from datetime import datetime
from pathlib import Path


def run_freshness_check(cfg: dict, data_dir: Path) -> str:
    """Generate a freshness report and write it to data/analysis/freshness_report.md.

    Returns the report content as a string.
    """
    lines = [
        "# Freshness Report",
        f"Generated: {datetime.utcnow().isoformat()}Z",
        "",
        "## Source Pin Status",
        "",
    ]

    from sentinel_data.ingestion.ingest import _all_sources
    for name, entry in _all_sources(cfg).items():
        if not entry.get("enabled"):
            continue
        pin = entry.get("pin", "")
        url = entry.get("url", "")
        connector = entry.get("connector", "git")

        if connector == "git" and url:
            upstream = _git_upstream_head(url)
            if upstream and pin and not upstream.startswith(pin):
                status = f"STALE — pinned={pin[:12]} upstream={upstream[:12]}"
            elif not pin:
                status = f"UNPINNED — upstream HEAD={upstream[:12] if upstream else 'unknown'}"
            else:
                status = "OK"
        else:
            status = f"UNCHECKED (connector={connector})"

        lines.append(f"- **{name}**: {status}")

    lines += [
        "",
        "## Slither Version",
        "",
    ]
    lines.append(_slither_version_check())

    report = "\n".join(lines) + "\n"
    out = data_dir / "analysis" / "freshness_report.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report)
    return report


def _git_upstream_head(url: str) -> str:
    """Get the HEAD commit SHA of the upstream repo without cloning."""
    try:
        result = subprocess.run(
            ["git", "ls-remote", url, "HEAD"],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode == 0 and result.stdout:
            return result.stdout.split()[0]
    except Exception:
        pass
    return ""


def _slither_version_check() -> str:
    """Compare installed slither version to latest PyPI release."""
    installed = _installed_slither_version()
    latest = _latest_pypi_version("slither-analyzer")

    if not installed:
        return "slither-analyzer: NOT INSTALLED in this venv (install with `poetry install --with pipeline`)"
    if not latest:
        return f"slither-analyzer: installed={installed} | could not check PyPI"
    if installed == latest:
        return f"slither-analyzer: OK ({installed})"
    return (
        f"slither-analyzer: STALE — installed={installed} latest={latest}. "
        "API changes between versions can silently break graph_extractor.py (see ADR-0002)."
    )


def _installed_slither_version() -> str:
    """Return the slither-analyzer version installed in the current Python env.

    The check is scoped to the current interpreter — falling back to a subprocess
    that calls `python3` would read a *different* Python's installed packages,
    which is misleading when the data venv and the ml venv have different
    slither versions. If slither is not installed in *this* venv, return "".
    """
    try:
        import importlib.metadata
        return importlib.metadata.version("slither-analyzer")
    except Exception:
        return ""


def _latest_pypi_version(package: str) -> str:
    try:
        import urllib.request, json as _json
        url = f"https://pypi.org/pypi/{package}/json"
        with urllib.request.urlopen(url, timeout=5) as r:
            data = _json.loads(r.read())
        return data["info"]["version"]
    except Exception:
        return ""
