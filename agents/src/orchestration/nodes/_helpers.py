"""
nodes/_helpers.py — Shared utilities used by multiple nodes.

Extracted from nodes.py (P2 split, 2026-06-24). Contains:
  _llm_enabled, _call_mcp_tool, _parse_aderyn_report, _run_aderyn_on_file,
  _extract_external_call_summary, _signals_for_class, _best_rag_score.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from loguru import logger

from src.orchestration.timeouts import (
    ENV_ADERYN_TIMEOUT_S,
    DEFAULT_ADERYN_TIMEOUT_S,
    get_timeout,
)
from src.orchestration.routing import CLASS_TO_DETECTORS


def _llm_enabled() -> bool:
    """False when AGENTS_DISABLE_LLM is truthy — nodes then use rule-based fallback."""
    return os.getenv("AGENTS_DISABLE_LLM", "").strip().lower() not in ("1", "true", "yes")


# ── MCP tool call ────────────────────────────────────────────────────────────

async def _call_mcp_tool(
    server_url: str,
    tool_name: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    from mcp import ClientSession
    from mcp.client.sse import sse_client

    async with sse_client(server_url) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, arguments)
            raw = result.content[0].text
            try:
                return json.loads(raw)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"MCP tool '{tool_name}' returned non-JSON response: {raw[:200]}"
                ) from exc


# ── Aderyn helpers ───────────────────────────────────────────────────────────

def _parse_aderyn_report(data: dict[str, Any]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for impact_label, bucket_key in [("High", "high_issues"), ("Low", "low_issues")]:
        for issue in data.get(bucket_key, {}).get("issues", []):
            instances = issue.get("instances", [])
            lines = sorted({
                inst["line_no"] for inst in instances
                if isinstance(inst.get("line_no"), int)
            })
            findings.append({
                "tool":           "aderyn",
                "detector":       issue.get("detector_name", issue.get("title", "unknown")),
                "impact":         impact_label,
                "confidence":     "Medium",
                "description":    issue.get("description", issue.get("title", "")),
                "lines":          lines,
                "function_names": [],
            })
    return findings


def _run_aderyn_on_file(contract_code: str) -> list[dict[str, Any]]:
    import json as _json

    findings: list[dict[str, Any]] = []
    tmpdir: str | None = None
    try:
        tmpdir = tempfile.mkdtemp(prefix="sentinel_aderyn_")
        sol_path = Path(tmpdir) / "contract.sol"
        sol_path.write_text(contract_code)
        report_path = Path(tmpdir) / "report.json"

        result = subprocess.run(
            ["aderyn", "--output", str(report_path), tmpdir],
            capture_output=True,
            text=True,
            timeout=get_timeout(ENV_ADERYN_TIMEOUT_S, DEFAULT_ADERYN_TIMEOUT_S),
        )
        if result.returncode != 0 or not report_path.exists():
            logger.debug(
                "_run_aderyn_on_file | exit={} stderr={}",
                result.returncode,
                result.stderr[:200],
            )
            return []

        data = _json.loads(report_path.read_text())
        findings = _parse_aderyn_report(data)

    except FileNotFoundError:
        logger.debug("_run_aderyn_on_file | aderyn not installed — skipping")
    except subprocess.TimeoutExpired:
        logger.warning("_run_aderyn_on_file | aderyn timed out after 90s")
    except Exception as exc:
        logger.warning("_run_aderyn_on_file | error (non-fatal): {}", exc)
    finally:
        if tmpdir:
            shutil.rmtree(tmpdir, ignore_errors=True)

    return findings


def _extract_external_call_summary(sl: Any) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    try:
        for contract in sl.contracts:
            if contract.is_interface:
                continue
            for fn in contract.functions_and_modifiers:
                for callee_contract, callee_fn in getattr(fn, "high_level_calls", []):
                    callee_name = (
                        callee_fn.name
                        if hasattr(callee_fn, "name")
                        else str(callee_fn)
                    )
                    calls.append({
                        "caller_contract":  contract.name,
                        "caller_function":  fn.name,
                        "callee_contract":  callee_contract.name,
                        "callee_function":  callee_name,
                        "callee_is_interface": getattr(callee_contract, "is_interface", False),
                    })
    except Exception as exc:
        logger.debug("_extract_external_call_summary | partial failure (non-fatal): {}", exc)
    return calls


# ── Signal detection ─────────────────────────────────────────────────────────

def _signals_for_class(class_name: str, static_findings: list[dict]) -> tuple[bool, bool]:
    detectors = CLASS_TO_DETECTORS.get(class_name, [])
    tokens = {tok for det in detectors for tok in det.split("-") if len(tok) > 3}

    slither_found = False
    aderyn_found = False
    for f in static_findings:
        det = str(f.get("detector", "")).lower()
        tool = f.get("tool", "")
        if tool == "slither" and det in detectors:
            slither_found = True
        elif tool == "aderyn" and any(tok in det for tok in tokens):
            aderyn_found = True
    return slither_found, aderyn_found


def _best_rag_score(class_name: str, rag_results: list[dict]) -> float:
    best = 0.0
    for c in rag_results:
        vt = c.get("metadata", {}).get("vulnerability_type", "")
        if vt == class_name:
            score = float(c.get("score", c.get("similarity", 0.0)))
            if score > best:
                best = score
    return best
