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
    """
    False when AGENTS_DISABLE_LLM is truthy OR SENTINEL_DETERMINISTIC is set.

    P5 (2026-06-26): SENTINEL_DETERMINISTIC=1 disables LLM calls to ensure
    reproducible verdicts (LLM outputs are non-deterministic even at temp=0).
    Nodes then use rule-based fallback.
    """
    if os.getenv("SENTINEL_DETERMINISTIC", "").strip().lower() in ("1", "true", "yes"):
        return False
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

class AderynRunError(RuntimeError):
    """
    Raised when Aderyn invocation fails for a runtime reason after the binary
    was resolved (timeout, non-zero exit, malformed report, missing report file).

    Rule 5C (CLAUDE.md): distinct from FileNotFoundError, which signals the
    binary itself is unresolvable. Callers catch both, but they need to
    distinguish "tool absent" from "tool ran and failed" in `tool_status`.
    """


def _resolve_aderyn_binary() -> str:
    """
    Resolve the Aderyn binary path.

    Searches in order:
        1. `shutil.which("aderyn")` (PATH-resolved — works when aderyn is on PATH
           via a venv, an explicit export, or a container build arg)
        2. `~/.cargo/bin/aderyn` (the canonical Rust/cargo install location —
           `cargo install aderyn` defaults here)

    The cargo fallback is the fix for the silent-skip bug: Aderyn 0.6.8 is
    installed at `~/.cargo/bin/aderyn` but that directory is NOT on PATH in
    many environments (including the agents venv). Bare `subprocess.run(["aderyn", …])`
    raised `FileNotFoundError` and the function returned `[]` — the same shape
    as "Aderyn ran clean" — which poisoned every downstream number that
    expected Aderyn's real signal. See CLAUDE.md Rule 5C.

    Raises:
        FileNotFoundError: with the exact lookup paths if Aderyn is not
            installed at any of them. Caller MUST surface this in
            `tool_status["aderyn"] = {"ran": False, "reason": ..., "resolved": ...}`,
            NOT swallow it as an empty findings list.
    """
    candidates: list[str] = []
    on_path = shutil.which("aderyn")
    if on_path:
        candidates.append(on_path)

    cargo_path = os.path.expanduser("~/.cargo/bin/aderyn")
    if os.path.isfile(cargo_path) and os.access(cargo_path, os.X_OK):
        if cargo_path not in candidates:
            candidates.append(cargo_path)

    if not candidates:
        raise FileNotFoundError(
            "Aderyn binary not found. Searched: "
            f"shutil.which('aderyn')={on_path!r}, "
            f"~/.cargo/bin/aderyn={cargo_path!r}. "
            "Install via `cargo install aderyn` (Rust toolchain required) or "
            "symlink an existing aderyn binary into a directory on PATH "
            "(e.g. `ln -s ~/.cargo/bin/aderyn agents/.venv/bin/aderyn`)."
        )

    return candidates[0]


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
    """
    Run Aderyn on the given Solidity source code and return parsed findings.

    Rule 5C contract (CLAUDE.md): this function NEVER returns `[]` to signal
    "Aderyn was absent or failed." Failure modes raise specific exceptions
    that callers (static_analysis.py, quick_screen.py) catch and surface in
    `state["tool_status"]["aderyn"] = {"ran": False, "reason": ..., ...}`.
    An empty return value here means "Aderyn ran and produced zero findings."

    Returns:
        list of normalised finding dicts (same shape as Slither findings)
        when Aderyn ran successfully — possibly empty if the contract is
        clean by Aderyn's detectors.

    Raises:
        FileNotFoundError: Aderyn binary not resolvable via PATH or
            `~/.cargo/bin/aderyn`. Caller surfaces as
            `tool_status["aderyn"].ran = False, reason = "binary not found"`.
        AderynRunError:     Aderyn timed out, exited non-zero, or produced
            a malformed report. Caller surfaces with the precise reason.
    """
    import json as _json

    aderyn_bin = _resolve_aderyn_binary()
    timeout_s = get_timeout(ENV_ADERYN_TIMEOUT_S, DEFAULT_ADERYN_TIMEOUT_S)
    findings: list[dict[str, Any]] = []
    tmpdir: str | None = None
    try:
        tmpdir = tempfile.mkdtemp(prefix="sentinel_aderyn_")
        sol_path = Path(tmpdir) / "contract.sol"
        sol_path.write_text(contract_code)
        report_path = Path(tmpdir) / "report.json"

        result = subprocess.run(
            [aderyn_bin, "--output", str(report_path), tmpdir],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        if result.returncode != 0 or not report_path.exists():
            stderr_snippet = (result.stderr or "")[:200]
            raise AderynRunError(
                f"aderyn exited with code {result.returncode} "
                f"(stderr first 200 chars: {stderr_snippet!r})"
            )

        try:
            data = _json.loads(report_path.read_text())
        except (ValueError, OSError) as exc:
            raise AderynRunError(
                f"aderyn report at {report_path} could not be parsed as JSON: {exc}"
            ) from exc

        findings = _parse_aderyn_report(data)

    except subprocess.TimeoutExpired as exc:
        raise AderynRunError(
            f"aderyn timed out after {timeout_s}s on contract "
            f"({len(contract_code)} bytes); check the contract for an OOM-style "
            "compile loop or raise the ADERYN_TIMEOUT_S env override."
        ) from exc

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
