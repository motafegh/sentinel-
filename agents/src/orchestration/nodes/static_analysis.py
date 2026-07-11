from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Any

from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.orchestration.state import AuditState
from src.orchestration.routing import CLASS_TO_DETECTORS
from src.orchestration.nodes._helpers import (
    AderynRunError,
    _run_aderyn_on_file,
    _extract_external_call_summary,
)


async def static_analysis(state: AuditState) -> dict[str, Any]:
    """
    Run Slither directly on the contract source and return per-finding dicts.

    RECALL — why direct Slither call, not MCP:
        Slither is a Python library installed in this process.
        Spawning it via MCP would add latency for no benefit.
        The result is merged into state alongside rag_research output — both
        run in parallel in the deep path (LangGraph fan-out semantics).

    RECALL — what this node produces:
        Two tools run on the same temp file; findings are combined in one list.

        Slither (scoped to ML-flagged classes):
            {tool="slither", detector, impact, confidence, description, lines, function_names}

        Aderyn (full scan — Aderyn has no per-class scoping):
            {tool="aderyn", detector, impact, confidence, description, lines, function_names}
            Non-fatal AT THE NODE LEVEL (intentional — keep pipeline going on
            Slither-only evidence) but the failure MUST surface in
            `state["tool_status"]["aderyn"]` (Rule 5C, CLAUDE.md). The
            `tool_status` reducer is `_merge_tool_status` in state.py —
            one-level-deep merge per tool key. Eval layer reads `tool_status`
            to distinguish "Aderyn ran clean" from "Aderyn was absent or
            failed" — the two are NOT the same.

        Having both tool names in the findings lets cross_validator and synthesizer
        reason about corroboration: "Slither AND Aderyn both found X" is stronger
        evidence than either tool alone.

    RECALL — scoped detectors (Slither only):
        Slither is run with only the detectors relevant to ML-flagged classes.
        CLASS_TO_DETECTORS in routing.py defines the mapping.
        This reduces runtime 3–8× vs running all 90+ detectors on large contracts.
        Any class above DEEP_THRESHOLDS contributes its detectors to the active set.
        If ml_result is empty or no class is flagged, all detectors run (safe fallback).
        Aderyn always runs its full detector set — it has no equivalent scope API.

    State updates:
        static_findings → combined Slither + Aderyn findings (may be empty)
        tool_status     → {tool: {"ran": bool, "reason": str, ...}} for Aderyn
                          (and Slither when relevant) — see Rule 5C.
        error           → set on Slither failure (non-fatal; returns empty list)
    """
    contract_code = state.get("contract_code", "")
    if not contract_code or not contract_code.strip():
        logger.warning("static_analysis | contract_code empty — skipping")
        return {"static_findings": []}

    # Collect detector names relevant to classes above DEEP_THRESHOLDS.
    # Prefer probabilities dict (all 10 classes) over legacy vulnerabilities list.
    ml_result     = state.get("ml_result", {})
    probabilities = ml_result.get("probabilities", {})
    if probabilities:
        flagged_classes = {cls for cls, prob in probabilities.items() if prob >= 0.35}
    else:
        flagged_classes = {
            v["vulnerability_class"]
            for v in ml_result.get("vulnerabilities", [])
            if v.get("probability", 0.0) >= 0.35
        }
    scoped_detectors: set[str] = set()
    for cls in flagged_classes:
        scoped_detectors.update(CLASS_TO_DETECTORS.get(cls, []))

    logger.info(
        "static_analysis | running Slither | address={} | classes={} | detectors={}",
        state.get("contract_address", "unknown"),
        sorted(flagged_classes) or ["all"],
        sorted(scoped_detectors) or ["all"],
    )

    tmp_path: str | None = None
    slither_status: dict = {"ran": False, "reason": "not_started"}
    try:
        import inspect

        from slither import Slither
        from slither.detectors import all_detectors
        from slither.detectors.abstract_detector import AbstractDetector

        # Slither requires a real file path — write to temp file.
        with tempfile.NamedTemporaryFile(
            suffix=".sol",
            prefix="sentinel_static_",
            mode="w",
            encoding="utf-8",
            delete=False,
        ) as tmp:
            tmp.write(contract_code)
            tmp_path = tmp.name

        sl = Slither(tmp_path)

        # CRITICAL (found 2026-06-21 via manual real-contract verification):
        # The Slither() constructor registers ZERO detectors — sl._detectors starts
        # empty. The CLI (slither/__main__.py) explicitly registers every detector
        # class via slither.register_detector() before calling run_detectors();
        # this in-process API call must do the same or static_analysis silently
        # finds NOTHING on every contract, regardless of vulnerability. Confirmed by
        # direct comparison: `slither contract.sol` on the CLI found reentrancy-eth
        # on a textbook reentrant Vault; this node (pre-fix) found 0.
        all_detector_classes = [
            d for d in (getattr(all_detectors, name) for name in dir(all_detectors))
            if inspect.isclass(d) and issubclass(d, AbstractDetector)
        ]
        for detector_cls in all_detector_classes:
            sl.register_detector(detector_cls)

        # If we have a scoped detector set, filter _detectors before running.
        # Slither stores detector classes in sl._detectors (list).
        # Filtering by ARGUMENT attribute (the detector's CLI name) is stable
        # across Slither versions as it's part of the public detector interface.
        if scoped_detectors:
            sl._detectors = [  # type: ignore[attr-defined]
                d for d in sl._detectors  # type: ignore[attr-defined]
                if getattr(d, "ARGUMENT", "") in scoped_detectors
            ]

        # ExternalBug structural gap fix: GNN sees call_target_typed=1.00 for
        # typed interface calls (looks safe). Extract inter-contract call graph
        # so rag_research and synthesizer can reason about oracle/price manipulation.
        external_calls: list[dict] = []
        if "ExternalBug" in flagged_classes:
            external_calls = _extract_external_call_summary(sl)
            if external_calls:
                logger.info(
                    "static_analysis | ExternalBug: {} inter-contract call(s) extracted",
                    len(external_calls),
                )

        findings: list[dict] = []
        for result in sl.run_detectors():
            for finding in result:
                elements = finding.get("elements", [])
                lines: list[int] = []
                fn_names: list[str] = []
                for elem in elements:
                    src = elem.get("source_mapping", {})
                    elem_lines = src.get("lines", [])
                    if isinstance(elem_lines, list):
                        lines.extend(int(ln) for ln in elem_lines if isinstance(ln, int))
                    if elem.get("type") == "function":
                        fn_names.append(elem.get("name", ""))

                detector_name = finding.get("check", "unknown")
                findings.append({
                    "tool":           "slither",
                    "detector":       detector_name,
                    "impact":         finding.get("impact", "Unknown"),
                    "confidence":     finding.get("confidence", "Unknown"),
                    "description":    finding.get("description", ""),
                    "lines":          sorted(set(lines)),
                    "function_names": fn_names,
                })

        slither_status = {"ran": True, "n_findings": len(findings)}

        # Run Aderyn on the same source — adds findings with tool="aderyn".
        # Rule 5C (CLAUDE.md): this call may raise FileNotFoundError
        # (binary unresolvable) or AderynRunError (timeout / non-zero exit /
        # malformed report). The node stays non-fatal at the pipeline level
        # (intentional — Slither-only evidence is still useful) but the
        # failure MUST be visible in `state["tool_status"]["aderyn"]` so the
        # eval layer can distinguish "Aderyn ran clean" from "Aderyn was
        # absent or failed." An empty aderyn_findings list alone is no
        # longer sufficient.
        try:
            aderyn_findings = _run_aderyn_on_file(contract_code)
        except FileNotFoundError as exc:
            logger.warning(
                "static_analysis | aderyn unavailable — falling back to Slither only | {}",
                exc,
            )
            aderyn_findings = []
            aderyn_status = {
                "ran": False,
                "reason": "binary_not_found",
                "detail": str(exc),
                "fallback": "slither-only",
            }
        except AderynRunError as exc:
            logger.warning(
                "static_analysis | aderyn run failed — falling back to Slither only | {}",
                exc,
            )
            aderyn_findings = []
            aderyn_status = {
                "ran": False,
                "reason": "run_error",
                "detail": str(exc),
                "fallback": "slither-only",
            }
        else:
            aderyn_status = {
                "ran": True,
                "n_findings": len(aderyn_findings),
            }
        findings.extend(aderyn_findings)

        logger.info(
            "static_analysis complete | slither={} aderyn={} aderyn_ran={} external_calls={} | contract_address={}",
            len(findings) - len(aderyn_findings),
            len(aderyn_findings),
            aderyn_status.get("ran"),
            len(external_calls),
            state.get("contract_address", "unknown"),
        )
        return {
            "static_findings": findings,
            "external_call_summary": external_calls,
            "tool_status": {"slither": slither_status, "aderyn": aderyn_status},
        }

    except ImportError:
        logger.warning("static_analysis | slither not installed — skipping")
        slither_status = {"ran": False, "reason": "not_installed"}
        return {
            "static_findings": [],
            "external_call_summary": [],
            "tool_status": {"slither": slither_status},
        }

    except Exception as exc:
        logger.error("static_analysis failed: {}", exc)
        slither_status = {"ran": False, "reason": "error", "detail": str(exc)}
        return {
            "static_findings": [],
            "external_call_summary": [],
            "tool_status": {"slither": slither_status},
            "error": f"static_analysis: {exc}",
        }

    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError as e:
                logger.warning("static_analysis | failed to delete temp file {}: {}", tmp_path, e)
