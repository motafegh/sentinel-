from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Any

from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.orchestration.state import AuditState
from src.orchestration.nodes._helpers import _llm_enabled
from src.orchestration.routing import CLASS_TO_DETECTORS
from src.orchestration.timing import step_timer
from src.orchestration.timeouts import (
    ENV_REFLECTION_TIMEOUT_S,
    DEFAULT_REFLECTION_TIMEOUT_S,
    get_timeout,
)


async def reflection(state: AuditState) -> dict[str, Any]:
    """
    Self-critique pass (A.3) — runs after synthesizer.

    Checks the assembled audit for internal consistency:
        - unused_evidence:    collected (RAG/static) but not reflected in verdicts
        - contradictions:     tools disagreeing (from cross_validator)
        - uncertain_verdicts: DISPUTED/WATCH or confidence < 0.7
        - failure_modes:      what could make this audit wrong

    Uses the strong LLM when available for a narrative `summary`; always
    computes the structured lists rule-based so the node is meaningful even
    with no LLM. Never raises.

    State updates:
        reflection_notes → {unused_evidence, contradictions, uncertain_verdicts,
                            failure_modes, summary, llm_used}
    """
    report = state.get("final_report", {}) or {}
    verdicts: dict[str, str] = report.get("verdicts", {}) or state.get("verdicts", {}) or {}
    vuln_verdicts = report.get("vulnerability_verdicts", []) or []
    static_findings = state.get("static_findings", []) or []
    rag_results = state.get("rag_results", []) or []
    contradictions = state.get("contradictions", {}) or {}
    confidence_by_class = state.get("confidence_by_class", {}) or {}
    truncated = report.get("ml_truncated", False)

    cited_classes = {v.get("vulnerability_class") for v in vuln_verdicts}

    # ── Rule-based structured critique ───────────────────────────────────────
    unused_evidence: list[str] = []
    if rag_results and not cited_classes:
        unused_evidence.append(
            f"{len(rag_results)} RAG exploit chunk(s) retrieved but no class was adjudicated."
        )
    uncited_static = [
        f.get("detector") for f in static_findings
        if f.get("impact") in ("High", "Medium")
        and not any(d in CLASS_TO_DETECTORS.get(c, []) for c in cited_classes
                    for d in [f.get("detector")])
    ]
    if uncited_static:
        unused_evidence.append(
            f"{len(uncited_static)} High/Medium static finding(s) not tied to a verdict: "
            f"{', '.join(str(d) for d in uncited_static[:5])}"
        )

    uncertain: list[str] = []
    for v in vuln_verdicts:
        cls = v.get("vulnerability_class", "?")
        verdict = v.get("verdict", "")
        conf = confidence_by_class.get(cls)
        if verdict in ("DISPUTED", "WATCH"):
            uncertain.append(f"{cls}: verdict {verdict}")
        elif isinstance(conf, (int, float)) and conf < 0.70:
            uncertain.append(f"{cls}: confidence {conf:.0%} below 0.70")

    failure_modes: list[str] = []
    if truncated:
        failure_modes.append(
            "Contract exceeded 512 CodeBERT tokens — tail code unanalysed; "
            "verdicts may miss vulnerabilities in the truncated region."
        )
    if any(v.get("vulnerability_class") == "ExternalBug" for v in vuln_verdicts):
        failure_modes.append(
            "ExternalBug flagged — the ML model is known to over-predict this class "
            "(Run 12 class-definition mismatch); treat as lower confidence unless "
            "corroborated by an inter-contract call finding."
        )
    if not static_findings:
        failure_modes.append(
            "No static-analysis findings available (tool unavailable or fast path) — "
            "verdicts rest on the ML signal alone."
        )

    contradiction_list = [
        f"{cls}: {'; '.join(reasons)}" for cls, reasons in contradictions.items()
    ]

    notes: dict[str, Any] = {
        "unused_evidence":    unused_evidence,
        "contradictions":     contradiction_list,
        "uncertain_verdicts": uncertain,
        "failure_modes":      failure_modes,
        "summary":            "",
        "llm_used":           False,
    }

    # ── Optional LLM narrative summary ───────────────────────────────────────
    if _llm_enabled() and vuln_verdicts:
        try:
            from langchain_core.messages import HumanMessage, SystemMessage
            from src.llm.client import get_strong_llm

            verdict_lines = "\n".join(
                f"  - {v.get('vulnerability_class')}: {v.get('verdict')} "
                f"(prob {v.get('probability', 0.0):.1%}, "
                f"conf {confidence_by_class.get(v.get('vulnerability_class'), 0.0):.0%})"
                for v in vuln_verdicts
            )
            system_msg = SystemMessage(content=(
                "You are a senior auditor performing a SELF-CRITIQUE of an audit "
                "already produced. In 3-5 sentences, flag any internal "
                "inconsistencies, over/under-confidence, or evidence that was "
                "ignored. Be skeptical and concrete. Output plain prose only."
            ))
            user_msg = HumanMessage(content=(
                f"Verdicts:\n{verdict_lines}\n\n"
                f"Unused evidence: {unused_evidence or 'none'}\n"
                f"Contradictions: {contradiction_list or 'none'}\n"
                f"Uncertain: {uncertain or 'none'}\n"
                f"Failure modes: {failure_modes or 'none'}"
            ))
            _timeout = get_timeout(ENV_REFLECTION_TIMEOUT_S, DEFAULT_REFLECTION_TIMEOUT_S)
            llm = get_strong_llm(max_tokens=int(os.getenv("REFLECTION_MAX_TOKENS", "1024")))
            with step_timer(
                "reflection.llm_summary",
                address=state.get("contract_address", "unknown"),
                budget_s=_timeout,
            ):
                loop = asyncio.get_running_loop()
                fut = loop.run_in_executor(None, llm.invoke, [system_msg, user_msg])
                resp = await asyncio.wait_for(fut, timeout=_timeout)
            notes["summary"] = resp.content.strip()
            notes["llm_used"] = True
            logger.info("reflection | LLM self-critique generated ({} chars)", len(notes["summary"]))
        except Exception as exc:
            logger.warning("reflection | LLM failed (rule-based notes kept): {}", exc)

    if not notes["summary"]:
        parts = []
        if uncertain:
            parts.append(f"{len(uncertain)} verdict(s) are uncertain")
        if unused_evidence:
            parts.append(f"{len(unused_evidence)} evidence item(s) unused")
        if failure_modes:
            parts.append(f"{len(failure_modes)} failure mode(s) noted")
        notes["summary"] = (
            "Self-critique: " + "; ".join(parts) + "."
            if parts else "Self-critique: audit is internally consistent; no concerns flagged."
        )

    logger.info(
        "reflection complete | uncertain={} unused={} failure_modes={} llm={}",
        len(uncertain), len(unused_evidence), len(failure_modes), notes["llm_used"],
    )
    return {"reflection_notes": notes}
