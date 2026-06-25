from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.orchestration.state import AuditState
from src.orchestration.nodes._helpers import _llm_enabled
from src.orchestration.routing import compute_overall_verdict, prob_to_severity
from src.orchestration.timing import step_timer
from src.orchestration.timeouts import (
    ENV_SYNTHESIZER_NARRATIVE_TIMEOUT_S,
    DEFAULT_SYNTHESIZER_NARRATIVE_TIMEOUT_S,
    get_timeout,
)
from src.ingestion.pipeline import REPORTS_DIR


async def synthesizer(state: AuditState) -> dict[str, Any]:
    """
    Assemble the final audit report from all available node outputs.

    RECALL — what this node does (M5 vs M6):
        M5 (current):  pure data assembly — no LLM call.
            Structured JSON report from available state fields.
            The "recommendation" field is rule-based for M5.

        M6 (next):     call the LLM synthesizer agent.
            The synthesizer agent uses STRONG model (gemma-4-e2b-it, was qwen3.5-9b-ud) to
            produce a natural-language audit summary + recommendation.
            The M5 structured report becomes the LLM's context.

    Report schema (Track 3 / multi-label) — matches SENTINEL-SPEC §8.1:
        contract_address:   str
        overall_label:      str          "vulnerable" | "safe" | "unknown"
        risk_probability:   float        max(probability) across detected vulns
        top_vulnerability:  str | None   class name with highest probability
        vulnerabilities:    list         [{vulnerability_class, probability}, ...]
        threshold:          float        per-class decision boundary used
        ml_truncated:       bool         True if contract exceeded 512 tokens
        num_nodes:          int          AST node count
        num_edges:          int          AST edge count
        rag_evidence:       list         matched exploit chunks (deep path)
        audit_history:      list         prior on-chain audit records
        static_findings:    dict | None  reserved for M6
        recommendation:     str          rule-based in M5, LLM-generated in M6
        error:              str | None   any non-fatal error during the run
        path_taken:         str          "deep" or "fast" — for observability

    BRIDGE (Issue #1):
        If contract_address is known, the final_report is persisted to
        data/reports/{contract_address}.json BEFORE this node returns.
        feedback_loop.py reads this file by contract_address when it
        processes the on-chain AuditSubmitted event, and uses
        report["top_vulnerability"] as the vuln_type metadata field.
        This replaces the hardcoded vuln_type="unknown" that made all
        on-chain RAG findings invisible to filtered searches.

    State updates:
        final_report → complete report dict
    """
    ml_result       : dict       = state.get("ml_result",       {})
    rag_results     : list       = state.get("rag_results",     [])
    audit_history   : list       = state.get("audit_history",   [])
    static_findings : list       = state.get("static_findings",  [])
    routing_decisions: list      = state.get("routing_decisions", [])
    error           : str | None = state.get("error")

    label      = ml_result.get("label",     "unknown")
    confirmed  = ml_result.get("confirmed",  [])
    suspicious = ml_result.get("suspicious", [])
    # All flagged classes (CONFIRMED + SUSPICIOUS) — used for verdicts and report.
    # Falls back to legacy `vulnerabilities` field if three-tier keys absent.
    all_flagged = confirmed + suspicious or ml_result.get("vulnerabilities", [])
    threshold  = ml_result.get("threshold", 0.50)
    truncated  = ml_result.get("truncated", False)
    num_nodes  = ml_result.get("num_nodes", 0)
    num_edges  = ml_result.get("num_edges", 0)

    # Derive risk_probability and top_vulnerability from CONFIRMED tier first.
    risk_source = confirmed or suspicious or ml_result.get("vulnerabilities", [])
    if risk_source:
        top_vuln      = max(risk_source, key=lambda v: v.get("probability", 0.0))
        risk_prob     = round(top_vuln.get("probability", 0.0), 4)
        top_vuln_name = top_vuln.get("vulnerability_class")
    else:
        risk_prob     = 0.0
        top_vuln_name = None

    # Determine which path was taken.
    path_taken = "deep" if (rag_results or static_findings) else "fast"

    # ── P2 Shape A: fuse() is the sole verdict producer ───────────────────────
    # Every channel upstream (consensus_engine, cross_validator) emits Evidence
    # objects to state["evidence_list"] (append-reducer). This node adds RAG
    # evidence, then fuse() de-correlates, aggregates, and applies FN/FP asymmetry
    # to produce verdict_provable (deterministic-only → ZK-anchorable) and
    # verdict_full (all evidence → human report).
    from src.orchestration.verdict.emit import emit_rag_evidence
    from src.orchestration.verdict.fuse import fuse
    from src.orchestration.verdict.evidence import Polarity

    evidence_list: list[Any] = list(state.get("evidence_list", []) or [])
    evidence_list.extend(emit_rag_evidence(rag_results))

    fused = fuse(evidence_list)
    verdict_provable: dict[str, str] = {}
    verdict_full: dict[str, str] = {}
    class_confidences: dict[str, float] = {}
    for cls, cv in fused.items():
        verdict_provable[cls] = cv.verdict_provable
        verdict_full[cls] = cv.verdict_full
        class_confidences[cls] = cv.confidence

    verdicts = verdict_full

    confirmations: dict[str, list[str]] = {}
    for ev in evidence_list:
        if ev.polarity == Polarity.SUPPORTS:
            confirmations.setdefault(ev.vuln_class, []).append(ev.source)

    has_supports: set[str] = set()
    has_refutes: set[str] = set()
    for ev in evidence_list:
        if ev.polarity == Polarity.SUPPORTS:
            has_supports.add(ev.vuln_class)
        elif ev.polarity == Polarity.REFUTES:
            has_refutes.add(ev.vuln_class)
    contradictions: dict[str, list[str]] = {
        cls: ["conflicting SUPPORTS/REFUTES evidence"]
        for cls in (has_supports & has_refutes)
    }

    flagged_by_cls = {v.get("vulnerability_class"): v for v in all_flagged}
    consensus_verdict: dict[str, dict] = state.get("consensus_verdict", {})

    all_classes = set(flagged_by_cls.keys()) | set(verdict_full.keys()) | set(consensus_verdict.keys())
    vuln_verdicts: list[dict] = []
    for cls in sorted(all_classes):
        vuln = flagged_by_cls.get(cls, {"vulnerability_class": cls, "probability": 0.0})
        prob = float(vuln.get("probability", 0.0))
        verdict = verdict_full.get(cls, "SAFE")
        ev_sources = list(dict.fromkeys(
            ev.source for ev in evidence_list if ev.vuln_class == cls
        ))
        vuln_verdicts.append({
            "vulnerability_class": cls,
            "probability":         prob,
            "verdict":             verdict,
            "evidence_sources":    ev_sources,
            "severity":            prob_to_severity(prob),
            "confidence":          class_confidences.get(cls, 0.0),
        })

    overall_verdict = compute_overall_verdict(verdicts)

    # ── Rule-based recommendation (fallback) ─────────────────────────────────
    # Used when the LLM is unavailable or times out.
    if not ml_result:
        recommendation = (
            "ML assessment failed — manual review required. "
            "Check that the inference server (port 8001) is running."
        )
    elif label in ("confirmed_vulnerable", "vulnerable") and risk_prob >= 0.70:
        rag_count    = len(rag_results)
        prior_count  = len(audit_history)
        slither_high = sum(
            1 for f in static_findings if f.get("impact") in ("High", "Medium")
        )
        recommendation = (
            f"HIGH RISK — top vulnerability: {top_vuln_name} "
            f"(probability {risk_prob:.1%}, CONFIRMED tier). "
            f"{rag_count} similar exploit pattern(s) found in DeFiHackLabs corpus. "
            f"{slither_high} Slither High/Medium finding(s). "
            f"{prior_count} prior on-chain audit(s). "
            "Recommend full manual audit before deployment."
        )
    elif label in ("confirmed_vulnerable", "suspicious", "vulnerable"):
        tier_note = (
            f"{len(confirmed)} confirmed, {len(suspicious)} suspicious"
            if (confirmed or suspicious)
            else f"probability {risk_prob:.1%}"
        )
        recommendation = (
            f"MODERATE RISK — top vulnerability: {top_vuln_name} "
            f"({tier_note}). "
            "Recommend targeted review of flagged patterns."
        )
    else:
        recommendation = (
            f"LOW RISK — no vulnerability exceeded detection threshold "
            f"(max probability: {risk_prob:.1%}). "
            "Standard due diligence recommended."
        )

    # ── LLM narrative (T3-A / Move 5) ────────────────────────────────────────
    # Attempt a structured Markdown security narrative from the strong LLM.
    # Falls back silently to the rule-based recommendation above on any failure
    # (LLM unavailable, timeout, malformed response).
    narrative: str | None = None
    if ml_result and _llm_enabled():
        try:
            from langchain_core.messages import HumanMessage, SystemMessage
            from src.llm.client import get_strong_llm

            # Fix applied 2026-06-21 ("narrative hallucination" incident, see
            # docs/changes/2026-06-21-agents-manual-verification-real-bugs-found.md):
            # vuln_lines used to list every ML-flagged class (down to weak
            # SUSPICIOUS-tier noise that never even reached the deep-path
            # threshold) with NO verdict attached. The model had no signal that
            # a listed class had actually been cleared, and wrote about it as
            # if it were a real finding — observed live: a narrative described
            # a "Reentrancy risk" on a contract whose Reentrancy verdict was
            # SAFE and which contains zero external calls. `verdicts` (computed
            # just above) is now attached to every line, mirroring the pattern
            # reflection's prompt already used correctly.
            vuln_lines = "\n".join(
                f"  - [{v.get('tier', 'CONFIRMED')}] "
                f"{v.get('vulnerability_class', '?')}: {v.get('probability', 0.0):.1%} "
                f"→ verdict: {verdicts.get(v.get('vulnerability_class', ''), 'PENDING')}"
                for v in all_flagged
            ) or "  (none detected)"

            rag_lines = "\n".join(
                f"  [{i + 1}] {c.get('metadata', {}).get('protocol', 'unknown')}: "
                f"{c.get('content', '')[:120]}..."
                for i, c in enumerate(rag_results[:3])
            ) if rag_results else "  (no matching exploit evidence retrieved)"

            slither_lines = "\n".join(
                f"  [{f.get('impact', '')}] {f.get('detector', '')}: "
                f"{f.get('description', '')[:100]}"
                for f in static_findings[:5]
                if f.get("impact") in ("High", "Medium")
            ) if static_findings else "  (no High/Medium static analysis findings)"

            code_snippet = state.get("contract_code", "")[:500].strip()

            # ExternalBug: add inter-contract call graph to prompt so LLM can
            # reason about oracle/price-feed dependency risks explicitly.
            ext_calls = state.get("external_call_summary", [])
            ext_flagged = any(
                v.get("vulnerability_class") == "ExternalBug" for v in all_flagged
            )
            ext_call_lines = ""
            if ext_flagged and ext_calls:
                ext_call_lines = "\n**Inter-contract call graph (ExternalBug context):**\n" + "\n".join(
                    f"  {c['caller_function']}({c['caller_contract']}) "
                    f"→ {c['callee_contract']}.{c['callee_function']}"
                    + (" [INTERFACE]" if c.get("callee_is_interface") else "")
                    for c in ext_calls[:8]
                )

            system_msg = SystemMessage(content=(
                "You are a senior smart contract security auditor. "
                "Produce a concise, structured Markdown security assessment with exactly "
                "these four sections:\n"
                "## Severity\n"
                "ONE of: CRITICAL | HIGH | MEDIUM | LOW | INFORMATIONAL\n"
                "## Vulnerability Summary\n"
                "2–3 sentences describing what was detected and why it is dangerous. "
                "Only discuss classes whose verdict below is CONFIRMED or LIKELY — "
                "if a class's verdict is SAFE or DISPUTED, do NOT describe it as a "
                "real risk even though it appears in the list, and do NOT introduce "
                "a vulnerability class that is not in the list at all.\n"
                "## Exploit Pattern\n"
                "How an attacker could exploit the CONFIRMED/LIKELY class(es) above — "
                "the RAG section below is general background on similar historical "
                "exploits, not necessarily evidence about THIS contract; only cite it "
                "if it genuinely matches a CONFIRMED/LIKELY class.\n"
                "## Recommended Fix\n"
                "Concrete, actionable mitigation steps specific to the detected vulnerability.\n"
                "Be concise. Output only the Markdown, no preamble."
            ))

            tier_summary = (
                f"{len(confirmed)} CONFIRMED (≥0.55), {len(suspicious)} SUSPICIOUS (0.25–0.54)"
                if (confirmed or suspicious) else ""
            )
            user_msg = HumanMessage(content=(
                f"**Contract address:** {state.get('contract_address', 'unknown')}\n"
                f"**ML model assessment:** {label}"
                + (f" — {tier_summary}" if tier_summary else "") + "\n\n"
                f"**ML-flagged classes (tier: class: probability: verdict):**\n{vuln_lines}\n\n"
                f"**RAG retrieved exploit patterns (general historical reference — "
                f"NOT necessarily about this contract; only use if it matches a "
                f"CONFIRMED/LIKELY class above):**\n{rag_lines}\n\n"
                f"**Static analysis findings (High/Medium):**\n{slither_lines}\n"
                + ext_call_lines + "\n\n"
                f"**Contract code snippet (first 500 chars):**\n```solidity\n{code_snippet}\n```"
            ))

            llm = get_strong_llm(max_tokens=int(os.getenv("SYNTHESIZER_MAX_TOKENS", "4096")))
            # Timeout configurable via SYNTHESIZER_TIMEOUT_S env var.
            # 2026-06-17: bumped default 45s → 120s. The narrative prompt is
            # longer than cross_validator (full contract snippet + multi-class
            # summary). 120s gives 2.5× headroom. Failure → narrative=None
            # (synthesizer rule-based fallback handles the rest).
            # FIX-17 (2026-06-17): pass max_tokens=4096 to LLM. Without this,
            # LM Studio's default ~2K is too small for the 4-section narrative
            # + reasoning content (model returns content="" → empty exception).
            _synthesizer_timeout = get_timeout(
                ENV_SYNTHESIZER_NARRATIVE_TIMEOUT_S, DEFAULT_SYNTHESIZER_NARRATIVE_TIMEOUT_S
            )
            with step_timer(
                "synthesizer.narrative",
                address=state.get("contract_address", "unknown"),
                budget_s=_synthesizer_timeout,
            ):
                loop = asyncio.get_running_loop()
                fut = loop.run_in_executor(None, llm.invoke, [system_msg, user_msg])
                response = await asyncio.wait_for(fut, timeout=_synthesizer_timeout)
            narrative = response.content.strip()
            logger.info("synthesizer | LLM narrative generated ({} chars)", len(narrative))

        except Exception as _llm_exc:
            logger.warning(
                "synthesizer | LLM narrative failed (using rule-based fallback): {}",
                _llm_exc,
            )
            narrative = None

    truncated_note = (
        "\n\n> **NOTE:** Contract exceeded 512 CodeBERT tokens — "
        "tail code was not analysed. Manual review of the unanalysed portion is recommended."
    ) if truncated else ""

    final_recommendation = (narrative or recommendation) + truncated_note

    report = {
        "contract_address":       state.get("contract_address", ""),
        "overall_label":          label,
        "overall_verdict":        overall_verdict,
        "risk_probability":       risk_prob,
        "top_vulnerability":      top_vuln_name,
        "confirmed":              confirmed,
        "suspicious":             suspicious,
        "vulnerabilities":        all_flagged,
        "probabilities":          ml_result.get("probabilities", {}),
        "tier_thresholds":        ml_result.get("tier_thresholds", {}),
        "vulnerability_verdicts": vuln_verdicts,
        "threshold":              threshold,
        "ml_truncated":           truncated,
        "num_nodes":              num_nodes,
        "num_edges":              num_edges,
        "rag_evidence":           rag_results,
        "audit_history":          audit_history,
        "static_findings":        static_findings,
        "external_call_summary":  state.get("external_call_summary", []),
        "routing_decisions":      routing_decisions,
        "consensus_verdict":      state.get("consensus_verdict", {}),
        "debate_transcript":      state.get("debate_transcript", {}),
        "recommendation":         final_recommendation,
        "narrative":              narrative,
        "error":                  error,
        "path_taken":             path_taken,
    }

    # ── BRIDGE (Issue #1): persist report for feedback_loop.py ──────────────
    # feedback_loop.py has no access to in-memory state — it runs as a
    # separate process listening to on-chain events. Writing the report to
    # disk by contract_address gives it the vulnerability_class it needs to
    # index on-chain findings with a meaningful vuln_type instead of "unknown".
    #
    # Only write if contract_address is known (it may be empty in test runs).
    # Failures are logged but never raise — the report is still returned
    # to the caller; a missing file only degrades RAG quality, not correctness.
    contract_address = state.get("contract_address", "").strip()
    if contract_address:
        try:
            REPORTS_DIR.mkdir(parents=True, exist_ok=True)
            report_path = REPORTS_DIR / f"{contract_address}.json"
            report_path.write_text(json.dumps(report, indent=2))
            logger.debug("synthesizer | report persisted → {}", report_path)
        except Exception as exc:
            logger.warning(
                "synthesizer | could not persist report for bridge (non-fatal): {}", exc
            )

    logger.info(
        "synthesizer complete | label={} | verdict={} | risk_prob={:.3f} | "
        "top_vuln={} | confirmed={} | suspicious={} | path={} | "
        "rag_chunks={} | prior_audits={} | static_findings={}",
        label,
        overall_verdict,
        risk_prob,
        top_vuln_name,
        len(confirmed),
        len(suspicious),
        path_taken,
        len(rag_results),
        len(audit_history),
        len(static_findings),
    )

    return {
        "final_report":        report,
        "verdicts":            verdicts,
        "confirmations":       confirmations,
        "contradictions":      contradictions,
        "confidence_by_class": class_confidences,
        "verdict_provable":    verdict_provable,
        "verdict_full":        verdict_full,
    }
