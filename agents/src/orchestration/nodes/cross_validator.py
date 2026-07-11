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
from src.orchestration.timing import step_timer
from src.orchestration.timeouts import (
    ENV_CROSS_VALIDATOR_SINGLE_PASS_TIMEOUT_S,
    DEFAULT_CROSS_VALIDATOR_SINGLE_PASS_TIMEOUT_S,
    ENV_DEBATE_TIMEOUT_S,
    DEFAULT_DEBATE_TIMEOUT_S,
    get_timeout,
)
from src.security import sanitize_for_prompt


async def cross_validator(state: AuditState) -> dict[str, Any]:
    """
    LLM-adjudicated per-class verdicts (Phase 2 deep path).

    Runs after audit_check (fan-in) and before synthesizer.
    For each ML-flagged class, prompts the strong LLM with all available
    evidence (ML tier + probability, Slither findings, RAG chunks, prior audits)
    and returns a structured per-class verdict.

    Verdict scale:
        CONFIRMED  — ML ≥ 0.55 AND Slither finding(s) agree
        LIKELY     — ML ≥ 0.35 AND at least one corroborating signal
        DISPUTED   — ML flagged but corroboration absent or contradictory
        WATCH      — ML 0.25–0.34, single weak signal; monitor only
        SAFE       — evidence points to false positive

    Falls back silently to empty dict on LLM failure — synthesizer then
    computes rule-based verdicts as before.

    State updates:
        verdicts       → {class: verdict_str}
        confirmations  → {class: [evidence_source, ...]}
        contradictions → {class: [description, ...]}
    """
    ml_result       = state.get("ml_result",       {})
    static_findings = state.get("static_findings",  [])
    rag_results     = state.get("rag_results",      [])
    audit_history   = state.get("audit_history",    [])

    confirmed  = ml_result.get("confirmed",  [])
    suspicious = ml_result.get("suspicious", [])
    all_flagged = confirmed + suspicious or ml_result.get("vulnerabilities", [])

    if not all_flagged:
        logger.info("cross_validator | no flagged classes — skipping")
        return {}

    if not _llm_enabled():
        logger.info("cross_validator | LLM disabled (AGENTS_DISABLE_LLM) — rule-based fallback")
        return {}

    # Real-audit finding (2026-06-21): ambiguous/small contracts can spread weak
    # "suspicious" signal (tier threshold 0.25) across most of the 10 classes.
    # Adjudicating all of them — especially with the 3-call debate plus the
    # contract source in every prompt — risks the FAST model overrunning
    # CROSS_VALIDATOR_TIMEOUT_S (3 sequential calls) on a tiny RTX 3070 box.
    # Cap to the top-N most probable classes; the rest fall back to rule-based
    # verdicts in synthesizer (they're weak signals anyway).
    #
    # WS1.5 (2026-06-21, Finding #13): the OLD sort was by raw ML probability
    # only — a class with strong tool corroboration but low ML score (e.g.
    # CallToUnknown at prob=0.249 with Slither+Aderyn agreeing) was excluded
    # from the debate even though consensus_engine (which already ran) had it
    # at CONFIRMED. Now sort by consensus confidence (which incorporates tool
    # corroboration) with ML prob as fallback, AND guarantee any class with a
    # tool hit is included regardless of rank.
    _max_classes = int(os.getenv("CROSS_VALIDATOR_MAX_CLASSES", "5"))
    consensus_verdict_state = state.get("consensus_verdict", {}) or {}
    if len(all_flagged) > _max_classes:
        all_flagged = sorted(all_flagged, key=lambda v: (
            consensus_verdict_state.get(v.get("vulnerability_class", ""), {}).get("confidence", 0.0),
            v.get("probability", 0.0),
        ), reverse=True)
        # Guarantee any class with a tool hit is adjudicated, regardless of rank.
        tool_classes = {
            c for c, v in consensus_verdict_state.items()
            if v.get("slither_match") or v.get("aderyn_match")
        }
        top = all_flagged[:_max_classes]
        for v in all_flagged[_max_classes:]:
            if v.get("vulnerability_class") in tool_classes:
                top.append(v)
        all_flagged = top

    logger.info("cross_validator | adjudicating {} class(es)", len(all_flagged))

    # Build Slither-finding index keyed by vuln class (without importing routing again)
    from src.orchestration.routing import CLASS_TO_DETECTORS

    slither_by_class: dict[str, list[str]] = {}
    for finding in static_findings:
        detector = finding.get("detector", "")
        for cls, detectors in CLASS_TO_DETECTORS.items():
            if detector in detectors:
                slither_by_class.setdefault(cls, []).append(
                    f"{finding.get('impact', '')} {detector}: "
                    f"{finding.get('description', '')[:80]}"
                )

    rag_topics = [
        c.get("metadata", {}).get("vulnerability_type", "")
        for c in rag_results[:5]
        if c.get("metadata", {}).get("vulnerability_type")
    ]
    prior_count = len(audit_history)

    class_lines = []
    eye_predictions = ml_result.get("eye_predictions", {}) or {}
    for vuln in all_flagged:
        cls    = vuln.get("vulnerability_class", "?")
        prob   = vuln.get("probability", 0.0)
        tier   = vuln.get("tier", "CONFIRMED")
        slither_hits = slither_by_class.get(cls, ["(no Slither findings)"])

        # D4 (WS3, 2026-06-22): per-eye clues — discountable hints showing
        # what kind of reasoning drives suspicion for this class.
        eye_line = ""
        if eye_predictions:
            per_eye = {
                eye: vals.get(cls, 0.0)
                for eye, vals in eye_predictions.items()
                if cls in vals
            }
            if per_eye:
                top_eye = max(per_eye, key=per_eye.get)
                eye_line = (
                    f"  Eye clues: "
                    f"{' '.join(f'{k}={v:.2f}' for k, v in sorted(per_eye.items()))} "
                    f"({top_eye} eye driving)\n"
                )

        class_lines.append(
            f"- {cls} [{tier}] prob={prob:.3f}\n"
            f"{eye_line}"
            f"  Slither: {'; '.join(slither_hits[:3])}\n"
            f"  RAG topics matched: {', '.join(rag_topics) or '(none)'}\n"
            f"  Prior audits: {prior_count}"
        )

    debate_transcript: dict[str, str] = {}
    try:
        from langchain_core.messages import HumanMessage, SystemMessage
        # 2026-06-17: Use FAST model (gemma-4-e2b-it) instead of STRONG.
        # Verdict picking from 5 options is a simple classification task.
        # STRONG (gemma-4-e2b-it, was qwen3.5-9b-ud) was taking 94s+ for 9 classes (TIMED OUT at 90s).
        # FAST runs ~3x faster, finishes in ~20-30s, quality is sufficient.
        # Override via CROSS_VALIDATOR_LLM_MODEL env var (model ID string).
        from src.llm.client import get_fast_llm, get_strong_llm
        _cv_model = os.getenv("CROSS_VALIDATOR_LLM_MODEL", "fast").lower()

        # WS4.1 (2026-06-21, Finding #6): per-role max_tokens cap. Previously
        # the 3 debate roles had NO output-length cap — each could generate
        # unlimited text, contributing to 75-115s per role (the prosecutor
        # alone took 178s on one contract). Caps + "be concise" instruction
        # cut the rambling while keeping the output complete.
        #
        # Tuning (2026-06-22 sweep on vulnerable_reentrant.sol):
        #   384/512 → LM Studio returns content="" (model's internal preamble
        #             hits the cap before producing output). Debate effectively
        #             doesn't run — verdict comes from consensus fallback.
        #   768    → Sweet spot. Prosecutor 657 chars, defender 900 chars,
        #             both non-empty, judge produces valid JSON. 28s debate
        #             (was 47s uncapped, 37s at 1024).
        #   1024   → Verbose (2021/1682 chars), 37s debate — no verdict
        #             improvement over 768.
        # The JUDGE is left uncapped (0) — LM Studio returns content="" when
        # the judge is capped (it generates reasoning before the JSON, hitting
        # the cap). Judge output is naturally short (~128 chars JSON).
        # Configurable via env vars.
        _prosecutor_max = int(os.getenv("DEBATE_PROSECUTOR_MAX_TOKENS", "768"))
        _defender_max = int(os.getenv("DEBATE_DEFENDER_MAX_TOKENS", "768"))
        _judge_max = int(os.getenv("DEBATE_JUDGE_MAX_TOKENS", "0"))  # 0 = uncapped
        if _cv_model == "strong":
            _llm_prosecutor = get_strong_llm(max_tokens=_prosecutor_max or None)
            _llm_defender = get_strong_llm(max_tokens=_defender_max or None)
            _llm_judge = get_strong_llm(max_tokens=_judge_max or None)
        else:
            _llm_prosecutor = get_fast_llm(max_tokens=_prosecutor_max or None)
            _llm_defender = get_fast_llm(max_tokens=_defender_max or None)
            _llm_judge = get_fast_llm(max_tokens=_judge_max or None)

        evidence_block = "\n".join(class_lines)

        # ── WS3 (2026-06-22): hotspot-guided source excerpt ──────────────
        # Replace the raw [:2000] char truncation with the specific functions
        # flagged by graph_explain's hotspot analysis. The debate gets the
        # vulnerable lines directly, PLUS the full source as reference.
        #
        # D3 resolved: hotspot-guided excerpt + sliding-window fallback.
        # P4 (2026-06-26): sanitize contract source before embedding in LLM prompt.
        raw_contract_code = state.get("contract_code", "") or ""
        sanitized_code, injection_matches = sanitize_for_prompt(raw_contract_code)
        contract_code = sanitized_code
        ml_hotspots = state.get("ml_hotspots", []) or []

        hotspot_excerpt_lines: list[str] = []
        if ml_hotspots and contract_code:
            source_lines = contract_code.splitlines()
            hotspots_by_cls: dict[str, list[dict]] = {}
            for h in ml_hotspots:
                c = h.get("class", "")
                if c:
                    hotspots_by_cls.setdefault(c, []).append(h)

            for vuln in all_flagged:
                cls = vuln.get("vulnerability_class", "")
                cls_hotspots = hotspots_by_cls.get(cls, [])
                if not cls_hotspots:
                    cls_hotspots = [
                        h for h in ml_hotspots
                        if cls in h.get("vulnerability_classes", [])
                    ]
                if not cls_hotspots:
                    continue

                hotspot_excerpt_lines.append(f"\n── {cls} ──")
                for h in cls_hotspots:
                    fn = h.get("fn_name", "?")
                    lines = h.get("lines", [])
                    score = h.get("score", 0.0)
                    signals = h.get("signals", [])
                    if lines:
                        label = f"lines {lines[0]}-{lines[-1]}"
                    else:
                        label = f"function {fn}"
                    hotspot_excerpt_lines.append(
                        f"  {fn} ({label}, score={score:.2f})"
                    )
                    if signals:
                        hotspot_excerpt_lines.append(
                            f"  Signals: {', '.join(signals)}"
                        )
                    if lines and source_lines:
                        block = ""
                        for ln in lines:
                            if 1 <= ln <= len(source_lines):
                                block += f"{ln:4d}: {source_lines[ln - 1]}\n"
                        if block:
                            hotspot_excerpt_lines.append(
                                f"```solidity\n{block}```"
                            )

        if hotspot_excerpt_lines:
            _full_for_ref = contract_code[:4000].strip()
            code_block = (
                "\n\nFocused code excerpts (flagged regions):"
                + "".join(hotspot_excerpt_lines)
                + ("\n\nFull contract source (for reference):\n"
                   f"```solidity\n{_full_for_ref}\n```" if _full_for_ref else "")
            )
        else:
            _code_raw = contract_code[:2000].strip()
            ml_windows = ml_result.get("windows_used", 1)
            fallback_note = (
                f"\nNote: ML used {ml_windows} sliding window(s) — "
                "no hotspot data to narrow the excerpt."
            ) if ml_windows > 1 else ""
            code_block = (
                "\n\nContract source (analyse it yourself — the ML signal is only a hint):\n"
                f"```solidity\n{_code_raw}\n```" + fallback_note
            ) if _code_raw else ""
        # ── A.4 Multi-LLM debate (Prosecutor → Defender → Judge) ─────────────
        # DEBATE_MODE (default on) runs three role-specific passes so the final
        # verdict reflects an adversarial exchange rather than one classification
        # call. Any failure raises and is caught below → silent rule-based
        # fallback in synthesizer.
        #
        # Real-audit finding (2026-06-21): an EARLIER version applied
        # CROSS_VALIDATOR_TIMEOUT_S (90s) PER CALL inside _ask(). With 3
        # sequential calls that allowed up to 270s worst case — which blew
        # past the calling script's own timeout (observed: a 200s script
        # timeout killed the process mid-debate, abandoning an
        # asyncio.to_thread() call that keeps running in its OS thread even
        # after cancellation, since to_thread cannot be cancelled).
        # FIX: ONE outer timeout bounds the entire debate (or single-pass
        # call) as a unit — DEBATE_TIMEOUT_S (default 240s) when debate is
        # on, CROSS_VALIDATOR_TIMEOUT_S (default 90s) for single-pass.
        _debate_on = os.getenv("DEBATE_MODE", "on").strip().lower() in ("1", "true", "on", "yes")
        _address = state.get("contract_address", "unknown")

        async def _ask(role: str, system: str, user: str, *, llm_instance=None) -> str:
            # Per-role timing (2026-06-21): each debate role is individually
            # logged so a live run shows exactly which of the 3 sequential
            # calls is slow, rather than only the aggregate debate duration.
            # WS4.1: per-role LLM instance with max_tokens cap.
            _llm = llm_instance or _llm_judge  # fallback for single-pass mode
            with step_timer(f"cross_validator.{role}", address=_address):
                loop = asyncio.get_running_loop()
                fut = loop.run_in_executor(
                    None,
                    _llm.invoke,
                    [SystemMessage(content=system), HumanMessage(content=user)],
                )
                resp = await fut
            return resp.content.strip()

        judge_system = (
            "You are the JUDGE in a smart contract security review. "
            "You have heard the prosecutor (argues vulnerable) and the defender "
            "(argues false-positive). For each vulnerability class below, return a "
            "JSON object mapping class name → verdict string.\n"
            "Verdict options: CONFIRMED | LIKELY | DISPUTED | WATCH | SAFE | INCONCLUSIVE\n"
            "Rules:\n"
            "  CONFIRMED:    ML ≥ 0.55 AND Slither finding(s) agree\n"
            "  LIKELY:       ML ≥ 0.35 AND at least one corroborating signal\n"
            "  DISPUTED:     ML flagged but Slither/RAG contradicts or is absent\n"
            "  WATCH:        ML 0.25–0.34, single weak signal; monitor only\n"
            "  SAFE:         evidence points to false positive\n"
            "  INCONCLUSIVE: debate timed out or could not adjudicate — do not guess\n"
            "Return ONLY valid JSON, no markdown fences, no explanation.\n"
            'Example: {"Reentrancy": "CONFIRMED", "IntegerUO": "LIKELY"}'
        )

        async def _run_debate() -> tuple[str, dict[str, str]]:
            prosecutor = await _ask(
                "prosecutor",
                "You are a security PROSECUTOR. Read the contract source yourself and "
                "argue concisely why it HAS the vulnerabilities below. Ground each "
                "claim in the actual code; cite supporting evidence (Slither detector, "
                "RAG match) where it agrees. Treat the ML probability as a weak hint "
                "only — do not assert a vulnerability the code does not support. "
                "Be concise: at most 3-4 sentences per class.",
                "Vulnerability evidence:\n" + evidence_block + code_block,
                llm_instance=_llm_prosecutor,
            )
            defender = await _ask(
                "defender",
                "You are a skeptical DEFENDER. Read the contract source yourself. "
                "Given the prosecutor's case, argue concisely why these findings may be "
                "false positives or low severity (e.g. typed interface calls, benign "
                "timestamp use, guarded external calls). The ML model is known to "
                "over-predict — challenge claims the code does not justify. "
                "Be concise: at most 3-4 sentences per class.",
                f"Prosecutor's case:\n{prosecutor}\n\nEvidence:\n{evidence_block}{code_block}",
                llm_instance=_llm_defender,
            )
            judge_raw = await _ask(
                "judge",
                judge_system,
                f"Prosecutor:\n{prosecutor}\n\nDefender:\n{defender}\n\n"
                f"Evidence:\n{evidence_block}",
                llm_instance=_llm_judge,
            )
            transcript = {"prosecutor": prosecutor, "defender": defender, "judge": judge_raw}
            return judge_raw, transcript

        # WS4.2 (2026-06-22): selective debate gating. Skip the LLM debate
        # when multiple independent tools (2+ of {ML, Slither, Aderyn}) already
        # agree every flagged class is CONFIRMED by consensus. NEVER skip because
        # cheap signals say "safe" (FN/FP asymmetry principle — Finding #8).
        _skip_debate = False
        if _debate_on and consensus_verdict_state:
            _all_confirmed = True
            for _vuln in all_flagged:
                _cls = _vuln.get("vulnerability_class", "")
                _cv = consensus_verdict_state.get(_cls, {})
                if _cv.get("verdict") != "CONFIRMED":
                    _all_confirmed = False
                    break
                _tool_count = (
                    (_cv.get("ml_signal", 0) or 0)
                    + (_cv.get("slither_match", 0) or 0)
                    + (_cv.get("aderyn_match", 0) or 0)
                )
                if _tool_count < 2:
                    _all_confirmed = False
                    break
            _skip_debate = _all_confirmed

        if _debate_on and not _skip_debate:
            _debate_timeout = get_timeout(ENV_DEBATE_TIMEOUT_S, DEFAULT_DEBATE_TIMEOUT_S)
            with step_timer("cross_validator.debate_total", address=_address, budget_s=_debate_timeout):
                raw, debate_transcript = await asyncio.wait_for(_run_debate(), timeout=_debate_timeout)
            logger.info("cross_validator | debate complete (3 roles)")
        elif _debate_on and _skip_debate:
            logger.info(
                "cross_validator | debate skipped — {} class(es) CONFIRMED by multi-tool consensus",
                len(all_flagged),
            )
            raw = json.dumps({
                v.get("vulnerability_class", ""): "CONFIRMED"
                for v in all_flagged
            })
        else:
            # Legacy single-pass classification (retained for perf/tests).
            _single_pass_timeout = get_timeout(
                ENV_CROSS_VALIDATOR_SINGLE_PASS_TIMEOUT_S,
                DEFAULT_CROSS_VALIDATOR_SINGLE_PASS_TIMEOUT_S,
            )
            raw = await asyncio.wait_for(
                _ask("single_pass", judge_system, "Vulnerability evidence:\n" + evidence_block),
                timeout=_single_pass_timeout,
            )

        raw = raw.strip()
        # Strip markdown fences if the model wraps the JSON anyway
        if raw.startswith("```"):
            parts = raw.split("```")
            raw = parts[1] if len(parts) > 1 else parts[0]
            if raw.lower().startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        llm_verdicts: dict[str, str] = json.loads(raw)

        valid_verdicts = {"CONFIRMED", "LIKELY", "DISPUTED", "WATCH", "SAFE"}
        flagged_class_set = {v.get("vulnerability_class", "") for v in all_flagged}
        verdicts: dict[str, str] = {
            cls: (v if v in valid_verdicts else "DISPUTED")
            for cls, v in llm_verdicts.items()
            if cls in flagged_class_set
        }

        # ── P6 Model Cascade: Strong model re-judges ambiguous cases ──
        # If CASCADE_ENABLED=true, use qwen2.5-coder-7b-instruct (strong model)
        # to re-judge classes with DISPUTED/WATCH verdict or confidence < 0.7.
        # This improves accuracy on ambiguous cases at the cost of latency.
        _cascade_enabled = os.getenv("CASCADE_ENABLED", "false").lower() in ("1", "true", "yes")
        if _cascade_enabled:
            from src.llm.client import get_coder_llm
            _cascade_threshold = float(os.getenv("CASCADE_CONFIDENCE_THRESHOLD", "0.7"))
            _cascade_verdicts = set(os.getenv("CASCADE_VERDICTS", "DISPUTED,WATCH").split(","))

            ambiguous_classes = []
            for cls, verdict in verdicts.items():
                confidence = consensus_verdict_state.get(cls, {}).get("confidence", 1.0)
                if verdict in _cascade_verdicts or confidence < _cascade_threshold:
                    ambiguous_classes.append(cls)

            if ambiguous_classes:
                logger.info("cross_validator | cascade: {} ambiguous class(es) → {}", len(ambiguous_classes), ambiguous_classes)
                _cascade_llm = get_coder_llm()

                for cls in ambiguous_classes:
                    # Build focused prompt with hotspot excerpts
                    cls_evidence = [line for line in class_lines if cls in line]
                    cls_evidence_str = "\n".join(cls_evidence) if cls_evidence else evidence_block

                    cascade_prompt = (
                        f"You are a smart contract security expert. Analyze the evidence and code below, "
                        f"then determine if the contract has a {cls} vulnerability.\n\n"
                        f"Evidence:\n{cls_evidence_str}\n\n"
                        f"Code:\n{code_block}\n\n"
                        f"Answer with a single verdict: CONFIRMED, LIKELY, DISPUTED, WATCH, or SAFE."
                    )

                    try:
                        with step_timer(f"cross_validator.cascade.{cls}", address=_address):
                            loop = asyncio.get_running_loop()
                            fut = loop.run_in_executor(
                                None,
                                _cascade_llm.invoke,
                                [SystemMessage(content="You are a smart contract security expert."),
                                 HumanMessage(content=cascade_prompt)],
                            )
                            cascade_resp = await asyncio.wait_for(fut, timeout=30.0)

                        cascade_verdict = cascade_resp.content.strip().upper()
                        # Extract verdict from response (might have extra text)
                        for v in valid_verdicts:
                            if v in cascade_verdict:
                                cascade_verdict = v
                                break
                        else:
                            cascade_verdict = "DISPUTED"  # Fallback if parsing fails

                        old_verdict = verdicts[cls]
                        verdicts[cls] = cascade_verdict
                        logger.info("cross_validator | cascade {} → {} (was {})", cls, cascade_verdict, old_verdict)
                    except Exception as exc:
                        logger.warning("cross_validator | cascade failed for {}: {} — keeping fast-model verdict", cls, exc)

        logger.info("cross_validator complete | verdicts={}", verdicts)
        # ── P2 Shape A: emit debate evidence (verdicts converted to Evidence objects) ──
        from src.orchestration.verdict.emit import emit_debate_evidence
        evidence_list: list[Any] = emit_debate_evidence(debate_transcript, verdicts)


        result: dict[str, Any] = {
            "evidence_list":  evidence_list,
            "injection_matches": injection_matches,
        }
        if debate_transcript:
            result["debate_transcript"] = debate_transcript
        return result

    except Exception as exc:
        logger.warning(
            "cross_validator | failed (synthesizer will use rule-based fallback): {}", exc
        )
        # WS4.1: preserve the debate transcript even on JSON parse failure —
        # it's valuable for debugging (can see what the 3 roles actually said
        # even when the judge's JSON was malformed/truncated).
        fallback: dict[str, Any] = {}
        if debate_transcript:
            fallback["debate_transcript"] = debate_transcript
        return fallback
