"""
emit.py — Per-source Evidence emission helpers (P2).

Each function converts raw tool output into zero or more Evidence items.
Used during dual-write inside consensus_engine, cross_validator, and synthesizer.
After P2 flip, these become the sole path from channels to verdicts.
"""

from __future__ import annotations

from typing import Any

from src.orchestration.verdict.evidence import Evidence, Polarity, Kind
from src.orchestration.verdict.reliability import get_reliability


def emit_ml_evidence(
    ml_result: dict[str, Any],
) -> list[Evidence]:
    """
    ML model output → Evidence list.
    One Evidence per flagged class (probability ≥ ml_positive_threshold).
    """
    from src.config import get_config

    cfg = get_config()
    threshold = cfg.consensus.ml_positive_threshold

    probabilities: dict[str, float] = ml_result.get("probabilities", {}) or {}
    if not probabilities:
        confirmed = ml_result.get("confirmed", [])
        suspicious = ml_result.get("suspicious", [])
        legacy = ml_result.get("vulnerabilities", [])
        for v in confirmed + suspicious + legacy:
            cls = v.get("vulnerability_class", "")
            prob = v.get("probability", 0.0)
            if cls and prob > 0:
                probabilities[cls] = float(prob)

    evidence: list[Evidence] = []
    for cls, prob in probabilities.items():
        if prob >= threshold:
            tier = ""
            for v in ml_result.get("confirmed", []):
                if v.get("vulnerability_class") == cls:
                    tier = "CONFIRMED"
                    break
            if not tier:
                for v in ml_result.get("suspicious", []):
                    if v.get("vulnerability_class") == cls:
                        tier = "SUSPICIOUS"
                        break
            if not tier and prob >= 0.55:
                tier = "CONFIRMED"
            elif not tier and prob >= 0.25:
                tier = "SUSPICIOUS"

            evidence.append(Evidence.ml(
                cls, prob,
                reliability=get_reliability("ml", cls),
                tier=tier,
                detail={"probability": prob, "tier": tier},
            ))
    return evidence


def emit_static_evidence(
    static_findings: list[dict[str, Any]],
) -> list[Evidence]:
    """
    Slither + Aderyn findings → Evidence list.
    Uses CLASS_TO_DETECTORS to map detector names → vulnerability classes.
    """
    from src.orchestration.routing import CLASS_TO_DETECTORS

    evidence: list[Evidence] = []
    seen: set[tuple[str, str, str]] = set()  # (source, cls, detector) dedup

    for finding in static_findings:
        tool = finding.get("tool", "")
        detector = finding.get("detector", "")
        impact = finding.get("impact", "Low")
        description = finding.get("description", "")
        lines = finding.get("lines", [])

        if tool not in ("slither", "aderyn"):
            continue

        for cls, detectors in CLASS_TO_DETECTORS.items():
            if tool == "slither" and detector.lower() in detectors:
                key = ("slither", cls, detector)
            elif tool == "aderyn":
                tokens = {tok for det in detectors for tok in det.split("-") if len(tok) > 3}
                if any(tok in detector.lower() for tok in tokens):
                    key = ("aderyn", cls, detector)
                else:
                    continue
            else:
                continue

            if key in seen:
                continue
            seen.add(key)

            if tool == "slither":
                evidence.append(Evidence.slither(
                    cls, impact, description,
                    reliability=get_reliability("slither", cls),
                    detector=detector, lines=lines,
                ))
            else:
                evidence.append(Evidence.aderyn(
                    cls, impact, description,
                    reliability=get_reliability("aderyn", cls),
                    detector=detector, lines=lines,
                ))

    return evidence


def emit_rag_evidence(
    rag_results: list[dict[str, Any]],
) -> list[Evidence]:
    """
    RAG results → Evidence list.
    One Evidence per chunk whose similarity exceeds the relevance floor.
    """
    from src.config import get_config

    cfg = get_config()
    floor = cfg.attribution.rag_relevance_floor

    evidence: list[Evidence] = []
    for chunk in rag_results:
        score = float(chunk.get("score", chunk.get("similarity", 0.0)))
        if score < floor:
            continue

        metadata = chunk.get("metadata", {}) or {}
        vt = metadata.get("vulnerability_type", "")
        if not vt:
            continue

        evidence.append(Evidence.rag(
            vt, score,
            reliability=get_reliability("rag", vt),
            chunk_id=chunk.get("id", chunk.get("chunk_id", "")),
            title=metadata.get("title", metadata.get("name", "")),
        ))
    return evidence


def emit_debate_evidence(
    debate_transcript: dict[str, str],
    pre_verdicts: dict[str, str],
) -> list[Evidence]:
    """
    Debate output → Evidence list.
    One Evidence per class the debate adjudicated. deterministic=False.
    """
    evidence: list[Evidence] = []
    for cls, verdict in pre_verdicts.items():
        # Confidence from the judge is not directly extractable from the
        # pre-verdicts dict (it's just the verdict string). We assign a
        # default strength based on the verdict.
        _strength = {
            "CONFIRMED": 0.85, "LIKELY": 0.65, "DISPUTED": 0.40,
            "WATCH": 0.25, "INCONCLUSIVE": 0.30, "SAFE": 0.15,
        }
        strength = _strength.get(verdict, 0.30)

        rationale = ""
        if debate_transcript:
            rationale = debate_transcript.get("judge", "")[:200]

        evidence.append(Evidence.debate(
            cls, verdict, strength,
            judge_rationale=rationale,
        ))
    return evidence


def emit_quick_screen_evidence(
    quick_screen_hits: dict[str, list[str]],
) -> list[Evidence]:
    """
    Quick-screen hits → Evidence list.
    High/Critical Slither/Aderyn findings that forced a deep-path escalation.
    """
    from src.orchestration.routing import CLASS_TO_DETECTORS

    evidence: list[Evidence] = []
    for tool, hits in quick_screen_hits.items():
        for hit in hits:
            for cls, detectors in CLASS_TO_DETECTORS.items():
                if tool == "slither" and hit.lower() in detectors:
                    evidence.append(Evidence.quick_screen(cls, hit, "High"))
                elif tool == "aderyn":
                    tokens = {tok for det in detectors for tok in det.split("-") if len(tok) > 3}
                    if any(tok in hit.lower() for tok in tokens):
                        evidence.append(Evidence.quick_screen(cls, hit, "High"))
    return evidence


def emit_consensus_evidence(
    consensus_verdict: dict[str, dict[str, Any]],
) -> list[Evidence]:
    """
    Consensus engine votes → Evidence list.

    Each class that received a consensus vote becomes an Evidence item with:
    - strength = consensus confidence (the agreed-upon signal)
    - reliability = 0.85 (consensus engine is the trusted aggregator)
    - deterministic = True (pure math, no LLM)
    - polarity = SUPPORTS if verdict is CONFIRMED/LIKELY/DISPUTED, NEUTRAL otherwise

    This ensures fuse() sees every class that legacy _reconcile_verdicts would process,
    including below-ML-threshold classes with tool corroboration.
    """
    evidence: list[Evidence] = []
    for cls, vote in consensus_verdict.items():
        verdict = vote.get("verdict", "SAFE")
        conf = float(vote.get("confidence", 0.0))

        _polarity_map = {
            "CONFIRMED": Polarity.SUPPORTS,
            "LIKELY": Polarity.SUPPORTS,
            "DISPUTED": Polarity.SUPPORTS,  # flagged — must not be silently cleared
            "WATCH": Polarity.NEUTRAL,
            "INCONCLUSIVE": Polarity.NEUTRAL,
            "SAFE": Polarity.NEUTRAL,
        }
        polarity = _polarity_map.get(verdict, Polarity.NEUTRAL)

        # Floor strength for overridden-from-SAFE classes: conf=0.0 gives zero
        # contribution, but the asymmetry principle says they must not be SAFE.
        strength = conf
        if vote.get("overridden_from_safe") and strength < 0.30:
            strength = 0.30

        evidence.append(Evidence(
            source="consensus",
            vuln_class=cls,
            polarity=polarity,
            strength=round(strength, 4),
            reliability=0.85,
            kind=Kind.STATISTICAL,
            deterministic=True,
            detail={
                "consensus_verdict": verdict,
                "consensus_confidence": conf,
                "ml_signal": vote.get("ml_signal", 0),
                "slither_match": vote.get("slither_match", 0),
                "aderyn_match": vote.get("aderyn_match", 0),
            },
        ))
    return evidence
