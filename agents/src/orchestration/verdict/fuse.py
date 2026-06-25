"""
fuse.py — Single verdict fusion function (proposal §5.2).

Takes a flat list of Evidence items, groups by class, de-correlates within
witness families, aggregates signed confidence, applies the FN/FP asymmetry
rule, maps confidence to a verdict band, and emits a dual verdict:
  - verdict_provable: computed over deterministic=True evidence only (ZK-anchored)
  - verdict_full:     computed over all evidence (human report)

This one function replaces consensus_engine's vote + the 8-case _reconcile_verdicts.
"""

from __future__ import annotations

from loguru import logger

from src.orchestration.verdict.evidence import Evidence, Polarity, Kind
from src.orchestration.verdict.verdict import ClassVerdict


# ── Witness families (proposal §5.2 step 2) ─────────────────────────────────
# Sources within the same family are correlated — their combined weight is
# discounted by 1/N so they don't multiply-count.
FAMILIES: dict[str, str] = {
    "ml":           "ML",
    "slither":      "STATIC_SYNTAX",
    "aderyn":       "STATIC_SYNTAX",
    "quick_screen": "STATIC_SYNTAX",
    "rag":          "RAG",
    "debate":       "LLM_DEBATE",
    "halmos":       "FORMAL",
    "z3":           "FORMAL",
    "gigahorse":    "FORMAL",
    "taint":        "FORMAL",
    "access_control": "FORMAL",
    "ityfuzz":      "ECONOMIC",
    "anvil":        "ECONOMIC",
}


# ── Strong-SUPPORTS definition (proposal §5.2 step 4, tunable) ──────────────
# A class with ANY of the following cannot be cleared to SAFE:
STRONG_SUPPORTS_RELIABILITY_STRENGTH = 0.5   # one SUPPORTS with rel×strength ≥ this
STRONG_SUPPORTS_MIN_STRENGTH = 0.3            # two+ SUPPORTS each with strength ≥ this
# FORMAL evidence is always "strong" (a proven invariant violation).


def _get_bands():
    """Lazy-read verdict band cutoffs from config (Rule B)."""
    from src.config import get_config
    cfg = get_config()
    return (
        cfg.consensus.confirmed_band,
        cfg.consensus.likely_band,
        cfg.consensus.disputed_band,
    )


def _band_label(confidence: float) -> str:
    confirmed, likely, disputed = _get_bands()
    if confidence >= confirmed:
        return "CONFIRMED"
    if confidence >= likely:
        return "LIKELY"
    if confidence >= disputed:
        return "DISPUTED"
    return "SAFE"


def _is_strong_supports(evidence_items: list[Evidence]) -> bool:
    """True if there exists at least one strong SUPPORTS signal."""
    supports = [e for e in evidence_items if e.polarity == Polarity.SUPPORTS]
    if not supports:
        return False

    # (a) One SUPPORTS with reliability × strength ≥ 0.5
    if any(e.reliability * e.strength >= STRONG_SUPPORTS_RELIABILITY_STRENGTH
           for e in supports):
        return True

    # (b) Two or more SUPPORTS each with strength ≥ 0.3
    strong_enough = [e for e in supports if e.strength >= STRONG_SUPPORTS_MIN_STRENGTH]
    if len(strong_enough) >= 2:
        return True

    # (c) One SUPPORTS with kind = FORMAL
    if any(e.kind == Kind.FORMAL for e in supports):
        return True

    return False


def _fuse_for_evidence(evidence: list[Evidence]) -> tuple[str, float, list[Evidence]]:
    """
    Core fusion for a single class's evidence list.

    Returns (verdict_label, confidence, driving_evidence).
    """
    if not evidence:
        return "SAFE", 0.0, []

    # ── Step 2: de-correlate by family ──────────────────────────────────
    # Group by family, count per family, scale each source's reliability by 1/N.
    family_counts: dict[str, int] = {}
    for e in evidence:
        family = FAMILIES.get(e.source, e.source)
        family_counts[family] = family_counts.get(family, 0) + 1

    # ── Step 3: aggregate signed reliability × strength ─────────────────
    positive_mass = 0.0
    driving: list[Evidence] = []

    for e in evidence:
        family = FAMILIES.get(e.source, e.source)
        discount = 1.0 / max(family_counts.get(family, 1), 1)
        discounted_rel = e.reliability * discount

        if e.polarity == Polarity.SUPPORTS:
            positive_mass += discounted_rel * e.strength
        elif e.polarity == Polarity.REFUTES:
            positive_mass -= discounted_rel * e.strength
        # NEUTRAL: contributes 0.0 to the signed sum

        driving.append(e)

    # Confidence = sum of signed (reliability × strength) contributions, clamped to [0,1].
    # Each piece of evidence adds/subtracts its discounted reliability × strength.
    # Weak signal from an unreliable source → low confidence; strong corroboration → high.
    confidence = max(0.0, min(1.0, positive_mass))

    # ── Step 5: map confidence to band ──────────────────────────────────
    verdict = _band_label(confidence)

    # ── Step 4: FN/FP asymmetry override ────────────────────────────────
    # A REFUTES can never clear a strong SUPPORTS — floor is DISPUTED.
    if _is_strong_supports(evidence) and verdict == "SAFE":
        verdict = "DISPUTED"

    return verdict, confidence, driving


def fuse(evidence: list[Evidence]) -> dict[str, ClassVerdict]:
    """
    Fuse all evidence into per-class verdicts.

    Args:
        evidence: Flat list of Evidence items across all classes and sources.

    Returns:
        dict[class_name, ClassVerdict] — one entry per class that has evidence.
        Classes with no evidence are not present in the output.
    """
    # ── Step 1: group by class ──────────────────────────────────────────
    by_class: dict[str, list[Evidence]] = {}
    for e in evidence:
        by_class.setdefault(e.vuln_class, []).append(e)

    # ── Step 6: dual emit per class ─────────────────────────────────────
    results: dict[str, ClassVerdict] = {}
    for cls, items in sorted(by_class.items()):
        # Provable tier: deterministic evidence only
        det_items = [e for e in items if e.deterministic]
        verdict_provable, conf_provable, _ = _fuse_for_evidence(det_items)

        # Full tier: all evidence
        verdict_full, conf_full, driving = _fuse_for_evidence(items)

        results[cls] = ClassVerdict(
            cls=cls,
            verdict_provable=verdict_provable,
            verdict_full=verdict_full,
            confidence=round(conf_full, 4),
            driving_evidence=driving,
        )

    logger.debug("fuse | {} classes from {} evidence items", len(results), len(evidence))
    return results
