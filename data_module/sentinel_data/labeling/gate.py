"""Go/No-Go minimum-viable-corpus gate — Task 3.11.

Reads merged .labels.json files and validates per-criterion thresholds
from config.yaml pipeline.min_viable_corpus. Exits with a pass/fail
report. Stage 3 is complete only when this gate passes (or the deferral
decision is documented).

Criteria checked (from config.yaml pipeline.min_viable_corpus):
  1. total_contracts_min        — total merged contracts ≥ threshold
  2. per_class_positive_min_major — Reentrancy, DoS, IntegerUO ≥ threshold
  3. per_class_positive_min_minor — all other 7 classes ≥ threshold
  4. call_to_unknown_min        — CallToUnknown positives; if below, flag for
                                   human review (merger CallToUnknown rule)

Criteria NOT checked here (require Stage 4 data):
  5. smartbugs_curated_recall_min — checked in Stage 4 verification
  6. forge_agreement_min         — checked in Stage 3.12 if FORGE is added
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from sentinel_data.labeling.schema import class_names

# Classes that require the higher "major" threshold
_MAJOR_CLASSES = {"Reentrancy", "DenialOfService", "IntegerUO"}


@dataclass
class GateCriterion:
    name: str
    actual: int | float
    threshold: int | float
    passed: bool
    note: str = ""


@dataclass
class GateResult:
    criteria: list[GateCriterion] = field(default_factory=list)
    gate_passed: bool = False
    call_to_unknown_review_needed: bool = False

    def __str__(self) -> str:
        lines = ["── Go/No-Go Gate Report ──────────────────────────"]
        for c in self.criteria:
            icon = "✓" if c.passed else "✗"
            lines.append(f"  {icon} {c.name}: {c.actual} (threshold={c.threshold}){' — ' + c.note if c.note else ''}")
        lines.append("")
        if self.call_to_unknown_review_needed:
            lines.append("  ⚠ CallToUnknown < threshold — human review required before merge")
        lines.append(f"  {'PASS ✓' if self.gate_passed else 'FAIL ✗ — see criteria above'}")
        lines.append("─────────────────────────────────────────────────")
        return "\n".join(lines)


def run_gate(data_dir: Path, cfg: dict) -> GateResult:
    """Run the minimum-viable-corpus gate against merged labels.

    Args:
        data_dir: Path to data/ directory.
        cfg: Full config dict (from config.yaml).

    Returns:
        GateResult with per-criterion pass/fail and overall verdict.
    """
    mvc = cfg.get("pipeline", {}).get("min_viable_corpus", {})
    total_min:   int = mvc.get("total_contracts_min", 4000)
    major_min:   int = mvc.get("per_class_positive_min_major", 300)
    minor_min:   int = mvc.get("per_class_positive_min_minor", 100)
    ctu_min:     int = mvc.get("call_to_unknown_min", 300)

    merged_dir = data_dir / "labels" / "merged"
    if not merged_dir.exists() or not any(merged_dir.glob("*.labels.json")):
        # Nothing merged yet — everything fails
        all_classes = class_names()
        result = GateResult()
        result.criteria.append(GateCriterion(
            "total_contracts", 0, total_min, False,
            "merged labels dir empty — run `sentinel-data label` first"
        ))
        result.gate_passed = False
        return result

    # Count positives per class across all merged files
    per_class: dict[str, int] = {cls: 0 for cls in class_names()}
    total = 0
    for lf in merged_dir.glob("*.labels.json"):
        try:
            lj = json.loads(lf.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        total += 1
        for cls, entry in lj["classes"].items():
            if entry["value"] == 1:
                per_class[cls] += 1

    result = GateResult()

    # Criterion 1: total contracts
    result.criteria.append(GateCriterion(
        "total_contracts", total, total_min, total >= total_min
    ))

    # Criteria 2+3: per-class positives
    for cls in class_names():
        count = per_class[cls]
        if cls in _MAJOR_CLASSES:
            threshold = major_min
        else:
            threshold = minor_min
        result.criteria.append(GateCriterion(
            f"class_{cls}", count, threshold, count >= threshold
        ))

    # CallToUnknown human-review flag (separate from pass/fail)
    ctu_count = per_class.get("CallToUnknown", 0)
    if ctu_count < ctu_min:
        result.call_to_unknown_review_needed = True
        # Find the CallToUnknown criterion and annotate it
        for c in result.criteria:
            if c.name == "class_CallToUnknown":
                c.note = "below threshold — human review: merge into ExternalBug?"

    # Gate passes if total + major classes all pass.
    # Minor class failures are reported but don't block (warn only).
    blocking = [c for c in result.criteria
                if not c.passed and
                (c.name == "total_contracts" or
                 any(c.name == f"class_{m}" for m in _MAJOR_CLASSES))]
    result.gate_passed = len(blocking) == 0

    return result
