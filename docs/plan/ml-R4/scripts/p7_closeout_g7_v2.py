#!/usr/bin/env python3
"""Robust Phase-7 G7 closeout wrapper.

The original closeout owns the evidence/governance logic. This wrapper replaces
only the active-document updater so Markdown wording changes cannot break G7
closure through brittle regex matching.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE_PATH = HERE / "p7_closeout_g7.py"
spec = importlib.util.spec_from_file_location("sentinel_p7_closeout_g7", BASE_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load {BASE_PATH}")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)


def replace_section(body: str, start_header: str, end_header: str, replacement: str) -> str:
    start = body.find(start_header)
    if start < 0:
        raise RuntimeError(f"missing section start: {start_header!r}")
    end = body.find(end_header, start + len(start_header))
    if end < 0:
        raise RuntimeError(f"missing section end: {end_header!r}")
    return body[:start] + replacement.rstrip() + "\n\n" + body[end:]


def replace_required(body: str, old: str, new: str, label: str) -> str:
    if new in body:
        return body
    if old not in body:
        raise RuntimeError(f"missing expected active-doc text for {label}: {old[:120]!r}")
    return body.replace(old, new, 1)


def update_active_docs() -> None:
    # Root README: move canonical project state from G6/pending-G7 to G7/Phase8-ready.
    root_readme = base.REPO / "README.md"
    body = base.text(root_readme)
    body = replace_required(
        body,
        "The current stable `main` baseline is the R4 **G6-passed** state from merge commit `91f795885` plus later documentation-only reconciliation.",
        "The current stable `main` baseline includes the R4 **G7-passed DATA vNext v2 implementation**, merged at `81d9c547d`, on top of the V3/runtime and canonical-documentation baseline.",
        "root README baseline",
    )
    body = replace_required(
        body,
        "Phase 7 implements the additive DATA vNext v2 semantic overlay on branch `r4/phase7-data-vnext-implementation`. Its remote semantic checks are green, but G7 is not complete until the existing local graph/token representations are physically bound and validated. Until G7 is merged, the v2 implementation branch is candidate work rather than canonical `main` runtime.",
        "Phase 7 is complete: DATA vNext v2 is canonical, its semantic overlay is deterministic, and the real local representation population was physically bound and validated for G7. Phase 8 is now the next authorized R4 step; no repaired teacher has been retrained yet.",
        "root README Phase7 state",
    )
    base.write(root_readme, body)

    # DATA entry point.
    data_readme = base.REPO / "data_module/README.md"
    body = base.text(data_readme)
    body = replace_required(body, "Stable `main` has passed R4 **G6**:", "Stable `main` has passed R4 **G7**:", "DATA README gate")
    body = replace_required(
        body,
        "Phase 7 DATA vNext implementation is active on `r4/phase7-data-vnext-implementation`. Remote semantic generation/validation is green, but G7 still requires local binding to the existing 21,657 physical representation triplets before merge.",
        "Phase 7 DATA vNext implementation is complete and canonical. The local G7 gate bound all 21,657 required representation contracts (64,971 graph/token/sidecar files) with zero missing files and zero mismatches; the committed v2 manifest is representation-bound and Phase 8 is authorized.",
        "DATA README Phase7 state",
    )
    body = body.replace("vnext/                     DATA vNext v2 implementation (Phase 7 branch until G7)", "vnext/                     canonical DATA vNext v2 implementation (G7-passed)")
    body = body.replace("On canonical `main` through G6:", "On canonical `main` through G7:")
    body = body.replace(
        "Phase-7 build/local commands belong to the dedicated Phase-7 branch until G7 is complete.",
        "The representation-bound v2 publication is now the canonical Phase-8 training-data lineage; use the committed vNext validator/loader and do not regenerate roles or fall back to historical v1 semantics.",
    )
    if "Stable `main` has passed R4 **G7**:" not in body or "64,971" not in body:
        raise RuntimeError("DATA README G7 update incomplete")
    base.write(data_readme, body)

    # DATA artifacts handbook chapter.
    data_artifacts = base.REPO / "docs/handbook/04_data_artifacts.md"
    body = base.text(data_artifacts)
    body = replace_required(
        body,
        "Remote semantic generation is deterministic. Final G7 requires local binding to the existing 21,657 represented contracts before the v2 candidate can be promoted/merged.",
        "Semantic generation is deterministic, and G7 is complete: the local gate bound all 21,657 required represented contracts / 64,971 graph-token-sidecar files with zero missing files or mismatches. The committed v2 manifest is representation-bound and Phase 8 may consume it.",
        "DATA artifacts G7 binding",
    )
    body = body.replace(
        "- Phase-7 v2 is not canonical main until G7 local representation binding passes and the branch merges.",
        "- DATA vNext v2 is canonical post-G7; Phase 8 must consume its exact manifest/roles/masks rather than rebuilding semantics implicitly.",
    )
    body = body.replace(
        "After G7 merges, use the committed vNext CLI/validator for v2 artifact verification.",
        "Use the committed vNext CLI/validator for v2 artifact verification; representation-required validation is now part of the G7 lineage.",
    )
    body = body.replace(
        "For repaired semantics, read R4 policy/schema/partition artifacts; after G7, the additive `data_module/sentinel_data/vnext` package is the v2 implementation source.",
        "For repaired semantics, read R4 policy/schema/partition artifacts and the canonical `data_module/sentinel_data/vnext` package, which is the G7-passed v2 implementation source.",
    )
    if "G7 is complete" not in body:
        raise RuntimeError("DATA artifacts G7 update incomplete")
    base.write(data_artifacts, body)

    # Current-status page: replace entire volatile sections by Markdown boundaries.
    current = base.REPO / "docs/handbook/16_current_status.md"
    body = base.text(current)
    summary = f"""## 30-second summary

The canonical post-G7 baseline includes DATA vNext implementation merge **`81d9c547d`** (2026-08-12): R4 DATA/ML repair has passed **G0 through G7**, the V3 registry/context protocol and read-only audit-MCP boundary remain implemented, and Run12 remains the historical operational teacher. The v2 semantic overlay is now physically bound to all {base.EXPECTED_REPRESENTED:,} required representations / {base.EXPECTED_FILES:,} graph-token-sidecar files with zero missing files and zero mismatches. **Phase 8 retraining is READY**, but no repaired teacher checkpoint has been trained or promoted yet.

The evidence limitations remain explicit: no confirmed-negative source exists in policy v1, threshold/calibration roles are unsupported/empty, and untouched acceptance is unsupported/empty/frozen. Historical July suite totals remain historical evidence rather than current-state proof."""
    body = replace_section(body, "## 30-second summary", "## Just-enough mental model", summary)

    phase7 = f"""### Phase 7 G7 state

DATA vNext v2 is canonical through implementation merge `{base.IMPLEMENTATION_MERGE[:9]}` and the G7 closeout records the locally bound publication.

Final G7 evidence:

- manifest status: `VALIDATED_G7_CANDIDATE`;
- contracts: {base.EXPECTED_CONTRACTS:,};
- contract×class rows: {base.EXPECTED_ROWS:,};
- positive targets: 1,007; negative targets: 0;
- STRONG signals: 403; WEAK signals: 604;
- effective loss cells: 852; outcome-metric cells: 118;
- required/checked representation contracts: {base.EXPECTED_REPRESENTED:,}/{base.EXPECTED_REPRESENTED:,};
- required/checked physical files: {base.EXPECTED_FILES:,}/{base.EXPECTED_FILES:,};
- missing files: 0; representation mismatches: 0;
- physical local path recorded: false;
- representation binding digest: `{base.EXPECTED_BINDING_DIGEST}`.

The v2 loader rejects silent historical-v1 fallback. Historical v1 artifacts remain immutable. Phase 8 may adapt the frozen training consumer to this exact lineage; it may not invent negatives, rebalance frozen roles, or manufacture unsupported threshold/calibration/acceptance populations."""
    body = replace_section(body, "### Phase 7 candidate state", "### Current ML state", phase7)

    # Stable point replacements elsewhere on the status page.
    body = body.replace("R4 G0–G6 PASS", "R4 G0–G7 PASS")
    body = body.replace(
        "Phase 7 branch\n  deterministic DATA vNext v2 semantic overlay\n  remote semantic checks PASS\n  local representation binding PENDING\n        ↓\nG7\n        ↓\nPhase 8 retrain existing architecture",
        "DATA vNext v2\n  deterministic semantic overlay PASS\n  local representation binding PASS\n  G7 PASS\n        ↓\nPhase 8 retrain existing architecture (READY)",
    )
    body = body.replace(
        "| 7 | pending G7 | implementation branch exists; local physical representation binding still required |",
        "| 7 | G7 PASS | v2 implementation merged; 21,657 representations / 64,971 files physically bound with zero mismatches |",
    )
    body = body.replace(
        "| 8–10 | waiting | retraining/evaluation/promotion not authorized until preceding gates |",
        "| 8 | READY | existing-architecture retraining authorized against the exact G7-passed v2 lineage |\n| 9–10 | waiting | evaluation/promotion remain gated by preceding phases |",
    )
    body = body.replace(
        "- Phase 7 still requires local physical representation binding;",
        "- Phase 7 physical representation binding passed; the remaining DATA/ML limitations are evidence limitations, not G7 implementation blockers;",
    )
    body = body.replace(
        "For Phase 7, use the active branch’s `p7_run_local_gate.py` command when the local representation root is available.",
        "For G7 evidence, use the committed vNext manifest, representation-binding report, and final G7 validation report. The local gate remains available for reproducibility, not because G7 is pending.",
    )
    body = body.replace(
        "For Phase 7, inspect its branch and gate scripts rather than assuming candidate code is already canonical main.",
        "For DATA vNext, inspect the canonical vNext package, G7 manifest/reports, and R4 decisions rather than historical v1 label/export assumptions.",
    )
    body = body.replace(
        "Today a correct statement is: “R4 G6 is canonical; Phase 7 remote semantics are green but local representation binding is pending; Run12 is historical operational inference; no retrained vNext teacher or untouched-acceptance claim exists.”",
        "Today a correct statement is: “R4 G7 is canonical; DATA vNext v2 is representation-bound; Phase 8 retraining is ready; Run12 remains historical operational inference; no retrained vNext teacher or untouched-acceptance claim exists.”",
    )
    body = body.replace(
        "Can you distinguish canonical main from Phase-7 candidate state, state all unsupported evaluation roles, identify the current teacher/proof/MCP protocol versions, and name the exact local blocker before G7?",
        "Can you identify the exact G7 DATA vNext manifest/binding lineage, state all unsupported evaluation roles, distinguish Run12 from the future repaired teacher, and explain what Phase 8 is and is not authorized to change?",
    )
    if "G0 through G7" not in body or "### Phase 7 G7 state" not in body or "| 8 | READY |" not in body:
        raise RuntimeError("current-status G7 update incomplete")
    base.write(current, body)


base.update_active_docs = update_active_docs

if __name__ == "__main__":
    raise SystemExit(base.main())
