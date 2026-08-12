#!/usr/bin/env python3
"""Deterministically close R4 Phase 7 after local + PR G7 validation."""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
R4 = REPO / "docs/plan/ml-R4"
OUT = REPO / "data_module/data/exports/sentinel-r4-vnext-v1"

IMPLEMENTATION_MERGE = "81d9c547d3610e2cfb12a5927a7a78b5693430c2"
LOCAL_G7_EVIDENCE_COMMIT = "5bd9c19eb46cd804b34ac0c2cd598767f10c7fad"
EXPECTED_BINDING_DIGEST = "7637461f6643d398c7a0446412fedd8877914c7b9ed41309dab45f18ed96f420"
EXPECTED_CONTRACTS = 22493
EXPECTED_ROWS = 224930
EXPECTED_REPRESENTED = 21657
EXPECTED_FILES = 64971
EXPECTED_EXCLUDED = 836
EXPECTED_TARGETS = {"1": 1007, "None": 223923}
EXPECTED_STRENGTH = {"NONE": 223923, "STRONG": 403, "WEAK": 604}

ARTIFACT_MARKER = "## Phase 7 artifacts — DATA vNext implementation and local representation binding (G7 PASS)"
LOG_MARKER = "### R4-LOG-20260812-010 — Phase 7 DATA vNext Implementation and G7 Closure"
ADR_PATH = R4 / "adrs/ADR-R4-007-data-vnext-implementation-and-g7-publication.md"
FINDING_PATH = R4 / "findings/09_phase7_data_vnext_implementation_and_g7.md"


def text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def load_json(path: Path) -> dict:
    return json.loads(text(path))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()


def replace_once(path: Path, old: str, new: str) -> None:
    body = text(path)
    if new in body:
        return
    if old not in body:
        raise RuntimeError(f"expected text not found in {path}: {old[:120]!r}")
    write(path, body.replace(old, new, 1))


def replace_regex_once(path: Path, pattern: str, replacement: str) -> None:
    body = text(path)
    updated, count = re.subn(pattern, replacement, body, count=1, flags=re.DOTALL)
    if count != 1:
        raise RuntimeError(f"regex replacement failed in {path}: {pattern[:100]!r}; count={count}")
    write(path, updated)


def validate_g7_inputs() -> dict:
    manifest = load_json(OUT / "manifest.json")
    rep = load_json(OUT / "representation_binding_report.json")
    g7 = load_json(OUT / "g7_validation_report.json")
    semantic = load_json(OUT / "validation_report.json")

    assert manifest["status"] == "VALIDATED_G7_CANDIDATE"
    assert manifest["dataset_version"] == "sentinel-r4-vnext-v1"
    assert manifest["export_schema_version"] == "v2"
    assert manifest["historical_artifacts_mutated"] is False
    assert manifest["population"] == {
        "contracts": EXPECTED_CONTRACTS,
        "contract_class_rows": EXPECTED_ROWS,
        "excluded_contracts": EXPECTED_EXCLUDED,
        "representation_required_contracts": EXPECTED_REPRESENTED,
    }
    assert manifest["semantic_counts"]["target_value"] == EXPECTED_TARGETS
    assert manifest["semantic_counts"]["training_strength"] == EXPECTED_STRENGTH
    assert manifest["unsupported_roles"] == {
        "CALIBRATION_FIT": "UNSUPPORTED_EMPTY",
        "THRESHOLD_FIT": "UNSUPPORTED_EMPTY",
        "UNTOUCHED_ACCEPTANCE": "UNSUPPORTED_EMPTY_FROZEN",
    }

    assert rep["status"] == "VALIDATED_LOCAL_G7"
    assert rep["passed"] is True
    assert rep["required_contracts"] == EXPECTED_REPRESENTED
    assert rep["checked_contracts"] == EXPECTED_REPRESENTED
    assert rep["expected_files"] == EXPECTED_FILES
    assert rep["checked_files"] == EXPECTED_FILES
    assert rep["missing_files_total"] == 0
    assert rep["mismatch_total"] == 0
    assert rep["binding_digest_sha256"] == EXPECTED_BINDING_DIGEST
    assert rep["physical_root_recorded"] is False
    assert rep["representation_root"] == "data_module/data/representations"

    assert g7["passed"] is True
    assert g7["require_representation_binding"] is True
    assert g7["contracts"] == EXPECTED_CONTRACTS
    assert g7["contract_class_rows"] == EXPECTED_ROWS
    assert g7["target_counts"] == EXPECTED_TARGETS
    assert g7["training_strength_counts"] == EXPECTED_STRENGTH
    assert g7["errors"] == []
    assert semantic["passed"] is True

    combined = "\n".join(text(p) for p in (OUT / "manifest.json", OUT / "representation_binding_report.json", OUT / "g7_validation_report.json"))
    assert "/home/" not in combined and "C:\\" not in combined

    return {"manifest": manifest, "rep": rep, "g7": g7}


def create_adr() -> None:
    if ADR_PATH.exists():
        return
    write(ADR_PATH, f"""# ADR-R4-007 — DATA vNext v2 Implementation and G7 Publication Acceptance

**Status:** Accepted  
**Date:** 2026-08-12  
**Deciders:** Ali Rajabi (routine technical/governance approval delegated), GPT-5.6 Sol  
**Scope:** R4 Phase-7 implementation acceptance and Phase-8 training input authority

## Context

R4 Phases 0–6 reconstructed the historical label defect, froze `data-vnext-policy-v1`, and assigned leakage-safe roles in `r4-vnext-roles-v1`. Phase 7 then implemented the approved semantics as an additive v2 overlay rather than rewriting historical v1 artifacts or duplicating the existing graph/token tensors.

Remote CI proved deterministic semantic generation. The required local gate then physically verified all **{EXPECTED_REPRESENTED:,}** non-excluded representation triplets (**{EXPECTED_FILES:,} files**) from the real protected representation tree with zero missing files and zero mismatches.

## Decision

Accept `sentinel-r4-vnext-v1` as the **G7-passed DATA vNext implementation** and the only authorized DATA input lineage for the first Phase-8 repaired-model retrain.

The accepted bundle includes:

- `label_states.parquet` — canonical 224,930-row contract×class semantic state;
- `ml_targets.parquet` — derived per-contract ten-class target/strength/mask/role projection;
- source/crosswalk/evidence/representation registries;
- deterministic semantic validation report;
- local physical representation-binding report;
- final representation-required G7 validation report;
- v2 format schema and fail-closed loader/validator/publication code.

The physical representation binding digest is:

`{EXPECTED_BINDING_DIGEST}`

The accepted semantics remain intentionally asymmetric:

- positive targets: **1,007**;
- confirmed-negative targets: **0**;
- STRONG signals: **403**;
- WEAK signals: **604**;
- GasException and UnusedReturn supervision remain disabled;
- threshold fit and calibration fit remain unsupported/empty;
- untouched acceptance remains unsupported/empty/frozen.

## Phase-8 authority

Phase 8 may add only the compatibility required to train the frozen four-eye architecture from this exact v2 lineage. It may not:

- silently rebuild/rebalance Phase-6 roles;
- reinterpret unknown/masked cells as negatives;
- silently fall back to historical v1 labels;
- change class order or graph schema;
- manufacture threshold/calibration/acceptance populations;
- treat a different DATA export as equivalent without a new versioned decision.

Any numeric weak-loss weight is a Phase-8 training-config decision and must be checkpoint-bound.

## Historical compatibility and rollback

Historical v1 artifacts remain immutable and reproducible as historical evidence. Rollback is selection of a prior hash-bound compatible bundle, never reverse mutation of v2 or v1 files.

## Evidence

- implementation merge: `{IMPLEMENTATION_MERGE}`;
- local G7 evidence commit: `{LOCAL_G7_EVIDENCE_COMMIT}`;
- local representation digest: `{EXPECTED_BINDING_DIGEST}`;
- branch + PR G7 workflows: PASS;
- historical G3–G6 regression gates: PASS on the integration tree.
""")


def create_finding(info: dict) -> None:
    if FINDING_PATH.exists():
        return
    rep = info["rep"]
    g7 = info["g7"]
    write(FINDING_PATH, f"""# 09 — Phase 7 DATA vNext Implementation and G7 Result

- **Phase:** R4 Phase 7 — DATA vNext Implementation
- **Gate:** G7 PASS
- **Dataset:** `sentinel-r4-vnext-v1`
- **Export schema:** `v2`
- **Graph schema:** `v9`
- **Implementation merge:** `{IMPLEMENTATION_MERGE}`
- **Local G7 evidence commit:** `{LOCAL_G7_EVIDENCE_COMMIT}`

## Result

DATA vNext v2 is now implemented as an additive semantic overlay over the existing representation lineage. Historical v1 artifacts and the graph/token representation bytes were not rewritten.

Final semantic population:

| Measure | Result |
|---|---:|
| contracts | {EXPECTED_CONTRACTS:,} |
| contract×class rows | {EXPECTED_ROWS:,} |
| represented / physically bound contracts | {EXPECTED_REPRESENTED:,} |
| excluded incomplete-representation contracts | {EXPECTED_EXCLUDED:,} |
| positive targets | {g7['target_counts']['1']:,} |
| negative targets | 0 |
| STRONG rows | {g7['training_strength_counts']['STRONG']:,} |
| WEAK rows | {g7['training_strength_counts']['WEAK']:,} |
| effective loss cells | {g7['effective_loss_cells']:,} |
| outcome-metric cells | {g7['outcome_metric_cells']:,} |

## Local representation binding

The protected/local representation tree was verified without DVC fetching and without recording the physical local filesystem path.

- required contracts: {rep['required_contracts']:,}
- checked contracts: {rep['checked_contracts']:,}
- expected files: {rep['expected_files']:,}
- checked files: {rep['checked_files']:,}
- missing files: {rep['missing_files_total']}
- mismatches: {rep['mismatch_total']}
- extractor: `v2.1-windowed-gcb`
- graph schema: `v9`
- physical path recorded: `false`
- binding digest: `{rep['binding_digest_sha256']}`

## Frozen limitations carried forward

G7 does **not** solve evidence that does not exist:

- no confirmed-negative training population exists in policy v1;
- GasException and UnusedReturn remain supervision-disabled;
- MODEL_SELECTION remains positive-only limited;
- THRESHOLD_FIT remains unsupported/empty;
- CALIBRATION_FIT remains unsupported/empty;
- UNTOUCHED_ACCEPTANCE remains unsupported/empty/frozen.

These limitations are inputs to Phase 8/9, not implementation defects to patch away.

## G7 assessment

**G7 PASS.** The versioned v2 bundle reproduces from frozen semantic inputs, validates independently, physically binds all required representations, preserves historical v1, and is suitable for the approved Phase-6 training roles.

Phase 8 may now adapt the existing training consumer to the exact v2 target/strength/mask/role contract and retrain the unchanged four-eye architecture.
""")


def update_r4_governance() -> None:
    status = R4 / "PLAN_STATUS_MATRIX.md"
    replace_once(
        status,
        "| 7 | `phases/08_PHASE_7_DATA_VNEXT_IMPLEMENTATION.md` | READY | G6 | G7 | Additive v2 implementation complete; local representation binding and branch/PR G7 validation passed; integration/merge closeout pending |",
        "| 7 | `phases/08_PHASE_7_DATA_VNEXT_IMPLEMENTATION.md` | PASSED | G6 | G7 | sentinel-r4-vnext-v1 implemented and locally bound to 21,657 representations / 64,971 files with zero mismatches; G7 PASS |",
    )
    replace_once(
        status,
        "| 8 | `phases/09_PHASE_8_EXISTING_MODEL_RETRAINING.md` | WAITING | G7 | G8 | Architecture frozen |",
        "| 8 | `phases/09_PHASE_8_EXISTING_MODEL_RETRAINING.md` | READY | G7 | G8 | Retrain the frozen four-eye architecture using the exact G7-passed v2 lineage; no historical-v1 fallback or unsupported evaluation roles |",
    )

    p7 = R4 / "phases/08_PHASE_7_DATA_VNEXT_IMPLEMENTATION.md"
    replace_once(p7, "**Status:** IN_PROGRESS — G6 SATISFIED", "**Status:** PASSED — G7 PASS")
    if "## G7 closeout" not in text(p7):
        write(p7, text(p7).rstrip() + f"""

## G7 closeout

The additive v2 implementation is merged and locally representation-bound.

- dataset: `sentinel-r4-vnext-v1`
- contracts: {EXPECTED_CONTRACTS:,}
- contract×class rows: {EXPECTED_ROWS:,}
- required/checked representations: {EXPECTED_REPRESENTED:,}
- required/checked physical files: {EXPECTED_FILES:,}
- missing files: 0
- mismatches: 0
- representation digest: `{EXPECTED_BINDING_DIGEST}`
- manifest state: `VALIDATED_G7_CANDIDATE`
- final representation-required validation: PASS

Historical v1 artifacts were not mutated. Threshold/calibration/untouched-acceptance roles remain intentionally unsupported/empty.

**G7 PASS.** Phase 8 is authorized to retrain the existing frozen architecture using this exact v2 lineage.
""" + "\n")

    p8 = R4 / "phases/09_PHASE_8_EXISTING_MODEL_RETRAINING.md"
    replace_once(p8, "**Status:** WAITING FOR G7", "**Status:** READY — G7 SATISFIED")
    if "## Phase-7 handoff" not in text(p8):
        write(p8, text(p8).rstrip() + f"""

## Phase-7 handoff

Training input authority is `sentinel-r4-vnext-v1` with local representation binding digest `{EXPECTED_BINDING_DIGEST}`. Phase 8 must preserve class order/schema, Phase-6 roles, nullable target semantics, STRONG/WEAK distinction, disabled-class masking, and the unsupported threshold/calibration/acceptance boundaries.
""" + "\n")

    decisions = R4 / "DECISION_REGISTER.md"
    body = text(decisions)
    if "| R4-D-007 |" not in body:
        row = (
            "| R4-D-007 | 2026-08-12 | ACCEPTED | DATA vNext implementation / G7 | "
            "Accept `sentinel-r4-vnext-v1` as the G7-passed v2 semantic/representation-bound lineage and sole Phase-8 training-data authority; historical v1 remains immutable and unsupported evaluation roles remain empty | "
            "ADR-R4-007; G7 local binding + final validation; implementation merge `81d9c547d` | "
            "Phase 8 consumes exact manifest/roles/masks and binds any training-weight choices to checkpoint config | "
            "select prior hash-bound compatible bundle; never rewrite v1/v2 in place | delegated technical owner / GPT-5.6 Sol |\n"
        )
        body = body.replace("\n## Decisions requiring ADR\n", "\n" + row + "\n## Decisions requiring ADR\n", 1)
        write(decisions, body)

    risks = R4 / "RISK_AND_BLOCKER_REGISTER.md"
    replace_once(
        risks,
        "| R4-R014 | Risk | Nominal DATA label orchestration is not reproducible: `dvc.yaml` invokes `sentinel-data label`, but current `_run_label()` is a no-op placeholder while lower-level parsers exist. | Medium | Phase 7 must implement a deterministic versioned vNext build path before G7. | OPEN | G7 |",
        "| R4-R014 | Risk | Historical nominal DATA label orchestration is incomplete: `sentinel-data label` remains a legacy no-op seam. | Medium | Phase 7 added a separate deterministic, versioned `sentinel_data.vnext` build/validate/load/publication path; historical `_run_label()` remains compatibility material and is not used to create vNext truth. | MITIGATED | G7 |",
    )


def register_artifacts() -> None:
    index = R4 / "ARTIFACT_INDEX.md"
    body = text(index)
    if ARTIFACT_MARKER in body:
        return

    entries = [
        ("R4-P7-PLN-001", "execution_plan", R4 / "runs/2026-08-12_PHASE7_data_vnext_implementation_plan.md", "Phase-7 implementation plan"),
        ("R4-P7-SCH-001", "export_schema", REPO / "data_module/sentinel_data/export/format_schema/v2.yaml", "Explicit DATA vNext v2 format schema"),
        ("R4-P7-EXP-001", "manifest", OUT / "manifest.json", "G7-passed DATA vNext publication root"),
        ("R4-P7-EXP-002", "label_states", OUT / "label_states.parquet", "224,930-row canonical contract×class semantic state"),
        ("R4-P7-EXP-003", "ml_targets", OUT / "ml_targets.parquet", "Derived per-contract ten-class target/strength/mask/role projection"),
        ("R4-P7-REG-001", "source_registry", OUT / "source_registry.json", "Frozen first-baseline source authority snapshot"),
        ("R4-P7-REG-002", "crosswalk_registry", OUT / "crosswalk_registry.json", "Frozen vNext crosswalk action snapshot"),
        ("R4-P7-BND-001", "evidence_snapshot", OUT / "evidence_snapshot.json", "Ledger/policy/partition evidence bindings"),
        ("R4-P7-BND-002", "representation_requirements", OUT / "representation_requirements.json", "Exact non-excluded representation requirement set"),
        ("R4-P7-VAL-001", "semantic_validation", OUT / "validation_report.json", "Independent semantic validation report"),
        ("R4-P7-VAL-002", "representation_binding", OUT / "representation_binding_report.json", "21,657-contract / 64,971-file local physical binding report"),
        ("R4-P7-VAL-003", "g7_validation", OUT / "g7_validation_report.json", "Final representation-required G7 validation report"),
        ("R4-P7-ADR-001", "ADR", ADR_PATH, "Accepted G7 publication/training-input authority"),
        ("R4-P7-FND-001", "findings", FINDING_PATH, "Phase-7 implementation and G7 result"),
    ]

    source_commit = IMPLEMENTATION_MERGE[:12]
    rows = []
    for aid, typ, path, note in entries:
        if not path.is_file():
            raise RuntimeError(f"missing Phase-7 artifact: {path}")
        rel = path.relative_to(REPO).as_posix()
        rows.append(
            f"| {aid} | 7 | {typ} | {rel} | {sha256(path)} | {source_commit} | New | AVAILABLE_VERIFIED | NO | {note} |"
        )

    block = (
        ARTIFACT_MARKER
        + "\n\n| Artifact ID | Phase | Type | Path/URI | SHA-256 | Source commit | Historical/New | Availability | Protected | Notes |\n"
        + "|---|---|---|---|---|---|---|---|---|---|\n"
        + "\n".join(rows)
        + "\n\n"
    )
    anchor = "## Availability\n"
    if anchor not in body:
        raise RuntimeError("ARTIFACT_INDEX availability anchor missing")
    write(index, body.replace(anchor, block + anchor, 1))


def append_execution_log(info: dict) -> None:
    log = R4 / "EXECUTION_LOG.md"
    body = text(log)
    if LOG_MARKER in body:
        return
    rep = info["rep"]
    g7 = info["g7"]
    entry = f"""

---

{LOG_MARKER}

- **Phase:** 7
- **Operator:** GPT-5.6 Sol under delegated routine technical/governance approval; local physical gate executed by project owner
- **Date:** 2026-08-12
- **Implementation merge:** `{IMPLEMENTATION_MERGE}`
- **Local G7 evidence commit:** `{LOCAL_G7_EVIDENCE_COMMIT}`
- **Dataset:** `sentinel-r4-vnext-v1` / export schema `v2` / graph schema `v9`
- **Semantic population:** {EXPECTED_CONTRACTS:,} contracts / {EXPECTED_ROWS:,} contract×class rows / {g7['target_counts']['1']:,} positive targets / 0 negative targets / {g7['training_strength_counts']['STRONG']:,} STRONG / {g7['training_strength_counts']['WEAK']:,} WEAK
- **Physical representation validation:** {rep['checked_contracts']:,}/{rep['required_contracts']:,} contracts; {rep['checked_files']:,}/{rep['expected_files']:,} files; missing=0; mismatches=0; physical path not recorded
- **Representation binding digest:** `{rep['binding_digest_sha256']}`
- **Unsupported roles preserved:** THRESHOLD_FIT empty; CALIBRATION_FIT empty; UNTOUCHED_ACCEPTANCE empty/frozen
- **Historical artifacts changed:** NO
- **Legacy v1 semantic path changed:** NO
- **Decision:** R4-D-007 / ADR-R4-007 Accepted
- **Gate effect:** **G7 PASS.** Phase 8 is authorized to adapt the existing frozen training consumer to this exact v2 lineage and retrain without acceptance leakage.
"""
    write(log, body.rstrip() + entry + "\n")


def update_active_docs() -> None:
    root_readme = REPO / "README.md"
    replace_once(
        root_readme,
        "The current stable `main` baseline is the R4 **G6-passed** state from merge commit `91f795885` plus later documentation-only reconciliation.",
        "The current stable `main` baseline includes the R4 **G7-passed DATA vNext v2 implementation**, merged at `81d9c547d`, on top of the V3/runtime and canonical-documentation baseline.",
    )
    replace_once(
        root_readme,
        "Phase 7 implements the additive DATA vNext v2 semantic overlay on branch `r4/phase7-data-vnext-implementation`. Its remote semantic checks are green, but G7 is not complete until the existing local graph/token representations are physically bound and validated. Until G7 is merged, the v2 implementation branch is candidate work rather than canonical `main` runtime.",
        "Phase 7 is complete: DATA vNext v2 is canonical, its semantic overlay is deterministic, and the real local representation population was physically bound and validated for G7. Phase 8 is now the next authorized R4 step; no repaired teacher has been retrained yet.",
    )

    data_readme = REPO / "data_module/README.md"
    replace_once(data_readme, "Stable `main` has passed R4 **G6**:", "Stable `main` has passed R4 **G7**:")
    replace_once(
        data_readme,
        "Phase 7 DATA vNext implementation is active on `r4/phase7-data-vnext-implementation`. Remote semantic generation/validation is green, but G7 still requires local binding to the existing 21,657 physical representation triplets before merge.",
        "Phase 7 DATA vNext implementation is complete and canonical. The local G7 gate bound all 21,657 required representation triplets (64,971 files) with zero missing files and zero mismatches; the committed v2 manifest is `VALIDATED_G7_CANDIDATE` and Phase 8 is authorized.",
    )
    replace_once(data_readme, "vnext/                     DATA vNext v2 implementation (Phase 7 branch until G7)", "vnext/                     canonical DATA vNext v2 implementation (G7-passed)")
    replace_once(data_readme, "On canonical `main` through G6:", "On canonical `main` through G7:")
    replace_once(
        data_readme,
        "Phase-7 build/local commands belong to the dedicated Phase-7 branch until G7 is complete.",
        "The representation-bound v2 publication is now the canonical Phase-8 training-data lineage; use the committed vNext validator/loader and do not regenerate roles or fall back to historical v1 semantics.",
    )

    data_artifacts = REPO / "docs/handbook/04_data_artifacts.md"
    replace_once(
        data_artifacts,
        "Remote semantic generation is deterministic. Final G7 requires local binding to the existing 21,657 represented contracts before the v2 candidate can be promoted/merged.",
        "Semantic generation is deterministic, and G7 is complete: the local gate bound all 21,657 required represented contracts / 64,971 graph-token-sidecar files with zero missing files or mismatches. The committed v2 manifest is representation-bound and Phase 8 may consume it.",
    )
    replace_once(
        data_artifacts,
        "- Phase-7 v2 is not canonical main until G7 local representation binding passes and the branch merges.",
        "- DATA vNext v2 is canonical post-G7; Phase 8 must consume its exact manifest/roles/masks rather than rebuilding semantics implicitly.",
    )
    replace_once(
        data_artifacts,
        "After G7 merges, use the committed vNext CLI/validator for v2 artifact verification.",
        "Use the committed vNext CLI/validator for v2 artifact verification; representation-required validation is now part of the G7 lineage.",
    )
    replace_once(
        data_artifacts,
        "For repaired semantics, read R4 policy/schema/partition artifacts; after G7, the additive `data_module/sentinel_data/vnext` package is the v2 implementation source.",
        "For repaired semantics, read R4 policy/schema/partition artifacts and the canonical `data_module/sentinel_data/vnext` package, which is the G7-passed v2 implementation source.",
    )

    current = REPO / "docs/handbook/16_current_status.md"
    body = text(current)
    summary_pattern = r"## 30-second summary\n\n.*?\n\nThis page intentionally does \*\*not\* carry the old July module-suite totals\."
    summary_replacement = f"""## 30-second summary

The canonical post-G7 baseline includes DATA vNext implementation merge **`81d9c547d`** (2026-08-12): R4 DATA/ML repair has passed **G0 through G7**, the V3 registry/context protocol and read-only audit-MCP boundary remain implemented, and Run12 remains the historical operational teacher. The v2 semantic overlay is now locally bound to all {EXPECTED_REPRESENTED:,} required representations / {EXPECTED_FILES:,} physical files with zero mismatches. **Phase 8 retraining is now READY**, but no repaired teacher checkpoint has been trained or promoted yet.

This page intentionally does **not** carry the old July module-suite totals."""
    updated, count = re.subn(summary_pattern, summary_replacement, body, count=1, flags=re.DOTALL)
    if count != 1:
        raise RuntimeError("could not replace handbook current-status summary")
    body = updated
    body = body.replace("R4 G0–G6 PASS", "R4 G0–G7 PASS")
    body = body.replace(
        "Phase 7 branch\n  deterministic DATA vNext v2 semantic overlay\n  remote semantic checks PASS\n  local representation binding PENDING\n        ↓\nG7\n        ↓\nPhase 8 retrain existing architecture",
        "DATA vNext v2\n  deterministic semantic overlay PASS\n  local representation binding PASS\n  G7 PASS\n        ↓\nPhase 8 retrain existing architecture (READY)",
    )
    body = body.replace("| 7 | pending G7 | implementation branch exists; local physical representation binding still required |", "| 7 | G7 PASS | v2 implementation merged; 21,657 representations / 64,971 files physically bound with zero mismatches |")
    body = body.replace("| 8–10 | waiting | retraining/evaluation/promotion not authorized until preceding gates |", "| 8 | READY | existing-architecture retraining authorized against the exact G7-passed v2 lineage |\n| 9–10 | waiting | evaluation/promotion remain gated by preceding phases |")

    phase7_pattern = r"### Phase 7 candidate state\n\n.*?\n\n### Current ML state"
    phase7_replacement = f"""### Phase 7 G7 state

DATA vNext v2 is now canonical through implementation merge `{IMPLEMENTATION_MERGE[:9]}` and the G7 closeout records the locally bound publication.

Final G7 evidence:

- manifest status: `VALIDATED_G7_CANDIDATE`;
- contracts: {EXPECTED_CONTRACTS:,};
- contract×class rows: {EXPECTED_ROWS:,};
- positive targets: 1,007; negative targets: 0;
- STRONG signals: 403; WEAK signals: 604;
- required/checked representation contracts: {EXPECTED_REPRESENTED:,}/{EXPECTED_REPRESENTED:,};
- required/checked physical files: {EXPECTED_FILES:,}/{EXPECTED_FILES:,};
- missing files: 0;
- representation mismatches: 0;
- physical local path recorded: false;
- representation binding digest: `{EXPECTED_BINDING_DIGEST}`.

The v2 loader rejects silent historical-v1 fallback. Historical v1 artifacts remain immutable. Phase 8 is authorized to adapt the training consumer to this exact lineage; it is not authorized to invent negatives or unsupported evaluation roles.

### Current ML state"""
    body, count = re.subn(phase7_pattern, phase7_replacement, body, count=1, flags=re.DOTALL)
    if count != 1:
        raise RuntimeError("could not replace Phase-7 current-status section")
    body = body.replace("- Phase 7 still requires local physical representation binding;", "- Phase 7 physical representation binding passed; the remaining DATA/ML limitations are evidence limitations, not G7 implementation blockers;")
    body = body.replace("For Phase 7, use the active branch’s `p7_run_local_gate.py` command when the local representation root is available.", "For G7 evidence, use the committed vNext manifest, representation-binding report, and final G7 validation report. The local gate is retained for reproducibility, not because G7 is still pending.")
    body = body.replace("For Phase 7, inspect its branch and gate scripts rather than assuming candidate code is already canonical main.", "For DATA vNext, inspect the canonical vNext package, G7 manifest/reports, and R4 decisions rather than historical v1 label/export assumptions.")
    body = body.replace("Today a correct statement is: “R4 G6 is canonical; Phase 7 remote semantics are green but local representation binding is pending; Run12 is historical operational inference; no retrained vNext teacher or untouched-acceptance claim exists.”", "Today a correct statement is: “R4 G7 is canonical; DATA vNext v2 is representation-bound; Phase 8 retraining is ready; Run12 remains historical operational inference; no retrained vNext teacher or untouched-acceptance claim exists.”")
    body = body.replace("Can you distinguish canonical main from Phase-7 candidate state, state all unsupported evaluation roles, identify the current teacher/proof/MCP protocol versions, and name the exact local blocker before G7?", "Can you identify the exact G7 DATA vNext manifest/binding lineage, state all unsupported evaluation roles, distinguish Run12 from the future repaired teacher, and explain what Phase 8 is and is not authorized to change?")
    write(current, body)


def update_handbook_validator() -> None:
    validator = REPO / "docs/handbook/tools/verify_handbook.py"
    body = text(validator)

    old_facts = '''    support = _json("docs/plan/ml-R4/manifests/p6_role_support_table.json")\n    status = _text("docs/plan/ml-R4/PLAN_STATUS_MATRIX.md")\n    return {"policy": policy, "partition": partition, "acceptance": acceptance, "support": support, "status_text": status}\n'''
    new_facts = '''    support = _json("docs/plan/ml-R4/manifests/p6_role_support_table.json")\n    status = _text("docs/plan/ml-R4/PLAN_STATUS_MATRIX.md")\n    g7_manifest = _json("data_module/data/exports/sentinel-r4-vnext-v1/manifest.json")\n    g7_representation = _json("data_module/data/exports/sentinel-r4-vnext-v1/representation_binding_report.json")\n    g7_validation = _json("data_module/data/exports/sentinel-r4-vnext-v1/g7_validation_report.json")\n    return {\n        "policy": policy,\n        "partition": partition,\n        "acceptance": acceptance,\n        "support": support,\n        "status_text": status,\n        "g7_manifest": g7_manifest,\n        "g7_representation": g7_representation,\n        "g7_validation": g7_validation,\n    }\n'''
    if new_facts not in body:
        if old_facts not in body:
            raise RuntimeError("handbook validator _r4_facts block changed unexpectedly")
        body = body.replace(old_facts, new_facts, 1)

    old_check = '    checks.append(Check("R4 phase status", "| 6 | `phases/07_PHASE_6_PARTITIONS_AND_ACCEPTANCE_FREEZE.md` | PASSED |" in r4["status_text"] and "| 7 | `phases/08_PHASE_7_DATA_VNEXT_IMPLEMENTATION.md` | READY |" in r4["status_text"], "Phase 6 PASSED / Phase 7 READY on canonical main"))\n'
    new_check = f'''    checks.append(Check("R4 phase status", "| 7 | `phases/08_PHASE_7_DATA_VNEXT_IMPLEMENTATION.md` | PASSED |" in r4["status_text"] and "| 8 | `phases/09_PHASE_8_EXISTING_MODEL_RETRAINING.md` | READY |" in r4["status_text"], "Phase 7 PASSED / Phase 8 READY on canonical main"))\n    g7_manifest = r4["g7_manifest"]\n    g7_rep = r4["g7_representation"]\n    g7_report = r4["g7_validation"]\n    checks.append(Check("R4 G7 publication", g7_manifest.get("status") == "VALIDATED_G7_CANDIDATE" and g7_manifest.get("export_schema_version") == "v2" and g7_manifest.get("population", {{}}).get("contracts") == {EXPECTED_CONTRACTS}, f"status={{g7_manifest.get('status')}}, schema={{g7_manifest.get('export_schema_version')}}"))\n    checks.append(Check("R4 G7 representation binding", g7_rep.get("passed") is True and g7_rep.get("checked_contracts") == {EXPECTED_REPRESENTED} and g7_rep.get("checked_files") == {EXPECTED_FILES} and g7_rep.get("missing_files_total") == 0 and g7_rep.get("mismatch_total") == 0 and g7_rep.get("physical_root_recorded") is False and g7_rep.get("binding_digest_sha256") == "{EXPECTED_BINDING_DIGEST}", f"contracts={{g7_rep.get('checked_contracts')}}, files={{g7_rep.get('checked_files')}}, missing={{g7_rep.get('missing_files_total')}}, mismatches={{g7_rep.get('mismatch_total')}}"))\n    checks.append(Check("R4 G7 final validation", g7_report.get("passed") is True and g7_report.get("require_representation_binding") is True and g7_report.get("target_counts") == {{"1": 1007, "None": 223923}} and g7_report.get("training_strength_counts") == {{"NONE": 223923, "STRONG": 403, "WEAK": 604}}, f"passed={{g7_report.get('passed')}}, targets={{g7_report.get('target_counts')}}"))\n'''
    if new_check not in body:
        if old_check not in body:
            raise RuntimeError("handbook validator R4 phase check changed unexpectedly")
        body = body.replace(old_check, new_check, 1)

    old_inventory = '            "acceptance_contracts": len(r4["acceptance"]["contract_ids"]),\n'
    new_inventory = old_inventory + '            "g7_status": r4["g7_manifest"]["status"],\n            "g7_binding_digest": r4["g7_representation"]["binding_digest_sha256"],\n            "g7_checked_contracts": r4["g7_representation"]["checked_contracts"],\n'
    if '"g7_status": r4["g7_manifest"]["status"]' not in body:
        if old_inventory not in body:
            raise RuntimeError("handbook validator inventory block changed unexpectedly")
        body = body.replace(old_inventory, new_inventory, 1)

    write(validator, body)


def verify_closeout(info: dict) -> None:
    assert "| 7 | `phases/08_PHASE_7_DATA_VNEXT_IMPLEMENTATION.md` | PASSED |" in text(R4 / "PLAN_STATUS_MATRIX.md")
    assert "| 8 | `phases/09_PHASE_8_EXISTING_MODEL_RETRAINING.md` | READY |" in text(R4 / "PLAN_STATUS_MATRIX.md")
    assert "**Status:** PASSED — G7 PASS" in text(R4 / "phases/08_PHASE_7_DATA_VNEXT_IMPLEMENTATION.md")
    assert "**Status:** READY — G7 SATISFIED" in text(R4 / "phases/09_PHASE_8_EXISTING_MODEL_RETRAINING.md")
    assert "| R4-D-007 |" in text(R4 / "DECISION_REGISTER.md")
    assert ARTIFACT_MARKER in text(R4 / "ARTIFACT_INDEX.md")
    assert LOG_MARKER in text(R4 / "EXECUTION_LOG.md")
    assert "G0 through G7" in text(REPO / "docs/handbook/16_current_status.md")
    assert "R4 G7 publication" in text(REPO / "docs/handbook/tools/verify_handbook.py")
    assert info["rep"]["binding_digest_sha256"] == EXPECTED_BINDING_DIGEST


def main() -> int:
    info = validate_g7_inputs()
    create_adr()
    create_finding(info)
    update_r4_governance()
    register_artifacts()
    append_execution_log(info)
    update_active_docs()
    update_handbook_validator()
    verify_closeout(info)
    print(json.dumps({
        "passed": True,
        "phase7": "PASSED_G7",
        "phase8": "READY",
        "dataset": "sentinel-r4-vnext-v1",
        "implementation_merge": IMPLEMENTATION_MERGE,
        "local_g7_evidence_commit": LOCAL_G7_EVIDENCE_COMMIT,
        "binding_digest_sha256": EXPECTED_BINDING_DIGEST,
        "checked_contracts": EXPECTED_REPRESENTED,
        "checked_files": EXPECTED_FILES,
        "artifact_manifest_sha256": sha256(OUT / "manifest.json"),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
