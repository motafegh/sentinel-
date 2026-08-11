#!/usr/bin/env python3
"""Promote and register the frozen R4 Phase-6 role/acceptance artifacts."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path("docs/plan/ml-R4")
M = ROOT / "manifests"
PARTITION = M / "p6_partition_manifest.json"
PHASE6 = ROOT / "phases/07_PHASE_6_PARTITIONS_AND_ACCEPTANCE_FREEZE.md"
PHASE7 = ROOT / "phases/08_PHASE_7_DATA_VNEXT_IMPLEMENTATION.md"
STATUS = ROOT / "PLAN_STATUS_MATRIX.md"
INDEX = ROOT / "ARTIFACT_INDEX.md"
LOG = ROOT / "EXECUTION_LOG.md"

ARTIFACT_MARKER = "## Phase 6 artifacts — role partitions and acceptance freeze (G6 PASS)"
LOG_MARKER = "### R4-LOG-20260812-009 — Phase 6 Leakage-Safe Roles and G6 Closure"
CLOSEOUT_MARKER = "## G6 closeout"

ARTIFACT_PATHS = [
    ("R4-P6-PLN-001", "execution_plan", ROOT / "runs/2026-08-12_PHASE6_partitions_acceptance_freeze_plan.md", "Phase-6 role/acceptance execution plan"),
    ("R4-P6-INV-001", "role_support_inventory", M / "p6_role_support_inventory.json", "Measured Phase-5-authorized strong/weak/unlabeled group support"),
    ("R4-P6-INV-002", "group_eligibility_inventory", M / "p6_group_eligibility_inventory.jsonl", "13,509 leakage-group eligibility rows"),
    ("R4-P6-MAN-001", "role_group_manifest", M / "p6_role_group_manifest.jsonl", "One frozen role per leakage group"),
    ("R4-P6-MAN-002", "contract_role_manifest", M / "p6_contract_role_manifest.jsonl", "One role per 22,493 contracts"),
    ("R4-P6-MAN-003", "role_support_table", M / "p6_role_support_table.json", "Per-role/class support and limitations"),
    ("R4-P6-MAN-004", "unsupported_roles", M / "p6_unsupported_roles.json", "Threshold/calibration/acceptance controlled empty roles"),
    ("R4-P6-MAN-005", "untouched_acceptance", M / "p6_untouched_acceptance_manifest.json", "Frozen empty unsupported untouched-acceptance manifest"),
    ("R4-P6-MAN-006", "partition_manifest", PARTITION, "r4-vnext-roles-v1 frozen partition root"),
    ("R4-P6-ADR-001", "ADR", ROOT / "adrs/ADR-R4-006-role-partition-and-acceptance-freeze.md", "Leakage-safe partition and empty-acceptance decision"),
    ("R4-P6-FND-001", "findings", ROOT / "findings/08_phase6_role_partition_and_acceptance_freeze.md", "Partition/support/exposure interpretation"),
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def promote_partition() -> None:
    data = json.loads(PARTITION.read_text())
    if data["status"] not in {"FROZEN_CANDIDATE_G6", "FROZEN_G6"}:
        raise RuntimeError(f"unexpected partition status {data['status']}")
    data["status"] = "FROZEN_G6"
    data["gate"] = "G6_PASS"
    data["decision_id"] = "R4-D-006"
    PARTITION.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def update_phase_docs() -> None:
    p6 = PHASE6.read_text()
    p6 = p6.replace("**Status:** READY — G5 SATISFIED", "**Status:** PASSED — G6 PASS")
    if CLOSEOUT_MARKER not in p6:
        p6 += """

## G6 closeout

G6 passed with `r4-vnext-roles-v1` frozen over all 22,493 active contracts / 13,509 leakage groups.

Frozen active roles:

- TRAIN_STRONG: 238 groups / 275 contracts
- MODEL_SELECTION: 51 / 56 (positive-only limited support)
- INTERNAL_AUDIT: 51 / 62
- TRAIN_WEAK: 465 / 773 (DIVE TOD weak signal only)
- TRAIN_UNLABELED: 11,869 / 20,491
- EXCLUDED: 835 / 836 (incomplete representation group)

`THRESHOLD_FIT` and `CALIBRATION_FIT` are `UNSUPPORTED_EMPTY`. `UNTOUCHED_ACCEPTANCE` is `UNSUPPORTED_EMPTY_FROZEN` with zero contracts/groups. No confirmed-negative rows were synthesized.

**G6 PASS.** Phase 7 may implement DATA vNext from the frozen Phase-5 policy and Phase-6 role manifests. It may not regenerate/rebalance roles or manufacture unsupported evaluation sets.
"""
    PHASE6.write_text(p6)

    p7 = PHASE7.read_text().replace("**Status:** WAITING FOR G6", "**Status:** READY — G6 SATISFIED")
    if "## Phase-6 handoff" not in p7:
        p7 += """

## Phase-6 handoff

Phase 7 must consume `r4-vnext-roles-v1` exactly. Threshold/calibration/untouched-acceptance roles are intentionally empty/unsupported; implementation must preserve that limitation. GasException and UnusedReturn remain supervision-disabled. The 836 contracts in incomplete-representation groups remain excluded unless a future versioned plan explicitly rebuilds and re-freezes roles.
"""
    PHASE7.write_text(p7)

    s = STATUS.read_text()
    old6 = "| 6 | `phases/07_PHASE_6_PARTITIONS_AND_ACCEPTANCE_FREEZE.md` | IN_PROGRESS | G5 | G6 | Building deterministic leakage-group roles; threshold/calibration/untouched-acceptance support must fail closed if trustworthy negatives/untouched groups are unavailable |"
    new6 = "| 6 | `phases/07_PHASE_6_PARTITIONS_AND_ACCEPTANCE_FREEZE.md` | PASSED | G5 | G6 | r4-vnext-roles-v1 covers 22,493 contracts/13,509 groups exactly once; 836 incomplete-representation contracts excluded; threshold/calibration/untouched acceptance frozen unsupported/empty; G6 PASS |"
    old7 = "| 7 | `phases/08_PHASE_7_DATA_VNEXT_IMPLEMENTATION.md` | WAITING | G6 | G7 | Versioned artifacts |"
    new7 = "| 7 | `phases/08_PHASE_7_DATA_VNEXT_IMPLEMENTATION.md` | READY | G6 | G7 | Implement accepted data-vnext-policy-v1 with frozen r4-vnext-roles-v1; no role rebalancing or semantic invention |"
    if old6 not in s and new6 not in s:
        raise RuntimeError("Phase-6 status matrix row not recognized")
    if old7 not in s and new7 not in s:
        raise RuntimeError("Phase-7 status matrix row not recognized")
    s = s.replace(old6, new6).replace(old7, new7)
    STATUS.write_text(s)


def register_artifacts() -> None:
    commit = head()[:12]
    rows = []
    for aid, typ, path, note in ARTIFACT_PATHS:
        if not path.is_file():
            raise RuntimeError(f"missing Phase-6 artifact {path}")
        rows.append(f"| {aid} | 6 | {typ} | {path.relative_to(ROOT)} | {sha256(path)} | {commit} | New | AVAILABLE_VERIFIED | NO | {note} |")
    block = ARTIFACT_MARKER + "\n\n| Artifact ID | Phase | Type | Path/URI | SHA-256 | Source commit | Historical/New | Availability | Protected | Notes |\n|---|---|---|---|---|---|---|---|---|---|\n" + "\n".join(rows) + "\n\n"
    text = INDEX.read_text()
    if ARTIFACT_MARKER not in text:
        anchor = "## Availability\n"
        if anchor not in text:
            raise RuntimeError("artifact index availability anchor missing")
        INDEX.write_text(text.replace(anchor, block + anchor, 1))


def append_log() -> None:
    text = LOG.read_text()
    if LOG_MARKER in text:
        return
    p = json.loads(PARTITION.read_text())
    entry = f"""

---

{LOG_MARKER}

- **Phase:** 6
- **Operator:** GPT-5.6 Sol under delegated routine technical/governance approval
- **Date:** 2026-08-12
- **Inputs:** Phase-3 ledger `{p['ledger_sha256']}`; accepted Phase-5 policy `{p['policy_sha256']}`; measured role-support inventory; repository exposure audit
- **Partition:** `{p['partition_version']}` / status `FROZEN_G6`
- **Population:** {p['population_contracts']} contracts / {p['population_groups']} leakage groups, exactly one role each
- **Roles:** TRAIN_STRONG 275 contracts; MODEL_SELECTION 56; INTERNAL_AUDIT 62; TRAIN_WEAK 773; TRAIN_UNLABELED 20,491; EXCLUDED 836
- **Evidence limitations:** zero confirmed-negative rows; MODEL_SELECTION positive-only; THRESHOLD_FIT and CALIBRATION_FIT unsupported empty; UNTOUCHED_ACCEPTANCE unsupported empty frozen
- **Exposure findings:** manual suite exposed to historical ML/AGENTS validation; quickstart contains invalid NonVulnerable mappings; Tier-E BCCC/tool-silence design is not confirmed-negative evidence; unavailable/deferred sources not imported
- **Protected artifacts changed:** NO
- **Implementation code changed:** NO
- **Decision:** R4-D-006 / ADR-R4-006 Accepted
- **Gate effect:** **G6 PASS.** Phase 7 is authorized to implement DATA vNext exactly from the frozen policy and role manifests; threshold/calibration/acceptance limitations must remain explicit.
"""
    LOG.write_text(text.rstrip() + entry + "\n")


def verify() -> None:
    p = json.loads(PARTITION.read_text())
    assert p["status"] == "FROZEN_G6"
    assert p["gate"] == "G6_PASS"
    assert p["decision_id"] == "R4-D-006"
    assert "**Status:** PASSED — G6 PASS" in PHASE6.read_text()
    assert "**Status:** READY — G6 SATISFIED" in PHASE7.read_text()
    assert ARTIFACT_MARKER in INDEX.read_text()
    assert LOG_MARKER in LOG.read_text()
    print(json.dumps({"passed": True, "partition_status": p["status"], "partition_sha256": sha256(PARTITION), "artifacts": len(ARTIFACT_PATHS)}, indent=2))


def main() -> int:
    promote_partition()
    update_phase_docs()
    register_artifacts()
    append_log()
    verify()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
