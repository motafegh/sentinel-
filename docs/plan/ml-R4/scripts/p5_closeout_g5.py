#!/usr/bin/env python3
"""Promote and register the accepted R4 Phase-5 DATA vNext policy."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path("docs/plan/ml-R4")
POLICY = ROOT / "specs/data_vnext_policy_v1.json"
SCHEMA = ROOT / "schemas/data_vnext_label_state_v1.schema.json"
SPEC = ROOT / "findings/07_data_vnext_policy_and_design_specification.md"
PLAN = ROOT / "runs/2026-08-12_PHASE5_data_vnext_policy_design_plan.md"
ARTIFACT_INDEX = ROOT / "ARTIFACT_INDEX.md"
EXECUTION_LOG = ROOT / "EXECUTION_LOG.md"

ADRS = [
    ROOT / "adrs/ADR-R4-001-label-state-and-training-signal.md",
    ROOT / "adrs/ADR-R4-002-source-class-authority-and-enablement.md",
    ROOT / "adrs/ADR-R4-003-crosswalk-and-aggregation-semantics.md",
    ROOT / "adrs/ADR-R4-004-export-and-ml-consumer-contract.md",
    ROOT / "adrs/ADR-R4-005-lineage-versioning-and-rollback.md",
]

ARTIFACT_MARKER = "## Phase 5 artifacts — DATA vNext policy and design (G5 PASS)"
LOG_MARKER = "### R4-LOG-20260812-008 — Phase 5 DATA vNext Policy and G5 Closure"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def promote_policy() -> None:
    policy = json.loads(POLICY.read_text(encoding="utf-8"))
    if policy["policy_version"] != "data-vnext-policy-v1":
        raise RuntimeError("unexpected policy version")
    if policy["status"] not in {"PROPOSED_FOR_G5_VALIDATION", "ACCEPTED_G5"}:
        raise RuntimeError(f"unexpected policy status: {policy['status']}")
    policy["status"] = "ACCEPTED_G5"
    POLICY.write_text(json.dumps(policy, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def register_artifacts() -> None:
    head = git_head()[:12]
    policy_sha = sha256(POLICY)
    schema_sha = sha256(SCHEMA)
    spec_sha = sha256(SPEC)
    plan_sha = sha256(PLAN)
    adr_shas = [sha256(p) for p in ADRS]

    block = f'''## Phase 5 artifacts — DATA vNext policy and design (G5 PASS)

| Artifact ID | Phase | Type | Path/URI | SHA-256 | Source commit | Historical/New | Availability | Protected | Notes |
|---|---|---|---|---|---|---|---|---|---|
| R4-P5-PLN-001 | 5 | execution_plan | runs/2026-08-12_PHASE5_data_vnext_policy_design_plan.md | {plan_sha} | {head} | New | AVAILABLE_VERIFIED | NO | Design-only Phase-5 execution plan |
| R4-P5-POL-001 | 5 | machine_policy | specs/data_vnext_policy_v1.json | {policy_sha} | {head} | New | AVAILABLE_VERIFIED | NO | Accepted DATA vNext source/class/state/role policy v1 |
| R4-P5-SCH-001 | 5 | schema | schemas/data_vnext_label_state_v1.schema.json | {schema_sha} | {head} | New | AVAILABLE_VERIFIED | NO | Contract×class label/outcome/training-signal schema |
| R4-P5-FND-001 | 5 | specification | findings/07_data_vnext_policy_and_design_specification.md | {spec_sha} | {head} | New | AVAILABLE_VERIFIED | NO | Implementation-facing semantic specification |
| R4-P5-ADR-001 | 5 | ADR | adrs/ADR-R4-001-label-state-and-training-signal.md | {adr_shas[0]} | {head} | New | AVAILABLE_VERIFIED | NO | Outcome truth separated from training signal |
| R4-P5-ADR-002 | 5 | ADR | adrs/ADR-R4-002-source-class-authority-and-enablement.md | {adr_shas[1]} | {head} | New | AVAILABLE_VERIFIED | NO | First-baseline source/class authority and disabled classes |
| R4-P5-ADR-003 | 5 | ADR | adrs/ADR-R4-003-crosswalk-and-aggregation-semantics.md | {adr_shas[2]} | {head} | New | AVAILABLE_VERIFIED | NO | No-target crosswalk states and evidence aggregation |
| R4-P5-ADR-004 | 5 | ADR | adrs/ADR-R4-004-export-and-ml-consumer-contract.md | {adr_shas[3]} | {head} | New | AVAILABLE_VERIFIED | NO | Explicit v2 export and masked ML consumer contract |
| R4-P5-ADR-005 | 5 | ADR | adrs/ADR-R4-005-lineage-versioning-and-rollback.md | {adr_shas[4]} | {head} | New | AVAILABLE_VERIFIED | NO | Immutable history, versioning, fail-closed publication, rollback |

'''

    text = ARTIFACT_INDEX.read_text(encoding="utf-8")
    if ARTIFACT_MARKER not in text:
        anchor = "## Availability\n"
        if anchor not in text:
            raise RuntimeError("ARTIFACT_INDEX availability anchor missing")
        text = text.replace(anchor, block + anchor, 1)
        ARTIFACT_INDEX.write_text(text, encoding="utf-8")


def append_log() -> None:
    if LOG_MARKER in EXECUTION_LOG.read_text(encoding="utf-8"):
        return

    policy_sha = sha256(POLICY)
    schema_sha = sha256(SCHEMA)
    spec_sha = sha256(SPEC)
    head = git_head()
    entry = f'''

---

### R4-LOG-20260812-008 — Phase 5 DATA vNext Policy and G5 Closure

- **Phase:** 5
- **Operator:** GPT-5.6 Sol under delegated routine technical/governance approval
- **Date:** 2026-08-12
- **Repository branch/commit:** `r4/phase5-data-vnext-policy-design` at `{head}` before deterministic closeout commit
- **Inputs:** Phase-0–4 evidence, Phase-3 ledger `3983cc2b3317515d546c784449b583ac9a7c23ac8da267ee10f5640857cd0ac7`, Phase-4 R4-GAP-002 review `7d7f0cce287c847df2376ac0f580abf6be05b46f6a2c90b5f00f9b34f8fc5473`
- **Outputs:** accepted machine policy SHA-256 `{policy_sha}`; label-state schema SHA-256 `{schema_sha}`; implementation specification SHA-256 `{spec_sha}`; five Accepted ADRs; R4-D-001..005 decision-register entries
- **Validation:** JSON-Schema row invariants + machine policy assertions + design-only branch scope guard passed in `R4 Phase 5 DATA vNext policy` CI
- **Key decisions:** target zero requires confirmed negative; no blanket negative source; SolidiFI/approved SmartBugs direct categories strong positive; DIVE weak TOD only and otherwise unlabeled/masked; SmartBugs bad_randomness/short_addresses/other no canonical target; GasException and UnusedReturn supervision disabled; export format v2 explicit; historical v1 immutable
- **Protected artifacts changed:** NO
- **Implementation code changed:** NO
- **Gate effect:** **G5 PASS.** Phase 6 is authorized to create leakage-safe dataset roles/partitions and freeze or explicitly declare unsupported acceptance support.
- **Next permitted action:** Phase 6 only; Phase-7 DATA implementation remains blocked until G6.
'''
    current = EXECUTION_LOG.read_text(encoding="utf-8").rstrip()
    EXECUTION_LOG.write_text(current + entry + "\n", encoding="utf-8")


def verify() -> None:
    policy = json.loads(POLICY.read_text(encoding="utf-8"))
    assert policy["status"] == "ACCEPTED_G5"
    index = ARTIFACT_INDEX.read_text(encoding="utf-8")
    log = EXECUTION_LOG.read_text(encoding="utf-8")
    assert index.count(ARTIFACT_MARKER) == 1
    assert log.count(LOG_MARKER) == 1
    for path in [POLICY, SCHEMA, SPEC, PLAN, *ADRS]:
        assert path.is_file(), path
    print(json.dumps({
        "passed": True,
        "policy_status": policy["status"],
        "policy_sha256": sha256(POLICY),
        "schema_sha256": sha256(SCHEMA),
        "spec_sha256": sha256(SPEC),
        "adrs": len(ADRS)
    }, indent=2))


def main() -> int:
    promote_policy()
    register_artifacts()
    append_log()
    verify()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
