"""Focused fail-closed tests for the full-population V10 transition audit V4."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


SCRIPT_DIR = Path(__file__).parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

SCRIPT = SCRIPT_DIR / "p8_audit_v10_transition_v4.py"
SPEC = importlib.util.spec_from_file_location("p8_audit_v10_transition_v4", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


def _fixture() -> tuple[dict, list[dict]]:
    index_rows = [
        {
            "contract": f"dive/index-{index}",
            "decision": AUDIT.INDEX_DECISION,
            "passed": True,
        }
        for index in range(AUDIT.EXPECTED_INDEX_IDENTITIES)
    ]
    write_rows = [
        {
            "contract": f"dive/write-{index}",
            "decision": AUDIT.WRITE_DECISION,
            "passed": True,
        }
        for index in range(AUDIT.EXPECTED_WRITE_IDENTITIES)
    ]
    source_audit_sha = "a" * 64
    source_binding = "b" * 64
    evidence_contracts = [
        {
            "contract": row["contract"],
            "target_groups": [
                {
                    "name": "EXPRESSION values.push(value)",
                    "source_lines": [10],
                    "coarse_type": "CFG_NODE",
                    "reference_multiplicity": 1,
                    "candidate_multiplicity": 1,
                    "write_proven": True,
                }
            ],
        }
        for row in write_rows
    ]
    evidence = {
        "schema": AUDIT.full_population.EVIDENCE_SCHEMA,
        "source_audit_sha256": source_audit_sha,
        "candidate_binding_digest_sha256": source_binding,
        "slither_analyzer": AUDIT.full_population.PRIMARY_SLITHER_VERSION,
        "unexpected_identities": AUDIT.EXPECTED_IDENTITIES,
        "contracts_with_write_drift": AUDIT.EXPECTED_WRITE_IDENTITIES,
        "target_groups": AUDIT.EXPECTED_WRITE_IDENTITIES,
        "duplicate_target_groups": 0,
        "storage_mutation_groups_proven": AUDIT.EXPECTED_WRITE_IDENTITIES,
        "unresolved_write_groups": [],
        "non_write_or_population_drift": [],
        "contracts": evidence_contracts,
    }
    probe = {
        "schema": AUDIT.PROBE_SCHEMA,
        "passed": True,
        "zero_unexplained_drift": True,
        "blocking_identities": [],
        "physical_acceptance": False,
        "training_authorized": False,
        "unexpected_identities": AUDIT.EXPECTED_IDENTITIES,
        "repeat_generations": AUDIT.EXPECTED_REPEAT_GENERATIONS,
        "semantic_evidence_repeats": AUDIT.EXPECTED_REPEAT_GENERATIONS,
        "semantic_evidence_stable": True,
        "decision_counts": {
            AUDIT.INDEX_DECISION: AUDIT.EXPECTED_INDEX_IDENTITIES,
            AUDIT.WRITE_DECISION: AUDIT.EXPECTED_WRITE_IDENTITIES,
        },
        "contracts": index_rows + write_rows,
        "source_audit_sha256": source_audit_sha,
        "candidate_binding_digest_sha256": source_binding,
    }
    return probe, [evidence, dict(evidence), dict(evidence)]


def test_full_population_evidence_requires_exact_355_split() -> None:
    probe, evidence = _fixture()
    index_set, write_set, targets = AUDIT._validate_full_population_evidence(
        probe, evidence
    )
    assert len(index_set) == AUDIT.EXPECTED_INDEX_IDENTITIES
    assert len(write_set) == AUDIT.EXPECTED_WRITE_IDENTITIES
    assert set(targets) == write_set


def test_full_population_evidence_rejects_acceptance_or_blockers() -> None:
    probe, evidence = _fixture()
    probe["physical_acceptance"] = True
    with pytest.raises(ValueError, match="physical acceptance"):
        AUDIT._validate_full_population_evidence(probe, evidence)

    probe, evidence = _fixture()
    probe["blocking_identities"] = ["dive/blocker"]
    with pytest.raises(ValueError, match="blocking identities"):
        AUDIT._validate_full_population_evidence(probe, evidence)


def test_full_population_evidence_rejects_unstable_or_misbound_semantics() -> None:
    probe, evidence = _fixture()
    evidence[2] = {**evidence[2], "candidate_binding_digest_sha256": "c" * 64}
    with pytest.raises(ValueError, match="candidate_binding"):
        AUDIT._validate_full_population_evidence(probe, evidence)

    probe, evidence = _fixture()
    evidence[1] = {**evidence[1], "target_groups": 999}
    with pytest.raises(ValueError, match="not_stable|not stable|differs"):
        AUDIT._validate_full_population_evidence(probe, evidence)
