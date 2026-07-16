from __future__ import annotations

import hashlib
import json
import uuid

from src.contracts.submission import normalize_submission
from src.persistence.report_writer import persist_report


def test_default_submission_is_explicitly_ineligible() -> None:
    assert normalize_submission() == {
        "schema_version": "1",
        "proof_scope": "none",
        "status": "not_requested",
        "policy_decision": "not_requested",
        "policy_reason": None,
        "verified_audit_eligible": False,
        "finality_ineligible_reason": "no_proof_scope",
    }


def test_unbound_policy_reason_is_preserved() -> None:
    record = normalize_submission({
        "proof_scope": "legacy_proxy_only_unbound",
        "status": "policy_rejected",
        "decision": "policy_rejected",
        "reason": "proof_scope_not_identity_bound",
    })
    assert record["policy_reason"] == "proof_scope_not_identity_bound"
    assert record["finality_ineligible_reason"] == "proof_scope_not_identity_bound"


def test_persisted_report_and_cas_are_byte_identical(tmp_path) -> None:
    job_id = str(uuid.uuid4())
    submission = normalize_submission()
    status = persist_report(
        {"job_id": job_id},
        {"overall_label": "unknown", "submission": submission},
        tmp_path,
    )["report_persistence"]
    report_bytes = (tmp_path / job_id / "report.json").read_bytes()
    cas_bytes = (tmp_path / status["cas_path"]).read_bytes()
    assert report_bytes == cas_bytes
    assert hashlib.sha256(report_bytes).hexdigest() == status["cas_sha256"]
    assert json.loads(report_bytes)["submission"] == submission


def test_truth_round_trip_remains_byte_equal() -> None:
    source = normalize_submission({
        "proof_scope": "legacy_proxy_only_unbound",
        "status": "policy_rejected",
        "policy_decision": "policy_rejected",
        "policy_reason": "proof_scope_not_identity_bound",
        "verified_audit_eligible": False,
        "finality_ineligible_reason": "proof_scope_not_identity_bound",
    })
    assert normalize_submission(source) == source
