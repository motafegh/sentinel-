from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).with_name("p8_validate_v10_v25_evidence_chain.py")
SPEC = importlib.util.spec_from_file_location("p8_validate_v10_v25_evidence_chain", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _write(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def test_chain_requires_exact_source_and_semantic_sha(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    semantic = tmp_path / "semantic.json"
    bounded = tmp_path / "bounded.json"
    _write(source, {"source": True})
    _write(semantic, {"semantic": True})
    _write(
        bounded,
        {
            "schema": MODULE.BOUNDED_SCHEMA,
            "bounded_v25_reproducibility_passed": True,
            "zero_unexplained_drift": True,
            "blocking_identities": [],
            "source_audit_sha256": MODULE._sha256(source),
            "semantic_evidence_sha256": MODULE._sha256(semantic),
            "unexpected_identities": 20,
            "index_equivalence_identities": 8,
            "semantic_correction_identities": 12,
            "repeat_generations": 3,
        },
    )

    result = MODULE.validate_chain(
        source_audit=source,
        bounded_report=bounded,
        semantic_evidence=semantic,
    )
    assert result["passed"] is True
    assert result["unexpected_identities"] == 20

    _write(source, {"source": "changed"})
    with pytest.raises(ValueError, match="source-audit SHA"):
        MODULE.validate_chain(
            source_audit=source,
            bounded_report=bounded,
            semantic_evidence=semantic,
        )


def test_chain_rejects_bounded_blocker(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    semantic = tmp_path / "semantic.json"
    bounded = tmp_path / "bounded.json"
    _write(source, {})
    _write(semantic, {})
    _write(
        bounded,
        {
            "schema": MODULE.BOUNDED_SCHEMA,
            "bounded_v25_reproducibility_passed": True,
            "zero_unexplained_drift": True,
            "blocking_identities": ["dive/fixture"],
            "source_audit_sha256": MODULE._sha256(source),
            "semantic_evidence_sha256": MODULE._sha256(semantic),
        },
    )

    with pytest.raises(ValueError, match="blocking identities"):
        MODULE.validate_chain(
            source_audit=source,
            bounded_report=bounded,
            semantic_evidence=semantic,
        )
