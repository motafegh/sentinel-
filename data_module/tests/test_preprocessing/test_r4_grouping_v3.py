"""Regression tests for corrected leakage grouping V3."""

from __future__ import annotations

import json

from sentinel_data.preprocessing.r4_grouping_v3 import build_grouping_v3
from sentinel_data.vnext.r4_v3_versions import GROUPING_VERSION_V3


def _write_meta(directory, sha, *, norm, addresses=(), source_records=()):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{sha}.meta.json").write_text(
        json.dumps(
            {
                "sha256": sha,
                "normalized_code_sha256": norm,
                "address_literals": list(addresses),
                "source_records": list(source_records),
            }
        )
    )


def test_shared_address_is_diagnostic_only_in_v3(tmp_path):
    source = tmp_path / "dive"
    common = "0x7a250d5630b4cf539739df2c5dacb4c659f2488d"
    _write_meta(source, "a" * 64, norm="1" * 64, addresses=(common,))
    _write_meta(source, "b" * 64, norm="2" * 64, addresses=(common,))

    out = tmp_path / "groups.json"
    result = build_grouping_v3({"dive": source}, out, verify_completeness=False)
    payload = json.loads(out.read_text())

    assert result.groups == 2
    assert result.address_edges == 0
    assert payload["grouping_version"] == GROUPING_VERSION_V3
    assert payload["address_diagnostics"]["used_as_grouping_authority"] is False
    assert payload["address_diagnostics"]["multi_artifact_address_keys"] == 1
    assert not any(
        edge["reason"] == "same_source_shared_address_candidate"
        for edge in payload["evidence_edges"]
    )


def test_normalized_identity_remains_global_grouping_authority(tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    _write_meta(a, "a" * 64, norm="n" * 64)
    _write_meta(b, "b" * 64, norm="n" * 64)

    out = tmp_path / "groups.json"
    result = build_grouping_v3({"a": a, "b": b}, out, verify_completeness=False)
    payload = json.loads(out.read_text())

    assert result.groups == 1
    assert result.normalized_edges == 1
    assert payload["artifact_to_group"]["a" * 64] == payload["artifact_to_group"]["b" * 64]


def test_explicit_family_remains_grouping_authority_within_source(tmp_path):
    source = tmp_path / "source"
    record = lambda value: {"ingestion_entry": {"base_family_id": value}}
    _write_meta(source, "a" * 64, norm="1" * 64, source_records=(record("family-7"),))
    _write_meta(source, "b" * 64, norm="2" * 64, source_records=(record("family-7"),))

    out = tmp_path / "groups.json"
    result = build_grouping_v3({"source": source}, out, verify_completeness=False)
    payload = json.loads(out.read_text())

    assert result.groups == 1
    assert result.explicit_family_edges == 1
    edge = next(
        edge
        for edge in payload["evidence_edges"]
        if edge["reason"] == "explicit_source_family"
    )
    assert edge["evidence_key"] == "source:base_family_id:family-7"


def test_same_explicit_family_value_from_different_sources_does_not_merge(tmp_path):
    source_a = tmp_path / "source-a"
    source_b = tmp_path / "source-b"
    record = lambda value: {"ingestion_entry": {"project_id": value}}
    _write_meta(
        source_a,
        "a" * 64,
        norm="1" * 64,
        source_records=(record("project-1"),),
    )
    _write_meta(
        source_b,
        "b" * 64,
        norm="2" * 64,
        source_records=(record("project-1"),),
    )

    out = tmp_path / "groups.json"
    result = build_grouping_v3(
        {"source-a": source_a, "source-b": source_b},
        out,
        verify_completeness=False,
    )
    payload = json.loads(out.read_text())

    assert result.groups == 2
    assert result.explicit_family_edges == 0
    assert payload["artifact_to_group"]["a" * 64] != payload["artifact_to_group"]["b" * 64]


def test_v3_group_ids_are_order_independent(tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    _write_meta(a, "a" * 64, norm="n" * 64)
    _write_meta(b, "b" * 64, norm="n" * 64)

    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    build_grouping_v3({"a": a, "b": b}, first, verify_completeness=False)
    build_grouping_v3({"b": b, "a": a}, second, verify_completeness=False)

    assert json.loads(first.read_text())["artifact_to_group"] == json.loads(second.read_text())["artifact_to_group"]
