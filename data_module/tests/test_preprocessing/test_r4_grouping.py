"""Tests for deterministic repaired leakage-family grouping."""

from __future__ import annotations

import json

from sentinel_data.preprocessing.r4_grouping import build_grouping


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


def test_normalized_identity_groups_across_sources(tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    _write_meta(a, "a" * 64, norm="n" * 64)
    _write_meta(b, "b" * 64, norm="n" * 64)
    out = tmp_path / "groups.json"
    result = build_grouping({"a": a, "b": b}, out, verify_completeness=False)
    payload = json.loads(out.read_text())
    assert result.groups == 1
    assert payload["artifact_to_group"]["a" * 64] == payload["artifact_to_group"]["b" * 64]


def test_same_source_shared_address_groups_but_preserves_both_artifacts(tmp_path):
    source = tmp_path / "solidifi"
    address = "0x1111111111111111111111111111111111111111"
    _write_meta(source, "a" * 64, norm="1" * 64, addresses=(address,))
    _write_meta(source, "b" * 64, norm="2" * 64, addresses=(address,))
    out = tmp_path / "groups.json"
    result = build_grouping({"solidifi": source}, out, verify_completeness=False)
    payload = json.loads(out.read_text())
    assert result.artifacts == 2
    assert result.groups == 1
    assert payload["artifact_to_group"]["a" * 64] == payload["artifact_to_group"]["b" * 64]


def test_cross_source_shared_address_alone_does_not_merge(tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    address = "0x2222222222222222222222222222222222222222"
    _write_meta(a, "a" * 64, norm="1" * 64, addresses=(address,))
    _write_meta(b, "b" * 64, norm="2" * 64, addresses=(address,))
    out = tmp_path / "groups.json"
    result = build_grouping({"a": a, "b": b}, out, verify_completeness=False)
    assert result.groups == 2


def test_explicit_family_provenance_groups_variants(tmp_path):
    source = tmp_path / "source"
    record = lambda value: {
        "ingestion_entry": {"base_family_id": value}
    }
    _write_meta(source, "a" * 64, norm="1" * 64, source_records=(record("family-7"),))
    _write_meta(source, "b" * 64, norm="2" * 64, source_records=(record("family-7"),))
    out = tmp_path / "groups.json"
    result = build_grouping({"source": source}, out, verify_completeness=False)
    assert result.groups == 1
    assert result.explicit_family_edges == 1


def test_group_ids_are_order_independent(tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    _write_meta(a, "a" * 64, norm="n" * 64)
    _write_meta(b, "b" * 64, norm="n" * 64)
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    build_grouping({"a": a, "b": b}, first, verify_completeness=False)
    build_grouping({"b": b, "a": a}, second, verify_completeness=False)
    assert json.loads(first.read_text())["artifact_to_group"] == json.loads(second.read_text())["artifact_to_group"]
