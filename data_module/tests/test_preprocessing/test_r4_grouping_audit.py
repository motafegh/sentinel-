from __future__ import annotations

from sentinel_data.preprocessing.r4_grouping_audit import audit_grouping_payload


def test_transitive_address_group_requests_review():
    payload = {
        "grouping_version": "r4-leakage-groups-v2",
        "groups": [
            {
                "group_id": "g1",
                "members": ["a", "b", "c"],
                "sources": ["dive"],
            }
        ],
        "artifact_to_group": {"a": "g1", "b": "g1", "c": "g1"},
        "evidence_edges": [
            {
                "reason": "same_source_shared_address_candidate",
                "evidence_key": "dive:0x111",
                "left": "a",
                "right": "b",
            },
            {
                "reason": "same_source_shared_address_candidate",
                "evidence_key": "dive:0x222",
                "left": "b",
                "right": "c",
            },
        ],
    }
    report = audit_grouping_payload(
        payload,
        high_frequency_address_threshold=2,
        large_group_threshold=3,
    )
    assert report["review_required"] is True
    assert report["transitive_multi_address_groups"][0]["group_id"] == "g1"
    assert report["address_connected_large_groups"][0]["member_count"] == 3


def test_normalized_identity_only_does_not_trigger_address_review():
    payload = {
        "grouping_version": "r4-leakage-groups-v2",
        "groups": [
            {
                "group_id": "g1",
                "members": ["a", "b"],
                "sources": ["dive", "solidifi"],
            }
        ],
        "artifact_to_group": {"a": "g1", "b": "g1"},
        "evidence_edges": [
            {
                "reason": "normalized_code_identity",
                "evidence_key": "norm",
                "left": "a",
                "right": "b",
            }
        ],
    }
    report = audit_grouping_payload(
        payload,
        high_frequency_address_threshold=2,
        large_group_threshold=2,
    )
    assert report["review_required"] is False
    assert report["high_frequency_address_keys"] == []
    assert report["transitive_multi_address_groups"] == []
