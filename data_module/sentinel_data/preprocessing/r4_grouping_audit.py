"""Read-only breadth diagnostics for repaired R4 leakage-family grouping.

The accepted repaired-v2 grouping remains immutable evidence.  This module
profiles whether source-scoped shared-address edges create unexpectedly broad
or transitive leakage families.  Flags are evidence requests, not automatic
claims that grouping is wrong.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

ADDRESS_REASON = "same_source_shared_address_candidate"


def _quantiles(values: list[int]) -> dict[str, int]:
    if not values:
        return {}
    ordered = sorted(int(value) for value in values)

    def at(q: float) -> int:
        return ordered[min(len(ordered) - 1, round((len(ordered) - 1) * q))]

    return {
        "min": ordered[0],
        "p50": at(0.50),
        "p95": at(0.95),
        "p99": at(0.99),
        "max": ordered[-1],
    }


def audit_grouping_payload(
    payload: dict[str, Any],
    *,
    high_frequency_address_threshold: int = 20,
    large_group_threshold: int = 20,
    top_n: int = 25,
) -> dict[str, Any]:
    """Profile group breadth and evidence-key connectivity.

    ``high_frequency_address_threshold`` and ``large_group_threshold`` are
    diagnostic review triggers only.  They do not mutate grouping and do not
    declare a group invalid.
    """

    if high_frequency_address_threshold < 2:
        raise ValueError("high_frequency_address_threshold must be >= 2")
    if large_group_threshold < 2:
        raise ValueError("large_group_threshold must be >= 2")
    if top_n < 1:
        raise ValueError("top_n must be >= 1")

    groups = list(payload.get("groups") or [])
    artifact_to_group = {
        str(key): str(value)
        for key, value in (payload.get("artifact_to_group") or {}).items()
    }
    if not groups or not artifact_to_group:
        raise ValueError("grouping payload lacks groups/artifact_to_group")

    group_members: dict[str, set[str]] = {}
    group_sources: dict[str, list[str]] = {}
    for raw in groups:
        group_id = str(raw["group_id"])
        members = {str(value) for value in (raw.get("members") or [])}
        if not members:
            raise ValueError(f"group {group_id} has no members")
        group_members[group_id] = members
        group_sources[group_id] = [str(value) for value in (raw.get("sources") or [])]
        for member in members:
            if artifact_to_group.get(member) != group_id:
                raise ValueError(
                    f"artifact_to_group mismatch for {member}: "
                    f"{artifact_to_group.get(member)!r} != {group_id!r}"
                )

    reason_counts: Counter[str] = Counter()
    evidence_key_members: dict[tuple[str, str], set[str]] = defaultdict(set)
    group_reason_counts: dict[str, Counter[str]] = defaultdict(Counter)
    group_address_keys: dict[str, set[str]] = defaultdict(set)

    for edge in payload.get("evidence_edges") or []:
        reason = str(edge.get("reason") or "")
        key = str(edge.get("evidence_key") or "")
        left = str(edge.get("left") or "")
        right = str(edge.get("right") or "")
        if not reason or not key or not left or not right:
            raise ValueError(f"invalid grouping evidence edge: {edge!r}")
        left_group = artifact_to_group.get(left)
        right_group = artifact_to_group.get(right)
        if left_group is None or right_group is None:
            raise ValueError(f"edge references unknown artifact: {edge!r}")
        if left_group != right_group:
            raise ValueError(
                f"evidence edge crosses final groups: {left_group} != {right_group}"
            )
        reason_counts[reason] += 1
        evidence_key_members[(reason, key)].update((left, right))
        group_reason_counts[left_group][reason] += 1
        if reason == ADDRESS_REASON:
            group_address_keys[left_group].add(key)

    group_sizes = [len(members) for members in group_members.values()]
    largest = sorted(
        group_members,
        key=lambda gid: (-len(group_members[gid]), gid),
    )[:top_n]
    largest_groups = [
        {
            "group_id": group_id,
            "member_count": len(group_members[group_id]),
            "sources": sorted(group_sources.get(group_id, [])),
            "edge_reasons": dict(sorted(group_reason_counts[group_id].items())),
            "address_key_count": len(group_address_keys[group_id]),
            "members_preview": sorted(group_members[group_id])[:10],
        }
        for group_id in largest
    ]

    address_keys = [
        {
            "evidence_key": key,
            "artifact_count": len(members),
            "artifacts_preview": sorted(members)[:10],
            "group_ids": sorted(
                {
                    artifact_to_group[artifact]
                    for artifact in members
                    if artifact in artifact_to_group
                }
            ),
        }
        for (reason, key), members in evidence_key_members.items()
        if reason == ADDRESS_REASON
    ]
    address_keys.sort(
        key=lambda row: (-int(row["artifact_count"]), str(row["evidence_key"]))
    )

    high_frequency_addresses = [
        row
        for row in address_keys
        if int(row["artifact_count"]) >= high_frequency_address_threshold
    ]
    large_groups = [
        row
        for row in largest_groups
        if int(row["member_count"]) >= large_group_threshold
    ]

    address_only_groups: list[dict[str, Any]] = []
    address_connected_large_groups: list[dict[str, Any]] = []
    transitive_address_groups: list[dict[str, Any]] = []
    for group_id, members in sorted(group_members.items()):
        if len(members) <= 1:
            continue
        reasons = set(group_reason_counts[group_id])
        record = {
            "group_id": group_id,
            "member_count": len(members),
            "sources": sorted(group_sources.get(group_id, [])),
            "address_key_count": len(group_address_keys[group_id]),
            "edge_reasons": dict(sorted(group_reason_counts[group_id].items())),
        }
        if reasons and reasons <= {ADDRESS_REASON}:
            address_only_groups.append(record)
        if len(members) >= large_group_threshold and ADDRESS_REASON in reasons:
            address_connected_large_groups.append(record)
        if len(group_address_keys[group_id]) > 1 and len(members) > 2:
            transitive_address_groups.append(record)

    address_only_groups.sort(
        key=lambda row: (-int(row["member_count"]), str(row["group_id"]))
    )
    address_connected_large_groups.sort(
        key=lambda row: (-int(row["member_count"]), str(row["group_id"]))
    )
    transitive_address_groups.sort(
        key=lambda row: (
            -int(row["member_count"]),
            -int(row["address_key_count"]),
            str(row["group_id"]),
        )
    )

    review_required = bool(
        high_frequency_addresses
        or address_connected_large_groups
        or transitive_address_groups
    )
    return {
        "schema": "sentinel-r4-grouping-breadth-audit-v1",
        "grouping_version": payload.get("grouping_version"),
        "automatic_defect_claim": False,
        "review_required": review_required,
        "thresholds": {
            "high_frequency_address_artifacts": high_frequency_address_threshold,
            "large_group_members": large_group_threshold,
        },
        "population": {
            "artifacts": len(artifact_to_group),
            "groups": len(group_members),
            "group_size_quantiles": _quantiles(group_sizes),
        },
        "evidence_edge_counts_by_reason": dict(sorted(reason_counts.items())),
        "address_evidence_keys": len(address_keys),
        "high_frequency_address_keys": high_frequency_addresses[:top_n],
        "address_only_multi_member_groups": address_only_groups[:top_n],
        "address_connected_large_groups": address_connected_large_groups[:top_n],
        "transitive_multi_address_groups": transitive_address_groups[:top_n],
        "largest_groups": largest_groups,
        "decision_boundary": (
            "Flags request local evidence review. They do not invalidate accepted "
            "repaired-v2 grouping. Any policy change must use a new grouping/"
            "partition version rather than rewriting accepted v2 evidence."
        ),
    }


__all__ = ["ADDRESS_REASON", "audit_grouping_payload"]
