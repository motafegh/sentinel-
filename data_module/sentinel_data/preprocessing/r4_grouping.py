"""Deterministic leakage-family grouping for repaired Phase-8 DATA.

Grouping is intentionally distinct from deduplication. Every content-distinct
artifact survives preprocessing; this module only decides which artifacts must
stay together when roles/splits are assigned.

Conservative grouping evidence:

* the same normalized-text SHA appearing in multiple sources is one contract
  identity with multiple source claims;
* identical normalized-code identity joins distinct artifact bytes globally;
* an explicit provenance family key joins artifacts carrying that key;
* a shared Ethereum address joins artifacts only *within the same source*.

The address rule prevents same-source variant families from being split across
roles, but never removes a record and never treats an address as label truth.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from sentinel_data.preprocessing.r4_versions import GROUPING_VERSION

_EXPLICIT_FAMILY_KEYS = (
    "base_family_id",
    "family_id",
    "project_group_id",
    "project_id",
)


class _UnionFind:
    def __init__(self, values: Iterable[str]):
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        a, b = self.find(left), self.find(right)
        if a == b:
            return
        small, large = sorted((a, b))
        self.parent[large] = small


@dataclass(frozen=True)
class GroupingResult:
    artifacts: int
    source_artifact_records: int
    groups: int
    normalized_edges: int
    address_edges: int
    explicit_family_edges: int
    cross_source_exact_identities: int
    output_path: str


def _group_id(members: list[str]) -> str:
    payload = GROUPING_VERSION + "\0" + "\0".join(sorted(members))
    return "r4grp-" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def _load_meta(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not value.get("sha256"):
        raise ValueError(f"invalid repaired preprocessing meta: {path}")
    return value


def _explicit_family_values(meta: dict[str, Any]) -> set[str]:
    values: set[str] = set()
    for record in meta.get("source_records", []):
        entry = record.get("ingestion_entry") or {}
        for key in _EXPLICIT_FAMILY_KEYS:
            value = entry.get(key)
            if value not in (None, ""):
                values.add(f"{key}:{value}")
    return values


def build_grouping(
    source_dirs: dict[str, Path],
    output_path: Path,
) -> GroupingResult:
    """Build and write the repaired group manifest from preprocessing metadata."""

    # One normalized-text SHA is one contract identity even if exact bytes were
    # observed in multiple source corpora. Keep every source-specific meta so
    # semantic claims are preserved, but use the SHA only once in group space.
    metas_by_artifact: dict[str, list[tuple[str, dict[str, Any]]]] = {}
    source_artifact_records = 0
    for source, directory in sorted(source_dirs.items()):
        for path in sorted(directory.glob("*.meta.json")):
            meta = _load_meta(path)
            artifact = str(meta["sha256"])
            metas_by_artifact.setdefault(artifact, []).append((source, meta))
            source_artifact_records += 1

    uf = _UnionFind(metas_by_artifact)
    normalized_index: dict[str, list[str]] = {}
    source_address_index: dict[tuple[str, str], list[str]] = {}
    explicit_index: dict[str, list[str]] = {}
    sources_by_artifact: dict[str, set[str]] = {
        artifact: {source for source, _ in entries}
        for artifact, entries in metas_by_artifact.items()
    }

    for artifact, entries in sorted(metas_by_artifact.items()):
        normalized_values = {
            str(meta.get("normalized_code_sha256") or "")
            for _, meta in entries
            if meta.get("normalized_code_sha256")
        }
        if len(normalized_values) > 1:
            raise ValueError(
                f"cross-source exact artifact {artifact} has conflicting normalized-code identities: "
                f"{sorted(normalized_values)}"
            )
        for normalized in normalized_values:
            normalized_index.setdefault(normalized, []).append(artifact)

        for source, meta in entries:
            for address in meta.get("address_literals") or []:
                source_address_index.setdefault(
                    (source, str(address).lower()), []
                ).append(artifact)
            for value in _explicit_family_values(meta):
                explicit_index.setdefault(value, []).append(artifact)

    evidence_edges: list[dict[str, str]] = []

    def join_members(reason: str, key: str, members: list[str]) -> int:
        unique = sorted(set(members))
        edges = 0
        for left, right in zip(unique, unique[1:]):
            uf.union(left, right)
            evidence_edges.append(
                {"reason": reason, "evidence_key": key, "left": left, "right": right}
            )
            edges += 1
        return edges

    normalized_edges = sum(
        join_members("normalized_code_identity", key, members)
        for key, members in sorted(normalized_index.items())
        if len(set(members)) > 1
    )
    explicit_edges = sum(
        join_members("explicit_source_family", key, members)
        for key, members in sorted(explicit_index.items())
        if len(set(members)) > 1
    )
    address_edges = sum(
        join_members(
            "same_source_shared_address_candidate",
            f"{source}:{address}",
            members,
        )
        for (source, address), members in sorted(source_address_index.items())
        if len(set(members)) > 1
    )

    components: dict[str, list[str]] = {}
    for artifact in sorted(metas_by_artifact):
        components.setdefault(uf.find(artifact), []).append(artifact)

    artifact_to_group: dict[str, str] = {}
    groups: list[dict[str, Any]] = []
    for members in sorted((sorted(v) for v in components.values()), key=lambda x: x[0]):
        group_id = _group_id(members)
        for artifact in members:
            artifact_to_group[artifact] = group_id
        groups.append(
            {
                "group_id": group_id,
                "members": members,
                "sources": sorted(
                    {source for item in members for source in sources_by_artifact[item]}
                ),
            }
        )

    cross_source_exact = {
        artifact: sorted(sources)
        for artifact, sources in sorted(sources_by_artifact.items())
        if len(sources) > 1
    }
    payload = {
        "status": "GROUPS_BUILT_PHYSICAL_ROLE_FREEZE_PENDING",
        "grouping_version": GROUPING_VERSION,
        "artifacts": len(metas_by_artifact),
        "source_artifact_records": source_artifact_records,
        "cross_source_exact_identities": cross_source_exact,
        "groups": groups,
        "artifact_to_group": dict(sorted(artifact_to_group.items())),
        "artifact_sources": {
            artifact: sorted(sources)
            for artifact, sources in sorted(sources_by_artifact.items())
        },
        "evidence_edges": sorted(
            evidence_edges,
            key=lambda row: (
                row["reason"], row["evidence_key"], row["left"], row["right"]
            ),
        ),
        "policy_notes": [
            "Grouping does not delete artifacts or alter label truth.",
            "Cross-source exact identity is one contract identity with all source claims preserved.",
            "Shared-address evidence is source-scoped and used only to prevent leakage-family role splitting.",
            "Cross-source address coincidence alone does not create a group edge.",
            "Role assignment must consume artifact_to_group atomically after this grouping is final."
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return GroupingResult(
        artifacts=len(metas_by_artifact),
        source_artifact_records=source_artifact_records,
        groups=len(groups),
        normalized_edges=normalized_edges,
        address_edges=address_edges,
        explicit_family_edges=explicit_edges,
        cross_source_exact_identities=len(cross_source_exact),
        output_path=str(output_path),
    )
