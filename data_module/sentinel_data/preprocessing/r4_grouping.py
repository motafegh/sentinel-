"""Deterministic leakage-family grouping for repaired Phase-8 DATA.

Grouping is intentionally distinct from deduplication.  Every content-distinct
artifact survives preprocessing; this module only decides which artifacts must
stay together when roles/splits are assigned.

Conservative grouping evidence:

* identical normalized-code identity joins artifacts across sources;
* an explicit provenance family key joins artifacts carrying that key;
* a shared Ethereum address joins artifacts only *within the same source*.

The address rule prevents the historical SolidiFI variant family from being
split across roles, but never removes a record and never treats the address as a
label/duplicate fact.  Cross-source address coincidence alone is not enough.
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
        # Lexicographic root makes the structure deterministic regardless of
        # input traversal order.
        small, large = sorted((a, b))
        self.parent[large] = small


@dataclass(frozen=True)
class GroupingResult:
    artifacts: int
    groups: int
    normalized_edges: int
    address_edges: int
    explicit_family_edges: int
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

    metas: dict[str, dict[str, Any]] = {}
    source_by_artifact: dict[str, str] = {}
    for source, directory in sorted(source_dirs.items()):
        for path in sorted(directory.glob("*.meta.json")):
            meta = _load_meta(path)
            artifact = str(meta["sha256"])
            if artifact in metas:
                raise ValueError(
                    f"artifact {artifact} appears in multiple source directories; "
                    "cross-source exact identity must be represented through provenance, not duplicate meta files"
                )
            metas[artifact] = meta
            source_by_artifact[artifact] = source

    uf = _UnionFind(metas)
    normalized_index: dict[str, list[str]] = {}
    source_address_index: dict[tuple[str, str], list[str]] = {}
    explicit_index: dict[str, list[str]] = {}

    for artifact, meta in sorted(metas.items()):
        normalized = str(meta.get("normalized_code_sha256") or "")
        if normalized:
            normalized_index.setdefault(normalized, []).append(artifact)
        for address in meta.get("address_literals") or []:
            source_address_index.setdefault(
                (source_by_artifact[artifact], str(address).lower()), []
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
    for artifact in sorted(metas):
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
                "sources": sorted({source_by_artifact[item] for item in members}),
            }
        )

    payload = {
        "status": "GROUPS_BUILT_PHYSICAL_ROLE_FREEZE_PENDING",
        "grouping_version": GROUPING_VERSION,
        "artifacts": len(metas),
        "groups": groups,
        "artifact_to_group": dict(sorted(artifact_to_group.items())),
        "evidence_edges": sorted(
            evidence_edges,
            key=lambda row: (row["reason"], row["evidence_key"], row["left"], row["right"]),
        ),
        "policy_notes": [
            "Grouping does not delete artifacts or alter label truth.",
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
        artifacts=len(metas),
        groups=len(groups),
        normalized_edges=normalized_edges,
        address_edges=address_edges,
        explicit_family_edges=explicit_edges,
        output_path=str(output_path),
    )
