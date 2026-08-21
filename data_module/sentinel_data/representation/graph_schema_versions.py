"""Version-aware graph-schema definitions for immutable v9 and candidate v10.

``graph_schema.py`` remains the historical/default v9 import surface.  New
representation lineages must select a schema explicitly through this module so
that adding v10 cannot silently reinterpret accepted v9 edge IDs.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping


_BASE_EDGE_TYPES: dict[str, int] = {
    "CALLS": 0,
    "READS": 1,
    "WRITES": 2,
    "EMITS": 3,
    "INHERITS": 4,
    "CONTAINS": 5,
    "CONTROL_FLOW": 6,
    "REVERSE_CONTAINS": 7,
    "CALL_ENTRY": 8,
    "RETURN_TO": 9,
    "DEF_USE": 10,
}


@dataclass(frozen=True)
class GraphSchemaDefinition:
    """Immutable edge-vocabulary and consumer semantics for one graph schema."""

    version: str
    extractor_version: str
    edge_types: Mapping[str, int]
    external_handoff_edge_names: tuple[str, ...]
    call_to_unknown_signal_edge_names: tuple[str, ...]
    library_edge_name: str | None = None

    @property
    def num_edge_types(self) -> int:
        values = tuple(self.edge_types.values())
        if not values:
            return 0
        expected = set(range(max(values) + 1))
        if set(values) != expected:
            raise ValueError(
                f"{self.version} edge IDs must be contiguous from zero: {values}"
            )
        return len(expected)

    def edge_ids(self, names: tuple[str, ...]) -> tuple[int, ...]:
        try:
            return tuple(self.edge_types[name] for name in names)
        except KeyError as exc:
            raise ValueError(
                f"{self.version} does not define required edge kind {exc.args[0]!r}"
            ) from exc


V9_EDGE_TYPES = MappingProxyType({**_BASE_EDGE_TYPES, "EXTERNAL_CALL": 11})

V10_EDGE_TYPES = MappingProxyType(
    {
        **_BASE_EDGE_TYPES,
        "HIGH_LEVEL_CALL": 11,
        "LOW_LEVEL_CALL": 12,
        "ETHER_TRANSFER": 13,
        "ETHER_SEND": 14,
        "LIBRARY_CALL": 15,
        "CONTRACT_CREATION": 16,
    }
)

V9_SCHEMA = GraphSchemaDefinition(
    version="v9",
    extractor_version="v2.2-r4-repaired",
    edge_types=V9_EDGE_TYPES,
    # Historical v9 semantics are recorded, not endorsed as adequate.
    external_handoff_edge_names=("EXTERNAL_CALL",),
    call_to_unknown_signal_edge_names=("EXTERNAL_CALL",),
)

V10_SCHEMA = GraphSchemaDefinition(
    version="v10",
    extractor_version="v2.3-r4-call-semantics",
    edge_types=V10_EDGE_TYPES,
    external_handoff_edge_names=(
        "HIGH_LEVEL_CALL",
        "LOW_LEVEL_CALL",
        "ETHER_TRANSFER",
        "ETHER_SEND",
        "CONTRACT_CREATION",
    ),
    # Coarse graph corroboration only; source review remains truth authority.
    call_to_unknown_signal_edge_names=("LOW_LEVEL_CALL", "ETHER_SEND"),
    library_edge_name="LIBRARY_CALL",
)

GRAPH_SCHEMAS: Mapping[str, GraphSchemaDefinition] = MappingProxyType(
    {V9_SCHEMA.version: V9_SCHEMA, V10_SCHEMA.version: V10_SCHEMA}
)


def get_graph_schema(version: str) -> GraphSchemaDefinition:
    """Return an exact schema definition or fail closed on unknown versions."""

    try:
        return GRAPH_SCHEMAS[version]
    except KeyError as exc:
        raise ValueError(
            f"unsupported graph schema {version!r}; expected one of "
            f"{sorted(GRAPH_SCHEMAS)}"
        ) from exc


__all__ = [
    "GraphSchemaDefinition",
    "GRAPH_SCHEMAS",
    "V9_EDGE_TYPES",
    "V9_SCHEMA",
    "V10_EDGE_TYPES",
    "V10_SCHEMA",
    "get_graph_schema",
]
