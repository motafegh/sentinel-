"""Fail-closed Solidity target selection for repaired representations.

Historical graph extraction used a heuristic over every non-interface Slither
declaration, which allowed libraries such as ``SafeMath`` to become the graph
root.  The repaired path resolves a target *before* Slither extraction and then
verifies that the resulting graph names the same declaration.

This module intentionally does not guess among multiple application contracts.
If provenance cannot identify one target, the repaired build fails closed and
records the ambiguity for local adjudication.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from sentinel_data.preprocessing.normalizer import strip_comments_lexically

_DECL_RE = re.compile(
    r"\b(?:(abstract)\s+)?(contract|library|interface)\s+([A-Za-z_][A-Za-z0-9_]*)\b"
)


class TargetSelectionError(ValueError):
    """Raised when the repaired graph target is absent or ambiguous."""


@dataclass(frozen=True)
class SolidityDeclaration:
    name: str
    kind: str  # contract | abstract_contract | library | interface
    source_offset: int


def _mask_strings(source: str) -> str:
    """Replace string contents with spaces while preserving source offsets."""

    chars = list(source)
    quote: str | None = None
    i = 0
    while i < len(chars):
        ch = chars[i]
        if quote is None:
            if ch in {"'", '"'}:
                quote = ch
                chars[i] = " "
            i += 1
            continue
        if ch == "\\" and i + 1 < len(chars):
            chars[i] = " "
            chars[i + 1] = " "
            i += 2
            continue
        chars[i] = "\n" if ch == "\n" else " "
        if ch == quote:
            quote = None
        i += 1
    return "".join(chars)


def declarations(source: str) -> tuple[SolidityDeclaration, ...]:
    """Return top-level declaration tokens visible to the repaired selector.

    This is deliberately a lexical declaration inventory rather than a Solidity
    parser.  Comments and strings are masked first, avoiding false ``contract``
    tokens from documentation or string constants.  Final compilation/Slither
    parsing remains authoritative for syntax.
    """

    no_comments, _ = strip_comments_lexically(source)
    code_only = _mask_strings(no_comments)
    found: list[SolidityDeclaration] = []
    for match in _DECL_RE.finditer(code_only):
        abstract, keyword, name = match.groups()
        kind = "abstract_contract" if keyword == "contract" and abstract else keyword
        found.append(
            SolidityDeclaration(
                name=name,
                kind=kind,
                source_offset=match.start(),
            )
        )
    return tuple(found)


def resolve_target_contract(
    source: str,
    *,
    explicit_target: str | None = None,
    provenance_contract_names: tuple[str, ...] | list[str] = (),
) -> str:
    """Resolve one application-contract target or fail closed.

    Selection precedence:

    1. an explicit provenance target, if it names a contract declaration;
    2. a unique contract intersecting ``provenance_contract_names``;
    3. the sole contract declaration in the file.

    Libraries and interfaces are never application targets.  Multiple remaining
    contracts are an ambiguity, not permission to reintroduce a heuristic.
    """

    decls = declarations(source)
    contracts = [
        item for item in decls if item.kind in {"contract", "abstract_contract"}
    ]
    libraries = [item.name for item in decls if item.kind == "library"]
    interfaces = [item.name for item in decls if item.kind == "interface"]

    by_name = {item.name: item for item in contracts}
    if explicit_target:
        if explicit_target in by_name:
            return explicit_target
        other_kind = next(
            (item.kind for item in decls if item.name == explicit_target),
            None,
        )
        if other_kind:
            raise TargetSelectionError(
                f"requested target {explicit_target!r} is a {other_kind}, not an application contract"
            )
        raise TargetSelectionError(
            f"requested target {explicit_target!r} not found; "
            f"contracts={sorted(by_name)}, libraries={libraries}, interfaces={interfaces}"
        )

    provenance_names = set(provenance_contract_names)
    provenance_matches = [item.name for item in contracts if item.name in provenance_names]
    if len(provenance_matches) == 1:
        return provenance_matches[0]
    if len(provenance_matches) > 1:
        raise TargetSelectionError(
            "provenance names identify multiple application contracts: "
            f"{provenance_matches}; explicit target provenance is required"
        )

    if len(contracts) == 1:
        return contracts[0].name
    if not contracts:
        raise TargetSelectionError(
            "no application contract declaration found; "
            f"libraries={libraries}, interfaces={interfaces}"
        )
    raise TargetSelectionError(
        "multiple application contracts remain ambiguous: "
        f"{[item.name for item in contracts]}; explicit target provenance is required"
    )
