"""Fail-closed Solidity target selection for repaired representations.

Historical graph extraction used a heuristic over every non-interface Slither
declaration, which allowed libraries such as ``SafeMath`` to become the graph
root.  The repaired path resolves a target *before* Slither extraction and then
verifies that the resulting graph names the same declaration.

This module intentionally does not guess among unrelated application contracts.
It can, however, resolve the unique inheritance leaf: a contract that is not a
base of another contract declared in the same file.  That rule is structural,
deterministic, and prevents common base contracts from replacing the deployed
application contract.  Remaining ambiguity fails closed for local adjudication.
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
    base_names: tuple[str, ...] = ()


def _inheritance_bases(code_only: str, name_end: int) -> tuple[str, ...]:
    """Read top-level inheritance names between a declaration name and ``{``."""

    brace = code_only.find("{", name_end)
    if brace < 0:
        return ()
    header = code_only[name_end:brace]
    match = re.match(r"\s+is\s+(.+?)\s*$", header, flags=re.DOTALL)
    if not match:
        return ()
    value = match.group(1)
    parts: list[str] = []
    start = 0
    depth = 0
    for index, character in enumerate(value):
        if character == "(":
            depth += 1
        elif character == ")" and depth:
            depth -= 1
        elif character == "," and depth == 0:
            parts.append(value[start:index])
            start = index + 1
    parts.append(value[start:])
    names: list[str] = []
    for part in parts:
        base = re.match(
            r"\s*([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*)",
            part,
        )
        if base:
            names.append(base.group(1).split(".")[-1])
    return tuple(names)


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
                base_names=_inheritance_bases(code_only, match.end()),
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
    2. the unique inheritance leaf among provenance-named contracts;
    3. the unique inheritance leaf among all contract declarations.

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
    if not contracts:
        raise TargetSelectionError(
            "no application contract declaration found; "
            f"libraries={libraries}, interfaces={interfaces}"
        )
    candidates = provenance_matches or [item.name for item in contracts]
    inherited_names = {
        base_name
        for item in contracts
        for base_name in item.base_names
    }
    leaves = [name for name in candidates if name not in inherited_names]
    if len(leaves) == 1:
        return leaves[0]
    raise TargetSelectionError(
        "multiple application contracts remain ambiguous after unique-inheritance-leaf selection: "
        f"candidates={candidates}, leaves={leaves}; explicit target provenance is required"
    )


def resolve_file_graph_targets(
    source: str,
    *,
    explicit_target: str | None = None,
    provenance_contract_names: tuple[str, ...] | list[str] = (),
) -> tuple[str, ...]:
    """Return the evidence-preserving target set for one file-level sample.

    A unique explicit target remains authoritative.  Otherwise every unrelated
    inheritance leaf is retained so a file-level label is not silently assigned
    to one guessed contract.  Inheritance parents are represented through each
    leaf by Slither.  Library-only files retain their libraries; interface-only
    files fail because they contain no executable implementation graph.
    """

    decls = declarations(source)
    if explicit_target:
        return (
            resolve_target_contract(
                source,
                explicit_target=explicit_target,
                provenance_contract_names=provenance_contract_names,
            ),
        )
    contracts = [
        item for item in decls if item.kind in {"contract", "abstract_contract"}
    ]
    provenance_names = set(provenance_contract_names)
    candidates = [item for item in contracts if item.name in provenance_names]
    if not candidates:
        candidates = contracts
    inherited_names = {
        base_name
        for item in contracts
        for base_name in item.base_names
    }
    leaves = tuple(item.name for item in candidates if item.name not in inherited_names)
    if leaves:
        return leaves
    libraries = tuple(item.name for item in decls if item.kind == "library")
    if not contracts and libraries:
        return libraries
    interfaces = [item.name for item in decls if item.kind == "interface"]
    raise TargetSelectionError(
        "no executable file-graph target found; "
        f"contracts={[item.name for item in contracts]}, interfaces={interfaces}"
    )


__all__ = [
    "SolidityDeclaration",
    "TargetSelectionError",
    "declarations",
    "resolve_file_graph_targets",
    "resolve_target_contract",
]
