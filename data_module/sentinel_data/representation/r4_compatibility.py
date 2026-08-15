"""Explicit Slither compatibility recovery for repaired R4 representations.

The normal graph path remains full Slither analysis of the exact promoted
Solidity bytes.  A small real-data tail is compile-valid but triggers known
Slither analysis/parser defects.  This module provides narrow, recorded
fallbacks without changing the token input or silently dropping artifacts.
"""

from __future__ import annotations

import ast
import hashlib
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sentinel_data.representation.graph_extractor import (
    GraphExtractionConfig,
    GraphExtractionError,
    extract_contract_graph,
)
from sentinel_data.representation.target_selector import TargetSelectionError


FULL_ANALYSIS = "slither_full_analysis"
PARSE_ONLY = "slither_parse_only"
FULL_ANALYSIS_CONSTANT_FOLD = "slither_full_analysis_constant_array_fold_v1"
PARSE_ONLY_CONSTANT_FOLD = "slither_parse_only_constant_array_fold_v1"
COMPATIBILITY_MODES = frozenset(
    {
        FULL_ANALYSIS,
        PARSE_ONLY,
        FULL_ANALYSIS_CONSTANT_FOLD,
        PARSE_ONLY_CONSTANT_FOLD,
    }
)


class CompatibilityExtractionError(RuntimeError):
    """All explicit graph compatibility attempts failed."""


@dataclass(frozen=True)
class CompatibilityExtraction:
    graphs: tuple[Any, ...]
    actual_targets: tuple[str, ...]
    mode: str
    fallback_errors: tuple[dict[str, str], ...]
    source_transform: dict[str, Any] | None = None

    @property
    def analysis_degraded(self) -> bool:
        return self.mode in {PARSE_ONLY, PARSE_ONLY_CONSTANT_FOLD}


_CONSTANT_DECLARATION = re.compile(
    r"\b[A-Za-z_]\w*(?:\s*\[[^\]\n]*\])?\s+"
    r"(?P<qualifiers>(?:(?:public|private|internal|external|constant)\s+)*)"
    r"(?P<name>[A-Za-z_]\w*)\s*=\s*(?P<expr>[^;\n]+);"
)
_FIXED_ARRAY_DECLARATION = re.compile(
    r"(?P<type>\b(?:u?int(?:[0-9]+)?|bytes(?:[0-9]+)?|address|bool|string|"
    r"[A-Z][A-Za-z0-9_]*)\s*)"
    r"\[(?P<expr>[^\]\n]+)\]"
    r"(?P<suffix>\s+(?:(?:public|private|internal|external|constant|immutable)\s+)*"
    r"[A-Za-z_]\w*\s*(?:[;=]))"
)
_ACTIVE_IMPORT = re.compile(r"^\s*import\b", re.MULTILINE)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _evaluate_constant(node: ast.AST, names: dict[str, int]) -> int:
    if isinstance(node, ast.Expression):
        return _evaluate_constant(node.body, names)
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return int(node.value)
    if isinstance(node, ast.Name) and node.id in names:
        return names[node.id]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _evaluate_constant(node.operand, names)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp):
        left = _evaluate_constant(node.left, names)
        right = _evaluate_constant(node.right, names)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, (ast.Div, ast.FloorDiv)):
            if right == 0:
                raise ValueError("division by zero")
            return left // right
        if isinstance(node.op, ast.Mod):
            if right == 0:
                raise ValueError("modulo by zero")
            return left % right
    raise ValueError(f"unsupported constant expression node: {type(node).__name__}")


def _safe_constant_value(expression: str, names: dict[str, int]) -> int:
    parsed = ast.parse(expression.strip(), mode="eval")
    value = _evaluate_constant(parsed, names)
    if value <= 0 or value > 10_000_000:
        raise ValueError(f"fixed-array length is outside the accepted range: {value}")
    return value


def fold_constant_array_lengths(source: str) -> tuple[str, list[dict[str, Any]]]:
    """Fold only compile-time fixed-array declaration lengths.

    Replacements are right-padded to preserve byte offsets and line numbers.
    Indexing expressions and dynamic arrays are not matched.  Names are
    accepted only when they resolve from an integer ``constant`` declaration.
    """

    constants: dict[str, int] = {}
    pending: list[tuple[str, str]] = []
    for match in _CONSTANT_DECLARATION.finditer(source):
        if "constant" not in match.group("qualifiers").split():
            continue
        pending.append((match.group("name"), match.group("expr")))

    while pending:
        unresolved: list[tuple[str, str]] = []
        changed = False
        for name, expression in pending:
            try:
                constants[name] = _safe_constant_value(expression, constants)
                changed = True
            except (SyntaxError, ValueError):
                unresolved.append((name, expression))
        if not changed:
            break
        pending = unresolved

    replacements: list[tuple[int, int, str, dict[str, Any]]] = []
    for match in _FIXED_ARRAY_DECLARATION.finditer(source):
        expression = match.group("expr")
        try:
            value = _safe_constant_value(expression, constants)
        except (SyntaxError, ValueError):
            continue
        literal = str(value)
        if len(literal) > len(expression):
            continue
        start, end = match.span("expr")
        replacement = literal.ljust(end - start)
        replacements.append(
            (
                start,
                end,
                replacement,
                {
                    "line": source.count("\n", 0, start) + 1,
                    "expression": expression.strip(),
                    "value": value,
                },
            )
        )

    transformed = source
    for start, end, replacement, _ in reversed(replacements):
        transformed = transformed[:start] + replacement + transformed[end:]
    if len(transformed.encode("utf-8")) != len(source.encode("utf-8")):
        raise CompatibilityExtractionError(
            "constant-array compatibility transform changed source byte length"
        )
    if transformed.count("\n") != source.count("\n"):
        raise CompatibilityExtractionError(
            "constant-array compatibility transform changed source line count"
        )
    return transformed, [record for *_, record in replacements]


def _cause_names(exc: BaseException) -> set[str]:
    names: set[str] = set()
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        names.add(type(current).__name__)
        current = current.__cause__ or current.__context__
    return names


def _error_record(exc: BaseException) -> dict[str, str]:
    return {
        "error_type": type(exc).__name__,
        "error": str(exc),
        "cause_types": ",".join(sorted(_cause_names(exc))),
    }


def _extract_components(
    sol_path: Path,
    targets: tuple[str, ...],
    *,
    solc_binary: Path | None,
    solc_version: str,
    skip_analyze: bool,
    allow_paths: str,
) -> tuple[tuple[Any, ...], tuple[str, ...]]:
    graphs: list[Any] = []
    actual_targets: list[str] = []
    for target in targets:
        config = GraphExtractionConfig(
            multi_contract_policy="by_name",
            target_contract_name=target,
            allow_paths=allow_paths,
            solc_binary=solc_binary,
            solc_version=solc_version,
            slither_skip_analyze=skip_analyze,
        )
        graph = extract_contract_graph(sol_path, config=config)
        actual = str(getattr(graph, "contract_name", ""))
        if actual != target:
            raise TargetSelectionError(
                f"graph target mismatch: requested={target!r}, actual={actual!r}"
            )
        graphs.append(graph)
        actual_targets.append(actual)
    return tuple(graphs), tuple(actual_targets)


def extract_components_with_compatibility(
    sol_path: Path,
    targets: tuple[str, ...],
    *,
    solc_binary: Path | None,
    solc_version: str,
) -> CompatibilityExtraction:
    """Extract all file targets with explicit, ordered compatibility modes."""

    sol_path = Path(sol_path)
    errors: list[dict[str, str]] = []
    common = {
        "solc_binary": solc_binary,
        "solc_version": solc_version,
        "allow_paths": str(sol_path.parent),
    }
    try:
        graphs, actual = _extract_components(
            sol_path, targets, skip_analyze=False, **common
        )
        return CompatibilityExtraction(graphs, actual, FULL_ANALYSIS, ())
    except GraphExtractionError as exc:
        errors.append(_error_record(exc))

    try:
        graphs, actual = _extract_components(
            sol_path, targets, skip_analyze=True, **common
        )
        return CompatibilityExtraction(
            graphs, actual, PARSE_ONLY, tuple(errors)
        )
    except GraphExtractionError as exc:
        errors.append(_error_record(exc))
        parse_only_error = exc

    if "NotConstant" not in _cause_names(parse_only_error):
        raise CompatibilityExtractionError(
            f"normal and parse-only Slither extraction failed for {sol_path.name}: "
            f"{errors}"
        ) from parse_only_error

    source = sol_path.read_text(encoding="utf-8")
    if _ACTIVE_IMPORT.search(source):
        raise CompatibilityExtractionError(
            "constant-array compatibility transform refuses source with active imports"
        ) from parse_only_error
    transformed, replacements = fold_constant_array_lengths(source)
    if not replacements or transformed == source:
        raise CompatibilityExtractionError(
            "Slither constant folding failed but no safe fixed-array replacement was found"
        ) from parse_only_error

    transform = {
        "schema": "r4-graph-source-compatibility-v1",
        "kind": "constant_array_length_fold",
        "original_sha256": _sha256_text(source),
        "transformed_sha256": _sha256_text(transformed),
        "byte_length_preserved": True,
        "line_count_preserved": True,
        "replacements": replacements,
    }
    with tempfile.TemporaryDirectory(prefix="sentinel-r4-graph-compat-") as tmp:
        transformed_path = Path(tmp) / sol_path.name
        transformed_path.write_text(transformed, encoding="utf-8")
        transformed_common = {
            "solc_binary": solc_binary,
            "solc_version": solc_version,
            "allow_paths": f"{sol_path.parent},{transformed_path.parent}",
        }
        try:
            graphs, actual = _extract_components(
                transformed_path, targets, skip_analyze=False, **transformed_common
            )
            return CompatibilityExtraction(
                graphs,
                actual,
                FULL_ANALYSIS_CONSTANT_FOLD,
                tuple(errors),
                transform,
            )
        except GraphExtractionError as exc:
            errors.append(_error_record(exc))
        try:
            graphs, actual = _extract_components(
                transformed_path, targets, skip_analyze=True, **transformed_common
            )
            return CompatibilityExtraction(
                graphs,
                actual,
                PARSE_ONLY_CONSTANT_FOLD,
                tuple(errors),
                transform,
            )
        except GraphExtractionError as exc:
            errors.append(_error_record(exc))
            raise CompatibilityExtractionError(
                f"all Slither compatibility modes failed for {sol_path.name}: {errors}"
            ) from exc


__all__ = [
    "COMPATIBILITY_MODES",
    "CompatibilityExtraction",
    "CompatibilityExtractionError",
    "FULL_ANALYSIS",
    "FULL_ANALYSIS_CONSTANT_FOLD",
    "PARSE_ONLY",
    "PARSE_ONLY_CONSTANT_FOLD",
    "extract_components_with_compatibility",
    "fold_constant_array_lengths",
]
