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
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

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
FULL_ANALYSIS_SINGLETON_CALL_TYPE = (
    "slither_full_analysis_singleton_call_type_v1"
)
FULL_ANALYSIS_KNOWN_TERNARY_FOLD = (
    "slither_full_analysis_known_state_initializer_ternary_fold_v1"
)
FULL_ANALYSIS_KNOWN_DISCARDED_TUPLE_FOLD = (
    "slither_full_analysis_known_discarded_tuple_component_v1"
)
COMPATIBILITY_MODES = frozenset(
    {
        FULL_ANALYSIS,
        PARSE_ONLY,
        FULL_ANALYSIS_CONSTANT_FOLD,
        PARSE_ONLY_CONSTANT_FOLD,
        FULL_ANALYSIS_SINGLETON_CALL_TYPE,
        FULL_ANALYSIS_KNOWN_TERNARY_FOLD,
        FULL_ANALYSIS_KNOWN_DISCARDED_TUPLE_FOLD,
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
    analyzer_repair: dict[str, Any] | None = None

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
_SLITHER_PATCH_LOCK = threading.RLock()

# This is a source-identity-bound reconciliation, not a generic ternary
# rewriter. The state value is fixed to 10 before the next initializer is
# evaluated during construction. Keeping the entry hash-bound prevents a
# similar-looking but semantically different source from being rewritten.
_KNOWN_STATE_INITIALIZER_TERNARY_FOLDS: dict[str, dict[str, Any]] = {
    "9703409b36b5e3bf0fe83d25c3f3377c6fd618e07c343b8a4f70b846dc80c235": {
        "line": 1566,
        "precondition": "uint256 public maxPerWallet = 10;",
        "expression": "maxPerWallet < 50 ? maxPerWallet : 50",
        "value": 10,
    }
}

# Slither 0.11.5 regresses on these Solidity 0.4.x tuple declarations because
# its IR converter does not return the TupleVariable that its assignment path
# asserts. The omitted component is discarded and never named; converting the
# LHS to the one declared variable retains the compiled assignment semantics.
# The graph-only rewrite is hash-bound and whitespace padded so offsets remain
# stable.
_KNOWN_DISCARDED_TUPLE_COMPONENT_FOLDS: dict[str, dict[str, Any]] = {
    "e5bffc7bdc50d329cc39ce5e88a4ac2ccae3b993e8d57375910e4c7879706a2a": {
        "line": 230,
        "expression": "(bool success, )",
    },
    "85abc72e5fb510a7af30774792efed1eaa799303843a78bad1b893bdbd775d0b": {
        "line": 24,
        "expression": "(bool success, )",
    },
    "9309110859776539318d5d92b9109fedf0974ff144dca4a64bed78b5893e27ef": {
        "line": 19,
        "expression": "(bool success, )",
    },
}


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


def _is_singleton_call_type_failure(exc: BaseException) -> bool:
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, TypeError) and str(current) == "unhashable type: 'list'":
            return True
        current = current.__cause__ or current.__context__
    return False


def _has_assertion_failure(exc: BaseException) -> bool:
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, AssertionError):
            return True
        current = current.__cause__ or current.__context__
    return False


@contextmanager
def _slither_singleton_call_type_repair() -> Iterator[list[dict[str, str]]]:
    """Temporarily unwrap only singleton Solidity type lists on high-level calls."""

    import slither.slithir.convert as slither_convert
    from slither.core.solidity_types.type import Type
    from slither.slithir.operations import HighLevelCall

    original = slither_convert.propagate_types
    records: list[dict[str, str]] = []

    def repaired(ir: Any, node: Any) -> Any:
        if isinstance(ir, HighLevelCall):
            destination = getattr(ir, "destination", None)
            destination_type = getattr(destination, "type", None)
            if (
                isinstance(destination_type, list)
                and len(destination_type) == 1
                and isinstance(destination_type[0], Type)
            ):
                destination.set_type(destination_type[0])
                records.append(
                    {
                        "function": str(getattr(node, "function", "")),
                        "node": str(node),
                        "unwrapped_type": str(destination_type[0]),
                    }
                )
        return original(ir, node)

    # Slither resolves propagate_types through its module global. Guard and
    # restore the process-local patch even when analysis raises.
    with _SLITHER_PATCH_LOCK:
        slither_convert.propagate_types = repaired
        try:
            yield records
        finally:
            slither_convert.propagate_types = original


def fold_known_state_initializer_ternary(
    source: str,
) -> tuple[str, list[dict[str, Any]]]:
    """Fold one hash-bound construction-time initializer Slither cannot lower."""

    source_sha256 = _sha256_text(source)
    spec = _KNOWN_STATE_INITIALIZER_TERNARY_FOLDS.get(source_sha256)
    if spec is None:
        return source, []
    precondition = str(spec["precondition"])
    expression = str(spec["expression"])
    if source.count(precondition) != 1 or source.count(expression) != 1:
        raise CompatibilityExtractionError(
            "known ternary reconciliation source no longer matches its bound form"
        )
    expression_start = source.index(expression)
    if source.index(precondition) >= expression_start:
        raise CompatibilityExtractionError(
            "known ternary reconciliation precondition is not evaluated first"
        )
    line = source.count("\n", 0, expression_start) + 1
    if line != int(spec["line"]):
        raise CompatibilityExtractionError(
            "known ternary reconciliation line no longer matches its bound form"
        )
    literal = str(spec["value"])
    if len(literal) > len(expression):
        raise CompatibilityExtractionError(
            "known ternary reconciliation cannot preserve source byte length"
        )
    replacement = literal.ljust(len(expression))
    transformed = (
        source[:expression_start]
        + replacement
        + source[expression_start + len(expression) :]
    )
    if len(transformed.encode("utf-8")) != len(source.encode("utf-8")):
        raise CompatibilityExtractionError(
            "known ternary reconciliation changed source byte length"
        )
    if transformed.count("\n") != source.count("\n"):
        raise CompatibilityExtractionError(
            "known ternary reconciliation changed source line count"
        )
    return transformed, [
        {
            "line": line,
            "precondition": precondition,
            "expression": expression,
            "value": int(spec["value"]),
        }
    ]


def fold_known_discarded_tuple_component(
    source: str,
) -> tuple[str, list[dict[str, Any]]]:
    """Remove one hash-bound, unnamed discarded tuple declaration component."""

    source_sha256 = _sha256_text(source)
    spec = _KNOWN_DISCARDED_TUPLE_COMPONENT_FOLDS.get(source_sha256)
    if spec is None:
        return source, []
    expression = str(spec["expression"])
    if source.count(expression) != 1 or expression != "(bool success, )":
        raise CompatibilityExtractionError(
            "known discarded-tuple reconciliation source no longer matches its bound form"
        )
    expression_start = source.index(expression)
    line = source.count("\n", 0, expression_start) + 1
    if line != int(spec["line"]):
        raise CompatibilityExtractionError(
            "known discarded-tuple reconciliation line no longer matches its bound form"
        )
    replacement = "bool success".ljust(len(expression))
    transformed = (
        source[:expression_start]
        + replacement
        + source[expression_start + len(expression) :]
    )
    if len(transformed.encode("utf-8")) != len(source.encode("utf-8")):
        raise CompatibilityExtractionError(
            "known discarded-tuple reconciliation changed source byte length"
        )
    if transformed.count("\n") != source.count("\n"):
        raise CompatibilityExtractionError(
            "known discarded-tuple reconciliation changed source line count"
        )
    return transformed, [
        {
            "line": line,
            "expression": expression,
            "replacement": replacement,
            "semantic_basis": "unnamed_tuple_component_is_discarded_and_the_named_bool_assignment_is_preserved",
        }
    ]


def _extract_components(
    sol_path: Path,
    targets: tuple[str, ...],
    *,
    solc_binary: Path | None,
    solc_version: str,
    skip_analyze: bool,
    allow_paths: str,
    graph_schema_version: str,
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
            graph_schema_version=graph_schema_version,
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
    graph_schema_version: str = "v9",
) -> CompatibilityExtraction:
    """Extract all file targets with explicit, ordered compatibility modes."""

    sol_path = Path(sol_path)
    errors: list[dict[str, str]] = []
    common = {
        "solc_binary": solc_binary,
        "solc_version": solc_version,
        "allow_paths": str(sol_path.parent),
        "graph_schema_version": graph_schema_version,
    }
    try:
        graphs, actual = _extract_components(
            sol_path, targets, skip_analyze=False, **common
        )
        return CompatibilityExtraction(graphs, actual, FULL_ANALYSIS, ())
    except GraphExtractionError as exc:
        errors.append(_error_record(exc))
        full_analysis_error = exc

    # Historical v9 behavior is frozen. V10 may attempt recorded full-analysis
    # repairs before falling back to parse-only evidence.
    if graph_schema_version == "v10":
        if _is_singleton_call_type_failure(full_analysis_error):
            try:
                with _slither_singleton_call_type_repair() as repair_records:
                    graphs, actual = _extract_components(
                        sol_path, targets, skip_analyze=False, **common
                    )
                if not repair_records:
                    raise CompatibilityExtractionError(
                        "singleton-call-type retry succeeded without recording a repair"
                    )
                analyzer_repair = {
                    "schema": "r4-slither-analyzer-compatibility-v1",
                    "kind": "singleton_high_level_call_destination_type_unwrap",
                    "repair_count": len(repair_records),
                    "records": repair_records,
                }
                return CompatibilityExtraction(
                    graphs,
                    actual,
                    FULL_ANALYSIS_SINGLETON_CALL_TYPE,
                    tuple(errors),
                    analyzer_repair=analyzer_repair,
                )
            except (GraphExtractionError, CompatibilityExtractionError) as exc:
                errors.append(_error_record(exc))

        source = sol_path.read_text(encoding="utf-8")
        transformed, replacements = fold_known_state_initializer_ternary(source)
        if replacements:
            transform = {
                "schema": "r4-graph-source-compatibility-v2",
                "kind": "known_state_initializer_ternary_fold",
                "original_sha256": _sha256_text(source),
                "transformed_sha256": _sha256_text(transformed),
                "byte_length_preserved": True,
                "line_count_preserved": True,
                "replacements": replacements,
            }
            with tempfile.TemporaryDirectory(
                prefix="sentinel-r4-v10-ternary-compat-"
            ) as tmp:
                transformed_path = Path(tmp) / sol_path.name
                transformed_path.write_text(transformed, encoding="utf-8")
                transformed_common = {
                    "solc_binary": solc_binary,
                    "solc_version": solc_version,
                    "allow_paths": f"{sol_path.parent},{transformed_path.parent}",
                    "graph_schema_version": graph_schema_version,
                }
                try:
                    graphs, actual = _extract_components(
                        transformed_path,
                        targets,
                        skip_analyze=False,
                        **transformed_common,
                    )
                    return CompatibilityExtraction(
                        graphs,
                        actual,
                        FULL_ANALYSIS_KNOWN_TERNARY_FOLD,
                        tuple(errors),
                        source_transform=transform,
                    )
                except GraphExtractionError as exc:
                    errors.append(_error_record(exc))

        if _has_assertion_failure(full_analysis_error):
            transformed, replacements = fold_known_discarded_tuple_component(source)
            if replacements:
                transform = {
                    "schema": "r4-graph-source-compatibility-v2",
                    "kind": "known_discarded_tuple_component_fold",
                    "original_sha256": _sha256_text(source),
                    "transformed_sha256": _sha256_text(transformed),
                    "byte_length_preserved": True,
                    "line_count_preserved": True,
                    "replacements": replacements,
                }
                with tempfile.TemporaryDirectory(
                    prefix="sentinel-r4-v10-discarded-tuple-compat-"
                ) as tmp:
                    transformed_path = Path(tmp) / sol_path.name
                    transformed_path.write_text(transformed, encoding="utf-8")
                    transformed_common = {
                        "solc_binary": solc_binary,
                        "solc_version": solc_version,
                        "allow_paths": f"{sol_path.parent},{transformed_path.parent}",
                        "graph_schema_version": graph_schema_version,
                    }
                    try:
                        graphs, actual = _extract_components(
                            transformed_path,
                            targets,
                            skip_analyze=False,
                            **transformed_common,
                        )
                        return CompatibilityExtraction(
                            graphs,
                            actual,
                            FULL_ANALYSIS_KNOWN_DISCARDED_TUPLE_FOLD,
                            tuple(errors),
                            source_transform=transform,
                        )
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
            "graph_schema_version": graph_schema_version,
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
    "FULL_ANALYSIS_KNOWN_DISCARDED_TUPLE_FOLD",
    "FULL_ANALYSIS_KNOWN_TERNARY_FOLD",
    "FULL_ANALYSIS_SINGLETON_CALL_TYPE",
    "PARSE_ONLY",
    "PARSE_ONLY_CONSTANT_FOLD",
    "extract_components_with_compatibility",
    "fold_constant_array_lengths",
    "fold_known_discarded_tuple_component",
    "fold_known_state_initializer_ternary",
]
