"""V10-only deterministic CFG classification guard for R4 Phase-8.

Slither 0.10.0 can expose different ``state_variables_written`` results across
otherwise-identical analyses when a state mutation is performed through a
member/index path backed by a storage reference. The mutable SlithIR
``ReferenceVariable.points_to`` chain is useful analysis state, but it is not a
stable representation identity boundary.

This module does not replace Slither's classifier and does not special-case
corpus hashes. During V10 extraction only it supplements the existing WRITE
classification with stable expression-tree information recorded before
SlithIR reference resolution:

* direct ``StateVariable`` lvalues are persistent writes;
* member/index writes rooted at a ``StateVariable`` are persistent writes;
* member/index writes rooted at a ``LocalVariable`` whose ``is_storage`` is
  true are persistent writes;
* declaring the storage-reference local itself is not treated as a mutation.
* Solidity collection ``push``/``pop`` calls whose receiver is rooted at a
  persistent-storage variable are persistent writes.

Historical v9 extraction remains on the unmodified classifier. The guard is
process-local, serialized with an ``RLock``, and restored after each V10 call.
"""

from __future__ import annotations

import functools
import threading
from contextlib import contextmanager
from typing import Any, Iterator


_CFG_GUARD_LOCK = threading.RLock()
_INSTALL_LOCK = threading.RLock()
_INSTALL_MARKER = "_sentinel_v10_deterministic_cfg_guard_v2"
_STORAGE_MUTATING_MEMBER_CALLS = frozenset({"push", "pop"})


def _expression_root_variable(expression: Any) -> Any | None:
    """Return the base variable for an identifier/member/index expression."""

    try:
        from slither.core.expressions.identifier import Identifier
        from slither.core.expressions.index_access import IndexAccess
        from slither.core.expressions.member_access import MemberAccess
    except ImportError:
        return None

    current = expression
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, Identifier):
            return getattr(current, "value", None)
        if isinstance(current, MemberAccess):
            current = getattr(current, "expression", None)
            continue
        if isinstance(current, IndexAccess):
            current = getattr(current, "expression_left", None)
            continue
        return None
    return None


def _expression_writes_persistent_storage(slither_node: Any) -> bool:
    """Return True when stable expression lvalues prove persistent storage mutation.

    ``variables_written_as_expression`` is populated by Slither's expression
    visitor before SlithIR reference propagation. Its order is irrelevant here;
    only membership and lvalue roots are consumed.

    A bare storage-reference local identifier is deliberately *not* enough. A
    declaration such as ``Struct storage alias = stateStruct`` creates a local
    reference but does not itself mutate the referenced state. Storage locals
    therefore qualify only when they are the root of a member/index lvalue.
    """

    try:
        from slither.core.expressions.identifier import Identifier
        from slither.core.expressions.index_access import IndexAccess
        from slither.core.expressions.member_access import MemberAccess
        from slither.core.variables.local_variable import LocalVariable
        from slither.core.variables.state_variable import StateVariable
    except ImportError:
        return False

    declaration = getattr(slither_node, "variable_declaration", None)
    written = list(
        getattr(slither_node, "variables_written_as_expression", None) or []
    )

    for expression in written:
        if isinstance(expression, Identifier):
            variable = getattr(expression, "value", None)
            if variable is declaration:
                continue
            if isinstance(variable, StateVariable):
                return True
            # A bare LocalVariable, including a storage reference, can be a
            # declaration/rebinding rather than a mutation. Member/index paths
            # are handled explicitly below.
            continue

        if not isinstance(expression, (MemberAccess, IndexAccess)):
            continue

        variable = _expression_root_variable(expression)
        if variable is None or variable is declaration:
            continue
        if isinstance(variable, StateVariable):
            return True
        if isinstance(variable, LocalVariable):
            try:
                if bool(variable.is_storage):
                    return True
            except Exception:
                # Classification evidence must be positive, not inferred from a
                # failed storage-location query.
                continue

    return False


def _call_mutates_persistent_storage(slither_node: Any) -> bool:
    """Return True for Solidity collection mutators on persistent storage.

    Slither 0.10 does not populate ``variables_written_as_expression`` for
    dynamic-array ``push``/``pop`` calls.  The call expression itself is stable:
    its callee is a ``MemberAccess`` and the callee receiver retains the same
    StateVariable/storage-LocalVariable root consumed by the existing lvalue
    rule.  Restricting the name set to Solidity's two mutating collection
    built-ins avoids treating arbitrary methods as writes.  Calls Slither
    already classifies as external/library CALL remain protected by the caller's
    higher-priority check.
    """

    try:
        from slither.core.expressions.call_expression import CallExpression
        from slither.core.expressions.member_access import MemberAccess
        from slither.core.variables.local_variable import LocalVariable
        from slither.core.variables.state_variable import StateVariable
    except ImportError:
        return False

    expression = getattr(slither_node, "expression", None)
    if not isinstance(expression, CallExpression):
        return False
    called = getattr(expression, "called", None)
    if not isinstance(called, MemberAccess):
        return False
    if str(getattr(called, "member_name", "") or "") not in _STORAGE_MUTATING_MEMBER_CALLS:
        return False

    receiver = getattr(called, "expression", None)
    variable = _expression_root_variable(receiver)
    if isinstance(variable, StateVariable):
        return True
    if not isinstance(variable, LocalVariable):
        return False
    try:
        return bool(variable.is_storage)
    except Exception:
        return False


@contextmanager
def v10_deterministic_cfg_classification() -> Iterator[None]:
    """Temporarily supplement ``_cfg_node_type`` with stable V10 WRITE evidence."""

    from sentinel_data.representation import graph_extractor

    with _CFG_GUARD_LOCK:
        original = graph_extractor._cfg_node_type
        node_types = graph_extractor.NODE_TYPES
        call_type = node_types["CFG_NODE_CALL"]
        write_type = node_types["CFG_NODE_WRITE"]

        def deterministic(slither_node: Any) -> int:
            classified = original(slither_node)
            # Preserve the historical priority contract: CALL always wins, and
            # every state write Slither already proved remains a WRITE.
            if classified in {call_type, write_type}:
                return classified
            if _expression_writes_persistent_storage(
                slither_node
            ) or _call_mutates_persistent_storage(slither_node):
                return write_type
            return classified

        graph_extractor._cfg_node_type = deterministic
        try:
            yield
        finally:
            graph_extractor._cfg_node_type = original


def install_v10_extraction_guard() -> None:
    """Install one version-aware wrapper around the canonical graph extractor.

    The wrapper changes no v9 call. For ``GraphExtractionConfig`` selecting
    graph schema ``v10`` it activates
    :func:`v10_deterministic_cfg_classification` only for the duration of that
    extraction, then restores the classifier.
    """

    from sentinel_data.representation import graph_extractor

    with _INSTALL_LOCK:
        current = graph_extractor.extract_contract_graph
        if getattr(current, _INSTALL_MARKER, False):
            return

        @functools.wraps(current)
        def guarded_extract(sol_path: Any, config: Any | None = None):
            version = (
                getattr(config, "graph_schema_version", "v9")
                if config is not None
                else "v9"
            )
            if version != "v10":
                return current(sol_path, config=config)
            with v10_deterministic_cfg_classification():
                return current(sol_path, config=config)

        setattr(guarded_extract, _INSTALL_MARKER, True)
        graph_extractor.extract_contract_graph = guarded_extract


__all__ = [
    "_call_mutates_persistent_storage",
    "install_v10_extraction_guard",
    "v10_deterministic_cfg_classification",
]
