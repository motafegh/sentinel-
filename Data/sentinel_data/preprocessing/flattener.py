"""Flattener — resolve import chains to a single .sol file using solc --flatten.

Falls back gracefully: if solc --flatten fails (e.g. forge-std imports),
the original file is passed through with flatten_status=skipped_error.
A second fallback strips unresolvable imports so the compile step can still
verify the file's parseability (lossy but lossy in a controlled way —
only removes `import "..."` lines whose target is not present on disk).

This two-stage fallback was added for DeFiHackLabs (2026-06-10): 717/738 PoCs
import `forge-std/Test.sol` which is not present in the cloned repo (forge
submodule not pulled). Without the strip fallback, those 717 files all fail
to compile and are dropped. The strip preserves the vulnerable test code
(our modeling target) while making the file standalone-compilable.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from sentinel_data.preprocessing.compiler import _available_versions, _extract_pragma, _solc_binary


@dataclass
class FlattenResult:
    content: str           # flattened source (or original/stripped if skipped)
    flatten_status: str    # "flattened" | "skipped_no_imports" | "skipped_error"
                           #   | "stripped_unresolved_imports"
    error: str = ""


_IMPORT_RE = re.compile(r'^\s*import\s+', re.MULTILINE)

# Symbols we conservatively assume a bare `import "forge-std/x.sol";` brings
# in. If we strip an import without removing the matching inheritance parent,
# the compile step fails with "Identifier not found". Better to assume the
# common ones and over-strip a few parents than to leave dangling references.
_ASSUMED_BARE_IMPORT_SYMBOLS = frozenset({
    "Test",       # forge-std/Test.sol base contract (used in 656/738 DeFiHackLabs PoCs)
    "console",    # forge-std/console.sol
    "console2",   # forge-std/console2.sol
    "Vm",         # forge-std/Vm.sol
    "ScriptUtils",
})
# `import "x";` or `import 'x';` or `import * as Y from "x";` or `import {a, b} from "x";`
_IMPORT_LINE_RE = re.compile(
    r"""^\s*import\s+(?:(?P<syms>[^'"]+?)\s+from\s+)?['"](?P<target>[^'"]+)['"]\s*;\s*$""",
    re.MULTILINE,
)
# `contract Foo is A, B, C {` — captures the inheritance list
_CONTRACT_INHERIT_RE = re.compile(
    r"(\bcontract\s+\w+\s+is\s+)([^{]+)( \{)",
)


def flatten_contract(sol_path: Path) -> FlattenResult:
    """Flatten `sol_path`. Returns FlattenResult with content ready for next step."""
    source = sol_path.read_text(errors="replace")

    # Skip flattening if there are no import statements (most contracts in DeFiHackLabs)
    if not _IMPORT_RE.search(source):
        return FlattenResult(content=source, flatten_status="skipped_no_imports")

    # Try solc --flatten with the version that matches this file's pragma
    pragma = _extract_pragma(source)
    solc_bin = _pick_solc(pragma)

    if solc_bin:
        result = subprocess.run(
            [str(solc_bin), "--flatten", str(sol_path)],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0 and result.stdout.strip():
            return FlattenResult(content=result.stdout, flatten_status="flattened")

    # Flatten failed — try stripping unresolvable imports as a second fallback.
    # The compile step will succeed if the only unresolvable imports are
    # external dependencies (forge-std, hardhat, openzeppelin) and the rest
    # of the file is self-contained.
    #
    # Recursion: when we encounter a relative import that DOES resolve on disk,
    # we also process that file's imports (since the relative import will pull
    # in its own transitive imports at compile time). This is what catches the
    # DeFiHackLabs pattern where `interface.sol` and `basetest.sol` are shared
    # relative files that themselves import forge-std.
    stripped, n_stripped, unresolved, removed_symbols, sub_strips = _strip_unresolved_imports_recursive(
        source, sol_path
    )
    if n_stripped > 0 and len(unresolved) == n_stripped:
        # If the stripped imports brought in named symbols (e.g. `Test` from
        # forge-std/Test.sol) and a contract inherits from one of them, also
        # remove that parent from the `is A, B, C` list. Without this the file
        # would still reference an undefined symbol and the compile would fail.
        if removed_symbols:
            stripped, n_inherit_stripped = _strip_unresolved_inheritance(
                stripped, removed_symbols
            )
        else:
            n_inherit_stripped = 0

        # When the recursive strip modified any transitive relative-imported
        # files (sub_strips is non-empty), we need to write those modified
        # files somewhere the compiler can see them, otherwise the compile
        # will read the originals from disk and re-introduce the unresolved
        # imports. The sub_strips dict maps original file Path -> stripped
        # content. We write each to a temp file alongside the original (so
        # the original's relative path resolution still works), and rewrite
        # the top-level import in `stripped` to point to the temp file.
        siblings: list[Path] = []
        if sub_strips:
            from ._transitive_strip import apply_sub_strips_to_source
            stripped, siblings = apply_sub_strips_to_source(
                stripped, sol_path, sub_strips
            )

        return FlattenResult(
            content=stripped,
            flatten_status="stripped_unresolved_imports",
            error=(
                f"stripped {n_stripped} unresolved import(s) "
                f"(incl. {len(sub_strips)} transitive), "
                f"{n_inherit_stripped} inheritance parent(s): "
                f"{', '.join(sorted(unresolved)[:5])}"
            ),
        )

    # Flatten failed and strip either didn't help or wasn't safe
    return FlattenResult(
        content=source,
        flatten_status="skipped_error",
        error="solc --flatten failed; using original source",
    )


def _strip_unresolved_imports(
    source: str, sol_path: Path
) -> tuple[str, int, list[str], set[str]]:
    """Single-file strip — no recursion. See `_strip_unresolved_imports_recursive`
    for the version that follows relative imports into shared files."""
    return _strip_unresolved_imports_recursive(source, sol_path, _seen=None, _sub_strips=None)


def _strip_unresolved_imports_recursive(
    source: str,
    sol_path: Path,
    _seen: set[Path] | None = None,
    _sub_strips: dict[Path, str] | None = None,
) -> tuple[str, int, list[str], set[str], dict[Path, str]]:
    """Strip `import "x";` lines whose target cannot be resolved on disk.

    Resolution rules (relative to sol_path's directory):
      - `import "x.sol";`           → ./x.sol
      - `import "dir/x.sol";`       → ./dir/x.sol
      - `import "../foo/x.sol";`    → ../foo/x.sol
      - `import "forge-std/x.sol";` → ./forge-std/x.sol  (NOT resolved if absent)

    Relative paths that don't resolve are stripped. Relative paths that DO
    resolve are kept (they'd be a transitive flatten failure, but we don't
    recursively flatten — the compile step handles the rest).

    Also collects the set of symbol names brought in by stripped imports
    (from `import {Test, console} from "forge-std/Test.sol";` or
    `import * as F from "forge-std/Test.sol";`). The caller uses this to
    remove matching inheritance parents via `_strip_unresolved_inheritance`.

    Returns (stripped_source, n_stripped, list_of_unresolved_targets,
             set_of_removed_symbol_names).
    """
    sol_dir = sol_path.parent
    n_stripped = 0
    unresolved: list[str] = []
    removed_symbols: set[str] = set()
    sub_strips: dict[Path, str] = _sub_strips if _sub_strips is not None else {}
    seen = _seen if _seen is not None else set()
    seen.add(sol_path.resolve())

    def _sub(m: re.Match) -> str:
        nonlocal n_stripped, unresolved, removed_symbols
        target = m.group("target")
        syms_str = m.group("syms")

        # Resolve the target and decide whether to recurse
        candidate = sol_dir / target
        try:
            candidate_resolved = candidate.resolve()
        except (OSError, ValueError):
            candidate_resolved = candidate

        # If the target is relative (./foo, ../bar) or a non-relative path that
        # happens to resolve on disk, recurse to strip its own unresolved imports.
        if target.startswith(".") or candidate.exists():
            if candidate_resolved in seen or not candidate_resolved.exists():
                return m.group(0)
            try:
                sub_source = candidate_resolved.read_text(errors="replace")
            except OSError:
                return m.group(0)
            # If we've already processed this sub_path, reuse the result.
            if candidate_resolved in sub_strips:
                return m.group(0)
            sub_stripped, sub_n, sub_unresolved, sub_symbols, _ = _strip_unresolved_imports_recursive(
                sub_source, candidate_resolved, seen, sub_strips
            )
            n_stripped += sub_n
            unresolved.extend(sub_unresolved)
            removed_symbols |= sub_symbols
            if sub_n > 0:
                sub_strips[candidate_resolved] = sub_stripped
            return m.group(0)

        # Non-resolvable, non-relative → strip it
        n_stripped += 1
        unresolved.append(target)

        # Collect symbols brought in by this import. Three cases:
        #   1. `import {A, B} from "x";`   → syms_str = "{A, B}"
        #   2. `import * as F from "x";`   → syms_str = "* as F"
        #   3. `import "x";`               → syms_str = None (no named symbols)
        if syms_str:
            for sym in _extract_imported_symbols(syms_str):
                removed_symbols.add(sym)
        else:
            # Bare import: conservatively assume the common forge-std symbols.
            # This is the pattern in 610/738 DeFiHackLabs PoCs.
            # The compile step will fail fast on any actually-undefined symbol
            # we missed, so being too aggressive here just costs us a few
            # extra dropped files, not silent corruption.
            for sym in _ASSUMED_BARE_IMPORT_SYMBOLS:
                removed_symbols.add(sym)

        return ""  # strip the import line

    stripped = _IMPORT_LINE_RE.sub(_sub, source)
    return stripped, n_stripped, unresolved, removed_symbols, sub_strips


def _extract_imported_symbols(syms_str: str) -> list[str]:
    """Parse the symbol portion of an import statement.

    Examples (input → output):
      '{A, B as C, D}'                 → ['A', 'C', 'D']
      '* as F'                          → ['F']
      'A'                               → ['A']   (legacy single-symbol form)
    """
    s = syms_str.strip()
    if not s:
        return []
    if s.startswith("{"):
        # Named imports: {A, B as C, D}
        out = []
        for piece in s.strip("{}").split(","):
            piece = piece.strip()
            if not piece:
                continue
            # `A as B` → keep B
            parts = piece.split(" as ")
            out.append(parts[-1].strip())
        return out
    if s.startswith("* as "):
        return [s[len("* as "):].strip()]
    # Legacy `import A from "x";` — single default import
    parts = s.split(" as ")
    return [parts[-1].strip()]


def _strip_unresolved_inheritance(source: str, removed_symbols: set[str]) -> tuple[str, int]:
    """Remove resolved-but-undefined parent names from `contract Foo is A, B, C {`.

    Returns (modified_source, n_stripped).
    """
    if not removed_symbols:
        return source, 0

    n_stripped = 0

    def _sub(m: re.Match) -> str:
        nonlocal n_stripped
        prefix = m.group(1)  # "contract Foo is "
        parents_str = m.group(2)
        brace = m.group(3)
        parents = [p.strip() for p in parents_str.split(",") if p.strip()]
        kept = [p for p in parents if p not in removed_symbols]
        n_stripped += len(parents) - len(kept)
        if not kept:
            # No parents left — drop the `is` clause entirely
            # `contract Foo is X {`  →  `contract Foo {`
            return prefix.rstrip(" is") + brace
        return prefix + ", ".join(kept) + brace

    return _CONTRACT_INHERIT_RE.sub(_sub, source), n_stripped


def _pick_solc(pragma: str):
    """Return a solc binary path that matches `pragma`, or None."""
    from sentinel_data.preprocessing.compiler import _parse_version, _satisfying_versions
    available = _available_versions()
    requested = _parse_version(pragma)
    if requested:
        from sentinel_data.preprocessing.compiler import _solc_binary
        b = _solc_binary(requested)
        if b:
            return b
    candidates = _satisfying_versions(pragma, available)
    for ver in reversed(candidates):
        from sentinel_data.preprocessing.compiler import _solc_binary
        b = _solc_binary(ver)
        if b:
            return b
    return None
