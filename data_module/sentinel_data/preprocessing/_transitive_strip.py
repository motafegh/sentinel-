"""Helper for the flattener: write transitive sub_strip files and rewrite
the top-level import to point at them.

When the flattener's recursive strip modifies a transitive relative-imported
file (e.g. `interface.sol` that itself imported `forge-std/Test.sol`), the
compiler must see the modified version. We can't safely modify the on-disk
file (it might be used by other PoCs in the same source), so we write a
sibling file with a `.sentinel_stripped.sol` suffix and rewrite the top-level
import to point at it. The sibling is auto-cleaned by the pipeline after
the compile step.

This keeps the original files untouched, ensures each PoC's compile sees a
consistent set of stripped dependencies, and survives the "shared
interface.sol" pattern in DeFiHackLabs.
"""

from __future__ import annotations

import re
from pathlib import Path

_SUFFIX = ".sentinel_stripped.sol"

# Same shape as _IMPORT_LINE_RE in flattener.py
_IMPORT_LINE_RE = re.compile(
    r"""(?P<full>^\s*import\s+(?:(?P<syms>[^'"]+?)\s+from\s+)?['"](?P<target>[^'"]+)['"]\s*;\s*$)""",
    re.MULTILINE,
)


def apply_sub_strips_to_source(
    source: str,
    sol_path: Path,
    sub_strips: dict[Path, str],
) -> str:
    """Rewrite the top-level `source` so its relative imports point to the
    sentinel-stripped sibling files for each entry in `sub_strips`, AND
    write the stripped contents alongside the original (so the importing
    file's directory resolves to the stripped version).

    The sentinel-stripped sibling is written NEXT TO the original (same dir
    as the relative-imported file), with a `.sentinel_stripped.sol` suffix.
    The top-level source's relative import is rewritten to point to this
    sibling using a path that's relative from the importer to the original's
    dir. Since the importer (temp file) lives in the same dir as the original
    `.sol` file (e.g. `2018-04/`), and the sibling lives in the imported
    file's dir (e.g. `src/test/`), we need a `../test/...sentinel_stripped.sol`
    style rewrite.

    Returns the rewritten source. The siblings are written to disk as a
    side effect; caller is responsible for cleanup.
    """
    sol_dir = sol_path.parent
    # Build: target_string -> new_target_string
    rewrites: dict[str, str] = {}
    siblings: list[Path] = []

    for orig_path, stripped_content in sub_strips.items():
        try:
            orig_resolved = orig_path.resolve()
        except (OSError, ValueError):
            orig_resolved = orig_path

        stripped_sibling = orig_resolved.with_name(orig_resolved.name + _SUFFIX)
        if not stripped_sibling.exists() or stripped_sibling.read_text(errors="replace") != stripped_content:
            stripped_sibling.write_text(stripped_content)
        siblings.append(stripped_sibling)

        # Compute the import-target rewrite: the original import was
        # `<some_path>/x.sol` from sol_path. Replace it with a path from
        # sol_dir to stripped_sibling.
        # e.g. orig: ../interface.sol, stripped: ../interface.sol.sentinel_stripped.sol
        try:
            rel = stripped_sibling.relative_to(sol_dir)
            new_target = str(rel)
        except ValueError:
            # stripped_sibling is not under sol_dir; compute from common ancestor
            new_target = str(stripped_sibling)

        # The original target string is what was in the import (e.g. "../interface.sol")
        # We need to find it in the source and replace. The recursive strip
        # recorded the original resolved path; we need the textual form.
        # Trick: for each sub_strip, look at all relative imports in source
        # that resolve to orig_resolved.
        for m in _IMPORT_LINE_RE.finditer(source):
            target = m.group("target")
            if not target.startswith("."):
                continue
            try:
                if (sol_dir / target).resolve() == orig_resolved:
                    rewrites[target] = new_target
            except (OSError, ValueError):
                continue

    def _do_rewrite(m: re.Match) -> str:
        target = m.group("target")
        if target in rewrites:
            new_target = rewrites[target]
            return m.group(0).replace(f'"{target}"', f'"{new_target}"').replace(
                f"'{target}'", f"'{new_target}'"
            )
        return m.group(0)

    return _IMPORT_LINE_RE.sub(_do_rewrite, source), siblings
