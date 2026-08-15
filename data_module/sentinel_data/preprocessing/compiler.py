"""Compiler selection and auditable Solidity invocation.

The repaired R4 path keeps the historical two-pass version resolution but makes
optional flags capability-aware.  In particular, Solidity 0.4.x binaries that
reject ``--allow-paths`` must never receive that flag.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

_SOLC_ARTIFACTS = Path.home() / ".solc-select" / "artifacts"
_PRAGMA_RE = re.compile(r"pragma\s+solidity\s+([^;]+);", re.MULTILINE)


@dataclass
class CompileResult:
    """Result of a two-pass solc compilation attempt."""

    success: bool
    solc_version: str
    pragma_raw: str
    error: str = ""
    attempted_versions: list[str] | None = None
    command_flags: list[str] | None = None

    def __post_init__(self) -> None:
        if self.attempted_versions is None:
            self.attempted_versions = []
        if self.command_flags is None:
            self.command_flags = []


def compile_contract(sol_path: Path) -> CompileResult:
    """Compile ``sol_path`` with the exact/nearest installed compatible solc."""

    source = sol_path.read_text(errors="replace")
    pragma_raw = _extract_pragma(source)
    requested = _parse_version(pragma_raw)
    attempted: list[str] = []
    last_err = ""
    last_flags: list[str] = []

    if requested:
        bin_path = _solc_binary(requested)
        if bin_path:
            attempted.append(requested)
            ok, err, flags = _run_solc(bin_path, sol_path, requested)
            last_flags = flags
            if ok:
                return CompileResult(
                    True,
                    requested,
                    pragma_raw,
                    attempted_versions=attempted,
                    command_flags=flags,
                )
            last_err = err

    for ver in _satisfying_versions(pragma_raw, _available_versions()):
        if ver in attempted:
            continue
        bin_path = _solc_binary(ver)
        if not bin_path:
            continue
        attempted.append(ver)
        ok, err, flags = _run_solc(bin_path, sol_path, ver)
        last_flags = flags
        if ok:
            return CompileResult(
                True,
                ver,
                pragma_raw,
                attempted_versions=attempted,
                command_flags=flags,
            )
        last_err = err

    return CompileResult(
        False,
        "",
        pragma_raw,
        error=f"all versions failed; last error: {last_err[:300]}",
        attempted_versions=attempted,
        command_flags=last_flags,
    )


def _extract_pragma(source: str) -> str:
    """Extract the pragma constraint and remove internal whitespace."""

    match = _PRAGMA_RE.search(source)
    return re.sub(r"\s+", "", match.group(1)) if match else ""


def _parse_version(pragma: str) -> str:
    """Extract a clean version string when a single floor/exact version exists."""

    match = re.fullmatch(r"=?(\d+\.\d+\.\d+)", pragma)
    if match:
        return match.group(1)
    match = re.fullmatch(r"[\^~>=]*(\d+\.\d+\.\d+)", pragma)
    return match.group(1) if match else ""


def _satisfying_versions(pragma: str, available: list[str]) -> list[str]:
    """Return installed Solidity versions satisfying the pragma, newest first.

    Solidity permits adjacent comparator clauses (for example
    ``>=0.6.2<0.8.0``), caret/tilde ranges, and ``||`` alternatives. Missing
    pragmas are not proof of invalid source; all installed versions are tried
    deterministically and the successful compiler becomes provenance.
    """

    if not pragma:
        return list(reversed(available))

    token_re = re.compile(r"(\^|~|>=|<=|>|<|=)?\s*(\d+(?:\.\d+){0,2})")

    def constraint_tuple(raw_version: str) -> tuple[int, int, int]:
        parts = raw_version.split(".")
        if not 1 <= len(parts) <= 3 or not all(part.isdigit() for part in parts):
            raise ValueError(f"invalid Solidity constraint version: {raw_version!r}")
        padded = [*parts, *(["0"] * (3 - len(parts)))]
        return tuple(int(part) for part in padded)  # type: ignore[return-value]

    def clause_matches(value: tuple[int, int, int], clause: str) -> bool:
        tokens = token_re.findall(clause)
        if not tokens:
            return False
        for operator, raw_version in tokens:
            bound = constraint_tuple(raw_version)
            precision = raw_version.count(".") + 1
            if operator in ("", "="):
                if precision == 3 and value != bound:
                    return False
                if precision == 2 and not bound <= value < (
                    bound[0],
                    bound[1] + 1,
                    0,
                ):
                    return False
                if precision == 1 and not bound <= value < (bound[0] + 1, 0, 0):
                    return False
            if operator == ">=" and value < bound:
                return False
            if operator == ">" and value <= bound:
                return False
            if operator == "<=" and value > bound:
                return False
            if operator == "<" and value >= bound:
                return False
            if operator == "^":
                if bound[0] > 0:
                    ceiling = (bound[0] + 1, 0, 0)
                elif bound[1] > 0:
                    ceiling = (0, bound[1] + 1, 0)
                else:
                    ceiling = (0, 0, bound[2] + 1)
                if not bound <= value < ceiling:
                    return False
            if operator == "~":
                ceiling = (bound[0], bound[1] + 1, 0)
                if not bound <= value < ceiling:
                    return False
        return True

    clauses = pragma.split("||")
    return [
        version
        for version in reversed(available)
        if any(clause_matches(_version_tuple(version), clause) for clause in clauses)
    ]


def _version_tuple(version: str) -> tuple[int, int, int]:
    parts = version.split(".")
    if len(parts) != 3 or not all(part.isdigit() for part in parts):
        raise ValueError(f"invalid solc version: {version!r}")
    return tuple(int(part) for part in parts)  # type: ignore[return-value]


def supports_allow_paths(version: str) -> bool:
    """Return whether this project authorizes ``--allow-paths`` for ``version``.

    Historical audit evidence proves 0.4.9 rejects the option.  The graph path
    already used 0.5.0 as the compatibility boundary; preprocessing now uses
    the same explicit policy.
    """

    return _version_tuple(version) >= (0, 5, 0)


def build_solc_command(
    bin_path: Path,
    sol_path: Path,
    version: str,
    *,
    allow_root: Path | None = None,
) -> list[str]:
    """Build the exact auditable command for one compiler invocation."""

    command = [str(bin_path), "--bin", str(sol_path)]
    if allow_root is not None and supports_allow_paths(version):
        command.extend(["--allow-paths", str(allow_root)])
    return command


def _available_versions() -> list[str]:
    if not _SOLC_ARTIFACTS.exists():
        return []
    versions = []
    for directory in _SOLC_ARTIFACTS.iterdir():
        if directory.is_dir() and directory.name.startswith("solc-"):
            version = directory.name[len("solc-") :]
            if re.fullmatch(r"\d+\.\d+\.\d+", version):
                versions.append(version)
    return sorted(versions, key=_version_tuple)


def _solc_binary(version: str) -> Path | None:
    path = _SOLC_ARTIFACTS / f"solc-{version}" / f"solc-{version}"
    return path if path.exists() else None


def _run_solc(
    bin_path: Path,
    sol_path: Path,
    version: str,
) -> tuple[bool, str, list[str]]:
    """Run solc and return ``(success, error_tail, optional_flags)``."""

    allow_root = sol_path.parent.parent
    command = build_solc_command(
        bin_path,
        sol_path,
        version,
        allow_root=allow_root,
    )
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=30,
    )
    flags = command[3:]
    if result.returncode == 0:
        return True, "", flags
    return False, (result.stderr or result.stdout)[-500:], flags
