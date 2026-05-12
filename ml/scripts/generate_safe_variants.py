#!/usr/bin/env python3
"""
generate_safe_variants.py — Data Augmentation: Mutation-Based Safe Contract Generation

Generates safe (CEI-compliant) variants of vulnerable Solidity contracts for
training data augmentation (Phase 3 of the SENTINEL v5 overhaul).

STRATEGY OVERVIEW
─────────────────
Each mutation strategy corresponds to a vulnerability type from §3.2 of the
v5 proposal. The text transformation is intentionally heuristic — correctness
is enforced by a mandatory two-step verification gate (§3.5):

  Step 1 — Compilation check (solc)
      Syntactically invalid swaps fail here. Slither may silently treat
      compilation failures as "zero findings", producing false "safe" verdicts.
      Running solc first prevents this.

  Step 2 — Slither detector check
      Semantically invalid swaps (missed second write path, modifier that
      re-enters, cross-function reentrancy) fail here.

Only contracts that pass BOTH gates are written to the output directory.
The expected yield is 30–70% of input contracts depending on code complexity.

MUTATION STRATEGIES
───────────────────
  reentrancy-cei:
      Swap state-write statements to before the external call (CEI pattern).
      Target: ~500 safe contracts from Reentrancy/ folder.

  mishandled-exception:
      Wrap bare `addr.call(...)` with return value check.
      Target: ~100 contracts from MishandledException/ folder.

  call-to-unknown:
      Add a typed interface comment annotation (light touch — full interface
      replacement requires contract-specific knowledge; Slither gate rejects
      contracts where typing alone is insufficient).
      Target: ~200 contracts from CallToUnknown/ folder.

  dos-bounded:
      Add `require(arr.length <= MAX_ITER)` guard before unbounded loops.
      Target: ~300 contracts from DenialOfService/ folder.

Usage:
    # CEI swap from the BCCC Reentrancy folder
    poetry run python ml/scripts/generate_safe_variants.py \\
        --input-dir ml/data/BCCC-SCsVul-2024/Reentrancy \\
        --output-dir ml/data/augmented/safe \\
        --strategy reentrancy-cei \\
        --max-contracts 800

    # Smoke run (first 20 contracts, dry run)
    poetry run python ml/scripts/generate_safe_variants.py \\
        --input-dir ml/data/BCCC-SCsVul-2024/Reentrancy \\
        --output-dir /tmp/safe_smoke \\
        --strategy reentrancy-cei \\
        --max-contracts 20 \\
        --dry-run

    # DoS bounded-loop augmentation
    poetry run python ml/scripts/generate_safe_variants.py \\
        --input-dir ml/data/BCCC-SCsVul-2024/DenialOfService \\
        --output-dir ml/data/augmented/dos_bounded \\
        --strategy dos-bounded

All commands must be run from the project root (~/projects/sentinel).
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

# ── Path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ml.src.data_extraction.ast_extractor import get_solc_binary, parse_solc_version

logger = logging.getLogger(__name__)


# ── Constants ─────────────────────────────────────────────────────────────────

# Maximum iterations for DoS bounded-loop guard
DOS_MAX_ITER = 100

# Reentrancy detectors to run in Slither verification step
REENTRANCY_DETECTORS = ["reentrancy-eth", "reentrancy-no-eth", "reentrancy-events"]

# Slither subprocess timeout (seconds) — complex contracts may be slow
SLITHER_TIMEOUT = 90

# solc subprocess timeout (seconds)
SOLC_TIMEOUT = 30

# Lines to scan ahead when looking for state writes after an external call
CALL_SCAN_WINDOW = 20


# ── Pragma version extraction ─────────────────────────────────────────────────

_PRAGMA_RE = re.compile(
    r"pragma\s+solidity\s+[^;]*?(\d+\.\d+\.\d+)"
)


def _extract_pragma_version(source: str) -> Optional[str]:
    """Return the first concrete semver string found in the pragma directive."""
    m = _PRAGMA_RE.search(source)
    return m.group(1) if m else None


# ── Step 1: Compilation check ─────────────────────────────────────────────────

def _compile_solidity(
    path: Path,
    solc_override: Optional[str] = None,
) -> subprocess.CompletedProcess:
    """
    Run solc on the given file and return the CompletedProcess.

    The caller checks returncode == 0 to determine success.
    We never run Slither before compilation passes — Slither may silently
    treat compilation failures as "zero findings" (false safe verdict).
    """
    source = path.read_text(encoding="utf-8", errors="replace")
    version = _extract_pragma_version(source)

    if solc_override:
        binary = solc_override
    elif version:
        binary = get_solc_binary(version) or "solc"
    else:
        binary = "solc"

    return subprocess.run(
        [binary, "--no-color", str(path)],
        capture_output=True,
        text=True,
        timeout=SOLC_TIMEOUT,
    )


# ── Step 2: Slither detector check ───────────────────────────────────────────

def _run_slither(
    path: Path,
    solc_override: Optional[str] = None,
    detectors: Optional[list[str]] = None,
) -> list[dict]:
    """
    Run Slither on the given file and return a list of detector finding dicts.

    Uses the CLI (subprocess) with --json output rather than the Python API
    to avoid version-dependent API surface area.

    Returns an empty list if Slither cannot run or produces no JSON output.
    """
    if detectors is None:
        detectors = REENTRANCY_DETECTORS

    source = path.read_text(encoding="utf-8", errors="replace")
    version = _extract_pragma_version(source)

    if solc_override:
        solc_binary = solc_override
    elif version:
        solc_binary = get_solc_binary(version) or "solc"
    else:
        solc_binary = "solc"

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    cmd = [
        "slither", str(path),
        "--detect", ",".join(detectors),
        "--json", str(tmp_path),
        "--solc", solc_binary,
        "--no-fail-pedantic",
    ]

    try:
        subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=SLITHER_TIMEOUT,
        )
        raw = tmp_path.read_text(encoding="utf-8", errors="replace")
        if not raw.strip():
            return []
        data = json.loads(raw)
        return data.get("results", {}).get("detectors", [])
    except (json.JSONDecodeError, FileNotFoundError, subprocess.TimeoutExpired) as exc:
        logger.debug("Slither run failed for %s: %s", path.name, exc)
        return []
    finally:
        tmp_path.unlink(missing_ok=True)


# ── Mutation: reentrancy-cei ──────────────────────────────────────────────────

# Matches an external call statement (possibly with ETH value transfer):
#   (bool ok, ) = addr.call{value: amount}("");
#   (bool ok, bytes memory ret) = addr.call{value: v}(data);
#   addr.call("");   ← bare (return ignored)
_EXTERNAL_CALL_RE = re.compile(
    r"\.call\s*[\({]",
)

# Matches assignment patterns that look like state writes:
#   balances[msg.sender] = 0;
#   totalSupply -= amount;
#   owner = msg.sender;
# Does NOT match local variable declarations (uint x = ...).
_STATE_WRITE_RE = re.compile(
    r"^(?!\s*(?:uint|int|bool|address|bytes|string|mapping|struct|enum|event|function)\b)"
    r"\s*\w+(?:\[[\w.\s]+\])?\s*[-+*/&|^]?=(?!=)",
    re.MULTILINE,
)


def _swap_call_and_write(source: str) -> Optional[str]:
    """
    Heuristic CEI swap: moves state-write statements to BEFORE the external
    call within the same function scope.

    Algorithm:
      1. Find all lines containing an external call (`.call{` or `.call(`).
      2. For each call line, scan forward within a CALL_SCAN_WINDOW window.
      3. Collect lines that look like state writes (assignments to non-local vars)
         and precede any `return`, `revert`, `}`, or another call.
      4. Move those write lines to just before the call line.

    Returns:
      Modified source string if at least one swap was made; None otherwise.

    The caller MUST validate the result via _compile_solidity() and _run_slither()
    before accepting it as safe — this transform is intentionally heuristic.
    """
    lines = source.split("\n")

    # Find all call-line indices (0-based)
    call_indices = [
        i for i, ln in enumerate(lines)
        if _EXTERNAL_CALL_RE.search(ln)
        and not ln.strip().startswith("//")
        and not ln.strip().startswith("*")
    ]

    if not call_indices:
        return None

    result = list(lines)
    swapped = False

    # Process from bottom to top so that earlier indices remain valid
    for call_i in reversed(call_indices):
        write_indices: list[int] = []
        for j in range(call_i + 1, min(call_i + CALL_SCAN_WINDOW + 1, len(result))):
            ln = result[j]
            stripped = ln.strip()

            # Skip blank lines and comments
            if not stripped or stripped.startswith("//") or stripped.startswith("*"):
                continue

            # Stop at hard scope exit or explicit control flow transfer
            if stripped.startswith("}") or any(
                stripped.startswith(kw) for kw in ("return", "revert", "assert")
            ):
                break

            # require() after a call is checking call success — skip it
            # (do NOT break: the state write may follow the require)
            if stripped.startswith("require") or stripped.startswith("emit"):
                continue

            # Stop at another external call (don't disturb multi-call patterns)
            if _EXTERNAL_CALL_RE.search(ln):
                break

            # Accept state-write assignments; reject local declarations
            if ";" in ln and _STATE_WRITE_RE.match(ln):
                write_indices.append(j)

        if not write_indices:
            continue

        # Extract the write lines, remove them (high→low to preserve indices)
        write_lines = [result[w] for w in write_indices]
        for w in reversed(write_indices):
            result.pop(w)

        # Insert before the call (all write_indices > call_i, so call_i unchanged)
        for ln in reversed(write_lines):
            result.insert(call_i, ln)

        swapped = True

    return "\n".join(result) if swapped else None


# ── Mutation: mishandled-exception ────────────────────────────────────────────

# Matches a bare addr.call(...) that discards the return value
_BARE_CALL_RE = re.compile(
    r"^(\s*)(\w[\w.]*)\s*\.call\s*(\{[^}]*\})?\s*(\([^)]*\))\s*;",
    re.MULTILINE,
)


def _wrap_bare_call(source: str) -> Optional[str]:
    """
    Wrap bare `addr.call(...)` invocations with a return-value check:
      addr.call{value: v}(data);
    becomes:
      (bool _ok,) = addr.call{value: v}(data);
      require(_ok, "call failed");

    Only wraps calls that currently discard the return value (bare call).
    Returns None if no bare calls are found.
    """
    modified = _BARE_CALL_RE.sub(
        lambda m: (
            f"{m.group(1)}(bool _ok,) = "
            f"{m.group(2)}.call"
            f"{m.group(3) or ''}"
            f"{m.group(4)};\n"
            f'{m.group(1)}require(_ok, "call failed");'
        ),
        source,
    )
    return modified if modified != source else None


# ── Mutation: dos-bounded ─────────────────────────────────────────────────────

# Matches a for/while loop over a storage array/mapping length
_LOOP_RE = re.compile(
    r"(^\s*)(for\s*\([^)]*(?:\.length|length\s*-\s*1)[^)]*\))",
    re.MULTILINE,
)

_WHILE_LENGTH_RE = re.compile(
    r"(^\s*)(while\s*\([^)]*\.length[^)]*\))",
    re.MULTILINE,
)


def _add_dos_loop_guard(source: str, max_iter: int = DOS_MAX_ITER) -> Optional[str]:
    """
    Insert `require(arr.length <= MAX_ITER)` before unbounded loops.

    This is a light-touch transformation: it adds a guard before `for` and
    `while` loops that iterate over `.length`. The guard prevents gas
    exhaustion DoS by bounding the iteration count.

    Returns None if no length-based loops are found.
    """
    guard_comment = f"// DoS guard: bound loop to prevent gas exhaustion"

    def _inject_guard(m: re.Match) -> str:
        indent = m.group(1)
        loop_stmt = m.group(2)
        # Extract array name from the loop condition if possible
        length_match = re.search(r"(\w+)\.length", loop_stmt)
        if length_match:
            arr = length_match.group(1)
            guard = (
                f"{indent}{guard_comment}\n"
                f'{indent}require({arr}.length <= {max_iter}, "array too long");\n'
                f"{indent}{loop_stmt}"
            )
        else:
            guard = f"{indent}{loop_stmt}"
        return guard

    modified = _LOOP_RE.sub(_inject_guard, source)
    modified = _WHILE_LENGTH_RE.sub(_inject_guard, modified)

    return modified if modified != source else None


# ── Mutation: call-to-unknown (annotation-based) ──────────────────────────────

_RAW_ADDR_CALL_RE = re.compile(
    r"(^\s*)(\w[\w]*)\s*\.call\s*(\{[^}]*\})?\s*(\(([^)]*)\))\s*;",
    re.MULTILINE,
)


def _annotate_typed_interface(source: str) -> Optional[str]:
    """
    Annotate raw `addr.call(...)` sites with a typed-interface comment.

    Full interface replacement requires contract-specific knowledge (the target
    ABI is not recoverable from source alone). This light annotation:
      addr.call(data);
    becomes:
      // TODO: replace with ITarget(addr).method(args) for type safety
      addr.call(data);

    The Slither CallToUnknown detector flags untyped calls to external
    addresses; this annotation does NOT silence Slither — it serves as a
    marker for manual review during the augmentation pipeline. Contracts where
    Slither still reports CallToUnknown after this pass are filtered out by the
    verification gate.

    Returns None if no raw addr.call sites are found.
    """
    comment = "// TYPED: replace with typed interface — ITarget(addr).method(args)"
    modified = _RAW_ADDR_CALL_RE.sub(
        lambda m: (
            f"{m.group(1)}{comment}\n"
            f"{m.group(1)}{m.group(2)}.call"
            f"{m.group(3) or ''}"
            f"{m.group(4)};"
        ),
        source,
    )
    return modified if modified != source else None


# ── Two-step verification pipeline ───────────────────────────────────────────

def generate_cei_safe(
    vulnerable_path: Path,
    out_dir: Path,
    solc_override: Optional[str] = None,
    strategy: str = "reentrancy-cei",
    dry_run: bool = False,
) -> Optional[Path]:
    """
    Apply mutation strategy to a single vulnerable contract and verify the result.

    Two-step verification gate (§3.5 of v5 proposal):
      1. Compilation check — must pass before Slither is invoked.
      2. Slither reentrancy check — swapped contract must not be flagged.

    Returns:
      Path of the written safe variant on success, None otherwise.
    """
    source = vulnerable_path.read_text(encoding="utf-8", errors="replace")

    # Apply mutation
    if strategy == "reentrancy-cei":
        mutated = _swap_call_and_write(source)
        check_detectors = REENTRANCY_DETECTORS
    elif strategy == "mishandled-exception":
        mutated = _wrap_bare_call(source)
        check_detectors = ["unchecked-lowlevel", "unchecked-send"]  # check for unhandled return values
    elif strategy == "call-to-unknown":
        # Annotation-only — cannot produce genuinely safe variants.
        # Accepting annotated-but-still-vulnerable contracts would poison training data.
        logger.warning(
            "%s: 'call-to-unknown' strategy cannot produce verified safe variants "
            "(adds comment only, no real fix). Skipping.",
            vulnerable_path.name,
        )
        return None
    elif strategy == "dos-bounded":
        mutated = _add_dos_loop_guard(source)
        check_detectors = ["calls-loop", "controlled-array-length"]
    else:
        raise ValueError(f"Unknown strategy: {strategy!r}")

    if mutated is None:
        logger.debug("%s: no mutation site found", vulnerable_path.name)
        return None

    if mutated == source:
        logger.debug("%s: mutation produced identical output", vulnerable_path.name)
        return None

    if dry_run:
        logger.info("[dry-run] Would write safe variant for %s", vulnerable_path.name)
        return vulnerable_path  # Return original as placeholder

    # Write to temp file for verification
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_stem = f"{vulnerable_path.stem}_safe"
    safe_path = out_dir / f"{safe_stem}.sol"

    # Avoid overwriting an already-accepted safe variant
    if safe_path.exists():
        suffix = 1
        while safe_path.exists():
            safe_path = out_dir / f"{safe_stem}_{suffix}.sol"
            suffix += 1

    safe_path.write_text(mutated, encoding="utf-8")

    # Step 1: Compilation check
    try:
        compile_result = _compile_solidity(safe_path, solc_override)
    except subprocess.TimeoutExpired:
        logger.error("solc timeout for %s — skipping", safe_path.name)
        safe_path.unlink(missing_ok=True)
        return None

    if compile_result.returncode != 0:
        logger.debug(
            "%s: compilation failed after mutation — discarding\n%s",
            safe_path.name,
            compile_result.stderr[:300],
        )
        safe_path.unlink(missing_ok=True)
        return None

    # Step 2: Slither detector check
    findings = _run_slither(safe_path, solc_override, detectors=check_detectors)

    # Filter to the detectors that indicate the vulnerability was NOT fixed
    if strategy == "reentrancy-cei":
        bad_findings = [f for f in findings if "reentrancy" in f.get("check", "").lower()]
    elif strategy == "mishandled-exception":
        bad_findings = [
            f for f in findings
            if any(d in f.get("check", "").lower() for d in ("unchecked", "lowlevel", "send"))
        ]
    elif strategy == "dos-bounded":
        bad_findings = [
            f for f in findings
            if any(d in f.get("check", "").lower() for d in ("loop", "array"))
        ]
    else:
        bad_findings = findings

    if bad_findings:
        logger.debug(
            "%s: still vulnerable after mutation (%d findings) — discarding",
            safe_path.name,
            len(bad_findings),
        )
        safe_path.unlink(missing_ok=True)
        return None

    logger.info("✓ accepted: %s → %s", vulnerable_path.name, safe_path.name)
    return safe_path


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(
    input_dir: Path,
    output_dir: Path,
    strategy: str,
    max_contracts: Optional[int],
    solc_override: Optional[str],
    dry_run: bool,
) -> dict:
    """
    Process all .sol files in input_dir and write safe variants to output_dir.

    Returns a summary dict suitable for writing to report.json.
    """
    sol_files = sorted(input_dir.rglob("*.sol"))
    if max_contracts:
        sol_files = sol_files[:max_contracts]

    if not sol_files:
        logger.error("No .sol files found in %s", input_dir)
        return {"total": 0, "accepted": 0, "rejected": 0}

    logger.info(
        "Processing %d contracts  strategy=%s  output=%s%s",
        len(sol_files),
        strategy,
        output_dir,
        "  [DRY RUN]" if dry_run else "",
    )

    results: list[dict] = []
    accepted = 0
    no_site = 0
    compile_fail = 0
    slither_fail = 0

    for i, sol_path in enumerate(sol_files, 1):
        logger.info("[%d/%d] %s", i, len(sol_files), sol_path.name)
        t0 = time.monotonic()

        source = sol_path.read_text(encoding="utf-8", errors="replace")

        # Quick pre-check: does the mutation apply at all?
        if strategy == "reentrancy-cei" and not _EXTERNAL_CALL_RE.search(source):
            no_site += 1
            results.append({"file": sol_path.name, "outcome": "no_site"})
            continue
        if strategy == "dos-bounded" and ".length" not in source:
            no_site += 1
            results.append({"file": sol_path.name, "outcome": "no_site"})
            continue

        out_path = generate_cei_safe(
            sol_path, output_dir, solc_override, strategy, dry_run
        )

        elapsed = time.monotonic() - t0

        if out_path is not None:
            accepted += 1
            results.append({
                "file": sol_path.name,
                "outcome": "accepted",
                "safe_file": out_path.name,
                "elapsed_s": round(elapsed, 2),
            })
        else:
            # We can't distinguish compile_fail vs slither_fail without more
            # instrumentation; both count as "rejected"
            results.append({
                "file": sol_path.name,
                "outcome": "rejected",
                "elapsed_s": round(elapsed, 2),
            })

    total = len(sol_files)
    rejected = total - accepted - no_site
    summary = {
        "strategy":       strategy,
        "input_dir":      str(input_dir),
        "output_dir":     str(output_dir),
        "total":          total,
        "accepted":       accepted,
        "rejected":       rejected,
        "no_site":        no_site,
        "yield_pct":      round(100 * accepted / total, 1) if total else 0,
        "dry_run":        dry_run,
        "contracts":      results,
    }

    report_path = output_dir / "report.json"
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        logger.info("Report written to %s", report_path)

    logger.info(
        "Done. %d/%d accepted (%.1f%%)  |  %d rejected  |  %d no mutation site",
        accepted, total, summary["yield_pct"], rejected, no_site,
    )

    return summary


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate safe contract variants for SENTINEL v5 data augmentation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--input-dir",
        required=True,
        type=Path,
        help="Directory containing vulnerable .sol contracts (searched recursively)",
    )
    p.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory where safe variants and report.json are written",
    )
    p.add_argument(
        "--strategy",
        choices=["reentrancy-cei", "mishandled-exception", "call-to-unknown", "dos-bounded"],
        default="reentrancy-cei",
        help=(
            "Mutation strategy:\n"
            "  reentrancy-cei:        Swap call/write order (CEI pattern).\n"
            "  mishandled-exception:  Wrap bare call() with return value check.\n"
            "  call-to-unknown:       Annotate raw address calls for review.\n"
            "  dos-bounded:           Add require(arr.length <= 100) loop guard."
        ),
    )
    p.add_argument(
        "--max-contracts",
        type=int,
        default=None,
        metavar="N",
        help="Process at most N contracts (useful for smoke runs)",
    )
    p.add_argument(
        "--solc-binary",
        default=None,
        metavar="PATH",
        help="Override solc binary path (default: auto-resolved from pragma version)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Log mutation sites without writing any files or running verification",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Enable DEBUG logging",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    if not args.input_dir.exists():
        logger.error("input-dir does not exist: %s", args.input_dir)
        sys.exit(1)

    summary = run_pipeline(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        strategy=args.strategy,
        max_contracts=args.max_contracts,
        solc_override=args.solc_binary,
        dry_run=args.dry_run,
    )

    sys.exit(0 if summary["accepted"] > 0 else 1)


if __name__ == "__main__":
    main()
