"""
session_close.py — L.5.1 Auto-Detection of Floating Findings.

WHY THIS EXISTS

Rule 3 (no floating findings) was enforced by humans remembering to
write findings. Humans forget, skip steps, or declare "I'll write it
later" then close the session. L.5.1 makes this automated.

This script:
1. Scans the conversation log / recent file changes for finding patterns
2. For each candidate, checks if a corresponding ISSUES.md entry exists
3. Refuses to close if findings are unwritten

USAGE

    # At session close
    python ml/scripts/session_close.py \\
        --recent-files <file1> <file2> ... \\
        --log <path/to/session.log>

    # Just check (no enforcement)
    python ml/scripts/session_close.py --dry-run \\
        --recent-files <files>

Exit codes:
    0  no floating findings
    1  floating findings detected (session should NOT close)
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

# Patterns that suggest a finding was discovered
FINDING_PATTERNS = [
    (r"\b(FP|false\s+positive)\b", "False positive"),
    (r"\b(FN|false\s+negative)\b", "False negative"),
    (r"\bbug\s+(is|was|discovered|found)\b", "Bug discovered"),
    (r"\bbroken\b", "Something broken"),
    (r"\bbug[_-](\d+)\b", "Bug ID mentioned"),
    (r"\bFIND[_-](\w+)\b", "Finding ID mentioned"),
    (r"\bdiscovered\s+(that|a\s+bug|an?\s+issue)\b", "Discovery noted"),
    (r"\b(label\s+noise|noisy\s+labels?)\b", "Label quality issue"),
    (r"\bthreshold\s+gaming\b", "Threshold gaming issue"),
    (r"\b(over|under)[- ]?predict", "Prediction issue"),
    (r"\b(model|system)\s+(is\s+)?(broken|wrong|fail)", "Model issue"),
]

# Files that should be checked for findings
ISSUES_FILES = [
    Path("ml/audit_docs/ISSUES.md"),
]


def scan_text_for_findings(text: str) -> list[dict[str, str]]:
    """Scan text for finding patterns. Returns list of {pattern, match, line}."""
    findings = []
    for line_no, line in enumerate(text.split("\n"), 1):
        for pattern, label in FINDING_PATTERNS:
            m = re.search(pattern, line, re.IGNORECASE)
            if m:
                findings.append({
                    "pattern": label,
                    "match": m.group(0),
                    "line": line_no,
                    "text": line.strip()[:200],
                })
    return findings


def is_finding_documented(finding: dict, issues_text: str) -> bool:
    """Check if a finding has a corresponding ISSUES.md entry."""
    # Heuristic: check if the keyword from the finding appears in ISSUES.md
    keyword = finding["match"]
    # Try variations
    candidates = [keyword, keyword.upper(), keyword.title()]
    if "FP" in finding["text"].upper() or "FALSE POSITIVE" in finding["text"].upper():
        candidates.extend(["FP", "false positive", "FALSE_POSITIVE"])
    for c in candidates:
        if c in issues_text:
            return True
    return False


def check_floating_findings(
    sources: list[tuple[str, str]],
    issues_text: str,
) -> tuple[list[dict], list[dict]]:
    """Check for floating findings.

    Args:
        sources: List of (source_name, content) tuples.
        issues_text: Full text of ISSUES.md (or empty if file missing).

    Returns:
        (documented_findings, floating_findings)
    """
    documented = []
    floating = []
    for src_name, content in sources:
        for finding in scan_text_for_findings(content):
            finding["source"] = src_name
            if is_finding_documented(finding, issues_text):
                documented.append(finding)
            else:
                floating.append(finding)
    return documented, floating


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="session_close",
        description=(
            "Auto-detect floating findings at session close. Refuses to "
            "let the session close if findings are unwritten. Implements L.5.1."
        ),
    )
    parser.add_argument("--recent-files", nargs="*", type=Path, default=[],
                        help="Files to scan for finding patterns.")
    parser.add_argument("--log", type=Path, default=None,
                        help="Path to a session log file to scan.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print findings but don't enforce.")
    parser.add_argument("--issues-file", type=Path, default=None,
                        help="Path to ISSUES.md. Default: ml/audit_docs/ISSUES.md")
    args = parser.parse_args()

    # Collect sources to scan
    sources: list[tuple[str, str]] = []
    for f in args.recent_files:
        if f.exists():
            try:
                sources.append((str(f), f.read_text()))
            except Exception as e:
                print(f"WARNING: could not read {f}: {e}")
    if args.log and args.log.exists():
        try:
            sources.append((str(args.log), args.log.read_text()))
        except Exception as e:
            print(f"WARNING: could not read {args.log}: {e}")

    if not sources:
        print("No sources provided. Pass --recent-files or --log.")
        return 0

    # Read ISSUES.md
    issues_path = args.issues_file or Path("ml/audit_docs/ISSUES.md")
    issues_text = ""
    if issues_path.exists():
        issues_text = issues_path.read_text()
    else:
        print(f"WARNING: {issues_path} not found. All findings will be flagged as floating.")

    # Check
    documented, floating = check_floating_findings(sources, issues_text)

    print("=" * 70)
    print("SESSION CLOSE — L.5.1 FLOATING FINDINGS CHECK")
    print("=" * 70)
    print()
    print(f"Documented findings: {len(documented)}")
    print(f"Floating findings:   {len(floating)}")
    print()
    if documented:
        print("✓ DOCUMENTED (have ISSUES.md entry):")
        for f in documented:
            print(f"  [{f['source']}:{f['line']}] {f['pattern']}: {f['text'][:100]}")
        print()
    if floating:
        print("✗ FLOATING (no ISSUES.md entry — must be written before close):")
        for f in floating:
            print(f"  [{f['source']}:{f['line']}] {f['pattern']}: {f['text'][:100]}")
        print()

    if args.dry_run:
        print("(DRY RUN — session close NOT blocked)")
        return 0

    if floating:
        print("=" * 70)
        print(f"SESSION CLOSE BLOCKED: {len(floating)} floating findings.")
        print("Write ISSUES.md entries for each, then re-run.")
        print("=" * 70)
        return 1

    print("No floating findings — session close OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
