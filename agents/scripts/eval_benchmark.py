#!/usr/bin/env python3
"""
SENTINEL agents pipeline evaluation comparator — thin CLI wrapper.

P0.1 (2026-06-23): delegates to `python -m src.eval.run_benchmark`.
Kept for backward compatibility; new callers should use the module runner.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import NoReturn

from src.eval.run_benchmark import main as runner_main


def main() -> None:
    # Filter out deprecated legacy args before delegating to the runner.
    deprecated = {"--output", "--metrics", "--positive-verdicts"}
    filtered = []
    skip_next = False
    has_name = False
    for i, arg in enumerate(sys.argv[1:], start=1):
        if skip_next:
            skip_next = False
            continue
        if arg in deprecated:
            # The next token is the value — skip both.
            if arg in ("--output", "--metrics", "--positive-verdicts"):
                skip_next = True
            continue
        if arg.startswith("--output=") or arg.startswith("--metrics=") or arg.startswith("--positive-verdicts="):
            continue
        if arg == "--name" or arg.startswith("--name="):
            has_name = True
        filtered.append(arg)

    # If --name not given, derive one.
    if not has_name:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        filtered = ["--name", f"legacy_{ts}"] + filtered

    # Mutate sys.argv so the runner sees the cleaned args.
    sys.argv[1:] = filtered
    runner_main()


if __name__ == "__main__":
    main()
