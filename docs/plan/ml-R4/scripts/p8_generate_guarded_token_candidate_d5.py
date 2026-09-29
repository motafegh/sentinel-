#!/usr/bin/env python3
"""Generate the protected-local D5 full guarded-token physical candidate.

This driver is intentionally separate from the bounded D4 validator. It derives
the exact 22,540-identity population from the immutable R4-D-011 parent, builds
fresh R4-D-012 guarded tokens, and writes generation evidence for later D6
review.

A successful D5 generation is not physical acceptance and does not authorize
training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

REPO_ROOT = Path(__file__).resolve().parents[4]
DATA_ROOT = REPO_ROOT / "data_module/data"

from sentinel_data.preprocessing.r4_versions import (  # noqa: E402
    GUARDED_REPRESENTATION_ROOT_NAME,
)
from sentinel_data.representation.r4_guarded_token_candidate import (  # noqa: E402
    GuardedBuildResult,
    build_guarded_token_full_candidate,
)

DEFAULT_ACCEPTANCE = (
    REPO_ROOT
    / "docs/plan/ml-R4/evidence/2026-09-02_v10_v26_physical_acceptance/acceptance.json"
)
DEFAULT_PREPROCESSED = DATA_ROOT / "sentinel-preprocessed-r4-v2"
DEFAULT_PARENT = (
    DATA_ROOT
    / "v10-v26-full-candidate-attempt-2026-09-01-a"
    / "representations-r4-v3-candidate"
)
REPORT_SCHEMA = "sentinel-r4-guarded-token-d5-generation-v1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _portable_path(path: Path) -> str:
    path = path.resolve()
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def _max_rss_mb() -> float:
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acceptance", type=Path, default=DEFAULT_ACCEPTANCE)
    parser.add_argument("--preprocessed-root", type=Path, default=DEFAULT_PREPROCESSED)
    parser.add_argument("--parent-root", type=Path, default=DEFAULT_PARENT)
    parser.add_argument(
        "--work-root",
        type=Path,
        required=True,
        help="Fresh protected-local D5 attempt directory.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=250,
        help="Print progress every N completed identities.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.progress_every < 1:
        raise ValueError("--progress-every must be >= 1")

    acceptance = args.acceptance.resolve()
    preprocessed_root = args.preprocessed_root.resolve()
    parent_root = args.parent_root.resolve()
    work_root = args.work_root.resolve()

    if work_root.exists() and any(work_root.iterdir()):
        raise FileExistsError(f"D5 work root is not empty; use a fresh path: {work_root}")
    work_root.mkdir(parents=True, exist_ok=True)
    output_root = work_root / GUARDED_REPRESENTATION_ROOT_NAME

    started = time.perf_counter()
    rss_before = _max_rss_mb()

    def progress(index: int, total: int, result: GuardedBuildResult) -> None:
        if index == 1 or index == total or index % args.progress_every == 0:
            elapsed = time.perf_counter() - started
            rate = float(index) / elapsed if elapsed > 0 else 0.0
            print(
                f"[D5] {index}/{total} "
                f"selector={result.effective_selector} "
                f"elapsed={elapsed:.1f}s rate={rate:.2f}/s",
                flush=True,
            )

    manifest = build_guarded_token_full_candidate(
        acceptance_path=acceptance,
        repo_root=REPO_ROOT,
        preprocessed_root=preprocessed_root,
        parent_root=parent_root,
        output_root=output_root,
        progress_callback=progress,
    )

    elapsed = time.perf_counter() - started
    manifest_path = output_root / "guarded_candidate_manifest.json"
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "status": "PASS_FULL_D5_REVIEW_REQUIRED",
        "source_commit": manifest["source_commit"],
        "acceptance_manifest_sha256": _sha256_file(acceptance),
        "parent_root": _portable_path(parent_root),
        "preprocessed_root": _portable_path(preprocessed_root),
        "work_root": _portable_path(work_root),
        "candidate_root": _portable_path(output_root),
        "candidate_manifest": _portable_path(manifest_path),
        "candidate_manifest_sha256": _sha256_file(manifest_path),
        "binding_digest_sha256": manifest["binding_digest_sha256"],
        "representation_lineage": manifest["representation_lineage"],
        "selector_policy": manifest["selector_policy"],
        "control_selector": manifest["control_selector"],
        "transformers_version": manifest["transformers_version"],
        "graph_schema_version": manifest["graph_schema_version"],
        "graph_extractor_version": manifest["graph_extractor_version"],
        "frozen_token_shape": manifest["frozen_token_shape"],
        "contracts_requested": manifest["contracts_requested"],
        "contracts_written": manifest["contracts_written"],
        "effective_selector_counts": manifest["effective_selector_counts"],
        "guarded_contracts": manifest["guarded_contracts"],
        "control_fallback_contracts": manifest["control_fallback_contracts"],
        "full_population": manifest["full_population"],
        "runtime": {
            "elapsed_seconds": elapsed,
            "max_rss_mb_before": rss_before,
            "max_rss_mb_after": _max_rss_mb(),
        },
        "physical_acceptance": False,
        "d6_authorized": False,
        "training_authorized": False,
        "review_required": True,
        "decision_boundary": (
            "A successful D5 generation creates and binds the full candidate only. "
            "It does not physically accept the lineage, authorize D6 acceptance "
            "without review, or authorize training."
        ),
    }
    report_path = work_root / "d5_guarded_token_generation_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
