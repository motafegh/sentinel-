#!/usr/bin/env python3
"""Deterministic local rebuild driver for the Phase-8 real-DATA repair.

This script is the supported local execution seam for the repaired v2 lineage.
It does not launch training and does not overwrite historical DATA artifacts.
Each stage writes to a new versioned path and fails closed when a destination is
already populated.

Typical execution from repository root, after pulling the exact repair commit:

    PYTHONPATH=.:data_module ./ml/.venv/bin/python \
      docs/plan/ml-R4/scripts/p8_rebuild_repaired_data.py prerequisites

    # Then run each stage in order so failures can be inspected before the next:
    ... p8_rebuild_repaired_data.py preprocess --source dive --workers 8
    ... p8_rebuild_repaired_data.py preprocess --source smartbugs_curated --workers 4
    ... p8_rebuild_repaired_data.py preprocess --source solidifi --workers 4
    ... p8_rebuild_repaired_data.py claims
    ... p8_rebuild_repaired_data.py grouping
    ... p8_rebuild_repaired_data.py represent --source dive
    ... p8_rebuild_repaired_data.py represent --source smartbugs_curated
    ... p8_rebuild_repaired_data.py represent --source solidifi
    ... p8_rebuild_repaired_data.py recover-representations --source dive \
          --failed-attempt-dir /path/to/immutable/failed/dive
    ... p8_rebuild_repaired_data.py publish
    ... p8_rebuild_repaired_data.py bind
    ... p8_rebuild_repaired_data.py summarize

A passing ``bind`` is necessary but not sufficient to re-authorize the
100-epoch run.  Token-coverage evidence and a bounded repaired-data GPU smoke
must still be reviewed and recorded.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[4]
DATA_ROOT = REPO_ROOT / "data_module" / "data"
POLICY_PATH = REPO_ROOT / "docs/plan/ml-R4/specs/data_vnext_policy_v1.json"
ACTIVE_SOURCES = ("dive", "smartbugs_curated", "solidifi")

DEFAULT_PREPROCESSED_ROOT = DATA_ROOT / "sentinel-preprocessed-r4-v2"
DEFAULT_REPRESENTATIONS_ROOT = DATA_ROOT / "representations-r4-v2"
DEFAULT_BUILD_ROOT = DATA_ROOT / "r4-v2-build"
DEFAULT_PUBLICATION_ROOT = DATA_ROOT / "exports" / "sentinel-r4-vnext-v2"
DEFAULT_DIVE_LABELS = DATA_ROOT / "raw_staging/dive_labels/DIVE_Labels.csv"


def _emit(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, default=str))


def _tracked_clean() -> tuple[bool, str]:
    try:
        status = subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "status", "--porcelain", "--untracked-files=no"],
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        return False, f"cannot inspect git status: {exc}"
    return not bool(status), status


def _git_head() -> str:
    return subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True
    ).strip()


def _source_raw_dir(source: str) -> Path:
    return DATA_ROOT / "raw" / source


def _source_manifest(source: str) -> Path:
    return _source_raw_dir(source) / "ingestion_manifest.json"


def _source_preprocessed(root: Path, source: str) -> Path:
    return root / source


def _source_representations(root: Path, source: str) -> Path:
    return root / source


def _ensure_fresh(path: Path, description: str) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(
            f"{description} already contains files: {path}. "
            "Repaired builds are immutable; inspect/archive the previous attempt "
            "and choose a fresh root rather than overwriting it."
        )


def _source_dirs(preprocessed_root: Path) -> dict[str, Path]:
    return {
        source: _source_preprocessed(preprocessed_root, source)
        for source in ACTIVE_SOURCES
    }


def cmd_prerequisites(args: argparse.Namespace) -> int:
    from sentinel_data.preprocessing.r4_raw_verifier import verify_manifest_source

    checks: list[dict[str, Any]] = []
    clean, detail = _tracked_clean()
    checks.append({"check": "tracked_worktree_clean", "passed": clean, "detail": detail})
    checks.append({"check": "git_head", "passed": True, "detail": _git_head()})
    checks.append(
        {
            "check": "accepted_policy_present",
            "passed": POLICY_PATH.is_file(),
            "detail": str(POLICY_PATH.relative_to(REPO_ROOT)),
        }
    )
    for source in ACTIVE_SOURCES:
        raw = _source_raw_dir(source)
        manifest = _source_manifest(source)
        checks.append(
            {
                "check": f"raw_source:{source}",
                "passed": raw.is_dir(),
                "detail": str(raw),
            }
        )
        verification = verify_manifest_source(source, raw, manifest)
        checks.append(
            {
                "check": f"raw_manifest_bytes:{source}",
                "passed": bool(verification.get("passed")),
                "detail": {
                    "manifest_records": verification.get("manifest_records"),
                    "manifest_sha256": verification.get("manifest_sha256"),
                    "errors_total": verification.get("errors_total"),
                    "errors": (verification.get("errors") or [])[:3],
                },
            }
        )
        checks.append(
            {
                "check": f"ingestion_manifest:{source}",
                "passed": manifest.is_file(),
                "detail": str(manifest),
            }
        )
    checks.append(
        {
            "check": "dive_labels_csv",
            "passed": args.dive_labels.is_file(),
            "detail": str(args.dive_labels),
        }
    )
    solc_root = Path.home() / ".solc-select" / "artifacts"
    installed = sorted(
        p.name.removeprefix("solc-")
        for p in solc_root.glob("solc-*")
        if p.is_dir()
    ) if solc_root.is_dir() else []
    checks.append(
        {
            "check": "historical_solc_artifacts",
            "passed": bool(installed),
            "detail": installed,
        }
    )
    try:
        import pyarrow  # noqa: F401
        import torch  # noqa: F401
        import torch_geometric  # noqa: F401
        import transformers  # noqa: F401
        dependency_ok = True
        dependency_detail = "pyarrow/torch/torch_geometric/transformers importable"
    except ImportError as exc:
        dependency_ok = False
        dependency_detail = str(exc)
    checks.append(
        {
            "check": "runtime_dependencies",
            "passed": dependency_ok,
            "detail": dependency_detail,
        }
    )
    passed = all(item["passed"] for item in checks)
    _emit({"passed": passed, "checks": checks})
    return 0 if passed else 2


def cmd_preprocess(args: argparse.Namespace) -> int:
    from sentinel_data.preprocessing.r4_pipeline import run_repaired_source

    out = _source_preprocessed(args.preprocessed_root, args.source)
    _ensure_fresh(out, f"repaired preprocessing output for {args.source}")
    result = run_repaired_source(
        args.source,
        _source_raw_dir(args.source),
        _source_manifest(args.source),
        out,
        n_workers=args.workers,
        limit=args.limit,
    )
    _emit(result.__dict__)
    # Explicit drops are part of the local acceptance evidence.  They remain in
    # dropped.csv and must be adjudicated by p8_audit_repaired_lineage.py; the
    # preprocessing command itself succeeded if it completed deterministically.
    return 0


def cmd_claims(args: argparse.Namespace) -> int:
    from sentinel_data.vnext.r4_source_claims import build_claim_index

    output = args.build_root / "source_claims.jsonl"
    if output.exists():
        raise FileExistsError(f"claim index already exists: {output}")
    for source, directory in _source_dirs(args.preprocessed_root).items():
        if not directory.is_dir():
            raise FileNotFoundError(f"missing repaired preprocessing for {source}: {directory}")
    result = build_claim_index(
        _source_dirs(args.preprocessed_root),
        POLICY_PATH,
        output,
        dive_labels_csv=args.dive_labels,
    )
    if result["target_zero_claims"] != 0:
        raise AssertionError("repaired claim index contains target zero")
    _emit(result)
    return 0


def cmd_grouping(args: argparse.Namespace) -> int:
    from sentinel_data.preprocessing.r4_grouping import build_grouping

    output = args.build_root / "grouping.json"
    if output.exists():
        raise FileExistsError(f"grouping manifest already exists: {output}")
    result = build_grouping(_source_dirs(args.preprocessed_root), output)
    _emit(result.__dict__)
    return 0


def cmd_represent(args: argparse.Namespace) -> int:
    from sentinel_data.representation.r4_orchestrator import represent_repaired_source

    preprocessed = _source_preprocessed(args.preprocessed_root, args.source)
    output = _source_representations(args.representations_root, args.source)
    _ensure_fresh(output, f"repaired representation output for {args.source}")
    result = represent_repaired_source(
        args.source,
        preprocessed,
        output,
        limit=args.limit,
        n_workers=args.workers,
    )
    _emit(result.__dict__)
    return 0 if result.representations_failed == 0 else 1


def cmd_recover_representations(args: argparse.Namespace) -> int:
    from sentinel_data.representation.r4_orchestrator import (
        recover_failed_representations,
    )

    preprocessed = _source_preprocessed(args.preprocessed_root, args.source)
    output = _source_representations(args.representations_root, args.source)
    _ensure_fresh(output, f"repaired representation recovery output for {args.source}")
    result = recover_failed_representations(
        args.source,
        preprocessed,
        args.failed_attempt_dir,
        output,
        n_workers=args.workers,
    )
    _emit(result.__dict__)
    return 0 if result.representations_failed == 0 else 1


def cmd_publish(args: argparse.Namespace) -> int:
    from sentinel_data.vnext.r4_builder import build_repaired_publication

    claims = args.build_root / "source_claims.jsonl"
    grouping = args.build_root / "grouping.json"
    ledger = args.build_root / "evidence_ledger_v2.parquet"
    ledger_manifest = args.build_root / "evidence_ledger_v2_manifest.json"
    for path in (claims, grouping, ledger, ledger_manifest):
        if not path.is_file():
            raise FileNotFoundError(path)
    _ensure_fresh(args.publication_root, "repaired vNext publication")
    manifest = build_repaired_publication(
        claims_path=claims,
        grouping_path=grouping,
        policy_path=POLICY_PATH,
        representation_root=args.representations_root,
        output_dir=args.publication_root,
        ledger_path=ledger,
        ledger_manifest_path=ledger_manifest,
    )
    _emit(manifest)
    return 0


def cmd_bind(args: argparse.Namespace) -> int:
    from sentinel_data.vnext.r4_binding import bind_repaired_publication

    report = bind_repaired_publication(
        publication_dir=args.publication_root,
        representations_root=args.representations_root,
    )
    _emit(report)
    return 0 if report["passed"] else 1


def cmd_summarize(args: argparse.Namespace) -> int:
    summary: dict[str, Any] = {
        "git_head": _git_head(),
        "preprocessed_root": str(args.preprocessed_root),
        "representations_root": str(args.representations_root),
        "publication_root": str(args.publication_root),
        "training_authorized": False,
        "reasons_training_not_authorized": [
            "physical repaired-DATA counts and attrition must be reviewed",
            "token coverage evidence must be reviewed; no adequacy threshold is approved",
            "bounded repaired-data GPU smoke has not yet been recorded",
        ],
    }
    manifest_path = args.publication_root / "manifest.json"
    binding_path = args.publication_root / "representation_binding_report.json"
    if manifest_path.is_file():
        summary["publication_manifest"] = json.loads(manifest_path.read_text())
    if binding_path.is_file():
        binding = json.loads(binding_path.read_text())
        summary["binding"] = {
            key: binding.get(key)
            for key in (
                "passed",
                "required_contracts",
                "checked_contracts",
                "checked_files",
                "missing_or_invalid_total",
                "binding_digest_sha256",
                "token_coverage",
            )
        }
    _emit(summary)
    return 0


def _add_common_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--preprocessed-root",
        type=Path,
        default=DEFAULT_PREPROCESSED_ROOT,
    )
    parser.add_argument(
        "--representations-root",
        type=Path,
        default=DEFAULT_REPRESENTATIONS_ROOT,
    )
    parser.add_argument("--build-root", type=Path, default=DEFAULT_BUILD_ROOT)
    parser.add_argument(
        "--publication-root",
        type=Path,
        default=DEFAULT_PUBLICATION_ROOT,
    )
    parser.add_argument("--dive-labels", type=Path, default=DEFAULT_DIVE_LABELS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    prereq = sub.add_parser("prerequisites")
    _add_common_paths(prereq)

    preprocess = sub.add_parser("preprocess")
    _add_common_paths(preprocess)
    preprocess.add_argument("--source", choices=ACTIVE_SOURCES, required=True)
    preprocess.add_argument("--workers", type=int, default=1)
    preprocess.add_argument("--limit", type=int)

    claims = sub.add_parser("claims")
    _add_common_paths(claims)

    grouping = sub.add_parser("grouping")
    _add_common_paths(grouping)

    represent = sub.add_parser("represent")
    _add_common_paths(represent)
    represent.add_argument("--source", choices=ACTIVE_SOURCES, required=True)
    represent.add_argument("--limit", type=int)
    represent.add_argument("--workers", type=int, default=1)

    recover = sub.add_parser("recover-representations")
    _add_common_paths(recover)
    recover.add_argument("--source", choices=ACTIVE_SOURCES, required=True)
    recover.add_argument("--failed-attempt-dir", type=Path, required=True)
    recover.add_argument("--workers", type=int, default=1)

    publish = sub.add_parser("publish")
    _add_common_paths(publish)

    bind = sub.add_parser("bind")
    _add_common_paths(bind)

    summarize = sub.add_parser("summarize")
    _add_common_paths(summarize)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        return {
            "prerequisites": cmd_prerequisites,
            "preprocess": cmd_preprocess,
            "claims": cmd_claims,
            "grouping": cmd_grouping,
            "represent": cmd_represent,
            "recover-representations": cmd_recover_representations,
            "publish": cmd_publish,
            "bind": cmd_bind,
            "summarize": cmd_summarize,
        }[args.command](args)
    except (OSError, ValueError, RuntimeError, AssertionError) as exc:
        print(f"R4 REPAIRED DATA ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
