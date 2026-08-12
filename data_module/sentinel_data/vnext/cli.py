"""Command-line surface for the additive DATA vNext v2 overlay."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .builder import DATASET_VERSION, build_vnext_overlay
from .publication import bind_semantic_validation_report, verify_publication_bindings
from .representations import bind_representation_report, verify_local_representations
from .validator import validate_vnext_overlay

_REPO_ROOT = Path(__file__).resolve().parents[3]
_R4_ROOT = _REPO_ROOT / "docs/plan/ml-R4"
_DEFAULT_EXPORT = _REPO_ROOT / "data_module/data/exports" / DATASET_VERSION


def _defaults() -> dict[str, Path]:
    return {
        "ledger": _R4_ROOT / "ledger/evidence_ledger_v1.parquet",
        "policy": _R4_ROOT / "specs/data_vnext_policy_v1.json",
        "roles": _R4_ROOT / "manifests/p6_contract_role_manifest.jsonl",
        "partition": _R4_ROOT / "manifests/p6_partition_manifest.json",
        "unsupported": _R4_ROOT / "manifests/p6_unsupported_roles.json",
        "acceptance": _R4_ROOT / "manifests/p6_untouched_acceptance_manifest.json",
        "label_schema": _R4_ROOT / "schemas/data_vnext_label_state_v1.schema.json",
        "export": _DEFAULT_EXPORT,
        "representations": _REPO_ROOT / "data_module/data/representations",
    }


def _print(value: object) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, default=str))


def _build(args: argparse.Namespace) -> int:
    manifest = build_vnext_overlay(
        ledger_path=args.ledger,
        policy_path=args.policy,
        contract_roles_path=args.roles,
        partition_manifest_path=args.partition,
        unsupported_roles_path=args.unsupported,
        acceptance_manifest_path=args.acceptance,
        label_schema_path=args.label_schema,
        output_dir=args.output,
        generation_commit=args.generation_commit,
    )
    report_path = args.output / "validation_report.json"
    report = validate_vnext_overlay(args.output, report_path=report_path)
    if not report["passed"]:
        _print(report)
        return 1
    bind_semantic_validation_report(args.output, report_path)
    publication = verify_publication_bindings(args.output)
    if not publication["passed"]:
        _print(publication)
        return 1
    _print({
        "passed": True,
        "manifest_status": manifest["status"],
        "output": str(args.output),
        "validation_report": str(report_path),
        "publication_bindings": publication,
    })
    return 0


def _validate(args: argparse.Namespace) -> int:
    report = validate_vnext_overlay(
        args.output,
        require_representation_binding=args.require_representation_binding,
        report_path=args.report,
    )
    publication = verify_publication_bindings(args.output)
    result = {
        "semantic_validation": report,
        "publication_bindings": publication,
        "passed": bool(report["passed"] and publication["passed"]),
    }
    _print(result)
    return 0 if result["passed"] else 1


def _verify_representations(args: argparse.Namespace) -> int:
    report_path = args.report or (args.output / "representation_binding_report.json")
    report = verify_local_representations(
        args.output,
        args.representations_root,
        report_path=report_path,
    )
    if report["passed"] and args.bind:
        bind_representation_report(args.output, report_path)
    _print(report)
    return 0 if report["passed"] else 1


def _local_gate(args: argparse.Namespace) -> int:
    pre = validate_vnext_overlay(args.output, require_representation_binding=False)
    publication_pre = verify_publication_bindings(args.output)
    if not pre["passed"] or not publication_pre["passed"]:
        _print({"passed": False, "stage": "semantic_precheck", "validation": pre, "publication": publication_pre})
        return 1

    rep_path = args.output / "representation_binding_report.json"
    rep = verify_local_representations(
        args.output,
        args.representations_root,
        report_path=rep_path,
    )
    if not rep["passed"]:
        _print({"passed": False, "stage": "representation_binding", "report": rep})
        return 1
    bind_representation_report(args.output, rep_path)

    final_path = args.output / "g7_validation_report.json"
    final = validate_vnext_overlay(
        args.output,
        require_representation_binding=True,
        report_path=final_path,
    )
    publication_final = verify_publication_bindings(args.output)
    result = {
        "passed": bool(final["passed"] and publication_final["passed"]),
        "stage": "g7_local_gate",
        "representation_binding_report": str(rep_path),
        "final_validation_report": str(final_path),
        "validation": final,
        "publication_bindings": publication_final,
    }
    _print(result)
    return 0 if result["passed"] else 1


def build_parser() -> argparse.ArgumentParser:
    d = _defaults()
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("build", help="Build and semantically validate the v2 overlay from frozen inputs")
    p.add_argument("--ledger", type=Path, default=d["ledger"])
    p.add_argument("--policy", type=Path, default=d["policy"])
    p.add_argument("--roles", type=Path, default=d["roles"])
    p.add_argument("--partition", type=Path, default=d["partition"])
    p.add_argument("--unsupported", type=Path, default=d["unsupported"])
    p.add_argument("--acceptance", type=Path, default=d["acceptance"])
    p.add_argument("--label-schema", type=Path, default=d["label_schema"])
    p.add_argument("--output", type=Path, default=d["export"])
    p.add_argument("--generation-commit", default=None)
    p.set_defaults(func=_build)

    p = sub.add_parser("validate", help="Validate a generated v2 overlay")
    p.add_argument("--output", type=Path, default=d["export"])
    p.add_argument("--report", type=Path, default=None)
    p.add_argument("--require-representation-binding", action="store_true")
    p.set_defaults(func=_validate)

    p = sub.add_parser("verify-representations", help="Verify local graph/token/sidecar files for all required contracts")
    p.add_argument("--output", type=Path, default=d["export"])
    p.add_argument("--representations-root", type=Path, default=d["representations"])
    p.add_argument("--report", type=Path, default=None)
    p.add_argument("--bind", action="store_true", help="Bind a successful report into manifest.json")
    p.set_defaults(func=_verify_representations)

    p = sub.add_parser("local-gate", help="Run the complete local physical-binding G7 gate")
    p.add_argument("--output", type=Path, default=d["export"])
    p.add_argument("--representations-root", type=Path, default=d["representations"])
    p.set_defaults(func=_local_gate)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
