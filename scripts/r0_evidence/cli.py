"""Command-line capture, environment manifest, and coverage validation."""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.r0_evidence.matrix import MATRIX_ROW_IDS
from scripts.r0_evidence.model import (
    canonical_json_bytes,
    load_evidence_records,
    redact_text,
    sha256_bytes,
    sha256_file,
    validate_coverage,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


_PLACEHOLDER_PATTERN = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")
_FULL_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _render_command_part(part: str, substitutions: dict[str, str]) -> str:
    """Replace only declared command placeholders, preserving code/JSON braces."""

    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in substitutions:
            raise ValueError(f"Unknown command placeholder {{{name}}}")
        return substitutions[name]

    return _PLACEHOLDER_PATTERN.sub(replace, part)


def validate_command_manifest(manifest: Any) -> list[str]:
    """Return every command-manifest error without executing a probe."""

    if not isinstance(manifest, dict):
        return ["command manifest must be an object"]

    errors: list[str] = []
    if manifest.get("schema_version") != "1" or manifest.get("kind") != "r0_command_manifest":
        errors.append("unsupported command manifest schema/kind")
    if not _FULL_GIT_SHA.fullmatch(str(manifest.get("baseline_commit", ""))):
        errors.append("baseline_commit must be a full lowercase Git SHA")

    fixtures = manifest.get("fixture_sha256", {})
    if not isinstance(fixtures, dict) or not all(
        isinstance(path, str)
        and path
        and not Path(path).is_absolute()
        and _SHA256.fullmatch(str(digest))
        for path, digest in fixtures.items()
    ):
        errors.append("fixture_sha256 must map relative paths to lowercase SHA-256 values")
    if fixtures and not manifest.get("fixture_root_variable"):
        errors.append("fixture_root_variable is required when fixtures are declared")

    probes = manifest.get("probes")
    if not isinstance(probes, list) or not probes:
        errors.append("probes must be a non-empty list")
        return errors

    probe_ids: list[str] = []
    row_ids: list[str] = []
    for index, probe in enumerate(probes):
        prefix = f"probes[{index}]"
        if not isinstance(probe, dict):
            errors.append(f"{prefix} must be an object")
            continue
        for field in ("probe_id", "matrix_row_id", "contract_version", "cwd"):
            if not probe.get(field):
                errors.append(f"{prefix}.{field} is required")
        if probe.get("contract_version") != "1":
            errors.append(f"{prefix}.contract_version must be 1")
        argv = probe.get("argv")
        if (
            not isinstance(argv, list)
            or not argv
            or not all(isinstance(item, str) and item for item in argv)
        ):
            errors.append(f"{prefix}.argv must be a non-empty string list")
        references = probe.get("test_references")
        if (
            not isinstance(references, list)
            or not references
            or not all(isinstance(item, str) and item for item in references)
        ):
            errors.append(f"{prefix}.test_references must be a non-empty string list")
        if probe.get("probe_id"):
            probe_ids.append(str(probe["probe_id"]))
        if probe.get("matrix_row_id"):
            row_ids.append(str(probe["matrix_row_id"]))

    if len(probe_ids) != len(set(probe_ids)):
        errors.append("probe_id values must be unique")
    if len(row_ids) != len(set(row_ids)):
        errors.append("matrix_row_id values must be unique")
    if set(row_ids) != MATRIX_ROW_IDS:
        missing = sorted(MATRIX_ROW_IDS - set(row_ids))
        unknown = sorted(set(row_ids) - MATRIX_ROW_IDS)
        if missing:
            errors.append(f"missing matrix rows: {', '.join(missing)}")
        if unknown:
            errors.append(f"unknown matrix rows: {', '.join(unknown)}")
    return errors


def _verify_command_fixtures(manifest: dict[str, Any], variables: dict[str, str]) -> None:
    fixtures = manifest.get("fixture_sha256", {})
    if not fixtures:
        return
    root_variable = str(manifest["fixture_root_variable"])
    if root_variable not in variables:
        raise ValueError(f"Missing fixture root variable: {root_variable}")
    root = Path(variables[root_variable]).resolve()
    for relative, expected in fixtures.items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"Command fixture is missing or outside its root: {relative}")
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(
                f"Command fixture digest mismatch for {relative}: expected {expected}, got {actual}"
            )


def create_environment_manifest(workspace: Path) -> dict[str, Any]:
    workspace = workspace.resolve()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=workspace, text=True).strip()
    status = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=workspace, text=True
    ).splitlines()

    locks: list[dict[str, str]] = []
    for relative in ("poetry.lock", "agents/poetry.lock", "ml/poetry.lock"):
        path = workspace / relative
        if path.is_file():
            locks.append({"path": relative, "sha256": sha256_file(path)})

    environment_contract = (
        f"{platform.system().lower()}-{platform.machine().lower()}-"
        f"python-{sys.version_info.major}.{sys.version_info.minor}"
    )
    return {
        "schema_version": "1",
        "kind": "r0_environment_manifest",
        "environment_contract": environment_contract,
        "workspace_commit": head,
        "workspace_dirty": bool(status),
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "lockfiles": locks,
    }


def capture_probe(
    *,
    command_manifest_path: Path,
    probe_id: str,
    phase: str,
    workspace: Path,
    environment_manifest_path: Path,
    output: Path,
    variables: dict[str, str],
) -> dict[str, Any]:
    manifest = json.loads(command_manifest_path.read_text(encoding="utf-8"))
    manifest_errors = validate_command_manifest(manifest)
    if manifest_errors:
        raise ValueError("Invalid command manifest: " + "; ".join(manifest_errors))
    probes = [probe for probe in manifest["probes"] if probe["probe_id"] == probe_id]
    if len(probes) != 1:
        raise ValueError(f"Expected one probe named {probe_id!r}, found {len(probes)}")
    probe = probes[0]

    workspace = workspace.resolve()
    environment = json.loads(environment_manifest_path.read_text(encoding="utf-8"))
    if environment.get("workspace_dirty") is not False:
        raise ValueError("Evidence capture requires a clean workspace manifest")
    expected_commit = (
        manifest["baseline_commit"] if phase == "before" else environment.get("workspace_commit")
    )
    if environment.get("workspace_commit") != expected_commit:
        raise ValueError(f"Environment workspace commit does not match the {phase} evidence commit")
    current_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=workspace, text=True
    ).strip()
    current_dirty = bool(
        subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=workspace, text=True
        ).splitlines()
    )
    if current_commit != environment.get("workspace_commit") or current_dirty:
        raise ValueError("Workspace changed after the environment manifest was created")

    _verify_command_fixtures(manifest, variables)
    substitutions = {"workspace": str(workspace), "python": sys.executable, **variables}
    argv = [_render_command_part(part, substitutions) for part in probe["argv"]]
    cwd = (workspace / probe.get("cwd", ".")).resolve()
    if not cwd.is_relative_to(workspace) or not cwd.is_dir():
        raise ValueError("Probe cwd must be an existing directory inside the workspace")

    started_at = _utc_now()
    completed = subprocess.run(
        argv,
        cwd=cwd,
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        check=False,
    )
    finished_at = _utc_now()
    stdout = redact_text(completed.stdout, workspace=workspace)
    stderr = redact_text(completed.stderr, workspace=workspace)

    payload: dict[str, Any]
    try:
        last_line = next(line for line in reversed(stdout.splitlines()) if line.strip())
        parsed = json.loads(last_line)
        if not isinstance(parsed, dict):
            raise ValueError("probe result must be an object")
        payload = parsed
    except (StopIteration, json.JSONDecodeError, ValueError) as exc:
        payload = {
            "invariant_passed": False,
            "status": "blocked",
            "assertions": [
                {
                    "name": "probe_result_contract",
                    "passed": False,
                    "detail": f"Probe did not emit a final JSON object: {exc}",
                }
            ],
        }

    if completed.returncode != 0:
        payload["invariant_passed"] = False
        payload["status"] = "blocked"
        payload.setdefault("assertions", []).append(
            {
                "name": "probe_exit_code",
                "passed": False,
                "detail": f"exit_code={completed.returncode}",
            }
        )

    comparison_material = {
        "probe_id": probe["probe_id"],
        "contract_version": probe["contract_version"],
        "argv_template": probe["argv"],
        "cwd": probe.get("cwd", "."),
        "test_references": probe["test_references"],
        "fixtures": {
            **manifest.get("fixture_sha256", {}),
            **probe.get("fixture_sha256", {}),
        },
        "environment_contract": environment["environment_contract"],
    }
    comparison_key = sha256_bytes(canonical_json_bytes(comparison_material))
    candidate_commit = current_commit if phase == "after" else None

    evidence_commit = candidate_commit or manifest["baseline_commit"]
    record = {
        "schema_version": "1",
        "kind": "r0_evidence_record",
        "record_id": (f"{probe['matrix_row_id']}:{phase}:{probe_id}:{evidence_commit[:12]}"),
        "matrix_row_id": probe["matrix_row_id"],
        "phase": phase,
        "baseline_commit": manifest["baseline_commit"],
        "candidate_commit": candidate_commit,
        "comparison_key": comparison_key,
        "probe": {
            "probe_id": probe["probe_id"],
            "contract_version": probe["contract_version"],
            "argv_template": probe["argv"],
            "resolved_argv": [redact_text(item, workspace=workspace) for item in argv],
            "cwd": probe.get("cwd", "."),
        },
        "environment_manifest": {
            "path": redact_text(str(environment_manifest_path.resolve()), workspace=workspace),
            "sha256": sha256_file(environment_manifest_path),
            "environment_contract": environment["environment_contract"],
        },
        "execution": {
            "started_at": started_at,
            "finished_at": finished_at,
            "exit_code": completed.returncode,
            "stdout_sha256": sha256_bytes(completed.stdout.encode("utf-8")),
            "stderr_sha256": sha256_bytes(completed.stderr.encode("utf-8")),
            "stdout": stdout,
            "stderr": stderr,
        },
        "outcome": {
            "status": payload.get("status", "pass" if payload.get("invariant_passed") else "fail"),
            "invariant_passed": bool(payload.get("invariant_passed", False)),
            "assertions": payload.get("assertions", []),
        },
        "test_references": probe["test_references"],
        "review": {"status": "pending", "reviewer": None, "decided_at": None},
    }
    _write_json(output, record)
    return record


def _parse_variables(values: list[str]) -> dict[str, str]:
    variables: dict[str, str] = {}
    for item in values:
        if "=" not in item:
            raise ValueError(f"--var must be NAME=VALUE, got {item!r}")
        name, value = item.split("=", 1)
        if not name:
            raise ValueError("--var name cannot be empty")
        variables[name] = value
    return variables


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m scripts.r0_evidence")
    subparsers = parser.add_subparsers(dest="command", required=True)

    environment_parser = subparsers.add_parser("environment")
    environment_parser.add_argument("--workspace", type=Path, required=True)
    environment_parser.add_argument("--output", type=Path, required=True)

    capture_parser = subparsers.add_parser("capture")
    capture_parser.add_argument("--command-manifest", type=Path, required=True)
    capture_parser.add_argument("--probe-id", required=True)
    capture_parser.add_argument("--phase", choices=("before", "after"), required=True)
    capture_parser.add_argument("--workspace", type=Path, required=True)
    capture_parser.add_argument("--environment-manifest", type=Path, required=True)
    capture_parser.add_argument("--output", type=Path, required=True)
    capture_parser.add_argument("--var", action="append", default=[])

    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--evidence-dir", type=Path, required=True)
    validate_parser.add_argument("--output", type=Path)

    args = parser.parse_args(argv)
    if args.command == "environment":
        payload = create_environment_manifest(args.workspace)
        _write_json(args.output, payload)
        print(json.dumps(payload, sort_keys=True))
        return 0
    if args.command == "capture":
        record = capture_probe(
            command_manifest_path=args.command_manifest,
            probe_id=args.probe_id,
            phase=args.phase,
            workspace=args.workspace,
            environment_manifest_path=args.environment_manifest,
            output=args.output,
            variables=_parse_variables(args.var),
        )
        print(json.dumps(record["outcome"], sort_keys=True))
        return 0

    report = validate_coverage(load_evidence_records(args.evidence_dir))
    if args.output:
        _write_json(args.output, report)
    print(json.dumps(report, sort_keys=True))
    return 0 if report["complete"] else 1


__all__ = [
    "capture_probe",
    "create_environment_manifest",
    "main",
    "validate_command_manifest",
]
