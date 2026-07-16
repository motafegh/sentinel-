"""Reproducible environment identity and isolated probe execution policy."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

from scripts.r0_evidence.model import canonical_json_bytes, sha256_bytes, sha256_file

_RUNTIME_INVENTORY_SCRIPT = """
import json
import platform
import sys
from importlib.metadata import distributions

packages = []
for distribution in distributions():
    name = distribution.metadata.get("Name")
    if not name:
        raise RuntimeError("installed distribution is missing its canonical name")
    packages.append([name.lower().replace("_", "-"), distribution.version])
packages.sort()
print(json.dumps({
    "implementation": platform.python_implementation(),
    "version": platform.python_version(),
    "packages": packages,
}, sort_keys=True, separators=(",", ":")))
"""

PROBE_ENVIRONMENT_POLICY = {
    "version": "1",
    "inherited_keys": [
        "LANG",
        "LC_ALL",
        "PATH",
        "REQUESTS_CA_BUNDLE",
        "SSL_CERT_DIR",
        "SSL_CERT_FILE",
        "TEMP",
        "TMP",
        "TMPDIR",
    ],
    "isolated_home": True,
    "forced": {
        "NO_COLOR": "1",
        "PYTHONHASHSEED": "0",
        "SENTINEL_EVIDENCE_PROBE": "1",
    },
}


def runtime_identity(executable: str) -> dict[str, Any]:
    """Fingerprint an interpreter and its installed distribution inventory."""

    runtime_path = Path(executable).expanduser().absolute()
    if not runtime_path.is_file():
        raise FileNotFoundError(f"Python runtime does not exist: {runtime_path}")
    completed = subprocess.run(
        [str(runtime_path), "-c", _RUNTIME_INVENTORY_SCRIPT],
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(completed.stdout)
    packages = payload.pop("packages")
    if not isinstance(packages, list):
        raise ValueError(f"Runtime inventory was not a list: {runtime_path}")
    return {
        "implementation": payload["implementation"],
        "version": payload["version"],
        "package_count": len(packages),
        "packages_sha256": sha256_bytes(canonical_json_bytes(packages)),
    }


def environment_comparison_material(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Select every environment field that must match across a measurement pair."""

    return {
        "comparison_contract_version": manifest.get("comparison_contract_version"),
        "environment_contract": manifest.get("environment_contract"),
        "lockfiles": manifest.get("lockfiles"),
        "platform": manifest.get("platform"),
        "probe_environment_policy": manifest.get("probe_environment_policy"),
        "python": manifest.get("python"),
        "runtimes": manifest.get("runtimes"),
    }


def create_environment_manifest(
    workspace: Path,
    *,
    runtimes: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Describe the clean source and every Python runtime used by evidence probes."""

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

    runtime_paths = {"harness_python": sys.executable, **dict(runtimes or {})}
    if not all(name and executable for name, executable in runtime_paths.items()):
        raise ValueError("Runtime names and executable paths must be non-empty")
    runtime_manifests = {
        name: runtime_identity(executable) for name, executable in sorted(runtime_paths.items())
    }

    environment_contract = (
        f"{platform.system().lower()}-{platform.machine().lower()}-"
        f"python-{sys.version_info.major}.{sys.version_info.minor}"
    )
    manifest = {
        "schema_version": "1",
        "kind": "r0_environment_manifest",
        "comparison_contract_version": "2",
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
        "runtimes": runtime_manifests,
        "probe_environment_policy": deepcopy(PROBE_ENVIRONMENT_POLICY),
    }
    manifest["comparison_fingerprint"] = sha256_bytes(
        canonical_json_bytes(environment_comparison_material(manifest))
    )
    return manifest


def validate_environment_manifest(manifest: Any) -> list[str]:
    """Reject incomplete or self-inconsistent environment evidence."""

    if not isinstance(manifest, dict):
        return ["environment manifest must be an object"]
    errors: list[str] = []
    if manifest.get("schema_version") != "1" or manifest.get("kind") != "r0_environment_manifest":
        errors.append("unsupported environment manifest schema/kind")
    if manifest.get("comparison_contract_version") != "2":
        errors.append("environment comparison_contract_version must be 2")
    if not isinstance(manifest.get("environment_contract"), str) or not manifest.get(
        "environment_contract"
    ):
        errors.append("environment_contract must be a non-empty string")
    if not isinstance(manifest.get("workspace_commit"), str) or not _is_digest(
        manifest.get("workspace_commit"), length=40
    ):
        errors.append("environment workspace_commit must be a full lowercase Git SHA")
    if not isinstance(manifest.get("workspace_dirty"), bool):
        errors.append("environment workspace_dirty must be boolean")
    if not _is_version_identity(manifest.get("python")):
        errors.append("environment python identity is invalid")
    platform_identity = manifest.get("platform")
    if not isinstance(platform_identity, dict) or not all(
        isinstance(platform_identity.get(field), str) and platform_identity[field]
        for field in ("system", "release", "machine")
    ):
        errors.append("environment platform identity is invalid")

    lockfiles = manifest.get("lockfiles")
    if not isinstance(lockfiles, list):
        errors.append("environment lockfiles must be a list")
    elif not all(
        isinstance(lock, dict)
        and isinstance(lock.get("path"), str)
        and lock["path"]
        and not Path(lock["path"]).is_absolute()
        and _is_digest(lock.get("sha256"))
        for lock in lockfiles
    ):
        errors.append("environment lockfile identities are invalid")
    runtimes = manifest.get("runtimes")
    if not isinstance(runtimes, dict) or not runtimes:
        errors.append("environment runtimes must be a non-empty object")
    elif not all(
        isinstance(name, str) and name and _is_runtime_identity(identity)
        for name, identity in runtimes.items()
    ):
        errors.append("environment runtime identities are invalid")
    if manifest.get("probe_environment_policy") != PROBE_ENVIRONMENT_POLICY:
        errors.append("environment probe policy does not match the supported policy")
    expected = sha256_bytes(canonical_json_bytes(environment_comparison_material(manifest)))
    if manifest.get("comparison_fingerprint") != expected:
        errors.append("environment comparison_fingerprint does not match its material")
    return errors


def _is_digest(value: Any, *, length: int = 64) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_version_identity(value: Any) -> bool:
    return isinstance(value, dict) and all(
        isinstance(value.get(field), str) and value[field]
        for field in ("implementation", "version")
    )


def _is_runtime_identity(value: Any) -> bool:
    return (
        _is_version_identity(value)
        and isinstance(value.get("package_count"), int)
        and value["package_count"] >= 0
        and _is_digest(value.get("packages_sha256"))
    )


def verify_runtime_bindings(
    manifest: Mapping[str, Any],
    environment: Mapping[str, Any],
    variables: Mapping[str, str],
) -> None:
    """Prove command placeholders resolve to the runtimes recorded in the environment."""

    bindings = manifest.get("runtime_bindings", {})
    for placeholder, runtime_name in bindings.items():
        executable = sys.executable if placeholder == "python" else variables.get(placeholder)
        if not executable:
            raise ValueError(f"Missing runtime command variable: {placeholder}")
        expected = environment["runtimes"].get(runtime_name)
        if expected is None:
            raise ValueError(f"Environment manifest lacks runtime: {runtime_name}")
        if runtime_identity(executable) != expected:
            raise ValueError(f"Runtime identity changed after manifest capture: {runtime_name}")


def probe_environment(isolated_home: Path) -> dict[str, str]:
    """Build the allowlisted environment used for every evidence subprocess."""

    inherited = {
        key: os.environ[key]
        for key in PROBE_ENVIRONMENT_POLICY["inherited_keys"]
        if key in os.environ
    }
    home = str(isolated_home.resolve())
    inherited.update(
        {
            **PROBE_ENVIRONMENT_POLICY["forced"],
            "HOME": home,
            "XDG_CACHE_HOME": str(isolated_home / "cache"),
            "XDG_CONFIG_HOME": str(isolated_home / "config"),
            "XDG_DATA_HOME": str(isolated_home / "data"),
        }
    )
    return inherited


__all__ = [
    "PROBE_ENVIRONMENT_POLICY",
    "create_environment_manifest",
    "environment_comparison_material",
    "probe_environment",
    "runtime_identity",
    "validate_environment_manifest",
    "verify_runtime_bindings",
]
