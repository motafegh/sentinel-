"""Fresh physical candidate assembly for the R4-D-012 guarded token lineage.

This module treats the accepted R4-D-011 V10 representation as an immutable
physical parent. It never regenerates graphs and never mutates parent files.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from sentinel_data.preprocessing.r4_versions import (
    PREPROCESSING_ARTIFACT_VERSION,
    V10_PRIMARY_SLITHER_VERSION,
    V10_REPRESENTATION_ROOT_NAME,
    V10_SLITHER_RUNTIME_EXCEPTIONS,
)
from sentinel_data.representation.r4_guarded_lineage import (
    CANDIDATE_ROOT_NAME,
    PARENT_ACCEPTANCE_SCHEMA,
    PARENT_BINDING_DIGEST_SHA256,
    PARENT_DECISION_ID,
    PARENT_EXTRACTOR_VERSION,
    PARENT_GRAPH_SCHEMA_VERSION,
    SELECTOR_CONFIG_SHA256,
    TOKEN_LINEAGE_ID,
    graph_parent_authority,
)
from sentinel_data.representation.r4_guarded_tokenizer import (
    tokenize_repaired_source_guarded,
)

CANDIDATE_BUILD_SCHEMA = "sentinel-r4-v10-v26-guarded-candidate-build-v1"
FAILURE_LEDGER_NAME = "guarded_candidate_failures.jsonl"
MANIFEST_NAME = "guarded_candidate_manifest.json"

_COVERAGE_FIELDS = (
    "coverage_schema_version",
    "pre_subsampling_window_count",
    "pre_subsampling_code_tokens",
    "selected_window_indices",
    "selected_code_token_ranges",
    "retained_unique_code_tokens",
    "retained_token_ratio",
    "content_tokens_per_window",
    "coverage_interpretation",
)


class GuardedCandidateBuildError(RuntimeError):
    """Raised when candidate-wide parent or output invariants are invalid."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GuardedCandidateBuildError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise GuardedCandidateBuildError(f"{label} must contain a JSON object")
    return value


def _load_acceptance(path: Path) -> dict[str, Any]:
    acceptance = _load_json(path, label="R4-D-011 acceptance record")
    if acceptance.get("schema") != PARENT_ACCEPTANCE_SCHEMA:
        raise GuardedCandidateBuildError("R4-D-011 acceptance schema mismatch")
    if acceptance.get("decision_id") != PARENT_DECISION_ID:
        raise GuardedCandidateBuildError("R4-D-011 decision identity mismatch")
    if acceptance.get("decision") != "ACCEPTED_IMMUTABLE_LOCAL_PHYSICAL_REPRESENTATION":
        raise GuardedCandidateBuildError("R4-D-011 is not recorded as immutable accepted")
    if acceptance.get("physical_acceptance") is not True:
        raise GuardedCandidateBuildError("R4-D-011 physical acceptance is not true")
    if acceptance.get("training_authorized") is not False:
        raise GuardedCandidateBuildError("R4-D-011 training authority unexpectedly changed")
    if acceptance.get("selector_promoted") is not False:
        raise GuardedCandidateBuildError("R4-D-011 selector promotion flag unexpectedly changed")

    lineage = acceptance.get("accepted_lineage")
    if not isinstance(lineage, dict):
        raise GuardedCandidateBuildError("R4-D-011 accepted_lineage is missing")
    expected = {
        "binding_digest_sha256": PARENT_BINDING_DIGEST_SHA256,
        "graph_schema_version": PARENT_GRAPH_SCHEMA_VERSION,
        "extractor_version": PARENT_EXTRACTOR_VERSION,
        "protected_local": True,
    }
    for field, value in expected.items():
        if lineage.get(field) != value:
            raise GuardedCandidateBuildError(
                f"R4-D-011 accepted_lineage {field} mismatch"
            )
    contracts = lineage.get("contracts")
    files = lineage.get("files")
    if isinstance(contracts, bool) or not isinstance(contracts, int) or contracts < 1:
        raise GuardedCandidateBuildError("R4-D-011 contract count is invalid")
    if files != contracts * 3:
        raise GuardedCandidateBuildError("R4-D-011 file count is not exactly three per identity")
    return acceptance


def _runtime_contract(
    acceptance: dict[str, Any],
) -> dict[tuple[str, str], str]:
    runtime_map: dict[tuple[str, str], str] = {}
    runtime_contracts = 0
    rows = acceptance.get("runtime_distribution")
    if not isinstance(rows, list) or not rows:
        raise GuardedCandidateBuildError("R4-D-011 runtime distribution is missing")
    for row in rows:
        if not isinstance(row, dict):
            raise GuardedCandidateBuildError("invalid R4-D-011 runtime distribution row")
        slither = row.get("slither_analyzer")
        crytic = row.get("crytic_compile")
        role = row.get("runtime_role")
        count = row.get("contracts")
        if (
            not isinstance(slither, str)
            or not slither
            or not isinstance(crytic, str)
            or not crytic
            or role not in {"primary", "identity_bound_exception"}
            or isinstance(count, bool)
            or not isinstance(count, int)
            or count < 1
        ):
            raise GuardedCandidateBuildError("invalid R4-D-011 runtime distribution row")
        key = (slither, role)
        if key in runtime_map and runtime_map[key] != crytic:
            raise GuardedCandidateBuildError("conflicting R4-D-011 crytic runtime identity")
        runtime_map[key] = crytic
        runtime_contracts += count
    if runtime_contracts != int(acceptance["accepted_lineage"]["contracts"]):
        raise GuardedCandidateBuildError(
            "R4-D-011 runtime distribution does not cover the accepted population"
        )
    return runtime_map


def _inventory(parent_root: Path) -> dict[tuple[str, str], Path]:
    inventory: dict[tuple[str, str], Path] = {}
    for sidecar in sorted(parent_root.glob("*/*.rep.json")):
        source = sidecar.parent.name
        contract_id = sidecar.name.removesuffix(".rep.json")
        if len(contract_id) != 64 or any(
            char not in "0123456789abcdef" for char in contract_id
        ):
            raise GuardedCandidateBuildError(
                f"invalid R4-D-011 representation identity {source}/{contract_id}"
            )
        key = (source, contract_id)
        if key in inventory:
            raise GuardedCandidateBuildError(
                f"duplicate R4-D-011 representation identity {source}/{contract_id}"
            )
        inventory[key] = sidecar
    if not inventory:
        raise GuardedCandidateBuildError("R4-D-011 parent root contains no representations")
    return inventory


def _validate_root_identity(
    *,
    acceptance: dict[str, Any],
    parent_root: Path,
    preprocessed_root: Path,
    candidate_root: Path,
) -> None:
    lineage = acceptance["accepted_lineage"]
    accepted_parent_name = Path(str(lineage.get("physical_root") or "")).name
    if accepted_parent_name != V10_REPRESENTATION_ROOT_NAME:
        raise GuardedCandidateBuildError(
            "R4-D-011 acceptance does not name the frozen V10 representation root"
        )
    if parent_root.name != accepted_parent_name:
        raise GuardedCandidateBuildError(
            f"parent root must be named {accepted_parent_name!r}"
        )

    accepted_preprocessed_name = Path(
        str(lineage.get("preprocessed_parent") or "")
    ).name
    if accepted_preprocessed_name != PREPROCESSING_ARTIFACT_VERSION:
        raise GuardedCandidateBuildError(
            "R4-D-011 acceptance does not name the frozen preprocessing parent"
        )
    if preprocessed_root.name != accepted_preprocessed_name:
        raise GuardedCandidateBuildError(
            f"preprocessed root must be named {accepted_preprocessed_name!r}"
        )

    if candidate_root.name != CANDIDATE_ROOT_NAME:
        raise GuardedCandidateBuildError(
            f"candidate root must be named {CANDIDATE_ROOT_NAME!r}"
        )
    parent_resolved = parent_root.resolve()
    candidate_resolved = candidate_root.resolve()
    if parent_resolved == candidate_resolved:
        raise GuardedCandidateBuildError("candidate and R4-D-011 parent roots must differ")
    if candidate_resolved.is_relative_to(parent_resolved):
        raise GuardedCandidateBuildError("candidate root must not be inside R4-D-011 parent")
    if parent_resolved.is_relative_to(candidate_resolved):
        raise GuardedCandidateBuildError("R4-D-011 parent must not be inside candidate root")
    if candidate_root.exists() and any(candidate_root.iterdir()):
        raise GuardedCandidateBuildError("candidate root must be fresh and empty")


def _required_runtime(
    contract_id: str,
    runtime_map: dict[tuple[str, str], str],
) -> tuple[str, str, str]:
    slither = V10_SLITHER_RUNTIME_EXCEPTIONS.get(
        contract_id,
        V10_PRIMARY_SLITHER_VERSION,
    )
    role = (
        "identity_bound_exception"
        if contract_id in V10_SLITHER_RUNTIME_EXCEPTIONS
        else "primary"
    )
    crytic = runtime_map.get((slither, role))
    if crytic is None:
        raise GuardedCandidateBuildError(
            f"R4-D-011 acceptance runtime distribution lacks {slither}/{role}"
        )
    return slither, crytic, role


def _validate_parent_sidecar(
    sidecar: dict[str, Any],
    *,
    source: str,
    contract_id: str,
    runtime_map: dict[tuple[str, str], str],
) -> tuple[str, ...]:
    if sidecar.get("sha256") != contract_id or sidecar.get("source") != source:
        raise ValueError("R4-D-011 sidecar identity mismatch")
    if sidecar.get("schema_version") != PARENT_GRAPH_SCHEMA_VERSION:
        raise ValueError("R4-D-011 graph schema mismatch")
    if sidecar.get("extractor_version") != PARENT_EXTRACTOR_VERSION:
        raise ValueError("R4-D-011 extractor version mismatch")
    if sidecar.get("token_lineage") != "accepted_v9_byte_copy":
        raise ValueError("R4-D-011 historical token lineage mismatch")

    requested = sidecar.get("requested_contract_names")
    actual = sidecar.get("actual_contract_names")
    if (
        not isinstance(requested, list)
        or not requested
        or any(not isinstance(name, str) or not name for name in requested)
        or len(set(requested)) != len(requested)
    ):
        raise ValueError("R4-D-011 requested target identity is invalid")
    if actual != requested:
        raise ValueError("R4-D-011 requested/actual target identity differs")

    runtime = sidecar.get("slither_runtime")
    if not isinstance(runtime, dict):
        raise ValueError("R4-D-011 Slither runtime binding is missing")
    required_slither, required_crytic, required_role = _required_runtime(
        contract_id,
        runtime_map,
    )
    if runtime.get("slither_analyzer") != required_slither:
        raise ValueError("R4-D-011 Slither version mismatch")
    if runtime.get("crytic_compile") != required_crytic:
        raise ValueError("R4-D-011 crytic-compile version mismatch")
    if runtime.get("runtime_role") != required_role:
        raise ValueError("R4-D-011 runtime role mismatch")
    if runtime.get("required_for_physical_acceptance") != required_slither:
        raise ValueError("R4-D-011 required Slither binding mismatch")

    mode = str(sidecar.get("graph_extraction_mode") or "")
    if not mode or mode.startswith("slither_parse_only"):
        raise ValueError("R4-D-011 graph extraction mode is invalid/degraded")
    if bool(sidecar.get("graph_analysis_degraded")):
        raise ValueError("R4-D-011 graph analysis is degraded")
    if list(sidecar.get("unclassified_call_ir") or []):
        raise ValueError("R4-D-011 contains unclassified call IR")
    if int(sidecar.get("unclassified_call_ir_count", 0)) != 0:
        raise ValueError("R4-D-011 reports unclassified call IR")
    if list(sidecar.get("call_mapping_errors") or []):
        raise ValueError("R4-D-011 contains call mapping errors")
    if sidecar.get("classified_call_ir_counts") != sidecar.get(
        "emitted_call_edge_counts"
    ):
        raise ValueError("R4-D-011 classified/emitted call counts differ")
    return tuple(requested)


def _load_parent_sidecar(path: Path, *, logical: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read R4-D-011 sidecar {logical}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"R4-D-011 sidecar {logical} must contain a JSON object")
    return value


def _parent_paths(
    sidecar_path: Path,
    contract_id: str,
) -> tuple[Path, Path, Path]:
    directory = sidecar_path.parent
    graph = directory / f"{contract_id}.pt"
    tokens = directory / f"{contract_id}.tokens.pt"
    if not graph.is_file() or not tokens.is_file():
        raise ValueError("R4-D-011 representation triple is incomplete")
    return graph, tokens, sidecar_path


def _triple_hashes(paths: tuple[Path, Path, Path]) -> tuple[str, str, str]:
    return tuple(_sha256_file(path) for path in paths)


def _coverage_mapping(token_data: dict[str, Any]) -> dict[str, Any]:
    return {field: token_data[field] for field in _COVERAGE_FIELDS}


def _write_failure_ledger(root: Path, failures: list[dict[str, str]]) -> None:
    path = root / FAILURE_LEDGER_NAME
    if not failures:
        if path.exists():
            path.unlink()
        return
    with path.open("w", encoding="utf-8") as handle:
        for row in failures:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _cleanup(paths: tuple[Path, Path, Path]) -> None:
    for path in paths:
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def assemble_guarded_candidate(
    *,
    parent_root: Path,
    preprocessed_root: Path,
    candidate_root: Path,
    acceptance_path: Path,
    tokenizer: Any | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    """Build a fresh guarded-token candidate from immutable R4-D-011 triples."""

    if limit is not None and (
        isinstance(limit, bool) or not isinstance(limit, int) or limit < 1
    ):
        raise GuardedCandidateBuildError("limit must be a positive integer")

    parent_root = Path(parent_root)
    preprocessed_root = Path(preprocessed_root)
    candidate_root = Path(candidate_root)
    acceptance_path = Path(acceptance_path)

    if not parent_root.is_dir():
        raise GuardedCandidateBuildError("R4-D-011 parent root does not exist")
    if not preprocessed_root.is_dir():
        raise GuardedCandidateBuildError("repaired preprocessing root does not exist")

    acceptance = _load_acceptance(acceptance_path)
    _validate_root_identity(
        acceptance=acceptance,
        parent_root=parent_root,
        preprocessed_root=preprocessed_root,
        candidate_root=candidate_root,
    )
    runtime_map = _runtime_contract(acceptance)
    inventory = _inventory(parent_root)
    expected_contracts = int(acceptance["accepted_lineage"]["contracts"])
    if len(inventory) != expected_contracts:
        raise GuardedCandidateBuildError(
            "R4-D-011 parent population does not match accepted contract count"
        )

    identities = sorted(inventory)
    if limit is not None:
        identities = identities[:limit]

    candidate_root.mkdir(parents=True, exist_ok=True)
    failures: list[dict[str, str]] = []
    records: list[dict[str, Any]] = []

    for source, contract_id in identities:
        logical = f"{source}/{contract_id}"
        parent_sidecar_path = inventory[(source, contract_id)]
        candidate_dir = candidate_root / source
        candidate_dir.mkdir(parents=True, exist_ok=True)
        candidate_paths = (
            candidate_dir / f"{contract_id}.pt",
            candidate_dir / f"{contract_id}.tokens.pt",
            candidate_dir / f"{contract_id}.rep.json",
        )
        try:
            parent_paths = _parent_paths(parent_sidecar_path, contract_id)
            parent_hashes_before = _triple_hashes(parent_paths)
            parent_sidecar = _load_parent_sidecar(
                parent_sidecar_path,
                logical=logical,
            )
            requested_names = _validate_parent_sidecar(
                parent_sidecar,
                source=source,
                contract_id=contract_id,
                runtime_map=runtime_map,
            )

            source_path = preprocessed_root / source / f"{contract_id}.sol"
            source_bytes = source_path.read_bytes()
            if hashlib.sha256(source_bytes).hexdigest() != contract_id:
                raise ValueError("repaired source bytes do not match representation identity")
            source_text = source_bytes.decode("utf-8")

            token_data = tokenize_repaired_source_guarded(
                source_text,
                contract_id=contract_id,
                requested_contract_names=requested_names,
                tokenizer=tokenizer,
            )

            shutil.copyfile(parent_paths[0], candidate_paths[0])
            if _sha256_file(candidate_paths[0]) != parent_hashes_before[0]:
                raise ValueError("candidate graph bytes differ from R4-D-011 parent")

            try:
                import torch
            except ImportError as exc:
                raise RuntimeError("guarded candidate assembly requires torch") from exc

            token_payload = {
                **token_data,
                "sha256": contract_id,
                "source": source,
            }
            torch.save(token_payload, candidate_paths[1])

            sidecar = dict(parent_sidecar)
            sidecar.update(
                {
                    "window_count": int(token_data["num_windows"]),
                    **_coverage_mapping(token_data),
                    "token_lineage": TOKEN_LINEAGE_ID,
                    "token_lineage_parent_decision": PARENT_DECISION_ID,
                    "token_lineage_parent_binding_digest_sha256": (
                        PARENT_BINDING_DIGEST_SHA256
                    ),
                    "selector_config_sha256": SELECTOR_CONFIG_SHA256,
                    "target_evidence": token_data["target_evidence"],
                    "token_selector": token_data["token_selector"],
                    "graph_parent": graph_parent_authority(),
                }
            )
            candidate_paths[2].write_text(
                json.dumps(sidecar, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

            parent_hashes_after = _triple_hashes(parent_paths)
            if parent_hashes_after != parent_hashes_before:
                raise GuardedCandidateBuildError(
                    f"R4-D-011 parent mutation detected for {logical}"
                )

            records.append(
                {
                    "source": source,
                    "contract_id": contract_id,
                    "parent_graph_sha256": parent_hashes_before[0],
                    "candidate_graph_sha256": _sha256_file(candidate_paths[0]),
                    "tokens_sha256": _sha256_file(candidate_paths[1]),
                    "sidecar_sha256": _sha256_file(candidate_paths[2]),
                    "effective_strategy": token_data["token_selector"][
                        "effective_strategy"
                    ],
                    "used_control_fallback": token_data["token_selector"][
                        "used_control_fallback"
                    ],
                }
            )
        except GuardedCandidateBuildError:
            _cleanup(candidate_paths)
            raise
        except Exception as exc:
            _cleanup(candidate_paths)
            failures.append(
                {
                    "source": source,
                    "contract_id": contract_id,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )

    records.sort(key=lambda row: (row["source"], row["contract_id"]))
    failures.sort(key=lambda row: (row["source"], row["contract_id"]))
    _write_failure_ledger(candidate_root, failures)

    complete = limit is None
    passed = not failures and len(records) == len(identities)
    manifest = {
        "schema": CANDIDATE_BUILD_SCHEMA,
        "status": (
            "CANDIDATE_BUILD_PASS"
            if passed and complete
            else "BOUNDED_BUILD_PASS"
            if passed
            else "CANDIDATE_BUILD_FAIL"
        ),
        "passed": passed,
        "physical_acceptance": False,
        "training_authorized": False,
        "candidate_root_name": CANDIDATE_ROOT_NAME,
        "token_lineage": TOKEN_LINEAGE_ID,
        "selector_config_sha256": SELECTOR_CONFIG_SHA256,
        "graph_parent": graph_parent_authority(),
        "accepted_parent_contracts": expected_contracts,
        "contracts_requested": len(identities),
        "representations_written": len(records),
        "representations_failed": len(failures),
        "complete_candidate_build": complete,
        "requested_limit": limit,
        "records": records,
        "failures": failures,
    }
    (candidate_root / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


__all__ = [
    "CANDIDATE_BUILD_SCHEMA",
    "FAILURE_LEDGER_NAME",
    "GuardedCandidateBuildError",
    "MANIFEST_NAME",
    "assemble_guarded_candidate",
]
