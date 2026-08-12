"""Build the R4 DATA vNext v2 semantic overlay from frozen inputs."""
from __future__ import annotations

import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .policy import (
    CLASS_NAMES,
    crosswalk_action,
    role_eligibility_for_row,
    semantic_decision,
    source_claim_state,
    validate_policy_surface,
)

DATASET_VERSION = "sentinel-r4-vnext-v1"
EXPORT_SCHEMA_VERSION = "v2"
EXPECTED_LEDGER_SHA256 = "3983cc2b3317515d546c784449b583ac9a7c23ac8da267ee10f5640857cd0ac7"
EXPECTED_CONTRACTS = 22493
EXPECTED_ROWS = 224930
EXPECTED_EXCLUDED = 836


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "UNKNOWN"


def _require_pyarrow():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("DATA vNext build requires pyarrow") from exc
    return pa, pq


def _policy_evidence_basis(row: dict[str, Any], policy: dict[str, Any]) -> list[str]:
    source = str(row.get("primary_source") or "")
    source_cfg = (policy.get("sources") or {}).get(source) or {}
    return [str(x) for x in (source_cfg.get("evidence_basis") or [])]


def _evidence_ids(row: dict[str, Any], policy: dict[str, Any]) -> list[str]:
    return sorted(set(
        [*(str(x) for x in (row.get("evidence_ids") or [])), *_policy_evidence_basis(row, policy)]
    ))


def _source_claim(row: dict[str, Any], policy: dict[str, Any]) -> dict[str, Any]:
    action, mapped = crosswalk_action(row)
    limitations = sorted(set(str(x) for x in (row.get("limitations") or [])))
    return {
        "source": str(row["primary_source"]),
        "source_record_id": row.get("source_record_id"),
        "source_claim_state": source_claim_state(row),
        "source_native_label": row.get("source_native_label"),
        "crosswalk_action": action,
        "mapped_class_name": mapped,
        "evidence_ids": _evidence_ids(row, policy),
        "limitations": limitations,
    }


def _label_state_row(
    row: dict[str, Any],
    policy: dict[str, Any],
    role: str,
) -> dict[str, Any]:
    decision = semantic_decision(row, policy, role)
    evidence_ids = _evidence_ids(row, policy)
    if decision.outcome_state == "CONFIRMED_POSITIVE" and not evidence_ids:
        raise ValueError(f"confirmed positive lacks evidence_ids: {row['contract_id']} {row['class_name']}")

    limitations = sorted(set(
        [*(str(x) for x in (row.get("limitations") or [])), decision.reason_code]
    ))

    return {
        "policy_version": policy["policy_version"],
        "contract_id": str(row["contract_id"]),
        "class_index": int(row["class_index"]),
        "class_name": str(row["class_name"]),
        "historical_state": str(row["historical_state"]),
        "source_claims": [_source_claim(row, policy)],
        "outcome_state": decision.outcome_state,
        "target_value": decision.target_value,
        "training_signal": decision.training_signal,
        "training_strength": decision.training_strength,
        "loss_eligible": decision.source_policy_loss_eligible,
        "outcome_metric_eligible": decision.outcome_metric_eligible,
        "role_eligibility": role_eligibility_for_row(role, decision),
        "policy_decision_id": decision.policy_decision_id,
        "evidence_ids": evidence_ids,
        "limitations": limitations,
    }


def _build_ml_projection(
    contract_id: str,
    source: str,
    group_id: str,
    role: str,
    semantic_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    ordered = sorted(semantic_rows, key=lambda r: r["class_index"])
    if [r["class_name"] for r in ordered] != list(CLASS_NAMES):
        raise ValueError(f"class order mismatch for {contract_id}")

    out: dict[str, Any] = {
        "contract_id": contract_id,
        "source": source,
        "group_id": group_id,
        "role": role,
        "representation_required": role != "EXCLUDED",
    }
    for row in ordered:
        i = int(row["class_index"])
        strength = row["training_strength"]
        source_eligible = bool(row["loss_eligible"])
        effective = source_eligible and (
            (strength == "STRONG" and role == "TRAIN_STRONG")
            or (strength == "WEAK" and role == "TRAIN_WEAK")
        )
        out[f"target_{i}"] = row["target_value"]
        out[f"strength_{i}"] = strength
        out[f"source_loss_eligible_{i}"] = source_eligible
        out[f"effective_loss_mask_{i}"] = effective
        out[f"outcome_metric_mask_{i}"] = bool(row["outcome_metric_eligible"])
        out[f"outcome_state_{i}"] = row["outcome_state"]
        out[f"policy_decision_id_{i}"] = row["policy_decision_id"]
    return out


def build_vnext_overlay(
    *,
    ledger_path: Path,
    policy_path: Path,
    contract_roles_path: Path,
    partition_manifest_path: Path,
    unsupported_roles_path: Path,
    acceptance_manifest_path: Path,
    label_schema_path: Path,
    output_dir: Path,
    generation_commit: str | None = None,
) -> dict[str, Any]:
    """Materialize the deterministic semantic overlay and return its manifest."""
    pa, pq = _require_pyarrow()

    for p in (
        ledger_path,
        policy_path,
        contract_roles_path,
        partition_manifest_path,
        unsupported_roles_path,
        acceptance_manifest_path,
        label_schema_path,
    ):
        if not p.is_file():
            raise FileNotFoundError(p)

    if _sha256(ledger_path) != EXPECTED_LEDGER_SHA256:
        raise ValueError("Phase-3 ledger SHA-256 mismatch")

    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    validate_policy_surface(policy)
    partition_manifest = json.loads(partition_manifest_path.read_text(encoding="utf-8"))
    if partition_manifest.get("status") != "FROZEN_G6":
        raise ValueError("Phase-6 partition manifest must be FROZEN_G6")
    if partition_manifest.get("partition_version") != "r4-vnext-roles-v1":
        raise ValueError("unexpected Phase-6 partition version")
    if partition_manifest.get("ledger_sha256") != EXPECTED_LEDGER_SHA256:
        raise ValueError("Phase-6 partition does not bind expected ledger")
    if partition_manifest.get("policy_sha256") != _sha256(policy_path):
        raise ValueError("Phase-6 partition does not bind current policy")

    unsupported = json.loads(unsupported_roles_path.read_text(encoding="utf-8"))
    acceptance = json.loads(acceptance_manifest_path.read_text(encoding="utf-8"))
    if unsupported["roles"]["THRESHOLD_FIT"]["status"] != "UNSUPPORTED_EMPTY":
        raise ValueError("threshold role unexpectedly enabled")
    if unsupported["roles"]["CALIBRATION_FIT"]["status"] != "UNSUPPORTED_EMPTY":
        raise ValueError("calibration role unexpectedly enabled")
    if acceptance.get("status") != "UNSUPPORTED_EMPTY_FROZEN" or acceptance.get("contract_ids") != []:
        raise ValueError("untouched acceptance must remain frozen empty")

    role_rows = _load_jsonl(contract_roles_path)
    role_by_contract: dict[str, tuple[str, str]] = {}
    for item in role_rows:
        cid = str(item["contract_id"])
        if cid in role_by_contract:
            raise ValueError(f"duplicate contract role: {cid}")
        role_by_contract[cid] = (str(item["role"]), str(item["group_id"]))
    if len(role_by_contract) != EXPECTED_CONTRACTS:
        raise ValueError(f"contract role count {len(role_by_contract)} != {EXPECTED_CONTRACTS}")

    ledger = pq.read_table(ledger_path).to_pylist()
    if len(ledger) != EXPECTED_ROWS:
        raise ValueError(f"ledger rows {len(ledger)} != {EXPECTED_ROWS}")
    ledger.sort(key=lambda r: (str(r["contract_id"]), int(r["class_index"])))

    semantic_rows: list[dict[str, Any]] = []
    contract_semantics: dict[str, list[dict[str, Any]]] = defaultdict(list)
    source_by_contract: dict[str, str] = {}
    representation_flag_by_contract: dict[str, bool] = {}

    for row in ledger:
        cid = str(row["contract_id"])
        if cid not in role_by_contract:
            raise ValueError(f"ledger contract missing frozen role: {cid}")
        role, _ = role_by_contract[cid]
        source = str(row["primary_source"])
        previous_source = source_by_contract.setdefault(cid, source)
        if previous_source != source:
            raise ValueError(f"contract primary source changes across class rows: {cid}")
        rep = bool(row["representation_available"])
        previous_rep = representation_flag_by_contract.setdefault(cid, rep)
        if previous_rep != rep:
            raise ValueError(f"representation flag changes across class rows: {cid}")

        out = _label_state_row(row, policy, role)
        semantic_rows.append(out)
        contract_semantics[cid].append(out)

    if len(contract_semantics) != EXPECTED_CONTRACTS:
        raise ValueError("semantic contract population mismatch")

    ml_rows: list[dict[str, Any]] = []
    for cid in sorted(contract_semantics):
        role, gid = role_by_contract[cid]
        if role != "EXCLUDED" and not representation_flag_by_contract[cid]:
            raise ValueError(f"non-excluded contract lacks representation flag: {cid}")
        ml_rows.append(_build_ml_projection(
            cid, source_by_contract[cid], gid, role, contract_semantics[cid]
        ))

    output_dir.mkdir(parents=True, exist_ok=True)
    label_states_path = output_dir / "label_states.parquet"
    ml_targets_path = output_dir / "ml_targets.parquet"
    source_registry_path = output_dir / "source_registry.json"
    crosswalk_registry_path = output_dir / "crosswalk_registry.json"
    evidence_snapshot_path = output_dir / "evidence_snapshot.json"
    representation_requirements_path = output_dir / "representation_requirements.json"
    manifest_path = output_dir / "manifest.json"

    pq.write_table(pa.Table.from_pylist(semantic_rows), label_states_path, compression="zstd")
    pq.write_table(pa.Table.from_pylist(ml_rows), ml_targets_path, compression="zstd")

    source_registry = {
        "schema": "sentinel-data-vnext-source-registry-v1",
        "policy_version": policy["policy_version"],
        "sources": policy["sources"],
    }
    _write_json(source_registry_path, source_registry)

    crosswalk_registry = {
        "schema": "sentinel-data-vnext-crosswalk-registry-v1",
        "policy_version": policy["policy_version"],
        "smartbugs_curated": {
            "approved_mappings": policy["sources"]["smartbugs_curated"]["approved_mappings"],
            "no_target_categories": policy["sources"]["smartbugs_curated"]["no_target_categories"],
        },
        "dive": {
            "mapped_category_policy": policy["sources"]["dive"]["mapped_category_policy"],
            "no_target_categories": policy["sources"]["dive"]["no_target_categories"],
            "unsupported_canonical_classes": policy["sources"]["dive"]["unsupported_canonical_classes"],
        },
        "solidifi": policy["sources"]["solidifi"]["direct_or_approved_mappings"],
        "aggregation": policy["aggregation"],
    }
    _write_json(crosswalk_registry_path, crosswalk_registry)

    evidence_snapshot = {
        "schema": "sentinel-data-vnext-evidence-snapshot-v1",
        "ledger": {"path": str(ledger_path), "sha256": _sha256(ledger_path)},
        "policy": {"path": str(policy_path), "sha256": _sha256(policy_path)},
        "label_schema": {"path": str(label_schema_path), "sha256": _sha256(label_schema_path)},
        "partition_manifest": {"path": str(partition_manifest_path), "sha256": _sha256(partition_manifest_path)},
        "contract_roles": {"path": str(contract_roles_path), "sha256": _sha256(contract_roles_path)},
        "unsupported_roles": {"path": str(unsupported_roles_path), "sha256": _sha256(unsupported_roles_path)},
        "untouched_acceptance": {"path": str(acceptance_manifest_path), "sha256": _sha256(acceptance_manifest_path)},
    }
    _write_json(evidence_snapshot_path, evidence_snapshot)

    role_counts = Counter(row["role"] for row in ml_rows)
    required = [r for r in ml_rows if r["representation_required"]]
    representation_requirements = {
        "schema": "sentinel-data-vnext-representation-requirements-v1",
        "graph_schema_version": "v9",
        "root": "data_module/data/representations",
        "path_patterns": {
            "graph": "<root>/<source>/<contract_id>.pt",
            "tokens": "<root>/<source>/<contract_id>.tokens.pt",
            "sidecar": "<root>/<source>/<contract_id>.rep.json",
        },
        "authoritative_contract_list": "ml_targets.parquet",
        "required_contracts": len(required),
        "excluded_contracts": role_counts["EXCLUDED"],
        "physical_binding_status": "PENDING_LOCAL_G7_GATE",
    }
    _write_json(representation_requirements_path, representation_requirements)

    artifacts = {}
    for name, path in (
        ("label_states", label_states_path),
        ("ml_targets", ml_targets_path),
        ("source_registry", source_registry_path),
        ("crosswalk_registry", crosswalk_registry_path),
        ("evidence_snapshot", evidence_snapshot_path),
        ("representation_requirements", representation_requirements_path),
    ):
        artifacts[name] = {
            "path": path.name,
            "sha256": _sha256(path),
            "bytes": path.stat().st_size,
        }

    target_counts = Counter()
    strength_counts = Counter()
    effective_loss_cells = 0
    outcome_metric_cells = 0
    for row in semantic_rows:
        target_counts[str(row["target_value"])] += 1
        strength_counts[row["training_strength"]] += 1
    for row in ml_rows:
        effective_loss_cells += sum(bool(row[f"effective_loss_mask_{i}"]) for i in range(10))
        outcome_metric_cells += sum(bool(row[f"outcome_metric_mask_{i}"]) for i in range(10))

    manifest = {
        "schema": "sentinel-data-vnext-overlay-manifest-v1",
        "dataset_version": DATASET_VERSION,
        "export_schema_version": EXPORT_SCHEMA_VERSION,
        "graph_schema_version": "v9",
        "status": "SEMANTIC_VALIDATED_REPRESENTATIONS_PENDING",
        "generation_commit": generation_commit or _git_head(),
        "class_order": list(CLASS_NAMES),
        "population": {
            "contracts": len(ml_rows),
            "contract_class_rows": len(semantic_rows),
            "excluded_contracts": role_counts["EXCLUDED"],
            "representation_required_contracts": len(required),
        },
        "role_contract_counts": dict(sorted(role_counts.items())),
        "semantic_counts": {
            "target_value": dict(sorted(target_counts.items())),
            "training_strength": dict(sorted(strength_counts.items())),
            "effective_loss_cells": effective_loss_cells,
            "outcome_metric_cells": outcome_metric_cells,
        },
        "unsupported_roles": {
            "THRESHOLD_FIT": "UNSUPPORTED_EMPTY",
            "CALIBRATION_FIT": "UNSUPPORTED_EMPTY",
            "UNTOUCHED_ACCEPTANCE": "UNSUPPORTED_EMPTY_FROZEN",
        },
        "inputs": evidence_snapshot,
        "artifacts": artifacts,
        "semantic_validation_report": None,
        "representation_binding_report": None,
        "historical_artifacts_mutated": False,
    }
    if manifest["population"]["contracts"] != EXPECTED_CONTRACTS:
        raise ValueError("manifest contract population mismatch")
    if manifest["population"]["contract_class_rows"] != EXPECTED_ROWS:
        raise ValueError("manifest row population mismatch")
    if manifest["population"]["excluded_contracts"] != EXPECTED_EXCLUDED:
        raise ValueError("manifest excluded population mismatch")
    if target_counts.get("0", 0):
        raise ValueError("policy v1 must not produce target=0")

    _write_json(manifest_path, manifest)
    return manifest


__all__ = [
    "DATASET_VERSION",
    "EXPORT_SCHEMA_VERSION",
    "build_vnext_overlay",
]
