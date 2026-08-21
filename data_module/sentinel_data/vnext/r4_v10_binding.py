"""Diagnostic binding for the R4-D-010 v10 representation candidate.

This module deliberately does not publish, accept, or authorize training from a
candidate population.  It proves three narrower mechanical properties:

* the candidate has exactly the same logical population as accepted v9;
* every candidate graph/sidecar conforms to the v10 interface; and
* every candidate token artifact is byte-identical to its accepted v9 input.

Source-backed call reconciliation and independent review remain separate stop
lines before physical acceptance can be considered.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from sentinel_data.preprocessing.r4_versions import (
    V10_GRAPH_SCHEMA_VERSION,
    V10_REPRESENTATION_EXTRACTOR_VERSION,
    V10_REPRESENTATION_ROOT_NAME,
)
from sentinel_data.representation.graph_schema_versions import get_graph_schema
from sentinel_data.vnext.r4_binding import _validate_graph, _validate_tokens


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _binding_digest(records: list[dict[str, Any]]) -> str:
    canonical = json.dumps(records, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _inventory(root: Path) -> dict[tuple[str, str], Path]:
    inventory: dict[tuple[str, str], Path] = {}
    for sidecar in sorted(root.glob("*/*.rep.json")):
        source = sidecar.parent.name
        contract_id = sidecar.name.removesuffix(".rep.json")
        if len(contract_id) != 64 or any(ch not in "0123456789abcdef" for ch in contract_id):
            raise ValueError(f"invalid representation identity: {source}/{contract_id}")
        key = (source, contract_id)
        if key in inventory:
            raise ValueError(f"duplicate representation identity: {source}/{contract_id}")
        inventory[key] = sidecar
    return inventory


def bind_v10_candidate(
    *,
    candidate_root: Path,
    accepted_v9_root: Path,
    report_path: Path | None = None,
    max_recorded_errors: int = 200,
) -> dict[str, Any]:
    """Validate a complete v10 candidate without changing either population."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - local build dependency
        raise RuntimeError("v10 candidate binding requires torch") from exc

    candidate_root = Path(candidate_root)
    accepted_v9_root = Path(accepted_v9_root)
    if candidate_root.name != V10_REPRESENTATION_ROOT_NAME:
        raise ValueError(
            f"candidate root must be named {V10_REPRESENTATION_ROOT_NAME!r}"
        )
    if not candidate_root.is_dir() or not accepted_v9_root.is_dir():
        raise FileNotFoundError("candidate and accepted-v9 representation roots must exist")
    if candidate_root.resolve() == accepted_v9_root.resolve():
        raise ValueError("candidate and accepted-v9 roots must be distinct")

    candidate = _inventory(candidate_root)
    accepted = _inventory(accepted_v9_root)
    candidate_keys = set(candidate)
    accepted_keys = set(accepted)
    missing = sorted(accepted_keys - candidate_keys)
    extra = sorted(candidate_keys - accepted_keys)
    schema = get_graph_schema(V10_GRAPH_SCHEMA_VERSION)

    errors: list[dict[str, str]] = []
    for source, contract_id in missing:
        errors.append(
            {
                "contract": f"{source}/{contract_id}",
                "reason": "missing_candidate_identity",
                "detail": "accepted v9 identity is absent from candidate",
            }
        )
    for source, contract_id in extra:
        errors.append(
            {
                "contract": f"{source}/{contract_id}",
                "reason": "extra_candidate_identity",
                "detail": "candidate identity is absent from accepted v9",
            }
        )

    records: list[dict[str, Any]] = []
    token_byte_matches = 0
    for source, contract_id in sorted(candidate_keys & accepted_keys):
        logical = f"{source}/{contract_id}"
        candidate_dir = candidate[(source, contract_id)].parent
        accepted_dir = accepted[(source, contract_id)].parent
        graph_path = candidate_dir / f"{contract_id}.pt"
        token_path = candidate_dir / f"{contract_id}.tokens.pt"
        sidecar_path = candidate_dir / f"{contract_id}.rep.json"
        accepted_token_path = accepted_dir / f"{contract_id}.tokens.pt"
        missing_files = [
            name
            for name, path in (
                ("candidate_graph", graph_path),
                ("candidate_tokens", token_path),
                ("accepted_v9_tokens", accepted_token_path),
            )
            if not path.is_file()
        ]
        if missing_files:
            errors.append(
                {
                    "contract": logical,
                    "reason": "missing_files",
                    "detail": ",".join(missing_files),
                }
            )
            continue

        try:
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
            if sidecar.get("sha256") != contract_id or sidecar.get("source") != source:
                raise ValueError("candidate sidecar identity mismatch")
            if sidecar.get("schema_version") != V10_GRAPH_SCHEMA_VERSION:
                raise ValueError("candidate graph schema version mismatch")
            if sidecar.get("extractor_version") != V10_REPRESENTATION_EXTRACTOR_VERSION:
                raise ValueError("candidate extractor version mismatch")
            if sidecar.get("token_lineage") != "accepted_v9_byte_copy":
                raise ValueError("candidate token lineage is not accepted-v9 byte copy")
            if sidecar.get("unclassified_call_ir") not in (None, []):
                raise ValueError("candidate contains unclassified call IR")
            if int(sidecar.get("unclassified_call_ir_count", 0)) != 0:
                raise ValueError("candidate reports unclassified call IR")
            if list(sidecar.get("call_mapping_errors") or []):
                raise ValueError("candidate reports call-to-graph mapping errors")
            if sidecar.get("classified_call_ir_counts") != sidecar.get(
                "emitted_call_edge_counts"
            ):
                raise ValueError("candidate classified/emitted call counts differ")

            graph = torch.load(graph_path, map_location="cpu", weights_only=False)
            if getattr(graph, "graph_schema_version", None) != V10_GRAPH_SCHEMA_VERSION:
                raise ValueError("graph payload schema version mismatch")
            if (
                getattr(graph, "representation_extractor_version", None)
                != V10_REPRESENTATION_EXTRACTOR_VERSION
            ):
                raise ValueError("graph payload extractor version mismatch")
            if list(getattr(graph, "unclassified_call_ir", []) or []):
                raise ValueError("graph payload contains unclassified call IR")
            if list(getattr(graph, "call_mapping_errors", []) or []):
                raise ValueError("graph payload contains call-to-graph mapping errors")
            if getattr(graph, "classified_call_ir_counts", None) != getattr(
                graph, "emitted_call_edge_counts", None
            ):
                raise ValueError("graph payload classified/emitted call counts differ")
            _validate_graph(torch, graph, sidecar, num_edge_types=schema.num_edge_types)

            token_payload = torch.load(token_path, map_location="cpu", weights_only=True)
            _validate_tokens(torch, token_payload, sidecar)
            if token_payload.get("sha256") != contract_id:
                raise ValueError("candidate token contract identity mismatch")
            if token_payload.get("source") != source:
                raise ValueError("candidate token source identity mismatch")

            candidate_token_hash = _sha256_file(token_path)
            accepted_token_hash = _sha256_file(accepted_token_path)
            if candidate_token_hash != accepted_token_hash:
                raise ValueError("candidate token bytes differ from accepted v9")
            token_byte_matches += 1
            records.append(
                {
                    "contract_id": contract_id,
                    "source": source,
                    "graph_sha256": _sha256_file(graph_path),
                    "tokens_sha256": candidate_token_hash,
                    "sidecar_sha256": _sha256_file(sidecar_path),
                    "schema_version": V10_GRAPH_SCHEMA_VERSION,
                    "extractor_version": V10_REPRESENTATION_EXTRACTOR_VERSION,
                }
            )
        except Exception as exc:
            errors.append(
                {
                    "contract": logical,
                    "reason": type(exc).__name__,
                    "detail": str(exc),
                }
            )

    records.sort(key=lambda row: (row["source"], row["contract_id"]))
    passed = not errors and len(records) == len(accepted)
    report = {
        "schema": "sentinel-r4-v10-candidate-binding-v1",
        "passed": passed,
        "status": "V10_CANDIDATE_DIAGNOSTIC_PASS" if passed else "V10_CANDIDATE_DIAGNOSTIC_FAIL",
        "physical_root_recorded": False,
        "candidate_root_name": V10_REPRESENTATION_ROOT_NAME,
        "accepted_v9_contracts": len(accepted),
        "candidate_contracts": len(candidate),
        "checked_contracts": len(records),
        "missing_candidate_contracts": len(missing),
        "extra_candidate_contracts": len(extra),
        "missing_or_invalid_total": len(errors),
        "errors": errors[:max_recorded_errors],
        "errors_truncated": max(0, len(errors) - max_recorded_errors),
        "token_byte_identical_contracts": token_byte_matches,
        "binding_digest_sha256": _binding_digest(records) if passed else None,
        "graph_schema_version": V10_GRAPH_SCHEMA_VERSION,
        "extractor_version": V10_REPRESENTATION_EXTRACTOR_VERSION,
        "physical_acceptance": False,
        "training_authorized": False,
        "remaining_stop_lines": [
            "full_source_call_reconciliation",
            "semantic_regression_review",
            "independent_review",
            "explicit_physical_acceptance_decision",
        ],
        "records": records,
    }

    if report_path is not None:
        report_path = Path(report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return report


__all__ = ["bind_v10_candidate"]
