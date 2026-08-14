"""Physical representation binding for the repaired R4-v2 publication.

This validator runs only after the local, Git-ignored repaired representation
population exists.  It binds every non-EXCLUDED ML row to graph/token/sidecar
files without recording the machine-specific root path.

A passing report is necessary for local DATA acceptance, but it is not by
itself G8/training authorization: long-contract coverage evidence and the
bounded repaired-data GPU smoke still require review.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from sentinel_data.preprocessing.r4_versions import (
    GRAPH_SCHEMA_VERSION,
    REPAIRED_DATA_PUBLICATION_ID,
    REPAIRED_REPRESENTATION_EXTRACTOR_VERSION,
    TOKEN_TENSOR_SHAPE,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_pyarrow():
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - local build dependency
        raise RuntimeError("repaired representation binding requires pyarrow") from exc
    return pq


def _require_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - local build dependency
        raise RuntimeError("repaired representation binding requires torch") from exc
    return torch


def _quantiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    ordered = sorted(float(value) for value in values)

    def at(fraction: float) -> float:
        index = min(len(ordered) - 1, round((len(ordered) - 1) * fraction))
        return ordered[index]

    return {
        "min": ordered[0],
        "p50": at(0.50),
        "p95": at(0.95),
        "p99": at(0.99),
        "max": ordered[-1],
    }


def _binding_digest(records: list[dict[str, Any]]) -> str:
    canonical = json.dumps(records, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def bind_repaired_publication(
    *,
    publication_dir: Path,
    representations_root: Path,
    report_path: Path | None = None,
) -> dict[str, Any]:
    """Validate all required repaired representation triples and write a report.

    ``representations_root`` is intentionally omitted from the persisted report;
    only logical ``source/contract_id`` identities and content hashes are bound.
    """

    pq = _require_pyarrow()
    torch = _require_torch()
    publication_dir = Path(publication_dir)
    manifest_path = publication_dir / "manifest.json"
    ml_targets_path = publication_dir / "ml_targets.parquet"
    if not manifest_path.is_file() or not ml_targets_path.is_file():
        raise FileNotFoundError(
            "repaired publication requires manifest.json and ml_targets.parquet"
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("dataset_version") != REPAIRED_DATA_PUBLICATION_ID:
        raise ValueError(
            f"unexpected repaired dataset version: {manifest.get('dataset_version')!r}"
        )
    if _sha256_file(ml_targets_path) != manifest["artifacts"]["ml_targets"]["sha256"]:
        raise ValueError("ml_targets.parquet no longer matches repaired manifest")

    rows = pq.read_table(ml_targets_path).to_pylist()
    records: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    coverage_ratios: list[float] = []
    pre_windows: list[float] = []
    coverage_over_four = 0
    role_counts: Counter[str] = Counter()

    for row in sorted(rows, key=lambda item: str(item["contract_id"])):
        role = str(row["role"])
        role_counts[role] += 1
        if role == "EXCLUDED" or not bool(row["representation_required"]):
            continue

        contract_id = str(row["contract_id"])
        source = str(row["source"])
        source_dir = Path(representations_root) / source
        graph_path = source_dir / f"{contract_id}.pt"
        token_path = source_dir / f"{contract_id}.tokens.pt"
        sidecar_path = source_dir / f"{contract_id}.rep.json"
        logical = f"{source}/{contract_id}"

        missing = [
            kind
            for kind, path in (
                ("graph", graph_path),
                ("tokens", token_path),
                ("sidecar", sidecar_path),
            )
            if not path.is_file()
        ]
        if missing:
            errors.append(
                {
                    "contract": logical,
                    "reason": "missing_files",
                    "detail": ",".join(missing),
                }
            )
            continue

        try:
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
            if sidecar.get("sha256") != contract_id:
                raise ValueError("sidecar sha256 mismatch")
            if sidecar.get("source") != source:
                raise ValueError("sidecar source mismatch")
            if sidecar.get("schema_version") != GRAPH_SCHEMA_VERSION:
                raise ValueError("graph schema version mismatch")
            if (
                sidecar.get("extractor_version")
                != REPAIRED_REPRESENTATION_EXTRACTOR_VERSION
            ):
                raise ValueError("extractor version mismatch")
            if sidecar.get("graph_target_policy") != "explicit_contract_fail_closed_v1":
                raise ValueError("graph target policy mismatch")
            if sidecar.get("requested_contract_name") != sidecar.get(
                "actual_contract_name"
            ):
                raise ValueError("requested/actual graph target mismatch")
            if sidecar.get("coverage_interpretation") != (
                "diagnostic_only_no_adequacy_threshold"
            ):
                raise ValueError("token coverage interpretation mismatch")

            token_payload = torch.load(token_path, map_location="cpu", weights_only=True)
            input_ids = token_payload.get("input_ids")
            attention_mask = token_payload.get("attention_mask")
            shape = tuple(int(v) for v in input_ids.shape)
            mask_shape = tuple(int(v) for v in attention_mask.shape)
            if shape != TOKEN_TENSOR_SHAPE or mask_shape != TOKEN_TENSOR_SHAPE:
                raise ValueError(
                    f"frozen token shape mismatch ids={shape} mask={mask_shape}"
                )
            if str(token_payload.get("sha256")) != contract_id:
                raise ValueError("token payload sha256 mismatch")
            if str(token_payload.get("source")) != source:
                raise ValueError("token payload source mismatch")

            ratio = float(sidecar["retained_token_ratio"])
            windows = int(sidecar["pre_subsampling_window_count"])
            retained = int(sidecar["retained_unique_code_tokens"])
            total = int(sidecar["pre_subsampling_code_tokens"])
            if not math.isfinite(ratio) or not 0.0 <= ratio <= 1.0:
                raise ValueError("invalid retained_token_ratio")
            if total < 0 or retained < 0 or retained > total:
                raise ValueError("invalid token coverage counts")
            coverage_ratios.append(ratio)
            pre_windows.append(float(windows))
            coverage_over_four += int(windows > TOKEN_TENSOR_SHAPE[0])

            records.append(
                {
                    "contract_id": contract_id,
                    "source": source,
                    "graph_sha256": _sha256_file(graph_path),
                    "tokens_sha256": _sha256_file(token_path),
                    "sidecar_sha256": _sha256_file(sidecar_path),
                    "schema_version": sidecar["schema_version"],
                    "extractor_version": sidecar["extractor_version"],
                    "requested_contract_name": sidecar["requested_contract_name"],
                    "actual_contract_name": sidecar["actual_contract_name"],
                    "pre_subsampling_window_count": windows,
                    "pre_subsampling_code_tokens": total,
                    "retained_unique_code_tokens": retained,
                    "retained_token_ratio": ratio,
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

    required = sum(
        str(row["role"]) != "EXCLUDED" and bool(row["representation_required"])
        for row in rows
    )
    passed = not errors and len(records) == required
    records.sort(key=lambda row: (row["source"], row["contract_id"]))
    digest = _binding_digest(records) if passed else None
    report = {
        "schema": "sentinel-r4-repaired-representation-binding-v1",
        "dataset_version": REPAIRED_DATA_PUBLICATION_ID,
        "passed": passed,
        "physical_root_recorded": False,
        "required_contracts": required,
        "checked_contracts": len(records),
        "checked_files": len(records) * 3,
        "missing_or_invalid_total": len(errors),
        "errors": errors[:200],
        "binding_digest_sha256": digest,
        "graph_schema_version": GRAPH_SCHEMA_VERSION,
        "extractor_version": REPAIRED_REPRESENTATION_EXTRACTOR_VERSION,
        "frozen_token_shape": list(TOKEN_TENSOR_SHAPE),
        "role_contract_counts": dict(sorted(role_counts.items())),
        "token_coverage": {
            "contracts_with_more_than_four_pre_subsampling_windows": coverage_over_four,
            "pre_subsampling_window_count": _quantiles(pre_windows),
            "retained_token_ratio": _quantiles(coverage_ratios),
            "adequacy_threshold": None,
            "interpretation": "diagnostic_only; long-contract strategy requires separate decision evidence",
        },
        "records": records,
    }

    if report_path is None:
        report_path = publication_dir / "representation_binding_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    # Bind the report into the local generated manifest only on a fully passing
    # physical check. A failing check must not make a candidate look accepted.
    if passed:
        logical_report = (
            report_path.relative_to(publication_dir).as_posix()
            if report_path.is_relative_to(publication_dir)
            else report_path.name
        )
        manifest["representation_binding_report"] = {
            "path": logical_report,
            "sha256": _sha256_file(report_path),
            "binding_digest_sha256": digest,
        }
        manifest["status"] = "REPAIRED_REPRESENTATION_BOUND_LOCAL_REVIEW_REQUIRED"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    return report
