"""Local physical representation binding for DATA vNext G7.

The semantic overlay is committed without copying graph/token tensors. This
module verifies the protected/local representation tree against the exact
non-excluded vNext population and produces a compact hash-bound report.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

EXPECTED_GRAPH_SCHEMA = "v9"
LOGICAL_REPRESENTATION_ROOT = "data_module/data/representations"
MAX_REPORTED_FAILURES = 100


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _require_pyarrow():
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("representation binding requires pyarrow") from exc
    return pq


def verify_local_representations(
    export_dir: Path,
    representations_root: Path,
    *,
    report_path: Path | None = None,
) -> dict[str, Any]:
    """Verify every non-excluded vNext contract has valid local representation files."""
    pq = _require_pyarrow()
    export_dir = Path(export_dir)
    representations_root = Path(representations_root)
    ml_targets_path = export_dir / "ml_targets.parquet"
    requirements_path = export_dir / "representation_requirements.json"

    if not ml_targets_path.is_file():
        raise FileNotFoundError(ml_targets_path)
    if not requirements_path.is_file():
        raise FileNotFoundError(requirements_path)
    if not representations_root.is_dir():
        raise FileNotFoundError(representations_root)

    requirements = json.loads(requirements_path.read_text(encoding="utf-8"))
    if requirements.get("graph_schema_version") != EXPECTED_GRAPH_SCHEMA:
        raise ValueError("representation requirements do not bind graph schema v9")

    rows = pq.read_table(
        ml_targets_path,
        columns=["contract_id", "source", "role", "representation_required"],
    ).to_pylist()
    required_rows = sorted(
        (r for r in rows if bool(r["representation_required"])),
        key=lambda r: (str(r["contract_id"]), str(r["source"])),
    )
    expected_required = int(requirements.get("required_contracts", -1))
    if len(required_rows) != expected_required:
        raise ValueError(
            f"ml_targets required population {len(required_rows)} != requirements {expected_required}"
        )
    if len({str(r["contract_id"]) for r in required_rows}) != len(required_rows):
        raise ValueError("duplicate required contract_id in ml_targets.parquet")

    missing_total = 0
    mismatch_total = 0
    missing: list[dict[str, str]] = []
    mismatches: list[dict[str, str]] = []
    source_counts: Counter[str] = Counter()
    extractor_versions: Counter[str] = Counter()
    checked_contracts = 0
    checked_files = 0
    total_bytes = 0
    aggregate = hashlib.sha256()

    for row in required_rows:
        cid = str(row["contract_id"])
        source = str(row["source"])
        source_counts[source] += 1
        source_dir = representations_root / source
        graph_path = source_dir / f"{cid}.pt"
        tokens_path = source_dir / f"{cid}.tokens.pt"
        sidecar_path = source_dir / f"{cid}.rep.json"
        expected = (("graph", graph_path), ("tokens", tokens_path), ("sidecar", sidecar_path))

        contract_missing = False
        for kind, path in expected:
            if not path.is_file() or path.stat().st_size <= 0:
                missing_total += 1
                contract_missing = True
                if len(missing) < MAX_REPORTED_FAILURES:
                    missing.append({
                        "contract_id": cid,
                        "source": source,
                        "kind": kind,
                        "relative_path": f"{source}/{path.name}",
                        "reason": "missing_or_empty",
                    })
        if contract_missing:
            continue

        try:
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            mismatch_total += 1
            if len(mismatches) < MAX_REPORTED_FAILURES:
                mismatches.append({
                    "contract_id": cid,
                    "source": source,
                    "kind": "sidecar",
                    "reason": f"invalid_json:{type(exc).__name__}",
                })
            continue

        sidecar_errors: list[str] = []
        if str(sidecar.get("sha256")) != cid:
            sidecar_errors.append("contract_id_mismatch")
        if str(sidecar.get("source")) != source:
            sidecar_errors.append("source_mismatch")
        if str(sidecar.get("schema_version")) != EXPECTED_GRAPH_SCHEMA:
            sidecar_errors.append("graph_schema_mismatch")
        extractor = str(sidecar.get("extractor_version") or "UNKNOWN")
        extractor_versions[extractor] += 1
        if sidecar_errors:
            mismatch_total += len(sidecar_errors)
            if len(mismatches) < MAX_REPORTED_FAILURES:
                mismatches.append({
                    "contract_id": cid,
                    "source": source,
                    "kind": "sidecar",
                    "reason": ",".join(sidecar_errors),
                })
            continue

        graph_sha = _sha256(graph_path)
        tokens_sha = _sha256(tokens_path)
        sidecar_sha = _sha256(sidecar_path)
        graph_bytes = graph_path.stat().st_size
        tokens_bytes = tokens_path.stat().st_size
        sidecar_bytes = sidecar_path.stat().st_size
        total_bytes += graph_bytes + tokens_bytes + sidecar_bytes
        checked_files += 3
        checked_contracts += 1
        aggregate.update(
            (
                f"{cid}\0{source}\0{graph_sha}\0{graph_bytes}\0"
                f"{tokens_sha}\0{tokens_bytes}\0{sidecar_sha}\0{sidecar_bytes}\n"
            ).encode("utf-8")
        )

    passed = (
        missing_total == 0
        and mismatch_total == 0
        and checked_contracts == expected_required
        and checked_files == expected_required * 3
    )
    report = {
        "schema": "sentinel-data-vnext-representation-binding-report-v1",
        "status": "VALIDATED_LOCAL_G7" if passed else "FAILED_LOCAL_G7",
        "passed": passed,
        "representation_root": LOGICAL_REPRESENTATION_ROOT,
        "physical_root_recorded": False,
        "graph_schema_version": EXPECTED_GRAPH_SCHEMA,
        "required_contracts": expected_required,
        "checked_contracts": checked_contracts,
        "checked_files": checked_files,
        "expected_files": expected_required * 3,
        "missing_files_total": missing_total,
        "mismatch_total": mismatch_total,
        "missing_files": missing,
        "mismatches": mismatches,
        "reported_failure_limit": MAX_REPORTED_FAILURES,
        "source_counts": dict(sorted(source_counts.items())),
        "extractor_version_counts": dict(sorted(extractor_versions.items())),
        "total_bytes": total_bytes,
        "binding_digest_sha256": aggregate.hexdigest() if passed else None,
        "binding_algorithm": "sha256(sorted contract_id/source + graph/token/sidecar sha256+size)",
    }
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def bind_representation_report(export_dir: Path, report_path: Path) -> dict[str, Any]:
    """Bind a successful local scan without yet claiming final G7 validation."""
    export_dir = Path(export_dir)
    report_path = Path(report_path)
    manifest_path = export_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    if not report_path.is_file():
        raise FileNotFoundError(report_path)

    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "VALIDATED_LOCAL_G7" or report.get("passed") is not True:
        raise ValueError("cannot bind an unsuccessful representation report")
    if report.get("missing_files_total") != 0 or report.get("mismatch_total") != 0:
        raise ValueError("representation report contains failures")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "SEMANTIC_VALIDATED_REPRESENTATIONS_PENDING":
        raise ValueError(f"unexpected manifest status before representation binding: {manifest.get('status')!r}")
    requirements = json.loads((export_dir / "representation_requirements.json").read_text(encoding="utf-8"))
    if int(report.get("required_contracts", -1)) != int(requirements.get("required_contracts", -2)):
        raise ValueError("representation report population does not match requirements")

    manifest["status"] = "REPRESENTATIONS_VALIDATED_G7_PENDING_FINAL"
    manifest["representation_binding_report"] = {
        "path": report_path.name,
        "sha256": _sha256(report_path),
        "bytes": report_path.stat().st_size,
        "status": report["status"],
        "required_contracts": report["required_contracts"],
        "binding_digest_sha256": report["binding_digest_sha256"],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


__all__ = ["bind_representation_report", "verify_local_representations"]
