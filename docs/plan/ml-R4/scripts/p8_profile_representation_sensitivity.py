#!/usr/bin/env python3
"""Profile repaired Phase-8 compatibility/file-union/worst-case sensitivity sets.

The output is read-only research evidence. It identifies exact contracts for
bounded exclusion/down-weighting and worst-case GPU comparisons without
changing the accepted repaired representation lineage. The report is
self-identifying: publication versions/hash, physical binding digest and source
commit are recorded so stale reports cannot be mixed silently.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "data_module"))

from sentinel_data.representation.r4_compatibility import FULL_ANALYSIS
from sentinel_data.representation.r4_sensitivity import (
    profile_representation_records,
)
from sentinel_data.vnext.policy import CLASS_NAMES

DATA_ROOT = REPO_ROOT / "data_module/data"
DEFAULT_OVERLAY = DATA_ROOT / "exports/sentinel-r4-vnext-v2"
DEFAULT_REPRESENTATIONS = DATA_ROOT / "representations-r4-v2"
DEFAULT_OUTPUT = DATA_ROOT / "r4-v2-build/representation_sensitivity_v1.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_commit() -> str:
    return subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        text=True,
    ).strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overlay", type=Path, default=DEFAULT_OVERLAY)
    parser.add_argument(
        "--representations-root",
        type=Path,
        default=DEFAULT_REPRESENTATIONS,
    )
    parser.add_argument("--top-n", type=int, default=25)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    import pyarrow.parquet as pq

    manifest_path = args.overlay / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    binding = manifest.get("representation_binding_report") or {}
    binding_digest = str(binding.get("binding_digest_sha256") or "")
    if not binding_digest:
        raise ValueError("sensitivity overlay manifest lacks representation binding digest")

    ml_targets_path = args.overlay / "ml_targets.parquet"
    expected_ml_sha = ((manifest.get("artifacts") or {}).get("ml_targets") or {}).get(
        "sha256"
    )
    if not expected_ml_sha or _sha256(ml_targets_path) != expected_ml_sha:
        raise ValueError("sensitivity overlay ml_targets.parquet hash mismatch")

    rows = pq.read_table(ml_targets_path).to_pylist()
    records: list[dict] = []
    missing: list[dict[str, str]] = []
    unexpected_metric_roles: set[str] = set()
    for row in rows:
        contract_id = str(row["contract_id"])
        source = str(row["source"])
        role = str(row["role"])
        sidecar_path = (
            args.representations_root / source / f"{contract_id}.rep.json"
        )
        if not sidecar_path.is_file():
            missing.append(
                {
                    "contract_id": contract_id,
                    "source": source,
                    "error": "missing representation sidecar",
                }
            )
            continue
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        optimizer_active = any(
            bool(row.get(f"effective_loss_mask_{index}"))
            for index in range(len(CLASS_NAMES))
        )
        metric_active = any(
            bool(row.get(f"outcome_metric_mask_{index}"))
            for index in range(len(CLASS_NAMES))
        )
        if metric_active and role not in {"MODEL_SELECTION", "INTERNAL_AUDIT"}:
            unexpected_metric_roles.add(role)
        model_selection_active = metric_active and role == "MODEL_SELECTION"
        internal_audit_active = metric_active and role == "INTERNAL_AUDIT"

        mode_value = sidecar.get("graph_extraction_mode")
        mode_inferred = mode_value is None
        if mode_inferred:
            # This is the same legacy-standard inference already accepted by
            # the repaired-v2 physical binder for byte-reused successful
            # sidecars from a failed-tail recovery build. A source transform
            # would contradict standard/full-analysis inference and therefore
            # remains fail-closed.
            if sidecar.get("graph_source_transform") is not None:
                raise ValueError(
                    f"{contract_id} lacks graph_extraction_mode but records "
                    "graph_source_transform"
                )
            graph_mode = FULL_ANALYSIS
        else:
            graph_mode = str(mode_value)

        records.append(
            {
                "contract_id": contract_id,
                "source": source,
                "role": role,
                "optimizer_active": optimizer_active,
                "model_selection_active": model_selection_active,
                "internal_audit_active": internal_audit_active,
                "graph_extraction_mode": graph_mode,
                "graph_extraction_mode_inferred_legacy_standard": mode_inferred,
                "graph_analysis_degraded": sidecar.get(
                    "graph_analysis_degraded"
                ),
                "graph_component_count": sidecar.get("graph_component_count"),
                "node_count": sidecar.get("node_count"),
                "edge_count": sidecar.get("edge_count"),
                "pre_subsampling_window_count": sidecar.get(
                    "pre_subsampling_window_count",
                    sidecar.get("window_count"),
                ),
            }
        )

    if missing:
        raise RuntimeError(
            "representation sensitivity requires complete bound sidecars; "
            f"missing={len(missing)} preview={missing[:5]}"
        )
    if unexpected_metric_roles:
        raise ValueError(
            "outcome metric masks appear on unexpected roles: "
            f"{sorted(unexpected_metric_roles)}"
        )

    report = profile_representation_records(records, top_n=args.top_n)
    report["lineage"] = {
        "dataset_version": manifest.get("dataset_version"),
        "grouping_version": manifest.get("grouping_version"),
        "partition_version": manifest.get("partition_version"),
        "publication_manifest_sha256": _sha256(manifest_path),
        "representation_binding_digest_sha256": binding_digest,
        "source_commit": _source_commit(),
    }
    report["overlay"] = str(args.overlay)
    report["representations_root"] = str(args.representations_root)
    report["full_training_authorized"] = False
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    print(text, end="")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
