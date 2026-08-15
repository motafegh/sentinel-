#!/usr/bin/env python3
"""Profile repaired-v2 compatibility/file-union/worst-case sensitivity sets.

The output is read-only research evidence. It identifies exact contracts for
bounded exclusion/down-weighting and worst-case GPU comparisons without
changing the accepted repaired-v2 representation lineage.
"""

from __future__ import annotations

import argparse
import json
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

    rows = pq.read_table(args.overlay / "ml_targets.parquet").to_pylist()
    records: list[dict] = []
    missing: list[dict[str, str]] = []
    for row in rows:
        contract_id = str(row["contract_id"])
        source = str(row["source"])
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
        selection_active = any(
            bool(row.get(f"outcome_metric_mask_{index}"))
            for index in range(len(CLASS_NAMES))
        )

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
                "role": str(row["role"]),
                "optimizer_active": optimizer_active,
                "model_selection_active": selection_active,
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

    report = profile_representation_records(records, top_n=args.top_n)
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
