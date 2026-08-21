#!/usr/bin/env python3
"""Audit existing v9 EXTERNAL_CALL edge semantics without mutating artifacts.

The audit compares type-11 graph self-loops with normalized Solidity source and
declared library names.  It is deliberately diagnostic: name-based matching can
prove some library-edge false positives but cannot prove that every remaining
edge is a genuine unknown-target call.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import torch


EXTERNAL_CALL_EDGE = 11
SCHEMA = "sentinel-r4-v9-external-call-semantics-audit-v1"
LIBRARY_DECLARATION_RE = re.compile(r"\blibrary\s+([A-Za-z_][A-Za-z0-9_]*)\b")
RAW_LOW_LEVEL_RE = re.compile(
    r"\.(?:call|callcode|delegatecall|staticcall|send)\s*(?:\.|\{|\()"
)
TRANSFER_RE = re.compile(r"\.\s*transfer\s*\(")
SEND_RE = re.compile(r"\.\s*send\s*\(")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_worktree_dirty(repo_root: Path) -> bool:
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return bool(result.stdout.strip())


def declared_library_names(source: str) -> tuple[str, ...]:
    return tuple(sorted(set(LIBRARY_DECLARATION_RE.findall(source))))


def classify_edge_name(name: str, libraries: Iterable[str]) -> dict[str, bool]:
    return {
        "declared_library": any(f"{library}." in name for library in libraries),
        "transfer": bool(TRANSFER_RE.search(name)),
        "send": bool(SEND_RE.search(name)),
        "raw_low_level": bool(RAW_LOW_LEVEL_RE.search(name)),
    }


def _graph_value(graph: Any, key: str) -> Any:
    if isinstance(graph, dict):
        return graph[key]
    return getattr(graph, key)


def audit_one(
    *,
    graph_path: Path,
    source_path: Path,
    sidecar_path: Path,
    queue_row: dict[str, Any] | None,
) -> dict[str, Any]:
    source = source_path.read_text(encoding="utf-8", errors="replace")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    graph = torch.load(graph_path, map_location="cpu", weights_only=False)
    edge_attr = _graph_value(graph, "edge_attr")
    edge_index = _graph_value(graph, "edge_index")
    metadata = _graph_value(graph, "node_metadata")
    libraries = declared_library_names(source)

    edge_positions = (
        (edge_attr == EXTERNAL_CALL_EDGE).nonzero(as_tuple=False).flatten().tolist()
    )
    edge_node_indices = [int(edge_index[0, position]) for position in edge_positions]
    edge_names = [str(metadata[index].get("name", "")) for index in edge_node_indices]
    classifications = [classify_edge_name(name, libraries) for name in edge_names]
    external_node_set = set(edge_node_indices)

    transfer_nodes = [
        index
        for index, item in enumerate(metadata)
        if TRANSFER_RE.search(str(item.get("name", "")))
    ]
    send_nodes = [
        index
        for index, item in enumerate(metadata)
        if SEND_RE.search(str(item.get("name", "")))
    ]
    raw_nodes = [
        index
        for index, item in enumerate(metadata)
        if RAW_LOW_LEVEL_RE.search(str(item.get("name", "")))
    ]

    contract_id = graph_path.stem
    result = {
        "contract_id": contract_id,
        "source": graph_path.parent.name,
        "graph_component_count": int(sidecar.get("graph_component_count", 0)),
        "retained_token_ratio": float(sidecar.get("retained_token_ratio", 0.0)),
        "declared_libraries": list(libraries),
        "external_call_edges": len(edge_positions),
        "external_call_edges_declared_library": sum(
            item["declared_library"] for item in classifications
        ),
        "external_call_edges_transfer_named": sum(
            item["transfer"] for item in classifications
        ),
        "external_call_edges_send_named": sum(item["send"] for item in classifications),
        "external_call_edges_raw_low_level_named": sum(
            item["raw_low_level"] for item in classifications
        ),
        "source_has_transfer_syntax": bool(TRANSFER_RE.search(source)),
        "source_has_send_syntax": bool(SEND_RE.search(source)),
        "source_has_raw_low_level_syntax": bool(RAW_LOW_LEVEL_RE.search(source)),
        "graph_transfer_nodes": len(transfer_nodes),
        "graph_transfer_nodes_with_external_edge": sum(
            index in external_node_set for index in transfer_nodes
        ),
        "graph_send_nodes": len(send_nodes),
        "graph_send_nodes_with_external_edge": sum(
            index in external_node_set for index in send_nodes
        ),
        "graph_raw_low_level_nodes": len(raw_nodes),
        "graph_raw_low_level_nodes_with_external_edge": sum(
            index in external_node_set for index in raw_nodes
        ),
    }
    if queue_row is not None:
        result["queue"] = {
            "candidate_id": queue_row["candidate_id"],
            "class_index": int(queue_row["class_index"]),
            "class_name": queue_row["class_name"],
            "queue_ordinal_within_class": int(
                queue_row["queue_ordinal_within_class"]
            ),
        }
    return result


def _increment_summary(summary: Counter[str], row: dict[str, Any]) -> None:
    summary["graphs_scanned"] += 1
    edge_count = int(row["external_call_edges"])
    library_count = int(row["external_call_edges_declared_library"])
    summary["external_call_edges"] += edge_count
    summary["external_call_edges_declared_library"] += library_count
    summary["graph_transfer_nodes"] += int(row["graph_transfer_nodes"])
    summary["graph_transfer_nodes_with_external_edge"] += int(
        row["graph_transfer_nodes_with_external_edge"]
    )
    summary["graph_send_nodes"] += int(row["graph_send_nodes"])
    summary["graph_send_nodes_with_external_edge"] += int(
        row["graph_send_nodes_with_external_edge"]
    )
    summary["graph_raw_low_level_nodes"] += int(row["graph_raw_low_level_nodes"])
    summary["graph_raw_low_level_nodes_with_external_edge"] += int(
        row["graph_raw_low_level_nodes_with_external_edge"]
    )
    if edge_count:
        summary["graphs_with_external_call_edge"] += 1
    if library_count:
        summary["graphs_with_declared_library_external_edge"] += 1
    if edge_count and library_count == edge_count:
        summary["graphs_all_external_edges_declared_library"] += 1
    if row["source_has_transfer_syntax"]:
        summary["graphs_with_transfer_syntax"] += 1
    if row["source_has_send_syntax"]:
        summary["graphs_with_send_syntax"] += 1
    if row["source_has_raw_low_level_syntax"]:
        summary["graphs_with_raw_low_level_syntax"] += 1
    if row["source_has_transfer_syntax"] and not row[
        "graph_transfer_nodes_with_external_edge"
    ]:
        summary["transfer_syntax_without_transfer_external_edge"] += 1
    if row["source_has_send_syntax"] and not row["graph_send_nodes_with_external_edge"]:
        summary["send_syntax_without_send_external_edge"] += 1
    if float(row["retained_token_ratio"]) < 0.5:
        summary["graphs_with_retained_token_ratio_below_0_5"] += 1


def _git_head(repo_root: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
    ).strip()


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    queue = json.loads(args.queue.read_text(encoding="utf-8"))
    queue_rows = list(queue.get("candidates") or [])
    if args.class_name:
        queue_rows = [row for row in queue_rows if row["class_name"] == args.class_name]
    queue_by_identity = {
        (str(row["source"]), str(row["contract_id"])): row for row in queue_rows
    }

    if args.scope == "queue":
        graph_paths = [
            args.representations_root / row["source"] / f"{row['contract_id']}.pt"
            for row in queue_rows
        ]
    else:
        graph_paths = sorted(
            path
            for path in args.representations_root.glob("*/*.pt")
            if not path.name.endswith(".tokens.pt")
        )

    summary: Counter[str] = Counter()
    by_source: dict[str, Counter[str]] = {}
    by_queue_class: dict[str, Counter[str]] = {}
    anomalies: list[dict[str, Any]] = []
    queue_records: list[dict[str, Any]] = []

    for ordinal, graph_path in enumerate(graph_paths, start=1):
        source_name = graph_path.parent.name
        contract_id = graph_path.stem
        source_path = args.preprocessed_root / source_name / f"{contract_id}.sol"
        sidecar_path = graph_path.with_name(f"{contract_id}.rep.json")
        if not graph_path.is_file() or not source_path.is_file() or not sidecar_path.is_file():
            raise FileNotFoundError(
                f"incomplete binding: graph={graph_path.is_file()} "
                f"source={source_path.is_file()} sidecar={sidecar_path.is_file()} "
                f"identity={source_name}/{contract_id}"
            )
        queue_row = queue_by_identity.get((source_name, contract_id))
        row = audit_one(
            graph_path=graph_path,
            source_path=source_path,
            sidecar_path=sidecar_path,
            queue_row=queue_row,
        )
        _increment_summary(summary, row)
        source_summary = by_source.setdefault(source_name, Counter())
        _increment_summary(source_summary, row)
        if queue_row is not None:
            class_summary = by_queue_class.setdefault(
                str(queue_row["class_name"]), Counter()
            )
            _increment_summary(class_summary, row)
            queue_records.append(row)

        if (
            row["external_call_edges_declared_library"]
            or (
                row["source_has_transfer_syntax"]
                and not row["graph_transfer_nodes_with_external_edge"]
            )
            or (
                row["source_has_send_syntax"]
                and not row["graph_send_nodes_with_external_edge"]
            )
        ) and len(anomalies) < args.max_examples:
            anomalies.append(row)

        if args.progress_every and ordinal % args.progress_every == 0:
            print(
                f"audited {ordinal}/{len(graph_paths)} graphs",
                file=sys.stderr,
                flush=True,
            )

    script_path = Path(__file__).resolve()
    repo_root = script_path.parents[4]
    report = {
        "schema": SCHEMA,
        "status": "PASS_DIAGNOSTIC_ONLY",
        "scope": args.scope,
        "class_name_filter": args.class_name,
        "repository_head": _git_head(repo_root),
        "repository_worktree_dirty": _git_worktree_dirty(repo_root),
        "audit_script_sha256": sha256_file(script_path),
        "lineage_note": (
            "repository_head identifies the checked-out base; audit_script_sha256 "
            "binds the exact audit implementation because the report may be generated "
            "before its documentation tranche is committed"
        ),
        "queue_sha256": sha256_file(args.queue),
        "external_call_edge_type": EXTERNAL_CALL_EDGE,
        "summary": dict(sorted(summary.items())),
        "by_source": {
            key: dict(sorted(value.items())) for key, value in sorted(by_source.items())
        },
        "queued_candidates_in_scope": len(queue_rows),
        "queue_records_observed": len(queue_records),
        "by_queue_class": {
            key: dict(sorted(value.items()))
            for key, value in sorted(by_queue_class.items())
        },
        "queue_records": sorted(
            queue_records,
            key=lambda row: (
                int(row["queue"]["class_index"]),
                int(row["queue"]["queue_ordinal_within_class"]),
            ),
        ),
        "anomaly_examples": anomalies,
        "limitations": [
            "Declared-library matching proves only calls whose CFG metadata names contain a library declared in the same normalized source; it can undercount using-for calls, imported libraries, aliases, and metadata-normalization variants.",
            "Transfer/send name matching does not by itself distinguish Ether address operations from token-interface methods.",
            "Type-11 presence and source regexes are representation diagnostics, not vulnerability truth or label authority.",
            "Token retained ratio is diagnostic only and has no current adequacy threshold.",
        ],
        "authorizations": {
            "training": False,
            "selector_promotion": False,
            "label_change": False,
            "artifact_mutation": False,
        },
    }
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preprocessed-root",
        type=Path,
        default=Path("data_module/data/sentinel-preprocessed-r4-v2"),
    )
    parser.add_argument(
        "--representations-root",
        type=Path,
        default=Path("data_module/data/representations-r4-v2"),
    )
    parser.add_argument(
        "--queue",
        type=Path,
        default=Path(
            "docs/plan/ml-R4/evidence/2026-08-15_phase8_logical_v3/"
            "confirmed_negative_review_queue_v1.json"
        ),
    )
    parser.add_argument("--scope", choices=("queue", "population"), default="queue")
    parser.add_argument("--class-name")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-examples", type=int, default=50)
    parser.add_argument("--progress-every", type=int, default=1000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_report(args)
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
