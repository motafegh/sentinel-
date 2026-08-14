#!/usr/bin/env python3
"""Read-only structural audit of every Phase-8 graph/token representation.

The G7 binding proves that the required files exist and match the frozen
manifest.  This profiler asks the next question: whether the tensors inside
those files are structurally usable and how the missing-representation
population changes the effective supervision.  It writes nothing and emits a
deterministic JSON report to stdout.  Progress messages go to stderr.
"""

from __future__ import annotations

import bisect
import csv
import json
import logging
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import torch


REPO_ROOT = Path(__file__).resolve().parents[4]
DATA_ROOT = REPO_ROOT / "data_module/data"
EXPORT_ROOT = DATA_ROOT / "exports/sentinel-r4-vnext-v1"
REPRESENTATIONS_ROOT = DATA_ROOT / "representations"
PREPROCESSED_ROOT = DATA_ROOT / "preprocessed"
CLASS_NAMES = (
    "CallToUnknown",
    "DenialOfService",
    "ExternalBug",
    "GasException",
    "IntegerUO",
    "MishandledException",
    "Reentrancy",
    "Timestamp",
    "TransactionOrderDependence",
    "UnusedReturn",
)
EXPECTED_GRAPH_KEYS = {
    "contract_name",
    "edge_attr",
    "edge_index",
    "has_cei_path",
    "node_metadata",
    "num_edges",
    "num_nodes",
    "x",
}
EXPECTED_TOKEN_KEYS = {
    "attention_mask",
    "input_ids",
    "max_length",
    "num_tokens",
    "num_windows",
    "sha256",
    "source",
    "stride",
    "tokenizer_name",
}


def _mask_comments_and_strings(text: str) -> str:
    """Preserve offsets/newlines while masking brace-like lexical content."""
    out = list(text)
    index = 0
    state = "code"
    quote = ""
    while index < len(text):
        if state == "code":
            if text.startswith("//", index):
                out[index] = out[index + 1] = " "
                index += 2
                state = "line_comment"
            elif text.startswith("/*", index):
                out[index] = out[index + 1] = " "
                index += 2
                state = "block_comment"
            elif text[index] in {'"', "'"}:
                quote = text[index]
                out[index] = " "
                index += 1
                state = "string"
            else:
                index += 1
        elif state == "line_comment":
            if text[index] == "\n":
                state = "code"
            else:
                out[index] = " "
            index += 1
        elif state == "block_comment":
            if text.startswith("*/", index):
                out[index] = out[index + 1] = " "
                index += 2
                state = "code"
            else:
                if text[index] != "\n":
                    out[index] = " "
                index += 1
        else:
            if text[index] == "\\" and index + 1 < len(text):
                out[index] = " "
                if text[index + 1] != "\n":
                    out[index + 1] = " "
                index += 2
            elif text[index] == quote:
                out[index] = " "
                index += 1
                state = "code"
            else:
                if text[index] != "\n":
                    out[index] = " "
                index += 1
    return "".join(out)


def _declaration_line_spans(text: str) -> dict[str, list[tuple[int, int]]]:
    """Return raw line spans for Solidity contracts/libraries/interfaces."""
    masked = _mask_comments_and_strings(text)
    newline_offsets = [-1] + [match.start() for match in re.finditer("\n", text)]
    spans: dict[str, list[tuple[int, int]]] = defaultdict(list)
    declaration = re.compile(
        r"\b(?:contract|library|interface)\s+([A-Za-z_$][A-Za-z0-9_$]*)"
    )
    for match in declaration.finditer(masked):
        opening = masked.find("{", match.end())
        if opening < 0:
            continue
        depth = 0
        closing = None
        for index in range(opening, len(masked)):
            if masked[index] == "{":
                depth += 1
            elif masked[index] == "}":
                depth -= 1
                if depth == 0:
                    closing = index
                    break
        if closing is None:
            continue
        start_line = bisect.bisect_right(newline_offsets, match.start())
        end_line = bisect.bisect_right(newline_offsets, closing)
        spans[match.group(1)].append((start_line, end_line))
    return dict(spans)


def _classify_extraction_failure(message: str) -> str:
    patterns = (
        ("expected_string_end_quote", r"Expected string end-quote"),
        ("expected_primary_expression", r"Expected primary expression"),
        (
            "declaration_expected",
            r"Function, variable, struct or modifier declaration expected",
        ),
        (
            "top_level_definition_expected",
            r"Expected pragma, import directive or contract",
        ),
        ("slither_ir_generation", r"Failed to generate IR"),
        ("no_contracts_found", r"No contract was found|No contracts were found"),
        ("stack_too_deep", r"Stack too deep"),
        ("different_compiler", r"requires different compiler|different compiler version"),
        ("source_not_found", r"Source .* not found"),
        ("empty_graph", r"No non-dependency contracts"),
    )
    return next(
        (name for name, pattern in patterns if re.search(pattern, message, re.I)),
        "other_failure",
    )


def _quantiles(values: list[int | float]) -> dict[str, int | float]:
    if not values:
        return {}
    ordered = sorted(values)

    def at(fraction: float) -> int | float:
        return ordered[min(len(ordered) - 1, round((len(ordered) - 1) * fraction))]

    return {
        "min": ordered[0],
        "p50": at(0.50),
        "p95": at(0.95),
        "p99": at(0.99),
        "max": ordered[-1],
    }


def _category(source: str, original_path: str) -> str:
    parts = Path(original_path).parts
    if source == "smartbugs_curated" and len(parts) > 1:
        return parts[1]
    if source == "solidifi" and len(parts) > 2:
        return parts[2]
    return "__source__"


def _target_cells(row: dict[str, Any]) -> list[tuple[int, str]]:
    return [
        (index, str(row[f"strength_{index}"]))
        for index in range(len(CLASS_NAMES))
        if row[f"target_{index}"] == 1
    ]


def _load_meta(source: str, contract_id: str) -> dict[str, Any]:
    path = PREPROCESSED_ROOT / source / f"{contract_id}.meta.json"
    return json.loads(path.read_text())


def _finalize_source(acc: dict[str, Any]) -> dict[str, Any]:
    return {
        "contracts": acc["contracts"],
        "roles": dict(sorted(acc["roles"].items())),
        "graph": {
            "node_count": _quantiles(acc["node_count"]),
            "edge_count": _quantiles(acc["edge_count"]),
            "isolated_node_count": _quantiles(acc["isolated_node_count"]),
            "graphs_with_zero_edges": acc["graphs_with_zero_edges"],
            "graphs_with_isolated_nodes": acc["graphs_with_isolated_nodes"],
            "graph_bytes": _quantiles(acc["graph_bytes"]),
        },
        "tokens": {
            "real_window_count": dict(sorted(acc["window_count"].items())),
            "attention_token_slots": _quantiles(acc["attention_token_slots"]),
            "contracts_at_four_window_cap": acc["contracts_at_four_window_cap"],
            "tokens_bytes": _quantiles(acc["tokens_bytes"]),
        },
        "source_shape": {
            "contract_names_count": _quantiles(acc["contract_names_count"]),
            "multi_contract_files": acc["multi_contract_files"],
            "target_cells_in_multi_contract_files": {
                key: value
                for key, value in sorted(acc["target_cells_in_multi_contract_files"].items())
            },
        },
    }


def main() -> int:
    rows = pq.read_table(EXPORT_ROOT / "ml_targets.parquet").to_pylist()
    rows.sort(key=lambda row: (str(row["source"]), str(row["contract_id"])))

    represented = [row for row in rows if bool(row["representation_required"])]
    excluded = [row for row in rows if not bool(row["representation_required"])]
    hard_failures: Counter[str] = Counter()
    selected_contract_mismatch_by_source: Counter[str] = Counter()
    selected_contract_mismatch_by_role: Counter[str] = Counter()
    selected_contract_mismatch_names: Counter[str] = Counter()
    selected_contract_mismatch_targets: Counter[tuple[str, str, str, str]] = Counter()
    selected_graph_names: dict[tuple[str, str], str] = {}
    sidecar_versions: Counter[tuple[str, str]] = Counter()
    graph_key_sets: Counter[tuple[str, ...]] = Counter()
    token_key_sets: Counter[tuple[str, ...]] = Counter()
    edge_types: Counter[int] = Counter()
    cei_values: Counter[int] = Counter()
    target_by_role_strength: Counter[tuple[str, str]] = Counter()
    effective_by_role_strength: Counter[tuple[str, str]] = Counter()
    metric_by_role_strength: Counter[tuple[str, str]] = Counter()
    capped_by_role: Counter[str] = Counter()
    capped_target_cells: Counter[tuple[str, str, str]] = Counter()
    feature_min = [math.inf] * 12
    feature_max = [-math.inf] * 12
    feature_negative_one = [0] * 12

    by_source: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "contracts": 0,
            "roles": Counter(),
            "node_count": [],
            "edge_count": [],
            "isolated_node_count": [],
            "graphs_with_zero_edges": 0,
            "graphs_with_isolated_nodes": 0,
            "graph_bytes": [],
            "window_count": Counter(),
            "attention_token_slots": [],
            "contracts_at_four_window_cap": 0,
            "tokens_bytes": [],
            "contract_names_count": [],
            "multi_contract_files": 0,
            "target_cells_in_multi_contract_files": Counter(),
        }
    )

    for ordinal, row in enumerate(represented, start=1):
        source = str(row["source"])
        contract_id = str(row["contract_id"])
        role = str(row["role"])
        root = REPRESENTATIONS_ROOT / source
        graph_path = root / f"{contract_id}.pt"
        tokens_path = root / f"{contract_id}.tokens.pt"
        sidecar_path = root / f"{contract_id}.rep.json"
        acc = by_source[source]
        acc["contracts"] += 1
        acc["roles"][role] += 1

        for index, strength in _target_cells(row):
            target_by_role_strength[(role, strength)] += 1
            if bool(row[f"effective_loss_mask_{index}"]):
                effective_by_role_strength[(role, strength)] += 1
            if bool(row[f"outcome_metric_mask_{index}"]):
                metric_by_role_strength[(role, strength)] += 1

        try:
            meta = _load_meta(source, contract_id)
        except (OSError, json.JSONDecodeError, KeyError):
            hard_failures["metadata_load"] += 1
            meta = {}
        contract_names = meta.get("contract_names") or []
        acc["contract_names_count"].append(len(contract_names))
        if len(contract_names) > 1:
            acc["multi_contract_files"] += 1
            for index, strength in _target_cells(row):
                key = f"{CLASS_NAMES[index]}|{strength}"
                acc["target_cells_in_multi_contract_files"][key] += 1

        try:
            sidecar = json.loads(sidecar_path.read_text())
        except (OSError, json.JSONDecodeError):
            hard_failures["sidecar_load"] += 1
            sidecar = {}
        sidecar_versions[
            (str(sidecar.get("schema_version")), str(sidecar.get("extractor_version")))
        ] += 1
        if sidecar.get("sha256") != contract_id:
            hard_failures["sidecar_contract_id"] += 1
        if sidecar.get("source") != source:
            hard_failures["sidecar_source"] += 1
        if sidecar.get("original_path") != meta.get("original_path"):
            hard_failures["sidecar_original_path"] += 1

        try:
            graph = torch.load(graph_path, map_location="cpu", weights_only=False)
        except Exception:  # noqa: BLE001 - corrupt artifacts must be counted, not abort audit
            hard_failures["graph_load"] += 1
            continue
        try:
            tokens = torch.load(tokens_path, map_location="cpu", weights_only=True)
        except Exception:  # noqa: BLE001 - corrupt artifacts must be counted, not abort audit
            hard_failures["tokens_load"] += 1
            continue

        graph_key_sets[tuple(sorted(graph.keys()))] += 1
        token_key_sets[tuple(sorted(tokens))] += 1
        if set(graph.keys()) != EXPECTED_GRAPH_KEYS:
            hard_failures["graph_key_set"] += 1
        if set(tokens) != EXPECTED_TOKEN_KEYS:
            hard_failures["token_key_set"] += 1

        x = graph.x
        edge_index = graph.edge_index
        edge_attr = graph.edge_attr
        if x.dtype != torch.float32 or x.ndim != 2 or x.shape[1] != 12:
            hard_failures["graph_x_shape_or_dtype"] += 1
        if not bool(torch.isfinite(x).all()):
            hard_failures["graph_x_nonfinite"] += 1
        if x.numel() and (float(x.min()) < -1.0 or float(x.max()) > 1.0):
            hard_failures["graph_x_outside_expected_range"] += 1
        if edge_index.dtype != torch.int64 or edge_index.ndim != 2 or edge_index.shape[0] != 2:
            hard_failures["edge_index_shape_or_dtype"] += 1
        if edge_attr.dtype != torch.int64 or edge_attr.ndim != 1:
            hard_failures["edge_attr_shape_or_dtype"] += 1

        node_count = int(x.shape[0])
        edge_count = int(edge_index.shape[1])
        if int(getattr(graph, "num_nodes", -1)) != node_count:
            hard_failures["graph_num_nodes_field"] += 1
        if int(getattr(graph, "num_edges", -1)) != edge_count:
            hard_failures["graph_num_edges_field"] += 1
        if len(edge_attr) != edge_count:
            hard_failures["edge_attr_length"] += 1
        if int(sidecar.get("node_count", -1)) != node_count:
            hard_failures["sidecar_node_count"] += 1
        if int(sidecar.get("edge_count", -1)) != edge_count:
            hard_failures["sidecar_edge_count"] += 1
        if node_count == 0:
            hard_failures["zero_node_graph"] += 1
        if edge_count and (int(edge_index.min()) < 0 or int(edge_index.max()) >= node_count):
            hard_failures["edge_index_out_of_bounds"] += 1
        if edge_attr.numel() and (int(edge_attr.min()) < 0 or int(edge_attr.max()) > 11):
            hard_failures["edge_type_out_of_range"] += 1

        metadata_rows = getattr(graph, "node_metadata", None)
        if not isinstance(metadata_rows, list) or len(metadata_rows) != node_count:
            hard_failures["node_metadata_length"] += 1
        else:
            line_limit = int(meta.get("n_normalized_lines") or 0)
            bad_nodes = 0
            out_of_range_lines = 0
            for node in metadata_rows:
                if not isinstance(node, dict) or not {"name", "type", "source_lines"} <= set(node):
                    bad_nodes += 1
                    continue
                for line in node.get("source_lines") or []:
                    if not isinstance(line, int) or line < 1 or (line_limit and line > line_limit):
                        out_of_range_lines += 1
            if bad_nodes:
                hard_failures["node_metadata_required_fields"] += 1
            if out_of_range_lines:
                hard_failures["node_metadata_source_line_range"] += 1

        graph_contract_name = str(getattr(graph, "contract_name", ""))
        if not graph_contract_name:
            hard_failures["blank_graph_contract_name"] += 1
        selected_graph_names[(source, contract_id)] = graph_contract_name
        if contract_names and graph_contract_name not in contract_names:
            selected_contract_mismatch_by_source[source] += 1
            selected_contract_mismatch_by_role[role] += 1
            selected_contract_mismatch_names[graph_contract_name] += 1
            for index, strength in _target_cells(row):
                selected_contract_mismatch_targets[
                    (source, CLASS_NAMES[index], strength, role)
                ] += 1

        if node_count:
            mins = x.amin(dim=0).tolist()
            maxs = x.amax(dim=0).tolist()
            negs = (x == -1.0).sum(dim=0).tolist()
            for index in range(min(12, len(mins))):
                feature_min[index] = min(feature_min[index], float(mins[index]))
                feature_max[index] = max(feature_max[index], float(maxs[index]))
                feature_negative_one[index] += int(negs[index])
        edge_types.update(map(int, edge_attr.tolist()))
        cei_values[int(getattr(graph, "has_cei_path", -1))] += 1
        degree = torch.bincount(edge_index.reshape(-1), minlength=node_count) if edge_count else torch.zeros(node_count, dtype=torch.long)
        isolated_nodes = int((degree == 0).sum())

        input_ids = tokens.get("input_ids")
        attention_mask = tokens.get("attention_mask")
        if not isinstance(input_ids, torch.Tensor) or input_ids.shape != (4, 512) or input_ids.dtype != torch.int64:
            hard_failures["input_ids_shape_or_dtype"] += 1
            continue
        if not isinstance(attention_mask, torch.Tensor) or attention_mask.shape != (4, 512) or attention_mask.dtype != torch.int64:
            hard_failures["attention_mask_shape_or_dtype"] += 1
            continue
        if not bool(((attention_mask == 0) | (attention_mask == 1)).all()):
            hard_failures["attention_mask_nonbinary"] += 1
        if bool((attention_mask[:, 1:] > attention_mask[:, :-1]).any()):
            hard_failures["attention_mask_not_right_padded"] += 1
        if input_ids.numel() and (int(input_ids.min()) < 0 or int(input_ids.max()) >= 50265):
            hard_failures["token_id_out_of_graphcodebert_vocab"] += 1

        window_count = int((attention_mask.sum(dim=1) > 0).sum())
        attention_token_slots = int(attention_mask.sum())
        if int(tokens.get("num_windows", -1)) != window_count:
            hard_failures["token_num_windows"] += 1
        if int(tokens.get("num_tokens", -1)) != attention_token_slots:
            hard_failures["token_num_tokens"] += 1
        if int(sidecar.get("window_count", -1)) != window_count:
            hard_failures["sidecar_window_count"] += 1
        if tokens.get("sha256") != contract_id:
            hard_failures["token_contract_id"] += 1
        if tokens.get("source") != source:
            hard_failures["token_source"] += 1
        if tokens.get("tokenizer_name") != "microsoft/graphcodebert-base":
            hard_failures["tokenizer_name"] += 1
        if int(tokens.get("max_length", -1)) != 512 or int(tokens.get("stride", -1)) != 256:
            hard_failures["tokenizer_window_config"] += 1
        if window_count == 0:
            hard_failures["zero_window_tokens"] += 1

        acc["node_count"].append(node_count)
        acc["edge_count"].append(edge_count)
        acc["isolated_node_count"].append(isolated_nodes)
        acc["graphs_with_zero_edges"] += edge_count == 0
        acc["graphs_with_isolated_nodes"] += isolated_nodes > 0
        acc["graph_bytes"].append(graph_path.stat().st_size)
        acc["window_count"][window_count] += 1
        acc["attention_token_slots"].append(attention_token_slots)
        acc["tokens_bytes"].append(tokens_path.stat().st_size)
        if window_count == 4:
            acc["contracts_at_four_window_cap"] += 1
            capped_by_role[role] += 1
            for index, strength in _target_cells(row):
                capped_target_cells[(source, CLASS_NAMES[index], strength)] += 1

        if ordinal % 2000 == 0 or ordinal == len(represented):
            print(f"audited {ordinal}/{len(represented)} representations", file=sys.stderr)

    excluded_components: Counter[tuple[str, str]] = Counter()
    excluded_by_source: Counter[str] = Counter()
    excluded_by_category: Counter[tuple[str, str]] = Counter()
    excluded_by_version: Counter[tuple[str, str]] = Counter()
    excluded_targets: Counter[tuple[str, str, str]] = Counter()
    excluded_lines: dict[str, list[int]] = defaultdict(list)
    excluded_zero_contract_names: Counter[str] = Counter()
    for row in excluded:
        source = str(row["source"])
        contract_id = str(row["contract_id"])
        root = REPRESENTATIONS_ROOT / source
        signature = "".join(
            (
                "G" if (root / f"{contract_id}.pt").is_file() else "-",
                "T" if (root / f"{contract_id}.tokens.pt").is_file() else "-",
                "S" if (root / f"{contract_id}.rep.json").is_file() else "-",
            )
        )
        excluded_components[(source, signature)] += 1
        excluded_by_source[source] += 1
        meta = _load_meta(source, contract_id)
        category = _category(source, str(meta.get("original_path", "")))
        excluded_by_category[(source, category)] += 1
        excluded_by_version[(source, str(meta.get("version_bucket")))] += 1
        excluded_lines[source].append(int(meta.get("n_normalized_lines") or 0))
        excluded_zero_contract_names[source] += not bool(meta.get("contract_names"))
        for index, strength in _target_cells(row):
            excluded_targets[(source, CLASS_NAMES[index], strength)] += 1
            target_by_role_strength[("EXCLUDED", strength)] += 1

    # Re-exercise the current graph extractor against every excluded contract.
    # The historical build did not retain a durable per-contract failure log,
    # so this is the only exact current-runtime classification available.
    sys.path.insert(0, str(REPO_ROOT / "data_module"))
    from sentinel_data.representation.graph_extractor import (  # noqa: PLC0415
        GraphExtractionConfig,
        extract_contract_graph,
    )
    from sentinel_data.representation.orchestrator import (  # noqa: PLC0415
        _resolve_solc_binary,
    )

    excluded_retry: Counter[tuple[str, str]] = Counter()
    excluded_retry_successes: list[dict[str, Any]] = []
    previous_logging_disable = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    try:
        for row in excluded:
            source = str(row["source"])
            contract_id = str(row["contract_id"])
            sol_path = PREPROCESSED_ROOT / source / f"{contract_id}.sol"
            meta = _load_meta(source, contract_id)
            config_kwargs: dict[str, Any] = {
                "allow_paths": str(sol_path.parent.parent.parent.parent),
            }
            solc_binary = _resolve_solc_binary(str(meta.get("solc_version", "")))
            if solc_binary is not None:
                config_kwargs["solc_binary"] = solc_binary
                config_kwargs["solc_version"] = meta["solc_version"]
            try:
                graph = extract_contract_graph(
                    sol_path,
                    config=GraphExtractionConfig(**config_kwargs),
                )
            except Exception as exc:  # noqa: BLE001 - classify all extractor failures
                failure_class = _classify_extraction_failure(str(exc))
                excluded_retry[(source, failure_class)] += 1
            else:
                excluded_retry[(source, "now_succeeds")] += 1
                target_cells = [
                    f"{CLASS_NAMES[index]}|{strength}"
                    for index, strength in _target_cells(row)
                ]
                excluded_retry_successes.append(
                    {
                        "source": source,
                        "contract_id": contract_id,
                        "original_path": meta.get("original_path"),
                        "selected_contract": str(graph.contract_name),
                        "node_count": int(graph.num_nodes),
                        "edge_count": int(graph.num_edges),
                        "target_cells": target_cells,
                    }
                )
    finally:
        logging.disable(previous_logging_disable)

    # SolidiFI publishes per-file injection logs.  Measure whether the one
    # contract selected for the GNN includes at least one recorded injection
    # line; the token branch still consumes the complete file.
    solidifi_coverage: Counter[tuple[str, str]] = Counter()
    solidifi_zero_coverage: list[dict[str, Any]] = []
    for row in represented:
        if row["source"] != "solidifi":
            continue
        contract_id = str(row["contract_id"])
        meta = _load_meta("solidifi", contract_id)
        raw_path = DATA_ROOT / "raw/solidifi" / str(meta["original_path"])
        bug_log = raw_path.with_name(
            f"BugLog_{raw_path.stem.rsplit('_', maxsplit=1)[-1]}.csv"
        )
        with bug_log.open(newline="") as handle:
            injection_lines = [int(item["loc"]) for item in csv.DictReader(handle)]
        selected_name = selected_graph_names[("solidifi", contract_id)]
        selected_spans = _declaration_line_spans(
            raw_path.read_text(errors="replace")
        ).get(selected_name, [])
        covered_lines = [
            line
            for line in injection_lines
            if any(start <= line <= end for start, end in selected_spans)
        ]
        category = raw_path.parent.name
        solidifi_coverage[(category, "files")] += 1
        solidifi_coverage[(category, "injection_sites")] += len(injection_lines)
        solidifi_coverage[(category, "selected_graph_injection_sites")] += len(
            covered_lines
        )
        if covered_lines:
            solidifi_coverage[(category, "files_with_selected_graph_injection")] += 1
        else:
            solidifi_coverage[(category, "files_without_selected_graph_injection")] += 1
            solidifi_zero_coverage.append(
                {
                    "contract_id": contract_id,
                    "original_path": meta["original_path"],
                    "selected_contract": selected_name,
                    "selected_contract_spans": selected_spans,
                    "injection_site_count": len(injection_lines),
                }
            )

    result = {
        "schema": "sentinel-r4-phase8-representation-audit-v1",
        "read_only": True,
        "population": {
            "overlay_contracts": len(rows),
            "represented_contracts": len(represented),
            "excluded_contracts": len(excluded),
        },
        "hard_structural_failures": dict(sorted(hard_failures.items())),
        "selected_contract_mismatch": {
            "contracts": sum(selected_contract_mismatch_by_source.values()),
            "by_source": dict(sorted(selected_contract_mismatch_by_source.items())),
            "by_role": dict(sorted(selected_contract_mismatch_by_role.items())),
            "selected_names": dict(
                selected_contract_mismatch_names.most_common()
            ),
            "target_cells": {
                f"{source}|{class_name}|{strength}|{role}": count
                for (source, class_name, strength, role), count in sorted(
                    selected_contract_mismatch_targets.items()
                )
            },
            "interpretation": (
                "The graph extractor may select libraries because Slither exposes them "
                "as non-interface contracts; preprocessing contract_names records only "
                "contract declarations. These are semantic selection mismatches, not "
                "tensor corruption."
            ),
        },
        "sidecar_versions": {
            f"{schema}|{extractor}": count
            for (schema, extractor), count in sorted(sidecar_versions.items())
        },
        "graph_key_sets": {
            "|".join(keys): count for keys, count in sorted(graph_key_sets.items())
        },
        "token_key_sets": {
            "|".join(keys): count for keys, count in sorted(token_key_sets.items())
        },
        "graph_features": {
            "minimum_by_index": feature_min,
            "maximum_by_index": feature_max,
            "negative_one_sentinel_count_by_index": feature_negative_one,
            "edge_type_counts": {str(key): value for key, value in sorted(edge_types.items())},
            "has_cei_path_counts": {str(key): value for key, value in sorted(cei_values.items())},
        },
        "by_source": {
            source: _finalize_source(acc) for source, acc in sorted(by_source.items())
        },
        "supervision": {
            "target_cells_by_role_and_strength": {
                f"{role}|{strength}": count
                for (role, strength), count in sorted(target_by_role_strength.items())
            },
            "effective_loss_cells_by_role_and_strength": {
                f"{role}|{strength}": count
                for (role, strength), count in sorted(effective_by_role_strength.items())
            },
            "outcome_metric_cells_by_role_and_strength": {
                f"{role}|{strength}": count
                for (role, strength), count in sorted(metric_by_role_strength.items())
            },
        },
        "four_window_cap": {
            "represented_contracts_by_role": dict(sorted(capped_by_role.items())),
            "target_cells_by_source_class_strength": {
                f"{source}|{class_name}|{strength}": count
                for (source, class_name, strength), count in sorted(capped_target_cells.items())
            },
            "interpretation": (
                "Saved artifacts record at most four selected windows and do not retain "
                "the pre-subsampling window count; cap saturation is measurable but exact "
                "token omission is not recoverable from the saved payload."
            ),
        },
        "excluded_population": {
            "by_source": dict(sorted(excluded_by_source.items())),
            "representation_component_signature": {
                f"{source}|{signature}": count
                for (source, signature), count in sorted(excluded_components.items())
            },
            "by_category": {
                f"{source}|{category}": count
                for (source, category), count in sorted(excluded_by_category.items())
            },
            "by_version_bucket": {
                f"{source}|{version}": count
                for (source, version), count in sorted(excluded_by_version.items())
            },
            "normalized_lines": {
                source: _quantiles(values) for source, values in sorted(excluded_lines.items())
            },
            "zero_contract_names_metadata": dict(sorted(excluded_zero_contract_names.items())),
            "target_cells": {
                f"{source}|{class_name}|{strength}": count
                for (source, class_name, strength), count in sorted(excluded_targets.items())
            },
            "current_extractor_retry": {
                "outcomes": {
                    f"{source}|{outcome}": count
                    for (source, outcome), count in sorted(excluded_retry.items())
                },
                "now_succeeds": excluded_retry_successes,
                "interpretation": (
                    "The frozen exclusion records do not retain failure logs. This "
                    "retries their normalized Solidity through the current graph "
                    "extractor without writing representations."
                ),
            },
        },
        "solidifi_selected_graph_injection_coverage": {
            "totals": {
                kind: sum(
                    count
                    for (category, item_kind), count in solidifi_coverage.items()
                    if item_kind == kind
                )
                for kind in (
                    "files",
                    "injection_sites",
                    "selected_graph_injection_sites",
                    "files_with_selected_graph_injection",
                    "files_without_selected_graph_injection",
                )
            },
            "by_category": {
                f"{category}|{kind}": count
                for (category, kind), count in sorted(solidifi_coverage.items())
            },
            "files_without_selected_graph_injection": solidifi_zero_coverage,
            "interpretation": (
                "Coverage joins each retained buggy_N.sol to BugLog_N.csv and asks "
                "whether a logged raw injection line falls inside the raw declaration "
                "span of the single contract selected for the GNN. The token branch "
                "still consumes the complete normalized file."
            ),
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 1 if hard_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
