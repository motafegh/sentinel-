"""Feature-based semantic checker — Stage 4.

Uses graph features from .pt files and rep.json metadata to verify that
labeled contracts actually contain evidence for each vulnerability class.

Coverage status per class:
  Reentrancy          — schema-aware has_cei_path flag
  Timestamp           — feat[2] uses_block_globals fires
  IntegerUO           — pragma < 0.8 OR feat[11] unchecked_block fires
  UnusedReturn        — feat[7] return_ignored fires
  MishandledException — feat[7] return_ignored fires (low-level call)
  CallToUnknown       — v10 LOW_LEVEL_CALL / ETHER_SEND coarse signal only
  ExternalBug         — NOT_EXTRACTABLE until a class-specific signal exists
  DenialOfService     — NOT_EXTRACTABLE (no loop-external-call feature in v9)
  GasException        — NOT_EXTRACTABLE (no unchecked-send feature in v9)
  TOD                 — NOT_EXTRACTABLE (no tx.origin feature in v9)

All checks require a graph .pt file. Without it the check is SKIP.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

from sentinel_data.representation.graph_schema_versions import get_graph_schema

log = logging.getLogger("sentinel_data.verification.semantic_checker")

_EXTERNAL_CALL_EDGE = 11  # historical v9 helper only

# Node type IDs from v9 schema (feat[0] * 13.0 = type_id)
_MAX_TYPE_ID = 13.0


class CheckVerdict(str, Enum):
    """Verdict for a single (class, contract) semantic check."""

    PASS = "PASS"           # semantic evidence present
    FAIL = "FAIL"           # labeled positive but no semantic evidence
    SKIP = "SKIP"           # no representation exists — cannot check
    NOT_EXTRACTABLE = "NOT_EXTRACTABLE"  # v9 schema has no feature for this class


@dataclass
class ContractCheckResult:
    """Result of a semantic check on one (class, contract) pair."""

    sha256: str
    class_name: str
    verdict: CheckVerdict
    note: str = ""


@dataclass
class ClassCheckSummary:
    """Aggregate semantic check results for a single vulnerability class."""

    class_name: str
    positives_checked: int = 0      # labeled positives with graph rep
    positives_skipped: int = 0      # labeled positives without graph rep
    pass_count: int = 0
    fail_count: int = 0
    not_extractable: int = 0
    details: list[ContractCheckResult] = field(default_factory=list)

    @property
    def pass_rate(self) -> Optional[float]:
        denom = self.pass_count + self.fail_count
        return self.pass_count / denom if denom > 0 else None

    @property
    def coverage(self) -> float:
        total = self.positives_checked + self.positives_skipped
        return self.positives_checked / total if total > 0 else 0.0


@dataclass
class SemanticCheckResult:
    """Top-level result container for the full semantic check run."""

    by_class: dict[str, ClassCheckSummary] = field(default_factory=dict)
    total_positives: int = 0
    total_checked: int = 0
    total_skipped: int = 0
    duration_s: float = 0.0


def _load_graph(rep_dir_for_source: Path, sha256: str):
    """Load a graph .pt file. Returns None if not found."""
    pt_path = rep_dir_for_source / f"{sha256}.pt"
    if not pt_path.exists():
        return None
    try:
        import torch
        return torch.load(str(pt_path), weights_only=False)
    except Exception as e:
        log.debug(f"Cannot load graph {sha256}: {e}")
        return None


def _load_rep(rep_dir_for_source: Path, sha256: str) -> Optional[dict]:
    """Load rep.json metadata. Returns None if not found."""
    rep_path = rep_dir_for_source / f"{sha256}.rep.json"
    if not rep_path.exists():
        return None
    try:
        return json.loads(rep_path.read_text())
    except Exception:
        return None


def _is_pre_08(rep: dict) -> bool:
    """Return True if contract is pre-0.8 Solidity (unchecked arithmetic)."""
    ver = rep.get("solc_version") or ""
    if not ver:
        return False
    try:
        parts = ver.lstrip("^~=v").split(".")
        major, minor = int(parts[0]), int(parts[1])
        return major == 0 and minor < 8
    except (IndexError, ValueError):
        return False


def _has_external_call_edge(graph) -> bool:
    """Return historical v9 EXTERNAL_CALL presence; not current class truth."""
    if not hasattr(graph, "edge_attr") or graph.edge_attr is None:
        return False
    return bool((graph.edge_attr == _EXTERNAL_CALL_EDGE).any())


def _graph_schema_version(graph, rep: Optional[dict]) -> str:
    """Resolve an exact schema, defaulting old unannotated graphs to v9."""

    graph_version = getattr(graph, "graph_schema_version", None) if graph is not None else None
    rep_version = None
    if rep is not None:
        rep_version = rep.get("schema_version") or rep.get("graph_schema_version")
    graph_version = graph_version if isinstance(graph_version, str) and graph_version else None
    rep_version = rep_version if isinstance(rep_version, str) and rep_version else None
    if graph_version and rep_version and graph_version != rep_version:
        raise ValueError(
            f"graph/sidecar schema mismatch: {graph_version!r} != {rep_version!r}"
        )
    resolved = graph_version or rep_version or "v9"
    get_graph_schema(resolved)
    return resolved


def _has_edge_kinds(graph, *, schema_version: str, edge_names: tuple[str, ...]) -> bool:
    """Return whether a graph contains any requested edge kind, fail closed."""

    if graph is None or not hasattr(graph, "edge_attr") or graph.edge_attr is None:
        return False
    schema = get_graph_schema(schema_version)
    edge_ids = schema.edge_ids(edge_names)
    return any(bool((graph.edge_attr == edge_id).any()) for edge_id in edge_ids)


def _check_class(class_name: str, graph, rep: Optional[dict]) -> tuple[CheckVerdict, str]:
    """Return (verdict, note) for one (class, contract) pair."""
    schema_version = _graph_schema_version(graph, rep)
    graph_schema = get_graph_schema(schema_version)
    if (
        schema_version == "v10"
        and rep is not None
        and rep.get("graph_analysis_degraded") is True
    ):
        return (
            CheckVerdict.NOT_EXTRACTABLE,
            "v10 graph uses degraded parse-only analysis; IR semantics unavailable",
        )

    if class_name == "Reentrancy":
        if graph is None:
            return CheckVerdict.SKIP, "no graph rep"
        cei = getattr(graph, "has_cei_path", None)
        if cei is None:
            return CheckVerdict.SKIP, "has_cei_path absent (old graph)"
        return (CheckVerdict.PASS, "CEI path present") if cei == 1 else (
            CheckVerdict.FAIL, "no CEI path found (EXTERNAL_CALL before WRITE)"
        )

    if class_name == "Timestamp":
        if graph is None:
            return CheckVerdict.SKIP, "no graph rep"
        feat2 = graph.x[:, 2].max().item() if hasattr(graph, "x") else 0.0
        return (CheckVerdict.PASS, "feat[2]=1.0 uses_block_globals") if feat2 > 0.5 else (
            CheckVerdict.FAIL, "feat[2]=0 on all nodes — no block.timestamp/now detected"
        )

    if class_name == "IntegerUO":
        if graph is not None and hasattr(graph, "x"):
            feat11 = graph.x[:, 11].max().item()
            if feat11 > 0.5:
                return CheckVerdict.PASS, "feat[11]=1.0 unchecked_block present"
        if rep is not None and _is_pre_08(rep):
            return CheckVerdict.PASS, f"pre-0.8 contract ({rep.get('solc_version','')})"
        if graph is None and rep is None:
            return CheckVerdict.SKIP, "no graph rep or rep.json"
        return CheckVerdict.FAIL, "no unchecked block and not pre-0.8"

    if class_name in ("UnusedReturn", "MishandledException"):
        if graph is None:
            return CheckVerdict.SKIP, "no graph rep"
        feat7 = graph.x[:, 7].max().item() if hasattr(graph, "x") else 0.0
        return (CheckVerdict.PASS, "feat[7]=1.0 return_ignored present") if feat7 > 0.5 else (
            CheckVerdict.FAIL, "feat[7]=0 on all nodes — no ignored return detected"
        )

    if class_name == "CallToUnknown":
        if graph is None:
            return CheckVerdict.SKIP, "no graph rep"
        if schema_version == "v9":
            return (
                CheckVerdict.NOT_EXTRACTABLE,
                "v9 EXTERNAL_CALL conflates library calls and omits call kinds",
            )
        signal_names = graph_schema.call_to_unknown_signal_edge_names
        if _has_edge_kinds(
            graph,
            schema_version=schema_version,
            edge_names=signal_names,
        ):
            return (
                CheckVerdict.PASS,
                "v10 LOW_LEVEL_CALL/ETHER_SEND coarse signal present",
            )
        return (
            CheckVerdict.FAIL,
            "no v10 LOW_LEVEL_CALL/ETHER_SEND coarse signal detected",
        )

    if class_name == "ExternalBug":
        return (
            CheckVerdict.NOT_EXTRACTABLE,
            "no class-specific source-backed graph signal is defined",
        )

    # DenialOfService, GasException, TransactionOrderDependence
    # — no dedicated v9 feature for these; cannot verify from graph alone
    return CheckVerdict.NOT_EXTRACTABLE, (
        f"no {schema_version} feature covers this class"
    )


def run_semantic_check(
    data_dir: Path,
    *,
    limit_per_class: Optional[int] = None,
) -> SemanticCheckResult:
    """Run graph-feature-based semantic checks on all labeled positives.

    For each (class, contract) pair where the merged label is positive,
    checks whether the graph representation contains the expected feature
    pattern for that class.

    Args:
        data_dir: Path to data/ directory.
        limit_per_class: If set, check at most this many positives per class
                         (useful for fast smoke tests).

    Returns:
        SemanticCheckResult with per-class summaries.
    """
    from sentinel_data.labeling.schema import class_names as _class_names

    merged_dir = data_dir / "labels" / "merged"
    if not merged_dir.exists():
        raise FileNotFoundError(f"Merged labels dir not found: {merged_dir}")

    classes = _class_names()
    result = SemanticCheckResult(by_class={c: ClassCheckSummary(class_name=c) for c in classes})
    t0 = time.monotonic()

    # Collect positives per class
    per_class_positives: dict[str, list[dict]] = {c: [] for c in classes}
    for lf in merged_dir.glob("*.labels.json"):
        try:
            lj = json.loads(lf.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        sha = lj["sha256"]
        sources = lj.get("sources", [])
        for cls in classes:
            if lj["classes"].get(cls, {}).get("value") == 1:
                per_class_positives[cls].append({"sha": sha, "sources": sources})

    # Check each positive
    for cls in classes:
        positives = per_class_positives[cls]
        if limit_per_class is not None:
            positives = positives[:limit_per_class]
        summary = result.by_class[cls]

        for item in positives:
            sha = item["sha"]
            sources = item["sources"]

            # Try each source's representation dir
            graph = None
            rep = None
            for src in sources:
                rep_dir = data_dir / "representations" / src
                if not rep_dir.exists():
                    continue
                g = _load_graph(rep_dir, sha)
                if g is not None:
                    graph = g
                    rep = _load_rep(rep_dir, sha)
                    break
                # Fallback: rep.json without .pt
                r = _load_rep(rep_dir, sha)
                if r is not None:
                    rep = r

            verdict, note = _check_class(cls, graph, rep)

            cr = ContractCheckResult(sha256=sha, class_name=cls, verdict=verdict, note=note)
            summary.details.append(cr)

            if verdict == CheckVerdict.SKIP:
                summary.positives_skipped += 1
                result.total_skipped += 1
            else:
                summary.positives_checked += 1
                result.total_checked += 1
                if verdict == CheckVerdict.PASS:
                    summary.pass_count += 1
                elif verdict == CheckVerdict.FAIL:
                    summary.fail_count += 1
                else:  # NOT_EXTRACTABLE
                    summary.not_extractable += 1

        result.total_positives += len(positives)
        log.info(
            f"  {cls}: {summary.pass_count} pass / {summary.fail_count} fail / "
            f"{summary.positives_skipped} skip / {summary.not_extractable} not_extractable"
        )

    result.duration_s = time.monotonic() - t0
    return result
