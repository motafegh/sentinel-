"""Guarded-token lineage identities and canonical metadata helpers.

This module is the single DATA owner for selector/target-evidence serialization
and digest semantics introduced by the fresh R4-D-012 physical candidate.
It contains no graph generation, tokenization, model, acceptance, or training
logic.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any

from sentinel_data.representation.r4_window_selector import (
    CONTROL_STRATEGY,
    FALLBACK_REASON_NOT_STRICTLY_GREATER,
    GREEDY_STRATEGY,
    GUARDED_STRATEGY,
    SelectorDecision,
    union_length,
)

SELECTOR_DECISION_SCHEMA = "sentinel-r4-token-selector-decision-v1"
TARGET_EVIDENCE_SCHEMA = "sentinel-r4-selector-target-evidence-v1"
TOKEN_LINEAGE_ID = "r4-v10-v26-target-aware-guarded-v1"
CANDIDATE_ROOT_NAME = "representations-r4-v10-v26-target-aware-guarded-v1-candidate"
BINDING_REPORT_SCHEMA = "sentinel-r4-v10-guarded-candidate-binding-v1"

PARENT_DECISION_ID = "R4-D-011"
PARENT_ACCEPTANCE_SCHEMA = "sentinel-r4-v10-v26-physical-acceptance-v1"
PARENT_BINDING_DIGEST_SHA256 = (
    "d9f925588913e66476cfbc097bace7daa7e673295fe2a243760313d0bef5ebdd"
)
PARENT_GRAPH_SCHEMA_VERSION = "v10"
PARENT_EXTRACTOR_VERSION = "v2.6-r4-call-semantics-deterministic-cfg-mutators"

TOKENIZER_NAME = "microsoft/graphcodebert-base"
WINDOW_SIZE = 512
STRIDE = 256
MAX_WINDOWS = 4

SELECTOR_CONFIG = MappingProxyType(
    {
        "requested_strategy": GUARDED_STRATEGY,
        "candidate_strategy": GREEDY_STRATEGY,
        "control_strategy": CONTROL_STRATEGY,
        "tokenizer_name": TOKENIZER_NAME,
        "window_size": WINDOW_SIZE,
        "stride": STRIDE,
        "max_windows": MAX_WINDOWS,
        "target_metric": "union_requested_target_token_coverage_v1",
        "guard": "candidate_target_coverage_strictly_greater_v1",
        "greedy_tie_break": "lowest_window_index_v1",
        "fill_policy": "historical_control_then_ascending_v1",
        "source_view": "repaired_preprocessed_bytes_v1",
    }
)


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    if isinstance(value, list):
        return [_plain(item) for item in value]
    if value is None or isinstance(value, (str, int, bool)):
        return value
    raise TypeError(f"unsupported canonical metadata value {type(value).__name__}")


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize lineage metadata with one stable canonical JSON contract."""

    return json.dumps(
        _plain(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def canonical_digest(value: Any) -> str:
    """Return SHA-256 over canonical JSON bytes."""

    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


SELECTOR_CONFIG_SHA256 = canonical_digest(SELECTOR_CONFIG)


def _require_sha256(value: Any, *, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a SHA-256 hex string")
    normalized = value.lower()
    if len(normalized) != 64 or any(ch not in "0123456789abcdef" for ch in normalized):
        raise ValueError(f"{field} must be a SHA-256 hex string")
    return normalized


def _require_contract_id(value: Any) -> str:
    return _require_sha256(value, field="contract_id")


def _require_names(values: Sequence[Any]) -> list[str]:
    if isinstance(values, (str, bytes)):
        raise ValueError("requested_contract_names must be a non-empty sequence")
    names: list[str] = []
    for value in values:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("requested_contract_names contains an invalid name")
        names.append(value.strip())
    if not names:
        raise ValueError("requested_contract_names must not be empty")
    if len(set(names)) != len(names):
        raise ValueError("requested_contract_names must be unique")
    return names


def _require_ranges(
    values: Sequence[Sequence[Any]],
    *,
    field: str,
) -> list[list[int]]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{field} must be a non-empty range sequence")
    normalized: list[list[int]] = []
    for index, value in enumerate(values):
        if len(value) != 2:
            raise ValueError(f"{field}[{index}] must contain exactly two integers")
        start, end = value
        if (
            isinstance(start, bool)
            or isinstance(end, bool)
            or not isinstance(start, int)
            or not isinstance(end, int)
            or start < 0
            or end <= start
        ):
            raise ValueError(f"{field}[{index}] is not a valid half-open range")
        normalized.append([start, end])
    if not normalized:
        raise ValueError(f"{field} must not be empty")
    return normalized


def graph_parent_authority() -> dict[str, str]:
    """Return the immutable governance identity of the accepted graph parent."""

    return {
        "decision_id": PARENT_DECISION_ID,
        "acceptance_schema": PARENT_ACCEPTANCE_SCHEMA,
        "binding_digest_sha256": PARENT_BINDING_DIGEST_SHA256,
        "graph_schema_version": PARENT_GRAPH_SCHEMA_VERSION,
        "extractor_version": PARENT_EXTRACTOR_VERSION,
    }


def validate_graph_parent_authority(value: Mapping[str, Any]) -> None:
    """Fail closed unless a parent record is exactly R4-D-011 authority."""

    if _plain(value) != graph_parent_authority():
        raise ValueError("graph_parent does not match immutable R4-D-011 authority")


def build_target_evidence(
    *,
    contract_id: str,
    requested_contract_names: Sequence[str],
    target_char_spans: Sequence[Sequence[int]],
    target_token_ranges: Sequence[Sequence[int]],
    target_tokens: int,
    selector_config_sha256: str = SELECTOR_CONFIG_SHA256,
) -> dict[str, Any]:
    """Build one canonical successful target-evidence record."""

    contract_id = _require_contract_id(contract_id)
    names = _require_names(requested_contract_names)
    char_spans = _require_ranges(target_char_spans, field="target_char_spans")
    token_ranges = _require_ranges(target_token_ranges, field="target_token_ranges")
    config_digest = _require_sha256(
        selector_config_sha256, field="selector_config_sha256"
    )
    if config_digest != SELECTOR_CONFIG_SHA256:
        raise ValueError("selector_config_sha256 does not match production config")
    if len(names) != len(char_spans) or len(names) != len(token_ranges):
        raise ValueError(
            "requested names, character spans and token ranges must have equal length"
        )
    if isinstance(target_tokens, bool) or not isinstance(target_tokens, int):
        raise ValueError("target_tokens must be an integer")
    expected_target_tokens = union_length(token_ranges)
    if target_tokens != expected_target_tokens:
        raise ValueError(
            f"target_tokens mismatch: {target_tokens} != {expected_target_tokens}"
        )

    payload: dict[str, Any] = {
        "schema": TARGET_EVIDENCE_SCHEMA,
        "contract_id": contract_id,
        "requested_contract_names": names,
        "target_char_spans": char_spans,
        "target_token_ranges": token_ranges,
        "target_tokens": target_tokens,
        "selector_config_sha256": config_digest,
    }
    payload["sha256"] = canonical_digest(payload)
    return payload


def validate_target_evidence(value: Mapping[str, Any]) -> None:
    """Validate schema, semantics and self-digest of target evidence."""

    raw = _plain(value)
    expected = build_target_evidence(
        contract_id=raw.get("contract_id"),
        requested_contract_names=raw.get("requested_contract_names") or (),
        target_char_spans=raw.get("target_char_spans") or (),
        target_token_ranges=raw.get("target_token_ranges") or (),
        target_tokens=raw.get("target_tokens"),
        selector_config_sha256=raw.get("selector_config_sha256") or "",
    )
    if raw != expected:
        raise ValueError("target_evidence is not canonical or its digest is invalid")


def _selector_mapping(
    decision: SelectorDecision,
    *,
    target_evidence_sha256: str,
    selector_config_sha256: str,
) -> dict[str, Any]:
    return {
        "schema": SELECTOR_DECISION_SCHEMA,
        "requested_strategy": decision.requested_strategy,
        "effective_strategy": decision.effective_strategy,
        "selector_config_sha256": selector_config_sha256,
        "selected_indices": list(decision.selected_indices),
        "control_indices": list(decision.control_indices),
        "candidate_indices": list(decision.candidate_indices),
        "used_control_fallback": decision.used_control_fallback,
        "fallback_reason": decision.fallback_reason,
        "total_windows": decision.total_windows,
        "max_windows": decision.max_windows,
        "target_evidence_sha256": target_evidence_sha256,
        "target_coverage_tokens": decision.target_coverage_tokens,
        "control_target_coverage_tokens": decision.control_target_coverage_tokens,
        "candidate_target_coverage_tokens": decision.candidate_target_coverage_tokens,
        "retained_tokens": decision.retained_tokens,
        "control_retained_tokens": decision.control_retained_tokens,
        "candidate_retained_tokens": decision.candidate_retained_tokens,
    }


def build_selector_metadata(
    decision: SelectorDecision,
    *,
    target_evidence_sha256: str,
    selector_config_sha256: str = SELECTOR_CONFIG_SHA256,
) -> dict[str, Any]:
    """Serialize one successful selector decision for token + sidecar artifacts."""

    if not isinstance(decision, SelectorDecision):
        raise ValueError("decision must be a SelectorDecision")
    evidence_digest = _require_sha256(
        target_evidence_sha256, field="target_evidence_sha256"
    )
    config_digest = _require_sha256(
        selector_config_sha256, field="selector_config_sha256"
    )
    if config_digest != SELECTOR_CONFIG_SHA256:
        raise ValueError("selector_config_sha256 does not match production config")
    return _selector_mapping(
        decision,
        target_evidence_sha256=evidence_digest,
        selector_config_sha256=config_digest,
    )


def validate_selector_metadata(value: Mapping[str, Any]) -> None:
    """Validate one canonical successful selector decision mapping."""

    raw = _plain(value)
    if raw.get("schema") != SELECTOR_DECISION_SCHEMA:
        raise ValueError("selector metadata schema mismatch")
    config_digest = _require_sha256(
        raw.get("selector_config_sha256"), field="selector_config_sha256"
    )
    if config_digest != SELECTOR_CONFIG_SHA256:
        raise ValueError("selector_config_sha256 does not match production config")
    evidence_digest = _require_sha256(
        raw.get("target_evidence_sha256"), field="target_evidence_sha256"
    )

    try:
        decision = SelectorDecision(
            requested_strategy=raw.get("requested_strategy"),
            effective_strategy=raw.get("effective_strategy"),
            selected_indices=tuple(raw.get("selected_indices") or ()),
            control_indices=tuple(raw.get("control_indices") or ()),
            candidate_indices=tuple(raw.get("candidate_indices") or ()),
            used_control_fallback=raw.get("used_control_fallback"),
            fallback_reason=raw.get("fallback_reason"),
            total_windows=raw.get("total_windows"),
            max_windows=raw.get("max_windows"),
            target_coverage_tokens=raw.get("target_coverage_tokens"),
            control_target_coverage_tokens=raw.get("control_target_coverage_tokens"),
            candidate_target_coverage_tokens=raw.get("candidate_target_coverage_tokens"),
            retained_tokens=raw.get("retained_tokens"),
            control_retained_tokens=raw.get("control_retained_tokens"),
            candidate_retained_tokens=raw.get("candidate_retained_tokens"),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid selector metadata: {exc}") from exc

    expected = _selector_mapping(
        decision,
        target_evidence_sha256=evidence_digest,
        selector_config_sha256=config_digest,
    )
    if raw != expected:
        raise ValueError("selector metadata is not canonical")


__all__ = [
    "BINDING_REPORT_SCHEMA",
    "CANDIDATE_ROOT_NAME",
    "MAX_WINDOWS",
    "PARENT_ACCEPTANCE_SCHEMA",
    "PARENT_BINDING_DIGEST_SHA256",
    "PARENT_DECISION_ID",
    "PARENT_EXTRACTOR_VERSION",
    "PARENT_GRAPH_SCHEMA_VERSION",
    "SELECTOR_CONFIG",
    "SELECTOR_CONFIG_SHA256",
    "SELECTOR_DECISION_SCHEMA",
    "STRIDE",
    "TARGET_EVIDENCE_SCHEMA",
    "TOKENIZER_NAME",
    "TOKEN_LINEAGE_ID",
    "WINDOW_SIZE",
    "build_selector_metadata",
    "build_target_evidence",
    "canonical_digest",
    "canonical_json_bytes",
    "graph_parent_authority",
    "validate_graph_parent_authority",
    "validate_selector_metadata",
    "validate_target_evidence",
]
