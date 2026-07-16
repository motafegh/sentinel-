"""Stable R0 acceptance rows and their canonical D2 finding mappings."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class MatrixRow:
    row_id: str
    invariant: str
    canonical_ids: tuple[str, ...]
    owner_package: str

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["canonical_ids"] = list(self.canonical_ids)
        return payload


MATRIX_ROWS: tuple[MatrixRow, ...] = (
    MatrixRow(
        row_id="R0-EVIDENCE-OUTAGE",
        invariant="ML/chain outage never becomes successful evidence",
        canonical_ids=("D2-AGT-001", "D2-AGT-012", "D2-AGT-016"),
        owner_package="R0.1",
    ),
    MatrixRow(
        row_id="R0-REPORT-CONTAINMENT",
        invariant="Report path cannot escape workspace",
        canonical_ids=("D2-AGT-002",),
        owner_package="R0.2",
    ),
    MatrixRow(
        row_id="R0-ARCHIVE-CONTAINMENT",
        invariant="Archive extraction cannot escape workspace",
        canonical_ids=("D2-DATA-001",),
        owner_package="R0.2",
    ),
    MatrixRow(
        row_id="R0-DATA-RELEASE-TRUST",
        invariant="Dataset release commitment binds semantics and exact files",
        canonical_ids=("D2-DATA-002",),
        owner_package="R0.5",
    ),
    MatrixRow(
        row_id="R0-AUTHORIZATION-LIMITS",
        invariant="Public mutation and expensive routes require auth/scope/quota",
        canonical_ids=("D2-X-001", "D2-AGT-011"),
        owner_package="R0.3",
    ),
    MatrixRow(
        row_id="R0-SIGNER-ISOLATION",
        invariant="Analysis process has no raw signing key",
        canonical_ids=("D2-X-001", "D2-ZKC-014"),
        owner_package="R0.3",
    ),
    MatrixRow(
        row_id="R0-PROOF-IDENTITY",
        invariant="Proof cannot support a cross-identity verified claim",
        canonical_ids=("D2-ZKC-001", "D2-ZKC-002"),
        owner_package="R0.4",
    ),
    MatrixRow(
        row_id="R0-TRANSACTION-TRUTH",
        invariant="Failed/reverted transaction cannot be reported submitted",
        canonical_ids=("D2-ZKC-003",),
        owner_package="R0.4",
    ),
)

MATRIX_ROW_IDS = frozenset(row.row_id for row in MATRIX_ROWS)


def matrix_manifest() -> dict[str, object]:
    return {
        "schema_version": "1",
        "kind": "r0_matrix_manifest",
        "rows": [row.to_dict() for row in MATRIX_ROWS],
    }


__all__ = ["MATRIX_ROWS", "MATRIX_ROW_IDS", "MatrixRow", "matrix_manifest"]
