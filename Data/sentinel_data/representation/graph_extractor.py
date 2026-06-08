"""Graph extractor stub — real implementation ported from ml/ in Stage 2.

Stage 0 role: provide importable symbols so the package installs and tests pass.
Stage 2 role: full port from ml/src/preprocessing/graph_extractor.py with a
              byte-identical regression test gating every line.
Stage 7 role: seam swap — sentinel-ml switches its import to this path; ml/ copy deleted.

IMPORTANT — 8 bugs that are already fixed in ml/ and must be preserved in Stage 2:
  A9  — now keyword miss in _compute_uses_block_globals (graph_extractor.py:587-605)
  A15 — def_map keyed by name instead of id() (graph_extractor.py:1147-1179)
  A20 — label=0 hardcode in ast_extractor.py batch extraction (ast_extractor.py:290)
  A34 — prefix sort uses wrong dimension (sentinel_model.py:356)
  A38 — NaN check ran after backward() instead of before (trainer.py)
  resume_overwrite — resume defaulted to model-only (trainer.py:383)
  return_ignored — always returned 0.0 due to lvalue identity check (graph_extractor.py)
  EMITS — edge bug (open, fix in Stage 7)

DO NOT re-fix any of the above — the Stage 2 regression test guards them.
"""

from typing import Any


class GraphExtractionError(Exception):
    """Raised when contract graph extraction fails."""


def extract_contract_graph(
    sol_path: str,
    label_vec: list[int] | None = None,
    schema_version: str = "v9",
    **kwargs: Any,
) -> Any:
    """Extract a PyG Data object from a Solidity source file.

    NOT IMPLEMENTED in Stage 0 — raises NotImplementedError.
    Real implementation is ported from ml/src/preprocessing/graph_extractor.py in Stage 2.

    Args:
        sol_path: Absolute path to the .sol file.
        label_vec: 10-element binary label vector (class order per CLASS_NAMES in graph_schema.py).
        schema_version: Must be "v9" — other values are rejected.
        **kwargs: Reserved for Stage 2 extractor options.

    Returns:
        torch_geometric.data.Data with x, edge_index, edge_attr, y, contract_path attributes.

    Raises:
        NotImplementedError: Always — implement in Stage 2.
        GraphExtractionError: For malformed inputs (raised in Stage 2+).
    """
    raise NotImplementedError(
        "extract_contract_graph() is a Stage 0 stub. "
        "The real implementation is ported from ml/src/preprocessing/graph_extractor.py "
        "in Stage 2 (docs/proposal/Data_Module_Proposals/actionable_plans/03_stage_2_representation.md)."
    )
