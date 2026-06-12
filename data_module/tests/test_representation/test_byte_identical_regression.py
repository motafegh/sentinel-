"""Byte-identical regression for Stage 2 thin-adapter port (Task 2.6).

Both import paths resolve to the same function object (thin adapter pattern),
so byte-identical output is guaranteed by construction. This suite serves as:
  1. Integration smoke: extraction works end-to-end on real SolidiFI contracts.
  2. Schema-dim gate: every graph has x.shape[-1] == NODE_FEATURE_DIM (12).
  3. Edge validity gate: all edge_attr values in [0, NUM_EDGE_TYPES - 1] = [0, 11].
  4. Forward regression guard: if the thin adapter is replaced with a code copy,
     any extraction divergence will be caught here.

Fixtures: the 10 smallest SolidiFI preprocessed contracts (by file size), which
are all 0.5.x era. Tests skip individually if the required solc binary is not
installed in ~/.solc-select/artifacts/.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
PREP_DIR  = REPO_ROOT / "data_module" / "data" / "preprocessed" / "solidifi"
SOLC_ROOT = Path.home() / ".solc-select" / "artifacts"


def _solc_bin(version: str) -> Path | None:
    p = SOLC_ROOT / f"solc-{version}" / f"solc-{version}"
    return p if p.exists() else None


def _collect_fixtures(n: int = 10) -> list[tuple[Path, dict]]:
    """Return the n smallest preprocessed SolidiFI (sol, meta) pairs."""
    if not PREP_DIR.exists():
        return []
    metas = sorted(PREP_DIR.glob("*.meta.json"), key=lambda p: p.stat().st_size)
    result = []
    for m in metas:
        # Stage 7B fix: m has suffix .meta.json; with_suffix(".sol") would give
        # <hash>.meta.sol (only replaces .json). Use replace to drop .meta.json.
        sol = m.with_name(m.name.replace(".meta.json", ".sol"))
        if not sol.exists():
            continue
        meta = json.loads(m.read_text())
        result.append((sol, meta))
        if len(result) == n:
            break
    return result


FIXTURES = _collect_fixtures(10)
FIXTURE_IDS = [f[0].stem[:16] for f in FIXTURES] if FIXTURES else ["no-fixtures"]


@pytest.mark.skipif(not FIXTURES, reason="SolidiFI preprocessed data not found")
@pytest.mark.parametrize("sol_path,meta", FIXTURES, ids=FIXTURE_IDS)
class TestByteIdentical:
    """Each test exercises one SolidiFI contract; skips if solc unavailable."""

    def _config(self, meta: dict):
        from ml.src.preprocessing.graph_extractor import GraphExtractionConfig
        version = meta.get("solc_version", "")
        solc = _solc_bin(version)
        if solc is None:
            pytest.skip(f"solc-{version} not in {SOLC_ROOT}")
        return GraphExtractionConfig(
            solc_binary=solc,
            solc_version=version,
            allow_paths=[str(REPO_ROOT)],
        )

    def test_old_and_new_path_are_same_object(self, sol_path, meta):
        from ml.src.preprocessing.graph_extractor import extract_contract_graph as old
        from sentinel_data.representation.graph_extractor import extract_contract_graph as new
        assert old is new, "thin adapter must re-export the SAME function object"

    def test_schema_dim_gate(self, sol_path, meta):
        """x.shape[-1] == NODE_FEATURE_DIM (12) — the schema-dim gate."""
        from sentinel_data.representation import extract_contract_graph, NODE_FEATURE_DIM
        cfg = self._config(meta)
        data = extract_contract_graph(str(sol_path), config=cfg)
        assert data.x.shape[-1] == NODE_FEATURE_DIM, (
            f"schema-dim gate: expected {NODE_FEATURE_DIM} features, "
            f"got {data.x.shape[-1]} in {sol_path.name}"
        )

    def test_edge_attr_within_valid_range(self, sol_path, meta):
        """All edge types are in [0, NUM_EDGE_TYPES - 1]."""
        from sentinel_data.representation import extract_contract_graph, NUM_EDGE_TYPES
        cfg = self._config(meta)
        data = extract_contract_graph(str(sol_path), config=cfg)
        if data.edge_attr is not None and data.edge_attr.numel() > 0:
            assert data.edge_attr.min() >= 0
            assert data.edge_attr.max() <= NUM_EDGE_TYPES - 1, (
                f"edge type {data.edge_attr.max()} out of range "
                f"[0, {NUM_EDGE_TYPES - 1}] in {sol_path.name}"
            )

    def test_byte_identical_old_vs_new(self, sol_path, meta):
        """Old and new paths produce identical tensor fields (trivially true for thin adapter)."""
        from ml.src.preprocessing.graph_extractor import extract_contract_graph as old_fn
        from sentinel_data.representation.graph_extractor import extract_contract_graph as new_fn
        cfg = self._config(meta)
        old_data = old_fn(str(sol_path), config=cfg)
        new_data = new_fn(str(sol_path), config=cfg)
        assert torch.equal(old_data.x, new_data.x), "x differs"
        assert torch.equal(old_data.edge_index, new_data.edge_index), "edge_index differs"
        if old_data.edge_attr is not None:
            assert torch.equal(old_data.edge_attr, new_data.edge_attr), "edge_attr differs"
        assert old_data.num_nodes == new_data.num_nodes
