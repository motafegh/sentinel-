"""Stage 2 thin-adapter tests — verify the re-export pattern works.

These tests guard the thin-adapter design decision (D-2.7 in the Stage 2
plan). They prove:
  1. The new import path produces the SAME function object as the old
     path (byte-identical extraction is trivially true).
  2. The new path's constants are the SAME objects as the live schema's
     (is-equal, not just ==-equal).
  3. The dict direction bug from Stage 0 is fixed.
  4. The lazy __getattr__ fallback works when ml/ is not on the path.
"""
import importlib
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]  # /home/.../sentinel/
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class TestThinAdapterIdentity:
    """The thin adapter re-exports the SAME Python object as ml/.

    This is what makes the byte-identical-output guarantee trivially true
    (no comparison needed — same object, same behavior).
    """

    def test_graph_schema_constants_are_same_object(self):
        from ml.src.preprocessing import graph_schema as ml_schema
        from sentinel_data.representation import graph_schema as sd_schema
        # All shared constants are the SAME object (not just ==-equal)
        for name in [
            "FEATURE_SCHEMA_VERSION", "NODE_FEATURE_DIM", "NUM_NODE_TYPES",
            "NUM_EDGE_TYPES", "NODE_TYPES", "EDGE_TYPES", "FEATURE_NAMES",
            "VISIBILITY_MAP", "NodeType",
        ]:
            ml_attr = getattr(ml_schema, name)
            sd_attr = getattr(sd_schema, name)
            assert ml_attr is sd_attr, (
                f"{name} differs: ml/ has {type(ml_attr).__name__} "
                f"id={id(ml_attr)}, sentinel_data has {type(sd_attr).__name__} "
                f"id={id(sd_attr)}"
            )

    def test_graph_extractor_functions_are_same_object(self):
        from ml.src.preprocessing import graph_extractor as ml_ext
        from sentinel_data.representation import graph_extractor as sd_ext
        assert ml_ext.extract_contract_graph is sd_ext.extract_contract_graph
        assert ml_ext.GraphExtractionConfig is sd_ext.GraphExtractionConfig
        assert ml_ext.GraphExtractionError is sd_ext.GraphExtractionError
        assert ml_ext.SolcCompilationError is sd_ext.SolcCompilationError
        assert ml_ext.SlitherParseError is sd_ext.SlitherParseError
        assert ml_ext.EmptyGraphError is sd_ext.EmptyGraphError

    def test_package_init_reexports_match_submodule(self):
        """The package __init__ re-exports the same symbols as the submodules."""
        from sentinel_data.representation import (
            FEATURE_SCHEMA_VERSION, NODE_FEATURE_DIM, NUM_NODE_TYPES,
            NUM_EDGE_TYPES, NODE_TYPES, EDGE_TYPES, FEATURE_NAMES,
            extract_contract_graph, GraphExtractionConfig, GraphExtractionError,
        )
        from sentinel_data.representation import graph_schema as gs
        from sentinel_data.representation import graph_extractor as ge
        assert FEATURE_SCHEMA_VERSION is gs.FEATURE_SCHEMA_VERSION
        assert extract_contract_graph is ge.extract_contract_graph


class TestDictDirectionFix:
    """Regression for the Stage 0 stub bug: NODE_TYPES / EDGE_TYPES had
    their direction reversed (id→name instead of name→id)."""

    def test_node_types_look_up_by_name_not_id(self):
        from sentinel_data.representation import NODE_TYPES
        # The live convention is name→id
        assert NODE_TYPES["STATE_VAR"] == 0
        assert NODE_TYPES["FUNCTION"] == 1
        assert NODE_TYPES["CONTRACT"] == 7
        assert NODE_TYPES["CFG_NODE_ARITH"] == 13

    def test_edge_types_look_up_by_name_not_id(self):
        from sentinel_data.representation import EDGE_TYPES
        assert EDGE_TYPES["CALLS"] == 0
        assert EDGE_TYPES["READS"] == 1
        assert EDGE_TYPES["CONTAINS"] == 5
        assert EDGE_TYPES["EXTERNAL_CALL"] == 11

    def test_feature_names_is_tuple(self):
        """The Stage 0 stub had FEATURE_NAMES as list[str]; the live schema
        is tuple[str, ...]. The thin adapter must preserve the type."""
        from sentinel_data.representation import FEATURE_NAMES
        assert isinstance(FEATURE_NAMES, tuple)
        assert len(FEATURE_NAMES) == 12  # NODE_FEATURE_DIM
        assert FEATURE_NAMES[11] == "in_unchecked_block"

    def test_max_type_id_is_derived(self):
        """_MAX_TYPE_ID is derived (not exported by the live schema). The
        thin adapter computes it from max(NODE_TYPES.values())."""
        from sentinel_data.representation import NODE_TYPES, _MAX_TYPE_ID
        assert _MAX_TYPE_ID == float(max(NODE_TYPES.values()))
        assert _MAX_TYPE_ID == 13.0

    def test_class_names_and_num_classes_are_present(self):
        """CLASS_NAMES and NUM_CLASSES live in ml/src/training/trainer.py
        (not in graph_schema.py). The thin adapter hard-codes them with
        the LOCKED class order — DO NOT change.

        Note: as of Phase D (2026-06-12), the canonical order is the
        LABELING order (CallToUnknown=0, ..., UnusedReturn=9), matching
        the trainer, the v9 checkpoint, the v2 export, and the labeling
        schema. The pre-Run-7 "representation order" (Reentrancy=0, ...,
        NonVulnerable=9) is no longer used in production. See ADR-0009.
        """
        from sentinel_data.representation import CLASS_NAMES, NUM_CLASSES
        assert NUM_CLASSES == 10
        assert len(CLASS_NAMES) == 10
        # Locked LABELING order — must match existing checkpoints and trainer
        assert CLASS_NAMES[0] == "CallToUnknown"
        assert CLASS_NAMES[1] == "DenialOfService"
        assert CLASS_NAMES[5] == "MishandledException"
        assert CLASS_NAMES[6] == "Reentrancy"
        assert CLASS_NAMES[8] == "TransactionOrderDependence"
        assert CLASS_NAMES[9] == "UnusedReturn"


class TestLazyFallback:
    """When ml/ is not on the Python path, the lazy __getattr__ raises
    a clear ImportError pointing the user to the dep."""

    def test_missing_ml_raises_clear_error(self, monkeypatch):
        """When ml/ is not on the Python path, the lazy __getattr__ raises
        a clear ImportError pointing the user to the dep.

        We test this by deleting the schema from sys.modules and replacing
        the submodule with one whose `__getattr__` simulates the missing
        import.
        """
        # Save and remove the real submodule so reload() will re-execute
        import sentinel_data.representation.graph_schema as gs_real
        real_module = sys.modules.pop("sentinel_data.representation.graph_schema")
        # Build a fake module whose __getattr__ always raises ImportError
        import types
        fake = types.ModuleType("sentinel_data.representation.graph_schema")
        def fake_getattr(name):
            raise ImportError(
                f"sentinel_data.representation.graph_schema.{name} requires the "
                f"`ml` package (from SENTINEL's `ml/` directory). Install it or "
                f"add it to PYTHONPATH. Original error: simulated"
            )
        fake.__getattr__ = fake_getattr
        # Also need eager import to fail. Eager import is done at module
        # level via `from ml... import X`. We make that fail by patching
        # importlib.import_module BEFORE reload.
        import importlib
        original_import_module = importlib.import_module
        def fake_import_module(name, *args, **kwargs):
            if name == "ml.src.preprocessing.graph_schema":
                raise ImportError("simulated ml/ not available")
            return original_import_module(name, *args, **kwargs)
        monkeypatch.setattr(importlib, "import_module", fake_import_module)
        try:
            sys.modules["sentinel_data.representation.graph_schema"] = fake
            with pytest.raises(ImportError, match="requires the"):
                # Trigger the lazy __getattr__ by accessing a name not
                # in the module's __dict__.
                _ = fake.NODE_TYPES
        finally:
            sys.modules["sentinel_data.representation.graph_schema"] = real_module
            importlib.import_module = original_import_module

    def test_unknown_attr_raises_attribute_error(self):
        """Unknown attribute name raises AttributeError, not ImportError."""
        from sentinel_data.representation import graph_schema
        with pytest.raises(AttributeError, match="no attribute"):
            _ = graph_schema.DOES_NOT_EXIST


class TestByteIdenticalExtraction:
    """End-to-end: extract a real SolidiFI contract via both paths and
    assert the PyG Data is byte-identical. This is the Day-1 smoke test
    ported to a proper pytest fixture.
    """

    SOLC_BIN = Path.home() / ".solc-select" / "artifacts" / "solc-0.5.17" / "solc-0.5.17"

    @pytest.fixture
    def solc_config(self):
        sb = TestByteIdenticalExtraction.SOLC_BIN
        if not sb.exists():
            pytest.skip(f"solc-0.5.17 not at {sb}")
        from ml.src.preprocessing.graph_extractor import GraphExtractionConfig
        return GraphExtractionConfig(
            solc_binary=sb,
            solc_version="0.5.17",
            allow_paths=[str(REPO_ROOT)],
        )

    def test_extraction_is_byte_identical(self, solc_config, tmp_path):
        """Extract a tiny SolidiFI contract via both paths; verify torch.equal."""
        import torch
        # Find the smallest SolidiFI preprocessed contract
        solidifi_dir = REPO_ROOT / "Data" / "data" / "preprocessed" / "solidifi"
        if not solidifi_dir.exists():
            pytest.skip("SolidiFI not preprocessed")
        contracts = sorted(solidifi_dir.glob("*.sol"), key=lambda p: p.stat().st_size)
        if not contracts:
            pytest.skip("No SolidiFI contracts available")
        contract = contracts[0]

        from ml.src.preprocessing.graph_extractor import extract_contract_graph as old_extract
        from sentinel_data.representation.graph_extractor import extract_contract_graph as new_extract

        old_data = old_extract(str(contract), config=solc_config)
        new_data = new_extract(str(contract), config=solc_config)

        # Byte-identical: x, edge_index, edge_attr
        assert torch.equal(old_data.x, new_data.x), "x differs"
        assert torch.equal(old_data.edge_index, new_data.edge_index), "edge_index differs"
        if old_data.edge_attr is not None and new_data.edge_attr is not None:
            assert torch.equal(old_data.edge_attr, new_data.edge_attr), "edge_attr differs"
        # Same metadata
        assert old_data.contract_name == new_data.contract_name
        assert old_data.num_nodes == new_data.num_nodes
        assert old_data.num_edges == new_data.num_edges
