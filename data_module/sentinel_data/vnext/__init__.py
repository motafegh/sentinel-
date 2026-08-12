"""SENTINEL DATA vNext v2 semantic overlay.

This package implements the R4 Phase-5 policy and Phase-6 frozen roles without
mutating the historical v1 label/export pipeline.
"""

from .builder import build_vnext_overlay
from .loader import VNextExport
from .representations import verify_local_representations
from .validator import validate_vnext_overlay

__all__ = [
    "VNextExport",
    "build_vnext_overlay",
    "validate_vnext_overlay",
    "verify_local_representations",
]
