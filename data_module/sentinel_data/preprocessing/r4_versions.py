"""Version identifiers for the Phase-8 real-data repair lineage.

These identifiers describe *interfaces and intended output paths*. They do not
claim that the protected local corpus has been rebuilt or accepted. Historical
v1 artifacts remain immutable.
"""

from __future__ import annotations

PREPROCESSING_ARTIFACT_VERSION = "sentinel-preprocessed-r4-v2"
PREPROCESSING_META_SCHEMA_VERSION = "2"
PROVENANCE_SCHEMA_VERSION = "r4-provenance-v1"
GROUPING_VERSION = "r4-leakage-groups-v2"
REPAIRED_DATA_PUBLICATION_ID = "sentinel-r4-vnext-v2"
REPAIRED_ROLE_PARTITION_ID = "r4-vnext-roles-v2"
REPAIRED_EVIDENCE_LEDGER_ID = "evidence-ledger-r4-v2"
REPAIRED_REPRESENTATION_EXTRACTOR_VERSION = "v2.2-r4-repaired"

# R4-D-010 repository candidate. These identifiers do not claim physical
# generation or acceptance; accepted v9 constants below remain unchanged.
# v2.6 keeps graph schema v10 and its call vocabulary unchanged while extending
# the deterministic CFG storage-WRITE seam to Solidity's storage-mutating
# collection built-ins (array/bytes ``push`` and ``pop``).  V2.5 artifacts stay
# immutable; this identifier applies only to fresh candidate generation.
V10_GRAPH_SCHEMA_VERSION = "v10"
V10_REPRESENTATION_EXTRACTOR_VERSION = (
    "v2.6-r4-call-semantics-deterministic-cfg-mutators"
)
V10_REPRESENTATION_ROOT_NAME = "representations-r4-v3-candidate"
V10_PRIMARY_SLITHER_VERSION = "0.10.0"
V10_SLITHER_RUNTIME_EXCEPTIONS = {
    # Slither 0.10.0 fails full analysis for this exact accepted source, while
    # 0.11.5 succeeds. Applying 0.11.5 population-wide is prohibited because
    # it materially changes otherwise-frozen node and edge structure.
    "caa35c1a5906269bbe5e70de780d105c2968ece4fc038d7f7208efee681aeec9": "0.11.5",
}

# Graph node/feature semantics and model tensor shape are intentionally frozen.
GRAPH_SCHEMA_VERSION = "v9"
TOKEN_TENSOR_SHAPE = (4, 512)


def preprocessed_root_name() -> str:
    """Return the versioned directory name used under ``data_module/data``."""

    return PREPROCESSING_ARTIFACT_VERSION


def representation_root_name() -> str:
    """Return the repaired representation root name used under ``data``."""

    return "representations-r4-v2"
