"""Version identifiers for the corrected Phase-8 logical DATA lineage.

The physical repaired corpus and representation triples remain the accepted v2
artifacts.  V3 changes only leakage grouping, role assignment, publication, and
research/evaluation reservations derived from those logical boundaries.
"""

from __future__ import annotations

GROUPING_VERSION_V3 = "r4-leakage-groups-v3"
ROLE_PARTITION_VERSION_V3 = "r4-vnext-roles-v3"
DATASET_VERSION_V3 = "sentinel-r4-vnext-v3"
LOGICAL_BUILD_VERSION_V3 = "r4-logical-lineage-v3"

# V3 deliberately reuses the already accepted role-independent semantic ledger
# and physical representation population.  No new negative truth is introduced.
SOURCE_EVIDENCE_LEDGER_VERSION = "evidence-ledger-r4-v2"
PHYSICAL_PREPROCESSING_VERSION = "sentinel-preprocessed-r4-v2"
PHYSICAL_REPRESENTATION_ROOT_VERSION = "representations-r4-v2"
PHYSICAL_REPRESENTATION_EXTRACTOR_VERSION = "v2.2-r4-repaired"
