"""sentinel_data.registry — Stage 5 submodule.

Implements the registry (catalog + lineage + hasher + dataset_diff +
changelog). See plan 06_stage_5_splitting_registry.md for the
full design rationale.

The registry is the single source of truth for "what dataset version
is the v2 baseline?" Every artifact is content-addressed (SHA-256);
the YAML mirror is for version control.
"""
from sentinel_data.registry.catalog import (
    Artifact, Catalog, DatasetVersion, Migration, Retirement,
    Source, SplitRecord, compute_dict_hash, compute_hash,
)
from sentinel_data.registry.lineage_tracker import (
    hash_artifact, hash_lineage, lineage_to_dot, record_lineage_step,
    record_training_run, verify_artifact,
)
from sentinel_data.registry.dataset_diff import (
    DatasetDiff, PerClassMetric, diff_dataset_versions, update_changelog,
)
