"""sentinel_data.splitting — Stage 5 submodule.

Implements the splitting pipeline:
  - 4 splitter strategies (random, stratified, project, temporal)
  - dedup_enforcer (BCCC-failure pattern fix)
  - leakage_auditor (post-split safety net)
  - NonVulnerable 3:1 cap (friend review)
  - split_manifest (versioned JSON contract)

See plan 06_stage_5_splitting_registry.md for the full design rationale.
"""
from sentinel_data.splitting.splitters import (
    DEFAULT_RATIOS,
    Contract,
    SPLITTERS,
    SplitMetadata,
    SplitName,
    Splits,
    apply_strategy,
    load_splits,
    random_split,
    stratified_split,
    project_split,
    temporal_split,
    write_manifest,
    write_splits,
)
from sentinel_data.splitting.dedup_enforcer import apply_dedup_enforcer
from sentinel_data.splitting.leakage_auditor import (
    DEFAULT_TEXT_SIMILARITY_THRESHOLD,
    LeakPair,
    LeakageReport,
    find_leaks,
    run_audit,
)
from sentinel_data.splitting.nonvulnerable_cap import (
    DEFAULT_CAP,
    apply_nonvulnerable_cap,
)
