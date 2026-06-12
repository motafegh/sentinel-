# `sentinel_data.registry` — Stage 5: The Versioned Artifact Catalog

> **Status: ✅ Fully implemented (3 files, 800 lines).** SQLite + YAML mirror + lineage tracker + dataset diff. The "artifact_hasher.py" file mentioned in the previous README doesn't exist — the equivalent lives as a thin wrapper in `lineage_tracker.py` (functions `hash_artifact`, `verify_artifact`).

## 1. Purpose

The registry is **the "library card catalog" for your data.** It maintains a SQLite + YAML mirror of every artifact produced by the pipeline — sources, representations, splits, exports, and dataset versions. Every artifact is content-addressed (SHA-256); the YAML mirror is for version control.

Six months from now, when someone asks "what dataset did Run 11 train on?" the answer is in the registry. Without it:

- You can't answer "what exact version of the dataset did this model use?"
- You can't verify that an export file hasn't been tampered with
- You can't compare two dataset versions to see what changed
- You can't trace an artifact's lineage back to its source commits

The BCCC dataset had none of this — the v1.4 labels, v8 graphs, v9 graphs, and v10 graphs were scattered across directories with no version tracking. The registry prevents that class of failure.

## 2. Source map

| File | Lines | Role |
|------|-------|------|
| `__init__.py` | 21 | Re-exports the public API: `Artifact`, `Catalog`, `DatasetVersion`, `Migration`, `Retirement`, `Source`, `SplitRecord`, `compute_dict_hash`, `compute_hash`, `hash_artifact`, `hash_lineage`, `lineage_to_dot`, `record_lineage_step`, `record_training_run`, `verify_artifact`, `DatasetDiff`, `PerClassMetric`, `diff_dataset_versions`, `update_changelog`. |
| `catalog.py` | 541 | The `Catalog` class — SQLite at `data/registry/catalog.db` + YAML mirror at `data/registry/catalog.yaml`. 4 base tables + 2 system tables. |
| `dataset_diff.py` | 161 | `diff_dataset_versions(metadata_old, metadata_new, ...) -> DatasetDiff` + `update_changelog(path, version_name, summary, metrics) -> None`. |
| `lineage_tracker.py` | 98 | `record_lineage_step(...)`, `record_training_run(...)`, `lineage_to_dot(lineage) -> str` (Graphviz), `hash_artifact(path) -> str` (alias for `catalog.compute_hash`), `verify_artifact(path, expected_hash) -> bool` (load-time gate). |

> **Note:** The previous README mentioned a separate `artifact_hasher.py` — that file does NOT exist. The artifact-hashing functions live as thin wrappers in `lineage_tracker.py` (lines 55-58 for `hash_artifact`, lines 85-97 for `verify_artifact`). Both call `catalog.compute_hash` as the canonical implementation (per AUDIT_PATCHES 5-P4).

**Sub-total: 800 lines** across 3 files.

## 3. Key concepts

### The 6-table schema (`catalog.py:210-288`)

| Table | Purpose | Type |
|-------|---------|------|
| `sources` | Per-source pin + last-fetched timestamp + enabled + n_contracts + tier + metadata | Base |
| `artifacts` | Per-exported-artifact name + sha256 + size_bytes + lineage (JSON) + created_at | Base |
| `splits` | Per-split-version version + seed + strategy + contract_counts (JSON) + metadata_hash + created_at | Base |
| `dataset_versions` | Named composite: source_set (JSON) + preprocessing_config_hash + split_version + label_schema_version + export_format + generated_at + verification_report_path + artifact_hash + artifact_path + metadata (JSON) | Base |
| `schema_migrations` | version (PRIMARY KEY) + description + applied_at | System |
| `dataset_version_retirements` | name (PRIMARY KEY) + superseded_by + retired_at + reason | System |

Schema migration policy is **append-only forward-only** — each `migrate()` call records the version. No down-migrations. The catalog is initialized with `SCHEMA_VERSION = 1` ("Initial schema: 4 base tables + 2 system tables").

### The shared hash function (`catalog.py:42-59`)

```python
def compute_hash(path: Path) -> str:
    """SHA-256 of the file bytes. Used by both the catalog and (post-Stage-7)
    the ML module's InferenceCache."""

def compute_dict_hash(d: dict) -> str:
    """Stable SHA-256 of a dict's canonical JSON form (sorted keys)."""
```

`compute_hash` is the **canonical content-addressing function**. Used by:
- The `Catalog.add_artifact()` flow (line 326-333) for `artifacts.sha256`
- The `Catalog.verify_artifact_hash()` flow (line 447-473) for the load-time gate
- The `lineage_tracker.hash_artifact()` / `verify_artifact()` thin wrappers
- (Post-Stage-7) the ML module's `InferenceCache` (per AUDIT_PATCHES 5-P4)

For large files (>1 GB) consider a chunked read — for the v2 baseline, files are <100 MB, so the current implementation is fine.

### The `Catalog` class — SQLite + YAML mirror (`catalog.py:168-541`)

**The core data structure**. Opens or creates `data/registry/catalog.db` with auto-commit, foreign keys ON, and row_factory=sqlite3.Row.

```python
class Catalog:
    SCHEMA_VERSION = 1
    
    def __init__(self, db_path: Path, yaml_mirror_path: Optional[Path] = None):
        """Open or create the catalog database.
        
        If yaml_mirror_path is provided, the catalog validates the YAML on
        init (warns on invalid format) and writes a new YAML mirror on
        every write.
        """
```

**Public API** (one CRUD per table):
- `add_source(source)`, `get_source(name)`, `list_sources()` — line 292-322
- `add_artifact(artifact)`, `get_artifact(name)` — line 326-343
- `add_split(split)`, `get_split(version)` — line 347-366
- `add_dataset_version(version)`, `get_dataset_version(name)`, `list_dataset_versions(include_retired=False)`, `retire_dataset_version(name, superseded_by, reason="")`, `is_retired(name)` — line 370-432

**Plus the ML-module-facing methods**:
- `load_artifact(name) -> Optional[DatasetVersion]` — line 436-445. The ML module's `SentinelDataset.__init__` calls this. Returns None if retired or missing.
- `verify_artifact_hash(path: Path) -> bool` — line 447-473. Checks BOTH the `artifacts` table and the `dataset_versions` table; returns True if EITHER matches.
- `write_yaml_mirror() -> None` — line 477-521. Exports the full catalog to YAML.
- `applied_migrations() -> list[Migration]` + `migrate(version, description)` — line 525-541. Records schema changes.

### The retirement chain (`catalog.py:419-432`)

Old dataset versions are **never deleted** — they're marked as "superseded" with a pointer to their replacement. The chain is a forensic record: you can always reconstruct what v1.4 looked like, even though it's been superseded by v10.

```
v1.4 BCCC labels → superseded_by: v8 BCCC graphs
v8 BCCC graphs   → superseded_by: v9 graphs
v9 graphs        → superseded_by: v10 deduped
v10 deduped      → superseded_by: sentinel-v2-gold-2026-08
```

`retire_dataset_version(name, superseded_by, reason="")` records the retirement. `load_artifact(name)` refuses to load retired versions (logs a warning, returns None).

### The YAML mirror (`catalog.py:477-521`)

Human-readable export, one YAML document per table:

```yaml
---
kind: sources
items:
  - name: solidifi
    pin: ""
    enabled: 1
    n_contracts: 283
    tier: "T0"
    ...
---
kind: artifacts
items: [...]
---
kind: dataset_versions
items: [...]
---
kind: retirements
items: [...]
---
kind: schema_migrations
items:
  - version: 1
    description: "Initial schema: 4 base tables + 2 system tables"
    applied_at: "2026-06-08T..."
```

JSON-stringified dicts (`source_set`, `metadata`, `contract_counts`, `lineage`) are parsed back to dicts for readability before writing.

**CI invariant**: the SQLite DB and YAML mirror are supposed to stay in sync. The catalog's `__init__` validates the YAML on load (line 191-196); the CLI calls `cat.write_yaml_mirror()` after every write.

### The dataset diff (`dataset_diff.py:76-142`)

Takes two dataset version metadata dicts and reports:

- **added_contracts** / **removed_contracts** (by sha256)
- **common_contracts** (intersection)
- **label_changes** (per-class value changes for common contracts)
- **per_class** — per-class count delta + tier breakdown + coarse F1 projection

```python
def diff_dataset_versions(
    metadata_old: dict, metadata_new: dict,
    name_old: str = "old", name_new: str = "new",
) -> DatasetDiff:
    """Compute the diff between two dataset versions.
    
    Inputs are the `metadata` dict of each DatasetVersion (which contains
    the per-contract labels and tier info). For large corpora, this is
    an O(N) comparison.
    """
```

The **per-class F1 projection** (per AUDIT_PATCHES 5-P7) is a coarse heuristic: `min(5.0, abs(delta_pct) * 0.1)` — more positives → likely higher recall (assuming the model was recall-bound on this class). Capped at 5% F1 delta. The model team uses this to predict "Run 11's per-class F1 will likely be X% better than Run 9's."

### The changelog (`dataset_diff.py:145-161`)

`update_changelog(path, version_name, summary, metrics)` appends a new entry to `data/changelog.md`:

```markdown
## sentinel-v2-gold-2026-08 (2026-06-08T...)

Removed 2,488 contracts that failed Phase 5 retry with solc 0.7.4 installed.
Re-classified 15,340 contracts based on the semantic_checker FAIL signal.

### Per-class metric projection

| Class | Old | New | Δ Count | Δ % |
|---|---|---|---|---|
| Reentrancy | 1699 | 1822 | +123 | +7.2% |
| CallToUnknown | 239 | 250 | +11 | +4.6% |
| ...
```

### The lineage tracker (`lineage_tracker.py`)

Lineage is the **audit trail** for every artifact: which ingestion connector, which preprocessing step, which labeling parser, which verification component, which splitter, which export writer produced it. Stored as a JSON field on the artifact.

```python
def record_lineage_step(lineage: dict, step: str, **details) -> dict:
    """Append a step to a lineage dict. Returns the updated dict."""

def record_training_run(lineage: dict, run_name: str, dataset_version: str, **details) -> dict:
    """Record that a training run consumed a dataset version (per AUDIT_PATCHES 5-P6)."""

def lineage_to_dot(lineage: dict) -> str:
    """Render lineage as Graphviz DOT for visualization."""

def hash_artifact(path: Path) -> str:
    """Alias for catalog.compute_hash."""

def verify_artifact(path: Path, expected_hash: str) -> bool:
    """The load-time gate. ML module's SentinelDataset.__init__ calls this."""
```

The lineage dict has two keys: `steps` (list of `{step, ts, ...details}`) and `parents` (list of sha256s of input artifacts). Each transformation appends one step. The training_run step additionally records the model-side outcome (per AUDIT_PATCHES 5-P6: the training-run link answers "what data did Run 11 train on, and how did it differ from Run 10?").

## 4. Public API

### The dataclasses — `catalog.py`

```python
@dataclass
class Source:
    name: str
    pin: str = ""
    last_fetched: str = ""
    enabled: bool = True
    n_contracts: int = 0
    tier: str = ""
    metadata: dict = field(default_factory=dict)

@dataclass
class Artifact:
    name: str                    # unique name (e.g., "preprocessed/dive/sha_abc.sol")
    sha256: str                  # content hash
    size_bytes: int = 0
    lineage: dict = field(default_factory=dict)
    created_at: str = field(default_factory=...)

@dataclass
class SplitRecord:
    version: str                 # "v1", "v2", etc.
    seed: int = 42
    strategy: str = "stratified"
    contract_counts: dict = field(default_factory=dict)
    metadata_hash: str = ""
    created_at: str = field(default_factory=...)

@dataclass
class DatasetVersion:
    name: str                    # "sentinel-v2-gold-2026-08"
    source_set: list[str] = field(default_factory=list)
    preprocessing_config_hash: str = ""
    split_version: str = ""
    label_schema_version: str = "v1"
    export_format: str = "v1"
    generated_at: str = field(default_factory=...)
    verification_report_path: str = ""
    artifact_hash: str = ""
    artifact_path: str = ""
    metadata: dict = field(default_factory=dict)

@dataclass
class Migration:
    version: int
    description: str
    applied_at: str = field(default_factory=...)

@dataclass
class Retirement:
    name: str
    superseded_by: str
    retired_at: str = field(default_factory=...)
    reason: str = ""
```

### The `Catalog` class — `catalog.py:168-541`

```python
class Catalog:
    SCHEMA_VERSION = 1
    
    def __init__(self, db_path: Path, yaml_mirror_path: Optional[Path] = None): ...
    
    # Source API
    def add_source(self, source: Source) -> None: ...
    def get_source(self, name: str) -> Optional[Source]: ...
    def list_sources(self) -> list[Source]: ...
    
    # Artifact API
    def add_artifact(self, artifact: Artifact) -> None: ...
    def get_artifact(self, name: str) -> Optional[Artifact]: ...
    
    # Split API
    def add_split(self, split: SplitRecord) -> None: ...
    def get_split(self, version: str) -> Optional[SplitRecord]: ...
    
    # Dataset version API
    def add_dataset_version(self, version: DatasetVersion) -> None: ...
    def get_dataset_version(self, name: str) -> Optional[DatasetVersion]: ...
    def list_dataset_versions(self, include_retired: bool = False) -> list[DatasetVersion]: ...
    def retire_dataset_version(self, name: str, superseded_by: str, reason: str = "") -> None: ...
    def is_retired(self, name: str) -> bool: ...
    
    # ML-module-facing
    def load_artifact(self, name: str) -> Optional[DatasetVersion]: ...
    def verify_artifact_hash(self, path: Path) -> bool: ...
    
    # YAML mirror
    def write_yaml_mirror(self) -> None: ...
    
    # Migrations
    def applied_migrations(self) -> list[Migration]: ...
    def migrate(self, version: int, description: str) -> None: ...
```

### Dataset diff + changelog — `dataset_diff.py`

```python
@dataclass
class PerClassMetric:
    class_name: str
    count_old: int
    count_new: int
    delta_count: int
    delta_pct: float
    tier_breakdown_old: dict = field(default_factory=dict)
    tier_breakdown_new: dict = field(default_factory=dict)
    predicted_f1_delta_pct: float = 0.0

@dataclass
class DatasetDiff:
    name_old: str
    name_new: str
    added_contracts: list[str] = field(default_factory=list)
    removed_contracts: list[str] = field(default_factory=list)
    common_contracts: list[str] = field(default_factory=list)
    label_changes: list[dict] = field(default_factory=list)
    per_class: list[PerClassMetric] = field(default_factory=list)
    computed_at: str = ""

def diff_dataset_versions(
    metadata_old: dict, metadata_new: dict,
    name_old: str = "old", name_new: str = "new",
) -> DatasetDiff: ...

def update_changelog(changelog_path: Path, version_name: str, summary: str,
                    metrics: dict) -> None: ...
```

### Lineage tracker — `lineage_tracker.py`

```python
def record_lineage_step(lineage: dict, step: str, **details) -> dict: ...
def record_training_run(lineage: dict, run_name: str, dataset_version: str, **details) -> dict: ...
def lineage_to_dot(lineage: dict) -> str: ...
def hash_artifact(path: Path) -> str: ...
def hash_lineage(lineage: dict) -> str: ...
def verify_artifact(path: Path, expected_hash: str) -> bool: ...
```

### Shared hash — `catalog.py:42-59`

```python
def compute_hash(path: Path) -> str: ...
def compute_dict_hash(d: dict) -> str: ...
```

## 5. Inputs → outputs

| Input | Where | What |
|-------|-------|------|
| `data/registry/catalog.db` (auto-created) | `Catalog.__init__` | The SQLite DB |
| `data/registry/catalog.yaml` (optional) | `Catalog.__init__` | The YAML mirror |
| `data/splits/v<N>/split_manifest.json` | `cli.py:_run_register` | The split manifest (for cataloging) |
| `data/verification/verification_report_*.md` | `cli.py:_run_register` | The verification report (linked from the dataset version) |
| `data/exports/<version>/` (post-Stage 7) | `Catalog.load_artifact` | The exported shards |

| Output | Where | What |
|--------|-------|------|
| `data/registry/catalog.db` | `Catalog.__init__` | The SQLite DB (all writes go here) |
| `data/registry/catalog.yaml` | `Catalog.write_yaml_mirror` | The YAML mirror (when `yaml_mirror_path` is set) |
| `data/changelog.md` | `update_changelog` | Append-only change log |

## 6. Pipeline interactions

| Stage | Direction | What |
|-------|-----------|------|
| Stage 5 (splitting) | ← | `cli.py:_run_register` reads `split_manifest.json` + computes artifact hash via `compute_hash` |
| `cli.py` | ↔ | `_run_register` at line 312 opens the catalog, registers a `DatasetVersion`, writes YAML mirror |
| `sentinel_data.verification.gate` | ← | `verification_report_path` is linked from the registered `DatasetVersion` |
| `ml/` training (SentinelDataset) | → | `Catalog.load_artifact(name)` + `verify_artifact_hash(path)` are the load-time gate (post-Stage-7 seam swap) |
| `sentinel_data.splitting` | ↔ | `split_manifest.json`'s metadata_hash is stored as `SplitRecord.metadata_hash` (post-Stage-7) |
| `sentinel_data.export` (Stage 7) | → | The export shard's hash becomes the `DatasetVersion.artifact_hash` |

## 7. Tests

**Location:** `data_module/tests/test_registry/test_catalog.py`

**Command:**
```bash
cd ~/projects/sentinel/data_module
poetry run pytest tests/test_registry/ -v
```

**Coverage:**
- `Catalog.__init__` creates all 6 tables
- `add_*` / `get_*` / `list_*` round-trip for every table
- `load_artifact` returns None for retired versions
- `verify_artifact_hash` returns True for matching hash, False for mismatch
- `write_yaml_mirror` produces parseable YAML
- `diff_dataset_versions` correctly identifies added/removed/common
- `update_changelog` appends a new entry
- `migrate` records the migration exactly once

## 8. See also

- Previous stage: `sentinel_data/splitting/README.md`
- Next stage: `sentinel_data/analysis/README.md` (analysis is typically run AFTER registry, though not strictly dependent)
- CLI entry: `sentinel_data/cli.py` (`_run_register` at line 312)
- Stage 5 plan: `docs/proposal/Data_Module_Proposals/actionable_plans/06_stage_5_splitting_registry.md`
- AUDIT_PATCHES 5-P2, 5-P3, 5-P4, 5-P6, 5-P7 (catalog tables, hash function, training-run link, per-class metric projection)
- Why append-only retirements: the audit trail is permanent
- Sealed seams: the ML module calls `load_artifact` + `verify_artifact_hash` as the only registry interface (post-Stage-7)
- Why shared `compute_hash`: per AUDIT_PATCHES 5-P4, the ML module's `InferenceCache` will also call this function (single source of truth)
