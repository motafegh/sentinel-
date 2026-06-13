# `sentinel_data.labeling.parsers` — Source-Specific Label Readers

> **Status: 2 parsers implemented (SolidiFI, DIVE).** Each parser is small (~160 lines) and follows the same shape: read `*.meta.json` → apply crosswalk → write `*.labels.json`. Adding a new source means adding a new parser module; the merger and gate are source-agnostic.

## 1. Purpose

This subpackage contains **one parser per data source** that knows how to translate that source's idiosyncratic label signal into SENTINEL's canonical 10-class taxonomy. Parsers are the **bridge between "how this source records labels" and "the canonical class dict every downstream stage expects"**.

The pattern is uniform:

1. **Read** every `data/preprocessed/<source>/<sha256>.meta.json` (Stage 1 sidecar)
2. **Resolve** which canonical classes are positive for that contract (source-specific — folder name, DAST category, multi-folder membership, …)
3. **Apply** the crosswalk YAML (`labeling/crosswalks/<source>.yaml`) to map source labels → canonical names
4. **Write** one `data/labels/<source>/<sha256>.labels.json` per contract

Adding a new source (`web3bugs`, `smartbugs_curated`, `defihacklabs`, …) is **just** one new `<source>.py` file + one `<source>.yaml` crosswalk. The merger, gate, and registry don't change.

## 2. Source map

| File | Lines | Role |
|------|-------|------|
| `__init__.py` | 7 | One-paragraph module docstring describing the parser contract. |
| `solidifi.py` | 162 | `SolidiFI parser` — maps SolidiFI's injection folder name (e.g. `Re-entrancy`) to a canonical class. Single-label per contract. |
| `dive.py` | 167 | `DIVE parser` — uses multi-folder membership in `data/raw/dive/repo/<Class>/<N>.sol` to assign multi-labels. Bad Randomness silently dropped (no canonical equivalent). |

**Sub-total: 336 lines** across 2 parsers.

## 3. Key concepts

### The parser contract

Every parser exposes a single function with the same shape:

```python
def label_source(
    data_dir: Path,
    *,
    force: bool = False,
    limit: int | None = None,
    output_dir: Path | None = None,
) -> LabelResult:
    """Run the <source> labeling parser.

    Reads data/preprocessed/<source>/<sha>.meta.json, applies the
    crosswalk, writes data/labels/<source>/<sha>.labels.json.

    Returns LabelResult with counts.
    """
```

The `LabelResult` dataclass aggregates counts: `contracts_seen`, `labels_written`, `labels_cached`, `labels_failed`, `duration_s`. DIVE adds `nonvulnerable_written` (its positive set is sparse so the nonvuln count is meaningful).

### The crosswalk YAML convention

Each parser loads its crosswalk from a fixed path:

```python
_CROSSWALK_PATH = Path(__file__).parents[1] / "crosswalks" / "solidifi.yaml"
#              = <data_module>/sentinel_data/labeling/crosswalks/solidifi.yaml
```

The crosswalk has at minimum:

```yaml
class_map:           # source-specific label → canonical class name
  "Re-entrancy": "Reentrancy"
  "Overflow-Underflow": "IntegerUO"
  ...
confidence_tier: "T0"   # the source's confidence tier (T0–T4)
```

> **⚠ STATUS NOTE (2026-06-11):** The `crosswalks/` directory does not yet exist on disk (only `parsers/` and `schema/` are present). The two existing parsers load their crosswalks from the path above; **the YAMLs themselves need to be created** before the parsers can run end-to-end. The DIVE crosswalk *was* committed in `docs/legacy/bccc_deep_dive/.../Phase5_LabelVerification_2026-06-08/` historically.

### SolidiFI: folder-name lookup (`solidifi.py:48-57`)

SolidiFI's structure is `data/raw/solidifi/repo/buggy_contracts/<FOLDER>/buggy_N.sol`. The injection folder name is the label.

```python
def _extract_folder(original_path: str) -> str | None:
    parts = Path(original_path).parts
    if len(parts) >= 3 and parts[1] == "buggy_contracts":
        return parts[2]
    return None
```

Then `class_map.get(folder)` resolves to a canonical class. Single-label per contract (SolidiFI's model). Tier is `T0` (injection-verified — the bug was deliberately injected).

### DIVE: multi-folder membership (`dive.py:54-69`)

DIVE stores all contracts flat in `data/raw/dive/repo/__source__/<N>.sol` and also creates symlinks in `data/raw/dive/repo/<VulnClass>/<N>.sol` for each positive class. So a single contract can appear in N vulnerability folders → multi-label.

```python
def _build_folder_index(raw_repo: Path, class_map: dict) -> dict[str, frozenset[str]]:
    """filename → frozenset of canonical class names.
    
    Scans every mapped folder under raw_repo for *.sol files and records
    which canonical classes each filename belongs to. Bad Randomness and
    any unmapped folders are silently skipped.
    """
```

The folder index is built **once** before the per-meta loop (O(F) where F = total DIVE files), making the per-contract lookup O(1). For DIVE's 22,073 contracts this is the difference between seconds and minutes.

Bad Randomness is **silently dropped** — no canonical equivalent, and inventing one would muddy the 10-class taxonomy. (The `bccc_class_to_sentinel` mapping in `verification/probe_trivials.py:16-27` shows the historical BCCC 12-class → SENTINEL 10-class mapping; "badrandomness" is not in it.)

### The canonical 10-class taxonomy

The labeling taxonomy in `schema/taxonomy.yaml` is the **single source of truth** for the 10-class vocabulary. Per ADR-0009 (Phase D, 2026-06-12), `representation/graph_schema.py:CLASS_NAMES` also uses this exact same order — the two are aligned.

**`labeling/schema/taxonomy.yaml`** (the canonical order — used by `class_names()`, parsers, merger, gate, verification, analysis, and now also the model):

| Idx | Class | Idx | Class |
|-----|-------|-----|-------|
| 0 | CallToUnknown | 5 | MishandledException |
| 1 | DenialOfService | 6 | Reentrancy |
| 2 | ExternalBug | 7 | Timestamp |
| 3 | GasException | 8 | TransactionOrderDependence |
| 4 | IntegerUO | 9 | UnusedReturn |

The DIVE crosswalk notably has **no entry for `TransactionOrderDependence`** in `taxonomy.yaml` despite the class existing — DIVE maps `front_running` to `Timestamp` instead. The `TransactionOrderDependence` slot in the taxonomy is used by **SolidiFI's `TOD` injection** which maps directly.

## 4. Public API

### `solidifi.label_source()` — `solidifi.py:87-161`

```python
def label_source(
    data_dir: Path,
    *,
    force: bool = False,
    limit: int | None = None,
    output_dir: Path | None = None,
) -> LabelResult:
    """Run the SolidiFI labeling parser.

    Reads data/preprocessed/solidifi/*.meta.json, maps each contract's
    injection folder to a canonical class via the crosswalk, writes
    data/labels/solidifi/<sha256>.labels.json.

    Args:
        force: Overwrite existing .labels.json files.
        limit: Process only the first N contracts.
        output_dir: Override output directory (default: data_dir/labels/solidifi).
    """
```

### `dive.label_source()` — `dive.py:94-166`

```python
def label_source(
    data_dir: Path,
    *,
    force: bool = False,
    limit: int | None = None,
    output_dir: Path | None = None,
) -> LabelResult:
    """Run the DIVE labeling parser.

    Args:
        force: Overwrite existing .labels.json files.
        limit: Process only the first N contracts.
        output_dir: Override output directory (default: data_dir/labels/dive).
    """
```

### `LabelResult` dataclasses

Both parsers return a `LabelResult` dataclass. Common fields: `contracts_seen`, `labels_written`, `labels_cached`, `labels_failed`, `duration_s`. DIVE's adds `nonvulnerable_written`.

## 5. Inputs → outputs

| Input | Where | What |
|-------|-------|------|
| `data/preprocessed/<source>/<sha256>.meta.json` | Stage 1 sidecar | Source-agnostic metadata (sha, original_path, solc_version, …) |
| `data/raw/<source>/repo/` | Stage 0 ingest | DIVE only — for the multi-folder index. SolidiFI uses `meta.original_path` instead. |
| `data/labeling/crosswalks/<source>.yaml` | Repo | Source-specific class map + confidence tier |

| Output | Where | What |
|--------|-------|------|
| `data/labels/<source>/<sha256>.labels.json` | New file | Full class dict: `{classes: {<ClassName>: {value: 0\|1, tier: "T0"\|None, source: <source>}}, n_pos: int, ...}` |

## 6. Pipeline interactions

| Stage | Direction | What |
|-------|-----------|------|
| Stage 1 (preprocess) | ← | Reads `*.meta.json` sidecars |
| `sentinel_data.labeling.merger` | → | Reads `data/labels/<source>/*.labels.json` to build merged labels |
| `sentinel_data.labeling.gate` | → | Reads merged labels to run the minimum-viable-corpus gate |
| `sentinel_data.labeling.schema.class_names()` | ↔ | Returns the 10 class names in labeling order (not model order) |

## 7. Tests

**Location:** `data_module/tests/test_labeling/`
- `test_parser_solidifi.py` — folder extraction, single-label write, cache behaviour
- `test_parser_dive.py` — multi-folder index, multi-label write, bad_randomness drop
- `test_crosswalk_solidifi.py`, `test_crosswalk_dive.py` — crosswalk YAML structure validation
- `test_taxonomy.py` — class order / count / string equality

**Command:**
```bash
cd ~/projects/sentinel/data_module
poetry run pytest tests/test_labeling/ -v
```

## 8. See also

- Parent: `sentinel_data.labeling/README.md`
- Crosswalks (to be created): `data_module/sentinel_data/labeling/crosswalks/<source>.yaml`
- Schema: `sentinel_data.labeling.schema`
- Merger: `sentinel_data.labeling.merger`
- Gate: `sentinel_data.labeling.gate`
- Stage 3 plan: `docs/proposal/Data_Module_Proposals/actionable_plans/04_stage_3_labeling.md`
- Probe trivials (uses class names): `sentinel_data.verification.probe_trivials`
- BCCC review batches (used by probe dataset): `data_module/docs/legacy/bccc_deep_dive/Phase5_LabelVerification_2026-06-08/outputs/review_batches/`
