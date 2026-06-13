# `sentinel_data.labeling.schema` — Canonical Taxonomy Loader

> **Status: 1 YAML + 1 Python loader (31 lines).** The schema is the single point of truth for "what are the 10 class names and what order are they in" — used by the entire pipeline and the model. Per ADR-0009 (Phase D, 2026-06-12), `representation/graph_schema.py:CLASS_NAMES` also uses this exact same order.

## 1. Purpose

This subpackage holds the **canonical 10-class taxonomy YAML** (`taxonomy.yaml`) and a tiny loader (`__init__.py`) that exposes the class names as a list and a name→index lookup. The taxonomy is **the only place class identity is defined** for the labeling layer; every parser, the merger, the gate, the verification stage, and the analysis tools import from here.

The schema is **referenced by everything that does string-based class lookups** (which is most of the pipeline). Anything that does **index-based** lookups (e.g. label array column 7) must use `class_names()` to stay in sync — never hardcode index 7 as "Timestamp" or similar.

## 2. Source map

| File | Lines | Role |
|------|-------|------|
| `__init__.py` | 31 | `load_taxonomy()` (cached) + `class_names()` + `class_index()`. |
| `taxonomy.yaml` | 166 | The 10-class taxonomy with id, name, dasp, severity, description, crosswalk_notes per class. Schema-versioned (`schema_version: "1"`, `locked: true`, `num_classes: 10`). |

**Sub-total: 197 lines.**

## 3. Key concepts

### The canonical source of truth for the 10-class vocabulary

Per ADR-0009 (Phase D, 2026-06-12), this file (`taxonomy.yaml` + `class_names()`) is the **single canonical source of truth** for the 10-class vocabulary across the entire pipeline — including the model. `representation/graph_schema.py:CLASS_NAMES` uses this exact same order. The historical "representation order" (with `NonVulnerable` at index 9) was the pre-Run-7 ordering and is no longer used in production.

### The YAML schema (this file)

```yaml
schema_version: "1"
locked: true
num_classes: 10

classes:
  - id: 0
    name: CallToUnknown
    dasp: "DASP-7 (Bad Randomness / Unchecked Low-Level Calls)"
    severity: high
    description: >
      Low-level calls (call/delegatecall/send) whose return value is not
      checked, or calls to unverified external addresses. ...
    crosswalk_notes: >
      DIVE maps unchecked_low_level_calls here.
      SmartBugs Curated maps unchecked_low_level_calls here.
      BCCC had 86.9% FP rate for this class (per Phase 5 verification).
      If verified count < 300 post-merger, the merger pauses and asks human.
```

Key fields:
- `id: 0..9` — the index in `class_names()` order
- `name` — the canonical class string (used as dict key everywhere)
- `dasp` — mapping to the Decentralized Application Security Project taxonomy
- `severity` — `high` / `medium` / `low` / `critical` (loose ordering, not enforced)
- `description` — multi-line; the "what is this class" definition
- `crosswalk_notes` — per-source mapping guidance, including known failure modes (BCCC FP rates)

The `locked: true` flag signals that **the order is preserved across runs** to maintain compatibility with downstream ML code that depends on positional alignment. Changing the order is a breaking change that requires a schema version bump.

### The class set (labeling order)

| Idx | Name | Severity | DASP |
|-----|------|----------|------|
| 0 | CallToUnknown | high | DASP-7 |
| 1 | DenialOfService | high | DASP-5 |
| 2 | ExternalBug | critical | DASP-2/10 |
| 3 | GasException | medium | DASP-5 |
| 4 | IntegerUO | high | DASP-3 |
| 5 | MishandledException | medium | DASP-4 |
| 6 | Reentrancy | critical | DASP-1 |
| 7 | Timestamp | low | DASP-8 |
| 8 | TransactionOrderDependence | medium | DASP-7 |
| 9 | UnusedReturn | low | DASP-4 |

Note: no `NonVulnerable` class in the labeling taxonomy. NonVulnerable is a *negative* label, recorded as `value=0` for all 10 classes.

## 4. Public API

### `load_taxonomy()` — `__init__.py:13-17`

```python
@lru_cache(maxsize=1)
def load_taxonomy() -> dict[str, Any]:
    """Load and return the taxonomy YAML (cached after first call)."""
```

Returns the parsed YAML as a nested dict. `@lru_cache(maxsize=1)` means the file is read from disk once per process.

### `class_names()` — `__init__.py:20-22`

```python
def class_names() -> list[str]:
    """Return the 10 class names in locked index order (index 0–9)."""
```

The most-used function in the pipeline. Returns e.g. `["CallToUnknown", "DenialOfService", "ExternalBug", "GasException", "IntegerUO", "MishandledException", "Reentrancy", "Timestamp", "TransactionOrderDependence", "UnusedReturn"]`.

### `class_index(name)` — `__init__.py:25-30`

```python
def class_index(name: str) -> int:
    """Return the integer index for a class name. Raises KeyError if unknown."""
```

Inverse of `class_names()`. Raises `KeyError` (not `ValueError`) for unknown names — consistent with the dict-keyed code that consumes it.

## 5. Inputs → outputs

| Input | Where | What |
|-------|-------|------|
| `taxonomy.yaml` | This directory | The full taxonomy definition |

| Output | Where | What |
|--------|-------|------|
| `list[str]` of length 10 | `class_names()` | Class names in locked order |
| `dict[str, Any]` | `load_taxonomy()` | The full parsed YAML (per-class dict with id, name, dasp, severity, description, crosswalk_notes) |
| `int` | `class_index(name)` | 0-9 for a known class |

## 6. Pipeline interactions

| Stage | Direction | What |
|-------|-----------|------|
| `parsers/solidifi.py`, `parsers/dive.py` | → | `class_names()` to build the full class dict in `*.labels.json` |
| `merger.py` | → | `class_names()` to iterate over classes when merging |
| `gate.py` | → | `class_names()` to enumerate per-class criteria |
| `verification/class_auditor.py` | → | `class_names()` for the per-class table |
| `verification/semantic_checker.py` | → | `class_names()` to enumerate classes for the semantic check |
| `verification/tool_validator.py`, `fp_estimator.py`, `negative_checker.py` | → | Same |
| `analysis/balance_viz.py`, `cooccurrence.py`, `feature_dist.py`, `drift_monitor.py` | → | Same |
| `splitting/splitters.py` | → | `class_names()` for the per-class stratification (indirect via Contract.classes) |
| `representation/graph_schema.py` | ↔ | Uses the same CLASS_NAMES order (per ADR-0009) — not a separate taxonomy anymore |

## 7. Tests

**Location:** `data_module/tests/test_labeling/test_taxonomy.py`

**Command:**
```bash
cd ~/projects/sentinel/data_module
poetry run pytest tests/test_labeling/test_taxonomy.py -v
```

**Coverage:**
- `class_names()` returns exactly 10 names in locked order
- `class_index()` round-trips: `class_index(class_names()[i]) == i`
- `class_index("NonExistent")` raises `KeyError`
- The 10 names match the expected canonical set (regression guard against accidental reorder)

## 8. See also

- Parent: `sentinel_data.labeling/README.md`
- Parsers: `sentinel_data.labeling.parsers`
- The representation schema (uses the same order): `sentinel_data.representation.graph_schema.CLASS_NAMES`
- Why two: see `data_module/docs/legacy/bccc_deep_dive/Phase5_LabelVerification_2026-06-08/` for the Phase 5 verified label set that the labeling taxonomy descends from
- DASP taxonomy: https://dasp.co/ (Decentralized Application Security Project)
- Locked order rationale: same file `taxonomy.yaml:1-12` (header comment)
