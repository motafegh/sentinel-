# `sentinel_data.labeling` — Stage 3: Assigning Meaning to Code

> **Status: ✅ Core shipped (merger + gate + 2 parsers).** `label` subcommand in CLI is a **STUB** (`cli.py:223-229` — "NOT IMPLEMENTED — implement in Stage 3") — the merger runs from a Python entry point or test harness, not the CLI. ⚠ See §3: the two taxonomies (labeling vs representation) disagree on order and class membership.

## 1. Purpose

The labeling stage takes preprocessed contracts and assigns them **vulnerability class labels** from a canonical 10-class taxonomy. This is the stage where "source code on disk" becomes "source code with semantic meaning."

The module implements four layers:

1. **Schema** — the locked 10-class taxonomy (`taxonomy.yaml`) + a `class_names()` / `class_index()` loader
2. **Parsers** — one per data source, reads the source's native format and applies the crosswalk
3. **Merger** — combines labels from multiple sources with tier-precedence conflict resolution and the 99% DoS↔Reentrancy noise detector
4. **Gate** — Go/No-Go minimum-viable-corpus check before downstream stages

The BCCC dataset (the predecessor) assigned labels based on folder names — if a contract was in the `reentrancy/` folder, it was labeled Reentrancy. This produced an **89% false positive rate for Reentrancy** and **86.9% for CallToUnknown** because contracts were copied across folders (38.8% duplication) and the folder name didn't reflect the actual vulnerability.

The labeling module fixes this by making every label mapping **explicit, human-reviewed, and version-controlled** in crosswalk YAMLs. The merger resolves multi-source conflicts by **tier precedence** (T0 exploit-verified overrides everything) and the gate enforces minimum-viable-corpus criteria before any representation/verification work runs.

## 2. Source map

| File | Lines | Role |
|------|-------|------|
| `__init__.py` | 7 | Module docstring. |
| `merger.py` | 240 | `run_merger(data_dir, sources, *, force=False, output_dir=None) -> MergeResult` — multi-source label merger with tier precedence + DoS↔Reentrancy co-occurrence noise detection. |
| `gate.py` | 139 | `run_gate(data_dir, cfg) -> GateResult` — Go/No-Go minimum-viable-corpus gate with 4 criteria + CallToUnknown human-review flag. |
| `parsers/` | (sub-folder) | Source-specific parsers. See `parsers/README.md`. |
| `schema/` | (sub-folder) | Canonical 10-class taxonomy. See `schema/README.md`. |
| `crosswalks/` | (DOES NOT EXIST YET) | Source-specific class maps + confidence tiers. NEED TO BE CREATED. See §3. |

**Sub-total: 386 lines** across 3 Python files (excluding `__init__.py` + sub-folders).

## 3. Key concepts

### ⚠ The two-taxonomy problem (READ BEFORE USING)

The labeling taxonomy in `schema/taxonomy.yaml` is **NOT the same as** the representation schema in `sentinel_data.representation.graph_schema.CLASS_NAMES`. They have:

1. **Different class orderings** (labeling starts with `CallToUnknown=0`; representation starts with `Reentrancy=0`)
2. **Different class sets** (labeling has `TransactionOrderDependence` at id=8; representation doesn't have it. Labeling has `UnusedReturn` at id=9; representation has `NonVulnerable` at id=9.)

**Why two?** The representation schema is **preserved from Runs 1–9** to keep all existing model checkpoints loadable. The labeling taxonomy is the **v2 design intent** — 10 vulnerability classes with `NonVulnerable` being a *negative* label (not a class) and `TransactionOrderDependence` being a *positive* class (added in v2 because it was missing from the v1 set).

**The representation schema is the one that matters for training** (its order matches the classifier head). If you write a new model from scratch, use the representation order. If you only do `class_names()` lookups (string-keyed dicts), both work — but be aware that **index 9 means `NonVulnerable` in representation and `UnusedReturn` in labeling**.

See `sentinel_data/labeling/schema/README.md` §3 for the full side-by-side tables and the historical context.

### The label output format (per-source, from parsers)

Every labeled contract has a `data/labels/<source>/<sha256>.labels.json` with this structure:

```json
{
  "sha256": "abc123...",
  "source": "solidifi",
  "injection_folder": "Re-entrancy",          // source-specific provenance
  "classes": {
    "Reentrancy":         {"value": 1, "tier": "T0", "source": "solidifi"},
    "CallToUnknown":      {"value": 0, "tier": null, "source": null},
    "DenialOfService":    {"value": 0, "tier": null, "source": null},
    // ... all 10 classes
  },
  "n_pos": 1
}
```

The **canonical 10 classes** are always present (every parser writes the full dict). For multi-label sources (DIVE), several `value: 1` entries appear. For sources with no positives (DISL — pure NonVulnerable pool), all are `value: 0`.

### The merged format (from merger.py)

`run_merger()` writes `data/labels/merged/<sha256>.labels.json` with the **same shape** PLUS:

- `sources: list[str]` — the sorted list of source names that contributed
- `flags: list[str]` — e.g. `["dos_reentrancy_cooccur_suspect"]` if the 99% rule fired
- `classes[<class>].source` — which source's value/tier was kept (per-class, not per-contract)

### The 5 confidence tiers

| Tier | Meaning | Example sources |
|------|---------|-----------------|
| T0 | Exploit-verified | SolidiFI (injection-tested), DeFiHackLabs (exploit PoC) |
| T1 | Expert-audited | ScaBench, Web3Bugs, Bastet |
| T2 | Curated | SmartBugs Curated, DIVE (Nature Sci. Data 2025) |
| T3 | Tool-generated | Slither-Audited, Messi-Q, OpenZeppelin |
| T4 | Heuristic/derived | DISL (unlabeled, used as NonVulnerable pool) |

The tier lives in the crosswalk YAML per source. `_SOURCE_TIER` in `merger.py:45-52` is the runtime lookup for the 5 critical-path sources.

### Tier-precedence conflict resolution (`merger.py:30-35, 76-97`)

When a contract appears in multiple sources with conflicting values for a class, the merger keeps the **highest-confidence** (lowest tier rank):

```python
_TIER_ORDER = ["T0", "T1", "T2", "T3", "T4", None]
# Lower index = higher confidence

def _merge_class_entries(entries):
    positives = [(src, e) for src, e in entries if e["value"] == 1]
    if positives:
        # Keep the highest-confidence positive (lowest tier rank)
        best_src, best_entry = min(positives, key=lambda x: _tier_rank(x[1]["tier"]))
        return {"value": 1, "tier": best_entry["tier"], "source": best_src}
    # No positives — keep the highest-confidence negative
    best_src, best_entry = min(entries, key=lambda x: _tier_rank(x[1]["tier"]))
    return {"value": 0, "tier": None, "source": best_src}
```

**Within a tier, positive wins over negative** — false negatives are worse than false positives.

### The 99% DoS↔Reentrancy co-occurrence de-duplication rule (`merger.py:100-124`)

The BCCC failure mode was a 99% co-occurrence between DoS and Reentrancy labels on the same contracts — clear label noise. The merger prevents this by **flagging** (not silently dropping) when:

1. **Both DoS and Reentrancy are positive** in the merged output
2. **Only one source contributed** (no independent attesting source)
3. **That source's tier is T3 or T4** (low-confidence)
4. **That source's per-source DoS+Reentrancy co-occurrence rate > 50%** (the `CO_OCCUR_NOISE_THRESHOLD`)

DIVE (T2) at 12% co-occurrence is **NOT flagged** — that's legitimate multi-label signal, not noise.

The flagged contracts get `"flags": ["dos_reentrancy_cooccur_suspect"]` in their merged label, which Stage 4 verification reads (via `audit.co_occurrence_flagged`).

### The CallToUnknown human-review rule (`merger.py:46-52` + `gate.py:122-129`)

If the merger produces fewer than 300 verified `CallToUnknown` contracts across all sources, **the gate pauses and asks a human** whether to merge CallToUnknown into ExternalBug. This is a safety valve (per friend review). The threshold is in `config.yaml` under `pipeline.min_viable_corpus.call_to_unknown_min: 300`.

The same threshold is also the gate trigger in `gate.py:124-130`: if CallToUnknown < 300, the criterion is annotated `"below threshold — human review: merge into ExternalBug?"` (a NOTE, not a failure — the gate still passes if the major classes are OK).

### The Go/No-Go gate (`gate.py:63-138`)

Four criteria, read from `config.yaml` under `pipeline.min_viable_corpus`:

| Criterion | Default | Behavior |
|-----------|---------|----------|
| `total_contracts_min` | 4000 | **BLOCKING** |
| `per_class_positive_min_major` (Reentrancy, DoS, IntegerUO) | 300 | **BLOCKING** |
| `per_class_positive_min_minor` (7 others) | 100 | Warn only |
| `call_to_unknown_min` | 300 | Human-review NOTE |

```python
result.gate_passed = len(blocking) == 0
# blocking = [total_contracts, Reentrancy, DoS, IntegerUO]
```

Major classes are **blocking** (per the friend review — Reentrancy, DoS, IntegerUO are the highest-priority classes and a model that can't learn them is useless). Minor class failures are reported but don't block.

### Crosswalk YAML convention (NEEDS TO BE IMPLEMENTED)

> **⚠ The `crosswalks/` directory does NOT exist yet on disk.** The two parsers (`solidifi.py:27` + `dive.py:32`) load crosswalks from `Path(__file__).parents[1] / "crosswalks" / "<source>.yaml"`, so the path resolution is ready — but the YAML files themselves need to be created before either parser can run end-to-end. The DIVE crosswalk *was* committed in the Phase 5 verification output historically.

A crosswalk has at minimum:

```yaml
class_map:           # source-specific label → canonical class name
  "Re-entrancy": "Reentrancy"
  "Overflow-Underflow": "IntegerUO"
  ...
confidence_tier: "T0"   # the source's tier
```

See `parsers/README.md` for the full convention and the historical crosswalk files in `docs/legacy/bccc_deep_dive/.../Phase5_LabelVerification_2026-06-08/outputs/`.

### The CLI is a STUB

`cli.py:223-229`:
```python
def _run_label(args: argparse.Namespace) -> None:
    print(f"[label] {STAGE_DESCRIPTIONS['label']}")
    print(f"  config : {args.config}")
    if args.dry_run:
        print("  (dry-run — no files written)")
        return
    print("  NOT IMPLEMENTED — implement in Stage 3")
```

The merger runs from a Python entry point or test harness today. The `label` CLI subcommand prints "NOT IMPLEMENTED". This is a known gap — the stage is logically complete (merger + gate + parsers exist and pass tests) but the CLI wiring is deferred.

## 4. Public API

### `run_merger(data_dir, sources, *, force=False, output_dir=None) -> MergeResult` — `merger.py:147-239`

```python
@dataclass
class MergeResult:
    contracts_merged: int = 0
    single_source: int = 0
    multi_source: int = 0
    co_occurrence_flagged: int = 0
    cached: int = 0
    failed: int = 0
    duration_s: float = 0.0

def run_merger(
    data_dir: Path, sources: list[str],
    *, force: bool = False, output_dir: Path | None = None,
) -> MergeResult:
    """Merge per-source label files into canonical merged labels."""
```

### `run_gate(data_dir, cfg) -> GateResult` — `gate.py:63-138`

```python
@dataclass
class GateCriterion:
    name: str
    actual: int | float
    threshold: int | float
    passed: bool
    note: str = ""

@dataclass
class GateResult:
    criteria: list[GateCriterion]
    gate_passed: bool
    call_to_unknown_review_needed: bool

def run_gate(data_dir: Path, cfg: dict) -> GateResult:
    """Run the minimum-viable-corpus gate against merged labels.
    
    Reads `pipeline.min_viable_corpus` from config.yaml.
    Returns GateResult with per-criterion pass/fail and overall verdict.
    """
```

### `class_names()` / `class_index()` / `load_taxonomy()` — `schema/__init__.py:13-30`

```python
def class_names() -> list[str]:
    """Return the 10 class names in locked index order (index 0–9)."""

def class_index(name: str) -> int:
    """Return the integer index for a class name. Raises KeyError if unknown."""

@lru_cache(maxsize=1)
def load_taxonomy() -> dict[str, Any]:
    """Load and return the taxonomy YAML (cached after first call)."""
```

### Per-source parsers — `parsers/{solidifi,dive}.py`

```python
# parsers/solidifi.py:87
def label_source(data_dir: Path, *, force: bool = False, limit: int | None = None,
                output_dir: Path | None = None) -> LabelResult: ...

# parsers/dive.py:94
def label_source(data_dir: Path, *, force: bool = False, limit: int | None = None,
                output_dir: Path | None = None) -> LabelResult: ...
```

## 5. Inputs → outputs

| Input | Where | What |
|-------|-------|------|
| `data/preprocessed/<source>/<sha256>.meta.json` | Stage 1b | For parsers (source_name, original_path) |
| `data/raw/<source>/repo/` | Stage 1a | For DIVE parser's multi-folder index |
| `data/labels/<source>/<sha256>.labels.json` | Parser output | For merger |
| `config.yaml` `pipeline.min_viable_corpus` | `gate.py:73-78` | Gate thresholds |
| `data/labeling/crosswalks/<source>.yaml` | **MUST EXIST** | Class map + tier (parser) |

| Output | Where | What |
|--------|-------|------|
| `data/labels/<source>/<sha256>.labels.json` | Parser | Per-source per-contract labels |
| `data/labels/merged/<sha256>.labels.json` | Merger | Canonical merged labels with multi-source provenance + flags |
| Gate verdict | CLI / Python | `gate_passed: bool` + `call_to_unknown_review_needed: bool` + per-criterion list |

## 6. Pipeline interactions

| Stage | Direction | What |
|-------|-----------|------|
| Stage 1b (preprocessing) | ← | Reads `data/preprocessed/<source>/*.meta.json` for source_name + original_path |
| `parsers/{solidifi,dive}.py` | ← | Per-source label resolution |
| `sentinel_data.verification.class_auditor` | → | Reads `data/labels/merged/*.labels.json` for per-class stats + co-occurrence matrix |
| `sentinel_data.verification.semantic_checker`, `tool_validator`, `fp_estimator`, `negative_checker` | → | Read `data/labels/merged/*.labels.json` to enumerate positives |
| `sentinel_data.splitting.splitters` | → | Reads `data/labels/merged/*.labels.json` to build `Contract` objects |
| `sentinel_data.analysis.{balance_viz,cooccurrence,feature_dist,drift_monitor}` | → | Same |
| `sentinel_data.cli` | ↔ | `_run_label` is a **STUB** (line 223-229); the merger must be called from Python today |

## 7. Tests

**Location:** `data_module/tests/test_labeling/`
- `test_parser_solidifi.py` — folder extraction, single-label write, cache behaviour
- `test_parser_dive.py` — multi-folder index, multi-label write, bad_randomness drop
- `test_merger.py` — single-source, multi-source, conflict resolution, 99% co-occurrence flag, force/cached semantics
- `test_gate.py` — every criterion pass/fail, blocking vs warn-only, CallToUnknown human-review note
- `test_crosswalk_solidifi.py`, `test_crosswalk_dive.py` — crosswalk YAML structure validation (against the not-yet-on-disk YAMLs)
- `test_taxonomy.py` — class order / count / string equality regression guard

**Command:**
```bash
cd ~/projects/sentinel/data_module
poetry run pytest tests/test_labeling/ -v
```

## 8. See also

- Parsers: `sentinel_data/labeling/parsers/README.md`
- Schema: `sentinel_data/labeling/schema/README.md` (the two-taxonomy problem explained)
- Next stage: `sentinel_data/verification/README.md`
- CLI entry: `sentinel_data/cli.py` (`_run_label` at line 223 — STUB)
- The OTHER taxonomy (model order): `sentinel_data/representation/graph_schema.py:73-84`
- Why two: see `data_module/docs/legacy/bccc_deep_dive/Phase5_LabelVerification_2026-06-08/` for the Phase 5 verified label set
- DASP taxonomy: https://dasp.co/
- Stage 3 plan: `docs/proposal/Data_Module_Proposals/actionable_plans/04_stage_3_labeling.md`
- 99% DoS↔Reentrancy: `merger.py:100-124` + AUDIT_PATCHES 3-P3
- CallToUnknown merge rule: friend review §6.3.1 + `gate.py:122-130`
- Stage 3 Go/No-Go gate decision: integration test report `docs/training/stage_0_2_integration_test_2026-06-11.md`
