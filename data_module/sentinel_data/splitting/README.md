# `sentinel_data.splitting` — Stage 5: Preparing Data for Training

> **Status: ✅ Fully implemented (4 files, 759 lines).** 4 splitter strategies + dedup_enforcer + leakage_auditor + NonVulnerable cap. `split` CLI subcommand fully wired (`cli.py:232-310`).

## 1. Purpose

This stage takes verified, labeled contracts and produces **deterministic, leak-free, stratified train/val/test splits** ready for model training. The output is versioned JSONL files with a manifest that records every decision made during the split.

The module is the **structural fix for the BCCC failure pattern**. BCCC had a 34.9% cross-split leakage rate in v6 — contracts that appeared in both train and test, inflating Run 9's F1 by an estimated ~0.05. The splitting module prevents this by:

1. **Enforcing deduplication BEFORE splitting** — near-dup groups stay in one split (via the dedup_enforcer)
2. **Auditing for leakage AFTER splitting** — an independent safety net using a different similarity algorithm (text shingle vs AST)
3. **Stratifying by source** — prevents one dominant source from skewing splits
4. **Capping NonVulnerable at 3:1** — prevents the "predict negative and win 99%+" failure mode

The module also implements the **4 splitting strategies** required by the plan, plus the **`Splits` / `SplitMetadata` / `Contract` dataclasses** that form the versioned contract between data and training.

## 2. Source map

| File | Lines | Role |
|------|-------|------|
| `__init__.py` | 39 | Re-exports the public API: `Contract`, `Splits`, `SplitMetadata`, `SPLITTERS`, the 4 splitter functions, `apply_strategy`, `load_splits`, `write_manifest`, `write_splits`, `apply_dedup_enforcer`, `find_leaks`, `run_audit`, `apply_nonvulnerable_cap`, `DEFAULT_TEXT_SIMILARITY_THRESHOLD`, `DEFAULT_CAP`. |
| `splitters.py` | 441 | `Contract` / `Splits` / `SplitMetadata` dataclasses + the 4 splitter strategies (`random_split`, `stratified_split`, `project_split`, `temporal_split`) + `SPLITTERS` dispatch + `apply_strategy()` + `write_manifest()` + `write_splits()` + `load_splits()`. |
| `dedup_enforcer.py` | 116 | `apply_dedup_enforcer(splits) -> Splits` — reassigns near-dup groups that straddle a split boundary to the majority split (ties → train). Records all reassignments in metadata. |
| `leakage_auditor.py` | 163 | `find_leaks(splits, *, texts, threshold, sources_for_text) -> LeakageReport` + `run_audit(splits, *, data_dir, threshold, sources)` high-level wrapper. Uses 3-shingle Jaccard similarity (different from dedup_enforcer's AST-based dedup_group). |
| `nonvulnerable_cap.py` | 163 | `apply_nonvulnerable_cap(splits, *, cap=3.0, seed=42) -> Splits` — subsample NonVulnerable to at most `cap × total_positive_count`. Stratified by source to preserve per-source distribution. |

**Sub-total: 759 lines** across 4 files.

## 3. Key concepts

### The 4 splitter strategies (`splitters.py:159-350`)

| Strategy | Function | When to use | Key property |
|----------|----------|-------------|--------------|
| **Random** | `random_split(contracts, ratios, seed)` | Sanity testing | Contracts randomly assigned to splits |
| **Stratified** | `stratified_split(contracts, ratios, seed, strata)` | Tool-derived datasets | Per-(class, source, tier) distribution preserved within ±2% |
| **Project** | `project_split(contracts, ratios, seed)` | Audit datasets (Bastet, ScaBench, Web3Bugs, DeFiHackLabs) | Entire project stays in one split |
| **Temporal** | `temporal_split(contracts, ratios, seed, cutoff_year=2023)` | Time-sensitive analyses | Pre-cutoff in train/val, post-cutoff in test (simulates "train on past, test on future") |

The `SPLITTERS` dispatch dict (`splitters.py:355-361`) maps string → function. Note the `project` ↔ `project_level` alias (per AUDIT_PATCHES 5-P1):

```python
SPLITTERS = {
    "random": random_split,
    "stratified": stratified_split,
    "project": project_split,
    "project_level": project_split,   # alias
    "temporal": temporal_split,
}
```

The default in `cli.py:285` is `stratified_split` with seed=42.

Default split ratios: **70% train, 15% val, 15% test** (`DEFAULT_RATIOS = (0.70, 0.15, 0.15)`, `splitters.py:45`).

### The `Contract` dataclass (`splitters.py:55-77`)

The unit of splitting. Designed for all 4 strategies + dedup_enforcer:

```python
@dataclass
class Contract:
    sha256: str
    source: str                        # "solidifi" / "dive" / "smartbugs_curated" / "disl"
    tier: str                          # "T0" / "T1" / "T2" / "T3" / "T4"
    classes: dict[str, int] = field(default_factory=dict)
    primary_class: str = "NonVulnerable"
    n_pos: int = 0
    loc: int = 0                       # for temporal splitting
    year: int = 0                      # for temporal splitting
    dedup_group: Optional[str] = None  # for dedup_enforcer
    project_id: Optional[str] = None   # for project-level splitter
    
    @property
    def is_nonvulnerable(self) -> bool:
        return all(v == 0 for v in self.classes.values())
```

A contract with all classes=0 is NonVulnerable (`is_nonvulnerable=True`). This is the trigger for `nonvulnerable_cap`.

### The dedup_enforcer — BCCC failure pattern fix (`dedup_enforcer.py:31-115`)

The BCCC 38.8% duplication rate meant many contracts appeared in BOTH train and test. The dedup_enforcer eliminates this.

**Two-pass split** (per plan D-5.2): `stratified_splitter` → `dedup_enforcer`.

**Algorithm**:
1. Build `group_id → {split_name → [contracts]}` map from the `dedup_group` field (set by Stage 1's deduplicator).
2. For each group with >1 members: count how many are in each split.
3. If the group straddles a split boundary (>1 split with members), reassign ALL members to the **majority split** (ties → train).
4. Rebuild train/val/test lists with the reassignments.
5. Record all reassignments in `Splits.metadata.reassignments`.

```python
# from dedup_enforcer.py:60-77 (simplified)
for group_id, by_split in group_to_splits.items():
    counts = {s: len(by_split[s]) for s in ("train", "val", "test")}
    straddles = sum(1 for s in ("train", "val", "test") if counts[s] > 0)
    if straddles <= 1:
        continue
    target = max(("train", "val", "test"), key=lambda s: (counts[s], s == "train"))
    # ... reassign all members of this group to `target`
```

The `dedup_group` field comes from `data/preprocessed/<source>/<sha256>.meta.json` (set by `preprocessing/deduplicator.py:32-79` in Stage 1b). Groups are pre-computed; this stage just does the fast lookup.

### The leakage_auditor — independent safety net (`leakage_auditor.py:77-163`)

Runs **after** the dedup_enforcer. Uses a **different similarity algorithm** (text shingle Jaccard, 3-grams) than the dedup_enforcer (which uses Stage 1's AST-based `dedup_group`). The two methods can disagree, and **that's the point** — the auditor catches what the enforcer misses.

**Algorithm** (O(N²) by design — for v2 baseline ~22K contracts this is ~500M comparisons, ~10-30 min):
1. Build a 3-shingle set per contract from the preprocessed `.sol` text
2. For every (train, val), (train, test), (val, test) pair: compute Jaccard similarity
3. Report all pairs above the threshold (default 0.5, per AUDIT_PATCHES 5-P3) as `LeakPair`s

```python
def find_leaks(splits, *, texts, threshold=0.5, sources_for_text=None) -> LeakageReport:
    # 3-shingle Jaccard similarity — independent of Stage 1's AST dedup
    # O(N²) for v2 baseline; LSH for v2.1+
```

**The auditor REPORTS only** — does NOT modify the splits. The `LeakageReport.n_pairs` is recorded in `split_manifest.json` for the data team to review. A non-empty leak list is a **bug to fix in dedup_enforcer** (or a config tweak), not a reason to discard the split.

For v2.1+, swap in an LSH (Locality-Sensitive Hashing) implementation to get O(N) candidate-pair finding. For the v2 baseline O(N²) is acceptable.

### The NonVulnerable 3:1 cap (`nonvulnerable_cap.py:47-162`)

The "friend review" insight: DISL provides 514,506 unlabeled contracts as NonVulnerable examples. With ~1,200 positives from the 5 critical-path sources, the default ratio is **428:1** — the SAME BCCC failure pattern at larger scale. A model that defaults to "predict negative" and is right 99%+ of the time never learns positive patterns.

**The cap** (`pipeline.negative.positive_ratio_max: 3.0` in config.yaml): NonVulnerable count ≤ 3 × total positive count.

```python
def apply_nonvulnerable_cap(splits, *, cap=3.0, seed=42) -> Splits:
    """Subsample NonVulnerable to at most cap * total_positive_count contracts.
    
    Stratified by source to preserve the per-source distribution.
    """
```

**Stratified subsampling** by source: the subsample is proportional to the original per-source distribution. Without stratification, the subsample could be 100% DISL (largest source). The implementation uses proportional allocation with deterministic remainder distribution (largest-remainder method, ties broken by source name sort).

**Audit log**: every subsample is recorded in `splits.metadata.nonvulnerable_cap` with the original count, the capped count, the per-source breakdown, and the cap value.

**Per-class override**: `pipeline.negative.per_class_ratio_max.<ClassName>: 5.0` allows per-class tuning. Default is 3:1.

**Why 3:1, not higher or lower**:
- Higher (5:1, 10:1) reproduces the BCCC problem
- Lower (1:1) starves the NonVulnerable signal
- 3:1 is the empirical sweet spot

### The split manifest (`splitters.py:381-387`)

The versioned contract between splitting and downstream training. Written to `data/splits/v<N>/split_manifest.json`:

```json
{
  "version": "v1",
  "seed": 42,
  "strategy": "stratified",
  "strategy_per_source": {"solidifi": "stratified", "dive": "stratified", ...},
  "ratios": [0.7, 0.15, 0.15],
  "contract_counts": {"train": 29103, "val": 6236, "test": 6237},
  "class_distributions": {"train": {"Reentrancy": 450, ...}, ...},
  "source_distributions": {"train": {"solidifi": 283, ...}, ...},
  "tier_distributions": {"train": {"T0": 1200, ...}, ...},
  "dedup_groups_resolved": 142,
  "reassignments": [{"group": "sha_abc", "from_split": "val", "to_split": "train", "contract_count": 1}, ...],
  "nonvulnerable_cap": {"cap": 3.0, "total_positive": 4622, "max_nonvuln": 13866, "per_source": {...}, "per_split": {...}},
  "leakage_audit": {"leaks_found": 0, "max_similarity": 0.23},
  "generated_at": "2026-..."
}
```

The manifest is the **complete audit trail** — answer to "what data did Run 11 train on?"

### The CLI flow (`cli.py:232-310`)

```python
def _run_split(args):
    # 1. Read merged labels, build Contract objects
    # 2. Run splitter (default: stratified, seed=args.seed)
    # 3. Apply dedup_enforcer (BCCC-failure pattern fix)
    # 4. Apply NonVulnerable cap (default 3:1)
    # 5. Write train/val/test JSONL + split_manifest.json
```

Order matters: dedup_enforcer BEFORE NonVulnerable cap (the cap is based on the dedup-resolved positive count).

CLI flags:
- `--version N` — split version number (default 1 = v1)
- `--seed N` — RNG seed (default 42)
- `--nonvuln-cap RATIO` — NonVulnerable : positive cap (default 3.0, per friend review §6.3.1)

## 4. Public API

### The `Contract` dataclass — `splitters.py:55-77`

(see §3 above for full definition)

### The 4 splitter functions — `splitters.py:159-350`

```python
def random_split(
    contracts: list[Contract],
    ratios: tuple[float, float, float] = DEFAULT_RATIOS,
    seed: int = 42,
) -> Splits: ...

def stratified_split(
    contracts: list[Contract],
    ratios: tuple[float, float, float] = DEFAULT_RATIOS,
    seed: int = 42,
    strata: list[str] | None = None,    # default: ["primary_class", "source", "tier"]
) -> Splits: ...

def project_split(
    contracts: list[Contract],
    ratios: tuple[float, float, float] = DEFAULT_RATIOS,
    seed: int = 42,
) -> Splits: ...

def temporal_split(
    contracts: list[Contract],
    ratios: tuple[float, float, float] = DEFAULT_RATIOS,
    seed: int = 42,
    cutoff_year: int = 2023,
) -> Splits: ...
```

### `apply_strategy(strategy, contracts, ratios, seed, **kwargs) -> Splits` — `splitters.py:364-376`

```python
def apply_strategy(strategy: str, contracts: list[Contract], ...) -> Splits:
    """Dispatch to the named splitter strategy."""
```

### `apply_dedup_enforcer(splits) -> Splits` — `dedup_enforcer.py:31-115`

### `find_leaks(splits, *, texts, threshold=0.5, sources_for_text=None) -> LeakageReport` — `leakage_auditor.py:77-137`

```python
@dataclass
class LeakPair:
    sha_a: str
    split_a: str
    sha_b: str
    split_b: str
    similarity: float

@dataclass
class LeakageReport:
    threshold: float
    pairs: list[LeakPair]
```

Plus `run_audit(splits, *, data_dir, threshold=0.5, sources=None) -> LeakageReport` — high-level wrapper that loads the preprocessed `.sol` texts and runs `find_leaks`.

### `apply_nonvulnerable_cap(splits, *, cap=3.0, seed=42) -> Splits` — `nonvulnerable_cap.py:47-162`

### `write_manifest(splits, output_dir) -> Path` + `write_splits(splits, output_dir) -> None` + `load_splits(output_dir) -> tuple[dict, SplitMetadata]` — `splitters.py:381-441`

### Constants

```python
DEFAULT_RATIOS = (0.70, 0.15, 0.15)
DEFAULT_TEXT_SIMILARITY_THRESHOLD = 0.5
SHINGLE_SIZE = 3
DEFAULT_CAP = 3.0
```

## 5. Inputs → outputs

| Input | Where | What |
|-------|-------|------|
| `data/labels/merged/*.labels.json` | Stage 3 | Read by `cli.py:_run_split` to build `Contract` objects |
| `data/preprocessed/<source>/<sha256>.meta.json` (for `dedup_group`) | Stage 1b | For dedup_enforcer's group lookup |
| `data/preprocessed/<source>/<sha256>.sol` (for leakage_auditor) | Stage 1b | For text shingle similarity (optional, not run by default CLI) |
| `config.yaml` `pipeline.negative.*` | `nonvulnerable_cap` | Cap value + per-class overrides |

| Output | Where | What |
|--------|-------|------|
| `data/splits/v<N>/train.jsonl` | `splitters.py:395` | One `Contract` per line as JSON |
| `data/splits/v<N>/val.jsonl` | Same |  |
| `data/splits/v<N>/test.jsonl` | Same |  |
| `data/splits/v<N>/split_manifest.json` | `splitters.py:384` | The versioned contract (see §3) |

## 6. Pipeline interactions

| Stage | Direction | What |
|-------|-----------|------|
| Stage 3 (labeling) | ← | Reads `data/labels/merged/*.labels.json` to build `Contract` objects |
| Stage 1b (preprocessing) | ← | Reads `data/preprocessed/<source>/<sha256>.meta.json` for `dedup_group` |
| Stage 6 (splitting CLI wires dedup + NonVuln cap; the leakage_auditor is called manually today) | → | Outputs JSONL + manifest |
| Stage 5 (registry) | → | `cli.py:_run_register` reads `split_manifest.json` + computes the artifact hash |
| `sentinel_data.splitting.leakage_auditor` | ↔ | `run_audit` reads `data/preprocessed/<source>/<sha256>.sol` (independent safety net) |
| `ml/` training (SentinelDataset) | → | Reads `data/splits/v<N>/{train,val,test}.jsonl` + `split_manifest.json` (post-seam-swap) |

## 7. Tests

**Location:** `data_module/tests/test_splitting/test_splitters.py`

**Command:**
```bash
cd ~/projects/sentinel/data_module
poetry run pytest tests/test_splitting/ -v
```

**Coverage:**
- Each splitter strategy's split counts add up to total
- Stratified split preserves per-class distribution within ±2%
- Project split keeps each project in one split
- Temporal split puts all post-cutoff in test
- dedup_enforcer reassigns straddling groups correctly
- NonVulnerable cap subsamples proportionally to source distribution
- Manifest is reproducible (same seed → byte-identical JSON)
- `write_splits` / `load_splits` round-trip
- Strategy dispatch (`apply_strategy("project")` works, `apply_strategy("unknown")` raises)

## 8. See also

- Previous stage: `sentinel_data/verification/README.md`
- Next stage: `sentinel_data/registry/README.md`
- CLI entry: `sentinel_data/cli.py` (`_run_split` at line 232, `_run_register` at line 312)
- Stage 5 plan: `docs/proposal/Data_Module_Proposals/actionable_plans/06_stage_5_splitting_registry.md`
- BCCC dedup story: AUDIT_PATCHES 5-P1 (project_level alias), 5-P2 (catalog tables), 5-P3 (text shingle threshold), 5-P4 (hash function)
- 99% DoS↔Reentrancy from BCCC: `merger.py:100-124`
- "Predict negative and win 99%+" failure mode: friend review §6.3.1
- AUDIT_PATCHES 5-P7: per-class metric projection in dataset_diff
- Manifest format: `splitters.py:78-96` (SplitMetadata fields) — locked
