# `sentinel_data.verification` — Stage 4: The BCCC-Failure Catcher

> **Status: ✅ 10 modules shipped + 10 per-class pattern YAMLs.** Stage 4 is fully implemented — class_auditor, semantic_checker, tool_validator, fp_estimator, negative_checker, gate, report_generator, slither_runner, probe_dataset, probe_trivials. The patterns/ YAMLs are documentation-only (semantic_checker reads graph features directly, not the YAMLs).

## 1. Purpose

This stage asks the question that BCCC failed to ask for 14 days of work: **"are these labels correct?"** It implements a **layered defense** against false positives, false negatives, and the kind of systematic label noise that destroyed Run 9.

The BCCC dataset had an **89% false positive rate for Reentrancy** and **86.9% for CallToUnknown**. These weren't subtle errors — they were massive label quality failures that would have been caught in minutes by the semantic checker:

- **Reentrancy FP pattern**: the BCCC pattern matched any external call + state write, even if the state write was BEFORE the call (which is not a reentrancy)
- **CallToUnknown FP pattern**: the BCCC pattern matched any `.call{}` without checking if the target was actually unknown

This module would have caught both in the first run, preventing the 14-day debugging session.

## 2. Source map

The previous README said "6 components" — the actual implementation has **10 Python modules** + 10 per-class pattern YAMLs:

| File | Lines | Role |
|------|-------|------|
| `__init__.py` | 7 | Module docstring listing the components. |
| `class_auditor.py` | 189 | `run_audit(data_dir) -> AuditResult` — per-class stats + 10×10 co-occurrence matrix with auto-flag for `P(B\|A) > 50%`. |
| `semantic_checker.py` | 283 | `run_semantic_check(data_dir, *, limit_per_class=None) -> SemanticCheckResult` — graph-feature-based checks per class. NOT_EXTRACTABLE for DoS / GasException / TOD. |
| `tool_validator.py` | 273 | `run_tool_validation(data_dir, *, limit_per_class=None, force=False, only_classes=None) -> ToolValidationResult` — Slither per-class agreement. |
| `fp_estimator.py` | 336 | `run_fp_estimation(data_dir, *, sample_size=50, seed=42, only_classes=None, force=False) -> FPEstimationResult` — stratified-by-(source, tier) FP estimation. |
| `negative_checker.py` | 235 | `run_negative_check(data_dir, *, warn_threshold=0.05, fail_threshold=0.10, detectors=None, limit=None, force=False) -> NonVulnResult` — Slither hit rate on NonVulnerable contracts. |
| `gate.py` | 286 | `run_gate(audit, semantic, *, tool_validation=None, fp_estimation=None, negative_check=None) -> GateResult` — per-class VERIFIED / PROVISIONAL / BEST-EFFORT / FAIL verdict. |
| `report_generator.py` | 272 | `generate_report(...) -> VerificationReport` + `.write(path)` — human-readable `verification_report_<ts>.md`. |
| `slither_runner.py` | 250 | Shared Slither runner with content-addressed cache at `data/slither_cache/<source>/<sha256>.slither.json`. Exports `CLASS_TO_DETECTORS` + `ALL_DETECTORS` + `SlitherFindings` + `run_on_contract()`. |
| `probe_dataset.py` | 361 | `build_probe_dataset(...) -> ProbeDataset` — 40 real contracts/class from BCCC review + DIVE, + 1 trivial positive + 1 trivial negative per class. Writes `data/probe_dataset/`. |
| `probe_trivials.py` | 226 | `TRIVIAL_POSITIVES: dict[str, str]` (10 hand-crafted minimal Solidity patterns) + `TRIVIAL_NEGATIVE: str` (1 clean OZ-style ERC20) + `bccc_class_to_sentinel()`. |
| `patterns/` | (sub-folder) | 10 per-class YAML pattern specs. **Documentation only — semantic_checker does NOT read them.** See `patterns/README.md`. |

**Sub-total: 2,709 lines** across 10 Python files.

## 3. Key concepts

### The 4 layers of the verification defense

1. **Semantic checker** — AST-level pattern matching via v9 graph features (fast, no Slither run)
2. **Tool validator** — Slither agreement per (class, contract) pair (corroborative, slow)
3. **FP estimator** — stratified (source × tier) sampling; upper-bound FP rate via Slither disagreement
4. **Negative checker** — Slither hit rate on NonVulnerable contracts (contamination check)

A 5th layer — the **gate** — combines all 4 into per-class verdicts (VERIFIED / PROVISIONAL / BEST-EFFORT / FAIL). A 6th — **report_generator** — writes the human-readable `verification_report.md`. A 7th — **slither_runner** — is the shared Slither infrastructure that tool_validator, fp_estimator, and negative_checker all use. And the 8th/9th — **probe_dataset** + **probe_trivials** — build the hand-curated contracts for model interpretability.

### `CLASS_TO_DETECTORS` — the canonical Slither mapping (`slither_runner.py:26-45`)

| Class | Slither detectors (v0.10) |
|-------|--------------------------|
| Reentrancy | `reentrancy-eth`, `reentrancy-no-eth`, `reentrancy-benign`, `reentrancy-events` |
| CallToUnknown | `low-level-calls`, `controlled-delegatecall`, `delegatecall-loop` |
| Timestamp | `timestamp` |
| **IntegerUO** | **(none — removed in Slither 0.10)** |
| MishandledException | `unchecked-send`, `unchecked-lowlevel`, `unchecked-transfer`, `return-bomb` |
| UnusedReturn | `unused-return` |
| ExternalBug | `arbitrary-send-eth`, `low-level-calls`, `tx-origin`, `controlled-delegatecall` |
| DenialOfService | `calls-loop`, `costly-loop`, `msg-value-loop` |
| GasException | `calls-loop`, `costly-loop`, `low-level-calls` |
| TransactionOrderDependence | `tx-origin`, `controlled-delegatecall` |

> ⚠ **IntegerUO has NO Slither detector in v0.10** — Solidity 0.8+ handles it natively. This means `tool_validator` skips IntegerUO entirely; `fp_estimator` skips it; only `semantic_checker` (via `feat[11] unchecked_block` or pre-0.8 solc) can corroborate.

`ALL_DETECTORS` (`slither_runner.py:48`) is the union across all classes — used by `negative_checker` for the contamination check.

### The semantic checker (`semantic_checker.py:136-185`)

Per-class feature check (no Slither needed):

| Class | Pass condition | Verdict code |
|-------|----------------|--------------|
| Reentrancy | `graph.has_cei_path == 1` (EXTERNAL_CALL BEFORE state WRITE) | PASS / FAIL / SKIP |
| Timestamp | `graph.x[:, 2].max() > 0.5` (feat[2] uses_block_globals) | PASS / FAIL / SKIP |
| IntegerUO | `graph.x[:, 11].max() > 0.5` (feat[11] unchecked_block) **OR** pre-0.8 solc | PASS / FAIL / SKIP |
| UnusedReturn / MishandledException | `graph.x[:, 7].max() > 0.5` (feat[7] return_ignored) | PASS / FAIL / SKIP |
| CallToUnknown / ExternalBug | any edge with `edge_attr == 11` (EXTERNAL_CALL) | PASS / FAIL / SKIP |
| **DoS, GasException, TOD** | (no v9 feature covers) | **NOT_EXTRACTABLE** |

3 of 10 classes are NOT_EXTRACTABLE from the v9 schema. For these, the tool_validator (Slither) is the only signal. NOT_EXTRACTABLE is **not the same as FAIL** — it means "can't verify from features alone; trust the source tier."

### The Slither runner with content-addressed cache (`slither_runner.py`)

```python
def run_on_contract(
    sha256: str, source: str, data_dir: Path,
    *, detectors: Optional[list[str]] = None, force: bool = False,
) -> Optional[SlitherFindings]:
    """Run Slither on a contract identified by sha256 + source.
    
    Looks up the preprocessed .sol and meta.json from
    data_dir/preprocessed/<source>/<sha256>.sol.
    
    Results are cached in data_dir/slither_cache/<source>/<sha256>.slither.json.
    """
```

- Uses Slither's **Python API** (not CLI subprocess) for performance
- Cache key includes the **list of detectors requested** — re-run with a new detector list is a fresh run for that detector (no false cache hit)
- `_get_detector_registry()` (line 67) builds the argument→class map from `slither.detectors.all_detectors` via `inspect.getmembers`. Reflective — picks up new detectors automatically when Slither is upgraded
- The `SlitherFindings.agrees_with_class(class_name)` method (line 106) is the per-class agreement check

**Cache key** (per `slither_runner.py:213-225`): `data_dir/slither_cache/<source>/<sha256>.slither.json` keyed on `(sha256, detectors_set)`. First run is slow (5–30s/contract for full Slither); cached runs are near-instant.

### The per-class gate (`gate.py:123-285`)

Produces per-class VERIFIED / PROVISIONAL / BEST-EFFORT / FAIL verdicts:

| Verdict | Criteria |
|---------|----------|
| **VERIFIED** | semantic pass rate > 90% AND no co-occurrence flag |
| | OR T0 (injection-verified) source with no semantic failures |
| | OR T0 source with no graph reps yet (ground truth by construction) |
| **PROVISIONAL** | semantic pass rate 60–90% |
| | OR T1 (expert-audited) source with no graph reps |
| | OR NOT_EXTRACTABLE class with T1 source |
| **BEST-EFFORT** | semantic pass rate 30–60% |
| | OR NOT_EXTRACTABLE class with T2+ source AND co-occurrence flag |
| | OR T3/T4 source with no graph reps |
| **FAIL** | semantic pass rate < 30% |
| | OR FP rate > 30% (from fp_estimator) |
| | OR co-occurrence flag on high-noise source |

**Hard gate**: any class with FAIL blocks downstream export. The override requires explicit `pipeline.verification.override_classes` in `config.yaml` with a documented reason.

**Negative checker FAIL** (hit rate > 10%) adds a special `__neg_check__` entry to `hard_fails` (also blocks export).

### The 99% DoS↔Reentrancy co-occurrence noise detector (`class_auditor.py:24-26`)

`CO_OCCUR_FLAG_THRESHOLD = 0.50` (50% — very conservative; BCCC was 99%). Computed as `P(class_b=1 | class_a=1)` for all ordered class pairs. Any pair with conditional probability > 50% is flagged in `AuditResult.flagged_pairs`. This is the **primary signal** that would have caught the BCCC failure mode in the first run.

### The FP estimator (`fp_estimator.py:147-225`)

**Stratified sampling by (source, tier)** — not just by class. The implementation (`_stratified_sample`, line 147) uses proportional allocation with deterministic remainder distribution:

1. Group all positives by `(source, tier)` cell
2. Compute proportional share per cell
3. Floor each share, distribute remainder to highest-remainder cells
4. Sample without replacement per cell; if underfilled, top up from global pool

**Per-stratum reporting** lets you see "T0 SolidiFI Reentrancy has 2% FP, T3 Slither-Audited Reentrancy has 40% FP" — the per-tier per-class breakdown is the operational signal.

**FP definition (v1)**: a sampled positive is a "likely FP" if NO Slither detector for that class fires on the contract. Upper-bound estimate (a positive that Slither misses is not necessarily a wrong label). The v2.1 enhancement is to compound Slither-disagreement with semantic-checker FAIL (the compound rate is the ground truth).

### The negative checker (`negative_checker.py:138-234`)

For every contract labeled NonVulnerable (all 10 sentinel classes value=0), runs Slither and reports the hit rate. Default thresholds: WARN > 5%, FAIL > 10% (BCCC had 41% — mass contamination).

Uses the **canonical `CLASS_TO_DETECTORS` union** (not a generic Slither run) — the point is to catch OZ-flagged patterns that should be false positives on clean code (SafeMath, Reentrancy in `nonReentrant`, etc.). A generic Slither run would flag too many FPs on clean code.

### The probe dataset (`probe_dataset.py` + `probe_trivials.py`)

**Per class**: 40 real contracts from the highest-quality source available + 1 trivial positive + 1 trivial negative. Total per class: 42.

Source priority for the 40 real contracts:
1. BCCC review_batches/ (KEEPs only) — 6 of 10 classes available (Reentrancy, CallToUnknown, DoS, Timestamp, ExternalBug, GasException)
2. DIVE positives (if a class is not in BCCC)
3. None available → class entry has 0 real contracts (trivial pos/neg still produced)

The BCCC review_batches CSVs have the contract source code embedded in the `source_snippet` column, so we can build the probe dataset without needing the BCCC source corpus on disk (BCCC is deferred).

**`probe_trivials.py`** has 10 hand-crafted trivial positives (one per class, the simplest possible Solidity contract that exhibits the pattern) + 1 trivial negative (a clean OZ-style ERC20). Used by the model interpretability suite (`ml/scripts/interpretability/`) to verify the model has learned the pattern, not surface features.

### The verification report (`report_generator.py:44-227`)

Human-readable markdown report aggregating everything. 9 sections:
1. Per-class gate (with verdict icon)
2. Per-class corpus stats (positives, prevalence, by source, by tier)
3. Co-occurrence matrix (flagged pairs)
4. Semantic check summary
5. Tool validation (Slither agreement)
6. FP estimation (stratified)
7. Negative checker (NonVulnerable contamination)
8. Hard failures
9. Known limitations

## 4. Public API

### `run_audit(data_dir) -> AuditResult` — `class_auditor.py:100-189`

```python
@dataclass
class CoOccurrencePair:
    class_a: str
    class_b: str
    count: int          # contracts where both are positive
    count_a: int        # contracts where class_a is positive
    rate: float         # P(b=1 | a=1)
    flagged: bool       # rate > 50%

@dataclass
class AuditResult:
    per_class: dict[str, ClassStats]
    co_occurrence: list[CoOccurrencePair]
    flagged_pairs: list[CoOccurrencePair]
    total_contracts: int
    duration_s: float
```

### `run_semantic_check(data_dir, *, limit_per_class=None) -> SemanticCheckResult` — `semantic_checker.py:188-283`

```python
class CheckVerdict(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    SKIP = "SKIP"
    NOT_EXTRACTABLE = "NOT_EXTRACTABLE"
```

### `run_tool_validation(data_dir, *, limit_per_class=None, force=False, only_classes=None) -> ToolValidationResult` — `tool_validator.py:139-273`

### `run_fp_estimation(data_dir, *, sample_size=50, seed=42, only_classes=None, force=False) -> FPEstimationResult` — `fp_estimator.py:228-335`

### `run_negative_check(data_dir, *, warn_threshold=0.05, fail_threshold=0.10, detectors=None, limit=None, force=False) -> NonVulnResult` — `negative_checker.py:147-234`

### `run_gate(audit, semantic, *, tool_validation=None, fp_estimation=None, negative_check=None) -> GateResult` — `gate.py:123-285`

```python
class Verdict(str, Enum):
    VERIFIED = "VERIFIED"
    PROVISIONAL = "PROVISIONAL"
    BEST_EFFORT = "BEST-EFFORT"
    FAIL = "FAIL"
```

### `generate_report(...) -> VerificationReport` + `.write(path)` — `report_generator.py:235-272`

### `run_on_contract(sha, source, data_dir, *, detectors=None, force=False) -> SlitherFindings | None` — `slither_runner.py:185-250`

Plus `CLASS_TO_DETECTORS: dict[str, list[str]]` and `ALL_DETECTORS: set[str]` (line 26-48) for direct import.

### `build_probe_dataset(*, data_dir=None, n_per_class=40, seed=42, output_dir=None, bccc_review_dir=None, add_trivial=True) -> ProbeDataset` — `probe_dataset.py:186-351`

`TRIVIAL_POSITIVES: dict[str, str]` + `TRIVIAL_NEGATIVE: str` from `probe_trivials.py:40-226`.

## 5. Inputs → outputs

| Input | Where | What |
|-------|-------|------|
| `data/labels/merged/*.labels.json` | Stage 3 | For all 4 checkers + the gate |
| `data/representations/<source>/<sha256>.pt` + `.rep.json` | Stage 2 | For semantic_checker (graph features) |
| `data/preprocessed/<source>/<sha256>.sol` + `.meta.json` | Stage 1b | For slither_runner (needs the .sol) |
| `docs/legacy/bccc_deep_dive/.../review_batches/review_class*.csv` | Frozen Phase 5 | For probe_dataset (BCCC KEEPs) |
| `data/raw/<source>/repo/` (DIVE only) | Stage 1a | For probe_dataset's DIVE fallback |

| Output | Where | What |
|--------|-------|------|
| `data/slither_cache/<source>/<sha256>.slither.json` | `slither_runner.py:246` | Per-contract Slither findings cache |
| `data/probe_dataset/<class>/{*.sol,trivial_*.sol,manifest.json}` | `probe_dataset.py` | Per-class real + trivial contracts + manifest |
| `data/verification/verification_report_<ts>.md` | `cli.py:466` | Human-readable aggregate report |

## 6. Pipeline interactions

| Stage | Direction | What |
|-------|-----------|------|
| Stage 3 (labeling) | ← | Reads `data/labels/merged/*.labels.json` |
| Stage 2 (representation) | ← | Reads `data/representations/<source>/<sha256>.pt` + `.rep.json` (semantic_checker only) |
| Stage 1b (preprocessing) | ← | Reads `data/preprocessed/<source>/<sha256>.sol` + `.meta.json` (slither_runner) |
| `cli.py` | ↔ | `_run_verify` at line 381 is the CLI wiring; runs all 5 sub-components in sequence with configurable skip flags |
| BCCC review_batches (frozen) | ← | probe_dataset (BCCC KEEPs only) |
| Stage 5 (splitting) | ← | Trusts the gate verdict; if FAIL on any class, blocks export (split manifest would still be generated but downstream trainer reads the gate verdict) |
| `sentinel_data.analysis.feature_dist` | ↔ | Re-uses the same `(sha, source)` lookup pattern; CFG artifacts from `--emit-cfg` are an optional input |

## 7. Tests

**Location:** `data_module/tests/test_verification/`
- `test_class_auditor.py` — co-occurrence matrix correctness, threshold flagging
- `test_semantic_checker.py` — per-class PASS/FAIL/NOT_EXTRACTABLE verdicts
- `test_tool_validator.py` — Slither agreement, NO_DETECTOR skip, error handling
- `test_fp_estimator.py` — stratified sampling, FP rate math, threshold fail
- `test_negative_checker.py` — hit rate, status thresholds
- `test_gate.py` — every verdict path, hard fail list, negative_check FAIL blocks
- `test_report_generator.py` — markdown structure, all 9 sections
- `test_patterns.py` — every YAML parses, every class has a YAML
- `test_probe_dataset.py` — 40+2 per class, BCCC KEEP filter, trivial contract contents
- `test_cli_verify.py` — CLI argument parsing, all --skip flags honored
- `test_bccc_regression.py` — the BCCC Phase 5 regression test: the new `verification/` module on the legacy BCCC corpus must produce a `verification_report.md` that matches the Phase 5 report (per-class drop counts within ±0.5%, per-class gate verdicts match exactly)
- `test_smartbugs_recall.py` — recall on SmartBugs Curated (≥ 70% on the major classes)

**Command:**
```bash
cd ~/projects/sentinel/data_module
poetry run pytest tests/test_verification/ -v
```

## 8. See also

- Patterns: `sentinel_data/verification/patterns/README.md`
- Previous stage: `sentinel_data/labeling/README.md`
- Next stage: `sentinel_data/splitting/README.md`
- CLI entry: `sentinel_data/cli.py` (`_run_verify` at line 381)
- The Slither detector list: `slither_runner.CLASS_TO_DETECTORS` (the source of truth)
- v2.1 roadmap: `docs/proposal/Data_Module_Proposals/actionable_plans/05_stage_4_verification.md`
- 99% DoS↔Reentrancy rule: `class_auditor.py:24-26` + `merger.py:100-124`
- BCCC failure retrospective: `docs/legacy/bccc_deep_dive/`
- Slither API drift: ADR-0002 + `freshness.py:79-93` (the `slither-analyzer` version check)
- Probe dataset consumer: `ml/scripts/interpretability/` (model-side scripts)
