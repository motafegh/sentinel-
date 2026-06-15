# `sentinel_data.verification.patterns` — Class-Specific YAML Patterns (v1 stub)

> **Status: 10 YAML files present, NOT YET consumed by verification code.** The semantic_checker in `verification/semantic_checker.py` uses **graph features directly** (feat[2] uses_block_globals, feat[7] return_ignored, edge_attr=11 EXTERNAL_CALL) rather than parsing these YAMLs. These files are kept as a human-readable pattern spec for reference and for the planned v2.1 tool_validator pattern overlay.

## 1. Purpose

This subpackage holds the **per-class semantic pattern definitions** — one YAML per canonical class, written by humans (not auto-generated). The intent (per the original Stage 4 design) was to make the verification rules **declarative, version-controlled, and auditable** rather than buried in Python code.

The current implementation in `verification/semantic_checker.py` does **not read these files** — it uses the v9 graph features directly (which is faster and avoids the regex maintenance burden). These YAMLs remain as:

1. A human-readable spec of "what does a positive Reentrancy / DoS / … look like"
2. Documentation for the v2.1 plan to overlay pattern-based rules on top of the feature-based checker
3. A reference for the probe_dataset trivial positives (which were written to match these patterns)

> **The source of truth for what each class *means* in this codebase is in two places:** the YAML here, and the `probe_trivials.py:40-159` trivial-positive examples. They should agree (and they do for the 10 classes that have entries in both).

## 2. Source map

| File | Approx lines | Role |
|------|--------------|------|
| `CallToUnknown.yaml` | 42 | Pattern: `.call{}` / `.delegatecall{}` / `.send{}` to dynamic address, return ignored, no SafeERC20 wrapper |
| `DenialOfService.yaml` | 37 | Pattern: unbounded loop with external call, or single-revert blocks payout |
| `ExternalBug.yaml` | 34 | Pattern: cross-contract call to non-interface target, or `tx.origin` in permission check |
| `GasException.yaml` | 26 | Pattern: unchecked `send()` / `transfer()`, large calldata, or OOG-prone loop |
| `IntegerUO.yaml` | 39 | Pattern: arithmetic op in pre-0.8 Solidity, or in `unchecked{}` block in 0.8.x |
| `MishandledException.yaml` | 30 | Pattern: low-level call with discarded `(bool, bytes)` return |
| `Reentrancy.yaml` | 38 | Pattern: external call before state write (CEI violation) |
| `Timestamp.yaml` | 30 | Pattern: `block.timestamp` / `now` in security-sensitive conditional |
| `TransactionOrderDependence.yaml` | 30 | Pattern: `tx.origin` in permission check, or approve/transferFrom race |
| `UnusedReturn.yaml` | 29 | Pattern: internal function call with discarded return value |

**Sub-total: ~335 lines** across 10 YAML files.

## 3. Key concepts

### The schema (per YAML, inferred from `Reentrancy.yaml` and `CallToUnknown.yaml`)

Each YAML follows a consistent shape (different files may have slightly different `v9_signal` sub-fields depending on the signal type):

```yaml
class: Reentrancy
description: >
  External call before state update (CEI violation). An attacker's fallback
  function can re-enter the vulnerable function before state is updated.

v9_signal:
  method: graph_attribute       # one of: graph_attribute, graph_edge, graph_feature, composite, NOT_EXTRACTABLE
  field: has_cei_path           # for graph_attribute: the field name on the PyG Data object
  positive_value: 1             # the value that indicates a positive signal
  note: >
    Computed by _compute_has_cei_path() in graph_extractor.py.

false_positive_risk:
  level: HIGH                   # LOW / MEDIUM / HIGH / VERY_HIGH
  bccc_fp_rate: "89%"           # historical FP rate in BCCC dataset
  bccc_root_cause: >            # why BCCC got it wrong
    BCCC matched any external call + state write regardless of ordering.

positive_example: |
  function withdraw(uint amount) public {
      require(balances[msg.sender] >= amount);
      msg.sender.call{value: amount}("");
      balances[msg.sender] -= amount;
  }

negative_example: |
  function withdraw(uint amount) public {
      require(balances[msg.sender] >= amount);
      balances[msg.sender] -= amount;
      msg.sender.call{value: amount}("");
  }

dasp_id: 1                     # DASP taxonomy mapping
tier_for_solidifi: T0          # confidence tier for SolidiFI source
```

> **Note on `v9_signal` variants** — the `method` field determines the sub-fields:
> - `graph_attribute` → `field`, `positive_value`, `note`
> - `graph_edge` → `edge_type`, `edge_id`, `note`
> - `graph_feature` → `feature_index`, `note`
> - `composite` → `signals` (list of sub-signals)
> - `NOT_EXTRACTABLE` → no additional fields (the class cannot be verified from v9 features alone)

### What `semantic_checker.py` actually uses (per `verification/semantic_checker.py`)

The current implementation reads graph features directly — **not these YAMLs**. Per-class check logic:

| Class | Pass condition | Source |
|-------|----------------|--------|
| Reentrancy | `graph.has_cei_path == 1` | `semantic_checker.py:138-146` |
| Timestamp | `graph.x[:, 2].max() > 0.5` (feat[2] uses_block_globals) | `semantic_checker.py:148-154` |
| IntegerUO | `graph.x[:, 11].max() > 0.5` (feat[11] unchecked_block) OR pre-0.8 solc | `semantic_checker.py:156-165` |
| UnusedReturn / MishandledException | `graph.x[:, 7].max() > 0.5` (feat[7] return_ignored) | `semantic_checker.py:167-173` |
| CallToUnknown / ExternalBug | any edge with `edge_attr == 11` (EXTERNAL_CALL) | `semantic_checker.py:175-181` |
| DenialOfService / GasException / TransactionOrderDependence | `NOT_EXTRACTABLE` (no v9 feature covers them) | `semantic_checker.py:183-185` |

Three of the ten classes (DoS, GasException, TOD) are **NOT_EXTRACTABLE** from the v9 schema. For these, the `tool_validator` (Slither) is the only signal.

## 4. Public API

**This subpackage has no Python code.** It is a **data-only** directory. Public consumption is via `yaml.safe_load(open(path))` from outside (currently nobody does this — the patterns are documentation-only).

> **A pattern loader exists in the Stage 4 plan** (`sentinel_data.verification.patterns.PatternLoader.load(class_name) -> dict`) but is **not yet implemented** as of 2026-06-11. The semantic_checker reads graph features directly.

## 5. Inputs → outputs

| Input | Where | What |
|-------|-------|------|
| `*.yaml` files in this directory | Human-authored | Pattern specs |

| Output | Where | What |
|--------|-------|------|
| *(none — no code consumes them yet)* | | |

## 6. Pipeline interactions

| Stage | Direction | What |
|-------|-----------|------|
| `verification/semantic_checker.py` | ✗ | Does NOT read these (uses graph features) |
| `verification/tool_validator.py` | ✗ | Reads Slither detectors (`slither_runner.CLASS_TO_DETECTORS`), not these patterns |
| `verification/probe_trivials.py` | ↔ | The trivial positives were written to match these patterns (manual cross-check) |
| Future `patterns/loader.py` (v2.1) | → | Will read these to overlay pattern-based rules on top of feature checks |

## 7. Tests

**Location:** `data_module/tests/test_verification/test_patterns.py`

**Command:**
```bash
cd ~/projects/sentinel/data_module
poetry run pytest tests/test_verification/test_patterns.py -v
```

**Coverage (inferred — file not yet read):**
- Every `*.yaml` is parseable YAML
- Every class in the taxonomy has a corresponding `*.yaml`
- The trivial positives in `probe_trivials.py` match the positive_example in each YAML (human cross-check, automated where possible)

## 8. See also

- Parent: `sentinel_data.verification/README.md`
- The actual semantic check: `sentinel_data.verification.semantic_checker`
- The probe dataset trivial positives: `sentinel_data.verification.probe_trivials`
- Slither detector mapping (the *other* pattern source): `sentinel_data.verification.slither_runner.CLASS_TO_DETECTORS`
- BCCC review batches (human-verified positive examples): `data_module/docs/legacy/bccc_deep_dive/Phase5_LabelVerification_2026-06-08/outputs/review_batches/`
- Stage 4 plan: `docs/proposal/Data_Module_Proposals/actionable_plans/05_stage_4_verification.md`
