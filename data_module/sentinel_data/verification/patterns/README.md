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
| `CallToUnknown.yaml` | ~15 | Pattern: `.call{}` / `.delegatecall{}` / `.send{}` to dynamic address, return ignored, no SafeERC20 wrapper |
| `DenialOfService.yaml` | ~15 | Pattern: unbounded loop with external call, or single-revert blocks payout |
| `ExternalBug.yaml` | ~15 | Pattern: cross-contract call to non-interface target, or `tx.origin` in permission check |
| `GasException.yaml` | ~15 | Pattern: unchecked `send()` / `transfer()`, large calldata, or OOG-prone loop |
| `IntegerUO.yaml` | ~15 | Pattern: arithmetic op in pre-0.8 Solidity, or in `unchecked{}` block in 0.8.x |
| `MishandledException.yaml` | ~15 | Pattern: low-level call with discarded `(bool, bytes)` return |
| `Reentrancy.yaml` | ~15 | Pattern: external call before state write (CEI violation) |
| `Timestamp.yaml` | ~15 | Pattern: `block.timestamp` / `now` in security-sensitive conditional |
| `TransactionOrderDependence.yaml` | ~15 | Pattern: `tx.origin` in permission check, or approve/transferFrom race |
| `UnusedReturn.yaml` | ~15 | Pattern: internal function call with discarded return value |

**Sub-total: ~150 lines** (each YAML is small — single-pattern spec with regex, positive/negative examples, structural features).

## 3. Key concepts

### The schema (per YAML, inferred from the simplest entry)

Each YAML follows a similar shape (NOT all files are byte-identical — older files may have simpler structure):

```yaml
class: Reentrancy
schema_version: "1"

description: >
  An external call is made before state is updated (CEI violation),
  allowing the callee to re-enter and manipulate the contract's state.

# Structural features that should fire (from v9 graph features):
v9_features_should_fire:
  - feat_2: 0         # uses_block_globals — should NOT fire for reentrancy
  - has_cei_path: 1   # the new (v9) structural signal: external call BEFORE state write
  - edge_attr_11: 1   # at least one EXTERNAL_CALL edge (cross-contract call)

# Anti-features (should NOT fire for a clean reentrancy label):
v9_features_should_not_fire:
  - has_reentrancy_guard: 1  # presence of nonReentrant modifier = probably not vulnerable

# Regex patterns in source text (optional, for source-level grep):
source_patterns:
  - pattern: '\\.call\\s*\\{'
    description: "low-level call to dynamic target"

# Positive example (BCCC review batch snippet):
positive_example: |
  function withdraw(uint amount) public {
      require(balances[msg.sender] >= amount);
      (bool ok,) = msg.sender.call{value: amount}("");
      require(ok);
      balances[msg.sender] -= amount;   // ← state write AFTER call = vulnerable
  }

# Negative example (clean OZ-style):
negative_example: |
  function transfer(address to, uint amount) external {
      _balances[msg.sender] -= amount;   // ← state write BEFORE call = safe
      _balances[to] += amount;
      emit Transfer(msg.sender, to, amount);
  }
```

> **The exact schema is informal** — different files were authored at different times. The Stage 4 plan D-4.1 says "pattern YAMLs are human-authored, version-controlled" but the schema version field is the only enforced contract.

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
