# `sentinel_data.verification` — The BCCC-Failure Catcher

## What This Module Does

The verification module is Stage 5 of the SENTINEL data pipeline. It asks the question that BCCC failed to ask for 14 days of work: **"are these labels correct?"**

The module implements 6 verification components that form a layered defense against false positives:

1. **Semantic Checker** — AST-level pattern matching that verifies each label has code-level evidence
2. **Tool Validator** — runs Slither (and optionally Mythril/Semgrep) to corroborate labels
3. **FP Estimator** — samples positives per class, runs all tools, reports empirical false positive rate
4. **Class Auditor** — per-class statistics, per-source breakdown, co-occurrence matrix
5. **Negative Checker** — verifies NonVulnerable contracts are actually clean
6. **Probe Dataset** — hand-curated 40-contracts-per-class set for model interpretability

## Why This Matters

The BCCC dataset had an **89% false positive rate for Reentrancy** and **86.9% for CallToUnknown**. These weren't subtle errors — they were massive label quality failures that would have been caught in minutes by the semantic checker:

- **Reentrancy FP pattern:** The BCCC pattern matched any external call + state write, even if the state write was BEFORE the call (which is not a reentrancy)
- **CallToUnknown FP pattern:** The BCCC pattern matched any `.call{}` without checking if the target was actually unknown

The verification module would have caught both of these in the first run, preventing the 14-day debugging session.

## Architecture Overview

```
Labeled contracts (.labels.json)
        │
        ▼
┌──────────────────────────────────────────────────┐
│            6 Verification Components              │
│                                                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────┐│
│  │ semantic_    │  │ tool_        │  │ fp_      ││
│  │ checker      │  │ validator    │  │ estimator││
│  └──────┬───────┘  └──────┬───────┘  └────┬─────┘│
│         │                 │               │       │
│  ┌──────┴───────┐  ┌──────┴───────┐  ┌────┴─────┐│
│  │ class_       │  │ negative_    │  │ probe_   ││
│  │ auditor      │  │ checker      │  │ dataset  ││
│  └──────────────┘  └──────────────┘  └──────────┘│
└──────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────┐
│          Per-Class Gate (VERIFIED/PROVISIONAL/     │
│          BEST-EFFORT/FAIL)                         │
│          + verification_report.md                  │
└──────────────────────────────────────────────────┘
```

## The 6 Verification Components

### 1. Semantic Checker (`semantic_checker.py`)

The most important component. For each (class, contract) pair, it asks: "does the contract's AST actually contain the code pattern implied by the class label?"

**Per-class patterns:**

| Class | Pattern | What It Catches |
|-------|---------|-----------------|
| Reentrancy | External call BEFORE state write (CEI ordering enforced) | 89% of BCCC's Reentrancy FPs |
| CallToUnknown | `.call{}` / `.delegatecall{}` with address target, NOT in OZ SafeERC20 wrapper | 86.9% of BCCC's CallToUnknown FPs |
| Timestamp | `block.timestamp` / `now` in a conditional | Timestamp dependence |
| IntegerUO | Arithmetic op in pre-0.8 or `unchecked{}` block | Integer overflow/underflow |
| ExternalBug | Cross-contract call to non-interface target | External vulnerability |
| GasException | Unchecked `send()` / `transfer()` | Gas griefing |
| MishandledException | Call with unused return value | Exception suppression |
| UnusedReturn | Internal function call with unused return | Similar to MishandledException |
| DoS | Loop with external call or unbounded iteration | Denial of service |
| TOD | `tx.origin` in permission check | Transaction order dependence |

The patterns are defined in `sentinel_data/verification/patterns/<class>.yaml` — one file per class, human-authored and reviewed.

### 2. Tool Validator (`tool_validator.py`)

Runs Slither (default) and optionally Mythril/Semgrep on every labeled positive. Reports the per-class agreement rate: "of 100 Reentrancy positives, how many does Slither also flag?"

**Key insight:** Tool agreement is **corroborative, not authoritative**. Slither has a 51.97% precision on reentrancy — so low agreement doesn't mean the label is wrong, and high agreement doesn't mean it's right. The tool validator is one signal among many.

### 3. FP Estimator (`fp_estimator.py`)

Samples N positives per class (default 50), runs all tools on the sample, and reports an empirical false positive rate. The sampling is **stratified by source AND confidence tier** — a class with 90% T3 labels has a very different FP rate than one with 90% T0 labels.

### 4. Class Auditor (`class_auditor.py`)

Produces per-class statistics:
- Total labeled positives
- Per-source breakdown
- Per-confidence-tier breakdown
- **Co-occurrence matrix** (directed + conditional form)

The co-occurrence matrix is the primary output that catches the 99% DoS↔Reentrancy pattern. It shows: "if class X is positive, what's the probability class Y is also positive?"

### 5. Negative Checker (`negative_checker.py`)

For every contract labeled `NonVulnerable`, runs Slither and reports what fraction has at least one tool hit. The threshold is **5%** (default) — anything above is flagged.

This catches the BCCC pattern where 41% of "NonVulnerable" contracts had Slither hits.

### 6. Probe Dataset (`probe_dataset.py`)

A hand-curated set of ~40 contracts per class where the vulnerability is visually obvious in the code. The model interpretability suite uses this set to verify the model has learned the right patterns, not shortcuts.

Each class gets:
- 40 real audit contracts (from Phase 5's `review_batches/`)
- 1 "trivial positive" (the simplest possible example)
- 1 "trivial negative" (a clean OZ contract of similar size)

## The Per-Class Gate

The verification produces a per-class gate:

| Gate | Criteria | Behavior |
|------|----------|----------|
| **VERIFIED** | Semantic check > 90%, tool agreement > 70%, FP estimate < 15% | Export allowed |
| **PROVISIONAL** | Semantic check 60–90%, tool agreement > 50%, FP estimate < 30% | Export with warning |
| **BEST-EFFORT** | Semantic check 30–60% | Export with strong warning |
| **FAIL** | Semantic check < 30% OR FP estimate > 30% | **Export blocked** |

**Hard gate:** Any class with `FAIL` blocks downstream export. The override requires explicit `pipeline.verification.override_classes` in `config.yaml` with a documented reason.

## The BCCC Regression Test

The module's critical test is reproducing the Phase 5 BCCC verification:

1. Run the new `verification/` module on the legacy BCCC corpus
2. Compare the output `verification_report.md` to the Phase 5 report
3. Per-class drop counts must match within ±0.5%
4. Per-class gate verdicts must match exactly

This is the proof that the module would have caught the BCCC failure in hours rather than weeks.

## How to Use

```bash
# Verify all enabled sources
sentinel-data verify

# Verify a single source
sentinel-data verify --source smartbugs_curated

# Strict mode (fail on PROVISIONAL gates)
sentinel-data verify --strict

# Dry-run
sentinel-data verify --dry-run
```

## Pipeline Position

```
Stage 4: Labeling (assign vulnerability classes)
    ↓
Stage 5: Verification ← YOU ARE HERE (are these labels correct?)
    ↓
Stage 6: Splitting + Registry (train/val/test splits)
```

## Design Decisions

1. **Per-class, not per-source** — a "good" Tier-1 source can compensate for a "bad" Tier-4 source
2. **AST patterns are human-authored** — not auto-generated from tool output
3. **Tool corroboration, not authority** — Slither's 51.97% reentrancy precision means it can't be the ground truth
4. **Sampling-based FP estimation** — running every tool on every contract is too expensive
5. **Hard gate on FAIL** — prevents the BCCC class of failure from reaching training
6. **Negative checker at 5%** — catches the "NonVulnerable has Slither hits" pattern early
