# BCCC-SCsVul-2024 — Phase 3 Plan: Deep Analysis, AutoML, Label Verification

**Date:** 2026-06-06
**Prerequisite:** Phase 1 (inventory) + Phase 2 (cleaning) complete.
**Goal:** Rigorous, tool-assisted validation of the cleaned dataset (67,311 contracts), deep EDA, feature engineering, AutoML baselines, and a head-to-head comparison of static-analysis tools, AutoML, and SENTINEL's GNN+CodeBERT — so that every label, feature, and modeling choice downstream is defensible.
**Navigation:** [`README.md`](README.md) — root table of contents for the whole BCCC deep dive
**Phase 3 entry point:** [`Phase3_DeepAnalysis_2026-06-06/README.md`](Phase3_DeepAnalysis_2026-06-06/README.md)
**Current headline results (Session 1+2):** [`Phase3_DeepAnalysis_2026-06-06/outputs/ws_i_agreement_report.md`](Phase3_DeepAnalysis_2026-06-06/outputs/ws_i_agreement_report.md) (Reentrancy P=0.93, CallToUnknown P=0.92, IntegerUO needs Aderyn)
**Mode:** Read-only on `BCCC-SCsVul-2024/`; reads `Data/Deep_Dive/.../Phase2_Validation_2026-06-06/outputs/`; writes only to `Data/Deep_Dive/.../Phase3_DeepAnalysis_2026-06-06/`.
**Out of scope (deferred to Phase 4+):** SENTINEL model retraining, on-chain/ZK work, agent-layer changes, BCCC paper re-derivation.

---

## 0. Scope & Deliverables

Phase 3 produces **12 workstreams (WS-I through WS-T)** that fall into 4 families:

| Family | Workstreams | Primary output |
|---|---|---|
| **Label verification** (correctness) | WS-I, WS-O, WS-N, WS-S | `labels/manual_inspections.md`, `labels/tool_consensus.csv`, `labels/dropped_review_analysis.md` |
| **Feature engineering** (model inputs) | WS-K, WS-P, WS-M | `features/contracts_clean_v12.csv` (slither+source features) |
| **Statistical EDA** (rigor) | WS-J | `eda/eda_report.md` + 12 PNG figures |
| **Modeling baselines** (SENTINEL ceiling) | WS-L, WS-Q, WS-R, WS-T | `automl/automl_report.md`, `automl/shap_*.png`, `automl/three_way_comparison.md` |

**12 workstreams × 3-5 hours each = 30-50 hours total.** Realistic split: 4-6 focused sessions, can run in parallel where dependencies allow.

---

## 1. Environment & Tool Inventory

### 1.1 Currently Available (verified 2026-06-06)

| Tool | Version | Path | Used by | Status |
|---|---|---|---|---|
| **slither-analyzer** | 0.11.5 (root), 0.10.0 (ml) | `.venv/bin/slither` (CLI broken — use Python API) | WS-I, WS-O, WS-K, WS-P | 101 detectors confirmed importable |
| **solc-select** | 1.2.0 | `ml/.venv/bin/solc-select` | All compile-dependent WS | 0.4.0 - 0.8.35 installed (current=0.4.24) |
| **slither Python API** | n/a | `from slither import Slither` | WS-I, WS-O, WS-K, WS-P | ✅ Works in root `.venv` |
| **Poetry** | 1.8.2 | `/usr/bin/poetry` | WS-O (mythril), WS-L (xgboost/lightgbm) | Available; all installs via Poetry |
| **pyproject.toml** | n/a | `pyproject.toml` | All WS | Has: slither 0.11.5, solc-select, sklearn, pandas, numpy, torch, transformers, matplotlib, seaborn |
| **scikit-learn** | 1.8.0 | `.venv` | WS-L, WS-M, WS-Q, WS-T | ✅ |
| **pandas / numpy** | 2.2 / 1.26 | `.venv` | All WS | ✅ |
| **matplotlib / seaborn** | 3.10 / 0.13 | `.venv` | WS-J (visualizations) | ✅ |
| **torch / transformers** | 2.10 / 5.1 | `.venv` | (deferred) | ✅ |

### 1.2 To Install via Poetry (before WS-O, WS-L start)

```bash
# Add to pyproject.toml — these are research-side dev-only
poetry add --group research xgboost lightgbm catboost optuna shap
poetry add --group research imbalanced-learn  # SMOTE, undersampling
poetry add --group research graphviz pydot    # slither graphviz dot export
poetry add --group research pydriller         # git history analysis (for SENTINEL vs old model comparison)

# Mythril — symbolic execution (large, requires docker fallback or native build)
poetry add --group research mythril
# If mythril install fails (common on WSL), use Docker:
#   docker pull mythril/myth && docker run -v $(pwd):/tmp mythril/myth analyze /tmp/contract.sol
# Or fall back to: mythril-analyzer (lighter CLI subset)
```

**D-P3-1: Mythril install → SUPERSEDED by D-P3-10 (Aderyn).** mythril works via Docker (`mythril/myth:latest`) but is too slow (3m/contract) for batch analysis. Container `mythril-sweep` retained for ad-hoc deep analysis if needed. The `solc-bin.ethereum.org` download URL is unreachable from WSL/Docker, so solc binaries are downloaded manually from GitHub releases via Python `urllib.request.urlretrieve` inside the container. 5 versions installed (0.4.26, 0.5.17, 0.6.12, 0.7.6, 0.8.28) at `/home/mythril/.solcx/solc-v{VERSION}` (flat-file layout, the format solcx expects).

**D-P3-10 (NEW, 2026-06-06): Aderyn replaces mythril as the 2nd static-analysis tool.** [Aderyn](https://github.com/Cyfrin/aderyn) by Cyfrin — Rust-based Solidity static analyzer. v0.6.8 (Jan 2026), 45K+ downloads, 88 detectors including reentrancy (`reentrancy-state-change`), weak-randomness, tx-origin, selfdestruct, unchecked-low-level-call, unchecked-send, dangerous-unary-operator, divide-before-multiplication, state-change-without-event, etc. **~3 seconds per contract** on the BCCC test contract (vs mythril's 3m16s — ~65x faster). Installed on host at `~/.cargo/bin/aderyn` via `curl ... | bash`. **Output formats:** JSON, Markdown, Sarif. **Caveat:** Aderyn does NOT have a dedicated integer-overflow detector for pre-0.8 contracts (slither fills that gap). **Test result (contract 00001c83…):** 88 detectors run, 7 low-severity issues found, completed in 3.1s.

### 1.3 Not Used in Phase 3 (with rationale)

- **securify** — academic tool, deprecated since 2020, only runs on solc 0.5.x; skip.
- **manticore** — symbolic execution, requires separate Python env setup; skip in favor of slither + aderyn.
- **echidna** — fuzzing-based, requires test harness per contract; not appropriate for label verification at 67K scale.
- **mythx** — closed-source SaaS, requires API key; skip.
- **solhint** — linter, not security detector; skip.
- **crytic-compile** — already pulled in as slither dependency; no separate install.
- **mythril** — DOCKER, reserved for ad-hoc deep analysis (D-P3-1). Too slow for batch (3m/contract × 5000 = 250h). Use Aderyn (D-P3-10) for batch.

### 1.4 Second Static Analyzer: Aderyn (added 2026-06-06 per D-P3-10)

Aderyn is the **second cross-tool** alongside slither for WS-O. It provides an independent (non-Crytic) implementation, which is valuable for the 2-way consensus — two different static analyzers agreeing on a label is much stronger evidence than two implementations from the same family.

**Tool comparison:**

| Tool | Engine | Detectors | Speed/contract | Int overflow | Reentrancy | TxOrigin | Weak PRNG |
|---|---|---:|---:|:---:|:---:|:---:|:---:|
| **slither 0.11.5** | Python (Slither IR) | 101 | 2-5s | ✅ | ✅ | ✅ | ✅ |
| **Aderyn 0.6.8** | Rust (AST) | 88 | ~3s | ❌ (pre-0.8 gap) | ✅ | ✅ | ✅ |
| mythril (Docker) | Python (Z3 symbolic exec) | 17 | **~3min** | ✅ | ⚠️ (didn't fire on test) | ✅ | ✅ |

**Key gaps filled by dual-tool approach:**
- Slither gives integer-overflow coverage; Aderyn gives faster breadth (88 vs 101 detectors but runs in Rust).
- Aderyn covers some detectors slither doesn't (e.g., `redundant-statement`, `state-change-without-event`, `enumerable-loop-removal`, `out-of-order-retryable`).
- The 2-way consensus (slither + Aderyn) on 5,000 contracts provides robust label verification at ~8 hours total.

---

## 2. Slither Detector → BCCC Class Mapping (101 detectors → 10 classes)

The 101 slither detectors map to BCCC's 10 SENTINEL classes as follows (preliminary; WS-I will refine with empirical evidence).

### 2.1 Mapping Table (best-effort)

| BCCC Class | Slither Detectors | Coverage | Notes |
|---|---|---|---|
| **Class01:ExternalBug** | `ArbitrarySendEth`, `ArbitrarySendErc20NoPermit`, `ArbitrarySendErc20Permit`, `ControlledDelegateCall`, `DelegatecallInLoop` | Strong | Most external-call-related bugs |
| **Class02:GasException** | `ConstantFunctionsAsm`, `ConstantFunctionsState`, `VoidConstructor`, `LockedEther`, `MissingEventsArithmetic` | Medium | Locked-ether overlap with MishandledException; assign by primary signal |
| **Class03:MishandledException** | `IncorrectReturn`, `ReturnInsteadOfLeave`, `UninitializedStateVars`, `UninitializedStorageVars`, `UninitializedLocalVars`, `MappingDeletionDetection`, `AssertStateChange`, `ArrayLengthAssignment`, `WriteAfterWrite`, `ReturnBomb`, `DomainSeparatorCollision` | Strong | Most error-handling bugs |
| **Class04:Timestamp** | `Timestamp`, `BadPRNG`, `GelatoUnprotectedRandomness` | Strong | Direct match |
| **Class05:TransactionOrderDependence** | (DROPPED per D-F1) | n/a | Not in SENTINEL v9 |
| **Class06:UnusedReturn** | `UncheckedTransfer`, `UncheckedSend`, `UncheckedLowLevel`, `UnusedReturnValues` | Strong | Direct match |
| **Class07:WeakAccessMod** | `Suicidal`, `TxOrigin`, `UnprotectedUpgradeable` | Medium | TxOrigin & Suicidal are classic access-mod issues |
| **Class08:CallToUnknown** | `MissingZeroAddressValidation`, `UninitializedFunctionPtrsConstructor` | Medium | Maps roughly to "calling untrusted addresses" |
| **Class09:DenialOfService** | `MultipleCallsInLoop`, `MsgValueInLoop`, `CostlyOperationsInLoop` | Strong | Direct match |
| **Class10:IntegerUO** | `DivideBeforeMultiply`, `IncorrectOperatorExponentiation`, `IncorrectStrictEquality`, `BooleanEquality`, `BooleanConstantMisuse`, `TypeBasedTautology`, `TautologicalCompare`, `ShiftParameterMixup`, `TooManyDigits`, `IncorrectUnaryExpressionDetection`, `StorageSignedIntegerArray`, `IncorrectUsingFor`, `ABIEncoderV2Array`, `ArrayByReference` | Strong | Most arithmetic/bounds issues |
| **Class11:Reentrancy** | `ReentrancyEth`, `ReentrancyNoGas`, `ReentrancyEvent`, `ReentrancyBalance`, `ReentrancyBenign`, `ReentrancyReadBeforeWritten` | Strong | All reentrancy variants |
| **Class12:NonVulnerable** | (no detector should fire; tracked as "0 detections from any of the above") | n/a | Used as 1-vs-rest for binary "is buggy" classifier |

### 2.2 Detectors NOT Mapped to Any BCCC Class (benign / quality-only)

These detectors do not match BCCC's 10 vulnerability classes but are useful for general code quality:

`Backdoor`, `ConstantPragma`, `IncorrectSolc`, `ReentrancyBenign`, `UnusedStateVars`, `CouldBeConstant`, `CouldBeImmutable`, `Assembly`, `LowLevelCalls` (overlaps with Class06), `NamingConvention`, `ExternalFunction`, `ShadowingAbstractDetection`, `StateShadowing`, `LocalShadowing`, `BuiltinSymbolShadowing`, `IncorrectERC20InterfaceDetection`, `IncorrectERC721InterfaceDetection`, `UnindexedERC20EventParameters`, `DeprecatedStandards`, `RightToLeftOverride`, `UnimplementedFunctionDetection`, `RedundantStatements`, `PredeclarationUsageLocal`, `MissingEventsAccessControl`, `ModifierDefaultDetection`, `MissingInheritance`, `EnumConversion`, `MultipleConstructorSchemes`, `PublicMappingNested`, `ReusedBaseConstructor`, `DeadCode`, `ProtectedVariables`, `CyclomaticComplexity`, `CacheArrayLength`, `EncodePackedCollision`, `OutOfOrderRetryable`, `ChronicleUncheckedPrice`, `FunctionInitializedState`.

**Used in WS-K** for general code quality features (24 detectors).

### 2.3 Mythril Detectors (DOCKER — reserved for ad-hoc use, NOT batch)

Mythril's EVM-level symbolic execution covers: reentrancy (`StateChangeAfterCall`), integer overflow (`IntegerArithmetics`), delegatecall (`ArbitraryDelegateCall`), suicide (`AccidentallyKillable`), unchecked send (`UncheckedRetval`), tx.origin (`TxOrigin`), timestamp dependence (`PredictableVariables`), weak PRNG, default visibility, locked ether (`UnexpectedEther`), function state mutability, gas exhaustion. ~17 detectors, complementary to slither.

**Mythril Docker setup (verified working 2026-06-06, container `mythril-sweep`):**
- Image: `mythril/myth:latest` (v0.24.8)
- `solc-bin.ethereum.org` unreachable from WSL/Docker → solc binaries downloaded manually from GitHub releases
- 5 solc versions installed in container at `/home/mythril/.solcx/solc-v{VERSION}` (flat-file layout that solcx expects):
  - 0.4.26, 0.5.17, 0.6.12, 0.7.6, 0.8.28
- Download method: `python3 -c "import urllib.request; urllib.request.urlretrieve('https://github.com/ethereum/solidity/releases/download/v{VER}/solc-static-linux', '/home/mythril/.solcx/solc-v{VER}')"`
- **Performance:** ~3m/contract for full analysis (depth-unbounded). See WS-O benchmark table for trade-offs.

---

## 3. Workstreams

### WS-I: Manual + Slither-Assisted Label Validation

**Goal:** Resolve the 766 NV+vuln contradictions, the 2 nine-folder "maxing" contracts, and a stratified sample of multi-positive contracts, using **slither output as evidence**.

**Why:** Phase 2 flagged these as actions but did not execute them. Slither is an independent third-party detector — agreement between BCCC labels and slither findings is strong evidence of label correctness; disagreement is a label-quality signal.

**Methodology:**

| Step | Action | Output |
|---|---|---|
| **I1** | For each of 766 NV+vuln contracts, run slither (Python API). Build a per-contract "slither detector hit set" (e.g., `{reentrancy_eth, unchecked_transfer}`). Cross-reference with the BCCC label: does slither agree with the BCCC vulnerability label? | `labels/i1_nv_vuln_slither.csv` (766 rows × {id, bccc_labels, slither_hits, agreement}) |
| **I2** | Compute agreement rate: (a) at contract level (any slither hit on a BCCC-vuln-positive class = agree), (b) at class level (per-class F1 between BCCC label and slither), (c) overall macro-F1. | `labels/i2_agreement_metrics.csv` |
| **I3** | Disagreement analysis: for the contracts where slither says "clean" but BCCC says "vuln", sample 30 and manually inspect the source. Verdict per contract: which is right? | `labels/i3_disagreement_inspections.md` (30 manual reviews) |
| **I4** | Same for the 2 nine-folder contracts: are they synthetic templates or real? Use file content + slither hits as evidence. | `labels/i4_nine_folder_inspections.md` |
| **I5** | Stratified sample of 50 multi-positive contracts (balanced across co-occurrence patterns). For each, verify BCCC's multi-label vector against slither hits + manual read. | `labels/i5_multi_pos_inspections.md` |
| **I6** | Final decision matrix: based on I3-I5, recommend adjustments to `review_pending=1` set (promote/demote individual contracts), or keep as-is with a confidence label. | `labels/i6_decision_matrix.md` (with updated `review_pending.csv`) |
| **I7** | Update `contracts_clean.csv` if any labels change (rare, conservative). Bump to `v1.1` with `metadata.json` updated. | `outputs/contracts_clean_v11.csv` (only if labels changed) |

**Slither invocation pattern (for all I-steps):**

```python
from slither import Slither
slither = Slither(contract_path)  # auto-uses solc-select if pragma matches
detector_results = slither.run_detectors()  # dict[check_name, list[result]]
hits = set()
for check, results in detector_results.items():
    if results:  # non-empty list = at least one finding
        hits.add(check)
```

**Sub-sampling for slither at scale:** Running slither on all 67,311 contracts would take ~1-2 sec each → 19-37 hours. **WS-I uses stratified samples (766 + 2 + 50 + 30 = 848 contracts)** for tractability. For WS-O we run slither on a larger stratified sample (5,000 contracts) to get statistical power.

**Done criteria:**
- I1-I2 done: 766 NV+vuln contracts scanned + agreement rate computed
- I3: 30 manual reviews of disagreements written
- I4-I5: 52 manual reviews total
- I6: decision matrix; review_pending set updated or kept with rationale
- I7: dataset version bumped (if needed) + CHANGELOG entry

**Est:** 4-5 hours (slither runs are slow; manual review is the bottleneck)

**Scripts:** `scripts/i_label_slither_validation.py` (master), `scripts/i_agreement_metrics.py` (compute F1), `scripts/i_manual_review_template.md` (template)

---

### WS-J: Statistical EDA

**Goal:** Apply formal statistical tests and produce publication-quality visualizations for every Phase 1+2 finding that was previously just a number.

**Methodology:**

| Step | Action | Output |
|---|---|---|
| **J1** | **Chi-square test of independence** for each pair of 10 BCCC classes (45 pairs). Null: classes are independent. Report p-values, effect sizes (Cramér's V), and significance after Bonferroni correction (α = 0.05/45 = 0.0011). Confirm or refute the "DoS+Reentrancy" co-occurrence claim with formal statistics. | `eda/j1_chi_square_class_pairs.csv` (45 rows × {pair, chi2, p, cramers_v, significant}) |
| **J2** | **Per-class distribution plots** (histograms) for: LOC, n_functions, n_events, n_modifiers, n_contracts. 6 features × 10 classes = 60 PNG figures (or 6 faceted plots with 10-class overlay). | `eda/figures/dist_loc.png`, ... `dist_n_modifiers.png` (6 figures) |
| **J3** | **Outlier analysis:** IQR method (Q1 - 1.5×IQR, Q3 + 1.5×IQR) and Z-score (|z| > 3) on LOC. Per-class outlier counts. | `eda/j3_outliers.csv` |
| **J4** | **Pragma × class contingency table** (top 10 pragma versions × 10 classes). Chi-square for each cell. | `eda/j4_pragma_class.csv` + heatmap `eda/figures/pragma_heatmap.png` |
| **J5** | **Correlation matrix** for 12 numeric features (LOC, funcs, events, mods, contracts, has_pragma, 242 BCCC features). 12×12 heatmap with hierarchical clustering. | `eda/figures/correlation_matrix.png` |
| **J6** | **Per-class co-occurrence heatmap** (10×10). Color = log2(lift vs random). Annotate cells with contract count. | `eda/figures/cooccurrence_heatmap.png` |
| **J7** | **Class-conditional pragma version distribution** (stacked bar chart: 10 classes × pragma versions). Reveals if vulnerable classes cluster in old pragmas. | `eda/figures/pragma_by_class.png` |
| **J8** | **Contracts-per-class-vs-derivation plot** (scatter: x=n_pos, y=LOC, color=primary_class). Shows whether "harder" labels = "bigger" contracts. | `eda/figures/n_pos_vs_loc.png` |
| **J9** | **Sample inspection (20 contracts):** read source code for 20 randomly selected contracts (2 per class). Qualitative notes on style, common patterns, what makes a Reentrancy contract look like a Reentrancy contract. | `eda/j9_qualitative_sample.md` |
| **J10** | **Master EDA report** compiling all the above. Each finding has: (a) statistical test, (b) p-value or effect size, (c) plot, (d) interpretation. | `eda/eda_report.md` |

**Library:** All `matplotlib` + `seaborn`; `scipy.stats` for chi-square; `pandas` for tabular work.

**Done criteria:**
- J1-J8: all 12+ figures generated at ≥150 DPI
- J9: 20 manual sample reviews written
- J10: master report with all findings + interpretation

**Est:** 2-3 hours (most of the time is on the manual sample review and writing the report)

**Scripts:** `scripts/j_statistical_eda.py` (master), `scripts/j_plot_distributions.py` (J2), `scripts/j_plot_heatmap.py` (J4, J5, J6)

---

### WS-K: Source-Level + Slither Feature Engineering

**Goal:** Augment `contracts_clean.csv` with **class-discriminative features** derived from (a) source-code text patterns, (b) slither detector hits. Bump dataset to `v1.2`.

**Why:** Phase 1 noted "ignore the 242 BCCC features" but never validated that decision. Phase 2 kept only basic complexity counts (LOC, funcs, events, mods, contracts, SPDX, pragma). We have no Reentrancy-specific features, no IntegerUO-specific features, no DoS-specific features. **The graph extraction pipeline (SENTINEL) builds these from the AST** — but tabular AutoML baselines need them as flat features.

**Two feature groups:**

**K1. Source-level text features (computed by reading .sol files):**

| Feature | Pattern (regex / count) | Discriminative for |
|---|---|---|
| `n_call_value` | `\.\s*call\s*\(\s*value` | Reentrancy |
| `n_send` | `\.\s*send\s*\(` | Reentrancy (deprecated), UnusedReturn |
| `n_transfer` | `\.\s*transfer\s*\(` | Reentrancy (0.4.x), UnusedReturn |
| `n_delegatecall` | `\.\s*delegatecall\s*\(` | ExternalBug |
| `n_callcode` | `\.\s*callcode\s*\(` | ExternalBug (deprecated) |
| `n_selfdestruct` | `\bselfdestruct\b` | MishandledException (locked-ether) |
| `n_assembly` | `\bassembly\s*[\(\{]` | ExternalBug (low-level) |
| `n_unchecked_block` | `\bunchecked\s*\{` | IntegerUO (0.8.x feature) |
| `n_inline_assembly_assign` | `assembly\s*\{[^}]*:=` | ExternalBug |
| `n_block_timestamp` | `\bblock\.timestamp\b` | Timestamp |
| `n_block_number` | `\bblock\.number\b` | Timestamp |
| `n_now` | `\bnow\b` | Timestamp (deprecated) |
| `n_keccak256` | `\bkeccak256\s*\(` | Timestamp (weak PRNG) |
| `n_sha3` | `\bsha3\s*\(` | Timestamp (deprecated) |
| `n_arithmetic_ops` | `[\+\-\*\/\%][^=]` (rough) | IntegerUO |
| `n_require` | `\brequire\s*\(` | All classes (general safety) |
| `n_assert` | `\bassert\s*\(` | All classes (general safety) |
| `n_revert` | `\brevert\s*\(` | All classes |
| `n_modifier_use` | `\bmodifier\s+\w+` | All classes |
| `n_payable_funcs` | `function\s+\w+[^;{]*payable` | Reentrancy (entry points) |
| `n_view_pure` | `\bview\b\|\bpure\b` | (negative correlate: NonVulnerable) |
| `n_state_vars` | `\b(mapping\|address\|uint\|int\|bool\|string\|bytes)\d*\s+(public\|private\|internal)\s+\w+` | All |
| `n_using_for` | `\busing\s+\w+\s+for` | All (OZ pattern) |
| `n_oz_imports` | regex on `import.*openzeppelin` | All (templating flag) |
| `n_interface_decl` | `\binterface\s+\w+` | All (architecture complexity) |
| `n_library_decl` | `\blibrary\s+\w+` | All |
| `n_abstract_decl` | `\babstract\s+contract` | All |
| `n_oracle_calls` | `\boracle\b\|\bchainlink\b\|\baggregator\b` | Timestamp (oracle-related) |
| `n_reentrancy_lock` | `\bnonReentrant\b\|\bReentrancyGuard\b` | Reentrancy (negative correlate) |
| `n_ownable` | `\bOwnable\b\|\bonlyOwner\b\|\bowner\s*=` | All (OZ template) |
| `n_safe_math` | `\bSafeMath\b` | IntegerUO (negative correlate) |

**31 source-level features.** Implementation: parallelized file read + regex count.

**K2. Slither-derived features (from WS-I's slither runs, computed on full corpus via WS-O's stratified sample):**

| Feature | Mapping |
|---|---|
| `slither_n_reentrancy_eth` | Class11 |
| `slither_n_reentrancy_no_gas` | Class11 |
| `slither_n_unchecked_transfer` | Class06 |
| `slither_n_unchecked_send` | Class06 |
| `slither_n_unchecked_lowlevel` | Class06 |
| `slither_n_unused_return` | Class06 |
| `slither_n_timestamp` | Class04 |
| `slither_n_bad_prng` | Class04 |
| `slither_n_tx_origin` | Class07 |
| `slither_n_suicidal` | Class07 |
| `slither_n_unprotected_upgrade` | Class07 |
| `slither_n_msg_value_in_loop` | Class09 |
| `slither_n_calls_in_loop` | Class09 |
| `slither_n_divide_before_multiply` | Class10 |
| `slither_n_tautological_compare` | Class10 |
| `slither_n_strict_equality` | Class10 |
| `slither_n_incorrect_exp` | Class10 |
| `slither_n_constant_funcs_asm` | Class02 |
| `slither_n_constant_funcs_state` | Class02 |
| `slither_n_void_constructor` | Class02 |
| `slither_n_locked_ether` | Class02/03 |
| `slither_n_arbitrary_send_eth` | Class01 |
| `slither_n_arbitrary_send_erc20` | Class01 |
| `slither_n_controlled_delegatecall` | Class01 |
| `slither_n_delegatecall_in_loop` | Class01 |
| `slither_n_missing_zero_validation` | Class08 |
| `slither_n_uninitialized_state` | Class03 |
| `slither_n_uninitialized_storage` | Class03 |
| `slither_n_uninitialized_local` | Class03 |
| `slither_n_incorrect_return` | Class03 |
| `slither_total_detector_hits` | sum of all above |
| `slither_n_distinct_detectors_fired` | count of distinct |

**32 slither-derived features.** Subset of contracts (those run through slither in WS-O's stratified sample of 5,000) will have these; the rest get NaN. WS-K will document this and the downstream AutoML will handle NaN gracefully (XGBoost/LightGBM both support it).

**Final schema (24 existing + 31 K1 + 32 K2 = 87 columns):**

```
contracts_clean_v12.csv:
  [existing 24 cols] + [31 K1 source features] + [32 K2 slither features]
```

**Methodology:**

| Step | Action | Output |
|---|---|---|
| **K1** | Implement source-feature extractor (regex-based, vectorized). Run on all 67,311 contracts (~30 min). | `features/k1_source_features.csv` (67,311 × 31) |
| **K2** | From WS-O's slither run, extract per-contract slither hit counts. Join to v12. | `features/k2_slither_features.csv` (5,000 × 32) |
| **K3** | Merge with `contracts_clean.csv` → `contracts_clean_v12.csv`. Update `metadata.json` with version bump. | `outputs/contracts_clean_v12.csv` + `.parquet` |
| **K4** | Feature distribution sanity check: each new feature has reasonable range, no all-zero columns, no NaN-explosions. | `features/k4_sanity_check.md` |
| **K5** | Master report: per-feature rationale, BCCC-class correlation, distribution, sample values. | `features/feature_engineering_report.md` |

**Done criteria:**
- 63 new features documented (31 K1 + 32 K2)
- `contracts_clean_v12.csv` + `.parquet` exists
- `metadata.json` updated to v1.2
- Sanity check passed (no degenerate columns)

**Est:** 4-5 hours (K1 regex design is the slowest part)

**Scripts:** `scripts/k1_source_features.py`, `scripts/k2_slither_features.py`, `scripts/k3_merge_v12.py`, `scripts/k4_sanity_check.py`

---

### WS-L: AutoML Baselines (XGBoost + LightGBM + CatBoost + LogReg + RF)

**Goal:** Establish **what a tabular model can achieve** on the 67,311 contracts with the 24 + 31 + 32 = 87 features. This is the "ceiling" for any non-graph model on the BCCC labels, and the baseline against which SENTINEL must justify its complexity.

**Why:** SENTINEL Run 7 F1=0.3074, Run 9 v11 ep14 F1=0.2586. We don't know if a 30-line XGBoost script beats that. If it does, SENTINEL's GNN+CodeBERT is over-engineered for the task and we should reconsider the architecture.

**Methodology:**

| Step | Action | Output |
|---|---|---|
| **L0** | Install xgboost, lightgbm, catboost, optuna, imbalanced-learn, shap via `poetry add --group research`. Verify imports. | (infra) |
| **L1** | Build train/val/test from `split_assignments.csv` (already 46,581 / 9,982 / 9,982). Load v12 features. Handle NaN for slither features (fillna(0) + indicator column). | `automl/l1_features_loaded.npz` |
| **L2** | **Logistic Regression** baseline (multi-label, one-vs-rest, L2). Fast (~5 min training). | `automl/l2_logreg_metrics.json` |
| **L3** | **Random Forest** baseline (200 trees, max_depth=20). Moderate (~15 min). | `automl/l3_rf_metrics.json` |
| **L4** | **XGBoost** with 5-fold CV + Optuna hyperparameter search (50 trials, 5-fold). ~2-4 hours. | `automl/l4_xgboost_metrics.json` + `automl/l4_xgboost_best_params.json` |
| **L5** | **LightGBM** with 5-fold CV + Optuna (50 trials, 5-fold). ~1-2 hours. | `automl/l5_lightgbm_metrics.json` + `automl/l5_lightgbm_best_params.json` |
| **L6** | **CatBoost** with 5-fold CV (no optuna, sensible defaults). ~1 hour. | `automl/l6_catboost_metrics.json` |
| **L7** | **Per-model evaluation on val + test:** macro-F1, micro-F1, per-class F1, ROC-AUC, hamming loss, subset accuracy, label ranking AP, coverage error, Brier score, calibration (ECE). | `automl/l7_all_metrics.csv` (5 models × 12 metrics) |
| **L8** | **Compare models** to **SENTINEL Run 7 (F1=0.3074) and Run 9 v11 (F1=0.2586)**. The headline question: does any tabular model match or exceed SENTINEL on the same labels? | `automl/l8_model_vs_sentinel.md` |
| **L9** | **Per-class comparison matrix:** 5 AutoML models × 10 classes × 4 metrics (F1, ROC-AUC, ECE, F1-best-threshold). | `automl/l9_per_class_matrix.csv` (50 rows × 12 cols) |
| **L10** | **Master report:** wins, losses, surprises, recommendations for SENTINEL architecture. | `automl/automl_report.md` |

**Hyperparameter search space (Optuna, for L4 and L5):**
- `n_estimators`: [100, 1000]
- `max_depth`: [3, 12]
- `learning_rate`: [0.01, 0.3] (log)
- `subsample`: [0.6, 1.0]
- `colsample_bytree`: [0.6, 1.0]
- `min_child_weight`: [1, 10]
- `reg_alpha`: [1e-8, 10] (log)
- `reg_lambda`: [1e-8, 10] (log)
- `scale_pos_weight`: per-class (10 values)

**Class imbalance handling:** Use `scale_pos_weight` per head; evaluate macro-F1 (not micro); per-class F1; report both.

**Calibration:** Apply `CalibratedClassifierCV` (Platt scaling) after each best model; compare ECE before/after.

**Done criteria:**
- All 5 models trained + evaluated
- Per-class matrix populated
- L8 verdict: which model (if any) matches SENTINEL
- L10 report written with architectural recommendation

**Est:** 3-4 hours L0-L3, 4-6 hours L4-L6 (Optuna is the bottleneck), 1-2 hours L7-L10 = **8-12 hours total**

**Scripts:** `scripts/l1_load_features.py`, `scripts/l2_logreg.py`, ..., `scripts/l10_report.py`

**Critical:** L8 is the most important deliverable. If AutoML beats SENTINEL, this is a major finding. If SENTINEL wins, the report should explain WHY (graph structure, code semantics, etc.) and quantify the gap.

---

### WS-M: BCCC 242-Feature Value Test

**Goal:** **Validate or refute** the Phase 1 decision to "ignore the 242 BCCC pre-extracted features" (`01_exploration_inventory.md:343`).

**Why:** Phase 1 said "ignore them" based on intuition (BCCC features are EVM-level, SENTINEL uses source-level graphs). **No one ever tested** if those features are actually informative. If they are, we have free features. If they aren't, we have a published reason to ignore them.

**Methodology:**

| Step | Action | Output |
|---|---|---|
| **M1** | Extract 242 BCCC features from `BCCC-SCsVul-2024.csv` for the 67,311 kept contracts. Many will be duplicated (CSV is long-format — collapse first). | `automl/m1_bccc_242_features.csv` (67,311 × 242) |
| **M2** | Quick correlation analysis: 242 features × 10 classes. Top 20 most-correlated features per class. | `automl/m2_bccc_feature_correlations.csv` |
| **M3** | Train XGBoost on **only the 242 BCCC features** (no complexity, no slither, no source-text). Report macro-F1 + per-class. | `automl/m3_xgb_242_only.json` |
| **M4** | Train XGBoost on **only the 12 complexity features** (LOC, funcs, events, mods, has_pragma, etc.). Report same metrics. | `automl/m4_xgb_complexity_only.json` |
| **M5** | Train XGBoost on **only the 31 K1 source-text features**. Report same metrics. | `automl/m5_xgb_k1_only.json` |
| **M6** | Train XGBoost on **only the 32 K2 slither features** (note: only available for 5,000 contracts from WS-O). | `automl/m6_xgb_k2_only.json` |
| **M7** | Train XGBoost on **all 87 features (v12)**. | `automl/m7_xgb_all.json` (already in L4, reuse) |
| **M8** | **Comparison table** of macro-F1 per feature subset. Verdict: which subset is most informative? Are 242 BCCC features worth keeping? | `automl/m8_feature_subset_comparison.csv` |
| **M9** | Master verdict: **KEEP or DROP** the 242 BCCC features? With quantification of F1 gain/loss. | `automl/m9_bccc_242_verdict.md` |

**Decision matrix (provisional):**

| Scenario | M3 F1 | M4 F1 | M5 F1 | M6 F1 | M7 F1 | Verdict |
|---|---|---|---|---|---|---|
| A | >0.60 | <0.40 | <0.40 | n/a | ~0.60 | KEEP 242 — they ARE informative |
| B | <0.40 | <0.40 | <0.40 | <0.40 | ~0.50 | DROP 242 — not informative, no subset dominates |
| C | <0.40 | <0.40 | >0.55 | <0.40 | ~0.55 | KEEP K1, DROP 242 — source-text beats bytecode |
| D | <0.40 | <0.40 | <0.40 | >0.50 | ~0.50 | KEEP K2 (slither), DROP 242 — domain knowledge wins |

**Done criteria:**
- M1-M7: 5 XGBoost models trained
- M8 comparison table
- M9 verdict with quantification
- Recommendation updated in `metadata.json` and any future SENTINEL data prep

**Est:** 1-2 hours (mostly M1's data extraction; M2-M7 are XGBoost runs)

**Scripts:** `scripts/m1_extract_242.py`, `scripts/m2_correlations.py`, `scripts/m3_to_m7_train_xgb.py`, `scripts/m8_compare.py`

---

### WS-N: Dropped + Review-Pending Deep-Dive

**Goal:** Characterize the 1,122 dropped + 766 review-pending contracts to understand **why they are different** and whether the hold-out decisions are correct.

**Why:** Phase 2 made the drop/hold decisions (D-F1, D-B2) without inspecting the contracts in detail. We should verify that dropped contracts are not systematically biased (e.g., are they all old? all multi-label? all in a particular class?) and that review-pending contracts are genuinely ambiguous.

**Methodology:**

| Step | Action | Output |
|---|---|---|
| **N1** | **Dropped contracts (1,122) profile:** per-class breakdown (which classes they were labeled with), LOC distribution, pragma distribution, n_pos, has_pragma%, n_contracts. Compare against the 67,311 kept on each dimension. | `labels/n1_dropped_profile.csv` |
| **N2** | Statistical test: are dropped contracts statistically different from kept? (Mann-Whitney U on LOC, chi-square on pragma version, chi-square on primary class.) | `labels/n2_dropped_vs_kept_tests.csv` |
| **N3** | **Sample 30 dropped contracts** (stratified across the 2 dropped classes). Read source. For each: why was the contract only labeled with a dropped class? Is the drop decision correct, or is the contract actually NonVulnerable? | `labels/n3_dropped_inspections.md` |
| **N4** | **Review-pending (766) profile:** same as N1 but for the 766. | `labels/n4_review_pending_profile.csv` |
| **N5** | **Manual review of 50 review-pending** (stratified): for each, decide (a) trust NV (drop vuln label), (b) trust vuln (drop NV), (c) keep both with high uncertainty, (d) drop entire contract. Verdicts in 4 categories. | `labels/n5_review_verdicts.md` |
| **N6** | **Resolution plan:** based on N5, propose a final `review_pending` list. Update `contracts_clean.csv` if any change (rare). | `labels/n6_resolution.csv` (final review_pending list) |
| **N7** | Master analysis report. | `labels/dropped_review_analysis.md` |

**Decision matrix for N5 (per review-pending contract):**

| BCCC label | Slither hit (if any, from WS-I) | Source inspection verdict | Action |
|---|---|---|---|
| NV + Reentrancy | ReentrancyEth | "Has reentrancy" | Drop NV label; keep Reentrancy |
| NV + Reentrancy | (none) | "Safe reentrancy-safe" | Keep NV; drop Reentrancy |
| NV + Reentrancy | (none) | "Real reentrancy but slither missed" | Keep both; flag "high uncertainty" |
| NV + Multi-vuln | (varied) | "Truly ambiguous" | Keep in review_pending; defer |

**Done criteria:**
- N1-N2: dropped/kept profile + statistical tests
- N3: 30 dropped manual reviews
- N4-N5: review-pending profile + 50 verdicts
- N6: final review_pending list (or "keep all 766" with rationale)
- N7: master report

**Est:** 2-3 hours (the 80 manual reviews are the bottleneck)

**Scripts:** `scripts/n1_dropped_profile.py`, `scripts/n2_stat_tests.py`, `scripts/n4_review_profile.py`

---

### WS-O: Cross-Tool Label Consensus (Slither + Aderyn)

**Goal:** Run **slither + Aderyn on a stratified 5,000-contract sample** to establish a 3-way consensus between BCCC labels, slither findings, and Aderyn findings.

**Why:** A single static analyzer can be wrong. Comparing 2-3 independent detectors' agreement is a strong signal of label correctness. If slither+Aderyn both agree with BCCC on 80%+ of contracts, labels are reliable. If they disagree on 30%+, we have a quality problem.

**Tool speed comparison (verified 2026-06-06, contract 00001c83…):**

| Tool | Time | Issues Found |
|--------|-----:|:-------------|
| **slither 0.11.5** | 2-5s | (varies, source-level AST) |
| **Aderyn 0.6.8** | **3.1s** | 7 (all low: centralization-risk, dead-code, empty-require-revert, etc.) |
| ~~mythril~~ (full, unbounded) | 3m16s | ~~4~~ (excluded from batch per D-P3-9 — too slow) |

**Mythril benchmark (verified 2026-06-06, contract 00001c83…):**

| Config | Time | Issues Found |
|--------|-----:|:-------------|
| `--max-depth 3-4, 30s timeout` | ~4s | 0 (false negative — too shallow) |
| `--max-depth 8-10, 15s timeout` | 35-51s | 0 |
| `--max-depth 22, 60s timeout, reentrancy only` | 2s | 0 (path not reached) |
| **Full unbounded (default), 90s timeout** | **3m16s** | **4** (2× Integer Overflow SWC-101, 2× External Call SWC-107 — but NOT reentrancy SWC-102) |

**Key finding:** mythril's `StateChangeAfterCall` reentrancy detector does NOT fire on the BCCC test contract even with 3 minutes of full analysis. It detects integer overflow and external-call risks but misses the subtle approve-then-call pattern. **At 3m/contract × 500 = 25h, mythril is not viable as a batch reentrancy detector for Phase 3.** Aderyn replaces mythril (D-P3-10): ~65x faster, broader detector coverage (88 vs 17), Rust-based AST analysis instead of symbolic execution.

**D-P3-9 (RESOLVED 2026-06-06): Mythril excluded from WS-O batch.** Replaced by Aderyn (D-P3-10). Mythril Docker is kept for ad-hoc deep analysis on individual contracts if disagreements need resolution.

**Sample design (5,000 contracts, stratified):**

| Stratum | n | Why |
|---|---:|---|
| All 766 review-pending | 766 | Highest uncertainty |
| All 1,122 dropped | 1,122 | Validate drop decision |
| 50 random from each of 10 classes (n_pos=1) | 500 | Single-label baseline |
| 50 random multi-label (n_pos ≥ 2) per class (10 classes × 50) | 500 | Multi-label edge cases |
| 500 random from NonVulnerable (n_pos=1) | 500 | Largest class, need proportional sample |
| 500 random from Reentrancy (n_pos=1) | 500 | Second largest, similar |
| 112 random from each of WeakAccessMod, Timestamp, TransactionOrderDependence (post-filter, n=0 since D-F1) | 0 | (excluded; classes dropped) |
| Random remainder | ~600 | Round up to 5,000 |

**Methodology:**

| Step | Action | Output |
|---|---|---|
| **O0** | Install Aderyn on host (`curl ... | bash`, 88 detectors). Mythril Docker available for ad-hoc use; not run in batch. | (infra) |
| **O1** | Build the 5,000-contract stratified sample. Save ID list. | `automl/o1_sample_5000.csv` |
| **O2** | Run slither on the 5,000 contracts (parallelized, 8 workers). ~2-4 hours. Per-contract output: list of detectors that fired. | `automl/o2_slither_5000.csv` (5000 rows × 101 cols) |
| **O3** | Run Aderyn on the 5,000 contracts (parallelized, 8 workers). ~3-5 hours. Per-contract output: list of detectors that fired. | `automl/o3_aderyn_5000.csv` (5000 rows × 88 cols) |
| **O4** | **BCCC label vs slither agreement:** for each contract, compare BCCC's positive class set to slither's hit set. Per-class precision, recall, F1. | `automl/o4_bccc_vs_slither.csv` |
| **O5** | **BCCC label vs Aderyn agreement:** same, for Aderyn. | `automl/o5_bccc_vs_aderyn.csv` |
| **O6** | **3-way consensus** (BCCC vs slither vs Aderyn): contracts where all 3 agree, where 2 agree, where all 3 disagree. | `automl/o6_three_way.csv` |
| **O7** | **Class-by-class consensus matrix:** for each of 10 classes, what % of BCCC-positive contracts do slither+Aderyn detect? | `automl/o7_per_class_consensus.csv` |
| **O8** | **Disagreement deep-dive:** for the 50 contracts with worst 3-way disagreement, manual inspect. Verdict: which tool is right? | `automl/o8_disagreement_inspections.md` |
| **O9** | Master report: tool agreement rates, class difficulty ranking (by consensus rate), recommendations for SENTINEL ensemble. | `automl/cross_tool_report.md` |

**Slither detector-class mapping (for O4):** Use the §2.1 table. For each BCCC class, the "slither positive" = at least one of the mapped detectors fired.

**Aderyn detector-class mapping (for O5):**
| Aderyn detector | BCCC class |
|---|---|
| `arbitrary-transfer-from` | Class01 |
| `delegate-call-unchecked-address` | Class01 |
| `selfdestruct` | Class07 (or Class03 if no auth) |
| `unchecked-low-level-call` | Class06 |
| `unchecked-send` | Class06 |
| `unchecked-return` | Class06 |
| `reentrancy-state-change` | Class11 |
| `non-reentrant-not-first` | Class11 |
| `unsafe-casting` | Class10 |
| `division-before-multiplication` | Class10 |
| `incorrect-caret-operator` | Class10 |
| `incorrect-shift-order` | Class10 |
| `tautological-compare` | Class10 |
| `tautology-or-contradiction` | Class10 |
| `tx-origin-used-for-auth` | Class07 |
| `weak-randomness` | Class04 |
| `block-timestamp-deadline` | Class04 |
| `state-change-without-event` | Class03 |
| `require-revert-in-loop` | Class09 |
| `msg-value-in-loop` | Class09 |
| `costly-loop` | Class09 |
| `delegatecall-in-loop` | Class09 |
| `return-bomb` | Class09 |
| `contract-locks-ether` | Class03 |
| `delete-nested-mapping` | Class03 |
| `storage-array-memory-edit` | Class03 |
| `state-no-address-check` | Class08 |
| `incorrect-erc20-interface` | Class08 |
| `eth-send-unchecked-address` | Class01 |
| `abiencode-packed-hash-collision` | Class03 |

**Done criteria:**
- O1: sample list with strata counts
- O2: slither output for 5,000 contracts
- O3: mythril output for 500 (or "skipped, with rationale")
- O4-O7: agreement metrics computed
- O8: 50 disagreement inspections
- O9: master report with per-class consensus rates

**Est:** 2-4 hours (slither runs) + 1-2 hours (analysis) = **3-6 hours total.** Mythril step removed per D-P3-9.

**Scripts:** `scripts/o1_build_sample.py`, `scripts/o2_run_slither.py` (parallelized), `scripts/o4_to_o7_metrics.py`, `scripts/o8_inspections.py`

**Failure modes:**
- ~~Mythril install fails → document, skip O3, O5, O6; report uses slither only.~~ **RESOLVED 2026-06-06:** Mythril works in Docker but excluded from batch (3m/contract too slow). 2-way consensus (BCCC vs slither) is the only comparison.
- Slither takes >2x expected time → reduce sample to 2,000; document.
- Contracts fail to compile (per WS-C, 27% fail) → skip those, report per-class compile rate.

---

### WS-P: Slither-Based Graph-Level Features

**Goal:** Extract **graph structure features** from slither's intermediate representation (CFG, call graph, inheritance graph) — features that go beyond per-line text patterns and capture structural complexity.

**Why:** SENTINEL's GNN uses CFG + ICFG + DEF_USE edges. Slither exposes these via its API. We can compute graph features (node count, edge count, branching factor, depth, etc.) for all 5,000 slither-analyzed contracts from WS-O. These features may be more discriminative than text counts.

**Methodology:**

| Step | Action | Output |
|---|---|---|
| **P1** | For each of 5,000 WS-O contracts, load slither IR. Extract: (a) CFG node count, edge count, max depth, branching factor; (b) call graph: in-degree, out-degree, max fan-in, max fan-out; (c) inheritance: depth of inheritance, # of parent contracts; (d) storage: # state vars, # mappings, total storage slots. | `features/p1_slither_graph_features.csv` (5,000 × ~30) |
| **P2** | Aggregate to per-contract "graph complexity score" (weighted sum of features). | `features/p2_complexity_scores.csv` |
| **P3** | Per-class statistics: for each of 10 classes, mean graph node count, edge count, etc. | `features/p3_per_class_graph_stats.csv` |
| **P4** | Visualization: scatter of node count vs LOC, colored by class. Reveals graph complexity patterns. | `eda/figures/graph_features.png` |
| **P5** | Master report. | `features/slither_graph_features.md` |

**Done criteria:**
- P1-P2: 5,000 contracts have graph features
- P3: per-class stats
- P4: visualization
- P5: report

**Est:** 2-3 hours (slither IR extraction is slow per contract)

**Scripts:** `scripts/p1_extract_graph.py`, `scripts/p2_aggregate.py`, `scripts/p3_per_class.py`, `scripts/p4_plot.py`

---

### WS-Q: SHAP Feature Importance

**Goal:** Use **SHAP (SHapley Additive exPlanations)** to identify **which of the 87 features drive each class prediction** in the best AutoML model (XGBoost from WS-L4).

**Why:** "SHAP = which features matter for which classes" is the standard ML interpretability tool. It tells us, for example, that "Class11:Reentrancy is mostly driven by `n_call_value` and `n_payable_funcs`, with secondary contribution from `slither_n_reentrancy_eth`." This is publishable insight.

**Methodology:**

| Step | Action | Output |
|---|---|---|
| **Q1** | Train final XGBoost (best params from L4) on full train set. | (model) |
| **Q2** | Compute SHAP TreeExplainer values on val set (9,982 contracts × 10 classes). | `automl/q2_shap_values.npy` (large) |
| **Q3** | **Per-class SHAP summary plot** (10 PNGs): for each class, top 20 features ranked by mean |SHAP value|. | `automl/shap_class01.png`, ... `shap_class11.png` |
| **Q4** | **Global SHAP summary plot** (1 PNG): all features × all classes, mean |SHAP|. | `automl/shap_global.png` |
| **Q5** | **Feature importance table:** per-class top-10 features. | `automl/q5_top10_per_class.csv` |
| **Q6** | **Class confusion via SHAP:** for each pair of frequently-confused classes (e.g., DoS + Reentrancy), which features distinguish them? | `automl/q6_class_discriminators.md` |
| **Q7** | Master report. | `automl/shap_report.md` |

**Done criteria:**
- Q1: best XGBoost trained
- Q2-Q4: 11 PNG figures
- Q5: per-class top-10 table
- Q6: confusion-pair discriminators
- Q7: report

**Est:** 2-3 hours (SHAP is fast on tree models, ~1 hour for 10K × 87 × 10)

**Scripts:** `scripts/q1_train_final.py`, `scripts/q2_compute_shap.py`, `scripts/q3_to_q6_plots.py`

---

### WS-R: 3-Way Model Comparison (SENTINEL vs AutoML vs Static Tools)

**Goal:** Build a single comparison report that puts **SENTINEL GNN+CodeBERT, AutoML (XGBoost best), and static-analysis tools (slither, optionally mythril)** in head-to-head on the same 10-class BCCC labels.

**Why:** This is the headline deliverable. A clear comparison: "AutoML achieves F1=X. Slither alone achieves F1=Y. SENTINEL achieves F1=Z. Recommendation: [architecture]."

**Methodology:**

| Step | Action | Output |
|---|---|---|
| **R1** | Pull SENTINEL's existing eval results (from MEMORY/CHANGELOG/eval logs): Run 7 macro-F1=0.3074, Run 9 v11 ep14=0.2586. | (input) |
| **R2** | Pull best AutoML results from WS-L4 (XGBoost). | (input) |
| **R3** | Compute slither-as-classifier metrics from WS-O4: for each contract, predict positive class = at least one mapped detector fired. Compute macro-F1, per-class F1. | `automl/r3_slither_as_classifier.json` |
| **R4** | ~~Compute mythril-as-classifier from WS-O5.~~ **SKIPPED** (no mythril batch output per D-P3-9). | `automl/r4_mythril_as_classifier.json` (skipped) |
| **R5** | Compute **hybrid slither+XGBoost:** use slither hits as additional features, then XGBoost. Compare to XGBoost alone. | `automl/r5_hybrid_metrics.json` |
| **R6** | **Confusion matrix comparison:** 4 models × 10 classes × {precision, recall, F1, ROC-AUC}. | `automl/r6_4way_matrix.csv` |
| **R7** | **Win/loss analysis:** per class, which model wins? Aggregate: SENTINEL wins X classes, AutoML wins Y, slither wins Z, hybrid wins W. | `automl/r7_win_loss.md` |
| **R8** | **Cost-benefit:** F1 per (model complexity, inference time, training time). | `automl/r8_cost_benefit.md` |
| **R9** | **Architectural recommendation:** based on R1-R8, recommend for SENTINEL Run 10+ (e.g., "use SENTINEL for the 3 hardest classes, XGBoost for the 7 easier classes" or "drop SENTINEL entirely if XGBoost matches it"). | `automl/r9_recommendation.md` |
| **R10** | **Master 3-way report** (publication-quality). | `automl/three_way_comparison.md` |

**Done criteria:**
- R1-R8: all metrics computed
- R9: clear recommendation
- R10: master report

**Est:** 2-3 hours (mostly report writing; data is already in place from L, O)

**Scripts:** `scripts/r1_to_r8_compute.py`, `scripts/r9_recommendation.py`, `scripts/r10_report.py`

---

### WS-S: BCCC ↔ SmartBugs Class Semantic Mapping (Deep)

**Goal:** Build a rigorous, evidence-based mapping between BCCC's 10 fine classes and SmartBugs' 8 broad classes, with verification on real contracts.

**Why:** Phase 2's WS-D (`cross_corpus/overlap_report.md:18-32`) gave a preliminary mapping, but it was based on intuition, not on label agreement between the two datasets. The mapping matters because SmartBugs is SENTINEL's OOD test set (per ADR-0005); if the mapping is wrong, OOD evaluation is misleading.

**Methodology:**

| Step | Action | Output |
|---|---|---|
| **S1** | Read SmartBugs README and any docs. List the 8 SmartBugs categories and their definitions. | `labels/s1_smartbugs_taxonomy.md` |
| **S2** | For each of 8 SmartBugs categories, sample 10 contracts. Read the source. Verify the category label is correct. | `labels/s2_smartbugs_samples.md` (80 manual reviews) |
| **S3** | For each SmartBugs category, identify which BCCC class(es) correspond. Use the 10 sampled contracts per category as evidence. | `labels/s3_smartbugs_to_bccc.csv` (8 rows × {smartbugs_cat, bccc_classes, evidence_count}) |
| **S4** | **Reverse direction:** for each of 10 BCCC classes, which SmartBugs categories would map? Verify with samples from the 143 SmartBugs contracts. | `labels/s4_bccc_to_smartbugs.csv` |
| **S5** | **Symmetry check:** the two mappings should be consistent (S3 ∘ S4 = identity where possible). Flag asymmetries. | `labels/s5_symmetry_check.md` |
| **S6** | **Mapping confidence matrix:** 8 × 10 matrix, cell = confidence (high/medium/low) that SmartBugs_cat maps to BCCC_class. | `labels/s6_mapping_confidence.csv` |
| **S7** | Master report with the verified mapping table and rationale. | `labels/bccc_smartbugs_mapping.md` |

**Key issues to resolve:**
- SmartBugs "other" (has no BCCC equivalent) — which BCCC contracts would a SmartBugs "other" sample be labeled with?
- SmartBugs "short_addresses" — no BCCC equivalent
- SmartBugs "access_control" → BCCC Class07 (WeakAccessMod), but the reverse direction (BCCC Class07 → SmartBugs) is ambiguous (could be access_control OR other)

**Done criteria:**
- S1-S2: SmartBugs taxonomy + 80 manual samples
- S3-S4: bidirectional mapping
- S5: symmetry check
- S6: confidence matrix
- S7: master mapping report

**Est:** 2-3 hours (80 manual reviews + writing)

**Scripts:** `scripts/s1_read_smartbugs.py`, `scripts/s2_sample_smartbugs.py`, `scripts/s3_to_s6_mapping.py`

---

### WS-T: Multi-Label Structure Test

**Goal:** Determine whether the multi-label structure in BCCC is **truly multi-label** (a single contract can have multiple vulnerabilities independently) or **systematic co-occurrence** (e.g., "all IntegerUO contracts also have UnusedReturn"). This affects how SENTINEL should model the output head.

**Why:** Phase 1 found DoS+Reentrancy co-occurrence = 12,381 (18% of corpus). If this is just dataset bias, SENTINEL's per-class F1 is overstating model capability. If it's true multi-label structure, SENTINEL should learn joint predictions.

**Methodology:**

| Step | Action | Output |
|---|---|---|
| **T1** | **Conditional probability matrix:** for each pair (A, B), P(B | A) and P(A | B). Compare to P(B) baseline. If P(B\|A) >> P(B), the pair is systematic. | `automl/t1_conditional_probs.csv` |
| **T2** | **Lift matrix** (already in Phase 1): confirm/refute with formal test. For each pair with lift > 5, classify as (a) template artifact (SafeMath+OZ import), (b) true joint vulnerability, (c) labeling noise. | `automl/t2_lift_classification.md` |
| **T3** | **Multi-label prediction test:** train 2 models — (a) independent per-class binary classifiers, (b) joint multi-label classifier. Compare F1. If (b) > (a), structure exists. | `automl/t3_independent_vs_joint.json` |
| **T4** | **Label co-occurrence ablation:** remove the most-co-occurring 10% of contracts, retrain XGBoost, measure per-class F1 change. If removing co-occurrences hurts F1, the model relies on co-occurrence patterns. | `automl/t4_cooccurrence_ablation.json` |
| **T5** | **Independence assumption test:** χ² for each pair (already in J1) → cross-reference with T1 conditional probs. Pairs that fail independence AND have high co-occurrence are systematic. | (cross-ref) |
| **T6** | Master report: is BCCC's multi-label structure true or artifactual? Recommendation for SENTINEL head design. | `automl/multi_label_structure.md` |

**Done criteria:**
- T1: conditional probability matrix
- T2: lift classification
- T3: independent vs joint comparison
- T4: co-occurrence ablation
- T5: cross-reference with J1
- T6: master report

**Est:** 2-3 hours

**Scripts:** `scripts/t1_conditional.py`, `scripts/t2_lift_classify.py`, `scripts/t3_joint.py`, `scripts/t4_ablation.py`, `scripts/t6_report.py`

---

## 4. Dependency Graph

```
           ┌──────────────────────────────────────────────┐
           │  Phase 1 + Phase 2 outputs (DONE)            │
           │  - contracts_clean.csv (67,311 × 24)         │
           │  - integrity, labels, complexity reports     │
           └──────────────┬───────────────────────────────┘
                          │
        ┌─────────────────┼──────────────────────┐
        │                 │                      │
        ▼                 ▼                      ▼
   ┌─────────┐      ┌──────────┐          ┌──────────┐
   │ WS-J    │      │ WS-I     │          │ WS-N     │
   │(EDA)    │      │(labels+  │          │(dropped+ │
   │independent│    │ slither) │          │ review)  │
   └────┬────┘      └────┬─────┘          └────┬─────┘
        │                │                     │
        │                │ depends on          │
        │                ▼                     │
        │          ┌──────────┐                │
        │          │ WS-O     │                │
        │          │(slither+ │                │
        │          │ mythril) │◀───────────────┘
        │          └────┬─────┘
        │               │ provides slither hits
        │               ▼
        │         ┌──────────┐    ┌──────────┐
        │         │ WS-K     │    │ WS-P     │
        │         │(features │    │(graph    │
        │         │+slither) │    │ features)│
        │         └────┬─────┘    └────┬─────┘
        │              │              │
        │              ▼              │
        │       ┌──────────┐          │
        │       │ WS-L     │          │
        │       │(AutoML   │          │
        │       │ baselines)│          │
        │       └────┬─────┘          │
        │            │                │
        │    ┌───────┼───────┐        │
        │    ▼       ▼       ▼        │
        │  ┌────┐ ┌─────┐ ┌────┐      │
        │  │WS-M│ │WS-Q │ │WS-T│     │
        │  │242 │ │SHAP │ │multi│     │
        │  │test│ │     │ │label│     │
        │  └────┘ └─────┘ └────┘     │
        │                            │
        │            ┌──────────┐    │
        └───────────▶│ WS-R     │◀───┘
                     │(3-way    │
                     │ compare) │
                     └────┬─────┘
                          │
                     ┌────▼────┐
                     │ WS-S    │(independent: BCCC↔SmartBugs mapping)
                     └─────────┘
```

**Critical path:** WS-I → WS-O → WS-K → WS-L → WS-R (longest dependency chain)
**Independent parallel work:** WS-J, WS-N, WS-S can run in parallel from session start.

---

## 5. Decision Points (User Input Required)

| ID | Decision | Default (if no input) | Options |
|---|---|---|---|
| **D-P3-1** | Mythril install | Try Poetry; fall back to Docker; fall back to "skip mythril" | Poetry / Docker / Skip |
| **D-P3-2** | WS-O sample size | 5,000 contracts | 5,000 / 2,000 (faster) / 10,000 (more statistical power) |
| **D-P3-3** | Manual review sample size for WS-I | 766 NV+vuln + 50 multi-pos + 30 disagreements = 846 | 846 / smaller (e.g., 100) / larger (e.g., 2,000) |
| **D-P3-4** | AutoML model set (WS-L) | XGBoost + LightGBM + CatBoost + LogReg + RF | All 5 / 3 only (XGB+LGBM+LogReg) |
| **D-P3-5** | Optuna trials for WS-L4/L5 | 50 trials × 5 folds | 50 / 100 / skip optuna (use defaults) |
| **D-P3-6** | Should WS-K update `contracts_clean.csv` in place, or create `v1.2` alongside? | Create v1.2 (safer) | v1.2 (recommended) / overwrite |
| **D-P3-7** | If AutoML beats SENTINEL in WS-R, what action? | Document finding; do NOT change SENTINEL architecture in Phase 3 | Document only / plan SENTINEL simplification in Phase 4 / drop SENTINEL entirely |
| **D-P3-8** | If mythril install fails, skip mythril? | ✅ **Skip (per D-P3-9 benchmark)** — mythril works in Docker but is too slow (3m/contract) for 500-sample batch. Continue with slither only. | Skip / Docker fallback / abort WS-O |

---

## 6. Risk Register

| ID | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| **R-P3-1** | Mythril install fails (heavy native deps) | Med | Med | D-P3-1 fallback: Docker or skip |
| **R-P3-2** | Slither on 5,000 contracts takes >8h | Low | Med | Reduce sample (D-P3-2) |
| **R-P3-3** | AutoML on 87 features × 67K rows is slow | Low | Low | XGBoost on GPU; reduce Optuna trials (D-P3-5) |
| **R-P3-4** | Slither can't compile 27% of contracts (per WS-C) | High | Med | Skip failed compiles; report per-class compile rate |
| **R-P3-5** | Manual review reveals label noise beyond 766 | Med | Med | Add to review_pending; defer resolution |
| **R-P3-6** | BCCC paper not found | High | Low | Document gap; proceed with assumptions from existing README + GitHub |
| **R-P3-7** | BCCC 242 features are uninformative (scenario B in M9) | Med | Low | Drop them; document decision |
| **R-P3-8** | BCCC 242 features ARE informative (scenario A in M9) | Low | Med | Add to v1.2; document in metadata |
| **R-P3-9** | AutoML beats SENTINEL (WS-R finding) | Med | High (architectural decision) | D-P3-7: document, don't act in Phase 3 |
| **R-P3-10** | Multi-label structure is artifactual (WS-T) | Med | Med | Recommend SENTINEL use independent per-class heads (or co-occurrence-aware loss) |
| **R-P3-11** | SmartBugs taxonomy mismatch with BCCC | Med | Med | S6 confidence matrix; honest reporting in mapping |
| **R-P3-12** | Time budget overrun | High | Med | Cap at 2-3 workstreams per session; allow multi-session completion |

---

## 7. Definition of Done (Phase 3)

- [ ] All 12 workstreams (WS-I through WS-T) have their `Done criteria` satisfied
- [ ] `outputs/contracts_clean_v12.csv` (87 columns) exists with documentation
- [ ] `automl/` contains: best XGBoost model, SHAP values, all metrics JSON, master report
- [ ] `features/` contains: K1 source features, K2 slither features, P1 graph features, master report
- [ ] `labels/` contains: I1-I7 manual inspections, N1-N7 dropped/review analysis, S1-S7 SmartBugs mapping
- [ ] `eda/` contains: 12+ PNG figures + master EDA report
- [ ] Master comparison report `automl/three_way_comparison.md` exists
- [ ] Architectural recommendation `automl/r9_recommendation.md` exists with verdict on SENTINEL vs AutoML
- [ ] CHANGELOG entry for Phase 3 (mirroring Phase 2's format)
- [ ] MEMORY.md updated with Phase 3 summary (≤200 lines maintained)

---

## 8. Effort Estimate & Session Planning

| Workstream | Est. (h) | Parallelizable? | Suggested session |
|---|---:|---|---|
| **WS-J** (statistical EDA) | 2-3 | Yes (independent) | Session 1 |
| **WS-S** (BCCC↔SmartBugs mapping) | 2-3 | Yes (independent) | Session 1 |
| **WS-N** (dropped/review deep-dive) | 2-3 | Yes (independent) | Session 1 |
| **WS-I** (slither label validation) | 4-5 | Yes (independent) | Session 2 |
| **WS-K** (K1 source features) | 2-3 | Yes (after WS-I) | Session 2 |
| **WS-K** (K2 slither features) | 2-3 | Needs WS-O | Session 3 |
| **WS-O** (slither + mythril) | 7-13 | Critical path | Session 3 |
| **WS-P** (graph features) | 2-3 | Needs WS-O | Session 3 |
| **WS-L** (AutoML) | 8-12 | Needs WS-K | Session 4 |
| **WS-M** (242-feature test) | 1-2 | Uses L4 | Session 4 |
| **WS-Q** (SHAP) | 2-3 | Uses L4 | Session 4 |
| **WS-T** (multi-label structure) | 2-3 | Uses L4 | Session 4 |
| **WS-R** (3-way comparison) | 2-3 | Uses L, I, O | Session 5 |
| **TOTAL** | **35-50** | | **5-6 sessions** |

**Realistic split (assuming 3-4 hours/session):**
- Session 1: WS-J + WS-S + start WS-N (3 WS in parallel)
- Session 2: WS-I (label validation) + WS-K-K1 (source features)
- Session 3: WS-O (slither+mythril runs in background) + WS-P (graph features)
- Session 4: WS-L (AutoML) + WS-M + WS-Q + WS-T
- Session 5: WS-R (3-way comparison) + finalize CHANGELOG/MEMORY

**Parallel execution:** Within a session, multiple scripts can run in parallel (e.g., WS-J plots + WS-N profile + WS-S reading simultaneously).

---

## 9. Output File Manifest

All writes go to `Data/Deep_Dive/BCCC-SCsVul-2024_Deep_Dive/Phase3_DeepAnalysis_2026-06-06/`:

```
Phase3_DeepAnalysis_2026-06-06/
├── README.md                              (this plan)
├── 00_session_log.md                      (chronological log)
├── scripts/                               (all Python scripts)
│   ├── list_slither_detectors.py          (preflight)
│   ├── i_label_slither_validation.py      (WS-I)
│   ├── j_statistical_eda.py               (WS-J)
│   ├── k1_source_features.py              (WS-K)
│   ├── k2_slither_features.py             (WS-K)
│   ├── l1_to_l10_automl.py                (WS-L)
│   ├── m1_to_m9_242_test.py               (WS-M)
│   ├── n1_to_n7_dropped_review.py         (WS-N)
│   ├── o1_to_o9_cross_tool.py             (WS-O)
│   ├── p1_to_p5_graph_features.py         (WS-P)
│   ├── q1_to_q7_shap.py                   (WS-Q)
│   ├── r1_to_r10_three_way.py             (WS-R)
│   ├── s1_to_s7_smartbugs_mapping.py      (WS-S)
│   └── t1_to_t6_multi_label.py            (WS-T)
├── labels/                                (WS-I, WS-N, WS-S outputs)
│   ├── i1_nv_vuln_slither.csv
│   ├── i2_agreement_metrics.csv
│   ├── i3_disagreement_inspections.md
│   ├── i4_nine_folder_inspections.md
│   ├── i5_multi_pos_inspections.md
│   ├── i6_decision_matrix.md
│   ├── n1_dropped_profile.csv
│   ├── n2_dropped_vs_kept_tests.csv
│   ├── n3_dropped_inspections.md
│   ├── n4_review_pending_profile.csv
│   ├── n5_review_verdicts.md
│   ├── n6_resolution.csv
│   ├── n7_dropped_review_analysis.md
│   ├── s1_smartbugs_taxonomy.md
│   ├── s2_smartbugs_samples.md
│   ├── s3_smartbugs_to_bccc.csv
│   ├── s4_bccc_to_smartbugs.csv
│   ├── s5_symmetry_check.md
│   ├── s6_mapping_confidence.csv
│   └── s7_bccc_smartbugs_mapping.md
├── features/                              (WS-K, WS-P outputs)
│   ├── k1_source_features.csv
│   ├── k2_slither_features.csv
│   ├── k3_v12_merge.py
│   ├── k4_sanity_check.md
│   ├── k5_feature_engineering_report.md
│   ├── p1_slither_graph_features.csv
│   ├── p2_complexity_scores.csv
│   ├── p3_per_class_graph_stats.csv
│   └── slither_graph_features.md
├── eda/                                   (WS-J outputs)
│   ├── j1_chi_square_class_pairs.csv
│   ├── j3_outliers.csv
│   ├── j4_pragma_class.csv
│   ├── j9_qualitative_sample.md
│   ├── eda_report.md
│   └── figures/
│       ├── dist_loc.png                   (and 5 more)
│       ├── pragma_heatmap.png
│       ├── correlation_matrix.png
│       ├── cooccurrence_heatmap.png
│       ├── pragma_by_class.png
│       ├── n_pos_vs_loc.png
│       └── graph_features.png
├── automl/                                (WS-L, M, O, P, Q, R, T outputs)
│   ├── o1_sample_5000.csv
│   ├── o2_slither_5000.csv
│   ├── o3_mythril_500.csv                 (or "skipped")
│   ├── o4_bccc_vs_slither.csv
│   ├── o5_bccc_vs_mythril.csv
│   ├── o6_three_way.csv
│   ├── o7_per_class_consensus.csv
│   ├── o8_disagreement_inspections.md
│   ├── cross_tool_report.md
│   ├── m1_bccc_242_features.csv
│   ├── m2_bccc_feature_correlations.csv
│   ├── m3_to_m7_xgb_subsets.json
│   ├── m8_feature_subset_comparison.csv
│   ├── m9_bccc_242_verdict.md
│   ├── l1_features_loaded.npz
│   ├── l2_logreg_metrics.json
│   ├── l3_rf_metrics.json
│   ├── l4_xgboost_metrics.json
│   ├── l4_xgboost_best_params.json
│   ├── l5_lightgbm_metrics.json
│   ├── l5_lightgbm_best_params.json
│   ├── l6_catboost_metrics.json
│   ├── l7_all_metrics.csv
│   ├── l8_model_vs_sentinel.md
│   ├── l9_per_class_matrix.csv
│   ├── automl_report.md
│   ├── q2_shap_values.npy
│   ├── q5_top10_per_class.csv
│   ├── q6_class_discriminators.md
│   ├── shap_class01.png ... shap_class11.png
│   ├── shap_global.png
│   ├── shap_report.md
│   ├── t1_conditional_probs.csv
│   ├── t2_lift_classification.md
│   ├── t3_independent_vs_joint.json
│   ├── t4_cooccurrence_ablation.json
│   ├── multi_label_structure.md
│   ├── r3_slither_as_classifier.json
│   ├── r4_mythril_as_classifier.json
│   ├── r5_hybrid_metrics.json
│   ├── r6_4way_matrix.csv
│   ├── r7_win_loss.md
│   ├── r8_cost_benefit.md
│   ├── r9_recommendation.md
│   └── three_way_comparison.md
└── outputs/                               (the final dataset)
    ├── contracts_clean_v12.csv            (87 cols × 67,311 rows)
    ├── contracts_clean_v12.parquet
    ├── v12_metadata.json
    └── README.md
```

**Total output: ~75 files, ~200 MB** (assuming per-contract slither output is ~5 KB/contract × 5,000 = 25 MB).

---

## 10. Out of Scope (Deferred to Phase 4+)

- **SENTINEL model retraining** on the new v12 features (Run 10+)
- **BCCC paper acquisition** if WS-S2 / WS-I5 reveals critical label-semantic issues
- **Per-class AutoML** (train one model per class, not multi-label)
- **Time-series analysis** (e.g., 4-eye + 3-phase attention weights as features)
- **Graph neural network baselines** (would compete with SENTINEL but is a separate effort)
- **External benchmarks** (run SENTINEL + AutoML on SB Curated, on other OOD sets)
- **Production deployment** (Phase 3 is research-side; inference is unchanged)
- **Active learning** (use SENTINEL's predictions to prioritize which contracts to label)
- **Adversarial testing** (deliberately craft contracts that fool slither, mythril, AutoML, SENTINEL)

---

## 11. Decisions Confirmed (2026-06-06 session)

| # | Question | Decision |
|---|----------|----------|
| 1 | Time budget | **Full 35-50h / 5-6 sessions** — all 12 workstreams at proposed depth |
| 2 | Mythril (D-P3-1) | **Try Poetry add → Docker → skip** (3-tier fallback) |
| 3 | WS-O sample (D-P3-2) | **5,000 stratified**; scale up if results suspicious |
| 4 | AutoML set (D-P3-4) | All 5 (LR, RF, XGB, LGBM, CatBoost) — default |
| 5 | v1.2 strategy (D-P3-6) | **New file** `contracts_clean_v12.csv` (87 cols) alongside v1.0 — default |
| 6 | Manual review (D-P3-3) | **846** = 766 NV+vuln + 50 multi-pos + 30 disagreements — default |
| 7 | SENTINEL response (D-P3-7) | **Document only**, no SENTINEL changes in Phase 3 — default |
| 8 | Mythril fallback (D-P3-8) | **Skip** WS-O mythril half if install fails — default |
| - | Optuna trials (D-P3-5) | **50 trials × 5 folds** — default |

**Session structure** (from §8):
- **Session 1** (5-7h, independent): WS-J (statistical EDA) + WS-S (BCCC↔SmartBugs mapping) + WS-N start (dropped/review deep-dive)
- **Session 2** (6-8h): WS-I (slither label validation, 846 contracts) + WS-K-K1 (31 source-text features via regex)
- **Session 3** (8-13h, bottleneck): WS-O (slither+mythril on 5,000) + WS-P (graph features from slither IR)
- **Session 4** (8-12h): WS-L (AutoML) + WS-M (242-feature test) + WS-Q (SHAP) + WS-T (multi-label structure)
- **Session 5** (4-6h): WS-R (3-way SENTINEL vs AutoML vs slither) + CHANGELOG + MEMORY update

---

**Plan approved. Session 1 starting: WS-J + WS-S + WS-N (the 3 independent workstreams).**
