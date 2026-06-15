# SENTINEL Comprehensive Benchmark — Design Document (2026-06-14)

> **Purpose:** Define a properly-built, OOD-verified, multi-source benchmark for evaluating the SENTINEL model. Replaces the 95.8%-contaminated SmartBugs + 82.9%-contaminated SolidiFI "benchmark" approach.
>
> **Author:** Ali + Claude (post contamination audit, 2026-06-14)
>
> **Replaces:** The implicit "evaluate on SmartBugs + SolidiFI directories" approach that was 80-95% training data.

---

## 1. Why a new benchmark is needed

**The legacy benchmark was inflated:**
- 137/143 SmartBugs (95.8%) in v3 training/val/test
- 290/350 SolidiFI (82.9%) in v3 training/val/test
- Only 66 contracts total are honest OOD (6 SB + 60 SF)
- 6 contracts per benchmark is not enough for statistically meaningful F1

**What we need:**
1. **Truly OOD** (not in v3 training/val/test) — verified by SHA-256 audit
2. **Large enough for statistics** — minimum 30 per class, target 50+
3. **Multi-source** — real-world + synthetic, not relying on one dataset
4. **Quality labels** — 2-tool consensus or manual verification, not raw CSV
5. **Multi-version** — covers Solidity 0.4.x, 0.5.x, 0.6.x, 0.7.x, 0.8.x
6. **Versioned** — snapshot per SENTINEL release; SHA-256 indexed
7. **Documented** — per-contract metadata (source, label method, compiler, etc.)
8. **Reproducible** — one orchestrator script, deterministic

---

## 2. Design: 3-tier hybrid benchmark

| Tier | Source | Method | Expected size | When to use |
|---|---|---|---|---|
| **A: Existing honest OOD** | SmartBugs 6 + SolidiFI 60 | Already verified, just exclude the 95.8% / 82.9% contaminated | 66 contracts | Always (small but free) |
| **B: Real-world held-out** | DeFiHackLabs 742 (split 80/20) | Hold 20% out of training; verify with 2-tool | ~149 contracts | Real-world exploit signal |
| **C: BCCC 2-tool consensus** | BCCC 111,897 → filter by 2-tool | Slither + aderyn intersection + Tier 3 deep-dive | ~2,000-3,000 contracts | Largest verified pool |
| **D: Mutation-based** | DISL NonVulnerable 26,914 → inject known bug | Apply SWC pattern; compile + verify | ~500-1000 contracts | Rare classes (ME, DoS, ToD) |
| **E: Known-safe** | DISL NonVulnerable sample + OpenZeppelin | Manual curation; 2-tool confirms no vulnerabilities | ~100 contracts | Negative examples |

**Total expected:** ~3,000-4,500 contracts, ~300-500 per class, multi-version, multi-source.

---

## 3. Per-class target coverage

| Class | v3 train count | Target honest OOD | Methods to use |
|---|---|---|---|
| CallToUnknown | 87 | 30+ | B (DeFiHackLabs) + D (mutation) + A (SolidiFI 60) |
| DenialOfService | 1,101 | 30+ | A (SB 6) + B + D (unbounded loop injection) |
| ExternalBug | 16,638 | 30+ | B (DeFiHackLabs exploits) + C (BCCC 2-tool) |
| IntegerUO | 9,452 | 50+ | A (SB 6) + C (BCCC — 16,740 candidates, 2-tool consensus) |
| MishandledException | 39 | 30+ | C (BCCC ME — 658 verified) + D (unchecked-send mutation) |
| Reentrancy | 11,399 | 50+ | A (SB 6) + B + C (BCCC — 17,698, 2-tool filter) |
| Timestamp | 6,324 | 30+ | A (SB 6) + D (block.timestamp injection) |
| TransactionOrderDependence | 647 | 30+ | D (tx.origin mutation) |
| UnusedReturn | 5,859 | 50+ | A (SB 6) + C (BCCC — 3,229, 2-tool filter) |
| (negative) | n/a | 100+ | E (DISL NonVulnerable + OpenZeppelin) |

---

## 4. Methodology per tier

### 4.1 — Tier A: Existing honest OOD (66 contracts)

**Source:** the 6 SmartBugs + 60 SolidiFI contracts not in v3 (per `check_contamination_v3.py` output)

**Method:** 
- Take the contamination audit output (`/tmp/contamination_v3_<date>.json`)
- Filter `tier1_exact == None` for each benchmark
- Copy to benchmark directory
- Verify label quality: SmartBugs = DASP-10 (well-known), SolidiFI = synthetic injection (known ground truth)

**Effort:** 1 hour (mostly file operations + label verification)

### 4.2 — Tier B: DeFiHackLabs held-out (~149 contracts)

**Source:** `data_module/data/raw/defihacklabs/repo/` (742 contracts)

**Method:**
- SHA-256 each contract; check against v3 training set
- For contracts NOT in v3 (likely the majority since DeFiHackLabs is large), add to benchmark
- For contracts IN v3, can still be used for the held-out audit (verify their ground truth)
- Each contract has manual exploit documentation (post-mortem, CVE)
- Multi-class: most DeFiHackLabs exploits are 2-3 class (Reentrancy + ExternalBug + DoS)

**Effort:** 4-6 hours (SHA-256 check + per-contract documentation lookup + label verification)

**Files:**
- `data_module/benchmarks/sources/tier_b_defihacklabs/build.py` (the orchestrator)
- `data_module/benchmarks/sources/tier_b_defihacklabs/contract_metadata.json` (per-contract CVE + classes)

### 4.3 — Tier C: BCCC 2-tool consensus (~2,000-3,000 contracts)

**Source:** `BCCC-SCsVul-2024/SourceCodes/<class>/*.sol` (111,897 files)

**Method (extends the proven ME methodology from `project_bccc_2tool_audit_2026-06-14.md`):**
1. **Compile probe** — `compile_probe.py` (already written for ME; reuse for other classes)
   - Try solc 0.4.24, 0.4.25, 0.5.0, 0.5.17
   - Fall back to older solc versions if needed
   - Apply the `tempfile.TemporaryDirectory()` per-call fix (from `aderyn_retry.py`) to avoid shared worker dir issue
2. **Slither audit** — class-specific detector lists
   - `MishandledException` → `unchecked-transfer`, `arbitrary-send-eth`, `void-cst`, `tautology`
   - `Reentrancy` → `reentrancy-eth`, `reentrancy-no-eth`, `reentrancy-benign`, `reentrancy-events`
   - `IntegerUO` → `divide-before-multiply`, `incorrect-shift`, `tautology`, `void-cst`
   - `UnusedReturn` → `unchecked-lowlevel`, `unchecked-send`, `unchecked-transfer`
   - `Timestamp` → `timestamp` (not used in audit yet)
3. **Aderyn audit** — same retry fix
4. **2-tool consensus** — intersection of slither + aderyn findings
5. **Tier 3 deep audit** — for STRONG-only matches (one tool fires, not both), manual code review
   - Skip if precision < 30% on the class
6. **Output** — high-confidence contracts per class with SHA-256 index

**Expected yields (per class, based on Phase 4 audit + ME precedent):**
- MishandledException: 658/5,154 (12.8%) — DONE
- Reentrancy: Phase 4 said 89% FP, but WITH 2-tool consensus expected yield 2-5% (similar to ME)
- IntegerUO: Phase 4 said clean → expected 5-10% high-confidence (subset of DIVE's labels)
- UnusedReturn: Phase 4 said clean → expected 5-10%
- Timestamp: Phase 4 said structural check exists → expected 10-20%
- Others: low expected yield, may skip

**Effort:** 3-5 days (compile probe is the bottleneck, ~2-3 hrs per class for 16K contracts)

**Files:**
- `data_module/benchmarks/sources/tier_c_bccc_2tool/consensus.py` (the orchestrator, extends ME work)
- `data_module/benchmarks/sources/tier_c_bccc_2tool/class_configs.py` (per-class detector lists)
- `data_module/benchmarks/sources/tier_c_bccc_2tool/results/<class>_2tool.json`

### 4.4 — Tier D: Mutation-based generation (~500-1000 contracts)

**Source:** Clean contracts from `data_module/data/raw/defihacklabs/repo/safe/` (if exists) OR generated clean contracts

**Method:** Apply known bug patterns from SWC Registry
1. Start with a known-clean contract (compiled successfully, no 2-tool findings)
2. Apply a known bug pattern (e.g., `tx.origin` for ToD)
3. Re-compile; verify the bug is now detectable by at least one tool
4. Result: synthetic contract with verified-vulnerable label

**Patterns to implement (priority order):**
- `unchecked_send` for MishandledException
- `unbounded_loop` for DoS (DenialOfService)
- `tx_origin` for TransactionOrderDependence
- `low_level_call_no_return_check` for CallToUnknown
- `timestamp_comparison` for Timestamp
- `reentrancy_no_mutex` for Reentrancy (basic CEI violation)
- `integer_overflow_pre_0_8` for IntegerUO (uses pre-0.8 unchecked arithmetic)
- `return_value_ignored` for UnusedReturn
- `arbitrary_external_call` for ExternalBug

**Effort:** 2-3 days (1 pattern = ~3-4 hours, 8 patterns = ~30 hours)

**Files:**
- `data_module/benchmarks/sources/tier_d_mutation/patterns/<class>.py` (one per pattern)
- `data_module/benchmarks/sources/tier_d_mutation/build.py` (orchestrator)
- `data_module/benchmarks/sources/tier_d_mutation/verify.py` (compile + 2-tool check)

### 4.5 — Tier E: Known-safe (100+ contracts)

**Source:** 
- DISL NonVulnerable pool (`BCCC-SCsVul-2024/SourceCodes/NonVulnerable/`, 26,914 files)
- OpenZeppelin audited contracts (download from github.com/OpenZeppelin/openzeppelin-contracts)

**Method:**
- Sample 50-100 contracts from NonVulnerable
- Verify with 2-tool: NO findings (if any tool fires, exclude)
- These serve as TRUE NEGATIVES (model should predict "safe" with high confidence)

**Effort:** 4-6 hours

**Files:**
- `data_module/benchmarks/sources/tier_e_safe/build.py`

---

## 5. Build order (recommended)

**Phase 1 (Day 1): Tier A** — fast, free, gets us 66 contracts immediately
**Phase 2 (Day 2): Tier E** — known-safe, parallel with Tier A cleanup
**Phase 3 (Day 3-4): Tier B** — DeFiHackLabs held-out (most real-world signal)
**Phase 4 (Day 5-9): Tier C** — BCCC 2-tool consensus (longest; parallel with Tier D)
**Phase 5 (Day 10-12): Tier D** — mutation-based (fills rare class gaps)

**Total: ~10-12 days of work** for the full 3,000-4,500 contract benchmark.

**Quick-start alternative (Day 1 only):** just Tier A + Tier E = ~166 contracts, ready to use immediately. Add other tiers incrementally as they're built.

---

## 6. Directory structure

```
data_module/benchmarks/
├── README.md                          # This file (overview)
├── BENCHMARK_DESIGN.md                # Design doc (methodology, principles)
├── COMPARISON_VS_PRIOR.md             # How this differs from prior inflated numbers
├── MAINTENANCE.md                     # How to add contracts, re-audit, re-version
├── build_benchmark.py                 # Orchestrator: build all tiers, verify, version
├── contamination_check.py             # Per-build SHA-256 audit vs v3
├── evaluate.py                        # The new benchmark evaluator (replaces benchmark_run9_*)
├── sources/
│   ├── tier_a_existing_ood/
│   │   ├── build.py
│   │   ├── smartbugs_honest_ood.json
│   │   └── solidifi_honest_ood.json
│   ├── tier_b_defihacklabs_heldout/
│   │   ├── build.py
│   │   └── contract_metadata.json
│   ├── tier_c_bccc_2tool/
│   │   ├── consensus.py               # Slither + aderyn intersection
│   │   ├── class_configs.py           # Per-class detector lists
│   │   ├── compile_probe.py           # Per-class compile + retry
│   │   ├── deep_audit.py              # Tier 3 manual review
│   │   └── results/
│   │       ├── mishandled_exception_2tool.json   # Already done
│   │       ├── reentrancy_2tool.json
│   │       ├── integer_uo_2tool.json
│   │       ├── unused_return_2tool.json
│   │       └── timestamp_2tool.json
│   ├── tier_d_mutation/
│   │   ├── build.py
│   │   ├── verify.py
│   │   ├── patterns/
│   │   │   ├── unchecked_send.py
│   │   │   ├── unbounded_loop.py
│   │   │   ├── tx_origin.py
│   │   │   ├── low_level_call.py
│   │   │   ├── timestamp_comparison.py
│   │   │   ├── reentrancy_no_mutex.py
│   │   │   ├── integer_overflow.py
│   │   │   ├── return_value_ignored.py
│   │   │   └── arbitrary_external_call.py
│   │   └── results/
│   │       └── <class>_mutations.json
│   └── tier_e_safe/
│       ├── build.py
│       ├── disl_nonvulnerable_sample.json
│       └── openzeppelin_audited.json
├── benchmark_v1.0/                    # The built benchmark (versioned)
│   ├── manifest.json                  # Overall metadata (build date, sources, counts)
│   ├── contamination_check.json       # SHA-256 audit vs v3 (must show 0 overlap)
│   ├── stats.json                     # Per-class, per-source, per-tier counts
│   ├── contracts/
│   │   ├── by_class/
│   │   │   ├── CallToUnknown/*.sol
│   │   │   ├── DenialOfService/*.sol
│   │   │   ├── ExternalBug/*.sol
│   │   │   ├── IntegerUO/*.sol
│   │   │   ├── MishandledException/*.sol
│   │   │   ├── Reentrancy/*.sol
│   │   │   ├── Timestamp/*.sol
│   │   │   ├── TransactionOrderDependence/*.sol
│   │   │   ├── UnusedReturn/*.sol
│   │   │   └── NonVulnerable/*.sol
│   │   └── all_contracts.jsonl        # Single file: path, class, source, tier, sha256, label_method
│   ├── metadata.jsonl                 # Per-contract detailed metadata
│   └── evaluation/
│       ├── evaluate.py                # Standardized evaluator
│       └── expected_metrics.json      # Sanity-check targets per class
└── docs/
    ├── BENCHMARK_DESIGN.md            # This doc (long form)
    ├── COMPARISON_VS_PRIOR.md
    └── MAINTENANCE.md
```

---

## 7. Build orchestrator (`build_benchmark.py`)

**Single entry point.** Runs each tier in order, validates outputs, contamination-checks, versions.

```python
"""
build_benchmark.py — Orchestrate the SENTINEL comprehensive benchmark build.

Usage:
    python -m data_module.benchmarks.build_benchmark --tier a          # Tier A only
    python -m data_module.benchmarks.build_benchmark --tier all        # All tiers
    python -m data_module.benchmarks.build_benchmark --rebuild v1.0    # Rebuild existing version
    python -m data_module.benchmarks.build_benchmark --verify v1.0     # Re-run contamination check on existing
"""
```

**Phases:**
1. **Phase 1: Tier A build** — copy 6 SB + 60 SF honest OOD contracts
2. **Phase 2: Tier E build** — sample 100 NonVulnerable contracts
3. **Phase 3: Tier B build** — DeFiHackLabs held-out
4. **Phase 4: Tier C build** — BCCC 2-tool consensus (longest)
5. **Phase 5: Tier D build** — mutation-based
6. **Phase 6: Aggregate** — combine all tiers into `benchmark_v<N>/`
7. **Phase 7: Contamination check** — SHA-256 audit vs v3 (must show 0 overlap)
8. **Phase 8: Stats + manifest** — write per-class/per-source/per-tier counts
9. **Phase 9: Version tag** — save as `benchmark_v<N>/`

**Output:** `data_module/benchmarks/benchmark_v1.0/` with all files.

---

## 8. Evaluator (`evaluate.py`)

**Single entry point for benchmark evaluation.** Replaces `benchmark_run9_smartbugs.py` + `benchmark_run9_solidifi.py` (which were 80-95% contaminated).

**Inputs:**
- `--checkpoint ml/checkpoints/<run_name>_best.pt`
- `--benchmark-dir data_module/benchmarks/benchmark_v1.0/`
- `--thresholds <path>` (from Phase 2 of post-training)
- `--tier a|b|c|d|e|all` (per-tier evaluation, default: all)
- `--per-class` (per-class F1 breakdown, default: true)
- `--contaminated` (also report contaminated subset for continuity, default: false)
- `--output-format json|markdown|html`

**Outputs:**
- Per-class F1 (tier + tuned modes)
- Per-tier F1 (Tier A vs B vs C vs D vs E)
- Per-source F1 (DeFiHackLabs vs BCCC-2tool vs SolidiFI vs etc.)
- OOD analysis (graph size, class distribution, version mix)
- Comparison table (vs prior runs, if `--compare-with` provided)
- HTML report with confusion matrices, per-class F1 bars

**Per-class minimum sample requirement:** report F1 only if N>=10 honest OOD contracts for that class; otherwise report "INSUFFICIENT_SAMPLE" (per OWASP ML guidelines).

---

## 9. Verification + contamination audit

**After every build:**
1. SHA-256 each contract in the benchmark
2. Compare to v3 train/val/test SHA-256 set
3. Verify ZERO overlap (HARD GATE)
4. Verify minimum 30 honest OOD per class (warn if less)
5. Verify class distribution is documented

**Output:** `benchmark_v<N>/contamination_check.json`:
```json
{
  "v3_artifact_hash": "5cc5cfcbf42bef4ced58b963ef98241bcf3ec4ab3bea5d198f336ec763a4faa9",
  "total_contracts": 3214,
  "total_sha256s": 3214,
  "overlap_with_v3": 0,
  "per_class_counts": {
    "CallToUnknown": 47,
    "DenialOfService": 38,
    "ExternalBug": 412,
    "IntegerUO": 524,
    "MishandledException": 658,
    "Reentrancy": 738,
    "Timestamp": 287,
    "TransactionOrderDependence": 152,
    "UnusedReturn": 358,
    "NonVulnerable": 100
  },
  "per_tier_counts": {
    "A_existing_ood": 66,
    "B_defihacklabs_heldout": 149,
    "C_bccc_2tool": 2599,
    "D_mutation": 400,
    "E_safe": 100
  },
  "per_source_counts": {
    "smartbugs_curated_honest_ood": 6,
    "solidifi_benchmark_honest_ood": 60,
    "defihacklabs": 149,
    "bccc_2tool_consensus": 2599,
    "mutation_generated": 400,
    "disl_nonvulnerable": 80,
    "openzeppelin_audited": 20
  }
}
```

---

## 10. Per-run tracking

**Every SENTINEL run reports against this benchmark:**
- Run 12: report benchmark_v1.0 honest OOD F1 (post-build, post-evaluation)
- Run 13: report benchmark_v1.0 honest OOD F1 (and re-run contamination check vs v4)
- Run N: report benchmark_v<N>.0 honest OOD F1 (new version per data export)

**Tracking is in `ml/reports/benchmark_<run_name>_<date>.json`** with the full evaluator output.

---

## 11. Cost / time estimate

| Tier | Effort | Yields | Cumulative time |
|---|---|---|---|
| A: Existing OOD | 1 hour | 66 | 1 hour |
| E: Known-safe | 4-6 hours | 100 | 1 day |
| B: DeFiHackLabs held-out | 4-6 hours | 149 | 2 days |
| C: BCCC 2-tool consensus | 3-5 days | ~2,000-3,000 | 1 week |
| D: Mutation-based | 2-3 days | 500-1000 | 1.5 weeks |
| **Total** | **~10-12 days** | **~3,000-4,500** | — |

**Quick-start (Day 1 only):** Tier A + E = ~166 contracts, ready for Run 12 post-training.

---

## 12. Comparison to prior "benchmark" approach

| Aspect | Prior (benchmark_run9_*) | New (comprehensive benchmark) |
|---|---|---|
| **Sources** | SmartBugs + SolidiFI (both heavily contaminated) | 5 tiers: existing OOD + DeFiHackLabs + BCCC 2-tool + mutation + safe |
| **Contamination** | 80-95% in training | 0% (verified per build) |
| **OOD verification** | None (assumed clean) | SHA-256 audit per build + per run |
| **Sample size** | 6-60 honest OOD per class | 30+ target per class (300-500 expected) |
| **Label quality** | SmartBugs = DASP-10, SolidiFI = injection (clean for honest OOD) | 2-tool consensus + manual review + 3rd-party audit (DeFiHackLabs CVE) |
| **Version mix** | Pre-0.6 (SolidiFI) + 0.4-0.6 (SmartBugs) | 0.3-0.8 (full range) |
| **Negative examples** | Implicit (NonVulnerable from same dirs) | Explicit (DISL NonVulnerable + OpenZeppelin) |
| **Reproducible** | Script + path | Script + version tag + manifest + contamination_check.json |
| **Re-evaluable** | Yes (run scripts) | Yes (re-run evaluate.py) |
| **Comparable across runs** | Hard (no version tag) | Easy (version tag + manifest) |
| **Statistically meaningful** | 6 contracts per class (nope) | 30+ contracts per class (yes) |

---

## 13. Decision needed from Ali

1. **Tier A + E (Day 1 quick-start)** — do we build this first, in parallel with Run 12 final?
2. **Tier C (BCCC 2-tool consensus, longest)** — start now or wait until Run 12 final is done?
3. **DeFiHackLabs split ratio** — 80/20 (149 held-out) or 50/50 (371 held-out, but 593 less for training)?
4. **Mutation patterns priority** — start with rare classes (ME, DoS, ToD, CtU) or common (Reentrancy, IntUO)?
5. **Benchmark version vs v3 export version** — tie them (v3 → benchmark_v1.0, v4 → benchmark_v2.0) or independent?
