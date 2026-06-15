# SENTINEL Comprehensive Benchmark

> A properly-built, OOD-verified, multi-source benchmark for evaluating the SENTINEL vulnerability detection model.
> Replaces the 95.8%-contaminated SmartBugs + 82.9%-contaminated SolidiFI approach (per contamination audit 2026-06-14).

## Status

- **v0.1 (quickstart)**: Tier A + E (66 + 100 = 166 contracts). Build in progress.
- **v1.0 (full)**: Tiers A+B+C+D+E (~3,000-4,500 contracts). ~10-12 days effort.
- **v2.0 (post-Run-13)**: Re-verify against v4 export.

## Quick-start (v0.1)

```bash
# Build the quickstart benchmark
cd /home/motafeq/projects/sentinel
ml/.venv/bin/python -m data_module.benchmarks.build_benchmark --tier quickstart

# Run contamination check
ml/.venv/bin/python -m data_module.benchmarks.contamination_check --version v0.1

# Evaluate a model
ml/.venv/bin/python -m data_module.benchmarks.evaluate \
    --checkpoint ml/checkpoints/GCB-P1-Run<N>-<tag>_best.pt \
    --benchmark-dir data_module/benchmarks/benchmark_v0.1_quickstart/ \
    --thresholds ml/checkpoints/<run_name>_thresholds.json
```

## Directory structure

```
data_module/benchmarks/
├── README.md                          # This file
├── BENCHMARK_DESIGN.md                # Design doc (methodology, principles)
├── build_benchmark.py                 # Orchestrator: build all tiers, verify, version
├── contamination_check.py             # Per-build SHA-256 audit vs v3
├── evaluate.py                        # New benchmark evaluator (replaces benchmark_run9_*)
├── sources/                           # Per-tier builder scripts
│   ├── tier_a_existing_ood/           # 6 SB + 60 SF honest OOD
│   ├── tier_b_defihacklabs_heldout/   # 149 DeFiHackLabs held-out
│   ├── tier_c_bccc_2tool/             # ~2,000-3,000 BCCC 2-tool consensus
│   ├── tier_d_mutation/               # 500-1000 mutation-based
│   └── tier_e_safe/                   # 100 known-safe contracts
├── benchmark_v0.1_quickstart/         # Tier A + E only (Day 1 deliverable)
└── docs/                              # Per-topic docs
```

## Design principles

1. **Truly OOD** — verified by SHA-256 audit against v3 (and future v4, v5, etc.) training
2. **Multi-source** — 5 tiers, not relying on any single dataset
3. **Quality labels** — 2-tool consensus OR manual verification OR hand-crafted mutation
4. **Multi-version** — covers Solidity 0.3.x, 0.4.x, 0.5.x, 0.6.x, 0.7.x, 0.8.x
5. **Versioned** — snapshot per SENTINEL release; SHA-256 indexed
6. **Documented** — per-contract metadata: source, version, label_method, compiler
7. **Reproducible** — one build script, deterministic output
8. **Statistically meaningful** — minimum 30 honest OOD per class (target 50+)

## See also

- `BENCHMARK_DESIGN.md` — full design document with methodology details
- `docs/COMPARISON_VS_PRIOR.md` — how this differs from the inflated 80-95% approach
- `docs/MAINTENANCE.md` — how to add contracts, re-audit, re-version
- `data_module/temp/live_plans/post_training_process_complete_2026-06-14.md` §4 — how this benchmark is used in the post-training process
