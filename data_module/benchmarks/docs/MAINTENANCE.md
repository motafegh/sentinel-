# SENTINEL Comprehensive Benchmark — Maintenance Guide (2026-06-14)

> How to add contracts, re-audit, re-version, and integrate with new data exports.

## When to add a new tier

- **A new data source becomes available** (e.g., new external dataset, new audit firm publication)
- **A class is under-represented** (e.g., MishandledException < 30 contracts)
- **A version is missing** (e.g., no Solidity 0.7.x contracts)
- **A real-world incident exposes a pattern** (e.g., new bug class)

## When to re-version

- **New data export** (v3 → v4): re-audit all contracts, possibly add new tier
- **New tier added**: version increment (v1.0 → v1.1, or v1.0 → v2.0 if material change)
- **Quarterly review**: check for new SmartBugs Wild / DeFiHackLabs additions

## Step-by-step: adding a new contract

1. **Get the source file** — place in `data_module/benchmarks/sources/<tier>/raw/`
2. **Compute SHA-256** and check against active data export:
   ```bash
   sha256sum <contract.sol>
   # Compare to v3 export: jq '.sha256' data_module/data/splits/v3/{train,val,test}.jsonl
   ```
3. **If NOT in export** (truly OOD): proceed
4. **Apply quality check**:
   - Tier A: assume clean (DASP-10 or injection)
   - Tier B: manual CVE check
   - Tier C: 2-tool consensus
   - Tier D: pattern + 2-tool verify
   - Tier E: 2-tool clean
5. **Copy to benchmark**:
   ```bash
   cp <contract.sol> data_module/benchmarks/benchmark_v<N>/contracts/by_class/<Class>/
   ```
6. **Update manifest** with metadata (source, class, sha256, size, label_method)
7. **Re-run contamination check**:
   ```bash
   ml/.venv/bin/python -m data_module.benchmarks.contamination_check --version v<N>
   ```
8. **Verify 0 overlap with active export**

## Step-by-step: re-versioning for a new data export

When v3 → v4:

1. **Build v4 export** (separate work, in `data_module/data/exports/sentinel-v4-.../`)
2. **Update Tier A + E** for any new honest OOD contracts
3. **Re-audit Tier B + C + D** for new honest OOD (BCCC 2-tool may yield more after v4 fixes)
4. **Build new benchmark version** (e.g., v2.0):
   ```bash
   ml/.venv/bin/python -m data_module.benchmarks.build_benchmark --tier all --version v2.0
   ```
5. **Run contamination check** against v4 splits (NOT v3)
6. **Update post-training process** to use v2.0:
   - Update `data_module/temp/live_plans/post_training_process_complete_2026-06-14.md` §4.0 references

## Integration with post-training process

Every Run N final report (Phase 1.3 of post-training) must include:
- Benchmark version used (e.g., `benchmark_v0.1_quickstart` or `benchmark_v1.0`)
- Per-class F1 (tier + tuned modes)
- Contamination check status (must be 0 overlap)
- Honest OOD F1 separately from contaminated-set F1 (if applicable)

`promote_model.py` should be extended (TODO) to FAIL if:
- The Run N final report is missing the benchmark section
- The contamination check has > 0 overlap
- The benchmark version doesn't match the data export version

## Audit log

All benchmark version builds should be logged in `~/.claude/projects/-home-motafeq-projects-sentinel/memory/MEMORY.md`:
- Build date
- Number of contracts per tier
- Number of contracts per class
- Contamination check result
- Active data export version at time of build

## Scripts inventory

| Script | Path | Purpose |
|---|---|---|
| `build_benchmark.py` | `data_module/benchmarks/build_benchmark.py` | Orchestrator |
| `contamination_check.py` | `data_module/benchmarks/contamination_check.py` | Per-build SHA-256 audit |
| `evaluate.py` | `data_module/benchmarks/evaluate.py` | Per-checkpoint evaluation |
| Tier A builder | `data_module/benchmarks/sources/tier_a_existing_ood/build.py` | Honest OOD from existing benchmarks |
| Tier B builder | `data_module/benchmarks/sources/tier_b_defihacklabs_heldout/build.py` | DeFiHackLabs held-out |
| Tier C builder | `data_module/benchmarks/sources/tier_c_bccc_2tool/consensus.py` | BCCC 2-tool consensus |
| Tier D builder | `data_module/benchmarks/sources/tier_d_mutation/build.py` | Mutation-based |
| Tier E builder | `data_module/benchmarks/sources/tier_e_safe/build.py` | Known-safe (negatives) |
| Legacy check | `ml/scripts/check_contamination.py` | SmartBugs vs BCCC only (v9/v10) — DO NOT USE for v3+ |
