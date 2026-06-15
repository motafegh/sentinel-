# Run 12 Post-Training Process — Completion Summary (2026-06-14)

> **Purpose:** Documents the 12-step post-training process executed for Run 12 (after PID 230342 was killed cleanly at ep51). Summarizes: validation, calibration, benchmark evaluation, behavior checks, promotion. Run 12 is now in MLflow Staging.
>
> **Companion to:** `project_run12_launch.md` (the launch context)
> **For the workflow PLAN:** see `data_module/temp/live_plans/post_training_process_complete_2026-06-14.md` (846 lines, 5-phase generic plan)
> **Incremental log:** `~/.claude/scratch/post_training_run12_20260614.md` (24 KB, every step documented)

---

## TL;DR

- **Run 12 killed cleanly at ep51** (2026-06-14 21:30 UTC, PID 230342, SIGTERM, 21h35m elapsed, 0 NaN)
- **Best f1_tuned = 0.7004 @ ep50** (NEW SOTA, +2.07x Run 11 ep1's 0.3384)
- **Phase 1-5 of post-training plan completed** in ~4 hours active work
- **MLflow Staging**: `sentinel-vulnerability-detector` v1, Run ID 4d8de6c485cc4991989e32b861d09ba7
- **NOT Production** (requires drift_baseline=warmup + statistical sig)
- **Bonus: SmartBugs Wild full eval LAUNCHED** (PID 984402, 47K contracts, resumable, atomic state, SIGTERM handler — running in background)

---

## 12-Step Execution Log

| # | Step | Output | Notes |
|---|---|---|---|
| 0 | Create scratch file | `~/.claude/scratch/post_training_run12_20260614.md` | 24 KB working memory |
| 1 | Kill PID 230342 (SIGTERM) | Clean exit, checkpoint intact | Graceful shutdown worked |
| 1.5 | Verify process dead + checkpoint intact | 281 MB best.pt, 51 epochs, 0 NaN | All artifacts preserved |
| 2 | Phase 1.1: Reproducibility (L.1) | PASS | RNG state saved, TRANSFORMERS_OFFLINE=1, export hash match, poetry.lock unchanged |
| 3 | Phase 1.2: Performance analysis | 6 tune points, DoS_F1 0.11→0.38 | Plateau signal clear (gain per 10 ep: 0.226→0.006) |
| 4 | Phase 1.3: Run 12 final report | `docs/training/GCB-P1-Run12-v3dospatched-analysis-2026-06-14.md` (9 KB, 12 sections) | Full hypothesis verification + comparison to prior runs |
| 5 | Phase 2.1: Threshold tuning | `*_thresholds.json` (3.3 KB, 10 per-class) | F1-macro 0.6823 on val. **Script audited first** — v3-aligned. |
| 6 | Phase 2.2: Temperature scaling | `temperatures_run12.json` + `_stats.json` + `_ece_comparison.png` (99 KB) | Mean ECE 0.1948 → 0.0346 (-82%). **Legacy script was v9/v10 INCOMPATIBLE — wrote v3 replacement at `/mnt/c/lenovo/AppData/Local/Temp/opencode/calibrate_temperature_v3.py` (272 lines)** |
| 7 | Phase 3.0: Contamination audit (re-verify) | Re-verified: SmartBugs 95.8% / SolidiFI 82.9% in v3 | Same as earlier finding |
| 8 | Phase 3.1: Run 12 on v0.1 quickstart benchmark | `ml/reports/Run12_benchmark_v0.1.json` (41 KB) | **First honest OOD F1: 0.8743 (tuned), 0.8291 (tier)** on 66 contracts (5 in-benchmark classes). 2.83x the inflated prior numbers. |
| 9 | Phase 4: Behaviour/API validation | `ml/reports/Run12_round_trip.json` (8 KB) | Predictor OK; round-trip on test_contracts/ was 7/16 (OOD-tiny limitation noted — test_contracts median ~20 nodes vs training 295) |
| 10 | Phase 5.1: Promote to Staging | MLflow `sentinel-vulnerability-detector` v1 | Dry-run clean, live promotion succeeded |
| 11 | Save artifacts | Immutable `_FINAL.pt`, `_archive/Run12_2026-06-14.tar.gz`, all reports | Backups + log archive |
| 12 | Update docs | MEMORY.md, CHANGELOG.md, architecture.md | 3 docs updated |

---

## Key artifacts

### Calibration (Phase 2)
- `ml/checkpoints/GCB-P1-Run12-v3dospatched-20260613_best_thresholds.json` (3.3 KB)
- `ml/calibration/temperatures_run12.json` (401 B) + `_stats.json` (1.6 KB) + `_ece_comparison.png` (99 KB)

### Benchmark (Phase 3)
- `ml/reports/Run12_benchmark_v0.1.json` (41 KB) — 65/66 evaluated, F1 tuned 0.8743, F1 tier 0.8291
- `ml/reports/Run12_round_trip.json` (8 KB) — 7/16 positive, 1/4 safe (OOD-tiny limitation)
- `data_module/benchmarks/benchmark_v0.1_quickstart/` — 66 contracts, 0% contamination

### Behaviour/API (Phase 4)
- 19 alerts in Run 12 training (all WARN, no KILL)
- 0 NaN, 0 loss spikes, VRAM 5.9 GB
- Predictor loads correctly with v3 checkpoint (architecture=four_eye_v8, thresholds_loaded=True)

### Promotion (Phase 5)
- MLflow: `sentinel-vulnerability-detector` v1
- Stage: **Staging** (NOT Production — I.2.2 gates not met)
- Run ID: 4d8de6c485cc4991989e32b861d09ba7
- Git commit: 344ce5e
- Experiment: `sentinel-retrain-v2`

### Immutable copies (Step 11)
- `ml/checkpoints/GCB-P1-Run12-v3dospatched-20260613_FINAL.pt` (281 MB)
- `ml/checkpoints/GCB-P1-Run12-v3dospatched-20260613_FINAL.state.json`
- `ml/checkpoints/GCB-P1-Run12-v3dospatched-20260613_FINAL_thresholds.json`
- `ml/logs/_archive/Run12_2026-06-14.tar.gz` (91 KB)

---

## Scripts audit findings (CRITICAL)

**`ml/scripts/calibrate_temperature.py` is INCOMPATIBLE with v3.** It uses legacy v9/v10 paths:
- `ml/data/cached_dataset_v9.pkl`
- `ml/data/processed/multilabel_index_deduped.csv`
- `ml/data/splits/deduped/`

If run unchanged, it would calibrate against WRONG data. **Wrote v3-aware replacement** at `/mnt/c/Users/lenovo/AppData/Local/Temp/opencode/calibrate_temperature_v3.py` (272 lines) following the `tune_threshold.py` pattern (which IS v3-aligned). **Future calibration runs should use the v3 version.**

`ml/scripts/tune_threshold.py` (616 lines) is **v3-aligned** and ran successfully without modification.

---

## SmartBugs Wild Eval (bonus, post-training)

After the 12-step post-training completed, the user asked to validate Run 12 on the SmartBugs Wild dataset (47,398 real-world Ethereum mainnet contracts, ~9.7M LOC).

### Speed test (N=100, N=1000)
- 849/1000 successful (15.1% errors — old pragmas or too-large contracts)
- mean 343ms, p50 197ms, p95 874ms, p99 3735ms
- Throughput: 2.91 predictions/sec
- Extrapolated: 4.5 hours for full 47K

### Famous contracts test
- **3/3 compilable famous vulnerable contracts correctly flagged for Reentrancy** (EtherDelta, DAOToken, SmartBillions)
- Famous contracts that failed (The DAO, Parity MultiSig, WithdrawDAO, Hacken) are pre-0.4.21 code that doesn't parse with modern slither 0.11.5 — **pipeline limitation, not model bug**

### Full 47K eval LAUNCHED (2026-06-14 22:26 UTC)
- PID 984402, currently at ~4500/47398 (9.49%) in 23 min
- ETA: 3h39m remaining (~02:30 UTC 2026-06-15)
- Resumable: checkpoint every 500 contracts (atomic write), resume with --resume
- SIGTERM graceful shutdown: state saved, exit 0
- Live log: `ml/logs/smartbugs_wild_eval_<date>.log`
- State: `ml/data/smartbugs_wild_eval_state.json`
- Watchdog: `bash /mnt/c/Users/lenovo/AppData/Local/Temp/opencode/watch_smartbugs_eval.sh {status|tail|watch|stop|resume|force-kill}`
- Final report: `ml/reports/Run12_smartbugs_wild_FULL_<date>.json` + `_summary.md`
- Incremental reports every 2000 contracts
- Reliability features all verified (checkpointing, resume, SIGTERM)

---

## Next steps

1. **Wait for SmartBugs Wild full eval to complete** (~3.5 hours remaining, PID 984402)
2. **Run 13 prep** (4 fixes: drop GasException, extend L4, strip Solidifi bug_*, inject 658 BCCC ME)
   - Plans: `data_module/temp/live_plans/run_12_to_13_handoff_2026-06-14.md` (handoff) + `run13_plan_2026-06-14.md` (4 fixes)
3. **Production promotion** of Run 12 or Run 13 (whichever wins) — requires drift_baseline=warmup + statistical sig test
4. **Comprehensive benchmark v1.0** (Tier B DeFiHackLabs + Tier C BCCC 2-tool + Tier D mutation + Tier E safe)

---

## References

- **Incremental log:** `~/.claude/scratch/post_training_run12_20260614.md` (24 KB, every step documented)
- **SmartBugs Wild log:** `~/.claude/scratch/smartbugs_wild_eval_20260614.md`
- **Plan:** `data_module/temp/live_plans/post_training_process_complete_2026-06-14.md` (846 lines)
- **Launch context:** `~/.claude/projects/.../memory/project_run12_launch.md`
- **Run 12 final report:** `docs/training/GCB-P1-Run12-v3dospatched-analysis-2026-06-14.md`
- **Architecture:** `data_module/docs/architecture.md` (414 lines)
- **CHANGELOG:** `docs/CHANGELOG.md` (3026 lines)
- **MEMORY:** `~/.claude/projects/.../memory/MEMORY.md`
