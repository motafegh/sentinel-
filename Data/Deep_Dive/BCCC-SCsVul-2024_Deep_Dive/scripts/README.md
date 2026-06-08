# Phase 1 Exploration Scripts

**Date:** 2026-06-06
**Purpose:** Read-only exploration of the BCCC-SCsVul-2024 dataset to understand its structure before any cleaning.

---

## Files

| Script | What it does | Output |
|---|---|---|
| `bccc_phase1_explore.py` | First pass: file tree, CSV schema, basic counts | Inventory findings 1-4 (multi-label, 68,433 unique, 12 folders, CSV long format) |
| `bccc_phase1_explore2.py` | Class distribution, imbalance, multi-label stats | Finding 5 (imbalance 4.9×, 60.6% single-label) |
| `bccc_phase1_explore3.py` | Cross-corpus: BCCC vs SmartBugs-curated | WS-D precursor; informs Phase 2 cross-corpus check |
| `bccc_phase1_explore4.py` | Co-occurrence matrix, NV+vuln contradictions, ID hash check | Findings 6, 7, 8 (766 contradictions, DoS+Reentrancy 18%, keccak-256 ID) |

**Status:** ✅ All 4 scripts complete. Output documented in [`../01_exploration_inventory.md`](../01_exploration_inventory.md) (435 lines, 10 findings).

---

## Reproducibility

```bash
cd /home/motafeq/projects/sentinel
source .venv/bin/activate  # root venv has pandas, hashlib, csv (no extra deps)

# All 4 scripts are read-only on BCCC source tree.
# Run in order:
python Data/Deep_Dive/BCCC-SCsVul-2024_Deep_Dive/scripts/bccc_phase1_explore.py
python Data/Deep_Dive/BCCC-SCsVul-2024_Deep_Dive/scripts/bccc_phase1_explore2.py
python Data/Deep_Dive/BCCC-SCsVul-2024_Deep_Dive/scripts/bccc_phase1_explore3.py
python Data/Deep_Dive/BCCC-SCsVul-2024_Deep_Dive/scripts/bccc_phase1_explore4.py
```

Each script writes a small summary to stdout. No CSV/JSON outputs (findings captured in the markdown report).

---

## Related

- [`../01_exploration_inventory.md`](../01_exploration_inventory.md) — The 10 findings from these scripts
- [`../README.md`](../README.md) — Root table of contents
- [`../02_validation_deep_dive_plan.md`](../02_validation_deep_dive_plan.md) — Phase 2 plan (used these findings as input)
- [`../Phase2_Validation_2026-06-06/README.md`](../Phase2_Validation_2026-06-06/README.md) — Phase 2 entry point
- [`../Phase3_DeepAnalysis_2026-06-06/README.md`](../Phase3_DeepAnalysis_2026-06-06/README.md) — Phase 3 entry point

---

**Last updated:** 2026-06-06
