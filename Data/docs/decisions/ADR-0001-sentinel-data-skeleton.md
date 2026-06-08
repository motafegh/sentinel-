# ADR-0001 — Why sentinel-data is a separate package

**Date:** 2026-06-09
**Status:** Accepted
**Deciders:** Ali Motafegh

---

## Context

Runs 1–9 of the SENTINEL model used the BCCC-SCsVul-2024 corpus directly through `ml/src/preprocessing/`. The Phase 1–5 investigation (2026-06-06 to 2026-06-08, preserved at `docs/legacy/bccc_deep_dive/`) found:

- **89.4% Reentrancy false-positive rate** — the BCCC folder structure assigned contracts to the Reentrancy class based on directory name alone; 89.4% of those contracts had no detectable reentrancy pattern.
- **86.9% CallToUnknown false-positive rate** — same root cause.
- **Run 9 F1 ceiling of 0.31** — the model was learning to predict label noise, not vulnerability patterns. The complexity proxy (total node count) was the dominant predictor for all 10 classes.
- **No separation between data quality and model quality** — when a run failed, it was unclear whether the cause was the architecture, the training procedure, or the data. The tightly coupled `ml/` structure made it impossible to version the dataset independently of the model.

The data side has fundamentally different concerns from the ML side:
- Different dependencies (slither, solc-select, huggingface-hub vs torch, transformers)
- Different cadence (data pipeline runs once per corpus version; model trains continuously)
- Different failure modes (connector errors, compile failures, label noise vs gradient collapse, OOM)
- Different ownership boundary (data quality is a pre-training concern; model quality is a training concern)

---

## Decision

Create `sentinel-data` as an independent installable Python package at `~/projects/sentinel/Data/`. It owns the entire pipeline from raw `.sol` ingestion through verified, versioned, multi-label dataset export. The ML module (`sentinel-ml`) consumes the exported shards; it never imports from `sentinel-data` source files.

The one-way dependency is enforced at install time: `sentinel-ml`'s `pyproject.toml` lists `sentinel-data ^0.1.0`; `sentinel-data`'s `pyproject.toml` has no reference to `sentinel-ml`. CI gate: `poetry show --tree | grep -i sentinel-ml` in `Data/` returns empty.

The BCCC deep-dive outputs are preserved at `docs/legacy/bccc_deep_dive/` as the historical record that motivated this decision. The v1.4 verified labels (24,021 contracts) may be re-introduced as a gold supplement in v2.1.

---

## Consequences

**Positive:**
- Data quality is independently verifiable and versioned (SQLite catalog + DVC)
- The 5 critical regression tests (Stage 2 byte-identical, Stage 4 Phase 5 regression, Stage 7 seam swap, 36-issue audit, 7 v2-readiness gates) are structurally impossible to bypass
- Each of the 17 data sources has a pinned version, SHA-256 manifest, and per-class label provenance
- The BCCC class of failure (folder-name labeling + no verification) is caught by Stage 4's verification module

**Negative:**
- The Stage 7 seam swap adds ~1 week of implementation and a byte-identical regression test
- Two separate venvs (`ml/.venv/` and `Data/.venv/`) require discipline to keep in sync on schema version
- The one-way boundary means `sentinel-data` cannot use any `sentinel-ml` utilities; duplicated utilities must be factored into a shared `sentinel-core` package (deferred to v3)

**Integration seam (Stage 7):**
The seam is `ml/src/preprocessing/graph_extractor.py` → `sentinel_data/representation/graph_extractor.py`. The Stage 7 seam swap produces byte-identical output for all 41,576 v10 graphs before deleting the old path.
