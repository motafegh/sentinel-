# 2026-05-02 — Pre-Execution Audit: Gaps Found Before Moves 3–8

**Session date:** 2026-05-02  
**Type:** Documentation audit + planning correction  
**Status:** Doc changes committed; no source code changes in this session  
**Related docs updated:** `docs/STATUS.md`, `docs/ROADMAP.md`

---

## Purpose

Before executing Moves 3–8 from the 2026-05-02 movements plan, a full review of
docs against actual source code was conducted. This changelog records what was
found, what was corrected, and why. Every doc change in this session is traceable
to a specific finding below.

---

## Finding 1 — Audit Item #13 Was Already Fixed (ROADMAP correction)

### Problem
`docs/ROADMAP.md` listed audit item #13 ("FocalLoss scalar not cast to `float()`")
as an open task in Move 8, implying it needed to be fixed before retraining.

### Source Code Verification
Actual code inspection (2026-05-02) confirmed it was already fixed on 2026-05-01:

- `ml/src/training/focalloss.py`: `predictions.float()` and `targets.float()` cast
  at the top of `FocalLoss.forward()` — labelled **Fix #6** in the audit changelog.
- `ml/src/training/trainer.py`: `_FocalFromLogits.forward()` applies `logits.float()`
  before calling `torch.sigmoid()` — labelled **Fix #2** in the audit changelog.

### Correction
- ROADMAP Move 8 table updated: item #13 removed; only #9 and #11 remain open.
- New ROADMAP Move 1 added: "Confirm Audit #13 Closed" — a 5-minute verification
  step so the closure is explicitly checked and recorded, not just assumed.
- STATUS.md open items table updated: #13 marked as already fixed with commit reference.

---

## Finding 2 — P0-B Graceful Degradation Changes Risk Profile of Dataset Gap

### Problem
The original audit flagged "no script confirms edge_attr tensors exist in .pt files"
as a potential retrain crash risk.

### Source Code Verification
`ml/src/models/gnn_encoder.py` (P0-B, 2026-05-02) was inspected. The `forward()` method
contains explicit graceful degradation:

```python
if edge_attr is not None:
    e = self.edge_emb(edge_attr)   # [E] → [E, edge_emb_dim]
else:
    # Graceful degradation: old .pt files may lack edge_attr.
    e = torch.zeros(edge_index.shape[1], self.edge_emb_dim, ...)
```

Note also from `graph_schema.py` (the `EDGE_TYPES` docstring):
> Pre-refactor .pt files on disk were produced by the old ast_extractor.py which
> stored shape [E, 1] — GNNEncoder ignores edge_attr in both cases.

### Corrected Risk Assessment
The risk is **not a crash** — it is a **silent quality regression**: the retrain
will complete successfully but will train on zero-vectors instead of real edge type
embeddings, gaining no benefit from P0-B.

### Correction
- ROADMAP: Move 0 rewritten to reflect the actual risk (silent quality loss, not crash).
  The validate script is still essential — the purpose is confirming we get P0-B's benefit,
  not preventing a crash.
- STATUS.md: graph dataset gap note updated to accurately describe graceful degradation behavior.

---

## Finding 3 — No Retrain Evaluation Protocol Documented

### Problem
The plan said "retrain with P0-A/B/C applied" with no specification of:
- What metric threshold constitutes success
- Whether the held-out split is fixed or re-randomized
- What the rollback rule is if quality regresses

Without a protocol, there is no way to know if the retrain improved anything,
and silent regressions on individual classes could be missed.

### Correction
Added **Retrain Evaluation Protocol** to both:
- `docs/STATUS.md`: full protocol table with baseline, split, success gate, per-class floor, rollback rule
- `docs/ROADMAP.md`: checklist under "Then: Retrain" section

Key parameters:
- Baseline: val F1-macro **0.4679** (epoch 34, `multilabel_crossattn_best.pt`)
- Split: `ml/data/splits/val_indices.npy` — fixed seed, do not regenerate
- Success gate: F1-macro > 0.4679 on same split
- Per-class floor: no class drops > 0.05 F1
- Rollback: revert checkpoint; investigate P0-B `edge_emb_dim=8` before re-running

---

## Finding 4 — Drift Detector Baseline Uses Wrong Data Source

### Problem
`compute_drift_baseline.py` was specified to walk `ml/data/graphs/` (training data)
to compute the KS test baseline. This is architecturally wrong:

- `ml/data/graphs/` contains BCCC-SCsVul-2024 contracts — a 2024 historical snapshot
- The KS test measures: "does this production request look like training data?"
- 2026 production contracts will differ structurally from 2024 training data by
  construction — the alert would fire on nearly every real request
- This makes the drift monitor useless from day one

### Correction
Drift baseline strategy updated in both `docs/STATUS.md` and `docs/ROADMAP.md` Move 7:
- Phase 1 (warm-up): collect feature statistics from first 500 real requests, emit no alerts
- Phase 2 (active): write `drift_baseline.json` from warm-up data, enable KS alerts
- `compute_drift_baseline.py` must support `--source [warmup|training]`
- `--source training` is available for offline testing with a prominent warning

---

## Finding 5 — M6 API Has No Auth/Security Design

### Problem
The M6 `api/` directory spec listed routes, files, and docker-compose structure
but contained no mention of authentication, API key management, rate limiting,
or contract confidentiality — despite this being a security-focused tool where
users would submit undeployed contract source code.

Building the routes before the security model is defined makes retrofitting
authentication disruptive (all route handlers need to be updated).

### Correction
Added **Security Design** section to ROADMAP under M6, specifying:
- Bearer token auth via FastAPI `Depends`
- `SENTINEL_API_KEYS` env var (never hardcoded)
- `slowapi` or Redis rate limiting: 10 audits/min per key
- Audit payloads logged at DEBUG only (contract confidentiality)
- Input validation: 500KB max, UTF-8 required before Slither

This must be designed before route implementation begins.

---

## Finding 6 — ZKML Pipeline Has No Resolution Path

### Problem
M2 (ZKML) has been marked "source complete, pipeline not yet run" since 2026-04-29.
Neither the Moves 3–8 list nor the Later Sprints table contained any item to
either run the pipeline or formally descope it. Left unaddressed, M2 will remain
in perpetual "source complete but never validated" limbo indefinitely.

### Correction
Added **ZKML Pipeline Resolution (S5.5)** section to ROADMAP with two explicit options:
- Option A: run the pipeline (environment setup, ONNX export, EZKL prove, verify)
- Option B: formally descope to S10

A decision must be made and recorded. STATUS.md M2 row updated to note the missing
resolution path.

---

## Finding 7 — No Unit Test Plan for New Stateful Modules

### Problem
Moves 4, 5, and 7 create three new stateful, IO-heavy modules:
- `cache.py` — file I/O with TTL, cache key construction
- `drift_detector.py` — rolling buffer, KS test, Prometheus counter
- `promote_model.py` — MLflow registry writes, CLI argument validation

None of these had test coverage planned. Bugs in IO-heavy stateful modules
are silent (wrong file written, wrong key used, buffer never flushed) and
difficult to diagnose in production without a test baseline.

### Correction
Added **Unit Test Plan for New Stateful Modules** table to ROADMAP specifying
the key test cases for each module alongside its corresponding Move.

---

## Summary of All Doc Changes Made

| File | Change type | Sections affected |
|------|-------------|-------------------|
| `docs/STATUS.md` | Updated | Module table (M2, M6 notes); open items table (#13 closed, new gaps added); new Retrain Protocol section; new Drift Baseline Note section |
| `docs/ROADMAP.md` | Updated | Move 0 (dataset validation); Move 1 (audit #13 closure check); Move 7 (drift baseline strategy); Move 8 (removed #13); M6 security design section; ZKML resolution section; Unit test plan section; Retrain protocol checklist |
| `docs/changes/2026-05-02-pre-execution-audit-gaps.md` | Created | This file |
| `docs/changes/INDEX.md` | Updated | New entry for this changelog |
