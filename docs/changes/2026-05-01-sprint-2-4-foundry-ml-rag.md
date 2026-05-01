# 2026-05-01 — Sprint 2–4: Foundry Tests, ML Unit Tests, RAG/Agent Hardening

## Summary

Major source-code sprint completing Phases 6–9 of the SENTINEL improvement plan.
All changes are source-only (no forge execution, no GPU inference required).

---

## Bug Fixes (Pre-Sprint)

### ZKML Dimension / Checkpoint Bugs Fixed

- **Z1 `zkml/src/distillation/train_proxy.py`**: Wrong checkpoint path (`run-alpha-tune_best.pt` → `multilabel_crossattn_best.pt`), wrong feature dimension comments (64 → 128), wrong distillation target shape (`[N,10]` → `[N]` scalar via `.mean(dim=1)`)
- **Z2 `zkml/src/distillation/export_onnx.py`**: `dummy_input = torch.randn(1, 64)` → `torch.randn(1, 128)` — ONNX circuit input shape must match CrossAttentionFusion output_dim=128
- **Z3 `zkml/src/distillation/generate_calibration.py`**: Feature extraction updated from 64-dim to 128-dim; calibration JSON now contains correct 128-float input vectors

---

## Sprint 2 — Foundry Test Suite

New files:
- `contracts/test/mocks/MockZKMLVerifier.sol` — configurable `setReturnValue(bool)` mock
- `contracts/test/SentinelToken.t.sol` — 14 unit tests (stake, unstake, slash, events)
- `contracts/test/AuditRegistry.t.sol` — 3-guard tests, pause, UUPS upgrade, audit history
- `contracts/test/InvariantAuditRegistry.t.sol` — stateful fuzzing with 3 invariants
- `contracts/script/Deploy.s.sol` — Sepolia deployment (env-driven, post-deploy sanity checks)

Infrastructure fix: removed blanket `contracts/` entry from `.gitignore` (line 89) so Solidity source files are now tracked. Build artifacts (`contracts/out/`, `contracts/cache/`) remain gitignored.

---

## Sprint 3 — ML Unit Tests + Trainer Improvements

New files:
- `ml/tests/test_model.py` — SentinelModel forward shapes, `_StubTransformer` avoids 500MB CodeBERT load
- `ml/tests/test_preprocessing.py` — error types (ValueError/RuntimeError/FileNotFoundError), shapes, hash consistency
- `ml/tests/test_dataset.py` — DualPathDataset length, getitem, collation (binary `[B]` and multi-label `[B,10]`)
- `ml/tests/test_trainer.py` — FocalLoss scalar, evaluate() metrics, 3-epoch loss decrease, checkpoint round-trip

Modified:
- `ml/src/training/trainer.py` — `TrainConfig.loss_fn: str = "bce"` added; FocalLoss wired via `_FocalFromLogits` inner class (applies sigmoid internally, preserves logit interface)
- `ml/pyproject.toml` — peft pinned to `>=0.13.0,<0.16.0` (was GitHub HEAD)

---

## Sprint 4 — RAG Score Field + Static Analysis + Parallel Graph

### S4.1 RAG Score

- `agents/src/rag/chunker.py` — `score: float = 0.0` field added to `Chunk` dataclass
- `agents/src/rag/retriever.py` — results constructed via explicit `Chunk(..., score=rrf_scores[i])` ctor for old-pickle backward compatibility
- `agents/tests/test_retriever_filters.py` — `TestSearchScores` class: score > 0 and descending order assertions

### S4.5 Static Analysis Node + Parallel Deep Path

- `agents/src/orchestration/state.py` — `static_findings: list[dict]` (was `dict`)
- `agents/src/orchestration/nodes.py` — `static_analysis` node: Slither direct call, temp file, per-finding dict `{tool, detector, impact, confidence, description, lines}`; synthesizer updated to count High+Medium findings
- `agents/src/orchestration/graph.py` — parallel fan-out: `_route_after_ml` returns `["rag_research", "static_analysis"]` for deep path; `audit_check` fan-in waits for both automatically

Graph topology (deep path):
```
ml_assessment → [rag_research ‖ static_analysis] → audit_check → synthesizer
```

---

## Documentation Updates

- `SENTINEL_ACTIVE_IMPROVEMENT_LEDGER_UPDATED_2026-04-28.md` — fully updated: new phases in §1, updated §2 current position, new §4 DONE entries (Z1/Z2/Z3 + Sprints 2-4), new §7.5/7.6 learning notes, updated §8 constraints, expanded §9 priority list through item 37
- `SENTINEL-SPEC.md` — targeted updates: routing logic comment (parallel vs sequential), AuditState `static_findings` type, conditional routing description, RAG score known-issue resolution, audit_server ABI fix note, ADR-031 (parallel LangGraph fan-out), ML file inventory (ml/tests/), contracts file inventory, agents test inventory
- `Additional skills, tools.md` — Implementation Status table added at top
