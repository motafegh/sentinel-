# 2026-05-09 — Per-Module Plans Overview

This document is the entry point for the six per-module plans authored on
2026-05-09. Each plan is grounded in the current state of the source tree
and references the relevant Project-Spec files in `docs/Project-Spec/`.

Branch: `claude/review-project-status-LFRYv`
Author: Claude (project-state review)
Active checkpoint at time of writing: `ml/checkpoints/multilabel-v3-fresh-60ep_best.pt`
v3 tuned F1-macro: **0.5069** (gate for v4)

---

## Plans

| Module | Path | Plan file |
|--------|------|-----------|
| M1 | `ml/` | [2026-05-09-M1-ml-plan.md](2026-05-09-M1-ml-plan.md) |
| M2 | `zkml/` | [2026-05-09-M2-zkml-plan.md](2026-05-09-M2-zkml-plan.md) |
| M3 | `ml/scripts/`, `ml/src/inference/drift_detector.py`, MLflow/DVC/Dagster | [2026-05-09-M3-mlops-plan.md](2026-05-09-M3-mlops-plan.md) |
| M4 | `agents/` | [2026-05-09-M4-agents-plan.md](2026-05-09-M4-agents-plan.md) |
| M5 | `contracts/` | [2026-05-09-M5-contracts-plan.md](2026-05-09-M5-contracts-plan.md) |
| M6 | `api/` (not yet built) | [2026-05-09-M6-api-plan.md](2026-05-09-M6-api-plan.md) |

---

## Recommended Execution Order

1. **M1 v4 retrain** (and autoresearch harness) — pushes accuracy past the
   v3 0.5069 gate; teacher checkpoint stabilises before any downstream
   distillation. *Detail: M1 plan §2–§3.*
2. **M3 ops hardening** — finalise drift baseline workflow + MLflow
   registry promotion using the v3 (and later v4) checkpoint. *M3 plan §1–§3.*
3. **M5 forge bring-up** — install Foundry, run `forge install` / `build` /
   `test`; resolve 3 open contract gaps before any deploy. *M5 plan §1–§4.*
4. **M2 ZKML resolution (S5.5)** — choose Option A (run pipeline against v3
   or v4 teacher) or Option B (descope to S10) and write the ADR. Can run
   in parallel with M5 once the teacher is locked. *M2 plan §1.*
5. **M4 hardening** — close the small open issues (LLM client, Dagster
   schedule wiring, feedback-loop reports path), then promote MCP servers
   to long-running. *M4 plan §1–§3.*
6. **M6 build** — only after M1, M3, M4 are green. Auth + rate-limit
   design first, then routes, then docker-compose. *M6 plan §1–§4.*

Move 9 (multi-contract parsing, `multi_contract_policy="all"`) is tracked
as a post-M6 enhancement and lives in `docs/ROADMAP.md` rather than this
batch of module plans.

---

## Cross-Cutting Constraints (apply to every plan)

From `docs/Project-Spec/SENTINEL-INDEX.md` §"Critical Cross-Cutting Rules":

```
fusion_output_dim = 128        — LOCKED; ZKML proxy depends on it
GNNEncoder in_channels = 8     — LOCKED; tied to 68,523 training graphs
NUM_CLASSES = 10               — WeakAccessMod excluded; append-only
edge_attr shape = [E] 1-D      — NOT [E, 1]
API response key = "thresholds" (list) — NOT "threshold" (single float; Fix #6)
NO "confidence" field anywhere — removed in Track 3; KeyError if accessed
TRANSFORMERS_OFFLINE = 1       — must be set at shell, not Python
weights_only:
  graph .pt   → weights_only=True  (with add_safe_globals)
  checkpoint .pt → weights_only=False (LoRA state dict)
```

Any plan that touches these values must explicitly call out the rebuild
trigger in its checklist (e.g. bumping `fusion_output_dim` cascades into
M2 `CIRCUIT_VERSION`, ONNX re-export, full EZKL rebuild, `ZKMLVerifier.sol`
redeploy on Sepolia, and `AuditRegistry` verifier-address upgrade).

---

## How These Plans Relate to Existing Docs

- `docs/STATUS.md` — current module state (authoritative for "what is
  built / broken / paused"). Plans here defer to it.
- `docs/ROADMAP.md` — ordered remaining work. Plans here decompose roadmap
  items into concrete file-level changes.
- `docs/Project-Spec/SENTINEL-*.md` — locked architecture, constants and
  ADRs. Plans here cite spec sections; they never override them.
- `docs/changes/INDEX.md` — chronological changelog index (each plan is
  added there).
