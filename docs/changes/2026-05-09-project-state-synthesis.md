# 2026-05-09 — Project State Synthesis (Cross-Module Briefing)

A consolidated end-to-end briefing across all six modules: **process,
design, data flow, expectations, and predictions**. Written as a
companion to the per-module plans dated 2026-05-09 so a reader can
understand the whole picture without opening every plan file.

Companion docs:
- `2026-05-09-module-plans-overview.md`
- `2026-05-09-M1-ml-plan.md` / `2026-05-09-M1-autoresearch-integration-plan.md`
- `2026-05-09-M2-zkml-plan.md`
- `2026-05-09-M3-mlops-plan.md`
- `2026-05-09-M4-agents-plan.md`
- `2026-05-09-M5-contracts-plan.md`
- `2026-05-09-M6-api-plan.md`

---

## 1. Where we are right now (the starting state)

- **M1 ML core** is the only module producing real output today. v3 checkpoint sits at tuned F1-macro **0.5069** on a frozen 10,278-graph val split.
- **M3 MLOps** (MLflow, DVC, Dagster, drift detector) is wired but the drift baseline is unanchored — there's no real production traffic to learn from yet.
- **M4 agents** (LangGraph + RAG + 3 MCP servers) runs end-to-end as a smoke test; some nodes (e.g. `static_analysis`) are referenced but probably not fully implemented; MCP clients are short-lived per call.
- **M2 ZKML** has clean source code but the proving pipeline has **never been executed**. No `proving_key.pk`, no real `ZKMLVerifier.sol`.
- **M5 contracts** has clean Solidity + tests but **`forge` has never been run** in this environment; `contracts/lib/` is empty.
- **M6 API** does not exist as a directory.

So the chain `audit request → ML → ZK → on-chain → feedback` is **complete on paper, intact only at the M1 link.** Everything we do next is closing the chain link by link.

---

## 2. The process — what happens, in order

The execution order from the overview doc:

```
M1 v4 retrain  →  M3 ops hardening  →  M5 forge bring-up  →  M2 ZKML decision
                                                                ↓
                                       M4 hardening  ←————— (parallel)
                                                                ↓
                                                              M6 API build
```

### Step-by-step

**Step 1 — M1 v4 retrain (1–3 days):** The autoresearch harness gets built first (`ml/scripts/auto_experiment.py` + `ml/autoresearch/program.md`). An agent runs short "smoke" trainings (~5 min each on subsampled data), keeps any candidate that beats v3's smoke equivalent, and only escalates to a full "confirm" run (~30–60 min, full val) for promising candidates. Goal: a v4 checkpoint with tuned F1 > 0.5069 and no per-class F1 below the v3 floors. Result is promoted via `promote_model.py --stage Staging`.

**Step 2 — M3 ops (parallel with v4):** While v4 is training, we exercise `drift_detector.py` with synthetic inputs to prove it alerts correctly, lock the MLflow promotion gates (Staging/Production rules), and document the run-name convention. No production traffic exists yet, so the warm-up baseline waits for M6.

**Step 3 — M5 forge (1 day):** Install Foundry, `forge install` OZ + forge-std, fix solc versions (0.8.20 for everything except `ZKMLVerifier.sol` which needs 0.8.17), run `forge test -vvv`. Add the missing reentrancy guard to `AuditRegistry.submitAudit`, verify slashing burns vs transfers, verify UUPS auth. **Don't deploy yet** — Sepolia deploy waits on M2.

**Step 4 — M2 decision (1 day for the ADR, ~2–4 hours for the pipeline if we run it):** Write ADR-039: Option A (run EZKL pipeline) or Option B (descope to S10). Recommendation in the plan: **Option A, but only after v4 lands** so we distil the proxy once against the final teacher. The pipeline is: train_proxy → export_onnx → setup_circuit → run_proof → handoff `ZKMLVerifier.sol` to M5 → solc-select 0.8.17 → forge build → deploy on Sepolia.

**Step 5 — M4 hardening (1–2 days):** Decide whether `static_analysis` is real or stub-and-remove; verify Dagster schedule fires; introduce an `MCPClientPool` so the gateway later (M6) doesn't pay per-call SSE setup; parameterise `build_graph(checkpointer=...)` so M6 can swap in a durable saver.

**Step 6 — M6 build (1–2 weeks, phased):** Phase 1 = scaffold + auth + rate-limit + size guard, calling `predict()` synchronously. Phase 2 = Celery + Postgres for `AuditJob`. Phase 3 = swap synchronous predict for `agents.build_graph().ainvoke()`. Phase 4 = wire `proof_task` (gated on M2). Phase 5 = `docker-compose up`. Phase 6 = CI.

---

## 3. The design — how the parts fit together

```
┌────────────────────────────────────────────────────────────────────┐
│                     User                                           │
└──────────────┬─────────────────────────────────────────────────────┘
               │  POST /v1/audit  (Bearer auth, ≤500KB, UTF-8)
               ▼
┌────────────────────────────────────────────────────────────────────┐
│  M6  api/  :8000  FastAPI gateway                                  │
│  - auth, rate-limit, size guard, correlation IDs                   │
│  - enqueue Celery task; return job_id (202)                        │
└──────────────┬───────────────────────────────────┬─────────────────┘
               │ Celery                            │ Postgres
               ▼                                   ▼
┌────────────────────────────────────────┐   AuditJob{id,status,result}
│  M4  agents/  build_graph().ainvoke()  │
│                                        │
│  ml_assessment ──► [high-risk?] ──► fan-out:                       │
│        │                  │   ├─ rag_research (FAISS+BM25+RRF)     │
│        │                  │   └─ static_analysis (Slither/Mythril) │
│        │                  ▼                ▼                       │
│        │             audit_check (on-chain history)                │
│        │                  │                                        │
│        ▼                  ▼                                        │
│        └──────► synthesizer (qwen3.5-9b-ud + rule fallback)        │
│                              │                                     │
│                              ▼                                     │
│                       final_report{}                               │
└──────────────┬─────────────────────────────────────────────────────┘
               │ (uses MCP SSE :8010/:8011/:8012)
               ▼
┌────────────────────────────────────────┐
│  M1  ml/src/inference/api.py  :8001    │
│  GAT + CodeBERT+LoRA + CrossAttention  │ ── teacher model (v4 ckpt)
│  windowed inference, /thresholds list  │
│  Prometheus + drift detector (M3)      │
└────────────────────────────────────────┘
               │
               ▼ (only for proof generation, async)
┌────────────────────────────────────────┐
│  M2  zkml/  EZKL pipeline              │
│  Proxy MLP (128→64→32→10, ~8K params)  │
│  proof π + publicSignals[65] → ~2 KB   │
└──────────────┬─────────────────────────┘
               │ submitAudit(addr, score, proof, signals)
               ▼
┌────────────────────────────────────────┐
│  M5  contracts/  AuditRegistry (UUPS)  │
│  on Sepolia                            │
│  3 guards: stake ≥ MIN, ZK verify,     │
│            consistency (sig[64]==score)│
│  emits AuditSubmitted event            │
└──────────────┬─────────────────────────┘
               │
               ▼
┌────────────────────────────────────────┐
│  M4  feedback_loop.py                  │
│  polls events → reads data/reports/    │
│  → re-indexes RAG corpus → BM25 rebuild│
└────────────────────────────────────────┘
```

---

## 4. The data — what flows through

| Stage | Shape | Where it lives |
|---|---|---|
| Raw input | Solidity source (UTF-8, ≤500 KB) | request body |
| Graph | PyG `Data` with `x [N,8]`, `edge_index [2,E]`, `edge_attr [E]` 1-D | built fresh per request via `graph_extractor.py` |
| Tokens | CodeBERT IDs, windowed if > 512 | built fresh via `preprocess.py` |
| Fused embedding | `[1, 128]` (CrossAttentionFusion output — **LOCKED**) | in-memory, internal |
| Per-class probs | `[10]` from teacher; per-class thresholds `[10]` from JSON sidecar | response key `thresholds` (list, never single float) |
| Proxy logits | `[10]` from ProxyMLP — only when proof requested | feeds EZKL |
| Proof | ~2 KB Groth16 + `publicSignals[65]` (64 features + 1 score) | M2 → M5 |
| On-chain record | `AuditResult{scoreFieldElement, proofHash, timestamp, agent, verified}` | Sepolia |
| RAG corpus | FAISS index + BM25 pickle + chunk metadata; updated by feedback loop | `agents/data/index/` |
| MLflow runs | per training run; tags: focal_gamma/alpha, lora_r, F1-macro, per-class F1 | `mlruns.db` |

The **scale factor 8192 = 2^13** is the bridge between float scores and on-chain `uint256`: `scoreFieldElement = round(prob * 8192)`. Decoded **little-endian** — getting that wrong silently produces garbage values (this bit our M2 source twice already).

---

## 5. Expectations (what we're aiming for)

**Concrete numbers to hit:**

| Module | Target | Source |
|---|---|---|
| M1 v4 | tuned F1-macro > 0.5069, no per-class F1 below v3 floor | `SENTINEL-EVAL-BACKLOG.md` |
| M1 autoresearch | smoke run < 8 min on 8 GB laptop; confirm reproduces v3 ±0.01 | autoresearch plan §12 |
| M3 drift | exercise script PASSes; alerts fire after 50 shifted requests | M3 plan §2.1 |
| M5 | 100 % `forge test` pass with `MockZKMLVerifier`, Slither no High/Critical | M5 plan §3, §7 |
| M2 (Option A) | proxy ≥ 95 % per-class agreement with teacher; on-chain `verifyProof` returns true | M2 plan §4 |
| M6 Phase 1 | 202 returned with valid bearer; 401 without; 429 above 10/min/key | M6 plan §4.1, §8 |

**Operational expectations:**

- **One module ships per ~1–2 days** for the small ones (M3, M5), one module per **~1 week** for the big ones (M6 phased).
- Every shipping change leaves a dated changelog under `docs/changes/` and updates `docs/STATUS.md`.
- The branch this work happens on is `claude/review-project-status-LFRYv` until something else is created; PR #34 stays draft until plans are accepted.

---

## 6. Predictions (the honest version)

These are the things expected to actually go wrong, ranked by likelihood:

1. **v4 doesn't beat 0.5069 on the first sweep.** The plateau at v3 was real. The recommended changes (focal γ=2, LoRA r=16, DoS sampler) are the obvious knobs but each only buys a few percent. Realistic outcome: 2–3 autoresearch sessions before a winner. Mitigation: per-class floor gating prevents accepting a regression dressed up as a macro-F1 win.

2. **8 GB VRAM bites us.** Batch 32 will OOM. LoRA r=32 will OOM. Some seemingly innocent agent edits (e.g. enabling more attention heads) will OOM. The locked-files hash guard handles structural drift; the per-knob bounds in `program.md` handle most of this; some runs will still die with exit code 2 and that's fine — autoresearch's loop is built around it.

3. **M2 Option A takes longer than the spec implies.** EZKL setup downloads a multi-GB SRS, the calibration step is finicky, and the auto-generated `ZKMLVerifier.sol` won't compile cleanly with newer solc. Plan on a full day even for a clean run.

4. **M5 `forge test` passes locally but Sepolia deploy hits gas/RPC issues.** Standard testnet noise. Recommendation: do a dry-run with a fork (`anvil --fork-url`) before mainnet broadcast.

5. **M4 `static_analysis` turns out to be a stub.** Flagged in the M4 plan because the `nodes.py` docstring says it's "added in M6" but `graph.py` already imports it. Either implement Slither for v1 or remove from `_route_after_ml`. Either way, one targeted fix.

6. **M6 Phase 1 will surface latency issues** that the MCP-per-call pattern hides today. That's why M4 plan §3.1 schedules the client pool *before* M6 Phase 3 wires the orchestrator in.

7. **The drift baseline question reopens once M6 ships traffic.** We're explicitly waiting for real audit requests; the synthetic exerciser is *only* a regression guard for the detector's own logic. First weeks of M6 traffic will probably need conservative thresholds before alerts are credible.

**What is NOT predicted:** that any of this requires changing the locked architecture constants. fusion_dim=128, NUM_CLASSES=10, edge_attr=[E] are sound and any change cascades through M2 (CIRCUIT_VERSION bump), M5 (verifier redeploy + `upgradeToAndCall`), and the dataset (re-extraction of 68k graphs). If anyone proposes touching them, the conversation is "do we actually need to" before anything else.

---

## 7. What to do next

The five plans are independent enough to start any of them, but the dependency graph and effort suggests:

- **Tonight or tomorrow:** Pick whether to start with **M1 autoresearch scaffolding** (`auto_experiment.py` + `program.md` + the hash sidecar) or **M5 `forge install + test`**. The first unblocks v4; the second unblocks M2/M5 deploy. They don't depend on each other.
- **Decide on M2 Option A vs B** — write the ADR. This is a 30-minute decision but it's been open for 10 days and it's blocking M5 deploy planning.
- **Don't start M6 yet.** Phase 1 is tempting but the value is low until M1 v4 lands and M4 client pool exists.
