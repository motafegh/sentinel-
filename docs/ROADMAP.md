# SENTINEL — Roadmap

**Last updated:** 2026-05-30  
**Current state:** Run 4 complete (F1=0.3362, ep32). Interpretability suite complete and validated (21 experiments, see `docs/interpretability/`). Data quality work + training fixes gate Run 5.

See `docs/STATUS.md` for current module state. Completed items are in `docs/changes/` and `docs/CHANGELOG.md`.

---

## Immediate — Data Quality (before Run 5)

These items fix known label noise, structural gaps, and training dynamics issues confirmed by the interpretability suite (2026-05-30). All should be done before launching Run 5.

### Quick wins (no retraining — do first)

| Item | ID | Description | Effort |
|------|----|-------------|--------|
| Fix solc-select | Interp-0 | `ml/.venv/bin/solc-select upgrade && solc-select install 0.8.25 && solc-select use 0.8.25` — unblocks EXP-L6 counterfactual validation | 5 min |
| Temperature scaling calibration | Interp-1 | Fit per-class temperature T on val set; wrap checkpoint in TemperatureScaler; re-run tune_threshold.py on calibrated outputs. ECE mean 0.252 → <0.05. See `docs/proposal/GNN_INTERPRETABILITY_FIXES_PROPOSAL.md` Fix 1. | 1 hr |

### Data quality (all required before Run 5)

| Item | ID | Description | Blocking |
|------|----|-------------|---------|
| CEI Reentrancy pattern injection | Sol-1 | Inject ~200 contracts with explicit CEI violations; labels confirmed by Slither `reentrancy-eth` | Yes |
| pragma/unchecked IntegerUO injection | Sol-2 | Inject contracts using `unchecked {}` blocks and unsafe arithmetic; verifiable by AST | Yes |
| Timestamp gating | Sol-3 | Remove Timestamp=1 labels on contracts where `block.timestamp` is only used in non-branching contexts. Interpretability confirmed size shortcut (d=0.643) — gating is essential. | Yes |
| re-extraction with fixed return_ignored | IMP-D1 | Run `reextract_graphs.py` to rebuild all 41K graphs with the IMP-D1 temporal ordering fix. Also raise max_nodes to 2048 (C-4 audit: 227 contracts exceed 1024, max=1735, all fit in 2048). | Yes |
| OpenZeppelin clean negatives | IMP-D2 / Sol-5 | Inject 100+ verified-clean OZ contracts as hard negatives to reduce false-positive rate | Yes |
| Fix EMITS edge extraction | Interp-6 | `graph_extractor.py` produces only 12 EMITS edges across 41K contracts. Fix Slither EventCall IR query. Enables UnusedReturn structural signal. | No |

**C-4 audit result (2026-05-30):** 227 / 41,576 contracts (0.55%) exceed 1024 nodes. Max observed: 1735. No contracts exceed 2048. Decision: **raise max_nodes to 2048** during IMP-D1 re-extraction. Change in `graph_schema.py` and the `--max-nodes` arg to `reextract_graphs.py`.

### Re-extraction command (IMP-D1)

```bash
PYTHONPATH=. python ml/scripts/reextract_graphs.py \
  --input-csv ml/data/processed/multilabel_index_cleaned.csv \
  --output-dir ml/data/graphs/ \
  --workers 10
```

Expected output: 41,576 graphs, schema v8, `return_ignored` now uses temporal ordering not global set.

---

## Training Run 5

**Trigger:** All Immediate data quality items complete + Interp-1 calibration validated.  
**Architecture:** Run 4 base + CEI auxiliary loss (Interp-2) + Timestamp size normalisation (Interp-3).  
**Target:** F1-macro > 0.40 on val set.

### New training changes from interpretability (add to Run 5)

| Item | ID | Description | Expected gain |
|------|----|-------------|---------------|
| CEI auxiliary loss | Interp-2 | Binary Phase 2 auxiliary loss head on Reentrancy-positive contracts. Forces Phase 2 to carry CEI signal (currently 0.014 Reentrancy drop vs 0.03 target). See `docs/proposal/GNN_INTERPRETABILITY_FIXES_PROPOSAL.md` Fix 3 for full code. | +0.03–0.05 on Reentrancy/TOD/ExternalBug |
| Timestamp size normalisation | Interp-3 | Either: (a) add contract node-count as explicit normalisation feature, or (b) size-stratified sampling in training batch. EXP-L7: F1 collapses from 1.0 (small) to 0.364 (large). | +0.02–0.04 on Timestamp |

### Gate criteria (must all pass to declare success)

| Criterion | Target |
|-----------|--------|
| Val F1-macro | > 0.38 |
| CallToUnknown F1 | > 0.30 |
| ExternalBug F1 | > 0.30 |
| TOD F1 | > 0.30 |
| No new class regressions vs Run 4 | All classes ≥ Run 4 − 0.03 |

### Resume command template

```bash
TRANSFORMERS_OFFLINE=1 TRITON_CACHE_DIR=/tmp/triton_cache PYTHONPATH=. nohup ml/.venv/bin/python ml/scripts/train.py \
  --gnn-layers 8 --gnn-prefix-k 48 --gnn-prefix-warmup-epochs 15 \
  --epochs 80 --batch-size 8 --gradient-accumulation-steps 8 \
  --loss-fn asl --compile --use-amp --phase2-edge-types 6 8 9 \
  --experiment-name sentinel-run5 --run-name GCB-P1-Run5 \
  --jk-entropy-reg-lambda 0.005 \
  >> ml/logs/run5-$(date +%Y%m%d).log 2>&1 &
```

---

## Agent Layer — Remaining Work

### Phase 1 completion

| Item | Description | Status |
|------|-------------|--------|
| `/hotspots` inference API endpoint | Expose true GNN attention weights from `prefix_attention_mean` and node-level attention scores via `GET /hotspots` | ⏳ PENDING |
| graph_inspector_server Phase 2 | Replace Slither-proxy scoring with real GNN attention weights from checkpoint | ⏳ PENDING — blocked on `/hotspots` endpoint |
| End-to-end MCP integration test | All four MCP servers (:8010/:8011/:8012/:8013) running simultaneously with a real contract | ⏳ PENDING |

### Phase 2 — Econ assessment node

New agent node for price manipulation simulation. Required inputs: contract bytecode + Slither call graph. Output: estimated manipulation cost (ETH) + affected functions.

Not yet designed. Depends on Phase 1 completion.

### Phase 3 — Full end-to-end integration test

Test harness that runs the full LangGraph topology against a real contract file with all MCP servers active. Verifies: ML result → routing decision → parallel tool calls → cross_validator verdicts → synthesizer report.

Blocked on Phase 1 completion + M6 Integration API.

---

## Infrastructure

### ZKML (M2)

| Item | Status | Notes |
|------|--------|-------|
| Source code (ezkl pipeline) | ✅ Complete | Never run |
| Proxy MLP | ✅ Valid | Linear(128→64→32→10) |
| ONNX export | ⏳ PENDING | Run after stable checkpoint (Run 5 gate) |
| ezkl gen-settings + calibrate | ⏳ PENDING | Blocked on ONNX export |
| ezkl prove | ⏳ PENDING | ~hours on CPU; GPU acceleration available |
| Solidity verifier deploy | ⏳ PENDING | Blocked on proof |

**Trigger:** Run 5 val F1 > 0.38 AND all gate criteria pass.

### Contracts (M5)

| Item | Status |
|------|--------|
| `contracts/lib/` populate | ❌ Not done — `forge install` never run |
| `forge build` | ❌ Not done |
| `forge test` | ❌ Not done |
| `AuditRegistry.sol` | Source exists; not compiled |
| `SentinelToken.sol` | Source exists; not compiled |
| `IZKMLVerifier.sol` | Source exists; not compiled |

No contracts milestone is blocking ML or agent work.

### M6 Integration API

`POST /v1/audit` endpoint does not exist. When built:
- Bearer token auth; env var `SENTINEL_API_KEYS`; rate limit 10 audits/min per key
- Input: `{ "source": "<solidity source>" }`; max 500 KB; reject non-UTF-8 before Slither
- Output: synthesizer report + ML result + cross_validator verdicts
- Wires ML predictor → agent graph → response

No design work started. Not blocking any current milestone.

---

## Explicitly Deferred (not building)

| Item | Reason |
|------|--------|
| Triage Agent as separate process | Adds orchestration complexity with no clear gain at current corpus size |
| Foundry fuzz tests in agent loop | Too slow for interactive audit; consider as async background job later |
| Cross-contract dependency graph | Requires multi-contract parsing (not implemented); defer to Phase 3 |
| Per-class tier threshold tuning in routing.py | TIER_CONFIRMED=0.55 / TIER_SUSPICIOUS=0.25 are fixed; per-class tuning after Run 5 |
| DEF_USE(10) edge type in Phase 2 | v8-AB result showed dilution of Reentrancy CEI; do not reintroduce without targeted test |
| S6 Observability (Prometheus/OTLP) | P3 — defer until end-to-end flow stable |
| S7 CI/CD (GitHub Actions) | P3 — defer until API surface stable |
| S8 Echidna/Halmos | P4 — property fuzzing + symbolic proofs |

---

## Milestone Sequence

```
Now
 │
 ├─ Sol-1 / Sol-2 / Sol-3 label fixes
 ├─ IMP-D1 re-extraction (41K graphs)
 ├─ IMP-D2 / Sol-5 OZ clean negatives
 ├─ C-4 max_nodes audit
 │
 ├─ Run 5 launch
 │   └─ Gate: F1 > 0.38, CallToUnknown/ExternalBug/TOD > 0.30
 │
 ├─ /hotspots inference endpoint (true GNN attention)
 ├─ graph_inspector Phase 2 (real attention weights)
 ├─ End-to-end MCP integration test
 │
 ├─ ZKML pipeline (ONNX → ezkl → Groth16 verifier) [if Run 5 gate passes]
 │
 ├─ Phase 2 agent: econ assessment node
 ├─ M6 Integration API (POST /v1/audit)
 ├─ M5 Contracts (forge build + test)
 │
 └─ Phase 3 agent: full end-to-end integration test
```
